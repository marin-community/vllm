# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import base64
import copy
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from infra.release.gpu_release import (
    GRUG_ARCHITECTURE,
    ReleaseError,
    assemble_candidate,
    build_matrix,
    extract_validation,
    finalize_release,
    inspect_wheel,
    load_json,
    sha256_file,
    validate_wheel_fragment,
    validation_matrix,
    verify_release_assets,
)

CONFIG_PATH = Path(__file__).parents[1] / "config.json"
FORK_COMMIT = "a" * 40
UPSTREAM_BASE = "b" * 40
BUILT_AT = "2026-08-03T12:00:00Z"
CANDIDATE_TAG = f"marin-vllm-gpu-candidate-{FORK_COMMIT[:12]}"


def write_wheel(
    path: Path,
    *,
    architecture: str,
    include_cumem: bool = True,
    version: str = "0.0.0.dev20260803+marin.test.cu129",
) -> None:
    dist_info = f"vllm-{version}.dist-info"
    metadata = (
        "Metadata-Version: 2.4\n"
        "Name: vllm\n"
        f"Version: {version}\n"
        "Requires-Python: >=3.10,<3.15\n"
        "Requires-Dist: torch==2.11.0\n"
        "Requires-Dist: transformers>=4.56.0\n"
        "\n"
    )
    wheel_metadata = (
        "Wheel-Version: 1.0\n"
        "Generator: test\n"
        "Root-Is-Purelib: false\n"
        f"Tag: cp38-abi3-linux_{architecture}\n"
        "\n"
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(f"{dist_info}/METADATA", metadata)
        archive.writestr(f"{dist_info}/WHEEL", wheel_metadata)
        archive.writestr("vllm/_C.cpython-312-test.so", b"compiled")
        if include_cumem:
            archive.writestr(
                "vllm/cumem_allocator.cpython-312-test.so", b"compiled"
            )
        archive.writestr(
            "vllm/model_executor/models/grugmoe.py",
            b"class GrugMoeForCausalLM:\n    pass\n",
        )


def fragment(
    tmp_path: Path,
    architecture: str,
    *,
    include_cumem: bool = True,
) -> dict:
    config = load_json(CONFIG_PATH)
    wheel = tmp_path / (
        "vllm-0.0.0.dev20260803+marin.test.cu129-cp38-abi3-"
        f"manylinux_2_28_{architecture}.whl"
    )
    write_wheel(wheel, architecture=architecture, include_cumem=include_cumem)
    return inspect_wheel(
        wheel,
        architecture=architecture,
        config=config,
        fork_commit=FORK_COMMIT,
        upstream_base=UPSTREAM_BASE,
        built_at=BUILT_AT,
        base_image=config["platforms"][architecture]["build_base_image"],
        base_image_digest=(
            f"example.invalid/{architecture}@"
            + config["platforms"][architecture]["build_base_image"].rsplit("@", 1)[-1]
        ),
        provenance={
            "system": "GitHub Actions",
            "run_id": "123",
            "runner_arch": architecture,
            "run_url": "https://github.com/marin-community/vllm/actions/runs/123",
        },
    )


def candidate(tmp_path: Path) -> dict:
    config = load_json(CONFIG_PATH)
    return assemble_candidate(
        [fragment(tmp_path, architecture) for architecture in config["platforms"]],
        config=config,
        repository="marin-community/vllm",
        candidate_tag=CANDIDATE_TAG,
        created_at=BUILT_AT,
    )


def validation(candidate_manifest: dict, architecture: str) -> dict:
    config = load_json(CONFIG_PATH)
    platform = next(
        item
        for item in candidate_manifest["platforms"]
        if item["architecture"] == architecture
    )
    validation_config = config["platforms"][architecture]["validation"]
    source_status = (
        "passed" if validation_config["run_source_tests"] else "not_run"
    )
    return {
        "schema_version": 1,
        "candidate_tag": candidate_manifest["release"]["tag"],
        "source_commit": candidate_manifest["source"]["fork_commit"],
        "architecture": architecture,
        "wheel": {
            "filename": platform["wheel"]["filename"],
            "sha256": platform["wheel"]["sha256"],
            "url": platform["wheel"]["url"],
        },
        "hardware": {
            "requested": validation_config["gpu"],
            "name": f"NVIDIA {validation_config['gpu']}",
            "compute_capability": validation_config["compute_capability"],
        },
        "environment": {
            "python_version": config["python_version"] + ".8",
            "torch_version": config["torch_version"],
            "torch_cuda_runtime": config["cuda_runtime_version"],
            "machine": architecture,
            "task_image": config["validation_task_image"],
            "vllm_package_path": (
                "/tmp/release/venv/lib/python3.12/site-packages/vllm/__init__.py"
            ),
        },
        "gates": {
            "wheel_sha256": {"status": "passed"},
            "distribution_metadata": {"status": "passed"},
            "vllm._C": {"status": "passed"},
            GRUG_ARCHITECTURE: {"status": "passed"},
            "cumem_allocator": {
                "status": "passed",
                "allocated_bytes": 16384,
                "checksum": 8_386_560.0,
            },
            "source_tests": {"status": source_status},
            "serve_smoke": {
                "status": "passed",
                "metrics": {
                    "completions": 4,
                    "min_completion_tokens": 32,
                    "output_tokens_per_second": 500.0,
                },
            },
        },
        "result": "passed",
    }


def test_inspect_wheel_records_release_identity_and_packaged_extensions(tmp_path):
    record = fragment(tmp_path, "x86_64")
    wheel = record["platform"]["wheel"]

    assert record["distribution"] == {
        "name": "vllm",
        "version": "0.0.0.dev20260803+marin.test.cu129",
        "requires_python": ">=3.10,<3.15",
        "torch_metadata_requirement": "torch==2.11.0",
    }
    assert record["source"]["fork_commit"] == FORK_COMMIT
    assert record["source"]["upstream_base"] == UPSTREAM_BASE
    assert record["build"]["python_version"] == "3.12"
    assert record["build"]["torch_version"] == "2.11.0+cu129"
    assert record["build"]["cuda_toolkit_version"] == "12.9.1"
    assert record["platform"]["wheel_tags"] == [
        "cp38-abi3-linux_x86_64"
    ]
    assert record["platform"]["filename_tag"] == (
        "cp38-abi3-manylinux_2_28_x86_64"
    )
    assert record["platform"]["packaged"] == {
        "vllm._C": "included",
        "vllm.cumem_allocator": "included",
        GRUG_ARCHITECTURE: "included",
    }
    wheel_path = tmp_path / wheel["filename"]
    assert wheel["sha256"] == sha256_file(wheel_path)
    assert wheel["size_bytes"] == wheel_path.stat().st_size


def test_missing_cumem_allocator_is_explicit_and_blocks_candidate(tmp_path):
    record = fragment(tmp_path, "aarch64", include_cumem=False)

    assert record["platform"]["packaged"]["vllm.cumem_allocator"] == "absent"
    with pytest.raises(ReleaseError, match="vllm.cumem_allocator.*absent"):
        validate_wheel_fragment(record)


def test_candidate_rejects_cross_arch_source_mismatch(tmp_path):
    config = load_json(CONFIG_PATH)
    x86 = fragment(tmp_path, "x86_64")
    arm = fragment(tmp_path, "aarch64")
    arm["source"]["upstream_base"] = "c" * 40

    with pytest.raises(ReleaseError, match="disagrees on source"):
        assemble_candidate(
            [x86, arm],
            config=config,
            repository="marin-community/vllm",
            candidate_tag=CANDIDATE_TAG,
            created_at=BUILT_AT,
        )


def test_validation_matrix_projects_iris_resources_from_release_config():
    config = load_json(CONFIG_PATH)

    matrix = validation_matrix(config)

    assert matrix == {
        "include": [
            {
                "architecture": architecture,
                "hardware": platform["validation"]["gpu"],
                "gpu_resource": platform["validation"]["gpu_resource"],
                "target_cluster": platform["validation"]["target_cluster"],
                "cpu": platform["validation"]["cpu"],
                "memory": platform["validation"]["memory"],
            }
            for architecture, platform in config["platforms"].items()
        ]
    }


def test_build_matrix_targets_only_the_validated_gpu():
    config = load_json(CONFIG_PATH)

    matrix = build_matrix(config)

    for item in matrix["include"]:
        platform = config["platforms"][item["architecture"]]
        assert item["sm_targets"] == platform["validation"]["compute_capability"]
        assert item["max_jobs"] == 1
        assert item["nvcc_threads"] == 1


def release_fixture(tmp_path: Path) -> tuple[dict, dict, list[dict], dict]:
    config = load_json(CONFIG_PATH)
    candidate_manifest = candidate(tmp_path)
    validations = [
        validation(candidate_manifest, architecture)
        for architecture in config["platforms"]
    ]
    manifest = finalize_release(
        candidate_manifest,
        validations,
        config=config,
        release_tag=f"marin-vllm-gpu-20260803-{FORK_COMMIT[:12]}",
        published_at="2026-08-04T00:00:00Z",
        provenance={"run_id": "456"},
    )
    return config, candidate_manifest, validations, manifest


def test_release_binds_passed_gpu_results_to_candidate_wheel_digests(tmp_path):
    _, _, _, manifest = release_fixture(tmp_path)

    assert manifest["release"]["status"] == "released"
    assert manifest["release"]["candidate_tag"] == CANDIDATE_TAG
    assert manifest["validation"]["status"] == "passed"
    assert {item["architecture"] for item in manifest["validation"]["targets"]} == {
        "x86_64",
        "aarch64",
    }
    for platform in manifest["platforms"]:
        assert f"/{manifest['release']['tag']}/" in platform["wheel"]["url"]


def test_release_rejects_allocator_absence_from_gpu_result(tmp_path):
    config, candidate_manifest, validations, _ = release_fixture(tmp_path)
    broken = copy.deepcopy(validations[0])
    broken["gates"]["cumem_allocator"] = {"status": "absent"}
    validations[0] = broken

    with pytest.raises(ReleaseError, match="cumem_allocator.*absent"):
        finalize_release(
            candidate_manifest,
            validations,
            config=config,
            release_tag=f"marin-vllm-gpu-20260803-{FORK_COMMIT[:12]}",
            published_at="2026-08-04T00:00:00Z",
            provenance={"run_id": "456"},
        )


def write_validation_assets(tmp_path: Path, validations: list[dict]) -> None:
    for result in validations:
        hardware = result["hardware"]["requested"].lower()
        path = tmp_path / f"marin-vllm-validation-{hardware}.json"
        path.write_text(json.dumps(result))


def test_final_release_verification_binds_urls_and_detached_results(tmp_path):
    config, _, validations, manifest = release_fixture(tmp_path)
    write_validation_assets(tmp_path, validations)

    verify_release_assets(manifest, tmp_path, config)

    broken = copy.deepcopy(manifest)
    broken["platforms"][0]["wheel"]["url"] = "https://example.com/mutable.whl"
    with pytest.raises(ReleaseError, match="URL is not tag-addressed"):
        verify_release_assets(broken, tmp_path, config)


def test_final_release_verification_rejects_changed_validation_asset(tmp_path):
    config, _, validations, manifest = release_fixture(tmp_path)
    write_validation_assets(tmp_path, validations)
    changed_path = tmp_path / "marin-vllm-validation-h100.json"
    changed_path.write_text(json.dumps({**validations[0], "result": "failed"}))

    with pytest.raises(ReleaseError, match="validation asset changed"):
        verify_release_assets(manifest, tmp_path, config)


def test_validation_log_extracts_machine_readable_result(tmp_path):
    expected = {"architecture": "aarch64", "result": "passed"}
    encoded = base64.b64encode(json.dumps(expected).encode()).decode()
    log_path = tmp_path / "validation.log"
    log_path.write_text(f"setup\nMARIN_GPU_VALIDATION_JSON={encoded}\ndone\n")

    assert extract_validation(log_path) == expected


def test_validation_log_records_missing_result_as_failure(tmp_path):
    log_path = tmp_path / "validation.log"
    log_path.write_text("Iris job exited before validation\n")

    result = extract_validation(log_path)

    assert result["result"] == "failed"
    assert result["failure"] == "Iris log did not contain a GPU validation record"


def test_wheel_tests_preload_installed_package_before_adding_checkout(tmp_path):
    site_packages = tmp_path / "site-packages"
    checkout = tmp_path / "checkout"
    (site_packages / "vllm").mkdir(parents=True)
    (site_packages / "vllm/__init__.py").write_text("ORIGIN = 'wheel'\n")
    (checkout / "vllm").mkdir(parents=True)
    (checkout / "vllm/__init__.py").write_text("ORIGIN = 'source'\n")
    (checkout / "tests").mkdir()
    (checkout / "tests/__init__.py").write_text("")
    test_path = checkout / "tests/test_import_origin.py"
    test_path.write_text(
        "import vllm\n\n"
        "def test_import_origin():\n"
        "    assert vllm.ORIGIN == 'wheel'\n"
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(site_packages)
    script = Path(__file__).parents[1] / "gpu_validation.py"

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "run-wheel-tests",
            "--source-root",
            str(checkout),
            "--",
            "-q",
            str(test_path),
        ],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
