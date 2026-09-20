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
import yaml

from infra.nightly.gpu_serve_smoke import server_command
from infra.release.gpu_release import (
    GRUG_ARCHITECTURE,
    SPARSE_NCCL_GATE,
    STABLE_LIBTORCH_GATE,
    assemble_candidate,
    build_matrix,
    extract_validation,
    finalize_release,
    inspect_wheel,
    validate_candidate,
    validate_wheel_fragment,
    validation_matrix,
    verify_main_lineage,
    verify_published_candidate,
    verify_release_assets,
)
from infra.release.release_common import ReleaseError, load_json, sha256_file

REPOSITORY_ROOT = Path(__file__).parents[3]
CONFIG_PATH = Path(__file__).parents[1] / "config.json"
GPU_CANDIDATE_WORKFLOW_PATH = (
    REPOSITORY_ROOT / ".github/workflows/marin-gpu-candidate.yaml"
)
GPU_RELEASE_WORKFLOW_PATH = (
    REPOSITORY_ROOT / ".github/workflows/marin-gpu-release.yaml"
)
FORK_COMMIT = "a" * 40
UPSTREAM_BASE = "b" * 40
BUILT_AT = "2026-08-03T12:00:00Z"
CANDIDATE_TAG = f"marin-vllm-gpu-candidate-{FORK_COMMIT[:12]}"


def test_publish_uses_current_release_automation_for_an_older_candidate():
    workflow = yaml.safe_load(GPU_RELEASE_WORKFLOW_PATH.read_text())
    checkout = workflow["jobs"]["publish"]["steps"][0]

    assert "ref" not in checkout.get("with", {})


def test_release_publishers_use_builtin_token_with_write_permission():
    workflow_paths = (
        GPU_CANDIDATE_WORKFLOW_PATH,
        GPU_RELEASE_WORKFLOW_PATH,
    )

    for workflow_path in workflow_paths:
        publish = yaml.safe_load(workflow_path.read_text())["jobs"]["publish"]
        assert publish["permissions"]["contents"] == "write"
        assert publish["env"]["GH_TOKEN"] == "${{ github.token }}"


def test_candidate_build_ignores_release_only_changes():
    workflow = yaml.load(
        GPU_CANDIDATE_WORKFLOW_PATH.read_text(), Loader=yaml.BaseLoader
    )
    ignored_paths = set(workflow["on"]["push"]["paths-ignore"])

    assert ignored_paths >= {
        ".github/workflows/marin-ci.yaml",
        ".github/workflows/marin-gpu-candidate.yaml",
        ".github/workflows/marin-gpu-release.yaml",
        "infra/release/gpu_validation.py",
        "infra/release/validation_common.py",
        "infra/release/tests/**",
    }


def test_candidate_gpu_modes_keep_single_architecture_builds_nonpublishing():
    workflow = yaml.load(
        GPU_CANDIDATE_WORKFLOW_PATH.read_text(), Loader=yaml.BaseLoader
    )
    inputs = workflow["on"]["workflow_dispatch"]["inputs"]
    publish = workflow["jobs"]["publish"]
    gpu_mode = inputs["gpu_mode"]

    assert gpu_mode["type"] == "choice"
    assert gpu_mode["default"] == "publish"
    assert gpu_mode["options"] == [
        "publish",
        "qualify-x86_64",
        "qualify-aarch64",
    ]
    assert publish["if"] == (
        "github.event_name == 'push' || inputs.gpu_mode == 'publish'"
    )


def test_server_command_pins_requested_attention_backend():
    command = server_command("Qwen/Qwen3-0.6B", 8000, "FLASH_ATTN")

    assert "--tensor-parallel-size" not in command
    assert command[-2:] == ["--attention-backend", "FLASH_ATTN"]


def test_server_command_binds_to_port_probe_interface():
    command = server_command("Qwen/Qwen3-0.6B", 8000, None)

    host_index = command.index("--host")
    assert command[host_index + 1] == "127.0.0.1"


def test_server_command_uses_requested_tensor_parallel_size():
    command = server_command("Qwen/Qwen3-0.6B", 8000, None, 8)

    assert command[-2:] == ["--tensor-parallel-size", "8"]


def write_wheel(
    path: Path,
    *,
    architecture: str,
    include_cumem: bool = True,
    version: str = "0.0.0.dev20260803+marin.test.cu130",
    metadata_platform_tag: str | None = None,
) -> None:
    if metadata_platform_tag is None:
        metadata_platform_tag = f"manylinux_2_28_{architecture}"
    dist_info = f"vllm-{version}.dist-info"
    metadata = (
        "Metadata-Version: 2.4\n"
        "Name: vllm\n"
        f"Version: {version}\n"
        "Requires-Python: >=3.10,<3.15\n"
        "Requires-Dist: torch==2.13.0\n"
        "Requires-Dist: transformers>=4.56.0\n"
        "\n"
    )
    wheel_metadata = (
        "Wheel-Version: 1.0\n"
        "Generator: test\n"
        "Root-Is-Purelib: false\n"
        f"Tag: cp38-abi3-{metadata_platform_tag}\n"
        "\n"
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(f"{dist_info}/METADATA", metadata)
        archive.writestr(f"{dist_info}/WHEEL", wheel_metadata)
        archive.writestr("vllm/_C_stable_libtorch.cpython-312-test.so", b"compiled")
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
    config: dict | None = None,
    metadata_platform_tag: str | None = None,
) -> dict:
    if config is None:
        config = load_json(CONFIG_PATH)
    wheel = tmp_path / (
        "vllm-0.0.0.dev20260803+marin.test.cu130-cp38-abi3-"
        f"manylinux_2_28_{architecture}.whl"
    )
    write_wheel(
        wheel,
        architecture=architecture,
        include_cumem=include_cumem,
        metadata_platform_tag=metadata_platform_tag,
    )
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


def test_published_candidate_rejects_changed_assets_and_target(tmp_path, monkeypatch):
    manifest = candidate(tmp_path)
    release = {
        "tag_name": CANDIDATE_TAG,
        "target_commitish": FORK_COMMIT,
        "draft": False,
        "prerelease": True,
        "assets": [{"name": "marin-vllm-gpu-manifest.json", "state": "uploaded"}],
    }
    for platform in manifest["platforms"]:
        wheel = platform["wheel"]
        release["assets"].append(
            {
                "name": wheel["filename"],
                "state": "uploaded",
                "size": wheel["size_bytes"],
                "digest": f"sha256:{wheel['sha256']}",
            }
        )

    def gh_api(args, **kwargs):
        return subprocess.CompletedProcess(
            args, 0, stdout=json.dumps(release), stderr=""
        )

    monkeypatch.setattr("infra.release.gpu_release.subprocess.run", gh_api)

    verify_published_candidate(manifest, "marin-community/vllm")
    release["assets"][1]["digest"] = "sha256:" + "0" * 64
    with pytest.raises(ReleaseError, match="wheel asset changed"):
        verify_published_candidate(manifest, "marin-community/vllm")
    release["assets"][1]["digest"] = (
        f"sha256:{manifest['platforms'][0]['wheel']['sha256']}"
    )
    release["target_commitish"] = "b" * 40
    with pytest.raises(ReleaseError, match="release identity changed"):
        verify_published_candidate(manifest, "marin-community/vllm")


@pytest.mark.parametrize("compare_status", ["identical", "ahead"])
def test_gpu_lineage_accepts_main_source_when_main_advances(
    monkeypatch, compare_status: str
):
    def gh_api(args, **kwargs):
        value = "main" if args[2] == "repos/marin-community/vllm" else compare_status
        return subprocess.CompletedProcess(args, 0, stdout=value, stderr="")

    monkeypatch.setattr("infra.release.gpu_release.subprocess.run", gh_api)

    # The same source can be current main or an ancestor after a long build.
    assert (
        verify_main_lineage(
            "marin-community/vllm", "refs/heads/main", FORK_COMMIT, CANDIDATE_TAG
        )
        is None
    )


def test_gpu_lineage_rejects_off_main_dispatch(monkeypatch):
    def gh_api(args, **kwargs):
        return subprocess.CompletedProcess(args, 0, stdout="main", stderr="")

    monkeypatch.setattr("infra.release.gpu_release.subprocess.run", gh_api)

    with pytest.raises(ReleaseError, match="workflow must run from refs/heads/main"):
        verify_main_lineage(
            "marin-community/vllm", "refs/heads/feature", FORK_COMMIT, CANDIDATE_TAG
        )


def test_gpu_lineage_rejects_diverged_candidate_with_context(monkeypatch):
    def gh_api(args, **kwargs):
        value = "main" if args[2] == "repos/marin-community/vllm" else "diverged"
        return subprocess.CompletedProcess(args, 0, stdout=value, stderr="")

    monkeypatch.setattr("infra.release.gpu_release.subprocess.run", gh_api)

    with pytest.raises(ReleaseError, match="source is not an ancestor of main") as exc:
        verify_main_lineage(
            "marin-community/vllm", "refs/heads/main", FORK_COMMIT, CANDIDATE_TAG
        )
    assert CANDIDATE_TAG in str(exc.value)
    assert FORK_COMMIT in str(exc.value)
    assert "workflow_ref=refs/heads/main" in str(exc.value)
    assert "expected_default_branch=main" in str(exc.value)


def test_candidate_rejects_abi_change(tmp_path):
    manifest = candidate(tmp_path)
    manifest["abi"]["torch_version"] = "0.0.0"

    with pytest.raises(ReleaseError, match="release ABI changed"):
        validate_candidate(manifest, load_json(CONFIG_PATH))


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
            "attention_backend": validation_config["attention_backend"],
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
            "torchaudio.resample": {"status": "passed", "max_error": 0.0},
            STABLE_LIBTORCH_GATE: {"status": "passed"},
            GRUG_ARCHITECTURE: {"status": "passed"},
            SPARSE_NCCL_GATE: {
                "status": "passed",
                "backend": "sparse_nccl",
                "checkpoint_shape": [2, 2],
                "patch_entries": 2,
            },
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
        "version": "0.0.0.dev20260803+marin.test.cu130",
        "requires_python": ">=3.10,<3.15",
        "torch_metadata_requirement": "torch==2.13.0",
    }
    assert record["source"]["fork_commit"] == FORK_COMMIT
    assert record["source"]["upstream_base"] == UPSTREAM_BASE
    assert record["platform"]["wheel_tags"] == [
        "cp38-abi3-manylinux_2_28_x86_64"
    ]
    assert record["platform"]["filename_tag"] == (
        "cp38-abi3-manylinux_2_28_x86_64"
    )
    assert record["platform"]["packaged"] == {
        STABLE_LIBTORCH_GATE: "included",
        "vllm.cumem_allocator": "included",
        GRUG_ARCHITECTURE: "included",
    }
    wheel_path = tmp_path / wheel["filename"]
    assert wheel["sha256"] == sha256_file(wheel_path)
    assert wheel["size_bytes"] == wheel_path.stat().st_size


def test_inspect_wheel_cli_runs_without_site_packages(tmp_path):
    config = load_json(CONFIG_PATH)
    architecture = "x86_64"
    platform = config["platforms"][architecture]
    wheel = tmp_path / (
        "vllm-0.0.0.dev20260803+marin.test.cu130-cp38-abi3-"
        f"manylinux_2_28_{architecture}.whl"
    )
    output = tmp_path / "fragment.json"
    write_wheel(wheel, architecture=architecture)

    completed = subprocess.run(
        [
            sys.executable,
            "-E",
            "-S",
            str(REPOSITORY_ROOT / "infra/release/gpu_release.py"),
            "inspect-wheel",
            "--config",
            str(CONFIG_PATH),
            "--wheel",
            str(wheel),
            "--architecture",
            architecture,
            "--fork-commit",
            FORK_COMMIT,
            "--upstream-base",
            UPSTREAM_BASE,
            "--built-at",
            BUILT_AT,
            "--base-image",
            platform["build_base_image"],
            "--base-image-digest",
            platform["build_base_image"],
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert load_json(output)["platform"]["wheel"]["filename"] == wheel.name


def test_inspect_wheel_rejects_filename_metadata_tag_mismatch(tmp_path):
    with pytest.raises(ReleaseError, match="do not match filename tag"):
        fragment(tmp_path, "x86_64", metadata_platform_tag="linux_x86_64")


def test_missing_cumem_allocator_is_explicit_and_blocks_candidate(tmp_path):
    record = fragment(tmp_path, "x86_64", include_cumem=False)

    assert record["platform"]["packaged"]["vllm.cumem_allocator"] == "absent"
    with pytest.raises(ReleaseError, match="vllm.cumem_allocator.*absent"):
        validate_wheel_fragment(record)


def test_candidate_rejects_cross_arch_source_mismatch(tmp_path):
    config = load_json(CONFIG_PATH)
    config["platforms"]["aarch64"] = copy.deepcopy(config["platforms"]["x86_64"])
    x86 = fragment(tmp_path, "x86_64")
    arm = fragment(tmp_path, "aarch64", config=config)
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

    assert {item["architecture"] for item in matrix["include"]} == {
        "x86_64",
        "aarch64",
    }
    for item in matrix["include"]:
        platform = config["platforms"][item["architecture"]]
        assert item["sm_targets"] == platform["validation"]["compute_capability"]
        assert item["max_jobs"] == 2
        assert item["max_wheel_size_mb"] == 800
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
    config, _, _, manifest = release_fixture(tmp_path)

    assert manifest["release"]["status"] == "released"
    assert manifest["release"]["candidate_tag"] == CANDIDATE_TAG
    assert manifest["validation"]["status"] == "passed"
    assert {item["architecture"] for item in manifest["validation"]["targets"]} == {
        *config["platforms"],
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


def test_release_rejects_missing_sparse_nccl_contract(tmp_path):
    config, candidate_manifest, validations, _ = release_fixture(tmp_path)
    broken = copy.deepcopy(validations[0])
    broken["gates"][SPARSE_NCCL_GATE] = {"status": "failed"}
    validations[0] = broken

    with pytest.raises(ReleaseError, match="sparse_nccl_contract.*failed"):
        finalize_release(
            candidate_manifest,
            validations,
            config=config,
            release_tag=f"marin-vllm-gpu-20260803-{FORK_COMMIT[:12]}",
            published_at="2026-08-04T00:00:00Z",
            provenance={"run_id": "456"},
        )


def test_release_rejects_wrong_serving_attention_backend(tmp_path):
    config, candidate_manifest, validations, _ = release_fixture(tmp_path)
    broken = copy.deepcopy(validations[0])
    broken["environment"]["attention_backend"] = "FLASHINFER"
    validations[0] = broken

    with pytest.raises(ReleaseError, match="attention_backend='FLASHINFER'"):
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
    log_path.write_text(
        "setup\n"
        f"Iris task=/vllm-ci/release/0 | MARIN_GPU_VALIDATION_JSON={encoded}\n"
        "done\n"
    )

    assert extract_validation(log_path) == expected


def test_validation_log_records_missing_result_as_failure(tmp_path):
    log_path = tmp_path / "validation.log"
    log_path.write_text("Iris job exited before validation\n")

    result = extract_validation(log_path)

    assert result["result"] == "failed"
    assert result["failure"] == "Iris log did not contain a GPU validation record"


def test_wheel_tests_use_installed_package_and_apply_exclusions(tmp_path):
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
        "import subprocess\n"
        "import sys\n\n"
        "import vllm\n\n"
        "def test_import_origin():\n"
        "    assert vllm.ORIGIN == 'wheel'\n"
        "    child = subprocess.run(\n"
        "        [sys.executable, '-c', 'import vllm; print(vllm.ORIGIN)'],\n"
        "        text=True, capture_output=True, check=True,\n"
        "    )\n"
        "    assert child.stdout.strip() == 'wheel'\n"
        "\n"
        "def test_excluded_failure():\n"
        "    assert False\n"
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(site_packages)
    script = Path(__file__).parents[1] / "gpu_validation.py"

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "run-wheel-tests",
            "--package-source-root",
            str(checkout),
            "--validation-source-root",
            str(checkout),
            "--exclude-test",
            "test_excluded_failure",
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
