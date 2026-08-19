# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
import zipfile
from pathlib import Path

import pytest

from infra.release.release_common import ReleaseError, load_json, sha256_file
from infra.release.tpu_release import (
    assemble_candidate,
    candidate_tag,
    finalize_release,
    release_tag,
    validate_candidate,
    validate_release,
    verify_assets,
    write_index,
)

CONFIG_PATH = Path(__file__).parents[1] / "tpu_config.json"
WORKFLOW_COMMIT = "c" * 40
VLLM_COMMIT = "a" * 40
TPU_INFERENCE_COMMIT = "b" * 40
EXCLUDE_NEWER = "2026-08-12T00:00:00Z"
CREATED_AT = "2026-08-08T00:30:00Z"
RUN_URL = "https://github.com/marin-community/vllm/actions/runs/123"
VLLM_VERSION = "0.20.1rc1.dev0+marin.aaaaaaaaaaaa.tpu"
TPU_INFERENCE_VERSION = "0.26.0+marin.bbbbbbbbbbbb"


def _write_wheel(
    path: Path,
    distribution: str,
    version: str,
    *,
    runtime_requirements: dict[str, str] | None = None,
) -> None:
    module_name = distribution.replace("-", "_")
    dist_info = f"{module_name}-{version}.dist-info"
    requirements = ""
    for name, required_version in (runtime_requirements or {}).items():
        specifier = (
            required_version
            if required_version.startswith(("<", ">", "=", "!", "~"))
            else f"=={required_version}"
        )
        requirements += f"Requires-Dist: {name}{specifier}\n"
    metadata = (
        "Metadata-Version: 2.4\n"
        f"Name: {distribution}\n"
        f"Version: {version}\n"
        "Requires-Python: >=3.10\n"
        f"{requirements}\n"
    )
    wheel_metadata = (
        "Wheel-Version: 1.0\n"
        "Generator: test\n"
        "Root-Is-Purelib: true\n"
        "Tag: py3-none-any\n\n"
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(f"{dist_info}/METADATA", metadata)
        archive.writestr(f"{dist_info}/WHEEL", wheel_metadata)
        archive.writestr(f"{module_name}/__init__.py", "")


def _wheels(tmp_path: Path) -> list[Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    vllm = tmp_path / f"vllm-{VLLM_VERSION}-py3-none-any.whl"
    tpu = tmp_path / f"tpu_inference-{TPU_INFERENCE_VERSION}-py3-none-any.whl"
    _write_wheel(vllm, "vllm", VLLM_VERSION)
    _write_wheel(
        tpu,
        "tpu-inference",
        TPU_INFERENCE_VERSION,
        runtime_requirements={
            "jax": "0.10.2",
            "jaxlib": "0.10.2",
            "libtpu": "0.0.43",
        },
    )
    return [vllm, tpu]


def _candidate(tmp_path: Path) -> dict:
    candidate = assemble_candidate(
        _wheels(tmp_path),
        config=load_json(CONFIG_PATH),
        repository="marin-community/vllm",
        vllm_commit=VLLM_COMMIT,
        tpu_inference_commit=TPU_INFERENCE_COMMIT,
        workflow_commit=WORKFLOW_COMMIT,
        workflow_run_url=RUN_URL,
        created_at=CREATED_AT,
        exclude_newer=EXCLUDE_NEWER,
    )
    write_index(candidate, tmp_path)
    return candidate


def _validation(candidate: dict) -> dict:
    return {
        "schema_version": 1,
        "candidate_tag": candidate["release"]["tag"],
        "source": candidate["source"],
        "workflow": candidate["workflow"],
        "qualification": {
            "run_url": "https://github.com/marin-community/vllm/actions/runs/456"
        },
        "index": candidate["index"],
        "wheels": {
            package["distribution"]: copy.deepcopy(package["wheel"])
            for package in candidate["packages"]
        },
        "hardware": {"selected": "v6e-8"},
        "gates": {
            "wheel_sha256": {"status": "passed"},
            "clean_install": {"status": "passed"},
            "serve_smoke": {"status": "passed"},
        },
        "result": "passed",
    }


def test_candidate_binds_both_sources_workflow_and_wheel_bytes(tmp_path: Path):
    config = load_json(CONFIG_PATH)
    manifest = _candidate(tmp_path)

    validate_candidate(manifest, config)
    verify_assets(manifest, tmp_path)
    assert manifest["release"]["tag"] == candidate_tag(
        config,
        manifest["source"],
        WORKFLOW_COMMIT,
        EXCLUDE_NEWER,
        {
            package["distribution"]: package["wheel"]["sha256"]
            for package in manifest["packages"]
        },
    )
    assert manifest["workflow"] == {
        "commit": WORKFLOW_COMMIT,
        "run_url": RUN_URL,
    }
    assert manifest["compatibility"]["runtime_requirements"] == {
        "jax": "0.10.2",
        "jaxlib": "0.10.2",
        "libtpu": "0.0.43",
    }
    index = manifest["index"]
    index_text = (tmp_path / index["filename"]).read_text()
    assert index["sha256"] == sha256_file(tmp_path / index["filename"])
    assert index_text.count("#sha256=") == 2
    for package in manifest["packages"]:
        expected_repository = config["repositories"][package["distribution"]]
        expected_commit = {
            "vllm": VLLM_COMMIT,
            "tpu-inference": TPU_INFERENCE_COMMIT,
        }[package["distribution"]]
        assert package["source_commit"] == expected_commit
        assert package["repository"] == expected_repository
        assert package["wheel"]["sha256"] == sha256_file(
            tmp_path / package["wheel"]["filename"]
        )
        assert manifest["release"]["tag"] in package["wheel"]["url"]


def test_candidate_rejects_unpinned_tpu_runtime(tmp_path: Path):
    config = load_json(CONFIG_PATH)
    wheels = _wheels(tmp_path)
    _write_wheel(
        wheels[1],
        "tpu-inference",
        TPU_INFERENCE_VERSION,
        runtime_requirements={
            "jax": "0.10.2",
            "jaxlib": ">=0.10.1",
            "libtpu": "0.0.41",
        },
    )

    with pytest.raises(ReleaseError, match="pin jaxlib with =="):
        assemble_candidate(
            wheels,
            config=config,
            repository="marin-community/vllm",
            vllm_commit=VLLM_COMMIT,
            tpu_inference_commit=TPU_INFERENCE_COMMIT,
            workflow_commit=WORKFLOW_COMMIT,
            workflow_run_url=RUN_URL,
            created_at=CREATED_AT,
            exclude_newer=EXCLUDE_NEWER,
        )


def test_candidate_rejects_source_or_asset_drift(tmp_path: Path):
    config = load_json(CONFIG_PATH)
    manifest = _candidate(tmp_path)
    changed_source = copy.deepcopy(manifest)
    changed_source["packages"][0]["source_commit"] = "d" * 40

    with pytest.raises(ReleaseError, match="source commit changed"):
        validate_candidate(changed_source, config)

    (tmp_path / manifest["packages"][0]["wheel"]["filename"]).write_bytes(b"changed")
    with pytest.raises(ReleaseError, match="SHA-256 mismatch"):
        verify_assets(manifest, tmp_path)


def test_candidate_rejects_requested_tag_or_extra_wheel(tmp_path: Path):
    config = load_json(CONFIG_PATH)
    manifest = _candidate(tmp_path)

    with pytest.raises(ReleaseError, match="requested release"):
        validate_candidate(manifest, config, expected_tag="other-candidate")

    extra = tmp_path / "extra-1.0-py3-none-any.whl"
    _write_wheel(extra, "extra", "1.0")
    with pytest.raises(ReleaseError, match="wheel set changed"):
        verify_assets(manifest, tmp_path)


def test_candidate_identity_changes_with_build_cutoff(tmp_path: Path):
    config = load_json(CONFIG_PATH)
    manifest = _candidate(tmp_path)

    changed = candidate_tag(
        config,
        manifest["source"],
        WORKFLOW_COMMIT,
        "2026-08-13T00:00:00Z",
        {
            package["distribution"]: package["wheel"]["sha256"]
            for package in manifest["packages"]
        },
    )

    assert changed != manifest["release"]["tag"]


def test_candidate_identity_changes_with_wheel_bytes(tmp_path: Path):
    before = _candidate(tmp_path / "before")
    wheels = _wheels(tmp_path / "after")
    with zipfile.ZipFile(wheels[0], "a") as archive:
        archive.writestr("vllm/build-marker", "different build")

    after = assemble_candidate(
        wheels,
        config=load_json(CONFIG_PATH),
        repository="marin-community/vllm",
        vllm_commit=VLLM_COMMIT,
        tpu_inference_commit=TPU_INFERENCE_COMMIT,
        workflow_commit=WORKFLOW_COMMIT,
        workflow_run_url=RUN_URL,
        created_at=CREATED_AT,
        exclude_newer=EXCLUDE_NEWER,
    )

    assert after["release"]["tag"] != before["release"]["tag"]


def test_candidate_rejects_a_wheel_with_the_wrong_source_version(tmp_path: Path):
    config = load_json(CONFIG_PATH)
    wheels = _wheels(tmp_path)
    _write_wheel(wheels[0], "vllm", "0.20.1rc1.dev0+marin.wrong.tpu")

    with pytest.raises(ReleaseError, match="vllm wheel version"):
        assemble_candidate(
            wheels,
            config=config,
            repository="marin-community/vllm",
            vllm_commit=VLLM_COMMIT,
            tpu_inference_commit=TPU_INFERENCE_COMMIT,
            workflow_commit=WORKFLOW_COMMIT,
            workflow_run_url=RUN_URL,
            created_at=CREATED_AT,
            exclude_newer=EXCLUDE_NEWER,
        )


def test_promotion_reuses_candidate_bytes_and_changes_only_release_labels(
    tmp_path: Path,
):
    config = load_json(CONFIG_PATH)
    candidate = _candidate(tmp_path)
    tag = release_tag(candidate)
    promoted = finalize_release(
        candidate,
        config=config,
        validation=_validation(candidate),
        tag=tag,
        published_at="2026-08-09T00:00:00Z",
    )
    (tmp_path / candidate["index"]["filename"]).unlink()
    write_index(promoted, tmp_path)

    validate_release(promoted, config)
    verify_assets(promoted, tmp_path)
    assert promoted["source"] == candidate["source"]
    assert promoted["workflow"] == candidate["workflow"]
    assert promoted["compatibility"] == candidate["compatibility"]
    assert promoted["validation"] == _validation(candidate)
    assert promoted["index"] != candidate["index"]
    for before, after in zip(candidate["packages"], promoted["packages"], strict=True):
        assert after["distribution"] == before["distribution"]
        assert after["version"] == before["version"]
        assert after["source_commit"] == before["source_commit"]
        assert after["wheel"]["filename"] == before["wheel"]["filename"]
        assert after["wheel"]["sha256"] == before["wheel"]["sha256"]
        assert after["wheel"]["url"] != before["wheel"]["url"]
        assert tag in after["wheel"]["url"]


def test_promotion_rejects_qualification_for_different_bytes(tmp_path: Path):
    config = load_json(CONFIG_PATH)
    candidate = _candidate(tmp_path)
    validation = _validation(candidate)
    validation["wheels"]["vllm"]["sha256"] = "d" * 64

    with pytest.raises(ReleaseError, match="qualification wheel pair changed"):
        finalize_release(
            candidate,
            config=config,
            validation=validation,
            tag=release_tag(candidate),
            published_at="2026-08-09T00:00:00Z",
        )
