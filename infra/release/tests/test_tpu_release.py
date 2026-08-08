# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
import zipfile
from pathlib import Path

import pytest

from infra.release.gpu_release import ReleaseError, load_json, sha256_file
from infra.release.tpu_release import (
    assemble_candidate,
    candidate_tag,
    finalize_release,
    release_tag,
    validate_candidate,
    validate_release,
    verify_assets,
)

CONFIG_PATH = Path(__file__).parents[1] / "tpu_config.json"
WORKFLOW_PATH = Path(__file__).parents[3] / ".github/workflows/marin-gpu-candidate.yaml"
WORKFLOW_COMMIT = "c" * 40
CREATED_AT = "2026-08-08T00:30:00Z"
RUN_URL = "https://github.com/marin-community/vllm/actions/runs/123"


def test_tpu_wheel_build_uses_the_repository_cpu_torch_environment():
    workflow = WORKFLOW_PATH.read_text()

    assert "--requirements requirements/build/tpu.txt" in workflow
    assert "--index-strategy unsafe-best-match" in workflow
    assert "--no-build-isolation" in workflow


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
        requirements += f"Requires-Dist: {name}=={required_version}\n"
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
    config = load_json(CONFIG_PATH)
    vllm = tmp_path / "vllm-0.20.1-py3-none-any.whl"
    tpu = tmp_path / "tpu_inference-0.23.0-py3-none-any.whl"
    _write_wheel(vllm, "vllm", "0.20.1")
    _write_wheel(
        tpu,
        "tpu-inference",
        "0.23.0",
        runtime_requirements=config["runtime_requirements"],
    )
    return [vllm, tpu]


def _candidate(tmp_path: Path) -> dict:
    return assemble_candidate(
        _wheels(tmp_path),
        config=load_json(CONFIG_PATH),
        repository="marin-community/vllm",
        workflow_commit=WORKFLOW_COMMIT,
        workflow_run_url=RUN_URL,
        created_at=CREATED_AT,
    )


def test_candidate_binds_both_sources_workflow_and_wheel_bytes(tmp_path: Path):
    config = load_json(CONFIG_PATH)
    manifest = _candidate(tmp_path)

    validate_candidate(manifest, config)
    verify_assets(manifest, tmp_path)
    assert manifest["release"]["tag"] == candidate_tag(config, WORKFLOW_COMMIT)
    assert manifest["workflow"] == {
        "commit": WORKFLOW_COMMIT,
        "run_url": RUN_URL,
    }
    for package in manifest["packages"]:
        expected = config["packages"][package["distribution"]]
        assert package["source_commit"] == expected["source_commit"]
        assert package["repository"] == expected["repository"]
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
        "0.23.0",
        runtime_requirements={"jax": "0.10.2", "jaxlib": "0.10.1", "libtpu": "0.0.41"},
    )

    with pytest.raises(ReleaseError, match="jax==0.10.1"):
        assemble_candidate(
            wheels,
            config=config,
            repository="marin-community/vllm",
            workflow_commit=WORKFLOW_COMMIT,
            workflow_run_url=RUN_URL,
            created_at=CREATED_AT,
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


def test_promotion_reuses_candidate_bytes_and_changes_only_release_labels(
    tmp_path: Path,
):
    config = load_json(CONFIG_PATH)
    candidate = _candidate(tmp_path)
    tag = release_tag(candidate)
    promoted = finalize_release(
        candidate,
        config=config,
        tag=tag,
        published_at="2026-08-09T00:00:00Z",
    )

    validate_release(promoted, config)
    verify_assets(promoted, tmp_path)
    assert promoted["source"] == candidate["source"]
    assert promoted["workflow"] == candidate["workflow"]
    assert promoted["compatibility"] == candidate["compatibility"]
    for before, after in zip(candidate["packages"], promoted["packages"], strict=True):
        assert after["distribution"] == before["distribution"]
        assert after["version"] == before["version"]
        assert after["source_commit"] == before["source_commit"]
        assert after["wheel"]["filename"] == before["wheel"]["filename"]
        assert after["wheel"]["sha256"] == before["wheel"]["sha256"]
        assert after["wheel"]["url"] != before["wheel"]["url"]
        assert tag in after["wheel"]["url"]
