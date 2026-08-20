# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import zipfile
from pathlib import Path

import pytest

from infra.release.release_common import ReleaseError, load_json
from infra.release.tpu_release import (
    assemble_candidate,
    finalize_release,
    release_tag,
    validate_release,
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
) -> None:
    module_name = distribution.replace("-", "_")
    dist_info = f"{module_name}-{version}.dist-info"
    metadata = (
        "Metadata-Version: 2.4\n"
        f"Name: {distribution}\n"
        f"Version: {version}\n"
        "Requires-Python: >=3.10\n"
        "\n"
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
    _write_wheel(tpu, "tpu-inference", TPU_INFERENCE_VERSION)
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
    return candidate


def _validation(candidate: dict) -> dict:
    return {
        "candidate_tag": candidate["release"]["tag"],
        "hardware": "v6e-8",
        "run_url": "https://github.com/marin-community/vllm/actions/runs/456",
    }


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


def test_promotion_reuses_candidate_bytes(tmp_path: Path):
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

    validate_release(promoted, config)
    assert {
        package["distribution"]: package["wheel"]["sha256"]
        for package in promoted["packages"]
    } == {
        package["distribution"]: package["wheel"]["sha256"]
        for package in candidate["packages"]
    }


def test_promotion_rejects_qualification_for_different_candidate(tmp_path: Path):
    config = load_json(CONFIG_PATH)
    candidate = _candidate(tmp_path)
    validation = _validation(candidate)
    validation["candidate_tag"] = "marin-vllm-tpu-candidate-other"

    with pytest.raises(ReleaseError, match="qualification candidate tag changed"):
        finalize_release(
            candidate,
            config=config,
            validation=validation,
            tag=release_tag(candidate),
            published_at="2026-08-09T00:00:00Z",
        )
