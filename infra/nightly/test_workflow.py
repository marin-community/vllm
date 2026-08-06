# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = REPOSITORY_ROOT / ".github/workflows/marin-nightly.yaml"
RUN_GPU_PATH = REPOSITORY_ROOT / "infra/nightly/run_gpu.sh"


def workflow_step(name: str) -> dict[str, str]:
    workflow = yaml.safe_load(WORKFLOW_PATH.read_text())
    steps = workflow["jobs"]["serve-smoke"]["steps"]
    return next(step for step in steps if step.get("name") == name)


def test_bundle_version_does_not_depend_on_release_tag_parsing():
    command = workflow_step(
        "Resolve the upstream base commit and the package version"
    )["run"]

    assert "SETUPTOOLS_SCM_PRETEND_VERSION=0.0.dev0+g$short_sha" in command
    assert "uvx --from setuptools-scm" not in command


def test_iris_pipeline_propagates_job_failure():
    command = workflow_step("Serve and gate on one accelerator")["run"]

    assert command.splitlines()[0] == "set -o pipefail"
    assert "2>&1 | tee nightly.log" in command


def test_accelerator_lanes_match_release_validation_contracts():
    command = workflow_step("Resolve the accelerator lane")["run"]

    for expected in (
        "gpu_resource=H100x1",
        "target_cluster=cw-rno2a",
        "spec=infra/nightly/specs/qwen3-0.6b-h100.json",
        "gpu_resource=GB200x1",
        "target_cluster=cw-us-east-08a",
        "spec=infra/nightly/specs/qwen3-0.6b-gb200.json",
    ):
        assert expected in command


def test_gpu_runner_covers_required_delta_contracts_and_selected_backend():
    command = RUN_GPU_PATH.read_text()

    for test_target in (
        "tests/models/test_grugmoe.py",
        "tests/v1/core/test_scheduler.py",
        "tests/engine/test_arg_utils.py::TestDpDeviceIdSharding",
        "tests/distributed/test_mq_connect_ip.py::test_mq_bind_with_local_ip",
        "tests/model_executor/test_routed_experts_capture.py",
        "tests/v1/executor/test_ray_utils.py",
    ):
        assert test_target in command
    assert '--attention-backend "$ATTENTION_BACKEND"' in command
