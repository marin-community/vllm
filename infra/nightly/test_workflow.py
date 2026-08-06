# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = REPOSITORY_ROOT / ".github/workflows/marin-nightly.yaml"


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
    command = workflow_step("Serve and gate on one H100")["run"]

    assert command.splitlines()[0] == "set -o pipefail"
    assert "2>&1 | tee nightly.log" in command
