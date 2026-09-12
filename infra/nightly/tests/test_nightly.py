# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import subprocess
import sys
from pathlib import Path

import yaml
from packaging.version import Version

REPO_ROOT = Path(__file__).parents[3]
RESOLVE_VERSION = REPO_ROOT / "infra/nightly/resolve_version.py"
RUN_H100 = REPO_ROOT / "infra/nightly/run_h100.sh"
NIGHTLY_WORKFLOW = REPO_ROOT / ".github/workflows/marin-nightly.yaml"


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=test", "-c", "user.email=test@example.com", *args],
        cwd=repo,
        check=True,
    )


def test_version_resolution_ignores_marin_release_tags(tmp_path: Path) -> None:
    _git(tmp_path, "init")
    _git(tmp_path, "commit", "--allow-empty", "-m", "base")
    _git(tmp_path, "tag", "v1.2.3")
    _git(tmp_path, "commit", "--allow-empty", "-m", "head")
    _git(tmp_path, "tag", "marin-vllm-gpu-20260815-a12602971f08")

    output = subprocess.check_output(
        [sys.executable, str(RESOLVE_VERSION)],
        cwd=tmp_path,
        text=True,
    )
    version = Version(output.strip())

    assert (version.release, version.dev) == ((1, 2, 4), 1)


def test_iris_failure_remains_nonzero_through_log_capture(tmp_path: Path) -> None:
    workflow = yaml.safe_load(NIGHTLY_WORKFLOW.read_text())
    steps = workflow["jobs"]["serve-smoke"]["steps"]
    iris_step = next(step for step in steps if step.get("id") == "iris")
    script = 'iris() { echo "current Iris run"; return 23; }\n' + iris_step["run"]

    completed = subprocess.run(["bash", "-e", "-c", script], cwd=tmp_path)

    assert completed.returncode == 23
    assert (tmp_path / "nightly.log").read_text() == "current Iris run\n"


def test_h100_test_dependencies_follow_cuda_constraints(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    uv_calls = tmp_path / "uv-calls"
    fake_uv = bin_dir / "uv"
    fake_uv.write_text(
        '#!/usr/bin/env bash\nprintf "%s\\n" "$@" >> "$UV_CALLS"\n'
        'printf "%s\\n" __CALL_END__ >> "$UV_CALLS"\n'
    )
    fake_uv.chmod(0o755)

    python_path = tmp_path / ".venv/bin/python"
    python_path.parent.mkdir(parents=True)
    python_path.write_text("#!/usr/bin/env bash\nexit 0\n")
    python_path.chmod(0o755)

    env = {
        **os.environ,
        "IRIS_WORKDIR": str(tmp_path),
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "UV_CALLS": str(uv_calls),
    }
    subprocess.run(["bash", str(RUN_H100)], env=env, check=True)

    calls = [
        call.splitlines()
        for call in uv_calls.read_text().split("__CALL_END__\n")
        if call
    ]
    assert [
        "pip",
        "install",
        "pytest",
        "tblib",
        "transformers",
        "--constraint",
        "requirements/test/cuda.txt",
    ] in calls
