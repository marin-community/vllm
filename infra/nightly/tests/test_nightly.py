import os
import subprocess
import sys
from pathlib import Path

import yaml
from packaging.version import Version

REPO_ROOT = Path(__file__).parents[3]
RESOLVE_VERSION = REPO_ROOT / "infra/nightly/resolve_version.py"
NIGHTLY_WORKFLOW = REPO_ROOT / ".github/workflows/marin-nightly.yaml"


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Nightly Test",
            "-c",
            "user.email=nightly-test@example.com",
            *args,
        ],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )


def test_version_resolution_ignores_marin_release_tags(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    (repo / "source.txt").write_text("base\n")
    _git(repo, "add", "source.txt")
    _git(repo, "commit", "-m", "base")
    _git(repo, "tag", "v1.2.3")

    (repo / "source.txt").write_text("head\n")
    _git(repo, "commit", "-am", "head")
    _git(repo, "tag", "marin-vllm-gpu-20260815-a12602971f08")
    _git(repo, "tag", "marin-vllm-gpu-candidate-deadbeef")

    completed = subprocess.run(
        [sys.executable, str(RESOLVE_VERSION)],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    version = Version(completed.stdout.strip())

    assert version.release == (1, 2, 4)
    assert version.dev == 1


def test_iris_failure_remains_nonzero_through_log_capture(tmp_path: Path) -> None:
    workflow = yaml.safe_load(NIGHTLY_WORKFLOW.read_text())
    iris_steps = [
        step
        for step in workflow["jobs"]["serve-smoke"]["steps"]
        if step.get("id") == "iris"
    ]
    [iris_step] = iris_steps

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_iris = fake_bin / "iris"
    fake_iris.write_text(
        "#!/usr/bin/env bash\n"
        "printf 'current Iris run\\n'\n"
        "exit 23\n"
    )
    fake_iris.chmod(0o755)

    tested_sha = "0123456789abcdef0123456789abcdef01234567"
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}{os.pathsep}{env['PATH']}",
            "TARGET_CLUSTER": "cw-rno2a",
            "JOB_NAME": f"vllm-123-1-{tested_sha}",
            "JOB_USER": "vllm-ci",
            "MODEL": "Qwen/Qwen3-0.6B",
            "VLLM_PRECOMPILED_WHEEL_COMMIT": tested_sha,
            "SETUPTOOLS_SCM_PRETEND_VERSION": "1.2.4.dev1",
            "GITHUB_SHA": tested_sha,
        }
    )

    completed = subprocess.run(
        ["bash", "-e", "-c", iris_step["run"]],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 23
    assert (tmp_path / "nightly.log").read_text() == "current Iris run\n"
