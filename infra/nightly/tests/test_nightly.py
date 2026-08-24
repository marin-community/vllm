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
