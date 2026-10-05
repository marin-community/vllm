"""Regression coverage for cold prefixes and unavailable cache transport."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from infra.release.compile_cache import transfer


@pytest.fixture
def gcloud_boundary(tmp_path, monkeypatch):
    executable = tmp_path / "bin" / "gcloud"
    executable.parent.mkdir()
    executable.write_text(
        "#!/bin/sh\n"
        'if [ "$2" = objects ]; then echo "[]"; exit 0; fi\n'
        "echo 'Did not find existing container' >&2\n"
        "exit 1\n"
    )
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", f"{executable.parent}:{os.environ['PATH']}")
    return executable


def test_empty_remote_prefix_records_cold_cache_without_transfer_error(
    tmp_path, gcloud_boundary,
):
    output = tmp_path / "evidence.json"
    transfer("restore", "cache", "test", "aarch64", tmp_path / "cache", output)
    records = json.loads(output.read_text())
    assert {record["stage"] for record in records} == {"cuda", "rust"}
    assert all(record["status"] == "complete" for record in records)
    assert all(record["payload_bytes"] == 0 for record in records)


def test_cli_cache_outage_preserves_local_outputs_and_records_failure(
    tmp_path, gcloud_boundary,
):
    gcloud_boundary.write_text("#!/bin/sh\nexit 1\n")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"bucket": "cache", "namespace": "test"}))
    output = tmp_path / "evidence.json"
    local = tmp_path / "cache"
    (local / "cuda").mkdir(parents=True)
    seed = local / "cuda" / "object"
    seed.write_bytes(b"previous compiler output")
    script = Path(__file__).parents[1] / "compile_cache.py"
    result = subprocess.run([
        sys.executable, str(script), "restore", "--config", str(config),
        "--architecture", "x86_64", "--directory", str(local),
        "--output", str(output),
    ], capture_output=True)
    assert result.returncode != 0
    assert seed.read_bytes() == b"previous compiler output"
    assert json.loads(output.read_text())[0]["status"] == "failed"
