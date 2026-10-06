"""Regression coverage for cold prefixes and unavailable cache transport."""

import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
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


@pytest.fixture
def gcloud_objects(tmp_path, monkeypatch):
    """File-backed object storage at the real gcloud subprocess boundary."""
    cloud = tmp_path / "cloud"
    executable = tmp_path / "bin" / "gcloud"
    executable.parent.mkdir()
    executable.write_text(f"#!{sys.executable}\n" + '''
import json,os,shutil,sys
from pathlib import Path
root=Path(os.environ["TEST_CACHE_CLOUD"])
def path(url):
    return root/url.removeprefix("gs://").split("/",1)[1].removesuffix("**")
if sys.argv[2:4]==["objects","list"]:
    prefix=path(sys.argv[4])
    files=prefix.rglob("*") if prefix.exists() else []
    print(json.dumps([{"name":str(p.relative_to(root)),"size":p.stat().st_size}
                      for p in files if p.is_file()]))
else:
    source,destination=sys.argv[3:5]
    source=path(source) if source.startswith("gs://") else Path(source)
    destination=(path(destination) if destination.startswith("gs://")
                 else Path(destination))
    for p in source.rglob("*"):
        if p.is_file():
            out=destination/p.relative_to(source)
            if not out.exists():
                out.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(p,out)
''')
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", f"{executable.parent}:{os.environ['PATH']}")
    monkeypatch.setenv("TEST_CACHE_CLOUD", str(cloud))
    return cloud


def test_weekly_rotation_preserves_used_objects_and_drops_unused_variants(
    tmp_path, gcloud_objects,
):
    old = gcloud_objects / "test/2026-10-05/x86_64/cuda"
    old.mkdir(parents=True)
    (old / "hot").write_bytes(b"compatible compiler output")
    (old / "unused").write_bytes(b"inactive source variant")
    local = tmp_path / "local"
    week = datetime(2026, 10, 12, tzinfo=UTC)
    transfer("restore", "cache", "test", "x86_64", local,
             tmp_path / "restore.json", now=week)
    assert (local / "cuda/unused").read_bytes() == b"inactive source variant"
    started = 2_000_000_000
    os.utime(local / "cuda/hot", ns=(started + 1, started + 1))
    os.utime(local / "cuda/unused", ns=(1_000_000_000, 1_000_000_000))
    transfer("save", "cache", "test", "x86_64", local,
             tmp_path / "save.json", now=week, used_since_ns=started)
    current = gcloud_objects / "test/2026-10-12/x86_64/cuda"
    assert (current / "hot").read_bytes() == b"compatible compiler output"
    assert not (current / "unused").exists()
    # Simulate lifecycle eviction of the original epoch, without touching a
    # real bucket. The next fresh builder must still reuse the carried hot key.
    shutil.rmtree(gcloud_objects / "test/2026-10-05")
    fresh = tmp_path / "fresh"
    transfer("restore", "cache", "test", "x86_64", fresh,
             tmp_path / "next.json", now=datetime(2026, 10, 19, tzinfo=UTC))
    assert (fresh / "cuda/hot").read_bytes() == b"compatible compiler output"
    assert not (fresh / "cuda/unused").exists()
