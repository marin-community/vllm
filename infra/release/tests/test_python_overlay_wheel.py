# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import binascii
import csv
import hashlib
import json
import subprocess
import zipfile

import pytest

from infra.release.python_overlay_provenance import annotate_fragment
from infra.release.python_overlay_wheel import repack_python_overlay


def _commit(repository, message):
    subprocess.run(["git", "add", "."], cwd=repository, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-m",
            message,
        ],
        cwd=repository,
        check=True,
        capture_output=True,
    )
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository, text=True
    ).strip()


def _base_wheel(path):
    dist_info = "vllm-1.0.0.dist-info"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("vllm/_C.abi3.so", b"compiled-bytes")
        archive.writestr("vllm/existing.py", "VALUE = 'base'\n")
        archive.writestr(
            "vllm/_version.py", "__version__ = '1.0.0'\n__version_tuple__ = (1, 0, 0)\n"
        )
        archive.writestr(
            f"{dist_info}/METADATA",
            "Metadata-Version: 2.4\nName: vllm\nVersion: 1.0.0\n",
        )
        archive.writestr(
            f"{dist_info}/WHEEL", "Wheel-Version: 1.0\nTag: cp38-abi3-linux_x86_64\n"
        )
        archive.writestr(f"{dist_info}/RECORD", "")


def test_repack_overlays_only_python_and_rewrites_identity(tmp_path):
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    package = repository / "vllm"
    package.mkdir()
    (package / "existing.py").write_text("VALUE = 'base'\n")
    base_commit = _commit(repository, "base")
    (package / "existing.py").write_text("VALUE = 'overlay'\n")
    (package / "added.py").write_text("ADDED = True\n")
    head_commit = _commit(repository, "overlay")
    base_wheel = tmp_path / "vllm-1.0.0-cp38-abi3-manylinux_2_28_x86_64.whl"
    _base_wheel(base_wheel)

    output = repack_python_overlay(
        base_wheel=base_wheel,
        repository=repository,
        base_commit=base_commit,
        head_commit=head_commit,
        version="1.0.0+overlay",
        output_dir=tmp_path / "dist",
    )

    assert output.name == "vllm-1.0.0+overlay-cp38-abi3-manylinux_2_28_x86_64.whl"
    with zipfile.ZipFile(output) as archive:
        assert archive.read("vllm/_C.abi3.so") == b"compiled-bytes"
        assert archive.read("vllm/existing.py") == b"VALUE = 'overlay'\n"
        assert archive.read("vllm/added.py") == b"ADDED = True\n"
        assert b"Version: 1.0.0+overlay" in archive.read(
            "vllm-1.0.0+overlay.dist-info/METADATA"
        )
        assert b"__version__ = version = '1.0.0+overlay'" in archive.read(
            "vllm/_version.py"
        )
        record = list(
            csv.reader(
                archive.read("vllm-1.0.0+overlay.dist-info/RECORD")
                .decode()
                .splitlines()
            )
        )
        record_by_path = {row[0]: row for row in record}
        digest = (
            binascii.b2a_base64(
                hashlib.sha256(b"VALUE = 'overlay'\n").digest(), newline=False
            )
            .rstrip(b"=")
            .translate(bytes.maketrans(b"+/", b"-_"))
            .decode()
        )
        assert record_by_path["vllm/existing.py"][2] == str(len(b"VALUE = 'overlay'\n"))
        assert record_by_path["vllm/existing.py"][1] == f"sha256={digest}"


def test_overlay_fragment_records_the_qualified_binary_base(tmp_path):
    fragment_path = tmp_path / "fragment.json"
    fragment_path.write_text(json.dumps({"build": {"provenance": {"run_id": "123"}}}))

    annotate_fragment(
        fragment_path,
        base_release="marin-vllm-gpu-20260805-fa50698a9a30",
        base_commit="fa50698a9a303f7282aa0e969f35717703de4911",
    )

    provenance = json.loads(fragment_path.read_text())["build"]["provenance"]
    assert provenance == {
        "binary_base_commit": "fa50698a9a303f7282aa0e969f35717703de4911",
        "binary_base_release": "marin-vllm-gpu-20260805-fa50698a9a30",
        "run_id": "123",
    }


def test_repack_rejects_binary_package_changes(tmp_path):
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    package = repository / "vllm"
    package.mkdir()
    (package / "existing.py").write_text("VALUE = 'base'\n")
    base_commit = _commit(repository, "base")
    (package / "_C.abi3.so").write_bytes(b"new-binary")
    head_commit = _commit(repository, "binary change")
    base_wheel = tmp_path / "vllm-1.0.0-cp38-abi3-manylinux_2_28_x86_64.whl"
    _base_wheel(base_wheel)

    with pytest.raises(RuntimeError, match="only vllm Python files"):
        repack_python_overlay(
            base_wheel=base_wheel,
            repository=repository,
            base_commit=base_commit,
            head_commit=head_commit,
            version="1.0.0+overlay",
            output_dir=tmp_path / "dist",
        )
