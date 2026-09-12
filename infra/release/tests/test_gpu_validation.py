# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import io
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
from infra.release import gpu_validation


class DownloadResponse(io.BytesIO):
    def __init__(self, content: bytes, *, status: int):
        super().__init__(content)
        self.status = status


class InterruptedResponse(DownloadResponse):
    def __init__(self, content: bytes, *, status: int, interrupt_after: int):
        super().__init__(content, status=status)
        self.interrupt_after = interrupt_after

    def read(self, size: int = -1) -> bytes:
        if self.tell() >= self.interrupt_after:
            raise urllib.error.URLError("transient interruption")
        return super().read(min(size, self.interrupt_after - self.tell()))


def test_download_wheel_resumes_after_transient_interruption(tmp_path, monkeypatch):
    content = b"candidate-wheel-bytes"
    requests: list[urllib.request.Request] = []

    def urlopen(request: urllib.request.Request, *, timeout: int):
        assert timeout == 900
        requests.append(request)
        if len(requests) == 1:
            return InterruptedResponse(content, status=200, interrupt_after=10)
        assert request.get_header("Range") == "bytes=10-"
        return DownloadResponse(content[10:], status=206)

    monkeypatch.setattr(gpu_validation.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(gpu_validation.time, "sleep", lambda _: None)
    destination = tmp_path / "candidate.whl"

    gpu_validation.download_wheel("https://example.invalid/candidate.whl", destination)

    assert destination.read_bytes() == content
    assert len(requests) == 2


def test_download_wheel_restarts_if_server_ignores_range(tmp_path, monkeypatch):
    content = b"candidate-wheel-bytes"
    destination = tmp_path / "candidate.whl"
    destination.write_bytes(b"stale-prefix")

    def urlopen(request: urllib.request.Request, *, timeout: int):
        assert timeout == 900
        assert request.get_header("Range") == "bytes=12-"
        return DownloadResponse(content, status=200)

    monkeypatch.setattr(gpu_validation.urllib.request, "urlopen", urlopen)

    gpu_validation.download_wheel("https://example.invalid/candidate.whl", destination)

    assert destination.read_bytes() == content


def test_validation_compiler_installs_configured_system_packages(tmp_path, monkeypatch):
    commands: list[list[str]] = []
    lookups = {"cc": None, "c++": None, "apt-get": "/usr/bin/apt-get"}

    def which(command: str):
        return lookups[command]

    def require_command(command, *, cwd, environment, phase):
        assert cwd == tmp_path
        assert environment == {"PATH": "/usr/bin"}
        assert phase
        commands.append(command)
        if command[1] == "install":
            lookups["cc"] = "/usr/bin/cc"
            lookups["c++"] = "/usr/bin/c++"

    monkeypatch.setattr(gpu_validation.shutil, "which", which)
    monkeypatch.setattr(gpu_validation, "require_command", require_command)

    gpu_validation.ensure_validation_compiler(
        tmp_path,
        {"validation_system_packages": ["build-essential"]},
        {"PATH": "/usr/bin"},
    )

    assert commands == [
        ["/usr/bin/apt-get", "update"],
        [
            "/usr/bin/apt-get",
            "install",
            "--yes",
            "--no-install-recommends",
            "build-essential",
        ],
    ]
