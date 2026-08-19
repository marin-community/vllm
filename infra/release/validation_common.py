#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared runtime helpers for Marin release qualification."""

from __future__ import annotations

import base64
import json
import shutil
import subprocess
import urllib.request
from pathlib import Path
from typing import Any


class ValidationFailure(RuntimeError):
    """A release validation phase failed."""


def gate(status: str, detail: str = "") -> dict[str, str]:
    value = {"status": status}
    if detail:
        value["detail"] = detail
    return value


def run_command(
    command: list[str],
    *,
    cwd: Path,
    environment: dict[str, str],
) -> int:
    print(f"::: running {' '.join(command)}", flush=True)
    completed = subprocess.run(command, cwd=cwd, env=environment, check=False)
    return completed.returncode


def require_command(
    command: list[str],
    *,
    cwd: Path,
    environment: dict[str, str],
    phase: str,
) -> None:
    return_code = run_command(command, cwd=cwd, environment=environment)
    if return_code != 0:
        raise ValidationFailure(f"{phase} exited with code {return_code}")


def download_wheel(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "marin-vllm-release"})
    with (
        urllib.request.urlopen(request, timeout=900) as response,
        destination.open("wb") as output,
    ):
        shutil.copyfileobj(response, output, length=1024 * 1024)


def emit_result(result: dict[str, Any], *, sentinel: str) -> None:
    payload = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    print(sentinel + base64.b64encode(payload).decode(), flush=True)
