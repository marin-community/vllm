#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared primitives for Marin wheel releases."""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
from dataclasses import dataclass
from email.message import Message
from email.parser import BytesParser
from pathlib import Path
from typing import Any
from urllib.parse import quote

RELEASE_REPOSITORY = "marin-community/vllm"


class ReleaseError(RuntimeError):
    """A release input violates the published artifact contract."""


@dataclass(frozen=True)
class WheelMetadata:
    """Release-relevant metadata read from one wheel archive."""

    metadata: Message
    tags: list[str]


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ReleaseError(f"{path} must contain a JSON object")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_distribution_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def requirement_name(requirement: str) -> str:
    match = re.match(r"\s*([A-Za-z0-9_.-]+)", requirement)
    if match is None:
        raise ReleaseError(f"cannot parse Requires-Dist entry {requirement!r}")
    return normalized_distribution_name(match.group(1))


def release_asset_url(repository: str, tag: str, filename: str) -> str:
    return (
        f"https://github.com/{repository}/releases/download/"
        f"{quote(tag, safe='')}/{quote(filename, safe='+._-')}"
    )


def wheel_metadata(wheel: Path) -> WheelMetadata:
    """Read package metadata and compatibility tags from a wheel."""
    with zipfile.ZipFile(wheel) as archive:
        metadata_names = [
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        ]
        wheel_names = [
            name for name in archive.namelist() if name.endswith(".dist-info/WHEEL")
        ]
        if len(metadata_names) != 1 or len(wheel_names) != 1:
            raise ReleaseError(
                f"{wheel.name} must contain one METADATA and one WHEEL file"
            )
        metadata = BytesParser().parsebytes(archive.read(metadata_names[0]))
        wheel_document = BytesParser().parsebytes(archive.read(wheel_names[0]))
    return WheelMetadata(metadata, wheel_document.get_all("Tag", []))
