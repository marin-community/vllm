# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
from pathlib import Path
from urllib.error import URLError

import pytest


class _MetadataResponse:

    def __init__(self, value: bytes):
        self.value = value

    def __enter__(self):
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.value


@pytest.fixture
def tpu_validation(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]))
    return importlib.import_module("tpu_validation")


def test_placed_tpu_reads_gcp_instance_metadata(
    monkeypatch: pytest.MonkeyPatch, tpu_validation
) -> None:
    request_details: dict[str, object] = {}

    def urlopen(request: object, timeout: int) -> _MetadataResponse:
        request_details.update(request=request, timeout=timeout)
        return _MetadataResponse(b"v6e-8\n")

    monkeypatch.setattr(tpu_validation.urllib.request, "urlopen", urlopen)

    assert tpu_validation.placed_tpu() == "v6e-8"
    request = request_details["request"]
    assert request.full_url == tpu_validation.GCP_TPU_TYPE_URL
    assert request.get_header("Metadata-flavor") == "Google"
    assert request_details["timeout"] == 2


def test_placed_tpu_rejects_empty_metadata(
    monkeypatch: pytest.MonkeyPatch, tpu_validation
) -> None:
    monkeypatch.setattr(
        tpu_validation.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: _MetadataResponse(b""),
    )

    with pytest.raises(
        tpu_validation.ValidationFailure,
        match="empty physical TPU type",
    ):
        tpu_validation.placed_tpu()


def test_placed_tpu_rejects_metadata_failure(
    monkeypatch: pytest.MonkeyPatch, tpu_validation
) -> None:
    def fail(*_args: object, **_kwargs: object) -> None:
        raise URLError("unavailable")

    monkeypatch.setattr(tpu_validation.urllib.request, "urlopen", fail)

    with pytest.raises(tpu_validation.ValidationFailure, match="could not read"):
        tpu_validation.placed_tpu()
