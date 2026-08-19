# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
from io import BytesIO
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request

import pytest


@pytest.fixture
def tpu_validation(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]))
    return importlib.import_module("tpu_validation")


def test_physical_tpu_type_reads_gcp_instance_metadata(
    monkeypatch: pytest.MonkeyPatch, tpu_validation
) -> None:
    def urlopen(request: Request, **_: object) -> BytesIO:
        assert request.get_header("Metadata-flavor") == "Google"
        return BytesIO(b"v6e-8\n")

    monkeypatch.setattr(tpu_validation.urllib.request, "urlopen", urlopen)

    assert tpu_validation.physical_tpu_type() == "v6e-8"


def test_physical_tpu_type_rejects_empty_metadata(
    monkeypatch: pytest.MonkeyPatch, tpu_validation
) -> None:
    monkeypatch.setattr(
        tpu_validation.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: BytesIO(),
    )

    with pytest.raises(
        tpu_validation.ValidationFailure,
        match="empty physical TPU type",
    ):
        tpu_validation.physical_tpu_type()


def test_physical_tpu_type_propagates_metadata_failure(
    monkeypatch: pytest.MonkeyPatch, tpu_validation
) -> None:
    def fail(*_args: object, **_kwargs: object) -> None:
        raise URLError("unavailable")

    monkeypatch.setattr(tpu_validation.urllib.request, "urlopen", fail)

    with pytest.raises(URLError, match="unavailable"):
        tpu_validation.physical_tpu_type()
