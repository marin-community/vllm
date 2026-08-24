#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate a published Marin TPU wheel pair on a TPU worker."""

from __future__ import annotations

import argparse
import os
import tempfile
import urllib.request
from pathlib import Path

from release_common import load_json, sha256_file
from tpu_release import (
    VALIDATION_SENTINEL,
    local_flat_index,
    validate_candidate,
)
from validation_common import (
    ValidationFailure,
    download_url,
    emit_result,
    require_command,
)

GCP_TPU_TYPE_URL = (
    "http://metadata.google.internal/computeMetadata/v1/instance/attributes/"
    "accelerator-type"
)


def physical_tpu_type() -> str:
    """Return the physical TPU type reported by GCP instance metadata."""
    request = urllib.request.Request(
        GCP_TPU_TYPE_URL,
        headers={"Metadata-Flavor": "Google"},
    )
    with urllib.request.urlopen(request, timeout=2) as response:
        tpu = response.read().decode().strip()
    if not tpu:
        raise ValidationFailure(
            "GCP metadata returned an empty physical TPU type"
        )
    return tpu


def download_and_verify(url: str, destination: Path, expected_sha256: str) -> None:
    print(f"::: downloading {url}", flush=True)
    download_url(url, destination)
    actual = sha256_file(destination)
    if actual != expected_sha256:
        raise ValidationFailure(
            f"SHA-256 mismatch for {destination.name}: "
            f"{actual} != {expected_sha256}"
        )


def validate(args: argparse.Namespace) -> int:
    config = load_json(args.config)
    candidate = load_json(args.candidate_manifest)
    serve_smoke = args.serve_smoke.resolve()
    spec = args.spec.resolve()
    environment = os.environ.copy()
    environment.update(
        {
            "UV_LINK_MODE": "copy",
            "UV_TORCH_BACKEND": "cpu",
            "VLLM_TARGET_DEVICE": config["vllm_target_device"],
        }
    )

    validate_candidate(candidate, config)
    selected_tpu = physical_tpu_type()
    expected_tpu = config["validation"]["hardware"]
    if selected_tpu != expected_tpu:
        raise ValidationFailure(
            f"Iris placed {selected_tpu}, expected {expected_tpu}"
        )

    # Iris mounts /tmp noexec. Keep the installed environment under its
    # executable task workdir so native libraries can be mapped.
    with tempfile.TemporaryDirectory(
        prefix="marin-vllm-tpu-release-", dir=Path.cwd()
    ) as directory:
        workdir = Path(directory)
        environment["UV_CACHE_DIR"] = str(workdir / "uv-cache")
        index = candidate["index"]
        index_path = workdir / index["filename"]
        download_and_verify(index["url"], index_path, index["sha256"])

        validation = config["validation"]
        with local_flat_index(index_path) as index_url:
            require_command(
                [
                    "uv",
                    "run",
                    "--no-project",
                    "--python",
                    config["python_version"],
                    "--prerelease",
                    "allow",
                    "--exclude-newer",
                    candidate["compatibility"]["exclude_newer"],
                    "--find-links",
                    index_url,
                    *(
                        item
                        for package in candidate["packages"]
                        for item in (
                            "--with",
                            f"{package['distribution']}=={package['version']}",
                        )
                    ),
                    "python",
                    str(serve_smoke),
                    "--spec",
                    str(spec),
                    "--tensor-parallel-size",
                    str(validation["tensor_parallel_size"]),
                    "--startup-timeout",
                    "1800",
                ],
                cwd=workdir,
                environment=environment,
                phase="clean wheel install and TPU serve smoke",
            )

    emit_result(
        {
            "candidate_tag": candidate["release"]["tag"],
            "hardware": selected_tpu,
            "run_url": args.qualification_run_url,
        },
        sentinel=VALIDATION_SENTINEL,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--serve-smoke", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--qualification-run-url", required=True)
    return validate(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
