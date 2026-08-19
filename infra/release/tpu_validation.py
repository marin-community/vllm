#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate a published Marin TPU wheel pair on a TPU worker."""

from __future__ import annotations

import argparse
import contextlib
import os
import tempfile
import traceback
import urllib.request
from pathlib import Path
from typing import Any

from release_common import load_json, sha256_file, write_json
from tpu_release import (
    VALIDATION_SENTINEL,
    local_flat_index,
    validate_candidate,
)
from validation_common import (
    ValidationFailure,
    download_wheel,
    emit_result,
    gate,
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


def initial_result(
    candidate: dict[str, Any], qualification_run_url: str
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "candidate_tag": candidate["release"]["tag"],
        "source": candidate["source"],
        "workflow": candidate["workflow"],
        "qualification": {"run_url": qualification_run_url},
        "index": candidate["index"],
        "wheels": {
            package["distribution"]: package["wheel"]
            for package in candidate["packages"]
        },
        "hardware": {},
        "gates": {
            "wheel_sha256": gate("not_run"),
            "clean_install": gate("not_run"),
            "serve_smoke": gate("not_run"),
        },
        "result": "failed",
    }


def download_and_verify(url: str, destination: Path, expected_sha256: str) -> None:
    print(f"::: downloading {url}", flush=True)
    download_wheel(url, destination)
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
    result = initial_result(candidate, args.qualification_run_url)
    active_gate: str | None = None
    smoke_result: Path | None = None
    environment = os.environ.copy()
    environment.update(
        {
            "UV_LINK_MODE": "copy",
            "VLLM_TARGET_DEVICE": config["vllm_target_device"],
        }
    )

    try:
        validate_candidate(candidate, config)
        selected_tpu = physical_tpu_type()
        expected_tpu = config["validation"]["hardware"]
        result["hardware"] = {"selected": selected_tpu}
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
            active_gate = "wheel_sha256"
            for package in candidate["packages"]:
                wheel = package["wheel"]
                download_and_verify(
                    wheel["url"], workdir / wheel["filename"], wheel["sha256"]
                )
            index = candidate["index"]
            index_path = workdir / index["filename"]
            download_and_verify(index["url"], index_path, index["sha256"])
            result["gates"]["wheel_sha256"] = gate("passed")

            active_gate = "clean_install"
            virtual_environment = workdir / "venv"
            require_command(
                [
                    "uv",
                    "venv",
                    "--python",
                    config["python_version"],
                    str(virtual_environment),
                ],
                cwd=workdir,
                environment=environment,
                phase="uv venv",
            )
            python = virtual_environment / "bin/python"
            with local_flat_index(index_path) as index_url:
                require_command(
                    [
                        "uv",
                        "pip",
                        "install",
                        "--python",
                        str(python),
                        "--prerelease",
                        "allow",
                        "--exclude-newer",
                        candidate["compatibility"]["exclude_newer"],
                        "--torch-backend",
                        "cpu",
                        "--find-links",
                        index_url,
                        *(
                            f"{package['distribution']}=={package['version']}"
                            for package in candidate["packages"]
                        ),
                    ],
                    cwd=workdir,
                    environment=environment,
                    phase="wheel pair install",
                )
            require_command(
                [
                    str(python),
                    "-c",
                    (
                        "import vllm; "
                        "from tpu_inference.runner.tpu_runner "
                        "import TPUModelRunner"
                    ),
                ],
                cwd=workdir,
                environment=environment,
                phase="installed pair import",
            )
            result["gates"]["clean_install"] = gate("passed")

            active_gate = "serve_smoke"
            smoke_result = workdir / "serve-smoke.json"
            validation = config["validation"]
            require_command(
                [
                    str(python),
                    str(serve_smoke),
                    "--spec",
                    str(spec),
                    "--tensor-parallel-size",
                    str(validation["tensor_parallel_size"]),
                    "--startup-timeout",
                    "1800",
                    "--result-json",
                    str(smoke_result),
                ],
                cwd=workdir,
                environment=environment,
                phase="serve smoke",
            )
            result["gates"]["serve_smoke"] = {
                "status": "passed",
                "metrics": load_json(smoke_result),
            }
            active_gate = None
        result["result"] = "passed"
    except Exception as exc:
        if (
            active_gate is not None
            and result["gates"][active_gate]["status"] == "not_run"
        ):
            result["gates"][active_gate] = gate("failed", str(exc))
            if active_gate == "serve_smoke" and smoke_result is not None:
                with contextlib.suppress(OSError, ValueError):
                    result["gates"][active_gate]["metrics"] = load_json(
                        smoke_result
                    )
        result["failure"] = str(exc)
        traceback.print_exc()

    write_json(args.output, result)
    emit_result(result, sentinel=VALIDATION_SENTINEL)
    return 0 if result["result"] == "passed" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--serve-smoke", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qualification-run-url", required=True)
    return validate(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
