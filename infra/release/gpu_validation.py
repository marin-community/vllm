#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate one Marin vLLM release wheel inside an Iris GPU job."""

from __future__ import annotations

import argparse
import base64
import gc
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import traceback
import urllib.request
from importlib import metadata
from pathlib import Path
from typing import Any

from gpu_release import (
    GRUG_ARCHITECTURE,
    VALIDATION_SENTINEL,
    load_json,
    sha256_file,
    validate_candidate,
    write_json,
)

SOURCE_TESTS = (
    "tests/models/test_grugmoe.py",
    "tests/v1/core/test_scheduler.py",
    "tests/engine/test_arg_utils.py::TestDpDeviceIdSharding",
    "tests/distributed/test_mq_connect_ip.py::test_mq_bind_with_local_ip",
)
SOURCE_TEST_DESELECTS = (
    "tests/models/test_grugmoe.py::test_grug_moe_parallel_config_rejects_tp_larger_than_attention_heads",
    "tests/v1/core/test_scheduler.py::test_async_scheduling_pp_allows_rescheduling_with_output_placeholders",
)


class ValidationFailure(RuntimeError):
    """A GPU release validation phase failed."""


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


def initial_result(
    candidate: dict[str, Any], architecture: str, hardware: str, task_image: str
) -> dict[str, Any]:
    platform_record = next(
        item for item in candidate["platforms"] if item["architecture"] == architecture
    )
    return {
        "schema_version": 1,
        "candidate_tag": candidate["release"]["tag"],
        "source_commit": candidate["source"]["fork_commit"],
        "architecture": architecture,
        "wheel": {
            "filename": platform_record["wheel"]["filename"],
            "sha256": platform_record["wheel"]["sha256"],
            "url": platform_record["wheel"]["url"],
        },
        "hardware": {"requested": hardware},
        "environment": {"task_image": task_image},
        "gates": {
            "wheel_sha256": gate("not_run"),
            "distribution_metadata": gate("not_run"),
            "vllm._C": gate("not_run"),
            GRUG_ARCHITECTURE: gate("not_run"),
            "cumem_allocator": gate("not_run"),
            "source_tests": gate("not_run"),
            "serve_smoke": gate("not_run"),
        },
        "result": "failed",
    }


def installed_probe(args: argparse.Namespace) -> int:
    import torch

    result: dict[str, Any] = {
        "environment": {
            "python_version": platform.python_version(),
            "machine": platform.machine(),
            "torch_version": torch.__version__,
            "torch_cuda_runtime": torch.version.cuda,
        },
        "hardware": {},
        "gates": {
            "distribution_metadata": gate("not_run"),
            "vllm._C": gate("not_run"),
            GRUG_ARCHITECTURE: gate("not_run"),
            "cumem_allocator": gate("not_run"),
        },
    }
    failed = False
    try:
        vllm = importlib.import_module("vllm")
        source_root = args.source_root.resolve()
        package_path = Path(vllm.__file__).resolve()
        result["environment"]["vllm_package_path"] = str(package_path)
        if package_path.is_relative_to(source_root):
            raise ValidationFailure(
                f"vllm imported from source checkout {package_path}, not the wheel"
            )
        distribution = metadata.distribution(args.distribution)
        actual_name = distribution.metadata["Name"]
        actual_version = distribution.version
        if actual_name != args.distribution or actual_version != args.version:
            raise ValidationFailure(
                f"installed {actual_name} {actual_version}, expected "
                f"{args.distribution} {args.version}"
            )
        if torch.__version__ != args.torch_version:
            raise ValidationFailure(
                f"installed Torch {torch.__version__}, expected {args.torch_version}"
            )
        if torch.version.cuda != args.cuda_runtime:
            raise ValidationFailure(
                f"Torch CUDA runtime {torch.version.cuda}, expected {args.cuda_runtime}"
            )
        result["gates"]["distribution_metadata"] = gate("passed")
    except Exception as exc:
        failed = True
        result["gates"]["distribution_metadata"] = gate("failed", str(exc))

    if not torch.cuda.is_available():
        failed = True
        result["hardware"] = {"status": "absent"}
    else:
        capability = torch.cuda.get_device_capability(0)
        capability_text = f"{capability[0]}.{capability[1]}"
        result["hardware"] = {
            "name": torch.cuda.get_device_name(0),
            "compute_capability": capability_text,
        }
        if capability_text != args.compute_capability:
            failed = True
            result["hardware"]["status"] = "unexpected"
        else:
            result["hardware"]["status"] = "passed"

    try:
        importlib.import_module("vllm._C")
        result["gates"]["vllm._C"] = gate("passed")
    except Exception as exc:
        failed = True
        result["gates"]["vllm._C"] = gate("failed", repr(exc))

    try:
        module = importlib.import_module("vllm.model_executor.models.grugmoe")
        model_class = getattr(module, GRUG_ARCHITECTURE)
        if model_class.__name__ != GRUG_ARCHITECTURE:
            raise ValidationFailure(f"loaded unexpected class {model_class!r}")
        result["gates"][GRUG_ARCHITECTURE] = gate("passed")
    except Exception as exc:
        failed = True
        result["gates"][GRUG_ARCHITECTURE] = gate("failed", repr(exc))

    try:
        importlib.import_module("vllm.cumem_allocator")
        from vllm.device_allocator.cumem import CuMemAllocator, cumem_available

        if not cumem_available:
            failed = True
            result["gates"]["cumem_allocator"] = gate(
                "absent", "vllm.device_allocator.cumem reported unavailable"
            )
        else:
            allocator = CuMemAllocator.get_instance()
            with allocator.use_memory_pool("marin-release-probe"):
                tensor = torch.arange(4096, dtype=torch.float32, device="cuda")
                torch.cuda.synchronize()
                usage = allocator.get_current_usage()
                checksum = tensor.sum().item()
                if usage < tensor.numel() * tensor.element_size():
                    raise ValidationFailure(
                        f"cuMem allocator reported only {usage} allocated bytes"
                    )
                if checksum != 8_386_560.0:
                    raise ValidationFailure(
                        f"cuMem allocation produced checksum {checksum}"
                    )
                del tensor
                gc.collect()
                torch.cuda.synchronize()
            result["gates"]["cumem_allocator"] = {
                "status": "passed",
                "allocated_bytes": usage,
                "checksum": checksum,
            }
    except ModuleNotFoundError as exc:
        failed = True
        result["gates"]["cumem_allocator"] = gate("absent", repr(exc))
    except Exception as exc:
        failed = True
        result["gates"]["cumem_allocator"] = gate("failed", repr(exc))

    write_json(args.output, result)
    return int(failed)


def run_wheel_tests(args: argparse.Namespace) -> int:
    """Run checkout test modules against the already imported wheel package."""
    import vllm

    source_root = args.source_root.resolve()
    package_path = Path(vllm.__file__).resolve()
    if package_path.is_relative_to(source_root):
        print(
            f"error: vllm imported from source checkout {package_path}",
            file=sys.stderr,
        )
        return 1
    print(f"::: wheel tests import vllm from {package_path}", flush=True)

    # `vllm` is now fixed in sys.modules with its site-packages __path__. Add the
    # checkout only so pytest can import the candidate commit's `tests` package.
    import pytest

    sys.path.insert(0, str(source_root))
    os.chdir(source_root)
    pytest_args = args.pytest_args
    if pytest_args and pytest_args[0] == "--":
        pytest_args = pytest_args[1:]
    return pytest.main(pytest_args)


def emit_result(result: dict[str, Any]) -> None:
    payload = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    print(VALIDATION_SENTINEL + base64.b64encode(payload).decode(), flush=True)


def validate(args: argparse.Namespace) -> int:
    config = load_json(args.config)
    candidate = load_json(args.candidate_manifest)
    validate_candidate(candidate, config)
    architecture = args.architecture
    platform_config = config["platforms"][architecture]
    expected_validation = platform_config["validation"]
    result = initial_result(
        candidate, architecture, expected_validation["gpu"], args.task_image
    )
    platform_record = next(
        item for item in candidate["platforms"] if item["architecture"] == architecture
    )
    repo_root = Path(__file__).resolve().parents[2]

    try:
        if platform.machine() != architecture:
            raise ValidationFailure(
                f"Iris task architecture is {platform.machine()}, "
                f"expected {architecture}"
            )
        if args.hardware != expected_validation["gpu"]:
            raise ValidationFailure(
                f"requested hardware is {args.hardware}, expected "
                f"{expected_validation['gpu']}"
            )
        if args.task_image != config["validation_task_image"]:
            raise ValidationFailure("Iris task image does not match release config")

        with tempfile.TemporaryDirectory(prefix="marin-vllm-release-") as directory:
            workdir = Path(directory)
            wheel = workdir / platform_record["wheel"]["filename"]
            print(f"::: downloading {platform_record['wheel']['url']}", flush=True)
            download_wheel(platform_record["wheel"]["url"], wheel)
            actual_digest = sha256_file(wheel)
            expected_digest = platform_record["wheel"]["sha256"]
            if actual_digest != expected_digest:
                result["gates"]["wheel_sha256"] = gate(
                    "failed", f"{actual_digest} != {expected_digest}"
                )
                raise ValidationFailure(
                    "downloaded wheel SHA-256 does not match manifest"
                )
            result["gates"]["wheel_sha256"] = gate("passed")

            environment = os.environ.copy()
            environment.update(
                {
                    "UV_LINK_MODE": "copy",
                    "VLLM_USE_FLASHINFER_SAMPLER": "0",
                }
            )
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
                phase="virtual environment creation",
            )
            python = virtual_environment / "bin/python"
            require_command(
                [
                    "uv",
                    "pip",
                    "install",
                    "--python",
                    str(python),
                    "--index-url",
                    config["torch_index_url"],
                    f"torch=={config['torch_version']}",
                ],
                cwd=workdir,
                environment=environment,
                phase="Torch installation",
            )
            require_command(
                [
                    "uv",
                    "pip",
                    "install",
                    "--python",
                    str(python),
                    "--index-strategy",
                    "unsafe-best-match",
                    "--extra-index-url",
                    config["torch_index_url"],
                    str(wheel),
                    "pytest",
                    "tblib",
                ],
                cwd=workdir,
                environment=environment,
                phase="wheel installation",
            )

            probe_path = workdir / "installed-probe.json"
            probe_command = [
                str(python),
                str(Path(__file__).resolve()),
                "probe-installed",
                "--distribution",
                candidate["distribution"]["name"],
                "--version",
                candidate["distribution"]["version"],
                "--torch-version",
                config["torch_version"],
                "--cuda-runtime",
                config["cuda_runtime_version"],
                "--compute-capability",
                expected_validation["compute_capability"],
                "--source-root",
                str(repo_root),
                "--output",
                str(probe_path),
            ]
            probe_return_code = run_command(
                probe_command, cwd=workdir, environment=environment
            )
            if not probe_path.is_file():
                raise ValidationFailure("installed-wheel probe produced no result")
            probe = load_json(probe_path)
            result["environment"].update(probe.get("environment", {}))
            result["hardware"].update(probe.get("hardware", {}))
            result["gates"].update(probe.get("gates", {}))
            if probe_return_code != 0:
                raise ValidationFailure(
                    f"installed-wheel probe exited with code {probe_return_code}"
                )

            if expected_validation["run_source_tests"]:
                source_test_command = [
                    str(python),
                    str(Path(__file__).resolve()),
                    "run-wheel-tests",
                    "--source-root",
                    str(repo_root),
                    "--",
                    "-v",
                    *SOURCE_TESTS,
                ]
                for test in SOURCE_TEST_DESELECTS:
                    source_test_command.extend(["--deselect", test])
                source_test_return_code = run_command(
                    source_test_command,
                    cwd=workdir,
                    environment=environment,
                )
                if source_test_return_code != 0:
                    result["gates"]["source_tests"] = gate(
                        "failed", f"pytest exited with code {source_test_return_code}"
                    )
                    raise ValidationFailure("source behavior tests failed")
                result["gates"]["source_tests"] = gate("passed")
            else:
                result["gates"]["source_tests"] = gate(
                    "not_run", "source behavior suite runs on the H100 lane"
                )

            serving_result = workdir / "serve-smoke.json"
            serve_return_code = run_command(
                [
                    str(python),
                    str(repo_root / "infra/nightly/gpu_serve_smoke.py"),
                    "--model",
                    args.model,
                    "--spec",
                    str(repo_root / expected_validation["spec"]),
                    "--result-json",
                    str(serving_result),
                ],
                cwd=workdir,
                environment=environment,
            )
            if serving_result.is_file():
                result["gates"]["serve_smoke"] = {
                    "status": "passed" if serve_return_code == 0 else "failed",
                    "metrics": load_json(serving_result),
                }
            else:
                result["gates"]["serve_smoke"] = gate(
                    "failed", "serve smoke produced no result"
                )
            if serve_return_code != 0:
                raise ValidationFailure(
                    f"serve smoke exited with code {serve_return_code}"
                )

        result["result"] = "passed"
    except Exception as exc:
        result["result"] = "failed"
        result["failure"] = str(exc)
        traceback.print_exc()
    emit_result(result)
    return 0 if result["result"] == "passed" else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--config", type=Path, required=True)
    validate_parser.add_argument("--candidate-manifest", type=Path, required=True)
    validate_parser.add_argument("--architecture", required=True)
    validate_parser.add_argument("--hardware", required=True)
    validate_parser.add_argument("--task-image", required=True)
    validate_parser.add_argument("--model", required=True)

    probe_parser = subparsers.add_parser("probe-installed")
    probe_parser.add_argument("--distribution", required=True)
    probe_parser.add_argument("--version", required=True)
    probe_parser.add_argument("--torch-version", required=True)
    probe_parser.add_argument("--cuda-runtime", required=True)
    probe_parser.add_argument("--compute-capability", required=True)
    probe_parser.add_argument("--source-root", type=Path, required=True)
    probe_parser.add_argument("--output", type=Path, required=True)

    tests_parser = subparsers.add_parser("run-wheel-tests")
    tests_parser.add_argument("--source-root", type=Path, required=True)
    tests_parser.add_argument("pytest_args", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "probe-installed":
        return installed_probe(args)
    if args.command == "run-wheel-tests":
        return run_wheel_tests(args)
    if args.command == "validate":
        return validate(args)
    raise AssertionError(f"unhandled command {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
