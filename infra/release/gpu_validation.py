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
    CUMEM_GATE,
    DISTRIBUTION_GATE,
    GRUG_ARCHITECTURE,
    SERVE_GATE,
    SOURCE_TESTS_GATE,
    VALIDATION_SENTINEL,
    WHEEL_SHA_GATE,
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
SOURCE_TEST_EXCLUDES = (
    "test_grug_moe_parallel_config_rejects_tp_larger_than_attention_heads",
    "test_async_scheduling_pp_allows_rescheduling_with_output_placeholders",
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
            WHEEL_SHA_GATE: gate("not_run"),
            DISTRIBUTION_GATE: gate("not_run"),
            "vllm._C": gate("not_run"),
            GRUG_ARCHITECTURE: gate("not_run"),
            CUMEM_GATE: gate("not_run"),
            SOURCE_TESTS_GATE: gate("not_run"),
            SERVE_GATE: gate("not_run"),
        },
        "result": "failed",
    }


def installed_vllm_path(*source_roots: Path) -> Path:
    """Import vllm and require its package to live outside the checkout."""
    import vllm  # noqa: PLC0415

    package_path = Path(vllm.__file__).resolve()
    for source_root in source_roots:
        if package_path.is_relative_to(source_root.resolve()):
            raise ValidationFailure(
                f"vllm imported from source checkout {package_path}, not the wheel"
            )
    return package_path


def probe_installed(args: argparse.Namespace) -> int:
    import torch  # noqa: PLC0415

    result: dict[str, Any] = {
        "environment": {
            "python_version": platform.python_version(),
            "machine": platform.machine(),
            "torch_version": torch.__version__,
            "torch_cuda_runtime": torch.version.cuda,
        },
        "hardware": {},
        "gates": {
            DISTRIBUTION_GATE: gate("not_run"),
            "vllm._C": gate("not_run"),
            GRUG_ARCHITECTURE: gate("not_run"),
            CUMEM_GATE: gate("not_run"),
        },
    }
    failed = False
    try:
        package_path = installed_vllm_path(
            args.package_source_root, args.validation_source_root
        )
        result["environment"]["vllm_package_path"] = str(package_path)
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
        result["gates"][DISTRIBUTION_GATE] = gate("passed")
    except Exception as exc:
        failed = True
        result["gates"][DISTRIBUTION_GATE] = gate("failed", str(exc))

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
            result["gates"][CUMEM_GATE] = gate(
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
            result["gates"][CUMEM_GATE] = {
                "status": "passed",
                "allocated_bytes": usage,
                "checksum": checksum,
            }
    except ModuleNotFoundError as exc:
        failed = True
        result["gates"][CUMEM_GATE] = gate("absent", repr(exc))
    except Exception as exc:
        failed = True
        result["gates"][CUMEM_GATE] = gate("failed", repr(exc))

    write_json(args.output, result)
    return int(failed)


def run_wheel_tests(args: argparse.Namespace) -> int:
    """Run checkout test modules against the already imported wheel package."""
    validation_source_root = args.validation_source_root.resolve()
    try:
        package_path = installed_vllm_path(
            args.package_source_root, validation_source_root
        )
    except ValidationFailure as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"::: wheel tests import vllm from {package_path}", flush=True)

    # `vllm` is now fixed in sys.modules with its site-packages __path__. Add the
    # checkout only so pytest can import the candidate commit's `tests` package.
    # Keep the process working directory outside the checkout: model inspection
    # launches fresh Python processes, and their first import path is the working
    # directory rather than this process's already-populated sys.modules.
    import pytest  # noqa: PLC0415

    sys.path.insert(0, str(validation_source_root))
    pytest_args = args.pytest_args
    if pytest_args and pytest_args[0] == "--":
        pytest_args = pytest_args[1:]
    if args.exclude_test:
        pytest_args.extend(
            ["-k", " and ".join(f"not {name}" for name in args.exclude_test)]
        )
    return pytest.main(pytest_args)


def source_node_id(source_root: Path, node_id: str) -> str:
    """Return a pytest node id whose file is absolute under ``source_root``."""
    relative_path, separator, selection = node_id.partition("::")
    absolute_path = source_root / relative_path
    if not separator:
        return str(absolute_path)
    return f"{absolute_path}::{selection}"


def emit_result(result: dict[str, Any]) -> None:
    payload = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    print(VALIDATION_SENTINEL + base64.b64encode(payload).decode(), flush=True)


def install_wheel_environment(
    workdir: Path,
    wheel: Path,
    config: dict[str, Any],
    environment: dict[str, str],
) -> Path:
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
    return python


def run_installed_probe(
    python: Path,
    workdir: Path,
    package_source_root: Path,
    validation_source_root: Path,
    candidate: dict[str, Any],
    config: dict[str, Any],
    expected_validation: dict[str, Any],
    environment: dict[str, str],
) -> tuple[dict[str, Any], int]:
    probe_path = workdir / "installed-probe.json"
    return_code = run_command(
        [
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
            "--package-source-root",
            str(package_source_root),
            "--validation-source-root",
            str(validation_source_root),
            "--output",
            str(probe_path),
        ],
        cwd=workdir,
        environment=environment,
    )
    if not probe_path.is_file():
        raise ValidationFailure("installed-wheel probe produced no result")
    return load_json(probe_path), return_code


def run_source_suite(
    python: Path,
    workdir: Path,
    package_source_root: Path,
    validation_source_root: Path,
    environment: dict[str, str],
) -> None:
    command = [
        str(python),
        str(Path(__file__).resolve()),
        "run-wheel-tests",
        "--package-source-root",
        str(package_source_root),
        "--validation-source-root",
        str(validation_source_root),
    ]
    for test in SOURCE_TEST_EXCLUDES:
        command.extend(["--exclude-test", test])
    command.extend(
        [
            "--",
            "-v",
            *(
                source_node_id(validation_source_root, test)
                for test in SOURCE_TESTS
            ),
        ]
    )
    return_code = run_command(command, cwd=workdir, environment=environment)
    if return_code != 0:
        raise ValidationFailure(f"source behavior tests exited with code {return_code}")


def run_serving_smoke(
    python: Path,
    workdir: Path,
    validation_source_root: Path,
    model: str,
    spec: str,
    environment: dict[str, str],
) -> dict[str, Any]:
    result_path = workdir / "serve-smoke.json"
    return_code = run_command(
        [
            str(python),
            str(validation_source_root / "infra/nightly/gpu_serve_smoke.py"),
            "--model",
            model,
            "--spec",
            str(validation_source_root / spec),
            "--result-json",
            str(result_path),
        ],
        cwd=workdir,
        environment=environment,
    )
    if not result_path.is_file():
        raise ValidationFailure("serve smoke produced no result")
    metrics = load_json(result_path)
    if return_code != 0:
        raise ValidationFailure(f"serve smoke exited with code {return_code}")
    return metrics


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
    package_source_root = Path(__file__).resolve().parents[2]
    validation_source_root = args.validation_source_root.resolve()

    try:
        if not validation_source_root.is_dir():
            raise ValidationFailure(
                "candidate validation source root does not exist: "
                f"{validation_source_root}"
            )
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
                result["gates"][WHEEL_SHA_GATE] = gate(
                    "failed", f"{actual_digest} != {expected_digest}"
                )
                raise ValidationFailure(
                    "downloaded wheel SHA-256 does not match manifest"
                )
            result["gates"][WHEEL_SHA_GATE] = gate("passed")

            environment = os.environ.copy()
            environment.update(
                {
                    "UV_LINK_MODE": "copy",
                    "VLLM_USE_FLASHINFER_SAMPLER": "0",
                }
            )
            python = install_wheel_environment(
                workdir, wheel, config, environment
            )
            probe, probe_return_code = run_installed_probe(
                python,
                workdir,
                package_source_root,
                validation_source_root,
                candidate,
                config,
                expected_validation,
                environment,
            )
            result["environment"].update(probe.get("environment", {}))
            result["hardware"].update(probe.get("hardware", {}))
            result["gates"].update(probe.get("gates", {}))
            if probe_return_code != 0:
                raise ValidationFailure(
                    f"installed-wheel probe exited with code {probe_return_code}"
                )

            if expected_validation["run_source_tests"]:
                try:
                    run_source_suite(
                        python,
                        workdir,
                        package_source_root,
                        validation_source_root,
                        environment,
                    )
                except ValidationFailure as exc:
                    result["gates"][SOURCE_TESTS_GATE] = gate("failed", str(exc))
                    raise
                result["gates"][SOURCE_TESTS_GATE] = gate("passed")
            else:
                result["gates"][SOURCE_TESTS_GATE] = gate(
                    "not_run", "source behavior suite runs on the H100 lane"
                )

            try:
                serve_metrics = run_serving_smoke(
                    python,
                    workdir,
                    validation_source_root,
                    args.model,
                    expected_validation["spec"],
                    environment,
                )
            except ValidationFailure as exc:
                result["gates"][SERVE_GATE] = gate("failed", str(exc))
                raise
            result["gates"][SERVE_GATE] = {
                "status": "passed",
                "metrics": serve_metrics,
            }

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
    validate_parser.add_argument(
        "--validation-source-root", type=Path, required=True
    )

    probe_parser = subparsers.add_parser("probe-installed")
    probe_parser.add_argument("--distribution", required=True)
    probe_parser.add_argument("--version", required=True)
    probe_parser.add_argument("--torch-version", required=True)
    probe_parser.add_argument("--cuda-runtime", required=True)
    probe_parser.add_argument("--compute-capability", required=True)
    probe_parser.add_argument("--package-source-root", type=Path, required=True)
    probe_parser.add_argument("--validation-source-root", type=Path, required=True)
    probe_parser.add_argument("--output", type=Path, required=True)

    tests_parser = subparsers.add_parser("run-wheel-tests")
    tests_parser.add_argument("--package-source-root", type=Path, required=True)
    tests_parser.add_argument("--validation-source-root", type=Path, required=True)
    tests_parser.add_argument("--exclude-test", action="append", default=[])
    tests_parser.add_argument("pytest_args", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "probe-installed":
        return probe_installed(args)
    if args.command == "run-wheel-tests":
        return run_wheel_tests(args)
    if args.command == "validate":
        return validate(args)
    raise AssertionError(f"unhandled command {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
