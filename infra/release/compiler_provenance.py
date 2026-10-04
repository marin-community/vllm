# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Install fixed compiler components and record the actual build inputs."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def install_compiler(inputs: dict, cuda_home: Path) -> dict:
    architecture = platform.machine()
    packages = inputs["platforms"][architecture]
    version = inputs["compiler_version"]
    with tempfile.TemporaryDirectory(prefix="marin-cuda-inputs-") as directory:
        requirements = Path(directory) / "compiler.txt"
        requirements.write_text(
            "".join(
                f"{item['name']} @ {item['url']} --hash=sha256:{item['sha256']}\n"
                for item in packages
            )
        )
        subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                sys.executable,
                "--no-deps",
                "--require-hashes",
                "-r",
                str(requirements),
            ],
            check=True,
        )

    # Keep the full digest-pinned toolkit prefix so native projects which use
    # /usr/local/cuda directly also receive the fixed compiler. Only files owned
    # by the six pinned compiler/runtime distributions are replaced.
    cuda_home.mkdir(parents=True, exist_ok=True)
    if not (cuda_home / "lib").exists():
        if (cuda_home / "lib64").exists():
            (cuda_home / "lib").symlink_to("lib64", target_is_directory=True)
        else:
            (cuda_home / "lib").mkdir()
    if not (cuda_home / "lib64").exists():
        (cuda_home / "lib64").symlink_to("lib", target_is_directory=True)

    files = {}
    for item in packages:
        distribution = importlib.metadata.distribution(item["name"])
        if distribution.version != version:
            raise RuntimeError(
                f"{item['name']} is {distribution.version}, expected {version}"
            )
        for entry in distribution.files or ():
            relative = Path(entry)
            try:
                target_relative = relative.relative_to("nvidia/cu13")
            except ValueError:
                continue
            if target_relative.parts[:2] == ("cuda_cccl", "include"):
                target_relative = Path("include/cccl", *target_relative.parts[2:])
            elif target_relative.parts[:2] == ("cuda_cccl", "lib"):
                target_relative = Path("lib", *target_relative.parts[2:])
            elif target_relative.parts[0] not in {"bin", "include", "lib", "nvvm"}:
                continue
            if ".." in target_relative.parts:
                raise RuntimeError(f"invalid compiler package path: {entry}")
            source = Path(distribution.locate_file(entry))
            destination = cuda_home / target_relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.is_symlink():
                destination.unlink()
            shutil.copy2(source, destination)
            files[str(target_relative)] = sha256(destination)

    runtime_link = cuda_home / "lib/libcudart.so"
    if runtime_link.is_symlink():
        runtime_link.unlink()
    if not runtime_link.exists():
        runtime_link.symlink_to("libcudart.so.13")

    tools = {}
    for name in ("nvcc", "ptxas"):
        executable = cuda_home / "bin" / name
        output = subprocess.check_output([str(executable), "--version"], text=True)
        if f"V{version}" not in output:
            raise RuntimeError(f"unexpected {name} version: {output}")
        tools[name] = {
            "path": str(executable),
            "sha256": sha256(executable),
            "version_output": output,
        }
    # Exercise headers, the compiler front end and both target assemblers before
    # the expensive wheel build. This does not claim GPU numerical validation.
    with tempfile.TemporaryDirectory(prefix="marin-cuda-probe-") as directory:
        source = Path(directory) / "probe.cu"
        source.write_text(
            "#include <cuda/std/array>\n"
            'extern "C" __global__ void probe(float* output, const float* input) {\n'
            " int i = threadIdx.x + blockIdx.x * blockDim.x;\n"
            " if (i < 32) { if (input[i] > 0) output[i] = input[i] * 2;\n"
            " else output[i] = -input[i]; }\n}\n"
        )
        output = Path(directory) / "probe.fatbin"
        subprocess.run(
            [
                str(cuda_home / "bin/nvcc"),
                "--fatbin",
                "--generate-code",
                "arch=compute_90,code=sm_90",
                "--generate-code",
                "arch=compute_100,code=sm_100",
                str(source),
                "--output-file",
                str(output),
            ],
            check=True,
        )
        compile_probe = {
            "status": "passed",
            "sm_targets": ["9.0", "10.0"],
            "sha256": sha256(output),
        }
    return {
        "schema_version": 1,
        "architecture": architecture,
        "cuda_home": str(cuda_home),
        "compiler_version": version,
        "inputs": packages,
        "files": files,
        "tools": tools,
        "compile_probe": compile_probe,
        "fix_reference": inputs["fix_reference"],
        "scope": (
            "native build compiler inputs; supplier and runtime JIT artifacts "
            "need separate evidence"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--cuda-home", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    record = install_compiler(json.loads(args.inputs.read_text()), args.cuda_home)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(
        f"Recorded {record['architecture']} CUDA compiler {record['compiler_version']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
