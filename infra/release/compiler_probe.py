"""Exercise real C++, NVCC and Rust cache compatibility on a native runner."""

import argparse
import hashlib
import json
import os
import subprocess
import time
from functools import partial
from pathlib import Path


def run(command: list[str], environment: dict[str, str]) -> None:
    subprocess.run(command, env=environment, check=True)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compile_case(
    case: str, command: list[str], expected: str, *,
    environment: dict[str, str], compiled: Path, output_arguments: list[str],
) -> dict:
    run(["sccache", "--zero-stats"], environment)
    started = time.monotonic()
    run(["sccache", *command, *output_arguments], environment)
    stats = json.loads(subprocess.check_output(
        ["sccache", "--show-stats", "--stats-format=json"], env=environment,
    ))["stats"]
    hits = sum(stats["cache_hits"]["counts"].values())
    misses = sum(stats["cache_misses"]["counts"].values())
    assert (hits, misses) == ((1, 0) if expected == "hit" else (0, 1)), (
        case, command, stats,
    )
    cached = digest(compiled)
    # Compile independently at the same output path and compare bytes.
    run([*command, *output_arguments], environment)
    independent = digest(compiled)
    assert cached == independent, (case, cached, independent)
    return {
        "case": case, "hits": hits, "misses": misses,
        "sha256": cached, "independent_sha256": independent,
        "seconds": round(time.monotonic() - started, 3), "stats": stats,
    }


def probe(expect: str, variant: str, output: Path) -> None:
    # Stable base inputs prove restore hits. Each run's mutations are new so
    # repeated warm probes can continue proving selective misses.
    changed_value = 8 + int.from_bytes(hashlib.sha256(variant.encode()).digest()[:3])
    source = Path("source")
    source.mkdir()
    Path("objects").mkdir()
    header = source / "input.h"
    header.write_text("#define VALUE 7\n")
    cpp = source / "input.cpp"
    cpp.write_text('#include "input.h"\nint value() { return VALUE; }\n')
    cuda = source / "input.cu"
    cuda.write_text(
        '#include "input.h"\n__global__ void value(int *p) { *p = VALUE; }\n'
    )
    rust = source / "input.rs"
    rust.write_text("pub fn value() -> u32 { 7 }\n")
    compilers = {
        "cpp": ["g++", "-O2", "-c", str(cpp)],
        # Keep intermediate names stable for this byte-parity fixture. The
        # production recipe's compiler flags remain unchanged.
        "cuda": ["nvcc", "--objdir-as-tempdir", "-O2", "-c", str(cuda)],
        "rust": [
            "rustc", "--crate-name=cache_probe", "--crate-type=rlib",
            "--emit=dep-info,link", str(rust),
        ],
    }
    records = []
    for language, base in compilers.items():
        environment = dict(os.environ)
        environment["SCCACHE_DIR"] = (
            "/root/.cache/sccache-rust" if language == "rust"
            else "/root/.cache/sccache"
        )
        suffix = "rlib" if language == "rust" else "o"
        compiled = Path(f"objects/{language}.{suffix}")
        output_arguments = ["-o", str(compiled)]
        if language == "rust":
            compiled = Path("objects/libcache_probe.rlib")
            output_arguments = ["--out-dir=objects"]
        compile_one = partial(
            compile_case, environment=environment, compiled=compiled,
            output_arguments=output_arguments,
        )
        cases = [compile_one("fresh-builder", base, expect)]
        cases.append(compile_one("repeat", base, "hit"))
        if language in ("cpp", "cuda"):
            header.write_text(f"#define VALUE {changed_value}\n")
            cases.append(compile_one("header-change", base, "miss"))
            header.write_text("#define VALUE 7\n")
            native_source = cpp if language == "cpp" else cuda
            original = native_source.read_text()
            native_source.write_text(
                original + f"int extra_value() {{ return {changed_value}; }}\n"
            )
            cases.append(compile_one("source-change", base, "miss"))
            native_source.write_text(original)
        else:
            rust.write_text(f"pub fn value() -> u32 {{ {changed_value} }}\n")
            cases.append(compile_one("source-change", base, "miss"))
            rust.write_text("pub fn value() -> u32 { 7 }\n")
        changed_flag = [*base, "-g", f"-DPROBE_VARIANT={changed_value}"]
        if language == "rust":
            changed_flag = [
                *base, "-C", "opt-level=2", "-C", f"metadata=probe_{changed_value}",
            ]
        cases.append(compile_one("flag-change", changed_flag, "miss"))
        if language == "cpp":
            changed_compiler = [
                "clang++", *base[1:], f"-DPROBE_VARIANT={changed_value}",
            ]
            cases.append(compile_one("compiler-change", changed_compiler, "miss"))
        cases.append(compile_one("return-to-original", base, "hit"))
        records.extend({"language": language, **case} for case in cases)
        run(["sccache", "--stop-server"], environment)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "architecture": os.uname().machine,
        "variant": variant,
        "toolchains": {name: subprocess.check_output(
            [command[0], "--version"], text=True,
        ) for name, command in compilers.items()},
        "records": records,
    }, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expect", choices=("hit", "miss"), required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    probe(args.expect, args.variant, args.output)


if __name__ == "__main__":
    main()
