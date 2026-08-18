# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Serve a model with this fork's OpenAI server and gate the result.

Runs inside an Iris accelerator job against either a vLLM built from the fork
commit under test or a cleanly installed release wheel. Boots `vllm serve`, waits
for the server to report ready, sends a fixed prompt set, and compares the run
against a checked-in spec: every prompt must return a non-empty answer of at
least `min_completion_tokens`, and decode throughput must clear the spec's floor.

Deliberately stdlib-only and it never imports vllm: it talks to the server over HTTP,
so a broken import in the model code surfaces as a serve failure rather than as a
crash in the harness.

    python infra/nightly/gpu_serve_smoke.py --spec <spec.json>
    python infra/nightly/gpu_serve_smoke.py --spec <spec.json> --record
"""

import argparse
import contextlib
import json
import logging
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger("gpu_serve_smoke")

# (prompt, a substring the answer must contain -- None where no single answer is right).
# Held fixed: the spec's throughput floor is only comparable across runs that ask the
# model for the same work. Decoding is greedy, so the expected substrings are
# deterministic, and they are what makes this a gate rather than a liveness probe: a
# server that has lost its kernels still returns tokens, it just stops returning correct
# ones.
#
# Every prompt here is one Qwen3-0.6B finishes on its own. An arithmetic prompt ("What
# is 17 + 25? Answer with the number only.") was dropped for the opposite behavior: the
# format constraint fights the reasoning, and the model second-guesses itself until it
# hits whatever cap it is given -- 512, then 1536 -- so its answer was always a
# truncated thought, which cannot tell us whether the model finished correctly.
PROMPTS = (
    ("What is the capital of France? Answer with one word.", "paris"),
    ("Write a Python function that reverses a string.", "def"),
    ("In one sentence, what is a large language model?", None),
    ("Name three primary colors.", None),
)
# Qwen3 thinks before it answers and the budget has to cover both -- thinking is what
# these models do in production, so this accommodates it rather than turning it off. At
# 64 tokens the whole budget went to the <think> block, every completion came back
# truncated, and the gate was passing on token count alone. 1024 clears the longest of
# the prompts above (474 tokens observed) with room to spare.
MAX_TOKENS = 1024
TEMPERATURE = 0.0
REQUEST_TIMEOUT = 120.0
READY_POLL_INTERVAL = 5.0


@dataclass(frozen=True)
class GateSpec:
    """The gate a nightly run must clear."""

    model: str
    expected_prompts: int
    min_completion_tokens: int
    min_output_tokens_per_second: float

    @classmethod
    def load(cls, path: Path) -> "GateSpec":
        raw = json.loads(path.read_text())
        fields = {f: raw[f] for f in cls.__dataclass_fields__}
        return cls(**fields)


@dataclass(frozen=True)
class SmokeResult:
    """What a run actually produced."""

    completions: int
    min_completion_tokens: int
    output_tokens_per_second: float


def post_json(url: str, payload: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def server_command(
    model: str,
    port: int,
    attention_backend: str | None,
    tensor_parallel_size: int = 1,
) -> list[str]:
    """Build the OpenAI server command for this smoke."""
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model,
        "--port",
        str(port),
        "--max-model-len",
        "4096",
        "--tensor-parallel-size",
        str(tensor_parallel_size),
    ]
    if attention_backend is not None:
        command.extend(("--attention-backend", attention_backend))
    return command


def pick_free_port() -> int:
    """Return a currently-free localhost TCP port.

    The GB200 validation lane runs with host networking, so a fixed serve port
    collides with anything already bound on the node (including a prior run's
    server that outlived its job). Letting the kernel assign a free port avoids
    that class of "address already in use" startup failure.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _terminate_process_group(server: subprocess.Popen) -> None:
    """Stop the server and every process in its group, escalating to SIGKILL."""
    for sig, timeout in ((signal.SIGTERM, 60), (signal.SIGKILL, 60)):
        if server.poll() is not None:
            return
        with contextlib.suppress(ProcessLookupError):
            os.killpg(os.getpgid(server.pid), sig)
        with contextlib.suppress(subprocess.TimeoutExpired):
            server.wait(timeout=timeout)


@contextlib.contextmanager
def serve(
    model: str,
    port: int,
    startup_timeout: float,
    attention_backend: str | None,
    tensor_parallel_size: int,
) -> Iterator[str]:
    """Run `vllm serve` for the duration of the block, yielding its base URL.

    Args:
        model: HF model id to serve.
        port: Port for the OpenAI server to listen on.
        startup_timeout: Seconds to wait for the server to report ready.
        attention_backend: Explicit vLLM attention backend, or auto-selection when
            unset.
        tensor_parallel_size: Number of accelerator devices used by the server.

    Yields:
        The server's OpenAI base URL.

    Raises:
        RuntimeError: If the server exits or fails to become ready in time.
    """
    command = server_command(
        model, port, attention_backend, tensor_parallel_size
    )
    logger.info("starting server: %s", " ".join(command))
    # Own a fresh process group so cleanup reaps vLLM's API-server and engine-core
    # children, not just the launcher. A leaked child keeps the serve port bound and
    # makes the next run on the same node fail with "address already in use".
    server = subprocess.Popen(command, start_new_session=True)
    try:
        base_url = f"http://127.0.0.1:{port}/v1"
        wait_until_ready(server, base_url, startup_timeout)
        yield base_url
    finally:
        _terminate_process_group(server)


def wait_until_ready(server: subprocess.Popen, base_url: str, timeout: float) -> None:
    """Block until the server answers /models, it dies, or `timeout` elapses."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        exit_code = server.poll()
        if exit_code is not None:
            raise RuntimeError(f"vllm serve exited with code {exit_code} at startup")
        try:
            with urllib.request.urlopen(f"{base_url}/models", timeout=10) as response:
                if response.status == 200:
                    logger.info("server ready")
                    return
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            pass
        time.sleep(READY_POLL_INTERVAL)
    raise RuntimeError(f"server was not ready within {timeout:.0f}s")


def run_prompts(base_url: str, model: str) -> SmokeResult:
    """Send every prompt to the served model and measure what comes back.

    Throughput is aggregate decode throughput across the (sequential) prompt set:
    total completion tokens over total wall-clock time spent in the requests.

    Raises:
        RuntimeError: If the server returns an empty or wrong answer for any prompt.
    """
    total_tokens = 0
    per_prompt_tokens = []
    started = time.monotonic()
    for prompt, expected in PROMPTS:
        response = post_json(
            f"{base_url}/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": MAX_TOKENS,
                "temperature": TEMPERATURE,
            },
            timeout=REQUEST_TIMEOUT,
        )
        answer = response["choices"][0]["message"]["content"]
        tokens = response["usage"]["completion_tokens"]
        if not answer or not answer.strip():
            raise RuntimeError(f"empty answer for prompt: {prompt!r}")
        if expected and expected not in answer.lower():
            raise RuntimeError(
                f"answer to {prompt!r} lacks {expected!r}: {answer.strip()!r}"
            )
        logger.info("[%2d tokens] %s -> %s", tokens, prompt, answer.strip()[:80])
        per_prompt_tokens.append(tokens)
        total_tokens += tokens
    elapsed = time.monotonic() - started

    return SmokeResult(
        completions=len(per_prompt_tokens),
        min_completion_tokens=min(per_prompt_tokens),
        output_tokens_per_second=total_tokens / elapsed,
    )


def failures(result: SmokeResult, spec: GateSpec) -> list[str]:
    """Return one message per way `result` misses `spec`; empty means it passed."""
    checks = []
    if result.completions != spec.expected_prompts:
        checks.append(
            f"answered {result.completions} prompts, "
            f"spec expects {spec.expected_prompts}"
        )
    if result.min_completion_tokens < spec.min_completion_tokens:
        checks.append(
            f"shortest answer was {result.min_completion_tokens} tokens, "
            f"spec floor is {spec.min_completion_tokens}"
        )
    if result.output_tokens_per_second < spec.min_output_tokens_per_second:
        checks.append(
            f"decode throughput {result.output_tokens_per_second:.1f} tok/s is below "
            f"the spec floor of {spec.min_output_tokens_per_second:.1f} tok/s"
        )
    return checks


def record(path: Path, spec: GateSpec, result: SmokeResult, model: str) -> None:
    """Overwrite the spec with floors derived from `result`, keeping provenance."""
    document = {
        **asdict(spec),
        "model": model,
        "expected_prompts": result.completions,
        "min_completion_tokens": max(1, result.min_completion_tokens // 2),
        "min_output_tokens_per_second": round(result.output_tokens_per_second / 2, 1),
        "provenance": {
            "recorded_at": datetime.now(UTC).isoformat(),
            "recorded_from": {
                "min_completion_tokens": result.min_completion_tokens,
                "output_tokens_per_second": round(result.output_tokens_per_second, 1),
            },
            "commit": os.environ.get("VLLM_COMMIT", "unknown"),
            "hardware": os.environ.get("SMOKE_HARDWARE", "unknown"),
            "note": (
                "Floors are half the observed values: this is a not-broken gate, not a "
                "performance benchmark. Re-record with --record after an intentional "
                "change to the prompt set or the serving path."
            ),
        },
    }
    path.write_text(json.dumps(document, indent=2) + "\n")
    logger.info("recorded spec to %s", path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True, help="Path to the spec.")
    parser.add_argument("--model", help="Override the model id in the spec.")
    parser.add_argument(
        "--port",
        type=int,
        default=0,
        help="Serve port; 0 (default) lets the kernel pick a free one.",
    )
    parser.add_argument(
        "--attention-backend",
        help="Pass an explicit attention backend to vLLM instead of auto-selecting.",
    )
    parser.add_argument(
        "--startup-timeout",
        type=float,
        default=900.0,
        help="Seconds to wait for the server to become ready.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of accelerator devices used by the server.",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Rewrite the spec from this run instead of gating against it.",
    )
    parser.add_argument(
        "--result-json",
        type=Path,
        help="Write the observed serving metrics as JSON.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    spec = GateSpec.load(args.spec)
    model = args.model or spec.model
    port = args.port or pick_free_port()

    with serve(
        model,
        port,
        args.startup_timeout,
        args.attention_backend,
        args.tensor_parallel_size,
    ) as base_url:
        result = run_prompts(base_url, model)

    logger.info("result: %s", result)

    if args.result_json is not None:
        args.result_json.write_text(json.dumps(asdict(result), indent=2) + "\n")

    if args.record:
        record(args.spec, spec, result, model)
        return 0

    gate_failures = failures(result, spec)
    for failure in gate_failures:
        logger.error("GATE FAILED: %s", failure)
    if gate_failures:
        return 1
    logger.info("gate passed against %s", args.spec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
