# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the nightly's gate. Runs on CPU in seconds; no GPU, no vLLM.

The serving half of gpu_serve_smoke.py needs an accelerator and is exercised by the
nightly itself. The half that decides pass/fail does not, so it is tested here: the
client's reading of an OpenAI response, and every way a run can miss the spec.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest
from gpu_serve_smoke import PROMPTS, GateSpec, SmokeResult, failures, run_prompts

SHIPPED_SPEC = Path(__file__).parent / "specs" / "qwen3-0.6b-h100.json"

SPEC = GateSpec(
    model="test-model",
    expected_prompts=len(PROMPTS),
    min_completion_tokens=4,
    min_output_tokens_per_second=1.0,
)


def serve_stub(answer: str, tokens: int) -> HTTPServer:
    """Run an OpenAI-shaped chat endpoint that always returns `answer`."""

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            body = json.dumps(
                {
                    "choices": [{"message": {"content": answer}}],
                    "usage": {"completion_tokens": tokens},
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


@pytest.fixture
def stub():
    servers = []

    def start(answer: str, tokens: int = 16) -> str:
        server = serve_stub(answer, tokens)
        servers.append(server)
        return f"http://127.0.0.1:{server.server_port}/v1"

    yield start
    for server in servers:
        server.shutdown()


def test_run_prompts_measures_every_prompt_served(stub):
    result = run_prompts(stub("a real answer", tokens=16), "test-model")

    assert result.completions == len(PROMPTS)
    assert result.min_completion_tokens == 16
    assert result.output_tokens_per_second > 0


def test_run_prompts_rejects_a_blank_answer(stub):
    # A server that returns 200 with nothing in it is the failure the nightly exists
    # to catch, and it must not be scored as a pass.
    with pytest.raises(RuntimeError, match="empty answer"):
        run_prompts(stub("   ", tokens=16), "test-model")


def test_failures_accepts_a_run_that_clears_the_spec():
    result = SmokeResult(
        completions=len(PROMPTS),
        min_completion_tokens=16,
        output_tokens_per_second=50.0,
    )

    assert failures(result, SPEC) == []


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        (
            SmokeResult(
                completions=len(PROMPTS),
                min_completion_tokens=16,
                output_tokens_per_second=0.5,
            ),
            "throughput",
        ),
        (
            SmokeResult(
                completions=len(PROMPTS),
                min_completion_tokens=1,
                output_tokens_per_second=50.0,
            ),
            "shortest answer",
        ),
        (
            SmokeResult(
                completions=1, min_completion_tokens=16, output_tokens_per_second=50.0
            ),
            "prompts",
        ),
    ],
    ids=["slow_decode", "truncated_answers", "missing_prompts"],
)
def test_failures_reports_each_way_a_run_misses_the_spec(result, expected):
    reported = failures(result, SPEC)

    assert len(reported) == 1
    assert expected in reported[0]


def test_shipped_spec_matches_the_gate_it_is_read_by():
    spec = GateSpec.load(SHIPPED_SPEC)

    assert spec.expected_prompts == len(PROMPTS)
