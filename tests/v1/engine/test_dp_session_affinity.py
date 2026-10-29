# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

from vllm import SamplingParams
from vllm.v1.engine import EngineCoreOutputs, EngineCoreRequest
from vllm.v1.engine.core_client import DPLBAsyncMPClient


def _client() -> DPLBAsyncMPClient:
    client = object.__new__(DPLBAsyncMPClient)
    client.client_count = 4
    client.reqs_in_flight = {}
    client.engine_inflight = Counter()
    client.core_engines = [bytes([i, 0]) for i in range(16)]
    client.lb_engines = [[0, 0, 0.0] for _ in client.core_engines]
    client.eng_start_index = 0
    return client


def _request(request_id: str, session_id: str | None) -> EngineCoreRequest:
    return EngineCoreRequest(
        request_id=request_id,
        prompt_token_ids=[1, 2, 3],
        mm_features=None,
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        arrival_time=0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        session_id=session_id,
    )


def test_session_stays_on_one_engine_across_clients_and_load():
    first = _client()
    second = _client()
    second.eng_start_index = 7
    second.lb_engines = [[100, 100, 0.9] for _ in second.core_engines]
    initial = first.get_core_engine_for_request(_request("first", "conversation-α"))
    for index in range(5):
        assert (
            second.get_core_engine_for_request(
                _request(f"branch-{index}", "conversation-α")
            )
            == initial
        )

    # Session routing must retain the normal completion/abort bookkeeping.
    asyncio.run(
        DPLBAsyncMPClient.process_engine_outputs(
            first, EngineCoreOutputs(finished_requests={"first"})
        )
    )
    assert first.engine_inflight[initial] == 0
    assert "first" not in first.reqs_in_flight


def test_explicit_rank_overrides_session_and_missing_session_uses_load():
    client = _client()
    request = _request("explicit", "conversation-α")
    request.data_parallel_rank = 9
    assert client.get_core_engine_for_request(request) == client.core_engines[9]

    client.lb_engines = [[2, 1, 0.0] for _ in client.core_engines]
    client.lb_engines[4] = [0, 0, 0.0]
    assert (
        client.get_core_engine_for_request(_request("unaffiliated", None))
        == client.core_engines[4]
    )


def test_session_assignment_survives_python_hash_randomization():
    script = (
        "import runpy; "
        f"ns=runpy.run_path({str(Path(__file__).resolve())!r}); "
        "client=ns['_client'](); "
        "request=ns['_request']('request', 'conversation-α'); "
        "print(client.get_core_engine_for_request(request).hex())"
    )
    choices = [
        subprocess.check_output(
            [sys.executable, "-c", script],
            env=dict(os.environ, PYTHONHASHSEED=seed),
            text=True,
        )
        .strip()
        .splitlines()[-1]
        for seed in ("17", "983")
    ]
    assert choices[0] == choices[1]
