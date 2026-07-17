# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import io

import httpx
import numpy as np
import pybase64 as base64
import pytest

from ...utils import VLLM_PATH, RemoteOpenAIServer

MODEL_NAME = "TitanML/tiny-mixtral"

# tiny-mixtral config: 8 local experts, top-2 routing, 2 hidden layers.
# The published config has sliding_window=4096, which produces
# SlidingWindowSpec kv-cache groups; RoutedExpertsManager requires a
# FullAttentionSpec group, so we override sliding_window=null below.
NUM_LOCAL_EXPERTS = 8
NUM_EXPERTS_PER_TOK = 2
NUM_HIDDEN_LAYERS = 2


@pytest.fixture(scope="module")
def server():
    args = [
        "--max-model-len",
        "256",
        "--max-num-seqs",
        "32",
        "--enforce-eager",
        "--enable-return-routed-experts",
        "--hf-overrides",
        '{"sliding_window": null}',
        "--chat-template",
        str(VLLM_PATH / "examples/template_chatml.jinja"),
    ]
    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest.mark.asyncio
async def test_routed_experts(server):
    """Test that /v1/completions returns routed_experts when enabled."""
    async with server.get_async_client() as client:
        result = await client.completions.create(
            model=MODEL_NAME,
            prompt="Hello, world",
            max_tokens=10,
            temperature=0,
            extra_body={"return_token_ids": True},
        )

        choice = result.model_dump()["choices"][0]

        assert choice["routed_experts"] is not None
        assert choice["token_ids"] is not None

        # routed_experts is base64-encoded .npy bytes; decode to ndarray.
        routed_experts = np.load(io.BytesIO(base64.b64decode(choice["routed_experts"])))
        assert routed_experts.ndim == 3
        num_tokens, num_layers, topk = routed_experts.shape
        assert num_tokens > 0
        assert num_layers == NUM_HIDDEN_LAYERS
        assert topk == NUM_EXPERTS_PER_TOK
        assert (routed_experts >= 0).all()
        assert (routed_experts < NUM_LOCAL_EXPERTS).all()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "messages",
    [
        [{"role": "user", "content": "Hello, world"}],
        [
            {"role": "user", "content": "Name a primary color."},
            {"role": "assistant", "content": "Red."},
            {"role": "user", "content": "Name another."},
        ],
    ],
)
async def test_chat_routed_experts_are_causally_aligned_nested_lists(server, messages):
    request = {
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": 4,
        "temperature": 0,
        "ignore_eos": True,
        "return_token_ids": True,
    }
    async with httpx.AsyncClient(timeout=60) as client:
        chat_response = await client.post(
            server.url_for("v1/chat/completions"),
            json=request,
        )
        chat_response.raise_for_status()
        chat = chat_response.json()
        chat_choice = chat["choices"][0]
        prompt_token_ids = chat["prompt_token_ids"]
        generated_token_ids = chat_choice["token_ids"]
        chat_routes = chat_choice["routed_experts"]

        completion_response = await client.post(
            server.url_for("v1/completions"),
            json={
                "model": MODEL_NAME,
                "prompt": prompt_token_ids,
                "max_tokens": 4,
                "temperature": 0,
                "ignore_eos": True,
                "return_token_ids": True,
            },
        )
        completion_response.raise_for_status()
        completion_choice = completion_response.json()["choices"][0]

    completion_routes = np.load(
        io.BytesIO(base64.b64decode(completion_choice["routed_experts"]))
    )
    assert completion_choice["token_ids"] == generated_token_ids
    assert len(chat_routes) == len(generated_token_ids) == 4
    assert chat_routes == completion_routes[-len(generated_token_ids) :].tolist()
    assert len(chat_routes[0]) == NUM_HIDDEN_LAYERS
    assert len(chat_routes[0][0]) == NUM_EXPERTS_PER_TOK
    assert all(
        isinstance(expert_id, int) and 0 <= expert_id < NUM_LOCAL_EXPERTS
        for token_routes in chat_routes
        for layer_routes in token_routes
        for expert_id in layer_routes
    )
