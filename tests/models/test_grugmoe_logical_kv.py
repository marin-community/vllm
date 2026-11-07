# SPDX-License-Identifier: Apache-2.0
# Copyright The Marin Authors

import pytest
import torch

from vllm import envs
from vllm.platforms import current_platform

HEAD_SIZE = 128
PAGE_SIZE = 32


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
@pytest.mark.parametrize(
    "query_heads,kv_heads",
    [(20, 5), (48, 6), (48, 12), (24, 3), (24, 6), (8, 1), (8, 2)],
)
@pytest.mark.parametrize("batch,length", [(1, 2049), (4, 8208)])
@torch.inference_mode()
def test_native_gqa_decode_matches_reference(
    query_heads: int, kv_heads: int, batch: int, length: int, monkeypatch,
    record_property,
) -> None:
    """Distinct cached heads must serve the right queries and sequence histories."""
    flashinfer = pytest.importorskip("flashinfer")
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    generator = torch.Generator(device="cuda").manual_seed(42)
    pages = (length + PAGE_SIZE - 1) // PAGE_SIZE
    history = torch.randn(
        (pages + batch) * PAGE_SIZE,
        kv_heads,
        2 * HEAD_SIZE,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    cache = history.reshape(-1, PAGE_SIZE, kv_heads, 2 * HEAD_SIZE)
    cache = cache.transpose(1, 2).contiguous()
    query = torch.randn(
        batch, query_heads, HEAD_SIZE,
        dtype=torch.bfloat16, device="cuda", generator=generator,
    )
    lengths = [length - index * 37 for index in range(batch)]
    block_tables = torch.arange(pages, dtype=torch.int32, device="cuda")[None]
    block_tables = block_tables + torch.arange(
        batch, dtype=torch.int32, device="cuda"
    )[:, None]
    workspace = torch.empty(
        envs.VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE, dtype=torch.uint8, device="cuda"
    )
    if current_platform.is_device_capability(100):
        output = flashinfer.decode.trtllm_batch_decode_with_kv_cache(
            query=query,
            kv_cache=cache.split(HEAD_SIZE, dim=-1),
            workspace_buffer=workspace,
            block_tables=block_tables,
            seq_lens=torch.tensor(lengths, dtype=torch.int32, device="cuda"),
            max_seq_len=length,
            bmm1_scale=HEAD_SIZE**-0.5,
            bmm2_scale=1.0,
            kv_layout="HND",
            backend="trtllm-gen",
        )
    else:
        counts = [(seq_len + PAGE_SIZE - 1) // PAGE_SIZE for seq_len in lengths]
        indptr = torch.tensor([0] + counts, dtype=torch.int32, device="cuda").cumsum(
            0, dtype=torch.int32
        )
        indices = torch.cat([row[:count] for row, count in zip(block_tables, counts)])
        tails = torch.tensor(
            [(seq_len - 1) % PAGE_SIZE + 1 for seq_len in lengths],
            dtype=torch.int32, device="cuda",
        )
        wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            workspace, "HND", use_tensor_cores=True
        )
        wrapper.plan(
            indptr, indices, tails, query_heads, kv_heads, HEAD_SIZE, PAGE_SIZE,
            q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16,
        )
        output = wrapper.run(query, cache.split(HEAD_SIZE, dim=-1))
    head = torch.arange(query_heads, device="cuda") // (query_heads // kv_heads)
    maximum_error = 0.0
    for index, seq_len in enumerate(lengths):
        sequence = history[index * PAGE_SIZE : index * PAGE_SIZE + seq_len]
        keys = sequence[:, head, :HEAD_SIZE].float().transpose(0, 1)
        values = sequence[:, head, HEAD_SIZE:].float().transpose(0, 1)
        scores = query[index].float()[:, None] @ keys.transpose(1, 2)
        expected = (
            torch.softmax(scores * HEAD_SIZE**-0.5, dim=-1) @ values
        ).squeeze(1)
        error = (output[index].float() - expected).abs().max().item()
        maximum_error = max(maximum_error, error)
        torch.testing.assert_close(
            output[index].float(), expected, atol=0.01, rtol=0.01
        )
    record_property("max_abs_error", maximum_error)
