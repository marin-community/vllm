# SPDX-License-Identifier: Apache-2.0
# Copyright The Marin Authors

"""Native decode kernels at the Snowball and Hero query/KV head geometries."""

import pytest
import torch

from vllm import envs
from vllm.platforms import current_platform

HEAD_SIZE = 128
PAGE_SIZE = 32
LONG_CONTEXT = 196_608
LONG_QUERY_SCALE = 3.0
SM100_ONLY = pytest.mark.skipif(
    not current_platform.is_device_capability(100),
    reason="Uses the SM100 TRT-LLM long-history decode kernel",
)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
@pytest.mark.parametrize(
    "query_heads,kv_heads,batch,length,page_size",
    [
        (20, 5, 1, 2049, PAGE_SIZE),
        (20, 5, 4, 8208, PAGE_SIZE),
        (48, 6, 1, 2049, PAGE_SIZE),
        (48, 6, 4, 8208, PAGE_SIZE),
        pytest.param(48, 6, 2, LONG_CONTEXT, 16, marks=SM100_ONLY),
        pytest.param(48, 6, 2, LONG_CONTEXT, PAGE_SIZE, marks=SM100_ONLY),
    ],
)
@torch.inference_mode()
def test_native_gqa_decode_matches_reference(
    query_heads: int,
    kv_heads: int,
    batch: int,
    length: int,
    page_size: int,
    monkeypatch,
    record_property,
) -> None:
    """Distinct cached heads must serve the right queries and sequence histories."""
    flashinfer = pytest.importorskip("flashinfer")
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    generator = torch.Generator(device="cuda").manual_seed(42)
    pages = (length + page_size - 1) // page_size
    long_history = length == LONG_CONTEXT
    page_stride = pages if long_history else 1
    total_pages = pages + (batch - 1) * page_stride + 1
    history = torch.randn(
        total_pages * page_size,
        kv_heads,
        2 * HEAD_SIZE,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    cache = history.reshape(-1, page_size, kv_heads, 2 * HEAD_SIZE)
    cache = cache.transpose(1, 2).contiguous()
    query = torch.randn(
        batch, query_heads, HEAD_SIZE,
        dtype=torch.bfloat16, device="cuda", generator=generator,
    )
    if long_history:
        # Sharper attention makes split-KV merge errors visible at the same tolerance.
        query.mul_(LONG_QUERY_SCALE)
    lengths = [length - index * 37 for index in range(batch)]
    block_tables = torch.arange(pages, dtype=torch.int32, device="cuda")[None]
    request_starts = torch.arange(batch, dtype=torch.int32, device="cuda") * page_stride
    block_tables = block_tables + request_starts[:, None]
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
        counts = [(seq_len + page_size - 1) // page_size for seq_len in lengths]
        indptr = torch.tensor([0] + counts, dtype=torch.int32, device="cuda").cumsum(
            0, dtype=torch.int32
        )
        indices = torch.cat([row[:count] for row, count in zip(block_tables, counts)])
        tails = torch.tensor(
            [(seq_len - 1) % page_size + 1 for seq_len in lengths],
            dtype=torch.int32, device="cuda",
        )
        wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            workspace, "HND", use_tensor_cores=True
        )
        wrapper.plan(
            indptr, indices, tails, query_heads, kv_heads, HEAD_SIZE, page_size,
            q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16,
        )
        output = wrapper.run(query, cache.split(HEAD_SIZE, dim=-1))
    maximum_error = 0.0
    for index, seq_len in enumerate(lengths):
        start = index * page_stride * page_size
        sequence = history[start : start + seq_len]
        keys = sequence[:, :, :HEAD_SIZE].float().transpose(0, 1)
        values = sequence[:, :, HEAD_SIZE:].float().transpose(0, 1)
        grouped_queries = query[index].float().reshape(kv_heads, -1, HEAD_SIZE)
        scores = grouped_queries @ keys.transpose(1, 2)
        expected = (torch.softmax(scores * HEAD_SIZE**-0.5, dim=-1) @ values).reshape(
            query_heads, HEAD_SIZE
        )
        error = (output[index].float() - expected).abs().max().item()
        maximum_error = max(maximum_error, error)
        torch.testing.assert_close(
            output[index].float(), expected, atol=0.01, rtol=0.01
        )
    record_property("max_abs_error", maximum_error)
