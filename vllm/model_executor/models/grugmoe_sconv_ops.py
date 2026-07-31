# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paged short-convolution kernels for GrugMoE."""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _fused_sconv_kernel(
    x_ptr,
    cache_ptr,
    weight_ptr,
    out_ptr,
    pos_ptr,
    seq_idx_ptr,
    slot_ptr,
    block_table_ptr,
    qstart_ptr,
    T,
    stride_x_t,
    stride_c_blk,
    stride_c_h,
    stride_c_n,
    stride_c_d,
    stride_w_d,
    stride_w_w,
    stride_bt_r,
    MAX_BLOCKS,
    N,
    W: tl.constexpr,
    OFF_S: tl.constexpr,
    WS: tl.constexpr,
    H: tl.constexpr,
    BT: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Insert current inputs and convolve without reading just-written slots."""
    pid_t = tl.program_id(0)
    pid_c = tl.program_id(1)
    toff = pid_t * BT + tl.arange(0, BT)
    coff = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    channels = H * WS
    t_mask = toff < T
    c_mask = coff < channels
    head = tl.minimum(coff // WS, H - 1)
    cache_d = OFF_S + coff % WS

    slot = tl.load(slot_ptr + toff, mask=t_mask, other=-1)
    valid = slot >= 0
    pos = tl.load(pos_ptr + toff, mask=t_mask, other=0)
    req = tl.load(seq_idx_ptr + toff, mask=t_mask, other=0)
    query_start = tl.load(qstart_ptr + toff, mask=t_mask, other=0)
    token_channel_mask = t_mask[:, None] & c_mask[None, :]

    x_value = tl.load(
        x_ptr + toff[:, None] * stride_x_t + coff[None, :],
        mask=token_channel_mask,
    )
    safe_slot = tl.maximum(slot, 0)
    destination = (
        cache_ptr
        + (safe_slot // N)[:, None] * stride_c_blk
        + head[None, :] * stride_c_h
        + (safe_slot % N)[:, None] * stride_c_n
        + cache_d[None, :] * stride_c_d
    )
    tl.store(
        destination,
        x_value,
        mask=token_channel_mask & valid[:, None],
    )

    accumulator = tl.zeros([BT, BLOCK_C], dtype=tl.float32)
    for tap in tl.static_range(W):
        source_position = pos - tap
        source_row = toff - tap
        in_window = valid & (source_position >= 0)
        current_forward = in_window & (source_row >= query_start)
        cached = in_window & (source_row < query_start)

        safe_row = tl.maximum(source_row, 0)
        current_value = tl.load(
            x_ptr + safe_row[:, None] * stride_x_t + coff[None, :],
            mask=c_mask[None, :] & current_forward[:, None],
            other=0.0,
        ).to(tl.float32)

        safe_source = tl.maximum(source_position, 0)
        logical_block = tl.minimum(safe_source // N, MAX_BLOCKS - 1)
        physical_block = tl.load(
            block_table_ptr + req * stride_bt_r + logical_block,
            mask=cached,
            other=0,
        ).to(tl.int64)
        cache_source = (
            cache_ptr
            + physical_block[:, None] * stride_c_blk
            + head[None, :] * stride_c_h
            + (safe_source % N)[:, None] * stride_c_n
            + cache_d[None, :] * stride_c_d
        )
        cached_value = tl.load(
            cache_source,
            mask=c_mask[None, :] & cached[:, None],
            other=0.0,
        ).to(tl.float32)
        tap_weight = tl.load(
            weight_ptr + coff * stride_w_d + tap * stride_w_w,
            mask=c_mask,
            other=0.0,
        ).to(tl.float32)
        accumulator += (current_value + cached_value) * tap_weight[None, :]

    tl.store(
        out_ptr + toff[:, None] * stride_x_t + coff[None, :],
        accumulator.to(out_ptr.dtype.element_ty),
        mask=token_channel_mask,
    )


def fused_sconv(
    x: torch.Tensor,
    weight: torch.Tensor,
    cache: torch.Tensor,
    positions: torch.Tensor,
    block_table: torch.Tensor,
    seq_idx: torch.Tensor,
    slot_mapping: torch.Tensor,
    query_start: torch.Tensor,
    off_s: int,
    width: int,
    block_size: int,
) -> torch.Tensor:
    """Insert and apply a depthwise causal convolution in one launch."""
    num_tokens = x.shape[0]
    out = torch.empty_like(x)
    if num_tokens == 0:
        return out
    if not x.is_contiguous():
        raise ValueError("sconv input must be contiguous")
    if cache.stride(3) != 1:
        raise ValueError("sconv cache D dimension must be contiguous")

    num_heads = cache.shape[1]
    kernel_size = weight.shape[1]
    channels = num_heads * width
    block_channels = min(triton.next_power_of_2(channels), 256)
    block_tokens = 8
    grid = (
        triton.cdiv(num_tokens, block_tokens),
        triton.cdiv(channels, block_channels),
    )
    _fused_sconv_kernel[grid](
        x,
        cache,
        weight,
        out,
        positions,
        seq_idx,
        slot_mapping,
        block_table,
        query_start,
        num_tokens,
        x.stride(0),
        cache.stride(0),
        cache.stride(1),
        cache.stride(2),
        cache.stride(3),
        weight.stride(0),
        weight.stride(1),
        block_table.stride(0),
        block_table.shape[1],
        block_size,
        W=kernel_size,
        OFF_S=off_s,
        WS=width,
        H=num_heads,
        BT=block_tokens,
        BLOCK_C=block_channels,
        num_warps=4,
    )
    return out


@triton.jit
def _seq_metadata_kernel(
    query_start_ptr,
    seq_idx_ptr,
    request_start_ptr,
    num_reqs,
    num_actual_tokens,
    num_padded_tokens,
    search_iterations,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    token = offsets.to(tl.int32)
    low = tl.zeros([BLOCK], tl.int32)
    high = tl.full([BLOCK], num_reqs - 1, tl.int32)
    for _ in range(search_iterations):
        middle = (low + high + 1) // 2
        below = tl.load(query_start_ptr + middle) <= token
        low = tl.where(below, middle, low)
        high = tl.where(below, high, middle - 1)
    actual = offsets < num_actual_tokens
    padded = offsets < num_padded_tokens
    request_start = tl.load(query_start_ptr + low)
    tl.store(seq_idx_ptr + offsets, tl.where(actual, low, 0), mask=padded)
    tl.store(
        request_start_ptr + offsets,
        tl.where(actual, request_start, 0),
        mask=padded,
    )


def sconv_seq_metadata(
    query_start_loc: torch.Tensor,
    num_reqs: int,
    num_actual_tokens: int,
    seq_idx_out: torch.Tensor,
    request_start_out: torch.Tensor,
    num_padded_tokens: int | None = None,
) -> None:
    """Build token-to-request metadata in persistent CUDA buffers."""
    if num_padded_tokens is None:
        num_padded_tokens = num_actual_tokens
    if num_padded_tokens < num_actual_tokens:
        raise ValueError("num_padded_tokens must cover all actual tokens")
    if num_padded_tokens > seq_idx_out.shape[0]:
        raise ValueError("seq_idx_out is too small")
    if num_padded_tokens > request_start_out.shape[0]:
        raise ValueError("request_start_out is too small")

    block = 256
    grid = (triton.cdiv(num_padded_tokens, block),)
    _seq_metadata_kernel[grid](
        query_start_loc,
        seq_idx_out,
        request_start_out,
        num_reqs,
        num_actual_tokens,
        num_padded_tokens,
        (num_reqs - 1).bit_length(),
        BLOCK=block,
    )


__all__ = ["fused_sconv", "sconv_seq_metadata"]
