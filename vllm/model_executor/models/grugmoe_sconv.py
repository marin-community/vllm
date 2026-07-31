# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefix-cache-aware short convolution for GrugMoE."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import torch
from torch import nn
from torch.nn.parameter import Parameter

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.utils import set_weight_attrs
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadata,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheSpec, SlidingWindowSpec

from .grugmoe_sconv_ops import fused_sconv, sconv_seq_metadata

_K, _V, _ATTN, _MLP = 0, 1, 2, 3


@dataclass
class GrugMoeSconvMetadata(AttentionMetadata):
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    seq_idx: torch.Tensor
    query_start: torch.Tensor
    block_size: int


class GrugMoeSconvMetadataBuilder(AttentionMetadataBuilder[GrugMoeSconvMetadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = (
        AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
    )

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        if not isinstance(kv_cache_spec, SlidingWindowSpec):
            raise TypeError("GrugMoE sconv requires a SlidingWindowSpec")
        self.block_size = kv_cache_spec.block_size
        max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.seq_idx_buffer = torch.empty(max_tokens, dtype=torch.int32, device=device)
        self.query_start_buffer = torch.empty(
            max_tokens, dtype=torch.int32, device=device
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> GrugMoeSconvMetadata:
        del common_prefix_len, fast_build
        num_reqs = common_attn_metadata.num_reqs
        num_actual_tokens = int(common_attn_metadata.query_start_loc_cpu[-1])
        num_padded_tokens = common_attn_metadata.slot_mapping.shape[0]
        if num_padded_tokens < num_actual_tokens:
            raise ValueError("padded token count is smaller than actual tokens")
        sconv_seq_metadata(
            common_attn_metadata.query_start_loc,
            num_reqs,
            num_actual_tokens,
            self.seq_idx_buffer,
            self.query_start_buffer,
            num_padded_tokens,
        )
        return GrugMoeSconvMetadata(
            block_table=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping[:num_padded_tokens],
            seq_idx=self.seq_idx_buffer[:num_padded_tokens],
            query_start=self.query_start_buffer[:num_padded_tokens],
            block_size=self.block_size,
        )


class GrugMoeSconvBackend(AttentionBackend):
    """Cache-management-only backend; convolution runs out of band."""

    @staticmethod
    def get_name() -> str:
        return "GRUGMOE_SCONV"

    @classmethod
    def indexes_kv_by_block_stride(cls) -> bool:
        return True

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        del cache_dtype_str
        return (num_blocks, num_kv_heads, block_size, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        if include_num_layers_dimension:
            return (0, 1, 2, 3, 4)
        return (0, 1, 2, 3)

    @staticmethod
    def get_impl_cls():
        raise NotImplementedError("GrugMoeSconvBackend has no attention implementation")

    @staticmethod
    def get_builder_cls() -> type[GrugMoeSconvMetadataBuilder]:
        return GrugMoeSconvMetadataBuilder


class GrugMoeConvState(nn.Module, AttentionLayerBase):
    """Paged state for one depthwise convolution stream."""

    def __init__(
        self,
        *,
        dim: int,
        num_heads: int,
        kernel_size: int,
        dtype: torch.dtype,
        prefix: str,
    ) -> None:
        super().__init__()
        self.prefix = prefix
        self.kv_cache = torch.tensor([])
        self.num_heads = num_heads
        if dim % num_heads != 0:
            raise ValueError("sconv dimension must be divisible by its cache heads")
        self.head_size = dim // num_heads
        self.sliding_window = kernel_size
        self.block_size = kernel_size
        vllm_config = get_current_vllm_config()
        self._dtype = dtype
        compilation_config = vllm_config.compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def forward(self): ...

    def get_attn_backend(self) -> type[AttentionBackend]:
        return GrugMoeSconvBackend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        del vllm_config
        return SlidingWindowSpec(
            block_size=self.block_size,
            num_kv_heads=self.num_heads,
            head_size=self.head_size,
            head_size_v=0,
            dtype=self._dtype,
            sliding_window=self.sliding_window,
        )


class GrugMoeShortConv(nn.Module):
    """A depthwise weight with a prefix-cache-aware paged history."""

    def __init__(
        self,
        dim: int,
        kernel_size: int,
        *,
        num_heads: int,
        dtype: torch.dtype,
        prefix: str,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.state = GrugMoeConvState(
            dim=dim,
            num_heads=num_heads,
            kernel_size=kernel_size,
            dtype=dtype,
            prefix=f"{prefix}.state",
        )
        self.weight = Parameter(
            torch.empty(dim, kernel_size, dtype=dtype),
            requires_grad=False,
        )
        set_weight_attrs(self.weight, {"weight_loader": self.weight_loader})

    def weight_loader(
        self,
        param: Parameter,
        loaded_weight: torch.Tensor,
    ) -> None:
        # Levanter stores [kernel, channels]. Inkling-style exports sometimes
        # use [channels, 1, kernel].
        if loaded_weight.ndim == 3 and loaded_weight.shape[1] == 1:
            loaded_weight = loaded_weight.squeeze(1)
        if loaded_weight.shape == (self.kernel_size, self.dim):
            loaded_weight = loaded_weight.T
        if loaded_weight.shape != param.shape:
            raise ValueError(
                f"sconv weight shape {tuple(loaded_weight.shape)} does not match "
                f"{tuple(param.shape)}"
            )
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        metadata = get_forward_context().attn_metadata
        if not isinstance(metadata, dict):
            raise RuntimeError("GrugMoE sconv requires paged forward metadata")
        stream_metadata = metadata.get(self.state.prefix)
        if stream_metadata is None:
            raise RuntimeError(
                f"GrugMoE sconv metadata is missing for {self.state.prefix}"
            )
        if not isinstance(stream_metadata, GrugMoeSconvMetadata):
            raise TypeError("unexpected GrugMoE sconv metadata")
        cache = self.state.kv_cache
        if cache.numel() == 0:
            raise RuntimeError(f"GrugMoE sconv cache is empty for {self.state.prefix}")
        return fused_sconv(
            x.contiguous(),
            self.weight,
            cache,
            positions,
            stream_metadata.block_table,
            stream_metadata.seq_idx,
            stream_metadata.slot_mapping,
            stream_metadata.query_start,
            0,
            self.state.head_size,
            stream_metadata.block_size,
        )


__all__ = [
    "GrugMoeConvState",
    "GrugMoeSconvBackend",
    "GrugMoeSconvMetadata",
    "GrugMoeSconvMetadataBuilder",
    "GrugMoeShortConv",
]
