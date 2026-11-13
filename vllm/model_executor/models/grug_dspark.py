# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Grug-layer DSpark draft model (speculators ``layer_type="grug"``).

The DSpark runtime is unchanged (parallel block drafting over context KV that is
precomputed from the target's hidden states, then a Markov head and a confidence
head). Only the draft decoder layer differs: it mirrors the Grug target layer
(``GrugMoeDecoderLayer``) instead of a Qwen3 layer.

  * q/k use a parameter-free RMSNorm and q is scaled by ``qk_mult``;
  * RoPE is partial through ``rope_parameters["partial_rotary_factor"]``;
  * the attention output loses its component along the token's own value (XSA)
    and is scaled by a per-head ``2 * sigmoid`` gate;
  * every RMSNorm is followed by a low-rank sigmoid "gated norm";
  * the FFN is the shared expert; routed experts are not implemented yet.

Attention is full and non-causal within the draft block, like the Qwen3 DSpark.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm import _custom_ops as ops
from vllm.config import CacheConfig, VllmConfig
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.transformers_utils.config import set_default_rope_theta
from vllm.v1.attention.backend import AttentionType

from .grugmoe import GrugMoeDenseMLP, GrugMoeGatedNorm
from .qwen3_dflash import DFlashQwen3Attention, _resolve_layer_attention
from .qwen3_dspark import Qwen3DSparkForCausalLM, Qwen3DSparkModel
from .utils import WeightsMapper

# Grug's gated norms are a fixed rank in the shared implementation.
_GATED_NORM_RANK = 128
_QK_NORM_EPS = 1e-6
_XSA_EPS = 1e-6
_ATTN_GATE_SCALE = 2.0


class GrugDSparkAttention(DFlashQwen3Attention):
    """Grug attention over the precomputed target context plus the query block."""

    def __init__(self, *, qk_mult: float, **kwargs) -> None:
        super().__init__(**kwargs)
        self.qk_mult = qk_mult
        # Parameter-free q/k norms (the base class built weighted ones).
        self.q_norm = RMSNorm(self.head_dim, eps=_QK_NORM_EPS, has_weight=False)
        self.k_norm = RMSNorm(self.head_dim, eps=_QK_NORM_EPS, has_weight=False)
        self.attn_gate = ColumnParallelLinear(
            self.hidden_size,
            self.total_num_heads,
            bias=False,
            gather_output=False,
            quant_config=None,
            prefix=f"{kwargs['prefix']}.attn_gate",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Context K/V are already in the KV cache (see
        ``precompute_and_store_context_kv``); this computes the query block."""
        num_tokens = hidden_states.shape[0]
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        q_shape, k_shape = q.shape, k.shape
        q = self.q_norm(q.view(num_tokens, self.num_heads, self.head_dim)).view(q_shape)
        k = self.k_norm(k.view(num_tokens, self.num_kv_heads, self.head_dim)).view(
            k_shape
        )
        q, k = self.rotary_emb(positions, q, k)
        q = q * self.qk_mult

        attn_output = self.attn(q, k, v)

        # XSA: drop the part of each head's output along the token's own value.
        attn_heads = attn_output.view(num_tokens, self.num_heads, self.head_dim)
        own_v = v.view(num_tokens, self.num_kv_heads, self.head_dim)
        own_v = own_v.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
        dot = (attn_heads * own_v).sum(dim=-1, keepdim=True)
        v_norm_sq = (own_v * own_v).sum(dim=-1, keepdim=True)
        attn_heads = attn_heads - (dot / (v_norm_sq + _XSA_EPS)) * own_v
        gate, _ = self.attn_gate(hidden_states)
        gate = _ATTN_GATE_SCALE * torch.sigmoid(gate)
        attn_heads = attn_heads * gate.unsqueeze(-1).to(attn_heads.dtype)

        output, _ = self.o_proj(attn_heads.reshape(num_tokens, self.q_size))
        return output


class GrugDSparkDecoderLayer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        *,
        config,
        layer_idx: int,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if getattr(config, "grug_num_experts", 0):
            raise NotImplementedError(
                "Grug DSpark drafts with routed experts are not supported yet"
            )
        if getattr(config, "grug_gated_norm_rank", _GATED_NORM_RANK) != (
            _GATED_NORM_RANK
        ):
            raise NotImplementedError(
                f"Grug DSpark gated norms support rank {_GATED_NORM_RANK} only"
            )
        self.hidden_size = config.hidden_size
        set_default_rope_theta(config, default_theta=10000)
        sliding_window, causal = _resolve_layer_attention(config, layer_idx)
        if sliding_window is not None:
            raise NotImplementedError("Grug DSpark drafts use full attention only")
        params_dtype = vllm_config.model_config.dtype

        rope_parameters = dict(config.rope_parameters)
        rope_parameters.setdefault(
            "partial_rotary_factor", getattr(config, "partial_rotary_factor", 1.0)
        )
        self.self_attn = GrugDSparkAttention(
            qk_mult=getattr(config, "grug_qk_mult", 1.75),
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            max_position=config.max_position_embeddings,
            num_kv_heads=config.num_key_value_heads,
            rms_norm_eps=config.rms_norm_eps,
            attention_bias=False,
            sliding_window=None,
            causal=causal,
            head_dim=getattr(config, "head_dim", None),
            cache_config=cache_config,
            quant_config=quant_config,
            rope_parameters=rope_parameters,
            prefix=f"{prefix}.self_attn",
            attn_type=AttentionType.DECODER,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn_gated_norm = GrugMoeGatedNorm(
            config.hidden_size,
            params_dtype,
            quant_config=quant_config,
            prefix=f"{prefix}.attn_gated_norm",
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.mlp_gated_norm = GrugMoeGatedNorm(
            config.hidden_size,
            params_dtype,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp_gated_norm",
        )
        self.shared_expert = GrugMoeDenseMLP(
            config.hidden_size,
            config.intermediate_size,
            params_dtype,
            quant_config=quant_config,
            prefix=f"{prefix}.shared_expert",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Same fused-residual convention as the Qwen3 DFlash layer: the true
        # residual stream is ``hidden_states + residual``. The gated norms sit
        # after each RMSNorm, as in the Grug target layer.
        if residual is not None:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        else:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.attn_gated_norm(hidden_states)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp_gated_norm(hidden_states)
        hidden_states = self.shared_expert(hidden_states)
        return hidden_states, residual


class GrugDSparkModel(Qwen3DSparkModel):
    """DSpark backbone built from Grug-style layers."""

    decoder_layer_cls = GrugDSparkDecoderLayer

    # The shared-expert / gated-norm projections are also called gate_proj/up_proj,
    # so only the attention projections are stacked (not the Qwen3 gate_up fusion).
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_substr={"midlayer.": "layers.0."},
        orig_to_new_stacked={
            ".q_proj": (".qkv_proj", "q"),
            ".k_proj": (".qkv_proj", "k"),
            ".v_proj": (".qkv_proj", "v"),
        },
    )

    def _build_context_kv_buffers(
        self,
        layers_attn: list[nn.Module],
        has_bias: bool,
    ) -> None:
        super()._build_context_kv_buffers(layers_attn, has_bias)
        # The context hidden states go through the draft's own weighted
        # ``hidden_norm`` (eps = rms_norm_eps); the parameter-free K norm uses
        # its own eps, so the two cannot share ``_rms_norm_eps``.
        self._hidden_norm_eps = self.hidden_norm.variance_epsilon
        self._k_norm_weights = torch.ones(
            len(layers_attn),
            layers_attn[0].head_dim,
            dtype=self._fused_kv_weight.dtype,
            device=self._fused_kv_weight.device,
        )

    def _project_context_kv(
        self,
        context_states: torch.Tensor,
        num_ctx: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normed_context_states = torch.empty_like(context_states)
        ops.rms_norm(
            normed_context_states,
            context_states,
            self._hidden_norm_weight,
            self._hidden_norm_eps,
        )
        all_kv_flat = F.linear(
            normed_context_states, self._fused_kv_weight, self._fused_kv_bias
        )
        all_kv = (
            all_kv_flat.view(num_ctx, num_layers, 2, num_kv_heads, head_dim)
            .permute(2, 1, 0, 3, 4)
            .contiguous()
        )
        return all_kv[0], all_kv[1]


class GrugDSparkForCausalLM(Qwen3DSparkForCausalLM):
    model_cls = GrugDSparkModel
    packed_modules_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}

