# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import Any

from transformers.configuration_utils import PretrainedConfig


def _coalesce(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def grug_moe_rope_theta(config: Any) -> float:
    rope_theta = getattr(config, "rope_theta", None)
    if rope_theta is not None:
        return float(rope_theta)

    rope_parameters = getattr(config, "rope_parameters", None)
    if isinstance(rope_parameters, dict):
        for key in ("rope_theta", "theta"):
            if key in rope_parameters:
                return float(rope_parameters[key])

    rope = getattr(config, "rope", None)
    if isinstance(rope, dict) and "theta" in rope:
        return float(rope["theta"])
    if hasattr(rope, "theta"):
        return float(rope.theta)
    return 10000.0


class GrugMoeConfig(PretrainedConfig):
    model_type = "grug_moe"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_dim: int | None = None,
        hidden_size: int | None = None,
        intermediate_dim: int | None = None,
        intermediate_size: int | None = None,
        moe_intermediate_size: int | None = None,
        shared_expert_intermediate_dim: int | None = None,
        shared_expert_intermediate_size: int | None = None,
        num_experts: int | None = None,
        num_local_experts: int | None = None,
        num_experts_per_token: int | None = None,
        num_experts_per_tok: int | None = None,
        num_layers: int | None = None,
        num_hidden_layers: int | None = None,
        num_heads: int | None = None,
        num_attention_heads: int | None = None,
        num_kv_heads: int | None = None,
        num_key_value_heads: int | None = None,
        head_dim: int | None = None,
        attention_head_dim: int | None = None,
        max_seq_len: int | None = None,
        max_position_embeddings: int | None = None,
        sliding_window: int | None = None,
        layer_norm_eps: float | None = None,
        rms_norm_eps: float | None = None,
        initializer_std: float | None = None,
        initializer_range: float | None = None,
        qk_mult: float = 1.0,
        qk_mult_long_scale: float = 1.0,
        disable_pko: bool = True,
        disable_long_rope: bool = True,
        rope: dict[str, Any] | None = None,
        rope_parameters: dict[str, Any] | None = None,
        rope_scaling: dict[str, Any] | None = None,
        rope_theta: float | None = None,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        **kwargs,
    ) -> None:
        hidden_size = int(_coalesce(hidden_size, hidden_dim, 2048))
        intermediate_size = int(
            _coalesce(intermediate_size, moe_intermediate_size, intermediate_dim, 5632)
        )
        shared_expert_intermediate_size = int(
            _coalesce(
                shared_expert_intermediate_size,
                shared_expert_intermediate_dim,
                5632,
            )
        )
        num_hidden_layers = int(_coalesce(num_hidden_layers, num_layers, 24))
        num_attention_heads = int(_coalesce(num_attention_heads, num_heads, 16))
        num_key_value_heads = int(
            _coalesce(num_key_value_heads, num_kv_heads, num_attention_heads)
        )
        head_dim = _coalesce(attention_head_dim, head_dim)
        max_position_embeddings = int(
            _coalesce(max_position_embeddings, max_seq_len, 4096)
        )
        rope_theta = grug_moe_rope_theta(
            SimpleNamespace(
                rope=rope,
                rope_parameters=rope_parameters,
                rope_theta=rope_theta,
            )
        )
        if rope is None:
            rope = {"theta": rope_theta}
        if rope_parameters is None:
            rope_parameters = {"rope_type": "default", "rope_theta": rope_theta}

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.hidden_dim = hidden_size
        self.intermediate_size = intermediate_size
        self.intermediate_dim = intermediate_size
        self.moe_intermediate_size = intermediate_size
        self.shared_expert_intermediate_size = shared_expert_intermediate_size
        self.shared_expert_intermediate_dim = shared_expert_intermediate_size
        self.num_experts = int(_coalesce(num_experts, num_local_experts, 8))
        self.num_local_experts = self.num_experts
        self.num_experts_per_tok = int(
            _coalesce(num_experts_per_tok, num_experts_per_token, 2)
        )
        self.num_experts_per_token = self.num_experts_per_tok
        self.num_hidden_layers = num_hidden_layers
        self.num_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_kv_heads = num_key_value_heads
        self.head_dim = head_dim
        self.attention_head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.max_seq_len = max_position_embeddings
        self.sliding_window = int(_coalesce(sliding_window, max_position_embeddings))
        self.rms_norm_eps = float(_coalesce(rms_norm_eps, layer_norm_eps, 1e-5))
        self.layer_norm_eps = self.rms_norm_eps
        self.initializer_range = float(
            _coalesce(initializer_range, initializer_std, 0.02)
        )
        self.initializer_std = self.initializer_range
        self.qk_mult = qk_mult
        self.qk_mult_long_scale = qk_mult_long_scale
        self.disable_pko = disable_pko
        self.disable_long_rope = disable_long_rope
        self.rope = rope
        self.rope_parameters = rope_parameters
        self.rope_scaling = rope_scaling
        self.rope_theta = rope_theta
        self.use_cache = use_cache

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


__all__ = ["GrugMoeConfig", "grug_moe_rope_theta"]
