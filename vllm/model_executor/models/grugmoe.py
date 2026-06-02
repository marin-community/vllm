# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness-first GrugMoE model.

This is a small reference implementation for the Marin Grug MoE architecture.
It intentionally avoids vLLM's fused MoE and attention/KV-cache kernels because
Grug's QB routing semantics differ from common MoE routers. The goal of this
initial implementation is exact tiny-model parity before performance work.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from vllm.config import VllmConfig
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.sequence import IntermediateTensors

_GATED_NORM_RANK = 128


def _get_config_attr(config: Any, names: tuple[str, ...], default: Any = None) -> Any:
    for name in names:
        if hasattr(config, name):
            return getattr(config, name)
    return default


def _get_rope_theta(config: Any) -> float:
    rope = getattr(config, "rope", None)
    if rope is not None:
        if isinstance(rope, dict) and "theta" in rope:
            return float(rope["theta"])
        if hasattr(rope, "theta"):
            return float(rope.theta)

    rope_parameters = getattr(config, "rope_parameters", None)
    if isinstance(rope_parameters, dict):
        for key in ("theta", "rope_theta"):
            if key in rope_parameters:
                return float(rope_parameters[key])

    return float(getattr(config, "rope_theta", 10000.0))


@dataclass(frozen=True)
class GrugMoeConfig:
    vocab_size: int
    hidden_dim: int = 2048
    intermediate_dim: int = 5632
    shared_expert_intermediate_dim: int = 5632
    num_experts: int = 8
    num_experts_per_token: int = 2
    num_layers: int = 24
    num_heads: int = 16
    num_kv_heads: int = 16
    head_dim: int | None = None
    max_seq_len: int = 4096
    sliding_window: int = 4096
    layer_norm_eps: float = 1e-5
    initializer_std: float = 0.02
    qk_mult: float = 1.0
    rope_theta: float = 10000.0

    @classmethod
    def from_hf_config(cls, config: Any) -> "GrugMoeConfig":
        max_seq_len = int(
            _get_config_attr(config, ("max_seq_len", "max_position_embeddings"), 4096)
        )
        num_heads = int(
            _get_config_attr(config, ("num_heads", "num_attention_heads"), 16)
        )
        return cls(
            vocab_size=int(_get_config_attr(config, ("vocab_size",))),
            hidden_dim=int(
                _get_config_attr(config, ("hidden_dim", "hidden_size"), 2048)
            ),
            intermediate_dim=int(
                _get_config_attr(
                    config,
                    ("intermediate_dim", "moe_intermediate_size", "intermediate_size"),
                    5632,
                )
            ),
            shared_expert_intermediate_dim=int(
                _get_config_attr(
                    config,
                    (
                        "shared_expert_intermediate_dim",
                        "shared_expert_intermediate_size",
                    ),
                    5632,
                )
            ),
            num_experts=int(
                _get_config_attr(config, ("num_experts", "num_local_experts"), 8)
            ),
            num_experts_per_token=int(
                _get_config_attr(
                    config, ("num_experts_per_token", "num_experts_per_tok"), 2
                )
            ),
            num_layers=int(
                _get_config_attr(config, ("num_layers", "num_hidden_layers"), 24)
            ),
            num_heads=num_heads,
            num_kv_heads=int(
                _get_config_attr(
                    config, ("num_kv_heads", "num_key_value_heads"), num_heads
                )
            ),
            head_dim=_get_config_attr(config, ("head_dim", "attention_head_dim")),
            max_seq_len=max_seq_len,
            sliding_window=int(
                _get_config_attr(config, ("sliding_window",), max_seq_len)
            ),
            layer_norm_eps=float(
                _get_config_attr(
                    config, ("layer_norm_eps", "rms_norm_eps"), 1e-5
                )
            ),
            initializer_std=float(
                _get_config_attr(config, ("initializer_std", "initializer_range"), 0.02)
            ),
            qk_mult=float(_get_config_attr(config, ("qk_mult",), 1.0)),
            rope_theta=_get_rope_theta(config),
        ).validate()

    @property
    def inferred_head_dim(self) -> int:
        if self.head_dim is not None:
            return int(self.head_dim)
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                f"hidden_dim={self.hidden_dim} is not divisible by "
                f"num_heads={self.num_heads}; set head_dim explicitly"
            )
        return self.hidden_dim // self.num_heads

    def validate(self) -> "GrugMoeConfig":
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.intermediate_dim <= 0:
            raise ValueError("intermediate_dim must be positive")
        if self.shared_expert_intermediate_dim < 0:
            raise ValueError("shared_expert_intermediate_dim must be non-negative")
        if self.num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if self.num_experts_per_token <= 0:
            raise ValueError("num_experts_per_token must be positive")
        if self.num_experts_per_token > self.num_experts:
            raise ValueError("num_experts_per_token must be <= num_experts")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if self.num_kv_heads <= 0:
            raise ValueError("num_kv_heads must be positive")
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        if self.inferred_head_dim <= 0:
            raise ValueError("head_dim must be positive")
        if self.inferred_head_dim % 2 != 0:
            raise ValueError("head_dim must be even for rotary embeddings")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        if self.sliding_window <= 1:
            raise ValueError("sliding_window must be greater than 1")
        return self


def _init_param(shape: tuple[int, ...], std: float) -> nn.Parameter:
    value = torch.empty(shape, dtype=torch.float32)
    nn.init.trunc_normal_(value, mean=0.0, std=std, a=-3 * std, b=3 * std)
    return nn.Parameter(value)


def _rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    dtype = x.dtype
    x_float = x.float()
    variance = torch.mean(torch.square(x_float), dim=-1, keepdim=True)
    x_norm = x_float * torch.rsqrt(variance + eps)
    if weight is not None:
        x_norm = x_norm * weight.float()
    return x_norm.to(dtype)


class GrugMoeRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _rms_norm(x, self.weight, self.eps)


class GrugMoeGatedNorm(nn.Module):
    def __init__(self, hidden_dim: int, initializer_std: float) -> None:
        super().__init__()
        self.w_down = _init_param((hidden_dim, _GATED_NORM_RANK), initializer_std)
        self.w_up = _init_param((_GATED_NORM_RANK, hidden_dim), initializer_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        gate_hidden = torch.matmul(x.float(), self.w_down.float())
        gate_hidden = F.silu(gate_hidden)
        gate = torch.sigmoid(torch.matmul(gate_hidden, self.w_up.float()))
        return x * gate.to(dtype)


class GrugMoeDenseMLP(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int,
        initializer_std: float,
    ) -> None:
        super().__init__()
        self.w_gate = _init_param((hidden_dim, intermediate_dim), initializer_std)
        self.w_up = _init_param((hidden_dim, intermediate_dim), initializer_std)
        self.w_down = _init_param((intermediate_dim, hidden_dim), initializer_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_flat = x.reshape(-1, orig_shape[-1])
        gate = torch.matmul(x_flat, self.w_gate.to(x_flat.dtype))
        up = torch.matmul(x_flat, self.w_up.to(x_flat.dtype))
        out = torch.matmul(F.silu(gate) * up, self.w_down.to(x_flat.dtype))
        return out.reshape(orig_shape)


class GrugMoeMLP(nn.Module):
    """QB-routed MoE with sigmoid combine weights."""

    def __init__(self, cfg: GrugMoeConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d, e, i = cfg.hidden_dim, cfg.num_experts, cfg.intermediate_dim
        self.router = _init_param((d, e), cfg.initializer_std)
        self.router_bias = nn.Parameter(torch.zeros(e, dtype=torch.float32))
        self.w_gate_up = _init_param((e, d, 2 * i), cfg.initializer_std)
        self.w_down = _init_param((e, i, d), cfg.initializer_std)

    def route(self, x_flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        router_logits = torch.matmul(x_flat.float(), self.router.float())
        biased_logits = router_logits + self.router_bias.float()
        topk = min(self.cfg.num_experts, self.cfg.num_experts_per_token + 1)
        _, selected = torch.topk(biased_logits, k=topk, dim=-1)
        selected = selected[:, : self.cfg.num_experts_per_token]
        unbiased_topk = torch.gather(router_logits, dim=-1, index=selected)
        combine_weights = torch.sigmoid(unbiased_topk).to(x_flat.dtype)
        return selected, combine_weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_flat = x.reshape(-1, orig_shape[-1])
        selected, combine_weights = self.route(x_flat)

        x_expert = x_flat.to(self.w_gate_up.dtype)
        expert_w13 = self.w_gate_up[selected]
        w13_out = torch.einsum("td,tkdi->tki", x_expert, expert_w13)
        gate, up = torch.split(w13_out, self.cfg.intermediate_dim, dim=-1)
        expert_w2 = self.w_down[selected]
        expert_out = torch.einsum("tki,tkid->tkd", F.silu(gate) * up, expert_w2)
        out = torch.sum(expert_out * combine_weights.unsqueeze(-1), dim=1)
        return out.to(x.dtype).reshape(orig_shape)


def _align_kv_heads(x: torch.Tensor, num_q_heads: int) -> torch.Tensor:
    num_kv_heads = x.shape[2]
    if num_q_heads == num_kv_heads:
        return x
    if num_q_heads % num_kv_heads != 0:
        raise ValueError("num_heads must be divisible by num_kv_heads")
    repeat = num_q_heads // num_kv_heads
    expanded = x.unsqueeze(3)
    expanded = expanded.expand(*x.shape[:3], repeat, x.shape[3])
    return expanded.reshape(*x.shape[:2], num_q_heads, x.shape[3])


def _causal_sliding_mask(positions: torch.Tensor, sliding_window: int) -> torch.Tensor:
    q_pos = positions[..., :, None]
    k_pos = positions[..., None, :]
    causal = k_pos <= q_pos
    in_window = k_pos >= q_pos - (sliding_window - 1)
    return causal & in_window


class GrugMoeAttention(nn.Module):
    def __init__(self, cfg: GrugMoeConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d, n, m, h = (
            cfg.hidden_dim,
            cfg.num_heads,
            cfg.num_kv_heads,
            cfg.inferred_head_dim,
        )
        self.w_q = _init_param((d, n * h), cfg.initializer_std)
        self.w_k = _init_param((d, m * h), cfg.initializer_std)
        self.w_v = _init_param((d, m * h), cfg.initializer_std)
        self.w_o = _init_param((n * h, d), cfg.initializer_std)
        self.attn_gate = nn.Parameter(torch.zeros(d, n, dtype=torch.float32))

    def _apply_rotary(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        half_dim = self.cfg.inferred_head_dim // 2
        inv_freq = 1.0 / (
            self.cfg.rope_theta
            ** (
                torch.arange(0, half_dim, device=q.device, dtype=torch.float32)
                / half_dim
            )
        )
        angles = positions.float().unsqueeze(-1) * inv_freq
        cos = torch.cos(angles).unsqueeze(2)
        sin = torch.sin(angles).unsqueeze(2)

        def _apply(x: torch.Tensor) -> torch.Tensor:
            x1, x2 = torch.split(x, half_dim, dim=-1)
            return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)

        return _apply(q), _apply(k)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        sliding_window: int,
    ) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        head_dim = self.cfg.inferred_head_dim
        q = torch.matmul(x, self.w_q.to(x.dtype)).reshape(
            bsz, seq_len, self.cfg.num_heads, head_dim
        )
        k = torch.matmul(x, self.w_k.to(x.dtype)).reshape(
            bsz, seq_len, self.cfg.num_kv_heads, head_dim
        )
        v = torch.matmul(x, self.w_v.to(x.dtype)).reshape(
            bsz, seq_len, self.cfg.num_kv_heads, head_dim
        )
        q = _rms_norm(q)
        k = _rms_norm(k)
        q, k = self._apply_rotary(q, k, positions)
        q = q * self.cfg.qk_mult

        k = _align_kv_heads(k, self.cfg.num_heads)
        v = _align_kv_heads(v, self.cfg.num_heads)
        scale = head_dim**-0.5
        scores = torch.einsum("bqhd,bkhd->bhqk", q * scale, k)
        mask = _causal_sliding_mask(positions, sliding_window)
        scores = torch.where(mask[:, None, :, :], scores, scores.new_tensor(-1e9))
        weights = torch.softmax(scores.float(), dim=-1).to(v.dtype)
        attn_out = torch.einsum("bhqk,bkhd->bqhd", weights, v)

        aligned_v = _align_kv_heads(v, self.cfg.num_heads)
        dot = torch.sum(attn_out * aligned_v, dim=-1, keepdim=True)
        v_norm_sq = torch.sum(aligned_v * aligned_v, dim=-1, keepdim=True)
        attn_out = attn_out - (dot / (v_norm_sq + 1e-6)) * aligned_v
        gate = 2 * torch.sigmoid(torch.matmul(x.float(), self.attn_gate.float()))
        attn_out = gate[..., None].to(attn_out.dtype) * attn_out
        attn_out = attn_out.reshape(bsz, seq_len, self.cfg.num_heads * head_dim)
        return torch.matmul(attn_out, self.w_o.to(attn_out.dtype))


class GrugMoeBlock(nn.Module):
    def __init__(self, cfg: GrugMoeConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.rms_attn = GrugMoeRMSNorm(cfg.hidden_dim, cfg.layer_norm_eps)
        self.attn_gated_norm = GrugMoeGatedNorm(
            cfg.hidden_dim, cfg.initializer_std
        )
        self.attn = GrugMoeAttention(cfg)
        self.rms_mlp = GrugMoeRMSNorm(cfg.hidden_dim, cfg.layer_norm_eps)
        self.mlp_gated_norm = GrugMoeGatedNorm(cfg.hidden_dim, cfg.initializer_std)
        self.mlp = GrugMoeMLP(cfg)
        self.shared = (
            GrugMoeDenseMLP(
                cfg.hidden_dim,
                cfg.shared_expert_intermediate_dim,
                cfg.initializer_std,
            )
            if cfg.shared_expert_intermediate_dim > 0
            else None
        )

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        sliding_window: int,
    ) -> torch.Tensor:
        attn_in = self.attn_gated_norm(self.rms_attn(x))
        x = x + self.attn(attn_in, positions, sliding_window)
        mlp_in = self.mlp_gated_norm(self.rms_mlp(x))
        mlp_out = self.mlp(mlp_in)
        if self.shared is not None:
            mlp_out = mlp_out + self.shared(mlp_in)
        return x + mlp_out


class GrugMoeModel(nn.Module):
    def __init__(self, cfg: GrugMoeConfig) -> None:
        super().__init__()
        self.config = cfg
        self.token_embed = _init_param(
            (cfg.vocab_size, cfg.hidden_dim), cfg.initializer_std
        )
        self.embed_norm = GrugMoeRMSNorm(cfg.hidden_dim, cfg.layer_norm_eps)
        self.embed_gated_norm = GrugMoeGatedNorm(
            cfg.hidden_dim, cfg.initializer_std
        )
        self.blocks = nn.ModuleList([GrugMoeBlock(cfg) for _ in range(cfg.num_layers)])
        self.final_norm = GrugMoeRMSNorm(cfg.hidden_dim, cfg.layer_norm_eps)
        self.final_gated_norm = GrugMoeGatedNorm(
            cfg.hidden_dim, cfg.initializer_std
        )
        self.output_proj = _init_param(
            (cfg.hidden_dim, cfg.vocab_size), cfg.initializer_std
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.token_embed[input_ids]

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if intermediate_tensors is not None:
            raise NotImplementedError("GrugMoE does not support pipeline parallelism")
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("input_ids or inputs_embeds must be provided")
            hidden = self.embed_input_ids(input_ids)
        else:
            hidden = inputs_embeds

        squeeze_batch = hidden.dim() == 2
        if squeeze_batch:
            hidden = hidden.unsqueeze(0)

        if positions.dim() == 1:
            positions = positions.unsqueeze(0)
        positions = positions.to(hidden.device)

        hidden = self.embed_gated_norm(self.embed_norm(hidden))
        short_window = self.config.sliding_window // 2
        for i, block in enumerate(self.blocks):
            layer_window = self.config.sliding_window if i % 4 == 3 else short_window
            hidden = block(hidden, positions, layer_window)
        hidden = self.final_gated_norm(self.final_norm(hidden))
        return hidden.squeeze(0) if squeeze_batch else hidden


class GrugMoeForCausalLM(nn.Module):
    fall_back_to_pt_during_load = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        del prefix
        hf_config = getattr(vllm_config.model_config, "hf_text_config", None)
        if hf_config is None:
            hf_config = vllm_config.model_config.hf_config
        self.config = GrugMoeConfig.from_hf_config(hf_config)
        self.tie_word_embeddings = bool(
            getattr(hf_config, "tie_word_embeddings", False)
        )
        self.model = GrugMoeModel(self.config)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(
            input_ids,
            positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        if self.tie_word_embeddings:
            output_proj = self.model.token_embed.transpose(0, 1)
        else:
            output_proj = self.model.output_proj
        return torch.matmul(hidden_states, output_proj.to(hidden_states.dtype))

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if name not in params_dict:
                continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


__all__ = [
    "GrugMoeAttention",
    "GrugMoeBlock",
    "GrugMoeConfig",
    "GrugMoeDenseMLP",
    "GrugMoeForCausalLM",
    "GrugMoeGatedNorm",
    "GrugMoeMLP",
    "GrugMoeModel",
    "GrugMoeRMSNorm",
]
