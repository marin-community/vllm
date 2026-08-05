# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness-first GPU implementation of Marin GrugMoE."""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

import vllm.envs as envs
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.grugmoe_sconv import GrugMoeShortConv
from vllm.model_executor.models.utils import (
    make_empty_intermediate_tensors_factory,
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.grugmoe import (
    grug_moe_layer_types,
    grug_moe_rope_theta,
)
from vllm.v1.attention.backend import AttentionType

logger = init_logger(__name__)

# The trained GrugMoE checkpoint uses a fixed rank-128 gated-norm bottleneck.
_GATED_NORM_RANK = 128
_ROUTER_COMBINE_WEIGHT_SUM = 2.5
_ROUTER_COMBINE_WEIGHT_EPS = 1e-9


def _config_attr(config: Any, names: tuple[str, ...], default: Any = None) -> Any:
    for name in names:
        if hasattr(config, name):
            return getattr(config, name)
    return default


def _rms_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    dtype = x.dtype
    x_float = x.float()
    variance = torch.mean(torch.square(x_float), dim=-1, keepdim=True)
    return (x_float * torch.rsqrt(variance + eps)).to(dtype)


def _align_kv_heads(value: torch.Tensor, num_q_heads: int) -> torch.Tensor:
    num_kv_heads = value.shape[1]
    if num_q_heads == num_kv_heads:
        return value
    if num_q_heads % num_kv_heads != 0:
        raise ValueError("num_heads must be divisible by num_kv_heads")
    repeat = num_q_heads // num_kv_heads
    value = value.unsqueeze(2).expand(
        value.shape[0], num_kv_heads, repeat, value.shape[2]
    )
    return value.reshape(value.shape[0], num_q_heads, value.shape[-1])


def _format_int_ranges(values: list[int]) -> str:
    if not values:
        return "[]"
    ranges: list[str] = []
    start = prev = values[0]
    for value in values[1:]:
        if value == prev + 1:
            prev = value
            continue
        ranges.append(str(start) if start == prev else f"{start}..{prev}")
        start = prev = value
    ranges.append(str(start) if start == prev else f"{start}..{prev}")
    return "[" + ", ".join(ranges) + "]"


def get_grug_moe_runtime_info(
    _vllm_config: VllmConfig,
    model: nn.Module,
) -> dict[str, Any]:
    """Return worker-local GrugMoE runtime state for logs and diagnostics."""
    grug_model = getattr(model, "model", model)
    layers = getattr(grug_model, "layers", None)
    if not layers:
        raise RuntimeError("GrugMoE runtime info requires at least one layer")
    first_layer = layers[0]
    moe_runner = first_layer.mlp.experts
    moe_config = moe_runner.moe_config
    moe_parallel_config = moe_config.moe_parallel_config
    expert_map_manager = moe_runner.expert_map_manager
    local_expert_ids = [
        int(expert_id) for expert_id in expert_map_manager.get_local_expert_ids()
    ]
    attention_backend = first_layer.self_attn.attn.attn_backend.get_name()
    runtime_config = getattr(grug_model, "config", first_layer.self_attn.cfg)
    return {
        "tp_size": int(moe_parallel_config.tp_size),
        "tp_rank": int(moe_parallel_config.tp_rank),
        "dp_size": int(moe_parallel_config.dp_size),
        "dp_rank": int(moe_parallel_config.dp_rank),
        "ep_size": int(moe_parallel_config.ep_size),
        "ep_rank": int(moe_parallel_config.ep_rank),
        "use_ep": bool(moe_parallel_config.use_ep),
        "num_experts": int(moe_config.num_experts),
        "num_logical_experts": int(moe_config.num_logical_experts),
        "num_local_experts": int(moe_config.num_local_experts),
        "top_k": int(moe_config.experts_per_token),
        "local_expert_ids": local_expert_ids,
        "local_expert_ownership": _format_int_ranges(local_expert_ids),
        "expert_placement_strategy": str(moe_runner.expert_placement_strategy),
        "all2all_backend": str(moe_parallel_config.all2all_backend),
        "attention_backend": str(attention_backend),
        "global_every": int(runtime_config.global_every),
        "local_kv_heads": int(runtime_config.resolved_local_kv_heads),
        "global_kv_heads": int(runtime_config.resolved_global_kv_heads),
        "stored_kv_heads": int(runtime_config.num_kv_heads),
        "sconv": bool(runtime_config.sconv),
        "sconv_kernel": int(runtime_config.sconv_kernel),
        "num_shared_experts": int(runtime_config.num_shared_experts),
        "rope_fused": bool(runtime_config.rope_fused),
        "gated_norm": bool(runtime_config.gated_norm),
        "attn_gate": bool(runtime_config.attn_gate),
        "xsa": bool(runtime_config.xsa),
        "qb_routing": bool(runtime_config.qb_routing),
        "mtp_depth": int(runtime_config.mtp_depth),
    }


def _log_grug_moe_runtime_info(
    vllm_config: VllmConfig,
    model: nn.Module,
) -> None:
    info = get_grug_moe_runtime_info(vllm_config, model)
    logger.info(
        "GrugMoE effective config: TP=%d/%d DP=%d/%d EP=%d/%d "
        "use_ep=%s experts=%d local=%d top_k=%d ownership=%s "
        "placement=%s all2all=%s attention=%s global_every=%d "
        "kv_heads=%d/%d(stored=%d) sconv=%s/%d shared=%d "
        "rope_fused=%s gated_norm=%s attn_gate=%s xsa=%s qb=%s "
        "mtp_depth=%d(serving=trunk-only)",
        info["tp_rank"],
        info["tp_size"],
        info["dp_rank"],
        info["dp_size"],
        info["ep_rank"],
        info["ep_size"],
        info["use_ep"],
        info["num_experts"],
        info["num_local_experts"],
        info["top_k"],
        info["local_expert_ownership"],
        info["expert_placement_strategy"],
        info["all2all_backend"],
        info["attention_backend"],
        info["global_every"],
        info["local_kv_heads"],
        info["global_kv_heads"],
        info["stored_kv_heads"],
        info["sconv"],
        info["sconv_kernel"],
        info["num_shared_experts"],
        info["rope_fused"],
        info["gated_norm"],
        info["attn_gate"],
        info["xsa"],
        info["qb_routing"],
        info["mtp_depth"],
    )


@dataclass(frozen=True)
class GrugMoeRuntimeConfig:
    vocab_size: int
    hidden_dim: int = 2048
    intermediate_dim: int = 5632
    shared_expert_intermediate_dim: int = 5632
    num_shared_experts: int = 1
    num_experts: int = 8
    num_experts_per_token: int = 2
    num_layers: int = 24
    num_heads: int = 16
    num_kv_heads: int = 16
    local_kv_heads: int | None = None
    global_kv_heads: int | None = None
    head_dim: int | None = None
    max_seq_len: int = 4096
    sliding_window: int = 4096
    global_every: int = 4
    layer_norm_eps: float = 1e-5
    initializer_std: float = 0.02
    qk_mult: float = 1.0
    qk_mult_long_scale: float = 1.0
    disable_pko: bool = True
    disable_long_rope: bool = True
    rope_fraction: float = 1.0
    rope_fused: bool = False
    gated_norm: bool = True
    attn_gate: bool = True
    xsa: bool = True
    qb_routing: bool = True
    legacy_input_output_gated_norm: bool = True
    mtp_depth: int = 0
    over_encoding_vocab_size: int = 0
    sconv: bool = False
    sconv_kernel: int = 4
    sconv_sites: tuple[str, ...] = ("k", "v", "attn", "mlp")
    rope_theta: float = 10000.0

    @classmethod
    def from_hf_config(cls, config: Any) -> "GrugMoeRuntimeConfig":
        max_seq_len = int(
            _config_attr(config, ("max_seq_len", "max_position_embeddings"), 4096)
        )
        num_heads = int(_config_attr(config, ("num_heads", "num_attention_heads"), 16))
        num_layers = int(_config_attr(config, ("num_layers", "num_hidden_layers"), 24))
        stored_kv_heads = int(
            _config_attr(config, ("num_kv_heads", "num_key_value_heads"), num_heads)
        )
        local_kv_heads = int(_config_attr(config, ("local_kv_heads",), stored_kv_heads))
        global_kv_heads = int(
            _config_attr(config, ("global_kv_heads",), stored_kv_heads)
        )
        return cls(
            vocab_size=int(_config_attr(config, ("vocab_size",))),
            hidden_dim=int(_config_attr(config, ("hidden_dim", "hidden_size"), 2048)),
            intermediate_dim=int(
                _config_attr(
                    config,
                    ("intermediate_dim", "moe_intermediate_size", "intermediate_size"),
                    5632,
                )
            ),
            shared_expert_intermediate_dim=int(
                _config_attr(
                    config,
                    (
                        "shared_expert_intermediate_dim",
                        "shared_expert_intermediate_size",
                    ),
                    5632,
                )
            ),
            num_shared_experts=int(_config_attr(config, ("num_shared_experts",), 1)),
            num_experts=int(
                _config_attr(config, ("num_experts", "num_local_experts"), 8)
            ),
            num_experts_per_token=int(
                _config_attr(
                    config, ("num_experts_per_token", "num_experts_per_tok"), 2
                )
            ),
            num_layers=num_layers,
            num_heads=num_heads,
            num_kv_heads=stored_kv_heads,
            local_kv_heads=local_kv_heads,
            global_kv_heads=global_kv_heads,
            head_dim=_config_attr(config, ("head_dim", "attention_head_dim")),
            max_seq_len=max_seq_len,
            sliding_window=int(_config_attr(config, ("sliding_window",), max_seq_len)),
            global_every=int(_config_attr(config, ("global_every",), 4)),
            layer_norm_eps=float(
                _config_attr(config, ("layer_norm_eps", "rms_norm_eps"), 1e-5)
            ),
            initializer_std=float(
                _config_attr(config, ("initializer_std", "initializer_range"), 0.02)
            ),
            qk_mult=float(_config_attr(config, ("qk_mult",), 1.0)),
            qk_mult_long_scale=float(
                _config_attr(config, ("qk_mult_long_scale",), 1.0)
            ),
            disable_pko=bool(_config_attr(config, ("disable_pko",), True)),
            disable_long_rope=bool(_config_attr(config, ("disable_long_rope",), True)),
            rope_fraction=float(_config_attr(config, ("rope_fraction",), 1.0)),
            rope_fused=bool(_config_attr(config, ("rope_fused",), False)),
            gated_norm=bool(_config_attr(config, ("gated_norm",), True)),
            attn_gate=bool(_config_attr(config, ("attn_gate",), True)),
            xsa=bool(_config_attr(config, ("xsa",), True)),
            qb_routing=bool(_config_attr(config, ("qb_routing",), True)),
            legacy_input_output_gated_norm=bool(
                _config_attr(
                    config,
                    ("legacy_input_output_gated_norm",),
                    True,
                )
            ),
            mtp_depth=int(_config_attr(config, ("mtp_depth",), 0)),
            over_encoding_vocab_size=int(
                _config_attr(config, ("over_encoding_vocab_size",), 0)
            ),
            sconv=bool(_config_attr(config, ("sconv",), False)),
            sconv_kernel=int(_config_attr(config, ("sconv_kernel",), 4)),
            sconv_sites=tuple(
                _config_attr(
                    config,
                    ("sconv_sites",),
                    ("k", "v", "attn", "mlp"),
                )
            ),
            rope_theta=grug_moe_rope_theta(config),
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

    @property
    def rotary_dim(self) -> int:
        if self.rope_fraction >= 1.0:
            return self.inferred_head_dim
        return (int(self.inferred_head_dim * self.rope_fraction) // 2) * 2

    @property
    def attention_layer_types(self) -> tuple[str, ...]:
        return tuple(grug_moe_layer_types(self.num_layers, self.global_every))

    def logical_kv_heads(self, is_long: bool) -> int:
        return (
            self.resolved_global_kv_heads if is_long else self.resolved_local_kv_heads
        )

    @property
    def resolved_local_kv_heads(self) -> int:
        return self.num_kv_heads if self.local_kv_heads is None else self.local_kv_heads

    @property
    def resolved_global_kv_heads(self) -> int:
        return (
            self.num_kv_heads if self.global_kv_heads is None else self.global_kv_heads
        )

    def validate(self) -> "GrugMoeRuntimeConfig":
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.intermediate_dim <= 0:
            raise ValueError("intermediate_dim must be positive")
        if self.shared_expert_intermediate_dim < 0:
            raise ValueError("shared_expert_intermediate_dim must be non-negative")
        if self.num_shared_experts <= 0:
            raise ValueError("num_shared_experts must be positive")
        if self.shared_expert_intermediate_dim % self.num_shared_experts != 0:
            raise ValueError(
                "shared_expert_intermediate_dim must be divisible by num_shared_experts"
            )
        if self.num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if self.num_experts_per_token <= 0:
            raise ValueError("num_experts_per_token must be positive")
        if self.num_experts_per_token >= self.num_experts:
            raise ValueError(
                "num_experts_per_token must be < num_experts for QB routing"
            )
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if self.num_kv_heads <= 0:
            raise ValueError("num_kv_heads must be positive")
        if (self.local_kv_heads is None) != (self.global_kv_heads is None):
            raise ValueError("local_kv_heads and global_kv_heads must be set together")
        if self.num_kv_heads != max(
            self.resolved_local_kv_heads,
            self.resolved_global_kv_heads,
        ):
            raise ValueError(
                "num_kv_heads must equal max(local_kv_heads, global_kv_heads)"
            )
        for name, value in (
            ("local_kv_heads", self.resolved_local_kv_heads),
            ("global_kv_heads", self.resolved_global_kv_heads),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            if self.num_heads % value != 0:
                raise ValueError(f"num_heads must be divisible by {name}")
        if self.inferred_head_dim <= 0:
            raise ValueError("head_dim must be positive")
        if self.inferred_head_dim % 2 != 0:
            raise ValueError("head_dim must be even for rotary embeddings")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        if self.sliding_window <= 1:
            raise ValueError("sliding_window must be greater than 1")
        if self.global_every <= 0:
            raise ValueError("global_every must be positive")
        if not 0.0 < self.rope_fraction <= 1.0:
            raise ValueError("rope_fraction must be in (0, 1]")
        if self.rotary_dim <= 0:
            raise ValueError("rope_fraction must select at least two rotary dimensions")
        if not self.gated_norm:
            raise NotImplementedError("GrugMoE currently requires gated_norm=true")
        if not self.attn_gate:
            raise NotImplementedError("GrugMoE currently requires attn_gate=true")
        if not self.xsa:
            raise NotImplementedError("GrugMoE currently requires xsa=true")
        if not self.qb_routing:
            raise NotImplementedError("GrugMoE currently requires qb_routing=true")
        if self.mtp_depth < 0:
            raise ValueError("mtp_depth must be non-negative")
        if self.over_encoding_vocab_size != 0:
            raise NotImplementedError(
                "GrugMoE serving does not support input over-encoding"
            )
        if self.sconv_kernel <= 0:
            raise ValueError("sconv_kernel must be positive")
        unknown_sconv_sites = set(self.sconv_sites) - {"k", "v", "attn", "mlp"}
        if unknown_sconv_sites:
            raise ValueError(f"unsupported sconv_sites: {sorted(unknown_sconv_sites)}")
        _ = self.attention_layer_types
        return self


class GrugMoeGatedNorm(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        params_dtype: torch.dtype,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.down_proj = ReplicatedLinear(
            hidden_dim,
            _GATED_NORM_RANK,
            bias=False,
            params_dtype=params_dtype,
            quant_config=None,
            disable_tp=True,
            prefix=f"{prefix}.down_proj",
        )
        self.up_proj = ReplicatedLinear(
            _GATED_NORM_RANK,
            hidden_dim,
            bias=False,
            params_dtype=params_dtype,
            quant_config=None,
            disable_tp=True,
            prefix=f"{prefix}.up_proj",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        gate_hidden = F.linear(x, self.down_proj.weight)
        gate_hidden = F.silu(gate_hidden)
        gate = F.linear(gate_hidden, self.up_proj.weight)
        return x * torch.sigmoid(gate).to(dtype)


class GrugMoeDenseMLP(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int,
        params_dtype: torch.dtype,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_proj = ReplicatedLinear(
            hidden_dim,
            intermediate_dim,
            bias=False,
            params_dtype=params_dtype,
            quant_config=None,
            disable_tp=True,
            prefix=f"{prefix}.gate_proj",
        )
        self.up_proj = ReplicatedLinear(
            hidden_dim,
            intermediate_dim,
            bias=False,
            params_dtype=params_dtype,
            quant_config=None,
            disable_tp=True,
            prefix=f"{prefix}.up_proj",
        )
        self.down_proj = ReplicatedLinear(
            intermediate_dim,
            hidden_dim,
            bias=False,
            params_dtype=params_dtype,
            quant_config=None,
            disable_tp=True,
            prefix=f"{prefix}.down_proj",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, _ = self.gate_proj(x)
        up, _ = self.up_proj(x)
        out, _ = self.down_proj(F.silu(gate) * up)
        return out


class GrugMoeRouter(BaseRouter):
    """Grug QB router: biased top-(K+1), drop threshold, normalized sigmoid."""

    def __init__(
        self,
        top_k: int,
        global_num_experts: int,
        bias: torch.Tensor,
        balanced_offset: int = 0,
    ) -> None:
        super().__init__(top_k=top_k, global_num_experts=global_num_experts)
        self.bias = bias
        self.balanced_offset = balanced_offset % global_num_experts

    @property
    def routing_method_type(self) -> RoutingMethodType:
        return RoutingMethodType.Custom

    def _compute_routing(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        indices_type: torch.dtype | None,
        *,
        input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del hidden_states, input_ids
        if envs.VLLM_GRUGMOE_ROUTING_FIXTURE == "balanced":
            # Deterministic batch-balanced control. Consecutive routed slots
            # walk the expert ring, so every expert receives the same number
            # of selections (within one when the slot count is not divisible
            # by the expert count). Each layer rotates the start of that ring
            # so the remainder does not favor the same EP rank throughout the
            # model. Only selection changes: combine weights still come from
            # the checkpoint's unbiased router logits.
            token_offsets = torch.arange(
                router_logits.shape[0],
                device=router_logits.device,
                dtype=torch.long,
            )
            slot_offsets = torch.arange(
                self.top_k,
                device=router_logits.device,
                dtype=torch.long,
            )
            selected = (
                token_offsets[:, None] * self.top_k
                + slot_offsets[None, :]
                + self.balanced_offset
            ) % self.global_num_experts
        else:
            _, selected = torch.topk(
                router_logits.float() + self.bias.float(),
                k=self.top_k + 1,
                dim=-1,
            )
            selected = selected[:, : self.top_k]
        topk_weights = torch.sigmoid(
            torch.gather(router_logits.float(), dim=-1, index=selected)
        )
        denom = topk_weights.sum(dim=-1, keepdim=True) + _ROUTER_COMBINE_WEIGHT_EPS
        topk_weights = topk_weights * (_ROUTER_COMBINE_WEIGHT_SUM / denom)
        topk_ids = selected.to(torch.int32 if indices_type is None else indices_type)
        return topk_weights, topk_ids


class GrugMoeMLP(nn.Module):
    """QB-routed MoE with normalized sigmoid combine weights."""

    def __init__(
        self,
        cfg: GrugMoeRuntimeConfig,
        params_dtype: torch.dtype | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.cfg = cfg
        params_dtype = params_dtype or torch.get_default_dtype()
        self.router = ReplicatedLinear(
            cfg.hidden_dim,
            cfg.num_experts,
            bias=True,
            params_dtype=torch.float32,
            quant_config=None,
            disable_tp=True,
            prefix=f"{prefix}.router",
        )
        prefix_parts = prefix.split(".")
        layer_index = next(
            (
                int(prefix_parts[index + 1])
                for index, part in enumerate(prefix_parts[:-1])
                if part == "layers" and prefix_parts[index + 1].isdigit()
            ),
            0,
        )
        moe_router = GrugMoeRouter(
            top_k=cfg.num_experts_per_token,
            global_num_experts=cfg.num_experts,
            bias=self.router.bias,
            # Every layer is individually as even as integer division allows.
            # Rotate its unavoidable remainder so the 48-layer aggregate does
            # not repeatedly favor the same experts or EP ranks.
            balanced_offset=layer_index * max(1, cfg.num_experts // 16),
        )
        self.experts = FusedMoE(
            num_experts=cfg.num_experts,
            top_k=cfg.num_experts_per_token,
            hidden_size=cfg.hidden_dim,
            intermediate_size=cfg.intermediate_dim,
            params_dtype=params_dtype,
            renormalize=False,
            quant_config=None,
            prefix=f"{prefix}.experts",
            router=moe_router,
            router_logits_dtype=torch.float32,
        )

    def _forward_torch_reference(self, x_flat: torch.Tensor) -> torch.Tensor:
        router_logits = F.linear(x_flat.float(), self.router.weight.float())
        combine_weights, selected = self.experts.router.select_experts(
            hidden_states=x_flat,
            router_logits=router_logits,
        )
        selected = selected.long()
        routed_experts = self.experts.routed_experts
        w13_weight = routed_experts.w13_weight
        gate_proj_weight = w13_weight[:, : self.cfg.intermediate_dim, :]
        up_proj_weight = w13_weight[:, self.cfg.intermediate_dim :, :]
        down_proj_weight = routed_experts.w2_weight

        gate_weight = gate_proj_weight[selected]
        up_weight = up_proj_weight[selected]
        down_weight = down_proj_weight[selected]
        gate = torch.einsum(
            "td,tkid->tki",
            x_flat,
            gate_weight.to(x_flat.dtype),
        )
        up = torch.einsum(
            "td,tkid->tki",
            x_flat,
            up_weight.to(x_flat.dtype),
        )
        expert_out = torch.einsum(
            "tki,tkdi->tkd",
            F.silu(gate) * up,
            down_weight.to(x_flat.dtype),
        )
        return torch.sum(expert_out * combine_weights.unsqueeze(-1), dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        hidden_dim = orig_shape[-1]
        x_flat = x.reshape(-1, hidden_dim)
        if not x_flat.is_cuda:
            out = self._forward_torch_reference(x_flat)
            return out.to(x.dtype).reshape(orig_shape)
        router_logits = F.linear(x_flat.float(), self.router.weight.float())
        out = self.experts(
            hidden_states=x_flat,
            router_logits=router_logits,
        )
        return out.to(x.dtype).reshape(orig_shape)


class GrugMoeAttention(nn.Module):
    def __init__(
        self,
        cfg: GrugMoeRuntimeConfig,
        cache_config: CacheConfig | None,
        params_dtype: torch.dtype,
        sliding_window: int | None,
        use_rope: bool,
        qk_mult_scale: float,
        logical_num_kv_heads: int | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.head_dim = cfg.inferred_head_dim
        self.q_size = cfg.num_heads * self.head_dim
        self.kv_size = cfg.num_kv_heads * self.head_dim
        self.logical_num_kv_heads = (
            cfg.num_kv_heads if logical_num_kv_heads is None else logical_num_kv_heads
        )
        self.logical_kv_size = self.logical_num_kv_heads * self.head_dim
        self.use_rope = use_rope
        self.qk_mult_scale = qk_mult_scale
        self.sconv_k = (
            GrugMoeShortConv(
                self.kv_size,
                cfg.sconv_kernel,
                num_heads=cfg.num_kv_heads,
                dtype=params_dtype,
                prefix=f"{prefix}.sconv_k",
            )
            if cfg.sconv and "k" in cfg.sconv_sites
            else None
        )
        self.sconv_v = (
            GrugMoeShortConv(
                self.kv_size,
                cfg.sconv_kernel,
                num_heads=cfg.num_kv_heads,
                dtype=params_dtype,
                prefix=f"{prefix}.sconv_v",
            )
            if cfg.sconv and "v" in cfg.sconv_sites
            else None
        )

        self.q_proj = ReplicatedLinear(
            cfg.hidden_dim,
            self.q_size,
            bias=False,
            params_dtype=params_dtype,
            quant_config=None,
            disable_tp=True,
            prefix=f"{prefix}.q_proj",
        )
        self.k_proj = ReplicatedLinear(
            cfg.hidden_dim,
            self.kv_size,
            bias=False,
            params_dtype=params_dtype,
            quant_config=None,
            disable_tp=True,
            prefix=f"{prefix}.k_proj",
        )
        self.v_proj = ReplicatedLinear(
            cfg.hidden_dim,
            self.kv_size,
            bias=False,
            params_dtype=params_dtype,
            quant_config=None,
            disable_tp=True,
            prefix=f"{prefix}.v_proj",
        )
        self.o_proj = ReplicatedLinear(
            self.q_size,
            cfg.hidden_dim,
            bias=False,
            params_dtype=params_dtype,
            quant_config=None,
            disable_tp=True,
            prefix=f"{prefix}.o_proj",
        )
        self.attn_gate = ReplicatedLinear(
            cfg.hidden_dim,
            cfg.num_heads,
            bias=False,
            params_dtype=params_dtype,
            quant_config=None,
            disable_tp=True,
            prefix=f"{prefix}.attn_gate",
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=cfg.max_seq_len,
            rope_parameters={
                "rope_theta": cfg.rope_theta,
                "rope_dim": cfg.rotary_dim,
            },
            is_neox_style=not cfg.rope_fused,
        )
        self.attn = Attention(
            cfg.num_heads,
            self.head_dim,
            self.head_dim**-0.5,
            num_kv_heads=self.logical_num_kv_heads,
            cache_config=cache_config,
            quant_config=None,
            per_layer_sliding_window=sliding_window,
            prefix=f"{prefix}.attn",
            attn_type=AttentionType.DECODER,
        )

    def apply_xsa(
        self,
        hidden_states: torch.Tensor,
        attn_output: torch.Tensor,
        value_states: torch.Tensor,
    ) -> torch.Tensor:
        num_tokens = hidden_states.shape[0]
        value_heads = value_states.view(
            num_tokens, self.logical_num_kv_heads, self.head_dim
        )
        aligned_value = _align_kv_heads(value_heads, self.cfg.num_heads)
        attn_heads = attn_output.view(num_tokens, self.cfg.num_heads, self.head_dim)

        dot = torch.sum(attn_heads * aligned_value, dim=-1, keepdim=True)
        value_norm_sq = torch.sum(aligned_value * aligned_value, dim=-1, keepdim=True)
        attn_heads = attn_heads - (dot / (value_norm_sq + 1e-6)) * aligned_value
        gate = F.linear(hidden_states, self.attn_gate.weight)
        gate = 2 * torch.sigmoid(gate)
        attn_heads = gate[..., None].to(attn_heads.dtype) * attn_heads
        return attn_heads.reshape(num_tokens, self.q_size)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        q, _ = self.q_proj(hidden_states)
        k, _ = self.k_proj(hidden_states)
        v, _ = self.v_proj(hidden_states)
        if self.sconv_k is not None:
            k = self.sconv_k(k, positions)
        if self.sconv_v is not None:
            v = self.sconv_v(v, positions)

        num_tokens = hidden_states.shape[0]
        q = _rms_norm(q.view(num_tokens, self.cfg.num_heads, self.head_dim))
        k = k.view(num_tokens, self.cfg.num_kv_heads, self.head_dim)
        v = v.view(num_tokens, self.cfg.num_kv_heads, self.head_dim)
        k = k[:, : self.logical_num_kv_heads, :]
        v = v[:, : self.logical_num_kv_heads, :]
        k = _rms_norm(k)
        q = q.reshape(num_tokens, self.q_size)
        k = k.reshape(num_tokens, self.logical_kv_size)
        v = v.reshape(num_tokens, self.logical_kv_size)
        if self.use_rope:
            q, k = self.rotary_emb(positions, q, k)
        q = q * self.cfg.qk_mult * self.qk_mult_scale

        attn_output = self.attn(q, k, v)
        attn_output = self.apply_xsa(hidden_states, attn_output, v)
        output, _ = self.o_proj(attn_output)
        return output


class GrugMoeDecoderLayer(nn.Module):
    def __init__(
        self,
        cfg: GrugMoeRuntimeConfig,
        cache_config: CacheConfig | None,
        params_dtype: torch.dtype,
        layer_index: int = 0,
        prefix: str = "",
    ) -> None:
        super().__init__()
        is_long = cfg.attention_layer_types[layer_index] == "full_attention"
        sliding_window = None if is_long else cfg.sliding_window
        self.input_layernorm = RMSNorm(cfg.hidden_dim, eps=cfg.layer_norm_eps)
        self.attn_gated_norm = GrugMoeGatedNorm(
            cfg.hidden_dim,
            params_dtype,
            prefix=f"{prefix}.attn_gated_norm",
        )
        self.self_attn = GrugMoeAttention(
            cfg,
            cache_config,
            params_dtype,
            sliding_window,
            logical_num_kv_heads=cfg.logical_kv_heads(is_long),
            use_rope=not (is_long and cfg.disable_long_rope),
            qk_mult_scale=cfg.qk_mult_long_scale if is_long else 1.0,
            prefix=f"{prefix}.self_attn",
        )
        self.post_attention_layernorm = RMSNorm(
            cfg.hidden_dim,
            eps=cfg.layer_norm_eps,
        )
        self.mlp_gated_norm = GrugMoeGatedNorm(
            cfg.hidden_dim,
            params_dtype,
            prefix=f"{prefix}.mlp_gated_norm",
        )
        self.mlp = GrugMoeMLP(
            cfg,
            params_dtype,
            prefix=f"{prefix}.mlp",
        )
        self.shared_expert = (
            GrugMoeDenseMLP(
                cfg.hidden_dim,
                cfg.shared_expert_intermediate_dim,
                params_dtype,
                prefix=f"{prefix}.shared_expert",
            )
            if cfg.shared_expert_intermediate_dim > 0 and cfg.num_shared_experts == 1
            else None
        )
        self.shared_experts = (
            nn.ModuleList(
                [
                    GrugMoeDenseMLP(
                        cfg.hidden_dim,
                        cfg.shared_expert_intermediate_dim // cfg.num_shared_experts,
                        params_dtype,
                        prefix=f"{prefix}.shared_experts.{expert_index}",
                    )
                    for expert_index in range(cfg.num_shared_experts)
                ]
            )
            if cfg.shared_expert_intermediate_dim > 0 and cfg.num_shared_experts > 1
            else None
        )
        self.sconv_attn = (
            GrugMoeShortConv(
                cfg.hidden_dim,
                cfg.sconv_kernel,
                num_heads=cfg.num_kv_heads,
                dtype=params_dtype,
                prefix=f"{prefix}.sconv_attn",
            )
            if cfg.sconv and "attn" in cfg.sconv_sites
            else None
        )
        self.sconv_mlp = (
            GrugMoeShortConv(
                cfg.hidden_dim,
                cfg.sconv_kernel,
                num_heads=cfg.num_kv_heads,
                dtype=params_dtype,
                prefix=f"{prefix}.sconv_mlp",
            )
            if cfg.sconv and "mlp" in cfg.sconv_sites
            else None
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        attn_in = self.attn_gated_norm(self.input_layernorm(hidden_states))
        attn_out = self.self_attn(positions, attn_in)
        if self.sconv_attn is not None:
            attn_out = self.sconv_attn(attn_out, positions)
        hidden_states = hidden_states + attn_out

        mlp_in = self.mlp_gated_norm(self.post_attention_layernorm(hidden_states))
        mlp_out = self.mlp(mlp_in)
        if self.shared_expert is not None:
            mlp_out = mlp_out + self.shared_expert(mlp_in)
        if self.shared_experts is not None:
            for shared_expert in self.shared_experts:
                mlp_out = mlp_out + shared_expert(mlp_in)
        if self.sconv_mlp is not None:
            mlp_out = self.sconv_mlp(mlp_out, positions)
        return hidden_states + mlp_out


@support_torch_compile
class GrugMoeModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        hf_config = getattr(vllm_config.model_config, "hf_text_config", None)
        if hf_config is None:
            hf_config = vllm_config.model_config.hf_config
        self.config = GrugMoeRuntimeConfig.from_hf_config(hf_config)
        if not self.config.disable_pko:
            raise NotImplementedError(
                "GrugMoE does not support Partial Key Offset; export a checkpoint "
                "with disable_pko=true"
            )
        self.params_dtype = vllm_config.model_config.dtype

        self.embed_tokens = VocabParallelEmbedding(
            self.config.vocab_size,
            self.config.hidden_dim,
            params_dtype=self.params_dtype,
            quant_config=None,
            prefix=f"{prefix}.embed_tokens",
        )
        self.embed_norm = RMSNorm(
            self.config.hidden_dim,
            eps=self.config.layer_norm_eps,
        )
        self.embed_gated_norm = (
            GrugMoeGatedNorm(
                self.config.hidden_dim,
                self.params_dtype,
                prefix=f"{prefix}.embed_gated_norm",
            )
            if self.config.legacy_input_output_gated_norm
            else None
        )
        self.layers = nn.ModuleList(
            [
                GrugMoeDecoderLayer(
                    self.config,
                    vllm_config.cache_config,
                    self.params_dtype,
                    layer_index,
                    prefix=f"{prefix}.layers.{layer_index}",
                )
                for layer_index in range(self.config.num_layers)
            ]
        )
        self.norm = RMSNorm(self.config.hidden_dim, eps=self.config.layer_norm_eps)
        self.final_gated_norm = (
            GrugMoeGatedNorm(
                self.config.hidden_dim,
                self.params_dtype,
                prefix=f"{prefix}.final_gated_norm",
            )
            if self.config.legacy_input_output_gated_norm
            else None
        )
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states"], self.config.hidden_dim
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if intermediate_tensors is not None:
            raise NotImplementedError("GrugMoE does not support pipeline parallelism")
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            if input_ids is None:
                raise ValueError(
                    "input_ids must be provided when inputs_embeds is None"
                )
            hidden_states = self.embed_input_ids(input_ids)

        hidden_states = self.embed_norm(hidden_states)
        if self.embed_gated_norm is not None:
            hidden_states = self.embed_gated_norm(hidden_states)
        for layer in self.layers:
            hidden_states = layer(positions, hidden_states)
        hidden_states = self.norm(hidden_states)
        if self.final_gated_norm is not None:
            hidden_states = self.final_gated_norm(hidden_states)
        return hidden_states


def _raise_for_unsupported_modes(vllm_config: VllmConfig) -> None:
    parallel_config = vllm_config.parallel_config
    if parallel_config.pipeline_parallel_size != 1:
        raise NotImplementedError("GrugMoE currently supports pipeline_parallel_size=1")
    if get_pp_group().world_size != 1:
        raise NotImplementedError("GrugMoE currently supports pipeline_parallel_size=1")
    if vllm_config.lora_config is not None:
        raise NotImplementedError("GrugMoE does not support LoRA")
    if vllm_config.quant_config is not None:
        raise NotImplementedError("GrugMoE does not support quantization")
    if parallel_config.tensor_parallel_size != 1:
        raise NotImplementedError(
            "GrugMoE expert-parallel GPU serving requires tensor_parallel_size=1"
        )
    if parallel_config.tensor_parallel_size != get_tensor_model_parallel_world_size():
        raise RuntimeError(
            "GrugMoE tensor_parallel_size does not match the initialized "
            "tensor-parallel world size: "
            f"{parallel_config.tensor_parallel_size} != "
            f"{get_tensor_model_parallel_world_size()}"
        )


_EXPERT_WEIGHT_MAPPING: tuple[tuple[str, str, str], ...] = (
    (
        "experts.gate_proj.weight",
        "experts.routed_experts.w13_weight",
        "w1",
    ),
    (
        "experts.down_proj.weight",
        "experts.routed_experts.w2_weight",
        "w2",
    ),
    (
        "experts.up_proj.weight",
        "experts.routed_experts.w13_weight",
        "w3",
    ),
)


def _try_load_grug_expert_weight(
    name: str,
    loaded_weight: torch.Tensor,
    params_dict: dict[str, nn.Parameter],
    *,
    expected_experts: int,
) -> str | None:
    for weight_name, param_name, shard_id in _EXPERT_WEIGHT_MAPPING:
        if weight_name not in name:
            continue
        mapped_name = name.replace(weight_name, param_name)
        param = params_dict.get(mapped_name)
        if param is None:
            raise ValueError(
                "Mapped Grug expert weight "
                f"{name!r} to missing parameter {mapped_name!r}"
            )
        weight_loader = getattr(param, "weight_loader", None)
        if weight_loader is None:
            raise ValueError(
                f"Grug expert parameter {mapped_name!r} has no FusedMoE weight_loader"
            )
        if loaded_weight.dim() != 3:
            raise ValueError(
                "Expected stacked 3D Grug expert weight "
                f"{name!r}, got shape {tuple(loaded_weight.shape)}"
            )
        if loaded_weight.shape[0] != expected_experts:
            raise ValueError(
                f"Expected {expected_experts} stacked Grug experts for "
                f"{name!r}, got {loaded_weight.shape[0]}"
            )
        loaded_experts = loaded_weight.unbind(dim=0)
        for expert_id, loaded_expert in enumerate(loaded_experts):
            weight_loader(
                param,
                loaded_expert,
                mapped_name,
                shard_id=shard_id,
                expert_id=expert_id,
                return_success=True,
            )
        return mapped_name
    return None


class GrugMoeForCausalLM(nn.Module):
    fall_back_to_pt_during_load = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        _raise_for_unsupported_modes(vllm_config)
        hf_config = getattr(vllm_config.model_config, "hf_text_config", None)
        if hf_config is None:
            hf_config = vllm_config.model_config.hf_config
        self.config = GrugMoeRuntimeConfig.from_hf_config(hf_config)
        self.tie_word_embeddings = bool(
            getattr(hf_config, "tie_word_embeddings", False)
        )
        self.quant_config: QuantizationConfig | None = None
        self.model = GrugMoeModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )
        _log_grug_moe_runtime_info(vllm_config, self)
        self.lm_head = ParallelLMHead(
            self.config.vocab_size,
            self.config.hidden_dim,
            params_dtype=vllm_config.model_config.dtype,
            quant_config=None,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        if self.tie_word_embeddings:
            self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)
        self.logits_processor = LogitsProcessor(self.config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

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
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        unused_substrings = (
            "rotary_pos_emb.inv_freq",
            "rotary_emb.inv_freq",
            "rotary_emb.cos_cached",
            "rotary_emb.sin_cached",
        )

        for name, loaded_weight in weights:
            if self.tie_word_embeddings and name.startswith("lm_head."):
                continue
            if any(substring in name for substring in unused_substrings):
                continue

            if (
                mapped_name := _try_load_grug_expert_weight(
                    name,
                    loaded_weight,
                    params_dict,
                    expected_experts=self.config.num_experts,
                )
            ) is not None:
                loaded_params.add(mapped_name)
                continue

            param = params_dict.get(name)
            if param is None:
                raise ValueError(
                    f"Unexpected GrugMoE weight {name!r}; parameter was not found"
                )
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)

        return loaded_params


__all__ = [
    "GrugMoeAttention",
    "GrugMoeDecoderLayer",
    "GrugMoeDenseMLP",
    "GrugMoeForCausalLM",
    "GrugMoeGatedNorm",
    "GrugMoeMLP",
    "GrugMoeModel",
    "GrugMoeRuntimeConfig",
    "GrugMoeRouter",
    "get_grug_moe_runtime_info",
]
