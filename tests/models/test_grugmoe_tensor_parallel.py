# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Distributed Grug dense projections, checkpoint loading, and cache geometry."""

import math
import os

import pytest
import torch
import torch.nn.functional as F
from torch.multiprocessing import spawn

from tests.utils import multi_gpu_marks
from vllm.config import CUDAGraphMode, KernelConfig, VllmConfig, set_current_vllm_config
from vllm.config.parallel import ParallelConfig
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.engine.arg_utils import EngineArgs
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.grugmoe import (
    GrugMoeAttention,
    GrugMoeDenseMLP,
    GrugMoeGatedNorm,
    GrugMoeMLP,
    GrugMoeRuntimeConfig,
    GrugMoeShortConv,
    _try_load_grug_expert_weight,
)
from vllm.platforms import current_platform
from vllm.transformers_utils.configs.grugmoe import GrugMoeConfig
from vllm.v1.worker.workspace import init_workspace_manager


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
@pytest.mark.parametrize("conflict", [
    pytest.param(conflict, marks=multi_gpu_marks(num_gpus=2))
    for conflict in (
        None, "defaults", "custom", "fusion", "fusion-default",
        "sequence-parallel", "gemm-comms",
        "VLLM_ALLREDUCE_USE_FLASHINFER", "VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC",
        "VLLM_ALLREDUCE_USE_SYMM_MEM", "VLLM_USE_NCCL_SYMM_MEM",
    )
])
def test_hero_tp2_launch_requires_qualified_reductions(tmp_path, monkeypatch, conflict):
    """Reject launches that could reintroduce the observed cache-state failures."""
    GrugMoeConfig(
        architectures=["GrugMoeForCausalLM"], grugmoe_artifact_schema_version=2,
        vocab_size=384, hidden_size=192, moe_intermediate_size=192,
        shared_expert_intermediate_size=192, num_experts=12, num_experts_per_tok=2,
        latent_dim=192, num_hidden_layers=2, num_attention_heads=48,
        num_key_value_heads=12, local_kv_heads=12, global_kv_heads=6, head_dim=64,
        max_position_embeddings=512, sliding_window=16, global_every=2,
        sconv=True, disable_pko=True,
    ).save_pretrained(tmp_path)
    for name in (
        "VLLM_ALLREDUCE_USE_FLASHINFER", "VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC",
        "VLLM_ALLREDUCE_USE_SYMM_MEM", "VLLM_USE_NCCL_SYMM_MEM",
    ):
        if conflict == "defaults":
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, "1" if conflict == name else "0")
    fusion = None if conflict == "fusion-default" else conflict == "fusion"
    args = EngineArgs(
        model=str(tmp_path), skip_tokenizer_init=True, dtype="bfloat16",
        tensor_parallel_size=2, disable_custom_all_reduce=conflict != "custom",
        compilation_config={
            "mode": 3, "cudagraph_mode": "PIECEWISE",
            "splitting_ops": ["vllm::grug_moe_short_conv"],
            "pass_config": {
                "fuse_allreduce_rms": fusion,
                "enable_sp": conflict == "sequence-parallel",
                "fuse_gemm_comms": conflict == "gemm-comms",
            },
        },
    )
    if conflict is not None:
        with pytest.raises(ValueError):
            args.create_engine_config()
        return
    config = args.create_engine_config()
    assert config.compilation_config.cudagraph_mode == CUDAGraphMode.PIECEWISE
    assert not config.compilation_config.pass_config.fuse_allreduce_rms


def _load_projection(module, weight):
    module.weight.weight_loader(module.weight, weight)


def _tp_projection_worker(rank, tp_size, init_file):
    torch.cuda.set_device(rank)
    config = VllmConfig(
        parallel_config=ParallelConfig(
            tensor_parallel_size=tp_size, enable_expert_parallel=True,
        ),
        kernel_config=KernelConfig(moe_backend="triton"),
    )
    with set_current_vllm_config(config):
        init_distributed_environment(
            world_size=tp_size, rank=rank, distributed_init_method=f"file://{init_file}",
            local_rank=rank, backend="nccl",
        )
        initialize_model_parallel(tp_size, 1)
        try:
            with torch.device(f"cuda:{rank}"), torch.no_grad():
                _check_tp_projections(rank, tp_size)
        finally:
            cleanup_dist_env_and_memory()


def _check_tp_projections(rank, tp_size):
    torch.manual_seed(341)
    cfg = GrugMoeRuntimeConfig(
        vocab_size=128, artifact_schema_version=2, hidden_dim=192,
        intermediate_dim=192, shared_expert_intermediate_dim=192,
        num_shared_experts=2, num_experts=12, num_experts_per_token=2,
        latent_dim=192, num_layers=1, num_heads=48, num_kv_heads=12,
        local_kv_heads=12, global_kv_heads=6, head_dim=64,
        max_seq_len=32, sliding_window=16, initializer_std=0.02,
        logical_kv_cache=True, disable_pko=True, sconv=True, sconv_kernel=4,
        sconv_sites=("k", "attn", "mlp"),
    ).validate()
    x = torch.randn(7, cfg.hidden_dim)

    norm = GrugMoeGatedNorm(cfg.hidden_dim, torch.float32)
    norm_down = torch.randn(128, cfg.hidden_dim) * 0.03
    norm_up = torch.randn(cfg.hidden_dim, 128) * 0.03
    _load_projection(norm.down_proj, norm_down)
    _load_projection(norm.up_proj, norm_up)
    torch.testing.assert_close(
        norm(x), x * torch.sigmoid(F.linear(F.silu(F.linear(x, norm_down)), norm_up))
    )

    shared = GrugMoeDenseMLP(cfg.hidden_dim, 192, torch.float32)
    gate, up, down = [torch.randn(192, 192) * 0.03 for _ in range(3)]
    for module, weight in zip(
        (shared.gate_proj, shared.up_proj, shared.down_proj), (gate, up, down)
    ):
        _load_projection(module, weight)
    assert shared.gate_proj.weight.numel() == gate.numel() // tp_size
    assert shared.down_proj.weight.numel() == down.numel() // tp_size
    torch.testing.assert_close(
        shared(x), F.linear(F.silu(F.linear(x, gate)) * F.linear(x, up), down),
        atol=1e-6, rtol=1e-5,
    )

    moe = GrugMoeMLP(cfg, torch.float32, prefix="latent")
    latent_down = torch.randn(192, 192) * 0.03
    latent_up = torch.randn(192, 192) * 0.03
    _load_projection(moe.latent_down_proj, latent_down)
    _load_projection(moe.latent_up_proj, latent_up)
    moe.latent_norm.weight.fill_(1)
    expected_latent = F.linear(x, latent_down)
    expected_latent *= torch.rsqrt(
        expected_latent.square().mean(-1, keepdim=True) + cfg.layer_norm_eps
    )
    actual_latent = moe._routed_input(x)
    torch.testing.assert_close(actual_latent, expected_latent, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(
        moe._expand_routed_output(actual_latent),
        F.linear(expected_latent, latent_up), atol=1e-6, rtol=1e-5,
    )
    assert moe.latent_down_proj.weight.numel() == latent_down.numel()
    assert moe.latent_up_proj.weight.numel() == latent_up.numel()

    for is_global in (False, True):
        attention = GrugMoeAttention(
            cfg, cache_config=None, params_dtype=torch.float32,
            sliding_window=None, use_rope=False, qk_mult_scale=1,
            is_global=is_global, prefix=f"attention.{is_global}",
        )
        weights = {
            "q_proj": torch.randn(48 * 64, 192) * 0.03,
            "k_proj": torch.randn(12 * 64, 192) * 0.03,
            "v_proj": torch.randn(12 * 64, 192) * 0.03,
            "o_proj": torch.randn(192, 48 * 64) * 0.03,
            "attn_gate": torch.randn(48, 192) * 0.03,
        }
        for name, weight in weights.items():
            module = getattr(attention, name)
            _load_projection(module, weight)
            heads = (6 if is_global else 12) if name in ("k_proj", "v_proj") else 12
            assert module.weight.numel() == weight.numel() * heads // 12 // tp_size
        conv = torch.randn(4, 12 * 64) * 0.03
        attention.sconv_k.weight.weight_loader(attention.sconv_k.weight, conv)
        logical_heads = 6 if is_global else 12
        head_indices = torch.arange(logical_heads)
        head_indices = head_indices.chunk(tp_size)[rank]
        expected_k = F.linear(x, weights["k_proj"]).view(7, 12, 64)
        expected_v = F.linear(x, weights["v_proj"]).view(7, 12, 64)
        actual_k = attention.k_proj(x)[0]
        actual_v = attention.v_proj(x)[0]
        torch.testing.assert_close(actual_k, expected_k[:, head_indices].flatten(1))
        torch.testing.assert_close(actual_v, expected_v[:, head_indices].flatten(1))
        torch.testing.assert_close(
            attention.sconv_k.weight, conv.view(4, 12, 64)[:, head_indices].flatten(1)
        )
        assert math.prod(attention.sconv_k.get_state_shape()[0]) == (
            logical_heads * 64 // tp_size * 3
        )
        q = attention.q_proj(x)[0]
        torch.testing.assert_close(
            q, F.linear(x, weights["q_proj"]).chunk(tp_size, -1)[rank]
        )

        # Head-local XSA and the real O-projection collective must recover
        # the unsharded, independent reference even across global-KV6 boundaries.
        full_attn = torch.randn(7, 48, 64)
        aligned_v = expected_v[:, :logical_heads].repeat_interleave(
            48 // logical_heads, dim=1
        )
        corrected = full_attn - (
            (full_attn * aligned_v).sum(-1, keepdim=True)
            / (aligned_v.square().sum(-1, keepdim=True) + 1e-6)
        ) * aligned_v
        corrected *= 2 * torch.sigmoid(F.linear(x, weights["attn_gate"]))[..., None]
        local_attn = full_attn.chunk(tp_size, dim=1)[rank].flatten(1)
        actual_xsa = attention.apply_xsa(x, local_attn, actual_v)
        actual = attention.o_proj(actual_xsa)[0]
        expected = F.linear(corrected.flatten(1), weights["o_proj"])
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-5)

    output_conv = GrugMoeShortConv(192, 4, torch.float32, None, "output_conv")
    assert math.prod(output_conv.get_state_shape()[0]) == 192 * 3
    head = ParallelLMHead(192, 192, params_dtype=torch.float32)
    head_weight = torch.randn(192, 192) * 0.03
    _load_projection(head, head_weight)
    logits = LogitsProcessor(192)(head, x)
    if logits is not None:
        torch.testing.assert_close(logits, F.linear(x, head_weight))


@pytest.mark.parametrize("tp_size", [
    pytest.param(2, marks=multi_gpu_marks(num_gpus=2)),
    pytest.param(6, marks=multi_gpu_marks(num_gpus=6)),
])
def test_grug_tp_projections_match_unsharded_checkpoint(tp_size, tmp_path):
    spawn(_tp_projection_worker, args=(tp_size, str(tmp_path / "init")),
          nprocs=tp_size, join=True)


def _tp_moe_worker(rank, tp_size, dp_size, init_file):
    torch.cuda.set_device(rank)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(tp_size * dp_size)
    dp_rank = rank // tp_size
    config = VllmConfig(
        parallel_config=ParallelConfig(
            tensor_parallel_size=tp_size, data_parallel_size=dp_size,
            data_parallel_rank=dp_rank, enable_expert_parallel=True,
            distributed_executor_backend="external_launcher",
        ),
        kernel_config=KernelConfig(moe_backend="triton"),
    )
    with set_current_vllm_config(config):
        init_distributed_environment(
            world_size=tp_size * dp_size, rank=rank,
            distributed_init_method=f"file://{init_file}", local_rank=rank,
            backend="nccl",
        )
        initialize_model_parallel(tp_size, 1)
        try:
            with torch.device(f"cuda:{rank}"), torch.no_grad():
                _check_tp_moe(config, dp_rank, dp_size)
        finally:
            cleanup_dist_env_and_memory()


def _check_tp_moe(config, dp_rank, dp_size):
    torch.manual_seed(823)
    # Preserve the original TP2/TP4 fixture; TP6 needs twelve experts.
    width, num_experts = (
        (192, 12) if config.parallel_config.tensor_parallel_size == 6 else (128, 8)
    )
    cfg = GrugMoeRuntimeConfig(
        vocab_size=width, artifact_schema_version=2, hidden_dim=width,
        intermediate_dim=width, shared_expert_intermediate_dim=width,
        num_shared_experts=2, num_experts=num_experts, num_experts_per_token=2,
        latent_dim=width, num_layers=1, num_heads=4, num_kv_heads=4,
        head_dim=32, max_seq_len=32, sliding_window=16, initializer_std=0.02,
        disable_pko=True,
    ).validate()
    module = GrugMoeMLP(cfg, torch.bfloat16, prefix="moe")
    latent_down = (torch.randn(width, width) * 0.03).bfloat16()
    latent_up = (torch.randn(width, width) * 0.03).bfloat16()
    gate = (torch.randn(num_experts, width, width) * 0.03).bfloat16()
    up = (torch.randn(num_experts, width, width) * 0.03).bfloat16()
    down = (torch.randn(num_experts, width, width) * 0.03).bfloat16()
    _load_projection(module.latent_down_proj, latent_down)
    _load_projection(module.latent_up_proj, latent_up)
    module.latent_norm.weight.fill_(1)
    module.router.weight.zero_()
    module.router.weight[:, :num_experts].copy_(3 * torch.eye(num_experts))
    module.router.bias.copy_(torch.arange(num_experts) * 0.01)
    params = dict(module.named_parameters())
    for name, bank in (("gate_proj", gate), ("up_proj", up), ("down_proj", down)):
        for expert_id, weight in enumerate(bank):
            _try_load_grug_expert_weight(
                f"experts.{expert_id}.{name}.weight", weight, params
            )
    experts = module.experts.routed_experts
    experts.quant_method.process_weights_after_loading(experts)

    # TP peers receive identical tokens; DP peers receive different token IDs
    # and lengths so EP gather/scatter cannot pass by returning another DP's batch.
    tokens = 7 + 2 * dp_rank
    x = (torch.randn(tokens, width) * 0.3).bfloat16()
    x[:, :num_experts] = F.one_hot(
        (torch.arange(tokens) + 3 * dp_rank) % num_experts, num_experts
    ).bfloat16()
    latent = F.linear(x, latent_down).float()
    latent *= torch.rsqrt(latent.square().mean(-1, keepdim=True) + cfg.layer_norm_eps)
    latent = latent.bfloat16()
    logits = F.linear(x.float(), module.router.weight).bfloat16().float()
    selected = (logits + module.router.bias).topk(3, dim=-1).indices[:, :2]
    combine = torch.sigmoid(logits.gather(1, selected))
    combine *= 2.5 / (combine.sum(-1, keepdim=True) + 1e-9)
    expected = torch.zeros(tokens, width, dtype=torch.float32)
    for token in range(tokens):
        for slot in range(2):
            expert_id = selected[token, slot]
            gate_value = F.linear(latent[token].float(), gate[expert_id].float())
            up_value = F.linear(latent[token].float(), up[expert_id].float())
            expert_value = F.linear(
                F.silu(gate_value) * up_value, down[expert_id].float()
            )
            expected[token] += combine[token, slot] * expert_value
    expected = F.linear(expected, latent_up.float())

    init_workspace_manager(torch.device("cuda"))
    with set_forward_context(
        None, config, num_tokens=tokens,
        num_tokens_across_dp=torch.tensor(
            [7 + 2 * dp for dp in range(dp_size)], device="cpu", dtype=torch.int32
        ),
    ):
        actual = module(x)
    # The independent oracle accumulates in FP32; the fused expert kernels and
    # latent projections round BF16 intermediate values. This absolute bound
    # applies to this fixed fixture, including output values near zero.
    max_error = (actual.float() - expected).abs().max().item()
    assert max_error < 5e-4, f"BF16 MoE max absolute error: {max_error}"
    assert actual.shape == (tokens, width)


@pytest.mark.parametrize(("tp_size", "dp_size"), [
    pytest.param(2, 1, marks=multi_gpu_marks(num_gpus=2)),
    pytest.param(4, 1, marks=multi_gpu_marks(num_gpus=4)),
    pytest.param(6, 1, marks=multi_gpu_marks(num_gpus=6)),
    pytest.param(2, 2, marks=multi_gpu_marks(num_gpus=4)),
])
def test_grug_tp_ep_bf16_moe_matches_unsharded_reference(tp_size, dp_size, tmp_path):
    spawn(_tp_moe_worker, args=(tp_size, dp_size, str(tmp_path / "init")),
          nprocs=tp_size * dp_size, join=True)
