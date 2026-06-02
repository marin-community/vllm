# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.nn.functional as F

from vllm.model_executor.models.grugmoe import GrugMoeConfig, GrugMoeMLP


def _tiny_config() -> GrugMoeConfig:
    return GrugMoeConfig(
        vocab_size=32,
        hidden_dim=4,
        intermediate_dim=3,
        shared_expert_intermediate_dim=0,
        num_experts=4,
        num_experts_per_token=2,
        num_layers=1,
        num_heads=2,
        num_kv_heads=1,
        head_dim=2,
        max_seq_len=8,
        sliding_window=8,
        initializer_std=0.02,
    ).validate()


def _reference_grug_moe(
    x: torch.Tensor,
    router: torch.Tensor,
    router_bias: torch.Tensor,
    w_gate_up: torch.Tensor,
    w_down: torch.Tensor,
    num_experts_per_token: int,
    intermediate_dim: int,
) -> torch.Tensor:
    router_logits = x.float() @ router.float()
    biased_logits = router_logits + router_bias.float()
    _, top_indices = torch.topk(
        biased_logits,
        k=num_experts_per_token + 1,
        dim=-1,
    )
    selected = top_indices[:, :num_experts_per_token]
    weights = torch.sigmoid(torch.gather(router_logits, dim=-1, index=selected))

    outs: list[torch.Tensor] = []
    for token_idx, token in enumerate(x):
        token_out = torch.zeros_like(token)
        for slot_idx, expert_idx in enumerate(selected[token_idx]):
            expert_id = int(expert_idx.item())
            projected = token @ w_gate_up[expert_id]
            gate, up = torch.split(projected, intermediate_dim, dim=-1)
            expert_out = (F.silu(gate) * up) @ w_down[expert_id]
            token_out = token_out + weights[token_idx, slot_idx] * expert_out
        outs.append(token_out)
    return torch.stack(outs, dim=0)


def test_grug_moe_uses_qb_bias_for_selection_and_unbiased_sigmoid_weights():
    cfg = _tiny_config()
    mlp = GrugMoeMLP(cfg)
    x = torch.tensor(
        [
            [0.3, -0.4, 0.2, 0.1],
            [-0.5, 0.7, -0.2, 0.4],
        ],
        dtype=torch.float32,
    )
    router = torch.tensor(
        [
            [0.7, -0.2, 0.1, -0.1],
            [-0.3, 0.4, -0.5, 0.2],
            [0.2, -0.6, 0.3, 0.8],
            [0.1, 0.5, -0.4, -0.3],
        ],
        dtype=torch.float32,
    )
    router_bias = torch.tensor([0.0, 0.0, 3.0, -2.0], dtype=torch.float32)

    generator = torch.Generator().manual_seed(1234)
    w_gate_up = torch.randn(
        cfg.num_experts,
        cfg.hidden_dim,
        2 * cfg.intermediate_dim,
        generator=generator,
    )
    w_down = torch.randn(
        cfg.num_experts,
        cfg.intermediate_dim,
        cfg.hidden_dim,
        generator=generator,
    )

    with torch.no_grad():
        mlp.router.copy_(router)
        mlp.router_bias.copy_(router_bias)
        mlp.w_gate_up.copy_(w_gate_up)
        mlp.w_down.copy_(w_down)

    actual = mlp(x)
    expected = _reference_grug_moe(
        x,
        router,
        router_bias,
        w_gate_up,
        w_down,
        cfg.num_experts_per_token,
        cfg.intermediate_dim,
    )

    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
