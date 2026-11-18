# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace

import pytest
import torch

from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts
from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEKernel
from vllm.model_executor.layers.fused_moe.prepare_finalize.naive_dp_ep import (
    MoEPrepareAndFinalizeNaiveDPEPModular,
)
from vllm.platforms import current_platform
from vllm.v1.worker import workspace


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    ("dispatchers", "dp", "sp", "gathered_tokens"),
    [(1, 1, 1, 5), (16, 16, 1, 80), (16, 8, 2, 48), (32, 16, 1, 80)],
)
@pytest.mark.parametrize("activation", [MoEActivation.SILU, MoEActivation.RELU2_NO_MUL])
@pytest.mark.parametrize(("hidden", "intermediate"), [(17, 29), (37, 11)])
def test_gathered_moe_workspace_never_grows_during_forward(
    monkeypatch,
    dispatchers: int,
    dp: int,
    sp: int,
    gathered_tokens: int,
    activation: MoEActivation,
    hidden: int,
    intermediate: int,
) -> None:
    """Reproduce the allocator-flush boundary without launching a collective."""
    # Exercise the GPU allocation path using real CPU tensor storage.
    monkeypatch.setattr(current_platform, "is_cpu", lambda: False)
    monkeypatch.setattr(current_platform, "is_cuda_alike", lambda: True)
    ubatch = [0]
    monkeypatch.setattr(workspace, "dbo_current_ubatch_id", lambda: ubatch[0])
    manager = workspace.WorkspaceManager(
        torch.device("cpu"), num_ubatches=2, num_lanes=2
    )
    monkeypatch.setattr(workspace, "_manager", manager)
    moe = FusedMoEConfig(
        num_experts=16,
        experts_per_token=2,
        hidden_dim=hidden,
        intermediate_size=intermediate,
        num_local_experts=1,
        num_logical_experts=16,
        activation=activation,
        device="cpu",
        routing_method=RoutingMethodType.TopK,
        moe_parallel_config=replace(
            FusedMoEParallelConfig.make_no_parallel(),
            dp_size=dp,
            sp_size=sp,
            ep_size=dispatchers,
            use_ep=True,
            all2all_backend="allgather_reducescatter",
        ),
        in_dtype=torch.bfloat16,
        max_num_tokens=5,
    )
    experts = TritonExperts(moe, FusedMoEQuantConfig.make(None))
    kernel = FusedMoEKernel(
        MoEPrepareAndFinalizeNaiveDPEPModular(
            num_dispatchers=dispatchers, is_sequence_parallel=sp > 1
        ),
        experts,
    )

    def blocked_cache_flush() -> None:
        raise RuntimeError("allocator waited on an in-flight dispatch")

    monkeypatch.setattr(torch.accelerator, "empty_cache", blocked_cache_flush)
    N = intermediate * (2 if activation.is_gated else 1)
    # Without setup reservation, this first request reaches the failing flush.
    kernel.impl._allocate_buffers(
        torch.bfloat16,
        torch.device("cpu"),
        1,
        1,
        N,
        hidden,
        2,
        16,
        1,
        None,
        activation,
    )
    # Captured graphs must keep the same backing allocations for every owner.
    manager.lock()
    # Rebuilding a kernel after a weight update must preserve those addresses.
    kernel.prepare_finalize.post_init_setup(experts)
    pointers = set()
    for ubatch_id in range(2):
        ubatch[0] = ubatch_id
        for lane in range(2):
            with workspace.use_workspace_lane(lane):
                pointer = None
                for tokens in (1, 3, gathered_tokens):
                    scratch13, scratch2, output = kernel.impl._allocate_buffers(
                        torch.bfloat16,
                        torch.device("cpu"),
                        tokens,
                        tokens,
                        N,
                        hidden,
                        2,
                        16,
                        1,
                        None,
                        activation,
                    )
                    assert output.shape == (tokens, hidden)
                    scratch13.fill_(1)
                    scratch2.fill_(2)
                    torch.testing.assert_close(scratch13, torch.ones_like(scratch13))
                    torch.testing.assert_close(scratch2, torch.full_like(scratch2, 2))
                    if pointer is None:
                        pointer = scratch13.data_ptr()
                        pointers.add(pointer)
                    assert scratch13.data_ptr() == pointer
                    assert output.data_ptr() == pointer
    assert len(pointers) == 4
    # TP ranks outside the actual gather group must not inflate the reservation.
    over_budget = gathered_tokens + moe.max_num_tokens
    with pytest.raises(AssertionError, match="Workspace is locked"):
        kernel.impl._allocate_buffers(
            torch.bfloat16,
            torch.device("cpu"),
            over_budget,
            over_budget,
            N,
            hidden,
            2,
            16,
            1,
            None,
            activation,
        )
