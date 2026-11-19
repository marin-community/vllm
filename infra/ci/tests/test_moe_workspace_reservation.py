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
def test_setup_reserves_the_current_slot_before_dispatch(
    monkeypatch,
    dispatchers: int,
    dp: int,
    sp: int,
    gathered_tokens: int,
    activation: MoEActivation,
    hidden: int,
    intermediate: int,
) -> None:
    """The setup reservation prevents the in-forward allocator flush.

    Reproduce the allocator-flush boundary without launching a collective:
    block torch.accelerator.empty_cache, build the kernel, and require every
    in-forward workspace request inside the dispatch bound to be satisfied by
    the setup-time reservation instead of a flush.
    """
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
    # The setup reservation mirrors the Triton MLA pattern: it grows the
    # requesting (ubatch, lane) slot only. The draft lane owns its buffer and
    # remaining ubatch slots grow during warmup.
    reserved = manager._workspace_size_bytes(manager._current_workspaces[0])
    assert reserved > 0
    for slot in range(1, 4):
        assert manager._current_workspaces[slot] is None
    # Every in-forward request at the reserved slot stays inside the bound and
    # needs no flush.
    pointer = None
    for tokens in (1, 3, gathered_tokens):
        ubatch[0] = 0
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
        assert scratch13.data_ptr() == pointer
        assert output.data_ptr() == pointer
    # Unreserved slots follow the upstream lazy-growth contract: the first
    # request reaches the allocator flush the setup reservation avoids.
    for slot_ubatch, lane in ((1, 0), (0, 1)):
        ubatch[0] = slot_ubatch
        with (
            workspace.use_workspace_lane(lane),
            pytest.raises(RuntimeError, match="in-flight dispatch"),
        ):
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
    ubatch[0] = 0
    manager.lock()
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
    # Unreserved slots stay locked out: growth there is not pre-grown either.
    ubatch[0] = 1
    with pytest.raises(AssertionError, match="Workspace is locked"):
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
    ubatch[0] = 0
    # Rebuilding a kernel after a weight update must preserve the reserved
    # addresses: the deferred setup does not grow or move the buffer.
    kernel.prepare_finalize.post_init_setup(experts)
    scratch13, _, output = kernel.impl._allocate_buffers(
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
    assert scratch13.data_ptr() == pointer
    assert output.data_ptr() == pointer


@pytest.mark.cpu_test
@pytest.mark.parametrize("activation", [MoEActivation.SILU])
def test_locked_setup_defers_to_the_graph_release_and_warmup_sequence(
    monkeypatch, activation: MoEActivation
) -> None:
    """Elastic scale-up rebuilds kernels while graphs lock the workspace.

    post_init_setup must neither grow nor fail in that state: the existing
    graph-release, unlock and warm-up sequence owns the growth before the next
    capture. After unlock, the same setup reservation applies again, still to
    the requesting slot only.
    """
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
        hidden_dim=17,
        intermediate_size=29,
        num_local_experts=1,
        num_logical_experts=16,
        activation=activation,
        device="cpu",
        routing_method=RoutingMethodType.TopK,
        moe_parallel_config=replace(
            FusedMoEParallelConfig.make_no_parallel(),
            dp_size=16,
            ep_size=16,
            use_ep=True,
            all2all_backend="allgather_reducescatter",
        ),
        in_dtype=torch.bfloat16,
        max_num_tokens=5,
    )
    experts = TritonExperts(moe, FusedMoEQuantConfig.make(None))
    # Initial setup: the reservation may flush the allocator, it runs outside
    # any forward. The kernel build that follows re-runs the reservation and
    # finds the slot large enough, so it must not grow again.
    experts.reserve_workspace(moe.max_num_tokens * 16)
    reserved = manager._workspace_size_bytes(manager._current_workspaces[0])
    assert reserved > 0
    kernel = FusedMoEKernel(
        MoEPrepareAndFinalizeNaiveDPEPModular(num_dispatchers=16),
        experts,
    )
    assert manager._workspace_size_bytes(manager._current_workspaces[0]) == reserved

    def blocked_cache_flush() -> None:
        raise RuntimeError("allocator waited on an in-flight dispatch")

    monkeypatch.setattr(torch.accelerator, "empty_cache", blocked_cache_flush)
    manager.lock()
    # Deferred: no flush attempt, no growth, no error.
    experts.reserve_workspace(moe.max_num_tokens * 16)
    kernel.prepare_finalize.post_init_setup(experts)
    assert manager._workspace_size_bytes(manager._current_workspaces[0]) == reserved
    for slot in range(1, 4):
        assert manager._current_workspaces[slot] is None
    manager.unlock()
    # After the release/unlock step the growth is legitimate again (the warm-up
    # dummy run owns it in the field); the setup reservation applies to the
    # requesting slot only. A larger bound (what scale-up needs) grows here.
    flushes: list[int] = []
    monkeypatch.setattr(torch.accelerator, "empty_cache", lambda: flushes.append(1))
    experts.reserve_workspace(moe.max_num_tokens * 24)
    assert flushes == [1]
    assert manager._workspace_size_bytes(manager._current_workspaces[0]) > reserved
    for slot in range(1, 4):
        assert manager._current_workspaces[slot] is None


@pytest.mark.cpu_test
def test_kernel_setup_without_a_workspace_manager_stays_inert(
    monkeypatch,
) -> None:
    """Without a manager, construction is inert and the flush boundary returns.

    The setup reservation is what moves the workspace growth out of the
    in-flight forward; with no reservation, the first forward reaches the
    allocator flush that wedged startup in the field.
    """
    monkeypatch.setattr(current_platform, "is_cpu", lambda: False)
    monkeypatch.setattr(current_platform, "is_cuda_alike", lambda: True)
    monkeypatch.setattr(workspace, "dbo_current_ubatch_id", lambda: 0)
    assert not workspace.is_workspace_manager_initialized()
    moe = FusedMoEConfig(
        num_experts=16,
        experts_per_token=2,
        hidden_dim=17,
        intermediate_size=29,
        num_local_experts=1,
        num_logical_experts=16,
        activation=MoEActivation.SILU,
        device="cpu",
        routing_method=RoutingMethodType.TopK,
        moe_parallel_config=replace(
            FusedMoEParallelConfig.make_no_parallel(),
            dp_size=16,
            ep_size=16,
            use_ep=True,
            all2all_backend="allgather_reducescatter",
        ),
        in_dtype=torch.bfloat16,
        max_num_tokens=5,
    )
    experts = TritonExperts(moe, FusedMoEQuantConfig.make(None))
    # Construction without a manager must not fail.
    kernel = FusedMoEKernel(
        MoEPrepareAndFinalizeNaiveDPEPModular(num_dispatchers=16),
        experts,
    )
    manager = workspace.WorkspaceManager(torch.device("cpu"))
    monkeypatch.setattr(workspace, "_manager", manager)

    def blocked_cache_flush() -> None:
        raise RuntimeError("allocator waited on an in-flight dispatch")

    monkeypatch.setattr(torch.accelerator, "empty_cache", blocked_cache_flush)
    N = 29 * 2
    with pytest.raises(RuntimeError, match="in-flight dispatch"):
        kernel.impl._allocate_buffers(
            torch.bfloat16,
            torch.device("cpu"),
            80,
            80,
            N,
            17,
            2,
            16,
            1,
            None,
            MoEActivation.SILU,
        )
