# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from vllm.config import VllmConfig
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
    fp8_w8a8_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts
from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEKernel
from vllm.model_executor.layers.fused_moe.prepare_finalize import (
    MoEPrepareAndFinalizeNoDPEPModular,
)
from vllm.model_executor.layers.fused_moe.prepare_finalize.naive_dp_ep import (
    MoEPrepareAndFinalizeNaiveDPEPModular,
)
from vllm.v1.worker.workspace import init_workspace_manager

from .modular_kernel_tools.parallel_utils import (
    ProcessGroupInfo,
    parallel_launch_with_config,
)


def _tensor_scale_worker(
    pgi: ProcessGroupInfo,
    vllm_config: VllmConfig,
    cpu_group,
    token_counts: tuple[int, int],
):
    init_workspace_manager(pgi.device)
    parallel = FusedMoEParallelConfig.make(
        tp_size_=1,
        pcp_size_=1,
        dp_size_=2,
        sp_size_=1,
        vllm_parallel_config=vllm_config.parallel_config,
    )
    moe = FusedMoEConfig(
        num_experts=4,
        experts_per_token=2,
        hidden_dim=64,
        intermediate_size=64,
        num_local_experts=2,
        num_logical_experts=4,
        activation=MoEActivation.SILU,
        device=pgi.device,
        routing_method=RoutingMethodType.TopK,
        moe_parallel_config=parallel,
        in_dtype=torch.bfloat16,
        max_num_tokens=sum(token_counts),
    )
    # Both ranks route to one expert on each GPU, but have different FP8 scales.
    inputs = torch.cat(
        [torch.full((count, 64), value, dtype=torch.bfloat16)
         for count, value in zip(token_counts, (1 / 64, 8 / 64))]
    )
    ids = (
        torch.tensor([[0, 2]], dtype=torch.int32)
        .expand(inputs.shape[0], -1)
        .contiguous()
    )
    scores = torch.full(ids.shape, 0.5, dtype=torch.float32)
    w1 = torch.full((4, 128, 64), 0.125, dtype=torch.float8_e4m3fn)
    w2 = torch.full((4, 64, 64), 0.125, dtype=torch.float8_e4m3fn)
    start = sum(token_counts[:pgi.rank])
    end = start + token_counts[pgi.rank]
    local_experts = slice(2 * pgi.rank, 2 * (pgi.rank + 1))
    expert_map = torch.full((4,), -1, dtype=torch.int32)
    expert_map[local_experts] = torch.arange(2, dtype=torch.int32)
    quant = fp8_w8a8_moe_quant_config(torch.ones(2), torch.ones(2))
    distributed = FusedMoEKernel(
        MoEPrepareAndFinalizeNaiveDPEPModular(), TritonExperts(moe, quant)
    )
    reference_moe = replace(
        moe, num_local_experts=4,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
    )
    reference_quant = fp8_w8a8_moe_quant_config(torch.ones(4), torch.ones(4))
    reference = FusedMoEKernel(
        MoEPrepareAndFinalizeNoDPEPModular(),
        TritonExperts(reference_moe, reference_quant),
    )
    with set_forward_context(
        None, vllm_config, num_tokens=token_counts[pgi.rank],
        num_tokens_across_dp=torch.tensor(token_counts, dtype=torch.int32),
    ):
        actual = distributed.apply(
            inputs[start:end], w1[local_experts], w2[local_experts],
            ids[start:end], scores[start:end], global_num_experts=4,
            expert_map=expert_map,
        )
        expected = reference.apply(
            inputs, w1, w2, ids, scores, global_num_experts=4,
        )[start:end]
    torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA GPUs")
@pytest.mark.parametrize("token_counts", [(4, 4), (3, 5)], ids=["equal", "unequal"])
def test_tensorwise_fp8_scales_survive_dp_ep_dispatch(token_counts):
    config = VllmConfig()
    config.model_config = SimpleNamespace(enforce_eager=True, is_moe=True)
    config.parallel_config.data_parallel_size = 2
    config.parallel_config.enable_expert_parallel = True
    config.parallel_config.all2all_backend = "allgather_reducescatter"
    parallel_launch_with_config(2, _tensor_scale_worker, config, None, token_counts)
