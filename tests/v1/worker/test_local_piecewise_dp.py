# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Exercise per-rank graph padding over a real four-process CPU collective."""

from types import SimpleNamespace

import torch.distributed as dist
import torch.multiprocessing as mp

from vllm.config import CUDAGraphMode
from vllm.v1.worker.gpu import dp_utils
from vllm.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor,
    CudaGraphManager,
)


def _check_rank(rank: int, store: str) -> None:
    dist.init_process_group("gloo", init_method=f"file://{store}", rank=rank, world_size=4)
    try:
        # Replace the process-group lookup with the actual fixture Gloo group.
        dp_utils.get_dp_group = lambda: SimpleNamespace(cpu_group=dist.group.WORLD)
        manager = CudaGraphManager.__new__(CudaGraphManager)
        manager._graphs_captured = True
        manager._lora_dispatch_map = {}
        manager._max_lora_case = 0
        captures = [1, 2, 4, 8, 16, 32]
        manager._candidates = {
            (count, 0): [
                BatchExecutionDescriptor(
                    CUDAGraphMode.PIECEWISE,
                    next(size for size in captures if size >= count),
                    None,
                )
            ]
            for count in range(1, 33)
        }
        cases = [
            ([2, 3, 5, 9], [2, 4, 8, 16]),
            ([17, 2, 9, 4], [32, 2, 16, 4]),
            ([1, 65, 3, 5], [1, 65, 3, 5]),
        ]
        for local in (False, True):
            manager.compilation_config = SimpleNamespace(cudagraph_local_dp=local)
            for offered, padded in cases:
                descriptor, sync = dp_utils.dispatch_cg_and_sync_dp(
                    manager,
                    1,
                    offered[rank],
                    None,
                    4,
                    rank,
                    max_query_len=offered[rank],
                )
                assert sync is not None
                fallback = max(offered) > max(captures)
                expected = padded if local or fallback else [max(padded)] * 4
                assert sync.num_tokens_across_dp.tolist() == expected
                assert descriptor.num_tokens == expected[rank]
                assert descriptor.cg_mode == (
                    CUDAGraphMode.NONE if fallback else CUDAGraphMode.PIECEWISE
                )
                assert sync.eager is fallback
            descriptor, sync = dp_utils.dispatch_cg_and_sync_dp(
                manager, 0, 0, None, 4, rank
            )
            assert sync is None
            assert descriptor.cg_mode == CUDAGraphMode.NONE
            assert descriptor.num_tokens == 0
    finally:
        dist.destroy_process_group()


def test_local_piecewise_dp_preserves_global_and_eager_fallback(tmp_path) -> None:
    mp.spawn(_check_rank, args=(str(tmp_path / "gloo-store"),), nprocs=4, join=True)
