# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from
# https://github.com/sgl-project/sglang/blob/bed301a5acaa9577c9aa706468bdf242f6a43051/python/sglang/srt/layers/moe/routed_experts_capturer.py

from __future__ import annotations

import logging

import numpy as np
import torch

from vllm.config import VllmConfig
from vllm.distributed.parallel_state import get_tp_group
from vllm.forward_context import get_forward_context
from vllm.platforms import current_platform
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig

logger = logging.getLogger(__name__)


def _get_num_experts_per_tok(hf_config) -> int:
    """Resolve the per-token expert count from the HF config.

    Different model families store this under different attribute names
    (e.g. ``num_experts_per_tok`` for DeepSeek, ``top_k_experts`` for Gemma 4).
    """
    val = getattr(hf_config, "num_experts_per_tok", None)
    if val is None:
        val = getattr(hf_config, "top_k_experts", None)
    if val is None:
        raise ValueError(
            "Cannot determine num_experts_per_tok: HF config has neither "
            "'num_experts_per_tok' nor 'top_k_experts'"
        )
    return val


def get_num_experts(hf_config) -> int:
    """Resolve ``num_experts`` across HuggingFace config naming conventions.

    Different MoE model families expose this under different keys:
      - ``num_experts``: Mixtral, Qwen2-MoE, Qwen3-MoE
      - ``n_routed_experts``: DeepSeek-V2/V3
      - ``num_local_experts``: Mixtral (older exports)
    """
    for key in ("num_experts", "n_routed_experts", "num_local_experts"):
        val = getattr(hf_config, key, None)
        if val is not None:
            return val
    raise ValueError(
        "Could not resolve num_experts from model config. "
        "Expected one of 'num_experts', 'n_routed_experts', "
        "or 'num_local_experts'."
    )


def _local_routed_experts(
    topk_ids: torch.Tensor,
    *,
    dp_rank: int,
    tp_size: int,
) -> tuple[torch.Tensor, int]:
    """Return the routing rows owned by one DP rank.

    The router can see a concatenated DP batch, an already-local batch, or
    one sequence-parallel shard. Keep this normalization in one place so the
    generated-route carrier and the aggregate route audit count the same
    logical token rows.
    """
    ctx = get_forward_context()
    if ctx.dp_metadata is None:
        return topk_ids, topk_ids.shape[0]

    num_tokens_dp = ctx.dp_metadata.num_tokens_across_dp_cpu
    token_num_per_dp = int(num_tokens_dp[dp_rank].item())
    total = int(num_tokens_dp.sum().item())
    n = topk_ids.shape[0]

    if n == total:
        cumsum = torch.cumsum(num_tokens_dp, dim=0)
        end_loc = int(cumsum[dp_rank].item())
        start_loc = end_loc - token_num_per_dp
        return topk_ids[start_loc:end_loc, :], token_num_per_dp

    if n == token_num_per_dp:
        return topk_ids, token_num_per_dp

    sp_expected = (token_num_per_dp + tp_size - 1) // tp_size if tp_size > 0 else -1
    if tp_size > 1 and n == sp_expected:
        gathered = get_tp_group().all_gather(topk_ids, dim=0)
        return gathered[:token_num_per_dp, :], token_num_per_dp

    raise AssertionError(
        "RoutedExpertsCapturer: unexpected topk_ids batch "
        f"dim {n} (expected {total}, {token_num_per_dp}, "
        f"or {sp_expected} for dp_rank={dp_rank}, tp_size={tp_size})"
    )


class RoutedExpertsCapturer:
    """Worker-side capturer for routed experts, lives on GPU.

    Layer-level hooks call :meth:`capture` from inside the forward pass
    with the per-layer ``topk_ids`` tensor. The tensor is sliced to the
    tokens owned by this DP rank and written into a preallocated device
    buffer. At the end of the step, :class:`GPUModelRunner` reads the
    device buffer, issues a D2H copy into a pinned CPU buffer, and hands
    the result to the scheduler via :class:`RoutedExpertsLists`.

    The device / pinned-CPU transit buffers use ``torch.int32`` (not a
    narrow ``uint8``/``uint16`` sized by ``num_experts``). This keeps the
    SP all-gather path free of dtype casts, matches the router's native
    ``topk_ids`` indices dtype more closely, and costs only a few MB per
    worker (``max_num_batched_tokens * num_layers * top_k * 4`` bytes).
    The scheduler-side slot buffer
    (``RoutedExpertsManager.routed_experts_by_slot``) still uses the
    narrow dtype -- numpy fancy-index assignment in ``store_batch``
    narrows the data on the way in.

    Invariants:
        - One instance per worker; shape is fixed at init and covers the
          worst-case step (``max_num_batched_tokens`` tokens).
        - :meth:`clear_buffer` is called at the start of every step, so
          unused slots stay zero.
        - ``device_buffer.dtype`` is ``torch.int32``.
    """

    def __init__(
        self,
        max_num_batched_tokens: int,
        vllm_config: VllmConfig,
    ) -> None:
        hf_config = vllm_config.model_config.hf_text_config
        num_experts_per_tok = _get_num_experts_per_tok(hf_config)
        self.device_buffer = torch.zeros(
            (
                max_num_batched_tokens,
                hf_config.num_hidden_layers,
                num_experts_per_tok,
            ),
            # Use int32 for the device / host transit buffers: it
            # matches the router's native topk_ids dtype, is universally
            # supported by NCCL (uint8/uint16 are version-dependent),
            # and the extra bytes are small (few MB per worker). The
            # big scheduler-side slot buffer stays narrow.
            dtype=torch.int32,
            device=current_platform.device_type,
        )
        self.dp_rank = vllm_config.parallel_config.data_parallel_rank
        self.tp_size = vllm_config.parallel_config.tensor_parallel_size

    def capture(self, layer_id: int, topk_ids: torch.Tensor) -> None:
        """Capture expert routing decisions for a specific layer.

        Under data parallelism, ``topk_ids`` may have three different batch
        layouts depending on where the DP combine happens and whether
        Sequence Parallelism (SP) is active for the MoE layer:
          - ``n == total`` (naive dispatch): all DP ranks' tokens are
            concatenated before routing; we slice out this rank's span
            using the cumulative per-rank counts.
          - ``n == token_num_per_dp`` (modular-kernel path): DP combine
            happens inside ``quant_method.apply``; ``select_experts`` only
            ever sees this rank's tokens, so we take the whole tensor.
          - ``n == ceil(token_num_per_dp / tp_size)`` (SP + modular-kernel
            path): tokens were split along dim=0 across the TP group by
            ``_sequence_parallel_context``
            (``moe_runner_base.py:_sequence_parallel_context``), so each
            TP rank only sees its shard. We all-gather along dim=0 to
            reconstruct this DP rank's full routing tensor. SP pads with
            ceil-div (see ``_compute_sp_num_tokens`` in
            ``forward_context.py``), so the gathered tensor may contain a
            few trailing padding rows which are trimmed by the downstream
            ``[:token_num_per_dp]`` slice.

        Args:
            layer_id: The layer index.
            topk_ids: Tensor of shape (batch_size, num_routed_experts).
        """

        local_topk_ids, token_num_per_dp = _local_routed_experts(
            topk_ids,
            dp_rank=self.dp_rank,
            tp_size=self.tp_size,
        )

        # Defensive: model may expose more layers than the capture buffer
        # was sized for (unusual, but guards against miss-config).
        if layer_id >= self.device_buffer.shape[1]:
            return

        self.device_buffer[:token_num_per_dp, layer_id, :] = local_topk_ids

    def clear_buffer(self) -> None:
        """Zero the device buffer. Called at the start of every step so
        slots belonging to finished / preempted tokens don't leak into
        the next step.
        """
        self.device_buffer.zero_()

    def get_device_buffer(self) -> torch.Tensor:
        """Return the underlying device buffer so the model runner can
        issue the D2H copy. The tensor is shared; callers must either
        clone or fully drain it before the next forward pass runs
        :meth:`clear_buffer`.
        """
        return self.device_buffer


class GrugMoeRouteAuditCapturer:
    """GPU-resident aggregate route counts for controlled GrugMoE runs.

    This records only a fixed ``num_layers x num_experts`` int64 histogram.
    Under expert parallelism, each worker masks the global selections to the
    experts physically owned by that worker. Summing worker histograms then
    counts every selected contribution once and also exposes per-rank load.
    """

    def __init__(self, vllm_config: VllmConfig, mode: str) -> None:
        if mode not in ("noop", "record"):
            raise ValueError(f"Unsupported GrugMoE route audit mode: {mode!r}")

        hf_config = vllm_config.model_config.hf_text_config
        self.mode = mode
        self.num_layers = int(hf_config.num_hidden_layers)
        self.num_experts = int(get_num_experts(hf_config))
        self.dp_rank = vllm_config.parallel_config.data_parallel_rank
        self.tp_size = vllm_config.parallel_config.tensor_parallel_size
        self.counts = torch.zeros(
            (self.num_layers, self.num_experts),
            dtype=torch.int64,
            device=current_platform.device_type,
        )
        self.local_expert_mask = torch.ones_like(self.counts)

    def set_expert_map(
        self,
        layer_id: int,
        expert_map: torch.Tensor | None,
    ) -> None:
        if not 0 <= layer_id < self.num_layers:
            raise AssertionError(
                f"GrugMoE route audit layer {layer_id} is outside "
                f"[0, {self.num_layers})"
            )
        if expert_map is None:
            self.local_expert_mask[layer_id].fill_(1)
            return
        if expert_map.shape[0] < self.num_experts:
            raise AssertionError(
                "GrugMoE route audit expert map is smaller than the global "
                f"expert count: {expert_map.shape[0]} < {self.num_experts}"
            )
        self.local_expert_mask[layer_id].copy_(
            (expert_map[: self.num_experts] >= 0).to(dtype=torch.int64)
        )

    def capture(self, layer_id: int, topk_ids: torch.Tensor) -> None:
        if self.mode == "noop":
            return
        if not 0 <= layer_id < self.num_layers:
            raise AssertionError(
                f"GrugMoE route audit layer {layer_id} is outside "
                f"[0, {self.num_layers})"
            )

        local_topk_ids, _ = _local_routed_experts(
            topk_ids,
            dp_rank=self.dp_rank,
            tp_size=self.tp_size,
        )
        flat_ids = local_topk_ids.reshape(-1).to(dtype=torch.long)
        local_weights = self.local_expert_mask[layer_id].index_select(0, flat_ids)
        self.counts[layer_id].scatter_add_(0, flat_ids, local_weights)

    def reset(self) -> None:
        self.counts.zero_()

    def snapshot(self) -> dict[str, object]:
        counts = self.counts.cpu().tolist()
        local_expert_mask = self.local_expert_mask.cpu().tolist()
        return {
            "mode": self.mode,
            "num_layers": self.num_layers,
            "num_experts": self.num_experts,
            "assignment_count": sum(sum(row) for row in counts),
            "counts": counts,
            "local_expert_mask": local_expert_mask,
        }


class RoutedExpertsManager:
    """Scheduler-side slot-indexed buffer for routed experts.

    Lives on CPU in the scheduler process. Each slot corresponds to
    ``block_id * block_size + offset_in_block`` where ``block_id`` is
    drawn from the physical KV-cache block pool, so routing data is
    tied to physical blocks and naturally survives preemption for
    prefix-cached blocks (prefix hits re-expose the same slots).

    Data flow per step:
      1. Worker D2Hs its device capture buffer into
         :class:`RoutedExpertsLists` and returns it via
         :class:`ModelRunnerOutput`.
      2. Scheduler calls :meth:`store_batch` with that step's
         ``(routing_data, slot_mapping)`` — a single CPU->CPU
         fancy-index assign, ~few MB per step.
      3. On request completion / abort / preemption, the scheduler
         calls :meth:`get` with the request's block IDs to recover
         the full per-token routing.

    Memory: ``routed_experts_by_slot`` is sized for the whole block
    pool (``num_blocks * block_size`` slots). For large block pools
    this can reach multiple GB; see the init log for the exact size.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
    ) -> None:
        # Pick the attention group for block/slot mapping. We require
        # a FullAttentionSpec group rather than any AttentionSpec to
        # stay consistent with the worker-side lookup in
        # ``GPUModelRunner._get_attention_kv_cache_gid``; hybrid models
        # (Mamba / linear attention) also have other AttentionSpec
        # groups whose slot layout differs.
        self.attn_gid = next(
            gid
            for gid, g in enumerate(kv_cache_config.kv_cache_groups)
            if isinstance(g.kv_cache_spec, FullAttentionSpec)
        )
        attn_group = kv_cache_config.kv_cache_groups[self.attn_gid]
        self.block_size = attn_group.kv_cache_spec.block_size

        # All kv_cache_groups share the same physical block pool, so
        # block IDs span [0, num_blocks) regardless of how many groups
        # exist. Sizing to the full pool avoids index-out-of-range
        # when different groups happen to land on the same block.
        hf_config = vllm_config.model_config.hf_text_config
        num_experts = get_num_experts(hf_config)
        num_experts_per_tok = _get_num_experts_per_tok(hf_config)
        max_num_slots = kv_cache_config.num_blocks * self.block_size
        # Expert IDs are 0..num_experts-1; uint8 fits 256 distinct
        # values so the boundary is ``<= 256`` (NOT ``< 256``). Keeping
        # this narrow matters because the slot buffer is sized for the
        # whole block pool and can reach multiple GB.
        expert_id_dtype = np.uint8 if num_experts <= 256 else np.uint16
        self.routed_experts_by_slot = np.zeros(
            (
                max_num_slots,
                hf_config.num_hidden_layers,
                num_experts_per_tok,
            ),
            dtype=expert_id_dtype,
        )
        logger.info(
            "RoutedExpertsManager CPU buffer: %.2f GB "
            "(slots=%d, layers=%d, top_k=%d, dtype=%s)",
            self.routed_experts_by_slot.nbytes / 1e9,
            max_num_slots,
            hf_config.num_hidden_layers,
            hf_config.num_experts_per_tok,
            self.routed_experts_by_slot.dtype.name,
        )

    def store_batch(self, data: np.ndarray, slot_mapping: np.ndarray) -> None:
        """Persist one step's routed experts into the slot buffer.

        Equivalent to ``slot_buffer[slot_mapping] = data``; numpy fancy
        indexing handles repeated / out-of-order indices. Called once
        per scheduler step in ``update_from_output``.
        """
        self.routed_experts_by_slot[slot_mapping] = data

    def get(
        self,
        block_ids: list[int],
        num_tokens: int,
        token_start: int = 0,
    ) -> np.ndarray:
        """Read routed experts data for a completed / preempted request.

        Reconstructs a per-token slot_mapping from the request's block
        IDs and returns the routing slice. Because numpy fancy indexing
        returns a **copy** (not a view), the returned ndarray is safe
        to hold across subsequent :meth:`store_batch` calls — do not
        replace the fancy index with a slice without re-verifying.

        Args:
            block_ids: Block IDs from the attention KV-cache group.
            num_tokens: Number of tokens that have gone through a forward
                pass and therefore have routing data written to their
                slots (typically ``request.num_tokens - 1``; the last
                sampled token has not been forwarded yet). Slots beyond
                ``request.num_computed_tokens`` are zero-initialized.
            token_start: Skip the first ``token_start`` tokens from the
                result. The slot_mapping is sliced before the fancy-index
                read, so only the requested slots are fetched — no large
                intermediate array is allocated. Clamped to
                ``[0, num_tokens]`` automatically.

        Returns:
            Array of shape (num_tokens - token_start, num_layers,
            num_experts_per_tok).
        """
        bs = self.block_size
        block_ids_array = np.array(block_ids, dtype=np.int32)
        block_offsets = np.arange(bs)
        # slot = block_id * block_size + offset_in_block; flatten the
        # (num_blocks, block_size) grid and trim to num_tokens, then
        # skip the first token_start entries so only the requested
        # range is fetched in a single fancy-index read.
        slot_mapping = (
            block_ids_array.reshape(-1, 1) * bs + block_offsets.reshape(1, -1)
        ).flatten()[:num_tokens]
        slot_mapping = slot_mapping[token_start:]
        return self.routed_experts_by_slot[slot_mapping]
