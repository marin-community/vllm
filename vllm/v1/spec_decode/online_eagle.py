# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Bounded training-state capture for online EAGLE-3 updates."""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch
from safetensors.torch import save_file
from torch import nn

_FORMAT_VERSION = 1
_MANIFEST_FILENAME = "manifest.json"
_MIN_TRAINING_WINDOW_TOKENS = 2
LM_HEAD_WEIGHT_NAME = "lm_head.weight"
TARGET_EMBEDDING_NAME = "model.embed_tokens.weight"


def _record_stream_for_async_copy(
    tensors: Sequence[torch.Tensor], stream: torch.cuda.Stream
) -> None:
    """Keep temporary CUDA storage alive until its side-stream copy finishes."""
    for tensor in tensors:
        tensor.record_stream(stream)


def _unique_parameter(model: nn.Module, name: str) -> nn.Parameter | None:
    matches = [
        parameter
        for parameter_name, parameter in model.named_parameters()
        if parameter_name == name or parameter_name.endswith(f".{name}")
    ]
    if len(matches) > 1:
        raise RuntimeError(
            f"Expected at most one {name} parameter, found {len(matches)}"
        )
    return matches[0] if matches else None


def target_embedding_weight(model: nn.Module) -> nn.Parameter:
    """Return the target embedding parameter through optional model wrappers."""
    embedding = _unique_parameter(model, TARGET_EMBEDDING_NAME)
    if embedding is None:
        raise RuntimeError(f"Target model has no {TARGET_EMBEDDING_NAME}")
    return embedding


def target_head_weight(model: nn.Module) -> nn.Parameter:
    """Return the target output head, falling back to tied embeddings."""
    head = _unique_parameter(model, LM_HEAD_WEIGHT_NAME)
    return target_embedding_weight(model) if head is None else head


def draft_vocab_target_ids(
    draft_model: nn.Module, target_vocab_size: int
) -> torch.Tensor:
    """Resolve vLLM's compact draft-vocabulary rows into target row IDs."""
    offsets = getattr(draft_model, "draft_id_to_target_id", None)
    if offsets is None or offsets.ndim != 1:
        raise ValueError("EAGLE draft has no target vocabulary row mapping")
    target_ids = torch.arange(offsets.numel(), device=offsets.device) + offsets
    if (
        target_ids.dtype not in {torch.int32, torch.int64}
        or target_ids.numel() == 0
        or int(target_ids.min()) < 0
        or int(target_ids.max()) >= target_vocab_size
        or target_ids.unique().numel() != target_ids.numel()
    ):
        raise ValueError("EAGLE draft vocabulary row mapping is invalid")
    return target_ids.to(dtype=torch.long)


def project_target_head(
    draft_model: nn.Module, target_head: torch.Tensor
) -> torch.Tensor:
    """Select target-head rows in the resident draft vocabulary order."""
    return target_head[draft_vocab_target_ids(draft_model, target_head.shape[0])]


def refresh_target_owned_draft_weights(
    target_model: nn.Module, draft_model: nn.Module | None
) -> None:
    """Refresh an embedding-free draft head after target synchronization."""
    if draft_model is None or not isinstance(
        getattr(draft_model, "draft_id_to_target_id", None), torch.Tensor
    ):
        return
    target_head = target_head_weight(target_model)
    draft_head = dict(draft_model.named_parameters()).get(LM_HEAD_WEIGHT_NAME)
    if draft_head is None:
        raise RuntimeError(f"Embedding-free EAGLE draft has no {LM_HEAD_WEIGHT_NAME}")
    projected = project_target_head(draft_model, target_head)
    if projected.shape != draft_head.shape:
        raise RuntimeError("Projected target head does not match the EAGLE draft head")
    with torch.no_grad():
        draft_head.copy_(projected)


@dataclass(frozen=True)
class OnlineEagleCaptureConfig:
    step: int
    max_tokens: int
    max_window_tokens: int
    worker_rank: int
    target_revision: str
    draft_revision: str
    aux_layer_ids: tuple[int, ...]
    capture_target_snapshot: bool = True

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> OnlineEagleCaptureConfig:
        allowed = set(cls.__dataclass_fields__)
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(
                "Unknown online EAGLE capture fields: " + ", ".join(sorted(unknown))
            )

        def positive_int(name: str) -> int:
            result = value.get(name)
            if isinstance(result, bool) or not isinstance(result, int) or result <= 0:
                raise ValueError(f"{name} must be a positive integer, got {result!r}")
            return result

        step = value.get("step")
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError(f"step must be a nonnegative integer, got {step!r}")
        worker_rank = value.get("worker_rank", 0)
        if not isinstance(worker_rank, int):
            raise ValueError("worker_rank must be an integer")
        aux_layer_ids = value.get("aux_layer_ids")
        if not isinstance(aux_layer_ids, Sequence) or isinstance(
            aux_layer_ids, (str, bytes)
        ):
            raise ValueError("aux_layer_ids must be a nonempty integer sequence")
        aux_layers = tuple(aux_layer_ids)
        if not aux_layers or any(
            isinstance(layer, bool) or not isinstance(layer, int) or layer < 0
            for layer in aux_layers
        ):
            raise ValueError("aux_layer_ids must be a nonempty integer sequence")
        target_revision = value.get("target_revision")
        draft_revision = value.get("draft_revision")
        if not isinstance(target_revision, str) or not target_revision:
            raise ValueError("target_revision must be a nonempty string")
        if not isinstance(draft_revision, str) or not draft_revision:
            raise ValueError("draft_revision must be a nonempty string")
        capture_target_snapshot = value.get("capture_target_snapshot", True)
        if not isinstance(capture_target_snapshot, bool):
            raise ValueError("capture_target_snapshot must be a boolean")
        return cls(
            step=step,
            max_tokens=positive_int("max_tokens"),
            max_window_tokens=positive_int("max_window_tokens"),
            worker_rank=worker_rank,
            target_revision=target_revision,
            draft_revision=draft_revision,
            aux_layer_ids=aux_layers,
            capture_target_snapshot=capture_target_snapshot,
        )


def resolve_online_eagle_capture_config(
    config: Mapping[str, Any],
    *,
    speculative_method: str | None,
    async_scheduling: bool,
    pipeline_parallel_size: int,
    worker_rank: int,
    max_window_tokens: int,
    aux_layer_ids: Callable[[], Sequence[int]],
) -> OnlineEagleCaptureConfig:
    """Add worker-owned fields and validate an online capture request."""
    if speculative_method != "eagle3":
        raise RuntimeError("Online EAGLE capture requires an EAGLE-3 drafter")
    if async_scheduling:
        raise RuntimeError("Online EAGLE capture requires synchronous scheduling")
    if pipeline_parallel_size != 1:
        raise RuntimeError("Online EAGLE capture requires pipeline_parallel_size=1")
    resolved = dict(config)
    reserved_fields = {"worker_rank", "aux_layer_ids"}.intersection(resolved)
    if reserved_fields:
        raise ValueError(
            "Online EAGLE capture fields are worker-owned: "
            + ", ".join(sorted(reserved_fields))
        )
    resolved["worker_rank"] = worker_rank
    resolved.setdefault("max_window_tokens", max_window_tokens)
    resolved["aux_layer_ids"] = list(aux_layer_ids())
    return OnlineEagleCaptureConfig.from_mapping(resolved)


@dataclass
class _PendingCopy:
    rows: list[tuple[str, int]]
    token_ids: torch.Tensor
    aux_hidden_states: torch.Tensor
    head_input_hidden_states: torch.Tensor
    event: torch.cuda.Event | None


@dataclass
class _PendingSamples:
    request_ids: tuple[str, ...]
    token_ids: torch.Tensor
    num_sampled_tokens: torch.Tensor
    event: torch.cuda.Event | None


@dataclass
class _RequestCapture:
    prompt_token_ids: tuple[int, ...]
    retention_floor: int = 0
    output_token_ids: tuple[int, ...] | None = None
    final_output_length: int | None = None
    provisional: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = field(
        default_factory=dict
    )


class OnlineEagleCapture:
    """Collect complete, token-keyed EAGLE windows in bounded host memory."""

    def __init__(self, config: OnlineEagleCaptureConfig):
        self.config = config
        self.requests: dict[str, _RequestCapture] = {}
        self._reserved_tokens = 0
        self._pending: list[_PendingCopy | _PendingSamples] = []
        self._copy_stream: torch.cuda.Stream | None = None
        self.dropped_requests = 0
        self.dropped_windows = 0
        self.captured_rows = 0

    def _copy_tensors_to_host(
        self, tensors: Sequence[torch.Tensor]
    ) -> tuple[tuple[torch.Tensor, ...], torch.cuda.Event | None]:
        if not tensors[0].is_cuda:
            return tuple(tensor.detach().cpu().clone() for tensor in tensors), None

        if self._copy_stream is None:
            self._copy_stream = torch.cuda.Stream(device=tensors[0].device)
        default_stream = torch.cuda.current_stream(tensors[0].device)
        with torch.cuda.stream(self._copy_stream):
            self._copy_stream.wait_stream(default_stream)
            host_tensors = tuple(
                torch.empty_like(tensor, device="cpu", pin_memory=True)
                for tensor in tensors
            )
            for host, source in zip(host_tensors, tensors, strict=True):
                host.copy_(source, non_blocking=True)
            _record_stream_for_async_copy(tensors, self._copy_stream)
            event = torch.cuda.Event()
            event.record(self._copy_stream)
        return host_tensors, event

    def admit_request(
        self,
        request_id: str,
        prompt_token_ids: Sequence[int] | None,
        max_completion_tokens: int,
    ) -> bool:
        """Admit one request before prefill and reserve a complete bounded window."""
        if (
            prompt_token_ids is None
            or isinstance(max_completion_tokens, bool)
            or max_completion_tokens <= 0
        ):
            self.dropped_requests += 1
            return False
        reserved = min(
            len(prompt_token_ids) + max_completion_tokens,
            self.config.max_window_tokens,
        )
        if (
            reserved < _MIN_TRAINING_WINDOW_TOKENS
            or self._reserved_tokens + reserved > self.config.max_tokens
        ):
            self.dropped_requests += 1
            return False
        self._reserved_tokens += reserved
        self.requests[request_id] = _RequestCapture(
            prompt_token_ids=tuple(int(token) for token in prompt_token_ids),
        )
        return True

    def finalize_request(
        self, request_id: str, output_token_ids: Sequence[int] | None
    ) -> None:
        request = self.requests.get(request_id)
        if request is None or output_token_ids is None:
            return
        request.output_token_ids = tuple(int(token) for token in output_token_ids)

    def finalize_request_length(self, request_id: str, output_length: int) -> None:
        """Record the scheduler's exact final output length for a V2 request."""
        request = self.requests.get(request_id)
        if request is None:
            return
        if output_length < 0:
            raise ValueError("output_length must be nonnegative")
        request.final_output_length = output_length
        if request.output_token_ids is not None:
            request.output_token_ids = request.output_token_ids[:output_length]

    def record_sampled(
        self,
        *,
        request_ids: Sequence[str],
        sampled_token_ids: torch.Tensor,
        num_sampled_tokens: torch.Tensor,
    ) -> None:
        """Append V2 sampler outputs without synchronizing the model stream."""
        self._drain_pending(wait=False)
        if sampled_token_ids.ndim != 2 or num_sampled_tokens.ndim != 1:
            raise ValueError("sampled token tensors have invalid ranks")
        if len(request_ids) != sampled_token_ids.shape[0] or len(request_ids) != len(
            num_sampled_tokens
        ):
            raise ValueError("sampled token metadata lengths do not match")

        host_tensors, event = self._copy_tensors_to_host(
            (sampled_token_ids, num_sampled_tokens)
        )
        self._pending.append(
            _PendingSamples(
                request_ids=tuple(request_ids),
                token_ids=host_tensors[0],
                num_sampled_tokens=host_tensors[1],
                event=event,
            )
        )
        self._drain_pending(wait=False)

    def _selected_forward_rows(
        self,
        request_ids: Sequence[str],
        num_scheduled_tokens: Sequence[int],
        num_computed_tokens: Sequence[int],
    ) -> tuple[list[int], list[tuple[str, int]]]:
        selected_rows: list[int] = []
        rows: list[tuple[str, int]] = []
        offset = 0
        for request_id, scheduled, computed in zip(
            request_ids,
            num_scheduled_tokens,
            num_computed_tokens,
            strict=True,
        ):
            request = self.requests.get(request_id)
            if request is not None:
                prompt_length = len(request.prompt_token_ids)
                confirmed_tokens = (
                    min(prompt_length, int(computed) + int(scheduled))
                    if int(computed) < prompt_length
                    else int(computed)
                )
                request.retention_floor = max(
                    request.retention_floor,
                    confirmed_tokens - self.config.max_window_tokens,
                )
                for query_offset in range(int(scheduled)):
                    position = int(computed) + query_offset
                    if position < request.retention_floor:
                        continue
                    selected_rows.append(offset + query_offset)
                    rows.append((request_id, position))
            offset += int(scheduled)
        return selected_rows, rows

    def _copy_selected_rows(
        self,
        *,
        rows: list[tuple[str, int]],
        selected_rows: list[int],
        input_ids: torch.Tensor,
        aux_hidden_states: Sequence[torch.Tensor],
        head_input_hidden_states: torch.Tensor,
    ) -> _PendingCopy:
        indices = torch.tensor(selected_rows, device=input_ids.device)
        selected_tokens = input_ids.index_select(0, indices)
        selected_aux = torch.cat(list(aux_hidden_states), dim=-1).index_select(
            0, indices
        )
        selected_head_inputs = head_input_hidden_states.index_select(0, indices)
        host_tensors, event = self._copy_tensors_to_host(
            (selected_tokens, selected_aux, selected_head_inputs)
        )
        return _PendingCopy(
            rows=rows,
            token_ids=host_tensors[0],
            aux_hidden_states=host_tensors[1],
            head_input_hidden_states=host_tensors[2],
            event=event,
        )

    def record_forward(
        self,
        *,
        request_ids: Sequence[str],
        num_scheduled_tokens: Sequence[int],
        num_computed_tokens: Sequence[int],
        input_ids: torch.Tensor,
        aux_hidden_states: Sequence[torch.Tensor],
        head_input_hidden_states: torch.Tensor,
    ) -> None:
        """Copy selected target rows to pinned host buffers on a side stream."""
        self._drain_pending(wait=False)
        if len(request_ids) != len(num_scheduled_tokens) or len(request_ids) != len(
            num_computed_tokens
        ):
            raise ValueError("request metadata lengths do not match")
        if len(aux_hidden_states) != len(self.config.aux_layer_ids):
            raise ValueError(
                "EAGLE auxiliary-state count does not match aux_layer_ids: "
                f"{len(aux_hidden_states)} != {len(self.config.aux_layer_ids)}"
            )

        selected_rows, rows = self._selected_forward_rows(
            request_ids,
            num_scheduled_tokens,
            num_computed_tokens,
        )
        if not selected_rows:
            return

        pending = self._copy_selected_rows(
            rows=rows,
            selected_rows=selected_rows,
            input_ids=input_ids,
            aux_hidden_states=aux_hidden_states,
            head_input_hidden_states=head_input_hidden_states,
        )
        self._pending.append(pending)
        self.captured_rows += len(rows)
        self._drain_pending(wait=False)

    def _store_pending_samples(self, pending: _PendingSamples) -> None:
        for index, request_id in enumerate(pending.request_ids):
            request = self.requests.get(request_id)
            if request is None:
                continue
            count = int(pending.num_sampled_tokens[index].item())
            if count < 0 or count > pending.token_ids.shape[1]:
                raise ValueError("sampled token count is out of bounds")
            sampled = tuple(
                int(token) for token in pending.token_ids[index, :count].tolist()
            )
            request.output_token_ids = (request.output_token_ids or ()) + sampled
            if request.final_output_length is not None:
                request.output_token_ids = request.output_token_ids[
                    : request.final_output_length
                ]

    def _store_pending_rows(self, pending: _PendingCopy) -> None:
        for index, (request_id, position) in enumerate(pending.rows):
            request = self.requests.get(request_id)
            if request is None:
                continue
            token_id = int(pending.token_ids[index].item())
            stale = [
                key
                for key in request.provisional
                if key[0] == position or key[0] < request.retention_floor
            ]
            for key in stale:
                del request.provisional[key]
            request.provisional[(position, token_id)] = (
                pending.aux_hidden_states[index].clone(),
                pending.head_input_hidden_states[index].clone(),
            )

    def _drain_pending(self, *, wait: bool) -> None:
        remaining: list[_PendingCopy | _PendingSamples] = []
        for pending in self._pending:
            if pending.event is not None:
                if not wait and not pending.event.query():
                    remaining.append(pending)
                    continue
                pending.event.synchronize()
            if isinstance(pending, _PendingSamples):
                self._store_pending_samples(pending)
            else:
                self._store_pending_rows(pending)
        self._pending = remaining

    @staticmethod
    def _target_snapshot(
        model: nn.Module, draft_model: nn.Module
    ) -> tuple[dict[str, torch.Tensor], dict]:
        embedding = target_embedding_weight(model)
        head = target_head_weight(model)
        tensors = {
            TARGET_EMBEDDING_NAME: embedding.detach().cpu().contiguous(),
            LM_HEAD_WEIGHT_NAME: project_target_head(draft_model, head)
            .detach()
            .cpu()
            .contiguous(),
        }
        return tensors, {
            name: {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
            for name, tensor in tensors.items()
        }

    def _window_for_request(
        self, request: _RequestCapture
    ) -> dict[str, torch.Tensor] | None:
        if request.output_token_ids is None:
            self.dropped_windows += 1
            return None
        prompt_length = len(request.prompt_token_ids)
        trajectory = request.prompt_token_ids + request.output_token_ids
        matching = {
            position: request.provisional[(position, token)]
            for position, token in enumerate(trajectory)
            if (position, token) in request.provisional
        }
        segments: list[list[int]] = []
        for position in sorted(matching):
            if not segments or position != segments[-1][-1] + 1:
                segments.append([position])
            else:
                segments[-1].append(position)
        eligible = [
            segment
            for segment in segments
            if len(segment) >= _MIN_TRAINING_WINDOW_TOKENS
            and segment[-1] >= prompt_length
        ]
        if not eligible:
            self.dropped_windows += 1
            return None
        positions = max(eligible, key=lambda segment: (segment[-1], len(segment)))[
            -self.config.max_window_tokens :
        ]
        aux = torch.stack([matching[position][0] for position in positions])
        head_inputs = torch.stack([matching[position][1] for position in positions])
        token_ids = torch.tensor(
            [trajectory[position] for position in positions], dtype=torch.long
        )
        loss_mask = torch.tensor(
            [position >= prompt_length for position in positions], dtype=torch.bool
        )
        if not loss_mask[1:].any():
            self.dropped_windows += 1
            return None
        return {
            "input_ids": token_ids,
            "hidden_states": aux,
            "head_input_hidden_states": head_inputs,
            "loss_mask": loss_mask,
            "position_ids": torch.tensor(positions, dtype=torch.long),
        }

    def _write_windows(self, staging: Path) -> list[dict[str, Any]]:
        windows: list[dict[str, Any]] = []
        for request_id, request in sorted(self.requests.items()):
            tensors = self._window_for_request(request)
            if tensors is None:
                continue
            window_path = staging / f"window-{len(windows):06d}.safetensors"
            save_file(tensors, str(window_path), metadata={"format": "pt"})
            windows.append(
                {
                    "path": window_path.name,
                    "request_id": request_id,
                    "tokens": int(tensors["input_ids"].shape[0]),
                    "supervised_tokens": int(tensors["loss_mask"].sum().item()),
                }
            )
        return windows

    def _write_target_snapshot(
        self,
        staging: Path,
        *,
        target_model: nn.Module,
        draft_model: nn.Module,
        target_config: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        if not self.config.capture_target_snapshot:
            return None
        target_tensors, target_inventory = self._target_snapshot(
            target_model, draft_model
        )
        target_path = staging / "target.safetensors"
        save_file(target_tensors, str(target_path), metadata={"format": "pt"})
        config_path = staging / "target-config.json"
        config_path.write_text(
            json.dumps(dict(target_config), sort_keys=True, separators=(",", ":"))
        )
        return {
            "weights_path": target_path.name,
            "config_path": config_path.name,
            "inventory": target_inventory,
            "lm_head_vocabulary": "draft",
        }

    def _manifest(
        self,
        windows: list[dict[str, Any]],
        target: dict[str, Any] | None,
    ) -> dict[str, Any]:
        return {
            "format": "vllm-online-eagle-capture",
            "format_version": _FORMAT_VERSION,
            "step": self.config.step,
            "worker_rank": self.config.worker_rank,
            "target_revision": self.config.target_revision,
            "draft_revision": self.config.draft_revision,
            "aux_layer_ids": list(self.config.aux_layer_ids),
            "head_input_semantics": "post_final_norm_target_lm_head_input",
            "windows": windows,
            "captured_rows": self.captured_rows,
            "dropped_requests": self.dropped_requests,
            "dropped_windows": self.dropped_windows,
            "target": target,
        }

    def seal(
        self,
        output_dir: str | os.PathLike[str],
        *,
        target_model: nn.Module,
        draft_model: nn.Module,
        target_config: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Write an immutable atomic capture directory and return its manifest."""
        self._drain_pending(wait=True)
        destination = Path(output_dir)
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = destination.with_name(f".{destination.name}.tmp-{uuid4().hex}")
        staging.mkdir(parents=True, exist_ok=False)
        try:
            windows = self._write_windows(staging)
            target = self._write_target_snapshot(
                staging,
                target_model=target_model,
                draft_model=draft_model,
                target_config=target_config,
            )
            manifest = self._manifest(windows, target)
            manifest_path = staging / _MANIFEST_FILENAME
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
            if destination.exists():
                raise FileExistsError(
                    f"Capture destination already exists: {destination}"
                )
            os.replace(staging, destination)
            return {**manifest, "path": str(destination / _MANIFEST_FILENAME)}
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise


__all__ = [
    "LM_HEAD_WEIGHT_NAME",
    "OnlineEagleCapture",
    "OnlineEagleCaptureConfig",
    "TARGET_EMBEDDING_NAME",
    "draft_vocab_target_ids",
    "project_target_head",
    "refresh_target_owned_draft_weights",
    "resolve_online_eagle_capture_config",
    "target_embedding_weight",
    "target_head_weight",
]
