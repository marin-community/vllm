# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Bounded training-state capture for online EAGLE-3 updates."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch
from safetensors.torch import load_file, save_file
from torch import nn

_FORMAT_VERSION = 1
_CANDIDATE_FORMAT = "marinskyrl-online-eagle-candidate"
_SKYRL_REQUEST_PREFIX = "skyrl-group-"
_MIN_TRAINING_WINDOW_TOKENS = 2


def _record_stream_for_async_copy(
    tensors: Sequence[torch.Tensor], stream: torch.cuda.Stream
) -> None:
    """Keep temporary CUDA storage alive until its side-stream copy finishes."""
    for tensor in tensors:
        tensor.record_stream(stream)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_candidate(
    candidate_dir: str | os.PathLike[str],
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """Load and validate a complete candidate before its collective broadcast."""
    directory = Path(candidate_dir)
    manifest_path = directory / "manifest.json"
    metadata = json.loads(manifest_path.read_text())
    if metadata.get("format") != _CANDIDATE_FORMAT:
        raise ValueError(f"Invalid online EAGLE candidate manifest: {manifest_path}")
    if not metadata.get("complete", False):
        raise ValueError(f"Incomplete online EAGLE candidate: {manifest_path}")
    weights_path = directory / metadata["weights_path"]
    digest = file_sha256(weights_path)
    if digest != metadata["weights_sha256"]:
        raise ValueError(
            "Online EAGLE candidate digest mismatch: "
            f"expected {metadata['weights_sha256']}, got {digest}"
        )
    candidate = load_file(weights_path)
    inventory = metadata["tensor_inventory"]
    if set(candidate) != set(inventory):
        raise ValueError(
            "Online EAGLE candidate tensor inventory does not match its manifest"
        )
    mismatched = [
        name
        for name, value in candidate.items()
        if inventory[name] != {"shape": list(value.shape), "dtype": str(value.dtype)}
    ]
    if mismatched:
        raise ValueError(
            "Online EAGLE candidate tensor metadata does not match its weights: "
            + ", ".join(sorted(mismatched))
        )
    forbidden = [
        name
        for name in candidate
        if "embed_tokens" in name or name.startswith("verifier_")
    ]
    if forbidden:
        raise ValueError(
            "Online EAGLE candidate contains target-owned tensors: "
            + ", ".join(sorted(forbidden))
        )
    return metadata, candidate


def request_group_from_id(request_id: str) -> str:
    """Return a SkyRL group digest, or the full ID for an ungrouped request."""
    if request_id.startswith(_SKYRL_REQUEST_PREFIX):
        remainder = request_id[len(_SKYRL_REQUEST_PREFIX) :]
        group, separator, _attempt = remainder.partition("-")
        if separator and group:
            return group
    return request_id


@dataclass(frozen=True)
class OnlineEagleCaptureConfig:
    step: int
    max_tokens: int
    max_window_tokens: int
    max_sequences_per_prompt_group: int
    trainer_rank: int
    worker_rank: int
    target_revision: str
    draft_revision: str
    aux_layer_ids: tuple[int, ...]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> OnlineEagleCaptureConfig:
        allowed = {
            "step",
            "max_tokens",
            "max_window_tokens",
            "max_sequences_per_prompt_group",
            "trainer_rank",
            "worker_rank",
            "target_revision",
            "draft_revision",
            "aux_layer_ids",
        }
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
        trainer_rank = value.get("trainer_rank", 0)
        worker_rank = value.get("worker_rank", 0)
        if not isinstance(trainer_rank, int) or not isinstance(worker_rank, int):
            raise ValueError("trainer_rank and worker_rank must be integers")
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
        return cls(
            step=step,
            max_tokens=positive_int("max_tokens"),
            max_window_tokens=positive_int("max_window_tokens"),
            max_sequences_per_prompt_group=positive_int(
                "max_sequences_per_prompt_group"
            ),
            trainer_rank=trainer_rank,
            worker_rank=worker_rank,
            target_revision=target_revision,
            draft_revision=draft_revision,
            aux_layer_ids=aux_layers,
        )


@dataclass
class _PendingCopy:
    rows: list[tuple[str, int]]
    token_ids: torch.Tensor
    aux_hidden_states: torch.Tensor
    head_input_hidden_states: torch.Tensor
    event: torch.cuda.Event | None


@dataclass
class _RequestCapture:
    group_id: str
    prompt_token_ids: tuple[int, ...]
    retention_floor: int = 0
    output_token_ids: tuple[int, ...] | None = None
    provisional: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = field(
        default_factory=dict
    )


class OnlineEagleCapture:
    """Collect complete, token-keyed EAGLE windows in bounded host memory."""

    def __init__(self, config: OnlineEagleCaptureConfig):
        self.config = config
        self.active = config.worker_rank == config.trainer_rank
        self.requests: dict[str, _RequestCapture] = {}
        self._group_counts: dict[str, int] = {}
        self._reserved_tokens = 0
        self._pending: list[_PendingCopy] = []
        self._copy_stream: torch.cuda.Stream | None = None
        self.dropped_requests = 0
        self.dropped_windows = 0
        self.captured_rows = 0

    def admit_request(
        self,
        request_id: str,
        prompt_token_ids: Sequence[int] | None,
        max_completion_tokens: int,
    ) -> bool:
        """Admit one request before prefill and reserve a complete bounded window."""
        if not self.active:
            return False
        if (
            prompt_token_ids is None
            or isinstance(max_completion_tokens, bool)
            or max_completion_tokens <= 0
        ):
            self.dropped_requests += 1
            return False
        group_id = request_group_from_id(request_id)
        if (
            self._group_counts.get(group_id, 0)
            >= self.config.max_sequences_per_prompt_group
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
        self._group_counts[group_id] = self._group_counts.get(group_id, 0) + 1
        self.requests[request_id] = _RequestCapture(
            group_id=group_id,
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
        selected_rows: list[int],
        input_ids: torch.Tensor,
        aux_hidden_states: Sequence[torch.Tensor],
        head_input_hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.cuda.Event | None]:
        indices = torch.tensor(selected_rows, device=input_ids.device)
        selected_tokens = input_ids.index_select(0, indices)
        selected_aux = torch.cat(list(aux_hidden_states), dim=-1).index_select(
            0, indices
        )
        selected_head_inputs = head_input_hidden_states.index_select(0, indices)
        if not selected_tokens.is_cuda:
            return (
                selected_tokens.detach().cpu().clone(),
                selected_aux.detach().cpu().clone(),
                selected_head_inputs.detach().cpu().clone(),
                None,
            )

        if self._copy_stream is None:
            self._copy_stream = torch.cuda.Stream(device=selected_tokens.device)
        default_stream = torch.cuda.current_stream(selected_tokens.device)
        with torch.cuda.stream(self._copy_stream):
            self._copy_stream.wait_stream(default_stream)
            host_tokens = torch.empty_like(
                selected_tokens, device="cpu", pin_memory=True
            )
            host_aux = torch.empty_like(selected_aux, device="cpu", pin_memory=True)
            host_head_inputs = torch.empty_like(
                selected_head_inputs, device="cpu", pin_memory=True
            )
            host_tokens.copy_(selected_tokens, non_blocking=True)
            host_aux.copy_(selected_aux, non_blocking=True)
            host_head_inputs.copy_(selected_head_inputs, non_blocking=True)
            _record_stream_for_async_copy(
                (selected_tokens, selected_aux, selected_head_inputs),
                self._copy_stream,
            )
            event = torch.cuda.Event()
            event.record(self._copy_stream)
        return host_tokens, host_aux, host_head_inputs, event

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
        if not self.active:
            return
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

        host_tokens, host_aux, host_head_inputs, event = self._copy_selected_rows(
            selected_rows=selected_rows,
            input_ids=input_ids,
            aux_hidden_states=aux_hidden_states,
            head_input_hidden_states=head_input_hidden_states,
        )
        self._pending.append(
            _PendingCopy(rows, host_tokens, host_aux, host_head_inputs, event)
        )
        self.captured_rows += len(rows)
        self._drain_pending(wait=False)

    def _drain_pending(self, *, wait: bool) -> None:
        remaining = []
        for pending in self._pending:
            if pending.event is not None:
                if not wait and not pending.event.query():
                    remaining.append(pending)
                    continue
                pending.event.synchronize()
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
        self._pending = remaining

    @staticmethod
    def _tensor_sha256(tensor: torch.Tensor) -> str:
        value = tensor.detach().cpu().contiguous()
        return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()

    @staticmethod
    def _target_snapshot(model: nn.Module) -> tuple[dict[str, torch.Tensor], dict]:
        parameters = dict(model.named_parameters())
        embedding = parameters.get("model.embed_tokens.weight")
        if embedding is None:
            raise ValueError("Target model has no model.embed_tokens.weight")
        head = parameters.get("lm_head.weight")
        if head is None:
            head = embedding
        tensors = {
            "model.embed_tokens.weight": embedding.detach().cpu().contiguous(),
            "lm_head.weight": head.detach().cpu().contiguous().clone(),
        }
        return tensors, {
            name: {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "sha256": OnlineEagleCapture._tensor_sha256(tensor),
            }
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

    def seal(
        self,
        output_dir: str | os.PathLike[str],
        *,
        target_model: nn.Module,
        target_config: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Write an immutable atomic capture directory and return its manifest."""
        if not self.active:
            return {"active": False, "worker_rank": self.config.worker_rank}
        self._drain_pending(wait=True)
        destination = Path(output_dir)
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = destination.with_name(f".{destination.name}.tmp-{uuid4().hex}")
        staging.mkdir(parents=True, exist_ok=False)
        windows: list[dict[str, Any]] = []
        try:
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
                        "group_id": request.group_id,
                        "tokens": int(tensors["input_ids"].shape[0]),
                        "supervised_tokens": int(tensors["loss_mask"].sum().item()),
                        "sha256": file_sha256(window_path),
                    }
                )

            target_tensors, target_inventory = self._target_snapshot(target_model)
            target_path = staging / "target.safetensors"
            save_file(target_tensors, str(target_path), metadata={"format": "pt"})
            config_path = staging / "target-config.json"
            config_path.write_text(
                json.dumps(dict(target_config), sort_keys=True, separators=(",", ":"))
            )
            manifest = {
                "format": "vllm-online-eagle-capture",
                "format_version": _FORMAT_VERSION,
                "active": True,
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
                "target": {
                    "weights_path": target_path.name,
                    "weights_sha256": file_sha256(target_path),
                    "config_path": config_path.name,
                    "config_sha256": file_sha256(config_path),
                    "inventory": target_inventory,
                },
            }
            manifest_path = staging / "manifest.json"
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
            if destination.exists():
                raise FileExistsError(
                    f"Capture destination already exists: {destination}"
                )
            os.replace(staging, destination)
            return {**manifest, "path": str(destination / "manifest.json")}
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise


__all__ = [
    "OnlineEagleCapture",
    "OnlineEagleCaptureConfig",
    "file_sha256",
    "load_candidate",
    "request_group_from_id",
]
