# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file
from torch import nn

from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.spec_decode.eagle import EagleProposer
from vllm.v1.spec_decode.online_eagle import (
    OnlineEagleCapture,
    OnlineEagleCaptureConfig,
)
from vllm.v1.worker import gpu_model_runner
from vllm.v1.worker.gpu import model_runner as gpu_model_runner_v2
from vllm.v1.worker.gpu_model_runner import GPUModelRunner


class _TargetModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(64, 2)
        self.lm_head = nn.Linear(2, 64, bias=False)


class _DraftModel(nn.Module):
    def __init__(self, embedding: nn.Embedding) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = embedding
        self.lm_head = nn.Linear(2, 4, bias=False)
        self.register_buffer("draft_id_to_target_id", torch.tensor([1, 2, 3, 4]))


@pytest.fixture
def should_do_global_cleanup_after_test() -> bool:
    """This module does not initialize distributed state."""
    return False


def _states(values: list[int]) -> tuple[list[torch.Tensor], torch.Tensor]:
    base = torch.tensor(values, dtype=torch.float32).reshape(-1, 1)
    aux = [base + offset for offset in (100, 200, 300)]
    return aux, torch.cat([base + 400, base + 500], dim=-1)


def _capture_config(**overrides) -> OnlineEagleCaptureConfig:
    values = {
        "step": 1,
        "max_tokens": 32,
        "max_window_tokens": 8,
        "worker_rank": 0,
        "target_revision": "target-0",
        "draft_revision": "draft-0",
        "aux_layer_ids": [2, 13, 23],
    }
    values.update(overrides)
    return OnlineEagleCaptureConfig.from_mapping(values)


def _runner_with_shared_embedding() -> tuple[_TargetModel, _DraftModel, GPUModelRunner]:
    target = _TargetModel()
    draft = _DraftModel(target.model.embed_tokens)
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.model = target
    runner.drafter = SimpleNamespace(model=draft)
    return target, draft, runner


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA side streams")
def test_async_capture_copy_survives_source_allocator_reuse() -> None:
    capture = OnlineEagleCapture(_capture_config())
    values = torch.arange(4096, device="cuda", dtype=torch.float32)
    aux = [values[:, None] + offset for offset in (100, 200, 300)]
    head = torch.stack((values + 400, values + 500), dim=-1)
    expected_rows = [17, 2049, 4095]

    copied = capture._copy_selected_rows(
        rows=[("request", position) for position in expected_rows],
        selected_rows=expected_rows,
        input_ids=values.to(torch.long),
        aux_hidden_states=aux,
        head_input_hidden_states=head,
    )
    del values, aux, head
    allocator_pressure = []
    for _ in range(512):
        allocator_pressure.extend(
            (
                torch.full((3,), -1, device="cuda", dtype=torch.long),
                torch.full((3, 3), -1.0, device="cuda"),
                torch.full((3, 2), -1.0, device="cuda"),
            )
        )
    assert copied.event is not None
    copied.event.synchronize()

    assert copied.token_ids.tolist() == expected_rows
    assert copied.aux_hidden_states[:, 0].tolist() == [117.0, 2149.0, 4195.0]
    assert copied.head_input_hidden_states[:, 0].tolist() == [417.0, 2449.0, 4495.0]
    # Retain these allocations through synchronization so the CUDA allocator
    # gets a chance to reuse the released source blocks.
    del allocator_pressure


def test_token_keyed_capture_discards_rejected_branch_and_keeps_replacement(
    tmp_path,
) -> None:
    config = _capture_config(
        step=7,
        target_revision="target-6",
        draft_revision="draft-6",
    )
    capture = OnlineEagleCapture(config)
    request_id = "skyrl-group-deadbeef-attempt0"
    assert capture.admit_request(request_id, [10, 11], 4)

    aux, head = _states([0, 1])
    capture.record_forward(
        request_ids=[request_id],
        num_scheduled_tokens=[2],
        num_computed_tokens=[0],
        input_ids=torch.tensor([10, 11]),
        aux_hidden_states=aux,
        head_input_hidden_states=head,
    )

    # Token 99 is a rejected speculative branch at position 3.
    aux, head = _states([2, 99])
    capture.record_forward(
        request_ids=[request_id],
        num_scheduled_tokens=[2],
        num_computed_tokens=[2],
        input_ids=torch.tensor([20, 99]),
        aux_hidden_states=aux,
        head_input_hidden_states=head,
    )
    # The target replacement reaches its real input state on the next forward.
    aux, head = _states([3])
    capture.record_forward(
        request_ids=[request_id],
        num_scheduled_tokens=[1],
        num_computed_tokens=[3],
        input_ids=torch.tensor([30]),
        aux_hidden_states=aux,
        head_input_hidden_states=head,
    )
    capture.finalize_request(request_id, [20, 30])

    destination = tmp_path / "capture"
    draft_model = SimpleNamespace(draft_id_to_target_id=torch.tensor([1, 2, 3, 4]))
    manifest = capture.seal(
        destination,
        target_model=_TargetModel(),
        draft_model=draft_model,
        target_config={"hidden_size": 2, "vocab_size": 64},
    )

    assert manifest["head_input_semantics"] == ("post_final_norm_target_lm_head_input")
    assert len(manifest["windows"]) == 1
    window = load_file(str(destination / manifest["windows"][0]["path"]))
    assert window["input_ids"].tolist() == [10, 11, 20, 30]
    assert window["loss_mask"].tolist() == [False, False, True, True]
    assert window["hidden_states"][-1].tolist() == [103.0, 203.0, 303.0]
    assert window["head_input_hidden_states"][-1].tolist() == [403.0, 503.0]
    assert not torch.any(window["hidden_states"] == 399)
    on_disk_manifest = json.loads((destination / "manifest.json").read_text())
    assert on_disk_manifest["target"]["inventory"]["lm_head.weight"]["shape"] == [
        4,
        2,
    ]


def test_sampled_capture_uses_scheduler_final_output_length(tmp_path) -> None:
    capture = OnlineEagleCapture(_capture_config())
    request_id = "request"
    assert capture.admit_request(request_id, [10, 11], 4)

    aux, head = _states([0, 1, 2, 3, 4])
    capture.record_forward(
        request_ids=[request_id],
        num_scheduled_tokens=[5],
        num_computed_tokens=[0],
        input_ids=torch.tensor([10, 11, 20, 30, 99]),
        aux_hidden_states=aux,
        head_input_hidden_states=head,
    )
    capture.record_sampled(
        request_ids=[request_id],
        sampled_token_ids=torch.tensor([[20, 30, 99]]),
        num_sampled_tokens=torch.tensor([3]),
    )
    capture.finalize_request_length(request_id, 2)

    destination = tmp_path / "capture"
    manifest = capture.seal(
        destination,
        target_model=_TargetModel(),
        draft_model=SimpleNamespace(draft_id_to_target_id=torch.tensor([1, 2, 3, 4])),
        target_config={"hidden_size": 2, "vocab_size": 64},
    )

    window = load_file(str(destination / manifest["windows"][0]["path"]))
    assert window["input_ids"].tolist() == [10, 11, 20, 30]


def test_capture_crops_a_long_prefill_before_copying() -> None:
    config = _capture_config(max_tokens=4, max_window_tokens=4)
    capture = OnlineEagleCapture(config)
    request_id = "skyrl-group-cafebabe-attempt0"
    prompt = list(range(10))
    assert capture.admit_request(request_id, prompt, 1)
    aux, head = _states(prompt)

    capture.record_forward(
        request_ids=[request_id],
        num_scheduled_tokens=[len(prompt)],
        num_computed_tokens=[0],
        input_ids=torch.tensor(prompt),
        aux_hidden_states=aux,
        head_input_hidden_states=head,
    )

    assert capture.captured_rows == 4


def test_capture_can_skip_target_snapshot(tmp_path) -> None:
    config = _capture_config(
        max_tokens=4,
        max_window_tokens=4,
        capture_target_snapshot=False,
    )

    manifest = OnlineEagleCapture(config).seal(
        tmp_path / "capture",
        target_model=_TargetModel(),
        draft_model=SimpleNamespace(),
        target_config={"hidden_size": 2, "vocab_size": 64},
    )

    assert manifest["target"] is None
    assert not (tmp_path / "capture" / "target.safetensors").exists()


def test_begin_capture_replaces_unsealed_scratch(monkeypatch) -> None:
    runner = GPUModelRunner.__new__(GPUModelRunner)
    old_capture = OnlineEagleCapture(_capture_config(step=1))
    runner.online_eagle_capture = old_capture
    runner.speculative_config = SimpleNamespace(method="eagle3")
    runner.use_async_scheduling = False
    runner.parallel_config = SimpleNamespace(data_parallel_rank=0)
    runner.effective_drafter_max_model_len = 8
    runner._get_eagle3_aux_layers_from_config = lambda: (2, 13, 23)
    monkeypatch.setattr(
        gpu_model_runner,
        "get_pp_group",
        lambda: SimpleNamespace(world_size=1),
    )

    request = asdict(_capture_config(step=2))
    del request["worker_rank"], request["aux_layer_ids"]
    result = runner.begin_online_eagle_capture(request)

    assert runner.online_eagle_capture is not old_capture
    assert result == {"active": True, "worker_rank": 0, "step": 2}


def test_begin_capture_rejects_worker_owned_fields(monkeypatch) -> None:
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.speculative_config = SimpleNamespace(method="eagle3")
    runner.use_async_scheduling = False
    monkeypatch.setattr(
        gpu_model_runner,
        "get_pp_group",
        lambda: SimpleNamespace(world_size=1),
    )

    with pytest.raises(ValueError, match="worker-owned: worker_rank"):
        runner.begin_online_eagle_capture({"worker_rank": 3})


def test_target_sync_refreshes_draft_vocabulary_head() -> None:
    target, draft, runner = _runner_with_shared_embedding()
    target.lm_head.weight.data.copy_(torch.arange(128).reshape(64, 2))

    runner.refresh_online_eagle_target_owned_weights()

    assert torch.equal(draft.lm_head.weight, target.lm_head.weight[[1, 3, 5, 7]])


def test_v2_target_sync_refreshes_draft_vocabulary_head() -> None:
    target = _TargetModel()
    draft = _DraftModel(target.model.embed_tokens)
    target.lm_head.weight.data.copy_(torch.arange(128).reshape(64, 2))
    runner = gpu_model_runner_v2.GPUModelRunner.__new__(
        gpu_model_runner_v2.GPUModelRunner
    )
    runner.model = target
    runner.get_draft_model = lambda: draft

    runner.refresh_online_eagle_target_owned_weights()

    assert torch.equal(draft.lm_head.weight, target.lm_head.weight[[1, 3, 5, 7]])


def test_aux_layers_fall_back_to_target_model_defaults(monkeypatch) -> None:
    runner = GPUModelRunner.__new__(GPUModelRunner)
    model = SimpleNamespace(
        get_eagle3_default_aux_hidden_state_layers=lambda: [2, 13, 23]
    )
    runner.model = model
    runner.speculative_config = SimpleNamespace(draft_model_config=None)
    monkeypatch.setattr(
        gpu_model_runner, "supports_eagle3", lambda value: value is model
    )

    assert runner._resolve_eagle3_aux_layers() == (2, 13, 23)


def test_kv_cache_specs_mark_proposer_discovered_draft_layers(monkeypatch) -> None:
    spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=8,
        dtype=torch.float32,
    )
    backend = SimpleNamespace(customize_spec=lambda value: value)
    attention = SimpleNamespace(
        get_kv_cache_spec=lambda _config: spec,
        get_attn_backend=lambda: backend,
    )
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.vllm_config = SimpleNamespace()
    runner.speculative_config = SimpleNamespace(method="eagle3")
    runner.shared_kv_cache_layers = {}
    runner.drafter = EagleProposer.__new__(EagleProposer)
    runner.drafter._draft_attn_layer_names = {"draft.attn"}
    monkeypatch.setattr(gpu_model_runner, "has_ec_transfer", lambda: False)
    monkeypatch.setattr(
        gpu_model_runner,
        "get_layers_from_vllm_config",
        lambda *_args: {"target.attn": attention, "draft.attn": attention},
    )

    specs = runner.get_kv_cache_spec()

    assert specs["draft.attn"].is_draft_attention is True
    assert specs["target.attn"].is_draft_attention is False
