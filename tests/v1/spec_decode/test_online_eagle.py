# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file
from torch import nn

from vllm.v1.spec_decode.online_eagle import (
    OnlineEagleCapture,
    OnlineEagleCaptureConfig,
    request_group_from_id,
)
from vllm.v1.worker import gpu_model_runner
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
        self.owned = nn.Linear(2, 2, bias=False)
        self.lm_head = nn.Linear(2, 4, bias=False)
        self.register_buffer("draft_id_to_target_id", torch.tensor([1, 2, 3, 4]))

    def load_weights(self, weights) -> None:
        for name, value in weights:
            if name == "owned.weight":
                self.owned.weight.data.copy_(value)


class _PrefixedDraftModel(nn.Module):
    def __init__(self, embedding: nn.Embedding) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = embedding
        self.model.owned = nn.Linear(2, 2, bias=False)

    def load_weights(self, weights) -> None:
        for name, value in weights:
            if name == "owned.weight":
                self.model.owned.weight.data.copy_(value)


class _SingleRankDPGroup:
    rank_in_group = 0

    @staticmethod
    def broadcast_object(value, src=0):
        assert src == 0
        return value

    @staticmethod
    def broadcast_tensor_dict(value, src=0):
        assert src == 0
        return value


@pytest.fixture
def should_do_global_cleanup_after_test() -> bool:
    """This module does not initialize distributed state."""
    return False


def _states(values: list[int]) -> tuple[list[torch.Tensor], torch.Tensor]:
    base = torch.tensor(values, dtype=torch.float32).reshape(-1, 1)
    aux = [base + offset for offset in (100, 200, 300)]
    return aux, torch.cat([base + 400, base + 500], dim=-1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA side streams")
def test_async_capture_copy_survives_source_allocator_reuse() -> None:
    capture = OnlineEagleCapture(
        OnlineEagleCaptureConfig.from_mapping(
            {
                "step": 1,
                "max_tokens": 32,
                "max_window_tokens": 8,
                "max_sequences_per_prompt_group": 1,
                "trainer_rank": 0,
                "worker_rank": 0,
                "target_revision": "target-0",
                "draft_revision": "draft-0",
                "aux_layer_ids": [2, 13, 23],
            }
        )
    )
    values = torch.arange(4096, device="cuda", dtype=torch.float32)
    aux = [values[:, None] + offset for offset in (100, 200, 300)]
    head = torch.stack((values + 400, values + 500), dim=-1)
    expected_rows = [17, 2049, 4095]

    tokens, selected_aux, selected_head, event = capture._copy_selected_rows(
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
    assert event is not None
    event.synchronize()

    assert tokens.tolist() == expected_rows
    assert selected_aux[:, 0].tolist() == [117.0, 2149.0, 4195.0]
    assert selected_head[:, 0].tolist() == [417.0, 2449.0, 4495.0]
    # Retain these allocations through synchronization so the CUDA allocator
    # gets a chance to reuse the released source blocks.
    del allocator_pressure


def test_token_keyed_capture_discards_rejected_branch_and_keeps_replacement(
    tmp_path,
) -> None:
    config = OnlineEagleCaptureConfig.from_mapping(
        {
            "step": 7,
            "max_tokens": 32,
            "max_window_tokens": 8,
            "max_sequences_per_prompt_group": 1,
            "trainer_rank": 0,
            "worker_rank": 0,
            "target_revision": "target-6",
            "draft_revision": "draft-6",
            "aux_layer_ids": [2, 13, 23],
        }
    )
    capture = OnlineEagleCapture(config)
    request_id = "skyrl-group-deadbeef-attempt0"
    assert request_group_from_id(request_id) == "deadbeef"
    assert capture.admit_request(request_id, [10, 11], 4)
    assert not capture.admit_request("skyrl-group-deadbeef-attempt1", [10], 4)

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


def test_capture_crops_a_long_prefill_before_copying() -> None:
    config = OnlineEagleCaptureConfig.from_mapping(
        {
            "step": 1,
            "max_tokens": 4,
            "max_window_tokens": 4,
            "max_sequences_per_prompt_group": 1,
            "trainer_rank": 0,
            "worker_rank": 0,
            "target_revision": "target-0",
            "draft_revision": "draft-0",
            "aux_layer_ids": [2, 13, 23],
        }
    )
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


def test_nonowner_capture_does_not_snapshot_target(tmp_path) -> None:
    config = OnlineEagleCaptureConfig.from_mapping(
        {
            "step": 1,
            "max_tokens": 4,
            "max_window_tokens": 4,
            "max_sequences_per_prompt_group": 1,
            "trainer_rank": 0,
            "worker_rank": 0,
            "target_revision": "target-0",
            "draft_revision": "draft-0",
            "aux_layer_ids": [2, 13, 23],
            "capture_target_snapshot": False,
        }
    )

    manifest = OnlineEagleCapture(config).seal(
        tmp_path / "capture",
        target_model=_TargetModel(),
        draft_model=SimpleNamespace(),
        target_config={"hidden_size": 2, "vocab_size": 64},
    )

    assert manifest["target"] is None
    assert not (tmp_path / "capture" / "target.safetensors").exists()


def test_candidate_install_is_in_place_and_preserves_shared_embedding(
    tmp_path, monkeypatch
) -> None:
    target = _TargetModel()
    draft = _DraftModel(target.model.embed_tokens)
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.model = target
    runner.drafter = SimpleNamespace(model=draft)
    runner.parallel_config = SimpleNamespace(data_parallel_rank=0)
    candidate_dir = tmp_path / "candidate"
    candidate_dir.mkdir()
    weights_path = candidate_dir / "model.safetensors"
    candidate = {"owned.weight": torch.full((2, 2), 7.0)}
    save_file(candidate, str(weights_path))
    weights_sha256 = hashlib.sha256(weights_path.read_bytes()).hexdigest()
    (candidate_dir / "manifest.json").write_text(
        json.dumps(
            {
                "format": "marinskyrl-online-eagle-candidate",
                "complete": True,
                "draft_revision": "draft-step-7",
                "weights_path": weights_path.name,
                "weights_sha256": weights_sha256,
                "tensor_inventory": {
                    "owned.weight": {
                        "shape": list(candidate["owned.weight"].shape),
                        "dtype": str(candidate["owned.weight"].dtype),
                    }
                },
            }
        )
    )
    monkeypatch.setattr(gpu_model_runner, "get_dp_group", _SingleRankDPGroup)
    parameter_id = id(draft.owned.weight)
    storage_pointer = draft.owned.weight.data_ptr()

    result = runner.install_online_eagle_speculator(str(candidate_dir), 0)

    assert torch.equal(draft.owned.weight, candidate["owned.weight"])
    assert id(draft.owned.weight) == parameter_id
    assert draft.owned.weight.data_ptr() == storage_pointer
    assert draft.model.embed_tokens is target.model.embed_tokens
    assert result == {
        "active": True,
        "worker_rank": 0,
        "draft_revision": "draft-step-7",
        "weights_sha256": weights_sha256,
        "tensor_count": 1,
    }


def test_candidate_tensor_install_and_snapshot_are_transactional() -> None:
    target = _TargetModel()
    draft = _DraftModel(target.model.embed_tokens)
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.model = target
    runner.drafter = SimpleNamespace(model=draft)
    runner.parallel_config = SimpleNamespace(data_parallel_rank=0)
    original = runner.snapshot_online_eagle_speculator(["owned.weight"])
    candidate = {"owned.weight": torch.full((2, 2), 7.0)}
    metadata = {
        "draft_revision": "draft-step-7",
        "weights_sha256": "payload-digest",
        "tensor_inventory": {
            "owned.weight": {
                "shape": [2, 2],
                "dtype": "torch.float32",
            }
        },
    }
    parameter_id = id(draft.owned.weight)
    storage_pointer = draft.owned.weight.data_ptr()

    installed = runner.install_online_eagle_speculator_tensors(metadata, candidate)
    runner.install_online_eagle_speculator_tensors(
        {
            **metadata,
            "draft_revision": "draft-initial",
            "weights_sha256": "incumbent-digest",
        },
        original,
    )

    assert installed["weights_sha256"] == "payload-digest"
    assert torch.equal(draft.owned.weight, original["owned.weight"])
    assert id(draft.owned.weight) == parameter_id
    assert draft.owned.weight.data_ptr() == storage_pointer


def test_candidate_snapshot_resolves_loader_model_prefix() -> None:
    target = _TargetModel()
    draft = _PrefixedDraftModel(target.model.embed_tokens)
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.model = target
    runner.drafter = SimpleNamespace(model=draft)

    snapshot = runner.snapshot_online_eagle_speculator(["owned.weight"])

    assert torch.equal(snapshot["owned.weight"], draft.model.owned.weight)


def test_direct_candidate_install_rejects_target_owned_head() -> None:
    target = _TargetModel()
    draft = _DraftModel(target.model.embed_tokens)
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.model = target
    runner.drafter = SimpleNamespace(model=draft)
    runner.parallel_config = SimpleNamespace(data_parallel_rank=0)
    candidate = {"lm_head.weight": torch.ones_like(draft.lm_head.weight)}
    metadata = {
        "draft_revision": "draft-step-7",
        "weights_sha256": "payload-digest",
        "tensor_inventory": {
            "lm_head.weight": {
                "shape": list(draft.lm_head.weight.shape),
                "dtype": str(draft.lm_head.weight.dtype),
            }
        },
    }

    with pytest.raises(ValueError, match="target-owned tensors"):
        runner.install_online_eagle_speculator_tensors(metadata, candidate)


def test_target_sync_refreshes_draft_vocabulary_head() -> None:
    target = _TargetModel()
    draft = _DraftModel(target.model.embed_tokens)
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.model = target
    runner.drafter = SimpleNamespace(model=draft)
    target.lm_head.weight.data.copy_(torch.arange(128).reshape(64, 2))

    runner.refresh_online_eagle_target_owned_weights()

    assert torch.equal(draft.lm_head.weight, target.lm_head.weight[[1, 3, 5, 7]])
