# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Checkpoint weight transfer from an exact Run:ai streamer URI."""

from dataclasses import dataclass

import torch

from vllm.config import VllmConfig
from vllm.config.weight_transfer import WeightTransferConfig
from vllm.distributed.weight_transfer.base import (
    WeightTransferEngine,
    WeightTransferInitInfo,
    WeightTransferUpdateInfo,
)
from vllm.model_executor.model_loader.weight_utils import (
    runai_safetensors_weights_iterator,
)

@dataclass
class RunaiStreamerWeightTransferInitInfo(WeightTransferInitInfo):
    """The Run:ai streamer receiver needs no rendezvous state."""


@dataclass
class RunaiStreamerWeightTransferUpdateInfo(WeightTransferUpdateInfo):
    """One immutable safetensors object to load into the selected model."""

    weights_path: str = ""

    def __post_init__(self) -> None:
        if not self.weights_path:
            raise ValueError("weights_path must be a nonempty exact object URI")


class RunaiStreamerWeightTransferEngine(
    WeightTransferEngine[
        RunaiStreamerWeightTransferInitInfo,
        RunaiStreamerWeightTransferUpdateInfo,
    ]
):
    """Stream a completed safetensors checkpoint into the active model."""

    init_info_cls = RunaiStreamerWeightTransferInitInfo
    update_info_cls = RunaiStreamerWeightTransferUpdateInfo

    def __init__(
        self,
        config: WeightTransferConfig,
        vllm_config: VllmConfig,
        device: torch.device,
        model: torch.nn.Module,
    ) -> None:
        super().__init__(config, vllm_config, device, model)

    def init_transfer_engine(
        self, init_info: RunaiStreamerWeightTransferInitInfo
    ) -> None:
        pass

    def start_weight_update(self) -> None:
        pass

    def receive_weights(
        self, update_info: RunaiStreamerWeightTransferUpdateInfo
    ) -> None:
        weights = runai_safetensors_weights_iterator(
            [update_info.weights_path], use_tqdm_on_load=False
        )
        self.model.load_weights(weights)

    def finish_weight_update(self) -> None:
        pass

    def shutdown(self) -> None:
        pass
