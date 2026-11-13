# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A Grug DSpark draft stores its routed-expert count as ``grug_num_experts``; ``ModelConfig.is_moe`` must see it."""

import pytest
from transformers import PretrainedConfig

import vllm.config  # noqa: F401  (imported first: importing the convertor module alone is a circular import)
from vllm.transformers_utils.model_arch_config_convertor import (
    ModelArchConfigConvertorBase,
)


@pytest.mark.parametrize(
    "config_kwargs, expected_experts",
    [
        # A Grug DSpark draft with routed experts is labelled MoE by its own key.
        ({"grug_num_experts": 8}, 8),
        # The usual (dense) Grug DSpark draft.
        ({"grug_num_experts": 0}, 0),
        # A target-style key takes precedence over the Grug key.
        ({"num_experts": 256, "grug_num_experts": 0}, 256),
    ],
)
def test_grug_draft_expert_count_is_read_from_config(
    config_kwargs: dict, expected_experts: int
):
    hf_config = PretrainedConfig(**config_kwargs)

    convertor = ModelArchConfigConvertorBase(hf_config, hf_config)

    assert convertor.get_num_experts() == expected_experts
