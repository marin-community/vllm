# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path
from typing import Literal

import pytest
from transformers import LlamaConfig, MixtralConfig

from vllm.config import ModelConfig, ParallelConfig, SpeculativeConfig


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    ("method", "draft_is_moe"),
    [("eagle3", False), ("draft_model", False), ("draft_model", True)],
)
def test_moe_target_accepts_dense_and_moe_drafts(
    tmp_path: Path, method: Literal["eagle3", "draft_model"], draft_is_moe: bool
) -> None:
    # Configuration must accept a dense draft without disabling the target's EP.
    dimensions = dict(
        vocab_size=32,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
    )
    target = tmp_path / "target"
    draft = tmp_path / "draft"
    moe = MixtralConfig(
        **dimensions,
        architectures=["MixtralForCausalLM"],
        num_local_experts=4,
        num_experts_per_tok=2,
    )
    moe.save_pretrained(target)
    draft_config = (
        moe
        if draft_is_moe
        else LlamaConfig(**dimensions, architectures=["LlamaForCausalLM"])
    )
    draft_config.save_pretrained(draft)
    target_model = ModelConfig(
        model=str(target), skip_tokenizer_init=True, dtype="bfloat16"
    )
    parallel = ParallelConfig(enable_expert_parallel=True)

    speculative = SpeculativeConfig(
        model=str(draft),
        method=method,
        num_speculative_tokens=3,
        target_model_config=target_model,
        target_parallel_config=parallel,
    )

    assert speculative.draft_parallel_config.enable_expert_parallel is draft_is_moe
    assert parallel.enable_expert_parallel
