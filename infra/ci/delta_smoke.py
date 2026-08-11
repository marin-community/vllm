# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check that the Marin delta still resolves against the upstream it is sitting on.

This fork's job is to carry GrugMoE on top of a moving upstream, so what actually breaks
is an upstream refresh renaming or reshaping something GrugMoE imports or registers
against. That is what this checks: the model module imports, the architecture is in the
model registry, and loading an HF config with `model_type: grug_moe` through vLLM's own
config path hands back GrugMoeConfig.

It runs on a CPU-only PR runner against a python-only build (VLLM_TARGET_DEVICE=empty),
and that bounds what it covers: vLLM cannot infer a device on such a build, so nothing
needing a VllmConfig -- including every behavior test in tests/models/test_grugmoe.py --
runs here. The nightly runs those on a real H100.
"""

import json
import logging
import tempfile
from pathlib import Path

from vllm.model_executor.models.grugmoe import GrugMoeForCausalLM
from vllm.model_executor.models.registry import ModelRegistry
from vllm.transformers_utils.config import get_config
from vllm.transformers_utils.configs.grugmoe import GrugMoeConfig

logger = logging.getLogger("delta_smoke")

MODEL_ARCH = "GrugMoeForCausalLM"
CONFIG_MODEL_TYPE = "grug_moe"

# Shape-only; enough for the config path to build a GrugMoeConfig.
HF_CONFIG = {
    "model_type": CONFIG_MODEL_TYPE,
    "architectures": [MODEL_ARCH],
    "vocab_size": 128,
    "hidden_size": 16,
    "intermediate_size": 32,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "num_local_experts": 4,
    "num_experts_per_tok": 2,
}


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    supported = ModelRegistry.get_supported_archs()
    if MODEL_ARCH not in supported:
        raise AssertionError(f"{MODEL_ARCH} is missing from the vLLM model registry")

    with tempfile.TemporaryDirectory() as directory:
        (Path(directory) / "config.json").write_text(json.dumps(HF_CONFIG))
        config = get_config(directory, trust_remote_code=False)

    if not isinstance(config, GrugMoeConfig):
        raise AssertionError(
            f"model_type {CONFIG_MODEL_TYPE!r} loaded as {type(config).__name__}, "
            f"not {GrugMoeConfig.__name__}"
        )

    logger.info(
        "%s imports, is registered, and %r loads as %s",
        GrugMoeForCausalLM.__name__,
        CONFIG_MODEL_TYPE,
        type(config).__name__,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
