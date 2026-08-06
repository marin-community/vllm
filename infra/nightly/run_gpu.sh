#!/usr/bin/env bash
# The nightly's accelerator work. Runs inside an Iris job with this repo bundled as the
# working directory; .github/workflows/marin-nightly.yaml selects H100 or GB200.
#
# Builds the fork from the bundled tree, runs the Marin delta's behavior tests, then
# serves Qwen/Qwen3-0.6B through the OpenAI server with an explicit attention backend
# and gates the answers and decode throughput against the hardware-specific spec.
#
# Iris bundles the working directory without its .git, so the two things setup.py would
# otherwise ask git for are passed in by the submitter, which has the real checkout:
# SETUPTOOLS_SCM_PRETEND_VERSION (the package version) and VLLM_PRECOMPILED_WHEEL_COMMIT
# (the upstream base this fork sits on, so the prebuilt kernels match it instead of
# silently falling back to upstream's nightly ones).
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
SPEC="${SPEC:-infra/nightly/specs/qwen3-0.6b-h100.json}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-FLASH_ATTN}"

cd "${IRIS_WORKDIR:-$PWD}"

echo "::: installing vLLM with upstream's prebuilt kernels"
# The Marin delta is Python-only, so upstream's kernels for this fork's base are the right
# ones and nothing needs compiling here.
uv venv --python 3.12
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto
# Ray is an optional runtime dependency, but its rank-remapping contract is part of this
# fork's SkyRL-facing delta. Match the version pinned by requirements/test/cuda.txt.
uv pip install pytest tblib "ray[cgraph,default]==2.48.0"

echo "::: running the delta's behavior tests on the accelerator"
# Two tests are deselected because they cannot run on a single GPU, not because they are
# failing: each builds a ParallelConfig with a world size larger than one (tensor_parallel
# 8, pipeline_parallel 2), and vLLM rejects that against the visible GPU count before the
# behavior under test is ever reached. They pass on CPU, where no such check exists.
.venv/bin/python -m pytest -v \
  tests/models/test_grugmoe.py \
  tests/v1/core/test_scheduler.py \
  tests/engine/test_arg_utils.py::TestDpDeviceIdSharding \
  tests/distributed/test_mq_connect_ip.py::test_mq_bind_with_local_ip \
  tests/model_executor/test_routed_experts_capture.py \
  tests/v1/executor/test_ray_utils.py \
  --deselect tests/models/test_grugmoe.py::test_grug_moe_parallel_config_rejects_tp_larger_than_attention_heads \
  --deselect tests/v1/core/test_scheduler.py::test_async_scheduling_pp_allows_rescheduling_with_output_placeholders

echo "::: serving ${MODEL} with ${ATTENTION_BACKEND} and gating against ${SPEC}"
# vLLM's default sampler is flashinfer's, which JIT-compiles its kernels on first use. The
# Iris task image is a slim Python image with no coherent CUDA toolkit, so use the native
# sampler. Explicit FLASHINFER dispatches still exercise FlashInfer attention; they do not
# claim coverage of the separate FlashInfer sampling path.
export VLLM_USE_FLASHINFER_SAMPLER=0
serve_args=(
  --model "$MODEL"
  --spec "$SPEC"
  --attention-backend "$ATTENTION_BACKEND"
)
# On H100, FlashInfer otherwise selects its TRTLLM XQA path, which JIT-compiles a
# machine-specific extension. The slim source-validation image deliberately has no CUDA
# development tree, so exercise native FlashInfer here and leave XQA to the release image.
if [[ "$ATTENTION_BACKEND" == FLASHINFER ]]; then
  serve_args+=(--disable-trtllm-attention)
fi
.venv/bin/python infra/nightly/gpu_serve_smoke.py "${serve_args[@]}"
