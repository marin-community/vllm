#!/usr/bin/env bash
# The nightly's GPU work. Runs INSIDE the Iris job, on one H100, with this repo bundled
# as the working directory; .github/workflows/marin-nightly.yaml submits it.
#
# Builds the fork from the bundled tree, runs the Marin delta's own tests -- including the
# CUDA-gated case no CPU runner can reach -- then serves Qwen/Qwen3-0.6B through the
# OpenAI server and gates the answers and decode throughput against the checked-in spec.
#
# Iris bundles the working directory without its .git, so the two things setup.py would
# otherwise ask git for are passed in by the submitter, which has the real checkout:
# SETUPTOOLS_SCM_PRETEND_VERSION (the package version) and VLLM_PRECOMPILED_WHEEL_COMMIT
# (the upstream base this fork sits on, so the prebuilt kernels match it instead of
# silently falling back to upstream's nightly ones).
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
SPEC="${SPEC:-infra/nightly/specs/qwen3-0.6b-h100.json}"

cd "${IRIS_WORKDIR:-$PWD}"

echo "::: installing vLLM with upstream's prebuilt kernels"
# The Marin delta is Python-only, so upstream's kernels for this fork's base are the right
# ones and nothing needs compiling here.
uv venv --python 3.12
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto
uv pip install pytest tblib

echo "::: running the delta's tests on the GPU"
# Two tests are deselected because they cannot run on a single GPU, not because they are
# failing: each builds a ParallelConfig with a world size larger than one (tensor_parallel
# 8, pipeline_parallel 2), and vLLM rejects that against the visible GPU count before the
# behavior under test is ever reached. They pass on CPU, where no such check exists.
.venv/bin/python -m pytest -v \
  tests/models/test_grugmoe.py \
  tests/v1/core/test_scheduler.py \
  --deselect tests/models/test_grugmoe.py::test_parallel_config_allows_tpu_heads_but_preserves_gpu_checks \
  --deselect tests/v1/core/test_scheduler.py::test_async_scheduling_pp_allows_rescheduling_with_output_placeholders

echo "::: serving ${MODEL} and gating against ${SPEC}"
# vLLM's default sampler is flashinfer's, which JIT-compiles its kernels on first use. The
# Iris task image is a slim Python image with no CUDA toolkit, so that compile cannot run:
# it wants nvcc, then ninja, and once both are installed from wheels the build still fails
# on "CUDA compiler and CUDA toolkit headers are incompatible" -- the nvcc wheel (13.2) and
# the cu13 runtime headers torch pulled in are not a matched pair. Pinning a coherent CUDA
# 13 set is possible but fragile, and it would have the nightly compiling kernels at
# runtime, which is neither what production does nor what this gate is about. The native
# sampler needs no toolchain and exercises the same serving path for our purposes; the
# consequence, stated plainly, is that the flashinfer sampling path is not covered here.
export VLLM_USE_FLASHINFER_SAMPLER=0
.venv/bin/python infra/nightly/gpu_serve_smoke.py --model "$MODEL" --spec "$SPEC"
