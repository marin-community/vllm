#!/usr/bin/env bash
# The nightly's GPU work. Runs INSIDE the Iris job, on one H100, with infra/nightly/
# bundled as the working directory; .github/workflows/marin-nightly.yaml submits it.
#
# Builds this fork at the commit under test, runs the Marin delta's own tests -- including
# the CUDA-gated case no CPU runner can reach -- then serves Qwen/Qwen3-0.6B through the
# OpenAI server and gates the answers and decode throughput against the checked-in spec.
#
# The source comes from a clone rather than the Iris bundle, which is not an accident:
# setuptools-scm needs a .git to version the package, and VLLM_USE_PRECOMPILED picks its
# wheel from `git merge-base <branch> upstream/main`, so the build needs a real repository
# with a real branch checked out. The bundle ships without .git, and this repo's ~98MB
# tracked tree would blow Iris's 25MB bundle cap regardless.
set -euo pipefail

REPO="${REPO:-https://github.com/marin-community/vllm.git}"
VLLM_COMMIT="${VLLM_COMMIT:?VLLM_COMMIT must name the commit under test}"
MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
SPEC="${SPEC:-infra/nightly/specs/qwen3-0.6b-h100.json}"
SRC="${IRIS_WORKDIR:-$PWD}/vllm-src"

echo "::: cloning ${REPO} at ${VLLM_COMMIT}"
git clone --quiet "$REPO" "$SRC"
cd "$SRC"

# A branch, not a detached HEAD. setup.py resolves the precompiled wheel from the merge-base
# of the *current branch* with upstream main; detached, that lookup fails and it falls back
# to upstream's nightly kernels, which this fork's base was never built against.
git checkout -q -B nightly "$VLLM_COMMIT"

echo "::: installing vLLM with upstream's prebuilt kernels"
# The Marin delta is Python-only, so upstream's kernels for this fork's base are the right
# ones and nothing needs compiling here.
uv venv --python 3.12
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto
uv pip install pytest tblib

echo "::: running the delta's tests on the GPU"
.venv/bin/python -m pytest tests/models/test_grugmoe.py tests/v1/core/test_scheduler.py -v

echo "::: serving ${MODEL} and gating against ${SPEC}"
.venv/bin/python infra/nightly/gpu_serve_smoke.py --model "$MODEL" --spec "$SPEC"
