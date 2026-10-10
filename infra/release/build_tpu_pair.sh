#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
set -euo pipefail

# Run in an empty /work directory in Dockerfile.tpu-build's x86_64 image.
vllm_commit=${1:?full vLLM commit required}
tpu_commit=${2:?full tpu-inference commit required}
exclude_newer=${3:?whole-second UTC dependency cutoff required}
[[ "$vllm_commit" =~ ^[0-9a-f]{40}$ && "$tpu_commit" =~ ^[0-9a-f]{40}$ ]]
[[ "$exclude_newer" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$ ]]

for distribution in vllm tpu-inference; do
    commit=$vllm_commit
    if [[ "$distribution" == tpu-inference ]]; then
        commit=$tpu_commit
    fi
    git init "$distribution"
    git -C "$distribution" remote add origin "https://github.com/marin-community/$distribution.git"
    git -C "$distribution" fetch --depth 1 origin "$commit"
    git -C "$distribution" checkout --detach FETCH_HEAD
    [[ "$(git -C "$distribution" rev-parse HEAD)" == "$commit" ]]
done

mkdir pair
dpkg-query -W > pair/system-packages.txt
g++ --version > pair/cxx-version.txt
sha256sum /usr/bin/x86_64-linux-gnu-g++-12 /usr/lib/gcc/x86_64-linux-gnu/12/cc1plus \
    /usr/bin/ld.bfd /usr/local/bin/uv > pair/compiler-inputs.sha256
uv --version > pair/uv-version.txt
python --version > pair/python-version.txt

cd vllm
uv venv --python 3.12.13 .tpu-wheel-build
uv pip install --python .tpu-wheel-build/bin/python \
    --exclude-newer "$exclude_newer" --index-strategy unsafe-best-match \
    --requirements requirements/build/tpu.txt
uv pip freeze --python .tpu-wheel-build/bin/python > /work/pair/vllm-build-dependencies.txt
VLLM_TARGET_DEVICE=tpu \
    VLLM_VERSION_OVERRIDE="0.0.0.dev20260929+marin.${vllm_commit:0:12}.tpu" \
    SOURCE_DATE_EPOCH="$(git show -s --format=%ct HEAD)" \
    uv build --wheel --no-build-isolation --python .tpu-wheel-build/bin/python \
    --exclude-newer "$exclude_newer" --out-dir /work/pair

cd /work/tpu-inference
VLLM_VERSION_OVERRIDE="0.30.0+marin.${tpu_commit:0:12}" \
    SOURCE_DATE_EPOCH="$(git show -s --format=%ct HEAD)" \
    uv build --wheel --python 3.12.13 \
    --exclude-newer "$exclude_newer" --out-dir /work/pair
sha256sum /work/pair/*.whl > /work/pair/wheels.sha256
