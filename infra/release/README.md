# Marin vLLM GPU releases

The GPU release flow publishes `vllm` wheels under commit-addressed Marin vLLM
GitHub release tags. It does not publish a `marin-vllm` distribution or maintain
a moving `latest` alias.

## Build configuration

[`config.json`](config.json) is the release ABI contract. It pins CPython 3.12,
Torch 2.11.0+cu129, CUDA 12.9.1, digest-pinned upstream manylinux builder
images, deployment-specific SM targets, Iris validation hardware, and the digest-pinned
multi-architecture validation image. Update the config and workflows in one PR
when an ABI changes.

The x86_64 and aarch64 builds reuse the `build` target in
[`docker/Dockerfile`](../../docker/Dockerfile). That is the same CUDA 12.9 path
used by upstream's release pipeline. Release code does not edit
`requirements/cuda.txt`, `requirements/build/cuda.txt`, or the `vllm`
distribution metadata.

Each native wheel contains code for the GPU on which it is promoted: SM90 for
the x86_64 H100 lane and SM100 for the aarch64 GB200 lane. These are Marin
deployment artifacts rather than general-purpose vLLM wheels. Limiting the
targets keeps native compilation within the memory available on GitHub-hosted
runners; `MAX_JOBS` and `NVCC_THREADS` are both one for the same reason. Add a
target only with corresponding Iris validation hardware.

## Candidate publication

[`marin-gpu-candidate.yaml`](../../.github/workflows/marin-gpu-candidate.yaml)
runs on every merge to `main`. It builds both native wheels, derives the
manylinux tag from each wheel's ELF symbols, and publishes a prerelease named
`marin-vllm-gpu-candidate-<12-character-sha>`.

The candidate manifest records:

- fork commit and upstream merge base;
- Python, Torch, CUDA, platform, and SM targets;
- builder image and GitHub Actions provenance;
- wheel filename, tags, size, and SHA-256;
- packaged `_C`, cuMem allocator, and Grug model state.

Candidate tags and assets are immutable. A rerun verifies an existing
candidate instead of replacing it.

## GPU validation and release

[`marin-gpu-release.yaml`](../../.github/workflows/marin-gpu-release.yaml) runs
on a schedule and through `workflow_dispatch`. The optional `candidate_tag`
input selects an exact candidate; an empty input selects the newest candidate.

The workflow runs two Iris jobs:

- H100x1 on `cw-rno2a` installs the x86_64 wheel, checks `_C` and
  `GrugMoeForCausalLM`, allocates through cuMem, runs the Marin delta tests, and
  serves Qwen/Qwen3-0.6B against the H100 spec.
- GB200x1 on `cw-us-east-08a` performs the same wheel, extension, allocator,
  and serving checks for the native aarch64 wheel. The source test suite stays
  on H100 because it is architecture-independent.

An absent cuMem extension is recorded as `absent` and fails promotion. Iris
setup failures and missing validation output also become explicit failed JSON
records.

The runtime probe and serving process run with the temporary venv outside the
checkout. The H100 test runner imports and verifies `vllm` from that venv before
adding the checkout to `sys.path` for the `tests` package. It rejects a `vllm`
path under the checkout, so the tests exercise the wheel's Python modules and
compiled extensions.

Both results must pass before the workflow creates
`marin-vllm-gpu-<UTC-date>-<12-character-sha>`. The final release contains the
unchanged candidate wheels, both validation records, and a final manifest that
binds every result to a wheel digest. The workflow never overwrites an existing
release tag or asset.

Dispatch a specific candidate with:

```bash
gh workflow run marin-gpu-release.yaml \
  --repo marin-community/vllm \
  -f candidate_tag=marin-vllm-gpu-candidate-0123456789ab
```
