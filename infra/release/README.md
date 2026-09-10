# Marin vLLM GPU releases

The GPU release flow publishes `vllm` wheels under commit-addressed Marin vLLM
GitHub release tags. It does not publish a `marin-vllm` distribution or maintain
a moving `latest` alias.

## Build configuration

[`config.json`](config.json) is the release ABI contract. It pins CPython 3.12,
Torch 2.13.0+cu132, CUDA 13.2.1, digest-pinned upstream manylinux builder
images, deployment-specific SM targets, Iris validation hardware, and the digest-pinned
multi-architecture validation image. Update the config and workflows in one PR
when an ABI changes.

The x86_64 build reuses the `wheel-build` target in
[`docker/Dockerfile`](../../docker/Dockerfile). That is the same build path
used by upstream's release pipeline. Release code does not edit
`requirements/cuda.txt`, `requirements/build/cuda.txt`, or the `vllm`
distribution metadata. A final scratch stage contains only `/dist`; BuildKit
exports that directory directly instead of loading the build image into the
runner's Docker image store.

The wheel targets SM90 on x86_64 H100. This release does not build or qualify
an aarch64 vLLM wheel. The configured H100 validation gates must all pass.
Compilation uses two jobs with one NVCC thread each and an 800 MiB wheel limit.

`gpu-constraints.txt` pins the Python build and runtime dependencies for CPython
3.12 on Linux x86_64. Both the release Docker build and wheel validation consume
it. Regenerate from `requirements/cuda.txt`, `requirements/build/cuda.txt`, and
`cuda-toolkit[nvcc,cccl]==13.2.1` with the configured PyTorch index when changing
the ABI. The toolkit extras pin the compiler and headers used by runtime JIT
compilation. Preserve the direct
TorchAudio CPU wheel constraint: the available CUDA 13.0 TorchAudio wheel
rejects Torch cu132, while audio preprocessing uses Torch's tensor operators.

The x86_64 candidate job removes unused Android, .NET, and GHC toolchains from
its ephemeral hosted runner before compiling. The wheel-only BuildKit export
also avoids duplicating the build toolchains and intermediate objects in the
runner's Docker image store. Together these keep compilation and artifact
export within the hosted runners' root filesystems.

## Candidate publication

[`marin-gpu-candidate.yaml`](../../.github/workflows/marin-gpu-candidate.yaml)
runs on every merge to `main`. It builds the H100 native wheel, derives the
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

The workflow qualifies the x86_64 wheel on H100:

- H100x1 on `cw-us-east-02a` installs the x86_64 wheel, checks `_C` and
  `GrugMoeForCausalLM`, allocates through cuMem, runs the Marin delta tests, and
  serves Qwen/Qwen3-0.6B against the H100 spec.

An absent cuMem extension is recorded as `absent` and fails promotion. Iris
setup failures and missing validation output also become explicit failed JSON
records.

The runtime probe and serving process run with the temporary venv outside the
checkout. The workflow extracts the candidate commit's tests and serving smoke
into a separate validation-source tree, while the release harness comes from
the workflow commit. The H100 test runner imports and verifies `vllm` from the
venv before adding that tree to `sys.path` for the `tests` package. It keeps its
working directory outside the tree as well, so model-inspection subprocesses
also import the wheel instead of an unbuilt source package.

The H100 result must pass before the workflow creates
`marin-vllm-gpu-<UTC-date>-<12-character-sha>`. The final release contains the
unchanged candidate wheel, its validation record, and a final manifest that
binds every result to a wheel digest. The workflow never overwrites an existing
release tag or asset.

Dispatch a specific candidate with:

```bash
gh workflow run marin-gpu-release.yaml \
  --repo marin-community/vllm \
  -f candidate_tag=marin-vllm-gpu-candidate-0123456789ab
```
