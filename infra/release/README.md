# Marin vLLM releases

The GPU release flow publishes `vllm` wheels under commit-addressed Marin vLLM
GitHub release tags. It does not publish a `marin-vllm` distribution or maintain
a moving `latest` alias.

## GPU build configuration

[`config.json`](config.json) is the release ABI contract. It pins CPython 3.12,
Torch 2.13.0+cu132, CUDA 13.2.1, digest-pinned upstream manylinux builder
images, deployment-specific SM targets, Iris validation hardware, and the digest-pinned
multi-architecture validation image. Update the config and workflows in one PR
when an ABI changes.

The x86_64 and aarch64 builds reuse the `wheel-build` target in
[`docker/Dockerfile`](../../docker/Dockerfile). That is the same build path
used by upstream's release pipeline. Release code does not edit
`requirements/cuda.txt`, `requirements/build/cuda.txt`, or the `vllm`
distribution metadata. A final scratch stage contains only `/dist`; BuildKit
exports that directory directly instead of loading the build image into the
runner's Docker image store.

The x86_64 wheel targets SM90 on H100, and the aarch64 wheel targets SM100 on
GB200. Every configured validation gate must pass. Compilation uses two jobs
with one NVCC thread each and an 800 MiB wheel limit.

`gpu-constraints.txt` pins the Python build and runtime dependencies for CPython
3.12 on manylinux 2.28 for both architectures. The release Docker build and
wheel validation consume it. Its direct inputs live in `gpu-constraints.in`,
including one `torchaudio==2.11.0+cpu` constraint and the Transformers and
Tokenizers versions qualified by the selected upstream CUDA test environment.
The generated file records the PyPI, CUDA 13.2, CPU Torch, and FlashInfer
indexes. uv selects compatible wheels for the builder architecture from those
indexes.

Use uv 0.11.21 and regenerate the file from the repository root with:

```bash
uv pip compile infra/release/gpu-constraints.in \
  --index-strategy unsafe-best-match \
  --index https://download.pytorch.org/whl/cu132 \
  --index https://download.pytorch.org/whl/cpu \
  --index https://flashinfer.ai/whl/ \
  --python-platform x86_64-manylinux_2_28 \
  --python-version 3.12 \
  --output-file infra/release/gpu-constraints.txt \
  --emit-index-url --no-annotate --no-header
```

The checked-in output seeds regeneration, so this command retains the frozen
dependency closure. Use `--upgrade` only when intentionally requalifying that
closure. The same inputs resolve for `aarch64-manylinux_2_28`; only the selected
platform wheel changes. The toolkit extras pin the compiler and headers used by
runtime JIT compilation. Preserve the CPU TorchAudio version constraint: the
available CUDA 13.0 TorchAudio wheel rejects Torch cu132, while audio
preprocessing uses Torch's tensor operators.

Each build job removes unused Android, .NET, and GHC toolchains from its
ephemeral hosted runner before compiling. The wheel-only BuildKit export also
avoids duplicating the build toolchains and intermediate objects in the
runner's Docker image store. Together these keep compilation and artifact
export within the hosted runners' root filesystems.

## GPU build, qualification, and publication

Review and land vLLM source changes on `main` before building wheels. This
applies to an upstream refresh and to patches authored in the fork. Marin keeps
its existing wheel until a separate adoption PR passes its tests and merges.

[`marin-gpu-candidate.yaml`](../../.github/workflows/marin-gpu-candidate.yaml)
builds, qualifies, and publishes GPU wheels in one workflow. Source changes on
`main` start it automatically; release-only and documentation changes are
excluded by `paths-ignore`. A manual dispatch must also select `main`:

```bash
gh workflow run marin-gpu-candidate.yaml \
  --repo marin-community/vllm --ref main -f lane=gpu
```

The workflow selects `marin-vllm-gpu-<source-UTC-date>-<12-character-sha>` and
skips the build if that release already exists. Both architecture builds,
dependency constraints, validation sources, and qualification code come from
the same workflow commit. A later change to `main` does not change a running
build.

The builds upload wheels and metadata to temporary GitHub Actions artifacts.
No GPU candidate release is published. The metadata records the source SHA,
upstream base, build ABI, builder image, wheel contents, and SHA-256. Schema 2
uses the final release tag throughout, including each qualification record's
`release_tag` and wheel URL.

Qualification runs on the configured Iris hardware:

- H100x1 on `cw-rno2a` installs the x86_64 wheel, checks the stable Torch
  extension and Grug model, validates sparse NCCL weight transfer, allocates
  through cuMem, runs the Marin delta tests, and serves Qwen/Qwen3-0.6B against
  the H100 spec.
- GB200x1 on `cw-us-east-08a` installs the aarch64 wheel, checks the stable
  Torch extension and Grug model, allocates through cuMem, and serves
  Qwen/Qwen3-0.6B against the GB200 spec.

Each Iris job downloads its wheel from the current workflow's Actions artifact
with a temporary read-only GitHub token. It checks the wheel's SHA-256 before
installation. The model probe and serving process run in a temporary environment
outside the checkout, so they import the installed wheel. The source tests use
a separate tree from the same commit.

Both qualification jobs must pass before publication. The workflow publishes
the built wheels once, alongside the manifest and qualification records. It
checks the recorded hashes and never overwrites a release. Failed qualification
publishes no GPU release. The temporary artifacts remain available for 14 days;
rerun the failed jobs in that workflow to retry. There is no separate GPU
promotion dispatch or reuse of another qualification run.

Download the published manifest in a Marin worktree, run
`config/update-external.py --promote-gpu-release <manifest>`, and run Marin's
Snowball parity test. Open a Marin PR with the resulting exact wheel pins and
test evidence. Merging that PR approves adoption. See Marin's
[GPU refresh guide](https://github.com/marin-community/marin/blob/main/.agents/skills/refresh-fork/docs/vllm.md).

## TPU wheel pairs

The TPU lane uses the same `main` source lineage and builds a separate vLLM
wheel with its own Torch, JAX, and libtpu environment. It never consumes or
promotes a `tpu` branch. `marin-gpu-candidate.yaml` accepts full vLLM and
tpu-inference commits plus a UTC dependency cutoff, builds both wheels once,
and publishes an immutable content-addressed prerelease and manifest.

`marin-gpu-release.yaml` redownloads that exact pair, verifies its hashes,
cold-installs it from the candidate index, and runs the Qwen3-0.6B TP8 gate on
one `v6e-8` in `us-east5`. A qualification dispatch with `promote=false` records
the physical TPU result without finalizing a release. Later promotion accepts
that successful qualification run's exact ID, revalidates its GitHub metadata
and artifact against the candidate, and reuses the same candidate bytes without
allocating another TPU.

The GPU and TPU lanes are dispatched separately. Advancing tpu-inference does
not rebuild a GPU wheel, and publishing a GPU release does not rebuild the TPU
pair.

```bash
gh workflow run marin-gpu-candidate.yaml \
  --repo marin-community/vllm \
  --ref <reviewed-workflow-ref> \
  -f lane=tpu \
  -f vllm_commit=<full-main-line-source-sha> \
  -f tpu_inference_commit=<full-tpu-inference-sha> \
  -f exclude_newer=<whole-second-utc-cutoff>

gh workflow run marin-gpu-release.yaml \
  --repo marin-community/vllm \
  --ref <same-reviewed-workflow-ref> \
  -f lane=tpu \
  -f candidate_tag=<exact-candidate-tag> \
  -f promote=false

# After source and consumer changes land, reuse the accepted qualification.
gh workflow run marin-gpu-release.yaml \
  --repo marin-community/vllm \
  --ref main \
  -f lane=tpu \
  -f candidate_tag=<same-exact-candidate-tag> \
  -f qualification_run_id=<successful-qualification-run-id> \
  -f promote=true
```
