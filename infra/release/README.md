# Marin vLLM releases

GPU releases publish `vllm` wheels under tags that name the source commit.
Marin adopts a release in a separate PR after its model tests pass.

## GPU build configuration

[`config.json`](config.json) pins the build and test environment: CPython 3.12,
Torch 2.14.1+cu132, CUDA 13.2.2, digest-pinned upstream manylinux builder
images, target GPU compute capabilities, Iris test hardware, and the
multi-architecture test image. Update the config and workflows in one PR when
these build settings change.

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

`gpu-constraints.txt` and `gpu-constraints-aarch64.txt` pin the Python build and
runtime dependencies for CPython 3.12 on manylinux 2.28. The release Docker build
and wheel validation consume the file for their architecture. Their direct
inputs live in `gpu-constraints.in`,
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
closure. For ARM, use `aarch64-manylinux_2_28` and write
`gpu-constraints-aarch64.txt`. That closure selects the maintained cuSPARSELt
wheel with its repaired ARM platform tag. The toolkit extras pin the compiler
and headers used by runtime JIT compilation. Preserve the CPU TorchAudio version
constraint: the
available CUDA 13.0 TorchAudio wheel rejects Torch cu132, while audio
preprocessing uses Torch's tensor operators.

The build installs CUDA compiler components 13.2.86 from the immutable URLs and
SHA-256 values in `cuda-compiler-inputs.json`. The digest-pinned builder supplies
the remaining toolkit. The manifest records the installed compiler provenance,
architecture-specific constraints, and packaged native payload hashes. Build
timestamps use the source commit time; caches and vendor binaries remain part
of the build inputs.

The qualified 0.30 port also retains its pinned Rust toolchain, frontend Cargo
profile, NVCC temporary-directory and source-seed flags, and bounded local
indices in the ARM quantization kernel. Those adaptations remain unchanged;
their measured reproduction limits and removal conditions are recorded in the
[qualified recipe](https://github.com/marin-community/vllm/blob/2e5249991b94a79d5d80ad363e8bb0ac30d304a6/infra/release/README.md#gpu-build-configuration).

Each build job removes unused Android, .NET, and GHC toolchains from its
ephemeral hosted runner before compiling. The wheel-only BuildKit export also
avoids duplicating the build toolchains and intermediate objects in the
runner's Docker image store. Together these keep compilation and artifact
export within the hosted runners' root filesystems.

Unchanged build layers use `ghcr.io/marin-community/vllm-build-cache:<architecture>`
with Marin/Iris's registry-cache settings. Compiler steps that must execute use
sccache 0.8.1's native GCS backend at
`gs://marin-public/build-cache/vllm/sccache-v1/<architecture>/`.
The existing `github-iris` workload identity is mounted as a BuildKit secret;
credentials are excluded from the build context and exported layers.
Layer hits skip compiler-cache requests. Missing credentials or unavailable
caches fall back to compilation. GCS expiry is handled separately by the cache
retention workflow. [Design and measurements](https://echo.oa.dev/wiki/613).

## GPU build, qualification, and publication

Review and land vLLM source changes on `main` before building wheels. This
applies to upstream refreshes and patches authored in the fork. Marin's
[GPU guide](https://github.com/marin-community/marin/blob/main/.agents/skills/refresh-fork/docs/vllm.md#triggers-and-tests)
defines the automatic checks, agent-run model tests, and adoption steps.

[`marin-gpu-candidate.yaml`](../../.github/workflows/marin-gpu-candidate.yaml)
builds, tests, and publishes GPU wheels in one workflow. Source changes on
`main` start it automatically. `paths-ignore` lists excluded documentation,
release-automation, and cache-maintenance files. A manual dispatch must also
select `main`:

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
upstream base, build settings, builder image, wheel contents, and SHA-256. Schema 2
uses the final release tag throughout, including each qualification record's
`release_tag` and wheel URL.

Qualification installs the x86_64 wheel on one H100 in `cw-rno2a` and the
aarch64 wheel on one GB200 in `cw-us-east-08a`. Both jobs check the
`vllm._C_stable_libtorch` extension and GrugMoE model implementation, allocate
through cuMem, and serve `Qwen/Qwen3-0.6B` against the hardware's smoke-test
spec. The H100 job also checks sparse NCCL weight transfer and runs the fork's
model, scheduler, and weight-transfer regression tests.

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
