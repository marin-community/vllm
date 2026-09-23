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

Each candidate job removes unused Android, .NET, and GHC toolchains from its
ephemeral hosted runner before compiling. The wheel-only BuildKit export also
avoids duplicating the build toolchains and intermediate objects in the
runner's Docker image store. Together these keep compilation and artifact
export within the hosted runners' root filesystems.

## GPU candidate publication

[`marin-gpu-candidate.yaml`](../../.github/workflows/marin-gpu-candidate.yaml)
runs on every merge to `main`. It builds x86_64 SM90 and aarch64 SM100 wheels,
derives the manylinux tag from each wheel's ELF symbols, and publishes a
prerelease named `marin-vllm-gpu-candidate-<12-character-sha>`.

The candidate manifest records:

- fork commit and upstream merge base;
- Python, Torch, CUDA, platform, and SM targets;
- builder image and GitHub Actions provenance;
- wheel filename, tags, size, and SHA-256;
- packaged `_C`, cuMem allocator, and Grug model state.

Candidate tags and assets are immutable. A rerun verifies an existing
candidate instead of replacing it.

GPU publication runs must use the repository's default branch, which this fork
expects to be `main`. A candidate from a prior `main` commit remains valid after
`main` advances. To build immediately after a merge, dispatch from `main`:

```bash
gh workflow run marin-gpu-candidate.yaml \
  --repo marin-community/vllm \
  --ref main \
  -f lane=gpu \
  -f gpu_mode=publish
```

A source refresh uses the separate `main-next` staging lane. `gpu_mode=stage`
requires the workflow source to equal the current remote `main-next` tip and
publishes under `marin-vllm-gpu-staged-candidate-<12-character-sha>`. Staged
candidates never match scheduled candidate selection. The manifest binds the
source and workflow commit, workflow ref and run, and both wheel digests.
Arbitrary branches cannot publish either candidate kind.

The `qualify-x86_64` and `qualify-aarch64` modes remain useful for branch-only
dependency checks. They build one short-lived Actions artifact and publish no
GitHub release. They do not replace the two-wheel staged candidate used by a
source refresh.

## GPU validation and release

[`marin-gpu-release.yaml`](../../.github/workflows/marin-gpu-release.yaml) runs
on a schedule and through `workflow_dispatch`. The optional `candidate_tag`
input selects an exact published candidate; an empty input selects the newest
published candidate by GitHub's `published_at` timestamp across release-list
pages. Drafts are ineligible. Invalid provenance, ABI, or assets fail the run;
it does not try an older candidate.

The workflow qualifies both wheels on their configured hardware:

- H100x1 on `cw-rno2a` installs the x86_64 wheel, checks `_C` and
  `GrugMoeForCausalLM`, validates the sparse NCCL trainer and worker contract,
  allocates through cuMem, runs the Marin delta tests, and serves
  Qwen/Qwen3-0.6B against the H100 spec.
- GB200x1 on `cw-us-east-08a` installs the aarch64 wheel, checks `_C` and
  `GrugMoeForCausalLM`, allocates through cuMem, and serves Qwen/Qwen3-0.6B
  against the GB200 spec.

An absent cuMem extension is recorded as `absent` and fails promotion. Iris
setup failures and missing validation output also become explicit failed JSON
records.

The runtime probe and serving process run with the temporary venv outside the
checkout. The workflow extracts the candidate commit's tests and serving smoke
into a separate validation-source tree, while the release harness comes from
the workflow commit. Each validation runner imports and verifies `vllm` from
the venv before adding that tree to `sys.path` for the `tests` package. It keeps
its working directory outside the tree as well, so model-inspection
subprocesses also import the wheel instead of an unbuilt source package.

Both GPU results must pass before the workflow creates
`marin-vllm-gpu-<UTC-date>-<12-character-sha>`. The final release contains the
unchanged candidate wheels, their validation records, and a final manifest that
binds every result to a wheel digest. The workflow never overwrites an existing
release tag or asset.

For a staged refresh, dispatch the release workflow from trusted `main` with
`qualification_only=true`. It accepts only the exact current `main-next` source,
runs the normal H100 and GB200 gates, and retains their artifacts without
publishing a final release. After an administrator promotes that exact source to
`main`, dispatch again with the successful qualification run ID. The promotion
rechecks the candidate assets, source lineage, workflow run, validation records,
and wheel digests, then publishes the same bytes without another GPU allocation.
Complete promotion within the validation artifacts' 14-day retention window;
after expiry, the exact qualification records cannot be reused.

```bash
gh workflow run marin-gpu-candidate.yaml \
  --repo marin-community/vllm \
  --ref main-next \
  -f lane=gpu \
  -f gpu_mode=stage

gh workflow run marin-gpu-release.yaml \
  --repo marin-community/vllm \
  --ref main \
  -f lane=gpu \
  -f candidate_tag=marin-vllm-gpu-staged-candidate-0123456789ab \
  -f qualification_only=true

# After the lease-checked main-next -> main promotion:
gh workflow run marin-gpu-release.yaml \
  --repo marin-community/vllm \
  --ref main \
  -f lane=gpu \
  -f candidate_tag=marin-vllm-gpu-staged-candidate-0123456789ab \
  -f qualification_run_id=<successful-qualification-run-id>
```

Dispatch a specific candidate with:

```bash
gh workflow run marin-gpu-release.yaml \
  --repo marin-community/vllm \
  --ref main \
  -f lane=gpu \
  -f candidate_tag=marin-vllm-gpu-candidate-0123456789ab
```

Check the published tag and final assets independently. [GitHub runs](https://docs.github.com/en/actions/concepts/workflows-and-actions/workflows)
the workflow file on the selected ref, so an older branch can still run its old
publication logic. Enforcing this against repository writers requires a publisher
outside branch-controlled workflow code.

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
not rebuild a GPU wheel, and promoting a GPU candidate does not rebuild the TPU
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
