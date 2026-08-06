# Debugging log for GPU nightly validation

Make the accelerator qualification report the real Iris job result and build the
bundled checkout without relying on a parseable release tag.

## Initial status

GitHub Actions run 31064299079 reported success, but its Iris log ended in
`JOB_STATE_FAILED`. No tests from the refreshed branch ran. The bundled build
received an empty `SETUPTOOLS_SCM_PRETEND_VERSION`, then failed because Iris omits
`.git` from the bundle.

## Hypothesis 1

Two independent shell behaviors mask the failure: setuptools-scm rejects the
fork's `marin-vllm-gpu-*` release tag inside command substitution, while the
surrounding `echo` still succeeds; later, `tee` replaces the failing Iris exit
status because the workflow shell does not enable `pipefail`.

## Changes to make

- Derive a PEP 440-safe disposable test-build version from the checked-out SHA.
- Enable `pipefail` before piping Iris logs through `tee`.
- Add workflow regression tests for both contracts.

## Results

The two workflow regression tests pass, and the diff-scoped Marin lint passes.
The fail-closed H100 rerun built the refreshed fork and ran the suite. It
reported 156 passed, 2 failed, and 2 deselected. Both failures were the two
parameters of `test_grammar_compile_error_finishes_only_request`: the future
contained `NameError("name 'logger' is not defined")` instead of the injected
`RuntimeError`.

## Hypothesis 2

The stepping point contains a short-lived upstream regression rather than a
Marin overlay conflict. Upstream commit `0416dab27` removed the structured-output
logger while leaving a new `logger.exception` call from `12213c679`. Upstream
commit `b91a40e72` restores the logger, but it follows the Torch 2.13 bump and is
therefore absent from this branch.

## Changes to make

Backport the exact four-line logger initialization from upstream `b91a40e72`
without taking the intervening Torch 2.13 dependency change.

## Results

The module compiles and imports with an initialized logger. The focused CPU
pytest invocation did not reach its assertions in the sandbox: scheduler setup
needed Hugging Face network access, and the empty-device build cannot run the
suite's accelerator cleanup.

The next fail-closed H100 run passed: 158 tests passed and the two documented
multi-GPU cases were deselected. The Grug suite exercised its CUDA fused-MoE
path. Qwen3-0.6B selected FlashAttention 3, answered all four prompts, and
decoded at 555.3 output tokens/second against the 306.1 floor.

The generalized runner's four configuration tests pass locally. Of the 23
newly selected focused test outcomes, 18 passed in the empty-device sandbox;
the other five were environment failures rather than assertion failures: one
test needed Hugging Face network access, three hit empty-accelerator cleanup,
and the local sandbox denied a ZeroMQ IPC bind. The accelerator jobs remain the
authoritative gate for these targets.

## Hypothesis 3

The first GB200 run built the aarch64 package and reported 177 passed, 1
failed, and 2 deselected. The only failure was the Ray rank-remap test:
`RayWorkerWrapper` was `None` because Ray is an optional dependency and the
lightweight runner had not installed it. The serving/backend gate did not run.

Install `ray[cgraph,default]==2.48.0`, matching the repository's CUDA test
requirements, and rerun the same GB200 lane. This preserves the rank-remap test
instead of weakening or deselecting it.

The rerun passed 178 tests with 2 documented single-GPU deselections, then
served Qwen3-0.6B through FlashAttention 4 on GB200 and passed the correctness
gate. Iris did not drain the final prompt metrics, so this run does not provide
a trustworthy GB200 throughput measurement.

## Hypothesis 4

H100/FlashInfer also passed 178 tests with 2 deselections and selected the
requested FlashInfer backend. FlashInfer then chose its TRTLLM XQA decode path,
which tried to JIT an extension and failed because the slim Iris image has no
discoverable `nvcc` or CUDA development tree.

The repository documents `--attention-config.use_trtllm_attention=0` as the
supported native-FlashInfer selector. Rerun with native FlashInfer to qualify
that backend without hiding the separate XQA/CUDA-toolchain gap.

The native-FlashInfer rerun passed the same 178 tests and selected native
FlashInfer, then its batch-prefill module also attempted runtime JIT compilation
and failed on the missing CUDA development tree. The experimental selector was
removed from the final branch: neither native FlashInfer nor XQA can be
meaningfully qualified in this slim image. A validation image with a coherent
CUDA toolkit and discoverable `CUDA_HOME`/`nvcc` is required.

## Final exact-head qualification

[H100/FlashAttention run 31067421923](https://github.com/marin-community/vllm/actions/runs/31067421923)
qualified commit `566686f02`: 178 tests passed with the same two documented
single-GPU deselections. The server selected `FLASH_ATTN` and FlashAttention 3,
then produced four Qwen3-0.6B completions with at least 65 tokens at 606.2 output
tokens/second. The H100 gate and Iris job both passed.

The server emitted an `EngineDeadError` while the harness shut it down after the
four completed requests. The result and passing gate were emitted afterward,
and the job returned `JOB_STATE_SUCCEEDED`; this is shutdown noise rather than a
failed request or engine startup. The fixed Iris job name causes its artifact to
include older attempts, so the package version `v0.0.dev0+g566686f02` identifies
the exact-head segment.

## Future work

- [x] Run the generalized non-publishing lane on GB200/FlashAttention.
- [x] Rerun H100/FlashAttention against the final executable branch shape.
- [ ] Qualify H100 FlashInfer native attention and XQA in a coherent CUDA JIT
  image.
- [ ] Serve and generate from a real Grug checkpoint on H100 and GB200; the
  registry's current Grug fixture is not publicly downloadable.
- [ ] Qualify real multi-GPU PP/DP/EP and SkyRL checkpoint/weight-sync
  lifecycle behavior outside this single-accelerator lane.
