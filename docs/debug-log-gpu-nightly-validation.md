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

## Future work

- [ ] Run the generalized non-publishing lane on GB200/FlashAttention and
  H100/FlashInfer.
- [ ] Qualify real multi-GPU PP/DP/EP and SkyRL checkpoint/weight-sync
  lifecycle behavior outside this single-accelerator lane.
