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
The H100 rerun is pending.

## Future work

- [ ] Generalize the non-publishing validation lane for GB200 and explicit
  attention backends after H100 is genuinely green.
