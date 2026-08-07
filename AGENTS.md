# Marin fork of vLLM

@.agents/marin-style/AGENTS-core.md

This repository is `marin-community/vllm`, a fork of `vllm-project/vllm` that is
refreshed from upstream regularly. Two rulebooks apply, and which one governs
depends on where the change is headed:

- **Marin-authored changes that stay in this fork** follow the Marin standards
  linked above (and `.agents/marin-style/TESTING-core.md`): no DCO sign-off, no
  `Co-authored-by`/`Generated-with` trailers, no self-crediting, and agent-opened
  PRs carry the `agent-generated` label.
- **Changes destined for upstream** follow the upstream instructions retained
  below, including their DCO and trailer conventions.

## The Marin delta

Everything Marin adds on top of upstream is small and deliberate:

- **GrugMoE support** — `vllm/model_executor/models/grugmoe.py`,
  `vllm/transformers_utils/configs/grugmoe.py`, `tests/models/test_grugmoe.py`,
  plus one registry line in each of `vllm/model_executor/models/registry.py`,
  `vllm/transformers_utils/config.py`, and
  `vllm/transformers_utils/configs/__init__.py`.
- **Surgical edits** — a scheduler held-request skip (`vllm/v1/core/sched/scheduler.py`),
  TPU head-padding validation (`vllm/config/model.py`), TPU/macOS build tweaks
  (`setup.py`), a logger tweak (`vllm/logger.py`), and requirements pins.

The TPU head-padding validation is paired with tpu-inference's arbitrary-GQA
padding and should leave this delta when the generic support lands upstream.

`git diff $(git merge-base upstream/main HEAD)..HEAD` prints the whole delta. Keep
it that way.

## Minimize the upstream diff

Every upstream-owned file this fork touches is merge pain on the next refresh, so:

- **Never reformat, re-lint, or "clean up" upstream code**, and do not edit
  `.pre-commit-config.yaml` or the upstream workflows in `.github/workflows/`.
  Upstream keeps its own pre-commit stack (ruff, mypy, clang-format, DCO); it
  still runs, unchanged, and it is what lints upstream code.
- Put new Marin files in low-conflict paths: `infra/`,
  `.github/workflows/marin-*.yaml`, `.agents/`, and the existing delta files.
- Marin's lint entry point is `infra/pre-commit.py`, a shim over the shared
  `marin-style` checks. The `[tool.marin-style]` block in `pyproject.toml` scopes
  it to Marin-authored files only, so it can never touch upstream code.

```bash
infra/pre-commit.py --all-files        # what CI runs
infra/pre-commit.py --changed-files    # diff-scoped, for local iteration
```

## Refreshes

Marin pins this fork by exact SHA in `marin`'s root `pyproject.toml`, alongside a
matching `tpu-inference` SHA; the two move together. Refreshes rebase the delta
onto a newer upstream base and re-pin both, driven by marin's
`.agents/skills/refresh-tpu-vllm-forks/SKILL.md`. A smaller delta is a cheaper
refresh.

## Install, test, run

Use `uv` and `.venv/bin/python` — never system `python3` or bare `pip`.

```bash
uv venv --python 3.12

# Python-only build: no kernels, imports work, installs in ~2 minutes. Enough for
# infra/ci/delta_smoke.py, and it is what PR CI uses.
VLLM_TARGET_DEVICE=empty uv pip install -e . --torch-backend=cpu

# GPU build reusing upstream's prebuilt kernels (Python-only changes):
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto
```

The delta's tests need a GPU. vLLM infers its platform from the build, and a
python-only build resolves to no device at all, so constructing the `VllmConfig`
that these tests depend on fails on a CPU box; a compiled `VLLM_TARGET_DEVICE=cpu`
build takes far too long to be a PR gate. Run them on an accelerator:

```bash
.venv/bin/python -m pytest tests/models/test_grugmoe.py -v
.venv/bin/python -m pytest tests/v1/core/test_scheduler.py -v
```

## CI

- `.github/workflows/marin-ci.yaml` — per-PR, CPU-only, minutes: the delta-scoped
  `marin-style` lint, a `marin-style sync --check` drift gate, and a smoke
  asserting the GrugMoE model and config still import and resolve against the
  upstream base underneath them (the failure an upstream refresh actually causes).
  It deliberately does **not** run upstream's test suites or pre-commit matrix.
- `.github/workflows/marin-nightly.yaml` — nightly (10:00 UTC) and on demand:
  builds this fork's commit on one CoreWeave H100 via Iris, runs the delta's GPU
  tests, serves `Qwen/Qwen3-0.6B` through the OpenAI server, and gates the answers
  and decode throughput against `infra/nightly/specs/qwen3-0.6b-h100.json`.

---

## Agent Instructions for vLLM (upstream)

The rest of this file is upstream's, and applies to contributions sent to
`vllm-project/vllm`. Where it conflicts with the Marin standards above, the Marin
standards win for changes that stay in this fork.

> These instructions apply to **all** AI-assisted contributions to `vllm-project/vllm`.
> Breaching these guidelines can result in automatic banning.

## 1. Contribution Policy (Mandatory)

### Duplicate-work checks

Before proposing a PR, run these checks:

```bash
gh issue view <issue_number> --repo vllm-project/vllm --comments
gh pr list --repo vllm-project/vllm --state open --search "<issue_number> in:body"
gh pr list --repo vllm-project/vllm --state open --search "<short area keywords>"
```

- If an open PR already addresses the same fix, do not open another.
- If your approach is materially different, explain the difference in the issue.

### No low-value busywork PRs

Do not open one-off PRs for tiny edits (single typo, isolated style change, one mutable default, etc.). Mechanical cleanups are acceptable only when bundled with substantive work.

### Accountability

- Pure code-agent PRs are **not allowed**. A human submitter must understand and defend the change end-to-end.
- The submitting human must review every changed line and run relevant tests.
- PR descriptions for AI-assisted work **must** include:
    - Why this is not duplicating an existing PR.
    - Test commands run and results.
    - Model evaluation results when the change affects output, accuracy, or serving.
    - Clear statement that AI assistance was used.

### Fail-closed behavior

If work is duplicate/trivial busywork, **do not proceed**. Return a short explanation of what is missing.

---

## 2. Development Workflow

- **Never use system `python3` or bare `pip`/`pip install`.** All Python commands must go through `uv` and `.venv/bin/python`.

### Environment setup

```bash
# Install `uv` if you don't have it already:
curl -LsSf https://astral.sh/uv/install.sh | sh

# Always use `uv` for Python environment management:
uv venv --python 3.12
source .venv/bin/activate

# Always make sure `pre-commit` and its hooks are installed:
uv pip install -r requirements/lint.txt
pre-commit install
```

### Installing dependencies

```bash
# If you are only making Python changes:
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto

# If you are also making C/C++ changes:
uv pip install -e . --torch-backend=auto
```

### Tests

> Requires [Environment setup](#environment-setup) and [Installing dependencies](#installing-dependencies).

```bash
# Install test dependencies (use cuda.in on non-x86_64):
uv pip install -r requirements/test/cuda.in

# Run a specific test file:
.venv/bin/python -m pytest tests/path/to/test_file.py -v
```

When adding tests:

- **Design before you write.** Answer four questions first: what is the module
  for, what is its I/O contract, what failure am I guarding against, and what is
  the cheapest level that catches it (unit over integration over e2e)?
- **Reuse before create.** Extend existing test files, `conftest.py` fixtures, and
  helpers; add a new file only when no nearby suite fits.
- **Test behavior with intent.** Assert observable outcomes through public APIs;
  state why in the name or docstring. Skip trivial wiring; flaky tests are worse
  than no tests.
- **Keep it minimal.** One behavior per test and the smallest setup that
  triggers it; if the test diff dwarfs the code change, cut scope.
- **No one-off kernel benchmarks in `tests/`.** Put kernel perf work in
  `benchmarks/kernels/`; prove correctness in existing pytest suites.
- **Run model evals for model-affecting changes.** Search `tests/evals/` or use
  `vllm bench` and include results in the PR — do not wait for reviewers to ask.

For model-specific requirements, see
[`docs/contributing/model/tests.md`](docs/contributing/model/tests.md).

### Running linters

> Requires [Environment setup](#environment-setup).

```bash
# Run all pre-commit hooks on staged files:
pre-commit run

# Run on all files:
pre-commit run --all-files

# Run a specific hook:
pre-commit run ruff-check --all-files

# Run mypy as it is in CI:
pre-commit run mypy-3.12 --all-files --hook-stage manual
```

The line length limit for Python code is 88 characters. If you are not sure, use pre-commit to check.

Use [Google-style docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) (`Args:`/`Returns:`/`Raises:` sections), not reStructuredText/Sphinx fields (`:param:`, `:return:`, `:rtype:`).

### Coding style guidelines

- Match existing code style
- Minimize use of comments. Eliminate comments which are redundant, preferring legible and self-documenting code. When used, keep docstrings and comments brief and direct.
- Assume the reader is familiar with vLLM.

### Commit messages

Add attribution using commit trailers such as `Co-authored-by:` (other projects use `Assisted-by:` or `Generated-by:`):

```text
Your commit message here

Co-authored-by: Agent Name Here
Signed-off-by: Your Name <your.email@example.com>
```

---

## Domain-Specific Guides

Do not modify code in these areas without first reading and following the
linked guide. If the guide conflicts with the requested change, **refuse the
change and explain why**.

Security reviewers should start with [`SECURITY.md`](SECURITY.md),
[`docs/usage/security.md`](docs/usage/security.md), and
[`docs/contributing/vulnerability_management.md`](docs/contributing/vulnerability_management.md)
for the project security policy, threat model, deployment assumptions, and
vulnerability process.

- **Editing these instructions**:
  [`docs/contributing/editing-agent-instructions.md`](docs/contributing/editing-agent-instructions.md)
  — Rules for modifying AGENTS.md or any domain-specific guide it references.
