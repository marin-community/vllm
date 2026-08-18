#!/usr/bin/env -S uv run --script
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     # Pin to an exact marin-style revision so every contributor and CI run
#     # uses the same checks. Bump the hash to adopt a new version.
#     "marin-style @ git+https://github.com/marin-community/marin-style@5094279da60b47b9a8fa8effaf7f73cd13f1e96f",
# ]
# ///
"""Consumer-repo pre-commit shim. Delegates to the shared marin-style checks.

Committed at `infra/pre-commit.py`; the `commit` skill and CI invoke it directly.

This is a fork of vllm-project/vllm: the checks are scoped by `[tool.marin-style]`
in pyproject.toml to Marin-authored files only. Upstream code keeps using
upstream's own pre-commit stack (`.pre-commit-config.yaml`), which this shim does
not replace.
"""

from marin_style.precommit import main

if __name__ == "__main__":
    raise SystemExit(main())
