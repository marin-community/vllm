# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Record the binary base of a pure-Python wheel overlay in a release fragment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def annotate_fragment(
    fragment_path: Path, *, base_release: str, base_commit: str
) -> None:
    fragment = json.loads(fragment_path.read_text())
    provenance = fragment["build"]["provenance"]
    additions = {
        "binary_base_release": base_release,
        "binary_base_commit": base_commit,
    }
    conflicts = {
        key: provenance[key]
        for key, value in additions.items()
        if key in provenance and provenance[key] != value
    }
    if conflicts:
        raise ValueError(
            f"overlay provenance conflicts with existing values: {conflicts}"
        )
    provenance.update(additions)
    fragment_path.write_text(json.dumps(fragment, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fragment", type=Path, required=True)
    parser.add_argument("--base-release", required=True)
    parser.add_argument("--base-commit", required=True)
    args = parser.parse_args()
    annotate_fragment(
        args.fragment,
        base_release=args.base_release,
        base_commit=args.base_commit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
