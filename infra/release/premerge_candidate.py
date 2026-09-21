"""Assemble and verify a branch wheel pair without entering the main release lane."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from infra.release.gpu_release import (
    _validate_manifest_common,
    assemble_candidate,
    release_notes,
    verify_manifest_assets,
)
from infra.release.release_common import ReleaseError, load_json, write_json


def verify(
    manifest: dict, *, config: dict, directory: Path, source_commit: str
) -> None:
    expected_tag = f"marin-vllm-gpu-premerge-{source_commit[:12]}"
    if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise ReleaseError("source commit must be a full lowercase SHA")
    if manifest["source"]["fork_commit"] != source_commit:
        raise ReleaseError("the wheel pair was built from a different source commit")
    if (
        manifest["release"]["tag"] != expected_tag
        or manifest["release"]["status"] != "candidate"
    ):
        raise ReleaseError("the premerge release identity changed")
    if manifest["validation"] != {"status": "pending", "targets": []}:
        raise ReleaseError("an unqualified branch release cannot claim validation")
    _validate_manifest_common(manifest, config)
    verify_manifest_assets(manifest, directory)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--created-at", required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--assemble", action="store_true")
    args = parser.parse_args()

    config = load_json(args.config)
    manifest_path = args.directory / "marin-vllm-gpu-manifest.json"
    if args.assemble:
        fragments = [
            load_json(path)
            for path in sorted(args.directory.glob("marin-vllm-*-fragment.json"))
        ]
        manifest = assemble_candidate(
            fragments,
            config=config,
            repository=args.repository,
            candidate_tag=f"marin-vllm-gpu-premerge-{args.source_commit[:12]}",
            created_at=args.created_at,
        )
        verify(
            manifest,
            config=config,
            directory=args.directory,
            source_commit=args.source_commit,
        )
        write_json(manifest_path, manifest)
        (args.directory / "release-notes.md").write_text(release_notes(manifest))
    else:
        verify(
            load_json(manifest_path),
            config=config,
            directory=args.directory,
            source_commit=args.source_commit,
        )


if __name__ == "__main__":
    main()
