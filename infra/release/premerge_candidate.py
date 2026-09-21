"""Assemble and verify a branch wheel pair without entering the main release lane."""

from __future__ import annotations

import argparse
import re
import zipfile
from email.parser import Parser
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
    expected_tag = f"marin-vllm-gpu-premerge-{source_commit[:12]}-immutable"
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
    required = {
        "apache-tvm-ffi": "0.1.12",
        "tilelang": "0.1.14",
        "tokenspeed-mla": "0.1.9",
    }
    for platform in manifest["platforms"]:
        wheel = directory / platform["wheel"]["filename"]
        with zipfile.ZipFile(wheel) as archive:
            metadata_path = next(
                name
                for name in archive.namelist()
                if name.endswith(".dist-info/METADATA")
            )
            metadata = Parser().parsestr(archive.read(metadata_path).decode())
        requirements = {
            requirement.lower().replace(" ", "")
            for requirement in metadata.get_all("Requires-Dist", [])
        }
        for name, pinned_version in required.items():
            if not any(
                requirement.startswith(f"{name}=={pinned_version}")
                for requirement in requirements
            ):
                raise ReleaseError(
                    f"{platform['architecture']} wheel does not pin "
                    f"{name}=={pinned_version}"
                )


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
            candidate_tag=(
                f"marin-vllm-gpu-premerge-{args.source_commit[:12]}-immutable"
            ),
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
