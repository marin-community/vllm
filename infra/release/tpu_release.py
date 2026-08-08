#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build and verify one immutable Marin vLLM/tpu-inference wheel pair."""

from __future__ import annotations

import argparse
import copy
import re
import sys
import zipfile
from pathlib import Path
from typing import Any

try:
    from infra.release.gpu_release import (
        RELEASE_REPOSITORY,
        ReleaseError,
        _wheel_document,
        load_json,
        normalized_distribution_name,
        release_asset_url,
        requirement_name,
        sha256_file,
        write_json,
    )
except ModuleNotFoundError:  # Direct script execution sets sys.path to this directory.
    from gpu_release import (  # type: ignore[no-redef]
        RELEASE_REPOSITORY,
        ReleaseError,
        _wheel_document,
        load_json,
        normalized_distribution_name,
        release_asset_url,
        requirement_name,
        sha256_file,
        write_json,
    )

CANDIDATE_TAG_PREFIX = "marin-vllm-tpu-candidate-"
RELEASE_TAG_PREFIX = "marin-vllm-tpu-"
REQUIRED_DISTRIBUTIONS = ("vllm", "tpu-inference")
COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
COMPATIBILITY_KEYS = (
    "python_version",
    "platform",
    "vllm_target_device",
    "exclude_newer",
    "runtime_requirements",
)


def _source(config: dict[str, Any], distribution: str) -> dict[str, str]:
    try:
        source = config["packages"][distribution]
    except KeyError as exc:
        raise ReleaseError(f"config has no {distribution} source") from exc
    if COMMIT_PATTERN.fullmatch(source.get("source_commit", "")) is None:
        raise ReleaseError(f"{distribution} source commit is not a full Git commit")
    return source


def candidate_tag(config: dict[str, Any], workflow_commit: str) -> str:
    """Address a candidate by both package sources and its workflow revision."""
    if COMMIT_PATTERN.fullmatch(workflow_commit) is None:
        raise ReleaseError("workflow commit is not a full Git commit")
    vllm = _source(config, "vllm")["source_commit"][:12]
    tpu = _source(config, "tpu-inference")["source_commit"][:12]
    return f"{CANDIDATE_TAG_PREFIX}{vllm}-{tpu}-{workflow_commit[:12]}"


def release_tag(candidate: dict[str, Any]) -> str:
    """Return the immutable promotion tag for a candidate."""
    created = candidate["release"]["created_at"]
    date = created[:10].replace("-", "")
    source = candidate["source"]
    workflow = candidate["workflow"]["commit"]
    return (
        f"{RELEASE_TAG_PREFIX}{date}-"
        f"{source['vllm']['commit'][:12]}-"
        f"{source['tpu-inference']['commit'][:12]}-{workflow[:12]}"
    )


def inspect_wheel(wheel: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Read the public identity and dependency contract from one wheel."""
    document = _wheel_document(wheel)
    metadata = document.metadata
    distribution = normalized_distribution_name(metadata.get("Name", ""))
    if distribution not in REQUIRED_DISTRIBUTIONS:
        raise ReleaseError(f"unexpected wheel distribution {distribution!r}")
    requirements = metadata.get_all("Requires-Dist", [])
    if distribution == "vllm" and any(
        requirement_name(requirement) == "tpu-inference" for requirement in requirements
    ):
        raise ReleaseError("vllm wheel must not hide the paired tpu-inference source")
    if distribution == "tpu-inference":
        normalized = {
            requirement.replace(" ", "").lower() for requirement in requirements
        }
        for name, version in config["runtime_requirements"].items():
            if f"{name}=={version}" not in normalized:
                raise ReleaseError(
                    f"tpu-inference wheel does not require {name}=={version}"
                )
    version = metadata.get("Version", "")
    if not version:
        raise ReleaseError(f"{distribution} wheel has no version")
    return {
        "distribution": distribution,
        "version": version,
        "requires_python": metadata.get("Requires-Python", ""),
        "requirements": requirements,
        "wheel": {
            "filename": wheel.name,
            "sha256": sha256_file(wheel),
            "size_bytes": wheel.stat().st_size,
            "tags": document.tags,
        },
    }


def assemble_candidate(
    wheels: list[Path],
    *,
    config: dict[str, Any],
    repository: str,
    workflow_commit: str,
    workflow_run_url: str,
    created_at: str,
) -> dict[str, Any]:
    """Bind two built wheels to their source and automation revisions."""
    packages = [inspect_wheel(wheel, config) for wheel in wheels]
    by_distribution = {package["distribution"]: package for package in packages}
    if len(by_distribution) != len(packages) or set(by_distribution) != set(
        REQUIRED_DISTRIBUTIONS
    ):
        raise ReleaseError(
            "candidate must contain one vllm and one tpu-inference wheel"
        )
    tag = candidate_tag(config, workflow_commit)
    for distribution, package in by_distribution.items():
        source = _source(config, distribution)
        package["repository"] = source["repository"]
        package["source_commit"] = source["source_commit"]
        wheel = package["wheel"]
        wheel["url"] = release_asset_url(repository, tag, wheel["filename"])
    return {
        "schema_version": config["schema_version"],
        "release": {
            "brand": config["brand"],
            "repository": repository,
            "tag": tag,
            "status": "candidate",
            "created_at": created_at,
        },
        "workflow": {
            "commit": workflow_commit,
            "run_url": workflow_run_url,
        },
        "source": {
            distribution: {
                "repository": by_distribution[distribution]["repository"],
                "commit": by_distribution[distribution]["source_commit"],
            }
            for distribution in REQUIRED_DISTRIBUTIONS
        },
        "compatibility": {key: config[key] for key in COMPATIBILITY_KEYS},
        "packages": [by_distribution[name] for name in REQUIRED_DISTRIBUTIONS],
    }


def _validate_common(manifest: dict[str, Any], config: dict[str, Any]) -> None:
    release = manifest["release"]
    if manifest["schema_version"] != config["schema_version"]:
        raise ReleaseError("release schema version changed")
    if release["brand"] != config["brand"]:
        raise ReleaseError("release brand changed")
    if release["repository"] != RELEASE_REPOSITORY:
        raise ReleaseError("release repository changed")
    workflow = manifest["workflow"]
    if COMMIT_PATTERN.fullmatch(workflow.get("commit", "")) is None:
        raise ReleaseError("workflow commit is not a full Git commit")
    expected_run_prefix = f"https://github.com/{RELEASE_REPOSITORY}/actions/runs/"
    if not workflow.get("run_url", "").startswith(expected_run_prefix):
        raise ReleaseError("workflow run URL changed")
    expected_compatibility = {key: config[key] for key in COMPATIBILITY_KEYS}
    if manifest["compatibility"] != expected_compatibility:
        raise ReleaseError("compatibility contract changed")
    packages = manifest["packages"]
    by_distribution = {package["distribution"]: package for package in packages}
    if len(packages) != len(by_distribution) or set(by_distribution) != set(
        REQUIRED_DISTRIBUTIONS
    ):
        raise ReleaseError("manifest does not contain the exact package pair")
    for distribution in REQUIRED_DISTRIBUTIONS:
        source = _source(config, distribution)
        package = by_distribution[distribution]
        if package["repository"] != source["repository"]:
            raise ReleaseError(f"{distribution} repository changed")
        if package["source_commit"] != source["source_commit"]:
            raise ReleaseError(f"{distribution} source commit changed")
        if manifest["source"][distribution] != {
            "repository": source["repository"],
            "commit": source["source_commit"],
        }:
            raise ReleaseError(f"{distribution} source record changed")
        if not package.get("version"):
            raise ReleaseError(f"{distribution} version is missing")
        wheel = package["wheel"]
        if SHA256_PATTERN.fullmatch(wheel.get("sha256", "")) is None:
            raise ReleaseError(f"{distribution} wheel SHA-256 is malformed")
        expected_url = release_asset_url(
            RELEASE_REPOSITORY, release["tag"], wheel["filename"]
        )
        if wheel.get("url") != expected_url:
            raise ReleaseError(f"{distribution} wheel URL is not tag-addressed")


def validate_candidate(manifest: dict[str, Any], config: dict[str, Any]) -> None:
    """Fail closed unless this is the configured immutable candidate."""
    _validate_common(manifest, config)
    if manifest["release"]["status"] != "candidate":
        raise ReleaseError("manifest is not a candidate")
    expected = candidate_tag(config, manifest["workflow"]["commit"])
    if manifest["release"]["tag"] != expected:
        raise ReleaseError(
            "candidate tag does not address its exact sources and workflow"
        )


def verify_assets(manifest: dict[str, Any], directory: Path) -> None:
    """Verify every wheel named by the manifest."""
    for package in manifest["packages"]:
        wheel = package["wheel"]
        path = directory / wheel["filename"]
        if not path.is_file():
            raise ReleaseError(f"release asset is missing: {path}")
        if sha256_file(path) != wheel["sha256"]:
            raise ReleaseError(f"SHA-256 mismatch for {path.name}")


def finalize_release(
    candidate: dict[str, Any],
    *,
    config: dict[str, Any],
    tag: str,
    published_at: str,
) -> dict[str, Any]:
    """Promote a candidate by relabeling its exact bytes."""
    validate_candidate(candidate, config)
    expected_tag = release_tag(candidate)
    if tag != expected_tag:
        raise ReleaseError(f"release tag must be {expected_tag}")
    manifest = copy.deepcopy(candidate)
    candidate_tag_value = manifest["release"]["tag"]
    manifest["release"].update(
        {
            "tag": tag,
            "status": "released",
            "candidate_tag": candidate_tag_value,
            "published_at": published_at,
        }
    )
    for package in manifest["packages"]:
        wheel = package["wheel"]
        wheel["url"] = release_asset_url(RELEASE_REPOSITORY, tag, wheel["filename"])
    return manifest


def validate_release(manifest: dict[str, Any], config: dict[str, Any]) -> None:
    """Verify a promoted manifest without trusting mutable release state."""
    _validate_common(manifest, config)
    release = manifest["release"]
    if release["status"] != "released":
        raise ReleaseError("manifest is not a release")
    candidate = copy.deepcopy(manifest)
    candidate["release"]["tag"] = release["candidate_tag"]
    candidate["release"]["status"] = "candidate"
    candidate["release"].pop("candidate_tag")
    candidate["release"].pop("published_at")
    for package in candidate["packages"]:
        wheel = package["wheel"]
        wheel["url"] = release_asset_url(
            RELEASE_REPOSITORY, candidate["release"]["tag"], wheel["filename"]
        )
    validate_candidate(candidate, config)
    if release["tag"] != release_tag(candidate):
        raise ReleaseError("release tag does not address its candidate")


def release_notes(manifest: dict[str, Any]) -> str:
    runtime = manifest["compatibility"]["runtime_requirements"]
    lines = [
        f"# {manifest['release']['brand']} {manifest['release']['status']}",
        "",
        "Immutable Python 3.12 TPU wheel pair:",
        "",
    ]
    for package in manifest["packages"]:
        lines.append(
            f"- `{package['distribution']}=={package['version']}` from "
            f"`{package['source_commit']}` (`{package['wheel']['sha256']}`)"
        )
    lines.extend(
        [
            "",
            f"Workflow revision: `{manifest['workflow']['commit']}`.",
            (
                f"Runtime: JAX {runtime['jax']}, JAXlib {runtime['jaxlib']}, "
                f"libtpu {runtime['libtpu']}."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    tag = commands.add_parser("candidate-tag")
    tag.add_argument("--config", type=Path, required=True)
    tag.add_argument("--workflow-commit", required=True)

    assemble = commands.add_parser("assemble-candidate")
    assemble.add_argument("--config", type=Path, required=True)
    assemble.add_argument("--wheel", action="append", type=Path, required=True)
    assemble.add_argument("--repository", required=True)
    assemble.add_argument("--workflow-commit", required=True)
    assemble.add_argument("--workflow-run-url", required=True)
    assemble.add_argument("--created-at", required=True)
    assemble.add_argument("--output", type=Path, required=True)

    for name in ("verify-candidate", "verify-release"):
        verify = commands.add_parser(name)
        verify.add_argument("--config", type=Path, required=True)
        verify.add_argument("--manifest", type=Path, required=True)
        verify.add_argument("--directory", type=Path, required=True)

    finalize = commands.add_parser("finalize-release")
    finalize.add_argument("--config", type=Path, required=True)
    finalize.add_argument("--candidate", type=Path, required=True)
    finalize.add_argument("--release-tag", required=True)
    finalize.add_argument("--published-at", required=True)
    finalize.add_argument("--output", type=Path, required=True)

    promoted_tag = commands.add_parser("release-tag")
    promoted_tag.add_argument("--candidate", type=Path, required=True)

    notes = commands.add_parser("release-notes")
    notes.add_argument("--manifest", type=Path, required=True)
    notes.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        if args.command == "candidate-tag":
            print(candidate_tag(load_json(args.config), args.workflow_commit))
        elif args.command == "assemble-candidate":
            write_json(
                args.output,
                assemble_candidate(
                    args.wheel,
                    config=load_json(args.config),
                    repository=args.repository,
                    workflow_commit=args.workflow_commit,
                    workflow_run_url=args.workflow_run_url,
                    created_at=args.created_at,
                ),
            )
        elif args.command == "verify-candidate":
            manifest = load_json(args.manifest)
            validate_candidate(manifest, load_json(args.config))
            verify_assets(manifest, args.directory)
        elif args.command == "verify-release":
            manifest = load_json(args.manifest)
            validate_release(manifest, load_json(args.config))
            verify_assets(manifest, args.directory)
        elif args.command == "finalize-release":
            write_json(
                args.output,
                finalize_release(
                    load_json(args.candidate),
                    config=load_json(args.config),
                    tag=args.release_tag,
                    published_at=args.published_at,
                ),
            )
        elif args.command == "release-tag":
            print(release_tag(load_json(args.candidate)))
        elif args.command == "release-notes":
            args.output.write_text(release_notes(load_json(args.manifest)))
        return 0
    except (KeyError, OSError, ReleaseError, zipfile.BadZipFile) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
