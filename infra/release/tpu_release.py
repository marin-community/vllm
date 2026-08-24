#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build and verify one hash-pinned Marin vLLM/tpu-inference wheel pair."""

from __future__ import annotations

import argparse
import base64
import contextlib
import copy
import functools
import hashlib
import html
import http.server
import json
import os
import subprocess
import sys
import threading
import zipfile
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

if __package__:
    from .release_common import (
        RELEASE_REPOSITORY,
        ReleaseError,
        load_json,
        normalized_distribution_name,
        release_asset_url,
        sha256_file,
        wheel_metadata,
        write_json,
    )
else:
    from release_common import (  # type: ignore[no-redef]
        RELEASE_REPOSITORY,
        ReleaseError,
        load_json,
        normalized_distribution_name,
        release_asset_url,
        sha256_file,
        wheel_metadata,
        write_json,
    )

CANDIDATE_TAG_PREFIX = "marin-vllm-tpu-candidate-"
RELEASE_TAG_PREFIX = "marin-vllm-tpu-"
VALIDATION_SENTINEL = "MARIN_TPU_VALIDATION_JSON="
REQUIRED_DISTRIBUTIONS = ("vllm", "tpu-inference")
INDEX_PREFIX = "marin-vllm-tpu-index-"
HEX_DIGITS = frozenset("0123456789abcdef")


class _QuietIndexHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        pass


@contextlib.contextmanager
def local_flat_index(index: Path) -> Iterator[str]:
    """Expose one verified HTML index to uv over loopback."""
    if not index.is_file():
        raise ReleaseError(f"flat index does not exist: {index}")
    handler = functools.partial(_QuietIndexHandler, directory=str(index.parent))
    with http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}/{quote(index.name)}"
        finally:
            server.shutdown()
            thread.join()


def _is_lower_hex(value: str, length: int) -> bool:
    return len(value) == length and set(value) <= HEX_DIGITS


def _validate_exclude_newer(value: Any) -> str:
    if not isinstance(value, str):
        raise ReleaseError("exclude_newer must be a UTC timestamp")
    try:
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise ReleaseError(
            "exclude_newer must be a whole-second UTC timestamp"
        ) from exc
    return value


def _source(
    config: dict[str, Any], distribution: str, commit: str
) -> dict[str, str]:
    if distribution not in REQUIRED_DISTRIBUTIONS:
        raise ReleaseError(f"unexpected source distribution {distribution!r}")
    if not _is_lower_hex(commit, 40):
        raise ReleaseError(f"{distribution} source commit is not a full Git commit")
    return {
        "repository": config["repositories"][distribution],
        "commit": commit,
    }


def candidate_tag(
    config: dict[str, Any],
    source: dict[str, dict[str, str]],
    workflow_commit: str,
    exclude_newer: str,
    wheel_sha256: dict[str, str],
) -> str:
    """Address a candidate by its sources, producer, contract, and wheel bytes."""
    if not _is_lower_hex(workflow_commit, 40):
        raise ReleaseError("workflow commit is not a full Git commit")
    for distribution in REQUIRED_DISTRIBUTIONS:
        expected = _source(config, distribution, source[distribution]["commit"])
        if source[distribution] != expected:
            raise ReleaseError(f"{distribution} source repository changed")
    if set(wheel_sha256) != set(REQUIRED_DISTRIBUTIONS) or any(
        not _is_lower_hex(digest, 64) for digest in wheel_sha256.values()
    ):
        raise ReleaseError("candidate wheel digest pair is malformed")
    build_contract = {
        "python_version": config["python_version"],
        "exclude_newer": _validate_exclude_newer(exclude_newer),
        "wheel_sha256": wheel_sha256,
    }
    contract_digest = hashlib.sha256(
        json.dumps(build_contract, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:12]
    return (
        f"{CANDIDATE_TAG_PREFIX}{source['vllm']['commit'][:12]}-"
        f"{source['tpu-inference']['commit'][:12]}-{workflow_commit[:12]}-"
        f"{contract_digest}"
    )


def release_tag(candidate: dict[str, Any]) -> str:
    """Return the non-overwriting promotion tag for a candidate."""
    created = candidate["release"]["created_at"]
    date = created[:10].replace("-", "")
    source = candidate["source"]
    workflow = candidate["workflow"]["commit"]
    return (
        f"{RELEASE_TAG_PREFIX}{date}-"
        f"{source['vllm']['commit'][:12]}-"
        f"{source['tpu-inference']['commit'][:12]}-{workflow[:12]}-"
        f"{candidate['release']['tag'][-12:]}"
    )


def expected_package_version(
    config: dict[str, Any], distribution: str, commit: str
) -> str:
    """Return the version the workflow must embed for one exact source."""
    _source(config, distribution, commit)
    package_version = config["package_versions"][distribution]
    return (
        f"{package_version['base']}+marin.{commit[:12]}"
        f"{package_version['suffix']}"
    )


def inspect_wheel(wheel: Path) -> dict[str, Any]:
    """Read the public identity from one wheel."""
    document = wheel_metadata(wheel)
    metadata = document.metadata
    distribution = normalized_distribution_name(metadata.get("Name", ""))
    if distribution not in REQUIRED_DISTRIBUTIONS:
        raise ReleaseError(f"unexpected wheel distribution {distribution!r}")
    version = metadata.get("Version", "")
    if not version:
        raise ReleaseError(f"{distribution} wheel has no version")
    return {
        "distribution": distribution,
        "version": version,
        "wheel": {
            "filename": wheel.name,
            "sha256": sha256_file(wheel),
        },
    }


def _index_document(manifest: dict[str, Any]) -> tuple[dict[str, str], bytes]:
    """Render a PEP 503-compatible flat index for the exact wheel pair."""
    links = []
    for package in sorted(
        manifest["packages"], key=lambda item: item["distribution"]
    ):
        wheel = package["wheel"]
        url = html.escape(wheel["url"], quote=True)
        filename = html.escape(wheel["filename"])
        links.append(
            f'<a href="{url}#sha256={wheel["sha256"]}">{filename}</a><br>'
        )
    content = (
        "<!doctype html>\n<html><body>\n"
        + "\n".join(links)
        + "\n</body></html>\n"
    ).encode()
    digest = hashlib.sha256(content).hexdigest()
    filename = f"{INDEX_PREFIX}{digest[:16]}.html"
    release = manifest["release"]
    return (
        {
            "filename": filename,
            "sha256": digest,
            "url": release_asset_url(
                release["repository"], release["tag"], filename
            ),
        },
        content,
    )


def write_index(manifest: dict[str, Any], directory: Path) -> Path:
    """Write the content-addressed flat index declared by a manifest."""
    expected, content = _index_document(manifest)
    if manifest.get("index") != expected:
        raise ReleaseError("flat index record changed")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / expected["filename"]
    path.write_bytes(content)
    return path


def assemble_candidate(
    wheels: list[Path],
    *,
    config: dict[str, Any],
    repository: str,
    vllm_commit: str,
    tpu_inference_commit: str,
    workflow_commit: str,
    workflow_run_url: str,
    created_at: str,
    exclude_newer: str,
) -> dict[str, Any]:
    """Bind two built wheels to their source and automation revisions."""
    packages = [inspect_wheel(wheel) for wheel in wheels]
    by_distribution = {package["distribution"]: package for package in packages}
    if len(by_distribution) != len(packages) or set(by_distribution) != set(
        REQUIRED_DISTRIBUTIONS
    ):
        raise ReleaseError(
            "candidate must contain one vllm and one tpu-inference wheel"
        )
    source = {
        "vllm": _source(config, "vllm", vllm_commit),
        "tpu-inference": _source(
            config, "tpu-inference", tpu_inference_commit
        ),
    }
    for distribution, package in by_distribution.items():
        expected_version = expected_package_version(
            config, distribution, source[distribution]["commit"]
        )
        if package["version"] != expected_version:
            raise ReleaseError(
                f"{distribution} wheel version is {package['version']!r}, "
                f"expected {expected_version!r}"
            )
    wheel_sha256 = {
        distribution: package["wheel"]["sha256"]
        for distribution, package in by_distribution.items()
    }
    tag = candidate_tag(
        config, source, workflow_commit, exclude_newer, wheel_sha256
    )
    for package in by_distribution.values():
        wheel = package["wheel"]
        wheel["url"] = release_asset_url(repository, tag, wheel["filename"])
    manifest = {
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
        "source": source,
        "compatibility": {
            "python_version": config["python_version"],
            "exclude_newer": _validate_exclude_newer(exclude_newer),
        },
        "packages": [by_distribution[name] for name in REQUIRED_DISTRIBUTIONS],
    }
    manifest["index"], _ = _index_document(manifest)
    return manifest


def _validate_common(manifest: dict[str, Any], config: dict[str, Any]) -> None:
    release = manifest["release"]
    if manifest["schema_version"] != config["schema_version"]:
        raise ReleaseError("release schema version changed")
    if release["brand"] != config["brand"]:
        raise ReleaseError("release brand changed")
    if release["repository"] != RELEASE_REPOSITORY:
        raise ReleaseError("release repository changed")
    workflow = manifest["workflow"]
    if not _is_lower_hex(workflow.get("commit", ""), 40):
        raise ReleaseError("workflow commit is not a full Git commit")
    expected_run_prefix = f"https://github.com/{RELEASE_REPOSITORY}/actions/runs/"
    if not workflow.get("run_url", "").startswith(expected_run_prefix):
        raise ReleaseError("workflow run URL changed")
    compatibility = manifest["compatibility"]
    if compatibility.get("python_version") != config["python_version"]:
        raise ReleaseError("compatibility python_version changed")
    _validate_exclude_newer(compatibility.get("exclude_newer"))
    packages = manifest["packages"]
    by_distribution = {package["distribution"]: package for package in packages}
    if len(packages) != len(by_distribution) or set(by_distribution) != set(
        REQUIRED_DISTRIBUTIONS
    ):
        raise ReleaseError("manifest does not contain the exact package pair")
    for distribution in REQUIRED_DISTRIBUTIONS:
        source = _source(
            config, distribution, manifest["source"][distribution]["commit"]
        )
        package = by_distribution[distribution]
        if manifest["source"][distribution] != source:
            raise ReleaseError(f"{distribution} source record changed")
        if not package.get("version"):
            raise ReleaseError(f"{distribution} version is missing")
        expected_version = expected_package_version(
            config, distribution, source["commit"]
        )
        if package["version"] != expected_version:
            raise ReleaseError(
                f"{distribution} wheel version is {package['version']!r}, "
                f"expected {expected_version!r}"
            )
        wheel = package["wheel"]
        if not _is_lower_hex(wheel.get("sha256", ""), 64):
            raise ReleaseError(f"{distribution} wheel SHA-256 is malformed")
        expected_url = release_asset_url(
            RELEASE_REPOSITORY, release["tag"], wheel["filename"]
        )
        if wheel.get("url") != expected_url:
            raise ReleaseError(f"{distribution} wheel URL is not tag-addressed")
    expected_index, _ = _index_document(manifest)
    if manifest.get("index") != expected_index:
        raise ReleaseError("flat index record changed")


def validate_candidate(
    manifest: dict[str, Any],
    config: dict[str, Any],
    *,
    expected_tag: str | None = None,
) -> None:
    """Fail closed unless this is the configured candidate."""
    _validate_common(manifest, config)
    if manifest["release"]["status"] != "candidate":
        raise ReleaseError("manifest is not a candidate")
    expected = candidate_tag(
        config,
        manifest["source"],
        manifest["workflow"]["commit"],
        manifest["compatibility"]["exclude_newer"],
        {
            package["distribution"]: package["wheel"]["sha256"]
            for package in manifest["packages"]
        },
    )
    if manifest["release"]["tag"] != expected:
        raise ReleaseError(
            "candidate tag does not address its exact sources and workflow"
        )
    if expected_tag is not None and manifest["release"]["tag"] != expected_tag:
        raise ReleaseError("candidate tag does not match the requested release")


def verify_assets(manifest: dict[str, Any], directory: Path) -> None:
    """Verify the manifested wheel pair and its content-addressed flat index."""
    expected_wheels = {package["wheel"]["filename"] for package in manifest["packages"]}
    observed_wheels = {path.name for path in directory.glob("*.whl")}
    if observed_wheels != expected_wheels:
        raise ReleaseError(
            "release wheel set changed: "
            f"expected {sorted(expected_wheels)}, found {sorted(observed_wheels)}"
        )
    for package in manifest["packages"]:
        wheel = package["wheel"]
        path = directory / wheel["filename"]
        if not path.is_file():
            raise ReleaseError(f"release asset is missing: {path}")
        if sha256_file(path) != wheel["sha256"]:
            raise ReleaseError(f"SHA-256 mismatch for {path.name}")
    index = manifest["index"]
    observed_indexes = {
        path.name for path in directory.glob(f"{INDEX_PREFIX}*.html")
    }
    if observed_indexes != {index["filename"]}:
        raise ReleaseError(
            "release index set changed: "
            f"expected {[index['filename']]}, found {sorted(observed_indexes)}"
        )
    index_path = directory / index["filename"]
    if sha256_file(index_path) != index["sha256"]:
        raise ReleaseError(f"SHA-256 mismatch for {index_path.name}")


def validate_result(
    result: dict[str, Any],
    candidate: dict[str, Any],
    config: dict[str, Any],
) -> None:
    """Require a passing qualification of this exact public candidate."""
    validate_candidate(candidate, config)
    if result.get("candidate_tag") != candidate["release"]["tag"]:
        raise ReleaseError("qualification candidate tag changed")
    expected_run_prefix = f"https://github.com/{RELEASE_REPOSITORY}/actions/runs/"
    if not result.get("run_url", "").startswith(expected_run_prefix):
        raise ReleaseError("qualification run URL changed")
    expected_hardware = config["validation"]["hardware"]
    if result.get("hardware") != expected_hardware:
        raise ReleaseError("qualification hardware changed")


def extract_validation(log: Path) -> dict[str, Any]:
    """Read the final structured TPU qualification record from an Iris log."""
    records = [
        line.split(VALIDATION_SENTINEL, 1)[1].strip()
        for line in log.read_text().splitlines()
        if VALIDATION_SENTINEL in line
    ]
    if not records:
        raise ReleaseError("Iris log contains no TPU qualification record")
    try:
        value = json.loads(base64.b64decode(records[-1]).decode())
    except (ValueError, json.JSONDecodeError) as exc:
        raise ReleaseError("TPU qualification record is malformed") from exc
    if not isinstance(value, dict):
        raise ReleaseError("TPU qualification record must be an object")
    return value


def finalize_release(
    candidate: dict[str, Any],
    *,
    config: dict[str, Any],
    validation: dict[str, Any],
    tag: str,
    published_at: str,
) -> dict[str, Any]:
    """Promote a candidate by relabeling its exact bytes."""
    validate_candidate(candidate, config)
    validate_result(validation, candidate, config)
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
    manifest["index"], _ = _index_document(manifest)
    manifest["validation"] = validation
    return manifest


def validate_release(manifest: dict[str, Any], config: dict[str, Any]) -> None:
    """Verify a promoted manifest without trusting mutable release state."""
    _validate_common(manifest, config)
    release = manifest["release"]
    if release["status"] != "released":
        raise ReleaseError("manifest is not a release")
    candidate = copy.deepcopy(manifest)
    validation = candidate.pop("validation")
    candidate["release"]["tag"] = release["candidate_tag"]
    candidate["release"]["status"] = "candidate"
    candidate["release"].pop("candidate_tag")
    candidate["release"].pop("published_at")
    for package in candidate["packages"]:
        wheel = package["wheel"]
        wheel["url"] = release_asset_url(
            RELEASE_REPOSITORY, candidate["release"]["tag"], wheel["filename"]
        )
    candidate["index"], _ = _index_document(candidate)
    validate_candidate(candidate, config)
    validate_result(validation, candidate, config)
    if release["tag"] != release_tag(candidate):
        raise ReleaseError("release tag does not address its candidate")


def release_notes(manifest: dict[str, Any]) -> str:
    lines = [
        f"# {manifest['release']['brand']} {manifest['release']['status']}",
        "",
        "Hash-pinned Python 3.12 TPU wheel pair:",
        "",
    ]
    for package in manifest["packages"]:
        source_commit = manifest["source"][package["distribution"]]["commit"]
        lines.append(
            f"- `{package['distribution']}=={package['version']}` from "
            f"`{source_commit}` (`{package['wheel']['sha256']}`)"
        )
    lines.extend(
        [
            "",
            f"Workflow revision: `{manifest['workflow']['commit']}`.",
            f"Flat package index: {manifest['index']['url']}",
            "",
        ]
    )
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    assemble = commands.add_parser("assemble-candidate")
    assemble.add_argument("--config", type=Path, required=True)
    assemble.add_argument("--wheel", action="append", type=Path, required=True)
    assemble.add_argument("--repository", required=True)
    assemble.add_argument("--vllm-commit", required=True)
    assemble.add_argument("--tpu-inference-commit", required=True)
    assemble.add_argument("--workflow-commit", required=True)
    assemble.add_argument("--workflow-run-url", required=True)
    assemble.add_argument("--created-at", required=True)
    assemble.add_argument("--exclude-newer", required=True)
    assemble.add_argument("--output", type=Path, required=True)

    for name in ("verify-candidate", "verify-release"):
        verify = commands.add_parser(name)
        verify.add_argument("--config", type=Path, required=True)
        verify.add_argument("--manifest", type=Path, required=True)
        verify.add_argument("--directory", type=Path, required=True)
        if name == "verify-candidate":
            verify.add_argument("--expected-tag")

    index = commands.add_parser("write-index")
    index.add_argument("--manifest", type=Path, required=True)
    index.add_argument("--directory", type=Path, required=True)

    finalize = commands.add_parser("finalize-release")
    finalize.add_argument("--config", type=Path, required=True)
    finalize.add_argument("--candidate", type=Path, required=True)
    finalize.add_argument("--validation", type=Path, required=True)
    finalize.add_argument("--release-tag", required=True)
    finalize.add_argument("--published-at", required=True)
    finalize.add_argument("--output", type=Path, required=True)

    promoted_tag = commands.add_parser("release-tag")
    promoted_tag.add_argument("--candidate", type=Path, required=True)

    package_version = commands.add_parser("package-version")
    package_version.add_argument("--config", type=Path, required=True)
    package_version.add_argument(
        "--distribution", choices=REQUIRED_DISTRIBUTIONS, required=True
    )
    package_version.add_argument("--source-commit", required=True)

    notes = commands.add_parser("release-notes")
    notes.add_argument("--manifest", type=Path, required=True)
    notes.add_argument("--output", type=Path, required=True)

    extract = commands.add_parser("extract-validation")
    extract.add_argument("--log", type=Path, required=True)
    extract.add_argument("--output", type=Path, required=True)

    result = commands.add_parser("validate-result")
    result.add_argument("--result", type=Path, required=True)
    result.add_argument("--candidate", type=Path, required=True)
    result.add_argument("--config", type=Path, required=True)

    run = commands.add_parser("run-with-index")
    run.add_argument("--index", type=Path, required=True)
    run.add_argument("argv", nargs=argparse.REMAINDER)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        if args.command == "assemble-candidate":
            write_json(
                args.output,
                assemble_candidate(
                    args.wheel,
                    config=load_json(args.config),
                    repository=args.repository,
                    vllm_commit=args.vllm_commit,
                    tpu_inference_commit=args.tpu_inference_commit,
                    workflow_commit=args.workflow_commit,
                    workflow_run_url=args.workflow_run_url,
                    created_at=args.created_at,
                    exclude_newer=args.exclude_newer,
                ),
            )
        elif args.command == "verify-candidate":
            manifest = load_json(args.manifest)
            validate_candidate(
                manifest,
                load_json(args.config),
                expected_tag=args.expected_tag,
            )
            verify_assets(manifest, args.directory)
        elif args.command == "verify-release":
            manifest = load_json(args.manifest)
            validate_release(manifest, load_json(args.config))
            verify_assets(manifest, args.directory)
        elif args.command == "write-index":
            write_index(load_json(args.manifest), args.directory)
        elif args.command == "finalize-release":
            write_json(
                args.output,
                finalize_release(
                    load_json(args.candidate),
                    config=load_json(args.config),
                    validation=load_json(args.validation),
                    tag=args.release_tag,
                    published_at=args.published_at,
                ),
            )
        elif args.command == "release-tag":
            print(release_tag(load_json(args.candidate)))
        elif args.command == "package-version":
            print(
                expected_package_version(
                    load_json(args.config),
                    args.distribution,
                    args.source_commit,
                )
            )
        elif args.command == "release-notes":
            args.output.write_text(release_notes(load_json(args.manifest)))
        elif args.command == "extract-validation":
            write_json(args.output, extract_validation(args.log))
        elif args.command == "validate-result":
            validate_result(
                load_json(args.result),
                load_json(args.candidate),
                load_json(args.config),
            )
        elif args.command == "run-with-index":
            command = args.argv[1:] if args.argv[:1] == ["--"] else args.argv
            if not command:
                raise ReleaseError("run-with-index requires a command after --")
            with local_flat_index(args.index.resolve()) as index_url:
                environment = {**os.environ, "UV_FIND_LINKS": index_url}
                return subprocess.run(command, env=environment, check=False).returncode
        return 0
    except (KeyError, OSError, ReleaseError, zipfile.BadZipFile) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
