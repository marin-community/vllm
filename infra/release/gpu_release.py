#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build and verify Marin vLLM GPU release metadata."""

from __future__ import annotations

import argparse
import base64
import copy
import json
import os
import re
import sys
import zipfile
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

if __package__:
    from .release_common import (
        RELEASE_REPOSITORY,
        ReleaseError,
        load_json,
        normalized_distribution_name,
        release_asset_url,
        requirement_name,
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
        requirement_name,
        sha256_file,
        wheel_metadata,
        write_json,
    )

MANIFEST_NAME = "marin-vllm-gpu-manifest.json"
VALIDATION_SENTINEL = "MARIN_GPU_VALIDATION_JSON="
SOURCE_REPOSITORY = "https://github.com/marin-community/vllm"
UPSTREAM_REPOSITORY = "https://github.com/vllm-project/vllm"
STABLE_LIBTORCH_GATE = "vllm._C_stable_libtorch"
REQUIRED_EXTENSIONS = (STABLE_LIBTORCH_GATE, "vllm.cumem_allocator")
GRUG_ARCHITECTURE = "GrugMoeForCausalLM"
CANDIDATE_TAG_PREFIX = "marin-vllm-gpu-candidate-"
RELEASE_TAG_PREFIX = "marin-vllm-gpu-"
BUILD_ABI_KEYS = (
    "python_version",
    "torch_version",
    "torch_index_url",
    "cuda_toolkit_version",
    "cuda_variant",
)
RELEASE_ABI_KEYS = (*BUILD_ABI_KEYS, "cuda_runtime_version")
WHEEL_SHA_GATE = "wheel_sha256"
DISTRIBUTION_GATE = "distribution_metadata"
CUMEM_GATE = "cumem_allocator"
TORCHAUDIO_GATE = "torchaudio.resample"
SPARSE_NCCL_GATE = "sparse_nccl_contract"
SOURCE_TESTS_GATE = "source_tests"
SERVE_GATE = "serve_smoke"
REQUIRED_RUNTIME_GATES = (
    WHEEL_SHA_GATE,
    DISTRIBUTION_GATE,
    STABLE_LIBTORCH_GATE,
    GRUG_ARCHITECTURE,
    CUMEM_GATE,
    TORCHAUDIO_GATE,
    SPARSE_NCCL_GATE,
    SERVE_GATE,
)


class CandidatePublicationAction(StrEnum):
    """The next safe transition for a candidate release."""

    CREATE = "create"
    RESUME = "resume"
    PUBLISH = "publish"
    VERIFIED = "verified"


@dataclass(frozen=True)
class CandidatePublicationPlan:
    """The next transition and assets needed to complete it."""

    action: CandidatePublicationAction
    missing_assets: tuple[str, ...]


def _expand_wheel_tag(tag: str) -> set[str]:
    """Expand the three dot-compressed wheel tag fields."""
    fields = tag.split("-")
    if len(fields) != 3 or any(not field for field in fields):
        raise ReleaseError(f"malformed wheel compatibility tag {tag!r}")
    python_tags, abi_tags, platform_tags = (field.split(".") for field in fields)
    return {
        f"{python_tag}-{abi_tag}-{platform_tag}"
        for python_tag in python_tags
        for abi_tag in abi_tags
        for platform_tag in platform_tags
    }


def _packaged_contents(wheel: Path) -> dict[str, str]:
    with zipfile.ZipFile(wheel) as archive:
        members = archive.namelist()
        grug_sources = [
            name
            for name in members
            if name == "vllm/model_executor/models/grugmoe.py"
        ]
        grug_state = "absent"
        if grug_sources:
            source = archive.read(grug_sources[0])
            if b"class GrugMoeForCausalLM" in source:
                grug_state = "included"
        return {
            STABLE_LIBTORCH_GATE: (
                "included"
                if any(
                    re.fullmatch(r"vllm/_C_stable_libtorch(?:\.[^/]*)?\.so", name)
                    for name in members
                )
                else "absent"
            ),
            "vllm.cumem_allocator": (
                "included"
                if any(
                    re.fullmatch(r"vllm/cumem_allocator(?:\.[^/]*)?\.so", name)
                    for name in members
                )
                else "absent"
            ),
            GRUG_ARCHITECTURE: grug_state,
        }


def inspect_wheel(
    wheel: Path,
    *,
    architecture: str,
    config: dict[str, Any],
    fork_commit: str,
    upstream_base: str,
    built_at: str,
    base_image: str,
    base_image_digest: str,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    if architecture not in config["platforms"]:
        raise ReleaseError(f"unsupported architecture {architecture!r}")
    document = wheel_metadata(wheel)
    metadata = document.metadata
    wheel_tags = document.tags
    packaged = _packaged_contents(wheel)
    distribution = metadata.get("Name", "")
    expected_distribution = config["distribution_name"]
    if normalized_distribution_name(distribution) != normalized_distribution_name(
        expected_distribution
    ):
        raise ReleaseError(
            f"{wheel.name} contains distribution {distribution!r}, "
            f"expected {expected_distribution!r}"
        )
    requirements = metadata.get_all("Requires-Dist", [])
    forbidden = [
        requirement
        for requirement in requirements
        if requirement_name(requirement) == "marin-vllm"
    ]
    if forbidden:
        raise ReleaseError(f"wheel moves dependencies to marin-vllm: {forbidden}")
    normalized_requirements = {item.replace(" ", "") for item in requirements}
    if config["torch_metadata_requirement"] not in normalized_requirements:
        raise ReleaseError(
            f"wheel does not require {config['torch_metadata_requirement']!r}"
        )
    compatible_tags = [tag for tag in wheel_tags if tag.endswith(f"_{architecture}")]
    filename_parts = wheel.stem.rsplit("-", 3)
    if len(filename_parts) != 4:
        raise ReleaseError(f"cannot parse wheel tags from {wheel.name}")
    filename_tag = "-".join(filename_parts[-3:])
    filename_platform_tag = filename_parts[-1]
    if not compatible_tags:
        raise ReleaseError(
            f"wheel metadata tags {wheel_tags!r} do not describe {architecture}"
        )
    if not filename_platform_tag.startswith("manylinux_") or not (
        filename_platform_tag.endswith(f"_{architecture}")
    ):
        raise ReleaseError(
            f"wheel filename tag {filename_tag!r} is not manylinux {architecture}"
        )
    metadata_tags = {
        expanded_tag
        for wheel_tag in wheel_tags
        for expanded_tag in _expand_wheel_tag(wheel_tag)
    }
    filename_tags = _expand_wheel_tag(filename_tag)
    if metadata_tags != filename_tags:
        raise ReleaseError(
            f"wheel metadata tags {wheel_tags!r} do not match filename tag "
            f"{filename_tag!r}"
        )

    platform_config = config["platforms"][architecture]
    return {
        "schema_version": config["schema_version"],
        "source": {
            "repository": SOURCE_REPOSITORY,
            "fork_commit": fork_commit,
            "upstream_repository": UPSTREAM_REPOSITORY,
            "upstream_base": upstream_base,
        },
        "distribution": {
            "name": distribution,
            "version": metadata.get("Version", ""),
            "requires_python": metadata.get("Requires-Python", ""),
            "torch_metadata_requirement": config["torch_metadata_requirement"],
        },
        "build": {
            **{key: config[key] for key in BUILD_ABI_KEYS},
            "base_image": base_image,
            "base_image_digest": base_image_digest,
            "built_at": built_at,
            "provenance": provenance,
        },
        "platform": {
            "architecture": architecture,
            "sm_targets": platform_config["sm_targets"],
            "wheel_tags": wheel_tags,
            "filename_tag": filename_tag,
            "packaged": packaged,
            "wheel": {
                "filename": wheel.name,
                "sha256": sha256_file(wheel),
                "size_bytes": wheel.stat().st_size,
            },
        },
    }


def validate_packaged_contents(packaged: dict[str, str]) -> None:
    absent = [
        name
        for name in (*REQUIRED_EXTENSIONS, GRUG_ARCHITECTURE)
        if packaged.get(name) != "included"
    ]
    if absent:
        states = {name: packaged.get(name, "absent") for name in absent}
        raise ReleaseError(f"required wheel contents are absent: {states}")


def validate_wheel_fragment(fragment: dict[str, Any]) -> None:
    validate_packaged_contents(fragment["platform"]["packaged"])


def assemble_candidate(
    fragments: list[dict[str, Any]],
    *,
    config: dict[str, Any],
    repository: str,
    candidate_tag: str,
    created_at: str,
) -> dict[str, Any]:
    expected_architectures = set(config["platforms"])
    by_architecture = {
        fragment["platform"]["architecture"]: fragment for fragment in fragments
    }
    if len(by_architecture) != len(fragments):
        raise ReleaseError("candidate contains duplicate architecture fragments")
    if set(by_architecture) != expected_architectures:
        raise ReleaseError(
            "candidate architectures do not match config: "
            f"expected {sorted(expected_architectures)}, "
            f"got {sorted(by_architecture)}"
        )
    for fragment in fragments:
        validate_wheel_fragment(fragment)

    first = by_architecture[sorted(by_architecture)[0]]
    common_fields = ("source", "distribution")
    for architecture, fragment in by_architecture.items():
        for field in common_fields:
            if fragment[field] != first[field]:
                raise ReleaseError(
                    f"{architecture} fragment disagrees on {field}: "
                    f"{fragment[field]!r} != {first[field]!r}"
                )
        expected_build = {key: first["build"][key] for key in BUILD_ABI_KEYS}
        actual_build = {key: fragment["build"][key] for key in expected_build}
        if actual_build != expected_build:
            raise ReleaseError(f"{architecture} fragment has a different build ABI")

    platforms = []
    for architecture in sorted(by_architecture):
        fragment = by_architecture[architecture]
        platform = copy.deepcopy(fragment["platform"])
        platform["build"] = fragment["build"]
        filename = platform["wheel"]["filename"]
        platform["wheel"]["url"] = release_asset_url(
            repository, candidate_tag, filename
        )
        platforms.append(platform)

    return {
        "schema_version": config["schema_version"],
        "release": {
            "brand": config["brand"],
            "repository": repository,
            "tag": candidate_tag,
            "status": "candidate",
            "created_at": created_at,
        },
        "source": first["source"],
        "distribution": first["distribution"],
        "abi": {key: config[key] for key in RELEASE_ABI_KEYS},
        "platforms": platforms,
        "validation": {"status": "pending", "targets": []},
    }


def verify_manifest_assets(manifest: dict[str, Any], directory: Path) -> None:
    for platform in manifest["platforms"]:
        wheel = platform["wheel"]
        path = directory / wheel["filename"]
        if not path.is_file():
            raise ReleaseError(f"release asset is missing: {path}")
        actual = sha256_file(path)
        if actual != wheel["sha256"]:
            raise ReleaseError(
                f"SHA-256 mismatch for {path.name}: {actual} != {wheel['sha256']}"
            )


def validate_candidate(manifest: dict[str, Any], config: dict[str, Any]) -> None:
    if manifest["release"]["status"] != "candidate":
        raise ReleaseError("manifest is not a candidate")
    _validate_manifest_common(manifest, config)
    expected_tag = (
        CANDIDATE_TAG_PREFIX + manifest["source"]["fork_commit"][:12]
    )
    if manifest["release"]["tag"] != expected_tag:
        raise ReleaseError("candidate tag does not match its fork commit")
    if manifest["validation"] != {"status": "pending", "targets": []}:
        raise ReleaseError("candidate validation state is not pending")


def _candidate_asset_digests(
    manifest: dict[str, Any], directory: Path, config: dict[str, Any]
) -> dict[str, str]:
    validate_candidate(manifest, config)
    manifest_path = directory / MANIFEST_NAME
    if load_json(manifest_path) != manifest:
        raise ReleaseError("candidate manifest file disagrees with the release input")
    verify_manifest_assets(manifest, directory)
    filenames = [
        MANIFEST_NAME,
        *(platform["wheel"]["filename"] for platform in manifest["platforms"]),
    ]
    return {filename: sha256_file(directory / filename) for filename in filenames}


def _validate_candidate_release_identity(
    manifest: dict[str, Any], release: dict[str, Any]
) -> None:
    if release.get("tag_name") != manifest["release"]["tag"]:
        raise ReleaseError("candidate release tag disagrees with its manifest")
    if release.get("target_commitish") != manifest["source"]["fork_commit"]:
        raise ReleaseError("candidate release targets a different commit")
    if release.get("prerelease") is not True:
        raise ReleaseError("candidate release is not a prerelease")
    if not isinstance(release.get("draft"), bool):
        raise ReleaseError("candidate release draft state is missing")


def _validate_existing_candidate_assets(
    release: dict[str, Any],
    existing_directory: Path,
    expected: dict[str, str],
) -> set[str]:
    release_assets = release.get("assets")
    if not isinstance(release_assets, list):
        raise ReleaseError("candidate release asset metadata is missing")
    by_name = {asset.get("name"): asset for asset in release_assets}
    if None in by_name or len(by_name) != len(release_assets):
        raise ReleaseError("candidate release contains unnamed or duplicate assets")
    asset_names = set(by_name)
    unexpected = asset_names - set(expected)
    if unexpected:
        raise ReleaseError(
            f"candidate release contains unexpected assets: {sorted(unexpected)}"
        )

    existing_paths = list(existing_directory.iterdir())
    if any(not path.is_file() for path in existing_paths):
        raise ReleaseError("downloaded candidate assets contain a non-file entry")
    downloaded_names = {path.name for path in existing_paths}
    if downloaded_names != asset_names:
        raise ReleaseError(
            "downloaded candidate assets disagree with release metadata: "
            f"expected {sorted(asset_names)}, got {sorted(downloaded_names)}"
        )

    for name, asset in by_name.items():
        if asset.get("state") != "uploaded":
            raise ReleaseError(f"candidate asset has not finished uploading: {name}")
        actual_digest = sha256_file(existing_directory / name)
        recorded_digest = asset.get("digest")
        if recorded_digest and recorded_digest != f"sha256:{actual_digest}":
            raise ReleaseError(f"candidate asset digest metadata changed: {name}")
        if actual_digest != expected[name]:
            raise ReleaseError(f"existing candidate asset differs: {name}")
    return asset_names


def plan_candidate_publication(
    manifest: dict[str, Any],
    *,
    candidate_directory: Path,
    config: dict[str, Any],
    release: dict[str, Any] | None,
    existing_directory: Path,
) -> CandidatePublicationPlan:
    """Validate candidate state and return its next safe release transition."""
    expected = _candidate_asset_digests(manifest, candidate_directory, config)
    if release is None:
        if any(existing_directory.iterdir()):
            raise ReleaseError(
                "candidate release is absent but downloaded assets exist"
            )
        return CandidatePublicationPlan(
            action=CandidatePublicationAction.CREATE,
            missing_assets=tuple(sorted(expected)),
        )

    _validate_candidate_release_identity(manifest, release)
    asset_names = _validate_existing_candidate_assets(
        release, existing_directory, expected
    )
    missing = tuple(sorted(set(expected) - asset_names))
    if release["draft"]:
        action = (
            CandidatePublicationAction.RESUME
            if missing
            else CandidatePublicationAction.PUBLISH
        )
        return CandidatePublicationPlan(action=action, missing_assets=missing)
    if missing:
        raise ReleaseError(
            f"published candidate is incomplete; missing assets: {list(missing)}"
        )
    return CandidatePublicationPlan(
        action=CandidatePublicationAction.VERIFIED,
        missing_assets=(),
    )


def _validate_manifest_common(
    manifest: dict[str, Any], config: dict[str, Any]
) -> None:
    release = manifest["release"]
    source = manifest["source"]
    if manifest["schema_version"] != config["schema_version"]:
        raise ReleaseError("release schema version changed")
    if release["brand"] != config["brand"]:
        raise ReleaseError("release brand changed")
    if release["repository"] != RELEASE_REPOSITORY:
        raise ReleaseError("release repository changed")
    if source["repository"] != SOURCE_REPOSITORY:
        raise ReleaseError("source repository changed")
    if source["upstream_repository"] != UPSTREAM_REPOSITORY:
        raise ReleaseError("upstream repository changed")
    for field in ("fork_commit", "upstream_base"):
        if re.fullmatch(r"[0-9a-f]{40}", source[field]) is None:
            raise ReleaseError(f"source {field} is not a full Git commit")
    if manifest["distribution"]["name"] != config["distribution_name"]:
        raise ReleaseError("distribution name changed")
    expected_abi = {key: config[key] for key in RELEASE_ABI_KEYS}
    if manifest["abi"] != expected_abi:
        raise ReleaseError("release ABI changed")
    architectures = {item["architecture"] for item in manifest["platforms"]}
    if (
        architectures != set(config["platforms"])
        or len(manifest["platforms"]) != len(config["platforms"])
    ):
        raise ReleaseError("release platform set is incomplete")
    if not manifest["distribution"].get("version"):
        raise ReleaseError("distribution version is missing")
    for platform in manifest["platforms"]:
        architecture = platform["architecture"]
        expected_platform = config["platforms"][architecture]
        if platform["sm_targets"] != expected_platform["sm_targets"]:
            raise ReleaseError(f"{architecture} SM targets changed")
        filename = platform["wheel"]["filename"]
        if re.fullmatch(
            rf".+-manylinux_[0-9]+_[0-9]+_{architecture}\.whl", filename
        ) is None:
            raise ReleaseError(f"{architecture} wheel filename is not manylinux")
        filename_parts = Path(filename).stem.rsplit("-", 3)
        if len(filename_parts) != 4 or platform["filename_tag"] != "-".join(
            filename_parts[-3:]
        ):
            raise ReleaseError(f"{architecture} wheel filename tag changed")
        if not any(
            tag.endswith(f"_{architecture}") for tag in platform["wheel_tags"]
        ):
            raise ReleaseError(f"{architecture} wheel metadata tags changed")
        expected_url = release_asset_url(
            release["repository"], release["tag"], filename
        )
        if platform["wheel"]["url"] != expected_url:
            raise ReleaseError(f"{architecture} wheel URL is not tag-addressed")
        if re.fullmatch(r"[0-9a-f]{64}", platform["wheel"]["sha256"]) is None:
            raise ReleaseError(f"{architecture} wheel SHA-256 is malformed")
        build = platform["build"]
        expected_build = {key: config[key] for key in BUILD_ABI_KEYS}
        expected_build["base_image"] = expected_platform["build_base_image"]
        for key, expected in expected_build.items():
            if build.get(key) != expected:
                raise ReleaseError(f"{architecture} build {key} changed")
        if re.fullmatch(
            r"[^@]+@sha256:[0-9a-f]{64}", build.get("base_image_digest", "")
        ) is None:
            raise ReleaseError(f"{architecture} base image digest is malformed")
        expected_digest = expected_platform["build_base_image"].rsplit("@", 1)[-1]
        if build["base_image_digest"].rsplit("@", 1)[-1] != expected_digest:
            raise ReleaseError(f"{architecture} base image digest changed")
        provenance = build.get("provenance", {})
        if provenance.get("system") != "GitHub Actions":
            raise ReleaseError(f"{architecture} build provenance is missing")
        if not provenance.get("run_url", "").startswith(
            f"{SOURCE_REPOSITORY}/actions/runs/"
        ):
            raise ReleaseError(f"{architecture} build run URL is missing")
        validate_packaged_contents(platform["packaged"])


def validate_release(manifest: dict[str, Any], config: dict[str, Any]) -> None:
    if manifest["release"]["status"] != "released":
        raise ReleaseError("manifest is not a final release")
    _validate_manifest_common(manifest, config)
    source_prefix = manifest["source"]["fork_commit"][:12]
    if manifest["release"].get("candidate_tag") != (
        f"{CANDIDATE_TAG_PREFIX}{source_prefix}"
    ):
        raise ReleaseError("release names the wrong candidate tag")
    if re.fullmatch(
        rf"{RELEASE_TAG_PREFIX}[0-9]{{8}}-{source_prefix}",
        manifest["release"]["tag"],
    ) is None:
        raise ReleaseError("release tag does not match its fork commit")
    if manifest["validation"].get("status") != "passed":
        raise ReleaseError("release validation status is not passed")
    validations = manifest["validation"].get("targets", [])
    index_validations(validations, config)

    candidate = copy.deepcopy(manifest)
    candidate["release"]["tag"] = manifest["release"]["candidate_tag"]
    candidate["release"]["status"] = "candidate"
    candidate["validation"] = {"status": "pending", "targets": []}
    for platform in candidate["platforms"]:
        wheel = platform["wheel"]
        wheel["url"] = release_asset_url(
            candidate["release"]["repository"],
            candidate["release"]["tag"],
            wheel["filename"],
        )
    validate_candidate(candidate, config)
    for result in validations:
        validate_validation_result(result, candidate, config)


def verify_release_assets(
    manifest: dict[str, Any], directory: Path, config: dict[str, Any]
) -> None:
    validate_release(manifest, config)
    verify_manifest_assets(manifest, directory)
    for result in manifest["validation"]["targets"]:
        hardware = result["hardware"]["requested"].lower()
        result_path = directory / f"marin-vllm-validation-{hardware}.json"
        if not result_path.is_file():
            raise ReleaseError(f"release validation asset is missing: {result_path}")
        if load_json(result_path) != result:
            raise ReleaseError(f"release validation asset changed: {result_path}")


def validate_validation_result(
    result: dict[str, Any], candidate: dict[str, Any], config: dict[str, Any]
) -> None:
    architecture = result.get("architecture")
    if architecture not in config["platforms"]:
        raise ReleaseError(f"validation has unknown architecture {architecture!r}")
    platform = next(
        item for item in candidate["platforms"] if item["architecture"] == architecture
    )
    expected_validation = config["platforms"][architecture]["validation"]
    if result.get("candidate_tag") != candidate["release"]["tag"]:
        raise ReleaseError(f"{architecture} validation names a different candidate")
    if result.get("source_commit") != candidate["source"]["fork_commit"]:
        raise ReleaseError(f"{architecture} validation names a different commit")
    if result.get("hardware", {}).get("requested") != expected_validation["gpu"]:
        raise ReleaseError(f"{architecture} validation used the wrong GPU")
    if result.get("hardware", {}).get("compute_capability") != expected_validation[
        "compute_capability"
    ]:
        raise ReleaseError(f"{architecture} validation used the wrong SM")
    expected_wheel = {
        key: platform["wheel"][key] for key in ("filename", "sha256", "url")
    }
    if result.get("wheel") != expected_wheel:
        raise ReleaseError(f"{architecture} validation used a different wheel")
    environment = result.get("environment", {})
    expected_environment = {
        "attention_backend": expected_validation["attention_backend"],
        "machine": architecture,
        "torch_version": config["torch_version"],
        "torch_cuda_runtime": config["cuda_runtime_version"],
        "task_image": config["validation_task_image"],
    }
    for key, expected in expected_environment.items():
        if environment.get(key) != expected:
            raise ReleaseError(
                f"{architecture} validation environment has {key}="
                f"{environment.get(key)!r}, expected {expected!r}"
            )
    if not environment.get("python_version", "").startswith(
        config["python_version"] + "."
    ):
        raise ReleaseError(f"{architecture} validation used the wrong Python")
    package_path = environment.get("vllm_package_path", "")
    if "/site-packages/vllm/" not in package_path:
        raise ReleaseError(
            f"{architecture} validation did not import the installed wheel: "
            f"{package_path!r}"
        )
    if result.get("result") != "passed":
        raise ReleaseError(
            f"{architecture} validation did not pass: {result.get('result')!r}"
        )
    required_gates = list(REQUIRED_RUNTIME_GATES)
    if expected_validation["run_source_tests"]:
        required_gates.append(SOURCE_TESTS_GATE)
    failed = {
        gate: result.get("gates", {}).get(gate, {}).get("status", "absent")
        for gate in required_gates
        if result.get("gates", {}).get(gate, {}).get("status") != "passed"
    }
    if failed:
        raise ReleaseError(f"{architecture} validation gates did not pass: {failed}")
    allocator_gate = result["gates"][CUMEM_GATE]
    if allocator_gate.get("allocated_bytes", 0) <= 0:
        raise ReleaseError(f"{architecture} cuMem gate did not allocate memory")
    serve_metrics = result["gates"][SERVE_GATE].get("metrics", {})
    if serve_metrics.get("completions", 0) <= 0:
        raise ReleaseError(f"{architecture} serving gate has no completions")
    if serve_metrics.get("output_tokens_per_second", 0) <= 0:
        raise ReleaseError(f"{architecture} serving gate has no throughput result")


def index_validations(
    validations: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    by_architecture = {result.get("architecture"): result for result in validations}
    if len(by_architecture) != len(validations):
        raise ReleaseError("release contains duplicate validation results")
    if set(by_architecture) != set(config["platforms"]):
        raise ReleaseError("release does not contain both GPU validation results")
    return by_architecture


def finalize_release(
    candidate: dict[str, Any],
    validations: list[dict[str, Any]],
    *,
    config: dict[str, Any],
    release_tag: str,
    published_at: str,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    validate_candidate(candidate, config)
    by_architecture = index_validations(validations, config)
    for result in validations:
        validate_validation_result(result, candidate, config)

    manifest = copy.deepcopy(candidate)
    candidate_tag = candidate["release"]["tag"]
    manifest["release"] = {
        "brand": config["brand"],
        "repository": candidate["release"]["repository"],
        "tag": release_tag,
        "status": "released",
        "published_at": published_at,
        "candidate_tag": candidate_tag,
        "provenance": provenance,
    }
    for platform in manifest["platforms"]:
        filename = platform["wheel"]["filename"]
        platform["wheel"]["url"] = release_asset_url(
            manifest["release"]["repository"], release_tag, filename
        )
    manifest["validation"] = {
        "status": "passed",
        "targets": [by_architecture[key] for key in sorted(by_architecture)],
    }
    return manifest


def build_matrix(config: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    includes = []
    for architecture, platform in config["platforms"].items():
        includes.append(
            {
                "architecture": architecture,
                "runner": platform["runner"],
                "build_base_image": platform["build_base_image"],
                # Build-arg values are sourced from config.json so the workflow
                # holds no toolchain literals that can drift from the base image.
                "cuda_version": config["cuda_toolkit_version"],
                "python_version": config["python_version"],
                "max_jobs": platform["max_jobs"],
                "max_wheel_size_mb": platform["max_wheel_size_mb"],
                "nvcc_threads": platform["nvcc_threads"],
                "sm_targets": " ".join(platform["sm_targets"]),
            }
        )
    return {"include": includes}


def validation_matrix(config: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    includes = []
    for architecture, platform in config["platforms"].items():
        validation = platform["validation"]
        includes.append(
            {
                "architecture": architecture,
                "hardware": validation["gpu"],
                "gpu_resource": validation["gpu_resource"],
                "target_cluster": validation["target_cluster"],
                "cpu": validation["cpu"],
                "memory": validation["memory"],
            }
        )
    return {"include": includes}


def extract_validation(log_path: Path) -> dict[str, Any]:
    encoded_payload = None
    with log_path.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if VALIDATION_SENTINEL in line:
                encoded_payload = line.partition(VALIDATION_SENTINEL)[2].strip()
    if encoded_payload is None:
        return {
            "schema_version": 1,
            "result": "failed",
            "failure": "Iris log did not contain a GPU validation record",
            "gates": {},
        }
    try:
        decoded = base64.b64decode(encoded_payload, validate=True)
        result = json.loads(decoded)
    except (ValueError, json.JSONDecodeError) as exc:
        raise ReleaseError("GPU validation record is not valid base64 JSON") from exc
    if not isinstance(result, dict):
        raise ReleaseError("GPU validation record must be a JSON object")
    return result


def release_notes(manifest: dict[str, Any]) -> str:
    source = manifest["source"]
    abi = manifest["abi"]
    lines = [
        f"Marin vLLM GPU wheels for `{source['fork_commit']}`.",
        "",
        f"Upstream base: `{source['upstream_base']}`",
        (
            f"Build ABI: CPython {abi['python_version']}, Torch "
            f"{abi['torch_version']}, CUDA {abi['cuda_toolkit_version']}"
        ),
        "",
    ]
    for platform in manifest["platforms"]:
        wheel = platform["wheel"]
        link = f"[{wheel['filename']}]({wheel['url']})"
        lines.extend(
            [
                f"- `{platform['architecture']}`: {link}",
                f"  SHA-256: `{wheel['sha256']}`",
            ]
        )
    manifest_url = release_asset_url(
        manifest["release"]["repository"],
        manifest["release"]["tag"],
        MANIFEST_NAME,
    )
    lines.extend(
        [
            "",
            f"Machine-readable provenance: [{MANIFEST_NAME}]({manifest_url})",
        ]
    )
    if manifest["release"]["status"] == "candidate":
        lines.extend(
            [
                "",
                "This candidate has not passed both Iris GPU validation lanes.",
            ]
        )
    return "\n".join(lines) + "\n"


def provenance_from_environment(
    base_image: str, base_image_digest: str
) -> dict[str, Any]:
    repository = os.environ.get("GITHUB_REPOSITORY", RELEASE_REPOSITORY)
    server = os.environ.get("GITHUB_SERVER_URL", "https://github.com")
    run_id = os.environ.get("GITHUB_RUN_ID", "unknown")
    provenance = {
        "system": "GitHub Actions",
        "workflow_ref": os.environ.get("GITHUB_WORKFLOW_REF", "unknown"),
        "run_id": run_id,
        "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT", "unknown"),
        "job": os.environ.get("GITHUB_JOB", "unknown"),
        "runner_arch": os.environ.get("RUNNER_ARCH", "unknown"),
        "run_url": f"{server}/{repository}/actions/runs/{run_id}",
    }
    if base_image:
        provenance["base_image"] = base_image
    if base_image_digest:
        provenance["base_image_digest"] = base_image_digest
    return provenance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    matrix_parser = subparsers.add_parser("build-matrix")
    matrix_parser.add_argument("--config", type=Path, required=True)

    validation_matrix_parser = subparsers.add_parser("validation-matrix")
    validation_matrix_parser.add_argument("--config", type=Path, required=True)

    inspect_parser = subparsers.add_parser("inspect-wheel")
    inspect_parser.add_argument("--config", type=Path, required=True)
    inspect_parser.add_argument("--wheel", type=Path, required=True)
    inspect_parser.add_argument("--architecture", required=True)
    inspect_parser.add_argument("--fork-commit", required=True)
    inspect_parser.add_argument("--upstream-base", required=True)
    inspect_parser.add_argument("--built-at", required=True)
    inspect_parser.add_argument("--base-image", required=True)
    inspect_parser.add_argument("--base-image-digest", required=True)
    inspect_parser.add_argument("--output", type=Path, required=True)

    assemble_parser = subparsers.add_parser("assemble-candidate")
    assemble_parser.add_argument("--config", type=Path, required=True)
    assemble_parser.add_argument(
        "--fragment", type=Path, action="append", required=True
    )
    assemble_parser.add_argument("--repository", default=RELEASE_REPOSITORY)
    assemble_parser.add_argument("--candidate-tag", required=True)
    assemble_parser.add_argument("--created-at", required=True)
    assemble_parser.add_argument("--output", type=Path, required=True)

    verify_parser = subparsers.add_parser("verify-assets")
    verify_parser.add_argument("--manifest", type=Path, required=True)
    verify_parser.add_argument("--directory", type=Path, required=True)
    verify_parser.add_argument("--config", type=Path)

    candidate_parser = subparsers.add_parser("validate-candidate")
    candidate_parser.add_argument("--manifest", type=Path, required=True)
    candidate_parser.add_argument("--config", type=Path, required=True)

    publication_parser = subparsers.add_parser("candidate-publication-plan")
    publication_parser.add_argument("--manifest", type=Path, required=True)
    publication_parser.add_argument("--directory", type=Path, required=True)
    publication_parser.add_argument("--config", type=Path, required=True)
    publication_parser.add_argument("--release", type=Path)
    publication_parser.add_argument(
        "--existing-directory", type=Path, required=True
    )
    publication_parser.add_argument("--output", type=Path, required=True)

    release_parser = subparsers.add_parser("verify-release")
    release_parser.add_argument("--manifest", type=Path, required=True)
    release_parser.add_argument("--directory", type=Path, required=True)
    release_parser.add_argument("--config", type=Path, required=True)

    validation_parser = subparsers.add_parser("validate-result")
    validation_parser.add_argument("--result", type=Path, required=True)
    validation_parser.add_argument("--candidate", type=Path, required=True)
    validation_parser.add_argument("--config", type=Path, required=True)

    finalize_parser = subparsers.add_parser("finalize-release")
    finalize_parser.add_argument("--candidate", type=Path, required=True)
    finalize_parser.add_argument(
        "--validation", type=Path, action="append", required=True
    )
    finalize_parser.add_argument("--config", type=Path, required=True)
    finalize_parser.add_argument("--release-tag", required=True)
    finalize_parser.add_argument("--published-at", required=True)
    finalize_parser.add_argument("--output", type=Path, required=True)

    extract_parser = subparsers.add_parser("extract-validation")
    extract_parser.add_argument("--log", type=Path, required=True)
    extract_parser.add_argument("--output", type=Path, required=True)

    notes_parser = subparsers.add_parser("release-notes")
    notes_parser.add_argument("--manifest", type=Path, required=True)
    notes_parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.command == "build-matrix":
            print(
                json.dumps(
                    build_matrix(load_json(args.config)), separators=(",", ":")
                )
            )
            return 0
        if args.command == "validation-matrix":
            print(
                json.dumps(
                    validation_matrix(load_json(args.config)),
                    separators=(",", ":"),
                )
            )
            return 0
        if args.command == "inspect-wheel":
            fragment = inspect_wheel(
                args.wheel,
                architecture=args.architecture,
                config=load_json(args.config),
                fork_commit=args.fork_commit,
                upstream_base=args.upstream_base,
                built_at=args.built_at,
                base_image=args.base_image,
                base_image_digest=args.base_image_digest,
                provenance=provenance_from_environment(
                    args.base_image, args.base_image_digest
                ),
            )
            write_json(args.output, fragment)
            validate_wheel_fragment(fragment)
            return 0
        if args.command == "assemble-candidate":
            manifest = assemble_candidate(
                [load_json(path) for path in args.fragment],
                config=load_json(args.config),
                repository=args.repository,
                candidate_tag=args.candidate_tag,
                created_at=args.created_at,
            )
            write_json(args.output, manifest)
            return 0
        if args.command == "verify-assets":
            manifest = load_json(args.manifest)
            if args.config:
                validate_candidate(manifest, load_json(args.config))
            verify_manifest_assets(manifest, args.directory)
            return 0
        if args.command == "validate-candidate":
            validate_candidate(load_json(args.manifest), load_json(args.config))
            return 0
        if args.command == "candidate-publication-plan":
            plan = plan_candidate_publication(
                load_json(args.manifest),
                candidate_directory=args.directory,
                config=load_json(args.config),
                release=load_json(args.release) if args.release else None,
                existing_directory=args.existing_directory,
            )
            write_json(args.output, asdict(plan))
            return 0
        if args.command == "verify-release":
            verify_release_assets(
                load_json(args.manifest), args.directory, load_json(args.config)
            )
            return 0
        if args.command == "validate-result":
            validate_validation_result(
                load_json(args.result),
                load_json(args.candidate),
                load_json(args.config),
            )
            return 0
        if args.command == "finalize-release":
            manifest = finalize_release(
                load_json(args.candidate),
                [load_json(path) for path in args.validation],
                config=load_json(args.config),
                release_tag=args.release_tag,
                published_at=args.published_at,
                provenance=provenance_from_environment("", ""),
            )
            write_json(args.output, manifest)
            return 0
        if args.command == "extract-validation":
            write_json(args.output, extract_validation(args.log))
            return 0
        if args.command == "release-notes":
            args.output.write_text(release_notes(load_json(args.manifest)))
            return 0
    except (KeyError, OSError, ReleaseError, zipfile.BadZipFile) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    raise AssertionError(f"unhandled command {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
