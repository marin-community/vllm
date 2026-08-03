#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build and verify Marin vLLM GPU release metadata."""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import os
import re
import sys
import zipfile
from email.parser import BytesParser
from pathlib import Path
from typing import Any
from urllib.parse import quote

MANIFEST_NAME = "marin-vllm-gpu-manifest.json"
VALIDATION_SENTINEL = "MARIN_GPU_VALIDATION_JSON="
RELEASE_REPOSITORY = "marin-community/vllm"
SOURCE_REPOSITORY = "https://github.com/marin-community/vllm"
UPSTREAM_REPOSITORY = "https://github.com/vllm-project/vllm"
REQUIRED_EXTENSIONS = ("vllm._C", "vllm.cumem_allocator")
GRUG_ARCHITECTURE = "GrugMoeForCausalLM"


class ReleaseError(RuntimeError):
    """A release input violates the published artifact contract."""


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ReleaseError(f"{path} must contain a JSON object")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_distribution_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def requirement_name(requirement: str) -> str:
    match = re.match(r"\s*([A-Za-z0-9_.-]+)", requirement)
    if match is None:
        raise ReleaseError(f"cannot parse Requires-Dist entry {requirement!r}")
    return normalized_distribution_name(match.group(1))


def release_asset_url(repository: str, tag: str, filename: str) -> str:
    return (
        f"https://github.com/{repository}/releases/download/"
        f"{quote(tag, safe='')}/{quote(filename, safe='+._-')}"
    )


def _wheel_document(wheel: Path) -> tuple[Any, list[str], dict[str, str]]:
    with zipfile.ZipFile(wheel) as archive:
        metadata_names = [
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        ]
        wheel_names = [
            name for name in archive.namelist() if name.endswith(".dist-info/WHEEL")
        ]
        if len(metadata_names) != 1 or len(wheel_names) != 1:
            raise ReleaseError(
                f"{wheel.name} must contain one METADATA and one WHEEL file"
            )
        metadata = BytesParser().parsebytes(archive.read(metadata_names[0]))
        wheel_metadata = BytesParser().parsebytes(archive.read(wheel_names[0]))
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
        packaged = {
            "vllm._C": (
                "included"
                if any(
                    re.fullmatch(r"vllm/_C(?:\.[^/]*)?\.so", name)
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
    return metadata, wheel_metadata.get_all("Tag", []), packaged


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
    metadata, wheel_tags, packaged = _wheel_document(wheel)
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
            "python_version": config["python_version"],
            "torch_version": config["torch_version"],
            "torch_index_url": config["torch_index_url"],
            "cuda_toolkit_version": config["cuda_toolkit_version"],
            "cuda_variant": config["cuda_variant"],
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


def validate_wheel_fragment(fragment: dict[str, Any]) -> None:
    packaged = fragment["platform"]["packaged"]
    absent = [
        name
        for name in (*REQUIRED_EXTENSIONS, GRUG_ARCHITECTURE)
        if packaged.get(name) != "included"
    ]
    if absent:
        states = {name: packaged.get(name, "absent") for name in absent}
        raise ReleaseError(f"required wheel contents are absent: {states}")


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
        expected_build = {
            key: first["build"][key]
            for key in (
                "python_version",
                "torch_version",
                "torch_index_url",
                "cuda_toolkit_version",
                "cuda_variant",
            )
        }
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
        "abi": {
            "python_version": config["python_version"],
            "torch_version": config["torch_version"],
            "torch_index_url": config["torch_index_url"],
            "cuda_toolkit_version": config["cuda_toolkit_version"],
            "cuda_runtime_version": config["cuda_runtime_version"],
            "cuda_variant": config["cuda_variant"],
        },
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
        "marin-vllm-gpu-candidate-" + manifest["source"]["fork_commit"][:12]
    )
    if manifest["release"]["tag"] != expected_tag:
        raise ReleaseError("candidate tag does not match its fork commit")
    if manifest["validation"] != {"status": "pending", "targets": []}:
        raise ReleaseError("candidate validation state is not pending")


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
    expected_abi = {
        "python_version": config["python_version"],
        "torch_version": config["torch_version"],
        "torch_index_url": config["torch_index_url"],
        "cuda_toolkit_version": config["cuda_toolkit_version"],
        "cuda_runtime_version": config["cuda_runtime_version"],
        "cuda_variant": config["cuda_variant"],
    }
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
        expected_build = {
            "python_version": config["python_version"],
            "torch_version": config["torch_version"],
            "torch_index_url": config["torch_index_url"],
            "cuda_toolkit_version": config["cuda_toolkit_version"],
            "cuda_variant": config["cuda_variant"],
            "base_image": expected_platform["build_base_image"],
        }
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
            "https://github.com/marin-community/vllm/actions/runs/"
        ):
            raise ReleaseError(f"{architecture} build run URL is missing")
        validate_wheel_fragment({"platform": platform})


def validate_release(manifest: dict[str, Any], config: dict[str, Any]) -> None:
    if manifest["release"]["status"] != "released":
        raise ReleaseError("manifest is not a final release")
    _validate_manifest_common(manifest, config)
    source_prefix = manifest["source"]["fork_commit"][:12]
    if manifest["release"].get("candidate_tag") != (
        f"marin-vllm-gpu-candidate-{source_prefix}"
    ):
        raise ReleaseError("release names the wrong candidate tag")
    if re.fullmatch(
        rf"marin-vllm-gpu-[0-9]{{8}}-{source_prefix}",
        manifest["release"]["tag"],
    ) is None:
        raise ReleaseError("release tag does not match its fork commit")
    if manifest["validation"].get("status") != "passed":
        raise ReleaseError("release validation status is not passed")
    validations = manifest["validation"].get("targets", [])
    by_architecture = {item.get("architecture"): item for item in validations}
    if len(by_architecture) != len(validations):
        raise ReleaseError("release contains duplicate validation results")
    if set(by_architecture) != set(config["platforms"]):
        raise ReleaseError("release does not contain both GPU validation results")

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
    required_gates = [
        "wheel_sha256",
        "distribution_metadata",
        "vllm._C",
        GRUG_ARCHITECTURE,
        "cumem_allocator",
        "serve_smoke",
    ]
    if expected_validation["run_source_tests"]:
        required_gates.append("source_tests")
    failed = {
        gate: result.get("gates", {}).get(gate, {}).get("status", "absent")
        for gate in required_gates
        if result.get("gates", {}).get(gate, {}).get("status") != "passed"
    }
    if failed:
        raise ReleaseError(f"{architecture} validation gates did not pass: {failed}")
    allocator_gate = result["gates"]["cumem_allocator"]
    if allocator_gate.get("allocated_bytes", 0) <= 0:
        raise ReleaseError(f"{architecture} cuMem gate did not allocate memory")
    serve_metrics = result["gates"]["serve_smoke"].get("metrics", {})
    if serve_metrics.get("completions", 0) <= 0:
        raise ReleaseError(f"{architecture} serving gate has no completions")
    if serve_metrics.get("output_tokens_per_second", 0) <= 0:
        raise ReleaseError(f"{architecture} serving gate has no throughput result")


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
    by_architecture = {result.get("architecture"): result for result in validations}
    if len(by_architecture) != len(validations):
        raise ReleaseError("release contains duplicate validation results")
    if set(by_architecture) != set(config["platforms"]):
        raise ReleaseError("release does not contain both GPU validation results")
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
                "max_jobs": platform["max_jobs"],
                "nvcc_threads": platform["nvcc_threads"],
                "sm_targets": " ".join(platform["sm_targets"]),
            }
        )
    return {"include": includes}


def extract_validation(log_path: Path) -> dict[str, Any]:
    sentinel = None
    with log_path.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if line.startswith(VALIDATION_SENTINEL):
                sentinel = line.removeprefix(VALIDATION_SENTINEL).strip()
    if sentinel is None:
        return {
            "schema_version": 1,
            "result": "failed",
            "failure": "Iris log did not contain a GPU validation record",
            "gates": {},
        }
    try:
        decoded = base64.b64decode(sentinel, validate=True)
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
