"""Transport immutable sccache objects outside the credential-free builder."""

import argparse
import json
import logging
import os
import subprocess
import tempfile
import time
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

STAGES = ("cuda", "rust")
TRANSFER_TIMEOUT = 900
CACHE_RETENTION_DAYS = 21


def local_objects(directory: Path) -> dict[str, int]:
    return {
        str(path.relative_to(directory)): path.stat().st_size
        for path in directory.rglob("*")
        if path.is_file() and not path.is_symlink()
    }


def remote_objects(bucket: str, prefix: str) -> dict[str, int]:
    result = subprocess.run(
        [
            "gcloud", "storage", "objects", "list", f"gs://{bucket}/{prefix}**",
            "--format=json(name,size)", "--quiet",
        ],
        check=True, stdout=subprocess.PIPE, text=True, timeout=120,
    )
    return {
        item["name"][len(prefix):]: int(item["size"])
        for item in json.loads(result.stdout)
        if item["name"].startswith(prefix)
    }


def transfer(
    operation: str, bucket: str, namespace: str, architecture: str,
    directory: Path, output: Path, *, used_since_ns: int | None = None,
    now: datetime | None = None,
) -> None:
    """Merge compiler objects without deleting or replacing remote variants.

    Records payload bytes and elapsed transport time. A failed transfer raises
    after recording its status; workflow cache steps are explicitly best effort.
    """
    if operation not in ("restore", "save"):
        raise ValueError(f"Unsupported cache operation: {operation}")
    if operation == "save" and used_since_ns is None:
        raise ValueError("Save requires the timestamp taken before compilation")
    namespace = namespace.strip("/")
    today = (now or datetime.now(UTC)).date()
    week = today - timedelta(days=today.weekday())
    weeks = [week] if operation == "save" else [week, week - timedelta(days=7)]
    records = []
    try:
        for stage in STAGES:
            started = time.monotonic()
            local = directory / stage
            local.mkdir(parents=True, exist_ok=True)
            record = {
                "stage": stage, "operation": operation,
                "status": "failed", "payload_bytes": 0,
                "weeks": [day.isoformat() for day in weeks],
            }
            records.append(record)
            try:
                before = local_objects(local)
                selected = {
                    key: size for key, size in before.items()
                    if used_since_ns is not None
                    and (local / key).stat().st_mtime_ns >= used_since_ns
                }
                record["remote_objects"] = record["remote_bytes"] = 0
                with tempfile.TemporaryDirectory(dir=directory.parent) as temporary:
                    upload = Path(temporary)
                    if operation == "save":
                        for key in selected:
                            destination = upload / key
                            destination.parent.mkdir(parents=True, exist_ok=True)
                            os.link(local / key, destination)
                    for day in weeks:
                        prefix = (
                            f"{namespace}/{day.isoformat()}/{architecture}/{stage}/"
                        )
                        remote = remote_objects(bucket, prefix)
                        record["remote_objects"] += len(remote)
                        record["remote_bytes"] += sum(remote.values())
                        cloud = f"gs://{bucket}/{prefix}"
                        source, destination = (
                            (cloud, str(local)) if operation == "restore"
                            else (str(upload), cloud)
                        )
                        # Missing prefixes are normal; never rsync an empty source.
                        if (operation == "restore" and remote) or (
                            operation == "save" and selected
                        ):
                            subprocess.run(
                                [
                                    "gcloud", "storage", "rsync", source, destination,
                                    "--recursive", "--no-clobber", "--quiet",
                                ],
                                check=True, timeout=TRANSFER_TIMEOUT,
                                stdout=subprocess.DEVNULL,
                            )
                after = local_objects(local)
                # Restore counts files actually added locally. Save counts the
                # planned new payload; concurrent identical writers may win a
                # race, so it is an upper bound rather than wire-byte telemetry.
                payload = (
                    {key: size for key, size in after.items() if key not in before}
                    if operation == "restore"
                    else {
                        key: size for key, size in selected.items() if key not in remote
                    }
                )
                record.update(
                    status="complete", payload_bytes=sum(payload.values()),
                    payload_objects=len(payload), local_bytes=sum(after.values()),
                    local_objects=len(after),
                )
            finally:
                record["seconds"] = round(time.monotonic() - started, 3)
                logging.info("Compile cache: %s", json.dumps(record, sort_keys=True))
        if operation == "save":
            # Existing shared buckets have other owners. Prune only this
            # namespace/architecture, after the entire week is 21 days old.
            cutoff = today - timedelta(days=CACHE_RETENTION_DAYS)
            expired = {
                parts[0] for key in remote_objects(bucket, f"{namespace}/")
                if len(parts := key.split("/", 2)) == 3
                and parts[1] == architecture
                and date.fromisoformat(parts[0]) + timedelta(days=7) <= cutoff
            }
            for day in sorted(expired):
                subprocess.run(
                    [
                        "gcloud", "storage", "rm",
                        f"gs://{bucket}/{namespace}/{day}/{architecture}/**",
                        "--recursive", "--quiet",
                    ],
                    check=True, timeout=TRANSFER_TIMEOUT,
                    stdout=subprocess.DEVNULL,
                )
    finally:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(records, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("restore", "save"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--architecture", choices=("x86_64", "aarch64"), required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--used-since-ns", type=int,
        help="Timestamp taken after cache injection and before compilation",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    transfer(
        args.operation, config["bucket"], config["namespace"], args.architecture,
        args.directory, args.output, used_since_ns=args.used_since_ns,
    )


if __name__ == "__main__":
    main()
