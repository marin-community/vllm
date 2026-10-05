"""Transport immutable sccache objects outside the credential-free builder."""

import argparse
import json
import logging
import subprocess
import time
from pathlib import Path

STAGES = ("cuda", "rust")
TRANSFER_TIMEOUT = 900


def local_objects(directory: Path) -> dict[str, int]:
    return {
        str(path.relative_to(directory)): path.stat().st_size
        for path in directory.rglob("*")
        if path.is_file() and not path.is_symlink()
    }


def remote_objects(bucket: str, prefix: str) -> dict[str, int]:
    result = subprocess.run(
        [
            "gcloud", "storage", "objects", "list", f"gs://{bucket}",
            "--format=json(name,size)", "--quiet",
        ],
        check=True, capture_output=True, text=True, timeout=120,
    )
    return {
        item["name"][len(prefix):]: int(item["size"])
        for item in json.loads(result.stdout)
        if item["name"].startswith(prefix)
    }


def transfer(
    operation: str, bucket: str, namespace: str, architecture: str,
    directory: Path, output: Path,
) -> None:
    """Merge compiler objects without deleting or replacing remote variants.

    Records payload bytes and elapsed transport time. A failed transfer raises
    after recording its status; workflow cache steps are explicitly best effort.
    """
    records = []
    try:
        for stage in STAGES:
            started = time.monotonic()
            local = directory / stage
            local.mkdir(parents=True, exist_ok=True)
            prefix = f"{namespace}/{architecture}/{stage}/"
            record = {
                "stage": stage, "operation": operation,
                "status": "failed", "payload_bytes": 0,
            }
            records.append(record)
            try:
                before = local_objects(local)
                remote = remote_objects(bucket, prefix)
                record["remote_objects"] = len(remote)
                record["remote_bytes"] = sum(remote.values())
                cloud = f"gs://{bucket}/{prefix}"
                source, destination = (
                    (cloud, str(local)) if operation == "restore"
                    else (str(local), cloud)
                )
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
                        key: size for key, size in before.items() if key not in remote
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
    finally:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(records, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("restore", "save"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--architecture", choices=("x86_64", "aarch64"), required=True)
    parser.add_argument(
        "--namespace", help="Owned test namespace; normally config namespace"
    )
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    namespace = args.namespace or config["namespace"]
    if not namespace or any(
        char not in "abcdefghijklmnopqrstuvwxyz0123456789-" for char in namespace
    ):
        raise ValueError("Namespace allows only lowercase letters, digits or hyphens")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    transfer(
        args.operation, config["bucket"], namespace, args.architecture,
        args.directory, args.output,
    )


if __name__ == "__main__":
    main()
