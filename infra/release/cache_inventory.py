"""Read GHCR manifest metadata without downloading image layers."""
import base64
import concurrent.futures
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

OWNER = "marin-community"
TOKEN = os.environ["GH_TOKEN"]
ACTOR = os.environ["GITHUB_ACTOR"]
ACCEPT = ",".join(("application/vnd.oci.image.manifest.v1+json", "application/vnd.oci.image.index.v1+json", "application/vnd.docker.distribution.manifest.v2+json", "application/vnd.docker.distribution.manifest.list.v2+json"))
MANIFEST_TYPES = set(ACCEPT.split(","))


def read_json(url, headers):
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=30) as response:
        return json.load(response), response.headers.get("Link", "")


def github_pages(path):
    url = "https://api.github.com/" + path
    result = []
    while url:
        page, link = read_json(url, {"Authorization": "Bearer " + TOKEN, "Accept": "application/vnd.github+json"})
        result.extend(page)
        url = next((part.split(";")[0].strip()[1:-1] for part in link.split(",") if 'rel="next"' in part), "")
    return result


def inventory_package(package):
    name = package["name"]
    result = {"package": name, "visibility": package.get("visibility"), "repository": (package.get("repository") or {}).get("full_name"), "complete": False}
    try:
        versions = github_pages(f"orgs/{OWNER}/packages/container/{urllib.parse.quote(name, safe='')}/versions?per_page=100")
        basic = base64.b64encode((ACTOR + ":" + TOKEN).encode()).decode()
        query = urllib.parse.urlencode({"service": "ghcr.io", "scope": f"repository:{OWNER}/{name}:pull"})
        auth, _ = read_json("https://ghcr.io/token?" + query, {"Authorization": "Basic " + basic})
        headers = {"Authorization": "Bearer " + auth["token"], "Accept": ACCEPT}

        def manifest_blobs(digest):
            manifest, _ = read_json(f"https://ghcr.io/v2/{OWNER}/{name}/manifests/{digest}", headers)
            blobs = {}
            config = manifest.get("config")
            if config:
                blobs[config["digest"]] = int(config["size"])
            for layer in manifest.get("layers", []):
                blobs[layer["digest"]] = int(layer["size"])
            for child in manifest.get("manifests", []):
                if child["mediaType"] in MANIFEST_TYPES:
                    blobs.update(manifest_blobs(child["digest"]))
                else:
                    blobs[child["digest"]] = int(child["size"])
            return blobs

        union = {}
        tagged = []
        entries = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            for version, blobs in zip(versions, pool.map(lambda version: manifest_blobs(version["name"]), versions)):
                tags = version["metadata"]["container"]["tags"]
                entry = {"id": version["id"], "digest": version["name"], "tags": tags, "created_at": version["created_at"], "referenced_blob_bytes": sum(blobs.values())}
                entries.append(entry)
                if tags:
                    tagged.append(entry)
                union.update(blobs)
        result.update(complete=True, version_count=len(versions), untagged_versions=sum(not v["metadata"]["container"]["tags"] for v in versions), unique_referenced_blob_bytes=sum(union.values()), unique_blob_count=len(union), tagged_snapshots=tagged, versions=entries)
    except urllib.error.HTTPError as error:
        result["error"] = f"HTTP {error.code} at {error.url}"
    return result


started = datetime.now(UTC).isoformat()
packages = github_pages(f"orgs/{OWNER}/packages?package_type=container&per_page=100")
candidates = [package for package in packages if "cache" in package["name"].lower()]
results = [inventory_package(package) for package in candidates]
report = {"started_at": started, "completed_at": datetime.now(UTC).isoformat(), "coverage": "Container packages readable by this repository token whose names contain cache.", "metric": "Unique compressed layer/config blob bytes referenced by retained manifests per package; not provider billing or network transfer bytes.", "packages": results}
Path("artifacts").mkdir(exist_ok=True)
Path("artifacts/cache-inventory.json").write_text(json.dumps(report, indent=2) + "\n")
for result in results:
    print(json.dumps({k: v for k, v in result.items() if k not in {"versions", "tagged_snapshots"}}))
