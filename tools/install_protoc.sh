#!/usr/bin/env bash
set -euo pipefail

# Install a pinned protoc binary from upstream GitHub releases.
#
# Distro protobuf-compiler packages vary widely in version (e.g.
# AlmaLinux/RHEL 8 ships protoc 3.5, predating the
# --experimental_allow_proto3_optional flag the rust frontend's build.rs
# passes), so we pin the protoc version here instead.
#
# Override the version via the PROTOC_VERSION env var.
# Requires: curl, unzip, root privileges.

if [[ $(id -u) -ne 0 ]]; then
  echo "Must be run as root" >&2
  exit 1
fi

VERSION="${PROTOC_VERSION:-34.2}"

ARCH="$(uname -m)"
case "${ARCH}" in
    # protoc release archives use "aarch_64" (with an underscore), not
    # "aarch64". Don't "fix" this.
    aarch64|arm64) URL_ARCH="aarch_64" ;;
    x86_64|amd64)  URL_ARCH="x86_64" ;;
    *) echo "Unsupported arch for protoc binary: ${ARCH}" >&2; exit 1 ;;
esac

URL="https://github.com/protocolbuffers/protobuf/releases/download/v${VERSION}/protoc-${VERSION}-linux-${URL_ARCH}.zip"
TMPDIR="$(mktemp -d)"
trap 'rm -rf "${TMPDIR}"' EXIT

echo "Downloading: ${URL}"
# Retry transient network failures; a bare one-shot curl flaked (exit 55) on a
# cold CI build where this download runs uncached every time. A bash loop covers
# every curl error and stays portable across the base image's older curl (which
# lacks --retry-all-errors).
downloaded=0
for attempt in 1 2 3 4 5; do
    if curl -fsSL --connect-timeout 30 -o "${TMPDIR}/protoc.zip" "${URL}"; then
        downloaded=1
        break
    fi
    echo "protoc download attempt ${attempt} failed; retrying in 2s..." >&2
    sleep 2
done
[[ "${downloaded}" -eq 1 ]] || { echo "protoc download failed after 5 attempts" >&2; exit 1; }
unzip -q -o "${TMPDIR}/protoc.zip" -d /usr/local
echo "Installed $(protoc --version)"
