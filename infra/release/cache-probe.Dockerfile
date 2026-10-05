ARG BUILD_BASE_IMAGE
FROM ${BUILD_BASE_IMAGE} AS probe
ARG TARGETARCH
ARG SCCACHE_VERSION=0.8.1
ARG RUST_TOOLCHAIN
RUN dnf install -y --setopt=install_weak_deps=False clang \
    && dnf clean all \
    && case "$TARGETARCH" in amd64) SCCACHE_ARCH=x86_64 ;; arm64) SCCACHE_ARCH=aarch64 ;; esac \
    && curl -fsSL "https://github.com/mozilla/sccache/releases/download/v${SCCACHE_VERSION}/sccache-v${SCCACHE_VERSION}-${SCCACHE_ARCH}-unknown-linux-musl.tar.gz" | tar -xz \
    && mv sccache-v${SCCACHE_VERSION}-${SCCACHE_ARCH}-unknown-linux-musl/sccache /usr/local/bin/ \
    && curl -fsSL https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain "$RUST_TOOLCHAIN"
ENV PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
ENV SCCACHE_IDLE_TIMEOUT=0
WORKDIR /probe
RUN /opt/python/cp312-cp312/bin/python3.12 -m venv .venv
COPY compiler_probe.py compiler_probe.py
ARG PROBE_EXPECT=miss
ARG PROBE_VARIANT
RUN --mount=type=cache,target=/root/.cache/sccache,sharing=locked \
    --mount=type=cache,target=/root/.cache/sccache-rust,sharing=locked \
    .venv/bin/python compiler_probe.py --expect "$PROBE_EXPECT" --variant "$PROBE_VARIANT" --output /results/probe.json

FROM scratch
COPY --from=probe /results/ /
