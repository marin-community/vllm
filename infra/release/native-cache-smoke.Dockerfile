FROM rust:1.85-slim AS base
RUN apt-get update && apt-get install -y --no-install-recommends gcc ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY sccache /usr/local/bin/sccache
FROM base AS compile
COPY probe.c probe.rs Cargo.toml /src/
WORKDIR /src
ARG CACHE_PREFIX
ARG CACHE_BUCKET=marin-public
ENV SCCACHE_GCS_BUCKET=${CACHE_BUCKET} \
    SCCACHE_GCS_KEY_PREFIX=${CACHE_PREFIX} \
    SCCACHE_GCS_RW_MODE=READ_WRITE \
    SCCACHE_GCS_KEY_PATH=/run/secrets/gcs-credentials \
    SCCACHE_IGNORE_SERVER_IO_ERROR=1
RUN --mount=type=secret,id=gcs-credentials,required=false \
    if [ -n "${SCCACHE_GCS_KEY_PATH}" ] && [ ! -f "$SCCACHE_GCS_KEY_PATH" ]; then unset SCCACHE_GCS_BUCKET SCCACHE_GCS_KEY_PREFIX SCCACHE_GCS_RW_MODE SCCACHE_GCS_KEY_PATH; fi \
    && mkdir -p /out \
    && sccache --zero-stats \
    && sccache gcc -c probe.c -o /out/probe.o \
    && RUSTC_WRAPPER=sccache cargo build --release --offline --target-dir /out \
    && sccache --show-stats --stats-format=json > /out/stats.json \
    && sccache --stop-server
FROM scratch
COPY --from=compile /out/ /
