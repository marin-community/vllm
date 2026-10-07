FROM rust:1.85-slim AS base
RUN apt-get update && apt-get install -y --no-install-recommends gcc ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY sccache /usr/local/bin/sccache
FROM base AS compile
COPY probe.c probe.rs /src/
WORKDIR /src
ARG CACHE_PREFIX
ARG CACHE_BUCKET=marin-public
ENV SCCACHE_GCS_BUCKET=${CACHE_BUCKET} \
    SCCACHE_GCS_KEY_PREFIX=${CACHE_PREFIX} \
    SCCACHE_GCS_RW_MODE=READ_WRITE \
    SCCACHE_GCS_KEY_PATH=/run/secrets/gcs \
    SCCACHE_IGNORE_SERVER_IO_ERROR=1
RUN --mount=type=secret,id=gcs \
    mkdir -p /out \
    && sccache --zero-stats \
    && sccache gcc -c probe.c -o /out/probe.o \
    && sccache rustc --crate-type=rlib probe.rs -o /out/libprobe.rlib \
    && sccache --show-stats --stats-format=json > /out/stats.json \
    && sccache --stop-server
FROM scratch
COPY --from=compile /out/ /
