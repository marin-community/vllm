# The candidate's already-pulled builder supplies cp and a shell. No compiler
# layers or credentials are exported. Change CACHE_EXPORT_NONCE on each export
# to read current mounts. --no-cache would clear those mounts before reading.
ARG CACHE_EXPORT_IMAGE
FROM ${CACHE_EXPORT_IMAGE} AS export
ARG CACHE_EXPORT_NONCE
RUN --mount=type=cache,target=/root/.cache/sccache,sharing=locked \
    --mount=type=cache,target=/root/.cache/sccache-rust,sharing=locked \
    test -n "$CACHE_EXPORT_NONCE" && mkdir -p /export/cuda /export/rust \
    && cp -a /root/.cache/sccache/. /export/cuda/ \
    && cp -a /root/.cache/sccache-rust/. /export/rust/

FROM scratch
COPY --from=export /export/ /
