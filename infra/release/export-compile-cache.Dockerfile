# The candidate's already-pulled builder supplies cp and a shell. No compiler
# layers or credentials are exported. Invoke with --no-cache after compilation.
ARG CACHE_EXPORT_IMAGE
FROM ${CACHE_EXPORT_IMAGE} AS export
RUN --mount=type=cache,target=/root/.cache/sccache,sharing=locked \
    --mount=type=cache,target=/root/.cache/sccache-rust,sharing=locked \
    mkdir -p /export/cuda /export/rust \
    && cp -a /root/.cache/sccache/. /export/cuda/ \
    && cp -a /root/.cache/sccache-rust/. /export/rust/

FROM scratch
COPY --from=export /export/ /
