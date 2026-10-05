# GPU compiler cache

The candidate workflow merges local CUDA and Rust sccache objects from a shared
GCS bucket before each build, then uploads new objects after compilation. Branch
names do not participate in storage paths. Architecture and stage do; sccache
0.8.1 still checks compiler, flags, source/header contents and dependencies.
Rust checks source files, extern crates, static libraries and tracked environment
inputs. Returning to an older compatible input can reuse its objects.

`compile-cache.json` supplies the bucket, namespace and identity provider.
The independent infrastructure root is
[`yonromai/infra/cloud/vllm-cache`](https://github.com/yonromai/infra/tree/vllm-cache-storage-01a10d2b/cloud/vllm-cache).
It declares Standard storage in us-central1, private access, object age deletion
after 14 days, and no soft deletion or versioning. Lifecycle deletion is
asynchronous. Reads do not refresh object age, so even hot objects eventually
expire and compile again. Loss of this disposable storage needs only a rebuild.

## Access and transport

The identity provider admits the numeric vLLM repository and Marin owner IDs,
from the candidate workflow on branches. Main pushes and manual branch builds
can authenticate; PR and tag events cannot. Repository writers control branch
workflows and are trusted cache producers. The bucket grants object read/list
and create permissions. It grants no replacement or deletion permission.

The runner obtains a short-lived federated token immediately before each
transfer. The token exists only in those steps' environment. Auth creates no
credential file, and BuildKit receives no cloud credentials. A cache failure is
reported and normal compilation continues. Cache export changes a nonce argument
on each run so it reads current mounts rather than an earlier exported layer.
Do not use `--no-cache`: BuildKit clears the cache mounts when it is set. The
legacy main GitHub CUDA snapshot is read as a bootstrap; its contents and retention are
unchanged.

`cache-restore.json` and `cache-save.json` report per-stage elapsed seconds,
object counts and payload bytes. Restore bytes count objects actually added
locally. Save bytes are an upper bound on new payload; an identical concurrent
writer may win the upload race. Neither count includes HTTP overhead. Native
steps print `compile-stage <stage> seconds=<seconds>` and sccache hit/miss
statistics in separate steps to avoid BuildKit's long-step log limit. The
workflow retains the full build log and total build/export time for 14 days.

## Storage choice and costs

GitHub cache restore can read the current and default branches, but cannot
read a sibling branch. Changing restore keys does not remove that restriction.
Snapshots are immutable and duplicate the object collection for each commit.
The October 5 inventory had 28 entries totaling 4.60 GiB, mostly snapshots of
the same native outputs. See
[GitHub's cache scope](https://docs.github.com/en/actions/reference/workflows-and-actions/dependency-caching#restrictions-for-accessing-a-cache).

GCS stores each sccache object once per architecture/stage namespace. New
variants accumulate until age eviction, and branch concurrency cannot discard
another branch's variants. Bulk transfers keep credentials outside the builder
and avoid a compiler daemon holding an expired token during a long build.
Direct sccache GCS access would transfer only requested objects but requires
credential refresh throughout compilation. Selective build-layer caching could
skip unchanged Rust stages, but transporting the dependency/toolchain layers is
much larger than compiler objects. Upstream uses warm builders and ECR caches;
this fork uses disposable hosted runners. See
[upstream's builder](https://github.com/vllm-project/vllm/blob/main/.buildkite/image_build/image_build.sh).

Use the measured cache volume and restored bytes to budget at the current
[GCS prices](https://cloud.google.com/storage/pricing): roughly $0.02 per
GiB-month for Standard storage in us-central1, plus request and network charges.
For example, 5 GiB retained and 50 monthly 1 GiB downloads at $0.12/GiB cost
about $6.10/month before requests. GitHub runner placement can change transfer
pricing. This is a planning example, not measured workload consumption. Set
build frequency from actual usage; retention alone does not cap traffic cost.

## Probes and recovery

Dispatch the existing candidate workflow with `gpu_mode=probe-x86_64` or
`probe-aarch64` and `probe_expect=miss` for an empty probe namespace. After the
first success, dispatch from a second trusted branch with `probe_expect=hit`
and the same `probe_namespace`. Choose a new namespace for a new cold/warm pair.
The probe uses the configured native builder, NVCC, Rust 1.95 and sccache 0.8.1.
Its NVCC fixture uses stable object intermediates for byte comparison; this
does not change production flags or establish production reproducibility.
It compares cached objects with independent compiler output and checks misses
after source/header, compiler and flag changes, then hits after reverting.
Each run uses fresh input variants for its expected misses, so further warm
probes remain valid. Probe data lives in its chosen namespace and expires with the bucket.
The probe stops on backend failure; production wheel builds continue.

For manual inspection, use `gcloud storage du --summarize` on the owned bucket.
To stop reuse without changing ABI, select a new `namespace` in the config.
Only the cache owner should delete old task data. A new namespace forces a cold
build and does not repair an invalid compiler/toolchain input.

## Serving refresh handoff

This change starts from main `39e62869693c46402b1a95fde4fc55ca1aab9ae1`.
The inspected serving recipe is `172b6f9773734e726d4004d84fa96b55746ef9d5`.
Apply the workflow's GCS transfer/auth and explicit mount export, the config and
`compile_cache.py`, and the Dockerfile's Rust compiler mount/wrapper and timing
blocks. The probes and this guide are optional operating tools; the existing
release test gains the new nonpublishing probe modes.

Keep the serving recipe's fixed CUDA compiler provenance, source date handling,
Rust C date header, dependency constraints and stable CUDA object intermediates.
Those change legitimate cache inputs and must retain their own compatibility
checks. Merge the Rust setup shell blocks so its CFLAGS/CXXFLAGS are established
before starting sccache/build_rust, and retain its provenance export stage.
Do not replace its Dockerfile with this older base recipe. Integrate and validate
on that owner's branch after these cache tests; do not publish or promote as
part of the cache handoff.
