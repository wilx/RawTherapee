# X-Trans Student-t GMR corpus and training tool

`rt-tgmr-train` is a standalone C++17 development tool for constructing,
auditing, fitting, and exporting RawTherapee's phase-conditioned X-Trans
Student-t GMR model. It is deliberately independent of `rtengine` and the GUI.
It uses only OpenMP, zlib, JPEG, PNG, TIFF, LittleCMS, and the small cJSON copy
already present in RawTherapee. Python is needed only for the optional
standard-library source downloader; it is not a training or runtime dependency.

This directory does **not** contain a production corpus or reviewed production
model yet. The original 200-image BSDS research corpus cannot be redistributed
or used to license a bundled production model. The release CMake gate remains
closed until the 5,000-source licensed corpus has been selected, independently
reviewed, packed, trained, and validated.

## Build

Ubuntu development packages:

```sh
sudo apt install build-essential cmake ninja-build pkg-config \
    libjpeg-dev libpng-dev libtiff-dev liblcms2-dev zlib1g-dev libomp-dev
```

Standalone GCC build:

```sh
cmake -S tools/tgmr_trainer -B /tmp/rt-tgmr-trainer \
    -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/rt-tgmr-trainer -j4
ctest --test-dir /tmp/rt-tgmr-trainer --output-on-failure
```

RawTherapee-tree build:

```sh
cmake --preset dev -DBUILD_TGMR_TRAINER=ON
cmake --build build/dev -j4 --target rt-tgmr-train
```

The canonical model-publishing backend is selected at run time with
`--backend canonical`. It uses one thread, a fixed reduction order, float64
fitting, and compiler contraction disabled. `--backend cpu` uses deterministic
per-thread sufficient statistics and fixed thread-order reduction. Its fitted
parameters need only satisfy the frozen validation tolerance; it is not the
official byte-identity producer. The `omp-target` name is reserved and currently
requires a separately compiled AMDGCN/NVPTX build. Builds without such a device
image fail closed. The target implementation rejects host fallback, keeps model
and sample arrays mapped while processing bounded batches, and offloads
likelihood, latent-weight, and sufficient-statistic reductions. It remains
experimental until the million-patch speed and image-quality gate passes.

Example AMDGCN configuration (the compiler and ROCm device bitcode must have
compatible LLVM versions):

```sh
cmake -S tools/tgmr_trainer -B /tmp/rt-tgmr-amdgcn -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang \
    -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ \
    -DOpenMP_CXX_FLAGS=-fopenmp -DOpenMP_CXX_LIB_NAMES=omp \
    -DOpenMP_omp_LIBRARY=/opt/rocm/llvm/lib/libomp.so \
    -DTGMR_ENABLE_OMP_TARGET=ON \
    -DTGMR_OMP_TARGET_FLAGS='-fopenmp -fopenmp-targets=amdgcn-amd-amdhsa --offload-arch=gfx1101 --rocm-path=/opt/rocm'
cmake --build /tmp/rt-tgmr-amdgcn -j4
```

NVPTX uses the corresponding compiler-supported target triple and
`--offload-arch=sm_XX`. Local invocations select a device with `--device N` and
may yield between batches using `--gpu-yield-ms N`; Unix niceness does not
schedule GPU work.

## Source manifest and reconstruction

The reviewed source list is a JSON Lines file governed by
[`corpus-source-manifest-v2.schema.json`](corpus-source-manifest-v2.schema.json).
The original [`corpus-v1.schema.json`](corpus-v1.schema.json) remains readable
for development fixtures. Each line is one source and line order is part of the
corpus identity. V2 additionally freezes the catalog snapshot, upstream source
and author identities, rights evidence, archive-member fallbacks, content tags,
people review, selection state, dHash, DCT pHash, and image histograms.

The frozen production mix is:

| Catalog | Candidate pool | Train | Validation | Test |
| --- | ---: | ---: | ---: | ---: |
| CVDF Open Images V4/V5 boxable subset | 10,000 | 2,000 | 250 | 250 |
| PASS | 6,000 | 1,200 | 150 | 150 |
| Wikimedia Commons | 3,000 | 480 | 60 | 60 |
| Smithsonian Open Access | 2,000 | 320 | 40 | 40 |

Catalog locations and immutable-snapshot procedure are recorded in
[`catalog-acquisition-v1.json`](catalog-acquisition-v1.json); selection policy
and exact quotas are frozen in
[`corpus-selection-v1.json`](corpus-selection-v1.json). Catalog metadata is
evidence, not automatic permission. Every selected record must have an approved
rights review and may use only CC0, Public Domain Mark, or CC BY 2.0/3.0/4.0.
NC, ND, SA, unknown, ambiguous, and Smithsonian records lacking explicit CC0
are rejected. People records additionally require an explicit non-sensitive,
no-obvious-minors review.

The CVDF Open Images tar mirror contains the 1,743,042-image V4/V5 boxable
training subset, not the complete roughly nine-million-image catalog. Its
candidate metadata comes from the matching
`train-images-boxable-with-rotation.csv` snapshot (638,407,721 bytes, SHA-256
`05f3d68dbbb03728d1a37e51479f4f35c062b871e1a6cae8c4cefbe0e0c80ed0`).
Open Images uses
`OriginalURL` and its advertised checksum as provenance; its changing
thumbnail URL is never canonical. PASS uses the official individual URL and
may name an authenticated Zenodo archive/member as a byte-identical fallback.
PASS's metadata `hash` is a variable-length hexadecimal source/filename
identity, not an image-content MD5; downloaded originals receive a local
SHA-256 before they become corpus inputs.
Commons freezes the upload revision, original URL, API SHA-1, and local SHA-256.
Smithsonian freezes exact anonymous Open Data on AWS index and metadata-shard
bytes, then selects a named high-resolution JPEG rendition. Both the record's
metadata usage and the selected image's media usage must be exactly CC0. No
Smithsonian REST API key or AWS account is required.

Reconstruct or audit the exact listed originals without third-party Python
modules:

```sh
python3 tools/tgmr_trainer/reconstruct_corpus.py \
    corpus-v1.jsonl /data/tgmr/source-cache --retry 2 --report /tmp/download.json
python3 tools/tgmr_trainer/reconstruct_corpus.py \
    corpus-v1.jsonl /data/tgmr/source-cache --offline-verify
python3 tools/tgmr_trainer/reconstruct_corpus.py \
    corpus-v1.jsonl --emit-fetch-list corpus-v1-fetch.tsv
```

The downloader reads in manifest order, writes `.part` files, retries transient
failures, publishes only bytes having the frozen SHA-256, and never silently
substitutes a changed fallback. V2 also authenticates complete PASS archives
and the selected archive member before publishing it under the source's frozen
identity. Original reconstruction is best effort because third-party URLs can
disappear. The authenticated TGPC patch corpus is the durable training input.

## Corpus workflow

The catalog preparation program uses only the Python standard library. Network
APIs are allowed only while creating a frozen snapshot; canonical rebuilds
consume the snapshot bytes and their pinned SHA-256, never a live API result.

```sh
TOOL=/tmp/rt-tgmr-trainer/rt-tgmr-train
CACHE=/data/tgmr/source-cache
PREP=tools/tgmr_trainer/prepare_corpus.py

# Download and authenticate the four catalog snapshots. The reviewed release
# records their actual SHA-256 values; examples omit them because this tree does
# not contain the external snapshots.
python3 "$PREP" snapshot \
    https://storage.googleapis.com/openimages/2018_04/train/train-images-boxable-with-rotation.csv \
    oi.csv
python3 "$PREP" snapshot PASS_METADATA_URL pass.csv
python3 "$PREP" snapshot PASS_URL_LIST pass-urls.txt

# Commons identifier lists are curated inputs. Smithsonian is collected from
# explicitly selected units and hexadecimal metadata shards in its public AWS
# bucket. Both commands freeze their live inputs; normalization is offline.
python3 "$PREP" collect-commons commons-file-titles.txt commons-api.jsonl
python3 "$PREP" collect-smithsonian smithsonian-aws.jsonl \
    --unit chndm --unit fsg --unit nmah --unit nmnhbirds --unit nmnhbotany \
    --unit nmnhento --unit nmnhminsci --unit nmnhpaleo --unit npm --unit saam \
    --prefix 00 --prefix 01 --prefix 02 --prefix 03 --per-unit-limit 200 \
    --snapshot-dir smithsonian-aws-metadata \
    --report smithsonian-aws-report.json

python3 "$PREP" normalize openimages oi.csv oi-candidates.jsonl \
    --revision cvdf-open-images-v5-boxable-05f3d68dbbb0 \
    --snapshot-sha256 \
    05f3d68dbbb03728d1a37e51479f4f35c062b871e1a6cae8c4cefbe0e0c80ed0 \
    --eligible-only --limit 10000
python3 "$PREP" normalize pass pass.csv pass-candidates.jsonl \
    --revision PASS_REVISION --snapshot-sha256 SHA256 --urls pass-urls.txt \
    --archive-index pass-archive-members.json --limit 6000
python3 "$PREP" normalize commons commons-api.jsonl commons-candidates.jsonl \
    --revision COMMONS_SNAPSHOT_REVISION --snapshot-sha256 SHA256 --limit 3000
python3 "$PREP" normalize smithsonian smithsonian-aws.jsonl \
    smithsonian-candidates.jsonl --revision SMITHSONIAN_SNAPSHOT_REVISION \
    --snapshot-sha256 SHA256 --limit 2000
python3 "$PREP" merge candidates.jsonl oi-candidates.jsonl \
    pass-candidates.jsonl commons-candidates.jsonl smithsonian-candidates.jsonl

# Download originals, preserving authenticated cache names and a machine-
# readable availability report. Classification is performed by the C++ tool.
python3 "$PREP" fetch candidates.jsonl "$CACHE" fetched.jsonl \
    --retry 2 --report fetch-report.json
"$TOOL" corpus classify fetched.jsonl "$CACHE" classifications.jsonl \
    --candidates

# Human review records bind rights evidence, content tags, and the controlled
# people review. They may also provide normalized_author_id when one person is
# represented differently in multiple catalogs. Assembly refuses mismatched
# source/cache identities.
python3 "$PREP" assemble fetched.jsonl classifications.jsonl "$CACHE" \
    reviewed-candidates.jsonl --reviews reviews.jsonl
"$TOOL" corpus report reviewed-candidates.jsonl --sources \
    --json candidate-statistics.json --csv candidate-statistics.csv \
    --html candidate-statistics.html

# Selection assigns whole normalized-author groups to one split, enforces exact
# per-catalog quotas and the author cap, rejects exact/perceptual duplicates,
# and balances the 27 training-derived brightness/chroma/texture cells.
"$TOOL" corpus select reviewed-candidates.jsonl \
    tools/tgmr_trainer/corpus-selection-v1.json selected-sources.jsonl \
    > selection-report.json

# Finalization creates 256 train or 128 validation/test fixed coordinates per
# source, with 75% spatial and 25% coverage samples and frozen augmentations.
"$TOOL" corpus finalize selected-sources.jsonl "$CACHE" corpus-v1.jsonl
"$TOOL" corpus validate-manifest corpus-v1.jsonl
"$TOOL" corpus verify-sources corpus-v1.jsonl "$CACHE"
"$TOOL" corpus pack corpus-v1.jsonl "$CACHE" tgmr-corpus-v1.tgpc
"$TOOL" corpus pack corpus-v1.jsonl "$CACHE" \
    tgmr-corpus-v1-sensor-noise.tgpc --noise sensor-v1
"$TOOL" corpus gzip tgmr-corpus-v1.tgpc tgmr-corpus-v1.tgpc.gz
"$TOOL" corpus inspect tgmr-corpus-v1.tgpc.gz
"$TOOL" corpus balance tgmr-corpus-v1.tgpc.gz
"$TOOL" corpus report tgmr-corpus-v1.tgpc.gz \
    --json corpus-statistics.json --csv corpus-statistics.csv \
    --html corpus-statistics.html

python3 "$PREP" release-manifest corpus-v1.jsonl tgmr-corpus-v1.tgpc \
    tgmr-corpus-v1.tgpc.gz CORPUS-NOTICE.txt corpus-statistics.json \
    rights-report.json tgmr-corpus-v1.release.json \
    --zenodo-doi DOI --github-release-url URL
```

`collect-smithsonian` uses standard-library HTTPS directly against the public
bucket. It streams every chosen shard, records SHA-256 and byte size for the
root index, unit indexes, and shards in the report, and emits only compact
eligible records. `--all-shards` may replace the explicit `--prefix` list for a
complete unit scan. `--snapshot-dir` preserves every upstream byte under stable
relative names. A later run with the same arguments plus `--offline-snapshot`
must reproduce the compact JSONL and report byte-for-byte without network
access. The REST command remains available as
`collect-smithsonian-api` for diagnostics involving an explicit record-ID list;
it is not part of the canonical corpus recipe.

The stdout from each `snapshot`, `collect-*`, `normalize`, `fetch`, `select`,
`balance`, and `release-manifest` command is canonical JSON and should be saved
with the release audit. A PASS archive-index JSONL line contains `hash`, `url`,
`sha256`, `member`, and `member_sha256`; extraction rejects absolute paths,
parent traversal, changed archives, and changed members.

Human review input is canonical JSONL. A typical line is:

```json
{"content_tags":["people","skin-hair-clothing"],"format":"rawtherapee-tgmr-source-review-v1","normalized_author_id":"flickr-user:stable-identity","people_review_status":"approved-no-minors-or-sensitive-content","rights_evidence_revision":"reviewed-upload-revision","rights_evidence_sha256":"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef","rights_evidence_url":"https://example.invalid/source-rights-page","rights_review_status":"approved","source_id":"openimages-cvdf-v5-boxable:source-id"}
```

`normalized_author_id` is optional, but it is required when catalog metadata
uses different identifiers for the same author. The preparation tool already
normalizes recognizable Flickr profile identities shared by Open Images and
PASS; manual review resolves remaining cross-catalog aliases.

The packer decodes and color-manages into linear sRGB, then applies frozen
exposure, white-balance, and train/evaluation-separated Fujifilm camera-matrix
augmentations. It emits 7x7 planar RGB patches as little-endian uint16. Packing
streams one source and one record at a time. The `.tgpc.gz` member is ordinary
gzip made through RawTherapee's existing required zlib dependency, with zero
timestamp, no filename/comment, fixed compression level, and a canonical OS
byte. Compressed input is streamed directly by the trainer.

`corpus pack` defaults to the frozen `none` recipe: transformed values are
bounded and quantized, but no synthetic noise is added. `--noise sensor-v1`
adds deterministic bounded integer-domain read and signal-dependent noise to
the 75% augmented patches while leaving identity patches unchanged. Its
nominal standard deviations are approximately 8 uint16 codes for read noise
and 64 codes at saturation for the signal term. The TGPC configuration digest
and per-record augmentation kind distinguish the recipes. This is a validation
candidate, not a claim about a particular camera's calibrated noise model.

`corpus classify-file` remains a useful intake/debugging form for one image.
`corpus classify --candidates` is the canonical batch path: it authenticates
the Python fetch output, decodes every available candidate, and produces the
classification input for assembly. `--all` classifies selected and unselected
records from an already assembled V2 manifest. Batch classification uses a
bounded C++ worker pool (`--jobs 4` by default). Each JPEG is read once: the
exact compressed bytes are SHA-256 authenticated and then decoded from memory.
Nested cache names are supported for sharded stores, but absolute paths,
`.`/`..` components, and symlink escapes outside the supplied cache are
rejected.

Long runs are restartable without changing their final bytes. By default the
tool writes immutable, authenticated result segments under
`OUTPUT.jsonl.work`, checkpoints every 128 completed images or 120 seconds,
and reports progress every 15 seconds. Put this work directory on a local
filesystem when the source cache is a network share:

```sh
rt-tgmr-train corpus classify fetched.jsonl /data/open-images \
    classifications.jsonl --candidates --jobs 4 \
    --work-dir /var/tmp/tgmr-classify-work \
    --checkpoint-images 128 --checkpoint-seconds 120 \
    --progress-seconds 15 --retries 2
```

Rerunning the same command validates and resumes its segments. Incomplete
`.tmp` files are ignored. A changed input list or full/proxy mode is rejected
instead of being combined with old state. Final output is merged in original
input order and is byte-identical across thread counts and interruptions.
Unresolved per-image failures are recorded in `failures.json`; successful
images remain checkpointed for the next run. Candidate-pool screening may use
the explicit `--allow-failures` option after reviewing that file: the final
classification JSONL then contains only successful records, while the failure
report remains beside the checkpoints as the rejection audit. The default
continues to fail closed, and reviewed/final corpus verification must not use
this option to hide missing selected sources.

For metadata shortlisting, `--proxy` asks libjpeg for a 1/8-resolution decode.
Its rows use the distinct
`rawtherapee-tgmr-image-proxy-classification-v1` identity and record both source
and proxy dimensions. Proxy results are approximate, JPEG-only, and must not
be passed to `assemble`; candidates selected from them require the normal full
classification pass. RGB and grayscale embedded ICC profiles are both honored;
grayscale JPEGs are expanded to neutral linear RGB through their one-component
profile rather than being incorrectly presented to LittleCMS as RGB-profile
input.

Bulk CVDF Open Images archives do not need a separate per-file SHA-256 pass.
The tar files contain only the V4/V5 boxable subset and therefore must be paired
with the matching boxable image-information snapshot, not the V6/V7
human-verified-label metadata. After metadata filtering has produced canonical
catalog-candidate JSONL, point the classifier at the extracted split root:

```sh
rt-tgmr-train corpus classify oi-shortlist.jsonl /data/open-images \
    oi-proxy.jsonl --open-images-cvdf train --proxy --jobs 4 \
    --work-dir /var/tmp/tgmr-oi-proxy
```

For an image ID `abcdef0123456789`, this mode requires
`train/a/b/c/abcdef0123456789.jpg`. It authenticates and decodes the same byte
buffer and records `source_sha256` in the classification row. Run the retained
shortlist again without `--proxy` for final metadata. `prepare_corpus.py
assemble` accepts the original catalog-candidate JSONL for this local mode,
requires the full classification's source digest, and reauthenticates the file
before creating the V2 source manifest. Because CVDF images are rescaled mirror
renditions rather than the Flickr originals, assembly records the matching
`https://open-images-dataset.s3.amazonaws.com/SPLIT/ID.jpg` object as the
reconstruction URL and does not apply the original-image MD5 to those bytes.

The intended large-catalog sequence is metadata/license/author filtering
first, proxy classification of a roughly 50,000-image shortlist second, and
full classification of only the roughly 10,000--16,000 candidates retained for
deduplication and balanced source selection. The tool can process a larger
list, but scanning every Open Images file is neither required nor the frozen
corpus recipe.

`corpus balance` reports the declared low/middle/high brightness, chroma, and
texture strata using cut points derived only from training patches. The old
fixed research thresholds remain available as `--fixed-v1-thresholds`.
`corpus report --sources` emits source/catalog/license/rights/tag and histogram
statistics before packing; ordinary `corpus report` emits canonical TGPC JSON,
CSV, and self-contained HTML. Final release files are published outside Git as
identical `.tgpc.gz` bytes on Zenodo and a RawTherapee/GitHub release. The Git
tree retains their manifest, DOI/URL, hashes, source JSONL, URL/checksum list,
statistics, and attribution notice.

Selection deliberately fails if the reviewed pool cannot satisfy a catalog,
split, people, author, quality, or deduplication constraint. The remedy is to
expand the corresponding candidate source, never to relax licensing or reuse
duplicates. Manual inspection remains required for rights evidence, borderline
near-duplicate clusters, controlled people content, and guardrail categories.
The program cannot turn a syntactically valid manifest into a legal conclusion.

## Fitting, checkpoints, export, and benchmark

```sh
"$TOOL" train tgmr-corpus-v1.tgpc.gz /data/tgmr/checkpoints \
    --backend canonical --components 32 --gaussian-iterations 10 \
    --student-iterations 30 --batch-size 4096 --maximum-memory-mib 8192

"$TOOL" resume tgmr-corpus-v1.tgpc.gz \
    /data/tgmr/checkpoints/phase-00-final.tgmrc \
    /data/tgmr/checkpoints/phase-00-resumed.tgmrc \
    --additional-student 1 --backend canonical

"$TOOL" benchmark tgmr-corpus-v1.tgpc.gz \
    --backend cpu --samples 102400 --phase 0 --components 32 \
    --gaussian-iterations 10 --student-iterations 30 --batch-size 4096
```

The frozen 250/500/1000/2000/4000-source convergence curve is generated from
the same TGPC by passing `--source-limit N` to `train`. Sources are selected in
their first-appearance order in the authenticated corpus; the limit and actual
sample count are stored in every checkpoint and companion manifest. `resume`
rejects a different limit. This avoids repacking subtly different curve
corpora while retaining one corpus payload identity.

For a target build, replace `--backend cpu` with `--backend omp-target` and add
`--device N`. A physical-device smoke test is not the release performance gate:
the backend is retained only if the production run is at least 1.5x faster than
optimized CPU and its exported model remains within 0.02 dB on validation.

Training uses 18 phase-conditioned 51-dimensional observations, deterministic
k-means++, full-covariance Gaussian EM initialization, covariance floor 1e-6,
and fixed-nu Student-t ECM with nu=3. Checkpoints are authenticated and atomic.
The final export specializes each phase/component into the fixed S9/top-8,
tau=0.0003, temperature-4 inference contract and emits the TGMR v2 model plus a
canonical companion manifest:

```sh
"$TOOL" export --checkpoints /data/tgmr/checkpoints model.tgmr \
    --corpus-sha256 HEX --configuration-sha256 HEX \
    --attribution-sha256 HEX --trainer-revision-sha256 HEX
"$TOOL" verify model.tgmr
"$TOOL" validate model.tgmr tgmr-corpus-v1.tgpc.gz \
    --split validation > model.validation.json
# Re-exporting the deterministic model bytes attaches the authenticated
# validation result to the final companion manifest.
"$TOOL" export --checkpoints /data/tgmr/checkpoints model.tgmr \
    --corpus-sha256 HEX --configuration-sha256 HEX \
    --attribution-sha256 HEX --trainer-revision-sha256 HEX \
    --validation-report model.validation.json --force
```

`validate` applies the exact scalar K32/S9/q8 inference equations to every
selected patch through all 18 X-Trans phases, restores the measured center
sample, and reports pooled/phase/source PSNR plus p99 and worst patch RMS. This
is the native measurement used for source-count curves, augmentation selection,
canonical/accelerated parity, and the frozen validation/test record.

The release model must be produced twice in clean pinned canonical environments
with byte-identical corpus, checkpoint/export inputs, model, and manifest. An
optimized CPU or future OpenMP-target fit is acceptable only after the specified
PSNR/image-quality parity tests; it can never publish the official artifact.
Release CMake configuration rejects a companion manifest whose validation
object is absent or whose model/corpus identities do not match.
The standalone validation report records elapsed time, but the final companion
manifest deliberately canonicalizes only its deterministic quality fields;
wall-clock time is excluded from the reproducible release identity.

## RawTherapee integration

RawTherapee exposes `Student-t GMR (experimental)` for X-Trans sensors, while
Markesteijn three-pass remains the default. A reviewed build installs
`models/xtrans-tgmr-v2.tgmr` and compiles its exact SHA-256 into the loader.
Until that artifact exists, developers can select an authenticated compatible
model explicitly:

```sh
RT_XTRANS_TGMR_MODEL=/absolute/path/model.tgmr rawtherapee-cli ...
```

The explicit environment variable is an override, not a search path. Loading,
CFA, allocation, or inference failure emits a diagnostic and causes complete
Markesteijn overwrite. No partial TGMR output is exposed.

Release configuration requires all four reviewed inputs:

```sh
-DTGMR_OFFICIAL_MODEL=/path/model.tgmr
-DTGMR_OFFICIAL_MODEL_SHA256=HEX
-DTGMR_OFFICIAL_MODEL_MANIFEST=/path/model.tgmr.json
-DTGMR_CORPUS_ATTRIBUTION=/path/CORPUS-NOTICE.txt
```

The model, corpus, checkpoints, and downloaded originals are ignored by Git.
They must not be committed before corpus/model redistribution review is complete.
