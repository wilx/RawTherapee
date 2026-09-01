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
[`corpus-v1.schema.json`](corpus-v1.schema.json). Each line is one source, and
line order is part of the corpus identity. Selected sources may use only CC0,
Public Domain Mark, or CC BY 2.0/3.0/4.0. Authors, decoded identities, and
perceptual identities may not cross train/validation/test boundaries.

The intended source population is 4,000 training, 500 validation, and 500 test
images. Open Images V7 is the primary catalog. Its image metadata supplies the
original URL, landing page, author, title, license, dimensions, rotation, and
advertised MD5. Wikimedia Commons may fill deficient strata, but every Commons
file needs an individual landing-page and license review. Changing thumbnails
are not authenticated substitutes for an unavailable original.

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
substitutes a changed fallback. Original reconstruction is best effort because
third-party URLs can disappear. The authenticated TGPC patch corpus is the
durable training input.

## Corpus workflow

```sh
TOOL=/tmp/rt-tgmr-trainer/rt-tgmr-train
CACHE=/data/tgmr/source-cache

"$TOOL" corpus verify-sources corpus-v1.jsonl "$CACHE"
# Bootstrap one candidate record before the final manifest exists.
"$TOOL" corpus classify-file openimages:IMAGE_ID "$CACHE/IMAGE_ID.jpg"
"$TOOL" corpus classify corpus-v1.jsonl "$CACHE" classifications.jsonl
"$TOOL" corpus propose-patches corpus-v1.jsonl "$CACHE" proposals.jsonl
# Review and merge classifications/proposals into the final canonical JSONL.
"$TOOL" corpus validate-manifest corpus-v1.jsonl
"$TOOL" corpus pack corpus-v1.jsonl "$CACHE" tgmr-corpus-v1.tgpc
"$TOOL" corpus pack corpus-v1.jsonl "$CACHE" \
    tgmr-corpus-v1-sensor-noise.tgpc --noise sensor-v1
"$TOOL" corpus gzip tgmr-corpus-v1.tgpc tgmr-corpus-v1.tgpc.gz
"$TOOL" corpus inspect tgmr-corpus-v1.tgpc.gz
"$TOOL" corpus balance tgmr-corpus-v1.tgpc.gz
"$TOOL" corpus report tgmr-corpus-v1.tgpc.gz \
    --json corpus-statistics.json --csv corpus-statistics.csv \
    --html corpus-statistics.html
```

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

`corpus classify-file` is the intake/bootstrap form: it accepts a stable source
ID and one downloaded image and emits decoded dimensions, orientation, ICC and
pixel identities, plus the complete classification object needed by a source
manifest record. It deliberately does not invent licensing, author, URL, or
split metadata. `corpus classify` is the authenticated batch form used after
those source records have been reviewed and merged.

`corpus balance` reports the declared low/middle/high brightness, chroma, and
texture strata. `corpus report` emits canonical JSON, CSV, and a self-contained
HTML report. The 5,000-image freeze additionally requires license review,
near-duplicate review, all minimum stratum counts, and the 250/500/1000/2000/4000
source-count learning curve; the tool does not pretend that a syntactically
valid manifest proves those external judgments.

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
