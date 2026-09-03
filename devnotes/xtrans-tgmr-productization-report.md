# X-Trans TGMR productization status

## Scope and release boundary

This work turns the successful fixed K32/S9/q8 Student-t GMR research method
into a reproducible product pipeline rather than treating the existing BSDS
artifact as shippable. The visible RawTherapee method and model container are
implemented, and a standalone native corpus/training tool now owns corpus
authentication, classification, patch construction, fitting, checkpointing,
export, inspection, and benchmarking.

The production release gate is intentionally **not yet open**. The existing
research weights came from 200 BSDS photographs and 102,400 distinct RGB
patches, with every patch projected through all 18 X-Trans observation phases.
That experiment established the method, but its learning curve had not
plateaued, its patch population was not balanced by visual characteristics, and
BSDS is restricted to non-commercial research use. It cannot be the source of
a model bundled with RawTherapee.

No corpus, checkpoint, model, or generated attribution list is committed by
this productization change. A model becomes installable only when CMake is
given an exact reviewed model, SHA-256, companion manifest, and corpus notice.
The automatic loader additionally requires the compiled official digest.

## Implemented foundation

### Reconstructible corpus

- Source-manifest schema v2 records stable source identity and split, exact
  catalog revision/snapshot digest, upstream source/Flickr/author identities,
  URLs and archive-member fallbacks, rights evidence and review state,
  attribution, original/decoded hashes, dimensions, orientation, ICC identity,
  classifications and histograms, deterministic patch seed, coordinates, and
  augmentation. The earlier v1 contract remains readable for development.
- Selected records accept only CC0-1.0, PDM-1.0, and CC-BY 2.0/3.0/4.0. NC,
  ND, SA, unknown, and ambiguous terms are rejected. Smithsonian records are
  accepted only with explicit CC0. People sources require a separate approved
  no-obvious-minors/non-sensitive-content review.
- Catalog acquisition is frozen for a deterministic 21,000-candidate pool:
  10,000 candidates from the CVDF Open Images V4/V5 boxable subset, 6,000
  PASS, 3,000 Wikimedia Commons, and 2,000 Smithsonian Open Access. The exact
  selected mix is respectively
  2,500/1,500/600/400, with each source preserving the 80/10/10 split.
- The matching Open Images boxable metadata snapshot contains 1,743,042 image
  records and is pinned at 638,407,721 bytes with SHA-256
  `05f3d68dbbb03728d1a37e51479f4f35c062b871e1a6cae8c4cefbe0e0c80ed0`.
  A deterministic 100-record pilot against the extracted three-level CVDF
  mirror classified all 100 inputs. Its candidate JSONL SHA-256 is
  `29b7fc12a65af58389aa80fb4ee6dd731628caa521521d8011335fb3eba4fa01` and
  classification JSONL SHA-256 is
  `ed02f664d805d62517d9153c2deba1959aee0f49cc78979e6564000233b50bc8`.
  The pilot exposed one grayscale JPEG with an embedded GRAY ICC profile;
  classification now passes the profile a one-component input to LittleCMS
  before producing neutral linear RGB. The full and 1/8-proxy paths have a
  native regression test for this case.
- A standard-library-only preparation program snapshots and authenticates
  catalogs, captures one-time Commons API responses and anonymous Smithsonian
  Open Data on AWS index/shard snapshots, preserves the complete consumed
  Smithsonian metadata bytes for byte-identical offline replay, normalizes all
  four catalogs, fetches originals, assembles C++ classifications plus human
  review, merges canonical JSONL, and creates the release manifest. Open Images
  changing thumbnails are excluded. PASS's variable-length hexadecimal `hash`
  field is treated as its source/filename identity rather than incorrectly as
  an MD5; retrieved image bytes are authenticated with RawTherapee SHA-256.
  PASS authenticated Zenodo archive members can serve as byte-identical
  fallback sources.
- Normalized author groups are assigned to one split before selection and are
  capped at five images. Compressed/decoded hashes, normalized Flickr IDs,
  64-bit dHash, and deterministic DCT pHash reject exact and near duplicates.
  The selection engine enforces exact catalog/split quotas, a reviewed 15%
  people subset, minimum size/area, and round-robin coverage of 27
  training-derived brightness/chroma/texture cells. It fails rather than
  weakening constraints when a candidate pool is inadequate. Its canonical
  decision report lists every exact/perceptual conflict and distance so the
  rejected borderline clusters can be audited manually.
- The production validator requires exactly 4,000/500/500 sources and
  256/128/128 unique patches per source. `corpus finalize` creates those
  1,152,000 patch records after selection, preserving the frozen 75% spatial /
  25% coverage policy and 25% identity augmentation.
- `reconstruct_corpus.py` uses only the Python standard library. It processes
  records in order, retries, writes `.part`, authenticates before publication,
  reports missing/changed URLs, supports partial/split and offline operation,
  emits a plain TSV fetch list, and safely extracts only the frozen member of an
  authenticated PASS archive. A mirror is accepted only when its bytes match
  the source identity; changed/unavailable content is never substituted.
- Source-corpus reports cover catalog, split, license, rights status, content
  tags, controlled people count, quality eligibility, and luminance/hue/
  saturation distributions. TGPC balance cut points are derived only from the
  training population; fixed research thresholds remain an explicit legacy
  control.
- A canonical release manifest binds the final source JSONL, TGPC, deterministic
  gzip, attribution notice, statistics, and rights report to upload locations.
  The intended durable publication is one identical `.tgpc.gz` on Zenodo and a
  RawTherapee/GitHub release; original photographs are not redistributed.

### Native classification and packing

- JPEG, PNG, and contiguous 8/16-bit RGB/grayscale TIFF are decoded using
  RawTherapee's existing dependencies. ICC-tagged files are transformed to
  linear sRGB with LittleCMS; untagged files use an explicit assumed-sRGB
  identity. JPEG EXIF and TIFF orientation are normalized before sampling while
  retaining the original orientation value in provenance.
- Classifications include linear channel/luminance statistics, chroma, hue,
  saturation, clipping, gradient and Laplacian energy, local contrast, JPEG
  blockiness, decoded-pixel SHA-256, and a deterministic dHash.
- Patch proposal produces distinct deterministic samples: 75% uniform spatial
  positions and 25% cyclic coverage of low/high brightness, low/high chroma,
  low/high texture, high contrast, clipping, saturation, and eight hue sectors.
- Exactly 25% of final patches retain identity linear sRGB. Remaining patches
  use deterministic -2..+2 stop exposure, product-normalized 0.5x..2x white
  balance, and frozen Fujifilm camera-matrix transforms. Training and held-out
  matrix IDs are disjoint and enforced while packing.
- The packer exposes two reproducible validation candidates: clipping-only
  (`--noise none`, the default) and bounded integer-domain `sensor-v1` noise on
  non-identity patches. The latter approximates 8 uint16-code read noise and a
  signal-dependent term reaching approximately 64 codes at saturation. It is
  bound into the TGPC configuration digest and augmentation identity; the
  simpler recipe remains preferred unless the frozen validation/tail study
  establishes a material advantage.
- TGPC v1 is an authenticated little-endian record stream containing planar
  7x7 RGB uint16 values and provenance. The packer does not materialize the
  corpus. The deterministic `.tgpc.gz` form uses existing required zlib,
  `mtime=0`, no original name/comment, fixed level, and canonical header bytes.
  The reader streams compressed or uncompressed data and authenticates the
  complete structure and payload.
- Corpus inspection, brightness/chroma/texture balance, and canonical JSON,
  CSV, and self-contained HTML statistics are implemented in `rt-tgmr-train`.

### Native fitting and artifact generation

- The C++17 trainer implements deterministic k-means++ initialization,
  full-covariance Gaussian EM, and fixed-nu Student-t ECM in float64.
- `--source-limit` selects the first N authenticated training sources without
  repacking the corpus, records the limit in checkpoints/manifests, and makes
  the mandated 250/500/1000/2000/4000 source-count curve reproducible from one
  TGPC identity. Resume rejects a conflicting limit.
- It fits 18 phase-conditioned, 51-dimensional K=32 models with nu=3,
  covariance floor 1e-6, stable log-domain responsibilities, Cholesky solves,
  deterministic component ordering, and atomic authenticated iteration
  checkpoints.
- The hot fitting loop accumulates batch sufficient statistics rather than
  storing an N-by-K responsibility matrix. Corpus phase copies are not stored.
  The configured batch and maximum memory limits are checked.
- `canonical` uses one worker and a fixed reduction order; compiler
  floating-point contraction is disabled. `cpu` uses OpenMP workers with
  deterministic per-thread statistics and thread-order reduction. Both use
  fixed-size compiler-vectorizable kernels.
- Export specializes the fitted model to S9/top-8, tau=0.0003, temperature 4,
  preserves all 18 phase contracts, and creates a TGMR v2 model plus canonical
  training manifest containing corpus/trainer/configuration identities,
  convergence, populations, phase backends, payload identity, and model hash.
  Export rejects phase sets with differing corpus identities, source limits,
  sample counts, seeds, covariance settings, backends, or convergence counts,
  as well as malformed or non-finite fitted arrays; a mixed checkpoint
  directory therefore cannot silently become one model.
  A final re-export accepts only a native validation report bound to that exact
  model and corpus and embeds it in the companion manifest; release CMake
  rejects an absent or limited/non-validation object. The standalone timing is
  retained in its validation report but omitted from the deterministic
  companion-manifest identity.
- Native model validation applies the exact scalar K32/S9/q8 inference
  equations to every selected patch through all 18 CFA phases, restores the
  measured component, and emits pooled, per-phase, and median-source PSNR plus
  p99/worst patch RMS. This keeps corpus/model selection inside the C++ tool.
- A native benchmark command measures selected phases and sample counts. On
  the Ryzen 9 5900X with 24 OpenMP workers, the external 102,400-patch research
  population completed one phase of the full 10-Gaussian/30-Student iteration
  recipe in 21.565369 seconds. Eighteen independent phases therefore project
  to approximately 6.47 minutes, below the ten-minute reproduction gate. A
  linear million-patch projection is approximately 63.2 minutes, below the
  two-hour gate, but both projections still require confirmation on the final
  packed production corpus.
- The optional OpenMP-target build now has a real fixed K32/D51 device path.
  It retains mapped samples/model parameters, computes likelihood and latent
  weights on device, reduces block sufficient statistics without floating-point
  atomics, uses bounded batches/yielding, supports explicit device selection,
  and rejects host fallback. A physical RX 7800 XT run on the corrected planar
  102,400-patch research population completed one Gaussian plus one Student-t
  iteration in 4.436398 seconds and matched the CPU mean log-likelihood
  (132.392527) at the reported precision. The equivalent 24-thread CPU run took
  1.465054 seconds. The current target kernel is therefore about 3.03 times
  slower, fails the required 1.5x speedup, and remains an opt-in diagnostic
  backend rather than a supported release backend. No 0.02 dB model-parity
  claim is made from matching one scalar likelihood.

### RawTherapee runtime

- `Student-t GMR (experimental)` is a visible X-Trans method. Markesteijn
  three-pass remains the default.
- The existing fixed scalar and AVX2/FMA component-bucketed inference paths are
  retained; an ARM64 NEON path implements the same numeric contract.
- TGMR v2 is a data-only, little-endian, SHA-256-authenticated fixed-contract
  container. The loaders reject unknown version/flags, malformed/reserved
  fields, wrong phase/component shapes, unexpected sizes, non-finite values,
  invalid Cholesky diagonals, trailing bytes, and digest mismatch.
- The installed reviewed model is loaded automatically. An explicit
  `RT_XTRANS_TGMR_MODEL` override permits compatible user-trained TGMR v2
  models and is reported as custom. There is no unreviewed implicit search.
- Any model, CFA, allocation, or inference failure emits a diagnostic and
  completely overwrites partial output through Markesteijn three-pass.
- Generic X-Trans PP3 serialization, edited-state, partial-paste, batch,
  preview/export, reset, and history wiring covers the new enum value. The
  English method label explicitly says experimental.

## Required production corpus audit

The final manifest must be reconstructible, but URL persistence cannot be
guaranteed. The canonical TGPC and deterministic gzip are therefore the durable
training artifacts. Each source nevertheless needs a human-verifiable landing
page and rights record so the patch corpus and derived model can be
redistributed.

Before freeze:

1. Download and authenticate the four frozen catalog snapshots and record their
   exact identities; curate 21,000 candidates at the fixed per-source quotas.
2. Download and authenticate original bytes; record advertised and RawTherapee
   hashes independently, including PASS archive-member fallbacks.
3. Review per-source license, attribution, people status, and content tags. A
   catalog license field is
   evidence to inspect, not an automatic legal conclusion.
4. Decode, orient, color-manage, classify, and compute exact, Flickr, dHash, and
   DCT-pHash identities plus signal-statistics histograms.
5. Assign normalized author groups to splits; reject duplicates and review
   borderline near-duplicate clusters before accepting exact source quotas.
6. Run deterministic selection, review category deficiencies, finalize fixed
   patch coordinates, audit the 75/25 policy, and pack TGPC twice.
7. Require at least 10,000 training and 1,000 validation/test patches in every
   declared critical brightness, chroma, and texture stratum. Add sources rather
   than duplicating deficient patches.
8. Compare identity-only, clipping, and bounded read/signal-noise augmentation
   recipes on validation. Prefer the simpler recipe inside 0.1 dB with
   equivalent tails.
9. Train the 250/500/1000/2000/4000 source-count curve. The last doubling must
   change validation PSNR by less than 0.1 dB and p99 error by less than 2%.
10. Freeze the corpus JSONL/TGPC/gzip hashes, complete the generated corpus
    notice and model redistribution review, then publish identical gzip bytes
    to Zenodo and the declared RawTherapee/GitHub release.

These are external data-selection and review tasks. Passing code tests cannot
substitute for them.

## PASS live acquisition pilot

A live PASS pilot was run against the official catalog artifacts on 2026-09-02.
The external pilot data remain under `~/TGPC/pilot-pass` and are not tracked in
Git.

- `pass_metadata.csv`: 152,590,171 bytes, 1,439,588 records, SHA-256
  `8b6fde80b48326bda9da0a7c48f92146a58a847f76a1dc40603b7e4f73f5e798`.
  Its published Zenodo MD5 `0b033707ea49365a5ffdd14615825511`
  also matched.
- `pass_urls.txt`: 158,356,435 bytes, 1,439,588 URLs, SHA-256
  `cc692c3e7094b7e51e218c8bc2e351acdbb419e861976be10472f16ba872566b`.
- The metadata and URL lists have exactly equal row counts. The current URLs
  resolve to the stable Multimedia Commons S3 host.
- A sequential 10-image fetch succeeded 10/10 with distinct compressed and
  decoded hashes. A second deterministic sample selected 20 records evenly
  over catalog indices 0 through 1,439,587 and also succeeded 20/20.
- Classification showed that all 20 distributed images are JPEGs with a long
  side no greater than 500 pixels: dimensions ranged from 280x479 through
  500x434, and areas ranged from 129,500 through 217,000 pixels. Consequently,
  0/20 meet the frozen shortest-side >=512 and area >=0.75-megapixel gate.
- The sample did provide diverse measurable content: linear luminance means
  ranged from 0.0175 to 0.4023 and chroma-ratio means from 0.0 to 2.6549.

The pilot therefore passes acquisition availability, licensing metadata,
normalization, authentication, and C++ classification, but it **fails the
production resolution gate structurally**. PASS remains useful for tooling and
lower-resolution research, but cannot supply its planned 1,500 production
sources without relaxing an already frozen quality rule. Under the corpus plan,
the correct response is to expand or replace the PASS source allocation rather
than upscale its images or silently lower the threshold.

## Open Images 10,000-source classification run

The first complete catalog-sized intake stage was run against the locally
mirrored CVDF Open Images V5 boxable training archive. The external images and
generated candidate artifacts remain under `~/TGPC` and are not tracked in
Git.

- The authenticated catalog input is
  `train-images-boxable-with-rotation.csv`: 638,407,721 bytes, 1,743,042 data
  records, SHA-256
  `05f3d68dbbb03728d1a37e51479f4f35c062b871e1a6cae8c4cefbe0e0c80ed0`.
- Deterministic candidate generation emitted exactly 10,000 records. The
  candidate JSONL SHA-256 is
  `e8ecba0eaa288da88a8efb962a6205417c76e0991bdec2c0b7ada353c521d9fb`.
- Full-resolution classification used 12 bounded workers and local durable
  checkpoints while reading the sharded CIFS image store. The main pass
  sustained approximately 28.6 images/s and 8.7 MiB/s. It completed 9,998
  candidates and retained two rejected records in the failure audit.
- Three otherwise valid CMYK JPEGs initially exposed a decoder limitation.
  The intake path now preserves CMYK/YCCK JPEG components and uses an embedded
  CMYK ICC profile through LittleCMS. It rejects unprofiled CMYK and component/
  profile mismatches instead of assigning an invented color interpretation.
- The final classification JSONL contains 9,998 records and has SHA-256
  `a2ef9ea919dd8edf8887e44600d227266594b4d6b0afc04fc942adf13499b6fb`.
  The two-record failure audit has SHA-256
  `be6734fe9ef9e4de31fb0bad22623c9a5afb5ed79ff90b1f7ab946546e24ab40`:
  one three-component JPEG carrying an incompatible CMYK profile and one
  four-component CMYK JPEG without any profile.
- Candidate screening now requires the explicit `--allow-failures` option to
  publish only successful classifications. The default remains fail-closed,
  and the option is not appropriate for a reviewed or selected corpus. The
  durable failure report remains beside the checkpoints as the rejection
  record.
- Assembly re-authenticated the source bytes and emitted 9,998 V2 records in
  pending-review state. Its JSONL SHA-256 is
  `279bd7e911b0d5cf3c15554036de19d6bffe09efa6d9941fe27b47bb260577ae`.
  The source-statistics JSON SHA-256 is
  `29bbe005fda89118a4211305bc343df47ab41981737f6f2b636faa8c6e3f9e77`.

Of the 9,998 decoded candidates, 4,668 meet the frozen 512-pixel shortest-side
and 0.75-megapixel gates. Before duplicate and content review, deterministic
author-group split assignment produces this capacity:

| Projected split | Size-eligible | After five-image author cap | Required Open Images quota |
| --- | ---: | ---: | ---: |
| train | 3,766 | 3,611 | 2,000 |
| validation | 441 | 428 | 250 |
| test | 461 | 431 | 250 |

The Open Images pool therefore has useful numerical surplus for its frozen
2,500-source quota. It is not selectable yet: all 9,998 records deliberately
remain `rights.review_status=pending`, content tags are empty, and the
controlled people subset has not undergone the required no-minors/sensitive-
content review. These are human audit gates rather than classifier outputs.

The run also confirms an operational distinction. Parallel decoding and
classification scale adequately even from the network store, but Python V2
assembly currently rehashes source files serially and classifier resume startup
canonicalizes all candidate paths on the CIFS mount. Both are bounded and
correct, but optimizing those preparation stages would improve iteration time
before processing substantially larger candidate pools.

## Incomplete plan items

The following remain deliberately open rather than being represented as done:

- Curating/downloading/reviewing the 5,000 licensed sources.
- Capturing and pinning the actual external catalog snapshots and SHA-256 values.
- Freezing and publishing `corpus-v1.jsonl`, TGPC, deterministic gzip, fetch
  list, and attribution notice.
- Source-count convergence and augmentation/noise selection.
- Canonical two-clean-run production training and the official ~6 MiB model.
- Production quality evaluation on the untouched licensed test set, every
  stratum, synthetic controls, X-Trans generation I-V RAFs, and `DSCF0771`.
- Full production-corpus validation of the OpenMP-target AMDGCN backend and a
  physical NVPTX run. No production GPU speed claim is made from the tiny smoke
  fixture.
- Confirmed all-phase and million-patch trainer performance on the final
  production corpus (the current 6.47-minute and 63.2-minute figures are
  projections from one measured research-corpus phase).
- Cross-compiler x86-64/ARM64 release matrix and physical ARM64 NEON execution.
- The final clean `codex/xtrans-tgmr-productization` branch from current
  `upstream/dev`.

The clean branch is intentionally last. It must port only the frozen trainer,
corpus tools, runtime, official reviewed model/notice, GUI/package integration,
focused tests, essential fixtures, and final documentation—not the broad
experimental history on this branch.

## Verification completed on the development branch

- Standalone trainer CMake configuration/build and its native tooling CTest.
- Python source-reconstruction unit tests without third-party packages.
- Python catalog-adapter tests for Open Images, PASS, Commons, and Smithsonian,
  including authenticated PASS archive-member recovery and V2 review assembly.
- Native deterministic source selection tests covering exact four-catalog
  quotas, source-level split counts, author cap, people quota, source reports,
  production-manifest validation, and training-derived balance thresholds.
- RawTherapee `dev` build of trainer/runtime targets using four jobs.
- Native TGMR contract tests with no external model.
- Reviewed development-model parity and crop/phase tests when
  `RT_XTRANS_TGMR_MODEL` points to the locally converted research artifact.
- GCC 13 release and strict-source builds, Clang 18 release builds, and a
  Clang 18 ASan/UBSan build. The focused trainer/runtime tests pass with and
  without the reviewed development model. Leak detection is disabled only for
  the sanitizer test invocation because LeakSanitizer cannot run under the
  desktop sandbox's tracing; AddressSanitizer and UndefinedBehaviorSanitizer
  remain active.
- Debug trainer/runtime builds and tests.
- A fresh default configuration with `BUILD_TGMR_TRAINER=OFF` exposes no
  trainer or trainer-test target, preserving ordinary RawTherapee builds.
- The release CMake gate accepts a correctly bound model/notice/full-validation
  manifest and rejects the same model with a missing validation object. Two
  validation runs with different elapsed times produced byte-identical model
  and final companion-manifest bytes after deterministic validation-field
  canonicalization.
- The trainer's direct ELF dependencies are OpenMP, JPEG, PNG, zlib, TIFF,
  LittleCMS, and the C/C++ runtimes. It has no direct Zstandard, curl, Python,
  Torch, ONNX, or machine-learning-library dependency. The Ubuntu TIFF library
  may itself load Zstandard transitively; that is an existing TIFF packaging
  detail, not a new trainer or corpus-format requirement.
- Direct AArch64 compilation of the NEON translation path.
- Canonical TGPC/gzip round trips, corrupted-container rejection,
  deterministic packing, deterministic checkpoint/resume, one-step
  Student-t reference arithmetic, phase contracts, model loading, PP3 round
  trip, custom-model override, and complete fallback coverage.

The sanitizer run also exposed and fixed a pre-existing null dereference in
the framing-profile save helper when a complete PP3 is saved without a
`ParamsEdited` object. The ordinary full-profile path now preserves its prior
behavior without undefined behavior.

RawTherapee builds used `-j4`.
The generated `Testing/` directory is unrelated workspace content and remains
untracked and untouched.
