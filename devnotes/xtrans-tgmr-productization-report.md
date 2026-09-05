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
- Catalog acquisition is frozen for a deterministic 19,000-position candidate
  pool: 14,000 positions from the CVDF Open Images V4/V5 boxable subset,
  3,000 Wikimedia Commons, and 2,000 Smithsonian Open Access. PASS remains a
  supported research/tooling input but contributes no corpus-v1 sources after
  its distributed images failed the frozen resolution gate. The exact selected
  mix is 4,000/600/400, with each source preserving the 80/10/10 split.
  Open Images grows in deterministic 2,000-record tranches whenever any
  projected split has less than 20% surplus after quality and author-cap gates.
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
  catalogs, captures a bounded one-time Commons category/API snapshot and anonymous Smithsonian
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
- Patch finalization is restartable at source granularity. Each completed
  source is serialized as an authenticated immutable checkpoint bound to the
  exact selected-source manifest. Packing is restartable at a configurable
  record boundary: it revalidates all durable partial records and split counts
  before appending, skips completed source ranges without reopening them, and
  produces byte-identical output to an uninterrupted run.
- The packer exposes two reproducible validation candidates: clipping-only
  (`--noise none`, the default) and bounded integer-domain `sensor-v1` noise on
  augmented training patches. The latter approximates 8 uint16-code read noise and a
  signal-dependent term reaching approximately 64 codes at saturation. It is
  bound into the TGPC configuration digest and augmentation identity; the
  simpler recipe remains preferred unless the frozen validation/tail study
  establishes a material advantage.
- A third `identity-only` training candidate removes all exposure,
  white-balance, and camera-matrix transforms from training records. Validation
  and test records remain byte-identical across identity-only, production-v1,
  and production-v1 plus sensor-noise corpora, preventing evaluation changes
  from being mistaken for model improvements.
- TGPC v1 is an authenticated little-endian record stream containing planar
  7x7 RGB uint16 values and provenance. The packer does not materialize the
  corpus. The deterministic `.tgpc.gz` form uses existing required zlib,
  `mtime=0`, no original name/comment, fixed level, and canonical header bytes.
  The reader streams compressed or uncompressed data and authenticates the
  complete structure and payload. Inspection also hashes each encoded split
  separately, making the required validation/test identity across the three
  training-augmentation candidates directly auditable.
- Corpus inspection, brightness/chroma/texture balance, and canonical JSON,
  CSV, and self-contained HTML statistics are implemented in `rt-tgmr-train`.
- Canonical attribution, per-source rights/evidence JSON, and reconstruction
  URL/checksum TSV generation are implemented in the same executable and
  require the complete production manifest contract.

### Native fitting and artifact generation

- The C++17 trainer implements deterministic k-means++ initialization,
  full-covariance Gaussian EM, and fixed-nu Student-t ECM in float64.
- `--source-limit` selects the first N authenticated training sources without
  repacking the corpus, records the limit in checkpoints/manifests, and makes
  the mandated 250/500/1000/2000/4000 source-count curve reproducible from one
  TGPC identity. `corpus order-training` first freezes nested prefixes with
  exact 80/12/8 Open Images/Commons/Smithsonian counts at every milestone and
  deterministic interleaving of the 27 source-level brightness/chroma/texture
  cells. Its order manifest binds the reviewed selected-source digest and the
  reordered manifest digest. Resume rejects a conflicting limit.
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
  The trainer-configuration digest is derived from the authenticated phase
  checkpoints. The trainer-revision digest is generated at build time from a
  sorted relative-path/SHA-256 manifest of every trainer source, header, CMake
  input, and the embedded cJSON implementation; release export no longer
  trusts caller-supplied arbitrary identity strings.
  A final re-export accepts only a native validation report bound to that exact
  model and corpus and embeds it in the companion manifest; release CMake
  rejects an absent or limited/non-validation object. Canonical validation JSON
  omits wall-clock timing so independent canonical results can be byte-identical.
- Native model validation applies the exact scalar K32/S9/q8 inference
  equations to every selected patch through all 18 CFA phases, restores the
  measured component, and emits pooled, per-phase, per-source, median-source,
  phase-spread, fixed corpus-v1 brightness/chroma/texture strata, and p99/worst
  patch metrics. This keeps corpus/model selection
  inside the C++ tool.
- A non-installed native population evaluator is built with the trainer and
  tests. It reconstructs the exact held-out linear augmentation around each
  frozen source coordinate, proves padded-crop Markesteijn center parity
  against a deterministic full-image run, assigns phases evenly, and compares
  TGMR with RawTherapee's actual three-pass Markesteijn implementation. It
  reports the same fixed signal strata separately for both methods. This
  closes the previous release gap where trainer validation had ground-truth
  TGMR metrics but no native Markesteijn population baseline.
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

1. Download and authenticate the frozen catalog and annotation snapshots and
   record their exact identities; curate the initial 15,000 candidates and any
   required deterministic Open Images expansion tranches.
2. Download and authenticate original bytes; record advertised and RawTherapee
   hashes independently, including PASS archive-member fallbacks.
3. Review per-source license, attribution, people status, and content tags.
   Complete authenticated Open Images catalog rows with whitelisted CC BY
   metadata are the automatic rights decision under the frozen recipe;
   people/minor/sensitive-content and ambiguous duplicate cases remain manual.
4. Decode, orient, color-manage, classify, and compute exact, Flickr, dHash, and
   DCT-pHash identities plus signal-statistics histograms.
5. Assign normalized author groups to splits; reject duplicates and review
   borderline near-duplicate clusters before accepting exact source quotas.
6. Run deterministic selection, review category deficiencies, finalize fixed
   patch coordinates, audit the 75/25 policy, and pack TGPC twice.
7. Require at least 10,000 training and 1,000 validation/test patches in every
   declared critical brightness, chroma, and texture stratum. Add sources rather
   than duplicating deficient patches.
   In addition, retain the source-level `astronomy-star-field` safety guardrail:
   at least 12 training, one validation, and one test source, identified through
   authenticated Commons astronomy-category provenance plus tracked stellar
   title patterns. Generic low-light images do not satisfy this minimum.
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

## Open Images 12,000-candidate preparation

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

| Projected split | Size-eligible | After five-image author cap | Initial required quota |
| --- | ---: | ---: | ---: |
| train | 3,766 | 3,611 | 3,200 |
| validation | 441 | 428 | 400 |
| test | 461 | 431 | 400 |

That initial pool could meet the exact quotas, but failed the separately frozen
20% surplus gate in every split. A deterministic 2,000-record tranche beginning
at eligible catalog offset 10,000 was therefore normalized and classified. All
2,000 additional images succeeded at about 12.1 images/s using four workers.
Assembly re-authenticated them and the merged 11,998-record V2 review input has
SHA-256
`27d03730f4e5c701085944284fdebeb3a2dec4d8da298f42eb805c9c75d04fd3`.

The content/review preparation authenticates and joins these snapshots:

| Snapshot | Bytes | SHA-256 |
| --- | ---: | --- |
| V7 boxable class descriptions | 12,064 | `1839e0e7e84130ae281f7f67413768601b031581c0c42e7fc17527b8e2a99aa9` |
| V5 positive human image labels | 376,764,810 | `f9bec2d40b4e12d67c9f726292b5db88285713267fc3dc6496ae72839b2fd9de` |
| V6 object boxes | 2,258,447,590 | `dfc9637907a6b105f87e435bac91a5ee9b29af3ff8391168f86c1d63879786b6` |
| tracked content-tag rules | 2,345 | `bbe9513cef61e5d40f8cafff63263c74236c62ac6c97a04d509ea66c4aa1c04f` |

The exact annotation join found 46,462 positive human-label rows and 101,355
box rows for the expanded candidates. It tagged 10,714 sources and approved the
rights evidence for all 11,998 complete catalog rows. Machine labels were not
used for rejection. After the frozen dimension/area and five-image author cap,
the expanded split capacity is:

| Projected split | Quality/author capacity | After conservative deduplication | Exact quota | 20% surplus target |
| --- | ---: | ---: | ---: | ---: |
| train | 4,302 | 4,293 | 3,200 | 3,840 |
| validation | 483 | 482 | 400 | 480 |
| test | 516 | 516 | 400 | 480 |

Ten candidates were conservatively rejected by the frozen pHash threshold;
there were no additional dHash, exact-byte, decoded-pixel, or Flickr-ID
collisions in this stable-order capacity pass. All three splits still pass the
surplus gate, although validation remains close to it. The deterministic
people-review queue contains 750 train, 94 validation,
and 94 test candidates, exactly 25% above the required 600/75/75 approvals. Its
JSONL SHA-256 is
`fdc4a67545bff5db4b7c17ef64cef6699dfeb694aaf5a4685b51526052d6aa18`;
the generated HTML and JSONL remain external under `~/TGPC`. Human review is
still required before Open Images candidates can enter final selection. The
review page is reject-only: unchecked entries are exported as approved, so the
reviewer scans the complete queue and marks only obvious minors or sensitive
content rather than clicking 938 individual approval controls.

The separate borderline-duplicate pass examines the 5,301 candidates that can
survive the projected split and five-image author cap. The frozen manual band
is dHash distance 6-7 or DCT pHash distance 9-10, immediately outside the hard
automatic thresholds of 5 and 8. It found 27 review pairs involving 46 sources,
plus 11 hard-close pairs that remain automatically excluded. Because pairwise
review would make transitive duplicate families awkward and error-prone, the
tool groups those pairs into 20 connected review clusters. The cluster queue
SHA-256 is
`f567b7db6055780e50b0208a2c5240d3533708a35b4612102f6f27ab54426ee6`;
the pending-manifest SHA-256 is
`4403a812791504403a9d03dde8d01c404d6d1ca28d87cf002eff0a401099aa0e`.
Every involved source is ineligible until its complete cluster is explicitly
reviewed and any redundant members are rejected.

The run also confirms an operational distinction. Parallel decoding and
classification scale adequately even from the network store. V2 assembly now
uses bounded parallel source authentication while preserving deterministic
record order. Classifier resume startup still canonicalizes all candidate paths
on the CIFS mount; it is bounded and correct, but remains an avoidable iteration
cost.

## Commons and Smithsonian metadata preparation

The two supplementary candidate catalogs have now been frozen at metadata
level. All generated snapshots and original images remain external under
`~/TGPC`.

The Commons collector uses the tracked bounded category recipe rather than a
manually edited title list. It combines subject-tagged Quality Images with
explicit CC BY 2.0/3.0/4.0 and CC0 license categories. This distinction is
necessary: a Quality-Images-only pilot found just 471 admissible files among
3,850 discoveries because 3,368 used ShareAlike or another excluded license.
The final author-balanced run discovered 23,850 unique files and froze exactly
3,000 records after rejecting 3,304 disallowed-license, 1,567 undersized, 677
unsupported-raster, and 14,051 above-author-cap candidates. It contains 1,063
authors; after applying the final five-image cap, its projected capacities are
1,814 train, 247 validation, and 180 test candidates, comfortably above the
480/60/60 quota. The snapshot SHA-256 is
`c03fb6265aae1462b0e37c06627caca4ddc6e25079fa3dd7ff4363d1a71af0b6`;
the normalized candidate JSONL SHA-256 is now
`81e98f99aa4b071023a12369c6671822eef90f167c49bc2ec8aadbcf37d2a470`
after binding the reviewed content-tag rules.
All 3,000 rows have complete whitelisted catalog rights evidence. The tracked
recipe SHA-256 is
`46ada0da3f88323e1497e704950530ddec95f453518890e1317f57090f7f8e99`.

The final metadata adds an `astronomy-star-field` guardrail distinct from
ordinary low-light content. Category provenance plus a conservative stellar
title rule identifies 19 sources, projected as 17 train, one validation, and
one test under the frozen author split. The selector requires 12/1/1 sources;
losing either held-out source to decoding, quality, author-cap, or duplicate
checks therefore triggers Commons expansion rather than silently dropping the
sparse-star safety class.

The Commons originals total about 12.95 GiB. The completed conservative
acquisition authenticated all 3,000 files with zero failures: 589 were adopted
from the restart cache and 2,411 were downloaded. Classification and assembly
also completed for all 3,000 records with zero failures. The service log
reported approximately 1.58 classified images/s and 6.97 MiB/s from the CIFS
store. The canonical fetch command uses one worker, a one-second global request
interval, a descriptive bot user agent, and process-wide `Retry-After`
handling.

The first assembled snapshot produced an empty people queue because the frozen
metadata contained only discovery/root categories. That was not accepted as
evidence that the corpus contained no people. A deterministic enrichment pass
now freezes each file page's actual non-hidden Commons categories before
normalization. It added 7,561 category memberships to the 3,000 records. The
original snapshot SHA-256 is
`c03fb6265aae1462b0e37c06627caca4ddc6e25079fa3dd7ff4363d1a71af0b6` and
the enriched snapshot SHA-256 is
`e2ff868e16589046810104e7104a37f0a171929f6b83cc742b18691f8be9970a`.
Category evidence plus conservative whole-word title terms now identify 231
people candidates: 184 projected train, 29 validation, and 18 test. The
canonical review-queue SHA-256 is
`acb641cd50a6b175c3ed4fdfbeb9be6a5f29fe46c2fdf25494e7e783192d9b65`.
These tags only require human review; they do not reject content automatically.
The completed review approved 220 candidates and rejected 11. The decision-file
SHA-256 is
`e65a32e2383daf479ceb0b9d6f9959de5ddc67d715bd92fccd082e78cc4b2823`;
the resulting 3,000-record reviewed Commons manifest has SHA-256
`d3651a9a043445803d0f436f7b4e23abf1375dc165540685735ee5b97db803bf`.

Smithsonian collection used anonymous Open Data on AWS metadata shards. A
first pass exposed 200 otherwise usable records without a stable landing page;
the collector now rejects those before normalization because the rights trail
would be incomplete. Offline replay of the already-authenticated shards emitted
1,576 initial and 800 expansion records, or 2,376 unique eligible records. The
merged compact snapshot SHA-256 is
`0e0973e40b657a12958d2a43c002ed9ebd56056aa54a9e9b98ad5e32e5254121`.
The deterministic first 2,000 normalized candidates have SHA-256
`d0ec606cb208b82fd51cabe9821db4021a2bf999465b290a0101b09867405346`;
all are explicit CC0 with complete metadata and media rights evidence.
Each candidate now carries the catalog-declared JPEG type, including 189
Smithsonian delivery-service URLs whose query identifiers have no filename
extension. This prevents authenticated JPEGs from being assigned a misleading
`.img` cache suffix.

Catalog category terms are mapped through the tracked, authenticated
`catalog-content-tags-v1.json`. It identifies 147 Smithsonian people candidates,
which remain ineligible without a manual no-obvious-minors/no-sensitive-content
decision. The selector can satisfy its fixed people quota entirely from the
reviewed Open Images queue and has more than enough non-people Smithsonian
capacity, so those records cannot enter the corpus accidentally. A generic
catalog contact-sheet/decision path is nevertheless available if they are to be
considered later.

All 2,000 Smithsonian originals were then downloaded and authenticated. The
portable-cache migration moved the 189 extensionless delivery-service objects
to their catalog-declared `.jpg` names without redownloading them. A retry of
just those records completed the classification, producing a 2,000-record
classification JSONL with SHA-256
`264d69b9e54f88c00fbe5d4b4501c97950f01c8f13666e53ca3a6b1cf230768c`.
Assembly re-authenticated every source and produced a V2 manifest with SHA-256
`65aa619e6ceb49c907e6ff564284879f03cd2d4b7cf1a5d36024362102f27082`.
After the frozen decoded-size gates, the non-people capacity is 1,417 train,
209 validation, and 182 test candidates, far above the required 320/40/40.

The optional Smithsonian people queue contains 116 train, 21 validation, and
10 test records and has SHA-256
`f7ad7809b2665e96122e8f2b4461a95360664ded62aedb66fd7b98dbede4e4de`.
The duplicate pass found 1,174 borderline pairs involving 280 sources and
grouped them into 25 connected clusters; 25 dHash-close and 911 pHash-close
pairs are hard exclusions under the frozen thresholds. The canonical cluster
queue has SHA-256
`0bc622a8a0d72375dc6ab6805ff05b111cf9824dfaaf2f42c1be447b414a3a87`.
The large pair count is concentrated in visually repetitive museum/artwork
families, which is precisely why cluster-level review is required rather than
1,174 independent pair decisions.

Human review is complete for the people records needed by the frozen quotas.
Open Images has 905 approved and 33 rejected people decisions; additional
people-tagged sources remain deliberately pending and ineligible. Commons has
220 approved and 11 rejected people decisions. All 147 Smithsonian people
records were accepted after review. The untouched pending records are not
implicitly approved.

The initial 12,000-position Open Images pool did not retain sufficient train
and validation capacity after the author cap and conservative global duplicate
review. The next deterministic 2,000-position tranche was therefore processed
from the already mirrored CVDF archive. It yielded 1,999 classified records;
one unprofiled CMYK JPEG was rejected. The tranche candidate, classification,
and reviewed-candidate SHA-256 values are respectively
`4f86c66de71fc95115b14205596701e80597a493818789f5eea5de21e440b266`,
`e1a2e117d447a42f48ffd656d66593a5fd22e7c106c396b05e62083871911ec8`,
and `cd24d3380c438a9f198d15ff1302045c11781de09ad176cadb28aa899789df82`.
The combined 13,997-record Open Images reviewed-candidate manifest has
SHA-256
`f85f09c03b1ee807848eed53c86217ee1edbeb17936a2d52353e9fae1f5cc412`.

The three reviewed catalogs now form an 18,997-record candidate manifest with
SHA-256
`b8d4a7b264e808504a0335847b75beb84e9e7b649c388f0844da7518723a759b`.
The merge rebases each catalog's cache filename under one common external root;
it does not copy images or change image identities. The global five-image
author cap leaves 8,540 candidates for perceptual duplicate analysis. The
fresh borderline review contains 66 clusters, 1,354 candidate pairs, and 425
distinct images. Its queue SHA-256 is
`8f2afa4c9593c0cd98444b2e2b43e057a39e1c499c99ae1bdb9e939a8e7aaa84`.

Fifty previously reviewed cluster identities remained byte-for-byte stable
and were carried forward. The other 16 clusters were treated as unreviewed;
under the selected fail-closed policy all 206 of their member images were
rejected, even when they were merely very similar rather than exact
duplicates. This deliberately trades source capacity for lower recurrence
bias in the patch population. The resulting 18,997-record reviewed manifest
has SHA-256
`7863a3342a43fedd066af7196b0b45276856c7abcb32c4f8da3ad47c683b855d`.

The frozen selector then chose exactly 5,000 sources with all catalog/split
quotas satisfied. It selected 714/133/75 approved people sources and the
required 12/1/1 astronomy sources for train/validation/test. The canonical
selected-source JSONL has SHA-256
`469b1c9639ef42aecd97c040e7c3712e9b958e444acd09aeb7cb82c469444bcc`;
the 14,000-position recipe has SHA-256
`b13250f271c07ed638ed15060a51cc3f3d3d8e916519561f1d73683edbc20ff1`.

## Production corpus freeze

The selected training population was reordered before patch finalization so
every learning-curve prefix is nested and catalog-proportional. The ordered
selected-source manifest has SHA-256
`cb703713bec1d373bd41b01c5b98f311b93513dd4a813966414aa03cc78bd909`;
its canonical order record has SHA-256
`981316001ad70d53abee3d41f6341f0efa0bb96997cee4f81a16f998f0854d81`.
The 250/500/1,000/2,000/4,000 prefixes contain exactly 80% Open Images, 12%
Commons, and 8% Smithsonian sources. Within each catalog, sources are
deterministically interleaved across the 27 training-derived
brightness/chroma/texture cells.

Two source-finalization runs used separate empty checkpoint directories and
produced byte-identical 5,000-line, 238,521,201-byte manifests. The canonical
expanded `corpus-v1.jsonl` has SHA-256
`70d87ed392b23b46c2b5e34ab0bd704261ae174aea555a31a4eab05905129749`.
The production validator accepts it with exactly 4,000/500/500 sources and
1,024,000/64,000/64,000 unique in-bounds patch coordinates. A separate pass
reauthenticated all 5,000 original files: zero were missing or changed.

The independent identity-only pack runs are also byte-identical. Each
uncompressed TGPC is exactly 442,368,256 bytes with SHA-256
`3f84e2e58f1c6a01d721168ca5e7d5ae4b3e7547be633c78db67b31350f7d357`;
each independently generated deterministic gzip is 220,511,678 bytes with
SHA-256
`a683ab9b76e00b782229670b2bf42f62c2d6a281a817294a8c5939412337fd27`.
The authenticated payload SHA-256 is
`7e7164cbf6ad888f234f717c8e9066720817c6707798551cae93675349114593`.
Its fixed-v1 balance report has no deficient brightness, chroma, or texture
stratum. The common held-out split identities are
`76e1ef73ca5f7ac590f97c0846b13300bb070630a2cce81e4e79b11bef5babe9`
for validation and
`658576612d7bf5e07e550a57f2dd153adadf1568e79374d4ac3eab0f8346cc12`
for test; the augmented pack candidates must reproduce both exactly.

The two augmented recipes also reproduced byte-for-byte in independent clean
pack runs. The production/no-noise TGPC has SHA-256
`acf8483c21e4d8b6f01679e92663d4345fd13244beb853e94a2b799827fd3c35`
and its 251,806,462-byte deterministic gzip has SHA-256
`e073c59d362df9ffb57e96d48acdb0d0347dc607da872bd9b9376d544532a59a`.
The production/sensor-noise TGPC has SHA-256
`c372673325a831d9d55b8267ac5b310f02120ec0a21817733608eccc9c7853f2`
and its 293,549,581-byte gzip has SHA-256
`e422430b7b9caabaee1b1cdc27b04f01d7efd78943e7f87af3ff18bed9dc4cea`.
All six uncompressed pack runs contain exactly 1,152,000 authenticated records
and satisfy the fixed brightness/chroma/texture balance gate. Their validation
and test split payload digests are identical; augmentation and noise affect
training records only.

The common release metadata was generated from the frozen source manifest.
The attribution notice, rights report, and reconstruction URL/checksum list
have SHA-256 values
`67c386e1046e250894c3e7c2867bdb1760211a43bc16c219c8a6bc6e19dbd1ad`,
`fd010f21852b946d42b9803d2b112ec9c001f54ecd006c91104ac9d119e79b40`,
and
`5864a39d2bd621703501668b131dc686a1376a07ff5b310001c20870c4b67b14`,
respectively. The deterministic compressed 5,000-source manifest is
18,651,262 bytes with SHA-256
`e84efc9ff0c87f6084f933b235e812a44416795dfdcbb4d6169b53cc925e5dd4`.

## Augmentation selection

All three pack candidates were trained with the same optimized CPU K32/D51
fit: ten Gaussian iterations, thirty fixed-nu Student-t iterations, nu=3,
covariance floor 1e-6, batch size 4,096, an 8 GiB ceiling, and all eighteen
X-Trans phases. Evaluation used the complete common validation split.

| Candidate | PSNR (dB) | p99 RMS | Worst RMS | Median source PSNR | Phase spread (dB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| identity-only | 33.323200 | 0.102573 | 0.328938 | 35.070346 | 0.786502 |
| production-v1, no noise | **34.127307** | **0.095750** | 0.250287 | 36.032642 | **0.811888** |
| production-v1, sensor-v1 | 34.095935 | 0.096155 | **0.245970** | **36.096526** | 0.907601 |

Production-v1 without synthetic noise is frozen as the corpus recipe. It wins
the primary pooled-PSNR ranking, has lower p99 error and phase spread than the
sensor-noise candidate, and is the simpler of the two augmented recipes. The
sensor candidate's 0.00532 lower worst RMS does not overturn those preceding
criteria. Identity-only is 0.804 dB behind and is not competitive.

The winning candidate model has SHA-256
`c795420c0517596cf304ca92f32fdcd4d2719f6d752febf52375691481bab993`;
its validation report has SHA-256
`935d6faf48ecaff620bc082bc113a55a9fa499602c8192f7011c00598faff517`.
These CPU-fit files are selection evidence, not official release weights.

## Source-count convergence

The selected production corpus was trained at every frozen nested milestone.
Each model used the same complete validation split; the sample count shown is
the number of distinct training patches supplied to each of the eighteen phase
fits.

| Sources | Patches/phase | Train time (s) | PSNR (dB) | p99 RMS | Worst RMS | Median source PSNR | Phase spread (dB) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 250 | 64,000 | 334.81 | 33.857031 | 0.098426 | 0.269474 | 35.671266 | 0.914878 |
| 500 | 128,000 | 597.19 | 34.026833 | 0.097076 | 0.261106 | 36.032155 | 0.748686 |
| 1,000 | 256,000 | 1,109.74 | 34.021443 | 0.096168 | 0.266298 | 36.042833 | 0.907544 |
| 2,000 | 512,000 | 2,082.15 | 34.068286 | 0.096865 | 0.245439 | 36.130428 | 0.743218 |
| 4,000 | 1,024,000 | about 4,029* | 34.127307 | 0.095750 | 0.250287 | 36.032642 | 0.811888 |

\*The original 4,000-source selection run predated explicit `/usr/bin/time`
capture. Its 4,029-second figure is the phase-00-first-checkpoint to
phase-17-final-checkpoint span and is therefore a small lower bound. The later
canonical release runs record complete wall time directly.

The decisive 2,000-to-4,000 change is 0.059021 dB in pooled PSNR and 1.1519%
in p99 RMS. Both are below the frozen absolute 0.1 dB and relative 2% limits,
so `tgmr-corpus-v1` passes the convergence gate. The non-monotonic smaller
milestones are retained rather than hidden; the gate was defined on the final
doubling before any results were observed.

The 250/500/1,000/2,000 model SHA-256 values are, respectively,
`ddbb953c1bcd729d9db124e1cb4fea5f7ab43a2441ada136558c5fc6ed74047f`,
`fed97775410226eaa7cf35ebd20a96338386080ea5b5551d412dc44a79143f0a`,
`6abd542542511678795c29013a22f459bc346d647d63f1e98a2b97bdee43f20c`,
and
`ce45a3cf67fc4171a13490c28527f699f6e525a2d112a87ababca2a08fddced9`.

## Canonical model and untouched-test gate

Two clean canonical fits used the pinned GCC 13.3.0 Release trainer, classic
locale, one canonical worker per independently fitted phase, disabled floating
point contraction, the complete 4,000-source nested order, and the frozen
production-v1/no-noise corpus. Six independent phases were scheduled at once;
this changes only phase completion order because phases share no mutable state
and every phase retains the canonical single-thread reduction order.

The A and B output trees contain 738 checkpoints each: ten Gaussian, thirty
Student-t, and one final checkpoint for each of eighteen phases. A recursive
byte comparison found no difference in any checkpoint. The complete runs took
6,235.26 and 6,232.96 seconds. `/usr/bin/time` reported about 529 MiB maximum
RSS for an individual phase process; six phase processes were active at once.

The final authenticated identities are:

- model: `5707fbd67d1998ed3bac646ecce967297a2022776821d62944a24dbbb8615285`;
- payload: `276b42a099de7ecfacdcb403d63e2bc8c5b0fa9f21ef4a63c9d82ff6ada4f4eb`;
- canonical configuration:
  `13bf483df2104617785cd0c9be5eeb9386a841a00b2e30087881811e7f19bcbf`;
- trainer revision:
  `df0fc05b7b62f35c87a8ad42caf07e219721bddc81307cbf2756c9ba14d71ee6`;
- validation report:
  `7a0be89dee806586c74e413db307d48c8797e16aa773e6ad5de59d3a8acc574d`;
- validation-attached companion manifest:
  `d753847f9389d5dd07a04d10013f81ffe85c4cfda1dd5120f90eb3e6e46d79af`.

Both model files and both companion manifests are byte-identical. Two complete
validation runs are also byte-identical and report 34.127307 dB pooled PSNR,
0.095750 p99 patch RMS, 0.250287 worst patch RMS, 36.032642 dB median source
PSNR, and 0.811888 dB phase spread over all 64,000 validation patches. Strict
TGMR v2 verification accepts the final model and all embedded identities.

Only after those identities were frozen was the 64,000-patch, 500-source test
split opened. The direct TGMR validation reports 34.055553 dB. The native
RawTherapee population evaluator then decoded the reviewed source images,
synthesized every X-Trans phase, and compared the same coordinates with real
three-pass Markesteijn. Its one full-image/cropped-center proof has exactly
zero difference, and its balanced phase assignment gives 3,556 patches to ten
phases and 3,555 to the other eight.

| Test metric | TGMR | Markesteijn | Result |
| --- | ---: | ---: | ---: |
| Pooled PSNR | 34.063575 dB | 32.316274 dB | +1.747301 dB |
| p99 patch RMS | 0.095072 | 0.115206 | 17.48% lower |
| Worst patch RMS | 0.408201 | 0.483703 | lower |
| Median source PSNR | 36.312600 dB | 34.701333 dB | +1.611267 dB |

TGMR wins 408 of 500 sources, and the median paired source gain is
1.625443 dB. It also wins every pooled low/middle/high brightness, chroma, and
texture stratum. However, 92 sources lose, 65 lose by more than 0.5 dB, and
the worst source loses 6.169889 dB. The native report has SHA-256
`9e2b799fa829d8cac33576a113594891caf0396a1f30037f6164474aff8a6dba`;
the run took 267.80 seconds and peaked at 3,675,560 KiB RSS.

The release gate therefore **fails** for two independent reasons: the pooled
advantage is below the required 2 dB, and the worst source loss is far beyond
the permitted 0.5 dB. The improved aggregate tail does not override a frozen
per-source safety gate. Control-suite, RAF, publication, bundled-model, and
clean-branch work stop here rather than tuning against the untouched test
sources or weakening a threshold after observing it.

## Incomplete plan items

The following remain deliberately open rather than being represented as done:

- Synthetic-control and real-RAF qualification was not run because the earlier
  untouched licensed-population gate failed.
- Publication, model bundling, installed-data selection, and GUI release status
  remain blocked by the failed quality gate and the still-required legal review.
- Full production-corpus validation of the OpenMP-target AMDGCN backend and a
  physical NVPTX run. No production GPU speed claim is made from the tiny smoke
  fixture.
- Cross-compiler x86-64/ARM64 release matrix and physical ARM64 NEON execution.
- The final clean `codex/xtrans-tgmr-productization` branch was not created,
  because the plan requires a passing release candidate before that port.

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
  deterministic packing, authenticated interruption/resume for source
  finalization and TGPC packing, train-only augmentation/noise with invariant
  validation/test records, deterministic release metadata, one-step
  Student-t reference arithmetic, phase contracts, model loading, PP3 round
  trip, custom-model override, and complete fallback coverage.

The sanitizer run also exposed and fixed a pre-existing null dereference in
the framing-profile save helper when a complete PP3 is saved without a
`ParamsEdited` object. The ordinary full-profile path now preserves its prior
behavior without undefined behavior.

RawTherapee builds used `-j4`.
The generated `Testing/` directory is unrelated workspace content and remains
untracked and untouched.

## Hard-case retraining follow-up

The failed release gate is being investigated without altering or replacing
the corpus-v1 evidence.  Every one of the 16,789 retained files under the
external release directory has been recorded in a canonical size/SHA-256
inventory.  The inventory digest is
`06eb9550e378a8075f5ca67064a08e25aa69508a6036b6379645a989c633ee73`.
Both `official-a.tgmr` and `official-b.tgmr` remain byte-identical with model
SHA-256
`5707fbd67d1998ed3bac646ecce967297a2022776821d62944a24dbbb8615285`.

The 500-source test is now diagnostic rather than eligible for selecting the
next model: its digital-frame failure directly motivated the follow-up.  The
new controlled experiment compares direct versus sensor-realistic natural
patch rendering, each with and without balanced synthetic hard cases.  It will
select only from validation and independent control corpora.  A candidate that
passes those gates must subsequently face a newly frozen author- and
duplicate-isolated 500-source test before release qualification.

The current production-v1 model has also been rendered on DSCF0771 and recorded
separately from the older research TGMR assets.  See
[the hard-case retraining report](xtrans-tgmr-hard-case-retraining-report.md)
for artifact identities, progress, and the complete decision protocol.

The default-off C++ corpus packer now provides the sensor-physical natural
renderer, exact 0--5-percent balanced synthetic replacement schedules,
independently seeded synthetic controls, and an explicit external-control
validation path. A complete legacy-default repack remained byte-identical to
the frozen production TGPC, confirming that these facilities do not alter the
existing corpus path.

The complete 1,000-source ratio screen has since finished. No candidate passed
the predeclared gate. The best direct candidate, with 5% synthetic
replacements, preserved ordinary validation but reduced held-out synthetic MSE
by only 15.83% rather than the required 50%. Sensor-physical candidates gained
up to 0.166 dB on matched physical controls, but lost 0.177--0.312 dB on
ordinary validation and achieved at most 8.85% synthetic-MSE reduction against
their matched no-synthetic baseline. The 4,000-source factorial and subsequent
qualification were therefore stopped. The permanent production-v1 model and
all existing training outputs remain retained; see
[the hard-case retraining report](xtrans-tgmr-hard-case-retraining-report.md)
for the full table and artifact identities.
