# X-Trans Student-t GMR RawTherapee integration report

## Result

Decision: **GO — visually useful experimental demosaicer.**

The frozen K32/S9/q8 Student-t Gaussian-mixture-regression model now runs at
RawTherapee's actual X-Trans demosaic stage as the GUI- and PP3-selectable
method `tgmr`, displayed as **TGMR (experimental)**.  The model remains an
external authenticated file.

The integration reproduces the independent research executor within the
required numerical tolerance, preserves native CFA samples exactly, maps all
18 X-Trans patterns, retains 79.2% of current standalone throughput, and falls
back loudly and completely to Markesteijn when the model cannot be used.

Four local X-T50 RAFs produced plausible output.  The strongest observed
benefit is `DSCF0771`: TGMR largely removes the magenta/cyan segmentation on
the metallic earring that remains with Markesteijn.  TGMR is generally a
little smoother in fine foliage and cobblestones.  No obvious global color
shift, tile seam, highlight failure, or correlated high-ISO artifact was seen
in this small set.

This is not a production-quality or cross-camera conclusion.  The evaluated
RAFs all came from one camera model, there is no ground truth, and the learned
model's redistribution terms remain unresolved.

## Frozen model and runtime contract

The engine accepts only:

- artifact: `tgmr32-native.bin`;
- byte size: `6,073,164`;
- SHA-256: `6279b6a593ef4b595b1aff246182682b60c7eea701373ee2eb9f80cfcb50485c`;
- 18 phase models, 7x7 support, K=32, central S9 coarse likelihood, q8
  shortlist;
- Student-t `nu=3`, posterior temperature `T=4`, mismatch floor
  `tau=0.0003`, and observed-RGB DC;
- float32 inference and exact measured-center restoration.

The loader authenticates the complete file before parsing.  It then checks
the magic, version, dimensions, observed-index permutations, center channel,
target channel permutation, S9 positions, positive Cholesky diagonals, unique
phase patterns, finite coefficients, and canonical end of file.  Successful
models are held in a thread-safe process-lifetime cache keyed by path; failed
loads are not cached.

The external path is supplied with:

```sh
export RT_XTRANS_TGMR_MODEL=/tmp/xtrans-tgmr-reduce/tgmr32-native.bin
```

There is no model lookup, download, embedded weight array, or model-path GUI
preference.

## RawTherapee input contract and normalization

`RawImageSource::preprocess()` calls `scaleColors()` before demosaicing.
For X-Trans, `scaleColors()`:

1. combines decoded and user-adjusted black offsets;
2. subtracts the selected channel's black value and floors negative values at
   zero;
3. constructs `scale_mul[c]` from `ref_pre_mul[c] / max(ref_pre_mul)` and
   `65535 / (white[c] - black[c])`;
4. multiplies every physical CFA sample by its channel's `scale_mul`.

Consequently the demosaic input is black-subtracted, preprocessing-balanced,
camera-native sensor RGB in RawTherapee's nominal 0..65535 float scale.  The
channel-dependent reference preprocessing multipliers are already part of
that scale.  TGMR does not apply another camera white balance or any color
matrix.  Values above the nominal range can survive when decoded data exceeds
the selected white level; the new method does not clamp them.

The exact model conversion is:

```text
model observation = rawData / 65535
RawTherapee prediction = model prediction * 65535
```

The physically sampled output channel is then restored from the original
`rawData` float, avoiding even division/multiplication roundoff.  There is no
TGMR-specific clipping, sample smoothing, sharpening, denoising, false-color
suppression, or image-specific adaptation.

Domain caveat: **the model was trained on normalized linear RGB images
remosaicked synthetically, whereas real RAF samples are camera-native sensor
RGB measurements.**  The small smoke set suggests useful transfer, but does
not establish general transfer across cameras, illuminants, or noise levels.

## Phase, boundary, SIMD, and threading behavior

The engine does not infer a phase from center color.  It obtains the actual
6x6 X-Trans matrix from `RawImage`, constructs each residue's complete 7x7
color pattern, and matches that pattern against the 18 authenticated model
patterns.  Unsupported or ambiguous matrices fail before output is accepted.
The flat test interface also takes a physical crop origin; all five nonzero
origin cases pass.

The 128x128 tiles are scheduling units only.  Every 7x7 observation is read
from the full input mosaic, so internal tile edges are not reflected.  Only
the actual image boundary uses reflection without edge repetition and a
three-pixel receptive halo.

On x86 GCC/Clang, runtime CPU detection selects small AVX2/FMA function-target
kernels; the rest of RawTherapee retains its baseline ISA.  Other CPUs compile
and use the scalar implementation.  The optimized path retains S9
coefficient-major AoSoA scoring and component-bucketed full q8 evaluation
across eight pixels.  OpenMP distributes 128x128 tiles and respects the
runtime thread limit.  The measured RAF runs used 18 workers; observed
parallel utilization and throughput show that this call was not serialized by
a surrounding OpenMP region.

Estimated TGMR workspace is 419,840 bytes per active worker plus the roughly
6 MB model and 210 KB prepared AoSoA cache.  At 18 workers this is about 7.2
MiB of worker scratch.  No pixels-by-49 full-frame observation matrix exists.

## Engine correctness

The independent research comparison used the exact deterministic mosaic
generator from `tgmr_optimized_benchmark`, 141x133 pixels, tile 128, chunk
512, and the same authenticated artifact.  RawTherapee's test adapter alone
performed the 65535 scale conversion.

| Test | RMS | Max | Native sample |
|---|---:|---:|---|
| scalar engine vs independent scalar research executor | `6.1550e-8` | `5.9605e-7` | exact |
| AVX2 engine vs scalar engine | `5.0299e-8` | `4.1723e-7` | exact |
| crop vs full frame, five physical origins | `9.6463e-9` | `3.5763e-7` | exact |

The crop differences are expected floating-point lane-grouping changes at an
internal tile boundary, not phase differences.  The maximum is about one to
six float32 ULPs depending on value and is far below the `1e-5` acceptance
bound.

Native CTest coverage includes:

- missing, truncated, and exact-size wrong-digest artifacts;
- canonical model loading and process-cache identity;
- constant, RGB gradient, chromatic gradient, edge, saturated point, tiny
  star, and periodic scenes;
- scalar/AVX2 parity and finite output;
- exact native sample restoration;
- all 18 unique X-Trans matrices;
- non-finite input rejection;
- nonzero crop origins `(1,0)`, `(0,1)`, `(3,3)`, `(5,5)`, and `(7,11)`;
- PP3 save/load round trip of `Method=tgmr`.

The model-backed tests skip with CTest code 77 when
`RT_XTRANS_TGMR_MODEL` is absent.  The mandatory loader/PP3 contract remains
self-contained.

## Performance

The synthetic comparison was rerun on the same Ryzen 9 5900X immediately
after integration, rather than comparing against only the older 6.85 MP/s
record.  Both paths used 18 threads, AVX2/FMA, 128x128 tiles, chunk 512, one
warm-up, and three measured runs.

| Image/geometry | Pixels | Threads | Research MP/s | Engine MP/s |
|---|---:|---:|---:|---:|
| deterministic 7738x5164 mosaic | 39,959,032 | 18 | 6.630 | 5.248 |

The engine retains 79.2% of standalone throughput.  Its 5.25 MP/s is above
the planned 4 MP/s investigation gate.  The remaining cost includes conversion
between RawTherapee's 65535 scale and model units and writes to three separate
engine planes.  No model reload or full-frame observation buffer occurs in
the timed kernel.

Complete CLI timings used identical minimal PP3 profiles with automatic
exposure, sharpening, capture sharpening, impulse denoising, directional
pyramid denoising, and false-color suppression disabled.  Times are one
measured run each and therefore descriptive rather than a formal benchmark.

| RAF | Resolution | ISO | Markesteijn time | TGMR time | TGMR kernel | Notes |
|---|---:|---:|---:|---:|---:|---|
| DSCF0771 | 7752x5178 | 320 | 3.78 s | 11.28 s | 7.47 s | people, hair, metallic earrings |
| DSCF5043 | 5178x7752 | 640 | 4.00 s | 10.78 s | 7.34 s | night architecture, cobbles, point lights |
| DSCF5595 | 7752x5178 | 800 | 3.58 s | 10.88 s | 7.46 s | saturated food, metal, printed packaging |
| DSCF3790 | 7752x5178 | 125 | 3.94 s | 11.14 s | 7.37 s | distant foliage and fine branches |

For DSCF0771, peak RSS was 1,996,832 KiB with TGMR and 2,004,632 KiB with
Markesteijn.  The difference is within measurement noise and supports the
bounded-workspace design.  Complete TGMR processing was about 3.0x the
Markesteijn wall time in this small sample.

## Real-RAW visual observations

The full-resolution TIFFs remain external under `/tmp/xtrans-tgmr-rt/`.  The
canonical DSCF0771 TGMR comparison assets are tracked alongside the earlier
methods:

- [one-third full frame](images/xtrans-neural/DSCF0771/DSCF0771-tgmr-full-third.png);
- [500-percent nearest-neighbour earring crop](images/xtrans-neural/DSCF0771/DSCF0771-tgmr-earring-500.png).

Their [separate manifest](images/xtrans-neural/DSCF0771/tgmr-manifest.json)
binds them to the RAW, TGMR model, and full-resolution source TIFF without
changing the identities of the earlier comparison manifests.  The comparison
order used below was Markesteijn then TGMR.  There is no ground truth, so
disagreement alone is not classified as error.

| RAF/crop | Markesteijn | TGMR observation |
|---|---|---|
| DSCF0771 earring, 500% nearest-neighbor | visible magenta/cyan segmented pixels on the gold hoop | segmentation is largely absent; hoop is more coherent and neutral; slightly smoother |
| DSCF0771 face/hair | ordinary Markesteijn texture | plausible color and structure; no global cast or repeating 6x6 texture seen |
| DSCF5043 clipped street lamp | stable highlight with surrounding flare | similarly stable highlight and flare; no colored ringing or point-light failure |
| DSCF5043 cobblestones | slightly more granular microtexture | slightly smoother surfaces and reduced worm-like texture; edges remain plausible |
| DSCF5595 bowl/metal edge, ISO 800 | fine colored edge pixels remain visible | somewhat cleaner dark/chromatic boundary; no obvious structured-noise conversion |
| DSCF3790 foliage | slightly crisper/granular fine foliage | slightly smoother foliage; no large false color, phase pattern, or seam seen |

The TIFF color appearance and full-frame channel balance were visually very
close in all four comparisons.  The limited set did not reveal a systematic
camera-domain hue shift.  Highlights and point-like light structures remained
stable.  The principal tradeoff observed so far is Markesteijn's extra
microcontrast versus TGMR's smoother, less artifact-prone fine reconstruction.

No star-field RAF was available in this smoke run.  The ISO 800 example is not
a substitute for a controlled high-ISO/noise evaluation.

## Failure behavior

With a missing model, the CLI emits once per demosaic invocation:

```text
TGMR X-Trans error [IO]: model=/tmp/does-not-exist-tgmr.bin: cannot open TGMR model: /tmp/does-not-exist-tgmr.bin; falling back to 3-pass (Markesteijn)
```

The TGMR engine never exposes a successful partial result.  Any load, CFA,
allocation, non-finite, or inference failure returns to `RawImageSource`,
which reruns Markesteijn three-pass over all output planes.  A real CLI
fallback export completed successfully.  Its decoded image differed from a
separate direct Markesteijn export in only 388 of roughly 120 million channel
samples, with normalized RMSE `5.09e-6`; this is consistent with the existing
parallel Markesteijn path's run-to-run float/quantization variation, not
residual TGMR tiles.

## User instructions

Start the current development GUI with the reviewed external model:

```sh
cd /home/wilx/RawTherapee
export RT_XTRANS_TGMR_MODEL=/tmp/xtrans-tgmr-reduce/tgmr32-native.bin
./build/dev/rtgui/rawtherapee
```

Open an X-Trans RAF, then select:

```text
Raw
  -> Demosaicing
  -> Method
  -> TGMR (experimental)
```

Switch between **TGMR (experimental)** and **3-pass (Markesteijn)** at 100%
to compare them.  Saving the processing profile writes:

```ini
[RAW X-Trans]
Method=tgmr
```

A verified minimal CLI profile is:

```ini
[Version]
AppVersion=5.12
Version=352

[RAW X-Trans]
Method=tgmr
CcSteps=0
Border=0
```

Process a RAF with:

```sh
RT_XTRANS_TGMR_MODEL=/tmp/xtrans-tgmr-reduce/tgmr32-native.bin \
OMP_NUM_THREADS=18 \
./build/dev/rtgui/rawtherapee-cli \
  -Y -o /tmp/tgmr-output.tif -p /tmp/tgmr.pp3 -t -b16 \
  -c /path/to/image.RAF
```

Successful processing prints the reviewed digest, K32/S9/q8 contract, tile
geometry, AVX2 status, worker count, bounded workspace, elapsed kernel time,
and throughput.  If that completion line is absent and a fallback warning is
present, the exported image is Markesteijn rather than TGMR.

## Required-question answers

- **Frozen algorithm:** yes.  Independent scalar and optimized parity establish
  that the engine implements the reviewed K32/S9/q8 model rather than a new
  approximation.
- **All phases/crops:** all 18 authenticated patterns and five nonzero origins
  pass.  RawTherapee currently invokes the full raw mosaic at physical origin
  zero; the adapter test entry preserves origin for crop/preview callers.
- **Normalization:** input is RawTherapee's black-subtracted,
  preprocessing-balanced nominal 0..65535 camera-linear sensor scale.  TGMR
  divides/multiplies by 65535 and adds no color transform or clamp.
- **SIMD:** the Ryzen 9 5900X selected AVX2/FMA.  The scalar path is functional
  and agrees within `5e-7` maximum on the independent probe.
- **Threading:** tile-level OpenMP uses 18 configured workers, bounded
  per-worker scratch, and no observed nested-parallel serialization.
- **Performance:** standalone engine inference is 5.248 MP/s; a 40.14 MP RAF
  spends about 7.3--7.5 seconds in TGMR and about 10.8--11.3 seconds in the
  complete minimal CLI pipeline, roughly 3x Markesteijn end-to-end.
- **External loading/fallback:** the environment-variable path, digest check,
  process cache, PP3/CLI method, loud failure, and complete Markesteijn
  fallback all work.
- **Real images:** output is visually plausible in the four-image X-T50 set.
  TGMR improves the known metallic-earring false color and is otherwise close
  in color, with somewhat smoother fine detail.  No severe highlight, seam,
  phase, or noise artifact was observed.
- **Domain mismatch:** no obvious systematic mismatch appeared in this limited
  single-camera evaluation, but broader camera/noise/illuminant coverage is
  required before claiming that synthetic linear-RGB training is generally
  adequate for camera-native X-Trans data.

The experiment therefore succeeds at its intended boundary: the statistically
successful and CPU-viable Student-t GMR remains useful when placed at
RawTherapee's real camera-RAW demosaic stage.  It should remain explicitly
experimental while broader RAF evaluation and model licensing/distribution
are unresolved.

## Production-v1 corpus model comparison

The later corpus-productization work produced a separately authenticated model
from 4,000 licensed training sources.  Its SHA-256 is
`5707fbd67d1998ed3bac646ecce967297a2022776821d62944a24dbbb8615285`.
It has now been run on DSCF0771 using the same neutral PP3 and derivative
geometry as the research model:

- [production-v1 one-third full frame](images/xtrans-neural/DSCF0771/DSCF0771-tgmr-production-v1-full-third.png);
- [production-v1 500-percent earring crop](images/xtrans-neural/DSCF0771/DSCF0771-tgmr-production-v1-earring-500.png);
- [production-v1 asset manifest](images/xtrans-neural/DSCF0771/tgmr-production-v1-manifest.json).

The complete TIFF authenticated the production-v1 model and did not fall back.
It remains external under the hard-case experiment directory.  These assets do
not replace the existing research-model pair: both generations remain available
for direct visual comparison.  The production-v1 earring remains coherent and
substantially less segmented than Markesteijn, with a few isolated colored
pixels still visible on the hoop.  Later hard-case retraining candidates must
preserve or improve this result.

The subsequent controlled 1,000-source hard-case screen tested all six frozen
synthetic ratios with both direct and sensor-physical natural-patch rendering.
No candidate passed its validation gates: the best direct model reduced
synthetic-control MSE by 15.83% rather than the required 50%, while every
sensor-physical model lost more than 0.1 dB on ordinary validation. Therefore
no retrained candidate was admitted to the 4,000-source stage or substituted
into RawTherapee. The runtime continues to use the preserved production-v1
model when that external artifact is selected, and the DSCF0771 images above
remain its authoritative comparison. Detailed results are in
[the hard-case retraining report](xtrans-tgmr-hard-case-retraining-report.md).
