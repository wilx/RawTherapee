# Phase 9 DemosaicNet X-Trans quality-gate report

## Status

Phase 9 is implemented as a developer-only experiment. The focused gamma-2.2
test is complete and the Gharbi drop-in experiment is **closed**. Phase 10 is
not approved. The native graph and both raw wrappers are correct against their
independent numerical references. Gamma-2.2 fixes the visible linear-wrapper
colour cast and most of its CFA-phase texture, but narrowly fails the agreed
per-channel phase-reduction gate and remains approximately 77x slower than
Markesteijn on the 40 MP test file.

The two hidden PP3 method strings remain:

* `demosaicnet-xtrans-linear`
* `demosaicnet-xtrans-gamma22`

They are intentionally absent from `XTransSensor::Method`, the method string
list, GUI, history, translations, defaults, and fast-export controls. An
explicit `RT_DEMOSAICNET_XTRANS_MODEL` path is required. Markesteijn three-pass
is the unconditional, diagnostic fallback.

## Correctness evidence

The tracked raw-wrapper corpus contains seven scaled scalar mosaics and
full-size normalized RGB results generated through the independent Python
RTNN reader and untiled PyTorch graph. It covers canonical linear and gamma
domains, translation, rotation, reflection, and both orientations of the
production 168-pixel core boundary. Its canonical manifest SHA-256 is:

```text
bd415eb33ecb10c01c7ef127039e6ef69267d9398d01edbb7a339a661790b526
```

Native output satisfies `5e-6 + 1e-5*abs(reference)` for every value. Separate
tests cover all 18 unique phase/orientation matrices, coordinate and colour
round trips, 1-pixel and sub-receptive images, signed-output clamping,
gamma-2.2, no sample reinjection, non-finite input, unsupported CFA rejection,
thread-safe model-cache identity, and horizontal/vertical seam cases.

The pinned upstream mosaic definition was checked directly. Its documented
cell is the same canonical matrix used here:

```text
G B G G R G
R G R B G B
G B G G R G
G R G G B G
B G B R G R
G R G G B G
```

Source: `mgharbi/demosaicnet`, revision
`959e9d1630976b421d5af5e35b2e2a01f5630e5c`,
`demosaicnet/mosaic.py`.

## Analytical smoke results

The 48-pixel neutral-DNG smoke run used false-colour suppression and unrelated
processing disabled. It authenticated RTNN SHA-256
`b4dd6ea4ba535e7f4aea249a2d589a80ca8584f60a605a5bce468c989529ccc2`
and reported one active worker with an 18,483,200-byte executor workspace.
There were no fallbacks.

Selected CPSNR results in dB:

| Scene | Linear | Gamma 2.2 | Markesteijn |
| --- | ---: | ---: | ---: |
| Constant | 54.50 | 54.03 | 88.34 |
| Gradient | 48.51 | 45.96 | 55.50 |
| Impulse | 40.15 | 45.88 | 37.94 |
| Frequency sweep | 10.31 | 10.14 | 10.41 |
| Saturated edges | 22.20 | 30.92 | 24.59 |
| Flat black | 52.82 | exact | exact |

Gamma is materially better on impulses, saturated edges, and flat black;
linear is better on the smooth gradient. These analytical cases do not replace
the planned ten-image upstream selection set.

## DSCF0771.RAF

Local source:

```text
/home/wilx/Pictures/2026-07-12 Áňa odjíždí na tábor/RAF/DSCF0771.RAF
SHA-256 26106d7da2ba9a87caebffd4bd5e5fb0371cc8a2b6139ec724b59ef4112842c0
Camera FUJIFILM X-T50, 40 MP
```

The initial direct-linear 7752x5178 export completed without fallback. The
portable executor used 24 workers; its reported workspace requirement is:

```text
18,483,200 bytes per worker
443,596,800 bytes for 24 active workers
```

The warm-up began after the analytical run at approximately 22:20:44 and
produced its 230 MiB TIFF at 22:30:28, giving a conservative elapsed lower
bound of 584 seconds. A separately timed Markesteijn export took 3.94 seconds
and peaked at 1,803,460 KiB RSS. Thus the demonstrated neural/Markesteijn time
ratio is greater than 148x, already far above the 10x limit. The remaining
median repetitions were stopped because they could not reverse the gate and
would have consumed roughly another hour.

The interrupted benchmark reused its timing filename before the stop, so a
reliable neural peak-RSS value was not retained. The benchmark now uses unique
per-invocation timing files. The 443.6 MB bounded executor workspace is well
below the revised 4 GiB allowance, but the memory criterion remains recorded
as not fully measured rather than passed.

Visual comparison of crop `(3450,1750,700,500)` shows a strong repeating
three-row/three-column CFA-phase texture across skin, hair, and the metallic
earrings in the linear result. Markesteijn is substantially smoother. The
pattern is not a 168-pixel tile seam: native tiled/untiled seam fixtures pass,
and row-modulo analysis identifies the CFA period. Linear therefore also fails
the real-image quality criterion on this file.

## Focused gamma-2.2 drop-in test

One full-resolution gamma-2.2 export was run with the same disabled unrelated
processing as the existing references. The completion diagnostic
authenticated RTNN SHA-256
`b4dd6ea4ba535e7f4aea249a2d589a80ca8584f60a605a5bce468c989529ccc2`;
there was no fallback. The 7752x5178 16-bit TIFF has SHA-256
`32dbb39586f0a49a402889e8600cfbad00930ea3da78b99a4037a163a30fb41e`.
All three comparison TIFFs have the same 748-byte ICC profile with SHA-256
`17aebbdf8a88c39b07eb881dcd824eb1cf9828914d5e74a7324d7a31045e8871`.

Execution measurements were:

```text
wall time                         301.72 seconds
engine demosaic time              298.359299 seconds
peak RSS                          1,807,360 KiB
active workers                    24
workspace per worker              18,483,200 bytes
total executor workspace          443,596,800 bytes
```

Against the 3.94-second Markesteijn reference this is approximately 76.6x
slower. The measured RSS difference is only 3,900 KiB over the 1,803,460 KiB
Markesteijn process peak, so the revised memory gate passes.

The comparison tool selected the lowest-detail half of 19,256 globally
aligned 6x6 crop blocks using Markesteijn alone, subtracted each block's
per-channel mean, and accumulated the residual by 3x3 pixel phase. Normalized
phase RMS was:

| Channel | Linear | Gamma 2.2 | Markesteijn | Gamma reduction from linear |
| --- | ---: | ---: | ---: | ---: |
| Red | 0.0059232 | 0.0012979 | 0.0005539 | 78.09% |
| Green | 0.0038893 | 0.0009719 | 0.0004460 | 75.01% |
| Blue | 0.0020701 | 0.0006668 | 0.0007821 | 67.79% |

Gamma is within the absolute limit `max(2 * Markesteijn, 0.0015)` for every
channel, and its absolute blue phase RMS is slightly lower than Markesteijn's.
However, the agreed contract also requires at least 75% reduction from linear
in every channel. Blue fails that condition, so the objective phase verdict
is fail.

The gamma-minus-Markesteijn crop mean difference is
`(-0.002216, +0.000415, +0.001811)` in normalized RGB. Its common luminance
shift is `0.0000035` and its channel-difference range is `0.004027`; both are
inside the `0.005` limits. The objective colour verdict therefore passes.

Visual review at 100% and nearest-neighbour 500% passes: gamma removes the
conspicuous repeating stipple and cooler cast of the linear wrapper, with no
material remaining global or local cast on the face and earring crop. The
final objective verdict nevertheless fails because the gate is conjunctive.
The canonical external comparison report has SHA-256
`7a1982a049edf2ba6be5305b3616112add32439c092a672cfdab7739a78f8659`;
the report and generated TIFFs remain untracked.

## Phase 10 criteria

| Criterion | Result | Evidence |
| --- | --- | --- |
| Wrapper selected on ten upstream scenes | Closed before full matrix | Gamma focused test failed the agreed objective drop-in gate. |
| Within 0.5 dB mean CPSNR of Markesteijn | Not measured | Requires all ten upstream scenes. |
| Wins at least 3 of 10 upstream scenes | Not measured | Requires all ten upstream scenes. |
| No >2 dB/severe analytical regression | Fail | Multiple smooth/challenge cases regress; linear flat black is not exact. |
| No seams/orientation/NaN/clipping faults | Pass for numerical corpus | All tracked Phase 9 parity and CFA tests pass. |
| No worse on DSCF0771 earrings | Pass for gamma visually | Gamma removes the obvious linear stipple and colour cast. |
| Focused gamma phase contract | **Fail** | Blue reduction is 67.79%, below the required 75%. |
| At most 10x Markesteijn | **Fail** | Gamma is approximately 76.6x slower than the 3.94 s baseline. |
| At most +4 GiB peak RSS | Pass | Gamma peak is 3,900 KiB above the measured Markesteijn peak. |
| Six public RAF generations reviewed | Not run | Cannot change the failed drop-in and runtime decisions. |

The developer method remains hidden. Under the agreed constraints there will
be no postprocessing, retraining, GUI exposure, model packaging, or inference
optimization. The implementation and measurements remain as a reproducible
research result.
