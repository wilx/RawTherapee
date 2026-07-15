# Phase 9 DemosaicNet X-Trans quality-gate report

## Status

Phase 9 is implemented as a developer-only experiment. Phase 10 is **not
approved** by the current evidence. The native graph and raw wrapper are
correct against their independent numerical references, but the portable
direct convolution is far outside the 10x runtime gate on the 40 MP test file,
and the direct-linear real-RAF output has conspicuous CFA-phase texture.

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

Gamma-2.2 was not exported at full resolution after the fatal speed result.
The wrapper-selection decision is consequently still open, but Phase 10 is
blocked independently of that decision.

## Phase 10 criteria

| Criterion | Result | Evidence |
| --- | --- | --- |
| Wrapper selected on ten upstream scenes | Not run | External corpus remains optional and the speed gate already fails. |
| Within 0.5 dB mean CPSNR of Markesteijn | Not measured | Requires all ten upstream scenes. |
| Wins at least 3 of 10 upstream scenes | Not measured | Requires all ten upstream scenes. |
| No >2 dB/severe analytical regression | Fail | Multiple smooth/challenge cases regress; linear flat black is not exact. |
| No seams/orientation/NaN/clipping faults | Pass for numerical corpus | All tracked Phase 9 parity and CFA tests pass. |
| No worse on DSCF0771 earrings | Fail for linear | Visible CFA-phase texture; gamma not run at full resolution. |
| At most 10x Markesteijn | **Fail** | Greater than 148x from completed linear warm-up versus 3.94 s baseline. |
| At most +4 GiB peak RSS | Not fully measured | Workspace is bounded at 443.6 MB; complete process peak was not retained. |
| Six public RAF generations reviewed | Not run | Cannot change the already-failed Phase 10 decision. |

The developer method should remain hidden. A future attempt should first
replace or substantially optimize the direct OIHW convolution; only then is it
worth completing the external ten-image and six-camera matrix. A camera-linear
model trained for RawTherapee's scaled raw domain is another plausible path if
gamma-2.2 continues to show real-image CFA-phase artefacts.
