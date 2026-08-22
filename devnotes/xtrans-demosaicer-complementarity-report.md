# X-Trans demosaicer complementarity and oracle upper bound

## Decision

**HIGH COMPLEMENTARITY and GO, narrowly focused on Markesteijn plus
corrected-final MLRI.** The current algorithms contain complementary
reconstruction information at practical patch sizes. On four authenticated,
camera-linear, remosaicked natural crops,
Markesteijn is the best individual method at 36.801 dB. A ground-truth oracle
choosing between Markesteijn and MLRI reaches 37.570 dB with 7x7 decisions and
37.310 dB with 15x15 decisions. Those gains are 0.769 dB/16.2% error reduction
and 0.509 dB/11.1% error reduction. A 15x15 convex blend oracle reaches 37.734
dB, 0.933 dB above Markesteijn and 0.423 dB above hard switching.

This is a research GO, not a production-method decision. The experiment proves
headroom, not that a selector can infer the oracle without ground truth. The
next justified experiment is therefore a deterministic, observable-feature
selector/blender for this pair. It is not justified to build a six-way hybrid:
at 15x15 on natural data, removal of MLRI costs 0.320 dB, while removal of
triangulated RGB, triangulated color difference, global B, and sparse alias
costs only 0.010, 0.031, 0.014, and 0.000 dB respectively.

The complete canonical measurements are in
[complementarity.json](images/xtrans-oracle/complementarity.json), SHA-256
`f433c4a3d241b3ab46c9f95a01edd160a39a96db823e92dbd8316d3d15c2c260`.

## Compared implementations

The native runner feeds one scalar float32 mosaic to the actual C++ methods and
collects planar camera-linear float32 output. It does not use the TIFF output
pipeline, gamma, color management, sharpening, denoising, false-color
suppression, or clipping.

| ID | Implementation used |
| --- | --- |
| `markesteijn` | RawTherapee three-pass/CIE-Lab Markesteijn |
| `mlri-final` | corrected blue guides and direct-final reconstruction, avoiding both apparent MATLAB copy/paste mistakes and the provisional/final blend implicated in blue overshoot |
| `triangulated-rgb` | independent fixed X-Trans triangulation |
| `triangulated-chroma` | green first, then triangulated R-G and B-G |
| `global-b` | whole-image quadratic green/color-difference solve, 50 PCG iterations |
| `sparse-alias` | replica-aware global Fourier OMP, with the three DC means estimated only from corresponding physically sampled CFA values |

Global B was selected over C because the preceding global experiment found B
the stronger general representative; C retained CFA-phase chromatic edge
points and was much slower. Rafinazari was not resurrected because its severe
saturated-edge and real-earring failures had already made it obsolete for this
comparison.

The new `demosaicMarkesteijnXTransReference()` entry point is only a developer
array adapter around the existing implementation. The ordinary production
member call now delegates to the same body with its original CFA and camera
matrix, so production processing is unchanged.

## Dataset and comparison contract

Thirteen deterministic 96x96 scenes cover the requested gradients, correlated
luma/chroma, vertical/horizontal/diagonal colored edges, saturated red/gray and
blue/gray edges, saturated quadrants, an impulse, frequency sweep,
near-Nyquist monochrome detail, periodic chroma, and a support-transition
scene.

The natural set uses center 192x192 crops from the authenticated scikit-image
`astronaut`, `coffee`, `hubble_deep_field`, and `rocket` sources. Their source
hashes, licenses, and exact crop coordinates are recorded in the canonical
JSON. Stored sRGB is decoded to linear light before X-Trans mosaicking; a
demosaiced X-Trans image is never used as ground truth. These four crops yield
well over a thousand natural 7x7 comparison blocks, but they are still a small
research set rather than a camera benchmark.

All metrics omit the same 16-pixel border. Patch decisions are nonoverlapping
blocks anchored at the evaluated interior's upper left. A smaller final block
is used at the right and bottom. Metrics include all RGB component values;
interpolation-only and physically measured-sample errors are also recorded.

All native methods preserve measured samples to float-conversion precision:
the greatest natural-set sample error is `8.57e-8` normalized for Markesteijn
and `2.88e-8` for the other four native methods. Sparse alias is a global
unconstrained reconstruction and changes measured samples by as much as
`0.0356`; this is reported rather than silently reinjected.

## Individual quality

Natural-set pooled camera-linear quality is:

| Method | PSNR |
| --- | ---: |
| Markesteijn | 36.801 dB |
| corrected-final MLRI | 34.298 dB |
| triangulated color difference | 33.920 dB |
| triangulated RGB | 31.987 dB |
| global B | 31.513 dB |
| sparse alias | 28.703 dB |

Markesteijn is decisively the best single natural-image method. MLRI is 2.50
dB behind in aggregate, so its value is complementary local information, not
replacement quality.

Across synthetic and natural scenes combined, sparse alias is the strongest
single result at 22.558 dB because it exactly recovers several specially
observable synthetic structures. On natural data it is the worst method. This
is why synthetic and natural conclusions are reported separately.

## Oracle headroom above Markesteijn

The all-method natural oracle gives:

| Decision block | Oracle PSNR | Gain over Markesteijn |
| ---: | ---: | ---: |
| pixel | 40.949 dB | +4.148 dB |
| 3x3 | 38.517 dB | +1.716 dB |
| 7x7 | 37.765 dB | +0.964 dB |
| 15x15 | 37.380 dB | +0.579 dB |
| 31x31 | 37.150 dB | +0.349 dB |

The pixel bound is deliberately not used as the GO signal. The 7x7 and 15x15
gains survive spatial aggregation, satisfying the experiment's negative
control. The best natural pair selected by 15x15 quality is Markesteijn plus
MLRI:

| Decision block | Pair PSNR | Gain | Error reduction |
| ---: | ---: | ---: | ---: |
| pixel | 39.060 dB | +2.259 dB | 40.6% |
| 3x3 | 38.029 dB | +1.228 dB | 24.6% |
| 7x7 | 37.570 dB | +0.769 dB | 16.2% |
| 15x15 | 37.310 dB | +0.509 dB | 11.1% |
| 31x31 | 37.138 dB | +0.337 dB | 7.5% |

Adding triangulated color difference makes the best three-method 15x15 result
37.356 dB. Adding all remaining methods reaches 37.380 dB. Thus almost all
actionable natural headroom is already present in the pair.

The natural-image headroom curves for the best pair, best three-method set, and
all methods are [patch-headroom.png](images/xtrans-oracle/patch-headroom.png).

## Pairwise complementarity

The pairwise natural matrix below is ranked by 15x15 gain over the better pair
member. This ranking must be read together with absolute PSNR: a large gain
between two weak methods can remain worse than Markesteijn.

| Pair | Better member | Pixel oracle | 7x7 | 15x15 | 15x15 gain | Centered RGB error correlation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| triangulated RGB + global B | 31.987 | 36.485 | 34.897 | 34.532 | +2.545 | 0.291 |
| MLRI + triangulated chroma | 34.298 | 38.119 | 36.477 | 36.119 | +1.821 | 0.399 |
| triangulated chroma + global B | 33.920 | 36.997 | 35.613 | 35.225 | +1.305 | 0.427 |
| MLRI + triangulated RGB | 34.298 | 37.695 | 35.915 | 35.563 | +1.265 | 0.301 |
| global B + sparse alias | 31.513 | 34.425 | 32.863 | 32.425 | +0.912 | 0.276 |
| triangulated RGB + sparse alias | 31.987 | 34.723 | 33.252 | 32.816 | +0.828 | 0.377 |
| **Markesteijn + MLRI** | **36.801** | **39.060** | **37.570** | **37.310** | **+0.509** | **0.529** |
| MLRI + sparse alias | 34.298 | 36.798 | 34.962 | 34.633 | +0.335 | 0.263 |
| MLRI + global B | 34.298 | 35.350 | 34.645 | 34.507 | +0.209 | 0.708 |
| Markesteijn + global B | 36.801 | 38.485 | 37.195 | 37.008 | +0.207 | 0.465 |
| triangulated chroma + sparse alias | 33.920 | 35.755 | 34.295 | 34.056 | +0.136 | 0.355 |
| Markesteijn + triangulated chroma | 36.801 | 37.992 | 36.934 | 36.848 | +0.047 | 0.618 |
| triangulated RGB + triangulated chroma | 33.920 | 34.840 | 33.994 | 33.943 | +0.023 | 0.658 |
| Markesteijn + triangulated RGB | 36.801 | 37.954 | 36.867 | 36.823 | +0.022 | 0.475 |
| Markesteijn + sparse alias | 36.801 | 37.938 | 36.859 | 36.801 | +0.000 | 0.328 |

The lowest natural centered RGB correlation is MLRI plus sparse alias at
0.263. The lowest combined-set correlation is triangulated chroma plus sparse
alias at 0.324. Neither is the most useful pair. Markesteijn plus MLRI has a
higher 0.529 correlation but is the only pair combining the best baseline with
large, persistent natural-image gain. It is therefore the strongest practical
pair even though it is not the lowest-correlation pair.

The full RGB/L/C1/C2 centered and uncentered matrices are in the JSON. The
natural-image centered RGB visualization is
[error-correlation.png](images/xtrans-oracle/error-correlation.png).

## Winner coherence and stability

Markesteijn/MLRI 15x15 selections remain balanced rather than degenerating to
one method. The MLRI fraction is 40%, 32%, 47%, and 35% for astronaut, coffee,
Hubble, and rocket. The maps contain 9--17 connected regions, and their
normalized boundary fractions are 0.013--0.028. Agreement between 7x7 and
15x15 labels is 0.747, 0.851, 0.683, and 0.679 respectively.

Pixel decisions are far more fragmented. On astronaut the pair has 2,325
pixel-winner regions and boundary fraction 0.393, while 7x7 has 53 regions and
15x15 has 9. The persistent patch result is therefore qualitatively different
from a misleading pixel oracle.

Representative pair and all-method maps are under
`devnotes/images/xtrans-oracle/`, including:

- [Markesteijn/MLRI astronaut, 7x7](images/xtrans-oracle/winner-mark-mlri-natural-astronaut-7.png)
- [Markesteijn/MLRI astronaut, 15x15](images/xtrans-oracle/winner-mark-mlri-natural-astronaut-15.png)
- [all-method astronaut, pixel](images/xtrans-oracle/winner-natural-astronaut-1.png)
- [all-method astronaut, 15x15](images/xtrans-oracle/winner-natural-astronaut-15.png)

Common-scale signed-error panels make the different residual structure visible
directly: [natural astronaut](images/xtrans-oracle/error-mark-mlri-natural-astronaut.png)
and [diagonal red/green edge](images/xtrans-oracle/error-mark-mlri-diagonal_red_green.png).
Neutral gray means zero error; RGB deviations show the sign and color of the
residual, with the shared 99.5th-percentile scale printed above both methods.

## Structural breakdown

The synthetic winners are interpretable rather than identical:

- triangulated RGB wins the smooth gradient and frequency sweep;
- triangulated color difference wins the correlated smooth-chroma field;
- corrected-final MLRI wins the diagonal red/green edge, saturated quadrant
  intersection, and near-Nyquist monochrome detail;
- global B wins the isolated impulse;
- Markesteijn exactly reconstructs the tested axis-aligned saturated red/gray
  and blue/gray transitions in the evaluated interior;
- sparse alias exactly reconstructs the observable vertical/horizontal
  red/green edges and periodic chroma case and wins the mixed support-transition
  scene.

The sparse-alias wins are real upper-bound evidence for its Fourier model, but
they do not transfer into a useful natural hybrid component. Against
Markesteijn its natural gain falls from 1.137 dB at pixel resolution to 0.058
dB at 7x7 and exactly 0.000 dB at 15x15. At 15x15 it is selected nowhere in
three natural crops and effectively nowhere in the fourth. This answers the
alias-focused hypothesis negatively for a practical hybrid.

The all-method 7x7 gain over each scene's best method is near zero on the
simple exact edge/texture cases, 0.21 dB on the correlated field, 0.04 dB on
the impulse, 0.12 dB on the frequency sweep, and 1.68 dB on the mixed
support-transition scene. At 15x15 only the mixed scene retains a large
synthetic gain (2.01 dB).

## Luminance, chroma, and frequency behavior

On natural crops, selecting one whole RGB result using chroma error gives
37.738 dB at 7x7 and 37.351 dB at 15x15. Luminance-based selection gives
37.681 and 37.354 dB. Independent selection of L, C1, and C2 reaches 38.027
and 37.592 dB, 0.262 and 0.212 dB above the corresponding hard RGB oracle.
There is therefore additional component-level complementarity, but it is
smaller than the basic Markesteijn/MLRI opportunity.

At pixel resolution the independent-component natural bound is 42.303 dB,
again demonstrating why pixel oracles must not drive the decision.

The synthetic radial spectra show materially different residual character.
MLRI's nonzero error cases place roughly 70% of C1 and 74% of C2 error energy
in the high-frequency band, whereas Markesteijn places about 35% and 39%
there. Markesteijn has two exact synthetic edge rows, so its averaged band
fractions sum below one when zero-energy cases contribute zeros. The result is
consistent with MLRI recovering some broad structures while leaving sharper
residuals; it does not by itself provide an observable selector.

## Hard selection versus convex blending

For Markesteijn plus MLRI on natural data:

| Block | Hard selection | Convex blend | Blend over hard | Blend over Markesteijn |
| ---: | ---: | ---: | ---: | ---: |
| pixel | 39.060 dB | 39.430 dB | +0.369 dB | +2.629 dB |
| 7x7 | 37.570 dB | 37.941 dB | +0.370 dB | +1.140 dB |
| 15x15 | 37.310 dB | 37.734 dB | +0.423 dB | +0.933 dB |

The blend advantage persists and slightly grows at 15x15. The algorithms
often contain complementary partial information rather than one output being
wholly correct. A future experiment should therefore evaluate both a hard
selector and a bounded blend confidence, while retaining Markesteijn exactly
in regions where confidence is low.

## Natural error tail

Using nonoverlapping 7x7 blocks, Markesteijn has median/p90/p95/maximum RMS
errors of `0.00403 / 0.02207 / 0.03209 / 0.10826`. The all-method oracle gives
`0.00359 / 0.01968 / 0.02792 / 0.08421`. Thus the oracle improves not only the
mean but also the high-error tail: p95 falls about 13% and the maximum falls
about 22%.

Worst-block visual comparisons are tracked for each source:

- [astronaut](images/xtrans-oracle/worst-natural-astronaut.png)
- [coffee](images/xtrans-oracle/worst-natural-coffee.png)
- [Hubble Deep Field](images/xtrans-oracle/worst-natural-hubble-deep-field.png)
- [rocket](images/xtrans-oracle/worst-natural-rocket.png)

These crops are shown as ground truth, Markesteijn, MLRI, and all-method 7x7
oracle. They are diagnostic nearest-neighbor renderings of linear results,
encoded to sRGB only for display.

## Runtime

Across the 17 small scenes, total native reconstruction time was 0.046 s for
Markesteijn, 10.586 s for MLRI, 0.008 s for triangulated RGB, 0.007 s for
triangulated chroma, and 0.102 s for global B. Python sparse alias recovery
took 26.187 s. Median per-scene times were 0.0026, 0.3603, 0.0003, 0.0003,
0.0047, and 1.2527 s respectively. These are offline small-image diagnostics,
not substitutes for the already recorded full-RAF timings. The final canonical
run spent 75.164 s on Python metric/oracle analysis and figure construction
after reconstruction. Memory was not separately measured because all cases are
at most 192x192 and no optimization claim is made.

## Limitations and next experiment

- The oracle knows ground truth. No deployable reliability feature has been
  identified yet.
- The natural set has four authenticated crops, not a large photographic
  corpus. A future selector must use held-out crops and additional linear RGB
  sources.
- MLRI remains vastly too slow in its present correctness-first
  implementation. Selector feasibility and MLRI optimization are separate
  gates.
- Nonoverlapping block selection establishes spatial headroom but imposes a
  grid. A practical method would require overlapping confidence and seam-safe
  routing or blending.
- Sparse alias has an unfairly favorable global signal model on the synthetic
  periodic cases, although its DC means are estimated from CFA samples rather
  than ground truth. Its natural 15x15 marginal contribution is nevertheless
  zero.

**GO:** Markesteijn and corrected-final MLRI have sufficiently complementary
errors that a 15x15 natural-image patch oracle improves Markesteijn by 0.509 dB
and reduces squared error by 11.1%, with coherent winner regions concentrated
in a modest number of contiguous patches. The remaining four methods add only
0.070 dB beyond this pair at 15x15 and do not justify a six-way hybrid.

The recommended follow-up is a **Markesteijn/MLRI deterministic selector and
blend-prediction experiment** using only observable mosaic/algorithm features.
It must train or tune on one subset and evaluate on held-out natural images,
and it must compare predicted selection to three baselines: Markesteijn,
15x15 hard oracle (+0.509 dB), and 15x15 convex oracle (+0.933 dB). Failure to
recover a meaningful fraction of that held-out headroom would close the hybrid
route despite this oracle GO.
