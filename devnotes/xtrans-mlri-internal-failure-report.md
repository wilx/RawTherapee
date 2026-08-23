# X-Trans corrected-final MLRI internal failure decomposition

## Decision

**PARTIAL — the dominant opportunity is green-direction / second-pass
selection, not a final-residual limiter.**

The experiment isolates the sparse/high-frequency failure upstream of the
last red/blue residual correction:

- the second pass and the final green-guided reconstruction increase error on
  catastrophic pixels, while the final residual correction reduces their
  mean absolute error;
- supplying the true green guide to the unchanged final red/blue stage raises
  held-out missing-red/blue PSNR from **21.335 to 27.782 dB**;
- one of the real green directional candidates is substantially better for
  **80.11%** of catastrophic green pixels;
- however, no tested observable median, trimmed, medoid, local-green, tree, or
  logistic selector recovers that oracle without giving up the large
  Brick/Grass/Gravel/Page gains.

This rejects a residual-clipping fix, but it does not close MLRI. It narrows a
future experiment to a pass-0/pass-1 or directional-green selector. No
production method, default, profile, or GUI behavior was changed.

## Scope and corpus

The corpus contains 76 deterministic normalized linear-light RGB cases:

- three authenticated crops each from Astronaut, Brick, Grass, Gravel,
  Hubble Deep Field, Page, and the independent NASA Hydra star field;
- sparse impulses, saturated R/G/B points, one-to-three-pixel dots, lines,
  intersections, star fields, microtexture, point-density and line-thickness
  transitions;
- gradients, correlated fields, chromatic edges, coherent high-frequency
  texture, a frequency sweep, and other prior analytical controls;
- all 18 distinct X-Trans phase cells in the canonicalized MLRI domain.

Training uses Astronaut/Grass/Gravel plus the training synthetic families.
Brick/Hubble and validation synthetics select thresholds. Hydra, Page,
held-out CFA placements, structure transitions, and the frequency controls
remain untouched tests. No demosaiced RAW is used as truth.

The canonical corpus and complete measurements are in
[dataset.json](images/xtrans-mlri-internal/dataset.json) and
[results.json](images/xtrans-mlri-internal/results.json).

## Exact corrected-final pipeline

The traced path is the real
`CORRECTED_BLUE_DIAGONAL_GUIDES_FINAL_ONLY` implementation, not a generic MLRI
diagram.

| Stage | Code | Operation and support | Decisions |
| --- | --- | --- | --- |
| CFA phase masks | `MlriMasks`, `phaseMask()` | 18 phase planes derived from the internal 6x6 cell | deterministic CFA phase only |
| Pass-0 guides | `makeGreenGuides()` | phase-specific horizontal/vertical 3-11-tap filters, 5-tap diagonals, and a 3x3 cross completion | four H/V/diagonal guide families |
| Pass-0 green | `interpolateGreen()`, sigma 2 | 5/7/13-tap Laplacians; 7x7 local guided-MLRI regression; fixed residual interpolation; eight half-plane/diagonal Gaussian candidates | inverse squared directional energy fuses eight candidates |
| Pass-0 provisional R/B | `interpolateChromaSites()`, `interpolateCenterGreenSites()`, `interpolateRemainingGreenSites()` | phase-specific guided regressions, residual filters, and completion of chroma and green sites | directional inverse-energy fusion |
| Pass-1 guides | `makeGreenGuides()` | reuses the entire pass-0 R/G/B result as every directional guide | no new direction construction |
| Pass-1 green and provisional R/B | same functions, sigma 1 | repeats the reconstruction with narrower half-Gaussians | same nonlinear regression and fusion |
| Final tentative R/B | `finalRedBlue()` | completed pass-1 green drives 5x5 asymmetric Laplacians and an 11x11 local guided-MLRI model | no final direction bank |
| Final residual | `finalRedBlue()` | measured-site residual `raw - tentative`; fixed 5x5, 7x5, and 5x7 X-Trans interpolation kernels | fixed phase-specific interpolation |
| Final output | `runMlri()` | `clip(tentative + correction)` for R/B plus clipped pass-1 green | final-only variant discards provisional R/B; no reinjection or false-color pass |

`guidedMlri()` is nonlinear: it fits local Laplacian-domain gain and DC
offset, then combines overlapping local models using inverse residual MSE.
The directional fusions are also nonlinear because their weights depend on
the reconstructed local gradients.

The trace records both pass green planes, both provisional R/B results, all
eight green-direction candidates in each pass, final tentative R/B,
measured-site residuals, interpolated correction, unclipped sum, clipped
output, and stage timings. The production tiled entry point never constructs
the trace.

## Stage-wise error

Whole-RGB metrics include all missing colors in the 16-pixel interior. The
held-out test pool deliberately includes the pathological frequency sweep, so
its aggregate PSNR is a stress metric rather than a natural-image score.

| Held-out stage | PSNR (dB) | p99 abs. error | RMS |
| --- | ---: | ---: | ---: |
| Markesteijn | 23.465 | 0.32885 | 0.06710 |
| pass 0 provisional RGB | 23.626 | 0.37106 | 0.06587 |
| pass 1 provisional RGB | 23.324 | 0.38602 | 0.06820 |
| pass-0 green through final R/B | 23.156 | 0.38842 | 0.06953 |
| final tentative / guide-only | 22.911 | 0.39496 | 0.07153 |
| current corrected-final | 23.213 | 0.38842 | 0.06908 |
| oracle true green through real final R/B | 30.635 | 0.00036 | 0.02939 |

For pixels that finally exceed the train-frozen catastrophic threshold, mean
absolute error grows by **0.01348** from pass 0 to pass 1 and by another
**0.01473** from pass 1 to the final tentative estimate. The final correction
then changes it by **-0.01009** on average and worsens only **37.67%** of these
catastrophic pixels. The failure therefore does not first arise in the final
residual.

### Natural-source red/blue behavior

The table evaluates only missing red/blue samples. It shows why simply
removing pass 1 is not an acceptable general fix.

| Source | Markesteijn | pass-0 guide final | corrected-final | oracle alpha |
| --- | ---: | ---: | ---: | ---: |
| Astronaut | 35.249 | 33.906 | 32.719 | 32.991 |
| Brick | 45.468 | 51.319 | 58.126 | 60.543 |
| Grass | 28.858 | 34.914 | 41.252 | 43.899 |
| Gravel | 33.846 | 39.009 | 45.352 | 47.971 |
| Hubble | 36.069 | 33.328 | 31.568 | 31.605 |
| Hydra (untouched) | 69.360 | 64.060 | 61.248 | 61.260 |
| Page (untouched) | 26.664 | 31.881 | 40.305 | 43.231 |

Pass 1 is strongly beneficial on coherent repeated/fine texture but harmful
on the two sparse star-field sources and Astronaut. This is precisely the
safe-versus-coherent distinction that prior final-output features failed to
resolve.

## Candidate oracle

At the completed red/blue level, only **10.58%** of 10,194 catastrophic
samples have a pass-0, pass-1, tentative, or pass-0-final candidate with at
most half the current final error. Thus **89.42%** are Type B at this late
stage: all retained completed-channel candidates are already wrong.

The earlier green direction bank is different:

- **80.11%** of 3,102 catastrophic missing-green pixels have a directional
  candidate with at most half the fused pass-1 error;
- pass-0 direction-oracle PSNR is **28.702 dB** on held-out green samples,
  versus **21.887 dB** for fused pass-1 green;
- combined pass-0/pass-1 7x7 direction-oracle PSNR is **29.085 dB**;
- pass-1 directional spread has **0.964 AUC** for identifying catastrophic
  green error.

This establishes candidate headroom, but not a usable selector. On held-out
green samples the best tested deterministic robust control is still the
existing pass-0 weighted result at 22.336 dB. Combined median reaches 22.081,
trimmed pass 0 reaches 22.167, and medoid pass 0 reaches 22.070 dB. Selecting
the direction closest to a local measured-green estimate is worse at 20.561
dB. The candidate identity—not merely candidate spread—remains unavailable.

At the completed-output level, an oracle choosing only between pass-0-guide
final and current pass-1 final reaches **21.728 dB** and p99 **0.50031**, versus
21.335 dB and 0.55109 for current MLRI. The unrestricted completed-candidate
oracle reaches 22.209 dB. There is therefore useful pass-selection headroom,
but much less than the directional-green or true-green ceilings.

## Final residual and correction strength

The correction helps **50.89%** and hurts **49.11%** of missing R/B samples;
4.71% exceed the train-frozen severe-harm threshold. Oracle correction
rejection improves held-out R/B from **21.335 to 21.449 dB**, and p99 only
from **0.55109 to 0.54118**. Allowing alpha in `[-0.5, 1.5]` reaches 21.632 dB,
still far below the green-guide oracle.

The practical alpha distribution is not a clean limiter split: 41.65% is at
zero, 42.41% at one, and 15.94% is interior. More importantly, validation
selects **no clipping** over every absolute, RMS-relative, MAD-relative, and
CFA-variation-relative threshold. Median/trimmed aggregation of completed
R/B candidates is also worse than current MLRI.

Raw correction magnitude correlates with catastrophic final error (AUC
0.846), but its normalized forms are weak: correction/MAD AUC is 0.600,
correction/CFA-variation 0.567, and correction/RMS 0.544. In contrast,
pass-0/pass-1 disagreement reaches 0.913. This again points to refinement,
not correction magnitude.

## Internal predictability

Two asymmetric gates were trained with source-level splits and a threshold
selected only on validation for at least 90% severe recall.

| Gate | Validation AUC / recall / rejection | Held-out AUC / recall / rejection | Held-out output |
| --- | --- | --- | --- |
| final-residual logistic | 0.968 / 90.19% / 8.37% | 0.955 / 98.95% / 26.22% | 21.026 dB, p99 0.58186 |
| pass-1-refinement logistic | 0.947 / 90.02% / 16.11% | 0.948 / 98.51% / 31.60% | 21.159 dB, p99 0.51857 |
| pass disagreement threshold | 0.953 / 90.02% / 11.77% | 0.921 / 95.41% / 31.53% | 21.174 dB, p99 0.51861 |

The internal features exceed the previous external detector's 77.10% severe
recall, including on the untouched safety pool, but the apparent
classification success does **not** produce a good reconstruction. Both
learned gates reject almost exactly the regions carrying MLRI's useful
texture gain. The residual gate collapses to guide-only quality; the
refinement gate collapses nearly to the complete pass-0 reconstruction.
Calibration also shifts sharply from validation to the held-out pool.

The pass-0/pass-1 oracle proves that safety gating alone could add 0.393 dB
and reduce held-out p99 by about 9.2%. The logistic gate instead falls from
21.335 to 21.159 dB despite 98.51% severe recall. It catches dangerous
refinements, but rejects too many indispensable texture refinements to recover
any of the oracle gain.

This is the important oracle/predictability separation: harmful refinements
are recognizable in aggregate, but the current features cannot preserve the
beneficial refinements with similar disagreement.

## Runtime profile

The traced research path spent approximately:

- 57.6% in the two provisional chroma/completion stages;
- 38.2% in the two green interpolation stages;
- 1.2% in initial guide construction;
- 3.0% in final red/blue reconstruction.

The final residual is therefore neither the main quality limiter nor the main
runtime cost. These are instrumented scalar research timings and include
trace capture; they are not a production benchmark.

## Diagnostic maps

Each map shows truth, final output, final and guide errors, green directional
spread, correction magnitude, oracle alpha, and harmful-correction locations:

- [Hubble](images/xtrans-mlri-internal/map-hubble-0.png)
- [Hydra](images/xtrans-mlri-internal/map-nasa-hydra-starfield-0.png)
- [isolated impulse](images/xtrans-mlri-internal/map-white-impulse.png)
- [Brick](images/xtrans-mlri-internal/map-brick-0.png)
- [Page](images/xtrans-mlri-internal/map-page-0.png)
- [saturated red edge](images/xtrans-mlri-internal/map-control-saturated_red_gray.png)

The Hubble map visibly places large guide error and directional spread on the
sparse bright structures. Harmful final corrections are much more dispersed
and are not the origin of those structures.

## Answers to the required questions

### Pipeline

Corrected-final MLRI consists of phase-specific pass-0 guides, an eight-way
guided/residual green interpolation, provisional R/B completion, a full
sigma-1 refinement pass, and a separate green-guided final R/B estimate plus
fixed residual interpolation. Final-only returns this last R/B result and
pass-1 green.

### Failure origin

Sparse/star-field error becomes large in the green reconstruction and is
amplified by the second pass and subsequent green-guided R/B tentative stage.
The final residual usually improves—not destroys—the already-wrong guide.

### Candidate information

A good actual directional green candidate exists for 80.11% of catastrophic
green pixels, but a good completed-channel/pass candidate exists for only
10.58% of catastrophic R/B pixels. Recoverable information is usually lost
during green candidate fusion/refinement before final R/B completion.

### Residual behavior

Absolute corrections are larger near failures, but normalized residual
outlier statistics are weak and fixed/adaptive clipping does not validate.
Pass and direction disagreement are much more diagnostic.

### Oracle limiter

Oracle residual rejection gives only 0.114 dB on the held-out R/B pool. It
cannot approach the 6.45 dB oracle-green headroom. It preserves natural
texture only because it changes relatively little; it does not solve sparse
failures.

### Predictability

Internal features greatly improve severe-event recall over the prior external
gate, but their selected reconstructions lose the aggregate benefit and do
not beat current corrected-final MLRI. They detect risk, not the correct
alternative.

### Practical opportunity

The mistake is localized, but not yet locally correctable without sacrificing
strong texture performance. The narrow justified next experiment is a
controlled pass-0/pass-1 green candidate selector or bounded blend that is
trained/evaluated directly for retained reconstruction quality—not another
residual limiter. Until such a selector succeeds, production MLRI should
remain unchanged.

## Verification

- native CTest contract and synthetic trace/decomposition tests pass;
- the 76-case run completed for all 18 internal CFA phase cells;
- every traced plane and reconstructed output was finite;
- the complete residual identity was checked before clipping;
- Python unit tests cover CFA phases, oracle alpha, local robust statistics,
  ROC/operating thresholds, and scalar metrics;
- generated data, results, maps, and their hashes are bound by
  [manifest.json](images/xtrans-mlri-internal/manifest.json).
