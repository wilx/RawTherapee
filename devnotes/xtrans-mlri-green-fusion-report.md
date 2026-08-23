# X-Trans MLRI green-candidate selection and bounded-fusion experiment

## Decision

**PARTIAL — the candidate bank contains substantial recoverable information,
but the tested MLRI-internal evidence cannot select it reliably enough to
justify a modified demosaicer.**

The validation-admissible winner is the simplest hard control: choose the
lowest-current-energy candidate from the eight pass-1 directions.  On the
untouched test pool it changes final RGB from **23.3352 to 23.3701 dB**, lowers
p99 absolute error from **0.38400 to 0.38074**, and captures only **2.58%** of
the 7x7 combined directional-oracle MSE gap.  It preserves 89-95% of the
pass-1 gain on Brick, Grass, Gravel, and Page, but improves Hubble by only
0.010 dB and worsens the independent Hydra star field by 0.133 dB.

This is not a GO.  The practical result is too small, does not fix both sparse
sources, and remains unstable on the controlled structure transitions.  No
production demosaicing method, profile, default, or GUI behavior was changed.

## Method and data discipline

The experiment reuses the exact corrected-final MLRI implementation and the
source-level split from the preceding internal decomposition.  It adds five
untouched test cases progressing from isolated points through aligned points,
a dotted line, a continuous line, and repeated microtexture.  The resulting
81-case normalized linear-light corpus contains:

- natural Astronaut, Brick, Grass, Gravel, Hubble, Page, and NASA Hydra crops;
- all 18 distinct X-Trans translation phases;
- impulses, colored dots, stars, thin lines, intersections, saturated edges,
  density and thickness transitions, coherent texture, and frequency stress
  cases;
- 41 training, 15 validation, and 25 untouched test cases.

Hubble participates only in validation.  Hydra, Page, the new coherence
transition, held-out CFA positions, and frequency controls remain untouched
tests.  No demosaiced RAW is used as ground truth.

The development trace now records, for both passes and all eight directions:

- the real green candidate value;
- the exact directional statistic recovered from `eightWeights()`;
- the normalized weight actually used by `weightedEight()`.

Native tests require the eight weights to sum to one at every missing-green
site.  Every experimental guide copies native green mosaic samples exactly
before the unchanged `finalRedBlue()` stage is run.  Per-candidate
`guidedMlri()` gain, offset, and fit MSE cannot be attributed after the current
candidate construction because the function returns only its completed model
estimate; this limitation is recorded rather than approximated as an exact
fit diagnostic.

The complete canonical measurements are in
[results.json](images/xtrans-mlri-green-fusion/results.json), with the corpus
contract in [dataset.json](images/xtrans-mlri-green-fusion/dataset.json).

## Validation selection policy

Three validation policies illustrate the central ambiguity:

1. Pooled validation PSNR selects a 3x3 pairwise combined-bank model.  It
   improves validation Hubble but loses 0.445 dB on the untouched aggregate
   and destroys much of Page.
2. Unconstrained source-balanced validation selects a 7x7 pass-0-only ridge
   score.  It gains 0.370 dB on the test stress aggregate and improves the two
   star fields, but it already loses 8.56 dB on validation Brick and later
   loses 10.45 dB on Page.  It is the prohibited collapse toward pass 0.
3. The final, prompt-aligned rule maximizes mean per-source validation PSNR
   only among methods retaining at least 80% of Brick's pass-1 MSE gain and
   not worsening Hubble.  This selects minimum-energy pass 1 and uses no test
   information.

The first two are retained as instability diagnostics.  All reported
practical conclusions use the third rule.

## Directional oracle structure

### Green-selected candidate oracles

Every selected green guide below is passed through the real final R/B stage.

| Candidate bank / spatial loss | Green PSNR (dB) | Final RGB PSNR (dB) | Final p99 |
| --- | ---: | ---: | ---: |
| current pass-1 fusion | 22.075 | 23.335 | 0.38400 |
| pass 0, pixel oracle | 28.828 | 26.948 | 0.11097 |
| pass 0, 3x3 oracle | 26.011 | 26.024 | 0.14532 |
| pass 0, 7x7 oracle | 23.708 | 24.678 | 0.27007 |
| pass 0, 15x15 oracle | 23.804 | 24.682 | 0.25628 |
| pass 1, pixel oracle | 23.292 | 23.346 | 0.36294 |
| pass 1, 7x7 oracle | 22.715 | 23.543 | 0.37214 |
| combined, pixel oracle | 29.125 | 27.017 | 0.10765 |
| combined, 3x3 oracle | 26.662 | 26.238 | 0.13624 |
| combined, 7x7 oracle | 24.458 | 24.951 | 0.26492 |
| combined, 15x15 oracle | 25.145 | 25.113 | 0.24876 |

The 7x7 oracle survives with a **1.615 dB final-RGB advantage** over current
MLRI, so useful headroom is not exclusively pixel-scale.  At the same time,
combined-oracle label boundary density falls only from 0.842 per pixel to
0.489 at 3x3, 0.394 at 7x7, and 0.280 at 15x15.  Identity is spatially more
coherent at larger support, but remains fragmented.  Connected-component
sizes are not very meaningful on the disconnected missing-green X-Trans
lattice, so boundary density is the safer statistic.

The combined oracle selects pass 1 for 55.0% of missing-green pixels, 63.8%
at 3x3, 72.5% at 7x7, and 74.9% at 15x15.  Pass 1 therefore adds genuinely
useful candidates rather than merely perturbing pass 0.  At 7x7, natural
patches select pass 1 86.2% of the time, while synthetic sparse-failure
patches select pass 0 87.4% of the time.  That strong conditional structure
explains both the oracle headroom and why a pass-0 collapse appears tempting.

## Green loss versus final-RGB loss

Selecting a candidate using downstream final-RGB patch error gives:

| Spatial support | Final-RGB oracle PSNR (dB) | Green-selected oracle PSNR (dB) |
| ---: | ---: | ---: |
| pixel | 27.089 | 27.017 |
| 3x3 | 26.485 | 26.238 |
| 7x7 | 25.030 | 24.951 |
| 15x15 | 26.276 | 25.113 |

The green and final-RGB oracle labels agree on only 32.8% of pixels and 35.9%
at 7x7.  Nevertheless, their reconstructed final RGB is close through 7x7;
green squared error is therefore a usable training objective at small support,
although not an exact proxy for downstream candidate identity.  At 15x15 the
larger 1.16 dB gap makes final-RGB optimization materially different.

## Convex and pass-level upper bounds

True green lies inside the scalar candidate interval for **90.3%** of pass-0,
**80.2%** of pass-1, and **94.0%** of combined missing-green samples.  The
corresponding combined convex-hull oracle reaches **29.327 dB green** and
**26.902 dB final RGB**, compared with 30.840 dB final RGB for the true-green
upper bound.  Blending has more headroom than hard selection; the candidate
bank usually brackets the needed value.

The bounded G0/G1 blend is much weaker: its 7x7 oracle reaches 23.463 dB final
RGB, only 0.128 dB above current.  A 7x7 oracle that decides whether to execute
pass 1 reaches 23.601 dB, chooses pass 0 for 26.3% of patches, and implies at
most a **12.2%** total traced-runtime saving.  The practical ridge pass blend
reaches only 23.346 dB.  Adaptive execution has theoretical headroom, but no
tested observable selector realizes it safely.

## Candidate predictability

The candidate-cost models use values, exact energies and weights, local
measured-green consistency, candidate gradients/Laplacians, pass
counterparts, robust spread, weight entropy, raw structure-tensor alignment,
local CFA variation, and phase encoding.

| Model | Test Spearman | Top-1 | Top-2 inclusion | Candidate green PSNR |
| --- | ---: | ---: | ---: | ---: |
| absolute ridge cost | 0.868 | 6.90% | 13.29% | 22.198 |
| pixel-centered ridge cost | 0.170 | 8.12% | 15.18% | 22.217 |
| depth-3 tree cost | 0.911 | 10.80% | 19.54% | 22.275 |
| pairwise logistic ranking | 0.018 | 8.91% | 15.56% | 21.331 |

The high absolute ridge/tree correlation mostly predicts whether a pixel is
generally difficult; it does not rank the 16 candidates at that pixel.  The
centered-cost control removes this common difficulty and barely improves
top-1 selection.  Pairwise AUC falls from 0.618 on validation to 0.598 on
test; margin-weighted accuracy falls from 92.65% to 74.22%.  Candidate identity
therefore remains poorly observable even though candidate error magnitude is
predictable.

The centered ridge's strongest variables are raw-green variation,
log-directional energy, combined spread, candidate local variation, mosaic
gradient, current weight, distance from fused green, and candidate gradient.
Geometric alignment is present but does not solve within-pixel ranking.  The
unavailable per-candidate guided-regression coefficients could contain new
information, but the present experiment provides no evidence that exposing
them would close the large ranking gap.

## Hard selection and bounded blending controls

Uniform and median candidate fusion are worse than current MLRI.  Altering
the current pass-1 weight exponent from 0 through 4 changes held-out final RGB
only within 23.293-23.373 dB and selects different exponents on validation and
test.  The best admissible hard control, minimum-energy pass 1, reaches 23.370
dB.  It is better than learned selection under the validation safety rule but
still recovers only 2.58% of the 7x7 oracle gap.

Minimum-energy pass 0 reaches 23.867 dB on the aggregate held-out stress pool,
but this result is not admissible: pass-0 methods fail the known validation
Brick constraint and sacrifice Page/coherent detail.  It is reported only as
evidence of split and structure dependence, not as a practical winner.

## Natural-source final RGB

| Source | Split | Markesteijn | current MLRI | pass 0 | selected practical | 7x7 oracle | convex oracle |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Astronaut | train | 37.555 | 34.967 | 36.221 | 35.097 | 36.378 | 39.860 |
| Brick | validation | 46.974 | 59.392 | 52.335 | 57.916 | 61.765 | 77.078 |
| Grass | train | 30.189 | 42.463 | 35.796 | 41.164 | 43.923 | 58.203 |
| Gravel | train | 35.136 | 46.547 | 39.891 | 45.109 | 48.252 | 62.709 |
| Hubble | validation | 38.382 | 33.710 | 35.603 | 33.719 | 36.934 | 39.800 |
| Hydra | test | 71.790 | 63.386 | 66.231 | 63.253 | 65.804 | 77.658 |
| Page | test | 27.855 | 41.324 | 32.617 | 40.169 | 43.106 | 56.419 |

The selected method retains 90.1% of Brick, 90.4% of Grass, 89.2% of Gravel,
and 95.3% of Page's pass-1 MSE gain.  That is enough to satisfy the texture
retention criterion, but it does not materially repair Hubble and it regresses
Hydra.  A practical method cannot be justified by the aggregate 0.035 dB gain.

## Density, thickness, and coherence transitions

The selected method always uses pass 1 and changes only its direction.  Its
behavior is not a stable sparse-to-coherent transition:

- isolated and aligned points are essentially unchanged;
- dotted-line final RGB falls from 28.06 to 27.69 dB;
- continuous-line final RGB rises from 26.85 to 30.42 dB;
- repeated texture falls from 21.10 to 20.81 dB;
- the three-point density case falls from 61.45 to 47.34 dB;
- 1-3-pixel lines improve slightly, while dense point patterns can regress.

The method recognizes some coherent line geometry but does not monotonically
move from sparse safety behavior to texture refinement.  This is another form
of the same candidate-ranking ambiguity.

## Runtime accounting

In the 81-case trace, pass 1 consumes 46.3% of MLRI stage time: 17.8% for
green and 28.5% for provisional chroma/completion.  The 7x7 adaptive-pass
oracle suggests only a 12.2% total saving because it skips pass 1 for 26.3%
of patches.  Feature extraction, model fitting, and 113-method evaluation are
research Python/native-batch timings and not production estimates.

## Diagnostic maps

Each map shows truth and both fused green passes, current error, ridge-selected
and 7x7-oracle identities, regret, spread, predicted cost, and practical final
RGB error:

- [Hubble](images/xtrans-mlri-green-fusion/map-hubble-0.png)
- [Hydra](images/xtrans-mlri-green-fusion/map-nasa-hydra-starfield-0.png)
- [Brick](images/xtrans-mlri-green-fusion/map-brick-0.png)
- [Grass](images/xtrans-mlri-green-fusion/map-grass-0.png)
- [Page](images/xtrans-mlri-green-fusion/map-page-0.png)
- [isolated impulse](images/xtrans-mlri-green-fusion/map-white-impulse.png)
- [saturated red edge](images/xtrans-mlri-green-fusion/map-control-saturated_red_gray.png)

## Required answers

### Oracle structure

Substantial directional headroom survives 3x3, 7x7, and 15x15 selection, but
candidate labels remain boundary-dense.  It is spatially structured, not
smoothly piecewise constant.

### Pass structure

Pass 0 dominates sparse synthetic failures; pass 1 dominates natural and
coherent patches and contributes most larger-window oracle selections.  Pass
1 adds useful candidate diversity rather than merely amplifying pass 0.

### Candidate predictability

Internal observables predict overall difficulty but rank candidates poorly.
Exact current energy is still the best safe practical selector.  Per-model
guided-regression fit statistics were not available for a faithful ablation.

### Hard versus blend

Hard practical selection slightly beats current weighting, while ordinary
mean/median and learned soft weights do not.  Oracle convex fusion is far
stronger than hard selection, proving that useful values often lie between
candidates, but practical weights do not approach it.

### Sparse structures

No.  The safe validation-selected method does not improve both Hubble and
Hydra.  Methods that strongly improve both do so by collapsing toward pass 0
and lose coherent texture.

### Coherent texture

The selected pass-1 control largely preserves Brick/Grass/Gravel/Page, but
learned pass-0 selectors do not.  The conflict remains unresolved.

### Final RGB

Better green does propagate strongly through the unchanged final R/B stage:
the 7x7 green oracle gains 1.615 dB final RGB, and the convex oracle gains
3.567 dB.  The limitation is candidate identification, not downstream R/B.

### Runtime

An adaptive-pass oracle could theoretically avoid about 12% of traced work,
but no quality-safe practical pass decision was found.

## Conclusion

The experiment answers the central question negatively for the tested local,
deterministic evidence:

> A good green candidate frequently exists inside MLRI, and the candidate
> interval very often contains true green, but the correct candidate or
> convex weights cannot be inferred reliably without sacrificing either
> sparse points or coherent texture.

The result remains PARTIAL rather than a pure NO-GO because the spatial and
convex oracles are large and minimum-energy pass 1 gives a small safe aggregate
gain.  It does not meet the 30-50% recovery target, does not repair both star
fields, and does not justify production implementation.  Further local
candidate-score tuning would need genuinely new per-candidate evidence—most
plausibly faithful guided-regression stability diagnostics—not more thresholds
over the current values, energies, or spreads.

## Verification

- the native trace and final-stage substitution tests pass;
- all 81 cases and all 18 X-Trans phase cells completed with finite values;
- measured green samples remain float32-identical in every experimental guide;
- all 113 guide methods were evaluated through the unchanged native final R/B
  path;
- seven Python unit tests cover measured-sample preservation, candidate patch
  selection, convex containment, bounded blending, temperature fusion, ridge
  fitting, and the coherence transition;
- [manifest.json](images/xtrans-mlri-green-fusion/manifest.json) binds the
  corpus, results, and diagnostic images.
