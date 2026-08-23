# X-Trans MLRI guided-regression stability and candidate-evidence experiment

## Decision

**NO-GO — faithful guided-regression state does not provide the missing local
candidate identity evidence. Further local MLRI candidate-selection tuning is
not justified.**

The validation-safe winner is the unchanged corrected-final pass-1 fusion.
It therefore recovers **0.0%** of the combined 7x7 directional-oracle final-RGB
gap. Several new controls improve the aggregate untouched-test score, but all
do so with the same unresolved sparse/coherent tradeoff:

- fit-MSE selection improves Brick, Grass, Gravel, and Page, but loses 0.395 dB
  on validation Hubble and 0.347 dB on untouched Hydra;
- the depth-3 tree improves Hubble by 1.190 dB and Hydra by 0.902 dB, but loses
  9.504 dB on Brick, 10.115 dB on Grass, 11.325 dB on Gravel, and 10.869 dB
  on untouched Page;
- pairwise ranking improves Hubble by 0.334 dB but fails the 90% Brick
  retention gate, regresses Hydra by 0.237 dB, and loses 0.901 dB on Page;
- ridge-weighted pruning improves Hubble and Hydra but substantially damages
  the coherent sources.

The central negative evidence is stronger than the final selector result.
Pass 1 makes its guided fits look dramatically more stable on both coherent
textures and sparse star fields. On Hubble, mean fit MSE falls from 10.110 to
0.875 and prediction variance from 51.925 to 0.952, while fused-green MSE
nearly doubles from 0.000230 to 0.000432. On Hydra the same quantities fall
from 0.0194 to 0.000623 and 0.0467 to 0.000271, while fused-green MSE doubles
from 2.33e-7 to 4.69e-7. The local regression is confidently fitting an
aliased explanation; stability is not correctness.

No production demosaicing method, profile, default, or GUI behavior changed.

## Instrumentation

The production computation in `guidedMlri()` fits one model at every window
center. The quantities that actually exist are:

- Laplacian-domain product `meanIp`, guide energy `meanII`, and the effective
  gain denominator `meanII + 0.01`;
- fitted gain `a = meanIp / (meanII + 0.01)`;
- observed-domain DC offset `b`;
- Laplacian-mask and observed-sample counts;
- the algebraic local residual MSE from TIP 2016 equation (8);
- inverse residual weight `1 / max(MSE, 0.01)`;
- the overlapping-model total weight and weighted mean gain/offset used by the
  reconstruction.

Normal MLRI keeps only the final weighted gain/offset long enough to form the
prediction. The numerator, denominator, local MSE, individual model weights,
support counts, and model-to-model disagreement are discarded.

The trace preserves those exact terms and derives, from the actual overlapping
models at each output position:

- weighted and minimum fit MSE;
- weighted gain/offset, gain variance and gain range;
- weighted numerator, denominator, guide energy, and support counts;
- prediction variance, range, weighted MAD, and weighted IQR;
- normalized model-weight entropy, effective model count, maximum normalized
  weight, and total weight;
- exact leave-one-overlapping-model prediction variance and maximum influence.

The final green directions are not direct `guidedMlri()` outputs. Each is a
directional G-C residual built from two guided regressions: green predicted at
a red/blue site and that chromatic channel predicted at green sites. The trace
therefore attributes each statistic through the same chromatic/green CFA-site
pairing, opposite-color completion, phase-dependent half-Gaussian support, and
red-versus-blue target selection used by the corresponding final candidate.
This is a defined aggregation of exact internal quantities; it is not labeled
as a per-candidate quantity that the original implementation explicitly had.

## Observational-only verification

Diagnostics are calculated only when the development trace pointer is present.
They are evaluated after the production weighted gain/offset and result are
formed and never feed back into reconstruction.

The native suite requires:

- all 22 statistics for both passes and all eight directions to have the exact
  image shape and finite values;
- repeated diagnostic traces to be float-identical;
- the instrumented final R/G/B planes to be float-identical to a separate
  uninstrumented corrected-final reference invocation;
- the eight current directional weights to continue summing to one at every
  missing-green site.

## Corpus and split discipline

The experiment reuses the prior 81-case normalized linear-light corpus:

- 21 natural crops from Astronaut, Brick, Grass, Gravel, Hubble, Page, and
  NASA Hydra;
- all 18 distinct X-Trans translation phases;
- impulses, colored dots, star fields, thin lines, intersections, saturated
  edges, density/thickness sweeps, frequency controls, and five untouched
  sparse-to-coherent transitions;
- 41 train, 15 validation, and 25 untouched test cases.

Hubble and Brick are validation sources. Hydra and Page remain untouched test
sources. The practical method is selected only if it retains at least 90% of
Brick's pass-1 MSE improvement over pass 0 and does not worsen Hubble. Test
sources do not influence selection.

Canonical measurements are in
[results.json](images/xtrans-mlri-regression-evidence/results.json), with the
corpus contract in
[dataset.json](images/xtrans-mlri-regression-evidence/dataset.json).

## Within-pixel candidate evidence

The table below is the untouched test split. Higher within-pixel Spearman and
AUC are better. Top-1/top-2 measure exact best-candidate identity.

| Observable cost | Within-pixel Spearman | Top-1 | Top-2 | Pair AUC | Close-pair AUC | High-margin AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| directional energy | 0.445 | 10.50% | 18.11% | 0.704 | 0.609 | 0.814 |
| guided fit MSE | 0.393 | 11.42% | 19.42% | 0.710 | 0.582 | 0.826 |
| gain instability | 0.026 | 8.87% | 15.02% | 0.510 | 0.577 | 0.396 |
| inverse denominator strength | 0.109 | 6.21% | 12.48% | 0.471 | 0.438 | 0.525 |
| model prediction variance | 0.384 | 8.43% | 16.54% | 0.695 | 0.560 | 0.805 |
| model prediction range | 0.336 | 7.31% | 14.35% | 0.669 | 0.549 | 0.794 |
| model prediction MAD | 0.363 | 8.84% | 16.30% | 0.688 | 0.549 | 0.784 |
| leave-one-model influence | 0.402 | 9.17% | 17.44% | 0.693 | 0.594 | 0.762 |
| weak support | 0.236 | 8.06% | 14.46% | 0.597 | 0.587 | 0.593 |
| hand stability composite | 0.396 | 9.66% | 18.06% | 0.685 | 0.605 | 0.769 |
| centered ridge cost | 0.286 | 8.18% | 15.10% | 0.656 | 0.540 | 0.793 |
| depth-3 tree | 0.206 | 9.16% | 17.77% | 0.612 | 0.497 | 0.760 |
| pairwise logistic ranker | 0.310 | 6.65% | 13.53% | 0.647 | 0.513 | 0.796 |

Fit MSE is the only simple new statistic to beat directional energy on top-1,
top-2, and aggregate pair AUC, and the margin is small. It is worse on mean
within-pixel rank correlation and close-pair discrimination. Prediction
dispersion and leave-one-model influence largely reproduce the same common
difficulty signal and do not identify the best direction more reliably.

Absolute pooled correlations are high—0.817 for fit MSE, 0.777 for prediction
variance, and 0.783 for leave-one-model influence—but fall to 0.409, 0.449,
and 0.483 after within-pixel centering. This confirms the previous warning:
the discarded regression state knows that a pixel is difficult much better
than it knows which candidate is correct at that pixel.

## Hypothesis results

### H1: low residual means a better candidate

Only weakly. Fit MSE gives 11.42% test top-1 versus 10.50% for current
directional energy, but has lower within-pixel Spearman and higher candidate
regret. Hard fit-MSE selection improves aggregate test RGB by 0.082 dB, yet
worsens both Hubble and Hydra. It is not safe candidate evidence.

### H2: extreme gain predicts failure

No. Gain instability is nearly random within a pixel (Spearman 0.026, AUC
0.510) and high-margin AUC is inverted at 0.396. The fitted gain's magnitude
is not a useful universal reliability score; assuming gain one is ideal would
not be justified by this formulation.

### H3: weak denominator predicts instability

No useful ranking signal was found. Inverse denominator strength has test AUC
0.471 and only 6.21% top-1 accuracy. It often identifies a different model
scale, not an erroneous candidate.

### H4: overlapping-model disagreement predicts wrong candidates

It mostly predicts difficulty. Prediction variance/range/MAD and
leave-one-model influence reach pair AUC 0.67-0.70, but top-1 remains 7.3-9.2%
and none beats directional energy consistently. Removing a single influential
overlapping model does not expose the sparse failures.

### H5: low residual can signal aliasing

Confirmed as a failure mechanism, not as a usable detector. Both Hubble and
Hydra obtain much lower pass-1 fit residual, prediction variance, and model
influence while their fused-green error increases. A low-residual plus
high-disagreement interaction is essentially non-informative (test Spearman
-0.033, AUC 0.500, top-1 5.68%).

## Pass-0 versus pass-1 behavior

| Source | Pass-0 fit MSE | Pass-1 fit MSE | Pass-0 prediction variance | Pass-1 prediction variance | Pass-0 fused-G MSE | Pass-1 fused-G MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Brick | 43.975 | 0.252 | 17.681 | 0.109 | 1.36e-5 | 2.38e-6 |
| Grass | 386.536 | 13.288 | 118.051 | 2.901 | 6.49e-4 | 1.21e-4 |
| Gravel | 230.376 | 5.728 | 66.637 | 1.130 | 2.53e-4 | 4.74e-5 |
| Page | 755.068 | 22.014 | 366.025 | 5.246 | 1.43e-3 | 1.71e-4 |
| Hubble | 10.110 | 0.875 | 51.925 | 0.952 | 2.30e-4 | 4.32e-4 |
| Hydra | 0.0194 | 0.000623 | 0.0467 | 0.000271 | 2.33e-7 | 4.69e-7 |

Pass 1 produces genuinely better-supported, more mutually consistent models
on coherent sources and improves their green reconstruction. It reports the
same stability improvement on stars while selecting the wrong alias family.
The internal diagnostics therefore do not reveal the distinction needed to
keep Page/Brick refinement and reject Hubble/Hydra refinement.

## Practical final-RGB controls

Untouched-test aggregate results:

| Method | PSNR (dB) | Delta from current | p99 absolute |
| --- | ---: | ---: | ---: |
| current corrected-final pass 1 | 23.335 | — | 0.3840 |
| minimum-energy pass 1 | 23.370 | +0.035 | 0.3807 |
| hard fit MSE | 23.417 | +0.082 | 0.3699 |
| ridge-weighted pruning, best 4 | 23.430 | +0.095 | 0.3807 |
| hard ridge | 23.495 | +0.159 | 0.3765 |
| hard pairwise | 23.537 | +0.202 | 0.3772 |
| hard depth-3 tree | 23.589 | +0.254 | 0.3272 |
| 7x7 directional oracle | 24.951 | +1.615 | 0.2649 |
| convex candidate oracle | 26.902 | +3.567 | 0.1069 |

The aggregate gains are not deployable. Tree/ridge improve star fields by
moving toward sparse-case behavior and destroy coherent texture. Fit MSE does
the reverse. Candidate pruning, uniform/median survivor fusion, soft
regression-cost modulation, and bounded current-weight modulation do not
produce a validation-safe improvement over current pass 1. The selected
method is therefore the unchanged current method, not the aggregate test
winner.

## Sparse/coherent transitions and CFA phase

No monotonic diagnostic transition appears. Pass-1 fit MSE is lower for
isolated and aligned points and repeated microtexture, but can be higher for
medium-density points and continuous lines. Prediction variance can increase
by orders of magnitude at dotted/continuous structures even when fit MSE
falls. The statistics describe support geometry but do not define a stable
sparse-to-coherent decision boundary.

Across the 18 identical CFA-relative impulse placements, fit-MSE top-1 ranges
from **0.84% to 6.41%** and within-pixel Spearman from **-0.056 to 0.209**.
The diagnostic does not track candidate error robustly across CFA phase.

## Perturbation scope and runtime

The trace computes exact leave-one-overlapping-model sensitivity essentially
for free once individual model predictions and weights are retained. It does
not refit every local 7x7 regression after deleting each underlying CFA
sample. Such support-sample leave-one-out would require many repeated local
fits per invocation. The cheaper and directly relevant model-influence test
already fails to separate candidate identity; the expensive refit was not
justified.

Research-only timings for the final 81-case run were:

- faithful native trace: 231.8 s;
- candidate table: 5.3 s;
- compact model fitting: 3.5 s;
- exhaustive rank/pairwise analysis: 96.9 s;
- guide construction: 20.7 s;
- native final-RGB evaluation: 47.1 s.

The production path pays none of the diagnostic trace cost. Existing scalar
quantities such as fit MSE, gain, denominator, and inverse model weight are
already computed and discarded; aggregating prediction dispersion and
influence would add storage and arithmetic. The quality result makes that
production cost moot.

## Required answers

### Guided regression

The exact coefficient, residual, support, and model-weight quantities exist
and can be faithfully attributed through the final candidate construction.
All except the final weighted coefficient averages are currently discarded.

### Candidate ranking and within-pixel information

Fit residual adds a very small amount of exact-identity information. Gain,
denominator, model disagreement, support concentration, and leave-one-model
influence do not. The new features remain much stronger at detecting common
pixel difficulty than distinguishing candidates at the same pixel.

### Pass behavior and stability

Pass 1 reduces residual and instability on both coherent textures and sparse
stars. It is correct on the former and confidently aliased on the latter.
Sparse failure is therefore not explained by a visibly unstable or
single-model-dominated regression fit.

### Practical fusion and final RGB

Hard fit-MSE selection can help coherent detail; learned/tree selection can
help stars; pruning and bounded modulation cannot reconcile them. The only
validation-safe selection is unchanged current pass 1, recovering 0% of the
7x7 oracle gap.

### Research decision

Faithful guided-regression stability information does **not** provide the
missing candidate-specific evidence. This was the last clearly identified
local MLRI signal. Further thresholds or compact models over local MLRI state
should stop here.

## Diagnostic maps

- [Hubble](images/xtrans-mlri-regression-evidence/map-hubble-0.png)
- [Hydra](images/xtrans-mlri-regression-evidence/map-nasa-hydra-starfield-0.png)
- [Brick](images/xtrans-mlri-regression-evidence/map-brick-0.png)
- [Page](images/xtrans-mlri-regression-evidence/map-page-0.png)
- [isolated points](images/xtrans-mlri-regression-evidence/map-transition-coherence-isolated-points.png)
- [continuous line](images/xtrans-mlri-regression-evidence/map-transition-coherence-continuous-line.png)

## Artifact identities

- dataset SHA-256:
  `5298fa8c5c6b39bf0aca96e2a718c1dad18910c46989774b2753edd8bbc27483`
- results SHA-256:
  `fd7969c4638d371d9bb6b2f58d2b17488d58001996a0688f164cefbdb43eb623`
- manifest SHA-256:
  `e56c913e4674fa8dc1896bfe94a47d9d7dba0918cb8cd154cf5603105e7e2630`

The manifest binds the canonical JSON and six diagnostic PNGs. Temporary
per-case trace planes are not tracked.
