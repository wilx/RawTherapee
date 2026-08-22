# Markesteijn-MLRI sparse-structure danger-gate experiment

## Result

**Decision: NO-GO.**

Inference-time disagreement features can identify part of MLRI's sparse-color
failure class, but they cannot meet the safety/utility tradeoff required for a
practical gate. The selected detector improves Hubble substantially, yet it
misses too many severe regressions on an independent star field, rejects too
much legitimate detail, worsens held-out p99 error, and retains less than 80%
of the frozen blender's aggregate gain.

The safety oracle confirms that gating itself has useful headroom. The failure
is prediction: observable 7x7 features do not adequately separate harmful
MLRI microstructure from detail for which the frozen blender is useful. This
closes the Markesteijn-corrected-final-MLRI hybrid route under the experiment's
stated criteria. No production demosaicer was added or changed.

Authoritative artifacts are in
[`images/xtrans-danger/`](images/xtrans-danger/). The canonical manifest is
SHA-256
`a75bb6b94e225c06ac16bedac7e7ec0f16433269322664c58abee58433d99cc5`.

## Question and frozen candidate

The experiment asks only whether a detector can decide, patch by patch,
whether to replace the existing blend with Markesteijn:

```text
output = Markesteijn, if danger is detected
         frozen blender, otherwise
```

The candidate reconstruction was not retrained. It is the previously selected
7x7 `per_pixel_box_7` logistic blender, authenticated by results SHA-256
`a1e18f4936518ae164cdb867e986518afb6212163ec73c36695b508d83bf4266`.
Its inputs and alpha quantization are reproduced exactly from the prior
experiment.

The patch regression target is:

```text
regression = (SSE(blender) - SSE(Markesteijn)) / RGB sample count
```

The fixed severe threshold is the 75th percentile of positive training
regressions, `2.894234e-6` normalized MSE. Target thresholds at zero,
the training-positive median, p75, and p90 were all fitted and compared. The
source-held-out test data did not choose the target threshold, model, feature
set, or operating threshold.

Patch SSE is rounded to 10 decimal places before target construction. This is
far below the smallest tested danger threshold after division by patch sample
count, and removes last-bit OpenMP scheduling variation without quantizing the
candidate images or inference-time features. Two complete runs in different
output directories produced byte-identical artifacts and manifests.

## Corpus and leakage boundary

The experiment evaluates 26,712 non-overlapping 7x7 patches:

| Split | Patches |
| --- | ---: |
| Train | 15,228 |
| Validation | 5,052 |
| Held-out natural | 6,000 |
| Held-out synthetic | 432 |

The natural corpus retains the previous source-level split. Hubble,
astronaut, brick, and page remain in the test split. An independent
catalog-derived Hydra star field was added from
[NASA/GSFC Scientific Visualization Studio](https://svs.gsfc.nasa.gov/4041/):
the external 3000x3000 TIFF has SHA-256
`bc5ae5f68fc1977545c0d16c1c60c20dafb4fdbeb045f55774e7a5d03498f6d0`.
Hubble, the NASA star field, and astronaut/specular content were never used to
fit the detector or select its operating point. Brick and page are untouched
fine-detail controls.

The deterministic hard-negative corpus contains 42 scenes covering:

- isolated white and saturated red, green, and blue samples;
- exact 1-, 2-, and 3-pixel colored dots;
- dark-background points, small specular highlights, and star fields;
- thin lines, line intersections, sparse/dense microtexture, and points beside
  ordinary edges;
- intensity, saturation, spacing, background, and CFA-relative variation;
- all 18 unique X-Trans phase/orientation matrices;
- held-out transitions from 1 to 72 points and from 1- to 5-pixel-thick lines.

The synthetic transitions are test-only. They do not influence model or
threshold selection.

## Observable features and models

Ground truth is absent from both feature APIs. The 68 deterministic features
comprise the prior 44 A-E features and 24 new impulsiveness/coherence features:

- disagreement maximum, RMS, median absolute deviation, and kurtosis;
- high-pass disagreement kurtosis;
- top-1, top-2, and top-4 chroma disagreement energy fractions;
- chroma peak-to-MAD and peak-to-surrounding-RMS ratios;
- luma, C1, C2, and chroma disagreement energy ratios;
- high-disagreement connected-component count and largest-component fraction;
- isolated-extremum and neighbor fractions;
- structure-tensor coherence and directional-to-point energy.

The following deterministic model families were fitted with severe false
negatives weighted much more heavily than false positives:

- a single-feature rule;
- weighted logistic regression at three regularizations;
- weighted depth-2 and depth-3 trees;
- a 24-stump weighted logistic boosting control.

Each family was evaluated with core-only, impulse-only, and combined feature
sets. No neural classifier or scikit-learn dependency is used.

No validation operating point simultaneously reached 90% severe recall and
retained 85% of blender gain. At at least 90% validation recall:

| Feature set | Lowest fallback | Recall | Gain retained |
| --- | ---: | ---: | ---: |
| Core A-E | 70.07% | 91.71% | 43.96% |
| Impulse/coherence | 62.21% | 91.71% | 47.70% |
| Combined | 67.00% | 90.26% | 65.58% |

Even when choosing the operating point that retained the most validation gain
while staying above 90% recall, the best result required 73.34% fallback and
retained only 69.93% of the gain. Adding the new features therefore improves
some operating points, but still does not approach the required tradeoff.

The selected minimum-fallback operating point is an impulse-only depth-3 tree
trained at target tau `5.997875e-7`, with threshold
`0.869575211478094`. Its splits use disagreement RGB RMS, chroma energy,
largest connected-component fraction, C1 energy fraction, and top-4 chroma
energy concentration. Validation ROC AUC is only `0.6838`.

## Oracle headroom

On held-out natural data, an oracle that chooses only between the practical
blender and Markesteijn obtains:

| Method | PSNR | p99 patch RMS | Maximum patch RMS |
| --- | ---: | ---: | ---: |
| Markesteijn | 34.118 dB | 0.08278 | 0.18890 |
| Frozen blender | 39.927 dB | 0.04279 | 0.12345 |
| Safety oracle | **41.354 dB** | **0.03391** | **0.08546** |

The oracle falls back on only 21.23% of patches. It improves the blender by
1.43 dB and eliminates all paired regressions. Safety gating therefore has
real theoretical value; the NO-GO is not caused by a lack of oracle headroom.

## Selected detector on held-out data

### Aggregate held-out natural data

| Metric | Frozen blender | Selected gate |
| --- | ---: | ---: |
| PSNR | **39.927 dB** | 37.759 dB |
| p95 patch RMS | **0.01835** | 0.02666 |
| p99 patch RMS | **0.04279** | 0.05839 |
| Maximum patch RMS | **0.12345** | 0.12740 |

The gate falls back on 27.93% of natural patches and retains 76.95% of the
blender's aggregate MSE gain. It recalls 77.10% of severe regressions and
misses 90 severe patches. Recall of the worst 1%, 5%, and 10% regressions is
83.33%, 83.67%, and 60.33%, respectively. The p99 error increases by 36.5%,
rather than decreasing.

### Held-out synthetic data

The detector recalls 87.5% of severe synthetic regressions with 24.31%
fallback, but it retains `-0.38%` of blender gain: the gated MSE is slightly
worse than Markesteijn. This fails the controlled hard-negative utility test
even though most severe events are found.

### Source detail

| Source | Fallback | Severe recall | Blender | Gated | Markesteijn | Oracle |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Hubble | 24.75% | 84.55% | 35.597 | 37.856 | 38.277 | 38.428 |
| NASA Hydra stars | 0.50% | 62.50% | 64.906 | 69.710 | 71.527 | 71.578 |
| Astronaut/specular | 63.33% | 67.88% | 38.038 | 37.699 | 37.595 | 38.326 |
| Brick | 18.25% | 100%* | 55.487 | 49.410 | 46.950 | 55.487 |
| Page | 32.83% | 100%* | 41.395 | 33.076 | 27.974 | 41.395 |

PSNR values are in dB. `*` Brick and page have no meaningful severe-regression
population at this threshold; their important result is the cost of false
fallback. Hubble is materially repaired but not brought to Markesteijn. The
independent star field remains under-detected. Brick and page demonstrate the
opposite error: legitimate high-frequency structure is rejected, destroying a
large part of MLRI's gain.

The diagnostic panels show the fixed operating point after selection:

- [Hubble danger map](images/xtrans-danger/danger-map-hubble.png)
- [NASA star-field danger map](images/xtrans-danger/danger-map-nasa-hydra-starfield.png)
- [Astronaut/specular danger map](images/xtrans-danger/danger-map-astronaut.png)
- [Brick danger map](images/xtrans-danger/danger-map-brick.png)
- [Page danger map](images/xtrans-danger/danger-map-page.png)

## Coherence-transition result

The intended behavior was increasing permission as isolated structures became
coherent. The selected detector does not show it.

- Point-pattern fallback rises from 2.78% for one point to 72.22% for 72
  points, although the denser cases have negative mean regression and should
  generally retain the blender.
- Line fallback is non-monotonic and ends at 55.56% for the thickest line.

The specialized statistics measure sparsity and coherence, but the mapping
from those observables to algorithmic error is not stable across scene type.
This directly answers the difficult distinction posed by the experiment:
legitimate fine texture and isolated erroneous microstructure remain too
confounded for this detector family and patch scale.

See the full [cost-curve dashboard](images/xtrans-danger/danger-cost-curve.png)
for recall, fallback, PSNR, p95/p99 error, maximum paired regression, and
retained gain over the complete threshold sweep. The canonical JSON contains
all candidate models, thresholds, calibration bins, per-source rows, and test
curves.

## GO / NO-GO criteria

| Criterion | Result |
| --- | --- |
| At least 90% severe recall on held-out natural data | **Fail: 77.10%** |
| At least 80% recall on both sparse sources | **Fail: 84.55%, 62.50%** |
| At least 80% recall on held-out synthetic data | Pass: 87.5% |
| Retain at least 80% of blender aggregate gain | **Fail: 76.95%** |
| Fallback no more than 35% | Pass: 27.93% |
| Reduce held-out p99 patch RMS by at least 20% | **Fail: p99 rises 36.5%** |
| Similar behavior across sparse sources and coherence transitions | **Fail** |

The result is therefore **NO-GO**, not PARTIAL. It catches a useful subset of
the known failure and proves that the Hubble defect is observable to a degree,
but the operating tradeoff is not stable across independent sources. Pursuing
a hidden production hybrid or optimizing corrected-final MLRI is not justified
by this route.

## Reproduction

```sh
nice -n 10 cmake --build build/dev \
  --target rawtherapee-xtrans-oracle-runner -j4

.venv/bin/python -m tools.xtrans_danger.generate \
  --runner build/dev/tests/xtransoracle/rawtherapee-xtrans-oracle-runner \
  --starfield /path/to/grail_free_air_stars1.tif \
  --force

PYTHONPATH=. .venv/bin/pytest -q tools/xtrans_danger/tests
```

The external NASA TIFF is authenticated before use. The runner's new
`run-pair-cfa` mode is research-only and enables the same Markesteijn/MLRI pair
to be evaluated against all 18 X-Trans CFA matrices. The committed corpus
contains derived diagnostic images and canonical results, not the NASA source
image.
