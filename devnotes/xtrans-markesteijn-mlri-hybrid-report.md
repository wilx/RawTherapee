# Markesteijn–MLRI deterministic hybrid experiment

## Decision

**PARTIAL — do not integrate a hybrid demosaicer.**

The frozen 7×7 logistic blender improves the aggregate held-out result by
5.80 dB over Markesteijn and recovers 77.8% of the convex-oracle headroom. The
validation-selected smoothed per-pixel field reaches 38.963 dB, a 5.814 dB
gain and 78.0% convex-headroom recovery. Those numbers are real, but they are
not source-stable: one of four untouched test sources, Hubble Deep Field,
regresses by 2.707 dB and its p95 patch RMS rises from 0.03029 to 0.04128.

The model is only moderately predictive and weakly calibrated. Its validation
gate selects no conservative fallback at all, and even that operating point has
a 0.0421 worst paired patch regression. The synthetic impulse and frequency
sweep also regress. Running corrected-final MLRI dominates the experimental
runtime. These failures violate the GO criteria despite the strong aggregate
score.

The missing capability is a reliable observable distinction between coherent
fine structure, where MLRI often helps, and sparse bright/chromatic point
structure, where its error can be severe. The Hubble and impulse failures show
that the present windowed disagreement, gradient, chroma, and texture features
do not provide that safety signal.

## Scope and artifacts

This is an offline, non-neural experiment. It changes no production demosaic
dispatch, PP3 method, GUI, defaults, or processing behavior. It uses only
RawTherapee's native three-pass Markesteijn and corrected-final MLRI engines.

The implementation and frozen outputs are:

- `tools/xtrans_hybrid/`: authenticated dataset binding, leakage-safe feature
  extractor, compact models, generator, and tests;
- `tests/xtransoracle/oracle_runner.cc`: a `run-pair` mode that avoids running
  unrelated candidates;
- `devnotes/images/xtrans-hybrid/dataset.json`: source hashes, provenance,
  licenses, source-level splits, and crop coordinates;
- `devnotes/images/xtrans-hybrid/results.json`: complete metrics, fitted model,
  ablations, gating sweeps, grouped folds, and synthetic controls;
- `devnotes/images/xtrans-hybrid/*.png`: calibration, ablation, gating, fold,
  alpha, oracle, disagreement, and regression diagnostics.

The canonical corpus manifest SHA-256 is:

`3a59d351c0276301fa7be0b68a79b0292bdab10ce6d598d99f501b53bbe1949f`

Two independent generations produced byte-identical JSON and PNG files.
NumPy/SciPy BLAS is restricted to one thread, observable features are
canonicalized to six decimal places, normalization parameters to twelve, and
bounded blend weights to 0.001 steps. The native pair remains four-threaded.
The alpha quantization changed quality only below the meaningful precision of
this experiment and provides 1001 deterministic blend levels.

## Dataset and leakage boundary

The corpus contains 20 authenticated sample images bundled with scikit-image
0.26.0 and three deterministic 168×168 crops from each source: 60 crops total.
Encoded RGB is decoded with the IEC 61966-2-1 inverse sRGB transfer function
before X-Trans remosaicking. Content includes portraits, biological detail,
fabric/fur, irregular and periodic texture, text, smooth material, saturated
objects, microscopy, and a dark astronomical field.

The frozen source split is 12 train / 4 validation / 4 test. No source crosses
a split. The two Middlebury motorcycle views share one validation group. The
final test sources are astronaut, brick, Hubble, and page.

This meets the requested 20-source minimum but is not large enough for a
production conclusion. Only four sources are in the final test split, three
crops from one source remain correlated, and the samples are mostly 8-bit
rendered RGB rather than camera-linear high-dynamic-range ground truth. The
large aggregate test gain is heavily influenced by brick and page.

The leakage audit is explicit:

- `feature_maps()` has no ground-truth argument and accepts only mosaic, CFA,
  Markesteijn, and MLRI data;
- source identity, split identity, oracle labels, oracle alpha, and errors are
  not features;
- ground truth is used only to build offline targets and report metrics;
- means and scales are fitted on training rows only;
- block size, model, regularization, gate, and spatial variant are selected on
  validation data only;
- the held-out result was not retuned after the Hubble failure was observed.

## Expanded oracle

The expanded 20-source corpus changes the earlier four-image result
substantially:

| 7×7 result | PSNR | Gain over Markesteijn |
| --- | ---: | ---: |
| Markesteijn | 34.781 dB | — |
| MLRI | 36.742 dB | +1.962 dB |
| Hard oracle | 40.083 dB | +5.302 dB |
| Convex oracle | 40.434 dB | +5.653 dB |

At 15×15, the hard and convex oracles reach 39.775 and 40.159 dB,
respectively. The expanded headroom is therefore much larger than the earlier
+0.769/+1.140 dB result. This does not mean the pair suddenly became universally
better; it shows that algorithm ranking and complementarity depend strongly on
content. In this corpus MLRI alone also outranks Markesteijn overall.

On the untouched four-source test set, 7×7 Markesteijn is 33.149 dB, MLRI is
36.966 dB, the hard oracle is 40.434 dB, and the convex oracle is 40.604 dB.
The corresponding 15×15 oracles are 40.339 and 40.500 dB.

## Frozen practical model

Validation selected:

- 7×7 decision scale;
- bounded logistic blending with regularization 0.0001;
- feature groups A–E (disagreement, mosaic gradients, candidate gradients,
  chroma/saturation, and texture);
- no effective confidence gate (`lower=0`, `upper=1.01`);
- a per-pixel alpha field followed by a 7×7 box filter.

The blockwise form, used for classification/calibration and source diagnostics,
scores 38.949 dB on test: +5.800 dB, 79.6% of hard-oracle headroom, and 77.8%
of convex-oracle headroom. The validation-selected spatial form scores
38.963 dB: +5.814 dB, 79.8% hard and 78.0% convex headroom. At 15×15 the
practical result is 38.431 dB, +5.282 dB and 71.9% convex headroom.

The 7×7 classification diagnostics are 84.5% accuracy, 67.8% balanced
accuracy, 0.724 ROC AUC, and 8.37% oracle-margin-weighted error. Blend-alpha
correlation is 0.511 and MAE is 0.189. Half of oracle alphas are strictly
between zero and one, so blending is genuinely useful, but the predictor is
overconfident: its mean alpha is 0.806, it changes 99.7% of test patches, and
87.8% receive an MLRI-majority blend.

Aggregate 7×7 error tails improve materially:

| Patch RMS | Markesteijn | Practical block blend |
| --- | ---: | ---: |
| median | 0.00493 | 0.00208 |
| p90 | 0.03488 | 0.01397 |
| p95 | 0.05235 | 0.02080 |
| p99 | 0.08803 | 0.04781 |
| maximum | 0.18890 | 0.12604 |

The aggregate tail, however, masks the source-specific regression:

| Test source | Markesteijn | MLRI | Practical | Gain | p95 Mark → practical |
| --- | ---: | ---: | ---: | ---: | ---: |
| astronaut | 37.595 | 35.103 | 37.993 | +0.398 dB | 0.02562 → 0.02503 |
| brick | 46.950 | 59.084 | 55.090 | +8.139 dB | 0.00942 → 0.00395 |
| Hubble | 38.277 | 33.722 | 35.570 | **−2.707 dB** | **0.03029 → 0.04128** |
| page | 27.974 | 41.559 | 41.525 | +13.551 dB | 0.08253 → 0.01709 |

The Hubble regression map shows the model assigning high MLRI weight around
sparse bright, colored microstructure where the oracle often prefers
Markesteijn. This is the principal blocker.

## Model, feature, and spatial ablations

The training-tuned fixed blend uses alpha 0.851 and reaches 37.386 dB on test.
It captures a large fraction of the corpus's MLRI-favoring bias, but the local
logistic blend adds another 1.56 dB. A depth-3 hard tree reaches 38.441 dB;
bounded blending is about 0.51 dB better. The single disagreement threshold is
only 33.249 dB, while a disagreement-only logistic blend reaches 38.019 dB.

On validation, chroma/saturation is the strongest single ridge group
(34.223 dB). All core groups reach 34.724 dB. Adding the optional phase group
raises this only 0.030 dB on validation and 0.038 dB on test, which does not
justify the extra X-Trans-specific feature complexity. Large standardized
coefficients include C2 disagreement RMS, candidate Laplacian energies,
disagreement-gradient energy, and Markesteijn corner/gradient terms. Because
these features are correlated, coefficient magnitude is descriptive rather
than causal importance.

All four source-grouped development folds improve, but gains vary widely:
3.336, 5.200, 3.636, and 9.839 dB. Convex-headroom recovery ranges from 80.1%
to 86.2%. The positive folds support a real signal; their variance and the
held-out Hubble failure show that it is content-dependent.

Spatial smoothing is inexpensive in quality terms. The constant 7×7 block
field scores 38.949 dB with 64 connected binary-majority regions, 0.00591
boundary fraction, and 0.528 routing-entropy bits. The selected per-pixel plus
7×7 box field scores 38.963 dB with 41 regions, 0.00478 boundary fraction, and
0.511 bits. Raw per-pixel alpha scores 38.965 dB on test but was worse on
validation and is more fragmented. Thus smoothing improves routing regularity,
but it cannot fix an incorrect high-confidence algorithm choice.

## Confidence gate and structural controls

The validation sweep records altered coverage, PSNR gain, p95, convex-headroom
recovery, and the maximum paired patch regression for every threshold. Its
selected point performs no conservative fallback: 98.1% validation coverage,
+3.154 dB, 76.0% convex-headroom recovery, and a 0.0421 worst paired patch
regression. No tested asymmetric threshold supplied the requested safety while
preserving the best validation quality.

Synthetic controls make the failure mode clearer. The model successfully
routes the diagonal red/green edge and high-frequency monochrome case toward
MLRI, and keeps Markesteijn for periodic chromatic texture. It misses most of
MLRI's large saturated-quadrant advantage, reduces the isolated impulse from
45.089 to 41.599 dB, and reduces the frequency sweep from 7.410 to 7.252 dB.
Those failures agree with the sparse-detail error seen in Hubble. Exact
Markesteijn gradients and axis-aligned edges receive tiny blends, although the
result remains above 80 dB and is numerically negligible.

## Runtime and memory

Two final deterministic runs used 60 168×168 natural crops. Timings are kept
out of the canonical manifest:

| Measurement | Run range |
| --- | ---: |
| Native Markesteijn sum | 0.196–0.205 s |
| Native corrected-final MLRI sum | 61.435–61.937 s |
| Both-scale feature extraction, 60 crops | 4.605–4.718 s |
| Selected-scale inference features, 12 test crops | 0.353–0.362 s |
| Selector evaluation, 12 test crops | 0.067–0.075 s |
| Complete experiment | 81.22–82.10 s |
| Peak RSS | 761,620–762,816 KiB |

In this small-crop harness MLRI takes roughly 300 times the measured native
Markesteijn time. These are not full-frame production benchmarks, but they show
that running both demosaicers—not the compact selector—is the dominant cost.

## Conclusion

**PARTIAL:** Observable local features recover about 78% of the 7×7 convex
oracle headroom and greatly improve aggregate error tails, but the gain is not
reliable across held-out images. Sparse bright/chromatic microstructure
regresses severely, alpha calibration is weak, confidence gating does not
protect Markesteijn, and the evaluation set remains small. Continue only with
targeted research into an impulse/sparse-color safety feature and a genuinely
Markesteijn-default benefit predictor. The current result does not justify
running both demosaicers in RawTherapee or adding a hidden/GUI hybrid method.
