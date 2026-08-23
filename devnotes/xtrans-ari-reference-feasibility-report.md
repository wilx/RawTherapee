# Adaptive Residual Interpolation reference reproduction and X-Trans feasibility

## Decision

**NO-GO - do not implement a RawTherapee X-Trans ARI demosaicer.**

The part of published ARI needed by this study was reproduced faithfully on
Bayer: the independent 11-iteration green stage agrees with the authenticated
MATLAB reference to `3.259e-8` aggregate RMS, `6.424e-6` maximum including
boundaries, and `3.127e-8` maximum in the 16-pixel interior.

ARI does introduce genuinely new information. On natural images, adding the RI
bank to the MLRI bank improves the pixel oracle by `0.572 dB`. It also gives a
very good sparse-source result: `41.677 dB` versus `36.654 dB` for corrected
MLRI on Hubble/Hydra. However, neither condition needed for a port is met:

- the direct horizontal/vertical X-Trans candidate architecture is intrinsically
  weak on coherent texture. Its pixel oracle is only `37.837 dB` on
  Brick/Grass/Gravel/Page, below corrected MLRI at `40.685 dB` and far below
  fixed high-refinement MLRI at `46.087 dB`;
- the published criterion is a poor within-pixel ranker on natural data:
  Spearman `0.331`, pairwise AUC `0.628`, top-1 `6.89%`, top-2 `11.32%`;
- full ARI is `0.469 dB` worse than MLRI-only adaptive on pooled natural data
  and `0.612 dB` worse on coherent natural data;
- recovery of the combined pixel-oracle gap is negative on pooled natural data
  (`-411%`) because ARI regresses relative to the stronger current green stage.

This is not just another threshold failure. Even an oracle cannot select a
good coherent reconstruction from the transferred 44-candidate bank. An
eight-direction, phase-specific design could add different candidates, but it
would no longer be a direct implementation of published ARI: it would require
new X-Trans guide, support, residual, and criterion choices of the kind already
shown unreliable in the prior selector studies.

Final adaptive R/B was therefore not implemented. None of triggers A-C is
satisfied strongly enough to justify that additional complexity.

## Authenticated reference and reuse boundary

The primary source is Monno, Kiku, Tanaka, and Okutomi, [*Adaptive Residual
Interpolation for Color and Multispectral Image Demosaicking*](https://doi.org/10.3390/s17122787),
Sensors 17(12), 2787, 2017. The earlier conference version is [ICIP
2015](https://doi.org/10.1109/ICIP.2015.7351687).

| Artifact | Version/date | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| `Sensors_ARI.zip` | 1.0, 2017-12-07 | 720,117 | `eecb94b92a2f1f4fb1bbdf41de93982697fbba09baa80cf8227104203dbd9272` |
| Sensors paper | published 2017 | 1,243,059 | `32c11929c42a7550b57b50c74b0cecbfb12f0f985af77dcfa60acfb5bedd2a16` |
| ICIP paper | published 2015 | 2,568,311 | `f763278c460ae0d40fdd8829513c44462e5748c4cf34d94de094ff3741a7cde5` |
| frozen Bayer manifest | local v1 | 24,099 | `0b45395709c1b0be6527970443b158cb1d17a72b69548c8d5ab6a9d83a6a4ded` |

The official `readme.txt` says the code is available only for research,
declares all rights reserved, and supplies no open-source license. No upstream
MATLAB file is tracked. The generator authenticates the external archive and
every invoked source file before making a temporary working copy.

### Octave compatibility and the reference defect

The guided-filter least-squares residual is mathematically nonnegative. Due to
floating-point cancellation, the official code can nevertheless produce a
tiny negative value before `dif.^0.5`. MATLAB tolerates the resulting complex
temporary in this path; Octave's image-package `imfilter` rejects it. The
external runner changes only the temporary authenticated copy to evaluate
`max(dif,0).^0.5`. This enforces the intended real RMS and is recorded in the
manifest.

Separately, `green_interpolation.m` line 178 stores vertical MLRI red-at-green
updates through `maskGr`, while the corresponding RI expression and the Bayer
geometry require `maskGb`. Iteration 1 agrees without special handling; later
iterations diverge strongly on chromatic periodic data. The independent Bayer
implementation has an explicit parity switch that preserves this exact source
behavior. The X-Trans study uses the corrected geometry and does not propagate
the typo.

## Native-Bayer reproduction

The frozen corpus contains:

- constant gray and saturated R/G/B;
- horizontal and vertical gradients;
- horizontal, vertical, and diagonal edges;
- isolated white impulse and saturated red point;
- periodic monochrome and periodic chromatic textures;
- all scenes in GRBG and selected phase-sensitive scenes in RGGB, GBRG, and
  BGGR, for 19 cases total.

For every case the corpus records normalized linear truth, the separated Bayer
mosaic, the official green result, the official complete RGB result, and the
independent green result. All are little-endian channel-major float32 with
individual SHA-256 identities.

| Bayer parity measure | Value |
| --- | ---: |
| Aggregate RMS | `3.258969e-8` |
| Complete-image maximum | `6.423683e-6` |
| 16-pixel-interior RMS | `4.340215e-9` |
| 16-pixel-interior maximum | `3.126100e-8` |

The green stage required by the X-Trans experiment is therefore independently
reproduced. Complete official R/B outputs are frozen, but the two adaptive R/B
stages were intentionally not independently ported because the green
feasibility gate failed.

## Exact published architecture

For each horizontal or vertical Bayer pair, iteration `k` uses the preceding
interpolation as the mutually guiding pair. RI estimates the affine gain from
values:

\[
a_{RI}=\frac{E[IG]-E[I]E[G]}{E[I^2]-E[I]^2+\epsilon},
\qquad b_{RI}=E[G]-a_{RI}E[I].
\]

MLRI estimates the gain from approximate Laplacians but still obtains the DC
offset in the value domain:

\[
a_{MLRI}=\frac{E[(\nabla^2 I)(\nabla^2 G)]}
{E[(\nabla^2 I)^2]+\epsilon},
\qquad b_{MLRI}=E[G]-a_{MLRI}E[I].
\]

Both form an observed-minus-tentative residual, linearly interpolate that
residual, add it to the tentative estimate, and use the completed result as the
next guide. RI starts with a `3x5` horizontal window and MLRI with `1x9`;
height and width each grow by two at every iteration. Green uses 11 iterations.
Each adaptive R/B stage uses two.

For the two paired guide differences, the paper defines magnitude `d`,
directional variation `delta d`, and exponents `(m,n)=(2,1)`. The reference
implementation smooths the two terms separately with a `5x5`, sigma-2 Gaussian
and evaluates:

\[
c_k = g(d_k)^2 g(\delta d_k).
\]

For every branch/direction, the candidate at the smallest criterion is
retained. The four retained RI-H, RI-V, MLRI-H, and MLRI-V results are then
combined with reciprocal criterion weights `1/(c+1e-10)`.

## Structural comparison

| Operation | corrected-final X-Trans MLRI | ULRI/fixed-pass reference | published ARI / this feasibility transfer |
| --- | --- | --- | --- |
| CFA | 18 canonical X-Trans phase cells | one source phase generalized locally | Bayer H/V green topology; phase-aware X-Trans H/V transfer |
| Guide initialization | phase-specific H/V/diagonal G/R/B filters | same | observed samples plus directional linear completion |
| RI branch | absent | absent | value-domain guided regression |
| MLRI branch | Laplacian-domain gain | same | independent Laplacian-domain branch |
| Residual | observed minus tentative, phase-specific interpolation | same | observed minus tentative, phase-aware line interpolation |
| Directions | eight split H/V/diagonal hypotheses | same | published H/V; this study retains H/V without inventing diagonal ARI |
| Iteration | fixed two passes, sigma 2 then 1 | globally chosen fixed 1-4 passes | 11 candidates per branch/direction, growing supports |
| Iteration criterion | none | none | smoothed convergence magnitude squared times smoothness |
| Fusion | inverse directional energy | same | reciprocal minimum criterion across RI/MLRI and H/V |
| Green output | fixed fused reconstruction | fixed fused reconstruction | adaptive 44-candidate bank |
| R/B | corrected direct final stage | source provisional/final blend | not transferred because green failed gate |
| Stop/selection | none | whole-image experiment only | per-pixel minimum criterion and weighted combination |

ARI's actual novelty is therefore real: a value-domain RI hypothesis family,
independent RI/MLRI iterative states, candidates from every iteration, and a
single criterion used for both iteration retention and branch fusion.

## X-Trans adaptation under test

X-Trans has no Bayer “R line,” no two green classes that partition neatly by
axis, and no published ARI support geometry. The least assumptive direct
transfer was used:

1. canonical scalar X-Trans observations remain the only measured samples;
2. each row/column linearly completes green and the paired R or B guide at the
   irregular measured positions;
3. RI and MLRI use the exact published regression domains, window starts, and
   support growth;
4. residuals are calculated only at physically measured sites and linearly
   interpolated along that axis;
5. the published criterion and all 11 candidates are retained independently
   for RI-H, RI-V, MLRI-H, and MLRI-V;
6. observed green is preserved exactly;
7. no diagonal branch, existing X-Trans energy feature, alias feature,
   postprocessing, or learned selector is retrofitted.

The complete bank is `2 branches x 2 directions x 11 iterations = 44`
candidates. The experiment covers the established 81 cases, including all 18
unique phase cells for impulses.

## Aggregate green results

All values evaluate only missing-green samples in the 16-pixel interior.

| Group/method | PSNR dB | RMS | p99 absolute | maximum absolute |
| --- | ---: | ---: | ---: | ---: |
| natural - Markesteijn | 30.427 | 0.030107 | 0.130095 | 0.685136 |
| natural - corrected MLRI | **38.343** | **0.012102** | 0.040295 | 0.613191 |
| natural - high refinement (`slow3`) | **38.353** | **0.012087** | **0.036550** | 0.721426 |
| natural - RI adaptive | 33.439 | 0.021285 | 0.093778 | 0.606306 |
| natural - MLRI adaptive | 35.336 | 0.017107 | 0.073392 | **0.434643** |
| natural - full ARI | 34.867 | 0.018057 | 0.078334 | 0.441655 |
| natural - combined pixel oracle | 39.880 | 0.010140 | 0.046073 | 0.363208 |
| natural - combined 7x7 oracle | 33.878 | 0.020234 | 0.085374 | 0.450358 |

The pixel oracle is fragmented and optimistic. At a practical 7x7 support the
bank falls below the existing algorithm, which is why the oracle-gap recovery
denominator is not positive for pooled natural/coherent data.

### Sparse versus coherent natural sources

| Group/method | sparse PSNR | sparse p99 | coherent PSNR | coherent p99 |
| --- | ---: | ---: | ---: | ---: |
| Markesteijn | 42.445 | 0.020826 | 28.175 | 0.162286 |
| corrected MLRI | 36.654 | 0.039316 | 40.685 | 0.034260 |
| low refinement (`slow0`) | 39.453 | 0.028900 | 32.285 | 0.098976 |
| high refinement (`slow3`) | 35.596 | 0.047327 | **46.087** | **0.017867** |
| RI adaptive | 41.469 | 0.023881 | 31.442 | 0.111899 |
| MLRI adaptive | **41.700** | 0.023584 | 33.628 | 0.083946 |
| full ARI | 41.677 | **0.022650** | 33.016 | 0.090315 |
| combined pixel oracle | 48.576 | 0.003992 | 37.837 | 0.058487 |
| combined 7x7 oracle | 43.157 | 0.020665 | 31.806 | 0.103629 |

On sparse natural data ARI recovers `88.3%` of the 7x7 oracle gap from
corrected MLRI. On coherent data there is no recoverable gap: even the oracle
bank is worse than the current method.

## Natural-source decision table

Green PSNR, dB:

| Source | Markesteijn | corrected MLRI | low refinement | high refinement | RI adaptive | MLRI adaptive | ARI | combined 7x7 oracle |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Astronaut | 38.542 | 35.994 | 37.595 | 34.281 | 37.996 | 37.486 | 38.273 | 38.865 |
| Brick | 44.422 | 56.241 | 48.707 | **61.946** | 49.826 | 48.911 | 49.887 | 49.052 |
| Grass | 27.184 | 39.186 | 31.875 | **43.344** | 30.202 | 32.156 | 31.547 | 30.829 |
| Gravel | 32.038 | 43.239 | 36.016 | **48.084** | 36.402 | 36.789 | 36.765 | 35.269 |
| Hubble | **39.428** | 33.640 | 36.439 | 32.583 | 38.451 | 38.683 | 38.659 | 40.139 |
| Hydra | 73.829 | 63.284 | 66.308 | 62.190 | 77.295 | 74.215 | **77.584** | 76.528 |
| Page | 24.540 | 37.663 | 28.387 | **44.431** | 27.758 | 30.439 | 29.681 | 28.231 |

ARI is not merely choosing a compromise pass count: it is excellent on Hydra
and nearly reaches Markesteijn on Hubble. But it forfeits the coherent detail
which motivated repeated X-Trans MLRI in the first place.

## Candidate novelty and oracle contribution

Across natural cases, corresponding RI/MLRI values correlate at `0.9943` and
their errors at `0.8830`; the branches are different but strongly related. RI
is the lower-error branch oracle at `41.38%` of missing-green samples and MLRI
at `58.62%`.

| Natural group | MLRI pixel oracle | combined pixel oracle | RI marginal | MLRI 7x7 oracle | combined 7x7 oracle | RI marginal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| all natural | 39.307 | 39.880 | +0.572 dB | 33.811 | 33.878 | +0.068 dB |
| sparse | 48.118 | 48.576 | +0.458 dB | 42.978 | 43.157 | +0.179 dB |
| coherent | 37.303 | 37.837 | +0.534 dB | 31.774 | 31.806 | +0.032 dB |

H1 is **partially confirmed**: original RI supplies complementary individual
pixels, especially on sparse structures, but almost all of that marginal
benefit disappears when a spatially coherent choice is required.

### Spatial-support oracle coherence

The pixel-oracle candidate identity is extremely fragmented. Increasing the
support does reduce its boundary density, but the 7x7 label agrees with the
pixel oracle at only 3-11% of missing-green sites in the representative first
phase. The branch map forms some large regions, yet retains a median component
size of one pixel because isolated RI/MLRI decisions remain widespread.

| Source | pixel candidate boundary density | 7x7 candidate boundary density | 7x7 agreement with pixel oracle | largest 7x7 branch region |
| --- | ---: | ---: | ---: | ---: |
| Brick | 0.940 | 0.259 | 7.91% | 1,774 px |
| Hubble | 0.962 | 0.471 | 5.49% | 5,819 px |
| Hydra | 0.947 | 0.569 | 3.14% | 1,996 px |
| Page | 0.914 | 0.296 | 10.53% | 6,369 px |

The 3x3, 7x7, and 15x15 results, candidate and branch boundary densities,
scale agreement, connected-region counts/sizes, and MLRI-direction coherence
are recorded per case in `results.json`. Spatial support therefore makes the
maps smoother by changing the hypothesis, not by discovering a stable region
version of the highly optimistic pixel oracle.

## Does the published criterion rank candidates?

| Group | within-pixel Spearman | pairwise AUC | top-1 | top-2 | selected-candidate regret MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| all natural | 0.331 | 0.628 | 6.89% | 11.32% | 2.592e-4 |
| natural sparse | 0.184 | 0.568 | 4.53% | 7.53% | 5.535e-5 |
| natural coherent | 0.430 | 0.669 | 8.71% | 14.09% | 3.914e-4 |
| synthetic sparse | 0.776 | 0.813 | 34.54% | 61.70% | 4.511e-4 |
| synthetic coherent | 0.468 | 0.683 | 7.84% | 14.88% | 5.895e-2 |

Pooled correlation is much stronger than within-pixel ranking because the
criterion identifies difficult regions. Calibration is directionally correct:
the lowest criterion decile has less error than the highest on Hubble, Hydra,
Brick, and Page. That does not make it a useful candidate selector. On the
sparse natural sources where alias safety matters most, its within-pixel
Spearman is only `0.184` and top-1 accuracy `4.53%`.

H2 is **rejected**. ARI repeats the known pattern: a convergence/residual score
can indicate local difficulty while failing to identify the correct
interpretation among candidates at the same pixel.

The aggregate-candidate comparison makes the failure concrete. For each
representative source below, the first candidate minimizes mean published
criterion while the second minimizes actual green RMS. Criterion succeeds on
Brick and Page, nearly succeeds on Hydra, but selects a different branch and
iteration on Hubble even though the wrong candidate appears substantially
more confident.

| Source | lowest-criterion candidate | iteration | mean criterion | true RMS | true-best candidate | iteration | its criterion | true RMS |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| Brick | RI-V | 10 | 35.80 | 0.004083 | RI-V | 10 | 35.80 | 0.004083 |
| Hubble | RI-H | 4 | 1,206 | 0.014608 | MLRI-H | 8 | 4,261 | 0.013351 |
| Hydra | RI-V | 10 | 0.002485 | 0.000165 | RI-H | 10 | 0.002613 | 0.000159 |
| Page | MLRI-H | 10 | 40,864 | 0.047898 | MLRI-H | 10 | 40,864 | 0.047898 |

## Iteration behavior and ablation

Best fixed ARI iteration by source is 10 for Brick, Grass, Gravel, and Page;
2 for Hubble; 6 for Astronaut; and 9 for Hydra. Thus the candidate bank still
contains the earlier sparse/coherent refinement conflict.

The published selector does discover part of that structure:

- Hubble RI-H/RI-V select iteration 0 at most pixels, with mean selected
  iterations 2.44/2.28;
- Brick and Page overwhelmingly select iteration 10 in their RI vertical and
  most MLRI/RI directions;
- Hydra is different: RI selects iteration 10 heavily while MLRI peaks at 9,
  yet its absolute errors are extremely small;
- the Hubble candidate oracle prefers MLRI at 62.9% of evaluated pixels, so
  “RI for stars” is not a universal branch rule;
- Brick/Page oracle choices are also mixed, not a clean RI-versus-MLRI region
  partition.

| Natural ablation | PSNR dB |
| --- | ---: |
| RI adaptive only | 33.439 |
| MLRI adaptive only | **35.336** |
| best globally fixed ARI iteration (10) | 35.005 |
| published full ARI weighting | 34.867 |
| combined pixel oracle | 39.880 |
| combined 7x7 oracle | 33.878 |

Branch combination is not beneficial: full ARI is `0.469 dB` below MLRI-only
adaptive on all natural sources and `0.612 dB` below it on coherent sources.
H3 is **partially confirmed as behavior but rejected as a solution**. H4 is
**rejected**.

The tracked maps show all four published iteration maps, the strongest final
branch/direction and iteration, and the ground-truth pixel-oracle label:

- [Hubble](images/xtrans-ari/map-hubble.png)
- [Hydra](images/xtrans-ari/map-nasa-hydra-starfield.png)
- [Brick](images/xtrans-ari/map-brick.png)
- [Page](images/xtrans-ari/map-page.png)
- [white impulse](images/xtrans-ari/map-white-impulse.png)
- [continuous line](images/xtrans-ari/map-transition-coherence-continuous-line.png)

## Synthetic controls and tail behavior

The same 81-case corpus includes isolated white and saturated R/G/B points,
1/2/3-pixel dots, star fields, thin lines and intersections, density/thickness
transitions, saturated and chromatic edges, high-frequency monochrome and
chromatic textures, a frequency sweep, and all 18 phase cells.

Synthetic sparse data is favorable to ARI (`31.589 dB` versus `25.664 dB`
corrected MLRI), with `97.3%` recovery of its 7x7 oracle gap. Synthetic
coherent/high-frequency controls are strongly unfavorable (`11.657 dB` ARI
versus `13.817 dB` corrected MLRI and `16.514 dB` Markesteijn). The complete
81-case p99 is `0.2130` for ARI versus `0.0946` corrected MLRI. This is a new
tail regression, not a safe exchange.

Complete 81-case absolute-error tails:

| Method | median | p90 | p95 | p99 | maximum |
| --- | ---: | ---: | ---: | ---: | ---: |
| Markesteijn | 0.001547 | 0.032417 | 0.056654 | 0.162376 | 0.979000 |
| corrected MLRI | **0.000821** | **0.012113** | **0.019303** | **0.094606** | 0.995063 |
| RI adaptive | 0.001347 | 0.023464 | 0.041874 | 0.218118 | **0.978821** |
| MLRI adaptive | 0.001291 | 0.021581 | 0.036009 | 0.212158 | 0.995653 |
| full ARI | 0.001241 | 0.021586 | 0.037187 | 0.213014 | 0.987492 |

## Runtime

The vectorized Python feasibility implementation is not a production timing
comparison. On a 168x168 natural case, medians were:

| Component | Seconds |
| --- | ---: |
| RI candidate branch | 0.3164 |
| MLRI candidate branch | 0.3318 |
| criterion smoothing (included above) | 0.0391 |
| complete 44-candidate bank/analysis input | 0.6641 |
| existing untiled corrected-final C++ reference | 1.0550 |
| existing untiled Markesteijn reference | 0.0046 |

The offline bank stores 44 green planes plus 44 criterion planes while a case
is evaluated. A production implementation would need streaming/minimum
retention to avoid that storage, but it would still execute 22 iterative
branch-direction sequences and their growing guided filters. Performance was
not the rejection reason; quality failed first.

## Hypotheses and trigger decision

| Hypothesis | Result |
| --- | --- |
| H1 - RI adds complementary information | **Partial.** +0.572 dB pixel-oracle headroom, but only +0.068 dB at 7x7 and no coherent-bank competitiveness. |
| H2 - published criterion predicts correctness | **Rejected.** Natural within-pixel top-1 6.89%; sparse Spearman 0.184. |
| H3 - adaptive iteration resolves fixed-pass conflict | **Rejected as a solution.** It recognizes some low/high iteration structure, but cannot recover coherent quality. |
| H4 - RI/MLRI combination matters | **Rejected.** Full ARI is worse than MLRI-only adaptive. |
| H5 - green gain propagates to final RGB | **Not triggered.** Green fails the coherent and tail gates, so final R/B was deliberately not run. |

Trigger A fails because held-out green is not a stable improvement and coherent
natural sources regress by `7.67 dB` from corrected MLRI. Trigger B fails the
practical requirement: combined 7x7 oracle is worse than corrected MLRI on
pooled natural data. Trigger C fails because branch/iteration selection is
mixed and inaccurate despite broad sparse/coherent trends.

## Required answers

- **Was ARI reproduced?** The official full outputs are authenticated and
  frozen; the independently required green stage reproduces them to float32
  precision. Full independent R/B was not needed after the green gate failed.
- **What is novel?** Parallel value-domain RI and Laplacian-domain MLRI,
  independent iterative candidates, per-pixel iteration retention, and
  criterion-weighted branch/direction fusion.
- **Does RI add candidates?** Yes at individual pixels, including Hubble/Hydra,
  but not enough spatially coherent headroom.
- **Does the criterion select correctly?** No. It calibrates difficulty but is
  unreliable within one pixel, especially on sparse natural structures.
- **Does iteration behave as desired?** Broadly on Hubble versus Brick/Page,
  but not consistently (Hydra is a counterexample) and not with adequate
  coherent reconstruction.
- **Sparse safety?** Strong mean/RMS behavior, but mixed branch choices and
  remaining large maxima; it does not rescue the full corpus.
- **Coherent detail?** Fails decisively. Even the pixel oracle remains below
  the established high-refinement MLRI result.
- **Oracle results?** Natural pixel oracles: RI 36.650 dB, MLRI 39.307 dB,
  combined 39.880 dB. Natural 7x7 oracles: 32.282, 33.811, and 33.878 dB.
- **Final RGB?** Not evaluated, per the predeclared trigger.

Most importantly:

> **ARI adds some genuinely complementary RI candidates and does perceive part
> of the sparse/coherent iteration conflict, but the direct published
> candidate architecture is not strong enough on X-Trans coherent texture and
> its criterion cannot rank natural candidates reliably. It does not break the
> fixed-pass conflict. Further RI/MLRI-family production work should stop.**

## Reproducibility

Tracked artifacts are in `devnotes/images/xtrans-ari/`:

- dataset SHA-256: `a0b56e18a0247e70ca10f592fce76143e730c13d71301003b1d3473e48c45097`;
- results SHA-256: `d71a2a51111aca28eb5cd8f631b01c1945262e2f7539ddce9ff5198bf76ad089`;
- artifact manifest SHA-256: `3518de4fd16f64c8877a2f8021d81dd539b6f524495d391c9afd2151ffcd6802`.

A second clean-directory generation was byte-identical, including all six PNG
maps. Wall-clock measurements are printed during generation and summarized in
this report, but deliberately excluded from canonical `results.json` so the
reviewed artifacts remain reproducible.

The external ARI source, papers, Hydra TIFF, and temporary candidate planes are
not tracked. Commands and exact identities are documented in
`tools/xtrans_ari/README.md`.
