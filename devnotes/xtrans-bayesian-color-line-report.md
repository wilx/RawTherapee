# X-Trans Bayesian Two-Color Neighborhood-Likelihood Feasibility

## Decision

**NO-GO at Gate 1.**  The Bennett neighborhood likelihood does not recover the
missing position evidence even when the local two-color endpoints are supplied
from ground truth.  Following the experiment's mandatory early-stop rule,
Stages B--V were not run and no production demosaicer was added.

The best adapted likelihood used the smallest tested support, 3x3, and the
preceding line experiment's lower channel cutoff.  On pooled natural data it
*increased* position RMS from 0.004833 to 0.005931, a 22.7% regression rather
than the required 30--50% reduction.  The faithful 3x3 result is 0.005964 and
the paper's 5x5 support is substantially worse at 0.009052.

The pooled sparse result is misleading if read alone.  The 3x3 likelihood
improves Hubble but still reduces Hydra from 80.960 to 79.894 dB.  With the
paper's channel cutoff, Hydra falls to 71.727 dB.  It therefore fails the
requirement to improve both independent star fields and does not preserve
individual sparse structure reliably.

## Authenticated reference and implementation status

The reference is:

> Eric P. Bennett, Matthew Uyttendaele, C. Lawrence Zitnick, Richard Szeliski,
> and Sing Bing Kang, "Video and Image Bayesian Demosaicing with a Two Color
> Image Prior," ECCV 2006, LNCS 3951, pp. 508--521,
> DOI `10.1007/11744023_40`.

The supplied `/home/wilx/Downloads/11744023_40.pdf` is 1,037,995 bytes with
SHA-256 `604f54015c22d362444dbe3063a62610c7878ee4bd84adc0f2de8b5238cfdeef`.
The Microsoft Research copy is 1,144,774 bytes with SHA-256
`7706f6bdcc16025966dd7f3bf9451810916efdc0f59812ef217e968adda45489`.
Their PDF containers differ, but `pdftotext -layout` produces byte-identical
text.  The paper identity and equations were also cross-checked against
Microsoft's patent publication WO2007089519A1 / US7706609B2.

The official Microsoft Research page, author publication pages, patent, and
targeted source searches exposed no executable author implementation.  This is
an evidence-bounded statement, not proof that private or unindexed code never
existed.  The experiment is an independent mathematical implementation and
copies no reference source.  The paper PDF does not state a software license;
the patent is not a source-code license.

Sources:

- <https://www.microsoft.com/en-us/research/publication/video-and-image-bayesian-demosaicing-with-a-two-color-image-prior/>
- <https://doi.org/10.1007/11744023_40>
- <https://patents.google.com/patent/WO2007089519A1/en>

## Exact Bennett formulation

The paper first bootstraps full RGB with Malvar--He--Cutler high-quality linear
interpolation.  At each target it clusters the 5x5 RGB neighborhood into two
representative colors using weighted two-means in RGB.  A sample's weight is
the inverse Euclidean distance from the kernel center; cluster sizes are not
balanced.  Samples farther than one standard deviation from their closest
cluster mean are rejected and clustering is repeated.  The resulting cluster
means are `J` and `K`.

For one target color,

```text
C(alpha) = (1 - alpha) J + alpha K.
```

For every physically measured CFA sample `s_i` in the target neighborhood,
whose sensor channel is `t_i`, Bennett assumes

```text
P(s_i | alpha,J,K) proportional to
    exp(-(s_i - (J[t_i] + alpha (K[t_i]-J[t_i])))^2 / (2 sigma_i^2)),

sigma_i = sigma_N[t_i] (1 + lambda distance_i),  lambda = 6.
```

The important interpretation is that every neighbor is treated as a noisy
observation of the *target's single alpha*.  The model does not estimate a
separate neighboring alpha and transport it to the target.  Under the paper's
independence assumption, the product of Gaussians gives

```text
alpha_i       = (s_i - J[t_i]) / (K[t_i] - J[t_i])
variance_i    = (sigma_i / (K[t_i] - J[t_i]))^2
alpha_star    = sum(alpha_i / variance_i) / sum(1 / variance_i).
```

Components with `abs(K[t_i]-J[t_i]) < 2` in the paper's 8-bit domain are
ignored.  This experiment normalizes the threshold to `2/255`.  All practical
coordinates are bounded to `[0,1]`.

The prior is `P(alpha)=1` at 0 and 1 and `eta` elsewhere, so only 0, 1, and the
bounded likelihood optimum need comparison.  The paper states only `eta < 1`
and says it approaches one under heavy noise; it gives no numeric value.  It
also does not publish how `sigma_N` is estimated.  Gate 1 therefore uses
`eta=1` to isolate the new measurement evidence, while `eta=0.5,0.1,0.01` is
reported separately.  Every tested endpoint bias worsens the flat result.

The paper says quantitative error can be reduced by forcing the native sensor
component back into the final output.  The experiment performs that forcing for
both A0 and A1.  Its metrics still measure alpha before forcing.

## Bayer reference reproduction

No original Bennett test images, exact `eta`, `sigma_N`, or executable output
were available, so the published Table 1 numbers cannot be reproduced exactly.
Two independent Bayer checks were completed before the X-Trans adaptation.

First, equations 5--11 were exercised on a deterministic RGGB corpus with exact
two-color endpoints:

| Bayer scene | alpha RMS | RGB RMS | maximum RGB error |
| --- | ---: | ---: | ---: |
| constant blend | 7.86e-17 | 4.14e-17 | 1.11e-16 |
| soft transition | 0.002118 | 0.001209 | 0.007055 |
| vertical edge | 0.031635 | 0.018186 | 0.109677 |
| horizontal edge | 0.031635 | 0.018186 | 0.109677 |
| diagonal edge | 0.039946 | 0.023097 | 0.129611 |

The exact constant result validates the Gaussian product and channel handling.
The edge error is expected behavior of the published local-similarity model:
neighboring samples on the opposite side are assigned nonzero likelihood
weight.

Second, the complete published structure was reproduced independently using
Malvar--He--Cutler 5x5 HQLI, a 5x5 weighted RGB two-means, one-standard-deviation
outlier rejection and re-fit, and the likelihood.  The paper leaves four
implementation details unspecified: K-means initialization, iteration count,
how literal inverse distance is made finite at the center, and how RGB cluster
variance becomes one scalar.  The deterministic reproduction uses minimum and
maximum mean-RGB initialization, eight iterations, `1/(1+distance)`, and the
standard deviation of Euclidean RGB distance.  These are inferences, not
claimed author choices.

| Bayer scene | HQLI RGB RMS | Full reproduced method RGB RMS |
| --- | ---: | ---: |
| constant blend | 4.00e-17 | 5.12e-17 |
| soft transition | 0.006959 | 0.003471 |
| vertical edge | 0.089658 | 0.036416 |
| horizontal edge | 0.089658 | 0.036416 |
| diagonal edge | 0.116703 | 0.068908 |

The full reproduction is exact on a constant and improves its HQLI bootstrap
on every nonconstant scene.  This reproduces the paper's qualitative Bayer
refinement behavior while remaining deliberately weaker than a claim of
numeric author-output parity.

## Stage A setup

- Corpus: the frozen 81-case color-line corpus in normalized camera-linear RGB.
- Endpoints: the prior experiment's ground-truth 3x3 local-PCA extrema.
- A0: previous bounded one-CFA-sample coordinate with its degeneracy fallback.
- A1: actual X-Trans sample identities, Bennett product likelihood, supports
  3x3, 5x5, 7x7, 11x11, and 15x15.
- A2: neighboring ground-truth RGB projected onto each target's endpoint line;
  the target itself is excluded.
- A3: a weighted alpha plane fitted to the same target-relative oracle
  projections; the target itself is excluded.
- Boundaries: reflected.  Native CFA components are forced after alpha
  estimation.
- Gate selection: A1 only, flat prior, minimum natural position RMS across the
  faithful supports and a 3x3 `1e-4` channel-cutoff control.  Oracle variants
  cannot select a practical result.

A2 and A3 coincide at the target because a complete symmetric square support
has zero weighted first moments: its fitted plane intercept is algebraically
the same as the weighted neighboring-alpha mean.  Keeping both calculations
and testing their equality prevents this geometry from being mistaken for two
independent upper bounds.

## Gate-1 results

### Pooled natural data

| Alpha method | Position RMS | Alpha RMS | RGB RMS | PSNR |
| --- | ---: | ---: | ---: | ---: |
| A0 one sample | 0.004833 | 0.09317 | 0.005335 | 45.457 dB |
| A1 3x3, `1e-4` cutoff | 0.005931 | 0.08428 | 0.006304 | 44.008 dB |
| A1 3x3 | 0.005964 | 0.20488 | 0.006333 | 43.968 dB |
| A1 5x5 (paper support) | 0.009052 | 0.20949 | 0.009288 | 40.642 dB |
| A1 7x7 | 0.011384 | 0.21857 | 0.011568 | 38.735 dB |
| A1 11x11 | 0.014644 | 0.23930 | 0.014783 | 36.605 dB |
| A1 15x15 | 0.016951 | 0.25606 | 0.017069 | 35.356 dB |

Lowering the channel cutoff improves 3x3 position RMS by only 0.6%; it does not
change the decision.  The monotonic damage as faithful support grows is the
signature of model mismatch, not
insufficient sample count.  Nearby CFA samples observe nearby pixels whose
blend coordinates often differ from the target.  Treating them as additional
measurements of one constant target alpha converts real spatial structure into
bias.

At the paper's 5x5 support, adding the unpublished endpoint prior also moves in
the wrong direction:

| `eta` | Position RMS | PSNR |
| ---: | ---: | ---: |
| 1.0 | 0.009052 | 40.642 dB |
| 0.5 | 0.009215 | 40.487 dB |
| 0.1 | 0.009487 | 40.240 dB |
| 0.01 | 0.009876 | 39.902 dB |

### Sparse sources

| Source | A0 position RMS | A1 3x3 position RMS | A0 PSNR | A1 3x3 PSNR |
| --- | ---: | ---: | ---: | ---: |
| Hubble | 0.008436 | 0.006835 | 40.607 dB | 42.162 dB |
| Hydra | 0.0000654 | 0.0000821 | 80.960 dB | 79.894 dB |

The best adaptation gains 1.56 dB on Hubble because the spatial likelihood
suppresses some off-line sample error.  It still loses 1.07 dB on Hydra because
its already well-located sparse structure is averaged with background/neighbor
evidence.  The faithful cutoff loses 9.23 dB on Hydra.  A method that makes one
known star field better and the independent one worse does not supply the
required observable sparse-star evidence.

The A2/A3 oracle-neighbor control is also negative: at 3x3 it gives 0.03922
natural position RMS and 0.00923 sparse position RMS.  Even perfect neighboring
RGB values do not determine the target coordinate under a local-constant-alpha
assumption.  This does not refute spatial models in general; it refutes this
particular way of treating neighbors as repeated measurements of the target.

## X-Trans phase behavior

All 18 distinct X-Trans phase placements were tested on an impulse, saturated
point, two-color edge, and tiny star with the same true scene line.

| Scene | A0 range | A1 low-cutoff 3x3 | A1 faithful 3x3 | A1 faithful 5x5 |
| --- | ---: | ---: | ---: | ---: |
| impulse | 1.28e-18 | 0.008918 | 0.008918 | 0.008543 |
| saturated point | 1.30e-18 | 0.008918 | 0.008918 | 0.008543 |
| two-color edge | 8.24e-19 | 0.001127 | 0.001127 | 0.001522 |
| tiny star | 0.00002281 | 0.006365 | 0.006335 | 0.006384 |

The one-component estimate is essentially phase invariant on the exact
two-color controls because every non-degenerate sampled component identifies
the same alpha.  The neighborhood likelihood introduces large phase dependence
because the irregular X-Trans neighborhood changes which off-target alpha
values contribute through which endpoint components.  It therefore fails the
specific X-Trans phase-sensitivity hypothesis.

## Error decomposition and interpretation

The failure is alpha inference, not endpoint inference: endpoints are already
oracle values in Stage A.  With the faithful cutoff, pooled-natural alpha RMS
rises from 0.0932 to 0.2049 and Hydra rises from 0.1133 to 0.3135.  The low
cutoff lowers unweighted alpha RMS to 0.0843, yet position RMS still worsens;
its remaining errors are concentrated on larger-separation lines where an
alpha error costs more RGB energy.  Supplying differently colored neighbor
observations does not create independent measurements of the target unless
those neighbors can legitimately share the target alpha.

This explains why the result can coexist with Bennett's reported Bayer gains.
Their complete method estimates cluster endpoints from a bootstrapped image and
uses the likelihood as a spatially regularized refinement/noise reducer.  In
this experiment, the target endpoints are already correct and the direct target
sample is highly informative.  The remaining neighboring terms primarily add
spatial bias.  The published `lambda=6` makes the target dominant, but not
dominant enough to avoid this damage; shrinking to 3x3 only limits it.

## Required questions

### Reference fidelity

Equations 5--12, 5x5 actual-sample support, `lambda=6`, the 2-level channel
cutoff, three-candidate prior, optional native-sample forcing, HQLI bootstrap,
weighted two-means, and one-sigma re-fit are reproduced directly.  X-Trans
adaptation changes only the CFA channel identities and uses the actual irregular
sample pattern.  The missing numeric `eta` and `sigma_N` estimator are explicit;
four clustering details are labeled as inferences above.  Full published parity
is unavailable because no author code, original data, exact noise estimate, or
numeric `eta` was found.

### Alpha inference and X-Trans

No.  With true endpoints, neighborhood likelihood is worse than the target's
single CFA component on pooled natural data and dramatically increases phase
sensitivity.  It helps Hubble but fails Hydra.

### Endpoint inference

Not evaluated for X-Trans.  Gate 1 fails before practical endpoint estimation.
The Bayer structural reproduction includes clustering, but X-Trans initializer
comparisons, endpoint asymmetry, and endpoint uncertainty were deliberately not
implemented.

### Error decomposition

Alpha is already the failing component under oracle endpoints.  There is no
basis for attributing the result to practical endpoint error.

### Posterior confidence

Not calibrated because Gate 1 failed.  Posterior variance from a misspecified
constant-alpha likelihood is not treated as a trustworthy gate merely because
it is mathematically available.

### Sparse stars

Hubble improves, independent Hydra regresses by 1.07 dB even under the best
cutoff control (9.23 dB with the faithful cutoff), and exact sparse-point phase
controls become strongly phase dependent.  The method does not reliably
preserve individual star intensity/color.

### Chromatic texture

The mandatory 6--10-source expansion was not performed after Gate 1 failed.
The existing full-color natural controls are sufficient to reject Stage A but
not to claim a complete chromatic-texture safety study.

### Research direction

The answer to the central question is **no for the Bennett single-alpha
neighborhood model**.  Neighboring X-Trans samples do add data, but the model
uses them as observations of the wrong latent quantity whenever alpha changes
spatially.  The true-line oracle remains inaccessible through this likelihood,
even before practical endpoint estimation is introduced.  The mandated outcome
is therefore **NO-GO**, not "endpoint problem isolated."

## Reproducibility

Focused tests:

```text
PYTHONPATH=. .venv/bin/python -m pytest -q \
  tools/xtrans_bayesian_color_line/tests
9 passed
```

The new and preceding color-line focused suites pass together: 14 tests.

Two clean full Stage-A runs were byte-identical.  The tracked canonical result
is `devnotes/images/xtrans-bayesian-color-line/stage-a.json`, 1,518,581 bytes,
SHA-256 `e4d61004279d1678a0feb1cdf4dafcee6770e07c70ac363ff79304b22affb76d`.
It contains all 81 case rows, aggregate tables, Bayer controls, all 18 phase
placements, methodology, and authenticated reference identities.  It records
no absolute paths, timestamps, host identity, or elapsed time.
