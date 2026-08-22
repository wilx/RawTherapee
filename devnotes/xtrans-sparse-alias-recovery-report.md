# X-Trans structured sparse alias-family recovery

## Decision

**PARTIAL:** the exact X-Trans replica structure is useful, oracle support
proves that observable chromatic edges need not be smoothed away, and
replica-aware OMP captures a substantial part of the sparse approximation
available in natural RGB windows. However, high-frequency families are only
moderately sparse, exact observation fits can still select the wrong source,
and the current confidence rule is conservative on the very chromatic edges
we want to preserve. Continue with local source classification and confidence
research, not a production demosaicer yet.

This experiment changes no RawTherapee demosaicing or GUI code.

## Reused sampling model

The implementation imports the carrier lattice, mask spectra, color basis,
and signature construction from `tools/xtrans_alias`; it does not define a
second sampling operator. The recovery dictionary contains 54 columns in
carrier-major order:

```text
18 carrier positions x (L, C1, C2)
```

with:

```text
L  = (R + G + B) / sqrt(3)
C1 = (R - B) / sqrt(2)
C2 = (R - 2G + B) / sqrt(6)
```

The maximum error between this dictionary and the previous 18x54 RGB
operator transformed into `L/C1/C2` is below `2e-16`. All 18 supported CFA
phase/orientation variants preserve its singular values to numerical
precision.

## Methods

### Oracle support

For a known support `S`, recovery uses double-precision SVD-backed least
squares on `A_S`. All 54 one-component supports and all 1,431 two-component
supports are tested. Larger supports use 500 deterministic random draws per
cardinality.

### Blind support

Replica-aware OMP scores each candidate with the complex inner product of the
complete predicted 18-bin constellation. After each selection it jointly
refits every selected coefficient. The control uses only one strongest
observed bin of each atom and ignores the companion replicas.

Both methods use identical dictionaries, stopping cardinalities, and least
squares refits; the scoring evidence is the intentional difference.

### Grouped real signals

On a 24x24 grid, each real-frequency/color candidate contains its sampled
cosine and sine quadratures. Positive and negative frequencies share one
group. Group scoring projects the observation onto the complete real 2-D
subspace, including half-carrier sideband collisions.

### Confidence

For every selected support, the analysis excludes each selected atom in turn
and searches for the best forced-different support. The normalized residual
gap between the selected and competing fits is the measured evidence for
uniqueness. This directly tests alternative explanations rather than relying
on one support's condition number.

### Natural RGB corpus

Four authenticated RGB images shipped by scikit-image 0.26.0 are used rather
than X-Trans demosaiced images:

| Image | Provenance | Intended structure |
| --- | --- | --- |
| astronaut | public domain | portrait and chromatic edges |
| coffee | CC0 | saturated objects and texture |
| Hubble Deep Field | NASA public domain | fine colored texture |
| rocket | SpaceX public domain | smooth areas, edges, fine detail |

Their source SHA-256 values are recorded in the canonical JSON. Pixels are
converted from sRGB to linear RGB. The analysis uses 24, 48, and 96-pixel
windows, 50% nominal overlap, deterministic subsampling to 16 windows per
image/size, per-channel mean removal, and a separable symmetric Hann window.
This produces 192 windows and 43,008 alias-family samples.

## Oracle-support results

[Oracle recoverability plot](images/xtrans-sparse-alias/oracle-recoverability.png)

| K | full-rank supports | median full-rank condition | p99 condition |
| ---: | ---: | ---: | ---: |
| 1 | 100% | 1.000 | 1.000 |
| 2 | 100% | 1.225 | 1.802 |
| 3 | 100% | 1.500 | 2.512 |
| 4 | 100% | 1.742 | 3.185 |
| 5 | 99.6% | 1.972 | 3.822 |
| 8 | 95.0% | 2.927 | 8.464 |
| 12 | 76.8% | 4.891 | 18.189 |
| 15 | 41.0% | 8.007 | 20.659 |
| 18 | 3.8% | 13.503 | 28.792 |
| 21/24 | 0% | -- | -- |

Full-rank supports reconstruct their coefficients to roughly `1e-15`. The
first exact dependency occurs at five columns, consistent with the known
one-luma/four-chroma null relation. Thus the unrestricted 36-dimensional null
space does not prevent sparse oracle recovery: supports through K=4 are
always recoverable, and most supports remain recoverable through K=10.

## Blind recovery

[Blind support plot](images/xtrans-sparse-alias/blind-support-recovery.png)

| K | replica-aware exact support | single-peak exact support | oracle full rank |
| ---: | ---: | ---: | ---: |
| 1 | 100.0% | 27.7% | 100.0% |
| 2 | 98.3% | 4.3% | 100.0% |
| 3 | 91.7% | 1.7% | 100.0% |
| 4 | 81.3% | 0% | 100.0% |
| 5 | 65.0% | 0% | 100.0% |
| 6 | 49.0% | 0% | 98.3% |
| 7 | 33.7% | 0% | 99.0% |
| 8 | 17.3% | 0% | 97.7% |

Replica-aware support inference is reliable through roughly K=3, useful at
K=4/5, and falls sharply afterward even though the oracle subsystems are
usually full rank. The failure beyond K=5 is primarily support identification,
not matrix conditioning.

For two components, exact recovery stays between 95.0% and 98.1% over the
tested relative phases and at or above 96.9% over amplitude ratios from 1:1
through 20:1. Luma/luma and luma/chroma pairs recover at 100.0% and 99.8%; the
slightly harder chroma/chroma category recovers at 95.8%.

The representative [replica fit](images/xtrans-sparse-alias/example-replica-fit.png)
recovers all three atoms to `2.61e-16` relative coefficient error. Its best
forced-different three-atom interpretation leaves 0.242 relative residual.

## Noise and real-signal grouping

[SNR plot](images/xtrans-sparse-alias/recovery-versus-snr.png)

At 20 dB observation SNR, exact replica-aware support recovery is:

| K | exact support rate |
| ---: | ---: |
| 1 | 100.0% |
| 2 | 96.1% |
| 3 | 91.1% |
| 4 | 81.7% |
| 5 | 64.4% |

The values are close to the noiseless rates. In this range, dictionary
coherence and support density are more limiting than Gaussian perturbations.
This is not a camera-noise result; it is only an observation-space robustness
test.

Grouped cosine/sine recovery removes most phase bias:

| Case | grouped exact group | independent-quadrature OMP |
| --- | ---: | ---: |
| generic real frequencies, 72 phases | 100.0% | 96.2% |
| half-carrier frequencies, 72 phases | 99.72% | 84.07% |

The grouped method remains 100% correct for one real group at 20, 30, and 40
dB in 200 deterministic trials per level. Correctly handling both quadratures
therefore materially reduces the half-carrier phase problem found by the
previous characterization.

## Confidence and ambiguity

[Confidence-gap plot](images/xtrans-sparse-alias/confidence-gap.png)

On K=1..5 random supports plus 50 explicit samples of the known deficient
five-column support:

| Population | count | median competing gap | p90 gap |
| --- | ---: | ---: | ---: |
| correct OMP | 434 | 0.312 | 0.866 |
| incorrect OMP | 66 | 0 | `1.02e-16` |
| fundamentally deficient | 50 | 0 | `7.99e-17` |

A threshold of 0.0540, selected by balanced accuracy on this diagnostic set,
gives 97.7% sensitivity for correct recoveries, 100% specificity for
incorrect/deficient recoveries, and 98.85% balanced accuracy. Cases classified
identifiable remain on the same support under eight 40 dB perturbations in
100% of trials; uncertain cases remain stable only 87.1% of the time.

The known four-chroma alternative reproduces one luma atom with relative
residual `8.29e-16` and receives a zero confidence gap. The detector therefore
does not become confidently wrong on the explicit null example.

These numbers are an in-sample threshold study, not a calibrated production
confidence probability. More importantly, some exact synthetic edges have
small competing gaps even when their true support is full rank. Confidence is
correctly warning that a different sparse support may still explain the same
unrestricted observation.

## Natural-image sparsity

[High-frequency energy concentration](images/xtrans-sparse-alias/natural-energy-concentration.png)
and [effective sparsity](images/xtrans-sparse-alias/natural-effective-sparsity.png)

Across all families, energy is strongly concentrated: the strongest three
coefficients capture 98.36% of L, 97.23% of C1, and 96.16% of C2 energy. Those
global numbers are dominated by low-frequency families and should not be used
alone to justify high-frequency reconstruction.

For families whose strongest component has radius at least 0.25 cycles/pixel:

| Direction | weighted median effective K | top 1 | top 3 | top 5 |
| --- | ---: | ---: | ---: | ---: |
| L | 2.73 | 58.7% | 82.2% | 90.3% |
| C1 | 3.74 | 49.0% | 76.4% | 86.8% |
| C2 | 5.70 | 36.0% | 64.4% | 77.3% |
| mixed 54-column family | 3.82 | 50.0% | 73.2% | 82.4% |

High-frequency chroma is therefore not one-atom sparse, especially in C2, but
it often has useful low-cardinality energy concentration. Counts at -30/-40 dB
are frequently dense because Hann leakage and weak texture populate many
coefficients; effective sparsity and energy-capture metrics give the more
relevant approximation picture.

On 4,096 sampled natural mixed families, fixed-cardinality recovery gives:

| K | oracle coefficient error | replica OMP error | peak-control error | replica top-K recall |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.364 | 0.367 | 0.937 | 97.6% |
| 2 | 0.200 | 0.209 | 0.541 | 95.5% |
| 3 | 0.137 | 0.154 | 0.530 | 91.6% |
| 5 | 0.106 | 0.141 | 0.527 | 80.8% |

These are energy-weighted means. At K=5, replica-aware OMP retains much of the
oracle approximation while the one-peak control remains poor. This supports
the claim that the complete X-Trans magnitude/phase constellation carries
useful source-identification evidence in real RGB windows.

## Synthetic image reconstruction

[Ground-truth/oracle/blind comparison](images/xtrans-sparse-alias/synthetic-oracle-reconstruction.png)

| Scene | oracle recoverable energy | oracle PSNR | blind replica PSNR | single-peak PSNR |
| --- | ---: | ---: | ---: | ---: |
| vertical red/green edge | 100% | 316.9 dB | 319.5 dB | 5.07 dB |
| saturated red/gray axis edge | 100% | 315.8 dB | 30.93 dB | 6.48 dB |
| diagonal red/green edge | 66.61% | 26.44 dB | 31.45 dB | 6.64 dB |
| A/B/C saturated quadrant | 80% | 17.60 dB | 25.94 dB | 4.13 dB |
| A/B/C frequency sweep | 0% | 12.93 dB | 7.84 dB | 7.86 dB |

The oracle exactly reconstructs both observable axis edges. Replica-aware OMP
also exactly reconstructs the vertical red/green edge and reaches 30.93 dB on
the saturated red/gray edge, far above the previous blanket-smoothness B/C
behavior. However, the saturated red/gray observation has alternative exact
sparse explanations: its median competing-support gap is zero, and OMP's
chosen coefficients are not the true coefficients despite an essentially zero
observation residual.

For deficient scenes, oracle support identifies the allowed columns but not a
unique coefficient vector. The reported oracle image is the least-squares
minimum-norm solution, not a claim of the best possible perceptual prior. This
is why blind OMP can have higher image PSNR than the oracle-support minimum-norm
solution on the diagonal and quadrant cases.

On the saturated quadrant, the blind result's 25.94 dB exceeds the previous
global experiments A (23.35 dB), B (14.95 dB), and C (12.68 dB), although it
still trails Markesteijn (31.64 dB). This is useful evidence that structured
replica selection recovers information discarded by the blanket chroma
regularizers, but it is not yet competitive with the reference demosaicer.

The frequency sweep remains a firm negative result. Every family is deficient,
oracle support itself is non-unique, and blind sparse recovery is worse than
the 12.93 dB minimum-norm oracle result. The previous A/B/C/Markesteijn results
were 12.23/9.36/9.16/9.56 dB respectively, so even knowing the true support
does not create a high-quality reconstruction of this densely occupied scene.

## Hypothesis decisions

### H1 — sparse-support recoverability: supported with limits

K<=4 supports are always oracle recoverable, most random supports remain full
rank through K=10, and natural energy is concentrated. High-frequency C2
families are notably less sparse than low-frequency families.

### H2 — replica constellations are useful: strongly supported

Replica-aware OMP beats the one-peak control at every tested cardinality,
including 100% versus 27.7% at K=1 and 81.3% versus 0% at K=4. Natural K=5
coefficient error is 0.141 versus 0.527.

### H3 — observable chromatic edges can be preserved: partially supported

Oracle support reconstructs both axis edges exactly. Blind OMP exactly
reconstructs vertical red/green and substantially improves saturated red/gray,
but the latter still admits competing sparse solutions and is not exact.

### H4 — ambiguity can be detected: supported diagnostically

The residual-gap rule separates random correct and incorrect/deficient cases
well and detects the known null. It is conservative on several dense edge
families and has not been calibrated on a windowed image reconstruction.

### H5 — natural images are sparse enough: partially supported

Natural windows are sparse in an energy-approximation sense, and replica OMP
captures much of the top-K oracle result. High-frequency chroma, especially
C2, is not sparse enough for a universal one-to-three-atom assumption.

## Go/no-go answers

- **Oracle:** observable axis chromatic edges are exactly recoverable. The
  quadrant and sweep retain fundamental ambiguity even with support known.
- **Blind:** reliable at K<=3, useful at K=4/5, and sharply unreliable beyond.
  Real quadrature grouping nearly eliminates the tested half-carrier phase
  regression.
- **Natural sparsity:** mixed high-frequency families have weighted effective
  K=3.82; their top five components contain 82.4% of energy.
- **Confidence:** known nulls and synthetic wrong supports are detected, but
  useful edge confidence remains too conservative and uncalibrated.
- **Algorithmic value:** yes, complete replica evidence preserves information
  that smooth-chroma solvers and peak selection discard. It does not produce a
  unique answer for every observable-looking edge.

## Why the decision is PARTIAL rather than GO

The experiment clears the mathematical and family-level value tests but has
not yet established a robust local image algorithm. The natural corpus is four
sRGB photographs rather than camera-linear ground truth; image categories are
assigned at image level; Hann leakage affects small coefficients; and the
blind image prototype is global rather than overlapped/windowed. Most
critically, an exact observation fit can still choose the wrong sparse source,
as the saturated red/gray case demonstrates.

The optional Bayer repeat was not rerun: the preceding alias-characterization
experiment already established the corresponding coherence control. SSIM is
also omitted because this stage is intended to characterize identifiability
and support recovery rather than rank perceptual image quality.

The next justified step is a windowed classifier that combines replica fit,
real quadrature grouping, neighboring-window support agreement, and the
competing-support gap. It should be evaluated as a confidence/source-selection
experiment before its output is used as a demosaicer.

## Reproducibility and artifacts

The canonical result is
`images/xtrans-sparse-alias/recovery-characterization.json`, SHA-256:

```text
f550c51edf78f34d7e1ed006ff02526c607e1f251e3c5e936db93b441e4455f1
```

The full run takes approximately 47.5 seconds and 255 MiB peak RSS on the test
host. Canonical output excludes timing, paths, host identity, and timestamps.
Random studies use fixed seeds. Independent native tests cover the dictionary
contract, all one/two-column oracle fits, OMP refitting, the known null
competitor, and 72 half-carrier quadrature phases.
