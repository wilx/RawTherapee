# X-Trans CFA-constrained sparse RGB dictionary reconstruction report

Date: 2026-08-23

Branch: `xtrans-neural-demosaic`

Scope: offline research only; no engine, PP3, GUI, or production changes

## Decision

**NO-GO — practical standalone sparse-dictionary demosaicing.**

The experiment establishes a real but insufficient population prior:

- a learned 5x5, 128-atom RGB dictionary beats the strongest same-support DCT
  control by **2.345 dB** at eight atoms on BSDS validation patches;
- with the four full-RGB-selected atoms supplied by an oracle, X-Trans samples
  retain **92.7%** of the representation MSE quality before measured-sample
  reinjection, with full rank in every held-out patch;
- blind four-atom OMP recovers **70.2%** of the oracle-support MSE advantage
  over blind DCT;
- but the best validation-selected blind solver is no longer very sparse. Its
  LASSO solution uses **25.8 of 128 atoms** on average and reaches only
  **28.517 dB** on the sampled held-out center pixels;
- the fixed global population LMMSE estimator reaches **34.195 dB on exactly
  the same pixels**, a **5.678 dB** advantage. Corrected-final MLRI reaches
  33.643 dB and Markesteijn reaches 30.450 dB there;
- explicitly selected bright targets collapse to 19.970 dB on Hubble and
  30.008 dB on Hydra. Uniform star-field sampling had hidden this by mostly
  selecting dark background;
- four-atom OMP changes support in 13 to 18 of the 18 CFA phases on the
  analytical controls and has 3.31 to 7.96 dB phase ranges.

The useful learned prior does not justify its inference complexity and is much
less effective than the simpler global Wiener predictor. Larger dictionaries,
content-specific dictionaries, overlap-add reconstruction, residual coding,
noise studies, and a C++ prototype were therefore stopped at Gate 4.

## Question and decomposition

The experiment asks whether one continuous sparse combination of learned RGB
structures can avoid the explicit content-classification failures of the MIX3
covariance selector. For an RGB patch `x`, dictionary `D`, phase-specific
X-Trans selector `M_p`, and observed CFA vector `y = M_p x`, it separates:

1. **representation**: encode complete RGB with `D`;
2. **known-support sensing**: take the RGB-selected support as an oracle and
   fit only its coefficients from `y`;
3. **blind inference**: select and fit atoms from `y` alone.

This distinction matters. A good RGB representation does not prove that its
atoms remain distinguishable after the CFA discards two color components at
every pixel.

## Corpus and split discipline

The experiment reuses the frozen LMMSE population corpus:

- first sorted 200 BSDS500 training images for dictionary fitting;
- 20 BSDS validation images for configuration, sparsity, and solver selection;
- 20 untouched BSDS test images for the final population result;
- six pinned, genuinely chromatic scikit-image controls, three 168x168 regions
  each, held out from fitting and selection;
- established Astronaut and Hubble sources plus the independently
  authenticated NASA Hydra star field;
- Hubble, Hydra, external controls, and all synthetic scenes remain completely
  outside dictionary training.

JPEG RGB is inverse-sRGB-EOTF decoded and evaluated in normalized linear RGB.
The BSDS binding remains subject to its non-commercial research and education
terms. The tracked dataset artifact authenticates all 240 BSDS bindings and
the six external sources.

Patch locations are selected without replacement by a SHA-256-derived seed:

- training: 24 patches/source = 4,800 patches at 200 sources;
- validation/test: 24 patches/source = 480 patches per split;
- external chromatic: 12 patches per fixed crop = 216 patches;
- established controls: 96 uniform patches/source;
- Hubble and Hydra additionally use 96 separated brightest-target patches.

The bright-target set is an evaluation control, not a tuning set.

## Implementation

The implementation is independent and uses only NumPy, SciPy, and Pillow.
Scikit-learn and upstream executable sparse-coding code are not required.

### Dictionary learning

- normalized float64 atoms;
- deterministic seed `0x58445247` plus configuration identity;
- alternating classical sparse coding / dictionary updates;
- 30 FISTA iterations for L1 codes;
- L1 regularization 0.015;
- six outer iterations;
- ridge-regularized method-of-optimal-directions update;
- phase-independent full-RGB dictionary;
- no semantic classes, neural features, or target-image adaptation.

The selected model is 5x5 RGB, 75 dimensions, 128 atoms, and observable-DC
normalization. Its float32 dictionary would occupy 38,400 bytes, although no
dictionary weights are written or committed.

### Normalization controls

`N0` uses absolute linear RGB. `N1` subtracts the scalar mean of the physically
observed CFA samples from all three patch channels and restores it afterward.
N1 is inference-observable and is dramatically better:

| Patch | Atoms | Normalization | Learned S=8 | Best DCT S=8 |
|---:|---:|---|---:|---:|
| 5 | 128 | absolute RGB | 25.444 | 25.328 |
| 5 | 128 | observable DC removed | **32.308** | 29.963 |
| 5 | 256 | observable DC removed | 31.850 | 29.963 |
| 7 | 256 | observable DC removed | 28.354 | 26.905 |

The overcomplete 256-atom dictionaries do not improve the 128-atom model.

### Analytic controls and patch sizes

Two complete orthonormal controls were tested: independent-channel RGB DCT and
spatial DCT crossed with an orthogonal luminance/opponent-color basis. The
opponent basis wins consistently.

| Patch | RGB DCT S=8 | Opponent DCT S=8 |
|---:|---:|---:|
| 3x3 | 31.730 | **36.031** |
| 5x5 | 26.900 | **29.963** |
| 7x7 | 24.498 | **26.905** |
| 9x9 | 23.691 | **25.536** |

Learned 5x5 and 7x7 dictionaries were the planned primary candidates. A larger
learned 9x9 sweep was not continued after the selected 5x5 model failed the
practical population gate.

### Inference

- Stage A and C1 use stable-tie OMP at S = 1, 2, 4, 8, and 16;
- Stage B uses ridge-stabilized least squares on the oracle support;
- C2 tests FISTA LASSO at 0.0001, 0.0003, 0.001, 0.003, and 0.01;
- C3 tests the same formulation with an elastic ridge of 0.001 at three L1
  values;
- all physical CFA samples are restored exactly after reconstruction;
- pre-reinjection values and errors are retained separately;
- no clipping, false-color cleanup, denoising, or blending is applied.

## Gate 1 — RGB representation

The selected learned dictionary materially beats DCT at every tested fixed
sparsity:

| Atoms | Learned PSNR | DCT PSNR | Mean energy captured | P(patch >=30 dB) | >=40 dB | >=50 dB |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 22.999 | 22.617 | 54.0% | 49.8% | 19.4% | 7.5% |
| 2 | 25.151 | 24.070 | 70.0% | 56.0% | 24.4% | 8.5% |
| 4 | 28.127 | 26.295 | 83.6% | 68.5% | 31.2% | 10.2% |
| 8 | **32.308** | 29.963 | 93.2% | 85.0% | 45.8% | 15.2% |
| 16 | **38.330** | 35.423 | 98.2% | 97.5% | 70.0% | 30.2% |

Gate 1 therefore passes by 2.345 dB at S=8. Natural patches are usefully
compressible, but high quality is not extremely sparse: only 45.8% of
validation patches reach 40 dB with eight atoms and the training L1 codes
retain about 33 nonzero coefficients on average after convergence.

### Training-size curve

| BSDS training sources | Validation representation PSNR, 5x5/K128/S8 |
|---:|---:|
| 25 | 31.966 |
| 50 | 31.897 |
| 100 | 32.168 |
| 200 | 32.212 |

The last 100 sources add only 0.044 dB. Training scale is not the explanation
for the later practical failure.

## Gate 2 — known-support X-Trans sensing

The central held-out test decomposition at validation-selected OMP S=4 is:

| Dictionary/method | RGB representation | Oracle CFA support, pre-enforcement | Oracle CFA support, enforced | Blind CFA, enforced |
|---|---:|---:|---:|---:|
| DCT | 25.123 | 24.834 | 26.235 | 20.932 |
| Learned | **27.111** | **26.782** | **28.251** | **24.614** |

Before native-sample restoration, the learned known-support error is only
7.86% above the full-RGB representation error, giving a quality-survival MSE
ratio of **0.927**. Every selected sensing matrix is full rank; median
condition number is 1.923, p99 is 3.071, and median smallest singular value is
0.371.

Gate 2 passes. X-Trans sensing does not destroy a known four-atom support.

## Masked dictionary ambiguity

| Dictionary | Full-RGB coherence | Minimum phase-masked coherence | Maximum phase-masked coherence |
|---|---:|---:|---:|
| DCT | ~0 | 0.467 | 0.915 |
| Learned | 0.682 | 0.695 | 0.918 |

The learned atoms are already correlated, but the more important result is
that both dictionaries contain atom pairs that become almost indistinguishable
after X-Trans projection. This explains the contrast between the well-
conditioned oracle support and unstable blind selection.

## Gate 3 — blind sparse inference

Four-atom OMP support recovery is weak:

- precision: 0.394;
- recall: 0.394;
- Jaccard: 0.288;
- p99 blind-minus-oracle MSE regret: 0.0200.

Nevertheless, learned blind OMP reaches 24.614 dB versus 20.932 dB for blind
DCT and recovers 70.2% of the learned oracle-support MSE advantage over blind
DCT. Gate 3 passes under the predeclared 30% rule.

The solver controls expose the catch:

| Solver | Parameter | Validation full-patch PSNR | Center PSNR | Mean nonzero atoms |
|---|---:|---:|---:|---:|
| OMP | S=4 | 25.363 | 26.965 | 4.0 |
| LASSO | 0.0001 | 27.611 | 29.404 | 79.1 |
| LASSO | 0.0003 | 28.027 | 29.952 | 56.0 |
| LASSO | 0.001 | 28.527 | 30.610 | 35.3 |
| LASSO | **0.003** | **28.795** | **30.756** | 23.6 |
| LASSO | 0.01 | 28.393 | 30.270 | 15.0 |
| Elastic net | 0.003 | 28.775 | 30.754 | 23.7 |

The selected test result uses LASSO 0.003. It reaches 27.778 dB on full 5x5
patches and 28.517 dB at centers, with 25.8 nonzero atoms. This is better than
OMP, but the successful blind representation is no longer low-cardinality.

## Gate 4 — exact-coordinate population comparison

To avoid comparing sampled dictionary centers with unrelated full-image
metrics, all baselines were rerun on each complete BSDS test image and measured
at the identical 24 deterministic center positions per source.

| Method | Pooled PSNR |
|---|---:|
| Markesteijn | 30.450 |
| corrected-final MLRI | 33.643 |
| global GMAX LMMSE | **34.195** |
| selected learned dictionary LASSO | **28.517** |

The dictionary loses 5.678 dB to GMAX, 5.126 dB to corrected MLRI, and 1.933
dB even to Markesteijn. This is a decisive Gate 4 failure.

### Untouched BSDS sources, identical sample coordinates

| Source | Markesteijn | corrected MLRI | GMAX | learned LASSO |
|---|---:|---:|---:|---:|
| 100007.jpg | 38.781 | 44.170 | 39.981 | 34.669 |
| 100039.jpg | 29.338 | 33.374 | 30.475 | 27.157 |
| 100099.jpg | 41.484 | 42.179 | 45.490 | 33.918 |
| 10081.jpg | 42.529 | 45.803 | 41.845 | 35.290 |
| 101027.jpg | 29.243 | 38.462 | 33.645 | 29.777 |
| 101084.jpg | 29.006 | 33.286 | 28.276 | 26.564 |
| 102062.jpg | 25.043 | 29.353 | 30.792 | 23.423 |
| 103006.jpg | 23.633 | 27.576 | 29.663 | 23.992 |
| 103029.jpg | 52.025 | 50.132 | 52.784 | 45.232 |
| 103078.jpg | 27.673 | 39.475 | 39.300 | 30.634 |
| 104010.jpg | 37.633 | 31.629 | 36.971 | 32.004 |
| 104055.jpg | 43.820 | 42.610 | 46.018 | 34.867 |
| 105027.jpg | 37.689 | 40.031 | 41.096 | 33.323 |
| 106005.jpg | 44.872 | 47.802 | 46.393 | 41.959 |
| 106047.jpg | 41.865 | 46.091 | 45.822 | 41.783 |
| 107014.jpg | 36.330 | 32.386 | 36.209 | 26.462 |
| 107045.jpg | 29.307 | 35.144 | 33.767 | 29.825 |
| 107072.jpg | 35.277 | 30.002 | 33.965 | 26.587 |
| 108004.jpg | 30.342 | 28.905 | 31.542 | 26.115 |
| 108036.jpg | 27.279 | 34.388 | 33.455 | 25.240 |

The learned method does not beat GMAX on a single source in this sampled test.

## Error tails

For the selected learned LASSO full-patch reconstruction:

| Metric | Normalized error |
|---|---:|
| median absolute | 0.00486 |
| p90 absolute | 0.05096 |
| p95 absolute | 0.08364 |
| p99 absolute | 0.18150 |
| maximum absolute | 0.63658 |

The tail is too large for a production demosaicer even before comparison with
the better baselines. No clipping was used to hide excursions.

## External chromatic texture

On the six held-out genuinely chromatic controls:

- OMP S=4 full-patch: 23.455 dB;
- oracle-support enforced: 28.392 dB;
- selected LASSO full-patch: 27.594 dB;
- selected LASSO centers: 28.585 dB;
- selected LASSO p99: 0.1719 and maximum: 0.6333.

The practical result remains far from safe despite visible oracle headroom.

## Sparse stars

Uniform sampling produces apparently strong star-field averages because most
patches are dark background. The added bright-target control changes the
conclusion:

| Source/sample | Selected LASSO full-patch | Center | p99 | Maximum |
|---|---:|---:|---:|---:|
| Hubble uniform | 34.306 | 39.905 | 0.09185 | 0.33279 |
| **Hubble brightest targets** | **19.970** | **19.105** | 0.34892 | 0.70716 |
| Hydra uniform | 65.007 | 71.854 | 0.00200 | 0.01844 |
| **Hydra brightest targets** | **30.008** | **30.274** | 0.11250 | 0.32849 |

The model estimates smooth dark background well but does not reconstruct the
bright star color/intensity. This is a domain/representation and blind-
inference failure precisely on the safety class that motivated replacing the
MIX3 selector.

## Smooth, edge, periodic, and phase controls

The analytical phase sweep uses OMP S=4 so that selected support can be
inspected directly:

| Scene | Min PSNR | Max PSNR | Phase range | Distinct supports / 18 | Maximum |
|---|---:|---:|---:|---:|---:|
| gray gradient | 19.603 | 23.267 | 3.664 | 13 | 0.336 |
| chromatic gradient | 11.838 | 17.794 | 5.956 | 18 | 0.920 |
| red/gray edge | 10.722 | 18.678 | **7.957** | 18 | 0.875 |
| blue/gray diagonal | 8.977 | 13.473 | 4.496 | 18 | 1.176 |
| saturated point | 11.518 | 16.452 | 4.933 | 18 | 1.489 |
| periodic chromatic | 5.192 | 9.405 | 4.214 | 18 | 1.744 |
| points, spacing 4 | 9.354 | 13.487 | 4.133 | 18 | 1.408 |
| points, spacing 2 | 6.049 | 9.364 | 3.315 | 18 | 1.472 |

The method is not safe on the smooth chromatic gradient that defeated MIX3,
and its blind support is strongly CFA-phase-dependent. The sparse-to-coherent
transition does not become stable within the tested point patterns.

## Learned atoms

The tracked atom montage shows a diverse mix of low-frequency gradients,
directional edges, opponent-color transitions, and localized chromatic
structures. It establishes that learning did not collapse to one structure
class. The montage is normalized per atom for visibility and is not a reusable
weight artifact.

This diversity explains the RGB representation gain but not reliable CFA
selection: several different RGB atoms project to nearly the same observed
samples.

## Complexity

The unoptimized, single-threaded Python measurements are:

- OMP S=4: approximately 0.17 ms/patch;
- selected 100-iteration LASSO: approximately 1.7–1.9 ms/patch;
- selected dictionary: 9,600 float32 values, 38.4 KiB;
- complete research run including exact-coordinate native baselines: about
  803 seconds under `nice -n 10` and single-threaded BLAS.

At one center patch per output pixel, 1.8 ms/patch implies roughly 20 hours for
a 40 MP image before overlap aggregation or C++ optimization. Performance was
not the deciding failure—quality already fails by 5.678 dB versus GMAX—but the
gap to the 242-MAC/pixel LMMSE path is enormous.

## Failure classification

### Representation failure: partial

The learned dictionary clearly improves DCT and reaches 38.33 dB with 16
oracle RGB-selected atoms. However, the trained L1 codes and practical blind
solutions use roughly 25–33 atoms, so the strongest practical mode is not very
sparse. Bright stars and some chromatic structures are poorly represented by
the population dictionary.

### Projection failure: not at known low-cardinality support

Known four-atom supports remain full-rank and well-conditioned. Gate 2 passes.
Projection still raises global atom coherence to 0.918 and creates serious
ambiguity for blind selection.

### Inference failure: yes

OMP recovers only 39.4% support precision/recall. LASSO improves quality by
using many atoms, but it remains far below GMAX and has unsafe tails and phase
sensitivity.

### Domain failure: yes on sparse bright structure

BSDS training captures common natural patches but does not preserve the
brightest Hubble/Hydra targets. Training-size convergence shows that simply
adding the remaining BSDS sources does not solve this.

## Early-stop consequences

The prompt explicitly permits stopping after each gate. Because Gate 4 fails
decisively, the following expensive downstream work was not performed:

- full-image overlap-add or Hann aggregation;
- source-specific and alternate-corpus dictionary oracles;
- LMMSE-residual dictionary formulation;
- noisy sigma 0.001/0.005 evaluation;
- larger 9x9/512–1024 atom learned dictionaries;
- whole-image runtime optimization or C++ integration.

These would not plausibly close a 5.678 dB gap without changing the method into
the larger class/content-selection system the experiment was intended to
avoid.

## Required questions

### Representation

- **Are natural RGB patches sparse?** Moderately, not strongly. Eight atoms
  capture 93.2% mean normalized energy, but only 45.8% of patches reach 40 dB.
- **How many atoms are required?** Sixteen give 38.33 dB representation; the
  selected practical LASSO uses 25.8 atoms and training codes use about 33.
- **Does learned beat DCT?** Yes, by 2.345 dB at S=8.

### X-Trans sensing

- **Does known support remain identifiable?** Yes at S=4: all sampled systems
  are full rank and 92.7% MSE quality survives before reinjection.
- **Does masked coherence increase?** Strongly. Learned maximum coherence rises
  from 0.682 to 0.918; orthogonal DCT rises from zero to 0.915.

### Blind inference

- **How much oracle quality does it recover?** OMP recovers 70.2% of the stated
  oracle-support advantage over blind DCT, but remains only 24.614 dB.
- **What causes wrong support?** Projected atom ambiguity and insufficient
  low-cardinality representation; support precision/recall are 0.394.

### Statistical prior

- **Does it beat global Wiener?** No. It loses 5.678 dB on identical held-out
  positions.
- **Where is it useful?** Full-RGB representation and known-support fitting;
  these are oracle properties, not a competitive blind demosaicer.

### Sparse stars

- **Hubble?** Uniform sampling looks good, but bright targets collapse to
  19.970 dB.
- **Hydra?** Same pattern: 65.007 dB uniform, only 30.008 dB at bright targets.
- **Tiny points?** Poor, phase-sensitive, and capable of >1.4 excursions.

### Chromatic and smooth texture

- **Chromatic fine texture?** Selected LASSO reaches only 27.594 dB with unsafe
  p99/max errors.
- **Gradients/edges?** Not safe. Chromatic-gradient and edge phase ranges are
  5.96–7.96 dB.

### Domain generalization

One dictionary is more expressive than one covariance, but it generalizes much
worse in blind use than the global population covariance. It does not solve
the sparse-star class.

### Phase

Blind low-cardinality supports are not stable: 18 distinct supports occur for
most analytical scenes, with multi-dB quality ranges.

### Complexity

The selected 128-atom dictionary is small, but a 100-iteration, approximately
26-active-atom LASSO per pixel is orders of magnitude more expensive than
GMAX and still much worse.

## Final answer

> **NO-GO — practical:** Population-learned RGB atoms genuinely improve sparse
> full-RGB representation, and correct four-atom supports remain identifiable
> through the X-Trans CFA. Blind inference is the break: low-cardinality OMP
> chooses ambiguous supports, while quality-selected LASSO needs about 26 atoms
> and still loses 5.678 dB to the much simpler global LMMSE predictor on the
> exact same held-out pixels. Bright Hubble/Hydra targets, smooth chromatic
> gradients, and phase stability all fail. Further dictionary complexity is
> not justified.

The central hypothesis therefore fails as a practical replacement:

\[
\boxed{
\text{A continuous sparse population prior does not resolve X-Trans ambiguity}
\text{ well enough to beat the global Wiener estimator.}
}
\]

## Verification and artifacts

Focused tests:

```text
26 passed in 0.33 s
```

Coverage includes exact sparse recovery, deterministic training, monotonic OMP
residual, FISTA convergence, constant fields, native-sample equality, and all
18 X-Trans phase cells.

Two complete pre-baseline experiment runs produced byte-identical dataset,
synthetic, and atom artifacts and identical scientific result fields after
excluding timing keys. The corresponding normalized scientific digest was
`7a39033a5e4b73307a6e7e6f838360657060818561226e39d53fec2f62ff697f`.
The final exact-coordinate run adds freshly measured baseline timings; timing
values are intentionally not treated as deterministic scientific data.

Tracked artifacts:

| Artifact | SHA-256 |
|---|---|
| `dataset.json` | `122c09b0bb6e4d5292849179c51833646c77e64fa243f06392f9f5900b597423` |
| `results.json` | `1aa7e97d2e6e07110b0b8cac7b6d735211fe445a035eee90668721d6cef936bc` |
| `synthetic.json` | `d2fd08bda20ed4a7a428cf4ee0db5f92e4f96e294b1491c61eae1c34279d2428` |
| `atoms.png` | `c250601317230fde0d4d7a6e6ef96a6e0163ce0a383d25d275bbc0dd2cd4bc62` |

External BSDS images, NASA Hydra TIFF, native runner products, and learned
dictionary weights are not committed.
