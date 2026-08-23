# X-Trans two-color / color-line prior feasibility

## Decision

**PARTIAL — the local color-line prior has strong oracle headroom, but the line
cannot yet be estimated reliably from inference-time data.**

A true 3×3 local RGB line raises pooled natural PSNR from 38.936 dB for
corrected-final MLRI to 45.457 dB.  It also improves the two held-out sparse
sources: Hubble reaches 40.607 dB versus 38.382 dB for Markesteijn and 33.710 dB
for corrected MLRI; Hydra reaches 80.960 dB versus 71.790 and 63.386 dB.

That gain does not survive observable line estimation.  The train-selected
`ulri-slow0-line-3` result is 38.326 dB on pooled natural data, below corrected
MLRI, and it is 38.444 dB on Hubble/Hydra pooled, below both Markesteijn
(41.390 dB) and its own `ulri-slow0` initializer (38.654 dB).  A one-step Huber
fit improves some coherent controls but makes sparse performance worse.  The
observable rank-1 residual is uncalibrated with actual reconstruction error.

The experiment therefore establishes a genuinely useful image prior, but not
a practical X-Trans demosaicer.  No engine or GUI method should be implemented
from this result.

## Scope and corpus

The tracked corpus contains 81 normalized linear-light RGB cases:

- 41 train, 15 validation, and 25 test cases;
- 21 natural crops from Astronaut, Grass, Gravel, Brick, Hubble, Page, and the
  independently authenticated NASA Hydra star field;
- 42 synthetic failure cases, 13 analytical success controls, and five
  coherence transitions;
- one impulse at every one of the 18 distinct X-Trans translation cells.

Model and scale selection use the train split only.  Hubble is validation;
Hydra and Page are test.  The native controls are Markesteijn,
corrected-final MLRI, zero-refinement MLRI (`ulri-slow0`), and high-refinement
MLRI (`ulri-slow3`).  Metrics exclude a 16-pixel boundary.

An important limitation is that the scikit-image Brick, Grass, Gravel, and
Page files are grayscale and are converted to equal-channel RGB.  They are
valuable coherent spatial-texture controls, but they are exactly rank-1 in RGB
by construction and do not establish performance on general chromatic texture.
Astronaut, Hubble, and Hydra provide the genuine full-color natural controls.

## Method

For each pixel and patch size 3, 5, 7, 11, and 15, the experiment computes the
RGB centroid and covariance, takes the dominant eigenvector, and records the
minimum and maximum local projections as bounded line endpoints.  The primary
oracle knows this ground-truth line but estimates the pixel coordinate from
only its actual scalar CFA observation.  The sampled channel is restored
exactly after reconstruction.

The experiment also tests:

- unbounded line coordinates;
- two-means endpoints on projected values;
- centroid, neighboring-coordinate, and spatial-plane handling when the
  sampled channel changes by less than `1e-4` across the endpoints;
- lines fitted from each native initializer at 3, 7, and 15 pixels;
- 7×7 leave-center-out fitting;
- a deterministic one-step Huber covariance reweighting;
- oracle corrected-MLRI/color-line gates with 3, 7, and 15-pixel support;
- modest synthetic noise at normalized sigma 0.001 and 0.005.

The selected true-line size and practical estimator are both chosen only from
training data: 3×3 and `ulri-slow0-line-3`, respectively.

## Main reconstruction results

PSNR in dB:

| Group | Markesteijn | Corrected MLRI | Low MLRI | High MLRI | True line 3×3 | Observable line | Huber line 7×7 | Oracle gate 7×7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| All 81 cases | 27.660 | 27.690 | 27.744 | 27.581 | 34.064 | 27.675 | 27.684 | 34.194 |
| Natural | 33.341 | 38.936 | 36.859 | 38.435 | 45.457 | 38.326 | 38.758 | 45.640 |
| Natural sparse | 41.390 | 36.715 | 38.654 | 36.028 | 43.617 | 38.444 | 38.025 | 43.665 |
| Natural coherent grayscale | 31.363 | 44.153 | 36.331 | 48.366 | effectively exact | 39.158 | 42.123 | effectively exact |
| Synthetic sparse | 27.540 | 25.425 | 26.134 | 25.088 | 43.985 | 25.859 | 26.144 | 43.985 |
| Edges and intersections | 25.539 | 25.087 | 26.146 | 25.019 | 35.750 | 26.031 | 26.003 | 40.009 |
| Pathological controls | 10.249 | 9.760 | 9.842 | 9.798 | 14.422 | 9.651 | 9.537 | 14.505 |

The true-line result improves every pooled class.  It reduces natural RMS from
0.01130 for corrected MLRI to 0.00534 and natural p99 absolute error from
0.03389 to 0.01643.  The oracle gate has only modest additional headroom because
it selects the true line for 93.4% of natural pixels at 3×3 support and 97.0% at
7×7 support.

The bounded coordinate is essential.  On natural data, the 7×7 bounded-extrema
oracle gives 41.087 dB, while its unbounded form collapses to 18.101 dB and a
70.04 normalized maximum error.  Projected two-means endpoints give only
33.759 dB.  Neighbor and plane degeneracy fallbacks are effectively tied at
41.0867 dB; centroid fallback is also nearly identical for natural data but is
worse on exact edges.

## Model validity and rank-2 control

At 3×3, the mean fraction of covariance energy outside the best line is:

| Source | Rank-1 residual ratio | Residual after rank-2 plane | Explained by line |
| --- | ---: | ---: | ---: |
| Astronaut | 0.0926 | 0.0197 | 0.9074 |
| Hubble | 0.1169 | 0.0216 | 0.8831 |
| Hydra | 0.0681 | 0.0083 | 0.9319 |
| Brick/Grass/Gravel/Page | numerical zero | numerical zero | 1.0000 |

The full-color sources are approximately, not exactly, two-color.  Rank 2
removes most of their remaining color-distribution error, showing that some
patches genuinely contain three-color variation.  The scalar CFA observation
cannot identify two rank-2 coordinates, so that control is not a demosaicer.

Larger patches improve the covariance ratio but worsen reconstruction because
the single line spans more structures.  For Hubble, mean rank-1 residual falls
from 0.1169 at 3×3 to 0.0761 at 15×15, yet oracle PSNR falls from 40.607 to
34.786 dB.  Hydra similarly falls from 80.960 to 76.363 dB.  Line fit quality
alone therefore does not select the right spatial scale.

## Sparse points and stars

The oracle preserves individual sparse colors rather than averaging them away.
At the brightest sampled points:

- Hubble RGB error falls from 0.0537 for Markesteijn and 0.1539 for corrected
  MLRI to 0.0277 for the true line.
- Hydra falls from 0.00822 and 0.02543 to 0.000131.  Its local line explains
  99.85% of covariance energy, and the bright endpoint closely matches the
  true star RGB.
- White, saturated red, green, and blue synthetic points are reconstructed to
  roughly `1e-8` RGB RMS when the true 3×3 line is supplied, even when the
  physically sampled channel is not the point's dominant color.

The observable line does not preserve these gains.  On Hubble it is 35.437 dB,
and on Hydra 66.233 dB, both well below Markesteijn.  Its diagnostic maps show
that it inherits the initializer's desaturation and point-color errors rather
than discovering the background/star endpoints independently.

## CFA identifiability and error decomposition

For the selected 3×3 true line, natural-data line-model RMS is 0.00237, while
the additional CFA-position RMS is 0.00483.  On Hubble/Hydra those values are
0.00288 and 0.00597.  Thus the dominant remaining oracle error is often not the
rank-1 approximation itself; it is locating the correct point on the line from
one sampled component.

The direct coordinate is degenerate for 1.55%, 1.67%, and 1.79% of natural
red-, green-, and blue-sampled pixels.  Sparse natural cases rise to 3.30%,
3.56%, and 3.39%.  Neighboring-coordinate fallback makes most such pixels
usable, but roughly one quarter of degenerate sparse positions still have no
usable local support.  Exact two-color edges have much higher degeneracy when
one channel is identical at both endpoints; this is expected and must be
handled explicitly rather than hidden with an epsilon.

The unconstrained CFA-only model is not identifiable.  A patch supplies `N`
scalar observations but has `N` independent alpha values plus two RGB
endpoints before gauge removal.  The tests construct two distinct bounded line
solutions that reproduce every observed CFA sample exactly.  Additional
spatial alpha structure or independently estimated endpoints are mandatory, so
no unconstrained nonlinear solve is presented as a demosaicer.

## Observable line estimation

Normal line fitting, leave-center-out fitting, and Huber covariance fitting all
remain initializer-dependent:

- On pooled natural data, low-MLRI line refinement improves its initializer by
  1.466 dB and recovers 33.2% of the oracle MSE gain, but still loses 0.610 dB
  to corrected MLRI.
- On natural sparse data it loses 0.210 dB to its own low-MLRI initializer.
- On coherent grayscale controls it improves low MLRI by 2.827 dB, but remains
  below corrected and high-refinement MLRI.
- The Huber 7×7 low-MLRI line reaches 38.758 dB on natural data but only
  38.025 dB on sparse natural data.  Robustly downweighting a few off-line
  initializer pixels does not recover the missing star colors.
- Leave-center-out fitting does not resolve the sparse failure either.

The selected estimator's line residual is not a useful applicability signal.
Spearman correlation is -0.032 with reconstruction error and -0.064 with
improvement.  Across residual-decile bins, the fraction improved stays in the
narrow and non-monotonic range 58.8%–63.6%.  Endpoint separation strongly
correlates with error (0.819), but mostly measures signal magnitude rather than
calibrated validity.  Because confidence is not calibrated, the conditional
practical gate was not built.

## Phase and noise

Scene color-line validity is essentially phase invariant: the mean rank-1
residual ratio across the 18 impulse placements varies by only `2.75e-9`.
Oracle reconstruction RMS nevertheless varies from `1.44e-9` to `3.55e-5`
because a scalar sample's ability to locate the point depends on which channel
is physically observed.  The huge 87.8 dB PSNR range is mostly the logarithmic
representation of these tiny absolute errors.

The observable estimator remains strongly phase-sensitive: its RMS spans
0.000649–0.05090, similar in character to corrected MLRI's
0.000294–0.05571.  Therefore the prior avoids the residual-alias trap only when
the correct line is known; the tested practical estimator does not.

With normalized sigma 0.001 noise, true-line oracle PSNR remains 40.63 dB on
Hubble, 59.71 dB on Hydra, and 59.42 dB on Brick.  At sigma 0.005 these fall to
38.97, 45.94, and 45.54 dB.  Direction estimates are particularly unstable in
Hydra's low-separation dark regions, confirming that a practical model needs an
explicit noise likelihood rather than raw PCA alone.

## Relation to prior art

[Bennett et al., *Video and Image Bayesian Demosaicing With A Two Color Image
Prior* (ECCV 2006)](https://www.microsoft.com/en-us/research/publication/video-and-image-bayesian-demosaicing-with-a-two-color-image-prior/),
DOI `10.1007/11744023_40`, is more than the per-pixel PCA oracle tested here.
It bootstraps full Bayer RGB, clusters a local neighborhood into two colors,
and estimates the blend with a product-of-Gaussians Bayesian likelihood over
nearby CFA samples.  Its gridless formulation is not intrinsically tied to one
sampling grid, although the published evaluation and details target Bayer.
No source was copied or ported.

[Zheng, Lin, and Yang, *Color filter array demosaicking with local color
distribution linearity*](https://www.microsoft.com/en-us/research/publication/color-filter-array-demosaicking-local-color-distribution-linearity/),
DOI `10.1117/1.1906084`, independently uses local color-distribution linearity
to reduce confetti and fringe artifacts.  It reinforces the relevance of the
prior but does not supply evidence that the line is directly identifiable from
X-Trans samples.

[Shakar, Li, and Randhawa, *Simultaneous CFA Demosaicking of Three Color Planes
for Improved Color Accuracy*](https://researchnow.flinders.edu.au/en/publications/simultaneous-cfa-demosaicking-of-three-color-planes-for-improved-),
DOI `10.17706/jcp.14.5.318-327`, uses the color-line property differently: it
generates directional RGB combinations and rejects artifact combinations.
That makes color-line consistency a candidate discriminator rather than the
complete reconstruction prior tested here.  Previous X-Trans candidate
selection failures make that distinction material; this experiment does not
claim to reproduce their Bayer algorithm.

## Answers to the required questions

- **Model validity:** full-color natural 3×3 patches are often strongly but not
  perfectly line-like (88.3%–93.2% mean explained variance on the sparse
  sources).  Grayscale coherent controls are exactly rank-1 by construction.
- **Sparse structures:** the true line preserves star color and intensity and
  substantially improves Hubble/Hydra.  An initializer-derived line does not.
- **CFA identifiability:** one component usually locates the point when the line
  is known, but position error exceeds line-model error and 3%–4% of sparse
  samples are directly degenerate.
- **Coherent texture:** the oracle preserves the grayscale texture controls and
  improves full-color Astronaut, but the corpus is insufficient to claim
  general chromatic-texture safety.
- **Oracle headroom:** large—6.52 dB over corrected MLRI on pooled natural data,
  2.23 dB over Markesteijn on sparse natural data, and 16.45 dB over
  Markesteijn on synthetic sparse data.
- **Estimation:** normal, leave-center-out, and Huber fits mostly reproduce the
  initializer's errors.  None recovers sparse-star oracle gains.
- **Confidence:** rank-1 residual/explained variance does not predict practical
  reconstruction error or improvement.
- **X-Trans specificity:** scene line validity is phase invariant, but scalar
  identifiability and practical line estimation remain CFA-phase sensitive.
- **Practical direction:** a color line contains genuinely new reconstruction
  information, but the tested observable estimators cannot access it.  A next
  experiment is justified only if it adds independent endpoint/alpha evidence,
  such as a faithful neighborhood Bayesian likelihood inspired by Bennett;
  more local PCA refinement is not justified.

## Artifacts and verification

Tracked artifacts are under `devnotes/images/xtrans-color-line/`.  The canonical
manifest SHA-256 is `14796c3322491cd0501c87371a0ad4e67439853058efa2d804e674055d410123`.
It binds the dataset/result JSON and six diagnostic maps.  The result JSON is
`102b983b76c0e390b8879ce7580fe0cd6e700a92bbca6fe38e6a1a5250e152ab`.

The focused test suite covers exact two-color reconstruction for all 18 phase
cells, rank-1/rank-2 ordering, robust-fit finiteness, explicit channel
degeneracy, native-sample preservation, spatial fallback, oracle gating, and a
constructive CFA-only non-uniqueness example.
