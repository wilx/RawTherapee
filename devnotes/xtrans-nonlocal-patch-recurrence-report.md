# X-Trans nonlocal patch-recurrence upper-bound experiment

## Outcome

**NO-GO for the tested direct patch-recurrence route.**  Observable CFA
distance predicts full-RGB patch distance remarkably well, but the recurring
patches are not accurate enough to improve the local demosaicers.  This is a
different negative result from the local hybrid study: matching is not the
main bottleneck here; donor fidelity is.

On the seven ordinary natural analysis sources, the frozen local blender
reached **36.322 dB**.  A magically RGB-selected best same-phase donor reached
only **24.783 dB**, and the CFA-selected best donor reached **24.729 dB**.
The RGB oracle therefore has **-11.538 dB**, not positive, headroom relative
to the local blender.  Direct cross-phase NLM reached **23.522 dB** and the
rank-4 completion control reached **23.495 dB**.

The result closes this bounded direct-donor/low-rank feasibility experiment.
It does not prove that the full iterative regularizer of Chang, Ding, and Li
cannot work: that method combines nonlocal groups with an inter-channel
gradient model and a global inverse-problem iteration, none of which is
implemented here.

## Literature boundary

Buades, Coll, Morel, and Sbert's 2007 preprint
[*Non local demosaicing*](https://www.researchgate.net/publication/228912399_Non_local_demosaicing)
motivates using image self-similarity when local geometry is ambiguous and
constructing estimates as weighted averages of similar CFA observations.

Chang, Ding, and Li,
[*Color image demosaicking using inter-channel correlation and nonlocal self-similarity*](https://doi.org/10.1016/j.image.2015.10.003),
Signal Processing: Image Communication 39 (2015), 264–279, combines
inter-channel gradient correlation with nonlocal low-rank regularization.  Its
sensing-matrix formulation can express arbitrary CFAs.  The present work is
an X-Trans information test, not an implementation of either complete paper.

## Frozen protocol

- 275 target patches: 25 per source over 11 sources.
- 175 ordinary natural analysis patches: Astronaut, Brick, Chelsea, Grass,
  Gravel, Page, and Rocket.
- 50 untouched sparse-point safety patches: Hubble Deep Field and the
  independently sourced NASA Hydra star field.
- 50 analytical patches: repeated saturated chromatic edges and a chromatic
  frequency sweep.
- Linear-light RGB truth, a globally phased canonical 6x6 X-Trans mosaic,
  7x7 patches, 24-pixel target spacing, and a 72-pixel search radius.
- Every target is compared with Markesteijn three-pass, corrected-final MLRI,
  and the frozen 7x7 practical blender from the preceding study.
- Same-phase distance is MSE over all 49 physically observed, color-aligned
  samples.  Cross-phase distance uses only corresponding offsets at which the
  two patches physically observe the same color, requiring at least eight.
- Neighbor counts 1, 8, and 32 are reported for full-RGB donor upper bounds.
  Ground truth selects the oracle groups; observable groups use CFA distance.
- The implementable controls use 32 cross-phase neighbors: direct observed
  donor averaging and ten iterations of rank-4 SVD projection.  Original CFA
  samples are reinserted after every reconstruction.
- Hubble and Hydra influence no parameter or method choice.

The search is nonlocal relative to the 7x7 reconstruction footprint but is
bounded to a 145x145 window.  It does not make a whole-image recurrence claim.

## Same-phase identifiability

Exact-phase matching is valid for testing recurrence and neighbor purity, but
it cannot by itself reconstruct the missing values.  Every patch in the group
has the same CFA mask.  Of the 147 RGB rows in a 7x7 patch, only 49 are observed
in every column and the same 98 rows are absent in every column.  Consequently:

- direct corresponding-pixel donation has zero missing-color coverage;
- ordinary matrix completion has no measurements for those 98 rows;
- a usable same-phase demosaicer needs an additional color/residual prior.

Cross-phase matching supplies that missing coverage.  The tested candidate
sets contain all 18 translated X-Trans phase patterns, and direct NLM attains
complete donor coverage.

## Neighbor purity

On ordinary natural analysis patches:

| Quantity | Result |
| --- | ---: |
| Mean Spearman correlation, CFA distance vs RGB distance | 0.9938 |
| Median RGB RMS of best CFA-selected same-phase neighbor | 0.04430 |
| Median RGB RMS of best RGB-oracle same-phase neighbor | 0.04430 |
| Median selected/oracle MSE ratio | 1.000 |
| Patches with selected-neighbor RGB RMS below 0.02 | 25.71% |
| Patches with selected-neighbor RGB RMS below 0.05 | 54.86% |

Thus the CFA measurements generally rank candidate patches correctly.  The
absolute error of even the correct neighbor, however, is much larger than the
remaining local demosaicing error.

## Reconstruction results

Pooled PSNR over the 175 ordinary natural patches:

| Reconstruction | PSNR (dB) | Patches beating Markesteijn |
| --- | ---: | ---: |
| Markesteijn three-pass | 31.108 | baseline |
| Corrected-final MLRI | 36.774 | 78.3% |
| Frozen practical blender | 36.322 | 88.6% |
| CFA-selected same-phase best donor upper bound | 24.729 | 1.7% |
| RGB-oracle same-phase best donor upper bound | 24.783 | 1.7% |
| CFA-selected cross-phase direct NLM | 23.522 | 0.6% |
| CFA-selected cross-phase rank-4 completion | 23.495 | 0.6% |
| RGB-oracle-neighbor cross-phase rank-4 completion | 27.051 | 4.0% |

The most favorable implementable control is direct NLM, but it is **12.800
dB worse** than the local blender.  Since the best full-RGB donor oracle is
already 11.538 dB worse, changing NLM weights or the SVD rank cannot recover a
missing positive upper bound.

### Per-source PSNR

| Source | Markesteijn | MLRI | Blender | CFA best donor | RGB best donor | CFA NLM | CFA low-rank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Astronaut | 37.93 | 37.34 | 39.51 | 27.94 | 28.02 | 25.34 | 24.18 |
| Brick | 44.50 | 57.15 | 52.70 | 32.39 | 32.39 | 31.95 | 32.12 |
| Chelsea | 35.94 | 36.51 | 37.32 | 28.92 | 29.40 | 25.87 | 28.40 |
| Grass | 28.97 | 41.95 | 41.56 | 21.46 | 21.46 | 20.76 | 20.68 |
| Gravel | 35.08 | 46.17 | 45.50 | 23.98 | 23.98 | 23.38 | 23.26 |
| Page | 27.33 | 41.73 | 41.70 | 21.82 | 21.82 | 20.58 | 20.34 |
| Rocket | 28.11 | 30.43 | 29.34 | 26.26 | 26.50 | 24.90 | 25.39 |

Brick and Page do contain discoverable recurrence, but even their oracle
donors are far behind MLRI and the blender.  Natural recurrence is approximate
at a scale too coarse for the remaining demosaicing error.

## Safety and analytical controls

On Hubble and Hydra together:

| Reconstruction | PSNR (dB) |
| --- | ---: |
| Markesteijn | 41.768 |
| Frozen blender | 39.264 |
| CFA-selected direct NLM | 33.457 |
| CFA-selected low-rank | 33.694 |
| RGB-oracle-neighbor low-rank | 37.630 |

The selected direct NLM has **6.78x Markesteijn's MSE**.  Dark backgrounds are
easy to match and make neighbor-purity statistics look excellent, but averaging
does not preserve each star's individual intensity and color.  This is exactly
the sparse-point failure the safety sources were intended to reveal.

The two analytical scenes show the opposite result: pooled local-blender PSNR
is 10.161 dB, direct NLM is 25.509 dB, and low-rank completion is 25.938 dB.
Repeated saturated stripes provide exact donor structure, while the frequency
sweep is pathological for the local methods.  These large synthetic gains are
real, but they must not select a natural-image method; doing so would hide the
12.8 dB natural regression.

## Decision criteria

| Criterion | Result |
| --- | --- |
| RGB-oracle same-phase donor gains at least 0.5 dB on natural data | **FAIL** (-11.538 dB) |
| Observable selection recovers at least 30% of positive oracle reduction | **FAIL** (there is no positive reduction) |
| At least 25% of repeated-structure patches have neighbor RMS below 0.05 | PASS |
| Implementable cross-phase method gains at least 0.1 dB on natural data | **FAIL** (-12.800 dB) |
| Sparse-point MSE no more than 10% above Markesteijn | **FAIL** (6.78x) |

The useful conclusion is narrower than “nonlocal methods never work.”  Direct
patch donation and generic low-rank completion do not provide a viable X-Trans
demosaicer here.  Pursuing the full Chang et al. formulation would be a new,
substantially larger algorithmic project whose possible benefit would come from
its inter-channel/global regularizers, not from demonstrated donor headroom in
this experiment.

## Reproduction and artifacts

Commands are documented in
[`tools/xtrans_nonlocal/README.md`](../tools/xtrans_nonlocal/README.md).
Canonical results and diagnostic images are in
[`devnotes/images/xtrans-nonlocal/`](images/xtrans-nonlocal/).

- Results SHA-256: `93c63c2760d3206fe22214a6c47ccce6c12170879b1f6bc48b83f01e516ae41f`
- Dataset SHA-256: `0a767b88db4ca0c9e7be8b6b58c27efc30fde85a24db02d05e8817531e333a27`
- Manifest SHA-256: `595b2258cce19f9cdc7415bb068b2cdda2b07243bf92118b6c1b25c19f4255c2`

The external NASA TIFF, source images, native float buffers, and temporary
reconstructions are not committed.
