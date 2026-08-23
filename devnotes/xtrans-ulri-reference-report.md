# X-Trans Unified Laplacian / ARI reference experiment

## Decision

**NO-GO — do not implement or transplant “ULRI” as a new RawTherapee
demosaicer. GO — a separate ARI reference experiment is now the next justified
residual-interpolation test.**

The decisive finding is provenance, not merely quality: the official X-Trans
source in *Unified Laplacian Residual Interpolation Demosaicing* 1.0.0 is the
same implementation already used as the engineering reference for
RawTherapee's MLRI experiment. The previous development copy and the official
source have identical 1,305 nonblank source lines after line-ending and blank
formatting differences are removed. The existing C++ `MATLAB_REFERENCE` path
is therefore already the independent ULRI reproduction.

Varying the source's `slow` parameter reveals an exceptionally clear version
of the known conflict:

- all six Hubble/Hydra crops select one pass (`slow=0`);
- all twelve Brick/Grass/Gravel/Page crops select four passes (`slow=3`);
- the pooled coherent natural set gains **4.212 dB** over corrected-final at
  `slow=3`, while the sparse natural set loses **0.688 dB**;
- Markesteijn still beats every ULRI setting on the pooled sparse natural set.

Thus more residual refinement does not recover new trustworthy chromatic
information. It reinforces coherent texture and an aliased sparse explanation
with the same machinery. There is no distinct ULRI operation to port.

ARI is not implemented here. It merits a separate reference experiment because
it adds something this source does not: parallel original-RI and MLRI candidate
banks, per-pixel iteration selection, and criterion-weighted combination. That
is a genuine architectural test, although the earlier finding that residual
criteria can be confidently wrong on X-Trans aliases remains a serious warning.

No production method, PP3 identifier, GUI behavior, or default changed.

## Frozen baselines

The source baseline is commit
`67b55e73c398e5b8228a6685f2770e73bcc6fadf`. The experiment makes no tuning
change to either comparator:

- **Markesteijn:** the developer array entry point runs RawTherapee's
  three-pass/CIE-Lab path, corresponding to ordinary “3-pass (best),” before
  later pipeline processing.
- **Corrected-final MLRI:**
  `CORRECTED_BLUE_DIAGONAL_GUIDES_FINAL_ONLY`, with the four reviewed blue
  guide substitutions, two green passes at sigma 2 then 1, epsilon 0.01, and
  direct separately reconstructed final red/blue.

Both receive the same 6x6 CFA mosaic in RawTherapee's float 0..65535
camera-linear domain. Evaluation converts outputs to normalized linear RGB,
excludes 16 pixels on every boundary, and applies no gamma, color management,
sharpening, denoise, or false-color suppression.

## Reference identity, fidelity, and license

The evaluated artifact is the [MathWorks File Exchange version
1.0.0](https://www.mathworks.com/matlabcentral/fileexchange/182302-unified-laplacian-residual-interpolation-demosaicing),
published 2025-10-14. Its page describes a unified two-pass solution for 15
CFA layouts, recommends `slow=1` as N+1 processing, and acknowledges redundant,
slow X-Trans steps.

| Artifact | Identity |
| --- | --- |
| official archive | 8,214,511 bytes; SHA-256 `0dab03107479153a68fe6feeef738e3fb9c058031634dc006e251f2fd7a81347` |
| official `function_demosaic_x_trans.m` | SHA-256 `bc8a2a557a326af2ea72f7b77b17d55d6785570fa36197f732288a2d54024b69` |
| earlier supplied development copy | SHA-256 `055d1807729cbd556406bf695a6617f189d2376f7dbd35f4477f44e671fffa0c` |
| license | BSD-3-Clause; reviewed `license.txt` SHA-256 `8da7c29a607b5d6450bb2dbf7b5c0a7a93cbe5bb69a8d35d0fb4de5d3940a7c4` |
| official reference manifest | SHA-256 `2441cb88637a5c50375093d11e97b6cb46912d68201b338c3f8d4a5c4e7114c3` |
| 81-case artifact manifest | SHA-256 `4394b8f6b9c3cde13267c944c6d10701dfc4c0cc7b3a1aaef4159cbb7b107487` |

The BSD license permits source reuse with attribution and the required notices.
This experiment nevertheless copies no upstream MATLAB implementation. The
small tracked Octave driver authenticates and executes the external source;
the C++ implementation remains independently written.

The official source's floating-input branch uses a MATLAB spelling unsupported
by Octave 8 (`max(A,[],"all")`). The generator does not patch it. Instead it
quantizes the analytical mosaic once and calls the source's native `uint16`
branch, which converts to single precision in 0..255 and returns uint16 RGB.
This also matches RawTherapee's reviewed source-domain contract.

Two independent corpus generations were byte-identical. The tracked corpus
contains truth, scalar mosaic, and official output for eleven analytical scenes
at `slow=0..3`. Native parity over the complete 48x48 outputs is:

| Mode | Full max abs. | Full RMS | Interior max abs. | Interior RMS |
| --- | ---: | ---: | ---: | ---: |
| `slow=0` | 0.0007411 | 1.006e-5 | 0.0000695 | 3.564e-6 |
| `slow=1` | 0.0215238 | 2.089e-4 | 0.0002742 | 1.403e-5 |
| `slow=2` | 0.0317515 | 3.127e-4 | 0.0003002 | 1.734e-5 |
| `slow=3` | 0.0455194 | 4.100e-4 | 0.0005191 | 1.894e-5 |

The larger maxima are confined to the 16-pixel boundary region, where small
Octave/C++ reduction-order differences pass repeatedly through ill-conditioned
local fits. Interior agreement is tight. Differences also include the
official path's final uint16 quantization. The native contract independently
requires `slow=1` to be float-identical to RawTherapee's frozen
`MATLAB_REFERENCE` entry point.

The official source hardcodes one 6x6 X-Trans cell. It cannot accept a
translated phase directly. The native reference uses the existing canonical
coordinate convention and successfully exercises all 18 distinct X-Trans
translation cells without non-finite output or channel swapping.

## Exact `slow` contract

The name “normal” is ambiguous, so results use the source parameter directly:

| Setting | Green passes | Pass sigma | Final red/blue |
| --- | ---: | --- | --- |
| `slow=0` | 1 | 2 | direct separate green-guided R/B |
| `slow=1` (source default/recommended) | 2 | 2, 1 | `sqrt(G/255)` blend of provisional and separate R/B |
| `slow=2` | 3 | 2, 1, 1 | same blend |
| `slow=3` | 4 | 2, 1, 1, 1 | same blend |

There is no convergence test or adaptive stop. `slow=N>0` means exactly N+1
complete green passes. Every additional pass rebuilds all green-direction
guides/candidates from the current reconstruction and reuses the same fixed
MLRI regression, residual interpolation, and directional fusion rules.

## Operation-by-operation decomposition

The X-Trans source first reconstructs green while retaining provisional R/B,
then reconstructs R/B from the completed green guide.

1. Construct 18 phase masks: ten green, four red, and four blue positions.
2. On the first green pass, form phase-dependent horizontal, vertical,
   diagonal, and anti-diagonal guides for G, R, and B using sparse 3- to
   11-tap filters and cross completion.
3. Form approximate guide and observed-sample Laplacians with direction- and
   phase-specific 5-, 7-, and 13-tap derivative kernels.
4. In each 7x7 local window, fit an affine tentative estimate `a*guide+b`.
   The gain minimizes observed residual Laplacian energy; overlapping models
   are averaged with inverse residual-MSE weights floored by epsilon 0.01.
5. At physically observed target samples, form `observed - tentative`,
   interpolate the residual through a direction-specific half-Gaussian support,
   and add it to the tentative estimate.
6. Pair green-at-chroma and chroma-at-green regressions into eight directional
   G-C candidates. Fuse them with normalized inverse squared directional
   color-difference-gradient energy.
7. Complete provisional opposite-color values using horizontal, vertical, and
   diagonal residual interpolation.
8. For a later green pass, substitute the prior full green image for the first
   sparse-green guide construction, lower sigma to 1, and repeat steps 3-7.
9. Clip green, then separately reconstruct red and blue from the final green
   guide using the same affine MLRI/residual mechanism.
10. For `slow>0`, blend provisional and separate R/B by `sqrt(G/255)`; for
    `slow=0`, return the separate R/B directly. Clip final R/B.

All correlations use zero extension to preserve MATLAB `conv2(...,"same")`
semantics. No sample reinjection, gamma, denoise, sharpening, or false-color
suppression is part of the reference.

## Structural comparison

| Operation | Corrected-final X-Trans MLRI | Official ULRI X-Trans | Full ARI literature |
| --- | --- | --- | --- |
| CFA | same 18 phase masks, generalized through canonical coordinates | one hardcoded X-Trans phase cell | published Bayer and multispectral layouts, not X-Trans |
| Initial guides | phase-specific G/R/B H/V/diagonal filters | identical | directional guide initialized from observed/interpolated bands |
| Tentative model | local affine `a*guide+b` | identical | separate original-RI and MLRI tentative models |
| Residual | observed minus tentative, interpolated and added back | identical | same RI framework for every branch/iteration |
| Laplacian role | MLRI gain fitted from approximate Laplacians | identical | used only by the MLRI branch; original-RI branch fits values |
| Model overlap | inverse residual-MSE weighting | identical | branch-specific iterative construction |
| Direction bank | eight X-Trans H/V/diagonal candidates | identical | published Bayer method uses H/V for green and diagonal/HV stages for R/B |
| Direction fusion | inverse squared directional energy | identical | criterion-weighted RI/MLRI and direction combination |
| Pass refinement | fixed two passes, sigma 2 then 1 | fixed 1..N+1 passes selected globally | candidate iterations generated up to a fixed maximum, best iteration selected per pixel |
| Blue diagonal guides | four apparent source copy/paste uses of red guides are corrected to blue | retains the four red-guide expressions | not applicable to the published Bayer construction |
| Final R/B | direct separately reconstructed R/B | `slow>0` uses square-root provisional/final blend | ARI applied to green and later R/B interpolation |
| Stop/selection rule | none | none | residual-magnitude/smoothness criterion selects iteration and weights branches per pixel |

Consequently, H1 is rejected. ULRI introduces no new information or
constraint relative to the current implementation. The only mathematical
experimental variable here is fixed pass count. The blue-guide substitutions
and final direct selection are controlled implementation corrections in
corrected-final, not a competing RI architecture.

## Corpus and comparison domain

The experiment reuses the exact 81-case normalized, camera-linear corpus from
the previous MLRI work:

- 21 natural crops from Astronaut, Brick, Grass, Gravel, Hubble, Page, and the
  independent NASA Hydra star field;
- 42 synthetic sparse/failure cases covering all 18 CFA-relative impulses,
  colored dots, stars, thin lines, intersections, and density/thickness sweeps;
- 13 synthetic coherent/control cases;
- five untouched sparse-to-coherent transition cases.

The same CFA mosaic feeds every method. Metrics exclude a 16-pixel border and
are pooled over actual pixels rather than averaging per-image PSNR. The output
domain is normalized linear RGB. Sampled/interpolated-channel metrics, R/G/B,
orthonormal L/C1/C2, percentiles, and maxima are all retained in
[results.json](images/xtrans-ulri/results.json).

Pooled over all 81 cases:

| Method | PSNR | RGB RMS | R RMS | G RMS | B RMS | missing-G RMS | missing-R/B RMS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Markesteijn | 27.660 | 0.041398 | 0.051891 | 0.028295 | 0.040598 | 0.042467 | 0.052817 |
| corrected-final | 27.690 | 0.041258 | 0.051699 | 0.032144 | 0.037423 | 0.048244 | 0.051164 |
| ULRI `slow=0` | 27.744 | 0.041003 | 0.052473 | 0.030463 | 0.036909 | 0.045721 | 0.051429 |
| ULRI `slow=1` | **27.772** | **0.040868** | 0.050754 | 0.032130 | 0.037448 | 0.048223 | **0.050563** |
| ULRI `slow=2` | 27.655 | 0.041425 | 0.051142 | 0.033180 | 0.037835 | 0.049799 | 0.050998 |
| ULRI `slow=3` | 27.581 | 0.041777 | 0.051347 | 0.033740 | 0.038222 | 0.050640 | 0.051315 |

| Method | L RMS | C1 RMS | C2 RMS | p90 abs. | p95 abs. | p99 abs. | max abs. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Markesteijn | **0.047871** | 0.039211 | **0.036226** | 0.020088 | 0.037854 | 0.120439 | 1.426584 |
| corrected-final | 0.049401 | 0.036207 | 0.036813 | 0.007751 | 0.013546 | **0.053404** | 1.000000 |
| ULRI `slow=0` | 0.048626 | 0.035991 | 0.037201 | 0.013062 | 0.023370 | 0.078450 | 1.000000 |
| ULRI `slow=1` | 0.049080 | **0.035653** | 0.036477 | 0.007614 | 0.013391 | 0.054275 | 1.000000 |
| ULRI `slow=2` | 0.049903 | 0.036261 | 0.036644 | 0.006311 | 0.011342 | 0.056119 | 1.000000 |
| ULRI `slow=3` | 0.050441 | 0.036665 | 0.036706 | **0.005806** | **0.010638** | 0.059059 | 1.000000 |

Lower p90/p95 with more passes coexists with worse RMS, p99, and sparse-source
quality. The iteration smooths many ordinary residuals while increasing rarer,
larger errors; those tail changes are precisely why aggregate PSNR alone is
insufficient.

## Natural-source results

PSNR in dB; higher is better.

| Source | Markesteijn | corrected-final | ULRI `slow=0` | ULRI `slow=1` | ULRI `slow=2` | ULRI `slow=3` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Astronaut | 37.555 | 35.023 | **36.217** | 34.738 | 34.041 | 33.504 |
| Brick | 46.974 | 59.392 | 52.374 | 59.545 | 62.589 | **64.308** |
| Grass | 30.189 | 42.463 | 35.791 | 42.581 | 44.661 | **45.501** |
| Gravel | 35.136 | 46.547 | 39.938 | 46.816 | 49.181 | **50.185** |
| Hubble | **38.382** | 33.710 | 35.648 | 33.709 | 33.251 | 33.022 |
| Hydra | **71.790** | 63.386 | 66.213 | 63.034 | 62.544 | 62.347 |
| Page | 27.855 | 41.325 | 32.516 | 41.576 | 45.256 | **46.958** |

The source-default `slow=1` is 0.069 dB below corrected-final over all natural
crops. It is not a new aggregate improvement. `slow=3` gains greatly on every
coherent source but loses on Astronaut and both star fields.

### Sparse versus coherent pooled results

| Group and method | PSNR | Delta from corrected-final | RGB RMS | p99 abs. | Maximum abs. |
| --- | ---: | ---: | ---: | ---: | ---: |
| natural sparse — Markesteijn | **41.390** | +4.675 | 0.008521 | 0.017918 | 0.577634 |
| natural sparse — corrected-final | 36.715 | — | 0.014596 | 0.028338 | 0.846309 |
| natural sparse — ULRI `slow=0` | 38.654 | +1.939 | 0.011676 | 0.021735 | 0.785914 |
| natural sparse — ULRI `slow=1` | 36.715 | -0.001 | 0.014597 | 0.028852 | 0.848861 |
| natural sparse — ULRI `slow=3` | 36.028 | -0.688 | 0.015799 | 0.032384 | 0.875150 |
| natural coherent — Markesteijn | 31.363 | -12.790 | 0.027030 | 0.114508 | 0.685136 |
| natural coherent — corrected-final | 44.153 | — | 0.006199 | 0.024965 | 0.140019 |
| natural coherent — ULRI `slow=0` | 36.331 | -7.823 | 0.015257 | 0.063901 | 0.394581 |
| natural coherent — ULRI `slow=1` | 44.356 | +0.203 | 0.006056 | 0.024324 | 0.146820 |
| natural coherent — ULRI `slow=3` | **48.366** | +4.212 | 0.003817 | 0.014933 | 0.041295 |

This rejects H2. One pass improves sparse safety only by moving toward the
less-refined result, while additional passes improve coherent structure and
progressively damage star color/intensity. ULRI does not improve Hubble/Hydra
while retaining MLRI's texture gains.

## Synthetic controls and transition behavior

Selected PSNR results:

| Case | Markesteijn | corrected-final | `slow=0` | `slow=1` | `slow=2` | `slow=3` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| white impulse | **40.036** | 34.797 | 34.778 | 34.796 | 34.796 | 34.796 |
| two-pixel red dot | 33.951 | 35.099 | **35.107** | 34.163 | 34.175 | 34.176 |
| thin red line | **22.773** | 20.567 | 21.394 | 20.489 | 20.494 | 20.491 |
| line intersection | **20.483** | 19.074 | 20.382 | 19.813 | 19.275 | 19.015 |
| high-frequency monochrome | 27.028 | 48.952 | 38.816 | 50.827 | 59.756 | **66.950** |
| periodic chromatic texture | **10.184** | 9.482 | 9.337 | 9.513 | 9.521 | 9.528 |
| frequency sweep | **7.293** | 6.897 | 7.112 | 6.970 | 6.935 | 6.932 |

The controlled transitions do not show a clean monotonic point-to-texture
switch. For example, the point-density progression usually prefers `slow=0`,
but its three-point case prefers `slow=1`; a five-pixel line prefers `slow=1`
while 1-, 2-, and 3-pixel lines prefer `slow=0`; continuous lines and dotted
lines can move in opposite directions. The reconstruction has no stable
coherence threshold.

The qualitative maps show the same mechanism. More passes progressively erase
or recolor sparse stellar detail while reducing structured brick/page error.
The third row is signed `ULRI - corrected-final`, with gray as zero:

- [Hubble](images/xtrans-ulri/map-hubble.png)
- [Hydra](images/xtrans-ulri/map-hydra.png)
- [Brick](images/xtrans-ulri/map-brick.png)
- [Page](images/xtrans-ulri/map-page.png)
- [Impulse](images/xtrans-ulri/map-impulse.png)
- [Continuous line](images/xtrans-ulri/map-continuous-line.png)
- [Saturated edge](images/xtrans-ulri/map-saturated-edge.png)

The prior frequency/alias diagnosis does not reveal a new suppression rule.
The frequency sweep is already pathological at 7.112 dB for `slow=0` and
declines to 6.932 dB at `slow=3`; periodic chromatic texture stays near 9.5 dB.
The point/star maps retain phase-local chromatic errors rather than converting
them into correctly reconstructed chroma. Because the residual and candidate
families are identical to current MLRI, these observations are consistent with
reinforcement of the same phase-sensitive alias explanation, not evidence for
a distinct ULRI alias rejection mechanism.

## Pass-count oracle

H3 is confirmed only as a conflict, not a solution. Among ULRI `slow=0..3`:

- the best whole-image pass count selects `slow=0` for 42 cases, `slow=1` for
  6, `slow=2` for 7, and `slow=3` for 26;
- all 6 natural sparse cases select `slow=0`;
- all 12 natural coherent cases select `slow=3`;
- the whole-image oracle reaches 28.090 dB versus 27.772 dB for the best fixed
  mode (`slow=1`), a 0.318 dB upper-bound gain;
- a 7x7 ground-truth oracle reaches 28.139 dB, only another 0.050 dB beyond the
  whole-image oracle.

The oracle confirms headroom in selecting “little refinement” versus “much
refinement,” but supplies no observable selector. The prior local-selection
experiments already found that regression residual, stability, candidate
spread, direction energy, and related local evidence cannot make this choice
safely on X-Trans aliases. No new selector is proposed here.

## Runtime

Median native untiled research-runner time over the 81 small cases:

| Method | Median seconds | Sum over corpus |
| --- | ---: | ---: |
| Markesteijn | 0.00277 | 0.295 |
| corrected-final | 0.19759 | 35.082 |
| ULRI `slow=0` | 0.10794 | 19.194 |
| ULRI `slow=1` | 0.19816 | 35.300 |
| ULRI `slow=2` | 0.29147 | 51.582 |
| ULRI `slow=3` | 0.38343 | 67.653 |

On the 48x48 diagonal reference case, separate Octave invocations took about
0.30, 0.58, 0.74, and 0.89 seconds for `slow=0..3`, including Octave startup
and output I/O. These numbers are reference overhead, not a production
benchmark. Native time grows approximately with pass count, as the source
contract predicts.

## Hypotheses and structural-divergence result

| Hypothesis | Result |
| --- | --- |
| H1 — materially different architecture | **Rejected.** It is the exact external reference behind the existing implementation. |
| H2 — changes sparse/coherent tradeoff safely | **Rejected.** Fewer passes help sparse cases; more passes help coherent cases. |
| H3 — iteration matters | **Confirmed, negatively.** Pass count strongly controls which side of the conflict wins. |
| H4 — a specific ULRI operation explains a new gain | **Rejected.** There is no new operation; only pass count, source copy/paste expressions, and final blend differ. |

Because the pipelines are structurally identical, internal divergence and
cross-substitution experiments are unnecessary and would be artificial. The
first meaningful divergences are already isolated:

- `slow=0` stops after the first green pass and omits the final blend;
- `slow=1` is the existing source-faithful two-pass path;
- `slow=2/3` append identical sigma-1 refinements;
- corrected-final changes four blue diagonal guide expressions and returns the
  separate final R/B without the square-root blend.

No different residual, guide, Laplacian, regression, or directional-fusion
stage exists to transplant. H4 therefore cannot trigger a smaller production
change.

## Porting and ARI decision

The ULRI C++ port trigger is not met:

- Trigger A fails because sparse stars do not improve without losing coherent
  texture.
- Trigger B fails because source-default ULRI is slightly worse than
  corrected-final on pooled natural data.
- Trigger C fails because no distinct beneficial operation exists.

In practical terms the “port” already exists, hidden, and extending its pass
count would only expose a globally selected quality tradeoff. That is not a
new demosaicer and should not enter production or the GUI.

The [ARI paper](https://doi.org/10.3390/s17122787) supplies a concrete reason
for one separate experiment. Unlike this fixed-pass X-Trans source, published
ARI generates iterative directional outputs for both original RI and MLRI,
selects the best iteration per pixel using a residual magnitude/smoothness
criterion, and combines the selected branches/directions by criterion-derived
weights. It applies adaptation to green and later red/blue reconstruction.

That is genuinely new candidate information and directly targets the measured
pass-dependent conflict. But it is not evidence that ARI will work on X-Trans:
the published construction assumes Bayer/multispectral arrangements, and our
prior experiments show that low/smooth residuals can validate a wrong alias.
The next step should therefore be **ARI reference reproduction and an
X-Trans-specific feasibility design on this same corpus**, not an immediate
RawTherapee port and not another heuristic selector around current MLRI.

## Answers to the required questions

- **Reference fidelity:** yes for the evaluation purpose. Version 1.0.0,
  BSD-3-Clause, is authenticated and reproducible; interior native agreement is
  within max `5.20e-4` and RMS `1.90e-5`, and the existing source-faithful C++
  path is locked by a float-identical contract.
- **Structural difference:** no new architecture exists. Corrected-final fixes
  four blue guides and removes the source's final blend; ULRI otherwise is the
  same X-Trans MLRI pipeline.
- **Natural quality:** source-default ULRI does not beat corrected-final
  overall. More passes win strongly on Brick, Grass, Gravel, and Page but lose
  on Astronaut, Hubble, and Hydra.
- **Sparse safety:** Hubble and Hydra prefer one pass; Markesteijn remains best.
  Impulses, thin colored lines, and intersections show the same risk.
- **Coherent texture:** every Brick/Grass/Gravel/Page crop prefers four passes.
- **Iteration:** N+1 meaningfully changes quality, but in opposing directions;
  it does not resolve the conflict.
- **Failure mechanism:** there is no new ULRI divergence stage. Repeated use of
  the same locally self-consistent residual model increasingly reinforces
  coherent structure and the wrong sparse alias alike.
- **Porting value:** none as a new algorithm or isolated transplant.
- **ARI:** yes, a separate reference experiment is justified because its
  RI/MLRI bank and per-pixel iteration adaptation are genuinely absent here.

Most importantly, changing the label to ULRI did not change the residual-
interpolation architecture. Changing only its fixed iteration count does not
solve any part of the sparse-alias/coherent-texture conflict.
