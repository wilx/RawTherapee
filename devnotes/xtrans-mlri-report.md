# Experimental X-Trans MLRI report

## Status

The developer-only `mlri-xtrans-2pass` method is implemented as a readable,
algorithmic X-Trans demosaicer. It is absent from the GUI and method enum;
Markesteijn three-pass remains the default and the complete fallback. This
report evaluates the independent C++ implementation against the authenticated
Octave reference, analytical ground truth, and `DSCF0771.RAF`.

The implementation is an X-Trans engineering generalization of residual
interpolation, not an X-Trans algorithm published by Kiku, Monno, Tanaka, or
Okutomi. The exact provenance and stage classification are recorded in
`xtrans-mlri-design.md`.

## Reference identity and numerical comparison

| Item | Identity |
| --- | --- |
| Development MATLAB source | `function_demosaic_x_trans.m` |
| Source SHA-256 | `055d1807729cbd556406bf695a6617f189d2376f7dbd35f4477f44e671fffa0c` |
| Published implementation | Unified Laplacian Residual Interpolation Demosaicing 1.0.0, 2025-10-14 |
| License | BSD-3-Clause |
| Octave corpus format | `rawtherapee-xtrans-mlri-golden-v1` |
| Corpus-manifest SHA-256 | `b39e6200a049a727a014faf560e8fe5bf94860991eb23ac63cc0c2699083e161` |

Two clean Octave regenerations were byte-identical to each other and to the
tracked five-case corpus. The C++ and Octave implementations use float32, but
their convolution reductions are evaluated in different orders. MLRI's
inverse residual-cost weighting magnifies those round-off differences in
pathological saturated and boundary inputs. The native comparison therefore
records explicit full-image and interior bounds instead of claiming bit parity.

| Case | Maximum absolute difference | RMS | Interior maximum | Interior RMS |
| --- | ---: | ---: | ---: | ---: |
| Flat | 69.887 | 1.929 | 0.340 | 0.045 |
| Gradient | 301.395 | 14.295 | 44.073 | 4.404 |
| Impulse | 14.854 | 0.473 | 9.315 | 0.394 |
| Saturated | 1,963.710 | 86.464 | 736.766 | 27.839 |
| Boundary | 642.113 | 68.208 | 354.420 | 65.297 |

Values are 16-bit code values. The worst single saturated-case difference is
about 3.0% of the normalized range; full-image RMS remains about 0.13%. This is
a bounded numerical comparison, not exact reference parity. The structure,
kernels, masks, clipping points, pass order, and the two documented apparent
source quirks were compared line by line.

## Native contract

The production path canonicalizes all 18 unique X-Trans phase/orientation
matrices, processes 384x384 output cores with a conservative 228-pixel halo,
uses global zero extension, and runs at most two OpenMP tile workers. A fixed
reduction order prevents tile-origin-dependent seams. Native tests require the
tiled result to remain within 0.02 16-bit code values of the untiled path at
and around a production boundary.

An interior 840x840 patch has a conservative workspace estimate of
508,032,000 bytes per worker. Two workers therefore have a reported ceiling of
1,016,064,000 bytes, excluding RawTherapee's ordinary image buffers. This
initial implementation deliberately prioritizes reviewability over speed.

Unsupported CFA layouts, sub-32-pixel dimensions, allocation failures, and
non-finite values return structured failures. The engine emits a loud marker
and runs Markesteijn three-pass over the complete output; no partially
reconstructed result is accepted by the caller.

## Analytical ground truth

The actual RawTherapee CLI mosaicked and exported nineteen deterministic 96x96
RGB scenes with unrelated processing and false-colour suppression disabled.
For every row the benchmark retains a float32 absolute-error TIFF and an
eight-times-amplified PNG error map in its external work directory. No row
fell back. Black is reconstructed exactly by both methods. CPSNR
removes a per-channel constant offset, so ordinary PSNR is included where the
two metrics lead to different conclusions.

| Scene | MLRI PSNR | Markesteijn PSNR | MLRI CPSNR | Markesteijn CPSNR |
| --- | ---: | ---: | ---: | ---: |
| Constant | 78.83 | 78.83 | 88.34 | 88.34 |
| Gradient | 69.48 | 71.21 | 70.72 | 71.85 |
| Impulses | 42.16 | 43.78 | 48.63 | 43.96 |
| Frequency sweep | 9.21 | 9.56 | 9.51 | 10.08 |
| Saturated edges | 42.45 | 31.64 | 42.82 | 62.42 |
| Asymmetric orientation | 22.35 | 21.26 | 23.13 | 21.31 |
| One-pixel lines | 29.47 | 19.61 | 29.63 | 20.24 |
| Diagonal lines | 28.29 | 26.35 | 28.32 | 27.16 |
| Concentric circles | 29.44 | 20.66 | 29.50 | 20.82 |
| Zone plate | 38.81 | 30.48 | 38.83 | 30.49 |
| Sinusoidal grating | 36.79 | 21.81 | 37.26 | 21.99 |
| Fine checkerboard | 18.77 | 81.42 | 24.53 | 89.19 |
| Red/green transition | 59.85 | 69.64 | 66.61 | 97.56 |
| Blue/green transition | 55.77 | 69.93 | 57.85 | undefined |
| Near-Nyquist achromatic | 48.03 | 80.98 | 48.20 | 88.40 |
| Near-Nyquist chromatic | 11.21 | 8.73 | 12.77 | undefined |
| Black/white text-like | 30.22 | 15.62 | 30.26 | 15.79 |
| Colored text-like | 12.02 | 12.27 | 12.47 | 12.72 |

The methods fail differently. MLRI strongly favors sparse, coherent geometric
detail: one-pixel lines, diagonals, rings, the zone plate, the sinusoidal
grating, and achromatic text-like edges. Markesteijn is dramatically stronger
on the one-pixel checkerboard and near-Nyquist achromatic grating, and it
handles hard red/green and blue/green transitions more faithfully. Both are
poor on the frequency sweep and near-Nyquist chromatic pattern. The divergent
saturated-edge PSNR/CPSNR result indicates a different channel offset or color
bias rather than an unqualified MLRI win.

These are deliberately adversarial signals, not a natural-image ranking. They
do establish the complementary failure modes that could justify a later
analytical reliability selector or ARI-style extension. No constants were
tuned against these scenes or the real RAF.

### Controlled blue-diagonal-guide correction

The authenticated MATLAB source uses red diagonal guides while fitting blue
samples in four expressions: diagonal and anti-diagonal blue Laplacians in the
green and chroma-reconstruction stages. The separate hidden method
`mlri-xtrans-2pass-corrected` substitutes the corresponding blue guides in
exactly those four expressions. It changes no mask, kernel, parameter, pass,
boundary rule, or postprocessing operation. The faithful method remains
available unchanged.

The same nineteen-scene benchmark was repeated for the corrected variant. The
table shows the complete faithful/corrected comparison; `Delta` is corrected
minus faithful CPSNR.

| Scene | Faithful PSNR | Corrected PSNR | Faithful CPSNR | Corrected CPSNR | Delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| Constant | 78.83 | 78.83 | 88.34 | 88.34 | +0.00 |
| Gradient | 69.48 | 69.65 | 70.72 | 70.78 | +0.06 |
| Impulses | 42.16 | 42.16 | 48.63 | 48.63 | +0.00 |
| Frequency sweep | 9.21 | 9.18 | 9.51 | 9.48 | -0.03 |
| Saturated edges | 42.45 | 42.49 | 42.82 | 42.93 | +0.11 |
| Black | exact | exact | undefined | undefined | n/a |
| Asymmetric orientation | 22.35 | 22.36 | 23.13 | 23.15 | +0.02 |
| One-pixel lines | 29.47 | 29.78 | 29.63 | 29.90 | +0.27 |
| Diagonal lines | 28.29 | 28.55 | 28.32 | 28.59 | +0.26 |
| Concentric circles | 29.44 | 29.78 | 29.50 | 29.86 | +0.35 |
| Zone plate | 38.81 | 39.28 | 38.83 | 39.30 | +0.47 |
| Sinusoidal grating | 36.79 | 35.75 | 37.26 | 35.90 | -1.36 |
| Fine checkerboard | 18.77 | 15.33 | 24.53 | 25.69 | +1.17 |
| Red/green transition | 59.85 | 61.57 | 66.61 | 67.93 | +1.32 |
| Blue/green transition | 55.77 | 59.97 | 57.85 | 63.41 | +5.57 |
| Near-Nyquist achromatic | 48.03 | 48.50 | 48.20 | 48.69 | +0.49 |
| Near-Nyquist chromatic | 11.21 | 10.40 | 12.77 | 16.63 | +3.86 |
| Black/white text-like | 30.22 | 30.12 | 30.26 | 30.16 | -0.09 |
| Colored text-like | 12.02 | 11.99 | 12.47 | 12.44 | -0.03 |

The correction helps the explicitly directional cases: diagonal lines gain
0.26 dB CPSNR, concentric circles 0.35 dB, the zone plate 0.47 dB, and the
blue/green transition 5.57 dB. It is not a universal improvement. The
sinusoidal grating loses 1.36 dB CPSNR; checkerboard PSNR loses 3.44 dB despite
its CPSNR increase; and near-Nyquist chromatic PSNR and SSIM both regress.
These mixed results are why this is a separate method rather than a silent
change to the source-compatible method.

## Controlled 2014/2016 paper-core comparison

The equation audit in `xtrans-mlri-paper-audit.md` confirms that the original
papers differ in their overlap reduction: the 2014 implementation uniformly
averages local affine coefficients, while the expanded 2016 implementation
uses inverse residual-error weights. Neither paper specifies the surrounding
X-Trans construction.

Two hidden methods therefore hold the corrected X-Trans guide geometry fixed,
run one sigma-2 green pass, and use the direct final green-guided red/blue
reconstruction without the later `sqrt(green/255)` blend:

- `mlri-xtrans-paper-core-2014`: uniform coefficient average.
- `mlri-xtrans-paper-core-2016`: residual-weighted coefficient average.

The existing faithful and corrected two-pass identifiers and their golden
corpus remain unchanged. Both paper-core variants completed the same nineteen
analytical scenes without fallback. CPSNR is shown below; undefined values are
the existing zero-variance cases.

| Scene | Markesteijn | Paper core 2014 | Paper core 2016 |
| --- | ---: | ---: | ---: |
| Constant | 88.34 | 87.27 | 88.34 |
| Gradient | 71.85 | 64.66 | 67.65 |
| Impulses | 43.96 | 44.87 | 44.87 |
| Frequency sweep | 10.08 | 9.51 | 9.50 |
| Saturated edges | 62.42 | 27.30 | 40.18 |
| Black | undefined | undefined | undefined |
| Asymmetric orientation | 21.31 | 37.15 | 40.32 |
| One-pixel lines | 20.24 | 21.39 | 22.40 |
| Diagonal lines | 27.16 | 23.10 | 21.44 |
| Concentric circles | 20.82 | 21.68 | 21.35 |
| Zone plate | 30.49 | 30.07 | 31.59 |
| Sinusoidal grating | 21.99 | 31.49 | 31.23 |
| Fine checkerboard | 89.19 | 39.70 | 48.29 |
| Red/green transition | 97.56 | 57.48 | 51.35 |
| Blue/green transition | undefined | undefined | undefined |
| Near-Nyquist achromatic | 88.40 | 47.82 | 47.46 |
| Near-Nyquist chromatic | undefined | 12.66 | 14.34 |
| Black/white text-like | 15.79 | 18.35 | 18.72 |
| Colored text-like | 12.72 | 12.68 | 12.93 |

Residual weighting is materially safer on saturated edges and the fine
checkerboard and improves the asymmetric scene, gradient, zone plate, and
several text/detail cases. Uniform averaging is better on the hard red/green
transition, diagonals, rings, and the sinusoidal grating. Thus the 2016 rule is
the stronger general paper core, but it does not dominate the 2014 rule or
Markesteijn.

Both full-resolution `DSCF0771.RAF` exports used 294 tiles and two workers and
completed without fallback:

| Method | Elapsed | Peak RSS | Crop phase RMS R/G/B |
| --- | ---: | ---: | --- |
| Paper core 2014 | 797.20 s | 1,964,204 KiB | 0.0003946 / 0.0002822 / 0.0006099 |
| Paper core 2016 | 937.65 s | 2,157,376 KiB | 0.0003902 / 0.0002839 / 0.0006050 |
| Corrected two-pass MLRI | 3,754.49 s | 1,902,432 KiB | 0.0003154 / 0.0003419 / 0.0004827 |
| Markesteijn three-pass | 4.20 s | 1,803,616 KiB | 0.0005539 / 0.0004460 / 0.0007821 |

Relative to Markesteijn, both paper cores have lower measured phase RMS in all
three channels. Relative to corrected two-pass MLRI, both improve green but
regress red and blue. The 2014/2016 crop means differ by only
`+0.0000345`, `-0.0000070`, and `-0.0000278` normalized RGB; their common
luminance difference is negligible. Their crop-mean color differences from
Markesteijn also remain below `0.00018` channel range.

At one-third full-frame scale and nearest-neighbour 500% earring scale the two
paper cores are not visibly distinguishable. Both remain very close to the
corrected two-pass image: no seam or new diagonal pattern is visible, but the
smoother earring/hair rendering and isolated purple point remain. Removing the
second pass and final blend therefore does not resolve the reported defect.

The external 16-bit TIFF SHA-256 values are
`8751aeaf6213ce1655c6215a6c9bfff7fd92213d6c9b425df4de25d91333d59d`
for 2014 and
`8d4cae242cd597dd4c7a23d55745b3fe88228283fcd44385d1ebe42d0e09ae14`
for 2016. The tracked comparison manifest is
`mlri-paper-core-manifest.json`, SHA-256
`bc5266ad199ab06c578ef00ef592e2c623d68c3273be50febb4548535b93f002`.
Its four PNG assets preserve the established generation rules and method
order.

The paper cores reduce full-RAF runtime by roughly four to five times relative
to corrected two-pass MLRI, but remain approximately 190 and 223 times slower
than Markesteijn. They are useful controls, not production candidates.

## DSCF0771 real-image comparison

All three methods processed the same authenticated RAF through neutral 16-bit
TIFF profiles with false-colour suppression and unrelated processing disabled.
Both MLRI outputs use 294 tiles, two workers, a 384x384 core, and a 228-pixel
halo. Neither run fell back.

| Method | Elapsed | Peak RSS | Crop phase RMS R/G/B |
| --- | ---: | ---: | --- |
| Corrected MLRI | 3,754.49 s | 1,902,432 KiB | 0.0003154 / 0.0003419 / 0.0004827 |
| Faithful MLRI | 4,602.98 s | 1,888,272 KiB | 0.0003176 / 0.0003432 / 0.0004884 |
| Markesteijn three-pass | 4.20 s | 1,803,616 KiB | 0.0005539 / 0.0004460 / 0.0007821 |

The corrected variant lowers the three measured phase RMS values by 0.7%,
0.4%, and 1.2% relative to faithful MLRI. Its crop mean differs from faithful
MLRI by only `+0.0000122`, `+0.0000007`, and `-0.0000078` in normalized RGB.
Across the crop, 99% of per-channel corrected/faithful absolute differences
are no greater than 85, 66, and 108 out of 65,535; normalized RMS is 0.000345,
0.000264, and 0.000506. Rare strong-edge pixels change more substantially,
which is consistent with the directional-guide substitution.

At the tracked one-third full-frame scale and nearest-neighbour 500% earring
scale, corrected and faithful MLRI are not visibly distinguishable. The
isolated purple points in the red object at the left of the earring crop remain.
Both MLRI variants make the metallic earring more neutral and continuous than
Markesteijn, but both are visibly smoother in adjacent hair and metal and may
suppress real fine detail. No seams, repeating CFA texture, global cast, or new
diagonal artifact are visible in this RAF.

The corrected source TIFF is 7752x5178 uint16 RGB with SHA-256
`b4abdb3f65dd0f14bfdc2c66f822831134b32e3d444b34a56d1242df99da5838`.
The tracked corrected assets are:

- `DSCF0771-mlri-corrected-full-third.png`, SHA-256
  `a3ff375b896c95be21318057df70ffc1b7474da6bb453d1bed2a60b5bc171ff9`.
- `DSCF0771-mlri-corrected-earring-500.png`, SHA-256
  `97cdf66e53f9b49c13f811ff19b42ac12420446de7eff7ebf5a4d9c9688bc229`.
- `mlri-corrected-manifest.json`, SHA-256
  `186fdc198c5a51bc028f61b11dec794e5ffae00fc1a9de33e92e151137541ec7`.

### Controlled final-reconstruction test

The hidden `mlri-xtrans-2pass-corrected-final-only` method holds the corrected
two-pass green reconstruction, masks, filters, parameters, boundary handling,
and tiling fixed. It changes only the final output selection: red and blue come
directly from the final green-guided reconstruction instead of blending its
chroma differences with the provisional reconstruction using
`sqrt(green / 255)`. Green is bit-identical to corrected two-pass MLRI in the
native controlled test, and observed CFA samples remain unchanged.

All nineteen analytical scenes completed without fallback. CPSNR is shown
below; `Delta` is direct-final minus corrected blended MLRI.

| Scene | Corrected blend | Direct final | Delta |
| --- | ---: | ---: | ---: |
| Constant | 88.34 | 88.34 | +0.00 |
| Gradient | 70.78 | 70.54 | -0.24 |
| Impulses | 48.63 | 48.63 | +0.00 |
| Frequency sweep | 9.48 | 9.29 | -0.19 |
| Saturated edges | 42.93 | 44.06 | +1.13 |
| Black | undefined | undefined | n/a |
| Asymmetric orientation | 23.15 | 39.69 | +16.54 |
| One-pixel lines | 29.90 | 28.93 | -0.97 |
| Diagonal lines | 28.59 | 28.48 | -0.11 |
| Concentric circles | 29.86 | 29.43 | -0.43 |
| Zone plate | 39.30 | 38.91 | -0.39 |
| Sinusoidal grating | 35.90 | 36.38 | +0.47 |
| Fine checkerboard | 25.69 | 49.40 | +23.71 |
| Red/green transition | 67.93 | 68.28 | +0.35 |
| Blue/green transition | 63.41 | undefined | n/a |
| Near-Nyquist achromatic | 48.69 | 49.08 | +0.39 |
| Near-Nyquist chromatic | 16.63 | 14.79 | -1.83 |
| Black/white text-like | 30.16 | 28.39 | -1.77 |
| Colored text-like | 12.44 | 11.90 | -0.55 |

Among scenes with finite CPSNR for both methods, direct final improves seven,
regresses nine, and ties one. Its +2.12 dB arithmetic mean delta is not a useful
general ranking because it is dominated by the asymmetric-orientation and
fine-checkerboard gains. The mixed result supports keeping the experiment
separate rather than silently replacing the corrected blend.

The full-resolution controlled export completed without fallback:

| Method | Elapsed | Peak RSS | Crop phase RMS R/G/B |
| --- | ---: | ---: | --- |
| Direct final | 4,566.36 s | 2,119,628 KiB | 0.0003382 / 0.0003286 / 0.0004941 |
| Corrected blend | 3,754.49 s | 1,902,432 KiB | 0.0003154 / 0.0003419 / 0.0004827 |
| Markesteijn three-pass | 4.20 s | 1,803,616 KiB | 0.0005539 / 0.0004460 / 0.0007821 |

The direct-final crop mean differs from the corrected blend by only
`-0.0000085`, `+0.0000360`, and `-0.0001096` normalized RGB. It slightly
improves green phase RMS and slightly regresses red and blue. The measured
76.1-minute runtime is not an added mathematical cost of selecting direct
output: this deliberately controlled implementation still computes the entire
provisional and final paths before choosing one. It remains an unoptimized
correctness implementation. The two single-run timings do not establish why
this export was slower; the direct selection itself adds no expensive stage.

The canonical red-object probe selects the ten greatest corrected-MLRI local
blue excursions in rectangle `(3510,1990,26,41)`. Local excess is the pixel's
normalized B-G value minus the median B-G of the other 24 pixels in its 5x5
neighborhood. Mean excess falls from `0.019592` for the corrected blend to
`0.006625` for direct final; Markesteijn measures `0.005577` at the same
coordinates. At the originally identified pixel `(3516,2009)`, the value falls
from `0.023362` to `0.005188` (`0.019821` for Markesteijn).

The tighter pixel view confirms that the blue/purple checker excursions in the
red object are reduced, while the established 500% earring view changes only
subtly. The corrected MLRI earring/hair character remains, not every colored
pixel disappears, and no seam, orientation error, or global cast is visible.
This passes the specific controlled overshoot hypothesis but does not make the
method production-ready. It shows that the square-root provisional/final blend
materially amplifies these outliers; it does not show that the blend is their
only cause.

The external direct-final TIFF is 7752x5178 uint16 RGB with SHA-256
`577a3385728504f4dbcdda5dca34b07d419fae7191ffb644bb9aae6f89ff6a37`.
The tracked full-third and earring PNG hashes are
`c91ddb15de4f299fbccb4ced5bb92f34118a338918fdbfbbc023c03505789371`
and
`8e12d4c56103187b633534d5199ca726d6a0ca1b1c6885f351c749e77ef82286`.
The canonical comparison manifest is
`mlri-corrected-final-only-manifest.json`, SHA-256
`917864bd0699cd45d8769290c654eafe34418e92df2e7764affd7b38739435ae`.

## Verification

- The seven native MLRI CTests pass in the normal build: error contract,
  Octave golden comparison, faithful/corrected behavior, controlled
  direct-final and paper-core variants, tiled/untiled seam parity, and
  all-orientation output parity.
- The complete 30-test native suite passes; external neural artifacts are the
  only skipped cases.
- The MLRI tests pass under Clang 18 ASan/UBSan. LeakSanitizer cannot operate
  under the test host's tracing mechanism and was disabled; address and
  undefined-behavior instrumentation remained active.
- GCC 13 strict compilation succeeds for `xtrans_mlri.cc` and the modified
  dispatch source. The complete strict target remains blocked by the existing
  unrelated `dcraw.cc` Foveon `-Wstringop-overflow` diagnostic.
- The Python development suite passes with 204 tests and 23 optional-artifact
  skips.
- `BUILD_TESTING=OFF` creates neither the MLRI test target nor the existing
  neural development targets.
- `git diff --check` passes.

## Decision

Retain all four identifiers as hidden developer-only experiments. The corrected
two-pass variant remains the internally consistent reference for further
algorithmic study, and the 2016 paper core is the stronger of the two
one-pass controls, but this test does not prove that the four MATLAB
expressions are accidental, nor
that they cause the diagonal pattern in Michael's X-T5 example. On the actual
requested photograph the correction is measurable but not visibly beneficial,
and it does not remove the reported purple pixels.

None of the MLRI methods is ready for GUI exposure. Corrected MLRI remains about
894 times slower than Markesteijn on this 40 MP export and trades false-colour
suppression for visibly softer fine detail. Markesteijn three-pass therefore
remains the default. The source-compatible method remains the audit baseline;
future correctness experiments should state explicitly which variant they use.
The paper-core comparison could not isolate the second pass from the final
blend. The controlled two-pass direct-final result now shows that the final
square-root blend materially amplifies the measured blue overshoot, although
residual colored pixels and MLRI's severe performance cost remain.
