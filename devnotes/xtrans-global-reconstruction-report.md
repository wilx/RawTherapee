# Three-phase global X-Trans reconstruction experiment

> Privacy update: DSCF0771 full-frame imagery is private and is not
> distributed. Only the reviewed earring crops may be shared. Historical
> full-frame asset references below describe private benchmark evidence,
> not current publication instructions.

## Result

The experiment answers its primary question negatively for this formulation.
Using all X-Trans samples in one global inverse solve can suppress some
CFA-period color variation, but it does not recover otherwise unavailable
detail. The quadratic model trades aliases for blur, performs very badly on
colored discontinuities, and is not better than the same objective on
overlapping 384- or 192-pixel tiles. The edge-aware variant recovers some
isolated luminance structure but also preserves CFA-phase chromatic points.

All three implementations remain hidden developer methods:

- `xtrans-global-spectral-rgb` (A)
- `xtrans-global-spectral-diff` (B)
- `xtrans-global-spectral-edge` (C)

They do not alter Markesteijn, the method enumeration, the GUI, defaults,
translations, or ordinary PP3 behavior.

## Objectives and numerical method

Let `L` be the symmetric four-neighbor graph Laplacian divided by eight. No
edge crosses the finite image boundary, which implements a zero-normal-
derivative, mirror/Neumann boundary. Its DCT eigenvalues are the normalized
discrete frequencies

```text
((2 - 2 cos(kx)) + (2 - 2 cos(ky))) / 8.
```

Consequently `L` and `L^2` are the matrix-free spatial forms of the requested
spectral penalties with exponents one and two. Solving an inverse system with
these operators gives every observation global support; the code is not merely
applying a local Laplacian filter. This representation also avoids the silent
opposite-edge coupling of a periodic FFT.

Phase A independently minimizes, for each `C` in `R,G,B`,

```text
||M_C C - y_C||^2 + lambda_rgb C' L^p C.
```

Phase B solves jointly for `G`, `D_R=R-G`, and `D_B=B-G`:

```text
||M_G G - y||^2
+ ||M_R (G + D_R) - y||^2
+ ||M_B (G + D_B) - y||^2
+ lambda_g G' L^p G
+ lambda_c (D_R' L^p D_R + D_B' L^p D_B).
```

Phase C retains Phase B's data and chroma terms. It replaces the green term by
a Charbonnier gradient prior. Each IRLS outer step forms horizontal and
vertical weights

```text
w = epsilon / sqrt(delta_G^2 + epsilon^2)
```

and solves the resulting weighted quadratic system.

The objectives deliberately use soft data fidelity so their absolute lambda
values remain meaningful. After the solve, each physically measured mosaic
value is copied exactly into its matching final RGB plane. The diagnostic
records both pre-projection sample error and the bit-exact final constraint.

All systems use matrix-free Jacobi-preconditioned conjugate gradient. Vectors
are float32; dot products use deterministic fixed 4096-value blocks accumulated
in float64. Phase A initializes from independent triangulated RGB; Phase B
initializes from triangulated green/color differences. Phase C initializes from
its Phase B result. A zero-filled initialization is available as a controlled
alternative.

## Parameters

The defaults and environment overrides are documented in
`tools/xtrans_global/README.md`.

| Parameter | Default | Values exercised |
| --- | ---: | --- |
| `lambda_rgb` | 0.02 | 0.02 |
| `lambda_g` | 0.01 | 0.01 |
| `lambda_c` | 0.10 | 0.01, 0.10, 1.0 |
| spectral exponent | 1 | 1, 2 |
| PCG limit | 20 | 5, 10, 20, 50 |
| relative tolerance | 1e-5 | 1e-5 |
| Charbonnier epsilon | 0.01 | 0.01 |
| C outer/inner iterations | 3 / 10 | 3 / 10 and 3 / 50 |
| tile size | whole frame | whole, 384, 192, 96 |
| initialization | triangulated | triangulated, zero-filled |

Optional diagnostics write normalized little-endian float32 R/G/B and, for
B/C, `D_R`/`D_B`, plus a residual CSV. No denoising, sharpening, false-color
suppression, median filter, clipping, or local cleanup is applied.

## Native analytical tests

On a small correlated luminance/chroma scene, normalized reconstruction RMS is:

| A | B | C |
| ---: | ---: | ---: |
| 0.00567682 | 0.00123052 | 0.000667863 |

This establishes that the implementation can show both the advantage of color
differences and an edge prior when the test image satisfies those assumptions.

The iteration sweep on a separate correlated field was:

| Limit | A RMS | A final residual | B RMS | B final residual |
| ---: | ---: | ---: | ---: | ---: |
| 5 | 0.00186149 | 0.0256601 | 0.000450766 | 0.886464 |
| 10 | 0.00186689 | 0.000325420 | 0.000448637 | 0.131921 |
| 20 | 0.00186690 | 0.000005395 | 0.000285445 | 0.042253 |
| 50 | 0.00186690 | 0.000005395 | 0.000277995 | 0.001741 |

A converges quickly toward the same flawed smooth solution. B benefits from
more iterations, but its gain from 20 to 50 is small on this scene. At 20
iterations, increasing `lambda_c` from 0.01 to 0.10 to 1.0 changes RMS from
`0.000576536` to `0.000285445` to `0.0000619332`; this is a deliberately smooth
correlated scene and does not predict colored-edge behavior. Exponents one and
two give `0.000285445` and `0.000277129` respectively.

Initialization matters greatly before convergence: B's triangulated and
zero-filled outputs differ by normalized RMS `0.227263` after five iterations,
but only `0.00036962` after 50. Finite-iteration comparisons therefore use the
same triangulated initialization.

All 18 unique phase/orientation matrices reproduce constants and preserve
samples. Repeated whole-image runs are bit-identical. A left-border impulse
does not behave as an opposite-edge periodic neighbor.

### Whole image versus tiles

For a 594x510 field containing broad structure plus diagonal fine detail:

| Support | Ground-truth RMS |
| --- | ---: |
| whole image | 0.000243981 |
| overlapping 384-pixel tiles | 0.000243636 |
| overlapping 192-pixel tiles | 0.000242641 |

Whole-versus-384 output RMS is `0.000000962`; whole-versus-192 is
`0.000003629`. The tiles are marginally better, not worse. A separate 96-pixel
control differs from whole B by `0.000003502`. There is no evidence here that
long-range support itself supplies useful missing information.

## Rendered ground-truth DNG benchmark

The development benchmark creates neutral 96x96 X-Trans DNGs, exports through
the same color pipeline, and compares encoded output to encoded ground truth.
The complete local record is `/tmp/xtrans-global/benchmark/report.json`.

| Scene | A PSNR / SSIM | B PSNR / SSIM | C PSNR / SSIM | Tri RGB | Tri diff | Markesteijn |
| --- | --- | --- | --- | ---: | ---: | ---: |
| gradient | 80.44 / 1.0000 | 80.40 / 1.0000 | 68.74 / 1.0000 | 80.44 | 80.44 | 74.05 |
| saturated edges | 23.35 / 0.9256 | 14.95 / 0.6138 | 12.68 / 0.4336 | 24.95 | 21.55 | 31.64 |
| impulse | 41.89 / 0.9919 | 52.24 / 0.9976 | 60.99 / 0.9996 | 41.14 | 38.47 | 44.35 |
| frequency sweep | 12.23 / 0.3244 | 9.36 / 0.0319 | 9.16 / 0.0104 | 11.52 | 9.15 | 9.56 |

Constants are reproduced equally by every method (`78.83 dB`, the remaining
error being rendering quantization), and flat black is exact.

The strongest positive result is C's isolated impulse. The decisive negative
result is saturated colored edges: B and C are 16.68 and 18.96 dB below
Markesteijn. C also has the worst frequency-sweep SSIM. Global smooth chroma is
therefore a poor prior exactly where channels contain real discontinuities.

## DSCF0771 real image

The reviewed 7752x5178 RAF export source has SHA-256
`26106d7da2ba9a87caebffd4bd5e5fb0371cc8a2b6139ec724b59ef4112842c0`.
All four exports used the same neutral profile, 16 threads, 16-bit uncompressed
TIFF, no false-color suppression, and no sharpening or denoise.

| Method | Demosaic wrapper | Complete export | Peak RSS | Pre-projection sample max / RMS |
| --- | ---: | ---: | ---: | ---: |
| A, defaults | 9.37 s | 12.99 s | 3,539,972 KiB | 0.005164 / 0.00003445 |
| B, 20 iterations | 13.10 s | 16.64 s | 5,264,392 KiB | 0.002366 / 0.00001592 |
| C, 3x10 IRLS | 34.87 s | 38.45 s | 5,578,808 KiB | 0.000319 / 0.00000740 |
| Markesteijn 3-pass | not isolated | 4.06 s | 2,002,248 KiB | n/a |

As a coarse iteration-cost diagnostic, wrapper time divided by completed PCG
updates is 0.276 s for A (34 updates summed across three single-plane solves),
0.655 s for B (20 joint three-plane updates), and 0.697 s for C (50 joint
updates including the Phase-B initializer). These figures describe unlike
operators and are useful for accounting, not as a direct kernel benchmark.

The final native-sample error is exactly zero for A/B/C. Default B stops at
relative residual `0.0485`; 50 iterations reach `0.00273` and take 30.58 s
complete. The two full rendered outputs differ by normalized RMS `0.001716`,
but the established earring crop does not show a qualitative improvement.

C's 3x10 result ends its last inner solve at relative residual `0.683`.
Increasing each inner limit to 50 takes 94.14 s, performs 170 total PCG
iterations, and reaches `0.239`. Its output differs from default C by RMS
`0.002296`, yet the colored points in the red object left of the earring remain.
They are therefore not merely a 10-iteration transient. Full convergence would
be much more expensive and is not supported as a useful direction by the
analytical colored-edge result.

Low-detail 3x3 CFA-phase RMS on crop `(3450,1750,700,500)` is:

| Method | R | G | B |
| --- | ---: | ---: | ---: |
| A | 0.000171 | 0.000489 | 0.000376 |
| B | 0.000292 | 0.000323 | 0.000408 |
| C | 0.000291 | 0.000328 | 0.000407 |
| Markesteijn | 0.000554 | 0.000446 | 0.000782 |

B and C reduce the measured phase texture overall, but this statistic does not
capture isolated edge-aligned colored points. In the 500-percent view, A has
obvious green/purple segmentation on the metal. B makes the earring much more
neutral but visibly smooths hair and fine metal structure. C is nearly B on the
earring; on the red object at the left-middle it makes a regular grid of dark
red/purple points slightly more conspicuous. The 3x50 run retains them. This is
consistent with edge reweighting protecting a CFA-phase excursion as if it were
real high-frequency green/luminance structure.

The tracked comparison order is `A | B | C | Markesteijn`:

- Full-frame image withheld for privacy (private benchmark only)
- [500-percent earring](images/xtrans-neural/DSCF0771/DSCF0771-global-comparison-earring-500.png)

The canonical metrics and asset identities are in
`images/xtrans-neural/DSCF0771/global-manifest.json`. Full TIFFs remain external:

- A: `/tmp/xtrans-global/DSCF0771-global-a.tif`, SHA-256
  `04e59c6fe60ea926da914929a4334e00be24447a33a610ca64d9d66eb710c327`
- B: `/tmp/xtrans-global/DSCF0771-global-b.tif`, SHA-256
  `ee704b8ef7834368a691937f0352a846b2d33f423c84d87adaaf61d8b214605d`
- C: `/tmp/xtrans-global/DSCF0771-global-c.tif`, SHA-256
  `b6ee01479d0c91ad22ab00e71e0ea4c9e70e7f2adc64e69c671b73b46e8457d1`
- B at 50: SHA-256
  `d976d99586faaa2ba915e2ec84a2daf2bc2e65e24b440d225c0d04d697222e85`
- C at 3x50: SHA-256
  `47293ada734fd8bb73e047ae9b478be430a948785ddcb6e608736aa9ced9b90d`

## Explicit answers

- **Does whole-frame reconstruction outperform tiles?** No. The 384- and
  192-pixel controls are marginally better on the controlled ground-truth
  field, and their outputs are extremely close to the whole solve.
- **Does B clearly outperform A?** On smooth correlated structure, isolated
  impulses, and the metallic earring, yes. On saturated colored edges and the
  frequency sweep, no; B is substantially worse.
- **Does C clearly outperform B?** No. It improves an isolated impulse and the
  correlated native test, but worsens colored edges and frequency content and
  retains conspicuous chromatic points on the real red object.
- **Do more iterations keep improving quality?** A simply converges to the same
  result. B gains numerically through 50 iterations without changing the
  qualitative earring conclusion. C changes substantially but retains the
  observed point artifact even after five times more inner work.
- **What benefits from long-range information?** Smooth gradients and smooth
  color differences can be reconstructed cleanly; isolated neutral impulses
  benefit from the joint/edge formulation.
- **What is harmed?** Saturated channel-specific edges, near-Nyquist color,
  repeating texture, and fine structure are smoothed, aliased, or assigned to
  the wrong component.
- **Does it reveal X-Trans-specific alias structure?** Yes. A exposes periodic
  chromatic segmentation; B suppresses much of it by moving the prior to color
  differences; C can protect phase-aligned excursions when its edge weights
  mistake them for real detail. It does not reveal recoverable information that
  the tile control loses.
- **Is the improvement worth the cost?** No. The useful earring neutralization
  comes with severe analytical failures, 3.1x to 9.5x complete-export time,
  and roughly 1.5 to 3.5 GiB more peak resident memory than Markesteijn.

## Recommendation

Keep these three methods and their fixtures as a documented negative/reference
experiment. Do not expose them in the GUI and do not add postprocessing to hide
their failures. The result argues against pursuing more whole-frame iterations
or a more elaborate optimizer for these objectives. A future algorithmic
experiment would need a materially different alias/data model—not merely more
global support or a stronger smoothness prior.

## Verification

- The complete GCC `dev-strict` build passed.
- The focused debug target built, and all five `xtrans-global` CTests passed in
  both strict/development and debug builds.
- The same five tests passed under combined ASan/UBSan. LeakSanitizer had to be
  disabled because it cannot operate under the managed runner's ptrace
  supervision; address and undefined-behavior instrumentation remained active.
- All eleven pre-existing MLRI and triangulation CTests passed.
- The complete neural/tooling Python suite passed: 209 tests passed and 23
  external-artifact tests skipped.
- The comparison-asset tests authenticate the committed PNGs and canonical
  manifest without requiring the external RAF or TIFF files.
