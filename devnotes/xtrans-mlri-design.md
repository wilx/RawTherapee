# X-Trans MLRI demosaicer design

## Purpose and provenance

This experiment evaluates a non-neural X-Trans demosaicer based on residual
interpolation with minimized Laplacian energy.  It must not be described as an
X-Trans algorithm published by Daisuke Kiku, Yusuke Monno, Masayuki Tanaka, or
Masatoshi Okutomi.  Their work defines RI, MLRI, and later ARI for Bayer and
multispectral layouts; the explicit 6x6 X-Trans construction used here is a
later independent engineering generalization.

Primary research references:

- D. Kiku, Y. Monno, M. Tanaka, and M. Okutomi, "Minimized-Laplacian
  Residual Interpolation for Color Image Demosaicking," Proc. SPIE 9023,
  2014, DOI 10.1117/12.2038425.
- D. Kiku, Y. Monno, M. Tanaka, and M. Okutomi, "Beyond Color Difference:
  Residual Interpolation for Color Image Demosaicking," IEEE Transactions on
  Image Processing 25(3), 2016, DOI 10.1109/TIP.2016.2518082.
- Y. Monno et al., "Adaptive Residual Interpolation for Color and
  Multispectral Image Demosaicking," Sensors 17(12), 2017,
  DOI 10.3390/s17122787.

Engineering reference:

- Rainbow-Johnny-Johnny-Image-Processing-Lim, "Unified Laplacian Residual
  Interpolation Demosaicing," MATLAB File Exchange 1.0.0, 2025-10-14.
  The reviewed X-Trans function and helper identities are recorded in
  `tools/xtrans_mlri/reference/README.md`.  The BSD license is retained in
  `licenses/MLRI_XTRANS_MATLAB_LICENSE`.

## Frozen first experiment

The faithful hidden PP3 method is `mlri-xtrans-2pass`.  It fixes the MATLAB
defaults `slow=1`, `sigma=2`, and `eps=0.01`; there are no GUI controls.
Markesteijn three-pass remains the default and is the complete fallback on
failure.

A second hidden identifier, `mlri-xtrans-2pass-corrected`, is a controlled
source-correction experiment.  In both the green reconstruction and later
chroma reconstruction, the reviewed MATLAB source forms the blue diagonal and
anti-diagonal Laplacians from the red directional guides (`Guiderd` and
`Guiderp`) while passing blue samples to the MLRI fit.  The corrected variant
uses the corresponding blue guides (`Guidebd` and `Guidebp`) in those four
expressions.  It changes no filter, pass, regularizer, clipping point, CFA
mapping, or final blend.  The faithful method and Octave golden corpus remain
the authority for reproducing the published implementation.

Two further hidden identifiers isolate the original paper-core distinction:
`mlri-xtrans-paper-core-2014` uses uniform overlap averaging of local affine
coefficients, while `mlri-xtrans-paper-core-2016` uses inverse-residual-error
weighting. Both share corrected blue diagonal guides, run one green pass at
sigma 2, and use the direct final green-guided red/blue reconstruction without
the later square-root blend. The equation-level audit and the exact boundary
between published mathematics and X-Trans engineering are in
`xtrans-mlri-paper-audit.md`.

RawTherapee's preprocessed mosaic is treated as a uint16-domain signal:

1. Clamp the scalar input to `[0,65535]` and divide by 257.
2. Run the reference arithmetic in the `[0,255]` domain using float values.
3. Preserve the reference's intermediate and final clipping points.
4. Multiply the final RGB values by 257 for the engine output.

This follows the `uint16` branch of the MATLAB entry point.  No gamma, second
white balance, sample reinjection, denoising, false-colour suppression, or
sharpening is part of the method.

## X-Trans sample topology

The MATLAB phase cell, using RawTherapee's `0=R, 1=G, 2=B`, is:

```
G R G G B G
B G B R G R
G R G G B G
G B G G R G
R G R B G B
G B G G R G
```

It contains ten distinguishable green phase masks, four red masks, and four
blue masks.  It is the existing shared canonical X-Trans cell translated by
three rows.  The implementation uses `findCanonicalXTransTransform()` with
this method-specific cell, retaining the shared deterministic transform order
and supporting all 18 unique phase/orientation representations.

## Processing map and classification

The algorithm first reconstructs green while also producing provisional red
and blue estimates, then reconstructs red and blue using the completed green
guide.

| Stage | Classification | Notes |
| --- | --- | --- |
| Tentative estimate as local `a*guide+b` | `ORIGINAL_RI` | RI framework and guided upsampling. |
| Gain fitted from sparse approximate Laplacians | `ORIGINAL_MLRI` | Expanded-paper equations 5-7; minimizes residual Laplacian energy. |
| Residual formation, interpolation, and addition | `ORIGINAL_RI` | Observed minus tentative, then tentative plus interpolated residual. |
| Residual-cost weighting of overlapping coefficients | `ORIGINAL_MLRI` | Expanded-paper equations 8-10. |
| Directional candidate combination from colour-difference gradients | `ARI_DERIVED` | Analytically weighted candidates; not the full published ARI iteration rule. |
| Eighteen phase masks and phase-dependent sparse filters | `XTRANS_GENERALIZATION` | Specific to the later 6x6 implementation. |
| Horizontal, vertical, and split diagonal candidate geometry | `XTRANS_GENERALIZATION` | Required by the X-Trans sampling distances. |
| Two complete green passes with sigma 2 then 1 | `MATLAB_AUTHOR_HEURISTIC` | `slow=1` means `N+1` processing in the reference. |
| Several phase-dependent Gaussian widths and directional supports | `MATLAB_AUTHOR_HEURISTIC` | No corresponding X-Trans derivation is published. |
| Final `sqrt(green/255)` blend of provisional and final R/B | `MATLAB_AUTHOR_HEURISTIC` | Retained solely for faithful v1.0.0 comparison. |

The source helper named `imgconv2` computes
`conv2(image,rot90(kernel,2),'same')`.  Because convolution reverses the
second operand, this is zero-padded, same-size correlation with the kernel as
written.  Asymmetric kernels therefore must not be silently reversed in C++.

## Implementation and memory model

The reference kernel operates on a rectangular canonical-coordinate tile.
The eighteen phase masks are materialized as float planes for the initial
readable experiment; they are derived from `(x mod 6,y mod 6)` and never kept
at full-frame size. Generic correlation, pointwise, guided-MLRI, Gaussian, and
directional-combination helpers keep the C++ structure readable without
importing MATLAB execution code.

An untiled invocation is retained for small native parity tests.  Production
uses 384x384 output cores whose origins are multiples of six.  A conservative
dependency audit propagates the support radius through both green passes and
the final red/blue stage; its final radius is the production halo.  Global
boundaries are zero extended. Tile boundaries are never treated as image
boundaries. The production halo is 228 pixels: a conservative support audit
gives 227 pixels and the value is rounded up to the six-pixel CFA period.
Local box reductions use a fixed order for every output sample; rolling sums
were rejected because their accumulated float round-off depended on the tile
origin and created a measurable boundary discontinuity even beyond the true
algorithmic support.

Each OpenMP task owns temporary planes for one bounded tile and writes a
non-overlapping core. At most two tasks run concurrently. The diagnostic
workspace ceiling is 508,032,000 bytes per worker for an interior 840x840
patch, or 1,016,064,000 bytes for two workers. This first implementation
deliberately favors traceability over allocation reuse and SIMD optimization.

## Developer interface and validation

The engine recognizes only the four literal hidden PP3 values documented
above. They are absent from the method enum, GUI, translations, history labels,
defaults, and fast-export controls. Successful CLI processing prints the
selected identifier, fixed parameters, tile geometry, worker count, workspace
estimate, and elapsed demosaic time. Failure prints a stable error code and the
caller immediately runs Markesteijn three-pass over the complete output.

The tracked Octave corpus contains flat, gradient, impulse, saturated, and
boundary cases. Its canonical manifest SHA-256 is
`b39e6200a049a727a014faf560e8fe5bf94860991eb23ac63cc0c2699083e161`.
Native tests cover deterministic repetition, finite/range guarantees, every
one of the 18 CFA phase/orientation representations, unsupported-CFA rejection,
and tiled-versus-untiled seams. The separate developer benchmark drives the
actual CLI on analytical DNGs and real RAF files with unrelated processing and
false-colour suppression disabled.

## Failure and evaluation boundary

Unsupported CFA matrices, images smaller than 32x32, allocation failure, and
non-finite results cause the caller to discard the partial result and run
Markesteijn three-pass.  A successful experiment is a faithful, stable, and
measured alternative; it is not required to beat Markesteijn.

Parameters will not be tuned against `DSCF0771.RAF`.  The corrected-blue-guide
variant is an explicit source-code hypothesis test, not parameter tuning.  If
neither controlled form is sufficient, the next research step is ARI or an
analytically justified candidate fusion.  The results and complementary
failure modes remain documented without GUI exposure.
