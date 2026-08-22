# MLRI paper-to-code audit

## Scope

This audit separates the published residual-interpolation mathematics from the
later X-Trans construction used by the developer-only RawTherapee experiment.
The papers and the authors' MATLAB packages describe Bayer demosaicing. They do
not define an X-Trans phase layout, diagonal X-Trans candidates, a two-pass
schedule, or the final square-root blend found in the 2025 X-Trans code.

The reviewed research artifacts are:

| Artifact | Source | SHA-256 |
| --- | --- | --- |
| Kiku et al., MLRI, SPIE 2014 | [paper](http://www.ok.sc.e.titech.ac.jp/res/DM/MLRI.pdf), DOI `10.1117/12.2038425` | `7926264f543d249895df55d7cf994e158bb1ae7a9f9bf72c50f1fd96128a4178` |
| Authors' 2014 MATLAB package | [MLRI.zip](http://www.ok.sc.e.titech.ac.jp/res/DM/MLRI.zip) | local file `guidedfilter.m`: `280dc9d195cc5b1e5064e58b0716eae269b850ebe8a2793b86a6389b781b7d77` |
| Kiku et al., expanded RI/MLRI, TIP 2016 | [paper](http://www.ok.sc.e.titech.ac.jp/res/DM/TIP_RI.pdf), DOI `10.1109/TIP.2016.2518082` | `bd7aa3157237083704253f1cf328ba7d41f75f56e436419abfb059788eb679f3` |
| Authors' 2016 MATLAB package | [TIP_RI.zip](http://www.ok.sc.e.titech.ac.jp/res/DM/TIP_RI.zip) | local file `guidedfilter_MLRI_wei.m`: `b66f2966b9af15235070d0f43686e85f0d01236da0073b5cd61f34998884d52a` |

The packages remain external. No author MATLAB source is copied into
RawTherapee.

## Published local model

For a guide `G` and a sparsely observed target colour `R`, both MLRI versions
fit a local affine model `q = a * G + b`, but determine `a` from approximate
Laplacians. In the notation used by `guidedMlri()`:

```
a = mean(laplacianGuide * laplacianObserved) /
    (mean(laplacianGuide * laplacianGuide) + epsilon)
b = mean(observed samples) - a * mean(guide samples)
```

This is the substance of 2014 equations (1)-(3), restated more explicitly as
2016 equations (5)-(7). `rtengine/xtrans_mlri.cc::guidedMlri()` computes the
same sparse window counts, Laplacian products, sample means, and affine
coefficients.

The important version difference is the overlap reduction:

| Version | Published/source rule | C++ rule |
| --- | --- | --- |
| 2014 | Uniformly average every overlapping local `a` and `b`; evaluate `meanA*G+meanB`. | `PAPER_CORE_2014`: `boxSum(a)/boxSum(ones)` and the corresponding expression for `b`. |
| 2016 | Compute each model's residual MSE, floor it, use its reciprocal as a weight, and normalize the weighted `a` and `b` sums; equations (8)-(10). | `PAPER_CORE_2016` and the existing source-compatible methods: residual-cost branch of `guidedMlri()`. |

The authors' 2014 helper additionally floors the Laplacian-guide energy at
`0.00001 * 255^2` while the 2016 helper uses the supplied epsilon directly.
The controlled RawTherapee comparison deliberately keeps the already reviewed
`epsilon=0.01` in both variants. Thus the experiment changes exactly the
coefficient-overlap rule rather than changing two numerical regularizers at
once.

Residual interpolation itself remains common: form `observed - tentative` at
known samples, interpolate that residual, and add it back to the tentative
estimate. This is implemented in the candidate reconstruction code surrounding
each `guidedMlri()` call.

## What is not in those papers

The following operations are necessary or inherited X-Trans engineering, not
claims from the cited Bayer papers:

- the 18 distinguishable positions of the 6x6 X-Trans cell;
- phase-dependent horizontal, vertical, diagonal, and anti-diagonal filters;
- inverse-gradient fusion of eight directional candidates;
- sigma-dependent half-Gaussian residual propagation;
- the 2025 source's second complete pass at sigma 1;
- the final `sqrt(green/255)` blend between provisional and separately
  reconstructed red/blue planes.

The published ARI work is also not represented merely by the directional
fusion above. Full ARI iterates RI and MLRI, selects the best iteration at each
pixel, and combines the two families by their criteria. That remains a separate
future algorithm.

## Controlled hidden methods

Two additional PP3/CLI-only methods hold the X-Trans geometry constant while
removing the two conspicuous later heuristics:

| Identifier | Coefficient average | Green passes | Blue diagonal guides | Final red/blue |
| --- | --- | ---: | --- | --- |
| `mlri-xtrans-paper-core-2014` | uniform 2014 rule | 1 at sigma 2 | corrected blue guides | direct green-guided RI/MLRI |
| `mlri-xtrans-paper-core-2016` | residual-weighted 2016 rule | 1 at sigma 2 | corrected blue guides | direct green-guided RI/MLRI |

The one-pass variants reconstruct green with the shared X-Trans candidate
geometry, clip it, then run the final green-guided red/blue RI/MLRI stage.
They do not run the provisional X-Trans red/blue completion needed only by the
later blend, and do not apply the square-root blend. Both use the corrected
blue diagonal guides so the apparent red-guide/blue-sample source inconsistency
does not become a third experimental variable.

The existing `mlri-xtrans-2pass` and
`mlri-xtrans-2pass-corrected` methods are unchanged. In particular, the
authenticated Octave golden corpus continues to test only the faithful 2025
source-compatible method.

## Interpretation boundary

These are paper-core *comparisons*, not implementations of a published
X-Trans algorithm. A quality difference can identify whether uniform versus
residual-weighted local-model averaging, the second pass, or the final blend is
important in this engineering construction. It cannot establish that either
paper proposed the surrounding X-Trans filters, and constants must not be tuned
against the `DSCF0771` earring crop.
