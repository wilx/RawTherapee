# X-Trans alias and frequency-confusion characterization

## Result

The central hypothesis is **partly supported, but not sufficient by itself**.
The 6x6 X-Trans sampling operator has exact, structured frequency/color
ambiguities. These ambiguities explain the previous frequency-sweep failure,
part of the saturated two-dimensional edge failure, and why CFA-relative phase
can make isolated grid-aligned artifacts unusually strong. They do not explain
all of the large B/C edge regression: several simple chromatic edges are
well-observable from the CFA and were damaged by the reconstruction prior.

The most important result is the exact per-alias-family model. For every base
frequency `omega`, X-Trans maps 54 possible complex RGB coefficients (three
colors at each of 18 carrier-related frequencies) to 18 observed mosaic
coefficients:

```text
A[r,(h,c)] = M_hat_c[r-h],   r,h in H, c in {R,G,B}.
```

`A` is 18x54, has rank 18, and has a 36-dimensional exact null space. Its 18
nonzero singular values are all one. This is true at every base frequency; the
operator is translated in frequency but does not otherwise change. A plot of
only the nonzero condition number would therefore be misleadingly perfect.
The [full-domain map](images/xtrans-alias/full-alias-nullity.png) correctly
shows nullity 36 everywhere.

Conversely, if the spatial frequency is already known and isolated, the 18x3
RGB matrix has singular values

```text
sqrt(5)/3, sqrt(2)/3, sqrt(2)/3
```

and condition number `sqrt(5/2) = 1.58114`. A single known-frequency RGB mode
is well determined through its complete replica constellation. Ambiguity
becomes fundamental when several carrier-related frequencies are allowed to be
present simultaneously.

No RawTherapee demosaicing or production code is changed by this experiment.

## Frequency and color convention

Frequencies use cycles/pixel on the half-open square `[-0.5,0.5)` for both
axes. NumPy's negative-exponent DFT is normalized by sample count. The
orthonormal color basis is:

```text
L  = (R + G + B) / sqrt(3)
C1 = (R - B) / sqrt(2)
C2 = (R - 2G + B) / sqrt(6)
```

The requested `R-G` and `B-G` directions are also present in the machine data,
but `L,C1,C2` makes ranks, singular values, and energy fractions interpretable
without color-basis correlation.

## Exact mask spectrum

The canonical cell is periodic under `(6,0)` and `(3,3)`, so the reciprocal
group contains 18 rather than 36 carriers. In the table below every mask
coefficient has the exact form `(a + b sqrt(3) i)/36`.

| DFT index | frequency | red `(a,b)` | green `(a,b)` | blue `(a,b)` |
| --- | --- | ---: | ---: | ---: |
| (0,0) | (0,0) | (8,0) | (20,0) | (8,0) |
| (2,0) | (1/3,0) | (-1,-1) | (2,2) | (-1,-1) |
| (4,0) | (-1/3,0) | (-1,1) | (2,-2) | (-1,1) |
| (1,1) | (1/6,1/6) | (0,0) | (0,0) | (0,0) |
| (3,1) | (-1/2,1/6) | (3,-3) | (0,0) | (-3,3) |
| (5,1) | (-1/6,1/6) | (0,0) | (0,0) | (0,0) |
| (0,2) | (0,1/3) | (-1,-1) | (2,2) | (-1,-1) |
| (2,2) | (1/3,1/3) | (2,-2) | (-4,4) | (2,-2) |
| (4,2) | (-1/3,1/3) | (-4,0) | (8,0) | (-4,0) |
| (1,3) | (1/6,-1/2) | (-3,3) | (0,0) | (3,-3) |
| (3,3) | (-1/2,-1/2) | (0,0) | (0,0) | (0,0) |
| (5,3) | (-1/6,-1/2) | (-3,-3) | (0,0) | (3,3) |
| (0,4) | (0,-1/3) | (-1,1) | (2,-2) | (-1,1) |
| (2,4) | (1/3,-1/3) | (-4,0) | (8,0) | (-4,0) |
| (4,4) | (-1/3,-1/3) | (2,2) | (-4,-4) | (2,2) |
| (1,5) | (1/6,-1/6) | (0,0) | (0,0) | (0,0) |
| (3,5) | (-1/2,-1/6) | (3,3) | (0,0) | (-3,-3) |
| (5,5) | (-1/6,-1/6) | (0,0) | (0,0) | (0,0) |

Exactly 13 rows are nonzero. The five zero carriers are `(1,1)`, `(5,1)`,
`(3,3)`, `(1,5)`, and `(5,5)`. The individual mask plots are:

- [red](images/xtrans-alias/mask-spectrum-red.png)
- [green](images/xtrans-alias/mask-spectrum-green.png)
- [blue](images/xtrans-alias/mask-spectrum-blue.png)

This reproduces Equation 6 of Rafinazari and Dubois, ICIP 2014 (DOI
`10.1109/ICIP.2014.7025132`) up to carrier row order and Fourier-sign
conjugation. It also reproduces their key statements that the apparent 6x6
cell is an 18-site `3x6` lattice and that only 13 of the 18 components are
nonzero.

The carrier rows separate naturally by color direction:

- Pure `L` passes through without modulation because the masks sum to one.
- `C1 = R-B` occupies the four carriers near `(-1/2,+/-1/6)` and
  `(+/-1/6,-1/2)`.
- `C2 = R-2G+B` occupies DC and the eight even one-third-grid carriers.

These are the two chroma families underlying the paper's component relations,
expressed in an orthonormal basis.

## Direct single-frequency validation

A 96x96 complex exponential at bin `(7,11)`, or frequency
`(0.0729167,0.114583)`, was sampled directly and compared with the predicted
mask convolution:

| Direction | significant observed peaks | maximum complex error |
| --- | ---: | ---: |
| L | 1 | `1.28e-15` |
| C1 | 4 | `5.49e-16` |
| C2 | 9 | `7.72e-16` |

The [replica plot](images/xtrans-alias/single-frequency-response.png) shows the
fixed constellations. Two independently generated mosaics also satisfy
`A(f1+f2)=Af1+Af2` exactly in sample space; their FFTs agree to `3.41e-13`
absolute error.

This gives a concrete future discriminator: a genuine isolated chroma mode
must produce the predicted relative complex magnitudes and phases at four or
nine carrier-related locations. A pure luma mode produces no companion
replicas at all.

## Direct confusion

For one complex source component, maximum observation correlation over a
different carrier displacement is:

| source | competitor | maximum similarity |
| --- | --- | ---: |
| L | L at another carrier | 0 |
| L | C1 | 0.500000 |
| L | C2 | 0.408248 |
| C1 | C1 at another carrier | 0.500000 |
| C1 | C2 | 0.306186 |
| C2 | C2 at another carrier | 0.250000 |

The [complex confusion matrix](images/xtrans-alias/complex-confusion.png) is
not close to singular for any individual pair. Pairwise comparison alone is
therefore insufficient: combinations of multiple sources create the exact
null space.

For example, the observed signature of one luma component at base frequency
is reproduced to relative residual `4.69e-16` by four chroma components:

```text
(1/sqrt(6) - i/sqrt(2)) C1 at (-1/2, +1/6)
-1/sqrt(2)              C2 at ( 0,    0)
(1/sqrt(6) + i/sqrt(2)) C1 at (-1/2, -1/6)
+1/sqrt(6)              C1 at (-1/2, -1/2)
```

The corresponding difference between the luma source and this chroma sum is
an exact sampling null vector. A demosaicer must choose between them using
scene evidence or a prior; the CFA observation alone cannot do so.

## Real phase and CFA translation

For real cosines, positive and negative frequency components coexist. When
`2*omega` equals a CFA carrier, those sidebands enter the same alias family and
their phases can reinforce or cancel.

Testing phases `0, pi/4, pi/2, 3pi/4` on a 24x24 grid gives:

| comparison | minimum worst-case | mean | maximum |
| --- | ---: | ---: | ---: |
| L against C1/C2 | 0.482963 | 0.501648 | 0.806898 |
| C1 against C2 | 0.295753 | 0.306564 | 0.570563 |

The strongest luma/chroma case has source frequency `(1/3,1/6)`, exactly a
half-carrier condition. The [luma/chroma](images/xtrans-alias/real-luma-chroma-confusion.png)
and [chroma/chroma](images/xtrans-alias/real-chroma-confusion.png) maps show the
sparse phase-sensitive hotspots.

Across all 18 supported X-Trans phase/orientation matrices, the carrier
magnitude multisets agree within `2.78e-17` and the alias-family singular
values within `6.66e-16`. Translation changes complex phase; rotation and
reflection permute/conjugate frequency coordinates. Rank, confusion magnitude,
and identifiability are invariant after the corresponding coordinate change.

## Edge and previous-benchmark analysis

For each exact 96x96 RGB scene, channel means were removed, the ground-truth
Fourier coefficients were grouped into 18-frequency alias families, and the
matrix containing only active source columns was ranked. The percentage below
is source energy residing in a family whose active columns are rank deficient.
It is not a claim that the same percentage of this particular image is lost;
it measures where an alternative source exists without changing the mosaic.

| Scene | rank-deficient family energy | maximum active columns | largest finite condition |
| --- | ---: | ---: | ---: |
| vertical/horizontal/diagonal luma edges | 0% | 3 / 3 / 9 | 1.000 |
| vertical/horizontal red-green edge | 0% | 6 | 1.732 |
| either diagonal red-green edge | 33.39% | 18 | 2.524 |
| saturated red-gray or blue-gray vertical edge | 0% | 9 | 2.000 |
| A/B/C saturated quadrant benchmark | 20.00% | 36 | 1.732 where full-rank |
| A/B/C frequency sweep | 100.00% | 36 | no active family full-rank |

The [edge-family chart](images/xtrans-alias/edge-family-ambiguity.png) makes the
orientation and scene differences explicit.

### Frequency sweep

After removing its neutral offset, the earlier sweep is entirely chromatic:
49.99% `C1`, 50.01% `C2`, and numerical-zero luma energy. Its chirped spectrum
populates carrier-related frequencies broadly enough that every active family
is rank deficient. This supports a sampling-identifiability explanation for
the poor PSNR (A 12.23, B 9.36, C 9.16, Markesteijn 9.56). A smooth chroma
prior can select one plausible member of the family, but the mosaic does not
identify it as the correct one.

### Saturated colored boundaries

The exact quadrant benchmark is also purely chromatic after mean removal and
places 20% of its energy in rank-deficient families. Its corners create a
two-dimensional spectrum with as many as 36 active source columns in one
18-observation family. This is genuine ambiguity and helps explain the scene's
difficulty.

It is not the whole explanation. Isolated axis-aligned red-green and saturated
red-gray edges are full-rank with condition at most 2, yet B and C impose a
strong smooth-chroma model across precisely such true chromatic discontinuities.
Their 14.95 and 12.68 dB saturated-edge results, versus Markesteijn's 31.64 dB,
therefore include avoidable algorithm error. The CFA preserved usable evidence
which the model rejected. Diagonal chromatic edges are intrinsically harder:
one third of their energy lies in deficient active families.

### CFA-grid colored points

The phase map makes the colored-point observation plausible: at sparse
half-carrier frequencies, real luma/chroma similarity rises from the generic
roughly 0.5 level to 0.807. An edge reweighting method can protect one such
phase-aligned interpretation as scene detail.

This is not proven to be the unique cause of the DSCF0771 points. A local
48x48 rendered patch at `(3504,1974)` was checked externally. B and C both
contain the pattern, and the C-minus-B difference has only 1.47% of its luma
energy and below 0.1% of either chroma energy at direct CFA carriers. The
artifact is consistent with phase-sensitive ambiguity, but the rendered local
spectrum does not identify one dominant carrier. The evidence does not justify
claiming more.

## Bayer context

The optional 2x2 RGGB calculation has the same fundamental two-thirds nullity:
its full matrix is 4x12, rank 4, nullity 8. A known frequency has condition
`sqrt(2)=1.414`, slightly better than X-Trans's 1.581 due to Bayer's 1:2:1
sample counts.

X-Trans is substantially better at separating individual complex modes:

| pair | X-Trans maximum | Bayer maximum |
| --- | ---: | ---: |
| L versus C1 | 0.500 | 0.707 |
| L versus C2 | 0.408 | 0.949 |
| shifted C1 versus C1 | 0.500 | 1.000 |

The [Bayer matrix](images/xtrans-alias/bayer-complex-confusion.png) shows why
X-Trans can suppress ordinary periodic color aliasing even though neither CFA
can make unrestricted RGB reconstruction identifiable.

## Answers to the requested questions

### Sampling structure

- The exact nonzero coefficients are the 13 rows in the table above.
- Source frequencies are coupled only within the 18-member group obtained by
  adding reciprocal offsets `(p/6,q/6)` with even `p+q`.
- The four `C1` carriers and nine `C2` carriers form distinct, fixed complex
  replica constellations. The strongest mask rows have magnitude proportional
  to the green-opponent coefficients at one-third offsets and the red-blue
  coefficients near the Nyquist axes.

### Identifiability

- A known isolated frequency is fully determined and modestly conditioned.
- Unrestricted RGB content over a complete alias family is never identifiable:
  rank 18, nullity 36 at every base frequency.
- There is no special base-frequency deterioration for complex exponentials;
  the periodic operator is translation invariant in frequency. Deterioration
  appears when the scene's active support puts multiple components in one
  family, or when real positive/negative sidebands collide.
- Exact null combinations exist; the four-term luma/chroma example is one.

### Color ambiguity

- One isolated luma mode has at most 0.5 correlation with one X-Trans chroma
  mode, but multiple chroma modes can reproduce it exactly.
- `C1`/`C2` single-mode correlation peaks at 0.306 complex and 0.571 for the
  tested real phases.
- Broad saturated chromatic structures are especially problematic when they
  populate multiple carrier-related frequencies; an isolated axis-aligned
  chromatic edge is not inherently lost.

### Orientation and phase

- Horizontal and vertical tests are equivalent, and the two diagonal senses
  are equivalent. Chromatic diagonals are much more ambiguous than axes because
  their two-dimensional spectral support overloads some families.
- Complex phase changes only a global signature phase. Real phase matters at
  `2*omega` carrier collisions and raises worst luma/chroma similarity to 0.807.
- CFA translations change predicted replica phases, which can make a local
  artifact phase-specific without changing global singular values.

### Previous experiments

- B improved smooth correlated content because a low-frequency chroma support
  leaves most alias families sparsely occupied and well determined.
- B/C failed on saturated edges for two reasons: the quadrant/corner spectrum
  contains genuine deficient families, and their smooth-chroma prior also
  damages simple chromatic edges that remain observable.
- The chirped frequency sweep densely occupies every alias family and is
  intrinsically underdetermined without a source prior.
- C's points are compatible with real-phase collision hotspots, but the local
  rendered diagnostic does not prove a unique carrier cause.

## Implications for a future demosaicer

A future method should use concrete carrier evidence rather than merely more
support or stronger smoothness:

1. Estimate all 18 observed coefficients in a local frequency family together.
2. Test candidate chroma against the predicted four-peak `C1` or nine-peak `C2`
   relative magnitude **and phase** constellation. Reject a candidate whose
   companion replicas disagree.
3. Treat an isolated component with consistent replicas as observable even at
   high spatial frequency; do not automatically smooth a chromatic edge.
4. Detect overloaded or nearly dependent active families. Only there is a
   prior mathematically necessary, and its uncertainty should be retained
   rather than converted into confident CFA-phase color.
5. At half-carrier real-frequency hotspots, compare both quadratures and
   neighboring spatial windows/CFA phases so a phase-specific cancellation is
   not mistaken for absent evidence.
6. Use directional support: axis chromatic edges are better determined than
   diagonal or two-dimensional chromatic corners in these tests.

The compact model `A[r,(h,c)] = M_hat_c[r-h]` is suitable for such a later
phase-aware algorithm. This experiment does not yet choose the spatial window,
sparsity model, or uncertainty rule needed to turn it into a demosaicer.

## Limits of this characterization

- The exact 18x54 operator describes unrestricted Fourier coefficients in one
  periodic alias family. Natural-image support and local window leakage can
  reduce or increase the practically relevant ambiguity.
- The real-signal maps sample four phases on a 24x24 FFT grid. They locate and
  explain half-carrier collisions, but are not a proof that no stronger phase
  case exists between grid points.
- Active-family edge rank uses exact synthetic ground truth. It separates CFA
  ambiguity from algorithm error, but it does not predict the error magnitude
  of a particular nonlinear demosaicer.
- The DSCF0771 check uses rendered B/C output rather than the raw local sampling
  operator, so it supports plausibility only and not causal identification.

## Verification

- Nine Python tests verify exact coefficients, periodicity, replicas, the full
  and known-frequency ranks, phase hotspots, all 18 variants, edge cases,
  linearity, Bayer context, deterministic generation, and the tracked corpus.
- Five native CTests independently verify the CFA contract, exact spectra,
  shifted replicas, variants, and linearity. They pass in normal, strict,
  debug, and combined ASan/UBSan builds.
- The complete related X-Trans native set passes 21/21 tests, and the combined
  neural/frequency Python set passes 218 tests with 23 external-artifact tests
  skipped.
- Canonical regeneration in two independent directories is byte-identical, and
  `git diff --check` reports no whitespace errors.

## Artifacts and reproducibility

The canonical machine-readable result is
`images/xtrans-alias/characterization.json`, SHA-256
`5f96d3a329516dd0396a0258553a8b83176a2ea3e3dec3efcc4ac34044709dd5`.
It contains exact coefficients, all singular values, replica phases,
confusion competitors, edge active-family results, variant errors, and plot
hashes. The generator uses NumPy 2.3.5 and Pillow 12.3.0 and produces
byte-identical JSON and PNG files in independent directories.
