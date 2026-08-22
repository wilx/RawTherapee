# Prior-art comparison of the experimental X-Trans triangulation demosaicers

## 1. Executive summary

This report compares commit `d2981e94945c1e612ebd52ae6791b79bd30c90ca`
against the closest triangulation and color-difference demosaicing work found.
It is a technical provenance analysis, not a patentability or freedom-to-operate
opinion.

The principal conclusions are:

1. **The independent RGB experiment is known in broad substance.** Separately
   representing each measured-color lattice as a triangular mesh and linearly
   interpolating missing channel values is explicit in Su and Willis for Bayer
   and in Yao et al. for individual MSFA bands. General linear triangular
   interpolation is much older still. The examined sources do not, however,
   disclose our exact combination of an infinite periodic X-Trans lattice,
   geometry-only Delaunay topology, deterministic periodic resolution of
   cocircular faces, and 36 compiled phase stencils.

2. **The chroma experiment's main signal-processing idea is established prior
   art.** Green-first reconstruction and interpolation of `R-G` and `B-G`
   exploit the long-established observation that color differences are often
   smoother than the color planes. Freeman, Laroche--Prescott,
   Hamilton--Adams, Pei--Tam, Kimmel, and Su--Willis all precede this experiment.
   Multispectral SD/ISD methods use the analogous inter-band-difference idea.

3. **The exact chroma construction is materially different from the two named
   papers.** Our method first builds one dense `Ghat`, samples
   `R-Ghat`/`B-Ghat` only at native R/B sites, interpolates each sparse
   difference with exactly the corresponding R/B Delaunay stencil, and adds the
   result to `Ghat`. Su--Willis constructs `G-R` and `G-B` estimates at several
   Bayer site types and uses content-selected two-point directions. Yao et al.
   forms differences between a measured sparse band and another spatially
   reconstructed band, interpolates those differences bilinearly, and combines
   contributions from multiple reference bands.

4. **“A non-adaptive triangulation-based demosaicer” is accurate; “a frozen
   Su--Willis DDT” is potentially misleading.** Both methods interpolate on
   triangles, but their topology spaces and selection rules differ. Su--Willis
   chooses one of two diagonals in each regular same-color Bayer square using
   image values. Our method computes a Delaunay triangulation of the irregular
   X-Trans same-color point lattice and resolves geometric degeneracy without
   image values. It is a non-adaptive member of the broad triangulation family,
   not a literal parameter limit of Su--Willis.

5. **The observed failures match the literature's assumptions and warnings.**
   Shared green structure makes the chroma version much better on neutral edges
   and patterns. It becomes worse when a colored checkerboard deliberately
   violates local smoothness of color differences. Fixed triangles cannot turn
   with a scene edge: Su--Willis introduced data-dependent diagonals for exactly
   that reason, and Yao et al. explicitly warns that triangles are not inherently
   aligned with image features and can produce zigzags or local distortion.

No searched academic paper or patent was found that describes the complete
fixed-periodic X-Trans method implemented here. That is a search result, not a
novelty claim; a professional claim-by-claim patent search would be broader.

## 2. Papers and sources examined

### 2.1 Primary comparison sources

- Dan Su and Philip J. Willis, **“Demosaicing of Colour Images Using Pixel
  Level Data-Dependent Triangulation,”** TPCG 2003, pp. 16--25,
  [DOI 10.1109/TPCG.2003.1206926](https://doi.org/10.1109/TPCG.2003.1206926).
  The full conference paper is reproduced at the end of Su's 2003 Bath PhD
  thesis. Thesis Chapter 5 supplies clearer formulas and boundary details than
  the typeset conference copy.

- Jean Yves Aristide Yao, Kacoutchy Jean Ayikpa, Pierre Gouton, and Tiemoman
  Kone, **“Enhancing multispectral image demosaicking from MSFA sensors using a
  triangular structure-aware bilinear interpolation: Application to one-shot
  and multi-shot systems,”** *Optics Communications* 605 (2026), 132842,
  [DOI 10.1016/j.optcom.2025.132842](https://doi.org/10.1016/j.optcom.2025.132842).
  Received 20 September 2025, accepted 26 December 2025, available online 27
  December 2025; © 2025 the authors, CC BY 4.0. The complete 18-page publisher
  PDF from the user-supplied ScienceDirect archive was examined.

### 2.2 Supporting primary sources

The backward citation trace included Su's thesis and general DDT source,
earlier color-difference methods, the MSFA SD/ISD sources cited by Yao et al.,
X-Trans-specific papers, and relevant patent families. Full references appear
in Section 16.

### 2.3 Access and verification notes

- Su's paper metadata was checked against the University of Bath record; the
  paper text was checked in the author thesis copy, not only against an
  abstract.
- Yao et al. was initially available as complete author-posted text while the
  direct PDF request was blocked. The supplied ScienceDirect archive later
  provided the complete publisher PDF and resolved that access limitation.
- Equation (4) of Yao et al. specifies three nonnegative weights summing to one,
  but does not state an explicit coordinate formula for them. “Convex
  three-vertex interpolation” is verified. Calling them standard barycentric
  coordinates is a plausible interpretation, but is not explicitly established
  by the equation or surrounding prose.

## 3. Exact reconstruction of our two algorithms

The implementation at the experiment commit is unchanged in the relevant files.

### 3.1 Infinite periodic lattices and offline geometry

The canonical X-Trans cell is

```text
G B G G R G
R G R B G B
G B G G R G
G R G G B G
B G B R G R
G R G G B G
```

The implementation evidence is concentrated in
`rtengine/xtrans_triangulation.cc:1-292` (tables, boundary extension, and
evaluation), `rtengine/xtrans_triangulation.cc:335-477` (both reconstruction
variants), `rtengine/xtrans_cfa.cc:10-57,143-162` (canonicalization and coordinate
mapping), and `tests/xtranstriangulation/triangulation_tests.cc:271-304` (the 18
unique representations).

For each color `c`, all integer points whose periodic CFA color is `c` form an
infinite same-color lattice `S_c`. A Delaunay triangulation is constructed for
each infinite periodic lattice. The runtime does not triangulate an image.

Because regular periodic point sets contain cocircular subsets, Delaunay
topology is not always unique. The offline construction assigns phase
`p = 6*(y mod 6) + (x mod 6)` and applies the deterministic symbolic
perturbation

```text
dx = 1e-6 * (((17*p + 11) mod 37) / 37 - 1/2)
dy = 1e-6 * (((29*p +  7) mod 41) / 41 - 1/2)
```

only while choosing topology. The compiled vertices retain their original
integer coordinates and the compiled barycentric weights are evaluated on those
unperturbed coordinates. This is a geometric tie-break, not edge evidence.

### 3.2 Meaning of the 36-phase stencil tables

There is one table of 36 stencils per color. The table index is the target's
absolute canonical phase `(y mod 6)*6 + (x mod 6)`. A stencil contains one,
two, or three `(dx, dy, weight)` entries:

- one entry `(0,0,1)` is an identity stencil at a native sample;
- two entries mean that the target lies exactly on a Delaunay edge and the
  third barycentric coordinate is zero;
- three entries are genuine two-dimensional triangle interpolation.

Offsets are relative to the target and may leave its displayed 6×6 cell. Those
vertices intentionally refer to neighboring repetitions of the infinite lattice,
so triangles are continuous across cell boundaries. In particular, many missing
R/B phases use three unequal or equal geometric weights; this is not merely a
Bayer midpoint average disguised as barycentric interpolation.

### 3.3 Runtime interpolation and boundaries

At an interior target, the runtime takes the fixed weighted sum of values at the
listed same-color vertices. Weights and vertices do not depend on the mosaic
values. Native samples are copied directly rather than recomputed.

If a listed vertex is outside the finite image, it is replaced by the nearest
in-frame native sample of the same color in canonical Euclidean distance. Ties
are resolved by lower canonical row and then lower column. This is constant
extension of each sparse color lattice, not reflection, zero fill, or use of a
differently colored sample.

### 3.4 CFA transformations

`XTransCfaView` tests the eight dihedral coordinate matrices in a fixed order,
then all 6×6 translations. Symmetry collapses these combinations to 18 unique
accepted representations. It maps the finite actual rectangle into canonical
coordinates and maps output positions back without exchanging R and B.

### 3.5 Independent RGB variant

For every output position `p` and channel `c`:

- if `p` is a native `c` site, copy the mosaic value exactly;
- otherwise evaluate the phase stencil for `c` from native `c` mosaic samples.

The three channels never influence one another.

### 3.6 Green/chroma variant

Pass 1 creates dense `Ghat`: native green is copied and missing green is
interpolated by the fixed green stencil. Consequently `Ghat` at every native red
or blue location is the two-point green estimate selected by that X-Trans phase.

Pass 2 copies a target's native R or B sample. At a missing red target it takes
the exact red stencil that the independent method would have used. At each red
stencil vertex `q` it evaluates `R(q)-Ghat(q)`, interpolates those differences
with the same weights, and adds the result to `Ghat(p)`. Blue is identical with
`B-Ghat`. Thus the sparse differences are attached to exactly the same native
R/B vertices and geometry as the independent variant.

Neither variant clips, adaptively weights, detects edges, reinjects non-native
samples, sharpens, denoises, suppresses false color, or performs a corrective
iteration. Inputs and outputs are checked for finite values. OpenMP changes only
row scheduling. Any reported failure causes the dispatch layer to run a complete
Markesteijn three-pass reconstruction over the result.

## 4. Su--Willis algorithm reconstruction

### 4.1 Geometry and topology

Su--Willis applies pixel-level DDT to Bayer. R and B each form a regular square
grid; G forms a square grid rotated by 45 degrees. Each 2×2 square of one
same-color mesh has two possible diagonals. For corner values `a,b,c,d`, the
basic model compares the two opposite-corner differences and connects the pair
with the smaller absolute difference:

```text
choose diagonal a--c if |a-c| < |b-d|
choose diagonal b--d if |b-d| < |a-c|
the examined text does not specify the equality tie-break
```

The intent is to isolate an outlier and make the diagonal follow, rather than
cross, a local image edge. The optional extended model (thesis Section 3.2.6)
can replace a central choice when at least six of eight neighboring squares
support the other orientation. This affects topology construction, not the later
interpolation formula.

The resulting one-bit-per-square lookup tables are image-specific. They may make
runtime interpolation cheap, but unlike our tables they must be initialized from
each image's pixel values.

### 4.2 Original-color demosaicing

For the Bayer geometry examined by Su--Willis, every missing sample lies on a
triangle boundary. Triangle interpolation therefore degenerates to the average
of two selected vertices. Missing R/B at some positions use a fixed row/column
edge; the ambiguous diagonal case uses the data-dependent diagonal. Missing G
at R/B uses one of the horizontal or vertical green pairs chosen by its green
mesh. There is no genuine three-positive-weight target in their Bayer
demosaicing formulas, although their general image-resampling method supports
triangle interiors.

This is close to our independent variant at the level “one same-color mesh per
plane, interpolate missing plane values.” It differs in CFA geometry, topology
selection, target location, and weights.

### 4.3 Color-difference demosaicing

Su--Willis uses `K_R=G-R` and `K_B=G-B`, the sign opposite to our stored
differences but algebraically equivalent when reconstruction uses subtraction.
The detailed thesis procedure (Chapter 5, pp. 66--68) is not simply “reconstruct
one dense G, then sample differences at native R/B sites”:

- at an R site, `K_R` uses the average of four surrounding measured G samples
  minus measured R; the B-site formula is analogous for `K_B`;
- at a G site, `K_R`/`K_B` use measured G minus the average of two surrounding
  R/B samples;
- DDT-selected pairs of these differences are averaged;
- missing G at an R/B site is reconstructed from the measured chroma plus an
  interpolated difference; missing R/B is reconstructed from G minus an
  interpolated difference.

The conference paper compresses these details, while the thesis prints the site
formulas. Su--Willis therefore has difference estimates at multiple Bayer site
types and reconstructs channels in a coupled order. Our method has one dense
green scaffold and defines each difference only where that chroma is measured.

### 4.4 Boundaries and postprocessing

Su's thesis supplies explicit boundary-specific averages along the same image
boundary line, nearest opposite-chroma copying where needed, and subsequent
cross-chroma calculation. This differs from our uniform nearest-native
sparse-lattice extension. The basic DDT and DDT-difference demosaicers have no
post-demosaic refinement; the optional neighboring-square vote is a topology
preparation pass.

## 5. Yao et al. TriSBI and spectral-difference reconstruction

### 5.1 TriSBI

The paper uses a 4×4 MSFA with eight bands, each appearing with probability
1/8. For each sparse band `Itilde_i`, Section 3.4 defines valid coordinates `P`,
their spectral values `V`, and missing coordinates `X`, then:

1. constructs a Delaunay triangulation on the finite valid points `P`;
2. finds the containing three-vertex triangle for each missing point;
3. evaluates Equation (4),
   `Ihat(x,y)=sum(j=1..3, lambda_j*Itilde(x_j,y_j))`, where
   `lambda_j >= 0` and `sum(lambda_j)=1`.

This is geometry-driven and does not use sample values to select topology or
detect edges. It is the closest examined paper to our independent variant's
geometry-only per-band Delaunay interpolation. Unlike our method, it describes a
finite triangulation, not an infinite periodic one or phase tables. It is defined
only inside the convex hull; the paper returns zero at edges or very empty
regions outside it.

The method name contains “bilinear,” but the authors themselves distinguish the
three-vertex triangular formula from conventional WB. The exact derivation of
`lambda` is not printed, so equivalence to our coordinate-derived barycentric
weights cannot be established operation by operation.

### 5.2 Spectral Difference and ISD

Section 3.7 gives the motivation directly: neighboring spectral bands are
strongly correlated and their difference varies more smoothly than the bands.
For target band `i` and a spatially reconstructed band `j`, Equation (14) forms

```text
Delta_tilde(i,j) = Itilde_i - Ihat_B^j * M_i
```

at positions measured by mask `M_i`. The prose then explicitly says each
spectral difference is interpolated **using a bilinear method** to obtain a dense
`Delta_hat(i,j)`. Equation (15) combines masked samples and difference estimates
from `k` reference bands with a luminance scale `tau`.

The paper also reports experiments where TriSBI replaces WB as the spatial
kernel used to create the spatial band estimates feeding SD/ISD. That does not
make Equation (14)'s residual interpolation itself Delaunay: the published
Section 3.7 still calls that step bilinear. This is an important difference from
our chroma method, which uses the R/B triangle stencils directly on the
differences.

ISD (Section 3.8, following Mizutani et al.) iteratively updates differences
according to spectral distance. Neither our variant nor non-iterative SD has
such refinement.

## 6. Operation-by-operation comparison matrix

### 6.1 Geometry and interpolation

| Operation | Our independent RGB | Our green/chroma | Su--Willis 2003 | Yao et al. 2026 | Assessment / significance |
|---|---|---|---|---|---|
| Filter geometry | 6×6 RGB X-Trans | Same | 2×2 Bayer | 4×4, eight-band MSFA | Different sampling problem. |
| Periodic vs finite geometry | Infinite periodic lattice, clipped to finite image at evaluation | Same | Periodic Bayer topology applied to finite image | Finite valid point set per band | Our infinite-periodic construction is implementation-specific. |
| Per-channel lattice | One irregular lattice for each R/G/B | Same | Regular R/B grids and rotated regular G grid | One sparse point set per band | Same broad decomposition. |
| Triangle vertices | Offline from CFA coordinates | Same for G and corresponding R/B differences | Two diagonal candidates in each same-color square | Finite Delaunay neighbors | Su vertices are constrained by Bayer squares; Yao is closer geometrically. |
| Triangulation criterion | Delaunay empty-circle geometry | Same | Pixel-level data-dependent diagonal, not Delaunay | Delaunay | Our geometry criterion matches Yao in kind, not Su. |
| Ambiguity handling | Deterministic phase-periodic symbolic perturbation | Same | Sample-difference comparison; equality behavior not fully specified | Delaunay tie handling not specified | Our cocircular rule is absent from both papers. |
| Content-dependent topology | No | No | Yes | No | The central distinction from Su. |
| Edge detection | None | None | Implicit in diagonal difference; optional neighbor vote | None | Yao later acknowledges lack of edge alignment. |
| Triangle selection at target | 36-phase table lookup | Same | Lookup image-specific diagonal table | Containing-triangle search | Same role, different realization and cost. |
| Target location | Identity, Delaunay edge, or true triangle interior | Same | Missing Bayer values always on an edge | Three-vertex containing triangle | Genuine 3-vertex R/B interpolation links ours more closely to MSFA TriSBI. |
| General barycentric interpolation | Yes, exact affine weights on integer vertices | Yes | General parent method yes; Bayer demosaic reduces to two points | Three convex weights; formula not derived | Exact weight equivalence to Yao is unclear. |
| Two-point special case | Some R/B phases; every missing G | Same | Every missing value | Not singled out | Bayer simplification does not describe all X-Trans phases. |
| Weight source | Geometry only | Geometry only | `1/2`; image values choose which pair | Nonnegative three weights sum to one | Su values affect support, not the final `1/2` weights. |
| Native samples | Explicit direct copy | Explicit direct copy | Measured primary retained | TriSBI defines interpolation only for missing set; SD's final equation does not state a separate reinjection invariant | Our preservation rule is explicit. |
| Neighboring periodic cells | Offsets may cross the 6×6 boundary | Same | Local Bayer squares naturally repeat | Finite mesh, no periodic-cell stencils described | X-Trans-specific table consequence. |
| Boundary handling | Nearest in-frame native sample of same color; deterministic ties | Same | Special same-boundary averages/copies | Zero outside convex hull | All four differ materially. |
| Orientation handling | 18 phase/orientation representations canonicalized | Same | One Bayer layout; rotations/reflections not discussed | One MSFA layout; transforms not discussed | RawTherapee integration requirement, not a new interpolation principle. |

### 6.2 Channel and band correlation

| Operation | Our independent RGB | Our green/chroma | Su--Willis 2003 | Yao et al. 2026 | Assessment / significance |
|---|---|---|---|---|---|
| Green/reference first | No | Yes, one dense `Ghat` | Coupled difference construction; no identical single `Ghat` pass | Spatial estimate of reference band(s) before SD | Established concept, different ordering. |
| Reference choice | None | Green | Green-centered `G-R`, `G-B` | Any other spectral band `j`; many references | Green is natural for RGB/X-Trans density, but not unique prior art. |
| Difference sign | None | `R-Ghat`, `B-Ghat` | `G-R`, `G-B` | `band_i - estimated band_j` | Sign is algebraically immaterial if reconstruction is consistent. |
| Where reference is obtained | N/A | Fixed green triangulation at every position | Four measured G around R/B; at G, measured G | Spatial reconstruction such as WB or TriSBI | Our one-scaffold design is not Su's sitewise construction. |
| Where differences exist | N/A | Native R sites or native B sites only | Constructed at several Bayer site types | Native positions of target band `i` | Ours and Yao share masked sparse residuals in broad form. |
| Difference geometry | N/A | Exactly same R/B Delaunay vertices and weights as independent RGB | DDT-selected pairs in corresponding color meshes | Published SD says bilinear residual interpolation | Exact same-stencil reuse was not found in the two papers. |
| Reconstruction | N/A | `Ghat + Interp(R-Ghat)` and blue analogue | `G-K_R`, `G-K_B`, or measured R/B plus K to recover G | Scaled sum over reference bands and masks, Eq. (15) | Same residual-addition family, different operands. |
| Iteration | No | No | No corrective iteration; optional topology voting pass | SD no; ISD yes | Our method is closest to non-iterative residual schemes. |
| Fine chromatic detail | No correlation model; independent aliasing | Assumes local smoothness of R-G/B-G; no exception | Same smooth-difference assumption, with edge-aligned support | Same inter-band smoothness motivation; sparse same-band support remains limiting | Predicts our colored-checkerboard regression. |

### 6.3 Engineering and complexity

| Operation | Our independent RGB | Our green/chroma | Su--Willis 2003 | Yao et al. 2026 | Assessment / significance |
|---|---|---|---|---|---|
| Geometry construction time | Offline once | Offline once | Per image, linear scan; optional second pass | Finite Delaunay construction; implementation complexity not specified | Our runtime is a fixed stencil filter. |
| Stored topology | Three compile-time 36-entry tables | Same | Three per-image one-bit square tables | No periodic lookup representation described | Periodic phase tables are not a general novelty by themselves; phase filters are known elsewhere. |
| Runtime order | One image pass, constant work per channel | Dense G pass then R/B pass | Build topology then selected pair averages | Build/search mesh; typical Delaunay cost is superlinear in sample count, but paper gives no implementation bound | All are practical piecewise-linear methods, but cost structures differ. |
| Clipping / denoising / sharpening | None | None | None in compared variants | None in TriSBI/SD formulas | No hidden quality stage explains differences. |
| Corrective postprocess | None | None | None | ISD is a separately evaluated iterative extension | Do not attribute ISD behavior to standalone TriSBI. |
| Failure policy | Loud Markesteijn overwrite in RawTherapee | Same | Not discussed | Outside-hull zero | Product integration, not literature-level interpolation. |

## 7. Normalized pseudocode comparison

Let `S_c` be measured positions for channel/band `c`, `m(q)` the mosaic
measurement, `T_c` a triangulation, and `Interp(T, values, p)` triangle
interpolation at `p`.

### 7.1 Our independent X-Trans triangulation

```text
offline, for c in {R,G,B}:
    S_c := infinite periodic X-Trans positions of c
    T_c := Delaunay(S_c), with deterministic periodic cocircular tie-break
    W_c[0..35] := integer-target vertices and barycentric weights from T_c

for each finite pixel p:
    for c in {R,G,B}:
        if p in S_c:
            Chat_c(p) := m(p)
        else:
            Chat_c(p) := sum((q,w) in W_c[phase(p)], w * m(boundary_c(q)))
```

### 7.2 Our green plus color-difference triangulation

```text
for each p:
    Ghat(p) := m(p) if p in S_G else Interp(W_G, m|S_G, p)

for c in {R,B}:
    for q in S_c:
        D_c(q) := m(q) - Ghat(q)
    for each p:
        Chat_c(p) := m(p) if p in S_c
                       else Ghat(p) + Interp(W_c, D_c, p)
```

### 7.3 Closest Su--Willis procedures

Original-color variant:

```text
for c in {R,G,B}:
    for each 2x2 square Q of the Bayer same-color grid:
        choose diagonal a-c if |value(a)-value(c)| is smaller
        choose diagonal b-d if |value(b)-value(d)| is smaller
        // equality tie-break is not specified in the examined text
        store one topology bit in L_c[Q]

for each p and each missing c:
    (q1,q2) := pair selected by L_c for p
    Chat_c(p) := (m(q1) + m(q2)) / 2
retain the measured primary at p
```

Color-difference variant, preserving its different data construction:

```text
construct image-dependent topology tables L_R, L_G, L_B

at native R q: K_R(q) := mean(four surrounding measured G) - R(q)
at native B q: K_B(q) := mean(four surrounding measured G) - B(q)
at native G q: K_R(q) := G(q) - mean(two surrounding measured R)
               K_B(q) := G(q) - mean(two surrounding measured B)

use the relevant L table to average the selected pair of K values
recover missing G from measured R/B plus K
recover missing R/B from G minus K_R/K_B
```

### 7.4 Yao TriSBI plus SD

```text
for each band i:
    P_i := finite measured coordinates selected by mask M_i
    T_i := Delaunay(P_i)
    for p inside convex_hull(P_i) and missing from band i:
        triangle := containing triangle in T_i
        Ihat_B_i(p) := sum(j=1..3, lambda_j * Itilde_i(vertex_j))
    outside convex hull: Ihat_B_i(p) := 0

for each target band i and reference band j:
    Delta_tilde(i,j) := (Itilde_i - Ihat_B_j) * M_i
    Delta_hat(i,j) := BilinearInterpolate(Delta_tilde(i,j))

Ihat_SD_i := tau * sum over j of the masked reference contribution
             corrected by Delta_hat(i,j)       // paper Eq. (15)

optional ISD only:
    repeat selected difference updates according to spectral distance
```

The final line is intentionally not rewritten as `reference + residual` at every
pixel: Equation (15) is a masked multi-reference sum and forcing it into our
single-green formula would hide a real difference.

## 8. What is clearly established prior art

- Linear interpolation on triangular meshes and Delaunay meshes predates both
  named demosaicing papers. Dyn--Levin--Rippa established data-dependent
  triangulation for piecewise-linear interpolation in 1990; Amidror's 2002
  survey treats linear triangular interpolation as standard scattered-data
  interpolation.
- Su--Willis explicitly represents Bayer R, G, and B samples as separate meshes
  and reconstructs missing primaries from triangles. Therefore the independent
  method's high-level recipe is not new in substance.
- Yao et al. explicitly performs per-band Delaunay triangulation and
  three-vertex convex interpolation for filter-array reconstruction. This is an
  even closer precedent for geometry-only triangular interpolation on a sparse,
  non-Bayer spectral layout.
- Green-first and color-difference interpolation are established CFA practice.
  Freeman's median method, Laroche--Prescott's three-pass method,
  Hamilton--Adams, Pei--Tam's `K_R=G-R`, `K_B=G-B` model, and Kimmel all
  precede this experiment.
- Brauers--Aach SD and Mizutani et al. ISD establish the analogous use of smooth
  inter-band differences for MSFA demosaicing.
- Precomputing different linear operations for each phase of a periodic
  non-Bayer CFA is also broadly known. WO2020139493A1, for example, describes
  `p^2` phase-indexed local resampling filters derived from a model. It is not
  triangulation, but it prevents treating phase-indexed precomputation alone as
  a strong research distinction.

Su--Willis was not the origin of DDT or triangle interpolation. Its contribution
was a simple pixel-level, edge-selected diagonal model and its application to
Bayer demosaicing, including color-difference space.

## 9. What differs because of X-Trans

- **Irregular same-color geometry:** R/B do not form Bayer's simple square
  sublattices. Missing targets can lie genuinely inside triangles, so three
  positive barycentric weights are mathematically substantive, not just an
  implementation generalization of a two-point average.
- **Six-by-six periodicity:** 36 target phases are needed for a phase-complete
  fixed implementation. This is substantive for support/weight selection but
  mechanical once the triangulation is chosen.
- **Cross-cell support:** a containing triangle near a cell edge can use native
  vertices in adjacent periods. Encoding such offsets is required to represent
  the infinite lattice correctly.
- **Cocircular ambiguity:** regular periodic geometry can admit multiple valid
  Delaunay triangulations. A deterministic, translation-compatible choice is
  necessary for reproducible fixed stencils, but the particular symbolic
  perturbation is an engineering choice rather than a new interpolation theory.
- **CFA canonicalization:** cameras and processing transforms expose translated,
  rotated, or reflected 6×6 matrices. Supporting the 18 representations is a
  RawTherapee integration requirement; it does not change the mathematics.
- **Dense green scaffold:** X-Trans has 20 green samples versus eight each of R
  and B per cell, making green the best-sampled fixed reference. Green-centered
  chroma is nevertheless established RGB prior art, not an X-Trans invention.

## 10. What appears implementation-specific

The following details were not present in the two primary papers and appear to
belong to this implementation:

- the exact canonical 6×6 coordinate convention;
- the phase formula and symbolic perturbation constants;
- compiling exact integer-coordinate barycentric stencils for all three infinite
  lattices;
- reuse of the exact R/B vertex/weight stencil for the sparse `R-Ghat` or
  `B-Ghat` field;
- nearest-native sparse-lattice extension with row/column tie-breaking;
- deterministic search order over dihedral transforms and translations;
- finite/non-finite validation, OpenMP row scheduling, and loud Markesteijn
  fallback.

These are concrete engineering distinctions. They should not individually be
described as novel without a wider search.

## 11. Search beyond the two papers

### 11.1 X-Trans-specific academic and patent search

Searches combined X-Trans with Delaunay, triangulation, piecewise-linear,
barycentric, green-first, and color-difference terms. The closest X-Trans works
found were:

- Rafinazari--Dubois (2014), a frequency-domain X-Trans analysis with adaptive
  and non-adaptive chroma reconstruction, not a triangular mesh method;
- Zhang et al. (2016), a universal method evaluated on X-Trans that estimates
  inter-pixel chrominance with distance and edge weights and a pseudoinverse
  color transform, not Delaunay interpolation;
- Fujifilm's X-Trans patent family, which specifies the repeating 6×6 CFA and
  describes four-direction correlation interpolation, not Delaunay or
  barycentric reconstruction.

No X-Trans-specific triangulation reconstruction matching the experiment was
found in the searched academic or patent records.

### 11.2 Multilingual searches

Targeted searches used, among others, `去马赛克 / 德劳内三角剖分` (Chinese),
`デモザイク / ドロネー三角形補間` (Japanese),
`Demosaicing / Delaunay-Dreiecksinterpolation` (German), and
`dématriçage / triangulation de Delaunay` (French), together with X-Trans,
CFA/MSFA, piecewise-linear, color-difference, and spectral-difference terms.

Materially relevant non-English records were limited:

| Original title | English title | Language/year | Actual relevance |
|---|---|---|---|
| `カラー撮像素子` (JP5095040B1 family) | “Color imaging element” | Japanese, 2012 | Fujifilm X-Trans 6×6 CFA family; directional correlation interpolation, no triangulation. Its Chinese/European/US records are family counterparts, not independent inventions. |
| `Détection des contours à partir d'images CFA de Bayer` | “Edge detection from Bayer CFA images” | French, Arezki Aberkane thesis, 2017 | Reviews Bayer demosaicing and color-difference concepts, but its contribution is direct edge detection on mosaics, not triangular demosaicing. |

Brauers--Aach came from a German venue but is written in English; Mizutani et
al. is a Japanese research contribution published in English. They are included
for their SD/ISD operations rather than language coverage. The searches found
no materially overlapping independent Chinese, Japanese, German, or French
fixed-Delaunay X-Trans demosaicer. Search-engine indexing and translation make
that a bounded finding, not proof of absence.

## 12. Relation to the experimental results

The measured behavior follows the prior-art models closely.

| Challenge | Independent RGB RMS | Green/chroma RMS | Interpretation |
|---|---:|---:|---|
| Vertical neutral edge | 0.037696 | 0 | Dense green carries the shared edge structure; smooth differences need not recreate it independently. |
| Horizontal neutral edge | 0.040298 | 0 | Same. |
| 45-degree neutral edge | 0.052177 | 0.035911 | Difference space helps, but fixed triangles can still cross the oblique edge. |
| 1-pixel neutral checkerboard | 0.601898 | 0.520349 | Shared structure helps, but sampling is far beyond what fixed sparse lattices can recover. |
| Diagonal lines | 0.439243 | 0.285466 | Strong benefit from correlated reconstruction, incomplete without edge adaptation. |
| Radial pattern | 0.359617 | 0.243803 | Same trend across changing orientations. |
| Colored checkerboard | 0.414932 | 0.434924 | The deliberately non-smooth chroma difference violates the model and makes the correlated method worse. |

This is precisely the tradeoff described by Pei--Tam and Su--Willis: color
differences are relatively flat in smooth natural regions, but not reliably so at
chromatic transitions. Su--Willis tries to avoid interpolating across those
transitions by changing the diagonal from image values. Our geometry cannot do
that.

On the `DSCF0771.RAF` crop, normalized RMS from Markesteijn improves from
`0.007335` for independent RGB to `0.005649` for green/chroma. The visual change
from fine alternating rainbow texture to broader green/yellow/purple earring
segments is also consistent: channel coupling suppresses independent phase
aliasing, but a fixed triangle that straddles a high-frequency metallic edge
spreads a wrong residual over a larger region. Yao et al., Section 5.2, makes the
same geometric warning: triangular relations cannot replace nearby same-band
samples, are not inherently aligned to scene features, and can yield zigzags or
local distortions at abrupt changes.

The result therefore supports neither “triangulation is ineffective” nor “adding
color differences solves it.” It supports the narrower conclusion that fixed
piecewise-linear geometry is a useful baseline, color-difference space is a
meaningful improvement on correlated neutral structure, and the missing
content-adaptive support remains visible on difficult real edges.

## 13. Patent considerations

The closest patent records fall into three groups:

1. **Established Bayer color-difference and adaptive interpolation:** Freeman
   US4724395A, Laroche--Prescott US5373322A, and Hamilton--Adams US5629734A
   disclose older color-correlation and directional reconstruction concepts.
2. **The X-Trans CFA family:** US20130048833A1 / US9313466B2, WO2012120705A1,
   JP5095040B1, EP2685711B1, and CN102870405B are jurisdictional members of one
   Fujifilm family, not five independent algorithms. The family describes 6×6
   periodic geometry and directional/correlation interpolation, but the examined
   text does not disclose Delaunay or barycentric stencils.
3. **Generic periodic non-Bayer conversion:** WO2020139493A1 describes
   phase-indexed local linear operations for periodic non-Bayer arrays, derived
   from a statistical/MAP model. It is adjacent to periodic precomputation but
   not the same mathematical reconstruction.

Patent status labels, claim scope, continuation families, and national law
require specialist analysis. The observations above compare disclosed
operations only. In particular, absence of the word “Delaunay” does not establish
non-infringement, and an expired US record says nothing by itself about foreign
family claims or later patents.

## 14. Open questions and useful next experiments

These are validation questions, not proposals for a third algorithm:

- Regenerate every compiled stencil from an independent exact-arithmetic tool
  and compare topology, weights, and phase continuity byte for byte.
- Visualize the fixed R/B triangle crossing the known earring artifact and
  compare it with the diagonal Su--Willis would choose from the local values.
  This would test the causal edge-crossing explanation without changing code.
- Separate interior and boundary metrics to quantify the contribution of our
  nearest-native extension relative to core interpolation.
- Use synthetic ground truth whose neutral and chromatic edges sweep all angles
  and all 36 phases, reporting error by triangle topology rather than only by
  image.
- If the exact TriSBI weight-generation code becomes available, determine
  whether Equation (4)'s `lambda` values are true barycentric coordinates and
  compare them to our stencil generator on an equivalent point set.
- Search patent claims professionally before attaching novelty, clearance, or
  distribution conclusions to the implementation-specific combination.

## 15. Provenance map

### Clearly established prior art

- per-channel/per-band triangular interpolation for CFA/MSFA reconstruction;
- data-dependent triangulation for piecewise-linear interpolation;
- Bayer pixel-level DDT;
- green/reference-first color-difference demosaicing;
- the smooth-local-color/spectral-difference assumption;
- interpolation and re-addition/subtraction of color/spectral residuals;
- phase-specific precomputed linear operations for periodic CFAs.

### Established concept, different realization

- geometry-only per-band Delaunay interpolation: explicit in Yao et al., but not
  as infinite periodic 6×6 compiled stencils;
- triangle interpolation in Su--Willis: image-selected Bayer edges and two-point
  averages rather than fixed X-Trans three-vertex stencils;
- spectral/color differences: Su and Yao construct and combine them differently
  from one dense `Ghat` plus sparse native chroma residuals.

### X-Trans-specific engineering adaptation

- the three irregular periodic color lattices and 36 phases;
- cross-cell triangle vertices;
- deterministic resolution of periodic cocircular Delaunay faces;
- handling 18 translated/rotated/reflected X-Trans representations;
- using the same R/B phase stencil on the corresponding sparse residual field.

### Not found in the examined literature

- the complete fixed-periodic X-Trans Delaunay/barycentric algorithm;
- the particular symbolic perturbation and exact 36-phase tables;
- the exact combination `fixed X-Trans Ghat -> native R-Ghat/B-Ghat -> same
  R/B Delaunay stencil -> add to Ghat`;
- the stated sparse-lattice boundary extension and RawTherapee orientation/failover
  contract.

### Potentially interesting research distinctions, not novelty claims

- genuine three-vertex interpolation at X-Trans R/B target phases, unlike the
  Bayer edge-degenerate formulas in Su--Willis;
- deterministic infinite-periodic Delaunay topology as a small constant-time
  baseline for comparing adaptive X-Trans methods;
- the empirical conversion of fine phase-alternating error into broader
  edge-crossing residual error when moving from independent to correlated
  interpolation.

## 16. Bibliography

1. D. Su and P. J. Willis, “Demosaicing of Colour Images Using Pixel Level
   Data-Dependent Triangulation,” *Theory and Practice of Computer Graphics*,
   2003, pp. 16--25. [doi:10.1109/TPCG.2003.1206926](https://doi.org/10.1109/TPCG.2003.1206926).
2. D. Su, *Pixel Level Data-Dependent Triangulation with Its Applications*, PhD
   thesis, University of Bath, 2003. [Bath author copy](https://purehost.bath.ac.uk/ws/portalfiles/portal/188132677/Dan_Su_thesis.pdf).
3. D. Su and P. Willis, “Image Interpolation by Pixel-Level Data-Dependent
   Triangulation,” *Computer Graphics Forum* 23(2), 2004, pp. 189--201.
   [doi:10.1111/j.1467-8659.2004.00752.x](https://doi.org/10.1111/j.1467-8659.2004.00752.x).
4. N. Dyn, D. Levin, and S. Rippa, “Data Dependent Triangulations for
   Piecewise Linear Interpolation,” *IMA Journal of Numerical Analysis* 10(1),
   1990, pp. 137--154. [doi:10.1093/imanum/10.1.137](https://doi.org/10.1093/imanum/10.1.137).
5. I. Amidror, “Scattered Data Interpolation Methods for Electronic Imaging
   Systems: A Survey,” *Journal of Electronic Imaging* 11(2), 2002,
   pp. 157--176. [doi:10.1117/1.1455013](https://doi.org/10.1117/1.1455013).
6. W. T. Freeman, “Median Filter for Reconstructing Missing Color Samples,”
   US4724395A, 1988. [Google Patents](https://patents.google.com/patent/US4724395A/en).
7. C. A. Laroche and M. A. Prescott, “Apparatus and Method for Adaptively
   Interpolating a Full Color Image Utilizing Chrominance Gradients,”
   US5373322A, 1994. [Google Patents](https://patents.google.com/patent/US5373322A/en).
8. J. F. Hamilton, Jr. and J. E. Adams, Jr., “Adaptive Color Plane
   Interpolation in Single Sensor Color Electronic Camera,” US5629734A, 1997.
   [Google Patents](https://patents.google.com/patent/US5629734A/en).
9. S.-C. Pei and I.-K. Tam, “Effective Color Interpolation in CCD Color Filter
   Arrays Using Signal Correlation,” *IEEE Transactions on Circuits and Systems
   for Video Technology* 13(6), 2003, pp. 503--513.
   [doi:10.1109/TCSVT.2003.813422](https://doi.org/10.1109/TCSVT.2003.813422).
10. R. Kimmel, “Demosaicing: Image Reconstruction from Color CCD Samples,”
    *IEEE Transactions on Image Processing* 8(9), 1999, pp. 1221--1228.
    [doi:10.1109/83.784434](https://doi.org/10.1109/83.784434).
11. J. Y. A. Yao, K. J. Ayikpa, P. Gouton, and T. Kone, “Enhancing
    Multispectral Image Demosaicking from MSFA Sensors Using a Triangular
    Structure-Aware Bilinear Interpolation: Application to One-Shot and
    Multi-Shot Systems,” *Optics Communications* 605, 2026, 132842.
    [doi:10.1016/j.optcom.2025.132842](https://doi.org/10.1016/j.optcom.2025.132842).
12. J. Brauers and T. Aach, “A Color Filter Array Based Multispectral Camera,”
    *12. Workshop Farbbildverarbeitung*, Ilmenau, 2006, pp. 55--64.
    [Author PDF](https://www.lfb.rwth-aachen.de/bibtexupload/pdf/BRA06a.pdf).
13. J. Mizutani, S. Ogawa, K. Shinoda, M. Hasegawa, and S. Kato,
    “Multispectral Demosaicking Algorithm Based on Inter-Channel Correlation,”
    VCIP 2014, pp. 474--477.
    [doi:10.1109/VCIP.2014.7051609](https://doi.org/10.1109/VCIP.2014.7051609).
14. M. Rafinazari and E. Dubois, “Demosaicking Algorithm for the Fujifilm
    X-Trans Color Filter Array,” ICIP 2014, pp. 660--663.
    [doi:10.1109/ICIP.2014.7025132](https://doi.org/10.1109/ICIP.2014.7025132).
15. C. Zhang, Y. Li, J. Wang, and P. Hao, “Universal Demosaicking of Color
    Filter Arrays,” *IEEE Transactions on Image Processing* 25(11), 2016,
    pp. 5173--5186. [doi:10.1109/TIP.2016.2601266](https://doi.org/10.1109/TIP.2016.2601266).
16. Fujifilm Corporation, “Color Imaging Element,” US20130048833A1 / US9313466B2,
    priority 9 March 2011. [Google Patents](https://patents.google.com/patent/US20130048833A1/en).
17. “Systems and Methods for Converting Non-Bayer Pattern Color Filter Array
    Image Data,” WO2020139493A1, 2020.
    [Google Patents](https://patents.google.com/patent/WO2020139493A1/en).
18. K. F. Mulchrone, “Application of Delaunay Triangulation to the Nearest
    Neighbour Method of Strain Analysis,” *Journal of Structural Geology* 25,
    2003, pp. 689--702.
    [doi:10.1016/S0191-8141(02)00067-6](https://doi.org/10.1016/S0191-8141(02)00067-6).
    This is Yao et al.'s Delaunay citation [45], but it is a geology application,
    not CFA/MSFA reconstruction prior art.
