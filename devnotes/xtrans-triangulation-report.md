# Experimental X-Trans triangulation demosaicers

> Privacy update: DSCF0771 full-frame imagery is private and is not
> distributed. Only the reviewed earring crops may be shared. Historical
> full-frame asset references below describe private benchmark evidence,
> not current publication instructions.

## Purpose and scope

This experiment isolates two deliberately non-adaptive geometric baselines:

- `xtrans-triangulated-rgb` independently interpolates the measured R, G, and
  B lattices.
- `xtrans-triangulated-chroma` interpolates G, forms R-G and B-G at native
  red/blue sites, interpolates those differences with the same red/blue
  geometry, and adds them back to G.

Both are hidden PP3/CLI methods. Neither is in the method enum or GUI. They add
no edge direction, content-dependent weight, denoising, sharpening, median
filter, false-colour suppression, clipping, or measured-sample reinjection
beyond exact preservation of the channel actually observed at each CFA site.

## CFA geometry and triangulation

RawTherapee's canonical cell uses `0=R`, `1=G`, and `2=B`:

```text
G B G G R G
R G R B G B
G B G G R G
G R G G B G
B G B R G R
G R G G B G
```

The native coordinates within `(x,y)=0..5` are:

- R: `(4,0) (0,1) (2,1) (4,2) (1,3) (3,4) (5,4) (1,5)`
- G: `(0,0) (2,0) (3,0) (5,0) (1,1) (4,1) (0,2) (2,2)
  (3,2) (5,2) (0,3) (2,3) (3,3) (5,3) (1,4) (4,4) (0,5) (2,5)
  (3,5) (5,5)`
- B: `(1,0) (3,1) (5,1) (1,2) (4,3) (0,4) (2,4) (4,5)`

Each color uses a Delaunay triangulation of its infinite periodic sample
lattice. A fixed phase-periodic symbolic perturbation resolves cocircular
topology choices. For `p=6*(y mod 6)+(x mod 6)`, it uses
`dx=1e-6*(((17p+11) mod 37)/37-1/2)` and
`dy=1e-6*(((29p+7) mod 41)/41-1/2)`. Barycentric weights are then calculated at
the real, unperturbed integer pixel centers. Runtime code contains only the
resulting three 36-phase stencil tables. It does not depend on a triangulation
library.

Native sites use identity stencils. In addition, four missing red phases and
four missing blue phases lie exactly on triangle edges and therefore use two
nonzero weights. Every missing-green pixel center lies on an edge between two
green samples. Neighbor-cell offsets are retained in the tables; indexing by
the absolute canonical coordinate modulo six makes the geometry join across
every cell boundary.

RawTherapee's existing `XTransCfaView` maps all 18 accepted rotations,
reflections, and translations to these tables without swapping RGB meaning.
At a finite image boundary, an exterior triangle vertex uses the nearest
in-frame native sample of the same color. Distance is measured in the
canonical orthonormal coordinates and ties use canonical row then column.

## Native verification

The CTest label `xtrans-triangulation` covers:

- stable method/error contracts and rejection of bad CFA/non-finite input;
- bit-exact native-sample preservation for both variants;
- exact constant reconstruction;
- channel-specific linear-plane reconstruction away from the finite border;
- all 18 CFA mappings and coordinate orientations;
- neutral vertical, horizontal, 45-degree, and shallow edges;
- checkerboard, one-pixel horizontal/vertical lines, diagonal lines, and a
  radial high-frequency pattern;
- a high-frequency colored checkerboard that violates the chroma assumption.

The largest interior linear-plane errors in RawTherapee float units were
`0.00195312` for independent RGB and `0.000976562` for green/chroma.

Normalized neutral-scene false-colour RMS (`R-G` and `B-G` combined) was:

| Scene | Independent RGB | Green/chroma |
| --- | ---: | ---: |
| Vertical edge | 0.037696 | 0.000000 |
| Horizontal edge | 0.040298 | 0.000000 |
| 45-degree edge | 0.052177 | 0.035911 |
| Shallow edge | 0.043485 | 0.037543 |
| 1-pixel checkerboard | 0.601898 | 0.520349 |
| Vertical lines | 0.551977 | 0.415366 |
| Horizontal lines | 0.551934 | 0.416042 |
| Diagonal lines | 0.439243 | 0.285466 |
| Radial pattern | 0.359617 | 0.243803 |

Thus the green scaffold reduces false colour on every tested neutral structure,
and eliminates it on the two axis-aligned step edges. The expected countercase
also appears: normalized reconstruction RMS on the colored checkerboard rises
from `0.414932` to `0.434924`.

## DSCF0771 comparison

The reviewed RAW is `DSCF0771.RAF`, SHA-256
`26106d7da2ba9a87caebffd4bd5e5fb0371cc8a2b6139ec724b59ef4112842c0`.
All exports used the same neutral PP3 controls, 16 OpenMP threads, 16-bit
uncompressed TIFF, no false-colour suppression, and no sharpening or denoise.

| Method | Engine demosaic | Complete export | Peak RSS |
| --- | ---: | ---: | ---: |
| Independent RGB | 0.339 s | 3.82 s | 1,981,600 KiB |
| Green/chroma | 0.399 s | 3.81 s | 1,981,556 KiB |
| Markesteijn 3-pass | not separately instrumented | 4.06 s | 2,002,248 KiB |

The complete TIFFs remain external:

- `/tmp/xtrans-triangulation/DSCF0771-triangulated-rgb.tif`, SHA-256
  `28b1251fad5fe3c12779405494041a7cb44ad9152562bcf195b6179f7f257019`
- `/tmp/xtrans-triangulation/DSCF0771-triangulated-chroma.tif`, SHA-256
  `5cf8b14031912cbad5ebc8353e2d17e642c8d8a27b128f5496eca9a6f25c0cee`
- `/tmp/xtrans-triangulation/DSCF0771-markesteijn.tif`, SHA-256
  `bd801f51b116eec35ac13a98bc6a13940a7ea01b9f0d58a57c7a27ee934cfa16`

On the established `(3450,1750,700,500)` crop, chroma versus Markesteijn has a
normalized RMS pixel difference of `0.005649`, versus `0.007335` for independent
RGB. The crop mean-color differences from Markesteijn are also smaller for the
chroma variant. These are similarity measurements, not ground-truth quality.

Visual inspection of the 500% earring crop gives a mixed rather than a simple
win. Green/chroma removes much of independent RGB's fine alternating rainbow
pattern, consistent with the synthetic tests, but it turns part of it into
broader green/yellow/purple segments around the metallic edge. Markesteijn is
still substantially more neutral there. Experiment 2 therefore reduces
high-frequency false colour, but does not eliminate the real-image artifact;
it changes some of its spatial character.

The canonical numeric record and all asset hashes are in
`devnotes/images/xtrans-neural/DSCF0771/triangulation-manifest.json`. The two
side-by-side files use the order `independent | chroma | Markesteijn`:

- `DSCF0771-triangulation-comparison-full-third.png`
- `DSCF0771-triangulation-comparison-earring-500.png`

Per the experiment boundary, no third algorithm or corrective heuristic was
introduced.
