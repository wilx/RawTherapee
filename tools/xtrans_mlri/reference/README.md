# X-Trans MLRI reference subset

This directory contains the small helper subset used to execute the published
X-Trans MATLAB reference under GNU Octave.  The main function is intentionally
kept as an authenticated external input until the generated golden corpus is
reviewed; pass it to the generator with `--source`.

The authoritative source is *Unified Laplacian Residual Interpolation
Demosaicing*, File Exchange version 1.0.0, published 2025-10-14 by
Rainbow-Johnny-Johnny-Image-Processing-Lim.  It is distributed under the
3-clause BSD license copied to `licenses/MLRI_XTRANS_MATLAB_LICENSE`.

Reviewed source identities:

| File | Published SHA-256 |
| --- | --- |
| `function_demosaic_x_trans.m` | `bc8a2a557a326af2ea72f7b77b17d55d6785570fa36197f732288a2d54024b69` |
| `guidedfilter_MLRI_wei.m` | `c26f06bef57e6ce99fca8369d8dc0ad414674978175623d4a0e5a9b0db59d0fd` |
| `imgconv2.m` | `dd7b68011c2fc7e9a20c5688bbc92e4d51b8ab387575b6d73826043bba95e388` |
| `clip.m` | `f66bed703a4d95336832815d83f2d15fbc33004043920ad08847753fd2488022` |
| `license.txt` | `8da7c29a607b5d6450bb2dbf7b5c0a7a93cbe5bb69a8d35d0fb4de5d3940a7c4` |

The development copy supplied for the RawTherapee experiment has SHA-256
`055d1807729cbd556406bf695a6617f189d2376f7dbd35f4477f44e671fffa0c`.
It has the same 1,305 nonblank source lines as the published file; only blank
line formatting differs.

The helper copies in this directory add comments that document the operation
being tested, so their tracked identities deliberately differ from the
published files:

| Tracked helper | SHA-256 |
| --- | --- |
| `guidedfilter_MLRI_wei.m` | `0c21bb652ef26cbc1ce0b2c842bc0ced7f8dbc5e0d289d35195b11733ab42f74` |
| `imgconv2.m` | `7a86a12f1380f756ade16e2bc5ad554458bce6ed1a23ac10c200a7ece0b37c8d` |
| `clip.m` | `a39b7149522e02f72d362e15e06ff9cd29dc50fd472f76b6638dbe9a9ff3807d` |

The committed golden manifest SHA-256 is
`b39e6200a049a727a014faf560e8fe5bf94860991eb23ac63cc0c2699083e161`.
Two independent regenerations must be byte-identical to each other and to the
tracked corpus.

The reference requires Ubuntu packages `octave` and `octave-image`.  It is a
development oracle only and is never called by RawTherapee.
