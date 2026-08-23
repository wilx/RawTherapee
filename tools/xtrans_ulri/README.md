# X-Trans ULRI reference-comparison experiment

This directory authenticates and executes version 1.0.0 of the MATLAB File
Exchange package *Unified Laplacian Residual Interpolation Demosaicing*, then
compares its X-Trans `slow` modes with RawTherapee's frozen Markesteijn and
corrected-final MLRI research baselines. It is development tooling only. It
does not add a demosaicing method to engine dispatch or the GUI.

## Reference identity and license

The reviewed package was published on 2025-10-14 under the BSD 3-clause
license. The external archive is authenticated before execution:

Source page: [Unified Laplacian Residual Interpolation
Demosaicing](https://www.mathworks.com/matlabcentral/fileexchange/182302-unified-laplacian-residual-interpolation-demosaicing).
The reviewed archive was obtained from the package's [versioned File Exchange
download](https://www.mathworks.com/matlabcentral/mlc-downloads/downloads/6fee805d-bc4d-4ada-aeaa-b721d372484b/837164e6-6dde-4386-a596-077cf3e07c65/packages/zip).

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `ulri-1.0.0.zip` | 8,214,511 | `0dab03107479153a68fe6feeef738e3fb9c058031634dc006e251f2fd7a81347` |
| `function_demosaic_x_trans.m` | - | `bc8a2a557a326af2ea72f7b77b17d55d6785570fa36197f732288a2d54024b69` |

The helper and license identities are frozen in `generate_reference.py`.
No upstream MATLAB implementation is copied here. The small Octave driver
loads the authenticated external source as an executable reference.

The development copy previously supplied for the RawTherapee MLRI experiment,
whose SHA-256 is
`055d1807729cbd556406bf695a6617f189d2376f7dbd35f4477f44e671fffa0c`,
is semantically identical to the published X-Trans source after ignoring
blank-line and line-ending differences. This identity is a central result of
the experiment: ULRI is the engineering reference from which the existing
MLRI implementation was reproduced, not an independent X-Trans architecture.

## Generate the official corpus

Ubuntu needs `octave` and `octave-image`. Generate from an authenticated,
external extraction of the official archive:

```sh
nice -n 10 .venv/bin/python -m tools.xtrans_ulri.generate_reference \
    --reference-root /tmp/ulri-1.0.0 \
    --archive /tmp/ulri-1.0.0.zip \
    --output /tmp/ulri-reference
diff -qr /tmp/ulri-reference tools/xtrans_ulri/reference_golden
```

The corpus covers eleven deterministic analytical scenes at `slow=0..3`.
Its canonical manifest SHA-256 is
`2441cb88637a5c50375093d11e97b6cb46912d68201b338c3f8d4a5c4e7114c3`.
The generator supplies `uint16` input because Octave 8 does not implement the
MATLAB `max(A,[],"all")` spelling used only by the source's floating-input
branch. The source itself is not patched; arithmetic therefore follows its
native `uint16 -> single 0..255 -> uint16` path.

## Build and run the comparison

```sh
/usr/bin/cmake --preset dev
nice -n 10 cmake --build build/dev \
    --target rawtherapee-xtrans-ulri-tests -j4
ctest --test-dir build/dev --output-on-failure -L xtrans-ulri -j4
```

Run the 81-case natural/synthetic comparison with the authenticated external
NASA Hydra star-field source used by the prior MLRI experiments:

```sh
nice -n 10 .venv/bin/python -m tools.xtrans_ulri.generate \
    --runner build/dev/tests/xtransulri/rawtherapee-xtrans-ulri-tests \
    --starfield /tmp/xtrans-danger-sources/grail_free_air_stars1.tif \
    --reference-golden tools/xtrans_ulri/reference_golden \
    --output /tmp/xtrans-ulri-results
```

The native development entry point exposes the source contract exactly:

- `slow=0`: one green pass at sigma 2 and direct final green-guided R/B;
- `slow=N`, for `N>0`: `N+1` green passes, sigma 2 then sigma 1, followed by
  the source's `sqrt(G/255)` blend of provisional and final R/B.

The official source hardcodes one 6x6 phase cell. The native harness uses the
already reviewed canonical-coordinate machinery and tests all 18 distinct
X-Trans translation cells. The official corpus itself therefore authenticates
the source's fixed cell; the C++ contract test covers generalized phase
handling.

See `devnotes/xtrans-ulri-reference-report.md` for the operation-level
comparison, measured results, and ARI decision.
