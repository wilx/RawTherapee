# X-Trans demosaicer complementarity experiment

This developer-only study measures error correlation and ground-truth oracle
headroom among the existing Markesteijn, corrected-final MLRI, triangulation,
global-B, and sparse-alias experiments. It changes no production demosaicing
behavior.

Build the native runner and generate the canonical report with:

```sh
nice -n 10 cmake --build build/dev -j4 \
    --target rawtherapee-xtrans-oracle-runner
.venv/bin/python -m tools.xtrans_oracle.generate \
    --runner build/dev/tests/xtransoracle/rawtherapee-xtrans-oracle-runner \
    --output devnotes/images/xtrans-oracle --force
```

The runner invokes the actual C++ implementations on one common float32 scalar
mosaic. The Python side adds sparse-alias recovery, performs the oracle
analysis, and writes only compact JSON/PNG research artifacts. Intermediate
mosaics and reconstructions stay in a temporary directory.

All comparisons use camera-linear values. The authenticated scikit-image
sources are decoded from sRGB before remosaicking. A 16-pixel border is omitted
uniformly. Patch oracles use nonoverlapping blocks anchored at the top-left of
the evaluated interior; incomplete blocks at the right and bottom remain
smaller but use one winner each.
