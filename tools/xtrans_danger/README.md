# X-Trans MLRI danger-gate experiment

This development-only experiment freezes the previously selected 7x7
Markesteijn/MLRI blender and asks a narrower question: can inference-time
features detect sparse or impulsive chromatic cases where the blender is
meaningfully worse than Markesteijn?

The detector is trained only from the mosaic, CFA, Markesteijn output, and
corrected-final MLRI output. Ground truth creates labels and evaluates quality;
it is not a feature. Hubble, the independent NASA Hydra star field, astronaut,
brick, and page sources remain untouched until final evaluation.

Patch SSE labels are canonicalized below the tested danger resolution so that
last-bit OpenMP differences cannot perturb fitted artifacts. Complete repeated
runs must produce byte-identical manifests.

Build the research runner and run:

```sh
nice -n 10 cmake --build build/dev --target rawtherapee-xtrans-oracle-runner -j4
.venv/bin/python -m tools.xtrans_danger.generate \
  --runner build/dev/tests/xtransoracle/rawtherapee-xtrans-oracle-runner \
  --starfield /path/to/grail_free_air_stars1.tif \
  --force
```

The NASA source is external and authenticated; it is not committed. The tool
records deterministic JSON, cost curves, and held-out diagnostic maps under
`devnotes/images/xtrans-danger/`. No production demosaicing path is changed.
