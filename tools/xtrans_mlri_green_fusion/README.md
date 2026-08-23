# MLRI directional green-candidate fusion experiment

This development-only experiment reuses the exact corrected-final MLRI trace,
the 16 pass-0/pass-1 directional green candidates, and the unchanged native
`finalRedBlue()` implementation.  It measures pixel and spatial candidate
oracles, scalar convex-hull headroom, bounded pass blending, current-weight
temperature controls, and deterministic shared candidate-cost models under
the existing source-level train/validation/test split.

Build and test:

```sh
cmake --preset dev
nice -n 10 cmake --build build/dev \
  --target rawtherapee-xtrans-mlri-internal-tests -j4
ctest --test-dir build/dev --output-on-failure -L xtrans-mlri-internal
.venv/bin/python -m pytest -q tools/xtrans_mlri_green_fusion/tests
```

Run the complete experiment:

```sh
nice -n 10 .venv/bin/python -m tools.xtrans_mlri_green_fusion.generate \
  --runner build/dev/tests/xtransmlriinternal/rawtherapee-xtrans-mlri-internal-tests \
  --starfield /tmp/xtrans-danger-sources/grail_free_air_stars1.tif \
  --output devnotes/images/xtrans-mlri-green-fusion
```

The trace and final-stage substitution APIs are research interfaces.  This
tool does not change production demosaicing, profiles, defaults, or the GUI.
