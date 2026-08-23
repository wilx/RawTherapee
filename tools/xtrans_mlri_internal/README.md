# Corrected-final MLRI internal failure experiment

This development-only tool captures the actual two-pass corrected-final MLRI
state, traces stage-wise ground-truth error, measures internal candidate and
residual-limiter oracle headroom, and tests whether simple internal-only
features predict harmful corrections on source-held-out data.

Build and test the native trace harness:

```sh
cmake --preset dev
nice -n 10 cmake --build build/dev \
  --target rawtherapee-xtrans-mlri-internal-tests -j4
ctest --test-dir build/dev --output-on-failure -L xtrans-mlri-internal
```

Run the complete experiment using the previously authenticated NASA Hydra
star-field file:

```sh
.venv/bin/python -m tools.xtrans_mlri_internal.generate \
  --runner build/dev/tests/xtransmlriinternal/rawtherapee-xtrans-mlri-internal-tests \
  --starfield /tmp/xtrans-danger-sources/grail_free_air_stars1.tif \
  --output devnotes/images/xtrans-mlri-internal
```

The trace API and executable are research interfaces. Production demosaicing,
method selection, profiles, defaults, and GUI behavior are unchanged.
