# MLRI guided-regression evidence experiment

This development-only experiment asks whether the local regression state that
`guidedMlri()` normally discards can rank the sixteen pass/direction green
candidates at the same missing-green pixel.

Build and run:

```sh
nice -n 10 cmake --build build/dev \
  --target rawtherapee-xtrans-mlri-internal-tests -j4

.venv/bin/python -m tools.xtrans_mlri_regression_evidence.generate \
  --runner build/dev/tests/xtransmlriinternal/rawtherapee-xtrans-mlri-internal-tests \
  --starfield /tmp/xtrans-danger-sources/grail_free_air_stars1.tif \
  --output devnotes/images/xtrans-mlri-regression-evidence
```

The native trace is observational only. It records exact model coefficients,
residuals, support terms, overlap-weight statistics and model-prediction
dispersion, then attributes them through the same CFA completion and
half-Gaussian supports as the final candidates. The test executable compares
the traced reconstruction bit-for-bit with the uninstrumented corrected-final
reference.

`leave-one-model-out` refers to removing one already-fitted overlapping local
model from the aggregate. It does not claim to refit every local regression
after removing one underlying CFA sample.
