# Markesteijn/MLRI deterministic hybrid experiment

This developer-only tool measures whether observable mosaic and candidate-output
features can recover the oracle complementarity between RawTherapee's actual
three-pass Markesteijn implementation and corrected-final MLRI. It does not add
a demosaicing method or modify production processing.

Build and run with:

```sh
nice -n 10 cmake --build build/dev -j4 \
    --target rawtherapee-xtrans-oracle-runner
.venv/bin/python -m tools.xtrans_hybrid.generate \
    --runner build/dev/tests/xtransoracle/rawtherapee-xtrans-oracle-runner \
    --output devnotes/images/xtrans-hybrid --force
```

The 20 source images come from authenticated scikit-image 0.26.0 bundled data.
The split is frozen by source image before fitting. Ground truth is used only to
construct offline targets and evaluation metrics; `feature_maps()` deliberately
has no ground-truth argument. The generator forces NumPy/SciPy BLAS backends to
one thread before importing them so parallel reduction order cannot alter the
canonical fitted coefficients; the native demosaicer runner still uses four
OpenMP threads. Derived observable features are canonicalized to six decimal
places before fitting or inference, matching the useful precision of the
float32 candidate images, and training means/scales to twelve decimal places.
This removes irrelevant float64 reduction noise from the fitted contract.
Bounded blend weights are then frozen to 0.001 steps, giving 1001 deterministic
alpha levels without changing hard routing or perceptible blend precision.
