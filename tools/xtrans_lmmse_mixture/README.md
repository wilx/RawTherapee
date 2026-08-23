# X-Trans three-bank adaptive LMMSE experiment

This development-only experiment asks whether three interpretable covariance
banks improve the fixed population LMMSE bank. It makes no RawTherapee engine,
PP3, or GUI changes and commits no learned filter weights.

The fixed comparison is:

- `G100`: the previous 100-source 11x11/M2/ridge-1e-3 bank;
- `GMAX`: the same bank trained on all 200 official BSDS training sources;
- `MIX3`: three 11x11/M2/ridge-1e-3 banks trained on the same 200 sources.

Classification uses no reconstructed colors. The primary statistics are:

- raw or per-observed-color high-pass variance in the scalar mosaic window;
- directional anisotropy from horizontal and vertical equal-CFA-color pairs at
  distances one, two, and three, with differences divided by distance;
- an alternate physical-sample chroma statistic formed from local observed
  R/G/B means.

Scale statistics are divided by their phase-specific training medians before
thresholding. One global threshold rule is then used for all 18 phases. Three
initial interpretations and four neighboring threshold controls are selected
using only BSDS training/validation data. Hubble, Hydra, untouched BSDS test,
and the external chromatic controls do not affect selection.

## Reproduction

```sh
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m pytest -q tools/xtrans_lmmse_mixture/tests

env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 \
  nice -n 10 .venv/bin/python -m tools.xtrans_lmmse_mixture.experiment \
    --output /tmp/xtrans-lmmse-mixture \
    --bsds-root /tmp/BSDS500 \
    --runner build/dev/tests/xtransulri/rawtherapee-xtrans-ulri-tests \
    --starfield /tmp/xtrans-danger-sources/grail_free_air_stars1.tif \
    --source-oracle devnotes/images/xtrans-lmmse/source-oracle.json \
    --previous-population devnotes/images/xtrans-lmmse/population.json

env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 \
  nice -n 10 .venv/bin/python -m tools.xtrans_lmmse_mixture.synthetic \
    --output /tmp/xtrans-lmmse-mixture/synthetic.json \
    --natural-result /tmp/xtrans-lmmse-mixture/mixture.json \
    --bsds-root /tmp/BSDS500 \
    --runner build/dev/tests/xtransulri/rawtherapee-xtrans-ulri-tests
```

The six external chromatic controls reuse authenticated scikit-image 0.26.0
sources from the earlier X-Trans corpus. Their three deterministic 168x168
crops remain source-held-out.

## Result

The result is **NO-GO**. MIX3 adds 0.264 dB over the equally trained global
bank on untouched BSDS and has substantial 7x7 oracle headroom, but it loses
3.406 dB on a smooth chromatic gradient and 0.889 dB on every phase of a tiny
colored star. The star-safe edge bank exists but the observable rule selects
the low-chroma bank. See `devnotes/xtrans-lmmse-mixture-report.md`.
