# X-Trans sparse RGB dictionary feasibility experiment

This development-only study tests a classical, non-neural population prior in
three deliberately separate stages:

1. full-RGB sparse representation;
2. X-Trans coefficient fitting with the RGB-selected support supplied by an
   oracle;
3. blind OMP support selection from physically observed CFA values only.

One RGB dictionary is used for all 18 X-Trans phases. The phase changes only
the observation rows. Every physically measured component is restored exactly
after reconstruction. No RawTherapee engine, PP3, GUI, postprocessing, or
learned model distribution changes are made.

Dictionary learning is an independent deterministic implementation of
alternating FISTA L1 sparse coding and regularized MOD updates. It needs only
the existing NumPy/SciPy/Pillow research environment; scikit-learn is not
required.

## Reproduction

```sh
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m pytest -q tools/xtrans_sparse_dictionary/tests

env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 \
  nice -n 10 .venv/bin/python -m tools.xtrans_sparse_dictionary.experiment \
    --output /tmp/xtrans-sparse-dictionary \
    --bsds-root /tmp/BSDS500 \
    --starfield /tmp/xtrans-danger-sources/grail_free_air_stars1.tif \
    --runner build/dev/tests/xtransulri/rawtherapee-xtrans-ulri-tests
```

BSDS images and the authenticated NASA Hydra TIFF remain external. Learned
dictionary weights are not written or committed. Only compact canonical
metrics, a corpus manifest, and an atom visualization are suitable for
tracking under `devnotes/`.

## Result

The result is **NO-GO — practical**. Learned representation and known-support
X-Trans sensing pass their gates, but the validation-selected LASSO solution
uses about 26 atoms and reaches 28.517 dB on held-out sample centers versus
34.195 dB for GMAX on exactly the same coordinates. Bright Hubble/Hydra
targets and analytical phase stability also fail. See
`devnotes/xtrans-sparse-dictionary-report.md`.
