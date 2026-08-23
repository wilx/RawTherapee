# X-Trans joint spatial-chromatic LMMSE experiment

This development-only experiment tests a compact, phase-specific Wiener
predictor before any RawTherapee engine implementation.  It learns only from
physically observed scalar X-Trans samples and restores the target's measured
component exactly.

The implementation follows the local joint spatial-chromatic formulation of
Portilla, Otaduy, and Dorronsoro (ICIP 2005), generalized from four Bayer
local mosaics to the 18 distinct X-Trans phase cells.  It does not copy a
reference implementation.

## Dependencies

- RawTherapee's existing `.venv` (Python 3.12, NumPy, SciPy, Pillow,
  scikit-image 0.26.0, pytest);
- the native `rawtherapee-xtrans-ulri-tests` research runner;
- an external BSDS500 checkout;
- the previously authenticated NASA Hydra TIFF.

Regenerating the optional report plots additionally needs Matplotlib. On
Ubuntu, the system `python3-matplotlib` package is sufficient; it is not a
runtime dependency of the experiment or RawTherapee.

The BSDS images are external and are not redistributed.  The official dataset
page permits non-commercial research and educational use.

## Reproduction

```sh
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m pytest -q tools/xtrans_lmmse/tests

env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  nice -n 10 .venv/bin/python -m tools.xtrans_lmmse.experiment \
    --output /tmp/xtrans-lmmse-output \
    --runner build/dev/tests/xtransulri/rawtherapee-xtrans-ulri-tests \
    --starfield /tmp/xtrans-danger-sources/grail_free_air_stars1.tif

git clone --depth 1 https://github.com/BIDS/BSDS500.git /tmp/BSDS500

env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  nice -n 10 .venv/bin/python -m tools.xtrans_lmmse.population \
    --output /tmp/xtrans-lmmse-output \
    --bsds-root /tmp/BSDS500 \
    --runner build/dev/tests/xtransulri/rawtherapee-xtrans-ulri-tests \
    --starfield /tmp/xtrans-danger-sources/grail_free_air_stars1.tif \
    --source-oracle /tmp/xtrans-lmmse-output/source-oracle.json

env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  nice -n 10 .venv/bin/python -m tools.xtrans_lmmse.selected_details \
    --bsds-root /tmp/BSDS500 \
    --population /tmp/xtrans-lmmse-output/population.json \
    --output /tmp/xtrans-lmmse-output/selected-details.json
```

The source-oracle gate passes strongly.  The population model is a **PARTIAL**
result: it beats corrected-final MLRI on the untouched BSDS subset but loses
about half the source-matched oracle gain on the established cross-domain
strips.  See `devnotes/xtrans-lmmse-wiener-report.md`.
