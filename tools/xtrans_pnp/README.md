# X-Trans plug-and-play BM3D feasibility experiment

This development-only study tests an exact X-Trans scalar forward model with
an external classical color BM3D prior in PnP-ADMM. It changes no RawTherapee
engine, PP3, CLI, or GUI behavior.

The BM3D implementation is deliberately not copied into this repository. The
tool accepts only the authenticated PyPI `bm3d==4.0.3` wrapper and
`bm4d==4.2.5` Linux payload used for the experiment. Their Tampere University
license permits non-commercial research use but is incompatible with ordinary
RawTherapee distribution.

## Environment

The local, ignored `.venv` used for the experiment additionally contains:

- `bm3d==4.0.3`, wheel SHA-256
  `fc4dfc0de0cd810fcb6ad198e1d0c6f99cf19d41f2ec69ff867674cfb9f2a775`;
- `bm4d==4.2.5`, wheel SHA-256
  `601338bbd54bcbd971d70b5a6961583a9ccaa564c982389cf3f82bd0a6d5bad5`;
- `colour-demosaicing==0.2.6` only for the public SCICO Bayer control.

No external package is a RawTherapee runtime or test dependency.

## Reproduction

```sh
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  nice -n 10 .venv/bin/python -m pytest -q tools/xtrans_pnp/tests

env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 \
  nice -n 10 .venv/bin/python -m tools.xtrans_pnp.bayer_reference \
    --image /tmp/scico-kodim23.png \
    --output /tmp/xtrans-pnp-bayer-reference.json

env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 \
  nice -n 10 .venv/bin/python -m tools.xtrans_pnp.experiment \
    --output /tmp/xtrans-pnp-bm3d \
    --bsds-root /tmp/BSDS500 \
    --starfield /tmp/xtrans-danger-sources/grail_free_air_stars1.tif \
    --runner build/dev/tests/xtransulri/rawtherapee-xtrans-ulri-tests
```

The experiment intentionally stops at the first failed quality gate. See
`devnotes/xtrans-pnp-bm3d-report.md` for the measured result.
