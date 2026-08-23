# X-Trans joint-color GMM feasibility experiment

This development-only experiment tests whether a soft Bayesian mixture of
joint RGB patch Gaussians can improve the successful population LMMSE prior
without MIX3's explicit content classifier. It adds no RawTherapee engine,
PP3, CLI method, or GUI behavior and commits no fitted model.

The implementation is independent. It follows the local conditional-Gaussian
model in Sandeep and Jacob, *Joint Color Space GMMs for CFA Demosaicking*, IEEE
Signal Processing Letters 26(2), 232-236, DOI
`10.1109/LSP.2018.2886466`. The supplied five-page PDF is 646,215 bytes with
SHA-256
`4b83b2e0d1c6bc2e529c4f2c2ef065b44e2f7347f83c0163563287c4a1128b6b`.
No author/reference implementation was found. The paper states that
supplementary material exists, but it was not present with the supplied PDF.

The faithful core is:

1. learn a GMM over contiguous CHW RGB patches;
2. project each component through the exact phase-specific X-Trans sampling
   matrix;
3. calculate observed-data component likelihoods;
4. calculate each component's conditional center RGB estimate;
5. restore the physically measured center channel exactly.

The paper uses hard observed-data MAP selection, 6x6 patches, K=150,
overlapping stride-one aggregation, and iterative image-adaptive GMM updates.
This bounded feasibility experiment instead starts with 3x3/5x5/7x7 patches,
K=1..64, center-only reconstruction, and both hard MAP and soft posterior-MMSE
estimation. It deliberately does not implement overlap aggregation, adaptive
GMM updates, or EPLL unless the local prior passes its population and safety
gates.

## Environment

The ignored `.venv` contains:

- Python 3.12;
- NumPy 2.3.5;
- SciPy 1.18.0;
- scikit-learn 1.7.2;
- pytest.

System Matplotlib is used only to render report figures.

## Reproduction

Build the native reference helper with the normal constrained build policy:

```sh
nice -n 10 cmake --build build/dev \
  --target rawtherapee-xtrans-ulri-tests -j4
build/dev/tests/xtransulri/rawtherapee-xtrans-ulri-tests contract
```

Run the mathematical tests:

```sh
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m pytest -q tools/xtrans_gmm/tests
```

Run the gated experiment. Fitted `.npz` models remain under `/tmp`:

```sh
nice -n 10 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  PYTHONUNBUFFERED=1 \
  .venv/bin/python -m tools.xtrans_gmm.experiment \
    --output /tmp/xtrans-gmm \
    --bsds-root /tmp/BSDS500 \
    --starfield /tmp/xtrans-danger-sources/grail_free_air_stars1.tif \
    --runner build/dev/tests/xtransulri/rawtherapee-xtrans-ulri-tests
```

The test driver now supports a backward-compatible `run-selected` command so
focused experiments can request only named native reference algorithms rather
than paying for every ULRI variant. The ordinary `run` command is unchanged.

Run post-selection controls without repeating native references:

```sh
nice -n 10 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  PYTHONUNBUFFERED=1 \
  .venv/bin/python -m tools.xtrans_gmm.supplement \
    --output /tmp/xtrans-gmm \
    --results /tmp/xtrans-gmm/results.json \
    --bsds-root /tmp/BSDS500 \
    --starfield /tmp/xtrans-danger-sources/grail_free_air_stars1.tif
```

Freeze timing/cache-state fields and render figures:

```sh
.venv/bin/python -m tools.xtrans_gmm.freeze \
  /tmp/xtrans-gmm/results.json \
  /tmp/xtrans-gmm/canonical-results.json

python3 tools/xtrans_gmm/render.py \
  --results /tmp/xtrans-gmm/results.json \
  --supplement /tmp/xtrans-gmm/supplement.json \
  --model /tmp/xtrans-gmm/models/p7-k16-observed-rgb.npz \
  --output devnotes/images/xtrans-gmm
```

## Result

The outcome is **PARTIAL**. The selected 7x7/K=16 soft GMM reaches 33.531 dB
on untouched BSDS, 1.477 dB above GMAX, and recovers 85.5% of validation
7x7 component-oracle headroom. It also improves the bright Hubble/Hydra
targets over GMAX. It fails the frozen safety gate because one external
chromatic crop loses 1.256 dB and the all-phase gray-gradient control loses
3.165 dB, although the latter remains an absolutely small 66.742 dB error.
The training-size curve is non-monotonic and further EM likelihood
optimization worsens validation reconstruction.

The diagonal control reaches only 22.347 dB, proving that joint
spatial/chromatic covariance is essential. EPLL is not triggered. See
`devnotes/xtrans-gmm-report.md` for the complete decomposition.
