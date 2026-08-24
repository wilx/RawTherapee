# X-Trans phase-conditioned GMR experiment

This development-only experiment replaces the previous 147-dimensional full
RGB patch GMM with 18 task-focused phase models over only the 49 physically
observed values of a 7x7 X-Trans patch and the two missing center channels.
It adds no production engine, PP3 method, CLI method, or GUI entry.

For each phase the model learns

```text
z_p = [y_p, t_p] in R^51
p_p(y,t) = sum_k pi_pk N([y,t]; mu_pk, Sigma_pk)
```

and calculates the Gaussian conditional target mean and soft observed-data
posterior exactly. Target order is `[G,B]` for measured R, `[R,B]` for measured
G, and `[R,G]` for measured B. The physical center CFA sample is restored
bit-exactly after posterior averaging.

## Reproduction

The experiment reuses the frozen BSDS/external/star corpus from
`tools/xtrans_gmm`. Fitted models and the decoded 102,400-patch matrix remain
external under `/tmp`.

```sh
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m pytest -q tools/xtrans_gmr/tests

nice -n 10 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  PYTHONUNBUFFERED=1 \
  .venv/bin/python -m tools.xtrans_gmr.experiment \
    --output /tmp/xtrans-gmr \
    --bsds-root /tmp/BSDS500 \
    --starfield /tmp/xtrans-danger-sources/grail_free_air_stars1.tif \
    --previous devnotes/images/xtrans-gmm/results.json

python3 tools/xtrans_gmr/render.py \
  --results /tmp/xtrans-gmr/results.json \
  --previous devnotes/images/xtrans-gmm/results.json \
  --output devnotes/images/xtrans-gmr
```

Training uses exact iteration checkpoints (`tol=0`). Scikit-learn therefore
emits expected `ConvergenceWarning` messages at the checkpoint boundary; model
validity is checked independently through finite values and positive-definite
Cholesky factors. Iterations 20 to 30 and 30 to 50 continue one saved EM
trajectory rather than refitting from initialization.

## Result

The result is **PARTIAL — safety stabilization incomplete**. The selected
7x7/K=64/10-iteration soft GMR reaches 34.835 dB on untouched BSDS, +2.781 dB
over GMAX and +1.304 dB over FULL-GMM. Seed and training-size behavior are
substantially more stable, and the old IHC-2 failure is fixed. One untouched
Motorcycle crop still loses 0.853 dB to GMAX, exceeding the frozen 0.5 dB
safety limit. Global covariance shrinkage does not improve validation and is
not selected. See `devnotes/xtrans-gmr-report.md` for the complete analysis.
