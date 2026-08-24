# X-Trans phase-conditioned Student-t GMR experiment

This development-only experiment asks whether heavy-tailed component
likelihoods can remove the last frozen external-domain failure of the
phase-conditioned Gaussian mixture regression (GMR). It adds no production
engine, PP3 method, CLI method, or GUI entry.

Everything except the component distribution is frozen: 7x7 support, 18
phase-specific K=64 models, 49 physical CFA observations, two missing center
components, observed-RGB DC handling, `tau=0.0003`, posterior temperature 4,
and all train/validation/test/external/star/synthetic splits.

## Mathematics and references

Stage A reinterprets the frozen Gaussian component location, scale, and weight
as a multivariate Student-t with fixed degrees of freedom. For observed
dimension `d=49`, responsibility uses

```text
log p(y|k) = const - 1/2 log|Sigma_yy|
             - (nu+d)/2 log(1 + delta_y/nu).
```

The exact conditional distribution follows Peng Ding, *On the Conditional
Distribution of the Multivariate t Distribution*, arXiv:1604.00561. Its
location is the same affine conditional predictor as the Gaussian, its degrees
of freedom are `nu+d`, and its scale is the Schur complement multiplied by
`(nu+delta_y)/(nu+d)`.

Stage B independently implements the latent-scale fixed-nu EM updates from
Peel and McLachlan, *Robust mixture modelling using the t distribution*,
Statistics and Computing 10 (2000), DOI `10.1023/A:1008981510081`. No
reference implementation source was copied.

## Reproduction

The selected Gaussian checkpoint and decoded training vectors remain external
under `/tmp/xtrans-gmr`. Student-t checkpoints remain external as well.

```sh
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m pytest -q tools/xtrans_tgmr/tests

nice -n 10 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m tools.xtrans_tgmr.experiment \
    --output /tmp/xtrans-tgmr/stage-a.json

nice -n 10 env OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=1 \
  .venv/bin/python -m tools.xtrans_tgmr.train \
    --vectors /tmp/xtrans-gmr/training-vectors.npy \
    --initial /tmp/xtrans-gmr/models/p7-k64-seed1481067858-i10-n200.npz \
    --degrees-of-freedom 3 --iterations 1 \
    --output /tmp/xtrans-tgmr/models/nu3-i1-n200.npz
```

After the fixed-nu/checkpoint/source-curve models have been generated, collect
the frozen evaluation with:

```sh
nice -n 10 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m tools.xtrans_tgmr.collect \
    --output /tmp/xtrans-tgmr/results.json \
    --models /tmp/xtrans-tgmr/models \
    --stage-a devnotes/images/xtrans-tgmr/stage-a.json \
    --learned-model /tmp/xtrans-tgmr-nu-learned-i5.npz \
    --learned-nu /tmp/xtrans-tgmr-nu-learned-i5.txt
```

## Result

**GO — robust Student-t GMR.** The validation-selected fixed model uses
`nu=3` and 30 Student-t continuation iterations. On untouched BSDS it reaches
35.518 dB, +3.464 dB over GMAX and +0.683 dB over the frozen Gaussian GMR.
The old `motorcycle-left-0` failure changes from -0.853 dB to +2.811 dB
relative to GMAX; the worst external delta is -0.251 dB, and both bright-star
controls remain above GMAX.

Canonical artifact identities:

```text
stage-a.json  40fd2b65cd93fc7f7f74b58ca05dc30bf2be4fb75ca3d3af30092305ca181b48
results.json  cd8156286cbd7f7f0135cdf11b1caa6ba060f7ea068db35ee54c6516512d5c4a
model         8c93e89fd46aafe24fc193a4dbdd0d05efbb0b87300e879c03903a494eabdd87
```

See `devnotes/xtrans-student-t-gmr-report.md` for the complete gates,
per-source tables, calibration, analytical phase behavior, and limitations.

The training CLI refuses replacement. Model `.npz` bytes are not a portable
identity because NumPy's ZIP container contains archive metadata; reports bind
the ordered float64 tensor content through the inherited logical model digest.

## Boundary

Fitted Student-t models, BSDS data, star fields, and temporary results are not
tracked. The tracked canonical result contains only metrics and identities.
EPLL, discriminative fallback, production C++, PP3/GUI exposure, and inference
optimization remain explicitly out of scope.
