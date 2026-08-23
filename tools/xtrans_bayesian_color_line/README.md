# X-Trans Bayesian two-color likelihood experiment

This development-only package evaluates the single-image two-color likelihood
from Bennett et al., *Video and Image Bayesian Demosaicing with a Two Color
Image Prior*, ECCV 2006, DOI `10.1007/11744023_40`, on the frozen X-Trans
color-line corpus.

It does not add a RawTherapee demosaicing method.  Gate 1 supplies the true
local line endpoints and asks whether nearby CFA observations estimate the
target blend coordinate better than the previous one-sample reconstruction.
The gate failed, so endpoint initialization and production integration are not
implemented.

Run the focused tests with:

```sh
PYTHONPATH=. .venv/bin/python -m pytest -q \
    tools/xtrans_bayesian_color_line/tests
```

Regenerate the Stage A artifact with the authenticated external Hydra image:

```sh
PYTHONPATH=. nice -n 10 .venv/bin/python \
    -m tools.xtrans_bayesian_color_line.stage_a \
    --starfield /path/to/grail_free_air_stars1.tif \
    --output /tmp/stage-a.json --force
```

The JSON is canonical and contains no paths, host identity, timestamps, or
elapsed time.  Two clean runs must have SHA-256
`e4d61004279d1678a0feb1cdf4dafcee6770e07c70ac363ff79304b22affb76d`.

The package also contains a small independent Bayer front-end reproduction:
Malvar--He--Cutler HQLI, 5x5 weighted two-means, one-sigma rejection, and the
same likelihood.  It labels the clustering details that the paper leaves
unspecified and is not presented as author-code parity.

The likelihood implementation follows paper equations 5--12.  The author did not publish
the numeric endpoint-prior strength `eta` or the procedure/value for estimating
global per-channel `sigma_N`; these are explicit parameters.  The Gate-1 result
uses `eta=1` to isolate the measurement likelihood.  The paper's endpoint prior
is reported as a sensitivity control and cannot silently select the result.
