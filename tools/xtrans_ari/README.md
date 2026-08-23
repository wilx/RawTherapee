# Adaptive Residual Interpolation reference experiment

This directory reproduces the published Bayer ARI reference and evaluates a
missing-green X-Trans adaptation. It is developer research tooling, not a
RawTherapee demosaicing method.

The authenticated upstream reference is Sensors ARI version 1.0, dated
2017-12-07. Its `readme.txt` permits research use only and reserves all rights,
so no upstream MATLAB source is tracked here. `generate_reference.py` requires
the external archive and papers, authenticates every source file, makes a
temporary copy, and applies one Octave-only numerical compatibility change:
the mathematically nonnegative guided-filter fit variance is clamped to zero
before its square root. The original files are never modified.

The independent Bayer implementation preserves the authenticated source's
vertical MLRI mask typo only in parity mode. The X-Trans experiment uses the
correct green-site mask.

## Frozen identities

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `Sensors_ARI.zip` | 720,117 | `eecb94b92a2f1f4fb1bbdf41de93982697fbba09baa80cf8227104203dbd9272` |
| Sensors 2017 paper | 1,243,059 | `32c11929c42a7550b57b50c74b0cecbfb12f0f985af77dcfa60acfb5bedd2a16` |
| ICIP 2015 paper | 2,568,311 | `f763278c460ae0d40fdd8829513c44462e5748c4cf34d94de094ff3741a7cde5` |
| Bayer golden manifest | 24,099 | `0b45395709c1b0be6527970443b158cb1d17a72b69548c8d5ab6a9d83a6a4ded` |

## Bayer regeneration

```sh
nice -n 10 .venv/bin/python -m tools.xtrans_ari.generate_reference \
    --reference-root /tmp/ari-reference/Sensors_ARI \
    --archive /tmp/xtrans-mlri-papers/ARI-2017.zip \
    --paper-2017 /tmp/xtrans-mlri-papers/ARI-2017.pdf \
    --paper-2015 /tmp/xtrans-mlri-papers/ARI-2015.pdf \
    --output /tmp/ari-bayer-reference

diff -qr /tmp/ari-bayer-reference tools/xtrans_ari/reference_golden
```

The independent green stage has aggregate RMS `3.259e-8` against the official
float32 output. The largest difference is `6.424e-6` at a boundary; the
16-pixel interior maximum is `3.127e-8`.

## X-Trans feasibility run

```sh
nice -n 10 .venv/bin/python -m tools.xtrans_ari.generate \
    --runner build/dev/tests/xtransulri/rawtherapee-xtrans-ulri-tests \
    --starfield /tmp/xtrans-danger-sources/grail_free_air_stars1.tif \
    --reference-golden tools/xtrans_ari/reference_golden \
    --output /tmp/xtrans-ari-results

diff -qr /tmp/xtrans-ari-results devnotes/images/xtrans-ari
```

The X-Trans bank contains 44 candidates: RI and MLRI, horizontal and vertical,
and 11 iterations. It applies the published window growth, residual update,
criterion, per-pixel iteration retention, and reciprocal-criterion branch
combination. Linear interpolation is generalized to the irregular X-Trans
sample positions along each row or column. It does not reconstruct final red
or blue. The canonical results also record median/p90/p95/p99/max error tails,
criterion calibration, and candidate/branch spatial coherence for 1x1, 3x3,
7x7, and 15x15 oracle supports.

Wall-clock measurements are printed while generating and summarized in the
report, but are intentionally excluded from canonical `results.json`. This
keeps the complete artifact directory byte-reproducible across runs while
still recording the observed feasibility cost.

Run tests with:

```sh
nice -n 10 .venv/bin/python -m pytest -q tools/xtrans_ari/tests
```

See `devnotes/xtrans-ari-reference-feasibility-report.md` for the design,
results, limitations, and decision.
