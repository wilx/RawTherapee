# X-Trans local color-line feasibility experiment

This development-only experiment asks whether a local affine rank-1 model in
linear RGB supplies useful information for X-Trans reconstruction.  It does not
add a RawTherapee demosaicing method.

The experiment fits local PCA color lines at 3, 5, 7, 11, and 15 pixels, then
uses only the physically observed X-Trans sample to locate each pixel on either
the true line or a line estimated from an initial demosaic.  Native samples are
re-injected exactly.  It also evaluates bounded and unbounded coordinates,
projected two-means endpoints, three degeneracy fallbacks, leave-center-out
fits, one-step Huber covariance fits, oracle applicability gates, all 18 CFA
phase cells, and modest synthetic noise.

The native baseline helper must already exist:

```sh
nice -n 10 .venv/bin/python -m tools.xtrans_color_line.generate \
    --runner build/dev/tests/xtransulri/rawtherapee-xtrans-ulri-tests \
    --starfield /tmp/xtrans-danger-sources/grail_free_air_stars1.tif \
    --output devnotes/images/xtrans-color-line \
    --force
```

The star-field file is authenticated by the reused NASA corpus binding.  The
generator invokes the native baseline helper with one OpenMP worker because
this is a reconstruction experiment, not a throughput benchmark, and the
small-patch Markesteijn path must be byte-stable.

Run the focused unit tests with:

```sh
nice -n 10 .venv/bin/python -m pytest -q tools/xtrans_color_line/tests
```

`results.json`, `dataset.json`, and the diagnostic maps are canonical tracked
artifacts.  They contain no absolute paths or timings.  See
`devnotes/xtrans-color-line-feasibility-report.md` for interpretation and the
GO/PARTIAL/NO-GO decision.
