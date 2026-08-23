# X-Trans nonlocal patch-recurrence feasibility study

This development-only experiment asks whether nonlocal self-similarity adds
information that the failed local Markesteijn–MLRI selector did not have.  It
does not add a RawTherapee demosaicing method.

The study first searches only exact X-Trans phase matches using physically
observed CFA samples.  This measures neighbor purity and an unattainable
full-RGB donor upper bound.  Exact-phase groups cannot directly reconstruct a
missing value: all columns have the same missing rows.  A second control then
searches all 18 translated phase relationships and tests direct observed-sample
donation plus iterative rank-4 completion.

Run with the pinned scikit-image 0.26.0 environment and external NASA Hydra
star-field source used by the preceding safety study:

```sh
nice -n 10 cmake --build build/dev --target rawtherapee-xtrans-oracle-runner -j4
.venv/bin/python -m tools.xtrans_nonlocal.generate \
  --runner build/dev/tests/xtransoracle/rawtherapee-xtrans-oracle-runner \
  --starfield /tmp/xtrans-danger-sources/grail_free_air_stars1.tif \
  --output devnotes/images/xtrans-nonlocal --force
.venv/bin/python -m pytest tools/xtrans_nonlocal/tests
```

The literature motivation is Antoni Buades et al., *Non local demosaicing*
(2007), and Kan Chang, Pak Lun Kevin Ding, and Baoxin Li, *Color image
demosaicking using inter-channel correlation and nonlocal self-similarity*,
Signal Processing: Image Communication 39 (2015), 264–279,
DOI `10.1016/j.image.2015.10.003`.  This experiment implements neither paper's
complete demosaicer.
