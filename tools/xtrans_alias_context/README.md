# X-Trans windowed alias-context experiment

This development-only package tests whether neighboring analysis windows can
select among competing sparse source explanations of one X-Trans alias family.
It builds directly on `tools/xtrans_alias` and `tools/xtrans_sparse_alias`.

The experiment regularizes source hypotheses—support, complex phase, source
frequency, color direction, and weak amplitude—not reconstructed RGB values.
It does not implement or register a RawTherapee demosaicing method.

Generate the canonical characterization and plots with:

```sh
.venv/bin/python -m tools.xtrans_alias_context.generate \
    --output-dir devnotes/images/xtrans-alias-context
```

The authenticated natural corpus is the same four scikit-image 0.26.0 RGB
sources used by the preceding sparse experiment. The generator refuses to
replace existing results unless `--force` is supplied.

Run focused Python tests with:

```sh
.venv/bin/python -m pytest -q tools/xtrans_alias_context/tests
```

The canonical JSON intentionally contains no timestamps, absolute paths, host
identity, or measured execution time.
