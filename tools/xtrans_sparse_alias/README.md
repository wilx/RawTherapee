# X-Trans structured sparse alias recovery

This development-only experiment reuses the validated 18x54 X-Trans alias
operator from `tools/xtrans_alias`. It measures oracle support recovery,
replica-aware OMP, a deliberately weak single-peak control, real
cosine/sine-group recovery, competing-support confidence, synthetic image
upper bounds, and natural-image alias-family sparsity. It does not contain or
register a RawTherapee demosaicer.

Generate the canonical report data and PNG diagnostics with:

```sh
.venv/bin/python -m tools.xtrans_sparse_alias.generate \
    --output-dir devnotes/images/xtrans-sparse-alias --force
```

The pinned natural corpus is the `astronaut`, `coffee`, `hubble_deep_field`,
and `rocket` RGB data shipped by scikit-image 0.26.0. Their file digests and
public-domain/CC0 provenance are authenticated before analysis. Inputs are
decoded with Pillow, converted from sRGB to linear RGB, locally mean-removed,
and analyzed through 24, 48, and 96-pixel separable-Hann windows.

Required development packages are NumPy 2.3.5, Pillow 12.3.0, and
scikit-image 0.26.0. Random studies use fixed seeds. Generated JSON contains
no elapsed time, path, host, or timestamp; the measured runtime belongs in the
research report because wall-clock time cannot be canonical.
