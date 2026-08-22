# X-Trans alias characterization

This development-only tool measures the periodic X-Trans sampling operator. It
does not demosaic or modify RawTherapee output.

Frequencies are expressed in cycles/pixel on `[-0.5, 0.5)` for each axis. The
tool computes the exact 6x6 mask coefficients, empirical shifted replicas,
complex and real-phase confusion, full alias-family singular values, explicit
null-space examples, all-18-variant invariance, and active-family diagnostics
for edges and the earlier A/B/C analytical scenes.

Generate the tracked corpus with the existing development environment:

```sh
.venv/bin/python -m tools.xtrans_alias.generate \
    --output-dir devnotes/images/xtrans-alias --force
```

Required development packages are `numpy==2.3.5` and `Pillow==12.3.0`. Output
is canonical JSON plus deterministic PNG diagnostics. Regeneration refuses to
replace files unless `--force` is supplied.

The full 18x54 RGB alias-family matrix always has a 36-dimensional null space.
Its 18 reported nonzero singular values are all one because X-Trans sampling is
a pointwise color selection. The zero singular values, not the nonzero
condition number, express the missing information.
