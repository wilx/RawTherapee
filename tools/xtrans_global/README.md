# Global X-Trans reconstruction experiment

The three hidden PP3 identifiers are:

```ini
[RAW X-Trans]
Method=xtrans-global-spectral-rgb
CcSteps=0
```

```ini
Method=xtrans-global-spectral-diff
```

```ini
Method=xtrans-global-spectral-edge
```

They are deliberately absent from the GUI and method enumeration. Runtime
parameters are development environment variables:

| Variable | Default | Meaning |
| --- | ---: | --- |
| `RT_XTRANS_GLOBAL_LAMBDA_RGB` | `0.02` | Phase-A spectral weight |
| `RT_XTRANS_GLOBAL_LAMBDA_G` | `0.01` | Phase-B/C green weight |
| `RT_XTRANS_GLOBAL_LAMBDA_C` | `0.10` | Phase-B/C color-difference weight |
| `RT_XTRANS_GLOBAL_P` | `1` | spectral exponent, `1` or `2` |
| `RT_XTRANS_GLOBAL_ITERATIONS` | `20` | Phase-A/B PCG limit and Phase-C initializer |
| `RT_XTRANS_GLOBAL_TOLERANCE` | `1e-5` | relative PCG residual threshold |
| `RT_XTRANS_GLOBAL_EDGE_OUTER` | `3` | Phase-C IRLS iterations |
| `RT_XTRANS_GLOBAL_EDGE_INNER` | `10` | PCG limit per IRLS solve |
| `RT_XTRANS_GLOBAL_EDGE_EPSILON` | `0.01` | normalized Charbonnier transition |
| `RT_XTRANS_GLOBAL_TILE` | `0` | whole frame, or `96`, `192`, `384` |
| `RT_XTRANS_GLOBAL_INITIALIZATION` | `triangulated` | `triangulated` or `zero` |
| `RT_XTRANS_GLOBAL_DEBUG_PREFIX` | unset | normalized float32 planes and residual CSV |

`compare_outputs.py` authenticates and compares the four full-resolution TIFFs
for the established DSCF0771 test, then creates the one-third full-frame and
nearest-neighbour 500-percent earring assets. Source TIFFs remain external.
