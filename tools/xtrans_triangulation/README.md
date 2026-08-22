# X-Trans triangulation experiments

The hidden PP3 identifiers are:

```ini
[RAW X-Trans]
Method=xtrans-triangulated-rgb
CcSteps=0
```

and:

```ini
[RAW X-Trans]
Method=xtrans-triangulated-chroma
CcSteps=0
```

They deliberately do not appear in RawTherapee's method enum or GUI. Both use
the same fixed periodic Delaunay stencils. The independent method interpolates
R, G, and B separately. The chroma method interpolates G, then R-G and B-G.
There is no adaptive interpolation, clipping, reinjection, denoising,
sharpening, or false-colour suppression.

`compare_outputs.py` validates three full-resolution 16-bit TIFFs, measures the
established DSCF0771 crop, and creates the one-third full-frame and 500% earring
assets. Source TIFFs remain external.
