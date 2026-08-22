# Developer-only X-Trans MLRI experiment

This directory contains the reproducibility tools for RawTherapee's hidden
`mlri-xtrans-2pass` method. The implementation and its provenance are described
in `devnotes/xtrans-mlri-design.md`; it is not a GUI method and is not claimed
to be an X-Trans algorithm published by the Tokyo Tech RI authors.

## Octave reference corpus

Ubuntu prerequisites:

```sh
sudo apt install octave octave-image
```

Regenerate the corpus from the authenticated MATLAB entry point:

```sh
.venv/bin/python tools/xtrans_mlri/generate_golden.py \
    --source /path/to/function_demosaic_x_trans.m \
    --output /tmp/mlri-golden
diff -qr /tmp/mlri-golden tools/xtrans_mlri/golden
```

The source must have SHA-256
`055d1807729cbd556406bf695a6617f189d2376f7dbd35f4477f44e671fffa0c`.
The corpus manifest SHA-256 is
`b39e6200a049a727a014faf560e8fe5bf94860991eb23ac63cc0c2699083e161`.
Octave may print `ignoring const execution_exception& while preparing to exit`;
the invocation is successful when it exits zero and all identities match.

## Native and CLI validation

With `BUILD_TESTING=ON`:

```sh
cmake --build build/dev --target rawtherapee-xtrans-mlri-tests
ctest --test-dir build/dev -L xtrans-mlri --output-on-failure
```

Run the analytical DNG benchmark without any neural-model files:

```sh
.venv/bin/python tools/benchmark_xtrans_demosaicnet.py \
    --rawtherapee-cli /path/to/rawtherapee-cli \
    --mlri --size 96 --work-dir /tmp/mlri-benchmark \
    --report /tmp/mlri-benchmark.json
```

The benchmark writes a neutral PP3 containing the hidden method string,
disables false-colour suppression and unrelated processing, rejects the loud
fallback marker, and retains its TIFFs outside the repository. For each
synthetic or ground-truth row it also writes a float32 absolute-error TIFF and
an explicitly eight-times-amplified PNG error map.

For a manual CLI export, set this literal value in an otherwise neutral PP3:

```ini
[RAW X-Trans]
Method=mlri-xtrans-2pass
CcSteps=0
```

The controlled blue-diagonal-guide correction is a separate hidden method:

```ini
[RAW X-Trans]
Method=mlri-xtrans-2pass-corrected
CcSteps=0
```

It changes only four source expressions from red to blue diagonal guides.  It
does not replace the faithful method or its Octave golden corpus.

The direct-final control keeps the corrected method's two passes and changes
only its last chroma-selection operation:

```ini
[RAW X-Trans]
Method=mlri-xtrans-2pass-corrected-final-only
CcSteps=0
```

It returns the separately reconstructed green-guided red and blue planes
instead of applying the source's luminance-dependent blend with provisional
chroma. Include `--mlri-corrected-final-only` in the analytical benchmark to
compare it with the corrected blend.

The controlled paper-core methods are also hidden:

```ini
[RAW X-Trans]
Method=mlri-xtrans-paper-core-2014
CcSteps=0
```

```ini
[RAW X-Trans]
Method=mlri-xtrans-paper-core-2016
CcSteps=0
```

They share one sigma-2 green pass, corrected blue diagonal guides, and direct
green-guided red/blue reconstruction. The 2014 method uniformly averages
overlapping affine coefficients; the 2016 method uses inverse residual-error
weights. See `devnotes/xtrans-mlri-paper-audit.md` before interpreting the
results: the surrounding X-Trans geometry is a later generalization, not a
published Tokyo Tech X-Trans algorithm.

The method is intentionally not available in the editor. A successful run
prints `MLRI X-Trans completed:` and its fixed pass/tile contract. Treat
`MLRI X-Trans error [...]` as a failed experiment even though the CLI will
complete after loudly falling back to Markesteijn three-pass.

`compare_outputs.py` validates the full-resolution DSCF0771 TIFF pair, records
the fixed diagnostic and timing data, calculates the existing crop/phase
metrics, and creates the tracked one-third full frame and 500% earring assets.
Its generated source TIFFs remain external.

`compare_corrected_outputs.py` authenticates those reviewed faithful and
Markesteijn TIFF identities, measures a corrected full-resolution export, and
creates matching `DSCF0771-mlri-corrected-*` comparison assets plus a separate
manifest.  It therefore leaves the original comparison manifest unchanged.

`compare_paper_core_outputs.py` authenticates the corrected and Markesteijn
baselines, measures both one-pass methods on the same crop, and creates their
matching one-third full-frame and 500% earring assets in a separate manifest.

`compare_final_only_outputs.py` authenticates the same corrected and
Markesteijn baselines, measures the corrected two-pass direct-final export, and
creates matching `DSCF0771-mlri-corrected-final-only-*` assets without changing
any earlier manifest.
