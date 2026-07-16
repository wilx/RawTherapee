# Developer-only X-veon ONNX experiment

## Scope and licensing boundary

This experiment independently integrates the public X-veon ONNX interface at
upstream revision `2e6b96c63559aa3909b0c7c1bc45dfd4b5dfe680`. At that revision,
the repository and `web/public/xtrans.onnx` contain no explicit license. The
model is therefore not downloaded, copied, installed, packaged, or exposed in
the RawTherapee GUI. ONNX Runtime is Apache-2.0 licensed, but that does not
supply a license for the model.

The external reviewed artifact is exactly 15,536,134 bytes with SHA-256:

```text
45b1fa22b0027868fd5c20ec7b59234ed5aeb35de89fbc0950a4bec67f328500
```

The hidden PP3 method is `xveon-xtrans-onnx`. It requires
`RT_XVEON_XTRANS_MODEL=/path/xtrans.onnx` and falls back loudly to Markesteijn
three-pass on every loader, CFA, allocation, runtime, or non-finite failure.

## Build contract

Ordinary builds remain independent of ONNX Runtime. The experiment is enabled
only with:

```sh
/usr/bin/cmake --preset dev \
    -DWITH_ONNXRUNTIME=ON \
    -DONNXRUNTIME_ROOT=/path/onnxruntime-linux-x64-1.27.0
cmake --build build/dev --target rawtherapee-cli rawtherapee-neuralmodel-tests
```

The root must contain `include/onnxruntime_c_api.h` and
`lib/libonnxruntime.so`. Configuration requires API version 27; the loader
also requires runtime version 1.27.0. The official x86-64 CPU archive used for
development has SHA-256
`547e40a48f1fe73e3f812d7c88a948612c23f896b91e4e2ee1e232d7b468246f`.

ONNX Runtime 1.27's C header contains declarations GCC does not accept from a
C++11 translation unit. `xveon_ort_bridge.c` consequently owns the vendor C
header and opaque C handles; all RawTherapee-facing code remains C++11. The
model is hashed before `CreateSessionFromArray`, and the same retained bytes
back the cached session. Only the CPU provider, sequential graph execution,
full graph optimization, default intra-op threading, and one inter-op thread
are used. Telemetry is disabled.

## Raw-domain and tiling contract

The network receives NCHW float32 `[1,4,288,288]`: the scaled scalar mosaic
followed by canonical red, green, and blue masks. `rawData / 65535` is passed
without clipping, gamma conversion, or a second white balance. Finite signed
RGB output is preserved and multiplied by 65535. There is no sample
reinjection, clipping, false-colour suppression, denoising, sharpening, or
highlight reconstruction in this wrapper.

The shared CFA utility maps all 18 X-Trans phase/orientation representations
to X-veon's architecture-specific 6x6 cell. Tiling reproduces the reviewed
web geometry: 288-pixel tiles, 48-pixel overlap, 240-pixel stride,
edge-repeating reflection for up to five leading alignment pixels, zero
extension on the right/bottom, and separable linear edge ramps with normalized
overlap accumulation. Tiles run sequentially through one cached session. The
wrapper allocates one full-frame weight plane and no full-frame feature maps.

## Developer verification

Mandatory CTests use a dependency-free mock runner to cover the 18 CFA
mappings, masks, leading reflection, trailing zero padding, tile geometry,
signed output, non-finite values, injected runtime failure, and disabled-build
behavior. The real test is conditional:

```sh
XVEON_XTRANS_ONNX=/path/xtrans.onnx \
    ctest --test-dir build/dev -R xveon_ --output-on-failure
```

It authenticates the model and metadata, checks cache identity under concurrent
lookups, and requires repeated tile inference to be bit-identical. The
independent Python reference is in `tools.neural_demosaic.xveon_reference` and
pins `onnxruntime==1.27.0` in `requirements-onnx.txt`. The native deterministic
tile and the independent Python ONNX Runtime result compare bit-for-bit equal;
the measured maximum and mean absolute differences are both zero. The separate
173x31 rotated/translated full-wrapper comparison, including reflection,
two-tile overlap, blending, and coordinate restoration, has maximum absolute
error `2.9802322e-8` and no sample outside `5e-6 + 1e-5 * abs(reference)`.

## DSCF0771 result

The full 7752x5178 export completed without fallback and produced TIFF
SHA-256 `8093d83b8c625bafc6424476220cea286f446bce8d89f9ea57562d226aefea0b`.
Three measured wall times were 45.60, 67.24, and 39.79 seconds; the 45.60-second
median is 11.57 times the 3.94-second Markesteijn reference and narrowly fails
the 10x gate. Peak RSS was 1,991,508 KiB, 188,048 KiB over Markesteijn, so the
4 GiB memory gate passes. The reported bounded working-buffer estimate was
180,740,422 bytes; ONNX Runtime's internal allocations are reflected in RSS,
not that lower-bound estimate.

On crop `(3450,1750,700,500)`, normalized X-veon CFA-phase RMS was
`(0.0007942, 0.0007060, 0.0006752)`, passing the limits
`(0.0015, 0.0015, 0.0015643)`. Its mean RGB delta from Markesteijn was
`(+0.0006301, +0.0000935, -0.0009562)`; common luminance delta
`-0.0000776` and RGB-delta range `0.0015863` both pass.

Visual review of the 500% metallic-earring crop passes. X-veon renders the
hoop more continuously and neutrally; Markesteijn shows stronger magenta/cyan
segmentation along its left edge. The earlier conservative interpretation of
those pixels was reversed after direct side-by-side review. X-veon nevertheless
remains a hidden research method because the runtime gate fails. Even a later
overall quality pass could not authorize packaging or GUI exposure without an
explicit compatible model license.

The 48x48 analytical matrix also exposes large regressions on gradients and
saturated edges, and the model does not return exact black. A repeated 40 MP
export is pixel-identical to the preserved TIFF; only run-dependent TIFF
metadata changes the whole-file digest.

The tracked compact comparison assets and their canonical identity manifest
are under `devnotes/images/xtrans-neural/DSCF0771/`; the RAF, full TIFFs, model,
timing logs, and canonical comparison JSON remain external.
