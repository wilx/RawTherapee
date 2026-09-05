# Developer-only X-veon ONNX experiment

> Privacy update: DSCF0771 full-frame imagery is private and is not
> distributed. Only the reviewed earring crops may be shared. Historical
> full-frame asset references below describe private benchmark evidence,
> not current publication instructions.

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

### Optional MIGraphX backend

Phase 11 adds a second default-off build switch for the reviewed ROCm 7.2.1 /
MIGraphX 2.15.0 development stack:

```sh
/usr/bin/cmake --preset dev \
    -DWITH_ONNXRUNTIME=ON \
    -DONNXRUNTIME_ROOT=/path/onnxruntime-linux-x64-1.27.0 \
    -DWITH_MIGRAPHX=ON \
    -DMIGRAPHX_ROOT=/opt/rocm
cmake --build build/dev --target rawtherapee-cli rawtherapee-neuralmodel-tests
```

The direct MIGraphX C API preserves RawTherapee's C++11 baseline. Select a
backend explicitly with `RT_XVEON_XTRANS_BACKEND=onnxruntime-cpu` or
`RT_XVEON_XTRANS_BACKEND=migraphx`. ONNX Runtime remains the default when both
are compiled. An unavailable explicitly selected backend is a loud failure;
it never silently executes the other neural backend.

MIGraphX parses only the already authenticated in-memory ONNX bytes, compiles
for `gpu` with offload copies, exhaustive tuning disabled, and strict math,
and validates the fixed input and output contracts before execution. Set
`RT_XVEON_MIGRAPHX_CACHE_DIR` to a user-owned mode-0700 directory to enable the
developer compiled-program cache. Cache files are bound to the model digest,
complete MIGraphX version, reviewed `gfx1101` ISA, compile settings, payload
size, and payload SHA-256. Symlinks, foreign ownership, and group/world-writable
directories or files are rejected. The compiled program is published last as
the atomic completion marker and receives a finite first-run check when loaded.
`RT_XVEON_MIGRAPHX_FAST_MATH=1` selects a separate evaluation-only cache key;
it is not recommended because the measured speedup was below the Phase 11
retention threshold.

ROCm GPU access requires the process to inherit the `render` group. A shell
started before group membership changed can use `sg render -c '...'`, or be
restarted. The compiled cache contains model values and remains ignored and
external for the same licensing reason as the ONNX model.

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

## Phase 11 MIGraphX result

The deterministic 288x288 tile comparison covers all 248,832 output floats.
Against ONNX Runtime CPU it measured maximum absolute error `0.000988603`, RMS
`0.000119834`, p99 `0.000316441`, and mean `0.000090861`; all reviewed aggregate
GPU bounds pass. The pre-existing tighter per-value diagnostic remains
unchanged and 17,974 values satisfy it. Fresh and cache-loaded MIGraphX output
is bit-identical.

Fresh compilation took 45.0 seconds. The authenticated 15,926,288-byte compiled
program has SHA-256
`2152adb0e16898cef340be7d961b099b1b86089176f746990eb66c82bebe8b04`;
cache loading took approximately 0.35 seconds. One hundred warmed strict tiles
averaged 2.214 ms in the RawTherapee driver. Fast math was output-identical but
averaged 2.176 ms, only 1.7 percent faster, so it fails the required 10 percent
benefit and strict math remains selected.

Three cached 7752x5178 exports took `6.97`, `6.92`, and `7.09` seconds, median
`6.97` seconds. This is 6.54 times faster than the 45.60-second CPU median and
1.77 times the 3.94-second Markesteijn reference. Peak host RSS was 2,373,852
KiB. Sampled VRAM rose by 399,572,992 bytes above an already occupied
10,697,834,496-byte baseline.

Full-frame CPU-versus-GPU comparison measured 68.30 dB CPSNR, SSIM
`0.99987655`, RMS `0.00038474`, p99 `0.00148013`, and maximum `0.00737011`.
Absolute channel-mean deltas were all below `0.000085`. On the agreed crop,
MIGraphX phase RMS was `(0.0007389, 0.0006894, 0.0006750)`, slightly lower in
all channels than the CPU result. Repeated GPU exports are pixel-identical;
whole TIFF hashes differ only because of run-dependent metadata. Visual review
of the full-third and 500-percent earring comparisons found no colour, texture,
seam, clipping, sharpness, or earring regression.

All seven analytical scenes retain the earlier qualitative findings. The
largest CPSNR change is -0.18 dB on the diagnostic black case; the saturated
edge and nonzero-black failures therefore remain model-quality findings rather
than backend changes. The Phase 11 acceleration/parity gate passes. The method
still stays hidden because the upstream model has no explicit license and the
earlier analytical quality gate remains unresolved.

The combined backend build and CTests pass in normal and debug configurations,
and the default-off CLI has no ONNX Runtime, ROCm, or MIGraphX dependency.
Clang 18 ASan/UBSan passes the complete self-contained neural suite and the
reviewed CPU/GPU parity tests. The Phase 11 C and C++ sources also compile with
the GCC strict-warning flags. The full GCC 13 strict build remains blocked by
the pre-existing `dcraw.cc` fortified-memmove warning, and GCC 13 additionally
ICEs in that file when compiling the whole target with ASan/UBSan; neither
failure involves a Phase 11 source.
