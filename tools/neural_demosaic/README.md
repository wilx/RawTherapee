# Neural demosaicer development tools

This directory contains development-only tooling for authenticating and
converting published neural demosaicing checkpoints. It is not part of the
RawTherapee runtime.

## PackedXTransNet Phase 12

The PackedXTransNet path accepts only upstream revision
`9c3cc5ab841c9afd2ed0bb702468950481043d06` and
`weights/packed_5183_3208.pt` (646,145 bytes, SHA-256
`1c78b888e3f885252f84c1b12f75dd0af179a62b48499c5808eeb773d1bfc161`).
The checkpoint is CC BY-NC 4.0 and must remain external.

Install the additional deterministic ONNX tooling, then convert:

```sh
.venv/bin/python -m pip install -r tools/neural_demosaic/requirements-onnx.txt
.venv/bin/python -m tools.neural_demosaic.inspect_packedxtrans_checkpoint \
    /path/packed_5183_3208.pt --output /tmp/packedxtrans-checkpoint.json
PACKED_XTRANS_CHECKPOINT=/path/packed_5183_3208.pt \
    .venv/bin/python -m pytest tools/neural_demosaic/tests/test_packedxtrans.py
.venv/bin/python -m tools.neural_demosaic.convert_packedxtrans \
    /path/packed_5183_3208.pt --output /tmp/packedxtransnet.onnx
```

The reviewed output is 1,673,648 bytes with SHA-256
`ad000f496fe9b4a8493bc891dedc3a1e379aec86c93b2fb53f8b8a66a2888e3c`;
its canonical companion manifest has SHA-256
`ebd978aef293d1cf35a5d15234ef185d0785ac222c9c10f6477903e74d4c338d`.
Use it only through the hidden `packedxtransnet-onnx` PP3 method and explicit
`RT_PACKED_XTRANS_MODEL`, `RT_PACKED_XTRANS_BACKEND`, and
`RT_PACKED_XTRANS_PRECISION` environment variables. A private
`RT_PACKED_XTRANS_MIGRAPHX_CACHE_DIR` enables authenticated compiled-program
caching. No checkpoint, ONNX, cache, or full comparison TIFF is tracked.

Phase 1 supports only the Gharbi DemosaicNet X-Trans checkpoint from upstream
revision `959e9d1630976b421d5af5e35b2e2a01f5630e5c`. The tool reads the complete
checkpoint, verifies its pinned size and SHA-256 before deserialization, loads
only tensor weights on the CPU, validates the exact tensor schema, and emits a
deterministic JSON inspection manifest.

The tools do not download checkpoints or run neural inference. Keep `.pth`,
`.rtnn`, and generated manifest files outside tracked source.

## Environment

From the repository root:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r tools/neural_demosaic/requirements.txt
.venv/bin/python -m pip install \
    -r tools/neural_demosaic/requirements-torch.txt \
    --index-url https://download.pytorch.org/whl/cpu
```

The `.venv` directory is ignored by Git. PyTorch, NumPy, and pytest are tooling
dependencies only; RawTherapee does not link or load them.

## Inspect the pinned checkpoint

Write the manifest outside the source tree:

```sh
.venv/bin/python -m tools.neural_demosaic.inspect_checkpoint \
    /path/to/demosaicnet/data/xtrans.pth \
    --output /tmp/demosaicnet-xtrans-v1.manifest.json
```

Omit `--output` to write canonical JSON to standard output. Existing output
files are not replaced unless `--force` is supplied.

The expected summary is:

```json
{
  "dtype": "float32",
  "parameter_count": 409923,
  "payload_bytes": 1639692,
  "tensor_count": 26
}
```

With the pinned checkpoint and this Phase 1 manifest format, the complete
canonical manifest has SHA-256:

```text
371a3e20bac66877238e44d36e349078953c0b6c4e256299f64d66bbd8b72848
```

## Validate the Phase 2 semantic schema

Phase 2 adds a tracked, canonical, language-neutral contract at
`schemas/demosaicnet-xtrans-v1.json`. It assigns architecture ID `1` to
`DEMOSAICNET_XTRANS_V1` and architecture-scoped tensor IDs `1` through `26` in
execution order. ID `0` is reserved as invalid.

The schema binds those IDs to the exact Phase 1 checkpoint and canonical
inspection manifest without changing the Phase 1 format or digest. Validate a
generated manifest with:

```sh
.venv/bin/python -m tools.neural_demosaic.validate_semantic_schema \
    /tmp/demosaicnet-xtrans-v1.manifest.json
```

The tracked canonical semantic schema has SHA-256:

```text
0ec34ea3d563f1357097181cb0dd90586a65e8d4c85b644b45c6fd3f81bcc151
```

The JSON is development-time input for the converter. Phase 3 serializes only
numeric semantic IDs into RTNN. Future RawTherapee C++ code will mirror the
stable enum values and will not parse this JSON at runtime.

## Convert to RTNN v1

Convert the authenticated checkpoint and write a deterministic companion
manifest as `<output>.json`:

```sh
.venv/bin/python -m tools.neural_demosaic.convert_checkpoint \
    /path/to/demosaicnet/data/xtrans.pth \
    --output /tmp/demosaicnet-xtrans-v1.rtnn
```

Existing RTNN or companion files are not replaced unless `--force` is used.
Both generated suffixes are ignored by Git pending the model-redistribution
decision. The exact container ABI is documented in
`devnotes/rtnn-v1-format.md`.

The pinned conversion produces:

| Artifact | Size | SHA-256 |
| --- | ---: | --- |
| RTNN | 1,642,432 | `b4dd6ea4ba535e7f4aea249a2d589a80ca8584f60a605a5bce468c989529ccc2` |
| Padded payload | 1,639,744 | `e0e501a3f3a4905e3c7bb1ab1f0e3acb5598da818d6406d5cf6ed030ab5af606` |
| Canonical manifest | 10,919 | `f9b5d784356a455327304cfbfe302b2041a5a1a1eb2970134e3c1dfb621ec447` |

RTNN stores all 409,923 float32 parameters without changing their bit
patterns. The manifest deliberately omits timestamps, local paths, filenames,
host details, and Python environment versions so conversions in different
directories remain byte-identical.

## Authenticate and inspect RTNN v1

Phase 4 adds a reader that is intentionally independent from the writer. It
duplicates the reviewed wire constants, parses every field explicitly, reads
no companion JSON, applies bounded checked arithmetic before copying tensor
payloads, and accepts only the pinned Gharbi model binding by default.

Inspect a converted file with:

```sh
.venv/bin/python -m tools.neural_demosaic.inspect_rtnn \
    /tmp/demosaicnet-xtrans-v1.rtnn
```

The command emits canonical JSON derived from the authenticated RTNN and its
reviewed semantic binding. It reports stable format, model, tensor, offset,
size, and digest metadata without recording the input path, timestamp, or host.
Malformed input is rejected with a stable error category.

The Phase 4 PyTorch reference locally reproduces the eleven main valid
convolutions, centered sparse-input crop and concatenation, post-convolution,
and RGB output layer without importing upstream executable model code. Against
the pinned checkpoint, checkpoint-backed and RTNN-backed models have identical
tensor bytes and bit-identical output on seven fixed sparse X-Trans inputs.

## Native RTNN verification

Phase 6 adds dependency-free C++ tests around the production Phase 5 loader.
They are built only with the standard CMake `BUILD_TESTING` option enabled:

```sh
cmake --preset dev
cmake --build build/dev \
    --target rawtherapee-neuralmodel-tests rawtherapee-rtnn-inspect
ctest --test-dir build/dev -L neural-model --output-on-failure
```

The mandatory synthetic, corruption, and native-graph tests require no
checkpoint, RTNN, Torch, Python, or third-party test framework. Reviewed
artifact, inspection parity, golden inference, and activation-trace cases skip
when `GHARBI_XTRANS_RTNN` is absent. Enable them with:

```sh
GHARBI_XTRANS_RTNN=/tmp/demosaicnet-xtrans-v1.rtnn \
    ctest --test-dir build/dev -L neural-model --output-on-failure
```

Inspect the reviewed artifact from C++ with:

```sh
build/dev/tests/neuralmodel/rawtherapee-rtnn-inspect \
    /tmp/demosaicnet-xtrans-v1.rtnn
```

The native output is byte-identical to `inspect_rtnn` and has pinned SHA-256
`026f992aa9fbc7277e16b1f13c4f55c56aeadf7457b5cd2090cc181847bb069b`.
It contains no path, filename, timestamp, hostname, or companion-manifest data.
Invalid input is reported as `error [CODE]: message` with exit status 2.

## Golden inference corpus

Phase 7 tracks seven exact network input/output pairs under
`golden/demosaicnet-xtrans-v1/`. The fourteen `.f32le` files contain contiguous
little-endian NCHW float32 data and total 114,600 bytes. Their canonical
10,513-byte manifest has SHA-256:

```text
ed4b6ff5544ef361d3613fdf354544acd89db56059de249468ac416c238f3b92
```

The corpus uses the Phase 4 zero, constant, seeded-random, three impulse, and
alternating-saturated inputs unchanged. The random case also records shapes and
SHA-256 values after all eleven main ReLUs, the sparse-input concatenation, and
the post-convolution ReLU. Intermediate activation bytes are not stored.

Regenerate into a new directory from the reviewed RTNN artifact with:

```sh
.venv/bin/python -m tools.neural_demosaic.export_golden_corpus \
    /tmp/demosaicnet-xtrans-v1.rtnn \
    --output /tmp/demosaicnet-xtrans-v1-golden
```

The exporter refuses an existing output directory. It authenticates RTNN with
the independent reader and runs only the local reference graph under pinned,
single-threaded deterministic CPU settings. The fixtures represent exact
sparse network tensors: they apply no gamma wrapper, raw normalization, CFA
orientation transform, clipping, sample reinjection, or postprocessing. The
tracked attribution and upstream MIT license apply to these generated tests;
the corpus contains no checkpoint or RTNN weights.

## Native inference and numeric trace

Phase 8 adds a standalone C++ `DemosaicNetXTransExecutor`. It consumes the
immutable reviewed model and contiguous planar CHW float32 input, runs the
eleven valid main convolutions, centered sparse-input concatenation, post
convolution, and RGB output, and returns an image reduced by 24 pixels in each
dimension. It intentionally defines no raw normalization, CFA orientation,
boundary, gamma, tiling, or postprocessing policy.

The direct OIHW kernel is single-threaded per executor, compiler-vectorized
across output columns, and uses two reusable 64-byte-aligned feature buffers.
It uses neither Boost/BLAS nor an `im2col` buffer. Floating-point contraction
and fast-math transformations are disabled so every output retains a fixed
input-channel and kernel-tap accumulation order. Phase 9 will create one
executor per parallel tile worker.

Portable native output is compared numerically rather than byte-for-byte with
PyTorch's MKL-backed convolution. Every golden value must satisfy:

```text
abs(native - reference) <= 5e-6 + 1e-5 * abs(reference)
```

On the current GCC 13 optimized build, all 2,661 final values have maximum
absolute error `2.98023224e-6`, mean absolute error `1.69540432e-7`, and RMS
error `3.93236628e-7`. The 4,175 sampled intermediate values have maximum
absolute error `6.85453415e-7`.

The compact Phase 8 trace is stored under `native_trace/`. It samples every
channel at the four corners and center of all thirteen Phase 7 activations.
Its 10,433-byte canonical manifest has SHA-256:

```text
1415ffa3c8c072740f39b2c084525483910cb09e71dcfba22061901f0cf1bf66
```

Regenerate the trace without changing the Phase 7 corpus with:

```sh
.venv/bin/python -m tools.neural_demosaic.export_native_trace \
    /tmp/demosaicnet-xtrans-v1.rtnn \
    --output /tmp/demosaicnet-xtrans-native-trace-v1
```

Run the non-CTest developer benchmark for 64, 128, 192, and 256 pixel inputs
with:

```sh
GHARBI_XTRANS_RTNN=/tmp/demosaicnet-xtrans-v1.rtnn \
    build/dev/tests/neuralmodel/rawtherapee-neuralmodel-tests inference-benchmark
```

It reports median execution time, output throughput, and retained workspace
bytes without imposing a machine-dependent performance threshold.

## Phase 9 raw wrapper and CLI experiment

Phase 9 adds two PP3 strings for command-line development only:

```text
demosaicnet-xtrans-linear
demosaicnet-xtrans-gamma22
```

They intentionally do not appear in the public X-Trans method list or GUI.
Supply the reviewed artifact explicitly:

```sh
RT_DEMOSAICNET_XTRANS_MODEL=/tmp/demosaicnet-xtrans-v1.rtnn \
    build/dev/rtgui/rawtherapee-cli -p linear.pp3 -o output.tif -c input.RAF
```

Success and fallback are both unconditional diagnostics. The benchmark treats
`falling back to 3-pass (Markesteijn)` as failure even when the CLI export
succeeds. Successful models are retained in a thread-safe process cache;
failed loads can be retried.

The independent raw-wrapper corpus lives under
`golden/demosaicnet-xtrans-raw-wrapper-v1/`. Its fourteen little-endian float32
blobs contain seven scaled scalar mosaics and full-size normalized CHW RGB
outputs. The canonical manifest SHA-256 is:

```text
bd415eb33ecb10c01c7ef127039e6ef69267d9398d01edbb7a339a661790b526
```

Regenerate it only from the strict RTNN reader:

```sh
.venv/bin/python -m tools.neural_demosaic.export_raw_wrapper_corpus \
    /tmp/demosaicnet-xtrans-v1.rtnn --output /tmp/raw-wrapper-corpus
```

Run the analytical/external/RAF benchmark with development-only image
dependencies installed:

```sh
.venv/bin/python tools/benchmark_xtrans_demosaicnet.py \
    --rawtherapee-cli build/dev/rtgui/rawtherapee-cli \
    --model /tmp/demosaicnet-xtrans-v1.rtnn \
    --ground-truth-dir /path/to/upstream-ten-images \
    --raf /path/to/DSCF0771.RAF \
    --crop DSCF0771:3450,1750,700,500 \
    --work-dir /tmp/xtrans-phase9
```

The native raw-wrapper CTest is unconditional; reviewed numerical parity skips
with code 77 unless `GHARBI_XTRANS_RTNN` is set. The current real-image and
performance verdict is recorded in `devnotes/xtrans-neural-phase9-report.md`.

## Tests

Unit tests generate ordinary local tensor dictionaries and never execute
untrusted pickle content:

```sh
.venv/bin/python -m pytest tools/neural_demosaic/tests
```

Run all optional integration tests against the actual pinned checkpoint and
RTNN artifact:

```sh
GHARBI_XTRANS_CHECKPOINT=/path/to/demosaicnet/data/xtrans.pth \
GHARBI_XTRANS_RTNN=/tmp/demosaicnet-xtrans-v1.rtnn \
    .venv/bin/python -m pytest tools/neural_demosaic/tests
```

The corresponding integration tests skip when either environment variable is
unset. They cover conversion identities, strict RTNN reading, metadata
equivalence, bit-exact Python reference-network parity, byte-identical golden
and trace regeneration, and bounded native C++ parity. Validation of the
committed corpus and trace, the independent reader, and the mandatory native
and corruption suites require no external checkpoint or model artifact.

## Developer-only X-veon ONNX experiment

X-veon is a separate hidden experiment. Build with ONNX Runtime CPU 1.27.0:

```sh
/usr/bin/cmake --preset dev \
    -DWITH_ONNXRUNTIME=ON \
    -DONNXRUNTIME_ROOT=/path/onnxruntime-linux-x64-1.27.0
cmake --build build/dev --target rawtherapee-cli rawtherapee-neuralmodel-tests
```

Use the literal PP3 method `xveon-xtrans-onnx` and provide the external model:

```sh
RT_XVEON_XTRANS_MODEL=/path/xtrans.onnx \
    build/dev/rtgui/rawtherapee-cli -p profile.pp3 -o output.tif -c input.RAF
```

The loader accepts only the 15,536,134-byte model with SHA-256
`45b1fa22b0027868fd5c20ec7b59234ed5aeb35de89fbc0950a4bec67f328500`
and verifies its fixed input/output and epoch/base-width/PSNR metadata. Real
native tests are optional:

```sh
XVEON_XTRANS_ONNX=/path/xtrans.onnx \
    ctest --test-dir build/dev -R xveon_ --output-on-failure
```

Install `requirements-onnx.txt` only for the independent Python reference.
The prior gamma comparison format is frozen; the four-method tool is invoked
as `python -m tools.neural_demosaic.compare_xveon_outputs`. See
`devnotes/xtrans-xveon-experiment.md` for the exact tiling contract, measured
quality result, and licensing boundary. Neither the upstream repository nor
the model had an explicit license at the pinned revision, so weights remain
external and the method remains absent from the GUI.

For the optional RX 7800 XT MIGraphX experiment, configure the same build with
`-DWITH_MIGRAPHX=ON -DMIGRAPHX_ROOT=/opt/rocm`, then use:

```sh
RT_XVEON_XTRANS_MODEL=/path/xtrans.onnx \
RT_XVEON_XTRANS_BACKEND=migraphx \
RT_XVEON_MIGRAPHX_CACHE_DIR=/path/to/private-cache \
    build/dev/rtgui/rawtherapee-cli -p profile.pp3 -o output.tif -c input.RAF
```

`compare_xveon_backends` reports deterministic float-tile error statistics;
`compare_xveon_migraphx` validates full-resolution TIFF parity, timings,
repeated pixel determinism, phase texture, and the Phase 11 gates. Real GPU
CTest coverage is `rtengine.neural_model.xveon_migraphx` and skips without the
external model or accessible `/dev/kfd`.
