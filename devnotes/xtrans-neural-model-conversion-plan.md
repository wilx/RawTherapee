# Plan: converting published X-Trans PyTorch models for RawTherapee

## Purpose

This plan covers the first implementation stage of the neural X-Trans
experiment: translating an upstream PyTorch .pth checkpoint into a deterministic,
safe, portable model artifact that RawTherapee C++ code can load without
Python, PyTorch, pickle, or another neural-network runtime.

The first supported input is the pretrained Gharbi DemosaicNet X-Trans
checkpoint. The converter and file format should leave room for other fixed
architectures, but version 1 must not become a general neural-network format.
Deep Demosaick should only be added after the Gharbi conversion and inference
path is validated.

This plan complements
[xtrans-neural-demosaicing.md](xtrans-neural-demosaicing.md), which evaluates
the models, their input domains, runtime costs, and integration risks.

## Outcomes

At the end of this stage the repository should have:

1. a development-only Python converter for the pinned Gharbi checkpoint;
2. a documented, versioned RawTherapee neural-model binary format;
3. a converted float32 model plus provenance and license information;
4. a small C++ loader that rejects malformed or incompatible files;
5. Python and C++ inspection tools that report identical tensor metadata;
6. a minimal CTest setup, integrated with the existing CMake build and guarded
   by the standard BUILD_TESTING option;
7. deterministic conversion, loader, and corruption/schema tests registered
   with CTest; and
8. golden reference data sufficient to implement C++ inference next.

This stage does not add the demosaicing method to the GUI and does not yet
implement the convolution engine.

## Non-goals for version 1

* Do not load .pth or any pickle format from RawTherapee.
* Do not embed Python, libtorch, ONNX Runtime, or another inference framework
  in the RTNN/Gharbi runtime. The later X-veon experiment is a separate,
  optional development target and leaves ordinary builds dependency-free.
* Do not convert arbitrary PyTorch models.
* Do not quantize weights to float16 or integers.
* Do not fuse, prune, retrain, fine-tune, or otherwise change model values.
* Do not encode input gamma, CFA transforms, tiling, or postprocessing in the
  weight artifact. Those belong to the inference method and its model contract.
* Do not convert the Kokkinos--Lefkimmiatis checkpoint in the first milestone.

## Pinned upstream input

The first conversion input is:

| Field | Value |
| --- | --- |
| Project | https://github.com/mgharbi/demosaicnet |
| Inspected revision | 959e9d1630976b421d5af5e35b2e2a01f5630e5c |
| Repository path | demosaicnet/data/xtrans.pth |
| File size | 1,644,547 bytes |
| SHA-256 | 3759a13296fecebae83a36a8f0c3890d8a2f7d33e5f8149ed9b76a70f8285bc7 |
| Upstream license | MIT |
| Architecture | XTransDemosaick, depth 11, width 64, no convolution padding |

The converter must refuse this source if its SHA-256 differs, unless the caller
explicitly supplies and records a different expected hash. A hash override is
for auditing a proposed upstream update, not for silently accepting an unknown
file.

The source .pth file should not be committed automatically. Whether the
converted model may be distributed with RawTherapee is a separate licensing
decision; local development can proceed with a downloaded checkpoint whose
origin and digest are recorded.

## Phase 1: create the isolated checkpoint-intake tool

### Proposed repository layout

Use a narrowly scoped directory rather than adding model conversion to an
existing benchmark script:

    tools/neural_demosaic/
        README.md
        __init__.py
        inspect_checkpoint.py
        schema.py
        requirements.txt
        requirements-torch.txt
        tests/
            test_inspect_checkpoint.py

Phase 1 writes only a deterministic inspection manifest to a developer-selected
location outside tracked source, for example:

    /tmp/demosaicnet-xtrans-v1.manifest.json

RTNN output and its install/search contract remain Phase 3 work. This keeps the
safe checkpoint boundary independently testable before freezing the binary
container.

### Python environment

The inspector is a developer tool, not a runtime dependency. Its verified
environment is Python 3.12 with CPU-only PyTorch 2.12.1, NumPy 2.3.5, and
pytest 8.4.2. The PyTorch dependency is kept in a separate requirements file so
it can be installed exclusively from the official CPU wheel index.

Load the state dictionary using the safest available API:

    torch.load(path, map_location="cpu", weights_only=True)

Do not import or execute the upstream model module merely to deserialize the
checkpoint. The artifact is an OrderedDict of tensors and should not require
upstream Python classes.

The Phase 1 inspector must:

1. read the source as bytes and compute SHA-256 before deserialization;
2. verify the expected source hash;
3. use weights-only loading and force all storage to CPU;
4. require an OrderedDict or mapping containing exactly the expected keys;
5. reject sparse, quantized, complex, non-floating, or non-finite tensors;
6. require float32, then detach and make each tensor contiguous without
   changing its bit pattern;
7. retain canonical PyTorch OIHW convolution order;
8. hash canonical little-endian float32 tensor bytes;
9. emit canonical JSON without timestamps or local paths; and
10. write to a temporary file, validate it, then atomically rename it.

The inspector never overwrites an existing manifest unless an explicit force
option is given. The pinned checkpoint currently produces 26 tensors, 409,923
parameters, 1,639,692 payload bytes, and canonical manifest SHA-256
`371a3e20bac66877238e44d36e349078953c0b6c4e256299f64d66bbd8b72848`.

## Phase 2: freeze the RTNN semantic tensor schema

Phase 1 validates these source keys and shapes before producing its inspection
manifest. Phase 2 assigns stable RTNN semantic tensor IDs and freezes their
mapping; the binary format must not depend on PyTorch key strings at runtime.
The authoritative contract is the canonical, language-neutral file
`tools/neural_demosaic/schemas/demosaicnet-xtrans-v1.json`, whose SHA-256 is
`0ec34ea3d563f1357097181cb0dd90586a65e8d4c85b644b45c6fd3f81bcc151`.

Architecture ID `1` identifies `DEMOSAICNET_XTRANS_V1`; ID `0` is invalid.
Tensor IDs are scoped to that architecture and assigned in execution order:

* main convolution layer `n` weight: `2n-1`, for `n=1..11`;
* main convolution layer `n` bias: `2n`;
* post-convolution weight and bias: `23` and `24`; and
* output weight and bias: `25` and `26`.

The schema also freezes the network graph: eleven valid 3x3 convolution-plus-
ReLU layers at width 64, centered sparse-input crop and concatenation, valid
3x3 post-convolution plus ReLU, and 1x1 RGB output. The result is 24 pixels
smaller in each dimension and has receptive-field radius 12.

The schema is bound to the pinned checkpoint SHA-256 and the unchanged Phase 1
manifest SHA-256
`371a3e20bac66877238e44d36e349078953c0b6c4e256299f64d66bbd8b72848`.
The Python validator cross-checks every source name, shape, layout, count, and
tensor digest. This JSON remains development-time input: Phase 3 serializes
numeric IDs, and C++ never parses it at runtime.

The source state-dictionary keys are:

    main_processor.conv1.weight
    main_processor.conv1.bias
    ...
    main_processor.conv11.weight
    main_processor.conv11.bias
    fullres_processor.post_conv.weight
    fullres_processor.post_conv.bias
    fullres_processor.output.weight
    fullres_processor.output.bias

There are 26 tensors with these exact shapes:

| Tensor group | Count | Expected shape |
| --- | ---: | --- |
| conv1 weight | 1 | [64, 3, 3, 3] |
| conv1 bias | 1 | [64] |
| conv2 through conv11 weights | 10 | [64, 64, 3, 3] |
| conv2 through conv11 biases | 10 | [64] |
| post_conv weight | 1 | [64, 67, 3, 3] |
| post_conv bias | 1 | [64] |
| output weight | 1 | [3, 64, 1, 1] |
| output bias | 1 | [3] |

The expected total is 409,923 float32 parameters, or 1,639,692 tensor payload
bytes. Check the parameter count independently from directory offsets.

Reject missing keys, additional keys, reordered semantic IDs, wrong ranks,
wrong dimensions, integer tensors, NaNs, and infinities. A future upstream
checkpoint with a different schema is a new model review, not an automatic
upgrade.

New weights with the same graph reuse architecture ID `1` and the existing
tensor IDs but require a separately reviewed model binding. Any graph,
tensor-role, shape, layout, or ID change requires a new architecture ID;
existing IDs are never renumbered or repurposed.

## Phase 3: RawTherapee neural-model file format version 1

RTNN v1 is now frozen and implemented. Its authoritative byte-level contract is
`devnotes/rtnn-v1-format.md`. The format transports validated tensor values and
identity metadata only; it does not describe or execute a graph.

The fixed layout uses a 192-byte header, 26 ascending 96-byte semantic tensor
records, and one 64-byte-aligned payload region. All integers and IEEE-754
float32 values are little-endian. Convolution tensors remain in canonical OIHW
order, vector tensors remain contiguous, and every gap and trailing alignment
byte is zero. The header binds architecture ID 1, model revision 1, the Phase 2
semantic-schema digest, the source-checkpoint digest, and the SHA-256 of the
complete padded payload. Every directory record also contains its tensor's
SHA-256.

The current artifact has 1,639,692 tensor bytes, a 1,639,744-byte padded payload
region, and a total size of 1,642,432 bytes. Its deterministic identities are:

| Artifact | SHA-256 |
| --- | --- |
| RTNN | `b4dd6ea4ba535e7f4aea249a2d589a80ca8584f60a605a5bce468c989529ccc2` |
| Padded payload | `e0e501a3f3a4905e3c7bb1ab1f0e3acb5598da818d6406d5cf6ed030ab5af606` |
| Companion manifest | `f9b5d784356a455327304cfbfe302b2041a5a1a1eb2970134e3c1dfb621ec447` |

The converter regenerates and validates the Phase 1 manifest in memory, checks
it against the authenticated Phase 2 schema, and only then serializes the
canonical tensor bytes. It publishes a canonical JSON companion next to the
RTNN file. That manifest records stable upstream provenance, semantic IDs,
source keys, shapes, layouts, offsets, sizes, and digests. It excludes
timestamps, local paths, output filenames, host information, and Python
environment details so both generated files are byte-identical across output
directories.

Generated `.rtnn` and `.rtnn.json` files remain ignored local artifacts until
the licensing and packaging gate is resolved. Runtime C++ will bind tensors by
numeric semantic enum and will parse neither the Phase 2 schema nor the
companion JSON.

## Phase 4: prove that conversion is lossless and deterministic

Phase 4 is implemented as a writer-independent Python reader plus an exact
local PyTorch reference graph. The reader intentionally imports no Phase 3
writer constants or parsing helpers and never reads the companion JSON. Its
public entry points accept only the reviewed Gharbi binding: architecture ID 1,
model revision 1, the exact schema and checkpoint identities, all 26 tensor
roles, and the pinned complete-file SHA-256.

The reader validates the fixed header and directory, bounded counts and sizes,
checked uint64 arithmetic, canonical record order, shapes, layouts, offsets,
64-byte alignment, non-overlap, minimal zero padding, tensor and payload
digests, finite float32 values, and finally the complete artifact digest. It
returns immutable canonical tensor bytes and exposes stable error categories
for I/O, format, limit, range, schema, digest, and finite-value failures.

An inspection CLI emits canonical metadata derived from the RTNN and its
reviewed binding without including a local path, timestamp, hostname, or
companion-manifest input. Its artifact, model, summary, and tensor sections
match the corresponding Phase 3 conversion-manifest sections exactly.

The local `DemosaicNetXTransReference` reproduces the pinned upstream graph
without importing upstream executable code. Separate factories populate it
from Phase 1 checkpoint bytes and from the strict RTNN reader. Under
single-threaded deterministic CPU execution with oneDNN disabled, all tensor
bytes and network outputs are bit-identical for seven fixed sparse inputs:

* zero, 32 by 32;
* constant observed R/G/B values, 31 by 35;
* seeded random observed values, 37 by 38;
* red, green, and blue impulses, each 36 by 36; and
* alternating zero/saturated observed samples, 35 by 36.

Every output is finite and exactly 24 pixels smaller in each spatial dimension.
The corruption suite uses an independently assembled synthetic RTNN and
exercises every header and directory field, truncation and extension, hard
limits, arithmetic overflow, order, dimensions, gaps, overlaps, alignment,
padding, internal hashes, full-file identity, and non-finite payloads without
requiring the external checkpoint. Checkpoint/pickle rejection remains Phase 1
coverage rather than being duplicated here.

## Phase 5: C++ loader

Phase 5 adds a C++11 loader independent of the demosaicing pipeline:

    rtengine/neuralmodel.h
    rtengine/neuralmodel.cc
    rtengine/rtnnreader_p.h
    rtengine/rtnnreader.cc
    rtengine/demosaicnetxtransmodel.h
    rtengine/demosaicnetxtransmodel.cc

`rtnnreader_p.h` is a private binding-driven parser interface retained for the
independent synthetic fixtures in Phase 6. RawTherapee exposes no generic or
unverified public RTNN loading mode. The public
`loadDemosaicNetXTransModel(const Glib::ustring&)` entry point accepts only the
reviewed architecture 1, model revision 1 artifact and returns either a complete
immutable model or a structured error. Its stable error names mirror the
Python reader, with `NONE` and `ALLOCATION` added for the C++ result contract.

The loader must:

1. open the file read-only and determine its actual size;
2. parse every scalar explicitly rather than casting a mapped byte buffer;
3. apply configured maximum file, directory, tensor, and dimension sizes;
4. validate header, ranges, alignment, non-overlap, counts, and checksums;
5. validate the exact DemosaicNet X-Trans schema;
6. copy float32 values into owned aligned storage;
7. decode little-endian float bits into native representation without aliasing;
8. verify all loaded floats are finite;
9. preserve canonical OIHW/vector order and every 64-byte tensor alignment; and
10. return a complete immutable model or a structured error, never a partially
    initialized model.

The implementation uses glibmm's existing SHA-256 support and parses every
integer explicitly rather than casting file bytes to native structures. It
authenticates the complete file before allocating returned tensor storage. A
portable C++11 over-allocation scheme owns a `float[]` and selects a 64-byte
aligned float element within it; C++17 `std::aligned_alloc` is neither available
nor required. Decoding through a little-endian `uint32_t` plus `memcpy` preserves
the reviewed float bits on little- and big-endian hosts.

The portable file and Phase 5 in-memory representation both remain canonical
OIHW/vector data. SIMD-specific repacking is deferred until a convolution
kernel defines the layout it consumes. Model discovery, process- or
engine-context caching, first-failure logging, licensing, and packaging are
also deferred to runtime integration. The Phase 5 loader is stateless and
performs no logging.

Phase 5 build and local smoke verification cover the reviewed artifact,
bit-for-bit native tensor values, repeated loads, alignment, and basic I/O and
size failures. It is not considered a completed security gate until the Phase
6 independent fixtures, full corruption matrix, sanitizers, and CTest suite
pass.

## Phase 6: native loader verification and CTest

Phase 6 introduces CTest through its standard `BUILD_TESTING` option. When the
option is enabled, CMake builds two non-installed development executables:
`rawtherapee-neuralmodel-tests` and `rawtherapee-rtnn-inspect`. When it is
disabled, neither target nor the test subdirectory enters the build graph, and
normal RawTherapee binaries remain unchanged.

The three production loader sources form the private static
`rtengine-neuralmodel-loader` target. Both `rtengine` and the native test tools
consume that same implementation. This permits strict and sanitizer builds of
the security boundary without compiling unrelated `dcraw.cc`, which triggers a
known GCC 13 internal compiler error under those configurations.

The native tests run through:

    ctest --test-dir build/dev --output-on-failure

All seven stable test names carry both the `rtengine` and `neural-model` labels.
Five mandatory tests use a dependency-free C++ fixture assembled independently
of the Phase 3 writer. The fixture has the frozen 26 tensor shapes and 409,923
parameters, deterministic finite float bits, independently calculated GLib
SHA-256 values, and a private parser binding. Its corruption tables exercise
the header, directory, payload, digest, limits, checked arithmetic, padding,
non-finite values, and file-I/O paths while asserting exact stable error names.

The frozen tensor lengths happen to be multiples of 64 except for the final
three-float bias. Therefore this model has no inter-tensor padding bytes: the
suite asserts that invariant, rejects inserted gaps as noncanonical, and tests
nonzero padding using the actual 52-byte trailing region.

The two optional cases read `GHARBI_XTRANS_RTNN`. They return CTest skip code 77
when it is absent, keeping the complete mandatory suite self-contained. When it
is present, the tests authenticate the reviewed file, compare first, middle,
and last values of every tensor against an independent little-endian decode of
the raw RTNN payload, require every pointer to be 64-byte aligned, and compare
two complete loads bit-for-bit. The Phase 3 companion manifest contains tensor
digests rather than selected float samples; raw-payload comparison plus the
pinned whole-file digest is the intended independent value check.

The native inspection utility accepts one reviewed artifact and emits the same
canonical JSON as the Python Phase 4 inspector. Architecture symbols, tensor
symbols, and Python source keys live only in its tool sources, not in the
runtime model. Its output SHA-256 is pinned as
`026f992aa9fbc7277e16b1f13c4f55c56aeadf7457b5cd2090cc181847bb069b`.
Failures use `error [CODE]: message` and exit status 2.

Run the optional cases with:

    GHARBI_XTRANS_RTNN=/tmp/demosaicnet-xtrans-v1.rtnn \
        ctest --test-dir build/dev -L neural-model --output-on-failure

The eventual convolution tests should consume the same immutable model object,
but convolution implementation is the following project stage.

## Phase 7: golden data for the C++ inference stage

Phase 7 freezes a small tracked corpus under
`tools/neural_demosaic/golden/demosaicnet-xtrans-v1`. A development-only
exporter accepts the reviewed RTNN artifact, loads it through the independent
Phase 4 reader, and executes the local PyTorch graph without `torch.load` or
upstream executable code.

The corpus contains the seven Phase 4 sparse inputs and their exact RGB
outputs as contiguous little-endian NCHW float32 files. The canonical manifest
binds every shape, byte count, and SHA-256 to the reviewed model, checkpoint,
semantic schema, RTNN, PyTorch 2.12.1 CPU build, and deterministic execution
settings. It contains no timestamp, path, filename from the invoking host, or
model weights. Its SHA-256 is
`ed4b6ff5544ef361d3613fdf354544acd89db56059de249468ac416c238f3b92`.

For the seeded-random case, the manifest also records shapes and SHA-256 values
after all eleven main ReLUs, the sparse-input concatenation, and the post ReLU.
This identifies the first divergent native layer without committing large
activation buffers. The random input and valid output span all 36 CFA phases
and exercise both sides of every valid-convolution boundary.

These are exact network tensors, not a raw-domain policy. No gamma wrapping,
raw normalization, orientation mapping, clipping, reinjection, or
postprocessing is represented. Golden parity therefore cannot select between
the direct-linear and gamma-wrapped wrappers. Attribution and the upstream MIT
license accompany the roughly 112 KiB of float fixtures; the checkpoint and
RTNN remain external.

Unconditional tests authenticate the tracked manifest and every blob. Optional
tests driven by `GHARBI_XTRANS_RTNN` regenerate twice into different
directories and require byte-identical corpus files.

## Phase 8: native fixed-graph inference

Phase 8 implements the fixed graph as a standalone
`DemosaicNetXTransExecutor` against the immutable Phase 5 model. The fallible
factory accepts only architecture 1, revision 1 with the complete frozen tensor
contract. Each non-copyable executor shares the model, owns two reusable
64-byte-aligned workspaces, accepts contiguous planar CHW float32 tensors, and
is deliberately not thread-safe. Phase 9 will use one executor per tile
worker.

The kernel consumes canonical OIHW weights directly. It accumulates in fixed
input-channel and kernel-tap order while the compiler vectorizes independent
output columns. Fast math and floating-point contraction are disabled. No
Boost/BLAS, `im2col`, handwritten SIMD, OpenMP, or architecture-specific
repacking is used. The centered three-channel crop is accumulated before the
64 feature channels without allocating the logical 67-channel activation in
production. Only a private diagnostic path materializes it for tests.

The standalone API accepts inputs of at least 25 by 25 and at most 262,144
input pixels, checks every size product and buffer count, rejects non-finite
input or activations, and copies the final result to the caller only after the
complete graph succeeds. The output is always 24 pixels smaller in both axes.
This cap enforces the Phase 9 tiling boundary rather than permitting accidental
full-frame feature allocation.

MKL and portable direct convolution use different float32 accumulation
implementations, so native parity is numerical rather than byte-exact. Every
value must satisfy `abs(native-reference) <= 5e-6 + 1e-5*abs(reference)`.
The current optimized GCC 13 build measured all 2,661 final values at maximum
`2.98023224e-6`, mean `1.69540432e-7`, and RMS `3.93236628e-7`. Repeated native
runs remain bit-identical.

A separate 16,700-byte Phase 8 trace samples all channels at the four corners
and center of the thirteen seeded-random activations. It is bound to the
unchanged Phase 7 manifest and full-activation hashes; its canonical manifest
SHA-256 is
`1415ffa3c8c072740f39b2c084525483910cb09e71dcfba22061901f0cf1bf66`.
Across its 4,175 values the current native maximum is `6.85453415e-7`.

Dependency-free CTest cases cover synthetic graph paths, ReLU and final signed
output, crop/concatenation order, dimensions, limits, workspace reuse,
non-finite rejection, unchanged output on failure, and repeated execution.
Optional reviewed-artifact cases compare all final and sampled intermediate
values and print maximum, mean, RMS, p90, and p99 statistics. A non-registered
developer benchmark reports the 64, 128, 192, and 256 input sizes without a
machine-dependent threshold. This phase remains outside the raw demosaic
pipeline and GUI.

## Phase 9: developer-only demosaic integration and quality gate

Phase 9 now provides two hidden PP3/CLI identifiers,
`demosaicnet-xtrans-linear` and `demosaicnet-xtrans-gamma22`, while keeping the
public method enum, GUI, translations, defaults, and history unchanged. It
requires an explicit `RT_DEMOSAICNET_XTRANS_MODEL` path and strongly caches
only successfully authenticated models. Any model, CFA, allocation,
non-finite, or inference error emits a stable diagnostic and reruns
Markesteijn three-pass.

The wrapper shares the deterministic 18-matrix CFA canonicalizer with the
Rafinazari experiment, scatters scaled raw values into sparse canonical RGB,
uses reflect-without-edge-repetition boundaries, and executes fixed 192x192
inputs with non-overlapping 168x168 cores. Core origins preserve the six-pixel
CFA phase. OpenMP creates one reusable executor only for workers that receive
a tile; no full-frame feature activation exists. Linear and the authors'
gamma-2.2 RAW wrapper add no reinjection or undocumented processing.

The independent tracked raw-wrapper corpus has manifest SHA-256
`bd415eb33ecb10c01c7ef127039e6ef69267d9398d01edbb7a339a661790b526`.
All seven native cases, including both seam orientations, pass the Phase 8
tolerance. The benchmark writes neutral profiles, rejects loud fallback,
supports analytical DNGs, external RGB ground truth, repeated RAFs and named
crops, and records quality, time, RSS, model, and workspace data.

The current quality gate fails. On `DSCF0771.RAF`, one completed direct-linear
40 MP warm-up took at least 584 seconds versus 3.94 seconds for Markesteijn
(greater than 148x) and showed strong CFA-phase texture. See
`devnotes/xtrans-neural-phase9-report.md`. Phase 10 must not begin on this
implementation.

## Phase 10: packaging and GUI

Only if Phase 9 demonstrates useful quality, stability, and acceptable speed,
resolve model distribution and expose “DemosaicNet X-Trans (experimental)” in
the demosaic GUI. Complete translations, history, profile editing, partial
paste, preview/export behavior, and missing-model handling in this phase. If
the quality gate fails, do not expose the method in the GUI.

## Phase 11: developer-only X-veon MIGraphX acceleration and CPU parity gate

Status: implemented and measured on the reviewed RX 7800 XT stack. The direct
backend, authenticated compiled cache, native parity coverage, full-resolution
comparison, and timing gate pass. Strict math is retained because fast math
improves warmed tile time by only 1.7 percent. See
`devnotes/xtrans-neural-phase9-report.md` for the recorded results.

Phase 11 is a separate continuation of the later X-veon experiment. It does
not reopen the failed Gharbi Phase 10 gate and does not authorize GUI exposure
or model distribution. Its purpose is to replace the current CPU-only X-veon
execution bottleneck with a directly linked MIGraphX GPU backend while keeping
the existing ONNX Runtime 1.27.0 CPU backend as the reviewed reference and
portable fallback build option.

The verified development stack for the first implementation is:

| Component | Reviewed value |
| --- | --- |
| GPU | AMD Radeon RX 7800 XT, `gfx1101`, 60 compute units, 16 GiB |
| ROCm | 7.2.1 |
| MIGraphX | 2.15.0, tweak `20250912-17-200-gde19b73ad` |
| X-veon model file | Repository `naorunaoru/x-veon`, revision `2e6b96c63559aa3909b0c7c1bc45dfd4b5dfe680`, regular file `web/public/xtrans.onnx` |
| X-veon ONNX | 15,536,134 bytes, SHA-256 `45b1fa22b0027868fd5c20ec7b59234ed5aeb35de89fbc0950a4bec67f328500` |
| CPU reference | ONNX Runtime 1.27.0 `CPUExecutionProvider` |

The required model is the actual `web/public/xtrans.onnx` file from that
revision, not a GitHub HTML page, Git LFS pointer, symlink, or MIGraphX compiled
cache file. After obtaining it, verify the size and SHA-256 above before setting
`RT_XVEON_XTRANS_MODEL` to its local path.

Initial feasibility checks are not parity acceptance results, but they prove
that the installed stack can execute this exact graph. ROCm enumerates the GPU
as `amdgcn-amd-amdhsa--gfx1101`; MIGraphX compiled and ran its native GPU GEMM
test; and MIGraphX parsed, compiled, and executed the complete X-veon
`1x4x288x288` to `1x3x288x288` FP16 U-Net. A fresh process spent approximately
46 seconds compiling the graph, while the driver's warmed 50-iteration timing
reported approximately 1.282 ms per tile. Phase 11 must measure end-to-end
RawTherapee behavior rather than treating that kernel-only number as an export
prediction.

### Build and backend selection

Add `WITH_MIGRAPHX`, default `OFF`, and a `MIGRAPHX_ROOT` cache path defaulting
to `/opt/rocm`. Discover the C header, version header, `libmigraphx_c.so`, and
the libraries required by its exported target explicitly. Require MIGraphX
2.15.0 for the first reviewed build and give development binaries a build
RPATH to the selected ROCm and MIGraphX library directories. Do not install or
bundle ROCm, MIGraphX, or their transitive libraries. `WITH_MIGRAPHX=OFF` must
leave the existing build graph and binaries unchanged.

Both `WITH_ONNXRUNTIME` and `WITH_MIGRAPHX` must be usable in one build so a
single native test process can compare both backends. Refactor model reading,
the 15,536,134-byte limit, SHA-256 authentication, immutable model bytes,
fixed tensor constants, and cache ownership out of the ONNX Runtime-specific
source. Keep `XVeonXTransRunner` as the demosaic wrapper's backend-neutral
interface.

Select the implementation with a development-only environment value:

```text
RT_XVEON_XTRANS_BACKEND=onnxruntime-cpu
RT_XVEON_XTRANS_BACKEND=migraphx
```

The default remains `onnxruntime-cpu` when it is compiled, preserving the
existing experiment. An explicitly selected unavailable or failed backend is
an error and triggers the existing loud Markesteijn overwrite; it must not
silently run the other neural backend and invalidate timing or parity results.
Cache successful runners by canonical model path plus backend and compile
options. Do not cache failures.

### Direct MIGraphX runner

Use MIGraphX's C API from C++11 RAII holders; do not introduce its C++ wrapper,
Python binding, or an ONNX Runtime MIGraphX build into RawTherapee. The runner
must:

1. retain and authenticate the same complete ONNX byte buffer used by the CPU
   path;
2. call `migraphx_parse_onnx_buffer` only after authentication;
3. require exactly one float32 parameter named `input` with shape
   `1x4x288x288` and one float32 result with shape `1x3x288x288`;
4. create the `gpu` target, enable host/device offload copies, disable
   exhaustive tuning, and initially disable fast math for the parity baseline;
5. wrap reusable host input storage in a MIGraphX argument, run sequentially
   under the runner's existing mutex, synchronize before reading output, and
   copy results to the caller only after successful finite-value validation;
6. preserve signed output and the existing tiling, CFA, blending, and fallback
   contracts unchanged; and
7. report backend, MIGraphX version, target/agent identity when available,
   compile source, compile time, and inference time separately in the
   developer diagnostic.

The exact authenticated model digest already binds the reviewed custom ONNX
metadata (`epoch=399`, `base_width=32`, and `best_val_psnr=45.78`). MIGraphX
still has to validate the executable parameter and output shapes after parsing;
it must not accept an arbitrary same-shaped model or weaken the existing file
identity gate. Map every C API operation to the existing structured
`NeuralModelErrorCode`; the MIGraphX C API's coarse status values must be
augmented with the failed operation name rather than exposed as an unexplained
generic error.

### Compiled-program cache

Process caching removes repeated compilation within one RawTherapee process,
but CLI exports start new processes. Add a developer-only compiled-program
cache only after fresh-compilation parity passes.

The cache identity must include the ONNX SHA-256, MIGraphX complete version,
GPU ISA, offload-copy/fast-math/exhaustive-tune settings, and a cache-format
revision. Write the compiled MIGraphX program and canonical identity manifest
through temporary files and atomic renames. Record the compiled payload's size
and SHA-256. Accept cache files only from a user-owned directory that is not
group- or world-writable, reject symlinks and ownership/mode violations, and
fall back to recompilation on any identity, digest, load, or first-run failure.
Never fall back to unauthenticated ONNX weights. Generated compiled programs
remain external and ignored because they contain the unlicensed model values
and are GPU/runtime specific.

Require a program loaded from cache to reproduce freshly compiled MIGraphX
output exactly on the deterministic tile. Measure and report cold parse,
compile, serialization, cache-load, first-run, and warmed-run times separately.
The cold compilation cost is not folded into steady-state export timing, but a
working cache is required before this backend can be considered usable outside
a single long-lived editor process.

### CPU-versus-GPU numerical comparison

The ONNX Runtime CPU runner remains the reference implementation. Generate the
same deterministic canonical X-veon tile already used by
`rawtherapee-xveon-tile`, run it repeatedly through both backends in the same
build, and compare all 248,832 output floats. Record exact matches, maximum,
mean, RMS, p90 and p99 absolute error, relative error away from zero, and the
number satisfying the existing `5e-6 + 1e-5 * abs(cpu)` diagnostic tolerance.
Do not require bit identity across CPU and GPU FP16 implementations and do not
silently widen that existing diagnostic tolerance.

The backend acceptance bounds are all values finite, maximum absolute error at
most `0.005`, RMS at most `0.0005`, and p99 absolute error at most `0.001` in
normalized network output. These aggregate bounds are additional, explicitly
reviewed GPU criteria; they do not change the tighter CPU/native parity tests.
Repeat the comparison with fast math enabled. Retain fast math only if it
passes the same numerical and image gates and improves warmed inference by at
least 10 percent; otherwise production experiment settings remain strict.

Run the existing 173x31 rotated/translated full-wrapper case through both
backends and add horizontal and vertical multi-tile cases. These compare CFA
canonicalization, leading reflection, right/bottom zero extension, overlap
ramps, accumulation, and coordinate restoration in addition to network
inference. Require deterministic repeated GPU output and no non-finite values,
seams, channel swaps, or fallback.

### Existing images and full-RAF comparison

Reuse the current CPU-only X-veon comparison assets under
`devnotes/images/xtrans-neural/DSCF0771/` as the visual baseline. Their
canonical manifest binds the source RAW, CPU X-veon TIFF, ICC profile,
full-third PNG, and 500-percent earring PNG. Do not regenerate or overwrite
those baseline files merely because a new backend is being tested.

For numerical comparison, reuse the external 16-bit CPU X-veon TIFF only when
its size, dimensions, ICC digest, and SHA-256
`8093d83b8c625bafc6424476220cea286f446bce8d89f9ea57562d226aefea0b`
match the tracked manifest; otherwise regenerate it with the pinned CPU
backend. Produce a fresh GPU TIFF with identical PP3 and unrelated processing
disabled. Compare every RGB sample and report CPU-versus-GPU CPSNR/PSNR, SSIM,
maximum/mean/RMS/p99 difference, channel means, observed-sample disagreement,
and 3x3 CFA-phase bias/RMS on crop `(3450,1750,700,500)`.

Require CPU-versus-GPU rendered CPSNR of at least 60 dB, SSIM of at least
0.999, absolute per-channel full-frame mean delta at most `0.0005`, and no
backend-induced crop phase-RMS increase greater than `0.00025`. Generate
untracked GPU full-third and nearest-neighbour 500-percent earring PNGs in the
same geometry as the tracked CPU files and inspect them side by side at 100
and 500 percent. There must be no visible colour, texture, seam, clipping,
sharpness, or earring regression. Commit separate GPU images only if they are
needed to document a reviewed difference; the existing CPU images remain the
stable baseline.

Also rerun the seven analytical TIFF cases through MIGraphX and compare both
against their CPU X-veon results and the existing Markesteijn references. GPU
acceleration must not change the already documented X-veon quality findings;
it cannot turn a numerical backend difference into a claimed demosaicing
improvement.

### Native tests, benchmark, and decision gate

Extend CTest with:

* dependency-free backend-selection, unavailable-backend, cache-key, and loud
  fallback tests;
* MIGraphX build/version and model-contract failures that do not require a
  GPU where possible;
* optional real-GPU deterministic tile, CPU parity, compiled-cache reload,
  full-wrapper, and injected-failure cases; and
* exact checks that a partial GPU result is discarded before Markesteijn
  overwrites all output planes.

Real-model/GPU cases require `XVEON_XTRANS_ONNX` and a usable `/dev/kfd`; they
return skip code 77 otherwise. Mandatory CI must not download the model or
require ROCm. Run normal, debug, strict-source, ASan/UBSan, and
`WITH_MIGRAPHX=OFF` builds; CTest with CPU only, MIGraphX built without a
model, and both backends with the reviewed model; repeated output determinism;
`ldd`; and `git diff --check`.

For `DSCF0771.RAF`, run one warm-up and three measured exports for the cached
GPU backend on the same host. Report cold compilation separately, then compare
the median cached total with the existing CPU X-veon median of 45.60 seconds
and Markesteijn's 3.94 seconds. Passing requires at least a 2x speedup over the
CPU X-veon median, no more than 10x Markesteijn, no more than 4 GiB additional
peak host RSS, bounded/reported VRAM use, and all numerical and visual parity
criteria above.

A Phase 11 pass retains MIGraphX as an optional hidden backend for the X-veon
experiment. It still does not permit model packaging or GUI exposure while
the upstream model lacks an explicit compatible license. A parity, stability,
cache, or quality failure leaves the existing CPU experiment unchanged and
the MIGraphX path disabled.

## Phase 12: developer-only PackedXTransNet experiment

Phase 12 evaluates Danylo Kelvich's compact PackedXTransNet through the same
hidden ONNX Runtime and direct MIGraphX infrastructure. It is technically
usable, but the checkpoint is explicitly licensed CC BY-NC 4.0. RawTherapee
must therefore neither package the weights nor expose this as a normal GUI
method. The checkpoint, converted ONNX, compiled GPU programs, full TIFFs, and
source RAWs remain external.

### Reviewed source and deterministic conversion

The only accepted source is revision
`9c3cc5ab841c9afd2ed0bb702468950481043d06` and the actual repository file
`weights/packed_5183_3208.pt` (not a mutable release link):

```text
https://github.com/danylo-kelvich/neural-demosaic/blob/9c3cc5ab841c9afd2ed0bb702468950481043d06/weights/packed_5183_3208.pt
```

Its size is 646,145 bytes and its SHA-256 is
`1c78b888e3f885252f84c1b12f75dd0af179a62b48499c5808eeb773d1bfc161`.
The safe inspector accepts exactly 39 dense finite CPU float32 tensors,
158,865 stored values and 635,460 payload bytes. Of these, 158,683 are learned
parameters and 182 are analytically checked CFA masks and tent kernels.

The independent converter does not import upstream executable code. Python
3.12, CPU `torch==2.12.1`, `onnx==1.22.0`, and opset 18 produce a fixed
`1x1x288x288` input / `1x3x288x288` output graph. It reproduces the full-size
chroma-difference baseline, 3x packing, nine mosaic plus one phase channel,
width-32 stem, eight residual blocks, 27-channel head, 3x unpacking, and
baseline addition. The reviewed deterministic artifacts are:

| Artifact | Size | SHA-256 |
| --- | ---: | --- |
| ONNX | 1,673,648 | `ad000f496fe9b4a8493bc891dedc3a1e379aec86c93b2fb53f8b8a66a2888e3c` |
| Canonical conversion manifest | generated locally | `ebd978aef293d1cf35a5d15234ef185d0785ac222c9c10f6477903e74d4c338d` |

ONNX Runtime output on the seeded reference tile differs from the independent
PyTorch graph by maximum `8.94e-7`, RMS `1.46e-7`, and p99 `3.87e-7`.

### Hidden runtime contract

The engine recognizes only the literal PP3 method `packedxtransnet-onnx` and
does not add it to method enums, GUI lists, defaults, translations, history,
or fast-export controls. Runtime selection is explicit:

```sh
RT_PACKED_XTRANS_MODEL=/path/packedxtransnet.onnx
RT_PACKED_XTRANS_BACKEND=onnxruntime-cpu|migraphx
RT_PACKED_XTRANS_PRECISION=fp32|fp16
RT_PACKED_XTRANS_MIGRAPHX_CACHE_DIR=/private/cache/directory
```

The generic fixed-tile bridges validate the model-specific input/output names,
shapes, types and counts while preserving the existing X-veon behavior. Both
backends authenticate the entire ONNX before session creation. A process cache
is keyed by canonical path, fixed digest, backend and precision. The optional
MIGraphX disk cache additionally binds the model digest, complete runtime
version, `gfx1101`, FP32/FP16 selection, compile settings, compiled-payload
size and payload SHA-256; it enforces private ownership/modes and rejects
symlinks or stale manifests.

The raw wrapper maps all 18 X-Trans phase/orientation matrices to the
architecture's canonical CFA, sends `rawData/65535` as a single linear scalar
mosaic, preserves signed finite output, and applies no gamma, clipping,
reinjection, second white balance, false-colour suppression, or other
postprocessing. It uses 288-pixel tiles and edge repetition. Any input, CFA,
model, runtime, allocation, or non-finite failure discards partial output and
loudly reruns Markesteijn three-pass.

The derived receptive radius is 56 pixels. A 12-pixel margin was compared with
the nearest CFA-aligned safe margin of 60 on the complete `DSCF0771.RAF`
export. Margin 12 had normalized maximum error `0.005005`, RMS `3.12e-5`, and
p99 `1.37e-4` against margin 60. It fails the agreed maximum `5e-4` and p99
`1e-4` limits, so the supported default is margin 60, stride 168. Margin 12
remains only a developer diagnostic override.

### Measured backend and quality result

The ONNX Runtime CPU export completed in 15.64 seconds wall time, with 12.23
seconds in the neural wrapper and peak RSS about 1.76 GiB. Direct MIGraphX FP32
matched CPU after rendering with 99.44 percent exact 16-bit samples and no
sample differing by more than one code value. With an authenticated compiled
cache, three full-RAF FP32 exports took 6.73, 7.06, and 7.33 seconds, median
7.06 seconds; wrapper medians were approximately 3.61 seconds and peak RSS was
2.23 GiB. Global VRAM sampling indicated an approximately 308 MiB increase;
this ROCm build reports the process VRAM field as `UNKNOWN`, so that figure is
an upper-bound delta rather than precise process attribution.

MIGraphX FP16 differs from ONNX Runtime by maximum `0.01431`, RMS `0.000338`,
and p99 `0.001175` in the rendered normalized TIFF. It nevertheless passed the
real-image phase, colour and visual gates and its cached runs were 6.37, 6.29,
and 6.23 seconds, median 6.29 seconds: 10.9 percent faster than FP32. FP16 is
therefore retained as an explicit developer option; FP32 remains the numerical
reference and default.

On crop `(3450,1750,700,500)`, PackedXTransNet phase RMS is
`[0.000540, 0.000464, 0.000746]`, versus Markesteijn
`[0.000554, 0.000446, 0.000782]`. Its common luminance delta is `1.82e-5` and
RGB-delta range `0.000281`, both well inside `0.005`. The metallic earring has
no repeating CFA texture or seam and is at least competitive with X-veon while
avoiding Markesteijn's colour segmentation. The separate tracked asset
manifest preserves its full-third and 500-percent earring images without
changing the earlier comparison manifest.

The small analytical suite is mixed and must not be hidden by the good real
crop: PackedXTransNet is excellent on constant/black fields but has substantial
regressions on the artificial saturated-edge and impulse challenges. The ten
unavailable upstream ground-truth images and the complete public generation
I-V RAF matrix were not fabricated or silently replaced; those remain open
coverage. Phase 12 therefore establishes a successful hidden technical and
`DSCF0771` experiment, not a general superiority claim.

CTest covers mock geometry, both margins, all 18 CFA representations, signed
outputs, non-finite/error propagation, artifact authentication, cache
concurrency, reviewed ONNX inference, and optional CPU/MIGraphX FP32 parity.
Python tests cover schema, fixed masks/kernels, safe checkpoint intake,
deterministic conversion, ONNX/PyTorch parity, and comparison metrics. The
comparison assets and detailed measurements are recorded in
`xtrans-neural-phase12-report.md`.

## Updating or upgrading the upstream checkpoint

Never edit a .pth file in place and never treat an upstream filename replacement
as the same model. An update follows this review workflow:

1. record the new repository revision, URL, size, SHA-256, and license state;
2. inspect it with weights-only loading in an isolated environment;
3. compare all keys, dtypes, dimensions, parameter counts, and tensor digests;
4. classify it as:
   * same architecture, new weights;
   * compatible format/schema extension; or
   * new architecture;
5. assign a new model revision for new weights;
6. bump the RTNN format only if container representation changes;
7. add a new architecture ID/schema if graph semantics change;
8. regenerate deterministic RTNN, manifest, and golden cases;
9. rerun Python parity, C++ loader, corruption, inference, and image-quality
   tests; and
10. keep the old artifact available until PP3/profile and fallback behaviour
    for the new model are decided.

A checkpoint with the same tensor shapes but different values is still a new
model revision and needs image-quality regression testing. A checkpoint with
new keys or shapes must never be accepted by weakening schema validation.

## Later conversion of Deep Demosaick

The Deep Demosaick checkpoint requires a separate architecture schema and
converter mode. In addition to ordinary convolution tensors it includes CUDA
storage tags, PReLU parameters, weight-normalization state, learned iteration
schedules, and MMNet data-consistency parameters.

Before supporting it, decide whether to:

* retain weight-normalization factors and reproduce normalization in C++; or
* fold weight normalization into ordinary convolution weights offline.

Folding is preferable for inference if Python parity proves the result. Record
both original and folded tensor digests in the manifest. The twenty-iteration
graph and projection operations also require their own golden cases and
architecture ID. Do not force this model into the Gharbi schema merely because
both source files use .pth.

## Licensing gate

Before committing a converted model under rtdata/models:

1. preserve the upstream MIT license and copyright notice;
2. ask the authors to confirm that pretrained checkpoint redistribution is
   intended under that license;
3. document training-dataset provenance and any redistribution implications;
4. ensure the source and generated artifact are covered in release packaging;
5. add the artifact and license to source-tarball/install manifests; and
6. confirm whether downstream distributions may rebuild the artifact without
   downloading non-free or mutable inputs.

If that gate is unresolved, commit the converter, schema, tests, and checksums
without committing the weights. Provide an explicit developer installation
step and a clear runtime missing-model fallback.

## Proposed implementation sequence

Keep reviewable changes separated:

1. **Conversion specification:** this plan plus a frozen RTNN v1 byte-layout
   document and Gharbi tensor schema.
2. **Python converter:** hash pinning, weights-only loading, schema validation,
   deterministic RTNN and manifest generation.
3. **Python parity and golden corpus:** round-trip, reproducibility, exact
   network outputs, and intermediate activation digests.
4. **CTest and C++ container loader:** BUILD_TESTING integration, bounds-checked
   RTNN parsing, registered loader tests, and corruption tests.
5. **Native inference:** fixed graph and golden parity without raw-pipeline
   integration.
6. **Developer demosaic integration:** CFA mapping, raw wrappers, tiling, and
   the real-image quality gate without GUI exposure.
7. **Packaging and GUI:** only after the developer method passes the quality
   gate.
8. **Separate X-veon acceleration:** retain the reviewed CPU backend, add the
   optional direct MIGraphX runner, prove CPU/GPU parity, and repeat the hidden
   method's quality and runtime gates before considering any wider exposure.

## Acceptance criteria

The conversion stage is complete when:

* the pinned upstream checkpoint converts without importing upstream executable
  model code;
* two conversions produce byte-identical RTNN output;
* all 409,923 normalized float32 parameters round-trip bit-for-bit;
* the generated manifest identifies source and output with SHA-256;
* a PyTorch model reconstructed from RTNN matches original-checkpoint output on
  all golden inputs;
* the C++ loader accepts the valid file and rejects every malformed fixture;
* CMake with BUILD_TESTING enabled registers the loader tests with CTest and
  ctest --test-dir build/dev --output-on-failure passes;
* CMake with BUILD_TESTING disabled builds RawTherapee without test targets;
* loader code is clean under normal, ASan, and UBSan builds;
* no Python or neural runtime dependency is added to ordinary RawTherapee
  builds; optional X-veon development backends remain default-off and
  independently gated;
* licensing determines whether the artifact is bundled or externally supplied;
  and
* the immutable C++ model object is ready for the tiled convolution engine.
