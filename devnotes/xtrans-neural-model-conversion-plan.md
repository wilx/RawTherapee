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
* Do not embed Python, libtorch, ONNX Runtime, or another inference framework.
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

The converter project should also produce small, deterministic golden cases.
For each input, preserve:

* the sparse three-plane input as little-endian float32;
* input dimensions;
* the valid output dimensions (input minus 24 in each axis);
* output RGB float32;
* input and output SHA-256;
* whether input is direct-linear or gamma-wrapped; and
* PyTorch/backend versions.

At least one case should expose every convolution border and CFA phase. Keep
fixtures small enough to commit if their provenance permits. Large RAF-derived
inputs and outputs should remain external benchmark data with recorded hashes.

Golden cases prove numerical implementation parity. They do not prove image
quality and should not be used to select the linear/gamma input contract.

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
3. **Python tests and golden cases:** round-trip, parity, reproducibility, and
   malformed input.
4. **CTest and C++ container loader:** minimal BUILD_TESTING integration,
   bounds-checked RTNN parser, registered loader tests, and corruption tests.
5. **Gharbi model binding:** exact 26-tensor schema, aligned storage, and
   internal weight repacking.
6. **Packaging decision:** either install the licensed converted model or
   document external developer provisioning.
7. **Inference implementation:** only after all preceding conversion and loader
   gates pass.

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
* no Python or neural runtime dependency is added to RawTherapee;
* licensing determines whether the artifact is bundled or externally supplied;
  and
* the immutable C++ model object is ready for the tiled convolution engine.
