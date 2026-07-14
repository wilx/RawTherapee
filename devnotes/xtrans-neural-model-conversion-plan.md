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

## Phase 1: create the isolated converter

### Proposed repository layout

Use a narrowly scoped directory rather than adding model conversion to an
existing benchmark script:

    tools/neural_demosaic/
        README.md
        convert_checkpoint.py
        inspect_model.py
        requirements.txt
        schemas/
            demosaicnet_xtrans_v1.py
        tests/
            test_conversion.py
            test_invalid_checkpoint.py

Proposed generated output, subject to the redistribution decision:

    rtdata/models/
        demosaicnet-xtrans-v1.rtnn
        demosaicnet-xtrans-v1.manifest.json
        demosaicnet-xtrans-LICENSE.txt

If the model is not distributable, keep generated output under an ignored build
directory and let developers install it explicitly. The C++ loader should use
the same model filename and search contract in either case.

### Python environment

The converter is a developer tool, not a runtime dependency. Pin a supported
Python and PyTorch version in requirements.txt and record both versions in the
manifest. Prefer a CPU-only PyTorch package.

Load the state dictionary using the safest available API:

    torch.load(path, map_location="cpu", weights_only=True)

Do not import or execute the upstream model module merely to deserialize the
checkpoint. The artifact is an OrderedDict of tensors and should not require
upstream Python classes.

The converter must:

1. read the source as bytes and compute SHA-256 before deserialization;
2. verify the expected source hash;
3. use weights-only loading and force all storage to CPU;
4. require an OrderedDict or mapping containing exactly the expected keys;
5. reject sparse, quantized, complex, non-floating, or non-finite tensors;
6. detach, convert to float32, and make each tensor contiguous;
7. retain canonical PyTorch OIHW convolution order;
8. serialize fields explicitly in little-endian order; and
9. write to a temporary file, validate it, then atomically rename it.

The converter should never overwrite an existing artifact with different
content unless an explicit output path or replace option is given.

## Phase 2: define and enforce the Gharbi tensor schema

Version 1 must validate model semantics, not just total byte size. The expected
state-dictionary keys are:

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

## Phase 3: RawTherapee neural-model file format version 1

Use a small container format, provisionally named RTNN. Its purpose is safe
transport of validated tensors, not graph execution.

### Design rules

* All integers and floats are explicitly little-endian.
* No C or C++ structure is written with a direct memory dump.
* Every offset and length is an unsigned fixed-width integer.
* All offset-plus-length operations use checked arithmetic.
* Tensor payloads are float32 in canonical OIHW order.
* Tensor payload starts are aligned to at least 64 bytes.
* Runtime code reads no JSON and executes no embedded instructions.
* Unknown format versions, architectures, scalar types, flags, or tensor IDs
  are rejected.
* Text fields have fixed maximum sizes and are UTF-8 only.

### Proposed header

The exact byte layout must be frozen in a format document before implementation.
The logical fields are:

| Field | Purpose |
| --- | --- |
| Magic | Eight-byte RTNN file signature |
| Format major/minor | Container compatibility |
| Header size | Allows safe extension within a major version |
| Endian marker | Detects byte-order mistakes |
| Architecture ID | DemosaicNet X-Trans v1 |
| Scalar type | IEEE-754 float32 |
| Tensor count | Must be 26 for this architecture |
| Directory offset/size | Bounds for tensor records |
| Payload offset/size | Bounds for tensor values |
| Payload digest | Detects corruption |
| Model UUID/revision | Identifies the converted logical model |
| Flags | Must contain only known bits |

Use a cryptographic digest for build/provenance verification. If adding SHA-256
to the runtime would introduce an unwanted dependency, use a small reviewed
implementation or a strong non-cryptographic corruption checksum at runtime
and keep SHA-256 in the generated manifest and build verification. Length,
shape, overlap, and bounds validation remain mandatory regardless of checksum.

### Tensor directory record

Each record should contain:

| Field | Purpose |
| --- | --- |
| Semantic tensor ID | Stable enum, not an arbitrary runtime name |
| Rank | One or four in version 1 |
| Dimensions | Up to four uint32 dimensions |
| Layout | Vector or OIHW |
| Scalar type | Must agree with the header |
| Payload offset | Relative to the payload region |
| Element count | Independently checked against dimensions |
| Byte length | Must equal element count times four |
| Optional tensor digest | Useful for conversion diagnostics |

Names may be present in the developer manifest, but the C++ loader should bind
weights by semantic enum. This avoids string lookup and ambiguous renamed
PyTorch keys while still enforcing one exact architecture schema.

### Manifest

Generate a human-readable JSON manifest next to the binary. Runtime code does
not need to parse it. It should contain:

* upstream project URL and revision;
* source path, byte size, and SHA-256;
* upstream license identifier and notice path;
* converter repository revision;
* Python, PyTorch, NumPy, and converter versions;
* conversion UTC timestamp, excluded from reproducibility comparisons;
* architecture name and schema version;
* output filename, size, and SHA-256;
* every source key, semantic ID, shape, dtype, element count, and tensor
  SHA-256; and
* golden-reference identifiers produced from this model.

To preserve reproducible binary output, timestamps and machine-specific paths
belong only in the manifest, never in the RTNN binary.

## Phase 4: prove that conversion is lossless and deterministic

The Python tests should perform four levels of verification.

### Tensor equivalence

After writing RTNN, read it back using a separate minimal Python reader and
compare every element with the normalized source tensor. Since source tensors
are already float32, equality should be bit-for-bit. Also compare tensor-level
SHA-256 values.

### Deterministic output

Convert the same checkpoint twice into different temporary directories.
The RTNN files must be byte-for-byte identical and have identical SHA-256
digests. Manifests may differ only in explicitly non-reproducible fields such as
timestamp and local source path; provide a canonical manifest mode that omits
those fields for tests and release generation.

### Independent network parity

Implement a small reference reader that reconstructs the PyTorch module from
RTNN tensors, without consulting the original .pth file. Compare its output
against the module loaded from the original state dictionary on fixed inputs:

* all zero;
* constant sampled values;
* deterministic pseudorandom sparse RGB;
* impulse at each CFA colour;
* saturated samples; and
* odd and even input sizes larger than the 24-pixel valid border.

The outputs should be bit-identical where PyTorch uses the same operations, or
within a documented tiny floating-point tolerance if backend differences
prevent exact equality.

### Negative and corruption tests

Test rejection of:

* incorrect source SHA-256;
* a non-state-dictionary pickle;
* missing and additional keys;
* wrong tensor shapes and dtypes;
* NaN and infinity;
* truncated header, directory, and payload;
* offset overflow and overlapping tensor ranges;
* duplicate or unknown semantic tensor IDs;
* unsupported version, architecture, layout, and scalar type;
* payload bit corruption; and
* unreasonable tensor count or allocation sizes.

Do not make test fixtures by executing untrusted pickle code. Construct invalid
state dictionaries locally and serialize only ordinary tensors.

## Phase 5: C++ loader

Add a loader independent of the demosaicing pipeline, for example:

    rtengine/neuralmodel.h
    rtengine/neuralmodel.cc
    rtengine/demosaicnetxtransmodel.h
    rtengine/demosaicnetxtransmodel.cc

Final names should follow existing rtengine conventions. Keep the generic
container parser small; place the exact 26-tensor schema and architecture
binding in the DemosaicNet-specific layer.

The loader must:

1. open the file read-only and determine its actual size;
2. parse every scalar explicitly rather than casting a mapped byte buffer;
3. apply configured maximum file, directory, tensor, and dimension sizes;
4. validate header, ranges, alignment, non-overlap, counts, and checksums;
5. validate the exact DemosaicNet X-Trans schema;
6. copy float32 values into owned aligned storage;
7. byte-swap on a big-endian host if RawTherapee still supports one;
8. verify all loaded floats are finite;
9. repack OIHW tensors into the selected internal SIMD layout; and
10. return a complete immutable model or a structured error, never a partially
    initialized model.

The portable RTNN file should remain OIHW. Architecture- or SIMD-specific
packing belongs in memory at load time. This keeps one distributed file valid
for scalar, SSE, AVX, ARM/NEON, and future kernels, and makes conversion parity
easy to audit. Loading and repacking 1.64 MB once is negligible relative to
full-image inference.

Load the model once per process or engine context and share immutable weights.
Do not reopen or revalidate the file per tile. Report the first model-loading
failure clearly, but avoid flooding logs during preview recomputation.

## Phase 6: C++ loader tests

Introduce CTest as part of this work. The top-level CMake configuration should
include CTest, use its standard BUILD_TESTING option, and add the model-test
subdirectory only when testing is enabled. Normal RawTherapee binaries must not
depend on the test executable or test-only libraries.

Register the model loader and corruption tests with add_test so they run through
the normal command:

    ctest --test-dir build/dev --output-on-failure

Keep the initial harness small and native to the repository. It should exercise
the production loader directly, rather than duplicating parsing code in a
standalone developer utility. Test names should be stable and grouped with a
neural-model or rtengine label so they can be selected independently.

Required checks:

* every tensor has the expected semantic ID, shape, and element count;
* selected first, middle, and last float bit patterns match the Python manifest;
* total parameter count is 409,923;
* C++ reports the expected artifact digest;
* every negative/corrupt RTNN fixture fails with the expected error class;
* no malformed input requests an unbounded allocation;
* loading is clean under ASan and UBSan; and
* repeated loads produce identical in-memory canonical values.

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
