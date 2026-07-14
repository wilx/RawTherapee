# Neural demosaicer development tools

This directory contains development-only tooling for authenticating and
converting published neural demosaicing checkpoints. It is not part of the
RawTherapee runtime.

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

## Tests

Unit tests generate ordinary local tensor dictionaries and never execute
untrusted pickle content:

```sh
.venv/bin/python -m pytest tools/neural_demosaic/tests
```

Run the optional integration tests against the actual pinned checkpoint:

```sh
GHARBI_XTRANS_CHECKPOINT=/path/to/demosaicnet/data/xtrans.pth \
    .venv/bin/python -m pytest tools/neural_demosaic/tests
```

The integration tests are skipped when `GHARBI_XTRANS_CHECKPOINT` is unset.
