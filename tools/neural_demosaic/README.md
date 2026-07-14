# Neural demosaicer development tools

This directory contains development-only tooling for inspecting published
neural demosaicing checkpoints. It is not part of the RawTherapee runtime.

Phase 1 supports only the Gharbi DemosaicNet X-Trans checkpoint from upstream
revision `959e9d1630976b421d5af5e35b2e2a01f5630e5c`. The tool reads the complete
checkpoint, verifies its pinned size and SHA-256 before deserialization, loads
only tensor weights on the CPU, validates the exact tensor schema, and emits a
deterministic JSON inspection manifest.

It does not download the checkpoint, run neural inference, or create an RTNN
file. Keep the `.pth` file and generated manifests outside tracked source.

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

The JSON is development-time input for the future converter. Phase 3 will
serialize only numeric semantic IDs into RTNN. RawTherapee C++ code will mirror
the stable enum values and will not parse this JSON at runtime.

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
