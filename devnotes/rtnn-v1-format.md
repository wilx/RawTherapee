# RawTherapee Neural Network container (RTNN) version 1

## Purpose and scope

RTNN is a deterministic, non-executable container for neural-network tensor
values used by RawTherapee. Version 1 transports the reviewed Gharbi
DemosaicNet X-Trans weights. It is not a general graph format and contains no
code, preprocessing rules, CFA metadata, text fields, compression, encryption,
or external references.

The authoritative semantic tensor mapping is
`tools/neural_demosaic/schemas/demosaicnet-xtrans-v1.json`. RTNN stores only
its numeric architecture and tensor IDs. Runtime code must not parse either
that JSON or the companion conversion manifest.

## Primitive representation

All integers and IEEE-754 float32 values are little-endian. Offsets and sizes
are unsigned and must be validated with checked arithmetic before use. No C or
C++ structure may be read directly from or written directly to the file.

Version 1 defines these values:

| Name | Value |
| --- | ---: |
| Architecture `DEMOSAICNET_XTRANS_V1` | 1 |
| Gharbi X-Trans model revision | 1 |
| Scalar type `FLOAT32` | 1 |
| Layout `VECTOR` | 1 |
| Layout `OIHW` | 2 |
| Alignment | 64 bytes |
| Flags | 0 |

Zero is invalid for architecture, model-revision, scalar-type, layout, and
semantic tensor IDs. All reserved bytes and fields must be zero. A version 1
reader must reject unknown major or minor versions, flags, enum values, tensor
IDs, noncanonical record ordering, or nonzero reserved fields.

## File organization

The file consists of a fixed header, 26 fixed-size tensor-directory records,
and one padded payload region:

```text
192-byte header
26 * 96-byte directory records
1,639,744-byte payload region
```

The directory begins immediately after the header. The payload begins
immediately after the directory at file offset 2,688, which is 64-byte aligned.
The complete reviewed file is 1,642,432 bytes.

### Header

The header is exactly 192 bytes:

| Offset | Size | Type | Field | Required v1 value |
| ---: | ---: | --- | --- | --- |
| 0 | 8 | bytes | Magic | `52 54 4e 4e 0d 0a 1a 0a` |
| 8 | 2 | uint16 | Format major | 1 |
| 10 | 2 | uint16 | Format minor | 0 |
| 12 | 4 | uint32 | Header size | 192 |
| 16 | 4 | uint32 | Endian marker | `0x01020304` |
| 20 | 4 | uint32 | Flags | 0 |
| 24 | 4 | uint32 | Architecture ID | 1 |
| 28 | 4 | uint32 | Model revision | 1 |
| 32 | 4 | uint32 | Scalar type | 1 |
| 36 | 4 | uint32 | Tensor count | 26 |
| 40 | 4 | uint32 | Directory-record size | 96 |
| 44 | 4 | uint32 | Reserved | 0 |
| 48 | 8 | uint64 | Directory offset | 192 |
| 56 | 8 | uint64 | Directory size | 2,496 |
| 64 | 8 | uint64 | Payload-region offset | 2,688 |
| 72 | 8 | uint64 | Padded payload-region size | 1,639,744 |
| 80 | 8 | uint64 | Sum of tensor byte lengths | 1,639,692 |
| 88 | 8 | uint64 | Total file size | 1,642,432 |
| 96 | 32 | bytes | Semantic-schema SHA-256 | Raw digest bytes |
| 128 | 32 | bytes | Padded payload-region SHA-256 | Raw digest bytes |
| 160 | 32 | bytes | Source-checkpoint SHA-256 | Raw digest bytes |

The schema digest is
`0ec34ea3d563f1357097181cb0dd90586a65e8d4c85b644b45c6fd3f81bcc151`.
The checkpoint digest is
`3759a13296fecebae83a36a8f0c3890d8a2f7d33e5f8149ed9b76a70f8285bc7`.

### Tensor directory

Directory records are exactly 96 bytes and occur in ascending semantic-ID
order from 1 through 26:

| Offset | Size | Type | Field |
| ---: | ---: | --- | --- |
| 0 | 4 | uint32 | Semantic tensor ID |
| 4 | 2 | uint16 | Rank; 1 or 4 |
| 6 | 2 | uint16 | Layout ID |
| 8 | 4 | uint32 | Scalar type; 1 |
| 12 | 4 | uint32 | Flags; 0 |
| 16 | 16 | 4 * uint32 | Dimensions; unused entries are zero |
| 32 | 8 | uint64 | Element count |
| 40 | 8 | uint64 | Tensor byte length |
| 48 | 8 | uint64 | Offset relative to payload-region start |
| 56 | 32 | bytes | Tensor-byte SHA-256; raw digest bytes |
| 88 | 8 | uint64 | Reserved; 0 |

Rank-one tensors use layout `VECTOR`, dimension `[N, 0, 0, 0]`, and exactly
`N * 4` bytes. Rank-four tensors use layout `OIHW`, dimensions
`[output channels, input channels, height, width]`, and the product of those
dimensions times four bytes. The recorded element count and byte length must be
checked independently from the dimensions.

### Payload region

Each tensor begins at a relative offset divisible by 64. Tensors occur in the
same ascending semantic-ID order as their records. Their values are contiguous
little-endian float32 data in `VECTOR` or PyTorch-compatible `OIHW` order.

All gaps and final alignment bytes are zero. The final payload size is rounded
up to 64 bytes. A tensor digest covers only that tensor's bytes. The header
payload digest covers the entire padded payload region, including all zero
padding. Overlapping, reordered, misaligned, out-of-bounds, non-finite, or
otherwise schema-incompatible tensors are invalid.

## Reviewed Gharbi artifact

Conversion of the pinned checkpoint produces these deterministic identities:

| Artifact | Size | SHA-256 |
| --- | ---: | --- |
| RTNN file | 1,642,432 | `b4dd6ea4ba535e7f4aea249a2d589a80ca8584f60a605a5bce468c989529ccc2` |
| Padded payload region | 1,639,744 | `e0e501a3f3a4905e3c7bb1ab1f0e3acb5598da818d6406d5cf6ed030ab5af606` |
| Canonical companion manifest | 10,919 | `f9b5d784356a455327304cfbfe302b2041a5a1a1eb2970134e3c1dfb621ec447` |

The companion JSON format is
`rawtherapee-rtnn-conversion-manifest-v1`. It records stable provenance,
container sizes and hashes, and the complete semantic-to-payload mapping. It
contains no timestamp, local path, output filename, host information, or tool
version. It is development metadata and is not needed by a runtime reader.

## Compatibility

An incompatible representation requires a new format major version. Version 1
readers accept only minor version zero until an extension policy is explicitly
defined. Existing numeric IDs are never renumbered or repurposed.

New reviewed weights with the identical graph retain architecture ID 1 and the
same tensor IDs, but increment the model revision and use new binding, tensor,
payload, file, and manifest digests. A graph, tensor-role, layout, or shape
change requires a new architecture ID.
