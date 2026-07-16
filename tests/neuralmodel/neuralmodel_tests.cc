#include "rtnn_inspection.h"
#include "demosaicnet_inference_tests.h"
#include "xtrans_demosaicnet_tests.h"
#include "xtrans_xveon_tests.h"

#include "rtengine/demosaicnetxtransmodel.h"
#include "rtengine/neuralmodel.h"
#include "rtengine/rtnnreader_p.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <glib.h>
#include <glib/gstdio.h>
#include <glibmm/checksum.h>

namespace
{

using rtengine::neural::NeuralModelErrorCode;
using rtengine::neural::NeuralTensorView;
using rtengine::neural::Sha256Digest;
using rtengine::neural::TensorLayout;
using rtengine::neural::detail::InternalLoadResult;
using rtengine::neural::detail::ModelBinding;
using rtengine::neural::detail::TensorBinding;

constexpr std::size_t HEADER_SIZE = 192;
constexpr std::size_t RECORD_SIZE = 96;
constexpr std::size_t TENSOR_COUNT = 26;
constexpr std::size_t DIRECTORY_SIZE = RECORD_SIZE * TENSOR_COUNT;
constexpr std::size_t PAYLOAD_OFFSET = HEADER_SIZE + DIRECTORY_SIZE;
constexpr std::size_t ALIGNMENT = 64;
constexpr std::uint64_t PARAMETER_COUNT = 409923;
constexpr std::uint64_t TENSOR_BYTES = 1639692;
constexpr std::uint64_t PAYLOAD_BYTES = 1639744;
constexpr std::uint64_t FILE_BYTES = 1642432;
constexpr std::uint64_t MAX_FILE_BYTES = 64 * 1024 * 1024;

struct TensorSpec final {
    std::uint16_t rank;
    TensorLayout layout;
    std::array<std::uint32_t, 4> dimensions;
    std::uint64_t count;
};

const std::array<TensorSpec, TENSOR_COUNT> SPECS{{
    {4, TensorLayout::OIHW, {{64, 3, 3, 3}}, 1728},
    {1, TensorLayout::VECTOR, {{64, 0, 0, 0}}, 64},
    {4, TensorLayout::OIHW, {{64, 64, 3, 3}}, 36864},
    {1, TensorLayout::VECTOR, {{64, 0, 0, 0}}, 64},
    {4, TensorLayout::OIHW, {{64, 64, 3, 3}}, 36864},
    {1, TensorLayout::VECTOR, {{64, 0, 0, 0}}, 64},
    {4, TensorLayout::OIHW, {{64, 64, 3, 3}}, 36864},
    {1, TensorLayout::VECTOR, {{64, 0, 0, 0}}, 64},
    {4, TensorLayout::OIHW, {{64, 64, 3, 3}}, 36864},
    {1, TensorLayout::VECTOR, {{64, 0, 0, 0}}, 64},
    {4, TensorLayout::OIHW, {{64, 64, 3, 3}}, 36864},
    {1, TensorLayout::VECTOR, {{64, 0, 0, 0}}, 64},
    {4, TensorLayout::OIHW, {{64, 64, 3, 3}}, 36864},
    {1, TensorLayout::VECTOR, {{64, 0, 0, 0}}, 64},
    {4, TensorLayout::OIHW, {{64, 64, 3, 3}}, 36864},
    {1, TensorLayout::VECTOR, {{64, 0, 0, 0}}, 64},
    {4, TensorLayout::OIHW, {{64, 64, 3, 3}}, 36864},
    {1, TensorLayout::VECTOR, {{64, 0, 0, 0}}, 64},
    {4, TensorLayout::OIHW, {{64, 64, 3, 3}}, 36864},
    {1, TensorLayout::VECTOR, {{64, 0, 0, 0}}, 64},
    {4, TensorLayout::OIHW, {{64, 64, 3, 3}}, 36864},
    {1, TensorLayout::VECTOR, {{64, 0, 0, 0}}, 64},
    {4, TensorLayout::OIHW, {{64, 67, 3, 3}}, 38592},
    {1, TensorLayout::VECTOR, {{64, 0, 0, 0}}, 64},
    {4, TensorLayout::OIHW, {{3, 64, 1, 1}}, 192},
    {1, TensorLayout::VECTOR, {{3, 0, 0, 0}}, 3},
}};

void require(bool condition, const std::string &message)
{
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::uint32_t readU32(const std::vector<std::uint8_t> &data, std::size_t offset)
{
    return static_cast<std::uint32_t>(data[offset]) |
           static_cast<std::uint32_t>(data[offset + 1]) << 8 |
           static_cast<std::uint32_t>(data[offset + 2]) << 16 |
           static_cast<std::uint32_t>(data[offset + 3]) << 24;
}

std::uint64_t readU64(const std::vector<std::uint8_t> &data, std::size_t offset)
{
    return static_cast<std::uint64_t>(readU32(data, offset)) |
           static_cast<std::uint64_t>(readU32(data, offset + 4)) << 32;
}

void writeU16(std::vector<std::uint8_t> &data, std::size_t offset, std::uint16_t value)
{
    data[offset] = static_cast<std::uint8_t>(value);
    data[offset + 1] = static_cast<std::uint8_t>(value >> 8);
}

void writeU32(std::vector<std::uint8_t> &data, std::size_t offset, std::uint32_t value)
{
    for (unsigned byte = 0; byte < 4; ++byte) {
        data[offset + byte] = static_cast<std::uint8_t>(value >> (byte * 8));
    }
}

void writeU64(std::vector<std::uint8_t> &data, std::size_t offset, std::uint64_t value)
{
    for (unsigned byte = 0; byte < 8; ++byte) {
        data[offset + byte] = static_cast<std::uint8_t>(value >> (byte * 8));
    }
}

Sha256Digest sha256(const std::uint8_t *data, std::size_t size)
{
    Glib::Checksum checksum(Glib::Checksum::CHECKSUM_SHA256);
    require(static_cast<bool>(checksum), "SHA-256 is unavailable to native tests");
    checksum.update(reinterpret_cast<const guchar *>(data), size);
    Sha256Digest digest{{}};
    gsize digestSize = digest.size();
    checksum.get_digest(digest.data(), &digestSize);
    require(digestSize == digest.size(), "unexpected SHA-256 length");
    return digest;
}

std::string hexDigest(const Sha256Digest &digest)
{
    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for (const std::uint8_t byte : digest) {
        output << std::setw(2) << static_cast<unsigned>(byte);
    }
    return output.str();
}

Sha256Digest digestFromHex(const std::string &text)
{
    require(text.size() == 64, "test digest must contain 64 hex characters");
    Sha256Digest result{{}};
    for (std::size_t index = 0; index < result.size(); ++index) {
        result[index] = static_cast<std::uint8_t>(std::strtoul(text.substr(index * 2, 2).c_str(), nullptr, 16));
    }
    return result;
}

void writeDigest(std::vector<std::uint8_t> &data, std::size_t offset, const Sha256Digest &digest)
{
    std::copy(digest.begin(), digest.end(), data.begin() + offset);
}

std::size_t alignUp(std::size_t value)
{
    return (value + ALIGNMENT - 1) / ALIGNMENT * ALIGNMENT;
}

std::uint32_t syntheticFloatBits(std::uint32_t tensorId, std::uint64_t index)
{
    const std::uint32_t exponent = static_cast<std::uint32_t>((index + tensorId * 17) % 254 + 1);
    const std::uint32_t mantissa = static_cast<std::uint32_t>(
        (index * UINT64_C(0x9e3779b1) + tensorId * UINT64_C(0x12345)) & 0x7fffff);
    return (static_cast<std::uint32_t>(index & 1) << 31) | (exponent << 23) | mantissa;
}

struct SyntheticFixture final {
    std::vector<std::uint8_t> data;
    std::array<std::size_t, TENSOR_COUNT> offsets{{}};
    std::array<std::string, TENSOR_COUNT> tensorDigests;
    std::array<TensorBinding, TENSOR_COUNT> tensorBindings;
    std::string schemaDigest = std::string(64, '1');
    std::string checkpointDigest = std::string(64, '2');
    std::string artifactDigest;
    ModelBinding binding{};

    SyntheticFixture()
    {
        data.assign(FILE_BYTES, 0);
        const std::uint8_t magic[8] = {'R', 'T', 'N', 'N', '\r', '\n', 0x1a, '\n'};
        std::copy(magic, magic + 8, data.begin());
        writeU16(data, 8, 1);
        writeU16(data, 10, 0);
        writeU32(data, 12, HEADER_SIZE);
        writeU32(data, 16, 0x01020304);
        writeU32(data, 20, 0);
        writeU32(data, 24, 1);
        writeU32(data, 28, 1);
        writeU32(data, 32, 1);
        writeU32(data, 36, TENSOR_COUNT);
        writeU32(data, 40, RECORD_SIZE);
        writeU32(data, 44, 0);
        writeU64(data, 48, HEADER_SIZE);
        writeU64(data, 56, DIRECTORY_SIZE);
        writeU64(data, 64, PAYLOAD_OFFSET);
        writeU64(data, 72, PAYLOAD_BYTES);
        writeU64(data, 80, TENSOR_BYTES);
        writeU64(data, 88, FILE_BYTES);
        writeDigest(data, 96, digestFromHex(schemaDigest));
        writeDigest(data, 160, digestFromHex(checkpointDigest));

        std::size_t payloadCursor = 0;
        for (std::size_t index = 0; index < TENSOR_COUNT; ++index) {
            payloadCursor = alignUp(payloadCursor);
            offsets[index] = payloadCursor;
            const TensorSpec &spec = SPECS[index];
            const std::size_t record = HEADER_SIZE + index * RECORD_SIZE;
            writeU32(data, record, static_cast<std::uint32_t>(index + 1));
            writeU16(data, record + 4, spec.rank);
            writeU16(data, record + 6, static_cast<std::uint16_t>(spec.layout));
            writeU32(data, record + 8, 1);
            writeU32(data, record + 12, 0);
            for (std::size_t dimension = 0; dimension < 4; ++dimension) {
                writeU32(data, record + 16 + dimension * 4, spec.dimensions[dimension]);
            }
            writeU64(data, record + 32, spec.count);
            writeU64(data, record + 40, spec.count * 4);
            writeU64(data, record + 48, payloadCursor);
            writeU64(data, record + 88, 0);

            const std::size_t absolute = PAYLOAD_OFFSET + payloadCursor;
            for (std::uint64_t element = 0; element < spec.count; ++element) {
                writeU32(data, absolute + static_cast<std::size_t>(element * 4),
                    syntheticFloatBits(static_cast<std::uint32_t>(index + 1), element));
            }
            refreshTensor(index);
            payloadCursor += static_cast<std::size_t>(spec.count * 4);
        }
        require(alignUp(payloadCursor) == PAYLOAD_BYTES, "fixture payload size differs from frozen model");
        refreshPayload();
        refreshArtifact();
        synchronizeBinding();
    }

    void synchronizeBinding()
    {
        for (std::size_t index = 0; index < TENSOR_COUNT; ++index) {
            const TensorSpec &spec = SPECS[index];
            tensorBindings[index] = TensorBinding{
                static_cast<std::uint32_t>(index + 1), spec.rank, spec.layout,
                spec.dimensions, spec.count, tensorDigests[index].c_str()};
        }
        binding = ModelBinding{
            1, 1, schemaDigest.c_str(), checkpointDigest.c_str(), artifactDigest.c_str(),
            PARAMETER_COUNT, TENSOR_BYTES, tensorBindings.data(), tensorBindings.size()};
    }

    void refreshTensor(std::size_t index)
    {
        const std::size_t byteLength = static_cast<std::size_t>(SPECS[index].count * 4);
        const Sha256Digest digest = sha256(data.data() + PAYLOAD_OFFSET + offsets[index], byteLength);
        tensorDigests[index] = hexDigest(digest);
        writeDigest(data, HEADER_SIZE + index * RECORD_SIZE + 56, digest);
        synchronizeBinding();
    }

    void refreshPayload()
    {
        writeDigest(data, 128, sha256(data.data() + PAYLOAD_OFFSET, static_cast<std::size_t>(readU64(data, 72))));
    }

    void refreshArtifact()
    {
        artifactDigest = hexDigest(sha256(data.data(), data.size()));
        synchronizeBinding();
    }
};

void expectFailure(
    const std::string &name,
    NeuralModelErrorCode expected,
    const std::function<void(SyntheticFixture &)> &mutate)
{
    SyntheticFixture fixture;
    mutate(fixture);
    fixture.synchronizeBinding();
    const InternalLoadResult result = rtengine::neural::detail::parseRtnnBytes(fixture.data, fixture.binding);
    require(!result, name + " unexpectedly loaded");
    require(result.error.code == expected,
        name + " returned " + rtengine::neural::neuralModelErrorCodeName(result.error.code) +
            ", expected " + rtengine::neural::neuralModelErrorCodeName(expected));
    require(!result.model, name + " exposed partial model state");
}

void testErrorNames()
{
    const std::array<std::pair<NeuralModelErrorCode, const char *>, 19> names{{
        {NeuralModelErrorCode::NONE, "NONE"},
        {NeuralModelErrorCode::IO, "IO"},
        {NeuralModelErrorCode::SIZE, "SIZE"},
        {NeuralModelErrorCode::MAGIC, "MAGIC"},
        {NeuralModelErrorCode::VERSION, "VERSION"},
        {NeuralModelErrorCode::ENDIAN, "ENDIAN"},
        {NeuralModelErrorCode::FLAGS, "FLAGS"},
        {NeuralModelErrorCode::ENUM, "ENUM"},
        {NeuralModelErrorCode::RESERVED, "RESERVED"},
        {NeuralModelErrorCode::LIMIT, "LIMIT"},
        {NeuralModelErrorCode::RANGE, "RANGE"},
        {NeuralModelErrorCode::ALIGNMENT, "ALIGNMENT"},
        {NeuralModelErrorCode::ORDER, "ORDER"},
        {NeuralModelErrorCode::SCHEMA, "SCHEMA"},
        {NeuralModelErrorCode::DIGEST, "DIGEST"},
        {NeuralModelErrorCode::NONFINITE, "NONFINITE"},
        {NeuralModelErrorCode::ALLOCATION, "ALLOCATION"},
        {NeuralModelErrorCode::RUNTIME, "RUNTIME"},
        {static_cast<NeuralModelErrorCode>(999), "UNKNOWN"},
    }};
    for (const auto &entry : names) {
        require(std::strcmp(rtengine::neural::neuralModelErrorCodeName(entry.first), entry.second) == 0,
            std::string("unstable error name for ") + entry.second);
    }
}

int syntheticValid()
{
    testErrorNames();
    SyntheticFixture fixture;
    const InternalLoadResult first = rtengine::neural::detail::parseRtnnBytes(fixture.data, fixture.binding);
    const InternalLoadResult second = rtengine::neural::detail::parseRtnnBytes(fixture.data, fixture.binding);
    require(first && second, "valid independently generated RTNN did not load");
    require(!first.error && !second.error, "successful loads carry errors");
    require(first.model->tensors.size() == TENSOR_COUNT, "tensor count differs");
    require(first.model->parameterCount == PARAMETER_COUNT, "parameter count differs");
    require(first.model->tensorPayloadBytes == TENSOR_BYTES, "tensor byte count differs");
    require(first.model->payloadRegionBytes == PAYLOAD_BYTES, "payload size differs");

    std::size_t expectedOffset = 0;
    for (std::size_t index = 0; index < TENSOR_COUNT; ++index) {
        expectedOffset = alignUp(expectedOffset);
        const NeuralTensorView &left = first.model->tensors[index];
        const NeuralTensorView &right = second.model->tensors[index];
        require(left.id == index + 1 && left.rank == SPECS[index].rank, "tensor identity differs");
        require(left.layout == SPECS[index].layout && left.dimensions == SPECS[index].dimensions,
            "tensor shape or layout differs");
        require(left.elementCount == SPECS[index].count && left.payloadOffset == expectedOffset,
            "tensor count or canonical offset differs");
        // Every frozen tensor except the final three-float bias has a byte
        // length divisible by 64. Consequently this architecture has no
        // inter-tensor padding to corrupt; only its trailing padding exists.
        require(left.payloadOffset == (index == 0 ? 0 : first.model->tensors[index - 1].payloadOffset + first.model->tensors[index - 1].byteLength),
            "frozen model unexpectedly contains inter-tensor padding");
        require(reinterpret_cast<std::uintptr_t>(left.data) % ALIGNMENT == 0,
            "materialized tensor is not 64-byte aligned");
        for (std::uint64_t element = 0; element < left.elementCount; ++element) {
            std::uint32_t leftBits = 0;
            std::uint32_t rightBits = 0;
            std::memcpy(&leftBits, left.data + element, sizeof(leftBits));
            std::memcpy(&rightBits, right.data + element, sizeof(rightBits));
            require(leftBits == syntheticFloatBits(static_cast<std::uint32_t>(index + 1), element),
                "synthetic float bits changed during native loading");
            require(leftBits == rightBits, "repeated native loads differ");
        }
        expectedOffset += static_cast<std::size_t>(left.byteLength);
    }
    return 0;
}

int headerCorruption()
{
    const auto byte = [](std::size_t offset, std::uint8_t value) {
        return [=](SyntheticFixture &f) { f.data[offset] = value; };
    };
    expectFailure("magic", NeuralModelErrorCode::MAGIC, byte(0, 'X'));
    expectFailure("major version", NeuralModelErrorCode::VERSION, [](SyntheticFixture &f) { writeU16(f.data, 8, 2); });
    expectFailure("minor version", NeuralModelErrorCode::VERSION, [](SyntheticFixture &f) { writeU16(f.data, 10, 1); });
    expectFailure("header size", NeuralModelErrorCode::SIZE, [](SyntheticFixture &f) { writeU32(f.data, 12, 191); });
    expectFailure("endian", NeuralModelErrorCode::ENDIAN, [](SyntheticFixture &f) { writeU32(f.data, 16, 0); });
    expectFailure("header flags", NeuralModelErrorCode::FLAGS, [](SyntheticFixture &f) { writeU32(f.data, 20, 1); });
    expectFailure("zero architecture", NeuralModelErrorCode::ENUM, [](SyntheticFixture &f) { writeU32(f.data, 24, 0); });
    expectFailure("architecture identity", NeuralModelErrorCode::SCHEMA, [](SyntheticFixture &f) { writeU32(f.data, 24, 2); });
    expectFailure("zero revision", NeuralModelErrorCode::ENUM, [](SyntheticFixture &f) { writeU32(f.data, 28, 0); });
    expectFailure("revision identity", NeuralModelErrorCode::SCHEMA, [](SyntheticFixture &f) { writeU32(f.data, 28, 2); });
    expectFailure("header scalar", NeuralModelErrorCode::ENUM, [](SyntheticFixture &f) { writeU32(f.data, 32, 2); });
    expectFailure("tensor count", NeuralModelErrorCode::SCHEMA, [](SyntheticFixture &f) { writeU32(f.data, 36, 25); });
    expectFailure("record size", NeuralModelErrorCode::SIZE, [](SyntheticFixture &f) { writeU32(f.data, 40, 95); });
    expectFailure("header reserved", NeuralModelErrorCode::RESERVED, [](SyntheticFixture &f) { writeU32(f.data, 44, 1); });
    expectFailure("directory offset", NeuralModelErrorCode::RANGE, [](SyntheticFixture &f) { writeU64(f.data, 48, 193); });
    expectFailure("directory size", NeuralModelErrorCode::RANGE, [](SyntheticFixture &f) { writeU64(f.data, 56, DIRECTORY_SIZE + 1); });
    expectFailure("payload offset", NeuralModelErrorCode::RANGE, [](SyntheticFixture &f) { writeU64(f.data, 64, PAYLOAD_OFFSET + 64); });
    expectFailure("tensor-byte summary", NeuralModelErrorCode::SCHEMA, [](SyntheticFixture &f) { writeU64(f.data, 80, TENSOR_BYTES + 4); });
    expectFailure("declared size", NeuralModelErrorCode::SIZE, [](SyntheticFixture &f) { writeU64(f.data, 88, FILE_BYTES - 1); });
    expectFailure("schema digest", NeuralModelErrorCode::SCHEMA, byte(96, 0));
    expectFailure("checkpoint digest", NeuralModelErrorCode::SCHEMA, byte(160, 0));
    expectFailure("truncated header", NeuralModelErrorCode::SIZE, [](SyntheticFixture &f) { f.data.resize(191); });
    expectFailure("extended file", NeuralModelErrorCode::SIZE, [](SyntheticFixture &f) { f.data.push_back(0); });
    return 0;
}

int directoryCorruption()
{
    const std::size_t record = HEADER_SIZE;
    expectFailure("reordered ID", NeuralModelErrorCode::ORDER, [=](SyntheticFixture &f) { writeU32(f.data, record, 2); });
    expectFailure("duplicate ID", NeuralModelErrorCode::ORDER, [](SyntheticFixture &f) { writeU32(f.data, HEADER_SIZE + RECORD_SIZE, 1); });
    expectFailure("rank limit", NeuralModelErrorCode::LIMIT, [=](SyntheticFixture &f) { writeU16(f.data, record + 4, 5); });
    expectFailure("rank enum", NeuralModelErrorCode::ENUM, [=](SyntheticFixture &f) { writeU16(f.data, record + 4, 2); });
    expectFailure("layout enum", NeuralModelErrorCode::ENUM, [=](SyntheticFixture &f) { writeU16(f.data, record + 6, 3); });
    expectFailure("record scalar", NeuralModelErrorCode::ENUM, [=](SyntheticFixture &f) { writeU32(f.data, record + 8, 2); });
    expectFailure("record flags", NeuralModelErrorCode::FLAGS, [=](SyntheticFixture &f) { writeU32(f.data, record + 12, 1); });
    expectFailure("zero dimension", NeuralModelErrorCode::SCHEMA, [=](SyntheticFixture &f) { writeU32(f.data, record + 16, 0); });
    expectFailure("dimension limit", NeuralModelErrorCode::LIMIT, [=](SyntheticFixture &f) { writeU32(f.data, record + 16, 1048577); });
    expectFailure("unused dimension", NeuralModelErrorCode::RESERVED, [](SyntheticFixture &f) { writeU32(f.data, HEADER_SIZE + RECORD_SIZE + 20, 1); });
    expectFailure("element limit", NeuralModelErrorCode::LIMIT, [=](SyntheticFixture &f) { writeU64(f.data, record + 32, 16777217); });
    expectFailure("element count", NeuralModelErrorCode::SCHEMA, [=](SyntheticFixture &f) { writeU64(f.data, record + 32, 1727); });
    expectFailure("byte length", NeuralModelErrorCode::SCHEMA, [=](SyntheticFixture &f) { writeU64(f.data, record + 40, 1); });
    expectFailure("tensor offset alignment", NeuralModelErrorCode::ALIGNMENT, [=](SyntheticFixture &f) { writeU64(f.data, record + 48, 1); });
    expectFailure("record digest", NeuralModelErrorCode::DIGEST, [=](SyntheticFixture &f) { f.data[record + 56] ^= 1; });
    expectFailure("record reserved", NeuralModelErrorCode::RESERVED, [=](SyntheticFixture &f) { writeU64(f.data, record + 88, 1); });
    expectFailure("overlap", NeuralModelErrorCode::RANGE, [](SyntheticFixture &f) { writeU64(f.data, HEADER_SIZE + RECORD_SIZE + 48, 0); });
    expectFailure("nonminimal gap", NeuralModelErrorCode::ALIGNMENT, [](SyntheticFixture &f) { writeU64(f.data, HEADER_SIZE + RECORD_SIZE + 48, 6976); });
    expectFailure("shape product overflow", NeuralModelErrorCode::RANGE, [=](SyntheticFixture &f) {
        for (std::size_t dimension = 0; dimension < 4; ++dimension) {
            writeU32(f.data, record + 16 + dimension * 4, 1048576);
        }
    });
    expectFailure("tensor out of payload", NeuralModelErrorCode::RANGE, [](SyntheticFixture &f) {
        f.data.resize(FILE_BYTES - 64);
        writeU64(f.data, 72, PAYLOAD_BYTES - 64);
        writeU64(f.data, 88, FILE_BYTES - 64);
    });
    return 0;
}

int payloadCorruption()
{
    expectFailure("nonminimal trailing payload", NeuralModelErrorCode::ALIGNMENT, [](SyntheticFixture &f) {
        f.data.resize(FILE_BYTES + 64, 0);
        writeU64(f.data, 72, PAYLOAD_BYTES + 64);
        writeU64(f.data, 88, FILE_BYTES + 64);
    });
    expectFailure("stale payload digest", NeuralModelErrorCode::DIGEST, [](SyntheticFixture &f) {
        f.data[PAYLOAD_OFFSET] ^= 1;
    });
    expectFailure("stale tensor digest", NeuralModelErrorCode::DIGEST, [](SyntheticFixture &f) {
        f.data[PAYLOAD_OFFSET] ^= 1;
        f.refreshPayload();
    });
    expectFailure("nonzero trailing padding", NeuralModelErrorCode::RESERVED, [](SyntheticFixture &f) {
        f.data.back() = 1;
        f.refreshPayload();
        f.refreshArtifact();
    });
    expectFailure("complete file digest", NeuralModelErrorCode::DIGEST, [](SyntheticFixture &f) {
        writeU32(f.data, PAYLOAD_OFFSET, syntheticFloatBits(1, 0) ^ 1);
        f.refreshTensor(0);
        f.refreshPayload();
    });
    expectFailure("NaN", NeuralModelErrorCode::NONFINITE, [](SyntheticFixture &f) {
        writeU32(f.data, PAYLOAD_OFFSET, 0x7fc00000);
        f.refreshTensor(0);
        f.refreshPayload();
        f.refreshArtifact();
    });
    expectFailure("infinity", NeuralModelErrorCode::NONFINITE, [](SyntheticFixture &f) {
        writeU32(f.data, PAYLOAD_OFFSET, 0x7f800000);
        f.refreshTensor(0);
        f.refreshPayload();
        f.refreshArtifact();
    });
    return 0;
}

std::string temporaryPath(const char *suffix)
{
    std::ostringstream path;
    path << g_get_tmp_dir() << G_DIR_SEPARATOR_S << "rawtherapee-rtnn-"
         << g_get_monotonic_time() << '-' << suffix;
    return path.str();
}

void writeFile(const std::string &path, const std::uint8_t *data, std::size_t size)
{
    std::unique_ptr<std::FILE, int (*)(std::FILE *)> file(g_fopen(path.c_str(), "wb"), std::fclose);
    require(static_cast<bool>(file), "cannot create native RTNN test file");
    require(std::fwrite(data, 1, size, file.get()) == size, "cannot write native RTNN test file");
}

int limitsAndIo()
{
    expectFailure("tensor count limit", NeuralModelErrorCode::LIMIT, [](SyntheticFixture &f) { writeU32(f.data, 36, 257); });
    expectFailure("payload limit", NeuralModelErrorCode::LIMIT, [](SyntheticFixture &f) { writeU64(f.data, 72, MAX_FILE_BYTES + 1); });

    const std::string missing = temporaryPath("missing.rtnn");
    g_remove(missing.c_str());
    InternalLoadResult result = rtengine::neural::detail::loadRtnnFile(missing, ModelBinding{});
    require(!result && result.error.code == NeuralModelErrorCode::IO, "missing file did not return IO");

    SyntheticFixture fixture;
    const std::string shortPath = temporaryPath("short.rtnn");
    writeFile(shortPath, fixture.data.data(), 100);
    result = rtengine::neural::detail::loadRtnnFile(shortPath, fixture.binding);
    g_remove(shortPath.c_str());
    require(!result && result.error.code == NeuralModelErrorCode::SIZE, "truncated file did not return SIZE");

    const std::string syntheticPath = temporaryPath("unreviewed.rtnn");
    writeFile(syntheticPath, fixture.data.data(), fixture.data.size());
    const rtengine::neural::DemosaicNetXTransLoadResult unreviewed =
        rtengine::neural::loadDemosaicNetXTransModel(syntheticPath);
    g_remove(syntheticPath.c_str());
    require(!unreviewed && unreviewed.error.code == NeuralModelErrorCode::SCHEMA,
        "public API accepted a synthetic unreviewed artifact");
    require(!unreviewed.model, "failed public load exposed a partial model");

    const std::string largePath = temporaryPath("large.rtnn");
    {
        std::unique_ptr<std::FILE, int (*)(std::FILE *)> file(g_fopen(largePath.c_str(), "wb"), std::fclose);
        require(static_cast<bool>(file), "cannot create sparse oversized RTNN");
        require(std::fseek(file.get(), static_cast<long>(MAX_FILE_BYTES), SEEK_SET) == 0,
            "cannot seek sparse oversized RTNN");
        require(std::fputc(0, file.get()) != EOF, "cannot finish sparse oversized RTNN");
    }
    result = rtengine::neural::detail::loadRtnnFile(largePath, fixture.binding);
    g_remove(largePath.c_str());
    require(!result && result.error.code == NeuralModelErrorCode::LIMIT,
        "64 MiB plus one file did not return LIMIT");
    return 0;
}

std::vector<std::uint8_t> readFile(const std::string &path)
{
    std::unique_ptr<std::FILE, int (*)(std::FILE *)> file(g_fopen(path.c_str(), "rb"), std::fclose);
    require(static_cast<bool>(file), "cannot open reviewed RTNN for independent comparison");
    require(std::fseek(file.get(), 0, SEEK_END) == 0, "cannot size reviewed RTNN");
    const long size = std::ftell(file.get());
    require(size >= 0 && std::fseek(file.get(), 0, SEEK_SET) == 0, "cannot rewind reviewed RTNN");
    std::vector<std::uint8_t> data(static_cast<std::size_t>(size));
    require(std::fread(data.data(), 1, data.size(), file.get()) == data.size(), "cannot read reviewed RTNN");
    return data;
}

const char *reviewedPath()
{
    const char *path = std::getenv("GHARBI_XTRANS_RTNN");
    return path && *path ? path : nullptr;
}

int reviewedArtifact()
{
    const char *path = reviewedPath();
    if (!path) {
        std::cout << "GHARBI_XTRANS_RTNN is not set; skipping reviewed artifact test\n";
        return 77;
    }
    const auto first = rtengine::neural::loadDemosaicNetXTransModel(path);
    const auto second = rtengine::neural::loadDemosaicNetXTransModel(path);
    require(first && second, "reviewed artifact did not load twice");
    const auto &leftModel = *first.model;
    const auto &rightModel = *second.model;
    require(leftModel.formatMajor() == 1 && leftModel.formatMinor() == 0, "reviewed format differs");
    require(leftModel.architectureId() == 1 && leftModel.modelRevision() == 1, "reviewed binding differs");
    require(leftModel.fileSize() == FILE_BYTES && leftModel.payloadRegionBytes() == PAYLOAD_BYTES,
        "reviewed container sizes differ");
    require(leftModel.tensorPayloadBytes() == TENSOR_BYTES && leftModel.parameterCount() == PARAMETER_COUNT,
        "reviewed tensor summaries differ");
    require(hexDigest(leftModel.artifactSha256()) == "b4dd6ea4ba535e7f4aea249a2d589a80ca8584f60a605a5bce468c989529ccc2",
        "reviewed artifact digest differs");
    require(hexDigest(leftModel.semanticSchemaSha256()) == "0ec34ea3d563f1357097181cb0dd90586a65e8d4c85b644b45c6fd3f81bcc151",
        "reviewed schema digest differs");
    require(hexDigest(leftModel.checkpointSha256()) == "3759a13296fecebae83a36a8f0c3890d8a2f7d33e5f8149ed9b76a70f8285bc7",
            "reviewed checkpoint digest differs");
    require(hexDigest(leftModel.payloadSha256()) == "e0e501a3f3a4905e3c7bb1ab1f0e3acb5598da818d6406d5cf6ed030ab5af606",
            "reviewed payload digest differs");
    require(leftModel.tensors().size() == TENSOR_COUNT, "reviewed tensor count differs");

    const std::vector<std::uint8_t> wire = readFile(path);
    require(wire.size() == FILE_BYTES, "reviewed raw file size differs");
    for (std::size_t index = 0; index < TENSOR_COUNT; ++index) {
        const NeuralTensorView &left = leftModel.tensors()[index];
        const NeuralTensorView &right = rightModel.tensors()[index];
        require(left.id == index + 1 && left.dimensions == SPECS[index].dimensions && left.elementCount == SPECS[index].count,
                "reviewed tensor contract differs");
        require(leftModel.tensor(static_cast<rtengine::neural::DemosaicNetXTransTensorId>(index + 1)) == &left,
                "reviewed semantic tensor lookup differs");
        require(reinterpret_cast<std::uintptr_t>(left.data) % ALIGNMENT == 0, "reviewed tensor is not aligned");
        const Sha256Digest independentlyHashed = sha256(
            wire.data() + PAYLOAD_OFFSET + static_cast<std::size_t>(left.payloadOffset),
            static_cast<std::size_t>(left.byteLength));
        require(independentlyHashed == left.sha256, "reviewed tensor digest differs from raw payload");
        const std::array<std::uint64_t, 3> samples{{0, left.elementCount / 2, left.elementCount - 1}};
        for (const std::uint64_t sample : samples) {
            const std::uint32_t wireBits = readU32(wire, PAYLOAD_OFFSET + static_cast<std::size_t>(left.payloadOffset + sample * 4));
            std::uint32_t nativeBits = 0;
            std::memcpy(&nativeBits, left.data + sample, sizeof(nativeBits));
            require(nativeBits == wireBits, "reviewed tensor differs from authenticated little-endian payload");
        }
        require(left.elementCount == right.elementCount, "repeated reviewed tensor count differs");
        require(std::memcmp(left.data, right.data, static_cast<std::size_t>(left.byteLength)) == 0,
            "repeated reviewed loads differ bit-for-bit");
    }
    return 0;
}

int inspectionParity()
{
    const char *path = reviewedPath();
    if (!path) {
        std::cout << "GHARBI_XTRANS_RTNN is not set; skipping inspection parity test\n";
        return 77;
    }
    const auto loaded = rtengine::neural::loadDemosaicNetXTransModel(path);
    require(static_cast<bool>(loaded), "reviewed artifact did not load for inspection parity");
    const std::string json = rtnn_test::canonicalInspectionJson(*loaded.model);
    require(!json.empty() && json.back() == '\n', "canonical inspection JSON lacks final newline");
    require(json.find(path) == std::string::npos, "inspection JSON includes its input path");
    require(hexDigest(sha256(reinterpret_cast<const std::uint8_t *>(json.data()), json.size())) ==
                "026f992aa9fbc7277e16b1f13c4f55c56aeadf7457b5cd2090cc181847bb069b",
        "native inspection JSON differs from the pinned Python output");
    return 0;
}

} // namespace

int main(int argc, char **argv)
{
    if (argc != 2) {
        std::cerr << "usage: rawtherapee-neuralmodel-tests MODE\n";
        return 2;
    }
    try {
        const std::string mode = argv[1];
        if (mode == "synthetic-valid") {
            return syntheticValid();
        }
        if (mode == "header-corruption") {
            return headerCorruption();
        }
        if (mode == "directory-corruption") {
            return directoryCorruption();
        }
        if (mode == "payload-corruption") {
            return payloadCorruption();
        }
        if (mode == "limits-io") {
            return limitsAndIo();
        }
        if (mode == "reviewed-artifact") {
            return reviewedArtifact();
        }
        if (mode == "inspection-parity") {
            return inspectionParity();
        }
        if (mode == "inference-synthetic") {
            return demosaicnet_inference_test::syntheticGraph();
        }
        if (mode == "inference-errors") {
            return demosaicnet_inference_test::errorsAndWorkspace();
        }
        if (mode == "inference-golden") {
            return demosaicnet_inference_test::reviewedGolden();
        }
        if (mode == "inference-trace") {
            return demosaicnet_inference_test::reviewedTrace();
        }
        if (mode == "xtrans-raw-contract") {
            return xtrans_demosaicnet_test::cfaAndContract();
        }
        if (mode == "xtrans-raw-reviewed") {
            return xtrans_demosaicnet_test::reviewedRawWrapper();
        }
        if (mode == "xveon-mock") {
            return xtrans_xveon_test::mockContract();
        }
        if (mode == "xveon-loader") {
            return xtrans_xveon_test::loaderContract();
        }
        if (mode == "xveon-reviewed") {
            return xtrans_xveon_test::reviewedModel();
        }
        if (mode == "inference-benchmark") {
            return demosaicnet_inference_test::benchmark();
        }
        std::cerr << "unknown test mode: " << mode << '\n';
        return 2;
    } catch (const std::exception &error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
