/*
 *  Strict RTNN v1 container reader.
 *
 *  The wire constants below intentionally mirror devnotes/rtnn-v1-format.md
 *  instead of sharing generated structures with the Python writer. Every
 *  scalar is decoded explicitly so untrusted bytes are never interpreted as a
 *  native C++ object.
 */

#include "rtnnreader_p.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <new>
#include <string>
#include <utility>

#include <glib/gstdio.h>
#include <glibmm/checksum.h>

namespace rtengine
{

namespace neural
{

namespace detail
{

namespace
{

constexpr std::uint8_t MAGIC[] = {'R', 'T', 'N', 'N', '\r', '\n', 0x1a, '\n'};
constexpr std::uint16_t FORMAT_MAJOR = 1;
constexpr std::uint16_t FORMAT_MINOR = 0;
constexpr std::uint32_t ENDIAN_MARKER = 0x01020304;
constexpr std::uint32_t HEADER_SIZE = 192;
constexpr std::uint32_t RECORD_SIZE = 96;
constexpr std::uint64_t ALIGNMENT = 64;
constexpr std::uint32_t SCALAR_FLOAT32 = 1;
constexpr std::uint32_t FLAGS = 0;

constexpr std::uint64_t MAX_FILE_BYTES = 64 * 1024 * 1024;
constexpr std::uint64_t MAX_PAYLOAD_BYTES = 64 * 1024 * 1024;
constexpr std::uint32_t MAX_TENSOR_COUNT = 256;
constexpr std::uint16_t MAX_RANK = 4;
constexpr std::uint32_t MAX_DIMENSION = 1048576;
constexpr std::uint64_t MAX_TENSOR_ELEMENTS = 16777216;

static_assert(sizeof(float) == 4, "RTNN v1 requires four-byte float");
static_assert(std::numeric_limits<float>::is_iec559, "RTNN v1 requires IEEE-754 float");
static_assert(ALIGNMENT % alignof(float) == 0, "RTNN alignment must preserve float alignment");

struct ParseFailure final {
    NeuralModelErrorCode code;
    std::string message;
};

struct ParsedRecord final {
    const TensorBinding *binding;
    std::uint64_t relativeOffset;
    std::uint64_t byteLength;
    Sha256Digest sha256;
};

[[noreturn]] void fail(NeuralModelErrorCode code, const std::string &message)
{
    throw ParseFailure{code, message};
}

std::uint16_t readU16(const std::vector<std::uint8_t> &data, std::size_t offset)
{
    return static_cast<std::uint16_t>(data[offset]) | static_cast<std::uint16_t>(data[offset + 1]) << 8;
}

std::uint32_t readU32(const std::vector<std::uint8_t> &data, std::size_t offset)
{
    return static_cast<std::uint32_t>(data[offset]) | static_cast<std::uint32_t>(data[offset + 1]) << 8 | static_cast<std::uint32_t>(data[offset + 2]) << 16 | static_cast<std::uint32_t>(data[offset + 3]) << 24;
}

std::uint64_t readU64(const std::vector<std::uint8_t> &data, std::size_t offset)
{
    return static_cast<std::uint64_t>(readU32(data, offset)) | static_cast<std::uint64_t>(readU32(data, offset + 4)) << 32;
}

std::uint64_t checkedAdd(std::uint64_t left, std::uint64_t right, const char *label)
{
    if (left > std::numeric_limits<std::uint64_t>::max() - right) {
        fail(NeuralModelErrorCode::RANGE, std::string(label) + " overflows uint64");
    }

    return left + right;
}

std::uint64_t checkedMultiply(std::uint64_t left, std::uint64_t right, const char *label)
{
    if (left && right > std::numeric_limits<std::uint64_t>::max() / left) {
        fail(NeuralModelErrorCode::RANGE, std::string(label) + " overflows uint64");
    }

    return left * right;
}

std::uint64_t alignUp(std::uint64_t value)
{
    const std::uint64_t remainder = value % ALIGNMENT;
    return remainder ? checkedAdd(value, ALIGNMENT - remainder, "aligned offset") : value;
}

int hexNibble(char value)
{
    if (value >= '0' && value <= '9') {
        return value - '0';
    }
    if (value >= 'a' && value <= 'f') {
        return value - 'a' + 10;
    }
    if (value >= 'A' && value <= 'F') {
        return value - 'A' + 10;
    }
    return -1;
}

Sha256Digest digestFromHex(const char *text)
{
    if (!text || std::strlen(text) != 64) {
        fail(NeuralModelErrorCode::SCHEMA, "compiled RTNN binding contains an invalid digest");
    }

    Sha256Digest digest{{}};
    for (std::size_t i = 0; i < digest.size(); ++i) {
        const int high = hexNibble(text[i * 2]);
        const int low = hexNibble(text[i * 2 + 1]);
        if (high < 0 || low < 0) {
            fail(NeuralModelErrorCode::SCHEMA, "compiled RTNN binding contains an invalid digest");
        }
        digest[i] = static_cast<std::uint8_t>((high << 4) | low);
    }
    return digest;
}

Sha256Digest digestBytes(const std::uint8_t *data, std::size_t size)
{
    Glib::Checksum checksum(Glib::Checksum::CHECKSUM_SHA256);
    if (!checksum) {
        fail(NeuralModelErrorCode::DIGEST, "SHA-256 is unavailable");
    }
    checksum.update(reinterpret_cast<const guchar *>(data), size);

    Sha256Digest digest{{}};
    gsize digestSize = digest.size();
    checksum.get_digest(digest.data(), &digestSize);
    if (digestSize != digest.size()) {
        fail(NeuralModelErrorCode::DIGEST, "SHA-256 returned an unexpected digest length");
    }
    return digest;
}

Sha256Digest copyDigest(const std::vector<std::uint8_t> &data, std::size_t offset)
{
    Sha256Digest digest{{}};
    std::copy_n(data.begin() + offset, digest.size(), digest.begin());
    return digest;
}

void requireZero(
    const std::vector<std::uint8_t> &data,
    std::uint64_t start,
    std::uint64_t end,
    const std::string &label)
{
    for (std::uint64_t offset = start; offset < end; ++offset) {
        if (data[static_cast<std::size_t>(offset)] != 0) {
            fail(NeuralModelErrorCode::RESERVED, label + " is not zero");
        }
    }
}

float decodeFloat(const std::vector<std::uint8_t> &data, std::size_t offset)
{
    const std::uint32_t bits = readU32(data, offset);
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

std::unique_ptr<ParsedRtnn> parseImpl(
    const std::vector<std::uint8_t> &data,
    const ModelBinding &binding)
{
    if (data.size() > MAX_FILE_BYTES) {
        fail(NeuralModelErrorCode::LIMIT, "RTNN exceeds the 64 MiB file limit");
    }
    if (data.size() < HEADER_SIZE) {
        fail(NeuralModelErrorCode::SIZE, "RTNN is shorter than its fixed header");
    }
    if (!std::equal(std::begin(MAGIC), std::end(MAGIC), data.begin())) {
        fail(NeuralModelErrorCode::MAGIC, "RTNN magic differs");
    }

    const std::uint16_t major = readU16(data, 8);
    const std::uint16_t minor = readU16(data, 10);
    if (major != FORMAT_MAJOR || minor != FORMAT_MINOR) {
        fail(NeuralModelErrorCode::VERSION, "unsupported RTNN version");
    }

    const std::uint32_t headerSize = readU32(data, 12);
    if (headerSize != HEADER_SIZE) {
        fail(NeuralModelErrorCode::SIZE, "RTNN header size differs");
    }
    if (readU32(data, 16) != ENDIAN_MARKER) {
        fail(NeuralModelErrorCode::ENDIAN, "RTNN endian marker differs");
    }
    if (readU32(data, 20) != FLAGS) {
        fail(NeuralModelErrorCode::FLAGS, "RTNN header contains unknown flags");
    }

    const std::uint32_t architectureId = readU32(data, 24);
    if (architectureId == 0) {
        fail(NeuralModelErrorCode::ENUM, "architecture ID zero is invalid");
    }
    if (architectureId != binding.architectureId) {
        fail(NeuralModelErrorCode::SCHEMA, "RTNN architecture is not the reviewed model");
    }

    const std::uint32_t modelRevision = readU32(data, 28);
    if (modelRevision == 0) {
        fail(NeuralModelErrorCode::ENUM, "model revision zero is invalid");
    }
    if (modelRevision != binding.modelRevision) {
        fail(NeuralModelErrorCode::SCHEMA, "RTNN model revision is not reviewed");
    }
    if (readU32(data, 32) != SCALAR_FLOAT32) {
        fail(NeuralModelErrorCode::ENUM, "unsupported RTNN scalar type");
    }

    const std::uint32_t tensorCount = readU32(data, 36);
    if (tensorCount > MAX_TENSOR_COUNT) {
        fail(NeuralModelErrorCode::LIMIT, "RTNN tensor count exceeds the configured limit");
    }
    if (tensorCount != binding.tensorCount) {
        fail(NeuralModelErrorCode::SCHEMA, "RTNN tensor count differs from the model schema");
    }

    const std::uint32_t recordSize = readU32(data, 40);
    if (recordSize != RECORD_SIZE) {
        fail(NeuralModelErrorCode::SIZE, "RTNN directory-record size differs");
    }
    if (readU32(data, 44) != 0) {
        fail(NeuralModelErrorCode::RESERVED, "RTNN header reserved field is not zero");
    }

    const std::uint64_t directoryOffset = readU64(data, 48);
    const std::uint64_t directorySize = readU64(data, 56);
    const std::uint64_t payloadOffset = readU64(data, 64);
    const std::uint64_t payloadSize = readU64(data, 72);
    const std::uint64_t tensorPayloadBytes = readU64(data, 80);
    const std::uint64_t declaredFileSize = readU64(data, 88);
    const Sha256Digest schemaDigest = copyDigest(data, 96);
    const Sha256Digest payloadDigest = copyDigest(data, 128);
    const Sha256Digest checkpointDigest = copyDigest(data, 160);

    if (payloadSize > MAX_PAYLOAD_BYTES) {
        fail(NeuralModelErrorCode::LIMIT, "RTNN payload exceeds the 64 MiB limit");
    }

    const std::uint64_t expectedDirectorySize = checkedMultiply(tensorCount, recordSize, "directory size");
    if (directoryOffset != headerSize || directorySize != expectedDirectorySize) {
        fail(NeuralModelErrorCode::RANGE, "RTNN directory placement is not canonical");
    }

    const std::uint64_t directoryEnd = checkedAdd(directoryOffset, directorySize, "directory end");
    if (payloadOffset != directoryEnd) {
        fail(NeuralModelErrorCode::RANGE, "RTNN payload does not immediately follow directory");
    }
    if (payloadOffset % ALIGNMENT != 0) {
        fail(NeuralModelErrorCode::ALIGNMENT, "RTNN payload region is not 64-byte aligned");
    }

    const std::uint64_t payloadEnd = checkedAdd(payloadOffset, payloadSize, "payload end");
    if (payloadEnd != declaredFileSize || declaredFileSize != data.size()) {
        fail(NeuralModelErrorCode::SIZE, "RTNN declared and actual file sizes differ");
    }
    if (schemaDigest != digestFromHex(binding.semanticSchemaSha256)) {
        fail(NeuralModelErrorCode::SCHEMA, "RTNN semantic-schema digest differs");
    }
    if (checkpointDigest != digestFromHex(binding.checkpointSha256)) {
        fail(NeuralModelErrorCode::SCHEMA, "RTNN checkpoint digest differs");
    }

    std::vector<ParsedRecord> records;
    records.reserve(tensorCount);
    std::uint64_t parameterCount = 0;
    std::uint64_t summedTensorBytes = 0;
    std::uint64_t previousEnd = 0;

    for (std::uint32_t index = 0; index < tensorCount; ++index) {
        const TensorBinding &expected = binding.tensors[index];
        const std::size_t recordOffset = static_cast<std::size_t>(directoryOffset + index * recordSize);
        const std::uint32_t tensorId = readU32(data, recordOffset);
        const std::uint16_t rank = readU16(data, recordOffset + 4);
        const std::uint16_t layoutId = readU16(data, recordOffset + 6);
        const std::uint32_t recordScalar = readU32(data, recordOffset + 8);
        const std::uint32_t recordFlags = readU32(data, recordOffset + 12);
        std::array<std::uint32_t, 4> dimensions{{}};
        for (std::size_t dimension = 0; dimension < dimensions.size(); ++dimension) {
            dimensions[dimension] = readU32(data, recordOffset + 16 + dimension * 4);
        }
        const std::uint64_t elementCount = readU64(data, recordOffset + 32);
        const std::uint64_t byteLength = readU64(data, recordOffset + 40);
        const std::uint64_t relativeOffset = readU64(data, recordOffset + 48);
        const Sha256Digest tensorDigest = copyDigest(data, recordOffset + 56);
        const std::uint64_t recordReserved = readU64(data, recordOffset + 88);

        if (tensorId != index + 1 || tensorId != expected.id) {
            fail(NeuralModelErrorCode::ORDER, "RTNN tensor IDs are not canonical");
        }
        if (rank > MAX_RANK) {
            fail(NeuralModelErrorCode::LIMIT, "RTNN tensor rank exceeds the limit");
        }
        if (rank != 1 && rank != 4) {
            fail(NeuralModelErrorCode::ENUM, "RTNN tensor rank is unsupported");
        }
        if (layoutId != static_cast<std::uint16_t>(TensorLayout::VECTOR) && layoutId != static_cast<std::uint16_t>(TensorLayout::OIHW)) {
            fail(NeuralModelErrorCode::ENUM, "RTNN tensor layout is unsupported");
        }
        if (recordScalar != SCALAR_FLOAT32) {
            fail(NeuralModelErrorCode::ENUM, "RTNN tensor scalar type is unsupported");
        }
        if (recordFlags != FLAGS) {
            fail(NeuralModelErrorCode::FLAGS, "RTNN tensor contains unknown flags");
        }
        if (recordReserved != 0) {
            fail(NeuralModelErrorCode::RESERVED, "RTNN tensor reserved field is nonzero");
        }

        for (std::size_t dimension = 0; dimension < rank; ++dimension) {
            if (dimensions[dimension] == 0) {
                fail(NeuralModelErrorCode::SCHEMA, "RTNN tensor has a zero dimension");
            }
            if (dimensions[dimension] > MAX_DIMENSION) {
                fail(NeuralModelErrorCode::LIMIT, "RTNN tensor dimension exceeds the limit");
            }
        }
        for (std::size_t dimension = rank; dimension < dimensions.size(); ++dimension) {
            if (dimensions[dimension] != 0) {
                fail(NeuralModelErrorCode::RESERVED, "RTNN tensor unused dimensions are nonzero");
            }
        }

        std::uint64_t shapeCount = 1;
        for (std::size_t dimension = 0; dimension < rank; ++dimension) {
            shapeCount = checkedMultiply(shapeCount, dimensions[dimension], "tensor element count");
        }
        if (elementCount > MAX_TENSOR_ELEMENTS) {
            fail(NeuralModelErrorCode::LIMIT, "RTNN tensor element count exceeds the limit");
        }
        if (elementCount != shapeCount) {
            fail(NeuralModelErrorCode::SCHEMA, "RTNN tensor element count differs from shape");
        }

        const std::uint64_t expectedBytes = checkedMultiply(elementCount, 4, "tensor byte length");
        if (byteLength != expectedBytes) {
            fail(NeuralModelErrorCode::SCHEMA, "RTNN tensor byte length differs");
        }
        if (rank != expected.rank || layoutId != static_cast<std::uint16_t>(expected.layout)) {
            fail(NeuralModelErrorCode::SCHEMA, "RTNN tensor rank or layout differs");
        }
        if (dimensions != expected.dimensions || elementCount != expected.elementCount) {
            fail(NeuralModelErrorCode::SCHEMA, "RTNN tensor shape differs");
        }
        if (tensorDigest != digestFromHex(expected.sha256)) {
            fail(NeuralModelErrorCode::DIGEST, "RTNN tensor recorded digest differs");
        }
        if (relativeOffset % ALIGNMENT != 0) {
            fail(NeuralModelErrorCode::ALIGNMENT, "RTNN tensor is not 64-byte aligned");
        }

        const std::uint64_t expectedOffset = alignUp(previousEnd);
        if (relativeOffset != expectedOffset) {
            if (relativeOffset < previousEnd) {
                fail(NeuralModelErrorCode::RANGE, "RTNN tensor overlaps its predecessor");
            }
            fail(NeuralModelErrorCode::ALIGNMENT, "RTNN tensor padding is not minimal");
        }

        const std::uint64_t tensorEnd = checkedAdd(relativeOffset, byteLength, "tensor end");
        if (tensorEnd > payloadSize) {
            fail(NeuralModelErrorCode::RANGE, "RTNN tensor exceeds the payload region");
        }

        parameterCount = checkedAdd(parameterCount, elementCount, "model parameter count");
        summedTensorBytes = checkedAdd(summedTensorBytes, byteLength, "model tensor byte count");
        records.push_back(ParsedRecord{&expected, relativeOffset, byteLength, tensorDigest});
        previousEnd = tensorEnd;
    }

    if (payloadSize != alignUp(previousEnd)) {
        fail(NeuralModelErrorCode::ALIGNMENT, "RTNN trailing payload padding is not minimal");
    }
    if (tensorPayloadBytes != summedTensorBytes) {
        fail(NeuralModelErrorCode::SCHEMA, "RTNN tensor-byte summary differs");
    }
    if (parameterCount != binding.parameterCount) {
        fail(NeuralModelErrorCode::SCHEMA, "RTNN parameter count differs from the binding");
    }
    if (summedTensorBytes != binding.tensorPayloadBytes) {
        fail(NeuralModelErrorCode::SCHEMA, "RTNN tensor-byte count differs from the binding");
    }

    const std::uint8_t *const payload = data.data() + static_cast<std::size_t>(payloadOffset);
    if (digestBytes(payload, static_cast<std::size_t>(payloadSize)) != payloadDigest) {
        fail(NeuralModelErrorCode::DIGEST, "RTNN payload digest differs");
    }

    previousEnd = 0;
    for (const ParsedRecord &record : records) {
        requireZero(
            data,
            payloadOffset + previousEnd,
            payloadOffset + record.relativeOffset,
            "RTNN tensor leading padding");

        const std::size_t absoluteOffset = static_cast<std::size_t>(payloadOffset + record.relativeOffset);
        if (digestBytes(data.data() + absoluteOffset, static_cast<std::size_t>(record.byteLength)) != record.sha256) {
            fail(NeuralModelErrorCode::DIGEST, "RTNN tensor payload digest differs");
        }
        for (std::uint64_t offset = 0; offset < record.byteLength; offset += sizeof(float)) {
            if (!std::isfinite(decodeFloat(data, absoluteOffset + static_cast<std::size_t>(offset)))) {
                fail(NeuralModelErrorCode::NONFINITE, "RTNN tensor contains a non-finite value");
            }
        }
        previousEnd = record.relativeOffset + record.byteLength;
    }
    requireZero(data, payloadOffset + previousEnd, payloadEnd, "RTNN trailing payload padding");

    const Sha256Digest artifactDigest = digestBytes(data.data(), data.size());
    if (artifactDigest != digestFromHex(binding.artifactSha256)) {
        fail(NeuralModelErrorCode::DIGEST, "RTNN complete-file digest differs");
    }

    std::unique_ptr<ParsedRtnn> parsed(new ParsedRtnn);
    const std::size_t payloadFloatCount = static_cast<std::size_t>(payloadSize / sizeof(float));
    constexpr std::size_t EXTRA_FLOATS = ALIGNMENT / sizeof(float) - 1;
    parsed->allocation.reset(new float[payloadFloatCount + EXTRA_FLOATS]);
    const std::uintptr_t unaligned = reinterpret_cast<std::uintptr_t>(parsed->allocation.get());
    const std::uintptr_t aligned = (unaligned + ALIGNMENT - 1) & ~(static_cast<std::uintptr_t>(ALIGNMENT) - 1);
    parsed->alignedData = reinterpret_cast<float *>(aligned);
    std::fill(parsed->alignedData, parsed->alignedData + payloadFloatCount, 0.0f);

    parsed->tensors.reserve(records.size());
    for (const ParsedRecord &record : records) {
        const std::size_t absoluteOffset = static_cast<std::size_t>(payloadOffset + record.relativeOffset);
        float *const destination = parsed->alignedData + record.relativeOffset / sizeof(float);
        for (std::uint64_t element = 0; element < record.binding->elementCount; ++element) {
            const float value = decodeFloat(
                data,
                absoluteOffset + static_cast<std::size_t>(element * sizeof(float)));
            std::memcpy(destination + element, &value, sizeof(value));
        }

        NeuralTensorView tensor;
        tensor.id = record.binding->id;
        tensor.rank = record.binding->rank;
        tensor.layout = record.binding->layout;
        tensor.dimensions = record.binding->dimensions;
        tensor.elementCount = record.binding->elementCount;
        tensor.payloadOffset = record.relativeOffset;
        tensor.byteLength = record.byteLength;
        tensor.sha256 = record.sha256;
        tensor.data = destination;
        parsed->tensors.push_back(tensor);
    }

    parsed->formatMajor = major;
    parsed->formatMinor = minor;
    parsed->architectureId = architectureId;
    parsed->modelRevision = modelRevision;
    parsed->fileSize = data.size();
    parsed->payloadRegionBytes = payloadSize;
    parsed->tensorPayloadBytes = summedTensorBytes;
    parsed->parameterCount = parameterCount;
    parsed->semanticSchemaSha256 = schemaDigest;
    parsed->checkpointSha256 = checkpointDigest;
    parsed->artifactSha256 = artifactDigest;
    parsed->payloadSha256 = payloadDigest;
    return parsed;
}

InternalLoadResult failureResult(NeuralModelErrorCode code, const std::string &message)
{
    InternalLoadResult result;
    result.error = NeuralModelError(code, message);
    return result;
}

} // namespace

InternalLoadResult parseRtnnBytes(
    const std::vector<std::uint8_t> &bytes,
    const ModelBinding &binding)
{
    try {
        InternalLoadResult result;
        result.model = parseImpl(bytes, binding);
        return result;
    } catch (const ParseFailure &error) {
        return failureResult(error.code, error.message);
    } catch (const std::bad_alloc &) {
        return failureResult(NeuralModelErrorCode::ALLOCATION, "cannot allocate RTNN model storage");
    }
}

InternalLoadResult loadRtnnFile(
    const Glib::ustring &path,
    const ModelBinding &binding)
{
    errno = 0;
    std::unique_ptr<std::FILE, int (*)(std::FILE *)> file(g_fopen(path.c_str(), "rb"), std::fclose);
    if (!file) {
        return failureResult(
            NeuralModelErrorCode::IO,
            std::string("cannot open RTNN model: ") + std::strerror(errno));
    }

    if (std::fseek(file.get(), 0, SEEK_END) != 0) {
        return failureResult(NeuralModelErrorCode::IO, "cannot determine RTNN file size");
    }
    const long end = std::ftell(file.get());
    if (end < 0) {
        return failureResult(NeuralModelErrorCode::IO, "cannot determine RTNN file size");
    }
    const std::uint64_t fileSize = static_cast<std::uint64_t>(end);
    if (fileSize > MAX_FILE_BYTES) {
        return failureResult(NeuralModelErrorCode::LIMIT, "RTNN exceeds the 64 MiB file limit");
    }
    if (std::fseek(file.get(), 0, SEEK_SET) != 0) {
        return failureResult(NeuralModelErrorCode::IO, "cannot rewind RTNN model");
    }

    try {
        std::vector<std::uint8_t> bytes(static_cast<std::size_t>(fileSize));
        const std::size_t read = bytes.empty() ? 0 : std::fread(bytes.data(), 1, bytes.size(), file.get());
        if (read != bytes.size()) {
            return failureResult(
                std::ferror(file.get()) ? NeuralModelErrorCode::IO : NeuralModelErrorCode::SIZE,
                std::ferror(file.get()) ? "cannot read RTNN model" : "RTNN changed while it was being read");
        }

        const int extra = std::fgetc(file.get());
        if (extra != EOF) {
            return failureResult(NeuralModelErrorCode::SIZE, "RTNN changed while it was being read");
        }
        if (std::ferror(file.get())) {
            return failureResult(NeuralModelErrorCode::IO, "cannot finish reading RTNN model");
        }
        return parseRtnnBytes(bytes, binding);
    } catch (const std::bad_alloc &) {
        return failureResult(NeuralModelErrorCode::ALLOCATION, "cannot allocate RTNN input buffer");
    }
}

} // namespace detail

} // namespace neural

} // namespace rtengine
