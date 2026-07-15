/*
 * Private RTNN reader interface. Binding injection is intentionally unavailable
 * through RawTherapee's public neural-model API. It exists here so the reviewed
 * model binding and future independent corruption fixtures use the same parser.
 */
#pragma once

#include "neuralmodel.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include <glibmm/ustring.h>

namespace rtengine
{

namespace neural
{

class DemosaicNetXTransModel;

namespace detail
{

struct TensorBinding final {
    std::uint32_t id;
    std::uint16_t rank;
    TensorLayout layout;
    std::array<std::uint32_t, 4> dimensions;
    std::uint64_t elementCount;
    const char *sha256;
};

struct ModelBinding final {
    std::uint32_t architectureId;
    std::uint32_t modelRevision;
    const char *semanticSchemaSha256;
    const char *checkpointSha256;
    const char *artifactSha256;
    std::uint64_t parameterCount;
    std::uint64_t tensorPayloadBytes;
    const TensorBinding *tensors;
    std::size_t tensorCount;
};

struct ParsedRtnn final {
    std::uint16_t formatMajor = 0;
    std::uint16_t formatMinor = 0;
    std::uint32_t architectureId = 0;
    std::uint32_t modelRevision = 0;
    std::uint64_t fileSize = 0;
    std::uint64_t payloadRegionBytes = 0;
    std::uint64_t tensorPayloadBytes = 0;
    std::uint64_t parameterCount = 0;
    Sha256Digest semanticSchemaSha256{{}};
    Sha256Digest checkpointSha256{{}};
    Sha256Digest artifactSha256{{}};
    Sha256Digest payloadSha256{{}};
    std::unique_ptr<float[]> allocation;
    float *alignedData = nullptr;
    std::vector<NeuralTensorView> tensors;
};

struct InternalLoadResult final {
    std::unique_ptr<ParsedRtnn> model;
    NeuralModelError error;

    explicit operator bool() const
    {
        return static_cast<bool>(model);
    }
};

class DemosaicNetXTransModelAccess final
{
public:
    static std::shared_ptr<const DemosaicNetXTransModel> create(
        std::unique_ptr<ParsedRtnn> parsed);
};

InternalLoadResult parseRtnnBytes(
    const std::vector<std::uint8_t> &bytes,
    const ModelBinding &binding);

InternalLoadResult loadRtnnFile(
    const Glib::ustring &path,
    const ModelBinding &binding);

} // namespace detail

} // namespace neural

} // namespace rtengine
