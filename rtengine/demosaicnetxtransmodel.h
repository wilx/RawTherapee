/*
 *  This file is part of RawTherapee.
 *
 *  Immutable binding for the reviewed Gharbi DemosaicNet X-Trans weights.
 *  The model contains canonical native-endian OIHW/vector tensors only; it
 *  performs no inference and no architecture-specific repacking.
 */
#pragma once

#include "neuralmodel.h"

#include <cstdint>
#include <memory>
#include <vector>

#include <glibmm/ustring.h>

namespace rtengine
{

namespace neural
{

enum class DemosaicNetXTransTensorId : std::uint32_t {
    MAIN_CONV1_WEIGHT = 1,
    MAIN_CONV1_BIAS = 2,
    MAIN_CONV2_WEIGHT = 3,
    MAIN_CONV2_BIAS = 4,
    MAIN_CONV3_WEIGHT = 5,
    MAIN_CONV3_BIAS = 6,
    MAIN_CONV4_WEIGHT = 7,
    MAIN_CONV4_BIAS = 8,
    MAIN_CONV5_WEIGHT = 9,
    MAIN_CONV5_BIAS = 10,
    MAIN_CONV6_WEIGHT = 11,
    MAIN_CONV6_BIAS = 12,
    MAIN_CONV7_WEIGHT = 13,
    MAIN_CONV7_BIAS = 14,
    MAIN_CONV8_WEIGHT = 15,
    MAIN_CONV8_BIAS = 16,
    MAIN_CONV9_WEIGHT = 17,
    MAIN_CONV9_BIAS = 18,
    MAIN_CONV10_WEIGHT = 19,
    MAIN_CONV10_BIAS = 20,
    MAIN_CONV11_WEIGHT = 21,
    MAIN_CONV11_BIAS = 22,
    POST_CONV_WEIGHT = 23,
    POST_CONV_BIAS = 24,
    OUTPUT_WEIGHT = 25,
    OUTPUT_BIAS = 26
};

class DemosaicNetXTransModel;

struct DemosaicNetXTransLoadResult final {
    std::shared_ptr<const DemosaicNetXTransModel> model;
    NeuralModelError error;

    explicit operator bool() const
    {
        return static_cast<bool>(model);
    }
};

class DemosaicNetXTransModel final
{
public:
    ~DemosaicNetXTransModel();

    DemosaicNetXTransModel(const DemosaicNetXTransModel &) = delete;
    DemosaicNetXTransModel &operator=(const DemosaicNetXTransModel &) = delete;

    std::uint16_t formatMajor() const;
    std::uint16_t formatMinor() const;
    std::uint32_t architectureId() const;
    std::uint32_t modelRevision() const;
    std::uint64_t fileSize() const;
    std::uint64_t payloadRegionBytes() const;
    std::uint64_t tensorPayloadBytes() const;
    std::uint64_t parameterCount() const;

    const Sha256Digest &semanticSchemaSha256() const;
    const Sha256Digest &checkpointSha256() const;
    const Sha256Digest &artifactSha256() const;
    const Sha256Digest &payloadSha256() const;

    const std::vector<NeuralTensorView> &tensors() const;
    const NeuralTensorView *tensor(DemosaicNetXTransTensorId id) const;

private:
    class Implementation;

    explicit DemosaicNetXTransModel(std::unique_ptr<Implementation> implementation);

    std::unique_ptr<Implementation> implementation_;

    friend DemosaicNetXTransLoadResult loadDemosaicNetXTransModel(const Glib::ustring &path);
};

DemosaicNetXTransLoadResult loadDemosaicNetXTransModel(const Glib::ustring &path);

} // namespace neural

} // namespace rtengine
