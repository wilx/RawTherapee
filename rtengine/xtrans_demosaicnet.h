/*
 *  This file is part of RawTherapee.
 *
 *  Developer-only raw-domain wrapper for the reviewed Gharbi DemosaicNet
 *  X-Trans model. The public processing-method list deliberately does not
 *  include these identifiers; Phase 9 profiles select them by literal PP3
 *  strings for controlled command-line evaluation.
 */
#pragma once

#include "array2D.h"
#include "demosaicnetxtransinference.h"

#include <cstdint>
#include <memory>

#include <glibmm/ustring.h>

namespace rtengine
{

constexpr const char *DEMOSAICNET_XTRANS_LINEAR_METHOD = "demosaicnet-xtrans-linear";
constexpr const char *DEMOSAICNET_XTRANS_GAMMA22_METHOD = "demosaicnet-xtrans-gamma22";

enum class DemosaicNetXTransDomain {
    LINEAR,
    GAMMA22
};

struct DemosaicNetXTransRunResult final {
    neural::NeuralModelError error;
    std::uint64_t tileCount = 0;
    std::uint32_t workerCount = 0;
    std::uint64_t workspaceBytesPerWorker = 0;

    explicit operator bool() const
    {
        return !error;
    }
};

// Successful reviewed models are retained for the process lifetime. Failures
// are intentionally not cached so a missing or replaced development artifact
// can be retried without restarting RawTherapee.
neural::DemosaicNetXTransLoadResult loadCachedDemosaicNetXTransModel(
    const Glib::ustring &path);

DemosaicNetXTransRunResult demosaicDemosaicNetXTrans(
    const array2D<float> &rawData,
    array2D<float> &red,
    array2D<float> &green,
    array2D<float> &blue,
    int width,
    int height,
    const int xtrans[6][6],
    DemosaicNetXTransDomain domain,
    std::shared_ptr<const neural::DemosaicNetXTransModel> model);

} // namespace rtengine
