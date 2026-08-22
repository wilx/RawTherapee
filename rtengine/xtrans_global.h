/*
 * Developer-only global reconstruction experiments for Fujifilm X-Trans.
 *
 * The three variants isolate whole-image support, color-difference modeling,
 * and an edge-preserving green prior.  They are reference implementations,
 * deliberately hidden from the GUI and ordinary method enumeration.
 */
#pragma once

#include "array2D.h"

#include <cstdint>
#include <string>
#include <vector>

namespace rtengine
{

constexpr const char *XTRANS_GLOBAL_SPECTRAL_RGB_METHOD =
    "xtrans-global-spectral-rgb";
constexpr const char *XTRANS_GLOBAL_SPECTRAL_DIFF_METHOD =
    "xtrans-global-spectral-diff";
constexpr const char *XTRANS_GLOBAL_SPECTRAL_EDGE_METHOD =
    "xtrans-global-spectral-edge";

enum class GlobalXTransVariant {
    INDEPENDENT_RGB,
    GREEN_COLOR_DIFFERENCE,
    EDGE_PRESERVING_GREEN
};

enum class GlobalXTransInitialization {
    TRIANGULATED,
    ZERO_FILLED
};

enum class GlobalXTransErrorCode {
    NONE,
    SIZE,
    CFA,
    PARAMETER,
    ALLOCATION,
    NONFINITE,
    SOLVER,
    IO,
    INTERNAL
};

const char *globalXTransErrorCodeName(GlobalXTransErrorCode code);

struct GlobalXTransOptions final {
    // Phase A uses lambdaRgb.  Phases B/C use lambdaGreen and lambdaChroma.
    double lambdaRgb = 0.02;
    double lambdaGreen = 0.01;
    double lambdaChroma = 0.10;
    int spectralExponent = 1;
    int maximumIterations = 20;
    double relativeTolerance = 1e-5;

    // Phase C starts with a Phase-B solve, followed by this many IRLS solves.
    int edgeOuterIterations = 3;
    int edgeInnerIterations = 10;
    double charbonnierEpsilon = 0.01;

    // Zero means one whole-image solve.  The controlled alternatives are
    // overlapping 96, 192, or 384-pixel tiles with half-tile stride.
    int tileSize = 0;
    GlobalXTransInitialization initialization =
        GlobalXTransInitialization::TRIANGULATED;

    // Optional developer prefix.  The implementation writes canonical
    // little-endian float32 planes and a residual CSV when non-empty.
    std::string debugOutputPrefix;
};

struct GlobalXTransRunResult final {
    GlobalXTransErrorCode code = GlobalXTransErrorCode::NONE;
    std::string message;
    std::uint64_t setupMicroseconds = 0;
    std::uint64_t solverMicroseconds = 0;
    std::uint64_t totalMicroseconds = 0;
    std::uint64_t workingBufferBytes = 0;
    std::uint64_t tileCount = 0;
    int completedIterations = 0;
    int completedOuterIterations = 0;
    double initialRelativeResidual = 0.0;
    double finalRelativeResidual = 0.0;
    double preProjectionSampleMaximum = 0.0;
    double preProjectionSampleRms = 0.0;
    double finalSampleMaximum = 0.0;
    double finalSampleRms = 0.0;
    std::vector<double> residualHistory;

    explicit operator bool() const
    {
        return code == GlobalXTransErrorCode::NONE;
    }
};

GlobalXTransRunResult demosaicGlobalXTrans(
    const array2D<float> &rawData,
    array2D<float> &red,
    array2D<float> &green,
    array2D<float> &blue,
    int width,
    int height,
    const int xtrans[6][6],
    GlobalXTransVariant variant,
    const GlobalXTransOptions &options);

// Native-test and developer-tool entry point.  All buffers are contiguous
// row-major float32 in actual image coordinates.
GlobalXTransRunResult demosaicGlobalXTransReference(
    const float *mosaic,
    float *red,
    float *green,
    float *blue,
    int width,
    int height,
    const int xtrans[6][6],
    GlobalXTransVariant variant,
    const GlobalXTransOptions &options);

} // namespace rtengine
