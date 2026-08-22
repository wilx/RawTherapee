/*
 * Developer-only X-Trans MLRI experiment.
 *
 * This is an independent C++ implementation of the residual-interpolation
 * equations described by Kiku et al. and the later 6x6 X-Trans engineering
 * generalization published by Rainbow-Johnny-Johnny-Image-Processing-Lim.
 * See devnotes/xtrans-mlri-design.md and
 * licenses/MLRI_XTRANS_MATLAB_LICENSE.
 */
#pragma once

#include "array2D.h"

#include <cstdint>
#include <string>

namespace rtengine
{

constexpr const char *MLRI_XTRANS_TWO_PASS_METHOD = "mlri-xtrans-2pass";
constexpr const char *MLRI_XTRANS_TWO_PASS_CORRECTED_METHOD =
    "mlri-xtrans-2pass-corrected";
constexpr const char *MLRI_XTRANS_TWO_PASS_CORRECTED_FINAL_ONLY_METHOD =
    "mlri-xtrans-2pass-corrected-final-only";
constexpr const char *MLRI_XTRANS_PAPER_CORE_2014_METHOD =
    "mlri-xtrans-paper-core-2014";
constexpr const char *MLRI_XTRANS_PAPER_CORE_2016_METHOD =
    "mlri-xtrans-paper-core-2016";

// Keep the authenticated MATLAB behavior available as a distinct algorithm.
// The corrected variant changes only the four blue diagonal/anti-diagonal
// Laplacian guide selections documented in devnotes/xtrans-mlri-design.md.
enum class MlriXTransVariant {
    MATLAB_REFERENCE,
    CORRECTED_BLUE_DIAGONAL_GUIDES,
    CORRECTED_BLUE_DIAGONAL_GUIDES_FINAL_ONLY,
    PAPER_CORE_2014,
    PAPER_CORE_2016
};

enum class MlriXTransErrorCode {
    NONE,
    SIZE,
    CFA,
    ALLOCATION,
    NONFINITE,
    INTERNAL
};

const char *mlriXTransErrorCodeName(MlriXTransErrorCode code);

struct MlriXTransRunResult final {
    MlriXTransErrorCode code = MlriXTransErrorCode::NONE;
    std::string message;
    std::uint64_t tileCount = 0;
    std::uint32_t workerCount = 0;
    std::uint64_t workspaceBytesPerWorker = 0;
    int coreSize = 0;
    int halo = 0;

    explicit operator bool() const
    {
        return code == MlriXTransErrorCode::NONE;
    }
};

MlriXTransRunResult demosaicMlriXTrans(
    const array2D<float> &rawData,
    array2D<float> &red,
    array2D<float> &green,
    array2D<float> &blue,
    int width,
    int height,
    const int xtrans[6][6],
    MlriXTransVariant variant = MlriXTransVariant::MATLAB_REFERENCE);

// Untiled entry point used only by the native parity suite.  Input and output
// are row-major canonical-coordinate arrays in the RawTherapee 0..65535
// domain.  Production dispatch always uses demosaicMlriXTrans().
MlriXTransRunResult demosaicMlriXTransReference(
    const float *mosaic,
    float *red,
    float *green,
    float *blue,
    int width,
    int height,
    int originX = 0,
    int originY = 0,
    MlriXTransVariant variant = MlriXTransVariant::MATLAB_REFERENCE);

} // namespace rtengine
