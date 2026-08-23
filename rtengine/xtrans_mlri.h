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

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

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

// Exact diagnostics retained from the local models that guidedMlri() already
// fits.  The research trace attributes these quantities to each final green
// direction by following the same CFA-site completion and half-Gaussian
// support used to form that candidate.  Values stay in the algorithm's native
// 8-bit-domain units; no diagnostic feeds back into reconstruction.
enum class MlriXTransRegressionStatistic : std::size_t {
    FIT_MSE,
    MIN_FIT_MSE,
    GAIN,
    GAIN_VARIANCE,
    GAIN_MIN,
    GAIN_MAX,
    OFFSET,
    GAIN_NUMERATOR,
    GAIN_DENOMINATOR,
    GUIDE_ENERGY,
    LAPLACIAN_COUNT,
    SAMPLE_COUNT,
    PREDICTION_VARIANCE,
    PREDICTION_RANGE,
    PREDICTION_MAD,
    PREDICTION_IQR,
    MODEL_WEIGHT_ENTROPY,
    EFFECTIVE_MODEL_COUNT,
    MAX_MODEL_WEIGHT,
    TOTAL_MODEL_WEIGHT,
    LOO_PREDICTION_VARIANCE,
    MAX_MODEL_INFLUENCE,
    COUNT
};

constexpr std::size_t MLRI_XTRANS_REGRESSION_STATISTIC_COUNT =
    static_cast<std::size_t>(MlriXTransRegressionStatistic::COUNT);

const char *mlriXTransRegressionStatisticName(
    MlriXTransRegressionStatistic statistic);

using MlriXTransCandidateRegressionTrace =
    std::array<std::vector<float>, MLRI_XTRANS_REGRESSION_STATISTIC_COUNT>;

// Development-only state capture for diagnosing the corrected-final path.
// Every plane is row-major in the public 0..65535 domain.  The production
// tiled entry point never constructs this object and incurs no trace storage.
struct MlriXTransInternalTrace final {
    int width = 0;
    int height = 0;
    std::vector<float> pass0Green;
    std::array<std::vector<float>, 8> pass0GreenDirectional;
    std::array<std::vector<float>, 8> pass0GreenDirectionalEnergy;
    std::array<std::vector<float>, 8> pass0GreenDirectionalWeight;
    std::array<MlriXTransCandidateRegressionTrace, 8> pass0GreenRegression;
    std::vector<float> pass0ProvisionalRed;
    std::vector<float> pass0ProvisionalBlue;
    std::vector<float> pass1Green;
    std::array<std::vector<float>, 8> pass1GreenDirectional;
    std::array<std::vector<float>, 8> pass1GreenDirectionalEnergy;
    std::array<std::vector<float>, 8> pass1GreenDirectionalWeight;
    std::array<MlriXTransCandidateRegressionTrace, 8> pass1GreenRegression;
    std::vector<float> pass1ProvisionalRed;
    std::vector<float> pass1ProvisionalBlue;
    std::vector<float> finalTentativeRed;
    std::vector<float> finalTentativeBlue;
    std::vector<float> finalRawResidualRed;
    std::vector<float> finalRawResidualBlue;
    std::vector<float> finalCorrectionRed;
    std::vector<float> finalCorrectionBlue;
    std::vector<float> finalUnclippedRed;
    std::vector<float> finalUnclippedBlue;
    std::vector<float> finalRed;
    std::vector<float> finalGreen;
    std::vector<float> finalBlue;
    double pass0GuideSeconds = 0.0;
    double pass0GreenSeconds = 0.0;
    double pass0ChromaSeconds = 0.0;
    double pass1GuideSeconds = 0.0;
    double pass1GreenSeconds = 0.0;
    double pass1ChromaSeconds = 0.0;
    double finalRedBlueSeconds = 0.0;
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

// Untiled, canonical-coordinate research entry point.  It traces the exact
// corrected-final algorithm and is intentionally not used by GUI/CLI dispatch.
MlriXTransRunResult demosaicMlriXTransInternalTraceReference(
    const float *mosaic,
    int width,
    int height,
    MlriXTransInternalTrace &trace,
    int originX = 0,
    int originY = 0);

// Development-only stage-substitution entry point.  It runs only the final
// R/B reconstruction with a caller-supplied green guide so the experiment can
// measure an oracle-green upper bound without changing production behavior.
MlriXTransRunResult demosaicMlriXTransFinalRedBlueReference(
    const float *mosaic,
    const float *greenGuide,
    float *red,
    float *blue,
    int width,
    int height,
    int originX = 0,
    int originY = 0);

} // namespace rtengine
