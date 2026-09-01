#pragma once

#include "tgmr/corpus.h"

#include <array>
#include <cstdint>
#include <string>

namespace tgmr
{

struct ValidationMetrics final {
    std::uint64_t patches = 0;
    std::uint64_t scalarValues = 0;
    double mse = 0.0;
    double psnr = 0.0;
    double patchRmsP99 = 0.0;
    double worstPatchRms = 0.0;
    double medianSourcePsnr = 0.0;
    std::array<double, 18> phasePsnr{};
};

struct ValidationReport final {
    std::string modelSha256;
    std::string corpusPayloadSha256;
    CorpusSplit split = CorpusSplit::VALIDATION;
    std::uint64_t requestedLimit = 0;
    double elapsedSeconds = 0.0;
    ValidationMetrics metrics;
};

// Evaluate every selected RGB patch through every frozen X-Trans phase using
// the same scalar K32/S9/q8 equations as the production engine. The physically
// measured center component is restored before error measurement.
ValidationReport validateModelOnCorpus(
    const std::string &modelPath,
    const std::string &corpusPath,
    CorpusSplit split,
    std::uint64_t patchLimit = 0);

std::string canonicalValidationJson(const ValidationReport &report);

} // namespace tgmr
