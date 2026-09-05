#pragma once

#include "tgmr/image.h"
#include "tgmr/manifest.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace tgmr
{

enum class SyntheticFamily : std::uint8_t {
    STEP = 0,
    THIN_LINE = 1,
    INTERSECTION = 2,
    DOTS_STARS = 3,
    SATURATED_HIGHLIGHT = 4,
    PERIODIC_DETAIL = 5,
    PROCEDURAL_STROKES = 6,
    FRAME_MATTE = 7,
};

constexpr std::uint16_t HARD_CASE_SYNTHETIC_SEED_VERSION = 1;
constexpr std::uint64_t HARD_CASE_TRAINING_SEED = UINT64_C(0x54524d5248415244);
constexpr std::uint64_t HARD_CASE_CONTROL_SEED = UINT64_C(0x54524d52434f4e54);

struct SensorPhysicalParameters final {
    double scale = 1.5;
    double sigma = 0.25;
    // Quarter-pixel offsets are represented exactly as -3,-1,+1,+3 eighths,
    // which is the centered form of the required four quarter-pixel positions.
    std::int8_t offsetXEighths = -3;
    std::int8_t offsetYEighths = -3;
};

struct SyntheticPatch final {
    SyntheticFamily family = SyntheticFamily::STEP;
    bool opticallyFiltered = false;
    std::uint32_t phasePlacement = 0;
    std::array<double, 7 * 7 * 3> rgb{};
};

// Choose one of the frozen 3 x 3 x 16 physical-renderer configurations from
// a stable source identity.  This is intentionally independent of host RNGs.
SensorPhysicalParameters sensorPhysicalParameters(
    const std::array<std::uint8_t, 32> &sourceIdentity);

// Render the selected 7x7 neighborhoods at a lower-resolution virtual sensor.
// The output proxy is bounded to selections.size() * 147 doubles; a full-frame
// feature or resampled-image buffer is never allocated.
std::vector<std::array<double, 7 * 7 * 3>> renderSensorPhysicalProxy(
    const LinearImage &image,
    const std::vector<PatchSelection> &selections,
    const SensorPhysicalParameters &parameters);

// Produce exactly round(total * basisPoints / 10000) evenly distributed
// replacement positions.  On success syntheticIndex is the dense 0-based
// index of the selected record.
bool syntheticReplacementAt(
    std::uint64_t ordinal,
    std::uint64_t total,
    std::uint16_t basisPoints,
    std::uint64_t &syntheticIndex);

std::uint64_t syntheticReplacementCount(
    std::uint64_t total,
    std::uint16_t basisPoints);

SyntheticPatch generateSyntheticPatch(
    std::uint64_t syntheticIndex,
    std::uint64_t seed = HARD_CASE_TRAINING_SEED);

const char *syntheticFamilyName(SyntheticFamily family);

} // namespace tgmr
