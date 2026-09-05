#include "tgmr/hard_cases.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace tgmr
{
namespace
{

constexpr double PI = 3.141592653589793238462643383279502884;
constexpr unsigned PATCH_EDGE = 7;
constexpr unsigned CHANNELS = 3;
constexpr unsigned OVERSAMPLE = 8;

std::uint64_t splitmix64(std::uint64_t value)
{
    value += UINT64_C(0x9e3779b97f4a7c15);
    value = (value ^ (value >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)) * UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31);
}

double unit(std::uint64_t value)
{
    return static_cast<double>(splitmix64(value) >> 11)
        * (1.0 / 9007199254740992.0);
}

std::int64_t reflectWithoutRepeat(std::int64_t value, std::int64_t size)
{
    if (size <= 1) return 0;
    const std::int64_t period = 2 * size - 2;
    value %= period;
    if (value < 0) value += period;
    return value < size ? value : period - value;
}

struct Weight final {
    std::uint32_t index = 0;
    double value = 0.0;
};

std::vector<Weight> integratedGaussianWeights(
    double center,
    double step,
    double sigmaOutput,
    std::uint32_t sourceExtent)
{
    if (!(step > 0.0) || !(sigmaOutput > 0.0) || sourceExtent == 0) {
        throw std::runtime_error("invalid physical-renderer geometry");
    }
    const double sigma = sigmaOutput * step;
    const double lower = center - 0.5 * step;
    const double upper = center + 0.5 * step;
    const double radius = 4.0 * sigma + 0.5 * step + 1.0;
    const std::int64_t first = static_cast<std::int64_t>(std::floor(center - radius));
    const std::int64_t last = static_cast<std::int64_t>(std::ceil(center + radius));
    const double denominator = std::sqrt(2.0) * sigma;
    std::vector<Weight> output;
    output.reserve(static_cast<std::size_t>(last - first + 1));
    for (std::int64_t virtualIndex = first; virtualIndex <= last; ++virtualIndex) {
        const double sampleCenter = static_cast<double>(virtualIndex) + 0.5;
        const double weight = 0.5 * (std::erf((upper - sampleCenter) / denominator)
                                   - std::erf((lower - sampleCenter) / denominator));
        if (weight <= 0.0) continue;
        const std::uint32_t index = static_cast<std::uint32_t>(
            reflectWithoutRepeat(virtualIndex, sourceExtent));
        auto existing = std::find_if(output.begin(), output.end(),
            [index](const Weight &candidate) { return candidate.index == index; });
        if (existing == output.end()) output.push_back({index, weight});
        else existing->value += weight;
    }
    double sum = 0.0;
    for (const Weight &weight : output) sum += weight.value;
    if (!(sum > 0.0) || !std::isfinite(sum)) {
        throw std::runtime_error("physical-renderer kernel has zero or non-finite mass");
    }
    for (Weight &weight : output) weight.value /= sum;
    return output;
}

std::array<double, PATCH_EDGE * PATCH_EDGE * CHANNELS> renderPatchAt(
    const LinearImage &image,
    std::uint32_t outputWidth,
    std::uint32_t outputHeight,
    std::int64_t centerX,
    std::int64_t centerY,
    const SensorPhysicalParameters &parameters)
{
    if (image.width == 0 || image.height == 0
        || image.rgb.size() != static_cast<std::size_t>(image.width) * image.height * 3
        || outputWidth < PATCH_EDGE || outputHeight < PATCH_EDGE) {
        throw std::runtime_error("source is too small for physical patch rendering");
    }
    const double stepX = static_cast<double>(image.width) / outputWidth;
    const double stepY = static_cast<double>(image.height) / outputHeight;
    const double offsetX = parameters.offsetXEighths / 8.0;
    const double offsetY = parameters.offsetYEighths / 8.0;
    std::array<std::vector<Weight>, PATCH_EDGE> xWeights;
    std::array<std::vector<Weight>, PATCH_EDGE> yWeights;
    for (unsigned index = 0; index < PATCH_EDGE; ++index) {
        const std::int64_t x = centerX + static_cast<std::int64_t>(index) - 3;
        const std::int64_t y = centerY + static_cast<std::int64_t>(index) - 3;
        const double sourceX = (static_cast<double>(x) + 0.5 + offsetX) * stepX;
        const double sourceY = (static_cast<double>(y) + 0.5 + offsetY) * stepY;
        xWeights[index] = integratedGaussianWeights(
            sourceX, stepX, parameters.sigma, image.width);
        yWeights[index] = integratedGaussianWeights(
            sourceY, stepY, parameters.sigma, image.height);
    }
    std::array<double, PATCH_EDGE * PATCH_EDGE * CHANNELS> output{};
    for (unsigned y = 0; y < PATCH_EDGE; ++y) {
        for (unsigned x = 0; x < PATCH_EDGE; ++x) {
            for (unsigned channel = 0; channel < CHANNELS; ++channel) {
                double value = 0.0;
                for (const Weight &wy : yWeights[y]) {
                    for (const Weight &wx : xWeights[x]) {
                        const std::size_t input =
                            (static_cast<std::size_t>(wy.index) * image.width + wx.index) * 3
                            + channel;
                        value += wy.value * wx.value * image.rgb[input];
                    }
                }
                if (!std::isfinite(value)) {
                    throw std::runtime_error("physical renderer produced a non-finite sample");
                }
                output[channel * 49 + y * 7 + x] = value;
            }
        }
    }
    return output;
}

std::array<double, 3> hsv(double hue, double saturation, double value)
{
    hue -= std::floor(hue);
    const double scaled = hue * 6.0;
    const int sector = static_cast<int>(std::floor(scaled)) % 6;
    const double fraction = scaled - std::floor(scaled);
    const double p = value * (1.0 - saturation);
    const double q = value * (1.0 - saturation * fraction);
    const double t = value * (1.0 - saturation * (1.0 - fraction));
    switch (sector) {
        case 0: return {{value, t, p}};
        case 1: return {{q, value, p}};
        case 2: return {{p, value, t}};
        case 3: return {{p, q, value}};
        case 4: return {{t, p, value}};
        default: return {{value, p, q}};
    }
}

std::array<double, 3> mix(
    const std::array<double, 3> &background,
    const std::array<double, 3> &foreground,
    double alpha)
{
    std::array<double, 3> output{};
    for (unsigned channel = 0; channel < 3; ++channel) {
        output[channel] = background[channel]
            + alpha * (foreground[channel] - background[channel]);
    }
    return output;
}

double sceneAlpha(
    SyntheticFamily family,
    double x,
    double y,
    double angle,
    double width,
    std::uint64_t random)
{
    const double cs = std::cos(angle);
    const double sn = std::sin(angle);
    const double offsetX = (unit(random + 1) - 0.5) * 1.5;
    const double offsetY = (unit(random + 2) - 0.5) * 1.5;
    const double dx = x - 3.5 - offsetX;
    const double dy = y - 3.5 - offsetY;
    const double u = cs * dx + sn * dy;
    const double v = -sn * dx + cs * dy;
    switch (family) {
        case SyntheticFamily::STEP:
            return u >= 0.0 ? 1.0 : 0.0;
        case SyntheticFamily::THIN_LINE:
            return std::abs(u) <= width * 0.5 ? 1.0 : 0.0;
        case SyntheticFamily::INTERSECTION: {
            const double second = std::abs(cs * dy - sn * dx);
            return std::abs(u) <= width * 0.5 || second <= width * 0.5 ? 1.0 : 0.0;
        }
        case SyntheticFamily::DOTS_STARS: {
            double nearest = std::hypot(dx, dy);
            const unsigned points = 1U + static_cast<unsigned>(splitmix64(random + 3) % 4U);
            for (unsigned point = 1; point < points; ++point) {
                const double px = (unit(random + 11 + point * 2) - 0.5) * 6.0;
                const double py = (unit(random + 12 + point * 2) - 0.5) * 6.0;
                nearest = std::min(nearest, std::hypot(dx - px, dy - py));
            }
            const double radius = 0.25 + 0.75 * unit(random + 4);
            return nearest <= radius ? 1.0 : 0.0;
        }
        case SyntheticFamily::SATURATED_HIGHLIGHT: {
            const double radius = 0.35 + 1.1 * unit(random + 4);
            const double distance = std::hypot(dx, dy);
            return std::exp(-0.5 * distance * distance / (radius * radius));
        }
        case SyntheticFamily::PERIODIC_DETAIL: {
            const double period = 0.7 + 1.8 * unit(random + 4);
            return std::sin(2.0 * PI * u / period) >= 0.0 ? 1.0 : 0.0;
        }
        case SyntheticFamily::PROCEDURAL_STROKES: {
            const double curve = u + 0.16 * v * v - 0.35;
            const double ring = std::abs(std::hypot(dx + 0.8, dy - 0.5)
                                         - (1.1 + unit(random + 4)));
            return std::abs(curve) <= width * 0.5 || ring <= width * 0.35 ? 1.0 : 0.0;
        }
        case SyntheticFamily::FRAME_MATTE: {
            const double halfWidth = 1.1 + 1.7 * unit(random + 4);
            const double halfHeight = 1.1 + 1.7 * unit(random + 5);
            const double border = 0.18 + 0.7 * width;
            const double rectangleDistance = std::min(
                std::abs(std::abs(dx) - halfWidth),
                std::abs(std::abs(dy) - halfHeight));
            const bool onExtent = std::abs(dx) <= halfWidth + border
                && std::abs(dy) <= halfHeight + border;
            return onExtent && rectangleDistance <= border ? 1.0 : 0.0;
        }
    }
    return 0.0;
}

LinearImage oversampledSyntheticScene(
    SyntheticFamily family,
    std::uint64_t random,
    double angle,
    double width,
    const std::array<double, 3> &background,
    const std::array<double, 3> &foreground)
{
    LinearImage image;
    image.width = PATCH_EDGE * OVERSAMPLE;
    image.height = PATCH_EDGE * OVERSAMPLE;
    image.sourceWidth = image.width;
    image.sourceHeight = image.height;
    image.orientation = 1;
    image.rgb.resize(static_cast<std::size_t>(image.width) * image.height * 3);
    for (unsigned y = 0; y < image.height; ++y) {
        for (unsigned x = 0; x < image.width; ++x) {
            const double px = (x + 0.5) / OVERSAMPLE;
            const double py = (y + 0.5) / OVERSAMPLE;
            double alpha = sceneAlpha(family, px, py, angle, width, random);
            if (family == SyntheticFamily::SATURATED_HIGHLIGHT) {
                alpha = std::min(1.4, alpha * (1.0 + 0.8 * unit(random + 40)));
            }
            const auto color = mix(background, foreground, alpha);
            const std::size_t index = (static_cast<std::size_t>(y) * image.width + x) * 3;
            std::copy(color.begin(), color.end(), image.rgb.begin() + index);
        }
    }
    return image;
}

} // namespace

SensorPhysicalParameters sensorPhysicalParameters(
    const std::array<std::uint8_t, 32> &sourceIdentity)
{
    static const double scales[] = {1.5, 2.0, 2.5};
    static const double sigmas[] = {0.25, 0.50, 0.75};
    static const std::int8_t offsets[] = {-3, -1, 1, 3};
    SensorPhysicalParameters output;
    output.scale = scales[sourceIdentity[0] % 3];
    output.sigma = sigmas[sourceIdentity[1] % 3];
    const unsigned placement = sourceIdentity[2] % 16;
    output.offsetXEighths = offsets[placement % 4];
    output.offsetYEighths = offsets[placement / 4];
    return output;
}

std::vector<std::array<double, 7 * 7 * 3>> renderSensorPhysicalProxy(
    const LinearImage &image,
    const std::vector<PatchSelection> &selections,
    const SensorPhysicalParameters &parameters)
{
    if (parameters.scale != 1.5 && parameters.scale != 2.0 && parameters.scale != 2.5) {
        throw std::runtime_error("physical renderer scale is not in the frozen set");
    }
    if (parameters.sigma != 0.25 && parameters.sigma != 0.50
        && parameters.sigma != 0.75) {
        throw std::runtime_error("physical renderer sigma is not in the frozen set");
    }
    const auto validOffset = [](std::int8_t value) {
        return value == -3 || value == -1 || value == 1 || value == 3;
    };
    if (!validOffset(parameters.offsetXEighths)
        || !validOffset(parameters.offsetYEighths)) {
        throw std::runtime_error("physical renderer offset is not in the frozen set");
    }
    const std::uint32_t outputWidth = static_cast<std::uint32_t>(
        std::floor(image.width / parameters.scale));
    const std::uint32_t outputHeight = static_cast<std::uint32_t>(
        std::floor(image.height / parameters.scale));
    if (outputWidth < PATCH_EDGE || outputHeight < PATCH_EDGE) {
        throw std::runtime_error("physical renderer output is smaller than 7x7");
    }
    std::vector<std::array<double, 7 * 7 * 3>> output;
    output.reserve(selections.size());
    for (const PatchSelection &selection : selections) {
        const double normalizedX = (selection.x + 3.5) / image.width;
        const double normalizedY = (selection.y + 3.5) / image.height;
        std::int64_t centerX = static_cast<std::int64_t>(
            std::llround(normalizedX * outputWidth - 0.5));
        std::int64_t centerY = static_cast<std::int64_t>(
            std::llround(normalizedY * outputHeight - 0.5));
        centerX = std::max<std::int64_t>(3,
            std::min<std::int64_t>(outputWidth - 4, centerX));
        centerY = std::max<std::int64_t>(3,
            std::min<std::int64_t>(outputHeight - 4, centerY));
        output.push_back(renderPatchAt(
            image, outputWidth, outputHeight, centerX, centerY, parameters));
    }
    return output;
}

std::uint64_t syntheticReplacementCount(
    std::uint64_t total,
    std::uint16_t basisPoints)
{
    if (basisPoints > 500) {
        throw std::runtime_error("synthetic ratio is outside the frozen 0..5 percent range");
    }
    // Divide first so even a defensive call outside the TGPC 16-million-record
    // limit cannot overflow uint64. The remainder product is below five
    // million because basisPoints is at most 500.
    return (total / 10000) * basisPoints
        + ((total % 10000) * basisPoints + 5000) / 10000;
}

bool syntheticReplacementAt(
    std::uint64_t ordinal,
    std::uint64_t total,
    std::uint16_t basisPoints,
    std::uint64_t &syntheticIndex)
{
    if (ordinal >= total) {
        throw std::runtime_error("synthetic replacement ordinal is outside the corpus");
    }
    if (total > 16'000'000) {
        throw std::runtime_error("synthetic replacement corpus exceeds the TGPC limit");
    }
    const std::uint64_t count = syntheticReplacementCount(total, basisPoints);
    if (count == 0) return false;
    // total is bounded by the TGPC record limit, so these products cannot
    // approach uint64 overflow after syntheticReplacementCount validates it.
    const std::uint64_t before = ordinal * count / total;
    const std::uint64_t after = (ordinal + 1) * count / total;
    if (after == before) return false;
    syntheticIndex = after - 1;
    return true;
}

SyntheticPatch generateSyntheticPatch(
    std::uint64_t syntheticIndex,
    std::uint64_t seed)
{
    const std::uint64_t random = splitmix64(seed ^ syntheticIndex);
    SyntheticPatch output;
    output.family = static_cast<SyntheticFamily>(syntheticIndex % 8);
    output.opticallyFiltered = ((syntheticIndex / 8) % 4) != 0;
    output.phasePlacement = static_cast<std::uint32_t>((syntheticIndex / 32) % 18);
    const double angle = (splitmix64(random + 6) % 32) * PI / 32.0;
    const double width = 0.35 + 2.65 * unit(random + 7);
    const double backgroundValue = 0.005 + 0.65 * unit(random + 8);
    const double foregroundValue = 0.35 + 1.05 * unit(random + 9);
    const double backgroundSaturation = 0.05 + 0.55 * unit(random + 10);
    const double foregroundSaturation = 0.35 + 0.65 * unit(random + 11);
    const auto background = hsv(unit(random + 12), backgroundSaturation, backgroundValue);
    const auto foreground = hsv(
        unit(random + 13), foregroundSaturation, foregroundValue);
    if (!output.opticallyFiltered) {
        const double phaseX = static_cast<double>(output.phasePlacement % 6) / 6.0;
        const double phaseY = static_cast<double>(output.phasePlacement / 6) / 3.0;
        for (unsigned y = 0; y < PATCH_EDGE; ++y) {
            for (unsigned x = 0; x < PATCH_EDGE; ++x) {
                double alpha = sceneAlpha(output.family, x + 0.5 + phaseX,
                                          y + 0.5 + phaseY, angle, width, random);
                if (output.family == SyntheticFamily::SATURATED_HIGHLIGHT) {
                    alpha = std::min(1.4, alpha * (1.0 + 0.8 * unit(random + 40)));
                }
                const auto color = mix(background, foreground, alpha);
                for (unsigned channel = 0; channel < 3; ++channel) {
                    output.rgb[channel * 49 + y * 7 + x] = color[channel];
                }
            }
        }
    } else {
        LinearImage scene = oversampledSyntheticScene(
            output.family, random, angle, width, background, foreground);
        SensorPhysicalParameters parameters;
        parameters.scale = 2.0;
        parameters.sigma = std::array<double, 3>{{0.25, 0.50, 0.75}}[
            splitmix64(random + 14) % 3];
        static const std::int8_t offsets[] = {-3, -1, 1, 3};
        parameters.offsetXEighths = offsets[output.phasePlacement % 4];
        parameters.offsetYEighths = offsets[(output.phasePlacement / 4) % 4];
        // The high-resolution synthetic scene is exactly 8x the desired
        // 7x7 output; renderPatchAt takes explicit output dimensions, so its
        // scale member is irrelevant to this internal path.
        output.rgb = renderPatchAt(scene, PATCH_EDGE, PATCH_EDGE, 3, 3, parameters);
    }
    return output;
}

const char *syntheticFamilyName(SyntheticFamily family)
{
    switch (family) {
        case SyntheticFamily::STEP: return "step";
        case SyntheticFamily::THIN_LINE: return "thin-line";
        case SyntheticFamily::INTERSECTION: return "intersection";
        case SyntheticFamily::DOTS_STARS: return "dots-stars";
        case SyntheticFamily::SATURATED_HIGHLIGHT: return "saturated-highlight";
        case SyntheticFamily::PERIODIC_DETAIL: return "periodic-detail";
        case SyntheticFamily::PROCEDURAL_STROKES: return "procedural-strokes";
        case SyntheticFamily::FRAME_MATTE: return "frame-matte";
    }
    return "invalid";
}

} // namespace tgmr
