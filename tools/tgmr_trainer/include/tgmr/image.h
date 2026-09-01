#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace tgmr
{

struct LinearImage final {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint16_t orientation = 1;
    std::string fileType;
    std::string iccIdentity;
    bool assumedSrgb = false;
    // Interleaved, row-major, linear-sRGB float64 values.  Values are not
    // clipped after the color transform.
    std::vector<double> rgb;
};

struct ImageClassification final {
    std::array<double, 3> channelMeans{};
    double luminanceMean = 0.0;
    double luminanceStddev = 0.0;
    double luminanceP01 = 0.0;
    double luminanceP99 = 0.0;
    double chromaRatioMean = 0.0;
    double hueDegrees = 0.0;
    double saturationMean = 0.0;
    double clippedBlackFraction = 0.0;
    double clippedWhiteFraction = 0.0;
    double gradientRms = 0.0;
    double laplacianRms = 0.0;
    double localContrast = 0.0;
    double jpegBlockiness = 0.0;
    std::string perceptualHash;
    std::string decodedPixelSha256;
};

LinearImage loadLinearImage(const std::string &path);
ImageClassification classifyImage(const LinearImage &image);
std::string canonicalClassificationJson(
    const LinearImage &image,
    const ImageClassification &classification,
    const std::string &sourceId);

} // namespace tgmr
