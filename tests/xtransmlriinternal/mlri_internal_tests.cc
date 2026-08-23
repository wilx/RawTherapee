#include "rtengine/xtrans_mlri.h"
#include "rtengine/xtrans_markesteijn.h"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace
{

constexpr int CFA[6][6] = {
    {1, 0, 1, 1, 2, 1},
    {2, 1, 2, 0, 1, 0},
    {1, 0, 1, 1, 2, 1},
    {1, 2, 1, 1, 0, 1},
    {0, 1, 0, 2, 1, 2},
    {1, 2, 1, 1, 0, 1}
};

void require(bool condition, const std::string &message)
{
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::uint32_t decodeU32(const unsigned char *bytes)
{
    return static_cast<std::uint32_t>(bytes[0]) |
           (static_cast<std::uint32_t>(bytes[1]) << 8) |
           (static_cast<std::uint32_t>(bytes[2]) << 16) |
           (static_cast<std::uint32_t>(bytes[3]) << 24);
}

void encodeU32(std::uint32_t value, unsigned char *bytes)
{
    bytes[0] = static_cast<unsigned char>(value);
    bytes[1] = static_cast<unsigned char>(value >> 8);
    bytes[2] = static_cast<unsigned char>(value >> 16);
    bytes[3] = static_cast<unsigned char>(value >> 24);
}

std::vector<float> readFloat32Le(const std::string &path, std::size_t count)
{
    std::FILE *file = std::fopen(path.c_str(), "rb");
    require(file != nullptr, "cannot open input " + path);
    std::vector<unsigned char> bytes(count * 4);
    const std::size_t read = std::fread(bytes.data(), 1, bytes.size(), file);
    const int extra = std::fgetc(file);
    std::fclose(file);
    require(read == bytes.size() && extra == EOF, "incorrect input payload size");
    std::vector<float> result(count);
    for (std::size_t i = 0; i < count; ++i) {
        const std::uint32_t bits = decodeU32(bytes.data() + i * 4);
        std::memcpy(&result[i], &bits, sizeof(bits));
        require(std::isfinite(result[i]), "input contains a non-finite value");
    }
    return result;
}

void writeFloat32Le(const std::string &path, const std::vector<float> &values)
{
    std::vector<unsigned char> bytes(values.size() * 4);
    for (std::size_t i = 0; i < values.size(); ++i) {
        require(std::isfinite(values[i]), "trace contains a non-finite value");
        const float normalized = values[i] / 65535.f;
        std::uint32_t bits = 0;
        std::memcpy(&bits, &normalized, sizeof(bits));
        encodeU32(bits, bytes.data() + i * 4);
    }
    std::FILE *file = std::fopen(path.c_str(), "wb");
    require(file != nullptr, "cannot create output " + path);
    require(std::fwrite(bytes.data(), 1, bytes.size(), file) == bytes.size(),
            "cannot write output " + path);
    require(std::fclose(file) == 0, "cannot close output " + path);
}

void writeUnitFloat32Le(const std::string &path, const std::vector<float> &values)
{
    std::vector<unsigned char> bytes(values.size() * 4);
    for (std::size_t i = 0; i < values.size(); ++i) {
        require(std::isfinite(values[i]), "unit trace contains a non-finite value");
        std::uint32_t bits = 0;
        std::memcpy(&bits, &values[i], sizeof(bits));
        encodeU32(bits, bytes.data() + i * 4);
    }
    std::FILE *file = std::fopen(path.c_str(), "wb");
    require(file != nullptr, "cannot create output " + path);
    require(std::fwrite(bytes.data(), 1, bytes.size(), file) == bytes.size(),
            "cannot write output " + path);
    require(std::fclose(file) == 0, "cannot close output " + path);
}

using TraceMember = std::vector<float> rtengine::MlriXTransInternalTrace::*;

const std::pair<const char *, TraceMember> PLANES[] = {
    {"pass0-green", &rtengine::MlriXTransInternalTrace::pass0Green},
    {"pass0-provisional-red", &rtengine::MlriXTransInternalTrace::pass0ProvisionalRed},
    {"pass0-provisional-blue", &rtengine::MlriXTransInternalTrace::pass0ProvisionalBlue},
    {"pass1-green", &rtengine::MlriXTransInternalTrace::pass1Green},
    {"pass1-provisional-red", &rtengine::MlriXTransInternalTrace::pass1ProvisionalRed},
    {"pass1-provisional-blue", &rtengine::MlriXTransInternalTrace::pass1ProvisionalBlue},
    {"final-tentative-red", &rtengine::MlriXTransInternalTrace::finalTentativeRed},
    {"final-tentative-blue", &rtengine::MlriXTransInternalTrace::finalTentativeBlue},
    {"final-raw-residual-red", &rtengine::MlriXTransInternalTrace::finalRawResidualRed},
    {"final-raw-residual-blue", &rtengine::MlriXTransInternalTrace::finalRawResidualBlue},
    {"final-correction-red", &rtengine::MlriXTransInternalTrace::finalCorrectionRed},
    {"final-correction-blue", &rtengine::MlriXTransInternalTrace::finalCorrectionBlue},
    {"final-unclipped-red", &rtengine::MlriXTransInternalTrace::finalUnclippedRed},
    {"final-unclipped-blue", &rtengine::MlriXTransInternalTrace::finalUnclippedBlue},
    {"final-red", &rtengine::MlriXTransInternalTrace::finalRed},
    {"final-green", &rtengine::MlriXTransInternalTrace::finalGreen},
    {"final-blue", &rtengine::MlriXTransInternalTrace::finalBlue},
};

rtengine::MlriXTransInternalTrace runTrace(
    const std::vector<float> &normalized, int width, int height, int originX, int originY)
{
    std::vector<float> input(normalized.size());
    for (std::size_t i = 0; i < input.size(); ++i) {
        input[i] = normalized[i] * 65535.f;
    }
    rtengine::MlriXTransInternalTrace trace;
    const auto result = rtengine::demosaicMlriXTransInternalTraceReference(
        input.data(), width, height, trace, originX, originY);
    require(static_cast<bool>(result),
            std::string("trace failed [") + rtengine::mlriXTransErrorCodeName(result.code) +
            "]: " + result.message);
    return trace;
}

int contract()
{
    constexpr int width = 32;
    constexpr int height = 32;
    std::vector<float> input(width * height, 0.25f);
    const auto trace = runTrace(input, width, height, 0, 0);
    require(trace.width == width && trace.height == height, "trace dimensions differ");
    for (const auto &plane : PLANES) {
        require((trace.*plane.second).size() == input.size(),
                std::string("trace plane size differs: ") + plane.first);
    }
    for (std::size_t direction = 0; direction < 8; ++direction) {
        require(trace.pass0GreenDirectional[direction].size() == input.size(),
                "pass-0 green directional trace size differs");
        require(trace.pass1GreenDirectional[direction].size() == input.size(),
                "pass-1 green directional trace size differs");
        require(trace.pass0GreenDirectionalEnergy[direction].size() == input.size(),
                "pass-0 green directional energy trace size differs");
        require(trace.pass1GreenDirectionalEnergy[direction].size() == input.size(),
                "pass-1 green directional energy trace size differs");
        require(trace.pass0GreenDirectionalWeight[direction].size() == input.size(),
                "pass-0 green directional weight trace size differs");
        require(trace.pass1GreenDirectionalWeight[direction].size() == input.size(),
                "pass-1 green directional weight trace size differs");
        for (std::size_t statistic = 0;
             statistic < rtengine::MLRI_XTRANS_REGRESSION_STATISTIC_COUNT;
             ++statistic) {
            require(trace.pass0GreenRegression[direction][statistic].size() == input.size(),
                    "pass-0 green regression trace size differs");
            require(trace.pass1GreenRegression[direction][statistic].size() == input.size(),
                    "pass-1 green regression trace size differs");
        }
    }
    float value = 1.f;
    rtengine::MlriXTransInternalTrace invalid;
    const auto tiny = rtengine::demosaicMlriXTransInternalTraceReference(
        &value, 1, 1, invalid);
    require(!tiny && tiny.code == rtengine::MlriXTransErrorCode::SIZE,
            "tiny trace input did not return SIZE");
    return 0;
}

int synthetic()
{
    constexpr int width = 64;
    constexpr int height = 64;
    const std::size_t pixels = width * height;
    std::vector<float> input(pixels);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const float channels[] = {
                0.1f + 0.7f * x / (width - 1),
                0.2f + 0.5f * y / (height - 1),
                0.3f + 0.3f * (x + y) / (width + height - 2)
            };
            input[static_cast<std::size_t>(y) * width + x] = channels[CFA[y % 6][x % 6]];
        }
    }
    const auto first = runTrace(input, width, height, 0, 0);
    const auto second = runTrace(input, width, height, 0, 0);
    for (const auto &plane : PLANES) {
        require(first.*plane.second == second.*plane.second,
                std::string("trace is nondeterministic: ") + plane.first);
    }
    require(first.pass0GreenDirectional == second.pass0GreenDirectional,
            "pass-0 green directional trace is nondeterministic");
    require(first.pass1GreenDirectional == second.pass1GreenDirectional,
            "pass-1 green directional trace is nondeterministic");
    require(first.pass0GreenDirectionalEnergy == second.pass0GreenDirectionalEnergy,
            "pass-0 green directional energy trace is nondeterministic");
    require(first.pass1GreenDirectionalEnergy == second.pass1GreenDirectionalEnergy,
            "pass-1 green directional energy trace is nondeterministic");
    require(first.pass0GreenDirectionalWeight == second.pass0GreenDirectionalWeight,
            "pass-0 green directional weight trace is nondeterministic");
    require(first.pass1GreenDirectionalWeight == second.pass1GreenDirectionalWeight,
            "pass-1 green directional weight trace is nondeterministic");
    require(first.pass0GreenRegression == second.pass0GreenRegression,
            "pass-0 green regression trace is nondeterministic");
    require(first.pass1GreenRegression == second.pass1GreenRegression,
            "pass-1 green regression trace is nondeterministic");
    std::vector<float> scaledInput(pixels), referenceRed(pixels), referenceGreen(pixels), referenceBlue(pixels);
    for (std::size_t i = 0; i < pixels; ++i) {
        scaledInput[i] = input[i] * 65535.f;
    }
    const auto reference = rtengine::demosaicMlriXTransReference(
        scaledInput.data(), referenceRed.data(), referenceGreen.data(), referenceBlue.data(),
        width, height, 0, 0,
        rtengine::MlriXTransVariant::CORRECTED_BLUE_DIAGONAL_GUIDES_FINAL_ONLY);
    require(static_cast<bool>(reference), "uninstrumented MLRI reference failed");
    require(referenceRed == first.finalRed,
            "guided-regression instrumentation changed final red");
    require(referenceGreen == first.finalGreen,
            "guided-regression instrumentation changed final green");
    require(referenceBlue == first.finalBlue,
            "guided-regression instrumentation changed final blue");
    for (std::size_t i = 0; i < pixels; ++i) {
        const int y = static_cast<int>(i / width);
        const int x = static_cast<int>(i % width);
        if (CFA[y % 6][x % 6] != 1) {
            float pass0WeightSum = 0.f;
            float pass1WeightSum = 0.f;
            for (std::size_t direction = 0; direction < 8; ++direction) {
                pass0WeightSum += first.pass0GreenDirectionalWeight[direction][i];
                pass1WeightSum += first.pass1GreenDirectionalWeight[direction][i];
            }
            require(std::abs(pass0WeightSum - 1.f) < 2e-5f,
                    "pass-0 normalized directional weights do not sum to one");
            require(std::abs(pass1WeightSum - 1.f) < 2e-5f,
                    "pass-1 normalized directional weights do not sum to one");
        }
        const float expectedR = first.finalTentativeRed[i] + first.finalCorrectionRed[i];
        const float expectedB = first.finalTentativeBlue[i] + first.finalCorrectionBlue[i];
        require(std::abs(expectedR - first.finalUnclippedRed[i]) < 0.02f,
                "red tentative/correction decomposition changed");
        require(std::abs(expectedB - first.finalUnclippedBlue[i]) < 0.02f,
                "blue tentative/correction decomposition changed");
    }
    return 0;
}

int runFiles(int argc, char **argv)
{
    require(argc == 9, "usage: mlri_internal_tests run INPUT TRUTH OUTPUT WIDTH HEIGHT ORIGIN_X ORIGIN_Y");
    const std::string inputPath = argv[2];
    const std::string truthPath = argv[3];
    const std::string outputDirectory = argv[4];
    const int width = std::stoi(argv[5]);
    const int height = std::stoi(argv[6]);
    const int originX = std::stoi(argv[7]);
    const int originY = std::stoi(argv[8]);
    require(width >= 32 && height >= 32, "dimensions must be at least 32x32");
    const auto input = readFloat32Le(inputPath, static_cast<std::size_t>(width) * height);
    const auto truth = readFloat32Le(truthPath, static_cast<std::size_t>(width) * height * 3);
    const auto trace = runTrace(input, width, height, originX, originY);
    for (const auto &plane : PLANES) {
        writeFloat32Le(outputDirectory + "/" + plane.first + ".f32le", trace.*plane.second);
    }
    for (std::size_t direction = 0; direction < 8; ++direction) {
        writeFloat32Le(
            outputDirectory + "/pass0-green-direction-" + std::to_string(direction) + ".f32le",
            trace.pass0GreenDirectional[direction]);
        writeFloat32Le(
            outputDirectory + "/pass1-green-direction-" + std::to_string(direction) + ".f32le",
            trace.pass1GreenDirectional[direction]);
        writeUnitFloat32Le(
            outputDirectory + "/pass0-green-energy-" + std::to_string(direction) + ".f32le",
            trace.pass0GreenDirectionalEnergy[direction]);
        writeUnitFloat32Le(
            outputDirectory + "/pass1-green-energy-" + std::to_string(direction) + ".f32le",
            trace.pass1GreenDirectionalEnergy[direction]);
        writeUnitFloat32Le(
            outputDirectory + "/pass0-green-weight-" + std::to_string(direction) + ".f32le",
            trace.pass0GreenDirectionalWeight[direction]);
        writeUnitFloat32Le(
            outputDirectory + "/pass1-green-weight-" + std::to_string(direction) + ".f32le",
            trace.pass1GreenDirectionalWeight[direction]);
        for (std::size_t statistic = 0;
             statistic < rtengine::MLRI_XTRANS_REGRESSION_STATISTIC_COUNT;
             ++statistic) {
            const auto statisticId = static_cast<rtengine::MlriXTransRegressionStatistic>(statistic);
            const std::string name = rtengine::mlriXTransRegressionStatisticName(statisticId);
            writeUnitFloat32Le(
                outputDirectory + "/pass0-green-regression-" + name + "-" +
                    std::to_string(direction) + ".f32le",
                trace.pass0GreenRegression[direction][statistic]);
            writeUnitFloat32Le(
                outputDirectory + "/pass1-green-regression-" + name + "-" +
                    std::to_string(direction) + ".f32le",
                trace.pass1GreenRegression[direction][statistic]);
        }
    }
    int cfa[6][6];
    for (int y = 0; y < 6; ++y) {
        for (int x = 0; x < 6; ++x) {
            const int yy = (y + originY % 6 + 6) % 6;
            const int xx = (x + originX % 6 + 6) % 6;
            cfa[y][x] = CFA[yy][xx];
        }
    }
    const std::size_t pixels = static_cast<std::size_t>(width) * height;
    std::vector<float> scaled(pixels);
    std::vector<float> markRed(pixels), markGreen(pixels), markBlue(pixels);
    for (std::size_t i = 0; i < pixels; ++i) {
        scaled[i] = input[i] * 65535.f;
    }
    const auto markStart = std::chrono::steady_clock::now();
    const auto markResult = rtengine::demosaicMarkesteijnXTransReference(
        scaled.data(), markRed.data(), markGreen.data(), markBlue.data(),
        width, height, cfa);
    require(static_cast<bool>(markResult),
            std::string("Markesteijn failed [") +
            rtengine::markesteijnXTransErrorCodeName(markResult.code) + "]: " +
            markResult.message);
    std::vector<float> markesteijn(3 * pixels);
    for (std::size_t i = 0; i < pixels; ++i) {
        markesteijn[i] = markRed[i];
        markesteijn[pixels + i] = markGreen[i];
        markesteijn[2 * pixels + i] = markBlue[i];
    }
    writeFloat32Le(outputDirectory + "/markesteijn.f32le", markesteijn);
    const double markSeconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - markStart).count();
    std::vector<float> truthGreen(pixels), oracleRed(pixels), oracleBlue(pixels);
    for (std::size_t i = 0; i < pixels; ++i) {
        truthGreen[i] = truth[pixels + i] * 65535.f;
    }
    const auto oracleResult = rtengine::demosaicMlriXTransFinalRedBlueReference(
        scaled.data(), truthGreen.data(), oracleRed.data(), oracleBlue.data(),
        width, height, originX, originY);
    require(static_cast<bool>(oracleResult),
            std::string("oracle-green final R/B failed [") +
            rtengine::mlriXTransErrorCodeName(oracleResult.code) + "]: " +
            oracleResult.message);
    writeFloat32Le(outputDirectory + "/oracle-green-red.f32le", oracleRed);
    writeFloat32Le(outputDirectory + "/oracle-green-blue.f32le", oracleBlue);
    std::vector<float> pass0Red(pixels), pass0Blue(pixels);
    const auto pass0Result = rtengine::demosaicMlriXTransFinalRedBlueReference(
        scaled.data(), trace.pass0Green.data(), pass0Red.data(), pass0Blue.data(),
        width, height, originX, originY);
    require(static_cast<bool>(pass0Result),
            std::string("pass-0-green final R/B failed [") +
            rtengine::mlriXTransErrorCodeName(pass0Result.code) + "]: " +
            pass0Result.message);
    writeFloat32Le(outputDirectory + "/pass0-guide-red.f32le", pass0Red);
    writeFloat32Le(outputDirectory + "/pass0-guide-blue.f32le", pass0Blue);
    std::cout << std::setprecision(17)
              << "pass0-guide\t" << trace.pass0GuideSeconds << '\n'
              << "pass0-green\t" << trace.pass0GreenSeconds << '\n'
              << "pass0-chroma\t" << trace.pass0ChromaSeconds << '\n'
              << "pass1-guide\t" << trace.pass1GuideSeconds << '\n'
              << "pass1-green\t" << trace.pass1GreenSeconds << '\n'
              << "pass1-chroma\t" << trace.pass1ChromaSeconds << '\n'
              << "final-red-blue\t" << trace.finalRedBlueSeconds << '\n'
              << "markesteijn\t" << markSeconds << '\n';
    return 0;
}

int finalFiles(int argc, char **argv)
{
    require(argc == 9,
            "usage: mlri_internal_tests final INPUT GREEN OUTPUT WIDTH HEIGHT ORIGIN_X ORIGIN_Y");
    const std::string inputPath = argv[2];
    const std::string greenPath = argv[3];
    const std::string outputDirectory = argv[4];
    const int width = std::stoi(argv[5]);
    const int height = std::stoi(argv[6]);
    const int originX = std::stoi(argv[7]);
    const int originY = std::stoi(argv[8]);
    require(width >= 32 && height >= 32, "dimensions must be at least 32x32");
    const std::size_t pixels = static_cast<std::size_t>(width) * height;
    const auto input = readFloat32Le(inputPath, pixels);
    const auto green = readFloat32Le(greenPath, pixels);
    std::vector<float> scaledInput(pixels), scaledGreen(pixels);
    std::vector<float> red(pixels), blue(pixels);
    for (std::size_t i = 0; i < pixels; ++i) {
        scaledInput[i] = input[i] * 65535.f;
        scaledGreen[i] = green[i] * 65535.f;
    }
    const auto result = rtengine::demosaicMlriXTransFinalRedBlueReference(
        scaledInput.data(), scaledGreen.data(), red.data(), blue.data(),
        width, height, originX, originY);
    require(static_cast<bool>(result),
            std::string("custom-green final R/B failed [") +
            rtengine::mlriXTransErrorCodeName(result.code) + "]: " + result.message);
    writeFloat32Le(outputDirectory + "/red.f32le", red);
    writeFloat32Le(outputDirectory + "/blue.f32le", blue);
    return 0;
}

int finalBankFiles(int argc, char **argv)
{
    require(argc == 10,
            "usage: mlri_internal_tests final-bank INPUT GREENS OUTPUT COUNT WIDTH HEIGHT ORIGIN_X ORIGIN_Y");
    const std::string inputPath = argv[2];
    const std::string greenPath = argv[3];
    const std::string outputPath = argv[4];
    const int count = std::stoi(argv[5]);
    const int width = std::stoi(argv[6]);
    const int height = std::stoi(argv[7]);
    const int originX = std::stoi(argv[8]);
    const int originY = std::stoi(argv[9]);
    require(count > 0 && count <= 256, "green bank count must be in 1..256");
    require(width >= 32 && height >= 32, "dimensions must be at least 32x32");
    const std::size_t pixels = static_cast<std::size_t>(width) * height;
    const auto input = readFloat32Le(inputPath, pixels);
    const auto green = readFloat32Le(greenPath, pixels * static_cast<std::size_t>(count));
    std::vector<float> scaledInput(pixels), scaledGreen(pixels);
    std::vector<float> red(pixels), blue(pixels), output(pixels * 3 * count);
    for (std::size_t i = 0; i < pixels; ++i) {
        scaledInput[i] = input[i] * 65535.f;
    }
    for (int bank = 0; bank < count; ++bank) {
        const std::size_t greenOffset = static_cast<std::size_t>(bank) * pixels;
        for (std::size_t i = 0; i < pixels; ++i) {
            scaledGreen[i] = green[greenOffset + i] * 65535.f;
        }
        const auto result = rtengine::demosaicMlriXTransFinalRedBlueReference(
            scaledInput.data(), scaledGreen.data(), red.data(), blue.data(),
            width, height, originX, originY);
        require(static_cast<bool>(result),
                std::string("green-bank final R/B failed [") +
                rtengine::mlriXTransErrorCodeName(result.code) + "]: " + result.message);
        const std::size_t outputOffset = static_cast<std::size_t>(bank) * pixels * 3;
        std::copy(red.begin(), red.end(), output.begin() + outputOffset);
        std::copy(scaledGreen.begin(), scaledGreen.end(), output.begin() + outputOffset + pixels);
        std::copy(blue.begin(), blue.end(), output.begin() + outputOffset + 2 * pixels);
    }
    writeFloat32Le(outputPath, output);
    return 0;
}

} // namespace

int main(int argc, char **argv)
{
    try {
        require(argc >= 2, "missing mode");
        const std::string mode = argv[1];
        if (mode == "contract") {
            return contract();
        }
        if (mode == "synthetic") {
            return synthetic();
        }
        if (mode == "run") {
            return runFiles(argc, argv);
        }
        if (mode == "final") {
            return finalFiles(argc, argv);
        }
        if (mode == "final-bank") {
            return finalBankFiles(argc, argv);
        }
        throw std::runtime_error("unknown mode " + mode);
    } catch (const std::exception &error) {
        std::cerr << "xtrans MLRI internal runner: " << error.what() << '\n';
        return 1;
    }
}
