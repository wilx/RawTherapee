#include "rtengine/xtrans_global.h"
#include "rtengine/xtrans_markesteijn.h"
#include "rtengine/xtrans_mlri.h"
#include "rtengine/xtrans_triangulation.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

constexpr int CFA[6][6] = {
    {1, 2, 1, 1, 0, 1},
    {0, 1, 0, 2, 1, 2},
    {1, 2, 1, 1, 0, 1},
    {1, 0, 1, 1, 2, 1},
    {2, 1, 2, 0, 1, 0},
    {1, 0, 1, 1, 2, 1}
};

constexpr const char *METHODS[] = {
    "markesteijn",
    "mlri-final",
    "triangulated-rgb",
    "triangulated-chroma",
    "global-b"
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
    std::vector<float> values(count);
    for (std::size_t index = 0; index < count; ++index) {
        const std::uint32_t bits = decodeU32(bytes.data() + index * 4);
        std::memcpy(&values[index], &bits, sizeof(bits));
        require(std::isfinite(values[index]), "input contains a non-finite value");
    }
    return values;
}

void writeFloat32Le(const std::string &path, const std::vector<float> &values)
{
    std::vector<unsigned char> bytes(values.size() * 4);
    for (std::size_t index = 0; index < values.size(); ++index) {
        require(std::isfinite(values[index]), "output contains a non-finite value");
        std::uint32_t bits = 0;
        std::memcpy(&bits, &values[index], sizeof(bits));
        encodeU32(bits, bytes.data() + index * 4);
    }
    std::FILE *file = std::fopen(path.c_str(), "wb");
    require(file != nullptr, "cannot create output " + path);
    require(std::fwrite(bytes.data(), 1, bytes.size(), file) == bytes.size(),
            "cannot write output " + path);
    require(std::fclose(file) == 0, "cannot close output " + path);
}

struct MethodOutput final {
    std::vector<float> values;
    double seconds = 0.0;
};

MethodOutput runMethod(
    const std::string &method,
    const std::vector<float> &normalizedMosaic,
    int width,
    int height,
    const int cfa[6][6])
{
    const std::size_t pixels = static_cast<std::size_t>(width) * height;
    require(normalizedMosaic.size() == pixels, "mosaic dimensions differ");
    array2D<float> raw(width, height);
    array2D<float> red(width, height);
    array2D<float> green(width, height);
    array2D<float> blue(width, height);
    for (std::size_t index = 0; index < pixels; ++index) {
        static_cast<float *>(raw)[index] = normalizedMosaic[index] * 65535.f;
    }

    const auto start = std::chrono::steady_clock::now();
    if (method == "markesteijn") {
        const auto result = rtengine::demosaicMarkesteijnXTransReference(
            static_cast<const float *>(raw), static_cast<float *>(red),
            static_cast<float *>(green), static_cast<float *>(blue),
            width, height, cfa);
        require(static_cast<bool>(result),
                std::string("Markesteijn failed [") +
                rtengine::markesteijnXTransErrorCodeName(result.code) + "]: " +
                result.message);
    } else if (method == "mlri-final") {
        const auto result = rtengine::demosaicMlriXTrans(
            raw, red, green, blue, width, height, cfa,
            rtengine::MlriXTransVariant::CORRECTED_BLUE_DIAGONAL_GUIDES_FINAL_ONLY);
        require(static_cast<bool>(result),
                std::string("MLRI failed [") +
                rtengine::mlriXTransErrorCodeName(result.code) + "]: " +
                result.message);
    } else if (method == "triangulated-rgb" || method == "triangulated-chroma") {
        const auto variant = method == "triangulated-rgb"
            ? rtengine::TriangulatedXTransVariant::INDEPENDENT_RGB
            : rtengine::TriangulatedXTransVariant::GREEN_CHROMA_DIFFERENCE;
        const auto result = rtengine::demosaicTriangulatedXTransReference(
            static_cast<const float *>(raw), static_cast<float *>(red),
            static_cast<float *>(green), static_cast<float *>(blue),
            width, height, cfa, variant);
        require(static_cast<bool>(result),
                std::string("triangulation failed [") +
                rtengine::triangulatedXTransErrorCodeName(result.code) + "]: " +
                result.message);
    } else if (method == "global-b") {
        rtengine::GlobalXTransOptions options;
        options.maximumIterations = 50;
        const auto result = rtengine::demosaicGlobalXTransReference(
            static_cast<const float *>(raw), static_cast<float *>(red),
            static_cast<float *>(green), static_cast<float *>(blue),
            width, height, cfa,
            rtengine::GlobalXTransVariant::GREEN_COLOR_DIFFERENCE, options);
        require(static_cast<bool>(result),
                std::string("global B failed [") +
                rtengine::globalXTransErrorCodeName(result.code) + "]: " +
                result.message);
    } else {
        throw std::runtime_error("unknown method " + method);
    }
    const auto end = std::chrono::steady_clock::now();

    MethodOutput output;
    output.values.resize(pixels * 3);
    const float *planes[] = {red, green, blue};
    for (int channel = 0; channel < 3; ++channel) {
        for (std::size_t index = 0; index < pixels; ++index) {
            output.values[static_cast<std::size_t>(channel) * pixels + index] =
                planes[channel][index] / 65535.f;
        }
    }
    output.seconds = std::chrono::duration<double>(end - start).count();
    return output;
}

int contract()
{
    using rtengine::MarkesteijnXTransErrorCode;
    require(std::strcmp(rtengine::markesteijnXTransErrorCodeName(
                            MarkesteijnXTransErrorCode::NONE), "NONE") == 0,
            "NONE name changed");
    require(std::strcmp(rtengine::markesteijnXTransErrorCodeName(
                            MarkesteijnXTransErrorCode::SIZE), "SIZE") == 0,
            "SIZE name changed");
    require(std::strcmp(rtengine::markesteijnXTransErrorCodeName(
                            MarkesteijnXTransErrorCode::CFA), "CFA") == 0,
            "CFA name changed");
    require(std::strcmp(rtengine::markesteijnXTransErrorCodeName(
                            MarkesteijnXTransErrorCode::ALLOCATION), "ALLOCATION") == 0,
            "ALLOCATION name changed");
    require(std::strcmp(rtengine::markesteijnXTransErrorCodeName(
                            MarkesteijnXTransErrorCode::NONFINITE), "NONFINITE") == 0,
            "NONFINITE name changed");
    require(std::strcmp(rtengine::markesteijnXTransErrorCodeName(
                            MarkesteijnXTransErrorCode::INTERNAL), "INTERNAL") == 0,
            "INTERNAL name changed");

    float input = 1.f;
    float output = 0.f;
    auto result = rtengine::demosaicMarkesteijnXTransReference(
        &input, &output, &output, &output, 1, 1, CFA);
    require(!result && result.code == MarkesteijnXTransErrorCode::SIZE,
            "tiny input did not return SIZE");
    std::vector<float> valid(32 * 32, 1000.f);
    std::vector<float> plane(32 * 32);
    int invalidCfa[6][6];
    std::memcpy(invalidCfa, CFA, sizeof(CFA));
    invalidCfa[0][0] = 0;
    result = rtengine::demosaicMarkesteijnXTransReference(
        valid.data(), plane.data(), plane.data(), plane.data(), 32, 32, invalidCfa);
    require(!result && result.code == MarkesteijnXTransErrorCode::CFA,
            "invalid CFA did not return CFA");
    valid[7] = std::numeric_limits<float>::quiet_NaN();
    result = rtengine::demosaicMarkesteijnXTransReference(
        valid.data(), plane.data(), plane.data(), plane.data(), 32, 32, CFA);
    require(!result && result.code == MarkesteijnXTransErrorCode::NONFINITE,
            "non-finite input did not return NONFINITE");
    return 0;
}

int synthetic()
{
    constexpr int width = 64;
    constexpr int height = 64;
    const std::size_t pixels = width * height;
    std::vector<float> mosaic(pixels);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const float channels[] = {
                0.15f + 0.5f * x / (width - 1),
                0.20f + 0.4f * y / (height - 1),
                0.25f + 0.3f * (x + y) / (width + height - 2)
            };
            mosaic[static_cast<std::size_t>(y) * width + x] =
                channels[CFA[y % 6][x % 6]];
        }
    }
    for (const char *method : METHODS) {
        const MethodOutput first = runMethod(method, mosaic, width, height, CFA);
        const MethodOutput second = runMethod(method, mosaic, width, height, CFA);
        require(first.values == second.values,
                std::string("repeated output differs for ") + method);
        for (float value : first.values) {
            require(std::isfinite(value),
                    std::string("non-finite output from ") + method);
        }
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                const std::size_t pixel = static_cast<std::size_t>(y) * width + x;
                const int channel = CFA[y % 6][x % 6];
                require(std::abs(first.values[static_cast<std::size_t>(channel) * pixels + pixel]
                                 - mosaic[pixel]) < 2e-6f,
                        std::string("native sample changed for ") + method);
            }
        }
    }
    return 0;
}

int runFiles(int argc, char **argv, bool pairOnly, bool customCfa)
{
    require(argc == (customCfa ? 7 : 6),
            "usage: oracle_runner run[-pair][-cfa] INPUT OUTPUT_DIR WIDTH HEIGHT [CFA36]");
    const std::string inputPath = argv[2];
    const std::string outputDirectory = argv[3];
    const int width = std::stoi(argv[4]);
    const int height = std::stoi(argv[5]);
    require(width >= 32 && height >= 32, "dimensions must be at least 32x32");
    int selectedCfa[6][6];
    std::memcpy(selectedCfa, CFA, sizeof(selectedCfa));
    if (customCfa) {
        const std::string encoded = argv[6];
        require(encoded.size() == 36, "CFA encoding must contain 36 digits");
        for (std::size_t index = 0; index < encoded.size(); ++index) {
            require(encoded[index] >= '0' && encoded[index] <= '2',
                    "CFA encoding contains an invalid color");
            selectedCfa[index / 6][index % 6] = encoded[index] - '0';
        }
    }
    const std::size_t pixels = static_cast<std::size_t>(width) * height;
    const std::vector<float> mosaic = readFloat32Le(inputPath, pixels);
    std::cout << std::setprecision(17);
    const std::size_t methodCount = pairOnly ? 2 : sizeof(METHODS) / sizeof(METHODS[0]);
    for (std::size_t methodIndex = 0; methodIndex < methodCount; ++methodIndex) {
        const char *method = METHODS[methodIndex];
        const MethodOutput output = runMethod(
            method, mosaic, width, height, selectedCfa);
        writeFloat32Le(outputDirectory + "/" + method + ".f32le", output.values);
        std::cout << method << '\t' << output.seconds << '\n';
    }
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
            return runFiles(argc, argv, false, false);
        }
        if (mode == "run-pair") {
            return runFiles(argc, argv, true, false);
        }
        if (mode == "run-pair-cfa") {
            return runFiles(argc, argv, true, true);
        }
        throw std::runtime_error("unknown mode " + mode);
    } catch (const std::exception &error) {
        std::cerr << "xtrans oracle runner: " << error.what() << '\n';
        return 1;
    }
}
