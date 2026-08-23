#include "rtengine/xtrans_markesteijn.h"
#include "rtengine/xtrans_mlri.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
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
    std::vector<unsigned char> bytes(count * sizeof(float));
    const std::size_t read = std::fread(bytes.data(), 1, bytes.size(), file);
    const int extra = std::fgetc(file);
    std::fclose(file);
    require(read == bytes.size() && extra == EOF, "incorrect input payload size");
    std::vector<float> result(count);
    for (std::size_t i = 0; i < count; ++i) {
        const std::uint32_t bits = decodeU32(bytes.data() + i * sizeof(float));
        std::memcpy(&result[i], &bits, sizeof(bits));
        require(std::isfinite(result[i]), "input contains a non-finite value");
    }
    return result;
}

void writeNormalizedRgb(
    const std::string &path,
    const std::vector<float> &red,
    const std::vector<float> &green,
    const std::vector<float> &blue)
{
    require(red.size() == green.size() && red.size() == blue.size(),
            "output plane sizes differ");
    std::vector<unsigned char> bytes(red.size() * 3 * sizeof(float));
    const std::vector<float> *planes[] = {&red, &green, &blue};
    for (std::size_t channel = 0; channel < 3; ++channel) {
        for (std::size_t i = 0; i < red.size(); ++i) {
            const float value = (*planes[channel])[i] / 65535.f;
            require(std::isfinite(value), "output contains a non-finite value");
            std::uint32_t bits = 0;
            std::memcpy(&bits, &value, sizeof(bits));
            encodeU32(bits, bytes.data() + (channel * red.size() + i) * sizeof(float));
        }
    }
    std::FILE *file = std::fopen(path.c_str(), "wb");
    require(file != nullptr, "cannot create output " + path);
    require(std::fwrite(bytes.data(), 1, bytes.size(), file) == bytes.size(),
            "cannot write output " + path);
    require(std::fclose(file) == 0, "cannot close output " + path);
}

std::vector<float> makeInput(int width, int height, int originX, int originY)
{
    std::vector<float> input(static_cast<std::size_t>(width) * height);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const float channels[] = {
                0.05f + 0.8f * x / (width - 1),
                0.12f + 0.7f * y / (height - 1),
                0.18f + 0.5f * (x + y) / (width + height - 2)
            };
            input[static_cast<std::size_t>(y) * width + x] =
                channels[CFA[(y + originY) % 6][(x + originX) % 6]] * 65535.f;
        }
    }
    return input;
}

struct Planes final {
    std::vector<float> red;
    std::vector<float> green;
    std::vector<float> blue;
};

Planes runUlri(
    const std::vector<float> &input,
    int width,
    int height,
    int originX,
    int originY,
    int slow)
{
    Planes output {
        std::vector<float>(input.size()),
        std::vector<float>(input.size()),
        std::vector<float>(input.size())
    };
    const auto result = rtengine::demosaicUlriXTransSlowReference(
        input.data(), output.red.data(), output.green.data(), output.blue.data(),
        width, height, originX, originY, slow);
    require(static_cast<bool>(result),
            std::string("ULRI failed [") +
                rtengine::mlriXTransErrorCodeName(result.code) + "]: " + result.message);
    return output;
}

int contract()
{
    constexpr int width = 42;
    constexpr int height = 40;
    const auto input = makeInput(width, height, 0, 0);
    const Planes first = runUlri(input, width, height, 0, 0, 1);
    const Planes second = runUlri(input, width, height, 0, 0, 1);
    require(first.red == second.red && first.green == second.green && first.blue == second.blue,
            "ULRI reference is nondeterministic");

    Planes existing {
        std::vector<float>(input.size()),
        std::vector<float>(input.size()),
        std::vector<float>(input.size())
    };
    const auto existingResult = rtengine::demosaicMlriXTransReference(
        input.data(), existing.red.data(), existing.green.data(), existing.blue.data(),
        width, height, 0, 0, rtengine::MlriXTransVariant::MATLAB_REFERENCE);
    require(static_cast<bool>(existingResult), "existing MATLAB-reference path failed");
    require(first.red == existing.red && first.green == existing.green && first.blue == existing.blue,
            "ULRI slow=1 differs from the frozen MATLAB-reference path");

    const Planes onePass = runUlri(input, width, height, 0, 0, 0);
    require(onePass.green != first.green || onePass.red != first.red || onePass.blue != first.blue,
            "ULRI slow=0 is indistinguishable from slow=1");
    for (const auto *plane : {&onePass.red, &onePass.green, &onePass.blue}) {
        require(std::all_of(plane->begin(), plane->end(),
                            [](float value) { return std::isfinite(value); }),
                "ULRI slow=0 produced a non-finite value");
    }

    float sample = 0.f;
    const auto low = rtengine::demosaicUlriXTransSlowReference(
        &sample, &sample, &sample, &sample, 1, 1, 0, 0, 0);
    require(!low && low.code == rtengine::MlriXTransErrorCode::SIZE,
            "tiny ULRI input did not return SIZE");
    std::vector<float> red(input.size()), green(input.size()), blue(input.size());
    const auto invalid = rtengine::demosaicUlriXTransSlowReference(
        input.data(), red.data(), green.data(), blue.data(),
        width, height, 0, 0, 9);
    require(!invalid && invalid.code == rtengine::MlriXTransErrorCode::SIZE,
            "invalid ULRI slow value did not return SIZE");

    std::set<std::string> phaseCells;
    for (int originY = 0; originY < 6; ++originY) {
        for (int originX = 0; originX < 6; ++originX) {
            std::string cell;
            for (int y = 0; y < 6; ++y) {
                for (int x = 0; x < 6; ++x) {
                    cell.push_back(static_cast<char>('0' +
                        CFA[(y + originY) % 6][(x + originX) % 6]));
                }
            }
            if (!phaseCells.insert(cell).second) {
                continue;
            }
            const auto phaseInput = makeInput(width, height, originX, originY);
            const Planes phase = runUlri(
                phaseInput, width, height, originX, originY, 1);
            require(std::all_of(phase.green.begin(), phase.green.end(),
                                [](float value) { return std::isfinite(value); }),
                    "ULRI phase reconstruction produced a non-finite value");
        }
    }
    require(phaseCells.size() == 18, "expected 18 distinct X-Trans phase cells");
    return 0;
}

int runFiles(int argc, char **argv)
{
    require(argc == 8,
            "usage: ulri_tests run INPUT OUTPUT WIDTH HEIGHT ORIGIN_X ORIGIN_Y");
    const std::string inputPath = argv[2];
    const std::string outputDirectory = argv[3];
    const int width = std::stoi(argv[4]);
    const int height = std::stoi(argv[5]);
    const int originX = std::stoi(argv[6]);
    const int originY = std::stoi(argv[7]);
    require(width >= 32 && height >= 32, "dimensions must be at least 32x32");
    const std::size_t pixels = static_cast<std::size_t>(width) * height;
    const auto normalized = readFloat32Le(inputPath, pixels);
    std::vector<float> input(pixels);
    for (std::size_t i = 0; i < pixels; ++i) {
        input[i] = normalized[i] * 65535.f;
    }

    for (int slow = 0; slow <= 3; ++slow) {
        const auto start = std::chrono::steady_clock::now();
        const Planes output = runUlri(input, width, height, originX, originY, slow);
        writeNormalizedRgb(
            outputDirectory + "/ulri-slow" + std::to_string(slow) + ".f32le",
            output.red, output.green, output.blue);
        std::cout << std::setprecision(17) << "ulri-slow" << slow << '\t'
                  << std::chrono::duration<double>(
                         std::chrono::steady_clock::now() - start).count() << '\n';
    }

    Planes corrected {
        std::vector<float>(pixels), std::vector<float>(pixels), std::vector<float>(pixels)
    };
    const auto correctedStart = std::chrono::steady_clock::now();
    const auto correctedResult = rtengine::demosaicMlriXTransReference(
        input.data(), corrected.red.data(), corrected.green.data(), corrected.blue.data(),
        width, height, originX, originY,
        rtengine::MlriXTransVariant::CORRECTED_BLUE_DIAGONAL_GUIDES_FINAL_ONLY);
    require(static_cast<bool>(correctedResult), "corrected-final MLRI failed");
    writeNormalizedRgb(
        outputDirectory + "/corrected-final.f32le",
        corrected.red, corrected.green, corrected.blue);
    std::cout << std::setprecision(17) << "corrected-final\t"
              << std::chrono::duration<double>(
                     std::chrono::steady_clock::now() - correctedStart).count() << '\n';

    int cfa[6][6];
    for (int y = 0; y < 6; ++y) {
        for (int x = 0; x < 6; ++x) {
            cfa[y][x] = CFA[(y + originY) % 6][(x + originX) % 6];
        }
    }
    Planes mark {
        std::vector<float>(pixels), std::vector<float>(pixels), std::vector<float>(pixels)
    };
    const auto markStart = std::chrono::steady_clock::now();
    const auto markResult = rtengine::demosaicMarkesteijnXTransReference(
        input.data(), mark.red.data(), mark.green.data(), mark.blue.data(),
        width, height, cfa);
    require(static_cast<bool>(markResult), "Markesteijn failed");
    writeNormalizedRgb(
        outputDirectory + "/markesteijn.f32le", mark.red, mark.green, mark.blue);
    std::cout << std::setprecision(17) << "markesteijn\t"
              << std::chrono::duration<double>(
                     std::chrono::steady_clock::now() - markStart).count() << '\n';
    return 0;
}

} // namespace

int main(int argc, char **argv)
{
    try {
        require(argc >= 2, "missing command");
        const std::string command = argv[1];
        if (command == "contract") {
            return contract();
        }
        if (command == "run") {
            return runFiles(argc, argv);
        }
        throw std::runtime_error("unknown command: " + command);
    } catch (const std::exception &error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
