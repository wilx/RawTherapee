#include "rtengine/xtrans_mlri.h"
#include "rtengine/xtrans_cfa.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <set>
#include <vector>

namespace
{

struct Case final {
    const char *name;
    int width;
    int height;
    int originX;
    int originY;
};

const std::array<Case, 5> CASES {{
    {"flat", 48, 48, 0, 0},
    {"gradient", 54, 48, 0, 0},
    {"impulse", 48, 48, 0, 0},
    {"saturated", 48, 48, 0, 0},
    {"boundary", 37, 35, 0, 0}
}};

void require(bool condition, const std::string &message)
{
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::vector<float> readFloats(const std::string &path, std::size_t count)
{
    std::FILE *file = std::fopen(path.c_str(), "rb");
    require(file != nullptr, "cannot open " + path);
    std::vector<float> result(count);
    const std::size_t read = std::fread(result.data(), sizeof(float), count, file);
    const int extra = std::fgetc(file);
    std::fclose(file);
    require(read == count && extra == EOF, "incorrect float payload size in " + path);
    return result;
}

void writeFloats(const std::string &path, const std::vector<float> &values)
{
    std::FILE *file = std::fopen(path.c_str(), "wb");
    require(file != nullptr, "cannot create " + path);
    require(std::fwrite(values.data(), sizeof(float), values.size(), file) == values.size(),
            "cannot write " + path);
    require(std::fclose(file) == 0, "cannot close " + path);
}

std::vector<float> run(
    const std::vector<float> &input,
    int width,
    int height,
    int originX,
    int originY,
    rtengine::MlriXTransVariant variant =
        rtengine::MlriXTransVariant::MATLAB_REFERENCE)
{
    const std::size_t pixels = static_cast<std::size_t>(width) * height;
    require(input.size() == pixels, "input size differs from dimensions");
    std::vector<float> output(pixels * 3);
    const rtengine::MlriXTransRunResult result = rtengine::demosaicMlriXTransReference(
        input.data(), output.data(), output.data() + pixels, output.data() + pixels * 2,
        width, height, originX, originY, variant);
    require(static_cast<bool>(result),
            std::string("MLRI failed [") + rtengine::mlriXTransErrorCodeName(result.code) +
            "]: " + result.message);
    for (float value : output) {
        require(std::isfinite(value), "MLRI returned a non-finite value");
        require(value >= 0.f && value <= 65535.f, "MLRI returned an out-of-range value");
    }
    return output;
}

int contract()
{
    require(std::strcmp(rtengine::MLRI_XTRANS_TWO_PASS_METHOD, "mlri-xtrans-2pass") == 0,
            "faithful method identifier changed");
    require(std::strcmp(rtengine::MLRI_XTRANS_TWO_PASS_CORRECTED_METHOD,
                        "mlri-xtrans-2pass-corrected") == 0,
            "corrected method identifier changed");
    require(std::strcmp(rtengine::MLRI_XTRANS_TWO_PASS_CORRECTED_FINAL_ONLY_METHOD,
                        "mlri-xtrans-2pass-corrected-final-only") == 0,
            "corrected final-only method identifier changed");
    require(std::strcmp(rtengine::MLRI_XTRANS_PAPER_CORE_2014_METHOD,
                        "mlri-xtrans-paper-core-2014") == 0,
            "2014 paper-core method identifier changed");
    require(std::strcmp(rtengine::MLRI_XTRANS_PAPER_CORE_2016_METHOD,
                        "mlri-xtrans-paper-core-2016") == 0,
            "2016 paper-core method identifier changed");
    require(std::strcmp(rtengine::mlriXTransErrorCodeName(rtengine::MlriXTransErrorCode::NONE), "NONE") == 0,
            "NONE error name changed");
    require(std::strcmp(rtengine::mlriXTransErrorCodeName(rtengine::MlriXTransErrorCode::SIZE), "SIZE") == 0,
            "SIZE error name changed");
    require(std::strcmp(rtengine::mlriXTransErrorCodeName(rtengine::MlriXTransErrorCode::CFA), "CFA") == 0,
            "CFA error name changed");
    require(std::strcmp(
                rtengine::mlriXTransErrorCodeName(rtengine::MlriXTransErrorCode::ALLOCATION),
                "ALLOCATION") == 0,
            "ALLOCATION error name changed");
    require(std::strcmp(rtengine::mlriXTransErrorCodeName(rtengine::MlriXTransErrorCode::NONFINITE), "NONFINITE") == 0,
            "NONFINITE error name changed");
    require(std::strcmp(rtengine::mlriXTransErrorCodeName(rtengine::MlriXTransErrorCode::INTERNAL), "INTERNAL") == 0,
            "INTERNAL error name changed");

    std::vector<float> tooSmall(31 * 32, 1000.f);
    std::vector<float> channel(31 * 32);
    rtengine::MlriXTransRunResult result = rtengine::demosaicMlriXTransReference(
        tooSmall.data(), channel.data(), channel.data(), channel.data(), 31, 32);
    require(!result && result.code == rtengine::MlriXTransErrorCode::SIZE,
            "sub-32 image did not return SIZE");

    std::vector<float> valid(32 * 32, 20000.f);
    valid[17] = std::numeric_limits<float>::quiet_NaN();
    channel.assign(32 * 32, 0.f);
    result = rtengine::demosaicMlriXTransReference(
        valid.data(), channel.data(), channel.data(), channel.data(), 32, 32);
    require(!result && result.code == rtengine::MlriXTransErrorCode::NONFINITE,
            "non-finite input did not return NONFINITE");

    std::fill(valid.begin(), valid.end(), 20000.f);
    const std::vector<float> first = run(valid, 32, 32, 0, 0);
    const std::vector<float> second = run(valid, 32, 32, 0, 0);
    require(first == second, "repeated MLRI reference executions differ");

    constexpr int canonical[6][6] = {
        {1,0,1,1,2,1}, {2,1,2,0,1,0}, {1,0,1,1,2,1},
        {1,2,1,1,0,1}, {0,1,0,2,1,2}, {1,2,1,1,0,1}
    };
    constexpr int matrices[8][4] = {
        {1,0,0,1}, {0,-1,1,0}, {-1,0,0,-1}, {0,1,-1,0},
        {-1,0,0,1}, {1,0,0,-1}, {0,1,1,0}, {0,-1,-1,0}
    };
    std::set<std::array<int, 36>> variants;
    for (const auto &matrix : matrices) {
        for (int oy = 0; oy < 6; ++oy) {
            for (int ox = 0; ox < 6; ++ox) {
                std::array<int, 36> actual {{}};
                for (int y = 0; y < 6; ++y) {
                    for (int x = 0; x < 6; ++x) {
                        actual[static_cast<std::size_t>(y * 6 + x)] = canonical[
                            rtengine::positiveModulo(matrix[2] * x + matrix[3] * y + oy, 6)
                        ][rtengine::positiveModulo(matrix[0] * x + matrix[1] * y + ox, 6)];
                    }
                }
                variants.insert(actual);
            }
        }
    }
    require(variants.size() == 18, "method-specific X-Trans cell does not have 18 unique mappings");
    for (const auto &variant : variants) {
        int actual[6][6];
        for (int y = 0; y < 6; ++y) {
            for (int x = 0; x < 6; ++x) {
                actual[y][x] = variant[static_cast<std::size_t>(y * 6 + x)];
            }
        }
        rtengine::XTransCfaTransform transform;
        require(rtengine::findCanonicalXTransTransform(actual, canonical, transform),
                "one of the 18 method-specific CFA mappings was rejected");
        const rtengine::XTransCfaView view(transform, 43, 37, canonical);
        require(view.valid(), "accepted method-specific CFA produced an invalid view");
        for (int y = 0; y < 37; ++y) {
            for (int x = 0; x < 43; ++x) {
                int u = 0;
                int v = 0;
                int roundTripX = 0;
                int roundTripY = 0;
                view.actualToCanonical(x, y, u, v);
                view.canonicalToActual(u, v, roundTripX, roundTripY);
                require(roundTripX == x && roundTripY == y,
                        "method-specific CFA coordinate round trip failed");
                require(view.colorAtCanonical(u, v) == actual[y % 6][x % 6],
                        "method-specific CFA mapping changed color meaning");
            }
        }
    }
    return 0;
}

int correctedVariant()
{
    constexpr int width = 48;
    constexpr int height = 48;
    const std::size_t pixels = static_cast<std::size_t>(width) * height;
    std::vector<float> input(pixels);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            input[static_cast<std::size_t>(y) * width + x] = static_cast<float>(
                1000 + ((x * 7919 + y * 104729 + x * y * 31) % 62000));
        }
    }
    const std::vector<float> faithful = run(input, width, height, 0, 0);
    const std::vector<float> corrected = run(
        input, width, height, 0, 0,
        rtengine::MlriXTransVariant::CORRECTED_BLUE_DIAGONAL_GUIDES);
    const std::vector<float> repeated = run(
        input, width, height, 0, 0,
        rtengine::MlriXTransVariant::CORRECTED_BLUE_DIAGONAL_GUIDES);
    require(corrected == repeated, "corrected MLRI execution is not deterministic");

    std::array<double, 3> maxima {{0.0, 0.0, 0.0}};
    for (std::size_t channel = 0; channel < 3; ++channel) {
        for (std::size_t pixel = 0; pixel < pixels; ++pixel) {
            maxima[channel] = std::max(
                maxima[channel],
                std::fabs(static_cast<double>(
                    corrected[channel * pixels + pixel] - faithful[channel * pixels + pixel])));
        }
        require(maxima[channel] > 0.0,
                "blue-guide correction did not propagate into every output channel");
    }
    std::cout << "corrected_vs_faithful_max_abs="
              << maxima[0] << ',' << maxima[1] << ',' << maxima[2] << '\n';
    return 0;
}

int correctedFinalOnlyVariant()
{
    constexpr int width = 48;
    constexpr int height = 48;
    constexpr int cfa[6][6] = {
        {1,0,1,1,2,1}, {2,1,2,0,1,0}, {1,0,1,1,2,1},
        {1,2,1,1,0,1}, {0,1,0,2,1,2}, {1,2,1,1,0,1}
    };
    const std::size_t pixels = static_cast<std::size_t>(width) * height;
    std::vector<float> input(pixels);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            input[static_cast<std::size_t>(y) * width + x] = static_cast<float>(
                300 + ((x * 3571 + y * 15401 + x * y * 101) % 12000));
        }
    }

    const std::vector<float> blended = run(
        input, width, height, 0, 0,
        rtengine::MlriXTransVariant::CORRECTED_BLUE_DIAGONAL_GUIDES);
    const std::vector<float> finalOnly = run(
        input, width, height, 0, 0,
        rtengine::MlriXTransVariant::CORRECTED_BLUE_DIAGONAL_GUIDES_FINAL_ONLY);
    const std::vector<float> repeated = run(
        input, width, height, 0, 0,
        rtengine::MlriXTransVariant::CORRECTED_BLUE_DIAGONAL_GUIDES_FINAL_ONLY);
    require(finalOnly == repeated, "corrected final-only MLRI is not deterministic");

    double redMaximum = 0.0;
    double blueMaximum = 0.0;
    for (std::size_t pixel = 0; pixel < pixels; ++pixel) {
        require(finalOnly[pixels + pixel] == blended[pixels + pixel],
                "final-only selection changed the corrected two-pass green plane");
        redMaximum = std::max(redMaximum, std::fabs(static_cast<double>(
            finalOnly[pixel] - blended[pixel])));
        blueMaximum = std::max(blueMaximum, std::fabs(static_cast<double>(
            finalOnly[pixels * 2 + pixel] - blended[pixels * 2 + pixel])));
    }
    require(redMaximum > 0.0 && blueMaximum > 0.0,
            "final-only selection did not change both reconstructed chroma planes");

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const std::size_t pixel = static_cast<std::size_t>(y) * width + x;
            const int observedChannel = cfa[y % 6][x % 6];
            require(std::fabs(static_cast<double>(
                        finalOnly[static_cast<std::size_t>(observedChannel) * pixels + pixel]
                        - input[pixel])) <= 0.02,
                    "final-only reconstruction changed an observed CFA sample");
        }
    }

    std::cout << "final_only_vs_blended_max_abs="
              << redMaximum << ",0," << blueMaximum << '\n';
    return 0;
}

int paperCoreVariants()
{
    constexpr int width = 48;
    constexpr int height = 48;
    const std::size_t pixels = static_cast<std::size_t>(width) * height;
    std::vector<float> input(pixels);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            input[static_cast<std::size_t>(y) * width + x] = static_cast<float>(
                500 + ((x * 3571 + y * 15401 + x * y * 101) % 64000));
        }
    }

    const std::vector<float> paper2014 = run(
        input, width, height, 0, 0,
        rtengine::MlriXTransVariant::PAPER_CORE_2014);
    const std::vector<float> paper2014Repeated = run(
        input, width, height, 0, 0,
        rtengine::MlriXTransVariant::PAPER_CORE_2014);
    const std::vector<float> paper2016 = run(
        input, width, height, 0, 0,
        rtengine::MlriXTransVariant::PAPER_CORE_2016);
    const std::vector<float> paper2016Repeated = run(
        input, width, height, 0, 0,
        rtengine::MlriXTransVariant::PAPER_CORE_2016);
    const std::vector<float> correctedTwoPass = run(
        input, width, height, 0, 0,
        rtengine::MlriXTransVariant::CORRECTED_BLUE_DIAGONAL_GUIDES);
    require(paper2014 == paper2014Repeated,
            "2014 paper-core execution is not deterministic");
    require(paper2016 == paper2016Repeated,
            "2016 paper-core execution is not deterministic");

    std::array<double, 3> averagingMaxima {{0.0, 0.0, 0.0}};
    std::array<double, 3> heuristicMaxima {{0.0, 0.0, 0.0}};
    for (std::size_t channel = 0; channel < 3; ++channel) {
        for (std::size_t pixel = 0; pixel < pixels; ++pixel) {
            const std::size_t index = channel * pixels + pixel;
            averagingMaxima[channel] = std::max(
                averagingMaxima[channel],
                std::fabs(static_cast<double>(paper2014[index] - paper2016[index])));
            heuristicMaxima[channel] = std::max(
                heuristicMaxima[channel],
                std::fabs(static_cast<double>(paper2016[index] - correctedTwoPass[index])));
        }
        require(averagingMaxima[channel] > 0.0,
                "uniform and residual-weighted coefficient averages are indistinguishable");
        require(heuristicMaxima[channel] > 0.0,
                "paper core is indistinguishable from the two-pass blended method");
    }
    std::cout << "paper_2014_vs_2016_max_abs="
              << averagingMaxima[0] << ',' << averagingMaxima[1] << ','
              << averagingMaxima[2] << '\n';
    std::cout << "paper_2016_vs_corrected_two_pass_max_abs="
              << heuristicMaxima[0] << ',' << heuristicMaxima[1] << ','
              << heuristicMaxima[2] << '\n';
    return 0;
}

int golden()
{
    for (const Case &test : CASES) {
        const std::string prefix = std::string(RT_MLRI_GOLDEN_DIR) + '/' + test.name;
        const std::size_t pixels = static_cast<std::size_t>(test.width) * test.height;
        const std::vector<float> input = readFloats(prefix + "-input.f32", pixels);
        const std::vector<float> expected = readFloats(prefix + "-output.f32", pixels * 3);
        const std::vector<float> actual = run(
            input, test.width, test.height, test.originX, test.originY);
        double squared = 0.0;
        double maximum = 0.0;
        double interiorSquared = 0.0;
        double interiorMaximum = 0.0;
        std::size_t interiorCount = 0;
        for (std::size_t i = 0; i < actual.size(); ++i) {
            const double difference = std::fabs(static_cast<double>(actual[i]) - expected[i]);
            maximum = std::max(maximum, difference);
            squared += difference * difference;
            const std::size_t pixel = i % pixels;
            const int x = static_cast<int>(pixel % test.width);
            const int y = static_cast<int>(pixel / test.width);
            if (x >= 12 && x < test.width - 12 && y >= 12 && y < test.height - 12) {
                interiorMaximum = std::max(interiorMaximum, difference);
                interiorSquared += difference * difference;
                ++interiorCount;
            }
        }
        const double rms = std::sqrt(squared / actual.size());
        const double interiorRms = std::sqrt(interiorSquared / interiorCount);
        // Octave's conv2 and the portable C++ correlation evaluate large
        // floating-point reductions in different orders.  The inverse-cost
        // weights make pathological saturated/boundary cases sensitive to
        // that rounding.  Keep explicit normalized bounds for both the full
        // zero-extended image and the region outside the active boundary.
        require(maximum / 65535.0 <= 0.031 && rms / 65535.0 <= 0.002,
                std::string(test.name) + " differs from Octave: max=" +
                std::to_string(maximum) + " rms=" + std::to_string(rms));
        require(interiorMaximum / 65535.0 <= 0.012 && interiorRms / 65535.0 <= 0.0011,
                std::string(test.name) + " interior differs from Octave: max=" +
                std::to_string(interiorMaximum) + " rms=" + std::to_string(interiorRms));
        std::cout << test.name << ": max_abs=" << maximum << " rms=" << rms
                  << " interior_max_abs=" << interiorMaximum
                  << " interior_rms=" << interiorRms << '\n';
    }
    return 0;
}

int tiledSeam()
{
    constexpr int width = 390;
    constexpr int height = 36;
    constexpr int cfa[6][6] = {
        {1,0,1,1,2,1}, {2,1,2,0,1,0}, {1,0,1,1,2,1},
        {1,2,1,1,0,1}, {0,1,0,2,1,2}, {1,2,1,1,0,1}
    };
    const std::size_t pixels = static_cast<std::size_t>(width) * height;
    std::vector<float> input(pixels);
    array2D<float> raw(width, height);
    array2D<float> tiledR(width, height);
    array2D<float> tiledG(width, height);
    array2D<float> tiledB(width, height);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const float value = static_cast<float>(
                1000 + ((x * 7919 + y * 104729 + x * y * 31) % 62000));
            input[static_cast<std::size_t>(y) * width + x] = value;
            raw[y][x] = value;
        }
    }
    const std::array<rtengine::MlriXTransVariant, 5> algorithms {{
        rtengine::MlriXTransVariant::MATLAB_REFERENCE,
        rtengine::MlriXTransVariant::CORRECTED_BLUE_DIAGONAL_GUIDES,
        rtengine::MlriXTransVariant::CORRECTED_BLUE_DIAGONAL_GUIDES_FINAL_ONLY,
        rtengine::MlriXTransVariant::PAPER_CORE_2014,
        rtengine::MlriXTransVariant::PAPER_CORE_2016
    }};
    for (const rtengine::MlriXTransVariant algorithm : algorithms) {
        const std::vector<float> untiled = run(input, width, height, 0, 0, algorithm);
        const rtengine::MlriXTransRunResult result = rtengine::demosaicMlriXTrans(
            raw, tiledR, tiledG, tiledB, width, height, cfa, algorithm);
        require(static_cast<bool>(result),
                std::string("tiled MLRI failed [") + rtengine::mlriXTransErrorCodeName(result.code) +
                "]: " + result.message);
        require(result.tileCount == 2 && result.coreSize == 384 && result.halo == 228,
                "production MLRI tile geometry changed");
        double maximum = 0.0;
        double seamMaximum = 0.0;
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                const std::size_t pixel = static_cast<std::size_t>(y) * width + x;
                const std::array<float,3> actual {{tiledR[y][x], tiledG[y][x], tiledB[y][x]}};
                for (int channelIndex = 0; channelIndex < 3; ++channelIndex) {
                    const double difference = std::fabs(
                        static_cast<double>(actual[static_cast<std::size_t>(channelIndex)]) -
                        untiled[static_cast<std::size_t>(channelIndex) * pixels + pixel]);
                    maximum = std::max(maximum, difference);
                    if (x >= 380 && x < 390) {
                        seamMaximum = std::max(seamMaximum, difference);
                    }
                }
            }
        }
        require(maximum <= 0.02 && seamMaximum <= 0.02,
                "tiled MLRI differs from untiled reference: max=" + std::to_string(maximum) +
                " seam_max=" + std::to_string(seamMaximum));
    }
    return 0;
}

int orientationParity()
{
    constexpr int actualWidth = 43;
    constexpr int actualHeight = 37;
    constexpr int canonical[6][6] = {
        {1,0,1,1,2,1}, {2,1,2,0,1,0}, {1,0,1,1,2,1},
        {1,2,1,1,0,1}, {0,1,0,2,1,2}, {1,2,1,1,0,1}
    };
    constexpr int matrices[8][4] = {
        {1,0,0,1}, {0,-1,1,0}, {-1,0,0,-1}, {0,1,-1,0},
        {-1,0,0,1}, {1,0,0,-1}, {0,1,1,0}, {0,-1,-1,0}
    };

    std::set<std::array<int, 36>> variants;
    for (const auto &matrix : matrices) {
        for (int oy = 0; oy < 6; ++oy) {
            for (int ox = 0; ox < 6; ++ox) {
                std::array<int, 36> actual {{}};
                for (int y = 0; y < 6; ++y) {
                    for (int x = 0; x < 6; ++x) {
                        actual[static_cast<std::size_t>(y * 6 + x)] = canonical[
                            rtengine::positiveModulo(matrix[2] * x + matrix[3] * y + oy, 6)
                        ][rtengine::positiveModulo(matrix[0] * x + matrix[1] * y + ox, 6)];
                    }
                }
                variants.insert(actual);
            }
        }
    }
    require(variants.size() == 18, "incorrect orientation fixture count");

    for (const auto &variant : variants) {
        int actualCfa[6][6];
        for (int y = 0; y < 6; ++y) {
            for (int x = 0; x < 6; ++x) {
                actualCfa[y][x] = variant[static_cast<std::size_t>(y * 6 + x)];
            }
        }
        rtengine::XTransCfaTransform transform;
        require(rtengine::findCanonicalXTransTransform(actualCfa, canonical, transform),
                "orientation fixture was not canonicalized");
        const rtengine::XTransCfaView view(
            transform, actualWidth, actualHeight, canonical);
        require(view.valid(), "orientation fixture produced an invalid view");

        const std::size_t canonicalPixels =
            static_cast<std::size_t>(view.width()) * view.height();
        std::vector<float> canonicalInput(canonicalPixels);
        for (int v = 0; v < view.height(); ++v) {
            for (int u = 0; u < view.width(); ++u) {
                const int color = view.colorAtCanonical(u, v);
                const unsigned pattern = static_cast<unsigned>(
                    (u + view.minimumX() + 96) * 7919 +
                    (v + view.minimumY() + 96) * 104729 + u * v * 31);
                canonicalInput[static_cast<std::size_t>(v) * view.width() + u] =
                    static_cast<float>(4000 + color * 16000 + pattern % 11000);
            }
        }
        const std::vector<float> expected = run(
            canonicalInput, view.width(), view.height(),
            view.minimumX(), view.minimumY());

        array2D<float> raw(actualWidth, actualHeight);
        array2D<float> red(actualWidth, actualHeight);
        array2D<float> green(actualWidth, actualHeight);
        array2D<float> blue(actualWidth, actualHeight);
        for (int y = 0; y < actualHeight; ++y) {
            for (int x = 0; x < actualWidth; ++x) {
                int u = 0;
                int v = 0;
                view.actualToCanonical(x, y, u, v);
                raw[y][x] = canonicalInput[static_cast<std::size_t>(v) * view.width() + u];
            }
        }
        const rtengine::MlriXTransRunResult result = rtengine::demosaicMlriXTrans(
            raw, red, green, blue, actualWidth, actualHeight, actualCfa);
        require(static_cast<bool>(result), "orientation demosaic failed");

        double maximum = 0.0;
        for (int y = 0; y < actualHeight; ++y) {
            for (int x = 0; x < actualWidth; ++x) {
                int u = 0;
                int v = 0;
                view.actualToCanonical(x, y, u, v);
                const std::size_t pixel = static_cast<std::size_t>(v) * view.width() + u;
                const std::array<float, 3> actual {{red[y][x], green[y][x], blue[y][x]}};
                for (int channel = 0; channel < 3; ++channel) {
                    maximum = std::max(maximum, std::fabs(
                        static_cast<double>(actual[static_cast<std::size_t>(channel)]) -
                        expected[static_cast<std::size_t>(channel) * canonicalPixels + pixel]));
                }
            }
        }
        require(maximum <= 0.02,
                "orientation output differs from canonical reference: max=" +
                std::to_string(maximum));
    }

    int invalidCfa[6][6] = {};
    array2D<float> raw(actualWidth, actualHeight);
    array2D<float> red(actualWidth, actualHeight);
    array2D<float> green(actualWidth, actualHeight);
    array2D<float> blue(actualWidth, actualHeight);
    for (int y = 0; y < actualHeight; ++y) {
        for (int x = 0; x < actualWidth; ++x) {
            raw[y][x] = 10000.f;
            red[y][x] = green[y][x] = blue[y][x] = -1234.f;
        }
    }
    const rtengine::MlriXTransRunResult failure = rtengine::demosaicMlriXTrans(
        raw, red, green, blue, actualWidth, actualHeight, invalidCfa);
    require(!failure && failure.code == rtengine::MlriXTransErrorCode::CFA,
            "unsupported CFA did not return CFA");
    for (int y = 0; y < actualHeight; ++y) {
        for (int x = 0; x < actualWidth; ++x) {
            require(red[y][x] == -1234.f && green[y][x] == -1234.f && blue[y][x] == -1234.f,
                    "unsupported CFA exposed partial output");
        }
    }
    return 0;
}

int dump(int argc, char **argv)
{
    require(argc == 8, "dump requires WIDTH HEIGHT ORIGIN_X ORIGIN_Y INPUT OUTPUT");
    const int width = std::stoi(argv[2]);
    const int height = std::stoi(argv[3]);
    const int originX = std::stoi(argv[4]);
    const int originY = std::stoi(argv[5]);
    const std::vector<float> input = readFloats(
        argv[6], static_cast<std::size_t>(width) * height);
    writeFloats(argv[7], run(input, width, height, originX, originY));
    return 0;
}

} // namespace

int main(int argc, char **argv)
{
    try {
        require(argc >= 2, "usage: rawtherapee-xtrans-mlri-tests MODE");
        const std::string mode = argv[1];
        if (mode == "contract") {
            require(argc == 2, "contract takes no arguments");
            return contract();
        }
        if (mode == "golden") {
            require(argc == 2, "golden takes no arguments");
            return golden();
        }
        if (mode == "corrected") {
            require(argc == 2, "corrected takes no arguments");
            return correctedVariant();
        }
        if (mode == "corrected-final-only") {
            require(argc == 2, "corrected-final-only takes no arguments");
            return correctedFinalOnlyVariant();
        }
        if (mode == "paper-core") {
            require(argc == 2, "paper-core takes no arguments");
            return paperCoreVariants();
        }
        if (mode == "tiled-seam") {
            require(argc == 2, "tiled-seam takes no arguments");
            return tiledSeam();
        }
        if (mode == "orientation-parity") {
            require(argc == 2, "orientation-parity takes no arguments");
            return orientationParity();
        }
        if (mode == "dump") {
            return dump(argc, argv);
        }
        throw std::runtime_error("unknown mode: " + mode);
    } catch (const std::exception &error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
