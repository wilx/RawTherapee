#include "rtengine/xtrans_cfa.h"
#include "rtengine/xtrans_triangulation.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <set>
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

struct Output final {
    int width;
    int height;
    std::vector<float> values;

    float at(int color, int x, int y) const
    {
        const std::size_t pixels = static_cast<std::size_t>(width) * height;
        return values[static_cast<std::size_t>(color) * pixels +
                      static_cast<std::size_t>(y) * width + x];
    }
};

void require(bool condition, const std::string &message)
{
    if (!condition) {
        throw std::runtime_error(message);
    }
}

Output run(
    const std::vector<float> &mosaic,
    int width,
    int height,
    const int cfa[6][6],
    rtengine::TriangulatedXTransVariant variant)
{
    const std::size_t pixels = static_cast<std::size_t>(width) * height;
    require(mosaic.size() == pixels, "mosaic size mismatch");
    Output output{width, height, std::vector<float>(pixels * 3)};
    const rtengine::TriangulatedXTransRunResult result =
        rtengine::demosaicTriangulatedXTransReference(
            mosaic.data(),
            output.values.data(),
            output.values.data() + pixels,
            output.values.data() + pixels * 2,
            width,
            height,
            cfa,
            variant);
    require(static_cast<bool>(result),
            std::string("triangulation failed [") +
            rtengine::triangulatedXTransErrorCodeName(result.code) + "]: " +
            result.message);
    for (float value : output.values) {
        require(std::isfinite(value), "triangulation produced a non-finite value");
    }
    return output;
}

template<typename Function>
std::vector<float> makeMosaic(int width, int height, const int cfa[6][6], Function value)
{
    std::vector<float> mosaic(static_cast<std::size_t>(width) * height);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const int color = cfa[y % 6][x % 6];
            mosaic[static_cast<std::size_t>(y) * width + x] = value(color, x, y);
        }
    }
    return mosaic;
}

void requireSamplePreservation(
    const Output &output,
    const std::vector<float> &mosaic,
    const int cfa[6][6])
{
    for (int y = 0; y < output.height; ++y) {
        for (int x = 0; x < output.width; ++x) {
            const int color = cfa[y % 6][x % 6];
            const float measured = mosaic[static_cast<std::size_t>(y) * output.width + x];
            require(output.at(color, x, y) == measured,
                    "a native CFA sample was not preserved bit-exactly");
        }
    }
}

double falseColorRms(const Output &output)
{
    double squared = 0.0;
    std::size_t count = 0;
    for (int y = 0; y < output.height; ++y) {
        for (int x = 0; x < output.width; ++x) {
            const double r = output.at(0, x, y);
            const double g = output.at(1, x, y);
            const double b = output.at(2, x, y);
            squared += (r - g) * (r - g) + (b - g) * (b - g);
            count += 2;
        }
    }
    return std::sqrt(squared / count) / 65535.0;
}

double reconstructionRms(
    const Output &output,
    const std::array<std::vector<float>, 3> &truth)
{
    double squared = 0.0;
    std::size_t count = 0;
    for (int color = 0; color < 3; ++color) {
        for (int y = 0; y < output.height; ++y) {
            for (int x = 0; x < output.width; ++x) {
                const std::size_t pixel = static_cast<std::size_t>(y) * output.width + x;
                const double difference = output.at(color, x, y) - truth[color][pixel];
                squared += difference * difference;
                ++count;
            }
        }
    }
    return std::sqrt(squared / count) / 65535.0;
}

int contract()
{
    using rtengine::TriangulatedXTransErrorCode;
    require(std::strcmp(rtengine::XTRANS_TRIANGULATED_RGB_METHOD,
                        "xtrans-triangulated-rgb") == 0,
            "independent method identifier changed");
    require(std::strcmp(rtengine::XTRANS_TRIANGULATED_CHROMA_METHOD,
                        "xtrans-triangulated-chroma") == 0,
            "chroma method identifier changed");
    require(std::strcmp(rtengine::triangulatedXTransErrorCodeName(
                            TriangulatedXTransErrorCode::NONE), "NONE") == 0,
            "NONE error name changed");
    require(std::strcmp(rtengine::triangulatedXTransErrorCodeName(
                            TriangulatedXTransErrorCode::SIZE), "SIZE") == 0,
            "SIZE error name changed");
    require(std::strcmp(rtengine::triangulatedXTransErrorCodeName(
                            TriangulatedXTransErrorCode::CFA), "CFA") == 0,
            "CFA error name changed");
    require(std::strcmp(rtengine::triangulatedXTransErrorCodeName(
                            TriangulatedXTransErrorCode::NONFINITE), "NONFINITE") == 0,
            "NONFINITE error name changed");
    require(std::strcmp(rtengine::triangulatedXTransErrorCodeName(
                            TriangulatedXTransErrorCode::INTERNAL), "INTERNAL") == 0,
            "INTERNAL error name changed");

    float input = 1.f;
    float r = 0.f;
    float g = 0.f;
    float b = 0.f;
    auto result = rtengine::demosaicTriangulatedXTransReference(
        &input, &r, &g, &b, 1, 1, CFA,
        rtengine::TriangulatedXTransVariant::INDEPENDENT_RGB);
    require(!result && result.code == TriangulatedXTransErrorCode::SIZE,
            "single-color tiny image did not fail with SIZE");

    int invalidCfa[6][6];
    std::memcpy(invalidCfa, CFA, sizeof(CFA));
    invalidCfa[0][0] = 0;
    std::vector<float> valid(12 * 12, 1000.f);
    std::vector<float> plane(12 * 12);
    result = rtengine::demosaicTriangulatedXTransReference(
        valid.data(), plane.data(), plane.data(), plane.data(), 12, 12, invalidCfa,
        rtengine::TriangulatedXTransVariant::INDEPENDENT_RGB);
    require(!result && result.code == TriangulatedXTransErrorCode::CFA,
            "unsupported CFA did not fail with CFA");

    valid[17] = std::numeric_limits<float>::quiet_NaN();
    result = rtengine::demosaicTriangulatedXTransReference(
        valid.data(), plane.data(), plane.data(), plane.data(), 12, 12, CFA,
        rtengine::TriangulatedXTransVariant::INDEPENDENT_RGB);
    require(!result && result.code == TriangulatedXTransErrorCode::NONFINITE,
            "non-finite mosaic did not fail with NONFINITE");
    return 0;
}

int synthetic()
{
    constexpr int width = 48;
    constexpr int height = 42;
    const std::array<rtengine::TriangulatedXTransVariant, 2> variants {{
        rtengine::TriangulatedXTransVariant::INDEPENDENT_RGB,
        rtengine::TriangulatedXTransVariant::GREEN_CHROMA_DIFFERENCE
    }};

    for (const auto variant : variants) {
        const std::vector<float> constant = makeMosaic(
            width, height, CFA, [](int, int, int) { return 23456.25f; });
        const Output constantOutput = run(constant, width, height, CFA, variant);
        requireSamplePreservation(constantOutput, constant, CFA);
        for (float value : constantOutput.values) {
            require(value == 23456.25f, "constant field was not reproduced exactly");
        }

        const auto planeValue = [](int color, int x, int y) {
            static const float bases[3] = {8000.f, 18000.f, 28000.f};
            static const float dx[3] = {73.f, -41.f, 29.f};
            static const float dy[3] = {-19.f, 61.f, 47.f};
            return bases[color] + dx[color] * x + dy[color] * y;
        };
        const std::vector<float> linear = makeMosaic(width, height, CFA, planeValue);
        const Output linearOutput = run(linear, width, height, CFA, variant);
        requireSamplePreservation(linearOutput, linear, CFA);
        double maximum = 0.0;
        for (int color = 0; color < 3; ++color) {
            for (int y = 3; y < height - 3; ++y) {
                for (int x = 3; x < width - 3; ++x) {
                    maximum = std::max(maximum, std::fabs(static_cast<double>(
                        linearOutput.at(color, x, y) - planeValue(color, x, y))));
                }
            }
        }
        require(maximum <= .01, "linear plane was not reproduced in the interior");
        std::cout << "linear_max_abs=" << maximum << '\n';
    }

    const std::vector<float> arbitrary = makeMosaic(
        17, 13, CFA, [](int color, int x, int y) {
            return static_cast<float>(500 + color * 7000 + x * 313 + y * 911);
        });
    for (const auto variant : variants) {
        const Output bordered = run(arbitrary, 17, 13, CFA, variant);
        requireSamplePreservation(bordered, arbitrary, CFA);
    }
    return 0;
}

std::set<std::array<int, 36>> cfaVariants()
{
    constexpr int matrices[8][4] = {
        {1, 0, 0, 1}, {0, -1, 1, 0}, {-1, 0, 0, -1}, {0, 1, -1, 0},
        {-1, 0, 0, 1}, {1, 0, 0, -1}, {0, 1, 1, 0}, {0, -1, -1, 0}
    };
    std::set<std::array<int, 36>> variants;
    for (const auto &matrix : matrices) {
        for (int oy = 0; oy < 6; ++oy) {
            for (int ox = 0; ox < 6; ++ox) {
                std::array<int, 36> actual {{}};
                for (int y = 0; y < 6; ++y) {
                    for (int x = 0; x < 6; ++x) {
                        actual[static_cast<std::size_t>(y) * 6 + x] = CFA[
                            rtengine::positiveModulo(matrix[2] * x + matrix[3] * y + oy, 6)
                        ][rtengine::positiveModulo(matrix[0] * x + matrix[1] * y + ox, 6)];
                    }
                }
                variants.insert(actual);
            }
        }
    }
    return variants;
}

int orientation()
{
    const auto variants = cfaVariants();
    require(variants.size() == 18, "expected exactly 18 CFA representations");
    constexpr int width = 43;
    constexpr int height = 37;
    for (const auto &flat : variants) {
        int cfa[6][6];
        for (int y = 0; y < 6; ++y) {
            for (int x = 0; x < 6; ++x) {
                cfa[y][x] = flat[static_cast<std::size_t>(y) * 6 + x];
            }
        }
        const auto planeValue = [](int color, int x, int y) {
            return 7000.f + color * 9000.f + x * (17.f + color * 5.f) +
                   y * (31.f - color * 3.f);
        };
        const std::vector<float> mosaic = makeMosaic(width, height, cfa, planeValue);
        for (const auto variant : {
                 rtengine::TriangulatedXTransVariant::INDEPENDENT_RGB,
                 rtengine::TriangulatedXTransVariant::GREEN_CHROMA_DIFFERENCE}) {
            const Output output = run(mosaic, width, height, cfa, variant);
            requireSamplePreservation(output, mosaic, cfa);
            for (int color = 0; color < 3; ++color) {
                for (int y = 3; y < height - 3; ++y) {
                    for (int x = 3; x < width - 3; ++x) {
                        require(std::fabs(output.at(color, x, y) - planeValue(color, x, y)) <= .01,
                                "orientation mapping broke interior linear interpolation");
                    }
                }
            }
        }
    }
    return 0;
}

int artifacts()
{
    constexpr int width = 96;
    constexpr int height = 84;
    struct Scene {
        const char *name;
        float (*value)(int, int);
    };
    const std::array<Scene, 9> scenes {{
        {"vertical-edge", [](int x, int) { return x < 48 ? 4000.f : 56000.f; }},
        {"horizontal-edge", [](int, int y) { return y < 42 ? 4000.f : 56000.f; }},
        {"diagonal-edge", [](int x, int y) { return x < y ? 4000.f : 56000.f; }},
        {"shallow-edge", [](int x, int y) { return x * 2 < y + 54 ? 4000.f : 56000.f; }},
        {"checkerboard", [](int x, int y) { return (x + y) & 1 ? 56000.f : 4000.f; }},
        {"vertical-lines", [](int x, int) { return x & 1 ? 56000.f : 4000.f; }},
        {"horizontal-lines", [](int, int y) { return y & 1 ? 56000.f : 4000.f; }},
        {"diagonal-lines", [](int x, int y) { return (x + y) % 4 < 2 ? 56000.f : 4000.f; }},
        {"radial", [](int x, int y) {
            const double dx = x - 47.5;
            const double dy = y - 41.5;
            return std::sin(std::sqrt(dx * dx + dy * dy) * 1.7) >= 0.0
                ? 56000.f : 4000.f;
        }}
    }};

    for (const Scene &scene : scenes) {
        const std::vector<float> mosaic = makeMosaic(
            width, height, CFA,
            [&scene](int, int x, int y) { return scene.value(x, y); });
        const Output independent = run(
            mosaic, width, height, CFA,
            rtengine::TriangulatedXTransVariant::INDEPENDENT_RGB);
        const Output chroma = run(
            mosaic, width, height, CFA,
            rtengine::TriangulatedXTransVariant::GREEN_CHROMA_DIFFERENCE);
        requireSamplePreservation(independent, mosaic, CFA);
        requireSamplePreservation(chroma, mosaic, CFA);
        std::cout << scene.name
                  << " independent_false_color_rms=" << falseColorRms(independent)
                  << " chroma_false_color_rms=" << falseColorRms(chroma) << '\n';
    }

    std::array<std::vector<float>, 3> truth;
    for (auto &plane : truth) {
        plane.resize(static_cast<std::size_t>(width) * height);
    }
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const std::size_t pixel = static_cast<std::size_t>(y) * width + x;
            const bool alternate = (x + y) & 1;
            truth[0][pixel] = alternate ? 58000.f : 6000.f;
            truth[1][pixel] = alternate ? 18000.f : 46000.f;
            truth[2][pixel] = alternate ? 5000.f : 57000.f;
        }
    }
    const std::vector<float> colored = makeMosaic(
        width, height, CFA,
        [&truth, width](int color, int x, int y) {
            return truth[color][static_cast<std::size_t>(y) * width + x];
        });
    const Output independent = run(
        colored, width, height, CFA,
        rtengine::TriangulatedXTransVariant::INDEPENDENT_RGB);
    const Output chroma = run(
        colored, width, height, CFA,
        rtengine::TriangulatedXTransVariant::GREEN_CHROMA_DIFFERENCE);
    std::cout << "colored-checkerboard independent_reconstruction_rms="
              << reconstructionRms(independent, truth)
              << " chroma_reconstruction_rms=" << reconstructionRms(chroma, truth) << '\n';
    return 0;
}

} // namespace

int main(int argc, char **argv)
{
    try {
        require(argc == 2, "expected one test-group argument");
        const std::string group = argv[1];
        if (group == "contract") return contract();
        if (group == "synthetic") return synthetic();
        if (group == "orientation") return orientation();
        if (group == "artifacts") return artifacts();
        throw std::runtime_error("unknown test group: " + group);
    } catch (const std::exception &error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
