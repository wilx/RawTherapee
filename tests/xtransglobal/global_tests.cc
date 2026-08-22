#include "rtengine/xtrans_cfa.h"
#include "rtengine/xtrans_global.h"

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

void require(bool condition, const std::string &message)
{
    if (!condition) {
        throw std::runtime_error(message);
    }
}

struct Output final {
    int width;
    int height;
    std::vector<float> values;
    rtengine::GlobalXTransRunResult result;

    float at(int color, int x, int y) const
    {
        const std::size_t pixels = static_cast<std::size_t>(width) * height;
        return values[static_cast<std::size_t>(color) * pixels +
                      static_cast<std::size_t>(y) * width + x];
    }
};

template<typename Function>
std::array<std::vector<float>, 3> makeTruth(int width, int height, Function function)
{
    const std::size_t pixels = static_cast<std::size_t>(width) * height;
    std::array<std::vector<float>, 3> truth {{
        std::vector<float>(pixels), std::vector<float>(pixels),
        std::vector<float>(pixels)}};
    for (int color = 0; color < 3; ++color) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                truth[color][static_cast<std::size_t>(y) * width + x] =
                    function(color, x, y);
            }
        }
    }
    return truth;
}

std::vector<float> mosaicFromTruth(
    const std::array<std::vector<float>, 3> &truth,
    int width,
    int height,
    const int cfa[6][6])
{
    std::vector<float> mosaic(static_cast<std::size_t>(width) * height);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const std::size_t pixel = static_cast<std::size_t>(y) * width + x;
            mosaic[pixel] = truth[cfa[y % 6][x % 6]][pixel];
        }
    }
    return mosaic;
}

Output run(
    const std::vector<float> &mosaic,
    int width,
    int height,
    const int cfa[6][6],
    rtengine::GlobalXTransVariant variant,
    const rtengine::GlobalXTransOptions &options)
{
    const std::size_t pixels = static_cast<std::size_t>(width) * height;
    Output output {width, height, std::vector<float>(pixels * 3), {}};
    output.result = rtengine::demosaicGlobalXTransReference(
        mosaic.data(), output.values.data(), output.values.data() + pixels,
        output.values.data() + 2 * pixels, width, height, cfa, variant, options);
    require(static_cast<bool>(output.result),
            std::string("global reconstruction failed [") +
            rtengine::globalXTransErrorCodeName(output.result.code) + "]: " +
            output.result.message);
    for (const float value : output.values) {
        require(std::isfinite(value), "global reconstruction produced a non-finite value");
    }
    return output;
}

void requireSamplePreservation(
    const Output &output,
    const std::vector<float> &mosaic,
    const int cfa[6][6])
{
    for (int y = 0; y < output.height; ++y) {
        for (int x = 0; x < output.width; ++x) {
            const std::size_t pixel = static_cast<std::size_t>(y) * output.width + x;
            const int color = cfa[y % 6][x % 6];
            require(output.at(color, x, y) == mosaic[pixel],
                    "a native CFA sample was not preserved bit-exactly");
        }
    }
    require(output.result.finalSampleMaximum == 0.0,
            "reported final sample error was not zero");
}

double rms(
    const Output &output,
    const std::array<std::vector<float>, 3> &truth,
    int margin = 0)
{
    double squared = 0.0;
    std::size_t count = 0;
    for (int color = 0; color < 3; ++color) {
        for (int y = margin; y < output.height - margin; ++y) {
            for (int x = margin; x < output.width - margin; ++x) {
                const std::size_t pixel = static_cast<std::size_t>(y) * output.width + x;
                const double difference = output.at(color, x, y) - truth[color][pixel];
                squared += difference * difference;
                ++count;
            }
        }
    }
    return std::sqrt(squared / count) / 65535.0;
}

double outputDifference(const Output &left, const Output &right)
{
    require(left.values.size() == right.values.size(), "output size mismatch");
    double squared = 0.0;
    for (std::size_t index = 0; index < left.values.size(); ++index) {
        const double difference = left.values[index] - right.values[index];
        squared += difference * difference;
    }
    return std::sqrt(squared / left.values.size()) / 65535.0;
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
                rtengine::XTransCfaTransform transform;
                int cfa[6][6];
                std::memcpy(cfa, actual.data(), sizeof(cfa));
                if (rtengine::findCanonicalXTransTransform(cfa, transform)) {
                    variants.insert(actual);
                }
            }
        }
    }
    return variants;
}

int contract()
{
    using rtengine::GlobalXTransErrorCode;
    require(std::strcmp(rtengine::XTRANS_GLOBAL_SPECTRAL_RGB_METHOD,
                        "xtrans-global-spectral-rgb") == 0,
            "Phase A identifier changed");
    require(std::strcmp(rtengine::XTRANS_GLOBAL_SPECTRAL_DIFF_METHOD,
                        "xtrans-global-spectral-diff") == 0,
            "Phase B identifier changed");
    require(std::strcmp(rtengine::XTRANS_GLOBAL_SPECTRAL_EDGE_METHOD,
                        "xtrans-global-spectral-edge") == 0,
            "Phase C identifier changed");
    const std::array<GlobalXTransErrorCode, 9> codes {{
        GlobalXTransErrorCode::NONE, GlobalXTransErrorCode::SIZE,
        GlobalXTransErrorCode::CFA, GlobalXTransErrorCode::PARAMETER,
        GlobalXTransErrorCode::ALLOCATION, GlobalXTransErrorCode::NONFINITE,
        GlobalXTransErrorCode::SOLVER, GlobalXTransErrorCode::IO,
        GlobalXTransErrorCode::INTERNAL}};
    const std::array<const char *, 9> names {{
        "NONE", "SIZE", "CFA", "PARAMETER", "ALLOCATION", "NONFINITE",
        "SOLVER", "IO", "INTERNAL"}};
    for (std::size_t index = 0; index < codes.size(); ++index) {
        require(std::strcmp(rtengine::globalXTransErrorCodeName(codes[index]),
                            names[index]) == 0,
                "stable error name changed");
    }

    std::vector<float> mosaic(12 * 12, 1000.f);
    std::vector<float> output(12 * 12 * 3);
    rtengine::GlobalXTransOptions options;
    options.maximumIterations = 2;
    options.lambdaRgb = -1.0;
    auto result = rtengine::demosaicGlobalXTransReference(
        mosaic.data(), output.data(), output.data() + 144,
        output.data() + 288, 12, 12, CFA,
        rtengine::GlobalXTransVariant::INDEPENDENT_RGB, options);
    require(!result && result.code == GlobalXTransErrorCode::PARAMETER,
            "invalid lambda did not fail with PARAMETER");

    options.lambdaRgb = 0.02;
    int badCfa[6][6];
    std::memcpy(badCfa, CFA, sizeof(CFA));
    badCfa[0][0] = 0;
    result = rtengine::demosaicGlobalXTransReference(
        mosaic.data(), output.data(), output.data() + 144,
        output.data() + 288, 12, 12, badCfa,
        rtengine::GlobalXTransVariant::INDEPENDENT_RGB, options);
    require(!result && result.code == GlobalXTransErrorCode::CFA,
            "invalid CFA did not fail with CFA");

    mosaic[17] = std::numeric_limits<float>::quiet_NaN();
    result = rtengine::demosaicGlobalXTransReference(
        mosaic.data(), output.data(), output.data() + 144,
        output.data() + 288, 12, 12, CFA,
        rtengine::GlobalXTransVariant::INDEPENDENT_RGB, options);
    require(!result && result.code == GlobalXTransErrorCode::NONFINITE,
            "non-finite input did not fail with NONFINITE");
    return 0;
}

int synthetic()
{
    constexpr int width = 54;
    constexpr int height = 48;
    rtengine::GlobalXTransOptions options;
    options.maximumIterations = 30;
    options.relativeTolerance = 1e-6;

    const auto constant = makeTruth(width, height,
        [](int, int, int) { return 23456.25f; });
    const std::vector<float> constantMosaic =
        mosaicFromTruth(constant, width, height, CFA);
    for (const auto variant : {
            rtengine::GlobalXTransVariant::INDEPENDENT_RGB,
            rtengine::GlobalXTransVariant::GREEN_COLOR_DIFFERENCE,
            rtengine::GlobalXTransVariant::EDGE_PRESERVING_GREEN}) {
        const Output output = run(
            constantMosaic, width, height, CFA, variant, options);
        requireSamplePreservation(output, constantMosaic, CFA);
        require(rms(output, constant) < 1e-7,
                "constant field was not reconstructed accurately");
    }

    const auto correlated = makeTruth(width, height,
        [](int color, int x, int y) {
            const double luminance = 12000.0 + 190.0 * x + 105.0 * y +
                2400.0 * std::sin((x + 2.0 * y) * 0.12);
            const double chroma = color == 0 ? 900.0 : color == 2 ? -700.0 : 0.0;
            return static_cast<float>(luminance + chroma +
                (x > 27 ? (color == 1 ? 6000.0 : 5700.0) : 0.0));
        });
    const std::vector<float> mosaic = mosaicFromTruth(correlated, width, height, CFA);
    const Output phaseA = run(
        mosaic, width, height, CFA,
        rtengine::GlobalXTransVariant::INDEPENDENT_RGB, options);
    const Output phaseB = run(
        mosaic, width, height, CFA,
        rtengine::GlobalXTransVariant::GREEN_COLOR_DIFFERENCE, options);
    const Output phaseC = run(
        mosaic, width, height, CFA,
        rtengine::GlobalXTransVariant::EDGE_PRESERVING_GREEN, options);
    requireSamplePreservation(phaseA, mosaic, CFA);
    requireSamplePreservation(phaseB, mosaic, CFA);
    requireSamplePreservation(phaseC, mosaic, CFA);
    const double rmsA = rms(phaseA, correlated, 3);
    const double rmsB = rms(phaseB, correlated, 3);
    const double rmsC = rms(phaseC, correlated, 3);
    std::cout << "correlated_rms_a=" << rmsA << " correlated_rms_b=" << rmsB
              << " correlated_rms_c=" << rmsC << '\n';
    require(rmsB < rmsA, "color-difference model did not beat independent RGB");
    require(phaseA.result.finalRelativeResidual < 1.0 &&
            phaseB.result.finalRelativeResidual < 1.0 &&
            phaseC.result.finalRelativeResidual < 1.0,
            "a solver did not reduce its initial residual");

    const Output repeated = run(
        mosaic, width, height, CFA,
        rtengine::GlobalXTransVariant::GREEN_COLOR_DIFFERENCE, options);
    require(repeated.values == phaseB.values,
            "whole-image result was not bit-deterministic");

    rtengine::GlobalXTransOptions strong = options;
    strong.lambdaChroma = 1.0;
    const Output strongOutput = run(
        mosaic, width, height, CFA,
        rtengine::GlobalXTransVariant::GREEN_COLOR_DIFFERENCE, strong);
    require(outputDifference(phaseB, strongOutput) > 1e-7,
            "changing lambda did not change the reconstruction");

    rtengine::GlobalXTransOptions p2 = options;
    p2.spectralExponent = 2;
    p2.maximumIterations = 10;
    const Output p2Output = run(
        mosaic, width, height, CFA,
        rtengine::GlobalXTransVariant::GREEN_COLOR_DIFFERENCE, p2);
    requireSamplePreservation(p2Output, mosaic, CFA);
    return 0;
}

int orientation()
{
    const std::set<std::array<int, 36>> variants = cfaVariants();
    require(variants.size() == 18, "expected exactly 18 X-Trans CFA representations");
    rtengine::GlobalXTransOptions options;
    options.maximumIterations = 4;
    const auto truth = makeTruth(24, 24,
        [](int, int, int) { return 17000.f; });
    for (const auto &variant : variants) {
        int cfa[6][6];
        std::memcpy(cfa, variant.data(), sizeof(cfa));
        const std::vector<float> mosaic = mosaicFromTruth(truth, 24, 24, cfa);
        const Output output = run(
            mosaic, 24, 24, cfa,
            rtengine::GlobalXTransVariant::GREEN_COLOR_DIFFERENCE, options);
        requireSamplePreservation(output, mosaic, cfa);
        require(rms(output, truth) < 1e-7,
                "constant reconstruction changed under CFA orientation");
    }
    return 0;
}

int tiled()
{
    constexpr int width = 137;
    constexpr int height = 119;
    const auto truth = makeTruth(width, height,
        [](int color, int x, int y) {
            const double smooth = 15000.0 + 70.0 * x + 45.0 * y +
                3000.0 * std::sin((x + y) * 0.07);
            return static_cast<float>(smooth + (color - 1) * 800.0);
        });
    const std::vector<float> mosaic = mosaicFromTruth(truth, width, height, CFA);
    rtengine::GlobalXTransOptions options;
    options.maximumIterations = 8;
    options.relativeTolerance = 1e-5;
    const Output whole = run(
        mosaic, width, height, CFA,
        rtengine::GlobalXTransVariant::GREEN_COLOR_DIFFERENCE, options);
    options.tileSize = 96;
    const Output tiledOutput = run(
        mosaic, width, height, CFA,
        rtengine::GlobalXTransVariant::GREEN_COLOR_DIFFERENCE, options);
    requireSamplePreservation(tiledOutput, mosaic, CFA);
    require(tiledOutput.result.tileCount > 1, "tiled control used only one tile");
    const double difference = outputDifference(whole, tiledOutput);
    std::cout << "whole_vs_tile96_rms=" << difference << '\n';
    require(difference > 0.0 && difference < 0.02,
            "whole/tiled control difference is implausible");

    // A bright left-border impulse may propagate through the global inverse,
    // but mirror boundaries must not make the opposite border its direct
    // periodic neighbour.
    auto impulseTruth = makeTruth(48, 36,
        [](int, int x, int y) {
            return x == 0 && y == 18 ? 60000.f : 1000.f;
        });
    const std::vector<float> impulse = mosaicFromTruth(impulseTruth, 48, 36, CFA);
    options.tileSize = 0;
    options.maximumIterations = 20;
    const Output bordered = run(
        impulse, 48, 36, CFA,
        rtengine::GlobalXTransVariant::INDEPENDENT_RGB, options);
    double left = 0.0;
    double right = 0.0;
    for (int color = 0; color < 3; ++color) {
        left += std::fabs(bordered.at(color, 0, 18) - 1000.0);
        right += std::fabs(bordered.at(color, 47, 18) - 1000.0);
    }
    require(right < left * 0.1,
            "border impulse behaved like a periodic wraparound neighbour");
    return 0;
}

int control()
{
    // A larger deterministic field makes 384- and 192-pixel overlapping
    // solves true controls rather than one-tile aliases of the whole solve.
    constexpr int width = 594;
    constexpr int height = 510;
    const auto truth = makeTruth(width, height,
        [](int color, int x, int y) {
            const double broad = 15000.0 + 9000.0 *
                std::sin(x * 0.013 + y * 0.007);
            const double fine = 1600.0 * std::sin((2 * x + 3 * y) * 0.19);
            const double chroma = color == 0 ? 700.0 : color == 2 ? -500.0 : 0.0;
            return static_cast<float>(broad + fine + chroma);
        });
    const std::vector<float> mosaic = mosaicFromTruth(truth, width, height, CFA);
    rtengine::GlobalXTransOptions options;
    options.maximumIterations = 20;
    const Output whole = run(
        mosaic, width, height, CFA,
        rtengine::GlobalXTransVariant::GREEN_COLOR_DIFFERENCE, options);
    options.tileSize = 384;
    const Output tile384 = run(
        mosaic, width, height, CFA,
        rtengine::GlobalXTransVariant::GREEN_COLOR_DIFFERENCE, options);
    options.tileSize = 192;
    const Output tile192 = run(
        mosaic, width, height, CFA,
        rtengine::GlobalXTransVariant::GREEN_COLOR_DIFFERENCE, options);
    requireSamplePreservation(tile384, mosaic, CFA);
    requireSamplePreservation(tile192, mosaic, CFA);
    std::cout << "control whole_rms=" << rms(whole, truth, 6)
              << " tile384_rms=" << rms(tile384, truth, 6)
              << " tile192_rms=" << rms(tile192, truth, 6)
              << " whole_vs_384=" << outputDifference(whole, tile384)
              << " whole_vs_192=" << outputDifference(whole, tile192) << '\n';

    constexpr int sweepWidth = 96;
    constexpr int sweepHeight = 84;
    const auto sweepTruth = makeTruth(sweepWidth, sweepHeight,
        [](int color, int x, int y) {
            const double light = 18000.0 + 83.0 * x + 47.0 * y +
                2800.0 * std::sin((x + 2 * y) * 0.14);
            return static_cast<float>(light + (color == 0 ? 850.0 : color == 2 ? -650.0 : 0.0));
        });
    const std::vector<float> sweepMosaic =
        mosaicFromTruth(sweepTruth, sweepWidth, sweepHeight, CFA);
    for (const int iterations : {5, 10, 20, 50}) {
        rtengine::GlobalXTransOptions selected;
        selected.maximumIterations = iterations;
        for (const auto variant : {
                rtengine::GlobalXTransVariant::INDEPENDENT_RGB,
                rtengine::GlobalXTransVariant::GREEN_COLOR_DIFFERENCE}) {
            const Output output = run(
                sweepMosaic, sweepWidth, sweepHeight, CFA, variant, selected);
            std::cout << "iteration_sweep variant=" <<
                (variant == rtengine::GlobalXTransVariant::INDEPENDENT_RGB ? 'A' : 'B')
                << " limit=" << iterations
                << " completed=" << output.result.completedIterations
                << " residual=" << output.result.finalRelativeResidual
                << " rms=" << rms(output, sweepTruth, 4) << '\n';
        }
    }
    for (const double lambda : {0.01, 0.1, 1.0}) {
        rtengine::GlobalXTransOptions selected;
        selected.maximumIterations = 20;
        selected.lambdaChroma = lambda;
        const Output output = run(
            sweepMosaic, sweepWidth, sweepHeight, CFA,
            rtengine::GlobalXTransVariant::GREEN_COLOR_DIFFERENCE, selected);
        std::cout << "lambda_sweep lambda_c=" << lambda
                  << " rms=" << rms(output, sweepTruth, 4) << '\n';
    }
    for (const int exponent : {1, 2}) {
        rtengine::GlobalXTransOptions selected;
        selected.maximumIterations = 20;
        selected.spectralExponent = exponent;
        const Output output = run(
            sweepMosaic, sweepWidth, sweepHeight, CFA,
            rtengine::GlobalXTransVariant::GREEN_COLOR_DIFFERENCE, selected);
        std::cout << "exponent_sweep p=" << exponent
                  << " rms=" << rms(output, sweepTruth, 4) << '\n';
    }

    for (const int iterations : {5, 50}) {
        rtengine::GlobalXTransOptions triangulated;
        triangulated.maximumIterations = iterations;
        rtengine::GlobalXTransOptions zero = triangulated;
        zero.initialization = rtengine::GlobalXTransInitialization::ZERO_FILLED;
        const Output first = run(
            sweepMosaic, sweepWidth, sweepHeight, CFA,
            rtengine::GlobalXTransVariant::GREEN_COLOR_DIFFERENCE, triangulated);
        const Output second = run(
            sweepMosaic, sweepWidth, sweepHeight, CFA,
            rtengine::GlobalXTransVariant::GREEN_COLOR_DIFFERENCE, zero);
        std::cout << "initialization_sweep limit=" << iterations
                  << " output_rms=" << outputDifference(first, second) << '\n';
    }
    return 0;
}

} // namespace

int main(int argc, char **argv)
{
    try {
        require(argc == 2, "usage: rawtherapee-xtrans-global-tests MODE");
        const std::string mode = argv[1];
        if (mode == "contract") {
            return contract();
        }
        if (mode == "synthetic") {
            return synthetic();
        }
        if (mode == "orientation") {
            return orientation();
        }
        if (mode == "tiled") {
            return tiled();
        }
        if (mode == "control") {
            return control();
        }
        throw std::runtime_error("unknown test mode: " + mode);
    } catch (const std::exception &error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
