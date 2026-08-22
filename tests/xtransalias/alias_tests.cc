#include "rtengine/xtrans_cfa.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstring>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

constexpr double PI = 3.141592653589793238462643383279502884;

constexpr std::array<std::array<int, 2>, 18> CARRIERS {{
    {{0, 0}}, {{2, 0}}, {{4, 0}},
    {{1, 1}}, {{3, 1}}, {{5, 1}},
    {{0, 2}}, {{2, 2}}, {{4, 2}},
    {{1, 3}}, {{3, 3}}, {{5, 3}},
    {{0, 4}}, {{2, 4}}, {{4, 4}},
    {{1, 5}}, {{3, 5}}, {{5, 5}}
}};

struct ExpectedCarrier final {
    std::array<int, 2> index;
    // Per RGB coefficient: real numerator, sqrt(3)-imaginary numerator.
    std::array<int, 6> numerator;
};

constexpr std::array<ExpectedCarrier, 18> EXPECTED {{
    {{{0, 0}}, {{ 8,  0, 20,  0,  8,  0}}},
    {{{2, 0}}, {{-1, -1,  2,  2, -1, -1}}},
    {{{4, 0}}, {{-1,  1,  2, -2, -1,  1}}},
    {{{1, 1}}, {{ 0,  0,  0,  0,  0,  0}}},
    {{{3, 1}}, {{ 3, -3,  0,  0, -3,  3}}},
    {{{5, 1}}, {{ 0,  0,  0,  0,  0,  0}}},
    {{{0, 2}}, {{-1, -1,  2,  2, -1, -1}}},
    {{{2, 2}}, {{ 2, -2, -4,  4,  2, -2}}},
    {{{4, 2}}, {{-4,  0,  8,  0, -4,  0}}},
    {{{1, 3}}, {{-3,  3,  0,  0,  3, -3}}},
    {{{3, 3}}, {{ 0,  0,  0,  0,  0,  0}}},
    {{{5, 3}}, {{-3, -3,  0,  0,  3,  3}}},
    {{{0, 4}}, {{-1,  1,  2, -2, -1,  1}}},
    {{{2, 4}}, {{-4,  0,  8,  0, -4,  0}}},
    {{{4, 4}}, {{ 2,  2, -4, -4,  2,  2}}},
    {{{1, 5}}, {{ 0,  0,  0,  0,  0,  0}}},
    {{{3, 5}}, {{ 3,  3,  0,  0, -3, -3}}},
    {{{5, 5}}, {{ 0,  0,  0,  0,  0,  0}}}
}};

void require(bool condition, const std::string &message)
{
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::complex<double> coefficient(const int cfa[6][6], int color, int kx, int ky)
{
    std::complex<double> result;
    for (int y = 0; y < 6; ++y) {
        for (int x = 0; x < 6; ++x) {
            if (cfa[y][x] == color) {
                const double phase = -2.0 * PI * (kx * x + ky * y) / 6.0;
                result += std::polar(1.0, phase);
            }
        }
    }
    return result / 36.0;
}

std::set<std::array<int, 36>> variants()
{
    constexpr int matrices[8][4] = {
        {1, 0, 0, 1}, {0, -1, 1, 0}, {-1, 0, 0, -1}, {0, 1, -1, 0},
        {-1, 0, 0, 1}, {1, 0, 0, -1}, {0, 1, 1, 0}, {0, -1, -1, 0}
    };
    std::set<std::array<int, 36>> result;
    for (const auto &matrix : matrices) {
        for (int oy = 0; oy < 6; ++oy) {
            for (int ox = 0; ox < 6; ++ox) {
                std::array<int, 36> actual {{}};
                for (int y = 0; y < 6; ++y) {
                    for (int x = 0; x < 6; ++x) {
                        actual[static_cast<std::size_t>(y) * 6 + x] =
                            rtengine::CANONICAL_XTRANS_CFA[
                                rtengine::positiveModulo(matrix[2] * x + matrix[3] * y + oy, 6)
                            ][rtengine::positiveModulo(matrix[0] * x + matrix[1] * y + ox, 6)];
                    }
                }
                result.insert(actual);
            }
        }
    }
    return result;
}

int contract()
{
    int counts[3] = {0, 0, 0};
    for (int y = 0; y < 6; ++y) {
        for (int x = 0; x < 6; ++x) {
            const int color = rtengine::CANONICAL_XTRANS_CFA[y][x];
            require(color >= 0 && color <= 2, "invalid canonical CFA color");
            ++counts[color];
            require(
                color == rtengine::CANONICAL_XTRANS_CFA[
                    rtengine::positiveModulo(y + 3, 6)
                ][rtengine::positiveModulo(x + 3, 6)],
                "canonical CFA does not have the documented (3,3) period");
        }
    }
    require(counts[0] == 8 && counts[1] == 20 && counts[2] == 8,
            "canonical CFA color counts changed");
    require(CARRIERS.size() == 18, "reciprocal carrier count changed");
    for (const auto &carrier : CARRIERS) {
        require((carrier[0] + carrier[1]) % 2 == 0,
                "carrier violates the reciprocal-lattice parity rule");
    }
    return 0;
}

int spectra()
{
    int nonzeroRows = 0;
    for (const ExpectedCarrier &expected : EXPECTED) {
        bool nonzero = false;
        for (int color = 0; color < 3; ++color) {
            const std::complex<double> actual = coefficient(
                rtengine::CANONICAL_XTRANS_CFA,
                color,
                expected.index[0],
                expected.index[1]);
            const std::complex<double> wanted(
                expected.numerator[color * 2] / 36.0,
                expected.numerator[color * 2 + 1] * std::sqrt(3.0) / 36.0);
            require(std::abs(actual - wanted) < 2e-15,
                    "exact X-Trans mask coefficient changed");
            nonzero |= std::abs(actual) > 1e-14;
        }
        nonzeroRows += nonzero;
    }
    require(nonzeroRows == 13,
            "expected the 13 nonzero Rafinazari-Dubois carrier rows");

    for (int ky = 0; ky < 6; ++ky) {
        for (int kx = 0; kx < 6; ++kx) {
            if ((kx + ky) % 2 != 0) {
                for (int color = 0; color < 3; ++color) {
                    require(std::abs(coefficient(
                                rtengine::CANONICAL_XTRANS_CFA, color, kx, ky)) < 2e-15,
                            "forbidden reciprocal-lattice bin was nonzero");
                }
            }
        }
    }
    return 0;
}

int replicas()
{
    constexpr int size = 24;
    constexpr int sourceX = 3;
    constexpr int sourceY = 5;
    const std::array<double, 3> direction {{
        1.0 / std::sqrt(6.0), -2.0 / std::sqrt(6.0), 1.0 / std::sqrt(6.0)
    }};
    std::vector<std::complex<double>> mosaic(size * size);
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            const int color = rtengine::CANONICAL_XTRANS_CFA[y % 6][x % 6];
            const double phase = 2.0 * PI * (sourceX * x + sourceY * y) / size;
            mosaic[static_cast<std::size_t>(y) * size + x] =
                direction[color] * std::polar(1.0, phase);
        }
    }
    for (int outputY = 0; outputY < size; ++outputY) {
        for (int outputX = 0; outputX < size; ++outputX) {
            std::complex<double> actual;
            for (int y = 0; y < size; ++y) {
                for (int x = 0; x < size; ++x) {
                    const double phase = -2.0 * PI *
                        (outputX * x + outputY * y) / size;
                    actual += mosaic[static_cast<std::size_t>(y) * size + x] *
                        std::polar(1.0, phase);
                }
            }
            actual /= static_cast<double>(size * size);
            std::complex<double> predicted;
            const int deltaX = rtengine::positiveModulo(outputX - sourceX, size);
            const int deltaY = rtengine::positiveModulo(outputY - sourceY, size);
            if (deltaX % (size / 6) == 0 && deltaY % (size / 6) == 0) {
                const int carrierX = deltaX / (size / 6);
                const int carrierY = deltaY / (size / 6);
                if ((carrierX + carrierY) % 2 == 0) {
                    for (int color = 0; color < 3; ++color) {
                        predicted += direction[color] * coefficient(
                            rtengine::CANONICAL_XTRANS_CFA,
                            color,
                            carrierX,
                            carrierY);
                    }
                }
            }
            require(std::abs(actual - predicted) < 2e-14,
                    "direct complex sampling did not match shifted mask replicas");
        }
    }

    // A monochrome/luma signal is observed at every pixel unchanged because
    // the three disjoint masks sum to one.
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            const double phase = 2.0 * PI * (sourceX * x + sourceY * y) / size;
            const std::complex<double> truth = std::polar(1.0, phase);
            const int color = rtengine::CANONICAL_XTRANS_CFA[y % 6][x % 6];
            const std::array<std::complex<double>, 3> rgb {{truth, truth, truth}};
            const std::complex<double> measured = rgb[color];
            require(measured == truth,
                    "monochrome sampling unexpectedly changed a value");
        }
    }
    return 0;
}

int variantSpectra()
{
    const auto all = variants();
    require(all.size() == 18, "expected exactly 18 phase/orientation matrices");
    std::array<std::vector<double>, 3> reference;
    for (int color = 0; color < 3; ++color) {
        for (const auto &carrier : CARRIERS) {
            reference[color].push_back(std::abs(coefficient(
                rtengine::CANONICAL_XTRANS_CFA, color, carrier[0], carrier[1])));
        }
        std::sort(reference[color].begin(), reference[color].end());
    }
    for (const auto &flat : all) {
        int cfa[6][6];
        std::memcpy(cfa, flat.data(), sizeof(cfa));
        rtengine::XTransCfaTransform transform;
        require(rtengine::findCanonicalXTransTransform(cfa, transform),
                "supported variant did not canonicalize");
        for (int color = 0; color < 3; ++color) {
            std::vector<double> magnitudes;
            for (const auto &carrier : CARRIERS) {
                magnitudes.push_back(std::abs(coefficient(
                    cfa, color, carrier[0], carrier[1])));
            }
            std::sort(magnitudes.begin(), magnitudes.end());
            for (std::size_t index = 0; index < magnitudes.size(); ++index) {
                require(std::abs(magnitudes[index] - reference[color][index]) < 2e-15,
                        "orientation changed the carrier-magnitude multiset");
            }
        }
    }
    return 0;
}

int linearity()
{
    constexpr int size = 24;
    std::vector<double> first(size * size);
    std::vector<double> second(size * size);
    std::vector<double> combined(size * size);
    double redGreenInner = 0.0;
    double redSquared = 0.0;
    double greenSquared = 0.0;
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            const std::size_t pixel = static_cast<std::size_t>(y) * size + x;
            const int color = rtengine::CANONICAL_XTRANS_CFA[y % 6][x % 6];
            const double wave1 = std::cos(2.0 * PI * (3 * x + 5 * y) / size + 0.31);
            const double wave2 = std::cos(2.0 * PI * (7 * x - 2 * y) / size - 0.47);
            first[pixel] = (color == 0 ? 1.0 : color == 2 ? -1.0 : 0.0) * wave1;
            second[pixel] = (color == 0 ? 1.0 : color == 1 ? -2.0 : 1.0) * wave2;
            combined[pixel] = first[pixel] + second[pixel];

            const double red = color == 0 ? wave1 : 0.0;
            const double green = color == 1 ? wave1 : 0.0;
            redGreenInner += red * green;
            redSquared += red * red;
            greenSquared += green * green;
        }
    }
    for (std::size_t index = 0; index < combined.size(); ++index) {
        require(combined[index] == first[index] + second[index],
                "sampling operator was not exactly linear");
    }
    require(redSquared > 0.0 && greenSquared > 0.0 && redGreenInner == 0.0,
            "same-frequency red and green observations were not orthogonal");

    double identical = 0.0;
    double norm = 0.0;
    for (double value : first) {
        identical += value * value;
        norm += value * value;
    }
    require(std::abs(identical / norm - 1.0) < 1e-15,
            "identical observation similarity was not one");
    return 0;
}

} // namespace

int main(int argc, char **argv)
{
    try {
        require(argc == 2, "usage: rawtherapee-xtrans-alias-tests MODE");
        const std::string mode = argv[1];
        if (mode == "contract") {
            return contract();
        }
        if (mode == "spectra") {
            return spectra();
        }
        if (mode == "replicas") {
            return replicas();
        }
        if (mode == "variants") {
            return variantSpectra();
        }
        if (mode == "linearity") {
            return linearity();
        }
        throw std::runtime_error("unknown test mode: " + mode);
    } catch (const std::exception &error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
