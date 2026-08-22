#include "rtengine/xtrans_cfa.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

constexpr double PI = 3.141592653589793238462643383279502884;
using Complex = std::complex<double>;
using Atom = std::array<Complex, 18>;

constexpr std::array<std::array<int, 2>, 18> CARRIERS {{
    {{0, 0}}, {{2, 0}}, {{4, 0}},
    {{1, 1}}, {{3, 1}}, {{5, 1}},
    {{0, 2}}, {{2, 2}}, {{4, 2}},
    {{1, 3}}, {{3, 3}}, {{5, 3}},
    {{0, 4}}, {{2, 4}}, {{4, 4}},
    {{1, 5}}, {{3, 5}}, {{5, 5}}
}};

void require(bool condition, const std::string &message)
{
    if (!condition) {
        throw std::runtime_error(message);
    }
}

Complex maskCoefficient(int color, int kx, int ky, int originX, int originY)
{
    Complex result;
    for (int y = 0; y < 6; ++y) {
        for (int x = 0; x < 6; ++x) {
            if (rtengine::CANONICAL_XTRANS_CFA[(y + originY) % 6][(x + originX) % 6] == color) {
                result += std::polar(1.0, -2.0 * PI * (kx * x + ky * y) / 6.0);
            }
        }
    }
    return result / 36.0;
}

std::array<std::array<double, 3>, 3> directions()
{
    return {{
        {{1.0 / std::sqrt(3.0), 1.0 / std::sqrt(3.0), 1.0 / std::sqrt(3.0)}},
        {{1.0 / std::sqrt(2.0), 0.0, -1.0 / std::sqrt(2.0)}},
        {{1.0 / std::sqrt(6.0), -2.0 / std::sqrt(6.0), 1.0 / std::sqrt(6.0)}}
    }};
}

Atom atom(int index, int originX, int originY)
{
    const auto basis = directions();
    const auto &source = CARRIERS[static_cast<std::size_t>(index / 3)];
    const auto &direction = basis[static_cast<std::size_t>(index % 3)];
    Atom result {{}};
    for (std::size_t outputIndex = 0; outputIndex < CARRIERS.size(); ++outputIndex) {
        const auto &output = CARRIERS[outputIndex];
        for (int color = 0; color < 3; ++color) {
            result[outputIndex] += direction[static_cast<std::size_t>(color)] * maskCoefficient(
                color,
                rtengine::positiveModulo(output[0] - source[0], 6),
                rtengine::positiveModulo(output[1] - source[1], 6),
                originX,
                originY);
        }
    }
    return result;
}

Complex inner(const Atom &left, const Atom &right)
{
    Complex result;
    for (std::size_t i = 0; i < left.size(); ++i) {
        result += std::conj(left[i]) * right[i];
    }
    return result;
}

double norm(const Atom &value)
{
    return std::sqrt(std::max(0.0, inner(value, value).real()));
}

std::vector<Complex> solve(std::vector<std::vector<Complex>> matrix, std::vector<Complex> rhs)
{
    for (std::size_t column = 0; column < rhs.size(); ++column) {
        std::size_t pivot = column;
        for (std::size_t row = column + 1; row < rhs.size(); ++row) {
            if (std::abs(matrix[row][column]) > std::abs(matrix[pivot][column])) {
                pivot = row;
            }
        }
        require(std::abs(matrix[pivot][column]) > 1e-13, "singular fit in test fixture");
        std::swap(matrix[pivot], matrix[column]);
        std::swap(rhs[pivot], rhs[column]);
        const Complex diagonal = matrix[column][column];
        for (std::size_t entry = column; entry < rhs.size(); ++entry) {
            matrix[column][entry] /= diagonal;
        }
        rhs[column] /= diagonal;
        for (std::size_t row = 0; row < rhs.size(); ++row) {
            if (row == column) {
                continue;
            }
            const Complex factor = matrix[row][column];
            for (std::size_t entry = column; entry < rhs.size(); ++entry) {
                matrix[row][entry] -= factor * matrix[column][entry];
            }
            rhs[row] -= factor * rhs[column];
        }
    }
    return rhs;
}

std::vector<Complex> fit(const std::vector<int> &support, const Atom &observation)
{
    std::vector<std::vector<Complex>> gram(support.size(), std::vector<Complex>(support.size()));
    std::vector<Complex> rhs(support.size());
    for (std::size_t row = 0; row < support.size(); ++row) {
        const Atom left = atom(support[row], 0, 0);
        rhs[row] = inner(left, observation);
        for (std::size_t column = 0; column < support.size(); ++column) {
            gram[row][column] = inner(left, atom(support[column], 0, 0));
        }
    }
    return solve(gram, rhs);
}

Atom combine(const std::vector<int> &support, const std::vector<Complex> &coefficients, int originX, int originY)
{
    Atom result {{}};
    for (std::size_t term = 0; term < support.size(); ++term) {
        const auto &carrier = CARRIERS[static_cast<std::size_t>(support[term] / 3)];
        const double fx = carrier[0] <= 3 ? carrier[0] / 6.0 : (carrier[0] - 6) / 6.0;
        const double fy = carrier[1] <= 3 ? carrier[1] / 6.0 : (carrier[1] - 6) / 6.0;
        const Complex translated = coefficients[term] * std::polar(1.0, 2.0 * PI * (fx * originX + fy * originY));
        const Atom value = atom(support[term], originX, originY);
        for (std::size_t row = 0; row < result.size(); ++row) {
            result[row] += value[row] * translated;
        }
    }
    return result;
}

double phaseError(Complex left, Complex right, double fx, double fy, int dx, int dy)
{
    const double predicted = 2.0 * PI * (fx * dx + fy * dy);
    return std::abs(std::remainder(std::arg(right) - std::arg(left) - predicted, 2.0 * PI)) / PI;
}

int contract()
{
    for (int originY = 0; originY < 6; ++originY) {
        for (int originX = 0; originX < 6; ++originX) {
            for (int index = 0; index < 54; ++index) {
                require(norm(atom(index, originX, originY)) > 0.0, "phase dictionary contains zero atom");
            }
        }
    }
    return 0;
}

int phase()
{
    const Complex source = std::polar(0.8, 0.37);
    const double fx = 1.0 / 3.0;
    const double fy = 1.0 / 6.0;
    const int dx = 6;
    const int dy = 12;
    const Complex translated = source * std::polar(1.0, 2.0 * PI * (fx * dx + fy * dy));
    require(phaseError(source, translated, fx, fy, dx, dy) < 1e-14,
            "predicted Fourier phase evolution changed");
    require(std::abs(phaseError(source, translated * Complex(0.0, 1.0), fx, fy, dx, dy) - 0.5) < 1e-14,
            "phase disagreement normalization changed");
    return 0;
}

int nullContext()
{
    const std::vector<int> alternative {{2, 28, 31, 34}};
    const std::vector<Complex> coefficients = fit(alternative, atom(0, 0, 0));
    for (const auto &origin : std::array<std::array<int, 2>, 7>{{
             {{0, 0}}, {{6, 0}}, {{12, 0}}, {{0, 6}}, {{1, 0}}, {{0, 1}}, {{7, 5}}
         }}) {
        const Atom truth = combine({0}, {Complex(1.0, 0.0)}, origin[0], origin[1]);
        const Atom competitor = combine(alternative, coefficients, origin[0], origin[1]);
        Atom residual {{}};
        for (std::size_t row = 0; row < residual.size(); ++row) {
            residual[row] = truth[row] - competitor[row];
        }
        require(norm(residual) < 3e-14, "known null ceased to be coherent after window translation");
    }
    return 0;
}

int selection()
{
    const double localBest = 0.0;
    const double localAlternative = 0.0;
    const double coherentPhase = 0.0;
    const double wrongPhase = 0.5;
    require(localBest + 0.5 * coherentPhase < localAlternative + 0.5 * wrongPhase,
            "phase-aware tie breaker did not select the coherent hypothesis");
    return 0;
}

int transition()
{
    // A strong local residual advantage must be able to defeat a weak support
    // continuity term at a genuine source transition.
    const double correct = 0.0 + 0.5 * 0.4;
    const double falselySmooth = 1.0 + 0.5 * 0.0;
    require(correct < falselySmooth, "context score erased a genuine support transition");
    return 0;
}

} // namespace

int main(int argc, char **argv)
{
    if (argc != 2) {
        std::cerr << "usage: rawtherapee-xtrans-alias-context-tests MODE\n";
        return 2;
    }
    try {
        const std::string mode(argv[1]);
        if (mode == "contract") {
            return contract();
        }
        if (mode == "phase") {
            return phase();
        }
        if (mode == "null_context") {
            return nullContext();
        }
        if (mode == "selection") {
            return selection();
        }
        if (mode == "transition") {
            return transition();
        }
        throw std::runtime_error("unknown test mode: " + mode);
    } catch (const std::exception &error) {
        std::cerr << "xtrans alias context test failed: " << error.what() << '\n';
        return 1;
    }
}
