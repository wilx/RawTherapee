#include "rtengine/xtrans_cfa.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <iostream>
#include <limits>
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

Complex maskCoefficient(int color, int kx, int ky)
{
    Complex result;
    for (int y = 0; y < 6; ++y) {
        for (int x = 0; x < 6; ++x) {
            if (rtengine::CANONICAL_XTRANS_CFA[y][x] == color) {
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

std::vector<Atom> dictionary()
{
    const auto basis = directions();
    std::vector<Atom> result;
    result.reserve(54);
    for (const auto &source : CARRIERS) {
        for (const auto &direction : basis) {
            Atom atom {{}};
            for (std::size_t outputIndex = 0; outputIndex < CARRIERS.size(); ++outputIndex) {
                const auto &output = CARRIERS[outputIndex];
                for (int color = 0; color < 3; ++color) {
                    atom[outputIndex] += direction[color] * maskCoefficient(
                        color,
                        rtengine::positiveModulo(output[0] - source[0], 6),
                        rtengine::positiveModulo(output[1] - source[1], 6));
                }
            }
            result.push_back(atom);
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

Atom combine(
    const std::vector<Atom> &atoms,
    const std::vector<int> &support,
    const std::vector<Complex> &coefficients)
{
    Atom result {{}};
    for (std::size_t term = 0; term < support.size(); ++term) {
        for (std::size_t row = 0; row < result.size(); ++row) {
            result[row] += atoms[support[term]][row] * coefficients[term];
        }
    }
    return result;
}

std::vector<Complex> solve(std::vector<std::vector<Complex>> matrix, std::vector<Complex> rhs)
{
    const std::size_t size = rhs.size();
    for (std::size_t column = 0; column < size; ++column) {
        std::size_t pivot = column;
        for (std::size_t row = column + 1; row < size; ++row) {
            if (std::abs(matrix[row][column]) > std::abs(matrix[pivot][column])) {
                pivot = row;
            }
        }
        require(std::abs(matrix[pivot][column]) > 1e-13, "selected support is singular");
        std::swap(matrix[pivot], matrix[column]);
        std::swap(rhs[pivot], rhs[column]);
        const Complex diagonal = matrix[column][column];
        for (std::size_t entry = column; entry < size; ++entry) {
            matrix[column][entry] /= diagonal;
        }
        rhs[column] /= diagonal;
        for (std::size_t row = 0; row < size; ++row) {
            if (row == column) {
                continue;
            }
            const Complex factor = matrix[row][column];
            for (std::size_t entry = column; entry < size; ++entry) {
                matrix[row][entry] -= factor * matrix[column][entry];
            }
            rhs[row] -= factor * rhs[column];
        }
    }
    return rhs;
}

std::vector<Complex> fit(
    const std::vector<Atom> &atoms,
    const std::vector<int> &support,
    const Atom &observation)
{
    std::vector<std::vector<Complex>> gram(
        support.size(), std::vector<Complex>(support.size()));
    std::vector<Complex> rhs(support.size());
    for (std::size_t row = 0; row < support.size(); ++row) {
        rhs[row] = inner(atoms[support[row]], observation);
        for (std::size_t column = 0; column < support.size(); ++column) {
            gram[row][column] = inner(atoms[support[row]], atoms[support[column]]);
        }
    }
    return solve(gram, rhs);
}

struct OmpResult final {
    std::vector<int> support;
    std::vector<Complex> coefficients;
    std::vector<double> residuals;
};

OmpResult omp(const std::vector<Atom> &atoms, const Atom &observation, int count)
{
    OmpResult result;
    Atom residual = observation;
    const double observationNorm = norm(observation);
    result.residuals.push_back(1.0);
    for (int iteration = 0; iteration < count; ++iteration) {
        int best = -1;
        double bestScore = -1.0;
        for (std::size_t candidate = 0; candidate < atoms.size(); ++candidate) {
            if (std::find(result.support.begin(), result.support.end(), candidate) != result.support.end()) {
                continue;
            }
            const double score = std::abs(inner(atoms[candidate], residual)) / norm(atoms[candidate]);
            if (score > bestScore) {
                best = static_cast<int>(candidate);
                bestScore = score;
            }
        }
        require(best >= 0, "OMP did not find a candidate");
        result.support.push_back(best);
        result.coefficients = fit(atoms, result.support, observation);
        const Atom predicted = combine(atoms, result.support, result.coefficients);
        for (std::size_t row = 0; row < residual.size(); ++row) {
            residual[row] = observation[row] - predicted[row];
        }
        result.residuals.push_back(norm(residual) / observationNorm);
        require(
            result.residuals.back() <= result.residuals[result.residuals.size() - 2] + 2e-14,
            "OMP residual increased");
    }
    return result;
}

int contract()
{
    const std::vector<Atom> atoms = dictionary();
    require(atoms.size() == 54, "sparse dictionary column count changed");
    for (const Atom &atom : atoms) {
        require(norm(atom) > 0.0, "sparse dictionary contains a zero atom");
    }
    require(std::abs(inner(atoms[0], atoms[0]).real() - 1.0 / 3.0) < 2e-15,
            "luma constellation energy changed");
    return 0;
}

int oracle()
{
    const std::vector<Atom> atoms = dictionary();
    for (std::size_t index = 0; index < atoms.size(); ++index) {
        const Complex truth(0.37 + index * 0.001, -0.22);
        const std::vector<Complex> recovered = fit(
            atoms, {static_cast<int>(index)}, combine(atoms, {static_cast<int>(index)}, {truth}));
        require(std::abs(recovered[0] - truth) < 2e-14, "single-column oracle recovery failed");
    }
    for (int first = 0; first < 54; ++first) {
        for (int second = first + 1; second < 54; ++second) {
            const std::vector<Complex> truth {{Complex(0.7, -0.2), Complex(-0.3, 0.5)}};
            const std::vector<int> support {{first, second}};
            const std::vector<Complex> recovered = fit(atoms, support, combine(atoms, support, truth));
            require(std::abs(recovered[0] - truth[0]) < 4e-13 &&
                    std::abs(recovered[1] - truth[1]) < 4e-13,
                    "two-column oracle recovery failed");
        }
    }
    return 0;
}

int ompTest()
{
    const std::vector<Atom> atoms = dictionary();
    const std::vector<int> support {{2, 19, 47}};
    const std::vector<Complex> coefficients {{
        Complex(1.0, 0.2), Complex(-0.55, 0.35), Complex(0.25, -0.4)
    }};
    const Atom observation = combine(atoms, support, coefficients);
    const OmpResult recovered = omp(atoms, observation, 3);
    std::vector<int> sorted = recovered.support;
    std::sort(sorted.begin(), sorted.end());
    require(sorted == support, "replica-aware OMP did not recover the known support");
    require(recovered.residuals.back() < 2e-14, "OMP did not reproduce the observation");
    return 0;
}

int ambiguity()
{
    const std::vector<Atom> atoms = dictionary();
    const int target = 0;
    const std::vector<int> alternative {{34, 2, 28, 31}};
    const std::vector<Complex> coefficients = fit(atoms, alternative, atoms[target]);
    const Atom reconstructed = combine(atoms, alternative, coefficients);
    Atom residual {{}};
    for (std::size_t row = 0; row < residual.size(); ++row) {
        residual[row] = atoms[target][row] - reconstructed[row];
    }
    require(norm(residual) < 2e-14, "known luma/chroma null competitor was not exact");
    require(std::find(alternative.begin(), alternative.end(), target) == alternative.end(),
            "null competitor reused the target atom");
    return 0;
}

double realInner(const std::vector<double> &left, const std::vector<double> &right)
{
    double result = 0.0;
    for (std::size_t i = 0; i < left.size(); ++i) {
        result += left[i] * right[i];
    }
    return result;
}

int grouped()
{
    constexpr int size = 24;
    const auto basis = directions();
    std::vector<double> cosine(size * size);
    std::vector<double> sine(size * size);
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            const int color = rtengine::CANONICAL_XTRANS_CFA[y % 6][x % 6];
            const double phase = 2.0 * PI * (8 * x + 4 * y) / size;
            cosine[static_cast<std::size_t>(y) * size + x] = basis[1][color] * std::cos(phase);
            sine[static_cast<std::size_t>(y) * size + x] = basis[1][color] * std::sin(phase);
        }
    }
    const double determinant = realInner(cosine, cosine) * realInner(sine, sine) -
        realInner(cosine, sine) * realInner(sine, cosine);
    require(std::abs(determinant) > 1e-10, "half-carrier quadratures collapsed");
    for (int phaseIndex = 0; phaseIndex < 72; ++phaseIndex) {
        const double phase = 2.0 * PI * phaseIndex / 72.0;
        std::vector<double> observation(size * size);
        for (std::size_t i = 0; i < observation.size(); ++i) {
            observation[i] = std::cos(phase) * cosine[i] - std::sin(phase) * sine[i];
        }
        const double rhsCos = realInner(cosine, observation);
        const double rhsSin = realInner(sine, observation);
        const double recoveredCos =
            (rhsCos * realInner(sine, sine) - rhsSin * realInner(cosine, sine)) / determinant;
        const double recoveredSin =
            (rhsSin * realInner(cosine, cosine) - rhsCos * realInner(sine, cosine)) / determinant;
        require(std::abs(recoveredCos - std::cos(phase)) < 2e-14 &&
                std::abs(recoveredSin + std::sin(phase)) < 2e-14,
                "grouped real quadrature recovery was phase biased");
    }
    return 0;
}

} // namespace

int main(int argc, char **argv)
{
    if (argc != 2) {
        std::cerr << "usage: rawtherapee-xtrans-sparse-alias-tests MODE\n";
        return 2;
    }
    try {
        const std::string mode(argv[1]);
        if (mode == "contract") {
            return contract();
        }
        if (mode == "oracle") {
            return oracle();
        }
        if (mode == "omp") {
            return ompTest();
        }
        if (mode == "ambiguity") {
            return ambiguity();
        }
        if (mode == "grouped") {
            return grouped();
        }
        throw std::runtime_error("unknown test mode: " + mode);
    } catch (const std::exception &error) {
        std::cerr << "xtrans sparse alias test failed: " << error.what() << '\n';
        return 1;
    }
}
