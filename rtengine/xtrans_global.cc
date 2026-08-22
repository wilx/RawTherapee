/*
 * Three developer-only global X-Trans reconstruction experiments.
 *
 * The quadratic "spectral" penalty is implemented as a normalized graph
 * Laplacian (p=1) or its square (p=2).  With mirror/zero-normal-derivative
 * image boundaries this is the spatial, matrix-free form of weighting the DCT
 * coefficients by the normalized discrete frequencies
 *
 *   w(kx,ky) = ((2-2 cos(kx)) + (2-2 cos(ky)))^p / 8^p.
 *
 * Solving the inverse problem by PCG gives every observation global influence;
 * this is not merely a local Laplacian filter.  The form avoids the implicit
 * left/right and top/bottom wraparound of a periodic FFT.
 *
 * All three objectives use soft data fidelity so their lambda parameters have
 * meaningful scale.  After solving, the physically measured value is copied
 * back to its matching RGB plane.  The result therefore preserves native CFA
 * samples bit-exactly while the pre-projection error remains measurable.
 */

#include "xtrans_global.h"

#include "xtrans_cfa.h"
#include "xtrans_triangulation.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <limits>
#include <new>
#include <sstream>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace rtengine
{
namespace
{

using Clock = std::chrono::steady_clock;
using Vector = std::vector<float>;

constexpr double RAW_SCALE = 65535.0;
constexpr std::size_t DOT_BLOCK_SIZE = 4096;
constexpr float MINIMUM_DIAGONAL = 1e-7f;

std::uint64_t microsecondsSince(const Clock::time_point &started)
{
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            Clock::now() - started).count());
}

bool checkedPixelCount(int width, int height, std::size_t &pixels)
{
    if (width <= 0 || height <= 0) {
        return false;
    }
    const std::size_t w = static_cast<std::size_t>(width);
    const std::size_t h = static_cast<std::size_t>(height);
    if (w > std::numeric_limits<std::size_t>::max() / h) {
        return false;
    }
    pixels = w * h;
    // The largest Phase-C reference workspace is below 32 float planes.
    return pixels <= std::numeric_limits<std::size_t>::max() / (32 * sizeof(float));
}

bool validOptions(const GlobalXTransOptions &options)
{
    const bool tileValid = options.tileSize == 0 || options.tileSize == 96 ||
        options.tileSize == 192 || options.tileSize == 384;
    return std::isfinite(options.lambdaRgb) && options.lambdaRgb > 0.0 &&
        std::isfinite(options.lambdaGreen) && options.lambdaGreen > 0.0 &&
        std::isfinite(options.lambdaChroma) && options.lambdaChroma > 0.0 &&
        (options.spectralExponent == 1 || options.spectralExponent == 2) &&
        options.maximumIterations > 0 && options.maximumIterations <= 1000 &&
        std::isfinite(options.relativeTolerance) && options.relativeTolerance > 0.0 &&
        options.relativeTolerance < 1.0 && options.edgeOuterIterations > 0 &&
        options.edgeOuterIterations <= 100 && options.edgeInnerIterations > 0 &&
        options.edgeInnerIterations <= 1000 &&
        std::isfinite(options.charbonnierEpsilon) &&
        options.charbonnierEpsilon > 0.0 && tileValid;
}

double deterministicDot(const Vector &left, const Vector &right)
{
    const std::size_t blocks =
        (left.size() + DOT_BLOCK_SIZE - 1) / DOT_BLOCK_SIZE;
    std::vector<double> partial(blocks, 0.0);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (std::ptrdiff_t block = 0;
            block < static_cast<std::ptrdiff_t>(blocks); ++block) {
        const std::size_t begin = static_cast<std::size_t>(block) * DOT_BLOCK_SIZE;
        const std::size_t end = std::min(left.size(), begin + DOT_BLOCK_SIZE);
        double sum = 0.0;
        for (std::size_t index = begin; index < end; ++index) {
            sum += static_cast<double>(left[index]) * right[index];
        }
        partial[static_cast<std::size_t>(block)] = sum;
    }
    double result = 0.0;
    for (const double value : partial) {
        result += value;
    }
    return result;
}

void applyLaplacian(
    const float *input,
    float *output,
    int width,
    int height)
{
    // This symmetric four-neighbour graph Laplacian has no edge crossing the
    // finite image boundary.  It is the zero-normal-derivative/mirror-boundary
    // operator, normalized by the infinite-grid maximum eigenvalue eight.
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const std::size_t pixel = static_cast<std::size_t>(y) * width + x;
            const float center = input[pixel];
            float value = 0.f;
            if (x > 0) {
                value += center - input[pixel - 1];
            }
            if (x + 1 < width) {
                value += center - input[pixel + 1];
            }
            if (y > 0) {
                value += center - input[pixel - width];
            }
            if (y + 1 < height) {
                value += center - input[pixel + width];
            }
            output[pixel] = value * (1.f / 8.f);
        }
    }
}

void addSpectralPenalty(
    const float *input,
    float *output,
    float *scratch,
    int width,
    int height,
    int exponent,
    float lambda)
{
    const std::size_t pixels = static_cast<std::size_t>(width) * height;
    if (exponent == 1) {
        applyLaplacian(input, scratch, width, height);
    } else {
        applyLaplacian(input, scratch, width, height);
        applyLaplacian(scratch, scratch + pixels, width, height);
        scratch += pixels;
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (std::ptrdiff_t index = 0;
            index < static_cast<std::ptrdiff_t>(pixels); ++index) {
        output[index] += lambda * scratch[index];
    }
}

float spectralDiagonal(int x, int y, int width, int height, int exponent)
{
    int degree = 0;
    degree += x > 0;
    degree += x + 1 < width;
    degree += y > 0;
    degree += y + 1 < height;
    if (exponent == 1) {
        return degree * (1.f / 8.f);
    }
    // diag(L_graph^2) = degree^2 + degree for an unweighted simple graph.
    return (degree * degree + degree) * (1.f / 64.f);
}

void formEdgeWeights(
    const float *green,
    Vector &horizontal,
    Vector &vertical,
    int width,
    int height,
    float epsilon)
{
    const std::size_t pixels = static_cast<std::size_t>(width) * height;
    horizontal.assign(pixels, 0.f);
    vertical.assign(pixels, 0.f);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const std::size_t pixel = static_cast<std::size_t>(y) * width + x;
            if (x + 1 < width) {
                const float difference = green[pixel + 1] - green[pixel];
                horizontal[pixel] = epsilon /
                    std::sqrt(difference * difference + epsilon * epsilon);
            }
            if (y + 1 < height) {
                const float difference = green[pixel + width] - green[pixel];
                vertical[pixel] = epsilon /
                    std::sqrt(difference * difference + epsilon * epsilon);
            }
        }
    }
}

void addWeightedGreenPenalty(
    const float *input,
    float *output,
    const Vector &horizontal,
    const Vector &vertical,
    int width,
    int height,
    float lambda)
{
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const std::size_t pixel = static_cast<std::size_t>(y) * width + x;
            const float center = input[pixel];
            float value = 0.f;
            if (x > 0) {
                value += horizontal[pixel - 1] * (center - input[pixel - 1]);
            }
            if (x + 1 < width) {
                value += horizontal[pixel] * (center - input[pixel + 1]);
            }
            if (y > 0) {
                value += vertical[pixel - width] * (center - input[pixel - width]);
            }
            if (y + 1 < height) {
                value += vertical[pixel] * (center - input[pixel + width]);
            }
            output[pixel] += lambda * value * (1.f / 8.f);
        }
    }
}

float weightedGreenDiagonal(
    std::size_t pixel,
    int x,
    int y,
    int width,
    int height,
    const Vector &horizontal,
    const Vector &vertical)
{
    float sum = 0.f;
    if (x > 0) {
        sum += horizontal[pixel - 1];
    }
    if (x + 1 < width) {
        sum += horizontal[pixel];
    }
    if (y > 0) {
        sum += vertical[pixel - width];
    }
    if (y + 1 < height) {
        sum += vertical[pixel];
    }
    return sum * (1.f / 8.f);
}

struct CgOutcome final {
    bool success = true;
    std::string message;
    int iterations = 0;
    double initialResidual = 0.0;
    double finalRelativeResidual = 0.0;
    std::vector<double> history;
};

using Operator = std::function<void(const Vector &, Vector &)>;

CgOutcome solvePcg(
    const Operator &apply,
    const Vector &rightHandSide,
    const Vector &inverseDiagonal,
    Vector &solution,
    int maximumIterations,
    double relativeTolerance)
{
    CgOutcome outcome;
    const std::size_t count = solution.size();
    Vector residual(count);
    Vector preconditioned(count);
    Vector direction(count);
    Vector applied(count);
    apply(solution, applied);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (std::ptrdiff_t index = 0;
            index < static_cast<std::ptrdiff_t>(count); ++index) {
        residual[index] = rightHandSide[index] - applied[index];
        preconditioned[index] = inverseDiagonal[index] * residual[index];
        direction[index] = preconditioned[index];
    }

    const double initialSquared = deterministicDot(residual, residual);
    if (!std::isfinite(initialSquared) || initialSquared < 0.0) {
        outcome.success = false;
        outcome.message = "initial PCG residual is non-finite";
        return outcome;
    }
    outcome.initialResidual = std::sqrt(initialSquared);
    outcome.history.push_back(outcome.initialResidual == 0.0 ? 0.0 : 1.0);
    if (outcome.initialResidual == 0.0) {
        return outcome;
    }

    double rho = deterministicDot(residual, preconditioned);
    if (!(rho > 0.0) || !std::isfinite(rho)) {
        outcome.success = false;
        outcome.message = "PCG preconditioned residual is not positive";
        return outcome;
    }

    for (int iteration = 0; iteration < maximumIterations; ++iteration) {
        apply(direction, applied);
        const double denominator = deterministicDot(direction, applied);
        if (!(denominator > 0.0) || !std::isfinite(denominator)) {
            outcome.success = false;
            outcome.message = "PCG operator is not numerically positive definite";
            return outcome;
        }
        const double alpha = rho / denominator;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (std::ptrdiff_t index = 0;
                index < static_cast<std::ptrdiff_t>(count); ++index) {
            solution[index] += static_cast<float>(alpha * direction[index]);
            residual[index] -= static_cast<float>(alpha * applied[index]);
        }
        const double squared = deterministicDot(residual, residual);
        if (!std::isfinite(squared) || squared < 0.0) {
            outcome.success = false;
            outcome.message = "PCG residual became non-finite";
            return outcome;
        }
        outcome.iterations = iteration + 1;
        outcome.finalRelativeResidual = std::sqrt(squared) / outcome.initialResidual;
        outcome.history.push_back(outcome.finalRelativeResidual);
        if (outcome.finalRelativeResidual <= relativeTolerance) {
            break;
        }

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (std::ptrdiff_t index = 0;
                index < static_cast<std::ptrdiff_t>(count); ++index) {
            preconditioned[index] = inverseDiagonal[index] * residual[index];
        }
        const double nextRho = deterministicDot(residual, preconditioned);
        if (!(nextRho > 0.0) || !std::isfinite(nextRho)) {
            outcome.success = false;
            outcome.message = "PCG recurrence became non-positive";
            return outcome;
        }
        const double beta = nextRho / rho;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (std::ptrdiff_t index = 0;
                index < static_cast<std::ptrdiff_t>(count); ++index) {
            direction[index] = preconditioned[index] +
                static_cast<float>(beta * direction[index]);
        }
        rho = nextRho;
    }
    return outcome;
}

void appendOutcome(GlobalXTransRunResult &result, const CgOutcome &outcome)
{
    result.completedIterations += outcome.iterations;
    if (result.residualHistory.empty()) {
        result.initialRelativeResidual = outcome.history.empty() ? 0.0 : outcome.history.front();
    }
    result.finalRelativeResidual = outcome.finalRelativeResidual;
    result.residualHistory.insert(
        result.residualHistory.end(), outcome.history.begin(), outcome.history.end());
}

bool makeInitialization(
    const Vector &mosaic,
    Vector &rgb,
    int width,
    int height,
    const int cfa[6][6],
    GlobalXTransInitialization initialization,
    TriangulatedXTransVariant triangulatedVariant,
    std::string &message)
{
    const std::size_t pixels = mosaic.size();
    rgb.assign(pixels * 3, 0.f);
    if (initialization == GlobalXTransInitialization::TRIANGULATED) {
        const TriangulatedXTransRunResult run =
            demosaicTriangulatedXTransReference(
                mosaic.data(), rgb.data(), rgb.data() + pixels,
                rgb.data() + 2 * pixels, width, height, cfa,
                triangulatedVariant);
        if (!run) {
            message = "triangulated initialization failed: " + run.message;
            return false;
        }
    } else {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                const std::size_t pixel = static_cast<std::size_t>(y) * width + x;
                rgb[static_cast<std::size_t>(cfa[y % 6][x % 6]) * pixels + pixel] =
                    mosaic[pixel];
            }
        }
    }
    for (float &value : rgb) {
        value = static_cast<float>(value / RAW_SCALE);
    }
    return true;
}

bool solveIndependent(
    const Vector &mosaic,
    const Vector &initialRgb,
    Vector &rgb,
    int width,
    int height,
    const int cfa[6][6],
    const GlobalXTransOptions &options,
    GlobalXTransRunResult &result)
{
    const std::size_t pixels = mosaic.size();
    rgb = initialRgb;
    Vector scratch(pixels * 2);
    for (int color = 0; color < 3; ++color) {
        Vector solution(
            rgb.begin() + static_cast<std::ptrdiff_t>(color * pixels),
            rgb.begin() + static_cast<std::ptrdiff_t>((color + 1) * pixels));
        Vector rightHandSide(pixels, 0.f);
        Vector inverseDiagonal(pixels);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                const std::size_t pixel = static_cast<std::size_t>(y) * width + x;
                const bool measured = cfa[y % 6][x % 6] == color;
                rightHandSide[pixel] = measured
                    ? static_cast<float>(mosaic[pixel] / RAW_SCALE) : 0.f;
                const float diagonal = (measured ? 1.f : 0.f) +
                    static_cast<float>(options.lambdaRgb) * spectralDiagonal(
                        x, y, width, height, options.spectralExponent);
                inverseDiagonal[pixel] = 1.f / std::max(diagonal, MINIMUM_DIAGONAL);
            }
        }
        const Operator apply = [&](const Vector &input, Vector &output) {
            std::fill(output.begin(), output.end(), 0.f);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    const std::size_t pixel = static_cast<std::size_t>(y) * width + x;
                    if (cfa[y % 6][x % 6] == color) {
                        output[pixel] = input[pixel];
                    }
                }
            }
            addSpectralPenalty(
                input.data(), output.data(), scratch.data(), width, height,
                options.spectralExponent, static_cast<float>(options.lambdaRgb));
        };
        const CgOutcome outcome = solvePcg(
            apply, rightHandSide, inverseDiagonal, solution,
            options.maximumIterations, options.relativeTolerance);
        appendOutcome(result, outcome);
        if (!outcome.success) {
            result.message = outcome.message;
            return false;
        }
        std::copy(solution.begin(), solution.end(),
                  rgb.begin() + static_cast<std::ptrdiff_t>(color * pixels));
    }
    result.workingBufferBytes = std::max<std::uint64_t>(
        result.workingBufferBytes,
        static_cast<std::uint64_t>((pixels * 16) * sizeof(float)));
    return true;
}

void buildDifferenceRightHandSide(
    const Vector &mosaic,
    Vector &rightHandSide,
    int width,
    int height,
    const int cfa[6][6])
{
    const std::size_t pixels = mosaic.size();
    rightHandSide.assign(pixels * 3, 0.f);
    float *const green = rightHandSide.data();
    float *const redDifference = green + pixels;
    float *const blueDifference = redDifference + pixels;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const std::size_t pixel = static_cast<std::size_t>(y) * width + x;
            const float value = static_cast<float>(mosaic[pixel] / RAW_SCALE);
            const int color = cfa[y % 6][x % 6];
            green[pixel] = value;
            if (color == 0) {
                redDifference[pixel] = value;
            } else if (color == 2) {
                blueDifference[pixel] = value;
            }
        }
    }
}

void initializeDifferences(const Vector &rgb, Vector &variables, std::size_t pixels)
{
    variables.resize(pixels * 3);
    const float *const red = rgb.data();
    const float *const green = red + pixels;
    const float *const blue = green + pixels;
    float *const outputGreen = variables.data();
    float *const redDifference = outputGreen + pixels;
    float *const blueDifference = redDifference + pixels;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (std::ptrdiff_t index = 0;
            index < static_cast<std::ptrdiff_t>(pixels); ++index) {
        outputGreen[index] = green[index];
        redDifference[index] = red[index] - green[index];
        blueDifference[index] = blue[index] - green[index];
    }
}

void addDifferenceDataOperator(
    const Vector &input,
    Vector &output,
    int width,
    int height,
    const int cfa[6][6])
{
    const std::size_t pixels = static_cast<std::size_t>(width) * height;
    const float *const green = input.data();
    const float *const redDifference = green + pixels;
    const float *const blueDifference = redDifference + pixels;
    float *const outputGreen = output.data();
    float *const outputRedDifference = outputGreen + pixels;
    float *const outputBlueDifference = outputRedDifference + pixels;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const std::size_t pixel = static_cast<std::size_t>(y) * width + x;
            const int color = cfa[y % 6][x % 6];
            if (color == 0) {
                const float value = green[pixel] + redDifference[pixel];
                outputGreen[pixel] = value;
                outputRedDifference[pixel] = value;
            } else if (color == 1) {
                outputGreen[pixel] = green[pixel];
            } else {
                const float value = green[pixel] + blueDifference[pixel];
                outputGreen[pixel] = value;
                outputBlueDifference[pixel] = value;
            }
        }
    }
}

bool solveDifferences(
    const Vector &mosaic,
    Vector &variables,
    int width,
    int height,
    const int cfa[6][6],
    const GlobalXTransOptions &options,
    int maximumIterations,
    const Vector *horizontalWeights,
    const Vector *verticalWeights,
    GlobalXTransRunResult &result)
{
    const std::size_t pixels = mosaic.size();
    Vector rightHandSide;
    buildDifferenceRightHandSide(mosaic, rightHandSide, width, height, cfa);
    Vector inverseDiagonal(pixels * 3);
    const float lambdaGreen = static_cast<float>(options.lambdaGreen);
    const float lambdaChroma = static_cast<float>(options.lambdaChroma);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const std::size_t pixel = static_cast<std::size_t>(y) * width + x;
            const int color = cfa[y % 6][x % 6];
            const float greenRegularizer = horizontalWeights
                ? weightedGreenDiagonal(
                    pixel, x, y, width, height,
                    *horizontalWeights, *verticalWeights)
                : spectralDiagonal(x, y, width, height, options.spectralExponent);
            const float chromaRegularizer = spectralDiagonal(
                x, y, width, height, options.spectralExponent);
            inverseDiagonal[pixel] = 1.f / std::max(
                1.f + lambdaGreen * greenRegularizer, MINIMUM_DIAGONAL);
            inverseDiagonal[pixels + pixel] = 1.f / std::max(
                (color == 0 ? 1.f : 0.f) + lambdaChroma * chromaRegularizer,
                MINIMUM_DIAGONAL);
            inverseDiagonal[2 * pixels + pixel] = 1.f / std::max(
                (color == 2 ? 1.f : 0.f) + lambdaChroma * chromaRegularizer,
                MINIMUM_DIAGONAL);
        }
    }

    Vector scratch(pixels * 2);
    const Operator apply = [&](const Vector &input, Vector &output) {
        std::fill(output.begin(), output.end(), 0.f);
        addDifferenceDataOperator(input, output, width, height, cfa);
        if (horizontalWeights) {
            addWeightedGreenPenalty(
                input.data(), output.data(), *horizontalWeights, *verticalWeights,
                width, height, lambdaGreen);
        } else {
            addSpectralPenalty(
                input.data(), output.data(), scratch.data(), width, height,
                options.spectralExponent, lambdaGreen);
        }
        addSpectralPenalty(
            input.data() + pixels, output.data() + pixels, scratch.data(),
            width, height, options.spectralExponent, lambdaChroma);
        addSpectralPenalty(
            input.data() + 2 * pixels, output.data() + 2 * pixels, scratch.data(),
            width, height, options.spectralExponent, lambdaChroma);
    };
    const CgOutcome outcome = solvePcg(
        apply, rightHandSide, inverseDiagonal, variables,
        maximumIterations, options.relativeTolerance);
    appendOutcome(result, outcome);
    result.workingBufferBytes = std::max<std::uint64_t>(
        result.workingBufferBytes,
        static_cast<std::uint64_t>((pixels * 26) * sizeof(float)));
    if (!outcome.success) {
        result.message = outcome.message;
        return false;
    }
    return true;
}

void differencesToRgb(const Vector &variables, Vector &rgb, std::size_t pixels)
{
    rgb.resize(pixels * 3);
    const float *const green = variables.data();
    const float *const redDifference = green + pixels;
    const float *const blueDifference = redDifference + pixels;
    float *const red = rgb.data();
    float *const outputGreen = red + pixels;
    float *const blue = outputGreen + pixels;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (std::ptrdiff_t index = 0;
            index < static_cast<std::ptrdiff_t>(pixels); ++index) {
        red[index] = green[index] + redDifference[index];
        outputGreen[index] = green[index];
        blue[index] = green[index] + blueDifference[index];
    }
}

void measureAndProjectSamples(
    const Vector &mosaic,
    Vector &rgb,
    int width,
    int height,
    const int cfa[6][6],
    GlobalXTransRunResult &result)
{
    const std::size_t pixels = mosaic.size();
    double squared = 0.0;
    double maximum = 0.0;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const std::size_t pixel = static_cast<std::size_t>(y) * width + x;
            const int color = cfa[y % 6][x % 6];
            const std::size_t output = static_cast<std::size_t>(color) * pixels + pixel;
            const double measured = mosaic[pixel] / RAW_SCALE;
            const double difference = rgb[output] - measured;
            squared += difference * difference;
            maximum = std::max(maximum, std::fabs(difference));
            // Exact final data constraint in RawTherapee's native scalar domain.
            rgb[output] = static_cast<float>(mosaic[pixel] / RAW_SCALE);
        }
    }
    result.preProjectionSampleMaximum = maximum;
    result.preProjectionSampleRms = std::sqrt(squared / pixels);
    result.finalSampleMaximum = 0.0;
    result.finalSampleRms = 0.0;
}

bool allFinite(const Vector &values)
{
    int invalid = 0;
#ifdef _OPENMP
#pragma omp parallel for reduction(|:invalid) schedule(static)
#endif
    for (std::ptrdiff_t index = 0;
            index < static_cast<std::ptrdiff_t>(values.size()); ++index) {
        invalid |= !std::isfinite(values[index]);
    }
    return !invalid;
}

bool writeFloatPlane(const std::string &path, const float *values, std::size_t count)
{
    std::ofstream stream(path.c_str(), std::ios::binary | std::ios::trunc);
    if (!stream) {
        return false;
    }
    for (std::size_t index = 0; index < count; ++index) {
        std::uint32_t bits = 0;
        std::memcpy(&bits, values + index, sizeof(bits));
        const unsigned char encoded[4] = {
            static_cast<unsigned char>(bits),
            static_cast<unsigned char>(bits >> 8),
            static_cast<unsigned char>(bits >> 16),
            static_cast<unsigned char>(bits >> 24)
        };
        stream.write(reinterpret_cast<const char *>(encoded), sizeof(encoded));
    }
    return static_cast<bool>(stream);
}

bool writeDiagnostics(
    const GlobalXTransOptions &options,
    const Vector &rgb,
    const Vector *variables,
    std::size_t pixels,
    const GlobalXTransRunResult &result)
{
    if (options.debugOutputPrefix.empty()) {
        return true;
    }
    if (!writeFloatPlane(options.debugOutputPrefix + "-r.f32", rgb.data(), pixels) ||
            !writeFloatPlane(options.debugOutputPrefix + "-g.f32", rgb.data() + pixels, pixels) ||
            !writeFloatPlane(options.debugOutputPrefix + "-b.f32", rgb.data() + 2 * pixels, pixels)) {
        return false;
    }
    if (variables &&
            (!writeFloatPlane(options.debugOutputPrefix + "-dr.f32", variables->data() + pixels, pixels) ||
             !writeFloatPlane(options.debugOutputPrefix + "-db.f32", variables->data() + 2 * pixels, pixels))) {
        return false;
    }
    std::ofstream residuals((options.debugOutputPrefix + "-residual.csv").c_str(),
                            std::ios::trunc);
    if (!residuals) {
        return false;
    }
    residuals << "record,relative_residual\n";
    for (std::size_t index = 0; index < result.residualHistory.size(); ++index) {
        residuals << index << ',' << result.residualHistory[index] << '\n';
    }
    return static_cast<bool>(residuals);
}

GlobalXTransRunResult runWhole(
    const Vector &mosaic,
    Vector &rgb,
    int width,
    int height,
    const int cfa[6][6],
    GlobalXTransVariant variant,
    const GlobalXTransOptions &options)
{
    GlobalXTransRunResult result;
    const Clock::time_point totalStarted = Clock::now();
    const Clock::time_point setupStarted = Clock::now();
    Vector initialization;
    if (!makeInitialization(
            mosaic, initialization, width, height, cfa,
            options.initialization,
            variant == GlobalXTransVariant::INDEPENDENT_RGB
                ? TriangulatedXTransVariant::INDEPENDENT_RGB
                : TriangulatedXTransVariant::GREEN_CHROMA_DIFFERENCE,
            result.message)) {
        result.code = GlobalXTransErrorCode::INTERNAL;
        return result;
    }
    result.setupMicroseconds = microsecondsSince(setupStarted);
    const Clock::time_point solverStarted = Clock::now();
    Vector variables;
    bool solved = false;
    if (variant == GlobalXTransVariant::INDEPENDENT_RGB) {
        solved = solveIndependent(
            mosaic, initialization, rgb, width, height, cfa, options, result);
    } else {
        initializeDifferences(initialization, variables, mosaic.size());
        solved = solveDifferences(
            mosaic, variables, width, height, cfa, options,
            options.maximumIterations, nullptr, nullptr, result);
        if (solved && variant == GlobalXTransVariant::EDGE_PRESERVING_GREEN) {
            Vector horizontal;
            Vector vertical;
            for (int outer = 0; outer < options.edgeOuterIterations; ++outer) {
                formEdgeWeights(
                    variables.data(), horizontal, vertical, width, height,
                    static_cast<float>(options.charbonnierEpsilon));
                solved = solveDifferences(
                    mosaic, variables, width, height, cfa, options,
                    options.edgeInnerIterations, &horizontal, &vertical, result);
                if (!solved) {
                    break;
                }
                ++result.completedOuterIterations;
            }
            result.workingBufferBytes = std::max<std::uint64_t>(
                result.workingBufferBytes,
                static_cast<std::uint64_t>((mosaic.size() * 28) * sizeof(float)));
        }
        if (solved) {
            differencesToRgb(variables, rgb, mosaic.size());
        }
    }
    result.solverMicroseconds = microsecondsSince(solverStarted);
    if (!solved) {
        result.code = GlobalXTransErrorCode::SOLVER;
        result.totalMicroseconds = microsecondsSince(totalStarted);
        return result;
    }
    if (!allFinite(rgb)) {
        result.code = GlobalXTransErrorCode::NONFINITE;
        result.message = "global reconstruction produced a non-finite value";
        result.totalMicroseconds = microsecondsSince(totalStarted);
        return result;
    }
    measureAndProjectSamples(mosaic, rgb, width, height, cfa, result);
    if (!writeDiagnostics(
            options, rgb, variables.empty() ? nullptr : &variables,
            mosaic.size(), result)) {
        result.code = GlobalXTransErrorCode::IO;
        result.message = "could not write requested global-reconstruction diagnostics";
        result.totalMicroseconds = microsecondsSince(totalStarted);
        return result;
    }
    result.tileCount = 1;
    result.totalMicroseconds = microsecondsSince(totalStarted);
    return result;
}

std::vector<int> tileOrigins(int length, int tileSize)
{
    if (length <= tileSize) {
        return std::vector<int>(1, 0);
    }
    const int stride = tileSize / 2;
    std::vector<int> result;
    for (int origin = 0; origin + tileSize < length; origin += stride) {
        result.push_back(origin);
    }
    const int finalOrigin = length - tileSize;
    if (result.empty() || result.back() != finalOrigin) {
        result.push_back(finalOrigin);
    }
    return result;
}

float tileAxisWeight(int coordinate, int length, bool hasBefore, bool hasAfter, int overlap)
{
    float weight = 1.f;
    if (hasBefore && coordinate < overlap) {
        weight = std::min(weight, static_cast<float>(coordinate + 1) / (overlap + 1));
    }
    if (hasAfter && coordinate >= length - overlap) {
        weight = std::min(weight, static_cast<float>(length - coordinate) / (overlap + 1));
    }
    return weight;
}

GlobalXTransRunResult runTiled(
    const Vector &mosaic,
    Vector &rgb,
    int width,
    int height,
    const int cfa[6][6],
    GlobalXTransVariant variant,
    const GlobalXTransOptions &options)
{
    const Clock::time_point started = Clock::now();
    GlobalXTransRunResult result;
    const std::size_t pixels = mosaic.size();
    rgb.assign(pixels * 3, 0.f);
    Vector weights(pixels, 0.f);
    const std::vector<int> originsX = tileOrigins(width, options.tileSize);
    const std::vector<int> originsY = tileOrigins(height, options.tileSize);
    const int overlap = options.tileSize / 2;
    GlobalXTransOptions tileOptions = options;
    tileOptions.tileSize = 0;
    tileOptions.debugOutputPrefix.clear();

    for (const int originY : originsY) {
        for (const int originX : originsX) {
            const int tileWidth = std::min(options.tileSize, width - originX);
            const int tileHeight = std::min(options.tileSize, height - originY);
            const std::size_t tilePixels =
                static_cast<std::size_t>(tileWidth) * tileHeight;
            Vector tileMosaic(tilePixels);
            for (int y = 0; y < tileHeight; ++y) {
                std::copy_n(
                    mosaic.begin() + static_cast<std::ptrdiff_t>(
                        static_cast<std::size_t>(originY + y) * width + originX),
                    tileWidth,
                    tileMosaic.begin() + static_cast<std::ptrdiff_t>(
                        static_cast<std::size_t>(y) * tileWidth));
            }
            int tileCfa[6][6];
            for (int y = 0; y < 6; ++y) {
                for (int x = 0; x < 6; ++x) {
                    tileCfa[y][x] = cfa[(originY + y) % 6][(originX + x) % 6];
                }
            }
            Vector tileRgb;
            GlobalXTransRunResult tile = runWhole(
                tileMosaic, tileRgb, tileWidth, tileHeight,
                tileCfa, variant, tileOptions);
            if (!tile) {
                return tile;
            }
            result.setupMicroseconds += tile.setupMicroseconds;
            result.solverMicroseconds += tile.solverMicroseconds;
            result.completedIterations += tile.completedIterations;
            result.completedOuterIterations += tile.completedOuterIterations;
            result.workingBufferBytes = std::max(
                result.workingBufferBytes, tile.workingBufferBytes +
                static_cast<std::uint64_t>((pixels * 4) * sizeof(float)));
            result.residualHistory.insert(
                result.residualHistory.end(), tile.residualHistory.begin(),
                tile.residualHistory.end());
            ++result.tileCount;

            const bool hasLeft = originX > 0;
            const bool hasRight = originX + tileWidth < width;
            const bool hasTop = originY > 0;
            const bool hasBottom = originY + tileHeight < height;
            for (int y = 0; y < tileHeight; ++y) {
                const float wy = tileAxisWeight(
                    y, tileHeight, hasTop, hasBottom, overlap);
                for (int x = 0; x < tileWidth; ++x) {
                    const float weight = wy * tileAxisWeight(
                        x, tileWidth, hasLeft, hasRight, overlap);
                    const std::size_t source = static_cast<std::size_t>(y) * tileWidth + x;
                    const std::size_t destination =
                        static_cast<std::size_t>(originY + y) * width + originX + x;
                    for (int color = 0; color < 3; ++color) {
                        rgb[static_cast<std::size_t>(color) * pixels + destination] +=
                            weight * tileRgb[static_cast<std::size_t>(color) * tilePixels + source];
                    }
                    weights[destination] += weight;
                }
            }
        }
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (std::ptrdiff_t pixel = 0;
            pixel < static_cast<std::ptrdiff_t>(pixels); ++pixel) {
        const float weight = weights[pixel];
        for (int color = 0; color < 3; ++color) {
            rgb[static_cast<std::size_t>(color) * pixels + pixel] /= weight;
        }
    }
    if (!allFinite(rgb)) {
        result.code = GlobalXTransErrorCode::NONFINITE;
        result.message = "tiled global reconstruction produced a non-finite value";
        return result;
    }
    measureAndProjectSamples(mosaic, rgb, width, height, cfa, result);
    if (!writeDiagnostics(options, rgb, nullptr, pixels, result)) {
        result.code = GlobalXTransErrorCode::IO;
        result.message = "could not write requested tiled diagnostics";
        return result;
    }
    result.totalMicroseconds = microsecondsSince(started);
    return result;
}

GlobalXTransRunResult run(
    const Vector &mosaic,
    Vector &rgb,
    int width,
    int height,
    const int cfa[6][6],
    GlobalXTransVariant variant,
    const GlobalXTransOptions &options)
{
    GlobalXTransRunResult result;
    if (!validOptions(options)) {
        result.code = GlobalXTransErrorCode::PARAMETER;
        result.message = "invalid global X-Trans solver parameter";
        return result;
    }
    if (variant != GlobalXTransVariant::INDEPENDENT_RGB &&
            variant != GlobalXTransVariant::GREEN_COLOR_DIFFERENCE &&
            variant != GlobalXTransVariant::EDGE_PRESERVING_GREEN) {
        result.code = GlobalXTransErrorCode::INTERNAL;
        result.message = "unknown global X-Trans variant";
        return result;
    }
    XTransCfaTransform transform;
    if (!findCanonicalXTransTransform(cfa, transform)) {
        result.code = GlobalXTransErrorCode::CFA;
        result.message = "unsupported 6x6 X-Trans CFA matrix";
        return result;
    }
    int colorCounts[3] = {0, 0, 0};
    int invalid = 0;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const std::size_t pixel = static_cast<std::size_t>(y) * width + x;
            invalid |= !std::isfinite(mosaic[pixel]);
            ++colorCounts[cfa[y % 6][x % 6]];
        }
    }
    if (invalid) {
        result.code = GlobalXTransErrorCode::NONFINITE;
        result.message = "mosaic contains a non-finite sample";
        return result;
    }
    if (!colorCounts[0] || !colorCounts[1] || !colorCounts[2]) {
        result.code = GlobalXTransErrorCode::SIZE;
        result.message = "finite image does not contain every measured CFA color";
        return result;
    }
    return options.tileSize == 0
        ? runWhole(mosaic, rgb, width, height, cfa, variant, options)
        : runTiled(mosaic, rgb, width, height, cfa, variant, options);
}

} // namespace

const char *globalXTransErrorCodeName(GlobalXTransErrorCode code)
{
    switch (code) {
        case GlobalXTransErrorCode::NONE: return "NONE";
        case GlobalXTransErrorCode::SIZE: return "SIZE";
        case GlobalXTransErrorCode::CFA: return "CFA";
        case GlobalXTransErrorCode::PARAMETER: return "PARAMETER";
        case GlobalXTransErrorCode::ALLOCATION: return "ALLOCATION";
        case GlobalXTransErrorCode::NONFINITE: return "NONFINITE";
        case GlobalXTransErrorCode::SOLVER: return "SOLVER";
        case GlobalXTransErrorCode::IO: return "IO";
        case GlobalXTransErrorCode::INTERNAL: return "INTERNAL";
    }
    return "INTERNAL";
}

GlobalXTransRunResult demosaicGlobalXTransReference(
    const float *mosaic,
    float *red,
    float *green,
    float *blue,
    int width,
    int height,
    const int xtrans[6][6],
    GlobalXTransVariant variant,
    const GlobalXTransOptions &options)
{
    GlobalXTransRunResult result;
    std::size_t pixels = 0;
    if (!mosaic || !red || !green || !blue) {
        result.code = GlobalXTransErrorCode::INTERNAL;
        result.message = "input and output pointers must not be null";
        return result;
    }
    if (!checkedPixelCount(width, height, pixels)) {
        result.code = GlobalXTransErrorCode::SIZE;
        result.message = "image dimensions are invalid or overflow";
        return result;
    }
    try {
        const Vector input(mosaic, mosaic + pixels);
        Vector output;
        result = run(input, output, width, height, xtrans, variant, options);
        if (!result) {
            return result;
        }
        for (std::size_t pixel = 0; pixel < pixels; ++pixel) {
            red[pixel] = output[pixel] * static_cast<float>(RAW_SCALE);
            green[pixel] = output[pixels + pixel] * static_cast<float>(RAW_SCALE);
            blue[pixel] = output[2 * pixels + pixel] * static_cast<float>(RAW_SCALE);
            const int color = xtrans[(pixel / width) % 6][(pixel % width) % 6];
            (color == 0 ? red : color == 1 ? green : blue)[pixel] = mosaic[pixel];
        }
    } catch (const std::bad_alloc &) {
        result = GlobalXTransRunResult();
        result.code = GlobalXTransErrorCode::ALLOCATION;
        result.message = "could not allocate global reconstruction workspace";
    }
    return result;
}

GlobalXTransRunResult demosaicGlobalXTrans(
    const array2D<float> &rawData,
    array2D<float> &red,
    array2D<float> &green,
    array2D<float> &blue,
    int width,
    int height,
    const int xtrans[6][6],
    GlobalXTransVariant variant,
    const GlobalXTransOptions &options)
{
    GlobalXTransRunResult result;
    std::size_t pixels = 0;
    if (!checkedPixelCount(width, height, pixels)) {
        result.code = GlobalXTransErrorCode::SIZE;
        result.message = "image dimensions are invalid or overflow";
        return result;
    }
    try {
        Vector mosaic(pixels);
        for (int y = 0; y < height; ++y) {
            std::copy_n(rawData[y], width,
                mosaic.begin() + static_cast<std::ptrdiff_t>(
                    static_cast<std::size_t>(y) * width));
        }
        Vector output;
        result = run(mosaic, output, width, height, xtrans, variant, options);
        if (!result) {
            return result;
        }
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                const std::size_t pixel = static_cast<std::size_t>(y) * width + x;
                red[y][x] = output[pixel] * static_cast<float>(RAW_SCALE);
                green[y][x] = output[pixels + pixel] * static_cast<float>(RAW_SCALE);
                blue[y][x] = output[2 * pixels + pixel] * static_cast<float>(RAW_SCALE);
                const int color = xtrans[y % 6][x % 6];
                (color == 0 ? red : color == 1 ? green : blue)[y][x] = rawData[y][x];
            }
        }
    } catch (const std::bad_alloc &) {
        result = GlobalXTransRunResult();
        result.code = GlobalXTransErrorCode::ALLOCATION;
        result.message = "could not allocate global reconstruction workspace";
    }
    return result;
}

} // namespace rtengine
