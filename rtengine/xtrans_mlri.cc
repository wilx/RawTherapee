/*
 * Two-pass residual-interpolation demosaicking for Fujifilm X-Trans.
 *
 * Research basis:
 *   D. Kiku, Y. Monno, M. Tanaka, and M. Okutomi,
 *   "Minimized-Laplacian Residual Interpolation for Color Image
 *   Demosaicking," Proc. SPIE 9023, 2014, DOI 10.1117/12.2038425.
 *
 *   D. Kiku, Y. Monno, M. Tanaka, and M. Okutomi,
 *   "Beyond Color Difference: Residual Interpolation for Color Image
 *   Demosaicking," IEEE TIP 25(3), 2016,
 *   DOI 10.1109/TIP.2016.2518082.
 *
 * X-Trans engineering reference:
 *   Rainbow-Johnny-Johnny-Image-Processing-Lim,
 *   "Unified Laplacian Residual Interpolation Demosaicing," MATLAB File
 *   Exchange 1.0.0, 2025.  BSD-3-Clause; see
 *   licenses/MLRI_XTRANS_MATLAB_LICENSE.
 *
 * The Tokyo Tech papers do not publish an X-Trans demosaicer.  This file is
 * an independently structured C++ implementation of the later 6x6
 * generalization, preserving its reviewed two-pass numerical contract.  The
 * provenance of every stage is documented in devnotes/xtrans-mlri-design.md.
 */

#include "xtrans_mlri.h"

#include "xtrans_cfa.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <initializer_list>
#include <limits>
#include <new>
#include <stdexcept>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace rtengine
{
namespace
{

constexpr float SOURCE_MAX = 255.f;
constexpr float SOURCE_SCALE = 257.f;
constexpr float EPSILON = 1e-2f;
constexpr int CORE_SIZE = 384;

// The source is finite-support throughout.  Radius propagation through the
// two complete green passes and the final R/B reconstruction gives 227 pixels;
// round up to a CFA-period boundary. Tests compare this tiled path with the
// untiled reference using high-contrast deterministic data across the core
// boundary.
constexpr int TILE_HALO = 228;

bool usesCorrectedBlueDiagonalGuides(MlriXTransVariant variant)
{
    return variant != MlriXTransVariant::MATLAB_REFERENCE;
}

bool usesUniformCoefficientAverage(MlriXTransVariant variant)
{
    return variant == MlriXTransVariant::PAPER_CORE_2014;
}

bool isPaperCore(MlriXTransVariant variant)
{
    return variant == MlriXTransVariant::PAPER_CORE_2014 ||
           variant == MlriXTransVariant::PAPER_CORE_2016;
}

bool usesFinalOnly(MlriXTransVariant variant)
{
    return variant == MlriXTransVariant::CORRECTED_BLUE_DIAGONAL_GUIDES_FINAL_ONLY;
}

// MATLAB's phase numbering, converted to zero-based indices.  The cell is a
// three-row translation of CANONICAL_XTRANS_CFA.
constexpr int MLRI_PHASE[6][6] = {
    { 0, 10,  1,  5, 14,  6},
    {16,  2, 17, 12,  7, 13},
    { 3, 11,  4,  8, 15,  9},
    { 5, 14,  6,  0, 10,  1},
    {12,  7, 13, 16,  2, 17},
    { 8, 15,  9,  3, 11,  4}
};

constexpr int MLRI_CFA[6][6] = {
    {1, 0, 1, 1, 2, 1},
    {2, 1, 2, 0, 1, 0},
    {1, 0, 1, 1, 2, 1},
    {1, 2, 1, 1, 0, 1},
    {0, 1, 0, 2, 1, 2},
    {1, 2, 1, 1, 0, 1}
};

struct Plane final {
    int width = 0;
    int height = 0;
    std::vector<float> values;

    Plane() = default;

    Plane(int w, int h, float value = 0.f) :
        width(w),
        height(h),
        values(static_cast<std::size_t>(w) * static_cast<std::size_t>(h), value)
    {
    }

    float &operator()(int x, int y)
    {
        return values[static_cast<std::size_t>(y) * width + x];
    }

    float operator()(int x, int y) const
    {
        return values[static_cast<std::size_t>(y) * width + x];
    }

    float zeroExtended(int x, int y) const
    {
        return x >= 0 && x < width && y >= 0 && y < height ? (*this)(x, y) : 0.f;
    }
};

void requireSame(const Plane &a, const Plane &b)
{
    if (a.width != b.width || a.height != b.height) {
        throw std::logic_error("MLRI plane shape mismatch");
    }
}

template<typename Function>
Plane binary(const Plane &a, const Plane &b, Function function)
{
    requireSame(a, b);
    Plane output(a.width, a.height);
    for (std::size_t i = 0; i < output.values.size(); ++i) {
        output.values[i] = function(a.values[i], b.values[i]);
    }
    return output;
}

Plane operator+(const Plane &a, const Plane &b)
{
    return binary(a, b, [](float x, float y) { return x + y; });
}

Plane operator-(const Plane &a, const Plane &b)
{
    return binary(a, b, [](float x, float y) { return x - y; });
}

Plane operator*(const Plane &a, const Plane &b)
{
    return binary(a, b, [](float x, float y) { return x * y; });
}

Plane operator/(const Plane &a, const Plane &b)
{
    return binary(a, b, [](float x, float y) { return x / y; });
}

Plane operator+(const Plane &a, float value)
{
    Plane output = a;
    for (float &x : output.values) {
        x += value;
    }
    return output;
}

Plane operator*(const Plane &a, float value)
{
    Plane output = a;
    for (float &x : output.values) {
        x *= value;
    }
    return output;
}

Plane operator*(float value, const Plane &a)
{
    return a * value;
}

Plane &operator+=(Plane &a, const Plane &b)
{
    requireSame(a, b);
    for (std::size_t i = 0; i < a.values.size(); ++i) {
        a.values[i] += b.values[i];
    }
    return a;
}

Plane clip(const Plane &input, float low = 0.f, float high = SOURCE_MAX)
{
    Plane output = input;
    for (float &value : output.values) {
        value = std::max(low, std::min(high, value));
    }
    return output;
}

Plane absolute(const Plane &input)
{
    Plane output = input;
    for (float &value : output.values) {
        value = std::fabs(value);
    }
    return output;
}

Plane squareRoot(const Plane &input)
{
    Plane output = input;
    for (float &value : output.values) {
        value = std::sqrt(std::max(0.f, value));
    }
    return output;
}

Plane oneMinus(const Plane &mask)
{
    Plane output = mask;
    for (float &value : output.values) {
        value = 1.f - value;
    }
    return output;
}

Plane minimum(const Plane &input, float minimumValue)
{
    Plane output = input;
    for (float &value : output.values) {
        value = std::max(value, minimumValue);
    }
    return output;
}

struct Kernel final {
    int width = 0;
    int height = 0;
    std::vector<float> values;

    Kernel() = default;

    Kernel(int w, int h, std::initializer_list<float> source) :
        width(w),
        height(h),
        values(source)
    {
        if (values.size() != static_cast<std::size_t>(w) * h) {
            throw std::logic_error("MLRI kernel shape mismatch");
        }
    }

    Kernel(int w, int h, std::vector<float> source) :
        width(w),
        height(h),
        values(std::move(source))
    {
        if (values.size() != static_cast<std::size_t>(w) * h) {
            throw std::logic_error("MLRI kernel shape mismatch");
        }
    }
};

Kernel horizontal(std::initializer_list<float> values)
{
    return Kernel(static_cast<int>(values.size()), 1, values);
}

Kernel vertical(std::initializer_list<float> values)
{
    return Kernel(1, static_cast<int>(values.size()), values);
}

Kernel transpose(const Kernel &source)
{
    std::vector<float> values(static_cast<std::size_t>(source.width) * source.height);
    for (int y = 0; y < source.height; ++y) {
        for (int x = 0; x < source.width; ++x) {
            values[static_cast<std::size_t>(x) * source.height + y] =
                source.values[static_cast<std::size_t>(y) * source.width + x];
        }
    }
    return Kernel(source.height, source.width, std::move(values));
}

Kernel rotate90(const Kernel &source, int turns)
{
    turns = positiveModulo(turns, 4);
    Kernel result = source;
    for (int turn = 0; turn < turns; ++turn) {
        std::vector<float> values(static_cast<std::size_t>(result.width) * result.height);
        const int newWidth = result.height;
        const int newHeight = result.width;
        for (int y = 0; y < result.height; ++y) {
            for (int x = 0; x < result.width; ++x) {
                // MATLAB rot90() is counter-clockwise for a positive turn.
                const int nx = y;
                const int ny = result.width - 1 - x;
                values[static_cast<std::size_t>(ny) * newWidth + nx] =
                    result.values[static_cast<std::size_t>(y) * result.width + x];
            }
        }
        result = Kernel(newWidth, newHeight, std::move(values));
    }
    return result;
}

Kernel diagonal(std::initializer_list<float> diagonalValues, bool anti = false)
{
    const int size = static_cast<int>(diagonalValues.size());
    std::vector<float> values(static_cast<std::size_t>(size) * size, 0.f);
    int index = 0;
    for (float value : diagonalValues) {
        const int x = anti ? size - 1 - index : index;
        values[static_cast<std::size_t>(index) * size + x] = value;
        ++index;
    }
    return Kernel(size, size, std::move(values));
}

Kernel scaled(const Kernel &source, float factor)
{
    Kernel output = source;
    for (float &value : output.values) {
        value *= factor;
    }
    return output;
}

Plane correlate(const Plane &input, const Kernel &kernel)
{
    Plane output(input.width, input.height);
    const int cx = kernel.width / 2;
    const int cy = kernel.height / 2;
    for (int y = 0; y < input.height; ++y) {
        for (int x = 0; x < input.width; ++x) {
            float sum = 0.f;
            for (int ky = 0; ky < kernel.height; ++ky) {
                for (int kx = 0; kx < kernel.width; ++kx) {
                    const float coefficient =
                        kernel.values[static_cast<std::size_t>(ky) * kernel.width + kx];
                    if (coefficient != 0.f) {
                        sum += input.zeroExtended(x + kx - cx, y + ky - cy) * coefficient;
                    }
                }
            }
            output(x, y) = sum;
        }
    }
    return output;
}

Plane boxSum(const Plane &input, int horizontalRadius, int verticalRadius)
{
    // Keep the reduction order local to each output sample.  A rolling sum
    // carries floating-point round-off from the beginning of a row/column;
    // consequently the same finite-support neighbourhood can produce a
    // different value when it is evaluated in an overlapping tile.  These
    // two separable fixed-order reductions make output depend only on the
    // documented window and therefore keep non-overlapping tile cores
    // seam-identical to an untiled invocation.
    Plane horizontalSum(input.width, input.height);
    for (int y = 0; y < input.height; ++y) {
        for (int x = 0; x < input.width; ++x) {
            float sum = 0.f;
            for (int dx = -horizontalRadius; dx <= horizontalRadius; ++dx) {
                sum += input.zeroExtended(x + dx, y);
            }
            horizontalSum(x, y) = sum;
        }
    }

    Plane output(input.width, input.height);
    for (int x = 0; x < input.width; ++x) {
        for (int y = 0; y < input.height; ++y) {
            float sum = 0.f;
            for (int dy = -verticalRadius; dy <= verticalRadius; ++dy) {
                sum += horizontalSum.zeroExtended(x, y + dy);
            }
            output(x, y) = sum;
        }
    }
    return output;
}

Kernel halfGaussian(float sigma, bool positive, bool verticalDirection = false,
                    bool diagonalDirection = false, bool antiDiagonal = false)
{
    // fspecial() and the subsequent half-kernel normalization are double
    // precision in MATLAB/Octave even though convolution returns single for
    // the single-precision image.  Normalize in double and cast only once.
    std::array<double, 17> gaussian {{}};
    double fullSum = 0.0;
    for (int i = 0; i < 17; ++i) {
        const int offset = i - 8;
        gaussian[static_cast<std::size_t>(i)] =
            std::exp(-(static_cast<double>(offset) * offset) /
                     (2.0 * static_cast<double>(sigma) * sigma));
        fullSum += gaussian[static_cast<std::size_t>(i)];
    }
    double retainedSum = 0.0;
    for (int i = 0; i < 17; ++i) {
        gaussian[static_cast<std::size_t>(i)] /= fullSum;
        const bool retained = positive ? i >= 8 : i <= 8;
        if (!retained) {
            gaussian[static_cast<std::size_t>(i)] = 0.0;
        }
        retainedSum += gaussian[static_cast<std::size_t>(i)];
    }
    std::vector<float> values(17);
    for (int i = 0; i < 17; ++i) {
        values[static_cast<std::size_t>(i)] =
            static_cast<float>(gaussian[static_cast<std::size_t>(i)] / retainedSum);
    }
    Kernel source(17, 1, std::move(values));
    if (diagonalDirection) {
        std::vector<float> diagonalValues = source.values;
        std::vector<float> matrix(17 * 17, 0.f);
        for (int i = 0; i < 17; ++i) {
            const int x = antiDiagonal ? 16 - i : i;
            matrix[static_cast<std::size_t>(i) * 17 + x] = diagonalValues[static_cast<std::size_t>(i)];
        }
        return Kernel(17, 17, std::move(matrix));
    }
    return verticalDirection ? transpose(source) : source;
}

Plane phaseMask(int width, int height, int originX, int originY,
                std::initializer_list<int> phases)
{
    std::array<bool, 18> selected {{false}};
    for (int phase : phases) {
        selected[static_cast<std::size_t>(phase)] = true;
    }
    Plane output(width, height);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const int phase = MLRI_PHASE[positiveModulo(y + originY, 6)]
                                        [positiveModulo(x + originX, 6)];
            output(x, y) = selected[static_cast<std::size_t>(phase)] ? 1.f : 0.f;
        }
    }
    return output;
}

Plane guidedMlri(
    const Plane &guide,
    const Plane &observed,
    const Plane &sampleMask,
    const Plane &laplacianGuide,
    const Plane &laplacianObserved,
    const Plane &laplacianMask,
    int horizontalRadius,
    int verticalRadius,
    MlriXTransVariant variant = MlriXTransVariant::MATLAB_REFERENCE)
{
    // SPIE 2014 equations (1)-(3), expanded as TIP 2016 equations (5)-(7).
    // The gain is fitted
    // in the approximate-Laplacian domain, while the DC component is fitted
    // to the observed samples.  The papers then differ only in how the
    // overlapping local models are combined: the 2014 method averages their
    // coefficients uniformly, while TIP 2016 equations (8)-(10) weight them
    // by inverse local residual MSE.  Keeping that choice here makes the two
    // paper-core variants a controlled comparison.
    Plane count = boxSum(laplacianMask, horizontalRadius, verticalRadius);
    for (float &value : count.values) {
        if (value == 0.f) {
            value = 1.f;
        }
    }
    Plane meanIp = boxSum(laplacianGuide * laplacianObserved * laplacianMask,
                          horizontalRadius, verticalRadius) / count;
    Plane meanII = boxSum(laplacianGuide * laplacianGuide * laplacianMask,
                          horizontalRadius, verticalRadius) / count;
    Plane a = meanIp / (meanII + EPSILON);

    Plane sampleCount = boxSum(sampleMask, horizontalRadius, verticalRadius);
    for (float &value : sampleCount.values) {
        if (value == 0.f) {
            value = 1.f;
        }
    }
    Plane mG = boxSum(guide * sampleMask, horizontalRadius, verticalRadius);
    Plane mR = boxSum(observed * sampleMask, horizontalRadius, verticalRadius);
    Plane b = mR / sampleCount - a * (mG / sampleCount);

    if (usesUniformCoefficientAverage(variant)) {
        // Every pixel is covered by one model centered at each valid window
        // position.  boxSum(ones) is the boundary-truncated overlap count used
        // by the authors' 2014 MATLAB helper; no data-dependent weighting is
        // introduced in this branch.
        const Plane modelCount = boxSum(
            Plane(guide.width, guide.height, 1.f),
            horizontalRadius,
            verticalRadius);
        const Plane meanA = boxSum(a, horizontalRadius, verticalRadius) / modelCount;
        const Plane meanB = boxSum(b, horizontalRadius, verticalRadius) / modelCount;
        return meanA * guide + meanB;
    }

    // TIP 2016 equations (8)-(10): residual MSE, inverse-error model weight,
    // and normalized weighted coefficient average.  The 0.01 floor is the
    // reviewed implementation's fixed numerical regularizer.
    Plane cost = (b * b) * sampleCount + ((a * b) * 2.f) * mG - b * (2.f * mR)
               - boxSum(observed * guide * sampleMask, horizontalRadius, verticalRadius) * (a * 2.f)
               + boxSum(guide * guide * sampleMask, horizontalRadius, verticalRadius) * (a * a)
               + boxSum(observed * observed * sampleMask, horizontalRadius, verticalRadius);
    cost = cost / sampleCount;
    for (float &value : cost.values) {
        value = 1.f / std::max(value, 0.01f);
    }
    Plane weightSum = minimum(boxSum(cost, horizontalRadius, verticalRadius), 0.01f);
    Plane meanA = boxSum(a * cost, horizontalRadius, verticalRadius) / weightSum;
    Plane meanB = boxSum(b * cost, horizontalRadius, verticalRadius) / weightSum;
    return meanA * guide + meanB;
}

Plane inverseSquaredEnergy(const Plane &gradient, const Kernel &support)
{
    Plane energy = correlate(gradient, support);
    for (float &value : energy.values) {
        value = 1.f / (value * value + 0.01f);
    }
    return energy;
}

Kernel westSupport()
{
    return Kernel(9, 3, {
        1,1,1,1,1,0,0,0,0,
        1,1,1,1,1,0,0,0,0,
        1,1,1,1,1,0,0,0,0
    });
}

Kernel eastSupport()
{
    return Kernel(9, 3, {
        0,0,0,0,1,1,1,1,1,
        0,0,0,0,1,1,1,1,1,
        0,0,0,0,1,1,1,1,1
    });
}

Kernel diagonalSupport()
{
    return Kernel(11, 11, {
        0,0,1,0,0,0,0,0,0,0,0,
        0,1,1,1,0,0,0,0,0,0,0,
        1,1,1,1,1,0,0,0,0,0,0,
        0,1,1,1,1,1,0,0,0,0,0,
        0,0,1,1,1,1,1,0,0,0,0,
        0,0,0,1,1,1,0,0,0,0,0,
        0,0,0,0,1,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0
    });
}

bool finite(const Plane &plane)
{
    for (float value : plane.values) {
        if (!std::isfinite(value)) {
            return false;
        }
    }
    return true;
}

struct MlriMasks final {
    Plane p[18];
    Plane green;
    Plane red;
    Plane blue;

    MlriMasks(int width, int height, int originX, int originY)
    {
        for (int phase = 0; phase < 18; ++phase) {
            p[phase] = phaseMask(width, height, originX, originY, {phase});
        }
        green = phaseMask(width, height, originX, originY, {0,1,2,3,4,5,6,7,8,9});
        red = phaseMask(width, height, originX, originY, {10,11,12,13});
        blue = phaseMask(width, height, originX, originY, {14,15,16,17});
    }

    Plane any(std::initializer_list<int> phases) const
    {
        Plane output(p[0].width, p[0].height);
        for (int phase : phases) {
            output += p[phase];
        }
        return output;
    }
};

struct RgbPlanes final {
    Plane red;
    Plane green;
    Plane blue;
};

struct DirectionalGuides final {
    Plane gh, gv, gd, gp;
    Plane rh, rv, rd, rp;
    Plane bh, bv, bd, bp;
};

Plane weightedEight(
    const std::array<Plane, 8> &weights,
    const std::array<Plane, 8> &candidates)
{
    Plane numerator(weights[0].width, weights[0].height);
    Plane denominator(weights[0].width, weights[0].height);
    for (std::size_t i = 0; i < weights.size(); ++i) {
        numerator += weights[i] * candidates[i];
        denominator += weights[i];
    }
    return numerator / denominator;
}

Plane weightedFour(
    const std::array<Plane, 4> &weights,
    const std::array<Plane, 4> &candidates)
{
    Plane numerator(weights[0].width, weights[0].height);
    Plane denominator(weights[0].width, weights[0].height);
    for (std::size_t i = 0; i < weights.size(); ++i) {
        numerator += weights[i] * candidates[i];
        denominator += weights[i];
    }
    return numerator / denominator;
}

std::array<Plane, 8> eightWeights(
    const Plane &horizontalDifference,
    const Plane &verticalDifference,
    const Plane &diagonalDifference,
    const Plane &antiDiagonalDifference)
{
    const Plane gh = absolute(correlate(horizontalDifference, horizontal({1,0,-1})));
    const Plane gv = absolute(correlate(verticalDifference, vertical({1,0,-1})));
    const Plane gd = absolute(correlate(diagonalDifference, diagonal({1,0,-1})));
    const Plane gp = absolute(correlate(antiDiagonalDifference, diagonal({1,0,-1}, true)));
    const Kernel west = westSupport();
    const Kernel east = eastSupport();
    const Kernel north = transpose(west);
    const Kernel south = transpose(east);
    const Kernel a = diagonalSupport();
    return {{
        inverseSquaredEnergy(gv, north),
        inverseSquaredEnergy(gv, south),
        inverseSquaredEnergy(gh, west),
        inverseSquaredEnergy(gh, east),
        inverseSquaredEnergy(gd, a),
        inverseSquaredEnergy(gd, scaled(rotate90(a, 2), 15.f / 23.f)),
        inverseSquaredEnergy(gp, scaled(rotate90(a, 3), 15.f / 23.f)),
        inverseSquaredEnergy(gp, scaled(rotate90(a, 1), 15.f / 23.f))
    }};
}

std::array<Plane, 4> fourWeights(
    const Plane &horizontalDifference,
    const Plane &verticalDifference)
{
    const Plane gh = absolute(correlate(horizontalDifference, horizontal({1,0,-1})));
    const Plane gv = absolute(correlate(verticalDifference, vertical({1,0,-1})));
    const Kernel west = westSupport();
    const Kernel east = eastSupport();
    return {{
        inverseSquaredEnergy(gv, transpose(west)),
        inverseSquaredEnergy(gv, transpose(east)),
        inverseSquaredEnergy(gh, west),
        inverseSquaredEnergy(gh, east)
    }};
}

DirectionalGuides makeGreenGuides(
    const Plane &greenRaw,
    const Plane &redRaw,
    const Plane &blueRaw,
    const MlriMasks &m,
    int pass,
    const RgbPlanes &previous)
{
    // XTRANS_GENERALIZATION: the first pass constructs phase-specific
    // horizontal, vertical, and split-diagonal guides on the irregular 6x6
    // lattice.  The second pass is the source author's N+1 refinement and
    // reuses the first-pass RGB result as every directional guide.
    if (pass > 0) {
        return {
            previous.green, previous.green, previous.green, previous.green,
            previous.red, previous.red, previous.red, previous.red,
            previous.blue, previous.blue, previous.blue, previous.blue
        };
    }

    const Plane m3 = m.p[2];
    const Plane m8 = m.p[7];
    const Plane m11_12 = m.any({10,11});
    const Plane m13_14 = m.any({12,13});
    const Plane m15_16 = m.any({14,15});
    const Plane m17_18 = m.any({16,17});
    const Plane maskY = m.red + m.blue;
    const Plane greenA = m.any({0,2,4,5,7,9});
    const Plane greenB = m.any({1,2,3,6,7,8});

    DirectionalGuides g;
    g.gh = greenRaw
         + m.any({10,11,14,15}) * correlate(greenRaw, horizontal({0.5f,0,0.5f}))
         + m.any({12,16}) * correlate(greenRaw, horizontal({1.f/3,0,0,2.f/3,0}))
         + m.any({13,17}) * correlate(greenRaw, horizontal({0,2.f/3,0,0,1.f/3}));
    g.rh = redRaw
         + m.any({0,1,3,4,5,6,8,9,14,15}) *
             correlate(redRaw, scaled(horizontal({1,2,3,4,5,0,5,4,3,2,1}), 1.f/6))
         + m8 * correlate(redRaw, horizontal({0.5f,0,0.5f}))
         + m3 * correlate(redRaw, horizontal({0.5f,0,0,0,0.5f}))
         + m.p[16] * correlate(redRaw, horizontal({0,0,0.75f,0,0,0,0.25f}))
         + m.p[17] * correlate(redRaw, horizontal({0.25f,0,0,0,0.75f,0,0}));
    g.bh = blueRaw
         + m.any({0,1,3,4,5,6,8,9,10,11}) *
             correlate(blueRaw, scaled(horizontal({1,2,3,4,5,0,5,4,3,2,1}), 1.f/6))
         + m3 * correlate(blueRaw, horizontal({0.5f,0,0.5f}))
         + m8 * correlate(blueRaw, horizontal({0.5f,0,0,0,0.5f}))
         + m.p[12] * correlate(blueRaw, horizontal({0,0,0.75f,0,0,0,0.25f}))
         + m.p[13] * correlate(blueRaw, horizontal({0.25f,0,0,0,0.75f,0,0}));

    g.gv = greenRaw
         + m.any({12,13,16,17}) * correlate(greenRaw, vertical({0.5f,0,0.5f}))
         + m.any({10,14}) * correlate(greenRaw, vertical({1.f/3,0,0,2.f/3,0}))
         + m.any({11,15}) * correlate(greenRaw, vertical({0,2.f/3,0,0,1.f/3}));
    g.rv = redRaw
         + m.any({0,1,3,4,5,6,8,9,16,17}) *
             correlate(redRaw, scaled(vertical({1,2,3,4,5,0,5,4,3,2,1}), 1.f/6))
         + m3 * correlate(redRaw, vertical({0.5f,0,0.5f}))
         + m8 * correlate(redRaw, vertical({0.5f,0,0,0,0.5f}))
         + m.p[14] * correlate(redRaw, vertical({0,0,0.75f,0,0,0,0.25f}))
         + m.p[15] * correlate(redRaw, vertical({0.25f,0,0,0,0.75f,0,0}));
    g.bv = blueRaw
         + m.any({0,1,3,4,5,6,8,9,12,13}) *
             correlate(blueRaw, scaled(vertical({1,2,3,4,5,0,5,4,3,2,1}), 1.f/6))
         + m8 * correlate(blueRaw, vertical({0.5f,0,0.5f}))
         + m3 * correlate(blueRaw, vertical({0.5f,0,0,0,0.5f}))
         + m.p[10] * correlate(blueRaw, vertical({0,0,0.75f,0,0,0,0.25f}))
         + m.p[11] * correlate(blueRaw, vertical({0.25f,0,0,0,0.75f,0,0}));

    const Kernel guideDiag = scaled(diagonal({1,2,0,2,1}), 1.f/3);
    const Kernel guideAnti = scaled(diagonal({1,2,0,2,1}, true), 1.f/3);
    const Kernel cross = Kernel(3, 3, {0,0.25f,0, 0.25f,0,0.25f, 0,0.25f,0});
    g.gd = greenRaw + maskY * correlate(greenRaw, guideDiag);
    g.rd = redRaw + correlate(redRaw, guideDiag);
    g.rd += greenA * correlate(g.rd, cross);
    g.bd = blueRaw + correlate(blueRaw, guideDiag);
    g.bd += greenA * correlate(g.bd, cross);
    g.gp = greenRaw + maskY * correlate(greenRaw, guideAnti);
    g.rp = redRaw + correlate(redRaw, guideAnti);
    g.rp += greenB * correlate(g.rp, cross);
    g.bp = blueRaw + correlate(blueRaw, guideAnti);
    g.bp += greenB * correlate(g.bp, cross);
    return g;
}

Plane interpolateGreen(
    const Plane &mosaic,
    const MlriMasks &m,
    const DirectionalGuides &g,
    float sigma,
    MlriXTransVariant variant)
{
    // ORIGINAL_MLRI supplies each guided tentative estimate below.  Residual
    // interpolation reconstructs the candidate; the eight-direction
    // inverse-energy fusion is an ARI-derived engineering rule rather than
    // the published ARI iteration/termination algorithm.
    const Plane greenRaw = mosaic * m.green;
    const Plane redRaw = mosaic * m.red;
    const Plane blueRaw = mosaic * m.blue;
    const Plane maskGC = m.any({2,7});
    const Plane maskGnC = m.any({0,1,3,4,5,6,8,9});
    const Plane maskY = m.red + m.blue;
    const Plane greenA = m.any({0,2,4,5,7,9});
    const Plane greenB = m.any({1,2,3,6,7,8});
    const Plane m11_12 = m.any({10,11});
    const Plane m13_14 = m.any({12,13});
    const Plane m15_16 = m.any({14,15});
    const Plane m17_18 = m.any({16,17});

    const Kernel lap13h = horizontal({-1,0,0,0,0,0,2,0,0,0,0,0,-1});
    const Kernel lap5h = horizontal({-1,0,2,0,-1});
    const Kernel lap7h = horizontal({-1,0,0,2,0,0,-1});
    const Kernel lap13v = transpose(lap13h);
    const Kernel lap5v = transpose(lap5h);
    const Kernel lap7v = transpose(lap7h);
    const Kernel lap7d = diagonal({-1,0,0,2,0,0,-1});
    const Kernel lap5d = diagonal({-1,0,2,0,-1});
    const Kernel lap7p = diagonal({-1,0,0,2,0,0,-1}, true);
    const Kernel lap5p = diagonal({-1,0,2,0,-1}, true);

    Plane difR = correlate(m11_12 * g.rh, lap13h) + correlate(m.any({2,12,13}) * g.rh, lap5h);
    Plane difG = correlate(m11_12 * g.gh, lap13h) + correlate(m.any({2,12,13}) * g.gh, lap5h);
    Plane tRh = clip(guidedMlri(g.gh, g.rh * m.red, m.red, difG, difR, m.red, 3, 3, variant));
    Plane difB = correlate(m15_16 * g.bh, lap13h) + correlate(m.any({7,16,17}) * g.bh, lap5h);
    Plane difGb = correlate(m15_16 * g.gh, lap13h) + correlate(m.any({7,16,17}) * g.gh, lap5h);
    Plane tBh = clip(guidedMlri(g.gh, g.bh * m.blue, m.blue, difGb, difB, m.blue, 3, 3, variant));
    difG = correlate(maskGC * g.gh, lap7h) + correlate((maskGnC + m11_12 + m15_16) * g.gh, lap5h);
    difR = correlate(maskGC * g.rh, lap7h) + correlate((maskGnC + m11_12 + m15_16) * g.rh, lap5h);
    difB = correlate(maskGC * g.bh, lap7h) + correlate((maskGnC + m11_12 + m15_16) * g.bh, lap5h);
    Plane tGrh = clip(guidedMlri(g.rh, g.gh * m.green, m.green, difR, difG, m.green, 3, 3, variant));
    Plane tGbh = clip(guidedMlri(g.bh, g.gh * m.green, m.green, difB, difG, m.green, 3, 3, variant));

    difR = correlate(m13_14 * g.rv, lap13v) + correlate(m.any({7,10,11}) * g.rv, lap5v);
    difG = correlate(m13_14 * g.gv, lap13v) + correlate(m.any({7,10,11}) * g.gv, lap5v);
    Plane tRv = clip(guidedMlri(g.gv, g.rv * m.red, m.red, difG, difR, m.red, 3, 3, variant));
    difB = correlate(m17_18 * g.bv, lap13v) + correlate(m.any({2,14,15}) * g.bv, lap5v);
    difGb = correlate(m17_18 * g.gv, lap13v) + correlate(m.any({2,14,15}) * g.gv, lap5v);
    Plane tBv = clip(guidedMlri(g.gv, g.bv * m.blue, m.blue, difGb, difB, m.blue, 3, 3, variant));
    difG = correlate(maskGC * g.gv, lap7v) + correlate((maskGnC + m13_14 + m17_18) * g.gv, lap5v);
    difR = correlate(maskGC * g.rv, lap7v) + correlate((maskGnC + m13_14 + m17_18) * g.rv, lap5v);
    difB = correlate(maskGC * g.bv, lap7v) + correlate((maskGnC + m13_14 + m17_18) * g.bv, lap5v);
    Plane tGrv = clip(guidedMlri(g.rv, g.gv * m.green, m.green, difR, difG, m.green, 3, 3, variant));
    Plane tGbv = clip(guidedMlri(g.bv, g.gv * m.green, m.green, difB, difG, m.green, 3, 3, variant));

    difR = correlate(m.red * g.rd, lap7d);
    difG = correlate(m.red * g.gd, lap7d);
    Plane tRd = clip(guidedMlri(g.gd, g.rd * m.red, m.red, difG, difR, m.red, 3, 3, variant));
    // The reviewed v1.0.0 reference forms the blue Laplacian from the red
    // directional guide here (and in the anti-diagonal branch), while the
    // guided samples remain blue.  The corrected experimental variant uses
    // the corresponding blue guides; no other equation or parameter changes.
    const bool correctedBlueDiagonals = usesCorrectedBlueDiagonalGuides(variant);
    difB = correlate(m.blue * (correctedBlueDiagonals ? g.bd : g.rd), lap7d);
    difGb = correlate(m.blue * g.gd, lap7d);
    Plane tBd = clip(guidedMlri(g.gd, g.bd * m.blue, m.blue, difGb, difB, m.blue, 3, 3, variant));
    difG = correlate(m.any({1,3,6,8}) * g.gd, lap7d) + correlate(greenA * g.gd, lap5d);
    difR = correlate(m.any({1,3,6,8}) * g.rd, lap7d) + correlate(greenA * g.rd, lap5d);
    difB = correlate(m.any({1,3,6,8}) * g.bd, lap7d) + correlate(greenA * g.bd, lap5d);
    Plane tGrd = clip(guidedMlri(g.rd, g.gd * m.green, m.green, difR, difG, m.green, 3, 3, variant));
    Plane tGbd = clip(guidedMlri(g.bd, g.gd * m.green, m.green, difB, difG, m.green, 3, 3, variant));

    difR = correlate(m.red * g.rp, lap7p);
    difG = correlate(m.red * g.gp, lap7p);
    Plane tRp = clip(guidedMlri(g.gp, g.rp * m.red, m.red, difG, difR, m.red, 3, 3, variant));
    difB = correlate(m.blue * (correctedBlueDiagonals ? g.bp : g.rp), lap7p);
    difGb = correlate(m.blue * g.gp, lap7p);
    Plane tBp = clip(guidedMlri(g.gp, g.bp * m.blue, m.blue, difGb, difB, m.blue, 3, 3, variant));
    difG = correlate(m.any({0,4,5,9}) * g.gp, lap7p) + correlate(greenB * g.gp, lap5p);
    difR = correlate(m.any({0,4,5,9}) * g.rp, lap7p) + correlate(greenB * g.rp, lap5p);
    difB = correlate(m.any({0,4,5,9}) * g.bp, lap7p) + correlate(greenB * g.bp, lap5p);
    Plane tGrp = clip(guidedMlri(g.rp, g.gp * m.green, m.green, difR, difG, m.green, 3, 3, variant));
    Plane tGbp = clip(guidedMlri(g.bp, g.gp * m.green, m.green, difB, difG, m.green, 3, 3, variant));

    const Plane m11121516 = m.any({10,11,14,15});
    const Plane m1317 = m.any({12,16});
    const Plane m1418 = m.any({13,17});
    const Plane m13141718 = m.any({12,13,16,17});
    const Plane m1115 = m.any({10,14});
    const Plane m1216 = m.any({11,15});
    const Kernel h101 = horizontal({0.5f,0,0.5f});
    const Kernel h10020 = horizontal({1.f/3,0,0,2.f/3,0});
    const Kernel h02001 = horizontal({0,2.f/3,0,0,1.f/3});
    const Kernel longResidual = scaled(horizontal({1,2,3,4,5,0,5,4,3,2,1}), 1.f/6);
    const Kernel cross = Kernel(3,3,{0,0.25f,0, 0.25f,0,0.25f, 0,0.25f,0});
    const Kernel residualDiag = scaled(diagonal({1,2,0,2,1}), 1.f/3);
    const Kernel residualAnti = scaled(diagonal({1,2,0,2,1}, true), 1.f/3);

    Plane rGrh = (greenRaw - tGrh) * m.green;
    rGrh += m11121516 * correlate(rGrh, h101)
          + m1317 * correlate(rGrh, h10020)
          + m1418 * correlate(rGrh, h02001);
    Plane rGbh = (greenRaw - tGbh) * m.green;
    rGbh += m11121516 * correlate(rGbh, h101)
          + m1317 * correlate(rGbh, h10020)
          + m1418 * correlate(rGbh, h02001);
    Plane rRh = (redRaw - tRh) * m.red;
    rRh += (maskGnC + m15_16) * correlate(rRh, longResidual)
         + m.p[7] * correlate(rRh, h101)
         + m.p[2] * correlate(rRh, horizontal({0.5f,0,0,0,0.5f}))
         + m.p[16] * correlate(rRh, horizontal({0,0,0.75f,0,0,0,0.25f}))
         + m.p[17] * correlate(rRh, horizontal({0.25f,0,0,0,0.75f,0,0}));
    Plane rBh = (blueRaw - tBh) * m.blue;
    rBh += (maskGnC + m11_12) * correlate(rBh, longResidual)
         + m.p[2] * correlate(rBh, h101)
         + m.p[7] * correlate(rBh, horizontal({0.5f,0,0,0,0.5f}))
         + m.p[12] * correlate(rBh, horizontal({0,0,0.75f,0,0,0,0.25f}))
         + m.p[13] * correlate(rBh, horizontal({0.25f,0,0,0,0.75f,0,0}));

    Plane rGrv = (greenRaw - tGrv) * m.green;
    rGrv += m13141718 * correlate(rGrv, transpose(h101))
          + m1115 * correlate(rGrv, transpose(h10020))
          + m1216 * correlate(rGrv, transpose(h02001));
    Plane rGbv = (greenRaw - tGbv) * m.green;
    rGbv += m13141718 * correlate(rGbv, transpose(h101))
          + m1115 * correlate(rGbv, transpose(h10020))
          + m1216 * correlate(rGbv, transpose(h02001));
    Plane rRv = (redRaw - tRv) * m.red;
    rRv += (maskGnC + m17_18) * correlate(rRv, transpose(longResidual))
         + m.p[2] * correlate(rRv, transpose(h101))
         + m.p[7] * correlate(rRv, vertical({0.5f,0,0,0,0.5f}))
         + m.p[14] * correlate(rRv, vertical({0,0,0.75f,0,0,0,0.25f}))
         + m.p[15] * correlate(rRv, vertical({0.25f,0,0,0,0.75f,0,0}));
    Plane rBv = (blueRaw - tBv) * m.blue;
    rBv += (maskGnC + m13_14) * correlate(rBv, transpose(longResidual))
         + m.p[7] * correlate(rBv, transpose(h101))
         + m.p[2] * correlate(rBv, vertical({0.5f,0,0,0,0.5f}))
         + m.p[10] * correlate(rBv, vertical({0,0,0.75f,0,0,0,0.25f}))
         + m.p[11] * correlate(rBv, vertical({0.25f,0,0,0,0.75f,0,0}));

    Plane rGrd = (greenRaw - tGrd) * m.green;
    rGrd += maskY * correlate(rGrd, residualDiag);
    Plane rGbd = (greenRaw - tGbd) * m.green;
    rGbd += maskY * correlate(rGbd, residualDiag);
    Plane rRd = (redRaw - tRd) * m.red;
    rRd += correlate(rRd, residualDiag);
    rRd += greenA * correlate(rRd, cross);
    Plane rBd = (blueRaw - tBd) * m.blue;
    rBd += correlate(rBd, residualDiag);
    rBd += greenA * correlate(rBd, cross);
    Plane rGrp = (greenRaw - tGrp) * m.green;
    rGrp += maskY * correlate(rGrp, residualAnti);
    Plane rGbp = (greenRaw - tGbp) * m.green;
    rGbp += maskY * correlate(rGbp, residualAnti);
    Plane rRp = (redRaw - tRp) * m.red;
    rRp += correlate(rRp, residualAnti);
    rRp += greenB * correlate(rRp, cross);
    Plane rBp = (blueRaw - tBp) * m.blue;
    rBp += correlate(rBp, residualAnti);
    rBp += greenB * correlate(rBp, cross);

    Plane Grh = clip((tGrh + rGrh) * m.red);
    Plane Gbh = clip((tGbh + rGbh) * m.blue);
    Plane Rh = clip((tRh + rRh) * m.green);
    Plane Bh = clip((tBh + rBh) * m.green);
    Plane Grv = clip((tGrv + rGrv) * m.red);
    Plane Gbv = clip((tGbv + rGbv) * m.blue);
    Plane Rv = clip((tRv + rRv) * m.green);
    Plane Bv = clip((tBv + rBv) * m.green);
    Plane Grd = clip((tGrd + rGrd) * m.red);
    Plane Gbd = clip((tGbd + rGbd) * m.blue);
    Plane Rd = clip((tRd + rRd) * m.green);
    Plane Bd = clip((tBd + rBd) * m.green);
    Plane Grp = clip((tGrp + rGrp) * m.red);
    Plane Gbp = clip((tGbp + rGbp) * m.blue);
    Plane Rp = clip((tRp + rRp) * m.green);
    Plane Bp = clip((tBp + rBp) * m.green);

    Plane drh = greenRaw + Grh - redRaw - Rh;
    Plane drv = greenRaw + Grv - redRaw - Rv;
    Plane drd = greenRaw + Grd - redRaw - Rd;
    Plane drp = greenRaw + Grp - redRaw - Rp;
    Plane dbh = greenRaw + Gbh - blueRaw - Bh;
    Plane dbv = greenRaw + Gbv - blueRaw - Bv;
    Plane dbd = greenRaw + Gbd - blueRaw - Bd;
    Plane dbp = greenRaw + Gbp - blueRaw - Bp;
    drh += m.blue * correlate(drh, h101);
    drv += m.blue * correlate(drv, transpose(h101));
    drd += m.blue * correlate(drd, scaled(diagonal({1,0,1}), 0.5f));
    drp += m.blue * correlate(drp, scaled(diagonal({1,0,1}, true), 0.5f));
    dbh += m.red * correlate(dbh, h101);
    dbv += m.red * correlate(dbv, transpose(h101));
    dbd += m.red * correlate(dbd, scaled(diagonal({1,0,1}), 0.5f));
    dbp += m.red * correlate(dbp, scaled(diagonal({1,0,1}, true), 0.5f));

    const std::array<Plane, 8> wr = eightWeights(drh, drv, drd, drp);
    const std::array<Plane, 8> wb = eightWeights(dbh, dbv, dbd, dbp);
    const Kernel kw1 = halfGaussian(sigma, false);
    const Kernel ke1 = halfGaussian(sigma, true);
    const Kernel kn1 = halfGaussian(sigma, false, true);
    const Kernel ks1 = halfGaussian(sigma, true, true);
    const Kernel kw2 = halfGaussian(2.f * sigma, false);
    const Kernel ke2 = halfGaussian(2.f * sigma, true);
    const Kernel kn2 = halfGaussian(2.f * sigma, false, true);
    const Kernel ks2 = halfGaussian(2.f * sigma, true, true);
    const Kernel kw3 = halfGaussian(3.f * sigma, false);
    const Kernel ke3 = halfGaussian(3.f * sigma, true);
    const Kernel kn3 = halfGaussian(3.f * sigma, false, true);
    const Kernel ks3 = halfGaussian(3.f * sigma, true, true);
    const Kernel ka15 = halfGaussian(1.5f * sigma, false, false, true, false);
    const Kernel kr15 = halfGaussian(1.5f * sigma, true, false, true, false);
    const Kernel kj15 = halfGaussian(1.5f * sigma, false, false, true, true);
    const Kernel kk15 = halfGaussian(1.5f * sigma, true, false, true, true);

    const Plane m13 = m.p[12];
    const Plane m14 = m.p[13];
    const Plane m17 = m.p[16];
    const Plane m18 = m.p[17];
    const Plane m11 = m.p[10];
    const Plane m12 = m.p[11];
    const Plane m15 = m.p[14];
    const Plane m16 = m.p[15];
    const std::array<Plane, 8> cr {{
        oneMinus(m11 + m13_14) * correlate(drv, kn1) + m11 * correlate(drv, kn2) + m13_14 * correlate(drv, kn3),
        oneMinus(m12 + m13_14) * correlate(drv, ks1) + m12 * correlate(drv, ks2) + m13_14 * correlate(drv, ks3),
        oneMinus(m13 + m11_12) * correlate(drh, kw1) + m13 * correlate(drh, kw2) + m11_12 * correlate(drh, kw3),
        oneMinus(m14 + m11_12) * correlate(drh, ke1) + m14 * correlate(drh, ke2) + m11_12 * correlate(drh, ke3),
        correlate(drd, ka15), correlate(drd, kr15), correlate(drp, kj15), correlate(drp, kk15)
    }};
    const std::array<Plane, 8> cb {{
        oneMinus(m15 + m17_18) * correlate(dbv, kn1) + m15 * correlate(dbv, kn2) + m17_18 * correlate(dbv, kn3),
        oneMinus(m16 + m17_18) * correlate(dbv, ks1) + m16 * correlate(dbv, ks2) + m17_18 * correlate(dbv, ks3),
        oneMinus(m17 + m15_16) * correlate(dbh, kw1) + m17 * correlate(dbh, kw2) + m15_16 * correlate(dbh, kw3),
        oneMinus(m18 + m15_16) * correlate(dbh, ke1) + m18 * correlate(dbh, ke2) + m15_16 * correlate(dbh, ke3),
        correlate(dbd, ka15), correlate(dbd, kr15), correlate(dbp, kj15), correlate(dbp, kk15)
    }};
    const Plane differenceR = weightedEight(wr, cr);
    const Plane differenceB = weightedEight(wb, cb);
    return mosaic + m.red * differenceR + m.blue * differenceB;
}

struct RedBluePair final {
    Plane red;
    Plane blue;
};

RedBluePair interpolateChromaSites(
    const Plane &mosaic,
    const MlriMasks &m,
    const DirectionalGuides &g,
    const Plane &green,
    float sigma,
    MlriXTransVariant variant)
{
    const Plane redRaw = mosaic * m.red;
    const Plane blueRaw = mosaic * m.blue;
    const Plane maskGC = m.any({2,7});
    const Plane maskGnC = m.any({0,1,3,4,5,6,8,9});
    const Plane maskY = m.red + m.blue;
    const Plane greenA = m.any({0,2,4,5,7,9});
    const Plane greenB = m.any({1,2,3,6,7,8});
    const Plane m11_12 = m.any({10,11});
    const Plane m13_14 = m.any({12,13});
    const Plane m15_16 = m.any({14,15});
    const Plane m17_18 = m.any({16,17});
    const Kernel lap13h = horizontal({-1,0,0,0,0,0,2,0,0,0,0,0,-1});
    const Kernel lap5h = horizontal({-1,0,2,0,-1});
    const Kernel lap13v = transpose(lap13h);
    const Kernel lap5v = transpose(lap5h);
    const Kernel lap7d = diagonal({-1,0,0,2,0,0,-1});
    const Kernel lap7p = diagonal({-1,0,0,2,0,0,-1}, true);

    Plane difR = correlate(m11_12 * g.rh, lap13h) + correlate(m.any({2,12,13}) * g.rh, lap5h);
    Plane difG = correlate(m11_12 * green, lap13h) + correlate(m.any({2,12,13}) * green, lap5h);
    Plane tRh = clip(guidedMlri(green, g.rh * m.red, m.red, difG, difR, m.red, 3, 3));
    Plane difB = correlate(m15_16 * g.bh, lap13h) + correlate(m.any({7,16,17}) * g.bh, lap5h);
    difG = correlate(m15_16 * green, lap13h) + correlate(m.any({7,16,17}) * green, lap5h);
    Plane tBh = clip(guidedMlri(green, g.bh * m.blue, m.blue, difG, difB, m.blue, 3, 3));

    difR = correlate(m13_14 * g.rv, lap13v) + correlate(m.any({7,10,11}) * g.rv, lap5v);
    difG = correlate(m13_14 * green, lap13v) + correlate(m.any({7,10,11}) * green, lap5v);
    Plane tRv = clip(guidedMlri(green, g.rv * m.red, m.red, difG, difR, m.red, 3, 3));
    difB = correlate(m17_18 * g.bv, lap13v) + correlate(m.any({2,14,15}) * g.bv, lap5v);
    difG = correlate(m17_18 * green, lap13v) + correlate(m.any({2,14,15}) * green, lap5v);
    Plane tBv = clip(guidedMlri(green, g.bv * m.blue, m.blue, difG, difB, m.blue, 3, 3));

    difR = correlate(m.red * g.rd, lap7d);
    difG = correlate(m.red * green, lap7d);
    Plane tRd = clip(guidedMlri(green, g.rd * m.red, m.red, difG, difR, m.red, 3, 3));
    // Preserve the same reviewed v1.0.0 cross-guide quirk used by the green
    // stage unless the separately selected corrected variant is active.
    const bool correctedBlueDiagonals = usesCorrectedBlueDiagonalGuides(variant);
    difB = correlate(m.blue * (correctedBlueDiagonals ? g.bd : g.rd), lap7d);
    difG = correlate(m.blue * green, lap7d);
    Plane tBd = clip(guidedMlri(green, g.bd * m.blue, m.blue, difG, difB, m.blue, 3, 3));
    difR = correlate(m.red * g.rp, lap7p);
    difG = correlate(m.red * green, lap7p);
    Plane tRp = clip(guidedMlri(green, g.rp * m.red, m.red, difG, difR, m.red, 3, 3));
    difB = correlate(m.blue * (correctedBlueDiagonals ? g.bp : g.rp), lap7p);
    difG = correlate(m.blue * green, lap7p);
    Plane tBp = clip(guidedMlri(green, g.bp * m.blue, m.blue, difG, difB, m.blue, 3, 3));

    const Kernel h101 = horizontal({0.5f,0,0.5f});
    const Kernel h10000 = horizontal({0.5f,0,0,0,0.5f});
    const Kernel longResidual = scaled(horizontal({1,2,3,4,5,0,5,4,3,2,1}), 1.f/6);
    const Kernel cross = Kernel(3,3,{0,0.25f,0, 0.25f,0,0.25f, 0,0.25f,0});
    const Kernel residualDiag = scaled(diagonal({1,2,0,2,1}), 1.f/3);
    const Kernel residualAnti = scaled(diagonal({1,2,0,2,1}, true), 1.f/3);

    Plane rRh = (redRaw - tRh) * m.red;
    rRh += (maskGnC + m15_16) * correlate(rRh, longResidual)
         + m.p[7] * correlate(rRh, h101)
         + m.p[2] * correlate(rRh, h10000)
         + m.p[16] * correlate(rRh, horizontal({0,0,0.75f,0,0,0,0.25f}))
         + m.p[17] * correlate(rRh, horizontal({0.25f,0,0,0,0.75f,0,0}));
    Plane rBh = (blueRaw - tBh) * m.blue;
    rBh += (maskGnC + m11_12) * correlate(rBh, longResidual)
         + m.p[2] * correlate(rBh, h101)
         + m.p[7] * correlate(rBh, h10000)
         + m.p[12] * correlate(rBh, horizontal({0,0,0.75f,0,0,0,0.25f}))
         + m.p[13] * correlate(rBh, horizontal({0.25f,0,0,0,0.75f,0,0}));
    Plane rRv = (redRaw - tRv) * m.red;
    rRv += (maskGnC + m17_18) * correlate(rRv, transpose(longResidual))
         + m.p[2] * correlate(rRv, transpose(h101))
         + m.p[7] * correlate(rRv, transpose(h10000))
         + m.p[14] * correlate(rRv, vertical({0,0,0.75f,0,0,0,0.25f}))
         + m.p[15] * correlate(rRv, vertical({0.25f,0,0,0,0.75f,0,0}));
    Plane rBv = (blueRaw - tBv) * m.blue;
    rBv += (maskGnC + m13_14) * correlate(rBv, transpose(longResidual))
         + m.p[7] * correlate(rBv, transpose(h101))
         + m.p[2] * correlate(rBv, transpose(h10000))
         + m.p[10] * correlate(rBv, vertical({0,0,0.75f,0,0,0,0.25f}))
         + m.p[11] * correlate(rBv, vertical({0.25f,0,0,0,0.75f,0,0}));
    Plane rRd = (redRaw - tRd) * m.red;
    rRd += correlate(rRd, residualDiag);
    rRd += greenA * correlate(rRd, cross);
    Plane rBd = (blueRaw - tBd) * m.blue;
    rBd += correlate(rBd, residualDiag);
    rBd += greenA * correlate(rBd, cross);
    Plane rRp = (redRaw - tRp) * m.red;
    rRp += correlate(rRp, residualAnti);
    rRp += greenB * correlate(rRp, cross);
    Plane rBp = (blueRaw - tBp) * m.blue;
    rBp += correlate(rBp, residualAnti);
    rBp += greenB * correlate(rBp, cross);

    Plane Rh = clip((tRh + rRh) * m.blue);
    Plane Bh = clip((tBh + rBh) * m.red);
    Plane Rv = clip((tRv + rRv) * m.blue);
    Plane Bv = clip((tBv + rBv) * m.red);
    Plane Rd = clip((tRd + rRd) * m.blue);
    Plane Bd = clip((tBd + rBd) * m.red);
    Plane Rp = clip((tRp + rRp) * m.blue);
    Plane Bp = clip((tBp + rBp) * m.red);

    Plane dh = redRaw + Rh - blueRaw - Bh;
    Plane dv = redRaw + Rv - blueRaw - Bv;
    Plane dd = redRaw + Rd - blueRaw - Bd;
    Plane dp = redRaw + Rp - blueRaw - Bp;
    dh += maskGnC * correlate(dh, scaled(horizontal({1,2,0,2,1}), 1.f/3))
        + maskGC * correlate(dh, h101);
    dv += maskGnC * correlate(dv, scaled(vertical({1,2,0,2,1}), 1.f/3))
        + maskGC * correlate(dv, transpose(h101));
    dd += m.any({1,3,6,8}) * correlate(dd, scaled(diagonal({1,0,1}), 0.5f));
    dd += greenA * correlate(dd, cross);
    dp += m.any({0,4,5,9}) * correlate(dp, scaled(diagonal({1,0,1}, true), 0.5f));
    dp += greenB * correlate(dp, cross);

    const std::array<Plane, 8> weights = eightWeights(dh, dv, dd, dp);
    const Kernel kw1 = halfGaussian(sigma, false);
    const Kernel ke1 = halfGaussian(sigma, true);
    const Kernel kn1 = halfGaussian(sigma, false, true);
    const Kernel ks1 = halfGaussian(sigma, true, true);
    const Kernel kw2 = halfGaussian(2.f*sigma, false);
    const Kernel ke2 = halfGaussian(2.f*sigma, true);
    const Kernel kn2 = halfGaussian(2.f*sigma, false, true);
    const Kernel ks2 = halfGaussian(2.f*sigma, true, true);
    const Kernel kw3 = halfGaussian(3.f*sigma, false);
    const Kernel ke3 = halfGaussian(3.f*sigma, true);
    const Kernel kn3 = halfGaussian(3.f*sigma, false, true);
    const Kernel ks3 = halfGaussian(3.f*sigma, true, true);
    const Kernel ka15 = halfGaussian(1.5f*sigma, false, false, true, false);
    const Kernel kr15 = halfGaussian(1.5f*sigma, true, false, true, false);
    const Kernel kj15 = halfGaussian(1.5f*sigma, false, false, true, true);
    const Kernel kk15 = halfGaussian(1.5f*sigma, true, false, true, true);
    const Plane horizontalLong = m.any({10,11,14,15});
    const Plane verticalLong = m.any({12,13,16,17});
    const Plane hNearWest = m.any({12,16});
    const Plane hNearEast = m.any({13,17});
    const Plane vNearNorth = m.any({10,14});
    const Plane vNearSouth = m.any({11,15});
    const std::array<Plane, 8> candidates {{
        oneMinus(vNearNorth + verticalLong) * correlate(dv, kn1)
            + vNearNorth * correlate(dv, kn2) + verticalLong * correlate(dv, kn3),
        oneMinus(vNearSouth + verticalLong) * correlate(dv, ks1)
            + vNearSouth * correlate(dv, ks2) + verticalLong * correlate(dv, ks3),
        oneMinus(hNearWest + horizontalLong) * correlate(dh, kw1)
            + hNearWest * correlate(dh, kw2) + horizontalLong * correlate(dh, kw3),
        oneMinus(hNearEast + horizontalLong) * correlate(dh, ke1)
            + hNearEast * correlate(dh, ke2) + horizontalLong * correlate(dh, ke3),
        correlate(dd, ka15), correlate(dd, kr15),
        correlate(dp, kj15), correlate(dp, kk15)
    }};
    const Plane difference = weightedEight(weights, candidates);
    return {
        mosaic * oneMinus(m.green) + difference * m.blue,
        mosaic * oneMinus(m.green) - difference * m.red
    };
}

RedBluePair interpolateCenterGreenSites(
    const Plane &mosaic,
    const MlriMasks &m,
    const Plane &green,
    const RedBluePair &chroma,
    const RgbPlanes &previous,
    int pass,
    float sigma)
{
    // The first R/B completion stage reconstructs the two central green
    // phases (G3/G8 in the reference notation).  These phases have usable
    // horizontal and vertical chroma neighbours, so the reference deliberately
    // omits diagonal candidates here.
    const Plane maskGC = m.any({2,7});
    const Plane maskGnC = m.any({0,1,3,4,5,6,8,9});
    const Plane maskY = m.red + m.blue;
    const Plane hvSites = m.any({10,11,14,15});
    const Plane vhSites = m.any({12,13,16,17});
    const Plane hLaplacianSites = maskGC + vhSites;
    const Plane vLaplacianSites = maskGC + hvSites;

    Plane guideRh;
    Plane guideRv;
    Plane guideBh;
    Plane guideBv;
    const Kernel nearestH = horizontal({0.5f,0,0.5f});
    const Kernel nearestV = transpose(nearestH);
    if (pass == 0) {
        guideRh = chroma.red + maskGC * correlate(chroma.red, nearestH);
        guideRv = chroma.red + maskGC * correlate(chroma.red, nearestV);
        guideBh = chroma.blue + maskGC * correlate(chroma.blue, nearestH);
        guideBv = chroma.blue + maskGC * correlate(chroma.blue, nearestV);
    } else {
        guideRh = chroma.red + maskGC * previous.red;
        guideRv = guideRh;
        guideBh = chroma.blue + maskGC * previous.blue;
        guideBv = guideBh;
    }

    const Kernel lap7h = horizontal({-1,0,0,2,0,0,-1});
    const Kernel lap5h = horizontal({-1,0,2,0,-1});
    const Kernel lap7v = transpose(lap7h);
    const Kernel lap5v = transpose(lap5h);
    Plane difG = correlate(hvSites * green, lap7h)
               + correlate(hLaplacianSites * green, lap5h);
    Plane difR = correlate(hvSites * guideRh, lap7h)
               + correlate(hLaplacianSites * guideRh, lap5h);
    Plane difB = correlate(hvSites * guideBh, lap7h)
               + correlate(hLaplacianSites * guideBh, lap5h);
    Plane tRh = clip(guidedMlri(green, guideRh * maskY, maskY,
                                difG, difR, maskY, 3, 3));
    Plane tBh = clip(guidedMlri(green, guideBh * maskY, maskY,
                                difG, difB, maskY, 3, 3));

    difG = correlate(vhSites * green, lap7v)
         + correlate(vLaplacianSites * green, lap5v);
    difR = correlate(vhSites * guideRv, lap7v)
         + correlate(vLaplacianSites * guideRv, lap5v);
    difB = correlate(vhSites * guideBv, lap7v)
         + correlate(vLaplacianSites * guideBv, lap5v);
    Plane tRv = clip(guidedMlri(green, guideRv * maskY, maskY,
                                difG, difR, maskY, 3, 3));
    Plane tBv = clip(guidedMlri(green, guideBv * maskY, maskY,
                                difG, difB, maskY, 3, 3));

    const Kernel broadH = scaled(horizontal({1,2,0,2,1}), 1.f / 3.f);
    const Kernel broadV = transpose(broadH);
    Plane rRh = (chroma.red - tRh) * maskY;
    Plane rRv = (chroma.red - tRv) * maskY;
    Plane rBh = (chroma.blue - tBh) * maskY;
    Plane rBv = (chroma.blue - tBv) * maskY;
    rRh += maskGC * correlate(rRh, nearestH) + maskGnC * correlate(rRh, broadH);
    rRv += maskGC * correlate(rRv, nearestV) + maskGnC * correlate(rRv, broadV);
    rBh += maskGC * correlate(rBh, nearestH) + maskGnC * correlate(rBh, broadH);
    rBv += maskGC * correlate(rBv, nearestV) + maskGnC * correlate(rBv, broadV);

    const Plane Rh = clip((tRh + rRh) * m.green);
    const Plane Rv = clip((tRv + rRv) * m.green);
    const Plane Bh = clip((tBh + rBh) * m.green);
    const Plane Bv = clip((tBv + rBv) * m.green);
    const Plane drh = chroma.red + Rh - green;
    const Plane drv = chroma.red + Rv - green;
    const Plane dbh = chroma.blue + Bh - green;
    const Plane dbv = chroma.blue + Bv - green;

    const std::array<Plane, 4> wr = fourWeights(drh, drv);
    const std::array<Plane, 4> wb = fourWeights(dbh, dbv);
    const Kernel kw = halfGaussian(1.5f * sigma, false);
    const Kernel ke = halfGaussian(1.5f * sigma, true);
    const Kernel kn = halfGaussian(1.5f * sigma, false, true);
    const Kernel ks = halfGaussian(1.5f * sigma, true, true);
    const std::array<Plane, 4> cr {{
        correlate(drv, kn), correlate(drv, ks),
        correlate(drh, kw), correlate(drh, ke)
    }};
    const std::array<Plane, 4> cb {{
        correlate(dbv, kn), correlate(dbv, ks),
        correlate(dbh, kw), correlate(dbh, ke)
    }};
    const Plane differenceR = weightedFour(wr, cr);
    const Plane differenceB = weightedFour(wb, cb);
    return {
        chroma.red + (mosaic + differenceR) * maskGC,
        chroma.blue + (mosaic + differenceB) * maskGC
    };
}

RedBluePair interpolateRemainingGreenSites(
    const Plane &mosaic,
    const MlriMasks &m,
    const Plane &green,
    const RedBluePair &centered,
    const RgbPlanes &previous,
    int pass,
    float sigma)
{
    // The second R/B completion stage fills the eight non-central green
    // phases.  This is the X-Trans-specific engineering layer: four RI/MLRI
    // directional estimates are combined with local inverse-gradient energy.
    const Plane maskGC = m.any({2,7});
    const Plane maskGnC = m.any({0,1,3,4,5,6,8,9});
    const Plane maskY = m.red + m.blue;
    const Plane maskYC = maskY + maskGC;
    const Plane target = oneMinus(maskYC);
    const Plane greenA = m.any({0,4,5,9});
    const Plane greenB = m.any({1,3,6,8});
    const Plane hvSites = m.any({10,11,14,15});
    const Plane vhSites = m.any({12,13,16,17});
    const Plane hLaplacianSites = maskGC + vhSites;
    const Plane vLaplacianSites = maskGC + hvSites;

    const Kernel broadHRed = scaled(horizontal({1,2,0,2,1}), 1.f / 3.f);
    const Kernel broadVRed = transpose(broadHRed);
    const Kernel broadHBlue = scaled(horizontal({1,2,0,2,1}), 0.5f);
    const Kernel broadVBlue = transpose(broadHBlue);
    const Kernel diagBroad = scaled(diagonal({1,2,0,2,1}), 1.f / 3.f);
    const Kernel diagNear = scaled(diagonal({1,0,1}), 0.5f);
    const Kernel antiBroad = scaled(diagonal({1,2,0,2,1}, true), 1.f / 3.f);
    const Kernel antiNear = scaled(diagonal({1,0,1}, true), 0.5f);

    Plane guideRh;
    Plane guideRv;
    Plane guideRd;
    Plane guideRp;
    Plane guideBh;
    Plane guideBv;
    Plane guideBd;
    Plane guideBp;
    if (pass == 0) {
        guideRh = centered.red + maskGnC * correlate(centered.red, broadHRed);
        guideRv = centered.red + maskGnC * correlate(centered.red, broadVRed);
        guideRd = centered.red + greenA * correlate(centered.red, diagBroad)
                              + greenB * correlate(centered.red, diagNear);
        guideRp = centered.red + greenB * correlate(centered.red, antiBroad)
                              + greenA * correlate(centered.red, antiNear);
        guideBh = centered.blue + correlate(centered.blue, broadHBlue);
        guideBv = centered.blue + correlate(centered.blue, broadVBlue);
        guideBd = centered.blue + greenA * correlate(centered.blue, diagBroad)
                               + greenB * correlate(centered.blue, diagNear);
        guideBp = centered.blue + greenB * correlate(centered.blue, antiBroad)
                               + greenA * correlate(centered.blue, antiNear);
    } else {
        const Plane priorR = previous.red * target;
        const Plane priorB = previous.blue * target;
        guideRh = centered.red + priorR;
        guideRv = guideRh;
        guideRd = guideRh;
        guideRp = guideRh;
        guideBh = centered.blue + priorB;
        guideBv = guideBh;
        guideBd = guideBh;
        guideBp = guideBh;
    }

    const Kernel lap7h = horizontal({-1,0,0,2,0,0,-1});
    const Kernel lap5h = horizontal({-1,0,2,0,-1});
    const Kernel lap7v = transpose(lap7h);
    const Kernel lap5v = transpose(lap5h);
    const Kernel lap7d = diagonal({-1,0,0,2,0,0,-1});
    const Kernel lap5d = diagonal({-1,0,2,0,-1});
    const Kernel lap7p = diagonal({-1,0,0,2,0,0,-1}, true);
    const Kernel lap5p = diagonal({-1,0,2,0,-1}, true);

    Plane difG = correlate(hvSites * green, lap7h)
               + correlate(hLaplacianSites * green, lap5h);
    Plane difR = correlate(hvSites * guideRh, lap7h)
               + correlate(hLaplacianSites * guideRh, lap5h);
    Plane difB = correlate(hvSites * guideBh, lap7h)
               + correlate(hLaplacianSites * guideBh, lap5h);
    Plane tRh = clip(guidedMlri(green, guideRh * maskYC, maskYC,
                                difG, difR, maskYC, 3, 3));
    Plane tBh = clip(guidedMlri(green, guideBh * maskYC, maskYC,
                                difG, difB, maskYC, 3, 3));

    difG = correlate(vhSites * green, lap7v)
         + correlate(vLaplacianSites * green, lap5v);
    difR = correlate(vhSites * guideRv, lap7v)
         + correlate(vLaplacianSites * guideRv, lap5v);
    difB = correlate(vhSites * guideBv, lap7v)
         + correlate(vLaplacianSites * guideBv, lap5v);
    Plane tRv = clip(guidedMlri(green, guideRv * maskYC, maskYC,
                                difG, difR, maskYC, 3, 3));
    Plane tBv = clip(guidedMlri(green, guideBv * maskYC, maskYC,
                                difG, difB, maskYC, 3, 3));

    difG = correlate(maskGC * green, lap7d) + maskY * correlate(green, lap5d);
    difR = correlate(maskGC * guideRd, lap7d) + maskY * correlate(guideRd, lap5d);
    difB = correlate(maskGC * guideBd, lap7d) + maskY * correlate(guideBd, lap5d);
    Plane tRd = clip(guidedMlri(green, guideRd * maskYC, maskYC,
                                difG, difR, maskYC, 3, 3));
    Plane tBd = clip(guidedMlri(green, guideBd * maskYC, maskYC,
                                difG, difB, maskYC, 3, 3));

    difG = correlate(maskGC * green, lap7p) + maskY * correlate(green, lap5p);
    difR = correlate(maskGC * guideRp, lap7p) + maskY * correlate(guideRp, lap5p);
    difB = correlate(maskGC * guideBp, lap7p) + maskY * correlate(guideBp, lap5p);
    Plane tRp = clip(guidedMlri(green, guideRp * maskYC, maskYC,
                                difG, difR, maskYC, 3, 3));
    Plane tBp = clip(guidedMlri(green, guideBp * maskYC, maskYC,
                                difG, difB, maskYC, 3, 3));

    Plane rRh = (centered.red - tRh) * maskYC;
    Plane rRv = (centered.red - tRv) * maskYC;
    Plane rRd = (centered.red - tRd) * maskYC;
    Plane rRp = (centered.red - tRp) * maskYC;
    Plane rBh = (centered.blue - tBh) * maskYC;
    Plane rBv = (centered.blue - tBv) * maskYC;
    Plane rBd = (centered.blue - tBd) * maskYC;
    Plane rBp = (centered.blue - tBp) * maskYC;
    rRh += correlate(rRh, broadHRed);
    rRv += correlate(rRv, broadVRed);
    rRd += greenA * correlate(rRd, diagBroad) + greenB * correlate(rRd, diagNear);
    rRp += greenB * correlate(rRp, antiBroad) + greenA * correlate(rRp, antiNear);
    rBh += correlate(rBh, broadHBlue);
    rBv += correlate(rBv, broadVBlue);
    rBd += greenA * correlate(rBd, diagBroad) + greenB * correlate(rBd, diagNear);
    rBp += greenB * correlate(rBp, antiBroad) + greenA * correlate(rBp, antiNear);

    const Plane Rh = clip((tRh + rRh) * target);
    const Plane Rv = clip((tRv + rRv) * target);
    const Plane Rd = clip((tRd + rRd) * target);
    const Plane Rp = clip((tRp + rRp) * target);
    const Plane Bh = clip((tBh + rBh) * target);
    const Plane Bv = clip((tBv + rBv) * target);
    const Plane Bd = clip((tBd + rBd) * target);
    const Plane Bp = clip((tBp + rBp) * target);
    const Plane drh = centered.red + Rh - green;
    const Plane drv = centered.red + Rv - green;
    const Plane drd = centered.red + Rd - green;
    const Plane drp = centered.red + Rp - green;
    const Plane dbh = centered.blue + Bh - green;
    const Plane dbv = centered.blue + Bv - green;
    const Plane dbd = centered.blue + Bd - green;
    const Plane dbp = centered.blue + Bp - green;

    const std::array<Plane, 8> wr = eightWeights(drh, drv, drd, drp);
    const std::array<Plane, 8> wb = eightWeights(dbh, dbv, dbd, dbp);
    const Kernel kw1 = halfGaussian(sigma, false);
    const Kernel ke1 = halfGaussian(sigma, true);
    const Kernel kn1 = halfGaussian(sigma, false, true);
    const Kernel ks1 = halfGaussian(sigma, true, true);
    const Kernel kw2 = halfGaussian(2.f * sigma, false);
    const Kernel ke2 = halfGaussian(2.f * sigma, true);
    const Kernel kn2 = halfGaussian(2.f * sigma, false, true);
    const Kernel ks2 = halfGaussian(2.f * sigma, true, true);
    const Kernel ka1 = halfGaussian(sigma, false, false, true, false);
    const Kernel kr1 = halfGaussian(sigma, true, false, true, false);
    const Kernel kj1 = halfGaussian(sigma, false, false, true, true);
    const Kernel kk1 = halfGaussian(sigma, true, false, true, true);
    const Kernel ka2 = halfGaussian(2.f * sigma, false, false, true, false);
    const Kernel kr2 = halfGaussian(2.f * sigma, true, false, true, false);
    const Kernel kj2 = halfGaussian(2.f * sigma, false, false, true, true);
    const Kernel kk2 = halfGaussian(2.f * sigma, true, false, true, true);
    const Plane maskLT = m.any({4,9});
    const Plane maskRT = m.any({3,8});
    const Plane maskLB = m.any({1,6});
    const Plane maskRB = m.any({0,5});
    const Plane maskRTRB = maskRT + maskRB;
    const Plane maskLTLB = maskLT + maskLB;
    const Plane maskLBRB = maskLB + maskRB;
    const Plane maskLTRT = maskLT + maskRT;
    const std::array<Plane, 8> cr {{
        oneMinus(maskLBRB) * correlate(drv, kn1) + maskLBRB * correlate(drv, kn2),
        oneMinus(maskLTRT) * correlate(drv, ks1) + maskLTRT * correlate(drv, ks2),
        oneMinus(maskRTRB) * correlate(drh, kw1) + maskRTRB * correlate(drh, kw2),
        oneMinus(maskLTLB) * correlate(drh, ke1) + maskLTLB * correlate(drh, ke2),
        oneMinus(maskRB) * correlate(drd, ka1) + maskRB * correlate(drd, ka2),
        oneMinus(maskLT) * correlate(drd, kr1) + maskLT * correlate(drd, kr2),
        oneMinus(maskLB) * correlate(drp, kj1) + maskLB * correlate(drp, kj2),
        oneMinus(maskRT) * correlate(drp, kk1) + maskRT * correlate(drp, kk2)
    }};
    const std::array<Plane, 8> cb {{
        oneMinus(maskLBRB) * correlate(dbv, kn1) + maskLBRB * correlate(dbv, kn2),
        oneMinus(maskLTRT) * correlate(dbv, ks1) + maskLTRT * correlate(dbv, ks2),
        oneMinus(maskRTRB) * correlate(dbh, kw1) + maskRTRB * correlate(dbh, kw2),
        oneMinus(maskLTLB) * correlate(dbh, ke1) + maskLTLB * correlate(dbh, ke2),
        oneMinus(maskRB) * correlate(dbd, ka1) + maskRB * correlate(dbd, ka2),
        oneMinus(maskLT) * correlate(dbd, kr1) + maskLT * correlate(dbd, kr2),
        oneMinus(maskLB) * correlate(dbp, kj1) + maskLB * correlate(dbp, kj2),
        oneMinus(maskRT) * correlate(dbp, kk1) + maskRT * correlate(dbp, kk2)
    }};
    const Plane differenceR = weightedEight(wr, cr);
    const Plane differenceB = weightedEight(wb, cb);
    return {
        centered.red + (green + differenceR) * target,
        centered.blue + (green + differenceB) * target
    };
}

RedBluePair finalRedBlue(
    const Plane &green,
    const Plane &mosaic,
    const MlriMasks &m,
    MlriXTransVariant variant)
{
    // MATLAB_AUTHOR_HEURISTIC: after the two MLRI/RI green passes, the source
    // performs a separate green-guided R/B reconstruction.  The faithful
    // path in runMlri()
    // blends it with the provisional chroma using sqrt(G/255); this is not a
    // step from the cited RI papers.
    // The final reconstruction is a conventional MLRI guided estimate of R
    // and B from the completed green plane.  The four asymmetric Laplacians
    // and residual kernels below are copied numerically from the reviewed
    // X-Trans reference; no sample reinjection or false-colour pass follows.
    const Plane redRaw = mosaic * m.red;
    const Plane blueRaw = mosaic * m.blue;
    const Kernel f1(5,5,{
        0,2,0,2,0, 0,0,0,0,0, 0,0,-11,0,0,
        2,0,0,0,2, 0,0,3,0,0});
    const Kernel f2(5,5,{
        0,0,3,0,0, 2,0,0,0,2, 0,0,-11,0,0,
        0,0,0,0,0, 0,2,0,2,0});
    const Kernel f3(5,5,{
        0,0,0,2,0, 2,0,0,0,0, 0,0,-11,0,3,
        2,0,0,0,0, 0,0,0,2,0});
    const Kernel f4(5,5,{
        0,2,0,0,0, 0,0,0,0,2, 3,0,-11,0,0,
        0,0,0,0,2, 0,2,0,0,0});
    const std::array<Kernel,4> filters {{f1,f2,f3,f4}};

    Plane lapRed(green.width, green.height);
    Plane lapGreenR(green.width, green.height);
    Plane lapBlue(green.width, green.height);
    Plane lapGreenB(green.width, green.height);
    for (int i = 0; i < 4; ++i) {
        lapRed += m.p[10 + i] * correlate(redRaw, filters[static_cast<std::size_t>(i)]);
        lapGreenR += m.p[10 + i] * correlate(green * m.red, filters[static_cast<std::size_t>(i)]);
        lapBlue += m.p[14 + i] * correlate(blueRaw, filters[static_cast<std::size_t>(i)]);
        lapGreenB += m.p[14 + i] * correlate(green * m.blue, filters[static_cast<std::size_t>(i)]);
    }
    Plane tentativeR = clip(guidedMlri(green, redRaw, m.red,
                                       lapGreenR, lapRed, m.red, 5, 5, variant));
    Plane tentativeB = clip(guidedMlri(green, blueRaw, m.blue,
                                       lapGreenB, lapBlue, m.blue, 5, 5, variant));
    Plane residualR = m.red * (redRaw - tentativeR);
    Plane residualB = m.blue * (blueRaw - tentativeB);

    const Kernel h1(5,5,{
        0,0,1,0,0, 0,0,2,0,0, 1,2,0,2,1,
        0,0,2,0,0, 0,0,1,0,0});
    const Kernel h2h(7,5,{
        0,3,0,0,0,3,0, 0,0,6,6,6,0,0,
        2,0,0,0,0,0,2, 0,0,6,6,6,0,0,
        0,3,0,0,0,3,0});
    const Kernel h2v(5,7,{
        0,0,2,0,0, 3,0,0,0,3, 0,6,0,6,0,
        0,6,0,6,0, 0,6,0,6,0, 3,0,0,0,3,
        0,0,2,0,0});
    const Kernel h34(5,5,{
        0,0,0,0,0, 0,2,2,2,0, 0,2,0,0,1,
        0,2,0,0,0, 0,0,1,0,1});
    const Kernel h33 = rotate90(h34, 3);
    const Kernel h32 = rotate90(h34, 1);
    const Kernel h31 = rotate90(h34, 2);
    const Plane maskGC = m.any({2,7});
    const Plane maskG1 = m.any({0,5});
    const Plane maskG2 = m.any({1,6});
    const Plane maskG3 = m.any({3,8});
    const Plane maskG4 = m.any({4,9});
    const Plane maskRv = m.any({10,11});
    const Plane maskRh = m.any({12,13});
    const Plane maskBv = m.any({14,15});
    const Plane maskBh = m.any({16,17});
    residualR += maskGC * correlate(residualR, scaled(h1, 1.f / 6.f));
    residualR += maskBv * correlate(residualR, scaled(h2h, 1.f / 34.f))
               + maskBh * correlate(residualR, scaled(h2v, 1.f / 34.f));
    residualR += maskG1 * correlate(residualR, scaled(h31, 1.f / 13.f))
               + maskG2 * correlate(residualR, scaled(h32, 1.f / 13.f))
               + maskG3 * correlate(residualR, scaled(h33, 1.f / 13.f))
               + maskG4 * correlate(residualR, scaled(h34, 1.f / 13.f));
    residualB += maskGC * correlate(residualB, scaled(h1, 1.f / 6.f));
    residualB += maskRv * correlate(residualB, scaled(h2h, 1.f / 34.f))
               + maskRh * correlate(residualB, scaled(h2v, 1.f / 34.f));
    residualB += maskG1 * correlate(residualB, scaled(h31, 1.f / 13.f))
               + maskG2 * correlate(residualB, scaled(h32, 1.f / 13.f))
               + maskG3 * correlate(residualB, scaled(h33, 1.f / 13.f))
               + maskG4 * correlate(residualB, scaled(h34, 1.f / 13.f));
    return {residualR + tentativeR, residualB + tentativeB};
}

RgbPlanes runMlri(
    const Plane &mosaic,
    int originX,
    int originY,
    MlriXTransVariant variant)
{
    MlriMasks masks(mosaic.width, mosaic.height, originX, originY);
    const Plane greenRaw = mosaic * masks.green;
    const Plane redRaw = mosaic * masks.red;
    const Plane blueRaw = mosaic * masks.blue;
    RgbPlanes previous {
        Plane(mosaic.width, mosaic.height),
        Plane(mosaic.width, mosaic.height),
        Plane(mosaic.width, mosaic.height)
    };

    // The source-compatible methods retain MATLAB slow=1: two complete
    // refinement passes at sigma 2 and 1.  The controlled paper-core methods
    // stop after the first green pass so the later two-pass heuristic cannot
    // influence the 2014-versus-2016 coefficient comparison.
    const int passCount = isPaperCore(variant) ? 1 : 2;
    for (int pass = 0; pass < passCount; ++pass) {
        const float sigma = pass == 0 ? 2.f : 1.f;
        const DirectionalGuides guides = makeGreenGuides(
            greenRaw, redRaw, blueRaw, masks, pass, previous);
        Plane green = interpolateGreen(mosaic, masks, guides, sigma, variant);
        if (isPaperCore(variant)) {
            // The direct final reconstruction is green-guided RI/MLRI.  It is
            // deliberately returned without the X-Trans source's provisional
            // R/B completion and sqrt(G/255) blend.
            green = clip(green);
            RedBluePair final = finalRedBlue(green, mosaic, masks, variant);
            return {clip(final.red), std::move(green), clip(final.blue)};
        }
        RedBluePair chroma = interpolateChromaSites(
            mosaic, masks, guides, green, sigma, variant);
        chroma = interpolateCenterGreenSites(
            mosaic, masks, green, chroma, previous, pass, sigma);
        chroma = interpolateRemainingGreenSites(
            mosaic, masks, green, chroma, previous, pass, sigma);
        previous = {std::move(chroma.red), std::move(green), std::move(chroma.blue)};
    }

    previous.green = clip(previous.green);
    const RedBluePair final = finalRedBlue(previous.green, mosaic, masks, variant);
    if (usesFinalOnly(variant)) {
        // Controlled overshoot experiment: retain the corrected two-pass green
        // reconstruction unchanged, but use the separately reconstructed,
        // green-guided chroma directly.  At low luminance the source's
        // sqrt(G/255) weight otherwise selects almost entirely the provisional
        // chroma planes, including their isolated blue overshoots.
        return {clip(final.red), std::move(previous.green), clip(final.blue)};
    }
    const Plane uG = squareRoot(previous.green * (1.f / SOURCE_MAX));
    previous.red = clip(oneMinus(uG) * previous.red + uG * final.red);
    previous.blue = clip(oneMinus(uG) * previous.blue + uG * final.blue);
    return previous;
}

} // namespace

const char *mlriXTransErrorCodeName(MlriXTransErrorCode code)
{
    switch (code) {
        case MlriXTransErrorCode::NONE: return "NONE";
        case MlriXTransErrorCode::SIZE: return "SIZE";
        case MlriXTransErrorCode::CFA: return "CFA";
        case MlriXTransErrorCode::ALLOCATION: return "ALLOCATION";
        case MlriXTransErrorCode::NONFINITE: return "NONFINITE";
        case MlriXTransErrorCode::INTERNAL: return "INTERNAL";
    }
    return "INTERNAL";
}

MlriXTransRunResult demosaicMlriXTransReference(
    const float *mosaic,
    float *red,
    float *green,
    float *blue,
    int width,
    int height,
    int originX,
    int originY,
    MlriXTransVariant variant)
{
    MlriXTransRunResult result;
    result.tileCount = 1;
    result.workerCount = 1;
    result.coreSize = width;
    result.halo = 0;
    if (!mosaic || !red || !green || !blue || width < 32 || height < 32) {
        result.code = MlriXTransErrorCode::SIZE;
        result.message = "MLRI requires non-null planes at least 32x32";
        return result;
    }
    if (static_cast<std::uint64_t>(width) * static_cast<std::uint64_t>(height)
            > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
        result.code = MlriXTransErrorCode::SIZE;
        result.message = "MLRI image size overflows addressable memory";
        return result;
    }

    try {
        Plane source(width, height);
        for (std::size_t i = 0; i < source.values.size(); ++i) {
            if (!std::isfinite(mosaic[i])) {
                result.code = MlriXTransErrorCode::NONFINITE;
                result.message = "MLRI input contains a non-finite sample";
                return result;
            }
            // Match the MATLAB uint16 path exactly: conversion to single,
            // division by 257, and an 8-bit-domain internal reconstruction.
            source.values[i] = std::max(0.f, std::min(65535.f, mosaic[i])) / SOURCE_SCALE;
        }
        const RgbPlanes output = runMlri(source, originX, originY, variant);
        if (!finite(output.red) || !finite(output.green) || !finite(output.blue)) {
            result.code = MlriXTransErrorCode::NONFINITE;
            result.message = "MLRI reconstruction produced a non-finite sample";
            return result;
        }
        for (std::size_t i = 0; i < source.values.size(); ++i) {
            red[i] = output.red.values[i] * SOURCE_SCALE;
            green[i] = output.green.values[i] * SOURCE_SCALE;
            blue[i] = output.blue.values[i] * SOURCE_SCALE;
        }
        result.workspaceBytesPerWorker =
            static_cast<std::uint64_t>(source.values.size()) * sizeof(float) * 180u;
        return result;
    } catch (const std::bad_alloc &) {
        result.code = MlriXTransErrorCode::ALLOCATION;
        result.message = "MLRI workspace allocation failed";
    } catch (const std::exception &error) {
        result.code = MlriXTransErrorCode::INTERNAL;
        result.message = error.what();
    } catch (...) {
        result.code = MlriXTransErrorCode::INTERNAL;
        result.message = "unknown MLRI failure";
    }
    return result;
}

MlriXTransRunResult demosaicMlriXTrans(
    const array2D<float> &rawData,
    array2D<float> &red,
    array2D<float> &green,
    array2D<float> &blue,
    int width,
    int height,
    const int xtrans[6][6],
    MlriXTransVariant variant)
{
    MlriXTransRunResult result;
    result.coreSize = CORE_SIZE;
    result.halo = TILE_HALO;
    if (width < 32 || height < 32) {
        result.code = MlriXTransErrorCode::SIZE;
        result.message = "MLRI requires an image at least 32x32";
        return result;
    }

    XTransCfaTransform transform;
    if (!findCanonicalXTransTransform(xtrans, MLRI_CFA, transform)) {
        result.code = MlriXTransErrorCode::CFA;
        result.message = "unsupported X-Trans CFA representation";
        return result;
    }
    const XTransCfaView view(transform, width, height, MLRI_CFA);
    if (!view.valid()) {
        result.code = MlriXTransErrorCode::CFA;
        result.message = "invalid canonical X-Trans coordinate transform";
        return result;
    }

    const int tilesX = (view.width() + CORE_SIZE - 1) / CORE_SIZE;
    const int tilesY = (view.height() + CORE_SIZE - 1) / CORE_SIZE;
    const std::uint64_t tileCount = static_cast<std::uint64_t>(tilesX) * tilesY;
    result.tileCount = tileCount;
#ifdef _OPENMP
    const int availableTiles = tileCount > static_cast<std::uint64_t>(std::numeric_limits<int>::max())
        ? std::numeric_limits<int>::max()
        : static_cast<int>(tileCount);
    const int workerCount = std::max(
        1, std::min(2, std::min(omp_get_max_threads(), availableTiles)));
#else
    const int workerCount = 1;
#endif
    result.workerCount = static_cast<std::uint32_t>(workerCount);
    const std::uint64_t maximumPatch = static_cast<std::uint64_t>(CORE_SIZE + 2 * TILE_HALO)
                                     * static_cast<std::uint64_t>(CORE_SIZE + 2 * TILE_HALO);
    result.workspaceBytesPerWorker = maximumPatch * sizeof(float) * 180u;

    std::atomic<bool> failed(false);
    MlriXTransErrorCode failureCode = MlriXTransErrorCode::NONE;
    std::string failureMessage;

#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic) num_threads(workerCount)
#endif
    for (std::int64_t tile = 0; tile < static_cast<std::int64_t>(tileCount); ++tile) {
        if (failed.load(std::memory_order_relaxed)) {
            continue;
        }
        try {
            const int tileX = static_cast<int>(tile % tilesX);
            const int tileY = static_cast<int>(tile / tilesX);
            const int coreX = tileX * CORE_SIZE;
            const int coreY = tileY * CORE_SIZE;
            const int coreWidth = std::min(CORE_SIZE, view.width() - coreX);
            const int coreHeight = std::min(CORE_SIZE, view.height() - coreY);
            const int patchStartX = std::max(0, coreX - TILE_HALO);
            const int patchStartY = std::max(0, coreY - TILE_HALO);
            const int patchEndX = std::min(view.width(), coreX + coreWidth + TILE_HALO);
            const int patchEndY = std::min(view.height(), coreY + coreHeight + TILE_HALO);
            const int patchWidth = patchEndX - patchStartX;
            const int patchHeight = patchEndY - patchStartY;
            Plane patch(patchWidth, patchHeight);
            for (int py = 0; py < patchHeight; ++py) {
                const int v = patchStartY + py;
                for (int px = 0; px < patchWidth; ++px) {
                    const int u = patchStartX + px;
                    int x = 0;
                    int y = 0;
                    view.canonicalToActual(u, v, x, y);
                    const float value = rawData[y][x];
                    if (!std::isfinite(value)) {
                        throw std::domain_error("MLRI input contains a non-finite sample");
                    }
                    patch(px, py) = std::max(0.f, std::min(65535.f, value)) / SOURCE_SCALE;
                }
            }

            const int originX = patchStartX + view.minimumX();
            const int originY = patchStartY + view.minimumY();
            const RgbPlanes output = runMlri(patch, originX, originY, variant);
            if (!finite(output.red) || !finite(output.green) || !finite(output.blue)) {
                throw std::domain_error("MLRI reconstruction produced a non-finite sample");
            }
            for (int cy = 0; cy < coreHeight; ++cy) {
                for (int cx = 0; cx < coreWidth; ++cx) {
                    int x = 0;
                    int y = 0;
                    view.canonicalToActual(coreX + cx, coreY + cy, x, y);
                    const int px = coreX - patchStartX + cx;
                    const int py = coreY - patchStartY + cy;
                    red[y][x] = output.red(px, py) * SOURCE_SCALE;
                    green[y][x] = output.green(px, py) * SOURCE_SCALE;
                    blue[y][x] = output.blue(px, py) * SOURCE_SCALE;
                }
            }
        } catch (const std::bad_alloc &) {
            bool expected = false;
            if (failed.compare_exchange_strong(expected, true)) {
                failureCode = MlriXTransErrorCode::ALLOCATION;
                failureMessage = "MLRI tile workspace allocation failed";
            }
        } catch (const std::domain_error &error) {
            bool expected = false;
            if (failed.compare_exchange_strong(expected, true)) {
                failureCode = MlriXTransErrorCode::NONFINITE;
                failureMessage = error.what();
            }
        } catch (const std::exception &error) {
            bool expected = false;
            if (failed.compare_exchange_strong(expected, true)) {
                failureCode = MlriXTransErrorCode::INTERNAL;
                failureMessage = error.what();
            }
        } catch (...) {
            bool expected = false;
            if (failed.compare_exchange_strong(expected, true)) {
                failureCode = MlriXTransErrorCode::INTERNAL;
                failureMessage = "unknown MLRI tile failure";
            }
        }
    }

    if (failed.load()) {
        result.code = failureCode;
        result.message = failureMessage;
    }
    return result;
}

} // namespace rtengine
