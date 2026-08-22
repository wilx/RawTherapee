/*
 * Fixed-geometry X-Trans triangulation experiments.
 *
 * The tables below are the integer-pixel evaluation stencils of three
 * periodic Delaunay triangulations, one for each measured-color lattice in
 * CANONICAL_XTRANS_CFA.  Cocircular Delaunay faces were resolved once with a
 * fixed phase-periodic symbolic perturbation; for phase p=6*(y mod 6)+(x mod
 * 6), dx=eps*(((17*p+11) mod 37)/37-1/2) and
 * dy=eps*(((29*p+7) mod 41)/41-1/2), with eps=1e-6. Only the resulting
 * unperturbed integer vertices and exact barycentric weights are compiled
 * here. The perturbation therefore selects topology but never changes
 * interpolation coordinates or weights.
 *
 * Every stencil repeats after six pixels.  Vertices with offsets outside the
 * 6x6 cell deliberately refer to the neighboring cell, which makes the
 * triangulation continuous across cell boundaries.  Native samples have the
 * one-point identity stencil.  A two-point stencil means that the pixel center
 * lies exactly on a Delaunay edge and the third barycentric weight is zero.
 */

#include "xtrans_triangulation.h"

#include "xtrans_cfa.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace rtengine
{
namespace
{

struct VertexWeight final {
    int dx;
    int dy;
    float weight;
};

struct Stencil final {
    VertexWeight vertices[3];
    int count;
};

constexpr Stencil RED_STENCILS[36] = {
    {{{-1, -2, .2f}, {0, 1, .6f}, {1, -1, .2f}}, 3},
    {{{-1, 1, .25f}, {0, -1, .5f}, {1, 1, .25f}}, 3},
    {{{-1, -1, .4f}, {0, 1, .4f}, {2, 0, .2f}}, 3},
    {{{-2, -1, .2f}, {-1, 1, .2f}, {1, 0, .6f}}, 3},
    {{{0, 0, 1.f}, {0, 0, 0.f}, {0, 0, 0.f}}, 1},
    {{{-1, 0, .4f}, {0, -2, .2f}, {1, 1, .4f}}, 3},
    {{{0, 0, 1.f}, {0, 0, 0.f}, {0, 0, 0.f}}, 1},
    {{{-1, 0, .5f}, {1, 0, .5f}, {0, 0, 0.f}}, 2},
    {{{0, 0, 1.f}, {0, 0, 0.f}, {0, 0, 0.f}}, 1},
    {{{-1, 0, .5f}, {1, -1, .25f}, {1, 1, .25f}}, 3},
    {{{0, -1, .5f}, {0, 1, .5f}, {0, 0, 0.f}}, 2},
    {{{-1, -1, .25f}, {-1, 1, .25f}, {1, 0, .5f}}, 3},
    {{{-2, 0, .2f}, {0, -1, .4f}, {1, 1, .4f}}, 3},
    {{{-1, -1, .25f}, {0, 1, .5f}, {1, -1, .25f}}, 3},
    {{{-1, 1, .2f}, {0, -1, .6f}, {1, 2, .2f}}, 3},
    {{{-1, -1, .4f}, {0, 2, .2f}, {1, 0, .4f}}, 3},
    {{{0, 0, 1.f}, {0, 0, 0.f}, {0, 0, 0.f}}, 1},
    {{{-1, 0, .6f}, {1, -1, .2f}, {2, 1, .2f}}, 3},
    {{{-2, -1, .2f}, {-1, 1, .2f}, {1, 0, .6f}}, 3},
    {{{0, 0, 1.f}, {0, 0, 0.f}, {0, 0, 0.f}}, 1},
    {{{-1, 0, .4f}, {0, -2, .2f}, {1, 1, .4f}}, 3},
    {{{-1, -2, .2f}, {0, 1, .6f}, {1, -1, .2f}}, 3},
    {{{-1, 1, .25f}, {0, -1, .5f}, {1, 1, .25f}}, 3},
    {{{-1, -1, .4f}, {0, 1, .4f}, {2, 0, .2f}}, 3},
    {{{-1, 0, .5f}, {1, -1, .25f}, {1, 1, .25f}}, 3},
    {{{0, -1, .5f}, {0, 1, .5f}, {0, 0, 0.f}}, 2},
    {{{-1, -1, .25f}, {-1, 1, .25f}, {1, 0, .5f}}, 3},
    {{{0, 0, 1.f}, {0, 0, 0.f}, {0, 0, 0.f}}, 1},
    {{{-1, 0, .5f}, {1, 0, .5f}, {0, 0, 0.f}}, 2},
    {{{0, 0, 1.f}, {0, 0, 0.f}, {0, 0, 0.f}}, 1},
    {{{-1, -1, .4f}, {0, 2, .2f}, {1, 0, .4f}}, 3},
    {{{0, 0, 1.f}, {0, 0, 0.f}, {0, 0, 0.f}}, 1},
    {{{-1, 0, .6f}, {1, -1, .2f}, {2, 1, .2f}}, 3},
    {{{-2, 0, .2f}, {0, -1, .4f}, {1, 1, .4f}}, 3},
    {{{-1, -1, .25f}, {0, 1, .5f}, {1, -1, .25f}}, 3},
    {{{-1, 1, .2f}, {0, -1, .6f}, {1, 2, .2f}}, 3}
};

constexpr Stencil GREEN_STENCILS[36] = {
    {{{0, 0, 1.f}, {0, 0, 0.f}, {0, 0, 0.f}}, 1},
    {{{-1, 0, .5f}, {1, 0, .5f}, {0, 0, 0.f}}, 2},
    {{{0, 0, 1.f}, {0, 0, 0.f}, {0, 0, 0.f}}, 1},
    {{{0, 0, 1.f}, {0, 0, 0.f}, {0, 0, 0.f}}, 1},
    {{{-1, 0, .5f}, {1, 0, .5f}, {0, 0, 0.f}}, 2},
    {{{0, 0, 1.f}, {0, 0, 0.f}, {0, 0, 0.f}}, 1},
    {{{0, -1, .5f}, {0, 1, .5f}, {0, 0, 0.f}}, 2},
    {{{0, 0, 1.f}, {0, 0, 0.f}, {0, 0, 0.f}}, 1},
    {{{0, -1, .5f}, {0, 1, .5f}, {0, 0, 0.f}}, 2},
    {{{0, -1, .5f}, {0, 1, .5f}, {0, 0, 0.f}}, 2},
    {{{0, 0, 1.f}, {0, 0, 0.f}, {0, 0, 0.f}}, 1},
    {{{0, -1, .5f}, {0, 1, .5f}, {0, 0, 0.f}}, 2},
    {{{0, 0, 1.f}, {0, 0, 0.f}, {0, 0, 0.f}}, 1},
    {{{-1, 0, .5f}, {1, 0, .5f}, {0, 0, 0.f}}, 2},
    {{{0, 0, 1.f}, {0, 0, 0.f}, {0, 0, 0.f}}, 1},
    {{{0, 0, 1.f}, {0, 0, 0.f}, {0, 0, 0.f}}, 1},
    {{{-1, 0, .5f}, {1, 0, .5f}, {0, 0, 0.f}}, 2},
    {{{0, 0, 1.f}, {0, 0, 0.f}, {0, 0, 0.f}}, 1},
    {{{0, 0, 1.f}, {0, 0, 0.f}, {0, 0, 0.f}}, 1},
    {{{-1, 0, .5f}, {1, 0, .5f}, {0, 0, 0.f}}, 2},
    {{{0, 0, 1.f}, {0, 0, 0.f}, {0, 0, 0.f}}, 1},
    {{{0, 0, 1.f}, {0, 0, 0.f}, {0, 0, 0.f}}, 1},
    {{{-1, 0, .5f}, {1, 0, .5f}, {0, 0, 0.f}}, 2},
    {{{0, 0, 1.f}, {0, 0, 0.f}, {0, 0, 0.f}}, 1},
    {{{0, -1, .5f}, {0, 1, .5f}, {0, 0, 0.f}}, 2},
    {{{0, 0, 1.f}, {0, 0, 0.f}, {0, 0, 0.f}}, 1},
    {{{0, -1, .5f}, {0, 1, .5f}, {0, 0, 0.f}}, 2},
    {{{0, -1, .5f}, {0, 1, .5f}, {0, 0, 0.f}}, 2},
    {{{0, 0, 1.f}, {0, 0, 0.f}, {0, 0, 0.f}}, 1},
    {{{0, -1, .5f}, {0, 1, .5f}, {0, 0, 0.f}}, 2},
    {{{0, 0, 1.f}, {0, 0, 0.f}, {0, 0, 0.f}}, 1},
    {{{-1, 0, .5f}, {1, 0, .5f}, {0, 0, 0.f}}, 2},
    {{{0, 0, 1.f}, {0, 0, 0.f}, {0, 0, 0.f}}, 1},
    {{{0, 0, 1.f}, {0, 0, 0.f}, {0, 0, 0.f}}, 1},
    {{{-1, 0, .5f}, {1, 0, .5f}, {0, 0, 0.f}}, 2},
    {{{0, 0, 1.f}, {0, 0, 0.f}, {0, 0, 0.f}}, 1}
};

constexpr Stencil BLUE_STENCILS[36] = {
    {{{-2, -1, .2f}, {-1, 1, .2f}, {1, 0, .6f}}, 3},
    {{{0, 0, 1.f}, {0, 0, 0.f}, {0, 0, 0.f}}, 1},
    {{{-1, 0, .4f}, {0, -2, .2f}, {1, 1, .4f}}, 3},
    {{{-1, -2, .2f}, {0, 1, .6f}, {1, -1, .2f}}, 3},
    {{{-1, 1, .25f}, {0, -1, .5f}, {1, 1, .25f}}, 3},
    {{{-1, -1, .4f}, {0, 1, .4f}, {2, 0, .2f}}, 3},
    {{{-1, 0, .5f}, {1, -1, .25f}, {1, 1, .25f}}, 3},
    {{{0, -1, .5f}, {0, 1, .5f}, {0, 0, 0.f}}, 2},
    {{{-1, -1, .25f}, {-1, 1, .25f}, {1, 0, .5f}}, 3},
    {{{0, 0, 1.f}, {0, 0, 0.f}, {0, 0, 0.f}}, 1},
    {{{-1, 0, .5f}, {1, 0, .5f}, {0, 0, 0.f}}, 2},
    {{{0, 0, 1.f}, {0, 0, 0.f}, {0, 0, 0.f}}, 1},
    {{{-1, -1, .4f}, {0, 2, .2f}, {1, 0, .4f}}, 3},
    {{{0, 0, 1.f}, {0, 0, 0.f}, {0, 0, 0.f}}, 1},
    {{{-1, 0, .4f}, {0, 2, .2f}, {1, -1, .4f}}, 3},
    {{{-1, 2, .2f}, {0, -1, .6f}, {1, 1, .2f}}, 3},
    {{{-1, -1, .25f}, {0, 1, .5f}, {1, -1, .25f}}, 3},
    {{{-1, 1, .2f}, {0, -1, .6f}, {1, 2, .2f}}, 3},
    {{{-1, -2, .2f}, {0, 1, .6f}, {1, -1, .2f}}, 3},
    {{{-1, 1, .25f}, {0, -1, .5f}, {1, 1, .25f}}, 3},
    {{{-1, -1, .2f}, {0, 1, .6f}, {1, -2, .2f}}, 3},
    {{{-1, 1, .4f}, {0, -2, .2f}, {1, 0, .4f}}, 3},
    {{{0, 0, 1.f}, {0, 0, 0.f}, {0, 0, 0.f}}, 1},
    {{{-1, 0, .4f}, {0, -2, .2f}, {1, 1, .4f}}, 3},
    {{{0, 0, 1.f}, {0, 0, 0.f}, {0, 0, 0.f}}, 1},
    {{{-1, 0, .5f}, {1, 0, .5f}, {0, 0, 0.f}}, 2},
    {{{0, 0, 1.f}, {0, 0, 0.f}, {0, 0, 0.f}}, 1},
    {{{-1, 0, .5f}, {1, -1, .25f}, {1, 1, .25f}}, 3},
    {{{0, -1, .5f}, {0, 1, .5f}, {0, 0, 0.f}}, 2},
    {{{-1, -1, .25f}, {-1, 1, .25f}, {1, 0, .5f}}, 3},
    {{{-2, 0, .2f}, {0, -1, .4f}, {1, 1, .4f}}, 3},
    {{{-1, -1, .25f}, {0, 1, .5f}, {1, -1, .25f}}, 3},
    {{{-1, 1, .2f}, {0, -1, .6f}, {1, 2, .2f}}, 3},
    {{{-1, -1, .4f}, {0, 2, .2f}, {1, 0, .4f}}, 3},
    {{{0, 0, 1.f}, {0, 0, 0.f}, {0, 0, 0.f}}, 1},
    {{{-1, 0, .6f}, {1, -1, .2f}, {2, 1, .2f}}, 3}
};

const Stencil *stencilsForColor(int color)
{
    switch (color) {
        case 0: return RED_STENCILS;
        case 1: return GREEN_STENCILS;
        case 2: return BLUE_STENCILS;
        default: return nullptr;
    }
}

int floorDivideBySix(int value)
{
    return value >= 0 ? value / 6 : -((-value + 5) / 6);
}

bool nearestCongruentCoordinate(int target, int residue, int length, int &result)
{
    if (length <= 0 || residue < 0 || residue >= 6 || residue >= length) {
        return false;
    }
    const int maximumK = (length - 1 - residue) / 6;
    const int lower = std::max(0, std::min(maximumK, floorDivideBySix(target - residue)));
    const int upper = std::max(0, std::min(maximumK, lower + 1));
    const int lowerCoordinate = residue + 6 * lower;
    const int upperCoordinate = residue + 6 * upper;
    result = std::abs(target - lowerCoordinate) <= std::abs(target - upperCoordinate)
        ? lowerCoordinate : upperCoordinate;
    return true;
}

bool findNearestNativeSample(
    const XTransCfaView &view,
    int targetU,
    int targetV,
    int color,
    int &actualX,
    int &actualY)
{
    std::int64_t bestDistance = std::numeric_limits<std::int64_t>::max();
    int bestU = 0;
    int bestV = 0;
    bool found = false;

    for (int phaseY = 0; phaseY < 6; ++phaseY) {
        for (int phaseX = 0; phaseX < 6; ++phaseX) {
            const int localResidueX = positiveModulo(phaseX - view.minimumX(), 6);
            const int localResidueY = positiveModulo(phaseY - view.minimumY(), 6);
            if (CANONICAL_XTRANS_CFA[phaseY][phaseX] != color) {
                continue;
            }
            int candidateU = 0;
            int candidateV = 0;
            if (!nearestCongruentCoordinate(targetU, localResidueX, view.width(), candidateU) ||
                    !nearestCongruentCoordinate(targetV, localResidueY, view.height(), candidateV)) {
                continue;
            }
            const std::int64_t du = static_cast<std::int64_t>(candidateU) - targetU;
            const std::int64_t dv = static_cast<std::int64_t>(candidateV) - targetV;
            const std::int64_t distance = du * du + dv * dv;
            if (!found || distance < bestDistance ||
                    (distance == bestDistance &&
                     (candidateV < bestV || (candidateV == bestV && candidateU < bestU)))) {
                found = true;
                bestDistance = distance;
                bestU = candidateU;
                bestV = candidateV;
            }
        }
    }

    if (!found) {
        return false;
    }
    view.canonicalToActual(bestU, bestV, actualX, actualY);
    // An in-range canonical coordinate maps back into the original image
    // rectangle even when the transform swaps width and height.
    return true;
}

bool locateVertex(
    const XTransCfaView &view,
    int u,
    int v,
    int color,
    int &actualX,
    int &actualY)
{
    if (u >= 0 && u < view.width() && v >= 0 && v < view.height()) {
        if (view.colorAtCanonical(u, v) != color) {
            return false;
        }
        view.canonicalToActual(u, v, actualX, actualY);
        return true;
    }
    // Constant extension on each sparse measured-color lattice: an exterior
    // triangle vertex takes the nearest finite-image sample of that same
    // color.  This never substitutes a differently colored mosaic value.
    return findNearestNativeSample(view, u, v, color, actualX, actualY);
}

template<typename Getter>
bool interpolate(
    const XTransCfaView &view,
    int u,
    int v,
    int color,
    const Getter &getter,
    float &output)
{
    const int phaseX = positiveModulo(u + view.minimumX(), 6);
    const int phaseY = positiveModulo(v + view.minimumY(), 6);
    const Stencil &stencil = stencilsForColor(color)[phaseY * 6 + phaseX];
    float value = 0.f;
    for (int index = 0; index < stencil.count; ++index) {
        const VertexWeight &vertex = stencil.vertices[index];
        int x = 0;
        int y = 0;
        if (!locateVertex(view, u + vertex.dx, v + vertex.dy, color, x, y)) {
            return false;
        }
        value += vertex.weight * getter(x, y);
    }
    output = value;
    return std::isfinite(value);
}

struct ArrayInput final {
    const array2D<float> &data;
    float operator()(int x, int y) const { return data[y][x]; }
};

struct ArrayOutput final {
    array2D<float> &red;
    array2D<float> &green;
    array2D<float> &blue;
    float greenAt(int x, int y) const { return green[y][x]; }
    void set(int color, int x, int y, float value)
    {
        (color == 0 ? red : color == 1 ? green : blue)[y][x] = value;
    }
};

struct PointerInput final {
    const float *data;
    int width;
    float operator()(int x, int y) const
    {
        return data[static_cast<std::size_t>(y) * width + x];
    }
};

struct PointerOutput final {
    float *red;
    float *green;
    float *blue;
    int width;
    float greenAt(int x, int y) const
    {
        return green[static_cast<std::size_t>(y) * width + x];
    }
    void set(int color, int x, int y, float value)
    {
        float *plane = color == 0 ? red : color == 1 ? green : blue;
        plane[static_cast<std::size_t>(y) * width + x] = value;
    }
};

template<typename Input, typename Output>
TriangulatedXTransRunResult run(
    const Input &input,
    Output &output,
    int width,
    int height,
    const int xtrans[6][6],
    TriangulatedXTransVariant variant)
{
    TriangulatedXTransRunResult result;
    if (width <= 0 || height <= 0) {
        result.code = TriangulatedXTransErrorCode::SIZE;
        result.message = "image dimensions must be positive";
        return result;
    }

    XTransCfaTransform transform;
    if (!findCanonicalXTransTransform(xtrans, transform)) {
        result.code = TriangulatedXTransErrorCode::CFA;
        result.message = "unsupported 6x6 X-Trans CFA matrix";
        return result;
    }
    const XTransCfaView view(transform, width, height);
    if (!view.valid()) {
        result.code = TriangulatedXTransErrorCode::CFA;
        result.message = "X-Trans coordinate transform is invalid";
        return result;
    }

    int colorCounts[3] = {0, 0, 0};
    for (int y = 0; y < std::min(height, 6); ++y) {
        for (int x = 0; x < std::min(width, 6); ++x) {
            ++colorCounts[xtrans[y][x]];
        }
    }
    int invalidInput = 0;
#ifdef _OPENMP
#pragma omp parallel for reduction(|:invalidInput) schedule(static)
#endif
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            invalidInput |= !std::isfinite(input(x, y));
        }
    }
    if (invalidInput) {
        result.code = TriangulatedXTransErrorCode::NONFINITE;
        result.message = "mosaic contains a non-finite sample";
        return result;
    }
    if (!colorCounts[0] || !colorCounts[1] || !colorCounts[2]) {
        result.code = TriangulatedXTransErrorCode::SIZE;
        result.message = "finite image does not contain every measured CFA color";
        return result;
    }

#ifdef _OPENMP
    result.workerCount = static_cast<std::uint32_t>(std::max(1, omp_get_max_threads()));
#else
    result.workerCount = 1;
#endif
    int failed = 0;

    if (variant == TriangulatedXTransVariant::INDEPENDENT_RGB) {
#ifdef _OPENMP
#pragma omp parallel for reduction(|:failed) schedule(static)
#endif
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int u = 0;
                int v = 0;
                view.actualToCanonical(x, y, u, v);
                const int measuredColor = xtrans[y % 6][x % 6];
                for (int color = 0; color < 3; ++color) {
                    float value = 0.f;
                    if (color == measuredColor) {
                        value = input(x, y);
                    } else if (!interpolate(view, u, v, color, input, value)) {
                        failed = 1;
                    }
                    output.set(color, x, y, value);
                }
            }
        }
    } else if (variant == TriangulatedXTransVariant::GREEN_CHROMA_DIFFERENCE) {
#ifdef _OPENMP
#pragma omp parallel for reduction(|:failed) schedule(static)
#endif
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int u = 0;
                int v = 0;
                view.actualToCanonical(x, y, u, v);
                float value = 0.f;
                if (xtrans[y % 6][x % 6] == 1) {
                    value = input(x, y);
                } else if (!interpolate(view, u, v, 1, input, value)) {
                    failed = 1;
                }
                output.set(1, x, y, value);
            }
        }

#ifdef _OPENMP
#pragma omp parallel for reduction(|:failed) schedule(static)
#endif
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int u = 0;
                int v = 0;
                view.actualToCanonical(x, y, u, v);
                const int measuredColor = xtrans[y % 6][x % 6];
                for (int color = 0; color <= 2; color += 2) {
                    float value = 0.f;
                    if (color == measuredColor) {
                        value = input(x, y);
                    } else {
                        const auto difference = [&input, &output](int sx, int sy) {
                            return input(sx, sy) - output.greenAt(sx, sy);
                        };
                        float chroma = 0.f;
                        if (!interpolate(view, u, v, color, difference, chroma)) {
                            failed = 1;
                        }
                        value = output.greenAt(x, y) + chroma;
                    }
                    if (!std::isfinite(value)) {
                        failed = 1;
                    }
                    output.set(color, x, y, value);
                }
            }
        }
    } else {
        result.code = TriangulatedXTransErrorCode::INTERNAL;
        result.message = "unknown triangulated X-Trans variant";
        return result;
    }

    if (failed) {
        result.code = TriangulatedXTransErrorCode::NONFINITE;
        result.message = "triangulated reconstruction failed or produced a non-finite value";
    }
    return result;
}

} // namespace

const char *triangulatedXTransErrorCodeName(TriangulatedXTransErrorCode code)
{
    switch (code) {
        case TriangulatedXTransErrorCode::NONE: return "NONE";
        case TriangulatedXTransErrorCode::SIZE: return "SIZE";
        case TriangulatedXTransErrorCode::CFA: return "CFA";
        case TriangulatedXTransErrorCode::NONFINITE: return "NONFINITE";
        case TriangulatedXTransErrorCode::INTERNAL: return "INTERNAL";
    }
    return "INTERNAL";
}

TriangulatedXTransRunResult demosaicTriangulatedXTrans(
    const array2D<float> &rawData,
    array2D<float> &red,
    array2D<float> &green,
    array2D<float> &blue,
    int width,
    int height,
    const int xtrans[6][6],
    TriangulatedXTransVariant variant)
{
    const ArrayInput input{rawData};
    ArrayOutput output{red, green, blue};
    return run(input, output, width, height, xtrans, variant);
}

TriangulatedXTransRunResult demosaicTriangulatedXTransReference(
    const float *mosaic,
    float *red,
    float *green,
    float *blue,
    int width,
    int height,
    const int xtrans[6][6],
    TriangulatedXTransVariant variant)
{
    TriangulatedXTransRunResult result;
    if (!mosaic || !red || !green || !blue) {
        result.code = TriangulatedXTransErrorCode::INTERNAL;
        result.message = "input and output pointers must not be null";
        return result;
    }
    const PointerInput input{mosaic, width};
    PointerOutput output{red, green, blue, width};
    return run(input, output, width, height, xtrans, variant);
}

} // namespace rtengine
