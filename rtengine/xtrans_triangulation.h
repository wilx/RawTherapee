/*
 * Developer-only geometry baselines for Fujifilm X-Trans demosaicking.
 *
 * Both variants use the same fixed periodic triangulation and barycentric
 * weights.  TRIANGULATED_RGB reconstructs the three measured-color lattices
 * independently.  TRIANGULATED_CHROMA reconstructs green first, then uses the
 * red/blue triangulations for R-G and B-G.
 */
#pragma once

#include "array2D.h"

#include <cstdint>
#include <string>

namespace rtengine
{

constexpr const char *XTRANS_TRIANGULATED_RGB_METHOD =
    "xtrans-triangulated-rgb";
constexpr const char *XTRANS_TRIANGULATED_CHROMA_METHOD =
    "xtrans-triangulated-chroma";

enum class TriangulatedXTransVariant {
    INDEPENDENT_RGB,
    GREEN_CHROMA_DIFFERENCE
};

enum class TriangulatedXTransErrorCode {
    NONE,
    SIZE,
    CFA,
    NONFINITE,
    INTERNAL
};

const char *triangulatedXTransErrorCodeName(TriangulatedXTransErrorCode code);

struct TriangulatedXTransRunResult final {
    TriangulatedXTransErrorCode code = TriangulatedXTransErrorCode::NONE;
    std::string message;
    std::uint32_t workerCount = 0;

    explicit operator bool() const
    {
        return code == TriangulatedXTransErrorCode::NONE;
    }
};

TriangulatedXTransRunResult demosaicTriangulatedXTrans(
    const array2D<float> &rawData,
    array2D<float> &red,
    array2D<float> &green,
    array2D<float> &blue,
    int width,
    int height,
    const int xtrans[6][6],
    TriangulatedXTransVariant variant);

// Native-test entry point.  The CFA and all planes use actual image
// coordinates; plane storage is contiguous row-major float32.
TriangulatedXTransRunResult demosaicTriangulatedXTransReference(
    const float *mosaic,
    float *red,
    float *green,
    float *blue,
    int width,
    int height,
    const int xtrans[6][6],
    TriangulatedXTransVariant variant);

} // namespace rtengine
