/* Developer-only PackedXTransNet X-Trans wrapper. */
#pragma once

#include "array2D.h"
#include "packedxtransmodel.h"

#include <cstdint>
#include <memory>

namespace rtengine
{
constexpr const char *PACKED_XTRANS_ONNX_METHOD = "packedxtransnet-onnx";
constexpr int PACKED_XTRANS_CFA[6][6] = {
    {0, 2, 1, 2, 0, 1}, {1, 1, 0, 1, 1, 2}, {1, 1, 2, 1, 1, 0},
    {2, 0, 1, 0, 2, 1}, {1, 1, 2, 1, 1, 0}, {1, 1, 0, 1, 1, 2}
};

struct PackedXTransRunResult final {
    neural::NeuralModelError error;
    std::uint64_t tileCount = 0;
    std::uint64_t workingBufferBytes = 0;
    int margin = 0;
    int stride = 0;
    explicit operator bool() const { return !error; }
};

PackedXTransRunResult demosaicPackedXTrans(
    const array2D<float> &rawData,
    array2D<float> &red,
    array2D<float> &green,
    array2D<float> &blue,
    int width,
    int height,
    const int xtrans[6][6],
    std::shared_ptr<neural::PackedXTransRunner> runner,
    int margin);
} // namespace rtengine
