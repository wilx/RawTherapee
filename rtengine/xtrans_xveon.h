/* Developer-only X-veon X-Trans wrapper. Not part of the public method list. */
#pragma once

#include "array2D.h"
#include "xveonxtransmodel.h"

#include <cstdint>
#include <memory>

namespace rtengine
{

constexpr const char *XVEON_XTRANS_ONNX_METHOD = "xveon-xtrans-onnx";

constexpr int XVEON_XTRANS_CFA[6][6] = {
    {0, 2, 1, 2, 0, 1},
    {1, 1, 0, 1, 1, 2},
    {1, 1, 2, 1, 1, 0},
    {2, 0, 1, 0, 2, 1},
    {1, 1, 2, 1, 1, 0},
    {1, 1, 0, 1, 1, 2}
};

struct XVeonXTransRunResult final {
    neural::NeuralModelError error;
    std::uint64_t tileCount = 0;
    std::uint64_t workingBufferBytes = 0;
    int paddedWidth = 0;
    int paddedHeight = 0;

    explicit operator bool() const
    {
        return !error;
    }
};

XVeonXTransRunResult demosaicXVeonXTrans(
    const array2D<float> &rawData,
    array2D<float> &red,
    array2D<float> &green,
    array2D<float> &blue,
    int width,
    int height,
    const int xtrans[6][6],
    std::shared_ptr<neural::XVeonXTransRunner> runner);

} // namespace rtengine
