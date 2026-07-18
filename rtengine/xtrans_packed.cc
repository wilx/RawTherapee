/*
 * Independent wrapper for PackedXTransNet at upstream revision
 * 9c3cc5ab841c9afd2ed0bb702468950481043d06. The CC-BY-NC checkpoint and
 * converted ONNX remain external; no upstream executable source is copied.
 */
#include "xtrans_packed.h"

#include "xtrans_cfa.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <new>
#include <vector>

namespace rtengine
{
namespace
{
constexpr int TILE = 288;
constexpr float RAW_SCALE = 65535.f;

neural::NeuralModelError error(neural::NeuralModelErrorCode code, const char *message)
{
    return neural::NeuralModelError(code, message);
}

bool checkedMultiply(std::uint64_t a, std::uint64_t b, std::uint64_t &result)
{
    if (a && b > std::numeric_limits<std::uint64_t>::max() / a) return false;
    result = a * b;
    return true;
}

int alignedCoordinate(int coordinate, int leading, int extent)
{
    const int source = coordinate < leading ? leading - 1 - coordinate : coordinate - leading;
    return std::max(0, std::min(source, extent - 1));
}
} // namespace

PackedXTransRunResult demosaicPackedXTrans(
    const array2D<float> &rawData,
    array2D<float> &red,
    array2D<float> &green,
    array2D<float> &blue,
    int width,
    int height,
    const int xtrans[6][6],
    std::shared_ptr<neural::PackedXTransRunner> runner,
    int margin)
{
    PackedXTransRunResult result;
    result.margin = margin;
    result.stride = TILE - 2 * margin;
    if (!runner) { result.error = error(neural::NeuralModelErrorCode::SCHEMA, "PackedXTransNet runner is null"); return result; }
    if (margin != 12 && margin != 60) { result.error = error(neural::NeuralModelErrorCode::RANGE, "PackedXTransNet margin must be 12 or 60"); return result; }
    if (width <= 0 || height <= 0) { result.error = error(neural::NeuralModelErrorCode::SIZE, "raw image dimensions are empty"); return result; }
    if (rawData.getWidth() != width || rawData.getHeight() != height || red.getWidth() != width ||
        red.getHeight() != height || green.getWidth() != width || green.getHeight() != height ||
        blue.getWidth() != width || blue.getHeight() != height) {
        result.error = error(neural::NeuralModelErrorCode::SIZE, "raw and RGB array dimensions differ"); return result;
    }
    XTransCfaTransform transform{};
    if (!findCanonicalXTransTransform(xtrans, PACKED_XTRANS_CFA, transform)) {
        result.error = error(neural::NeuralModelErrorCode::SCHEMA, "unsupported PackedXTransNet X-Trans CFA"); return result;
    }
    const XTransCfaView view(transform, width, height, PACKED_XTRANS_CFA);
    if (!view.valid()) { result.error = error(neural::NeuralModelErrorCode::RANGE, "PackedXTransNet CFA mapping is invalid"); return result; }
    const int padLeft = positiveModulo(view.minimumX(), 6);
    const int padTop = positiveModulo(view.minimumY(), 6);
    if (view.width() > std::numeric_limits<int>::max() - padLeft || view.height() > std::numeric_limits<int>::max() - padTop) {
        result.error = error(neural::NeuralModelErrorCode::RANGE, "PackedXTransNet aligned dimensions overflow"); return result;
    }
    const int alignedWidth = view.width() + padLeft;
    const int alignedHeight = view.height() + padTop;
    const std::uint64_t across = (static_cast<std::uint64_t>(alignedWidth) + result.stride - 1) / result.stride;
    const std::uint64_t down = (static_cast<std::uint64_t>(alignedHeight) + result.stride - 1) / result.stride;
    if (!checkedMultiply(across, down, result.tileCount)) {
        result.error = error(neural::NeuralModelErrorCode::LIMIT, "PackedXTransNet tile grid overflow"); return result;
    }
    try {
        std::vector<float> input(neural::PACKED_XTRANS_INPUT_FLOATS);
        std::vector<float> output(neural::PACKED_XTRANS_OUTPUT_FLOATS);
        red.fill(0.f); green.fill(0.f); blue.fill(0.f);
        const std::size_t tilePixels = static_cast<std::size_t>(TILE) * TILE;
        for (std::uint64_t tileIndex = 0; tileIndex < result.tileCount; ++tileIndex) {
            const int coreX = static_cast<int>(tileIndex % across) * result.stride;
            const int coreY = static_cast<int>(tileIndex / across) * result.stride;
            for (int py = 0; py < TILE; ++py) {
                const int alignedY = std::max(0, std::min(coreY + py - margin, alignedHeight - 1));
                const int v = alignedCoordinate(alignedY, padTop, view.height());
                for (int px = 0; px < TILE; ++px) {
                    const int alignedX = std::max(0, std::min(coreX + px - margin, alignedWidth - 1));
                    const int u = alignedCoordinate(alignedX, padLeft, view.width());
                    int actualX = 0, actualY = 0;
                    view.canonicalToActual(u, v, actualX, actualY);
                    const float observed = rawData[actualY][actualX];
                    if (!std::isfinite(observed)) { result.error = error(neural::NeuralModelErrorCode::NONFINITE, "raw mosaic contains NaN or infinity"); return result; }
                    input[static_cast<std::size_t>(py) * TILE + px] = observed / RAW_SCALE;
                }
            }
            const neural::NeuralModelError inference = runner->run(input.data(), input.size(), output.data(), output.size());
            if (inference) { result.error = inference; return result; }
            for (int py = margin; py < TILE - margin; ++py) {
                const int alignedY = coreY + py - margin;
                if (alignedY < padTop || alignedY >= padTop + view.height()) continue;
                for (int px = margin; px < TILE - margin; ++px) {
                    const int alignedX = coreX + px - margin;
                    if (alignedX < padLeft || alignedX >= padLeft + view.width()) continue;
                    const std::size_t pixel = static_cast<std::size_t>(py) * TILE + px;
                    const float r = output[pixel], g = output[tilePixels + pixel], b = output[2 * tilePixels + pixel];
                    if (!std::isfinite(r) || !std::isfinite(g) || !std::isfinite(b)) {
                        result.error = error(neural::NeuralModelErrorCode::NONFINITE, "PackedXTransNet output contains NaN or infinity"); return result;
                    }
                    int actualX = 0, actualY = 0;
                    view.canonicalToActual(alignedX - padLeft, alignedY - padTop, actualX, actualY);
                    red[actualY][actualX] = r * RAW_SCALE;
                    green[actualY][actualX] = g * RAW_SCALE;
                    blue[actualY][actualX] = b * RAW_SCALE;
                }
            }
        }
        result.workingBufferBytes = static_cast<std::uint64_t>((input.size() + output.size()) * sizeof(float)) + runner->workingBufferBytes();
    } catch (const std::length_error &) {
        result.error = error(neural::NeuralModelErrorCode::LIMIT, "PackedXTransNet workspace exceeds container limits");
    } catch (const std::bad_alloc &) {
        result.error = error(neural::NeuralModelErrorCode::ALLOCATION, "cannot allocate PackedXTransNet workspace");
    }
    return result;
}
} // namespace rtengine
