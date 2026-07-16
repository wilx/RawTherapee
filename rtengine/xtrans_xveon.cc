/*
 * Developer-only RawTherapee wrapper for the pinned X-veon X-Trans model.
 *
 * This independently reproduces the observed public model interface and web
 * tiling contract at upstream revision
 * 2e6b96c63559aa3909b0c7c1bc45dfd4b5dfe680. No upstream source is copied.
 * The upstream repository and model have no explicit license at that revision,
 * so the method is intentionally hidden and requires external weights.
 */
#include "xtrans_xveon.h"

#include "xtrans_cfa.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <new>
#include <stdexcept>
#include <vector>

namespace rtengine
{

namespace
{

constexpr int PATCH_SIZE = 288;
constexpr int OVERLAP = 48;
constexpr int STRIDE = PATCH_SIZE - OVERLAP;
constexpr float RAW_SCALE = 65535.f;

neural::NeuralModelError error(neural::NeuralModelErrorCode code, const char *message)
{
    return neural::NeuralModelError(code, message);
}

bool checkedMultiply(std::uint64_t left, std::uint64_t right, std::uint64_t &result)
{
    if (left && right > std::numeric_limits<std::uint64_t>::max() / left) {
        return false;
    }
    result = left * right;
    return true;
}

bool checkedAdd(std::uint64_t left, std::uint64_t right, std::uint64_t &result)
{
    if (right > std::numeric_limits<std::uint64_t>::max() - left) {
        return false;
    }
    result = left + right;
    return true;
}

bool paddedExtent(int extent, int &padded, std::uint64_t &tiles)
{
    if (extent <= 0) {
        return false;
    }
    const std::uint64_t steps = extent <= OVERLAP
        ? 0
        : (static_cast<std::uint64_t>(extent - OVERLAP) + STRIDE - 1) / STRIDE;
    const std::uint64_t value = steps * STRIDE + PATCH_SIZE;
    if (value > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
        return false;
    }
    padded = static_cast<int>(value);
    tiles = steps + 1;
    return true;
}

int reflectedLeadingCoordinate(int paddedCoordinate, int padding, int extent)
{
    const int coordinate = paddedCoordinate < padding
        ? padding - 1 - paddedCoordinate
        : paddedCoordinate - padding;
    return std::min(coordinate, extent - 1);
}

} // namespace

XVeonXTransRunResult demosaicXVeonXTrans(
    const array2D<float> &rawData,
    array2D<float> &red,
    array2D<float> &green,
    array2D<float> &blue,
    int width,
    int height,
    const int xtrans[6][6],
    std::shared_ptr<neural::XVeonXTransRunner> runner)
{
    XVeonXTransRunResult result;
    if (!runner) {
        result.error = error(neural::NeuralModelErrorCode::SCHEMA, "X-veon runner is null");
        return result;
    }
    if (width <= 0 || height <= 0) {
        result.error = error(neural::NeuralModelErrorCode::SIZE, "raw image dimensions are empty");
        return result;
    }
    if (rawData.getWidth() != width || rawData.getHeight() != height ||
        red.getWidth() != width || red.getHeight() != height ||
        green.getWidth() != width || green.getHeight() != height ||
        blue.getWidth() != width || blue.getHeight() != height) {
        result.error = error(neural::NeuralModelErrorCode::SIZE, "raw and RGB array dimensions differ");
        return result;
    }

    XTransCfaTransform transform{};
    if (!findCanonicalXTransTransform(xtrans, XVEON_XTRANS_CFA, transform)) {
        result.error = error(neural::NeuralModelErrorCode::SCHEMA,
            "X-Trans CFA is not a supported phase or orientation of X-veon's canonical pattern");
        return result;
    }
    const XTransCfaView view(transform, width, height, XVEON_XTRANS_CFA);
    if (!view.valid()) {
        result.error = error(neural::NeuralModelErrorCode::RANGE, "X-veon canonical CFA mapping is invalid");
        return result;
    }

    const int padLeft = positiveModulo(view.minimumX(), 6);
    const int padTop = positiveModulo(view.minimumY(), 6);
    if (view.width() > std::numeric_limits<int>::max() - padLeft ||
        view.height() > std::numeric_limits<int>::max() - padTop) {
        result.error = error(neural::NeuralModelErrorCode::RANGE, "X-veon aligned dimensions overflow");
        return result;
    }
    const int alignedWidth = view.width() + padLeft;
    const int alignedHeight = view.height() + padTop;
    std::uint64_t tilesAcross = 0;
    std::uint64_t tilesDown = 0;
    if (!paddedExtent(alignedWidth, result.paddedWidth, tilesAcross) ||
        !paddedExtent(alignedHeight, result.paddedHeight, tilesDown) ||
        !checkedMultiply(tilesAcross, tilesDown, result.tileCount)) {
        result.error = error(neural::NeuralModelErrorCode::LIMIT, "X-veon tile grid exceeds supported limits");
        return result;
    }

    std::uint64_t pixels = 0;
    std::uint64_t weightBytes = 0;
    if (!checkedMultiply(static_cast<std::uint64_t>(width), static_cast<std::uint64_t>(height), pixels) ||
        !checkedMultiply(pixels, sizeof(float), weightBytes) ||
        pixels > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) ||
        pixels > static_cast<std::uint64_t>(std::vector<float>().max_size())) {
        result.error = error(neural::NeuralModelErrorCode::LIMIT, "X-veon image storage exceeds supported limits");
        return result;
    }

    try {
        std::vector<float> weights(static_cast<std::size_t>(pixels), 0.f);
        std::vector<float> input(neural::XVEON_INPUT_FLOATS, 0.f);
        std::vector<float> output(neural::XVEON_OUTPUT_FLOATS, 0.f);
        std::array<float, PATCH_SIZE> ramp{};
        ramp.fill(1.f);
        for (int index = 0; index < OVERLAP; ++index) {
            ramp[index] = static_cast<float>(index) / OVERLAP;
            ramp[PATCH_SIZE - 1 - index] = static_cast<float>(index) / OVERLAP;
        }

        red.fill(0.f);
        green.fill(0.f);
        blue.fill(0.f);
        const std::size_t patchPixels = static_cast<std::size_t>(PATCH_SIZE) * PATCH_SIZE;
        for (std::uint64_t tileIndex = 0; tileIndex < result.tileCount; ++tileIndex) {
            const int tileX = static_cast<int>(tileIndex % tilesAcross) * STRIDE;
            const int tileY = static_cast<int>(tileIndex / tilesAcross) * STRIDE;
            std::fill(input.begin(), input.begin() + patchPixels, 0.f);

            for (int py = 0; py < PATCH_SIZE; ++py) {
                const int paddedY = tileY + py;
                for (int px = 0; px < PATCH_SIZE; ++px) {
                    const int paddedX = tileX + px;
                    const std::size_t tilePixel = static_cast<std::size_t>(py) * PATCH_SIZE + px;
                    const int channel = XVEON_XTRANS_CFA[paddedY % 6][paddedX % 6];
                    input[(static_cast<std::size_t>(channel) + 1) * patchPixels + tilePixel] = 1.f;

                    if (paddedX >= alignedWidth || paddedY >= alignedHeight) {
                        continue; // Upstream production path zero-extends right and bottom.
                    }
                    const int u = reflectedLeadingCoordinate(paddedX, padLeft, view.width());
                    const int v = reflectedLeadingCoordinate(paddedY, padTop, view.height());
                    int actualX = 0;
                    int actualY = 0;
                    view.canonicalToActual(u, v, actualX, actualY);
                    const float observed = rawData[actualY][actualX];
                    if (!std::isfinite(observed)) {
                        result.error = error(neural::NeuralModelErrorCode::NONFINITE,
                            "raw mosaic contains NaN or infinity");
                        return result;
                    }
                    input[tilePixel] = observed / RAW_SCALE;
                }
            }

            const neural::NeuralModelError inference = runner->run(
                input.data(), input.size(), output.data(), output.size());
            if (inference) {
                result.error = inference;
                return result;
            }

            for (int py = 0; py < PATCH_SIZE; ++py) {
                const int paddedY = tileY + py;
                if (paddedY < padTop || paddedY >= padTop + view.height()) {
                    continue;
                }
                for (int px = 0; px < PATCH_SIZE; ++px) {
                    const int paddedX = tileX + px;
                    if (paddedX < padLeft || paddedX >= padLeft + view.width()) {
                        continue;
                    }
                    const std::size_t tilePixel = static_cast<std::size_t>(py) * PATCH_SIZE + px;
                    const float r = output[tilePixel];
                    const float g = output[patchPixels + tilePixel];
                    const float b = output[2 * patchPixels + tilePixel];
                    if (!std::isfinite(r) || !std::isfinite(g) || !std::isfinite(b)) {
                        result.error = error(neural::NeuralModelErrorCode::NONFINITE,
                            "X-veon output contains NaN or infinity");
                        return result;
                    }
                    int actualX = 0;
                    int actualY = 0;
                    view.canonicalToActual(paddedX - padLeft, paddedY - padTop, actualX, actualY);
                    const std::size_t actualPixel = static_cast<std::size_t>(actualY) * width + actualX;
                    const float weight = ramp[py] * ramp[px];
                    red[actualY][actualX] += r * weight;
                    green[actualY][actualX] += g * weight;
                    blue[actualY][actualX] += b * weight;
                    weights[actualPixel] += weight;
                }
            }
        }

        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                const std::size_t pixel = static_cast<std::size_t>(y) * width + x;
                if (weights[pixel] > 1e-8f) {
                    const float scale = RAW_SCALE / weights[pixel];
                    red[y][x] *= scale;
                    green[y][x] *= scale;
                    blue[y][x] *= scale;
                }
                if (!std::isfinite(red[y][x]) || !std::isfinite(green[y][x]) || !std::isfinite(blue[y][x])) {
                    result.error = error(neural::NeuralModelErrorCode::NONFINITE,
                        "normalized X-veon output contains NaN or infinity");
                    return result;
                }
            }
        }

        const std::uint64_t tileBytes = static_cast<std::uint64_t>(
            (input.size() + output.size()) * sizeof(float));
        std::uint64_t localBytes = 0;
        if (!checkedAdd(weightBytes, tileBytes, localBytes) ||
            !checkedAdd(localBytes, runner->workingBufferBytes(), result.workingBufferBytes)) {
            result.error = error(neural::NeuralModelErrorCode::LIMIT,
                "X-veon working-buffer accounting overflowed");
            return result;
        }
    } catch (const std::length_error &) {
        result.error = error(neural::NeuralModelErrorCode::LIMIT,
            "X-veon tiling workspace exceeds container limits");
    } catch (const std::bad_alloc &) {
        result.error = error(neural::NeuralModelErrorCode::ALLOCATION,
            "cannot allocate X-veon tiling workspace");
    }
    return result;
}

} // namespace rtengine
