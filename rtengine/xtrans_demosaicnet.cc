/*
 *  This file is part of RawTherapee.
 *
 *  Phase 9 developer integration for:
 *
 *    M. Gharbi, G. Chaurasia, S. Paris, and F. Durand,
 *    "Deep Joint Demosaicking and Denoising," ACM TOG 35(6), 2016.
 *
 *  This wrapper deliberately contains only the documented raw-domain
 *  transforms and phase-stable tiling. It does not reinject samples, suppress
 *  false colour, sharpen, denoise, blend, or reconstruct highlights.
 */
#include "xtrans_demosaicnet.h"

#include "xtrans_cfa.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace rtengine
{

namespace
{

constexpr int INPUT_TILE = 192;
constexpr int RECEPTIVE_HALO = 12;
constexpr int OUTPUT_CORE = INPUT_TILE - 2 * RECEPTIVE_HALO;
static_assert(OUTPUT_CORE % 6 == 0, "neural tile cores must preserve the X-Trans phase");
constexpr float RAW_SCALE = 65535.f;
constexpr float GAMMA = 2.2f;

std::mutex modelCacheMutex;
std::map<std::string, std::shared_ptr<const neural::DemosaicNetXTransModel>> modelCache;

int reflectWithoutEdgeRepetition(int coordinate, int size)
{
    if (size <= 1) {
        return 0;
    }

    const int period = 2 * (size - 1);
    const int folded = positiveModulo(coordinate, period);
    return folded < size ? folded : period - folded;
}

float normalizeObserved(float value, DemosaicNetXTransDomain domain)
{
    value = std::max(0.f, std::min(1.f, value / RAW_SCALE));
    if (domain == DemosaicNetXTransDomain::GAMMA22) {
        value = std::pow(value, 1.f / GAMMA);
    }
    return value;
}

float decodeOutput(float value, DemosaicNetXTransDomain domain)
{
    value = std::max(0.f, std::min(1.f, value));
    if (domain == DemosaicNetXTransDomain::GAMMA22) {
        value = std::pow(value, GAMMA);
    }
    return value * RAW_SCALE;
}

neural::NeuralModelError makeError(neural::NeuralModelErrorCode code, const char *message)
{
    return neural::NeuralModelError(code, message);
}

} // namespace

neural::DemosaicNetXTransLoadResult loadCachedDemosaicNetXTransModel(
    const Glib::ustring &path)
{
    if (path.empty()) {
        return {nullptr, makeError(
            neural::NeuralModelErrorCode::IO,
            "RT_DEMOSAICNET_XTRANS_MODEL is unset or empty")};
    }

    const std::string key = path.raw();
    std::lock_guard<std::mutex> lock(modelCacheMutex);
    const auto found = modelCache.find(key);
    if (found != modelCache.end()) {
        return {found->second, neural::NeuralModelError()};
    }

    neural::DemosaicNetXTransLoadResult loaded =
        neural::loadDemosaicNetXTransModel(path);
    if (loaded) {
        modelCache.emplace(key, loaded.model);
    }
    return loaded;
}

DemosaicNetXTransRunResult demosaicDemosaicNetXTrans(
    const array2D<float> &rawData,
    array2D<float> &red,
    array2D<float> &green,
    array2D<float> &blue,
    int width,
    int height,
    const int xtrans[6][6],
    DemosaicNetXTransDomain domain,
    std::shared_ptr<const neural::DemosaicNetXTransModel> model)
{
    DemosaicNetXTransRunResult result;
    if (!model) {
        result.error = makeError(neural::NeuralModelErrorCode::SCHEMA, "neural model is null");
        return result;
    }
    if (width <= 0 || height <= 0) {
        result.error = makeError(neural::NeuralModelErrorCode::SIZE, "raw image dimensions are empty");
        return result;
    }
    if (rawData.getWidth() != width || rawData.getHeight() != height ||
        red.getWidth() != width || red.getHeight() != height ||
        green.getWidth() != width || green.getHeight() != height ||
        blue.getWidth() != width || blue.getHeight() != height) {
        result.error = makeError(neural::NeuralModelErrorCode::SIZE, "raw and RGB array dimensions differ");
        return result;
    }
    if (domain != DemosaicNetXTransDomain::LINEAR &&
        domain != DemosaicNetXTransDomain::GAMMA22) {
        result.error = makeError(neural::NeuralModelErrorCode::ENUM, "raw-domain transform is unsupported");
        return result;
    }

    XTransCfaTransform transform {};
    if (!findCanonicalXTransTransform(xtrans, transform)) {
        result.error = makeError(
            neural::NeuralModelErrorCode::SCHEMA,
            "X-Trans CFA is not a supported phase or orientation of the canonical 6x6 pattern");
        return result;
    }

    const XTransCfaView view(transform, width, height);
    if (!view.valid()) {
        result.error = makeError(neural::NeuralModelErrorCode::RANGE, "canonical CFA image mapping is invalid");
        return result;
    }

    const std::uint64_t tilesAcross =
        (static_cast<std::uint64_t>(view.width()) + OUTPUT_CORE - 1) / OUTPUT_CORE;
    const std::uint64_t tilesDown =
        (static_cast<std::uint64_t>(view.height()) + OUTPUT_CORE - 1) / OUTPUT_CORE;
    if (tilesAcross > std::numeric_limits<std::uint64_t>::max() / tilesDown ||
        tilesAcross * tilesDown > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
        result.error = makeError(neural::NeuralModelErrorCode::LIMIT, "neural tile count exceeds the supported limit");
        return result;
    }
    result.tileCount = tilesAcross * tilesDown;

    std::atomic<bool> failed(false);
    std::atomic<std::uint32_t> workersUsed(0);
    std::mutex errorMutex;
    neural::NeuralModelError firstError;
    std::uint64_t workspaceBytes = 0;
    std::mutex workspaceMutex;

#ifdef _OPENMP
    #pragma omp parallel
#endif
    {
        bool workerActive = false;
        bool workspaceRecorded = false;
        std::unique_ptr<neural::DemosaicNetXTransExecutor> executor;
        std::vector<float> input;
        std::vector<float> output;

#ifdef _OPENMP
        #pragma omp for schedule(dynamic, 1)
#endif
        for (int tileIndex = 0; tileIndex < static_cast<int>(result.tileCount); ++tileIndex) {
            if (failed.load(std::memory_order_relaxed)) {
                continue;
            }
            if (!workerActive) {
                try {
                    neural::DemosaicNetXTransExecutorCreateResult created =
                        neural::createDemosaicNetXTransExecutor(model);
                    if (!created) {
                        std::lock_guard<std::mutex> lock(errorMutex);
                        if (!failed.exchange(true)) {
                            firstError = created.error;
                        }
                        continue;
                    }
                    executor = std::move(created.executor);
                    input.resize(3u * INPUT_TILE * INPUT_TILE);
                    output.resize(3u * OUTPUT_CORE * OUTPUT_CORE);
                    workerActive = true;
                    workersUsed.fetch_add(1, std::memory_order_relaxed);
                } catch (const std::bad_alloc &) {
                    std::lock_guard<std::mutex> lock(errorMutex);
                    if (!failed.exchange(true)) {
                        firstError = makeError(neural::NeuralModelErrorCode::ALLOCATION, "neural worker allocation failed");
                    }
                    continue;
                }
            }

            const int tileX = static_cast<int>(tileIndex % tilesAcross) * OUTPUT_CORE;
            const int tileY = static_cast<int>(tileIndex / tilesAcross) * OUTPUT_CORE;
            std::fill(input.begin(), input.end(), 0.f);

            bool tileFinite = true;
            for (int iy = 0; iy < INPUT_TILE && tileFinite; ++iy) {
                const int sourceV = reflectWithoutEdgeRepetition(
                    tileY + iy - RECEPTIVE_HALO, view.height());
                for (int ix = 0; ix < INPUT_TILE; ++ix) {
                    const int sourceU = reflectWithoutEdgeRepetition(
                        tileX + ix - RECEPTIVE_HALO, view.width());
                    int actualX = 0;
                    int actualY = 0;
                    view.canonicalToActual(sourceU, sourceV, actualX, actualY);
                    const float observed = rawData[actualY][actualX];
                    if (!std::isfinite(observed)) {
                        tileFinite = false;
                        break;
                    }
                    const int channel = view.colorAtCanonical(sourceU, sourceV);
                    const std::size_t offset =
                        (static_cast<std::size_t>(channel) * INPUT_TILE + iy) * INPUT_TILE + ix;
                    input[offset] = normalizeObserved(observed, domain);
                }
            }

            if (!tileFinite) {
                std::lock_guard<std::mutex> lock(errorMutex);
                if (!failed.exchange(true)) {
                    firstError = makeError(neural::NeuralModelErrorCode::NONFINITE, "raw mosaic contains NaN or infinity");
                }
                continue;
            }

            const neural::NeuralModelError inferenceError = executor->run(
                input.data(), input.size(), INPUT_TILE, INPUT_TILE,
                output.data(), output.size());
            if (inferenceError) {
                std::lock_guard<std::mutex> lock(errorMutex);
                if (!failed.exchange(true)) {
                    firstError = inferenceError;
                }
                continue;
            }
            if (!workspaceRecorded) {
                std::lock_guard<std::mutex> lock(workspaceMutex);
                workspaceBytes = std::max(workspaceBytes, executor->workspaceBytes());
                workspaceRecorded = true;
            }

            const int copyWidth = std::min(OUTPUT_CORE, view.width() - tileX);
            const int copyHeight = std::min(OUTPUT_CORE, view.height() - tileY);
            for (int oy = 0; oy < copyHeight && !failed.load(std::memory_order_relaxed); ++oy) {
                for (int ox = 0; ox < copyWidth; ++ox) {
                    int actualX = 0;
                    int actualY = 0;
                    view.canonicalToActual(tileX + ox, tileY + oy, actualX, actualY);
                    const std::size_t pixel = static_cast<std::size_t>(oy) * OUTPUT_CORE + ox;
                    const float r = decodeOutput(output[pixel], domain);
                    const float g = decodeOutput(output[OUTPUT_CORE * OUTPUT_CORE + pixel], domain);
                    const float b = decodeOutput(output[2 * OUTPUT_CORE * OUTPUT_CORE + pixel], domain);
                    if (!std::isfinite(r) || !std::isfinite(g) || !std::isfinite(b)) {
                        std::lock_guard<std::mutex> lock(errorMutex);
                        if (!failed.exchange(true)) {
                            firstError = makeError(neural::NeuralModelErrorCode::NONFINITE, "neural output contains NaN or infinity");
                        }
                        break;
                    }
                    red[actualY][actualX] = r;
                    green[actualY][actualX] = g;
                    blue[actualY][actualX] = b;
                }
            }
        }
    }

    result.workerCount = workersUsed.load();
    result.workspaceBytesPerWorker = workspaceBytes;
    if (failed.load()) {
        result.error = firstError.code == neural::NeuralModelErrorCode::NONE
            ? makeError(neural::NeuralModelErrorCode::ALLOCATION, "neural demosaic failed")
            : firstError;
    }
    return result;
}

} // namespace rtengine
