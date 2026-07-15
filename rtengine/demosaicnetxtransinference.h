/*
 *  This file is part of RawTherapee.
 *
 *  Standalone native execution of the reviewed Gharbi DemosaicNet X-Trans
 *  graph. This interface operates on exact network tensors only; it does not
 *  define raw normalization, CFA orientation, tiling, or image boundaries.
 */
#pragma once

#include "demosaicnetxtransmodel.h"

#include <cstdint>
#include <memory>

namespace rtengine
{

namespace neural
{

namespace detail
{

using DemosaicNetXTransActivationObserver = void (*)(
    const char *name,
    const float *data,
    std::uint32_t channels,
    std::uint32_t width,
    std::uint32_t height,
    void *context);

class DemosaicNetXTransExecutorAccess;

} // namespace detail

class DemosaicNetXTransExecutor;

struct DemosaicNetXTransExecutorCreateResult final {
    std::unique_ptr<DemosaicNetXTransExecutor> executor;
    NeuralModelError error;

    explicit operator bool() const
    {
        return static_cast<bool>(executor);
    }
};

class DemosaicNetXTransExecutor final
{
public:
    static const std::uint32_t CHANNELS = 3;
    static const std::uint32_t OUTPUT_SHRINK = 24;
    static const std::uint64_t MAX_INPUT_PIXELS = 262144;

    ~DemosaicNetXTransExecutor();

    DemosaicNetXTransExecutor(const DemosaicNetXTransExecutor &) = delete;
    DemosaicNetXTransExecutor &operator=(const DemosaicNetXTransExecutor &) = delete;

    // Input and output are contiguous planar CHW float32 arrays. The executor
    // owns reusable scratch storage and consequently must not be used by two
    // threads concurrently. Independent executors may share the same model.
    NeuralModelError run(
        const float *input,
        std::uint64_t inputElements,
        std::uint32_t width,
        std::uint32_t height,
        float *output,
        std::uint64_t outputElements);

    std::uint64_t workspaceBytes() const;

private:
    class Implementation;

    explicit DemosaicNetXTransExecutor(std::unique_ptr<Implementation> implementation);

    NeuralModelError runInternal(
        const float *input,
        std::uint64_t inputElements,
        std::uint32_t width,
        std::uint32_t height,
        float *output,
        std::uint64_t outputElements,
        detail::DemosaicNetXTransActivationObserver observer,
        void *observerContext);

    std::unique_ptr<Implementation> implementation_;

    friend DemosaicNetXTransExecutorCreateResult createDemosaicNetXTransExecutor(
        std::shared_ptr<const DemosaicNetXTransModel> model);
    friend class detail::DemosaicNetXTransExecutorAccess;
};

DemosaicNetXTransExecutorCreateResult createDemosaicNetXTransExecutor(
    std::shared_ptr<const DemosaicNetXTransModel> model);

} // namespace neural

} // namespace rtengine
