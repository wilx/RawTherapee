/*
 * Private diagnostic access for native inference tests. Production callers
 * intentionally receive no intermediate-activation API.
 */
#pragma once

#include "demosaicnetxtransinference.h"

namespace rtengine
{

namespace neural
{

namespace detail
{

class DemosaicNetXTransExecutorAccess final
{
public:
    static NeuralModelError runWithObserver(
        DemosaicNetXTransExecutor &executor,
        const float *input,
        std::uint64_t inputElements,
        std::uint32_t width,
        std::uint32_t height,
        float *output,
        std::uint64_t outputElements,
        DemosaicNetXTransActivationObserver observer,
        void *observerContext)
    {
        return executor.runInternal(
            input,
            inputElements,
            width,
            height,
            output,
            outputElements,
            observer,
            observerContext);
    }
};

} // namespace detail

} // namespace neural

} // namespace rtengine
