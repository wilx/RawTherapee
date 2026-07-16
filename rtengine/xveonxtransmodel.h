/*
 * Developer-only binding for the pinned X-veon X-Trans ONNX model.
 *
 * The model is intentionally external: the pinned upstream revision and
 * weights have no explicit license.  This interface exposes only the reviewed
 * fixed-shape inference contract and is backed by ONNX Runtime only when the
 * optional build switch is enabled.
 */
#pragma once

#include "neuralmodel.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include <glibmm/ustring.h>

namespace rtengine
{

namespace neural
{

constexpr std::size_t XVEON_INPUT_FLOATS = 4u * 288u * 288u;
constexpr std::size_t XVEON_OUTPUT_FLOATS = 3u * 288u * 288u;

class XVeonXTransRunner
{
public:
    virtual ~XVeonXTransRunner() = default;

    virtual NeuralModelError run(
        const float *input,
        std::size_t inputCount,
        float *output,
        std::size_t outputCount) = 0;

    virtual const std::string &artifactSha256() const = 0;
    virtual const std::string &runtimeVersion() const = 0;
    virtual const std::string &provider() const = 0;
    virtual std::uint64_t workingBufferBytes() const = 0;
};

struct XVeonXTransLoadResult final {
    std::shared_ptr<XVeonXTransRunner> runner;
    NeuralModelError error;

    explicit operator bool() const
    {
        return runner && !error;
    }
};

// Successful sessions are retained and shared by canonical model path.
// Failures are not cached, allowing an external development model to appear
// or be replaced without restarting RawTherapee.
XVeonXTransLoadResult loadCachedXVeonXTransRunner(const Glib::ustring &path);

} // namespace neural

} // namespace rtengine
