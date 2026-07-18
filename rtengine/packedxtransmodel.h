/* Developer-only binding for the reviewed PackedXTransNet ONNX conversion. */
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

constexpr std::size_t PACKED_XTRANS_INPUT_FLOATS = 1u * 288u * 288u;
constexpr std::size_t PACKED_XTRANS_OUTPUT_FLOATS = 3u * 288u * 288u;
constexpr const char *PACKED_XTRANS_ONNX_SHA256 = "ad000f496fe9b4a8493bc891dedc3a1e379aec86c93b2fb53f8b8a66a2888e3c";

class PackedXTransRunner
{
public:
    virtual ~PackedXTransRunner() = default;
    virtual NeuralModelError run(const float *, std::size_t, float *, std::size_t) = 0;
    virtual const std::string &artifactSha256() const = 0;
    virtual const std::string &runtimeVersion() const = 0;
    virtual const std::string &provider() const = 0;
    virtual const std::string &precision() const = 0;
    virtual const std::string &compileSource() const = 0;
    virtual std::uint64_t compilationMicroseconds() const = 0;
    virtual std::uint64_t lastInferenceMicroseconds() const = 0;
    virtual std::uint64_t workingBufferBytes() const = 0;
};

struct PackedXTransLoadResult final {
    std::shared_ptr<PackedXTransRunner> runner;
    NeuralModelError error;
    explicit operator bool() const { return runner && !error; }
};

PackedXTransLoadResult loadCachedPackedXTransRunner(const Glib::ustring &path);

} // namespace neural
} // namespace rtengine
