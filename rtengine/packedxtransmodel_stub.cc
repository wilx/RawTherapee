#include "packedxtransmodel.h"

namespace rtengine
{
namespace neural
{
PackedXTransLoadResult loadCachedPackedXTransRunner(const Glib::ustring &)
{
    return {nullptr, NeuralModelError(NeuralModelErrorCode::RUNTIME,
        "RawTherapee was built without ONNX Runtime or MIGraphX")};
}
} // namespace neural
} // namespace rtengine
