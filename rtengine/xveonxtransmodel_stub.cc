#include "xveonxtransmodel.h"

namespace rtengine
{

namespace neural
{

XVeonXTransLoadResult loadCachedXVeonXTransRunner(const Glib::ustring &)
{
    return {
        nullptr,
        NeuralModelError(
            NeuralModelErrorCode::RUNTIME,
            "RawTherapee was built without an X-veon neural backend")
    };
}

} // namespace neural

} // namespace rtengine
