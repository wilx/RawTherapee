/* Development-only canonical inspection output for the reviewed RTNN model. */
#pragma once

#include <string>

namespace rtengine
{
namespace neural
{
class DemosaicNetXTransModel;
}
} // namespace rtengine

namespace rtnn_test
{

std::string canonicalInspectionJson(
    const rtengine::neural::DemosaicNetXTransModel &model);

} // namespace rtnn_test
