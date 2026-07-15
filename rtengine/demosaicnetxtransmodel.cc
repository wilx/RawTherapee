/*
 *  This file is part of RawTherapee.
 *
 *  Exact C++ binding for architecture ID 1, model revision 1. The numeric
 *  tensor IDs, shapes, layouts, and hashes mirror the reviewed Phase 2 schema;
 *  PyTorch source-key strings intentionally do not enter the runtime model.
 */

#include "demosaicnetxtransmodel.h"

#include "rtnnreader_p.h"

#include <array>
#include <new>
#include <utility>

namespace rtengine
{

namespace neural
{

namespace
{

const std::array<detail::TensorBinding, 26> TENSOR_BINDINGS{{{1, 4, TensorLayout::OIHW, {{64, 3, 3, 3}}, 1728, "3a5634f46ffdfed5c2df0d7d9ff37672c43cdd213a97b6cd47b3e8c4ba232977"},
    {2, 1, TensorLayout::VECTOR, {{64, 0, 0, 0}}, 64, "aee3ec0800aa3fad56b23e16769f3bf85c2c8e9c62240828fba87084a0a18661"},
    {3, 4, TensorLayout::OIHW, {{64, 64, 3, 3}}, 36864, "9b59fcd95b6bf6b882336a4780b5674a7700295b959d2b895cf89404008de13a"},
    {4, 1, TensorLayout::VECTOR, {{64, 0, 0, 0}}, 64, "8b068ff7ed25d5060c7a459cb699eb8cccd4226272aa838711df9bc08ff39cb7"},
    {5, 4, TensorLayout::OIHW, {{64, 64, 3, 3}}, 36864, "38f7d56162b27788c17c6135731839c5885e80c873a8eea5fd976f4e09239359"},
    {6, 1, TensorLayout::VECTOR, {{64, 0, 0, 0}}, 64, "55ad1b0055983e8c2b48c249b9d25b7784deef5b28bb266ae900601d19dadb4b"},
    {7, 4, TensorLayout::OIHW, {{64, 64, 3, 3}}, 36864, "a1c04998a3e0f8e4988b038422cc30a035d7541e687a97aba1eb35d9f6d156d7"},
    {8, 1, TensorLayout::VECTOR, {{64, 0, 0, 0}}, 64, "8a1cd3ce53b7751c9af80e9aef41ff30604a502bc02909d2e7c35f9cfcd16198"},
    {9, 4, TensorLayout::OIHW, {{64, 64, 3, 3}}, 36864, "31e8d1eb3c443291f7e60b09a25bff1fe5c46fc6da2cea703139300be822a79d"},
    {10, 1, TensorLayout::VECTOR, {{64, 0, 0, 0}}, 64, "ff6daf5ec7b4137ed899b3604d3c5c70d20da98f291dc82807a1e132f36a2767"},
    {11, 4, TensorLayout::OIHW, {{64, 64, 3, 3}}, 36864, "cc0491d220a8a3cb330be5df0048df2713653944dd407e4af84b0f0add548bce"},
    {12, 1, TensorLayout::VECTOR, {{64, 0, 0, 0}}, 64, "692de686c1e75e66c6eb4133cf7f50ee7aecfd88399f4904a031e49a0511f1fc"},
    {13, 4, TensorLayout::OIHW, {{64, 64, 3, 3}}, 36864, "d9c05548a1db7c044c00deb11fcfd645667bdd50cc8f38f77dca3adad4f09922"},
    {14, 1, TensorLayout::VECTOR, {{64, 0, 0, 0}}, 64, "7383483b46b242e08415f9d7d59c080ab5875b369260f3527eb76feacbbe5e7b"},
    {15, 4, TensorLayout::OIHW, {{64, 64, 3, 3}}, 36864, "34d210c7ab146f96eec068e60f7163a39a44212bb2e6094fe152cc25ea2ef0a7"},
    {16, 1, TensorLayout::VECTOR, {{64, 0, 0, 0}}, 64, "d216797211439bebf2b93d8c5ed816700583da2b32ad3dc0ce9f1b4cdaeb7284"},
    {17, 4, TensorLayout::OIHW, {{64, 64, 3, 3}}, 36864, "2057e4538ceef9a55d10ccf1282b076d16e60e5ca3a04f41a6ed4d344b27d3fe"},
    {18, 1, TensorLayout::VECTOR, {{64, 0, 0, 0}}, 64, "92414fe7076834257db027a8ebedd4621bdc1e311420a3857e4be4373d241dbb"},
    {19, 4, TensorLayout::OIHW, {{64, 64, 3, 3}}, 36864, "3ac5984898de94656bb11059ed6172e8466efde2c714687b052e1471ff7cd6f3"},
    {20, 1, TensorLayout::VECTOR, {{64, 0, 0, 0}}, 64, "d2fb001b13b1a569bbdd866b2dd7da2c5ee7bb464982cfed1df6d1224db6457e"},
    {21, 4, TensorLayout::OIHW, {{64, 64, 3, 3}}, 36864, "07fcfc4d6142fcfd99485773fdd71bcb658995e12f568f8c8cbf3af36bcf7a6c"},
    {22, 1, TensorLayout::VECTOR, {{64, 0, 0, 0}}, 64, "3b2e6fb9781a72a1b883ac9ce166f68eb4b9aaf34ee2e26cb539bedd06751942"},
    {23, 4, TensorLayout::OIHW, {{64, 67, 3, 3}}, 38592, "5a4a7ff15de350a4e3221a7b539307d148647d82e2ca15b8c61c69cd9c3c89de"},
    {24, 1, TensorLayout::VECTOR, {{64, 0, 0, 0}}, 64, "52e63f1f963b457b96cf09cc025935e9500f4fbd419ea0929bb13a966bdff480"},
    {25, 4, TensorLayout::OIHW, {{3, 64, 1, 1}}, 192, "cfb9febc7cc60c444aa946cdaa78d36c32f015a3a75855558970a890642a7717"},
    {26, 1, TensorLayout::VECTOR, {{3, 0, 0, 0}}, 3, "30e7b89987e4ed6f321a9b22dcff3a2506236587c42e909d8c33953949313b49"}}};

const detail::ModelBinding MODEL_BINDING{
    1,
    1,
    "0ec34ea3d563f1357097181cb0dd90586a65e8d4c85b644b45c6fd3f81bcc151",
    "3759a13296fecebae83a36a8f0c3890d8a2f7d33e5f8149ed9b76a70f8285bc7",
    "b4dd6ea4ba535e7f4aea249a2d589a80ca8584f60a605a5bce468c989529ccc2",
    409923,
    1639692,
    TENSOR_BINDINGS.data(),
    TENSOR_BINDINGS.size()};

} // namespace

class DemosaicNetXTransModel::Implementation final
{
public:
    explicit Implementation(std::unique_ptr<detail::ParsedRtnn> parsed) :
        parsed(std::move(parsed))
    {
    }

    std::unique_ptr<detail::ParsedRtnn> parsed;
};

DemosaicNetXTransModel::DemosaicNetXTransModel(std::unique_ptr<Implementation> implementation) :
    implementation_(std::move(implementation))
{
}

DemosaicNetXTransModel::~DemosaicNetXTransModel() = default;

std::uint16_t DemosaicNetXTransModel::formatMajor() const
{
    return implementation_->parsed->formatMajor;
}

std::uint16_t DemosaicNetXTransModel::formatMinor() const
{
    return implementation_->parsed->formatMinor;
}

std::uint32_t DemosaicNetXTransModel::architectureId() const
{
    return implementation_->parsed->architectureId;
}

std::uint32_t DemosaicNetXTransModel::modelRevision() const
{
    return implementation_->parsed->modelRevision;
}

std::uint64_t DemosaicNetXTransModel::fileSize() const
{
    return implementation_->parsed->fileSize;
}

std::uint64_t DemosaicNetXTransModel::payloadRegionBytes() const
{
    return implementation_->parsed->payloadRegionBytes;
}

std::uint64_t DemosaicNetXTransModel::tensorPayloadBytes() const
{
    return implementation_->parsed->tensorPayloadBytes;
}

std::uint64_t DemosaicNetXTransModel::parameterCount() const
{
    return implementation_->parsed->parameterCount;
}

const Sha256Digest &DemosaicNetXTransModel::semanticSchemaSha256() const
{
    return implementation_->parsed->semanticSchemaSha256;
}

const Sha256Digest &DemosaicNetXTransModel::checkpointSha256() const
{
    return implementation_->parsed->checkpointSha256;
}

const Sha256Digest &DemosaicNetXTransModel::artifactSha256() const
{
    return implementation_->parsed->artifactSha256;
}

const Sha256Digest &DemosaicNetXTransModel::payloadSha256() const
{
    return implementation_->parsed->payloadSha256;
}

const std::vector<NeuralTensorView> &DemosaicNetXTransModel::tensors() const
{
    return implementation_->parsed->tensors;
}

const NeuralTensorView *DemosaicNetXTransModel::tensor(DemosaicNetXTransTensorId id) const
{
    const std::uint32_t numericId = static_cast<std::uint32_t>(id);
    if (numericId == 0 || numericId > implementation_->parsed->tensors.size()) {
        return nullptr;
    }

    const NeuralTensorView &result = implementation_->parsed->tensors[numericId - 1];
    return result.id == numericId ? &result : nullptr;
}

DemosaicNetXTransLoadResult loadDemosaicNetXTransModel(const Glib::ustring &path)
{
    detail::InternalLoadResult loaded = detail::loadRtnnFile(path, MODEL_BINDING);
    DemosaicNetXTransLoadResult result;
    if (!loaded) {
        result.error = std::move(loaded.error);
        return result;
    }

    try {
        std::unique_ptr<DemosaicNetXTransModel::Implementation> implementation(
            new DemosaicNetXTransModel::Implementation(std::move(loaded.model)));
        result.model.reset(new DemosaicNetXTransModel(std::move(implementation)));
        return result;
    } catch (const std::bad_alloc &) {
        result.error = NeuralModelError(
            NeuralModelErrorCode::ALLOCATION,
            "cannot allocate reviewed DemosaicNet X-Trans model");
        return result;
    }
}

std::shared_ptr<const DemosaicNetXTransModel> detail::DemosaicNetXTransModelAccess::create(
    std::unique_ptr<detail::ParsedRtnn> parsed)
{
    std::unique_ptr<DemosaicNetXTransModel::Implementation> implementation(
        new DemosaicNetXTransModel::Implementation(std::move(parsed)));
    return std::shared_ptr<const DemosaicNetXTransModel>(
        new DemosaicNetXTransModel(std::move(implementation)));
}

} // namespace neural

} // namespace rtengine
