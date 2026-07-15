/*
 *  This file is part of RawTherapee.
 *
 *  Portable fixed-graph inference for Gharbi et al.'s X-Trans DemosaicNet.
 *  OIHW weights are consumed directly. The innermost output-column loop is
 *  intentionally suitable for compiler vectorization, while every output
 *  sample retains a fixed input-channel and kernel-tap accumulation order.
 */

#include "demosaicnetxtransinference.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <new>
#include <utility>
#include <vector>

namespace rtengine
{

namespace neural
{

namespace
{

constexpr std::uint32_t FEATURE_CHANNELS = 64;
constexpr std::uint32_t MAIN_LAYERS = 11;
constexpr std::size_t ALIGNMENT = 64;
constexpr std::size_t FLOATS_PER_ALIGNMENT = ALIGNMENT / sizeof(float);

const std::array<const char *, MAIN_LAYERS> MAIN_ACTIVATION_NAMES{{
    "main_processor.relu1",
    "main_processor.relu2",
    "main_processor.relu3",
    "main_processor.relu4",
    "main_processor.relu5",
    "main_processor.relu6",
    "main_processor.relu7",
    "main_processor.relu8",
    "main_processor.relu9",
    "main_processor.relu10",
    "main_processor.relu11",
}};

bool checkedMultiply(std::uint64_t left, std::uint64_t right, std::uint64_t &result)
{
    if (left && right > std::numeric_limits<std::uint64_t>::max() / left) {
        return false;
    }
    result = left * right;
    return true;
}

bool finiteBuffer(const float *data, std::uint64_t count)
{
    for (std::uint64_t index = 0; index < count; ++index) {
        if (!std::isfinite(data[index])) {
            return false;
        }
    }
    return true;
}

bool activateReluAndValidate(float *data, std::uint64_t count)
{
    for (std::uint64_t index = 0; index < count; ++index) {
        const float value = data[index];
        if (!std::isfinite(value)) {
            return false;
        }
        data[index] = value > 0.f ? value : 0.f;
    }
    return true;
}

void initializePlanes(
    float *destination,
    std::uint32_t channels,
    std::uint64_t planeElements,
    const float *bias)
{
    for (std::uint32_t channel = 0; channel < channels; ++channel) {
        std::fill(
            destination + channel * planeElements,
            destination + (channel + 1) * planeElements,
            bias[channel]);
    }
}

void accumulate3x3(
    const float *source,
    std::uint32_t sourceChannels,
    std::uint32_t sourceWidth,
    std::uint64_t sourcePlaneElements,
    float *destination,
    std::uint32_t destinationChannels,
    std::uint32_t destinationWidth,
    std::uint32_t destinationHeight,
    const float *weights,
    std::uint32_t weightInputChannels,
    std::uint32_t weightChannelOffset)
{
    const std::uint64_t destinationPlaneElements =
        static_cast<std::uint64_t>(destinationWidth) * destinationHeight;

    for (std::uint32_t outputChannel = 0; outputChannel < destinationChannels; ++outputChannel) {
        float *destinationPlane = destination + outputChannel * destinationPlaneElements;
        for (std::uint32_t inputChannel = 0; inputChannel < sourceChannels; ++inputChannel) {
            const float *sourcePlane = source + inputChannel * sourcePlaneElements;
            const std::uint64_t weightBase =
                (static_cast<std::uint64_t>(outputChannel) * weightInputChannels +
                 weightChannelOffset + inputChannel) * 9;
            for (std::uint32_t kernelRow = 0; kernelRow < 3; ++kernelRow) {
                for (std::uint32_t kernelColumn = 0; kernelColumn < 3; ++kernelColumn) {
                    const float weight = weights[weightBase + kernelRow * 3 + kernelColumn];
                    for (std::uint32_t row = 0; row < destinationHeight; ++row) {
                        const float *sourceRow = sourcePlane +
                            static_cast<std::uint64_t>(row + kernelRow) * sourceWidth + kernelColumn;
                        float *destinationRow = destinationPlane +
                            static_cast<std::uint64_t>(row) * destinationWidth;
                        for (std::uint32_t column = 0; column < destinationWidth; ++column) {
                            destinationRow[column] += sourceRow[column] * weight;
                        }
                    }
                }
            }
        }
    }
}

bool convolution3x3Relu(
    const float *source,
    std::uint32_t sourceChannels,
    std::uint32_t sourceWidth,
    std::uint32_t sourceHeight,
    float *destination,
    const float *weights,
    const float *bias)
{
    const std::uint32_t destinationWidth = sourceWidth - 2;
    const std::uint32_t destinationHeight = sourceHeight - 2;
    const std::uint64_t destinationPlaneElements =
        static_cast<std::uint64_t>(destinationWidth) * destinationHeight;
    initializePlanes(destination, FEATURE_CHANNELS, destinationPlaneElements, bias);
    accumulate3x3(
        source,
        sourceChannels,
        sourceWidth,
        static_cast<std::uint64_t>(sourceWidth) * sourceHeight,
        destination,
        FEATURE_CHANNELS,
        destinationWidth,
        destinationHeight,
        weights,
        sourceChannels,
        0);
    return activateReluAndValidate(destination, FEATURE_CHANNELS * destinationPlaneElements);
}

bool postConvolutionRelu(
    const float *original,
    std::uint32_t originalWidth,
    std::uint32_t originalHeight,
    const float *features,
    float *destination,
    const float *weights,
    const float *bias)
{
    const std::uint32_t featureWidth = originalWidth - 22;
    const std::uint32_t featureHeight = originalHeight - 22;
    const std::uint32_t destinationWidth = originalWidth - 24;
    const std::uint32_t destinationHeight = originalHeight - 24;
    const std::uint64_t destinationPlaneElements =
        static_cast<std::uint64_t>(destinationWidth) * destinationHeight;
    initializePlanes(destination, FEATURE_CHANNELS, destinationPlaneElements, bias);

    // The centered crop is a view into the original planes. Accumulating its
    // three channels before the feature channels preserves concatenation order
    // without materializing a 67-channel activation.
    const float *cropped = original + static_cast<std::uint64_t>(11) * originalWidth + 11;
    accumulate3x3(
        cropped,
        3,
        originalWidth,
        static_cast<std::uint64_t>(originalWidth) * originalHeight,
        destination,
        FEATURE_CHANNELS,
        destinationWidth,
        destinationHeight,
        weights,
        67,
        0);
    accumulate3x3(
        features,
        FEATURE_CHANNELS,
        featureWidth,
        static_cast<std::uint64_t>(featureWidth) * featureHeight,
        destination,
        FEATURE_CHANNELS,
        destinationWidth,
        destinationHeight,
        weights,
        67,
        3);
    return activateReluAndValidate(destination, FEATURE_CHANNELS * destinationPlaneElements);
}

bool outputConvolution(
    const float *source,
    std::uint32_t width,
    std::uint32_t height,
    float *destination,
    const float *weights,
    const float *bias)
{
    const std::uint64_t planeElements = static_cast<std::uint64_t>(width) * height;
    initializePlanes(destination, 3, planeElements, bias);
    for (std::uint32_t outputChannel = 0; outputChannel < 3; ++outputChannel) {
        float *destinationPlane = destination + outputChannel * planeElements;
        for (std::uint32_t inputChannel = 0; inputChannel < FEATURE_CHANNELS; ++inputChannel) {
            const float weight = weights[outputChannel * FEATURE_CHANNELS + inputChannel];
            const float *sourcePlane = source + inputChannel * planeElements;
            for (std::uint64_t index = 0; index < planeElements; ++index) {
                destinationPlane[index] += sourcePlane[index] * weight;
            }
        }
    }
    return finiteBuffer(destination, 3 * planeElements);
}

const NeuralTensorView *requiredTensor(
    const DemosaicNetXTransModel &model,
    DemosaicNetXTransTensorId id)
{
    const NeuralTensorView *tensor = model.tensor(id);
    return tensor && tensor->data ? tensor : nullptr;
}

bool tensorMatches(
    const DemosaicNetXTransModel &model,
    DemosaicNetXTransTensorId id,
    std::uint16_t rank,
    TensorLayout layout,
    const std::array<std::uint32_t, 4> &dimensions,
    std::uint64_t count)
{
    const NeuralTensorView *tensor = model.tensor(id);
    return tensor && tensor->data && tensor->rank == rank && tensor->layout == layout &&
           tensor->dimensions == dimensions && tensor->elementCount == count &&
           tensor->byteLength == count * sizeof(float);
}

bool reviewedModelContract(const DemosaicNetXTransModel &model)
{
    if (model.architectureId() != 1 || model.modelRevision() != 1 ||
        model.tensors().size() != 26 || model.parameterCount() != 409923) {
        return false;
    }
    const std::array<std::uint32_t, 4> biasDimensions{{64, 0, 0, 0}};
    for (std::uint32_t layer = 1; layer <= MAIN_LAYERS; ++layer) {
        const std::uint32_t inputChannels = layer == 1 ? 3 : FEATURE_CHANNELS;
        if (!tensorMatches(
                model,
                static_cast<DemosaicNetXTransTensorId>(layer * 2 - 1),
                4,
                TensorLayout::OIHW,
                {{FEATURE_CHANNELS, inputChannels, 3, 3}},
                static_cast<std::uint64_t>(FEATURE_CHANNELS) * inputChannels * 9) ||
            !tensorMatches(
                model,
                static_cast<DemosaicNetXTransTensorId>(layer * 2),
                1,
                TensorLayout::VECTOR,
                biasDimensions,
                FEATURE_CHANNELS)) {
            return false;
        }
    }
    return tensorMatches(
               model,
               DemosaicNetXTransTensorId::POST_CONV_WEIGHT,
               4,
               TensorLayout::OIHW,
               {{64, 67, 3, 3}},
               64 * 67 * 9) &&
           tensorMatches(
               model,
               DemosaicNetXTransTensorId::POST_CONV_BIAS,
               1,
               TensorLayout::VECTOR,
               biasDimensions,
               64) &&
           tensorMatches(
               model,
               DemosaicNetXTransTensorId::OUTPUT_WEIGHT,
               4,
               TensorLayout::OIHW,
               {{3, 64, 1, 1}},
               192) &&
           tensorMatches(
               model,
               DemosaicNetXTransTensorId::OUTPUT_BIAS,
               1,
               TensorLayout::VECTOR,
               {{3, 0, 0, 0}},
               3);
}

} // namespace

class DemosaicNetXTransExecutor::Implementation final
{
public:
    explicit Implementation(std::shared_ptr<const DemosaicNetXTransModel> model) :
        model(std::move(model))
    {
    }

    bool reserve(std::uint64_t elementsPerBuffer)
    {
        if (elementsPerBuffer <= bufferCapacity) {
            return true;
        }
        std::uint64_t totalElements = 0;
        if (!checkedMultiply(elementsPerBuffer, 2, totalElements) ||
            totalElements > std::numeric_limits<std::size_t>::max() - FLOATS_PER_ALIGNMENT) {
            return false;
        }
        std::unique_ptr<float[]> replacement(
            new float[static_cast<std::size_t>(totalElements) + FLOATS_PER_ALIGNMENT]);
        const std::uintptr_t address = reinterpret_cast<std::uintptr_t>(replacement.get());
        const std::uintptr_t alignedAddress = (address + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
        allocation = std::move(replacement);
        aligned = reinterpret_cast<float *>(alignedAddress);
        bufferCapacity = elementsPerBuffer;
        return true;
    }

    float *firstBuffer()
    {
        return aligned;
    }

    float *secondBuffer()
    {
        return aligned + bufferCapacity;
    }

    std::shared_ptr<const DemosaicNetXTransModel> model;
    std::unique_ptr<float[]> allocation;
    float *aligned = nullptr;
    std::uint64_t bufferCapacity = 0;
};

DemosaicNetXTransExecutor::DemosaicNetXTransExecutor(std::unique_ptr<Implementation> implementation) :
    implementation_(std::move(implementation))
{
}

DemosaicNetXTransExecutor::~DemosaicNetXTransExecutor() = default;

std::uint64_t DemosaicNetXTransExecutor::workspaceBytes() const
{
    return implementation_->bufferCapacity * 2 * sizeof(float);
}

NeuralModelError DemosaicNetXTransExecutor::run(
    const float *input,
    std::uint64_t inputElements,
    std::uint32_t width,
    std::uint32_t height,
    float *output,
    std::uint64_t outputElements)
{
    return runInternal(
        input,
        inputElements,
        width,
        height,
        output,
        outputElements,
        nullptr,
        nullptr);
}

NeuralModelError DemosaicNetXTransExecutor::runInternal(
    const float *input,
    std::uint64_t inputElements,
    std::uint32_t width,
    std::uint32_t height,
    float *output,
    std::uint64_t outputElements,
    detail::DemosaicNetXTransActivationObserver observer,
    void *observerContext)
{
    if (!input || !output) {
        return NeuralModelError(NeuralModelErrorCode::RANGE, "neural input and output must be non-null");
    }
    if (width <= OUTPUT_SHRINK || height <= OUTPUT_SHRINK) {
        return NeuralModelError(NeuralModelErrorCode::SIZE, "neural input must be at least 25 by 25");
    }

    std::uint64_t inputPixels = 0;
    if (!checkedMultiply(width, height, inputPixels)) {
        return NeuralModelError(NeuralModelErrorCode::RANGE, "neural input dimensions overflow");
    }
    if (inputPixels > MAX_INPUT_PIXELS) {
        return NeuralModelError(NeuralModelErrorCode::LIMIT, "neural input exceeds the standalone executor limit");
    }
    std::uint64_t requiredInputElements = 0;
    if (!checkedMultiply(inputPixels, CHANNELS, requiredInputElements) ||
        inputElements < requiredInputElements) {
        return NeuralModelError(NeuralModelErrorCode::SIZE, "neural input buffer is too small");
    }

    const std::uint32_t outputWidth = width - OUTPUT_SHRINK;
    const std::uint32_t outputHeight = height - OUTPUT_SHRINK;
    std::uint64_t outputPixels = 0;
    std::uint64_t requiredOutputElements = 0;
    if (!checkedMultiply(outputWidth, outputHeight, outputPixels) ||
        !checkedMultiply(outputPixels, CHANNELS, requiredOutputElements) ||
        outputElements < requiredOutputElements) {
        return NeuralModelError(NeuralModelErrorCode::SIZE, "neural output buffer is too small");
    }
    if (!finiteBuffer(input, requiredInputElements)) {
        return NeuralModelError(NeuralModelErrorCode::NONFINITE, "neural input contains a non-finite value");
    }

    const std::uint64_t firstFeaturePixels =
        static_cast<std::uint64_t>(width - 2) * (height - 2);
    std::uint64_t bufferElements = 0;
    if (!checkedMultiply(FEATURE_CHANNELS, firstFeaturePixels, bufferElements)) {
        return NeuralModelError(NeuralModelErrorCode::RANGE, "neural workspace size overflows");
    }
    try {
        if (!implementation_->reserve(bufferElements)) {
            return NeuralModelError(NeuralModelErrorCode::RANGE, "neural workspace size overflows");
        }
    } catch (const std::bad_alloc &) {
        return NeuralModelError(NeuralModelErrorCode::ALLOCATION, "cannot allocate neural inference workspace");
    }

    float *first = implementation_->firstBuffer();
    float *second = implementation_->secondBuffer();
    const DemosaicNetXTransModel &model = *implementation_->model;
    const float *source = input;
    std::uint32_t sourceChannels = 3;
    std::uint32_t sourceWidth = width;
    std::uint32_t sourceHeight = height;

    for (std::uint32_t layer = 1; layer <= MAIN_LAYERS; ++layer) {
        const auto weightId = static_cast<DemosaicNetXTransTensorId>(layer * 2 - 1);
        const auto biasId = static_cast<DemosaicNetXTransTensorId>(layer * 2);
        const NeuralTensorView *weights = requiredTensor(model, weightId);
        const NeuralTensorView *bias = requiredTensor(model, biasId);
        if (!weights || !bias) {
            return NeuralModelError(NeuralModelErrorCode::SCHEMA, "neural model lacks a main convolution tensor");
        }
        float *destination = layer & 1 ? first : second;
        if (!convolution3x3Relu(
                source,
                sourceChannels,
                sourceWidth,
                sourceHeight,
                destination,
                weights->data,
                bias->data)) {
            return NeuralModelError(NeuralModelErrorCode::NONFINITE, "main convolution produced a non-finite value");
        }
        source = destination;
        sourceChannels = FEATURE_CHANNELS;
        sourceWidth -= 2;
        sourceHeight -= 2;
        if (observer) {
            observer(
                MAIN_ACTIVATION_NAMES[layer - 1],
                source,
                FEATURE_CHANNELS,
                sourceWidth,
                sourceHeight,
                observerContext);
        }
    }

    const NeuralTensorView *postWeights = requiredTensor(model, DemosaicNetXTransTensorId::POST_CONV_WEIGHT);
    const NeuralTensorView *postBias = requiredTensor(model, DemosaicNetXTransTensorId::POST_CONV_BIAS);
    if (!postWeights || !postBias) {
        return NeuralModelError(NeuralModelErrorCode::SCHEMA, "neural model lacks post-convolution tensors");
    }
    if (observer) {
        try {
            const std::uint64_t featurePlaneElements =
                static_cast<std::uint64_t>(sourceWidth) * sourceHeight;
            std::vector<float> concatenated(static_cast<std::size_t>(67 * featurePlaneElements));
            for (std::uint32_t channel = 0; channel < 3; ++channel) {
                const float *originalPlane = input +
                    static_cast<std::uint64_t>(channel) * width * height;
                float *croppedPlane = concatenated.data() + channel * featurePlaneElements;
                for (std::uint32_t row = 0; row < sourceHeight; ++row) {
                    std::copy(
                        originalPlane + static_cast<std::uint64_t>(row + 11) * width + 11,
                        originalPlane + static_cast<std::uint64_t>(row + 11) * width + 11 + sourceWidth,
                        croppedPlane + static_cast<std::uint64_t>(row) * sourceWidth);
                }
            }
            std::copy(
                source,
                source + FEATURE_CHANNELS * featurePlaneElements,
                concatenated.data() + 3 * featurePlaneElements);
            observer(
                "fullres_processor.input_concat",
                concatenated.data(),
                67,
                sourceWidth,
                sourceHeight,
                observerContext);
        } catch (const std::bad_alloc &) {
            return NeuralModelError(NeuralModelErrorCode::ALLOCATION, "cannot allocate diagnostic concatenation");
        }
    }
    if (!postConvolutionRelu(
            input,
            width,
            height,
            source,
            second,
            postWeights->data,
            postBias->data)) {
        return NeuralModelError(NeuralModelErrorCode::NONFINITE, "post-convolution produced a non-finite value");
    }
    if (observer) {
        observer(
            "fullres_processor.post_relu",
            second,
            FEATURE_CHANNELS,
            outputWidth,
            outputHeight,
            observerContext);
    }

    const NeuralTensorView *outputWeights = requiredTensor(model, DemosaicNetXTransTensorId::OUTPUT_WEIGHT);
    const NeuralTensorView *outputBias = requiredTensor(model, DemosaicNetXTransTensorId::OUTPUT_BIAS);
    if (!outputWeights || !outputBias) {
        return NeuralModelError(NeuralModelErrorCode::SCHEMA, "neural model lacks output tensors");
    }
    if (!outputConvolution(
            second,
            outputWidth,
            outputHeight,
            first,
            outputWeights->data,
            outputBias->data)) {
        return NeuralModelError(NeuralModelErrorCode::NONFINITE, "output convolution produced a non-finite value");
    }

    std::memmove(output, first, static_cast<std::size_t>(requiredOutputElements * sizeof(float)));
    return NeuralModelError();
}

DemosaicNetXTransExecutorCreateResult createDemosaicNetXTransExecutor(
    std::shared_ptr<const DemosaicNetXTransModel> model)
{
    DemosaicNetXTransExecutorCreateResult result;
    if (!model || !reviewedModelContract(*model)) {
        result.error = NeuralModelError(
            NeuralModelErrorCode::SCHEMA,
            "executor requires the reviewed DemosaicNet X-Trans model");
        return result;
    }
    try {
        std::unique_ptr<DemosaicNetXTransExecutor::Implementation> implementation(
            new DemosaicNetXTransExecutor::Implementation(std::move(model)));
        result.executor.reset(new DemosaicNetXTransExecutor(std::move(implementation)));
        return result;
    } catch (const std::bad_alloc &) {
        result.error = NeuralModelError(
            NeuralModelErrorCode::ALLOCATION,
            "cannot allocate DemosaicNet X-Trans executor");
        return result;
    }
}

} // namespace neural

} // namespace rtengine
