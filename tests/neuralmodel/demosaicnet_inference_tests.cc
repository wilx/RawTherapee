#include "demosaicnet_inference_tests.h"

#include "rtengine/demosaicnetxtransinference.h"
#include "rtengine/demosaicnetxtransinference_p.h"
#include "rtengine/rtnnreader_p.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <glib/gstdio.h>

namespace
{

using rtengine::neural::DemosaicNetXTransExecutor;
using rtengine::neural::DemosaicNetXTransModel;
using rtengine::neural::NeuralModelError;
using rtengine::neural::NeuralModelErrorCode;
using rtengine::neural::NeuralTensorView;
using rtengine::neural::TensorLayout;
using rtengine::neural::detail::DemosaicNetXTransModelAccess;
using rtengine::neural::detail::ParsedRtnn;

constexpr std::size_t ALIGNMENT = 64;
constexpr std::size_t TENSOR_COUNT = 26;
constexpr float ABSOLUTE_TOLERANCE = 5e-6f;
constexpr float RELATIVE_TOLERANCE = 1e-5f;

struct TensorSpec final {
    std::uint16_t rank;
    TensorLayout layout;
    std::array<std::uint32_t, 4> dimensions;
    std::uint64_t count;
};

const std::array<TensorSpec, TENSOR_COUNT> SPECS{{
    {4, TensorLayout::OIHW, {{64, 3, 3, 3}}, 1728},
    {1, TensorLayout::VECTOR, {{64, 0, 0, 0}}, 64},
    {4, TensorLayout::OIHW, {{64, 64, 3, 3}}, 36864},
    {1, TensorLayout::VECTOR, {{64, 0, 0, 0}}, 64},
    {4, TensorLayout::OIHW, {{64, 64, 3, 3}}, 36864},
    {1, TensorLayout::VECTOR, {{64, 0, 0, 0}}, 64},
    {4, TensorLayout::OIHW, {{64, 64, 3, 3}}, 36864},
    {1, TensorLayout::VECTOR, {{64, 0, 0, 0}}, 64},
    {4, TensorLayout::OIHW, {{64, 64, 3, 3}}, 36864},
    {1, TensorLayout::VECTOR, {{64, 0, 0, 0}}, 64},
    {4, TensorLayout::OIHW, {{64, 64, 3, 3}}, 36864},
    {1, TensorLayout::VECTOR, {{64, 0, 0, 0}}, 64},
    {4, TensorLayout::OIHW, {{64, 64, 3, 3}}, 36864},
    {1, TensorLayout::VECTOR, {{64, 0, 0, 0}}, 64},
    {4, TensorLayout::OIHW, {{64, 64, 3, 3}}, 36864},
    {1, TensorLayout::VECTOR, {{64, 0, 0, 0}}, 64},
    {4, TensorLayout::OIHW, {{64, 64, 3, 3}}, 36864},
    {1, TensorLayout::VECTOR, {{64, 0, 0, 0}}, 64},
    {4, TensorLayout::OIHW, {{64, 64, 3, 3}}, 36864},
    {1, TensorLayout::VECTOR, {{64, 0, 0, 0}}, 64},
    {4, TensorLayout::OIHW, {{64, 64, 3, 3}}, 36864},
    {1, TensorLayout::VECTOR, {{64, 0, 0, 0}}, 64},
    {4, TensorLayout::OIHW, {{64, 67, 3, 3}}, 38592},
    {1, TensorLayout::VECTOR, {{64, 0, 0, 0}}, 64},
    {4, TensorLayout::OIHW, {{3, 64, 1, 1}}, 192},
    {1, TensorLayout::VECTOR, {{3, 0, 0, 0}}, 3},
}};

void require(bool condition, const std::string &message)
{
    if (!condition) {
        throw std::runtime_error(message);
    }
}

struct MutableSyntheticModel final {
    std::shared_ptr<const DemosaicNetXTransModel> model;
    std::array<float *, TENSOR_COUNT> tensors{{}};
};

MutableSyntheticModel syntheticModel()
{
    std::unique_ptr<ParsedRtnn> parsed(new ParsedRtnn());
    parsed->architectureId = 1;
    parsed->modelRevision = 1;
    parsed->parameterCount = 409923;
    parsed->tensorPayloadBytes = 1639692;
    parsed->allocation.reset(new float[409923 + ALIGNMENT / sizeof(float)]);
    const std::uintptr_t address = reinterpret_cast<std::uintptr_t>(parsed->allocation.get());
    parsed->alignedData = reinterpret_cast<float *>((address + ALIGNMENT - 1) & ~(ALIGNMENT - 1));
    std::fill(parsed->alignedData, parsed->alignedData + 409923, 0.f);

    MutableSyntheticModel result;
    std::uint64_t cursor = 0;
    for (std::size_t index = 0; index < SPECS.size(); ++index) {
        const TensorSpec &spec = SPECS[index];
        NeuralTensorView view;
        view.id = static_cast<std::uint32_t>(index + 1);
        view.rank = spec.rank;
        view.layout = spec.layout;
        view.dimensions = spec.dimensions;
        view.elementCount = spec.count;
        view.byteLength = spec.count * sizeof(float);
        view.data = parsed->alignedData + cursor;
        parsed->tensors.push_back(view);
        result.tensors[index] = parsed->alignedData + cursor;
        cursor += spec.count;
    }
    require(cursor == 409923, "synthetic model parameter count differs");
    result.model = DemosaicNetXTransModelAccess::create(std::move(parsed));
    return result;
}

std::unique_ptr<DemosaicNetXTransExecutor> executorFor(
    const std::shared_ptr<const DemosaicNetXTransModel> &model)
{
    auto result = rtengine::neural::createDemosaicNetXTransExecutor(model);
    require(static_cast<bool>(result), "cannot construct synthetic inference executor");
    require(!result.error, "successful executor construction carries an error");
    return std::move(result.executor);
}

std::vector<float> run(
    DemosaicNetXTransExecutor &executor,
    const std::vector<float> &input,
    std::uint32_t width,
    std::uint32_t height)
{
    const std::uint64_t outputElements =
        static_cast<std::uint64_t>(width - 24) * (height - 24) * 3;
    std::vector<float> output(static_cast<std::size_t>(outputElements), -1234.f);
    const NeuralModelError error = executor.run(
        input.data(), input.size(), width, height, output.data(), output.size());
    require(!error, std::string("native inference failed: ") + error.message);
    return output;
}

std::uint64_t weight3x3Index(
    std::uint32_t outputChannel,
    std::uint32_t inputChannel,
    std::uint32_t inputChannels,
    std::uint32_t row,
    std::uint32_t column)
{
    return (static_cast<std::uint64_t>(outputChannel) * inputChannels + inputChannel) * 9 + row * 3 + column;
}

std::uint32_t readU32(const unsigned char *bytes)
{
    return static_cast<std::uint32_t>(bytes[0]) |
           static_cast<std::uint32_t>(bytes[1]) << 8 |
           static_cast<std::uint32_t>(bytes[2]) << 16 |
           static_cast<std::uint32_t>(bytes[3]) << 24;
}

std::vector<float> readFloatFile(const std::string &path, std::size_t expectedElements)
{
    std::unique_ptr<std::FILE, int (*)(std::FILE *)> file(g_fopen(path.c_str(), "rb"), std::fclose);
    require(static_cast<bool>(file), "cannot open golden float file: " + path);
    std::vector<unsigned char> bytes(expectedElements * sizeof(float));
    require(std::fread(bytes.data(), 1, bytes.size(), file.get()) == bytes.size(),
        "cannot read complete golden float file: " + path);
    require(std::fgetc(file.get()) == EOF, "golden float file has trailing bytes: " + path);
    std::vector<float> result(expectedElements);
    for (std::size_t index = 0; index < expectedElements; ++index) {
        const std::uint32_t bits = readU32(bytes.data() + index * 4);
        std::memcpy(&result[index], &bits, sizeof(bits));
    }
    return result;
}

struct DifferenceStats final {
    std::uint64_t count = 0;
    std::uint64_t exact = 0;
    long double sum = 0;
    long double squaredSum = 0;
    double maximum = 0;
    std::size_t maximumIndex = 0;
    std::vector<double> differences;
};

void accumulateDifferences(
    DifferenceStats &stats,
    const std::vector<float> &actual,
    const std::vector<float> &expected,
    const std::string &caseId)
{
    require(actual.size() == expected.size(), caseId + " output size differs");
    for (std::size_t index = 0; index < actual.size(); ++index) {
        require(std::isfinite(actual[index]), caseId + " output is non-finite");
        const double difference = std::abs(static_cast<double>(actual[index]) - expected[index]);
        const double tolerance = ABSOLUTE_TOLERANCE +
            RELATIVE_TOLERANCE * std::abs(static_cast<double>(expected[index]));
        if (difference > tolerance) {
            std::ostringstream message;
            message << caseId << " differs at flat index " << index
                    << ": actual=" << std::setprecision(10) << actual[index]
                    << ", expected=" << expected[index]
                    << ", absolute difference=" << difference
                    << ", tolerance=" << tolerance;
            throw std::runtime_error(message.str());
        }
        std::uint32_t actualBits = 0;
        std::uint32_t expectedBits = 0;
        std::memcpy(&actualBits, &actual[index], sizeof(actualBits));
        std::memcpy(&expectedBits, &expected[index], sizeof(expectedBits));
        stats.exact += actualBits == expectedBits;
        stats.sum += difference;
        stats.squaredSum += difference * difference;
        if (difference > stats.maximum) {
            stats.maximum = difference;
            stats.maximumIndex = index;
        }
        stats.differences.push_back(difference);
        ++stats.count;
    }
}

double percentile(std::vector<double> values, double percentage)
{
    require(!values.empty(), "cannot calculate percentile of empty differences");
    std::sort(values.begin(), values.end());
    const double position = percentage / 100.0 * (values.size() - 1);
    const std::size_t lower = static_cast<std::size_t>(position);
    const std::size_t upper = std::min(lower + 1, values.size() - 1);
    const double fraction = position - lower;
    return values[lower] * (1.0 - fraction) + values[upper] * fraction;
}

void printStats(const std::string &label, const DifferenceStats &stats)
{
    const double mean = static_cast<double>(stats.sum / stats.count);
    const double rms = std::sqrt(static_cast<double>(stats.squaredSum / stats.count));
    std::cout << std::scientific << std::setprecision(8)
              << label << ": count=" << stats.count
              << ", exact=" << 100.0 * stats.exact / stats.count << "%"
              << ", max=" << stats.maximum
              << ", mean=" << mean
              << ", rms=" << rms
              << ", p90=" << percentile(stats.differences, 90)
              << ", p99=" << percentile(stats.differences, 99) << '\n';
}

const char *reviewedPath()
{
    const char *path = std::getenv("GHARBI_XTRANS_RTNN");
    return path && *path ? path : nullptr;
}

struct GoldenCase final {
    const char *id;
    std::uint32_t width;
    std::uint32_t height;
};

const std::array<GoldenCase, 7> GOLDEN_CASES{{
    {"zero-even-32x32", 32, 32},
    {"constant-odd-31x35", 35, 31},
    {"random-37x38", 38, 37},
    {"impulse-red-36x36", 36, 36},
    {"impulse-green-36x36", 36, 36},
    {"impulse-blue-36x36", 36, 36},
    {"alternating-saturated-35x36", 36, 35},
}};

struct TraceSpec final {
    const char *name;
    std::uint32_t channels;
    std::uint32_t width;
    std::uint32_t height;
};

const std::array<TraceSpec, 13> TRACE_SPECS{{
    {"main_processor.relu1", 64, 36, 35},
    {"main_processor.relu2", 64, 34, 33},
    {"main_processor.relu3", 64, 32, 31},
    {"main_processor.relu4", 64, 30, 29},
    {"main_processor.relu5", 64, 28, 27},
    {"main_processor.relu6", 64, 26, 25},
    {"main_processor.relu7", 64, 24, 23},
    {"main_processor.relu8", 64, 22, 21},
    {"main_processor.relu9", 64, 20, 19},
    {"main_processor.relu10", 64, 18, 17},
    {"main_processor.relu11", 64, 16, 15},
    {"fullres_processor.input_concat", 67, 16, 15},
    {"fullres_processor.post_relu", 64, 14, 13},
}};

struct TraceCollector final {
    std::size_t next = 0;
    DifferenceStats aggregate;
};

void collectTrace(
    const char *name,
    const float *data,
    std::uint32_t channels,
    std::uint32_t width,
    std::uint32_t height,
    void *context)
{
    TraceCollector &collector = *static_cast<TraceCollector *>(context);
    require(collector.next < TRACE_SPECS.size(), "native trace contains an extra activation");
    const TraceSpec &spec = TRACE_SPECS[collector.next++];
    require(
        std::strcmp(name, spec.name) == 0 && channels == spec.channels &&
        width == spec.width && height == spec.height,
        "native activation identity or shape differs");
    const std::array<std::pair<std::uint32_t, std::uint32_t>, 5> positions{{
        {0, 0},
        {0, width - 1},
        {height / 2, width / 2},
        {height - 1, 0},
        {height - 1, width - 1},
    }};
    std::vector<float> actual;
    actual.reserve(static_cast<std::size_t>(channels) * positions.size());
    const std::uint64_t planeElements = static_cast<std::uint64_t>(width) * height;
    for (const auto &position : positions) {
        for (std::uint32_t channel = 0; channel < channels; ++channel) {
            actual.push_back(data[
                static_cast<std::uint64_t>(channel) * planeElements +
                static_cast<std::uint64_t>(position.first) * width + position.second]);
        }
    }
    const std::string path = std::string(RT_NEURAL_TRACE_DIR) + "/activations/" +
        name + ".samples.f32le";
    const std::vector<float> expected = readFloatFile(path, actual.size());
    DifferenceStats activation;
    accumulateDifferences(activation, actual, expected, name);
    printStats(name, activation);
    accumulateDifferences(collector.aggregate, actual, expected, name);
}

} // namespace

namespace demosaicnet_inference_test
{

int syntheticGraph()
{
    {
        MutableSyntheticModel synthetic = syntheticModel();
        synthetic.tensors[25][0] = 0.25f;
        synthetic.tensors[25][1] = -0.5f;
        synthetic.tensors[25][2] = 1.f;
        auto executor = executorFor(synthetic.model);
        std::vector<float> input(3 * 27 * 29, 0.75f);
        const std::vector<float> output = run(*executor, input, 29, 27);
        const std::size_t plane = 5 * 3;
        for (std::size_t index = 0; index < plane; ++index) {
            require(output[index] == 0.25f, "output red bias differs");
            require(output[plane + index] == -0.5f, "final layer unexpectedly applies ReLU");
            require(output[2 * plane + index] == 1.f, "output blue bias differs");
        }
    }

    {
        MutableSyntheticModel synthetic = syntheticModel();
        std::uint32_t rowOffset = 0;
        std::uint32_t columnOffset = 0;
        for (std::uint32_t layer = 1; layer <= 11; ++layer) {
            const std::uint32_t kernelRow = layer % 3;
            const std::uint32_t kernelColumn = (layer * 2) % 3;
            const std::uint32_t inputChannel = layer == 1 ? 1 : 0;
            const std::uint32_t inputChannels = layer == 1 ? 3 : 64;
            synthetic.tensors[(layer - 1) * 2][weight3x3Index(
                0, inputChannel, inputChannels, kernelRow, kernelColumn)] = 1.f;
            rowOffset += kernelRow;
            columnOffset += kernelColumn;
        }
        synthetic.tensors[22][weight3x3Index(0, 3, 67, 2, 1)] = 1.f;
        synthetic.tensors[24][0] = 1.f;
        rowOffset += 2;
        columnOffset += 1;
        auto executor = executorFor(synthetic.model);
        const std::uint32_t width = 31;
        const std::uint32_t height = 30;
        const std::uint64_t pixels = static_cast<std::uint64_t>(width) * height;
        std::vector<float> input(3 * pixels, 0.f);
        for (std::uint32_t row = 0; row < height; ++row) {
            for (std::uint32_t column = 0; column < width; ++column) {
                input[pixels + static_cast<std::uint64_t>(row) * width + column] =
                    static_cast<float>(row * width + column + 1);
            }
        }
        const std::vector<float> output = run(*executor, input, width, height);
        const std::uint32_t outputWidth = width - 24;
        const std::uint32_t outputHeight = height - 24;
        for (std::uint32_t row = 0; row < outputHeight; ++row) {
            for (std::uint32_t column = 0; column < outputWidth; ++column) {
                const float expected = input[pixels +
                    static_cast<std::uint64_t>(row + rowOffset) * width + column + columnOffset];
                require(output[static_cast<std::uint64_t>(row) * outputWidth + column] == expected,
                    "main convolution kernel order differs");
            }
        }
    }

    {
        MutableSyntheticModel synthetic = syntheticModel();
        synthetic.tensors[22][weight3x3Index(0, 2, 67, 2, 1)] = 1.f;
        synthetic.tensors[24][2 * 64] = 1.f;
        auto executor = executorFor(synthetic.model);
        const std::uint32_t width = 29;
        const std::uint32_t height = 28;
        const std::uint64_t pixels = static_cast<std::uint64_t>(width) * height;
        std::vector<float> input(3 * pixels, 0.f);
        for (std::uint32_t row = 0; row < height; ++row) {
            for (std::uint32_t column = 0; column < width; ++column) {
                input[2 * pixels + static_cast<std::uint64_t>(row) * width + column] =
                    static_cast<float>(row * width + column + 1);
            }
        }
        const std::vector<float> output = run(*executor, input, width, height);
        const std::uint64_t outputPixels = static_cast<std::uint64_t>(width - 24) * (height - 24);
        for (std::uint32_t row = 0; row < height - 24; ++row) {
            for (std::uint32_t column = 0; column < width - 24; ++column) {
                const float expected = input[2 * pixels +
                    static_cast<std::uint64_t>(row + 13) * width + column + 12];
                require(output[2 * outputPixels + static_cast<std::uint64_t>(row) * (width - 24) + column] == expected,
                    "centered sparse-input crop differs");
            }
        }
    }
    return 0;
}

int errorsAndWorkspace()
{
    auto missing = rtengine::neural::createDemosaicNetXTransExecutor(nullptr);
    require(!missing && missing.error.code == NeuralModelErrorCode::SCHEMA,
        "null model did not return SCHEMA");

    MutableSyntheticModel synthetic = syntheticModel();
    auto executor = executorFor(synthetic.model);
    std::vector<float> input(3 * 25 * 25, 0.f);
    std::vector<float> output(3, 17.f);
    require(!executor->run(input.data(), input.size(), 25, 25, output.data(), output.size()),
        "minimum input did not run");
    const std::uint64_t firstWorkspace = executor->workspaceBytes();
    require(firstWorkspace > 0 && firstWorkspace % ALIGNMENT == 0,
        "workspace byte count is not aligned");

    input.assign(3 * 35 * 31, 0.f);
    output.assign(3 * 11 * 7, 17.f);
    require(!executor->run(input.data(), input.size(), 35, 31, output.data(), output.size()),
        "odd non-square input did not run");
    require(executor->workspaceBytes() >= firstWorkspace, "workspace did not grow monotonically");
    const std::vector<float> repeated = output;
    require(!executor->run(input.data(), input.size(), 35, 31, output.data(), output.size()),
        "repeated input did not run");
    require(std::memcmp(output.data(), repeated.data(), output.size() * sizeof(float)) == 0,
        "repeated native inference is not bit-identical");

    NeuralModelError error = executor->run(nullptr, 0, 25, 25, output.data(), output.size());
    require(error.code == NeuralModelErrorCode::RANGE, "null input did not return RANGE");
    error = executor->run(input.data(), input.size(), 24, 25, output.data(), output.size());
    require(error.code == NeuralModelErrorCode::SIZE, "small dimension did not return SIZE");
    error = executor->run(input.data(), 3 * 25 * 25 - 1, 25, 25, output.data(), output.size());
    require(error.code == NeuralModelErrorCode::SIZE, "short input did not return SIZE");
    error = executor->run(input.data(), input.size(), 25, 25, output.data(), 2);
    require(error.code == NeuralModelErrorCode::SIZE, "short output did not return SIZE");
    float dummy = 0.f;
    error = executor->run(&dummy, UINT64_MAX, 513, 512, &dummy, UINT64_MAX);
    require(error.code == NeuralModelErrorCode::LIMIT, "oversized standalone input did not return LIMIT");

    input.assign(3 * 25 * 25, 0.f);
    input[7] = std::numeric_limits<float>::quiet_NaN();
    output.assign(3, 123.f);
    error = executor->run(input.data(), input.size(), 25, 25, output.data(), output.size());
    require(error.code == NeuralModelErrorCode::NONFINITE, "NaN input did not return NONFINITE");
    require(std::all_of(output.begin(), output.end(), [](float value) { return value == 123.f; }),
        "failed inference modified output");
    return 0;
}

int reviewedGolden()
{
    const char *path = reviewedPath();
    if (!path) {
        std::cout << "GHARBI_XTRANS_RTNN is not set; skipping native golden parity\n";
        return 77;
    }
    const auto loaded = rtengine::neural::loadDemosaicNetXTransModel(path);
    require(static_cast<bool>(loaded), "reviewed RTNN did not load for native inference");
    auto executor = executorFor(loaded.model);
    DifferenceStats aggregate;

    for (const GoldenCase &golden : GOLDEN_CASES) {
        const std::uint64_t inputElements = static_cast<std::uint64_t>(golden.width) * golden.height * 3;
        const std::uint64_t outputElements =
            static_cast<std::uint64_t>(golden.width - 24) * (golden.height - 24) * 3;
        const std::string base = std::string(RT_NEURAL_GOLDEN_DIR) + "/cases/" + golden.id;
        const std::vector<float> input = readFloatFile(base + ".input.f32le", inputElements);
        const std::vector<float> expected = readFloatFile(base + ".output.f32le", outputElements);
        const std::vector<float> actual = run(*executor, input, golden.width, golden.height);
        const std::vector<float> repeated = run(*executor, input, golden.width, golden.height);
        require(std::memcmp(actual.data(), repeated.data(), actual.size() * sizeof(float)) == 0,
            std::string(golden.id) + " is not bit-identical on repeat");
        DifferenceStats perCase;
        accumulateDifferences(perCase, actual, expected, golden.id);
        printStats(golden.id, perCase);
        accumulateDifferences(aggregate, actual, expected, golden.id);
    }
    printStats("aggregate", aggregate);
    return 0;
}

int reviewedTrace()
{
    const char *path = reviewedPath();
    if (!path) {
        std::cout << "GHARBI_XTRANS_RTNN is not set; skipping native activation parity\n";
        return 77;
    }
    const auto loaded = rtengine::neural::loadDemosaicNetXTransModel(path);
    require(static_cast<bool>(loaded), "reviewed RTNN did not load for activation parity");
    auto executor = executorFor(loaded.model);
    const GoldenCase &golden = GOLDEN_CASES[2];
    const std::uint64_t inputElements = static_cast<std::uint64_t>(golden.width) * golden.height * 3;
    const std::uint64_t outputElements =
        static_cast<std::uint64_t>(golden.width - 24) * (golden.height - 24) * 3;
    const std::string base = std::string(RT_NEURAL_GOLDEN_DIR) + "/cases/" + golden.id;
    const std::vector<float> input = readFloatFile(base + ".input.f32le", inputElements);
    std::vector<float> output(static_cast<std::size_t>(outputElements));
    TraceCollector collector;
    const NeuralModelError error =
        rtengine::neural::detail::DemosaicNetXTransExecutorAccess::runWithObserver(
            *executor,
            input.data(),
            input.size(),
            golden.width,
            golden.height,
            output.data(),
            output.size(),
            collectTrace,
            &collector);
    require(!error, std::string("traced native inference failed: ") + error.message);
    require(collector.next == TRACE_SPECS.size(), "native trace lacks an activation");
    printStats("activation aggregate", collector.aggregate);
    return 0;
}

int benchmark()
{
    const char *path = reviewedPath();
    if (!path) {
        std::cout << "GHARBI_XTRANS_RTNN is not set; skipping native inference benchmark\n";
        return 77;
    }
    const auto loaded = rtengine::neural::loadDemosaicNetXTransModel(path);
    require(static_cast<bool>(loaded), "reviewed RTNN did not load for benchmark");
    auto executor = executorFor(loaded.model);
    const std::array<std::uint32_t, 4> sizes{{64, 128, 192, 256}};
    for (const std::uint32_t size : sizes) {
        std::vector<float> input(static_cast<std::size_t>(3) * size * size);
        for (std::size_t index = 0; index < input.size(); ++index) {
            input[index] = static_cast<float>((index * 37 + 11) % 1024) / 1024.f;
        }
        std::vector<float> output(static_cast<std::size_t>(3) * (size - 24) * (size - 24));
        require(!executor->run(input.data(), input.size(), size, size, output.data(), output.size()),
            "benchmark warmup failed");
        std::vector<double> milliseconds;
        for (unsigned iteration = 0; iteration < 3; ++iteration) {
            const auto start = std::chrono::steady_clock::now();
            require(!executor->run(input.data(), input.size(), size, size, output.data(), output.size()),
                "benchmark inference failed");
            const auto stop = std::chrono::steady_clock::now();
            milliseconds.push_back(std::chrono::duration<double, std::milli>(stop - start).count());
        }
        std::sort(milliseconds.begin(), milliseconds.end());
        const double outputPixels = static_cast<double>(size - 24) * (size - 24);
        std::cout << size << 'x' << size
                  << ": median_ms=" << milliseconds[1]
                  << ", output_megapixels_per_second=" << outputPixels / milliseconds[1] / 1000.0
                  << ", workspace_bytes=" << executor->workspaceBytes() << '\n';
    }
    return 0;
}

} // namespace demosaicnet_inference_test
