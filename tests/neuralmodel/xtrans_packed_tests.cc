#include "xtrans_packed_tests.h"

#include "rtengine/xtrans_cfa.h"
#include "rtengine/xtrans_packed.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <unistd.h>

#include <glib/gstdio.h>

namespace
{

void require(bool condition, const std::string &message)
{
    if (!condition) throw std::runtime_error(message);
}

void cfaFor(int a, int b, int c, int d, int ox, int oy, int result[6][6])
{
    for (int y = 0; y < 6; ++y) {
        for (int x = 0; x < 6; ++x) {
            result[y][x] = rtengine::PACKED_XTRANS_CFA[
                rtengine::positiveModulo(c * x + d * y + oy, 6)
            ][rtengine::positiveModulo(a * x + b * y + ox, 6)];
        }
    }
}

std::string cfaKey(const int cfa[6][6])
{
    std::string result;
    for (int y = 0; y < 6; ++y) for (int x = 0; x < 6; ++x) {
        result.push_back(static_cast<char>('0' + cfa[y][x]));
    }
    return result;
}

class MockRunner final : public rtengine::neural::PackedXTransRunner
{
public:
    enum class Mode { ECHO, CONSTANT, FAIL, NONFINITE };

    explicit MockRunner(Mode mode = Mode::ECHO) : mode_(mode) {}

    rtengine::neural::NeuralModelError run(
        const float *input, std::size_t inputCount, float *output, std::size_t outputCount) override
    {
        ++calls;
        if (!input || !output || inputCount != rtengine::neural::PACKED_XTRANS_INPUT_FLOATS ||
            outputCount != rtengine::neural::PACKED_XTRANS_OUTPUT_FLOATS) {
            return {rtengine::neural::NeuralModelErrorCode::SIZE, "mock tensor size differs"};
        }
        if (firstInput.empty()) firstInput.assign(input, input + inputCount);
        if (mode_ == Mode::FAIL && calls == failAt) {
            return {rtengine::neural::NeuralModelErrorCode::RUNTIME, "injected PackedXTransNet failure"};
        }
        constexpr std::size_t pixels = 288u * 288u;
        for (std::size_t p = 0; p < pixels; ++p) {
            if (mode_ == Mode::CONSTANT) {
                output[p] = -0.25f;
                output[pixels + p] = 0.5f;
                output[2 * pixels + p] = 1.25f;
            } else {
                output[p] = input[p];
                output[pixels + p] = input[p] + 0.125f;
                output[2 * pixels + p] = input[p] - 0.125f;
            }
        }
        if (mode_ == Mode::NONFINITE) output[144u * 288u + 144u] = std::numeric_limits<float>::infinity();
        return {};
    }

    const std::string &artifactSha256() const override { return artifact; }
    const std::string &runtimeVersion() const override { return runtime; }
    const std::string &provider() const override { return providerName; }
    const std::string &precision() const override { return precisionName; }
    const std::string &compileSource() const override { return compileSourceName; }
    std::uint64_t compilationMicroseconds() const override { return 0; }
    std::uint64_t lastInferenceMicroseconds() const override { return 0; }
    std::uint64_t workingBufferBytes() const override { return 1234; }

    Mode mode_;
    int calls = 0;
    int failAt = 2;
    std::vector<float> firstInput;
    std::string artifact = rtengine::neural::PACKED_XTRANS_ONNX_SHA256;
    std::string runtime = "mock";
    std::string providerName = "mock";
    std::string precisionName = "fp32";
    std::string compileSourceName = "mock";
};

rtengine::PackedXTransRunResult runCase(
    int width, int height, const int cfa[6][6], const std::shared_ptr<MockRunner> &runner,
    int margin, array2D<float> *redResult = nullptr, array2D<float> *greenResult = nullptr,
    array2D<float> *blueResult = nullptr)
{
    array2D<float> raw(width, height), red(width, height), green(width, height), blue(width, height);
    for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x) {
        raw[y][x] = static_cast<float>((1000 + y * width + x) % 65536);
    }
    const auto result = rtengine::demosaicPackedXTrans(
        raw, red, green, blue, width, height, cfa, runner, margin);
    if (redResult) *redResult = std::move(red);
    if (greenResult) *greenResult = std::move(green);
    if (blueResult) *blueResult = std::move(blue);
    return result;
}

#ifdef RT_TEST_WITH_ONNXRUNTIME
std::string temporaryPath(const char *suffix)
{
    return std::string(g_get_tmp_dir()) + "/rt-packed-test-" + std::to_string(g_random_int()) + suffix;
}
#endif

struct EnvironmentGuard final {
    explicit EnvironmentGuard(const char *name) : name(name)
    {
        const char *current = std::getenv(name);
        if (current) { hadValue = true; value = current; }
    }
    ~EnvironmentGuard()
    {
        if (hadValue) setenv(name.c_str(), value.c_str(), 1);
        else unsetenv(name.c_str());
    }
    std::string name;
    std::string value;
    bool hadValue = false;
};

} // namespace

namespace xtrans_packed_test
{

int mockContract()
{
    constexpr int matrices[8][4] = {
        {1,0,0,1}, {0,-1,1,0}, {-1,0,0,-1}, {0,1,-1,0},
        {-1,0,0,1}, {1,0,0,-1}, {0,1,1,0}, {0,-1,-1,0}
    };
    std::set<std::string> unique;
    for (const auto &matrix : matrices) {
        for (int oy = 0; oy < 6; ++oy) for (int ox = 0; ox < 6; ++ox) {
            int cfa[6][6];
            cfaFor(matrix[0], matrix[1], matrix[2], matrix[3], ox, oy, cfa);
            if (!unique.insert(cfaKey(cfa)).second) continue;
            auto runner = std::make_shared<MockRunner>();
            array2D<float> red, green, blue;
            const auto result = runCase(5, 7, cfa, runner, 60, &red, &green, &blue);
            require(result && result.tileCount == 1 && result.margin == 60 && result.stride == 168,
                "one of 18 PackedXTransNet CFA mappings failed");
            rtengine::XTransCfaTransform found{};
            require(rtengine::findCanonicalXTransTransform(cfa, rtengine::PACKED_XTRANS_CFA, found),
                "PackedXTransNet CFA transform was not found");
            rtengine::XTransCfaView view(found, 5, 7, rtengine::PACKED_XTRANS_CFA);
            for (int y = 0; y < 7; ++y) for (int x = 0; x < 5; ++x) {
                int u, v, ax, ay;
                view.actualToCanonical(x, y, u, v);
                view.canonicalToActual(u, v, ax, ay);
                require(ax == x && ay == y && view.colorAtCanonical(u, v) == cfa[y % 6][x % 6],
                    "PackedXTransNet CFA coordinate round trip differs");
            }
        }
    }
    require(unique.size() == 18, "PackedXTransNet CFA mappings do not collapse to 18 patterns");

    int canonical[6][6];
    cfaFor(1, 0, 0, 1, 0, 0, canonical);
    for (const int margin : {12, 60}) {
        auto runner = std::make_shared<MockRunner>(MockRunner::Mode::CONSTANT);
        array2D<float> red, green, blue;
        const auto result = runCase(401, 173, canonical, runner, margin, &red, &green, &blue);
        require(result && result.tileCount > 1 && result.workingBufferBytes > 1234,
            "PackedXTransNet multi-tile geometry or accounting differs");
        for (int y = 0; y < 173; ++y) for (int x = 0; x < 401; ++x) {
            require(std::abs(red[y][x] + 0.25f * 65535.f) < 0.02f &&
                    std::abs(green[y][x] - 0.5f * 65535.f) < 0.02f &&
                    std::abs(blue[y][x] - 1.25f * 65535.f) < 0.02f,
                "PackedXTransNet output was clipped, reinjected, or left unwritten at a seam");
        }
    }

    auto failing = std::make_shared<MockRunner>(MockRunner::Mode::FAIL);
    auto result = runCase(400, 31, canonical, failing, 60);
    require(!result && result.error.code == rtengine::neural::NeuralModelErrorCode::RUNTIME,
        "injected PackedXTransNet failure did not propagate");
    auto nonfinite = std::make_shared<MockRunner>(MockRunner::Mode::NONFINITE);
    result = runCase(100, 100, canonical, nonfinite, 60);
    require(!result && result.error.code == rtengine::neural::NeuralModelErrorCode::NONFINITE,
        "non-finite PackedXTransNet output was accepted");

    array2D<float> raw(1, 1), red(1, 1), green(1, 1), blue(1, 1);
    raw[0][0] = std::numeric_limits<float>::quiet_NaN();
    result = rtengine::demosaicPackedXTrans(raw, red, green, blue, 1, 1, canonical,
        std::make_shared<MockRunner>(), 60);
    require(!result && result.error.code == rtengine::neural::NeuralModelErrorCode::NONFINITE,
        "non-finite PackedXTransNet input was accepted");
    int invalid[6][6] = {};
    raw[0][0] = 1.f;
    result = rtengine::demosaicPackedXTrans(raw, red, green, blue, 1, 1, invalid,
        std::make_shared<MockRunner>(), 60);
    require(!result && result.error.code == rtengine::neural::NeuralModelErrorCode::SCHEMA,
        "unsupported PackedXTransNet CFA was accepted");
    result = rtengine::demosaicPackedXTrans(raw, red, green, blue, 1, 1, canonical,
        std::make_shared<MockRunner>(), 13);
    require(!result && result.error.code == rtengine::neural::NeuralModelErrorCode::RANGE,
        "unsupported PackedXTransNet margin was accepted");
    return 0;
}

int loaderContract()
{
#if !defined(RT_TEST_WITH_ONNXRUNTIME) && !defined(RT_TEST_WITH_MIGRAPHX)
    auto disabled = rtengine::neural::loadCachedPackedXTransRunner("anything.onnx");
    require(!disabled && disabled.error.code == rtengine::neural::NeuralModelErrorCode::RUNTIME,
        "runtime-disabled PackedXTransNet load did not return RUNTIME");
    return 0;
#else
    EnvironmentGuard backend("RT_PACKED_XTRANS_BACKEND");
    EnvironmentGuard precision("RT_PACKED_XTRANS_PRECISION");
    setenv("RT_PACKED_XTRANS_BACKEND", "not-a-backend", 1);
    auto invalid = rtengine::neural::loadCachedPackedXTransRunner("anything.onnx");
    require(!invalid && invalid.error.code == rtengine::neural::NeuralModelErrorCode::ENUM,
        "unknown PackedXTransNet backend did not return ENUM");
    setenv("RT_PACKED_XTRANS_BACKEND", "onnxruntime-cpu", 1);
    setenv("RT_PACKED_XTRANS_PRECISION", "fp16", 1);
    invalid = rtengine::neural::loadCachedPackedXTransRunner("anything.onnx");
    require(!invalid && invalid.error.code == rtengine::neural::NeuralModelErrorCode::ENUM,
        "CPU FP16 PackedXTransNet request did not return ENUM");
    setenv("RT_PACKED_XTRANS_PRECISION", "fp32", 1);
    auto missing = rtengine::neural::loadCachedPackedXTransRunner("/definitely/missing/packed.onnx");
#if defined(RT_TEST_WITH_ONNXRUNTIME)
    require(!missing && missing.error.code == rtengine::neural::NeuralModelErrorCode::IO,
        "missing PackedXTransNet model did not return IO");
    const std::string shortPath = temporaryPath(".onnx");
    {
        std::unique_ptr<std::FILE, int (*)(std::FILE *)> file(g_fopen(shortPath.c_str(), "wb"), std::fclose);
        require(static_cast<bool>(file) && std::fputc(0, file.get()) == 0, "cannot create short PackedXTransNet fixture");
    }
    auto shortModel = rtengine::neural::loadCachedPackedXTransRunner(shortPath);
    g_remove(shortPath.c_str());
    require(!shortModel && shortModel.error.code == rtengine::neural::NeuralModelErrorCode::SIZE,
        "short PackedXTransNet model did not return SIZE");
    const std::string digestPath = temporaryPath("-digest.onnx");
    {
        std::unique_ptr<std::FILE, int (*)(std::FILE *)> file(g_fopen(digestPath.c_str(), "wb"), std::fclose);
        require(static_cast<bool>(file) && std::fseek(file.get(), 1673648L - 1, SEEK_SET) == 0 &&
                std::fputc(0, file.get()) == 0, "cannot create PackedXTransNet digest fixture");
    }
    auto digest = rtengine::neural::loadCachedPackedXTransRunner(digestPath);
    g_remove(digestPath.c_str());
    require(!digest && digest.error.code == rtengine::neural::NeuralModelErrorCode::DIGEST,
        "wrong PackedXTransNet digest did not return DIGEST");
#endif
    return 0;
#endif
}

int reviewedModel()
{
    const char *path = std::getenv("PACKED_XTRANS_ONNX");
    if (!path || !*path) { std::printf("SKIP: PACKED_XTRANS_ONNX is not set\n"); return 77; }
#ifndef RT_TEST_WITH_ONNXRUNTIME
    std::printf("SKIP: RawTherapee was built without ONNX Runtime\n");
    return 77;
#else
    EnvironmentGuard backend("RT_PACKED_XTRANS_BACKEND");
    EnvironmentGuard precision("RT_PACKED_XTRANS_PRECISION");
    setenv("RT_PACKED_XTRANS_BACKEND", "onnxruntime-cpu", 1);
    setenv("RT_PACKED_XTRANS_PRECISION", "fp32", 1);
    const auto first = rtengine::neural::loadCachedPackedXTransRunner(path);
    require(first && first.runner->artifactSha256() == rtengine::neural::PACKED_XTRANS_ONNX_SHA256 &&
            first.runner->runtimeVersion() == "1.27.0" && first.runner->provider() == "CPUExecutionProvider",
        "reviewed PackedXTransNet identity differs: " + first.error.message);
    std::array<std::shared_ptr<rtengine::neural::PackedXTransRunner>, 8> cached;
    std::array<std::thread, 8> threads;
    for (std::size_t i = 0; i < threads.size(); ++i) threads[i] = std::thread([&, i]() {
        cached[i] = rtengine::neural::loadCachedPackedXTransRunner(path).runner;
    });
    for (auto &thread : threads) thread.join();
    for (const auto &runner : cached) require(runner == first.runner, "PackedXTransNet cache identity differs");
    std::vector<float> input(rtengine::neural::PACKED_XTRANS_INPUT_FLOATS);
    for (std::size_t i = 0; i < input.size(); ++i) input[i] = static_cast<float>((i * 17u + 31u) % 1024u) / 1023.f;
    std::vector<float> a(rtengine::neural::PACKED_XTRANS_OUTPUT_FLOATS), b(a.size());
    require(!first.runner->run(input.data(), input.size(), a.data(), a.size()) &&
            !first.runner->run(input.data(), input.size(), b.data(), b.size()) && a == b,
        "reviewed PackedXTransNet inference is not deterministic");
    for (float value : a) require(std::isfinite(value), "reviewed PackedXTransNet output is non-finite");
    return 0;
#endif
}

int migraphxParity()
{
#if !defined(RT_TEST_WITH_ONNXRUNTIME) || !defined(RT_TEST_WITH_MIGRAPHX)
    std::printf("SKIP: both ONNX Runtime and MIGraphX are required\n");
    return 77;
#else
    const char *path = std::getenv("PACKED_XTRANS_ONNX");
    if (!path || !*path) { std::printf("SKIP: PACKED_XTRANS_ONNX is not set\n"); return 77; }
    if (access("/dev/kfd", R_OK | W_OK) != 0) { std::printf("SKIP: /dev/kfd is unavailable\n"); return 77; }
    EnvironmentGuard backend("RT_PACKED_XTRANS_BACKEND");
    EnvironmentGuard precision("RT_PACKED_XTRANS_PRECISION");
    setenv("RT_PACKED_XTRANS_BACKEND", "onnxruntime-cpu", 1);
    setenv("RT_PACKED_XTRANS_PRECISION", "fp32", 1);
    const auto cpu = rtengine::neural::loadCachedPackedXTransRunner(path);
    require(static_cast<bool>(cpu), "cannot load PackedXTransNet CPU reference: " + cpu.error.message);
    setenv("RT_PACKED_XTRANS_BACKEND", "migraphx", 1);
    const auto gpu = rtengine::neural::loadCachedPackedXTransRunner(path);
    require(static_cast<bool>(gpu), "cannot load PackedXTransNet MIGraphX runner: " + gpu.error.message);
    std::vector<float> input(rtengine::neural::PACKED_XTRANS_INPUT_FLOATS);
    for (std::size_t i = 0; i < input.size(); ++i) input[i] = static_cast<float>((i * 17u + 31u) % 1024u) / 1023.f;
    std::vector<float> reference(rtengine::neural::PACKED_XTRANS_OUTPUT_FLOATS), candidate(reference.size());
    require(!cpu.runner->run(input.data(), input.size(), reference.data(), reference.size()) &&
            !gpu.runner->run(input.data(), input.size(), candidate.data(), candidate.size()),
        "PackedXTransNet CPU/GPU inference failed");
    double squared = 0.0, maximum = 0.0;
    std::vector<double> absolute;
    absolute.reserve(reference.size());
    for (std::size_t i = 0; i < reference.size(); ++i) {
        const double difference = std::abs(static_cast<double>(reference[i]) - candidate[i]);
        absolute.push_back(difference); squared += difference * difference; maximum = std::max(maximum, difference);
    }
    std::sort(absolute.begin(), absolute.end());
    const double rms = std::sqrt(squared / absolute.size());
    const double p99 = absolute[static_cast<std::size_t>(std::ceil(0.99 * absolute.size())) - 1];
    require(maximum <= 5e-4 && rms <= 5e-5 && p99 <= 1e-4,
        "PackedXTransNet MIGraphX FP32 exceeds the reviewed reference tolerance");
    return 0;
#endif
}

} // namespace xtrans_packed_test
