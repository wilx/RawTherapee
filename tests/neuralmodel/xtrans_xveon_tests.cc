#include "xtrans_xveon_tests.h"

#include "rtengine/xtrans_cfa.h"
#include "rtengine/xtrans_xveon.h"
#ifdef RT_TEST_WITH_ONNXRUNTIME
#include "rtengine/xveon_ort_bridge.h"
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
            result[y][x] = rtengine::XVEON_XTRANS_CFA[
                rtengine::positiveModulo(c * x + d * y + oy, 6)
            ][rtengine::positiveModulo(a * x + b * y + ox, 6)];
        }
    }
}

std::string cfaKey(const int cfa[6][6])
{
    std::string result;
    for (int y = 0; y < 6; ++y) for (int x = 0; x < 6; ++x) result.push_back(static_cast<char>('0' + cfa[y][x]));
    return result;
}

class MockRunner final : public rtengine::neural::XVeonXTransRunner
{
public:
    enum class Mode { CONSTANT, ECHO, FAIL, NONFINITE };

    explicit MockRunner(Mode mode = Mode::CONSTANT) : mode_(mode) {}

    rtengine::neural::NeuralModelError run(
        const float *input, std::size_t inputCount, float *output, std::size_t outputCount) override
    {
        ++calls;
        if (inputCount != rtengine::neural::XVEON_INPUT_FLOATS || outputCount != rtengine::neural::XVEON_OUTPUT_FLOATS) {
            return {rtengine::neural::NeuralModelErrorCode::SIZE, "mock tensor size differs"};
        }
        if (firstInput.empty()) firstInput.assign(input, input + inputCount);
        if (mode_ == Mode::FAIL && calls == failAt) {
            return {rtengine::neural::NeuralModelErrorCode::RUNTIME, "injected runner failure"};
        }
        const std::size_t pixels = 288u * 288u;
        for (std::size_t pixel = 0; pixel < pixels; ++pixel) {
            if (mode_ == Mode::ECHO) {
                output[pixel] = input[pixel];
                output[pixels + pixel] = input[pixel] + 0.25f;
                output[2 * pixels + pixel] = input[pixel] - 0.25f;
            } else {
                output[pixel] = -0.25f;
                output[pixels + pixel] = 0.5f;
                output[2 * pixels + pixel] = 1.25f;
            }
        }
        if (mode_ == Mode::NONFINITE) output[50u * 288u + 50u] = std::numeric_limits<float>::infinity();
        return {};
    }

    const std::string &artifactSha256() const override { return artifact; }
    const std::string &runtimeVersion() const override { return runtime; }
    const std::string &provider() const override { return providerName; }
    const std::string &compileSource() const override { return source; }
    std::uint64_t compilationMicroseconds() const override { return 0; }
    std::uint64_t lastInferenceMicroseconds() const override { return 0; }
    std::uint64_t workingBufferBytes() const override { return 1234; }

    Mode mode_;
    int calls = 0;
    int failAt = 2;
    std::vector<float> firstInput;
    std::string artifact = std::string(64, 'a');
    std::string runtime = "mock";
    std::string providerName = "mock";
    std::string source = "mock";
};

rtengine::XVeonXTransRunResult runCase(
    int width, int height, const int cfa[6][6], const std::shared_ptr<MockRunner> &runner,
    array2D<float> *rawResult = nullptr, array2D<float> *redResult = nullptr,
    array2D<float> *greenResult = nullptr, array2D<float> *blueResult = nullptr)
{
    array2D<float> raw(width, height), red(width, height), green(width, height), blue(width, height);
    for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x) raw[y][x] = static_cast<float>(1000 + y * width + x);
    const auto result = rtengine::demosaicXVeonXTrans(raw, red, green, blue, width, height, cfa, runner);
    if (rawResult) *rawResult = std::move(raw);
    if (redResult) *redResult = std::move(red);
    if (greenResult) *greenResult = std::move(green);
    if (blueResult) *blueResult = std::move(blue);
    return result;
}

#if defined(RT_TEST_WITH_ONNXRUNTIME) || defined(RT_TEST_WITH_MIGRAPHX) || defined(RT_TEST_WITH_TVM_VULKAN)
std::string temporaryPath(const char *suffix)
{
    return std::string(g_get_tmp_dir()) + "/rt-xveon-test-" + std::to_string(g_random_int()) + suffix;
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

namespace xtrans_xveon_test
{

int mockContract()
{
    constexpr int matrices[8][4] = {
        {1,0,0,1}, {0,-1,1,0}, {-1,0,0,-1}, {0,1,-1,0},
        {-1,0,0,1}, {1,0,0,-1}, {0,1,1,0}, {0,-1,-1,0}
    };
    std::set<std::string> unique;
    for (const auto &matrix : matrices) {
        for (int oy = 0; oy < 6; ++oy) {
            for (int ox = 0; ox < 6; ++ox) {
                int cfa[6][6];
                cfaFor(matrix[0], matrix[1], matrix[2], matrix[3], ox, oy, cfa);
                if (!unique.insert(cfaKey(cfa)).second) continue;
                auto runner = std::make_shared<MockRunner>();
                const auto result = runCase(5, 7, cfa, runner);
                require(static_cast<bool>(result), "one of 18 X-veon CFA mappings failed");
                rtengine::XTransCfaTransform found{};
                require(rtengine::findCanonicalXTransTransform(cfa, rtengine::XVEON_XTRANS_CFA, found),
                    "X-veon CFA transform was not found");
                rtengine::XTransCfaView view(found, 5, 7, rtengine::XVEON_XTRANS_CFA);
                for (int y = 0; y < 7; ++y) for (int x = 0; x < 5; ++x) {
                    int u, v, ax, ay;
                    view.actualToCanonical(x, y, u, v);
                    view.canonicalToActual(u, v, ax, ay);
                    require(ax == x && ay == y && view.colorAtCanonical(u, v) == cfa[y % 6][x % 6],
                        "X-veon CFA coordinate or colour mapping differs");
                }
            }
        }
    }
    require(unique.size() == 18, "X-veon CFA mappings do not collapse to 18 unique patterns");

    int canonical[6][6];
    cfaFor(1, 0, 0, 1, 0, 0, canonical);
    auto masks = std::make_shared<MockRunner>(MockRunner::Mode::ECHO);
    array2D<float> raw, red, green, blue;
    auto result = runCase(3, 2, canonical, masks, &raw, &red, &green, &blue);
    require(result && result.tileCount == 1 && result.paddedWidth == 288 && result.paddedHeight == 288,
        "tiny X-veon tile grid differs");
    const std::size_t pixels = 288u * 288u;
    for (int y = 0; y < 288; ++y) for (int x = 0; x < 288; ++x) {
        const std::size_t p = static_cast<std::size_t>(y) * 288 + x;
        const int expected = rtengine::XVEON_XTRANS_CFA[y % 6][x % 6];
        require(masks->firstInput[pixels + p] == (expected == 0 ? 1.f : 0.f) &&
                masks->firstInput[2 * pixels + p] == (expected == 1 ? 1.f : 0.f) &&
                masks->firstInput[3 * pixels + p] == (expected == 2 ? 1.f : 0.f),
            "X-veon canonical channel masks differ");
        if (x >= 3 || y >= 2) require(masks->firstInput[p] == 0.f, "right/bottom extension is not zero");
    }

    int translated[6][6] = {};
    rtengine::XTransCfaTransform found{};
    bool foundLeadingPadding = false;
    for (int oy = 0; oy < 6 && !foundLeadingPadding; ++oy) {
        for (int ox = 0; ox < 6 && !foundLeadingPadding; ++ox) {
            cfaFor(1, 0, 0, 1, ox, oy, translated);
            if (rtengine::findCanonicalXTransTransform(translated, rtengine::XVEON_XTRANS_CFA, found)) {
                rtengine::XTransCfaView candidate(found, 4, 3, rtengine::XVEON_XTRANS_CFA);
                foundLeadingPadding = rtengine::positiveModulo(candidate.minimumX(), 6) > 0 &&
                    rtengine::positiveModulo(candidate.minimumY(), 6) > 0;
            }
        }
    }
    require(foundLeadingPadding, "could not construct an X-veon leading-reflection case");
    auto reflected = std::make_shared<MockRunner>(MockRunner::Mode::ECHO);
    result = runCase(4, 3, translated, reflected);
    require(static_cast<bool>(result), "translated reflection case failed");
    rtengine::XTransCfaView view(found, 4, 3, rtengine::XVEON_XTRANS_CFA);
    const int padX = rtengine::positiveModulo(view.minimumX(), 6);
    const int padY = rtengine::positiveModulo(view.minimumY(), 6);
    require(padX > 0 && padY > 0, "test translation did not exercise leading reflection");
    int ax = 0, ay = 0;
    view.canonicalToActual(std::min(padX - 1, view.width() - 1), std::min(padY - 1, view.height() - 1), ax, ay);
    const float expectedReflected = static_cast<float>(1000 + ay * 4 + ax) / 65535.f;
    require(reflected->firstInput[0] == expectedReflected, "top/left edge-repeating reflection differs");

    auto signedRunner = std::make_shared<MockRunner>();
    result = runCase(100, 100, canonical, signedRunner, nullptr, &red, &green, &blue);
    require(result && result.workingBufferBytes > 1234, "ordinary X-veon run/accounting failed");
    require(std::abs(red[50][50] + 0.25f * 65535.f) < 0.01f &&
            std::abs(green[50][50] - 0.5f * 65535.f) < 0.01f &&
            std::abs(blue[50][50] - 1.25f * 65535.f) < 0.02f,
        "signed or over-range X-veon output was altered");

    auto failing = std::make_shared<MockRunner>(MockRunner::Mode::FAIL);
    result = runCase(300, 31, canonical, failing);
    require(!result && result.error.code == rtengine::neural::NeuralModelErrorCode::RUNTIME,
        "injected X-veon inference failure did not propagate");
    auto nonfinite = std::make_shared<MockRunner>(MockRunner::Mode::NONFINITE);
    result = runCase(100, 100, canonical, nonfinite);
    require(!result && result.error.code == rtengine::neural::NeuralModelErrorCode::NONFINITE,
        "non-finite X-veon output was accepted");

    array2D<float> badRaw(1, 1), badR(1, 1), badG(1, 1), badB(1, 1);
    badRaw[0][0] = std::numeric_limits<float>::quiet_NaN();
    result = rtengine::demosaicXVeonXTrans(badRaw, badR, badG, badB, 1, 1, canonical, signedRunner);
    require(!result && result.error.code == rtengine::neural::NeuralModelErrorCode::NONFINITE,
        "non-finite X-veon input was accepted");
    int invalid[6][6] = {};
    badRaw[0][0] = 1.f;
    result = rtengine::demosaicXVeonXTrans(badRaw, badR, badG, badB, 1, 1, invalid, signedRunner);
    require(!result && result.error.code == rtengine::neural::NeuralModelErrorCode::SCHEMA,
        "unsupported X-veon CFA was accepted");
    return 0;
}

int loaderContract()
{
#if defined(RT_TEST_WITH_ONNXRUNTIME) || defined(RT_TEST_WITH_MIGRAPHX) || defined(RT_TEST_WITH_TVM_VULKAN)
    EnvironmentGuard backendGuard("RT_XVEON_XTRANS_BACKEND");
    setenv("RT_XVEON_XTRANS_BACKEND", "not-a-backend", 1);
    auto invalidBackend = rtengine::neural::loadCachedXVeonXTransRunner("anything.onnx");
    require(!invalidBackend && invalidBackend.error.code == rtengine::neural::NeuralModelErrorCode::ENUM,
        "unknown X-veon backend did not return ENUM");

#ifdef RT_TEST_WITH_ONNXRUNTIME
    const char *availableBackend = "onnxruntime-cpu";
    const long reviewedBytes = 15536134L;
#elif defined(RT_TEST_WITH_MIGRAPHX)
    const char *availableBackend = "migraphx";
    const long reviewedBytes = 15536134L;
#else
    const char *availableBackend = "tvm-vulkan";
    const long reviewedBytes = 32594896L;
#endif
    setenv("RT_XVEON_XTRANS_BACKEND", availableBackend, 1);

    auto missing = rtengine::neural::loadCachedXVeonXTransRunner("/definitely/missing/xtrans.onnx");
    require(!missing && missing.error.code == rtengine::neural::NeuralModelErrorCode::IO,
        "missing X-veon model did not return IO");
    const std::string shortPath = temporaryPath(".onnx");
    {
        std::unique_ptr<std::FILE, int (*)(std::FILE *)> file(g_fopen(shortPath.c_str(), "wb"), std::fclose);
        require(static_cast<bool>(file), "cannot create short ONNX fixture");
        std::fputc(0, file.get());
    }
    auto shortModel = rtengine::neural::loadCachedXVeonXTransRunner(shortPath);
    g_remove(shortPath.c_str());
    require(!shortModel && shortModel.error.code == rtengine::neural::NeuralModelErrorCode::SIZE,
        "short X-veon model did not return SIZE");

    const std::string retryPath = temporaryPath("-retry.onnx");
    auto missingRetry = rtengine::neural::loadCachedXVeonXTransRunner(retryPath);
    require(!missingRetry && missingRetry.error.code == rtengine::neural::NeuralModelErrorCode::IO,
        "initial missing X-veon retry fixture did not return IO");
    {
        std::unique_ptr<std::FILE, int (*)(std::FILE *)> file(g_fopen(retryPath.c_str(), "wb"), std::fclose);
        require(static_cast<bool>(file) && std::fputc(0, file.get()) == 0,
            "cannot create X-veon retry fixture");
    }
    auto retried = rtengine::neural::loadCachedXVeonXTransRunner(retryPath);
    g_remove(retryPath.c_str());
    require(!retried && retried.error.code == rtengine::neural::NeuralModelErrorCode::SIZE,
        "failed X-veon model load was incorrectly cached");

    const std::string wrongDigestPath = temporaryPath("-wrong-digest.onnx");
    {
        std::unique_ptr<std::FILE, int (*)(std::FILE *)> file(g_fopen(wrongDigestPath.c_str(), "wb"), std::fclose);
        require(static_cast<bool>(file), "cannot create wrong-digest ONNX fixture");
        require(std::fseek(file.get(), reviewedBytes - 1, SEEK_SET) == 0 && std::fputc(0, file.get()) == 0,
            "cannot size wrong-digest ONNX fixture");
    }
    auto wrongDigest = rtengine::neural::loadCachedXVeonXTransRunner(wrongDigestPath);
    g_remove(wrongDigestPath.c_str());
    require(!wrongDigest && wrongDigest.error.code == rtengine::neural::NeuralModelErrorCode::DIGEST,
        "same-size wrong-digest X-veon model did not return DIGEST");

#ifndef RT_TEST_WITH_TVM_VULKAN
    const std::string excessivePath = temporaryPath("-excessive.onnx");
    {
        std::unique_ptr<std::FILE, int (*)(std::FILE *)> file(g_fopen(excessivePath.c_str(), "wb"), std::fclose);
        require(static_cast<bool>(file), "cannot create excessive ONNX fixture");
        require(std::fseek(file.get(), 64L * 1024L * 1024L, SEEK_SET) == 0 && std::fputc(0, file.get()) == 0,
            "cannot size excessive ONNX fixture");
    }
    auto excessive = rtengine::neural::loadCachedXVeonXTransRunner(excessivePath);
    g_remove(excessivePath.c_str());
    require(!excessive && excessive.error.code == rtengine::neural::NeuralModelErrorCode::LIMIT,
        "64 MiB-plus-one X-veon model did not return LIMIT");
#endif

#ifdef RT_TEST_WITH_ONNXRUNTIME
    RtXveonOrtSession *malformedSession = nullptr;
    char malformedMessage[1024] = {};
    const unsigned char malformedModel = 0;
    const int malformed = rt_xveon_ort_create(
        &malformedModel, 1, &malformedSession, malformedMessage, sizeof(malformedMessage));
    rt_xveon_ort_release(malformedSession);
    require(malformed == RT_XVEON_ORT_RUNTIME && malformedMessage[0],
        "malformed ONNX did not produce a bounded runtime error");
#endif
#else
    auto disabled = rtengine::neural::loadCachedXVeonXTransRunner("anything.onnx");
    require(!disabled && disabled.error.code == rtengine::neural::NeuralModelErrorCode::RUNTIME,
        "ONNX-disabled build did not return RUNTIME");
#endif
    return 0;
}

int reviewedModel()
{
    const char *path = std::getenv("XVEON_XTRANS_ONNX");
    if (!path || !*path) {
        std::printf("SKIP: XVEON_XTRANS_ONNX is not set\n");
        return 77;
    }
#ifndef RT_TEST_WITH_ONNXRUNTIME
    std::printf("SKIP: RawTherapee was built without ONNX Runtime\n");
    return 77;
#else
    const auto first = rtengine::neural::loadCachedXVeonXTransRunner(path);
    require(static_cast<bool>(first), "cannot load reviewed X-veon model: " + first.error.message);
    require(first.runner->artifactSha256() == "45b1fa22b0027868fd5c20ec7b59234ed5aeb35de89fbc0950a4bec67f328500" &&
            first.runner->runtimeVersion() == "1.27.0" && first.runner->provider() == "CPUExecutionProvider",
        "reviewed X-veon identity differs");

    std::vector<unsigned char> wrongContract(15536134);
    {
        std::unique_ptr<std::FILE, int (*)(std::FILE *)> file(g_fopen(path, "rb"), std::fclose);
        require(static_cast<bool>(file) &&
                std::fread(wrongContract.data(), 1, wrongContract.size(), file.get()) == wrongContract.size(),
            "cannot read reviewed model for private contract fixture");
    }
    const std::array<unsigned char, 5> inputName{{'i', 'n', 'p', 'u', 't'}};
    bool renamed = false;
    for (auto position = wrongContract.begin();;) {
        position = std::search(position, wrongContract.end(), inputName.begin(), inputName.end());
        if (position == wrongContract.end()) break;
        position[4] = 'X';
        renamed = true;
        ++position;
    }
    require(renamed, "private wrong-input-name fixture did not change the model");
    RtXveonOrtSession *wrongContractSession = nullptr;
    char wrongContractMessage[1024] = {};
    const int wrongContractStatus = rt_xveon_ort_create(
        wrongContract.data(), wrongContract.size(), &wrongContractSession,
        wrongContractMessage, sizeof(wrongContractMessage));
    rt_xveon_ort_release(wrongContractSession);
    require(wrongContractStatus == RT_XVEON_ORT_SCHEMA && wrongContractMessage[0],
        "valid ONNX with a wrong input contract did not return SCHEMA");

    std::array<std::shared_ptr<rtengine::neural::XVeonXTransRunner>, 8> cached;
    std::array<std::thread, 8> threads;
    for (std::size_t i = 0; i < threads.size(); ++i) threads[i] = std::thread([&, i]() {
        cached[i] = rtengine::neural::loadCachedXVeonXTransRunner(path).runner;
    });
    for (auto &thread : threads) thread.join();
    for (const auto &runner : cached) require(runner == first.runner, "X-veon cache identity differs under concurrency");

    std::vector<float> input(rtengine::neural::XVEON_INPUT_FLOATS, 0.f);
    std::vector<float> left(rtengine::neural::XVEON_OUTPUT_FLOATS), right(left.size());
    const std::size_t pixels = 288u * 288u;
    for (int y = 0; y < 288; ++y) for (int x = 0; x < 288; ++x) {
        const std::size_t p = static_cast<std::size_t>(y) * 288 + x;
        const int channel = rtengine::XVEON_XTRANS_CFA[y % 6][x % 6];
        input[p] = static_cast<float>((x * 17 + y * 31) % 1024) / 1023.f;
        input[(static_cast<std::size_t>(channel) + 1) * pixels + p] = 1.f;
    }
    require(!first.runner->run(input.data(), input.size(), left.data(), left.size()), "reviewed X-veon inference failed");
    require(!first.runner->run(input.data(), input.size(), right.data(), right.size()), "repeated X-veon inference failed");
    require(left == right, "repeated X-veon output differs");
    for (float value : left) require(std::isfinite(value), "reviewed X-veon output is non-finite");
    return 0;
#endif
}

int migraphxParity()
{
#if !defined(RT_TEST_WITH_ONNXRUNTIME) || !defined(RT_TEST_WITH_MIGRAPHX)
    std::printf("SKIP: both ONNX Runtime and MIGraphX are required\n");
    return 77;
#else
    const char *path = std::getenv("XVEON_XTRANS_ONNX");
    if (!path || !*path) {
        std::printf("SKIP: XVEON_XTRANS_ONNX is not set\n");
        return 77;
    }
    if (access("/dev/kfd", R_OK | W_OK) != 0) {
        std::printf("SKIP: /dev/kfd is unavailable to this process\n");
        return 77;
    }

    setenv("RT_XVEON_XTRANS_BACKEND", "onnxruntime-cpu", 1);
    const auto cpu = rtengine::neural::loadCachedXVeonXTransRunner(path);
    require(static_cast<bool>(cpu), "cannot load CPU X-veon reference: " + cpu.error.message);
    setenv("RT_XVEON_XTRANS_BACKEND", "migraphx", 1);
    const auto gpu = rtengine::neural::loadCachedXVeonXTransRunner(path);
    require(static_cast<bool>(gpu), "cannot load MIGraphX X-veon runner: " + gpu.error.message);
    require(gpu.runner->provider().find("MIGraphX-gpu") == 0 &&
            gpu.runner->runtimeVersion() == "2.15.0-20250912-17-200-gde19b73ad",
        "MIGraphX runner identity differs");

    std::vector<float> input(rtengine::neural::XVEON_INPUT_FLOATS, 0.f);
    std::vector<float> reference(rtengine::neural::XVEON_OUTPUT_FLOATS);
    std::vector<float> candidate(reference.size()), repeated(reference.size());
    const std::size_t pixels = 288u * 288u;
    for (int y = 0; y < 288; ++y) for (int x = 0; x < 288; ++x) {
        const std::size_t p = static_cast<std::size_t>(y) * 288 + x;
        const int channel = rtengine::XVEON_XTRANS_CFA[y % 6][x % 6];
        input[p] = static_cast<float>((y * 31 + x * 17) % 1024) / 1023.f;
        input[(static_cast<std::size_t>(channel) + 1) * pixels + p] = 1.f;
    }
    require(!cpu.runner->run(input.data(), input.size(), reference.data(), reference.size()),
        "CPU X-veon reference inference failed");
    require(!gpu.runner->run(input.data(), input.size(), candidate.data(), candidate.size()),
        "MIGraphX X-veon inference failed");
    require(!gpu.runner->run(input.data(), input.size(), repeated.data(), repeated.size()),
        "repeated MIGraphX X-veon inference failed");
    require(candidate == repeated, "repeated MIGraphX tile output differs");

    std::vector<double> absolute;
    absolute.reserve(reference.size());
    double squared = 0.0;
    for (std::size_t i = 0; i < reference.size(); ++i) {
        require(std::isfinite(reference[i]) && std::isfinite(candidate[i]),
            "CPU or MIGraphX tile output is non-finite");
        const double difference = std::abs(static_cast<double>(reference[i]) - candidate[i]);
        absolute.push_back(difference);
        squared += difference * difference;
    }
    std::sort(absolute.begin(), absolute.end());
    const double rms = std::sqrt(squared / absolute.size());
    const double p99 = absolute[static_cast<std::size_t>(std::ceil(0.99 * absolute.size())) - 1];
    require(absolute.back() <= 0.005 && rms <= 0.0005 && p99 <= 0.001,
        "MIGraphX deterministic tile exceeds the reviewed aggregate tolerance");

    int cfa[6][6];
    cfaFor(0, -1, 1, 0, 2, 3, cfa);
    const int dimensions[3][2] = {{173, 31}, {400, 31}, {31, 400}};
    for (const auto &dimension : dimensions) {
        const int width = dimension[0];
        const int height = dimension[1];
        array2D<float> raw(width, height), cpuR(width, height), cpuG(width, height), cpuB(width, height);
        array2D<float> gpuR(width, height), gpuG(width, height), gpuB(width, height);
        for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x) {
            raw[y][x] = static_cast<float>((y * 197 + x * 113) % 65536);
        }
        const auto cpuResult = rtengine::demosaicXVeonXTrans(
            raw, cpuR, cpuG, cpuB, width, height, cfa, cpu.runner);
        const auto gpuResult = rtengine::demosaicXVeonXTrans(
            raw, gpuR, gpuG, gpuB, width, height, cfa, gpu.runner);
        require(cpuResult && gpuResult, "CPU/GPU X-veon wrapper comparison failed");
        for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x) {
            require(std::abs(cpuR[y][x] - gpuR[y][x]) / 65535.f <= 0.005f &&
                    std::abs(cpuG[y][x] - gpuG[y][x]) / 65535.f <= 0.005f &&
                    std::abs(cpuB[y][x] - gpuB[y][x]) / 65535.f <= 0.005f,
                "MIGraphX wrapper exceeds the reviewed per-sample bound");
        }
    }
    unsetenv("RT_XVEON_XTRANS_BACKEND");
    return 0;
#endif
}

int tvmVulkanParity()
{
#ifndef RT_TEST_WITH_TVM_VULKAN
    std::printf("SKIP: RawTherapee was built without TVM Vulkan\n");
    return 77;
#else
    const char *modulePath = std::getenv("XVEON_XTRANS_TVM_MODULE");
    if (!modulePath || !*modulePath) {
        std::printf("SKIP: XVEON_XTRANS_TVM_MODULE is not set\n");
        return 77;
    }
    EnvironmentGuard backend("RT_XVEON_XTRANS_BACKEND");
    setenv("RT_XVEON_XTRANS_BACKEND", "tvm-vulkan", 1);
    const auto tvm = rtengine::neural::loadCachedXVeonXTransRunner(modulePath);
    require(static_cast<bool>(tvm), "cannot load X-veon TVM module: " + tvm.error.message);
    require(
        tvm.runner->artifactSha256() ==
            "8648e3741a98345c8bc76b9e1c853a3b4ef58155b65ed726f9c2fda0226c206d" &&
            tvm.runner->provider().find("TVM-Vulkan/") == 0 &&
            tvm.runner->compileSource() == "ahead-of-time",
        "X-veon TVM runner identity differs");

    std::vector<float> input(rtengine::neural::XVEON_INPUT_FLOATS, 0.f);
    constexpr std::size_t pixels = 288u * 288u;
    for (int y = 0; y < 288; ++y) for (int x = 0; x < 288; ++x) {
        const std::size_t p = static_cast<std::size_t>(y) * 288 + x;
        const int channel = rtengine::XVEON_XTRANS_CFA[y % 6][x % 6];
        input[p] = static_cast<float>((y * 31 + x * 17) % 1024) / 1023.f;
        input[(static_cast<std::size_t>(channel) + 1) * pixels + p] = 1.f;
    }
    std::vector<float> candidate(rtengine::neural::XVEON_OUTPUT_FLOATS);
    std::vector<float> repeated(candidate.size());
    require(!tvm.runner->run(input.data(), input.size(), candidate.data(), candidate.size()) &&
            !tvm.runner->run(input.data(), input.size(), repeated.data(), repeated.size()),
        "X-veon TVM Vulkan inference failed");
    require(candidate == repeated, "repeated X-veon TVM output differs");
    for (float value : candidate) require(std::isfinite(value), "X-veon TVM output is non-finite");

#ifdef RT_TEST_WITH_ONNXRUNTIME
    const char *onnxPath = std::getenv("XVEON_XTRANS_ONNX");
    if (onnxPath && *onnxPath) {
        setenv("RT_XVEON_XTRANS_BACKEND", "onnxruntime-cpu", 1);
        const auto cpu = rtengine::neural::loadCachedXVeonXTransRunner(onnxPath);
        require(static_cast<bool>(cpu), "cannot load X-veon ONNX reference: " + cpu.error.message);
        std::vector<float> reference(candidate.size());
        require(!cpu.runner->run(input.data(), input.size(), reference.data(), reference.size()),
            "X-veon ONNX reference inference failed");
        std::vector<double> absolute;
        absolute.reserve(reference.size());
        double squared = 0.0;
        std::size_t tightFailures = 0;
        for (std::size_t i = 0; i < reference.size(); ++i) {
            const double difference = std::abs(static_cast<double>(reference[i]) - candidate[i]);
            absolute.push_back(difference);
            squared += difference * difference;
            if (difference > 5e-6 + 1e-5 * std::abs(static_cast<double>(reference[i]))) {
                ++tightFailures;
            }
        }
        std::sort(absolute.begin(), absolute.end());
        const double rms = std::sqrt(squared / absolute.size());
        const double p99 = absolute[static_cast<std::size_t>(std::ceil(0.99 * absolute.size())) - 1];
        std::printf("X-veon TVM parity: max=%.9g rms=%.9g p99=%.9g tight_failures=%zu\n",
            absolute.back(), rms, p99, tightFailures);
        require(absolute.back() <= 0.005 && rms <= 0.0005 && p99 <= 0.001,
            "X-veon TVM Vulkan output exceeds the reviewed aggregate tolerance");
    }
#endif
    return 0;
#endif
}

} // namespace xtrans_xveon_test
