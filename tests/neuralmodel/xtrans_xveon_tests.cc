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
    std::uint64_t workingBufferBytes() const override { return 1234; }

    Mode mode_;
    int calls = 0;
    int failAt = 2;
    std::vector<float> firstInput;
    std::string artifact = std::string(64, 'a');
    std::string runtime = "mock";
    std::string providerName = "mock";
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

#ifdef RT_TEST_WITH_ONNXRUNTIME
std::string temporaryPath(const char *suffix)
{
    return std::string(g_get_tmp_dir()) + "/rt-xveon-test-" + std::to_string(g_random_int()) + suffix;
}
#endif

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
#ifdef RT_TEST_WITH_ONNXRUNTIME
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
        require(std::fseek(file.get(), 15536134L - 1, SEEK_SET) == 0 && std::fputc(0, file.get()) == 0,
            "cannot size wrong-digest ONNX fixture");
    }
    auto wrongDigest = rtengine::neural::loadCachedXVeonXTransRunner(wrongDigestPath);
    g_remove(wrongDigestPath.c_str());
    require(!wrongDigest && wrongDigest.error.code == rtengine::neural::NeuralModelErrorCode::DIGEST,
        "same-size wrong-digest X-veon model did not return DIGEST");

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

    RtXveonOrtSession *malformedSession = nullptr;
    char malformedMessage[1024] = {};
    const unsigned char malformedModel = 0;
    const int malformed = rt_xveon_ort_create(
        &malformedModel, 1, &malformedSession, malformedMessage, sizeof(malformedMessage));
    rt_xveon_ort_release(malformedSession);
    require(malformed == RT_XVEON_ORT_RUNTIME && malformedMessage[0],
        "malformed ONNX did not produce a bounded runtime error");
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

} // namespace xtrans_xveon_test
