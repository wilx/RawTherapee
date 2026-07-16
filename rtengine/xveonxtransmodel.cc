#include "xveonxtransmodel.h"

#include "xveon_ort_bridge.h"

#include <cstdio>
#include <map>
#include <mutex>
#include <new>
#include <stdexcept>
#include <utility>
#include <vector>

#include <glib.h>
#include <glib/gstdio.h>
#include <glibmm/checksum.h>

namespace rtengine
{

namespace neural
{

namespace
{

constexpr std::size_t MODEL_BYTES = 15536134;
constexpr const char *MODEL_SHA256 = "45b1fa22b0027868fd5c20ec7b59234ed5aeb35de89fbc0950a4bec67f328500";
constexpr std::uint64_t MAX_MODEL_BYTES = 64u * 1024u * 1024u;

std::string sha256(const std::vector<unsigned char> &bytes)
{
    Glib::Checksum checksum(Glib::Checksum::CHECKSUM_SHA256);
    if (!checksum) throw std::runtime_error("SHA-256 is unavailable");
    checksum.update(bytes.data(), bytes.size());
    return checksum.get_string();
}

std::vector<unsigned char> readAuthenticatedModel(const std::string &path)
{
    std::unique_ptr<std::FILE, int (*)(std::FILE *)> file(g_fopen(path.c_str(), "rb"), std::fclose);
    if (!file) throw NeuralModelError(NeuralModelErrorCode::IO, "cannot open X-veon ONNX model");
    if (std::fseek(file.get(), 0, SEEK_END) != 0) throw NeuralModelError(NeuralModelErrorCode::IO, "cannot size X-veon ONNX model");
    const long size = std::ftell(file.get());
    if (size < 0 || std::fseek(file.get(), 0, SEEK_SET) != 0) throw NeuralModelError(NeuralModelErrorCode::IO, "cannot rewind X-veon ONNX model");
    if (static_cast<unsigned long long>(size) > MAX_MODEL_BYTES) throw NeuralModelError(NeuralModelErrorCode::LIMIT, "X-veon ONNX model exceeds 64 MiB");
    if (static_cast<std::size_t>(size) != MODEL_BYTES) throw NeuralModelError(NeuralModelErrorCode::SIZE, "X-veon ONNX model size differs");
    std::vector<unsigned char> bytes(MODEL_BYTES);
    if (std::fread(bytes.data(), 1, bytes.size(), file.get()) != bytes.size()) {
        throw NeuralModelError(std::ferror(file.get()) ? NeuralModelErrorCode::IO : NeuralModelErrorCode::SIZE,
            "X-veon ONNX model was truncated while reading");
    }
    if (std::fgetc(file.get()) != EOF) throw NeuralModelError(NeuralModelErrorCode::SIZE, "X-veon ONNX model changed while reading");
    if (sha256(bytes) != MODEL_SHA256) throw NeuralModelError(NeuralModelErrorCode::DIGEST, "X-veon ONNX model SHA-256 differs");
    return bytes;
}

NeuralModelError bridgeError(int code, const char *message)
{
    NeuralModelErrorCode mapped = NeuralModelErrorCode::RUNTIME;
    if (code == RT_XVEON_ORT_VERSION) mapped = NeuralModelErrorCode::VERSION;
    else if (code == RT_XVEON_ORT_SCHEMA) mapped = NeuralModelErrorCode::SCHEMA;
    else if (code == RT_XVEON_ORT_ALLOCATION) mapped = NeuralModelErrorCode::ALLOCATION;
    return NeuralModelError(mapped, message ? message : "unknown ONNX Runtime error");
}

class OrtXVeonRunner final : public XVeonXTransRunner
{
public:
    explicit OrtXVeonRunner(std::vector<unsigned char> bytes) :
        bytes_(std::move(bytes)), artifact_(MODEL_SHA256), runtime_(rt_xveon_ort_version()), provider_("CPUExecutionProvider")
    {
        char message[1024] = {};
        const int status = rt_xveon_ort_create(bytes_.data(), bytes_.size(), &session_, message, sizeof(message));
        if (status != RT_XVEON_ORT_OK) throw bridgeError(status, message);
    }

    ~OrtXVeonRunner() override
    {
        rt_xveon_ort_release(session_);
    }

    NeuralModelError run(const float *input, std::size_t inputCount, float *output, std::size_t outputCount) override
    {
        if (!input || !output) return NeuralModelError(NeuralModelErrorCode::RANGE, "X-veon input and output must be non-null");
        std::lock_guard<std::mutex> lock(runMutex_);
        char message[1024] = {};
        const int status = rt_xveon_ort_run(session_, input, inputCount, output, outputCount, message, sizeof(message));
        return status == RT_XVEON_ORT_OK ? NeuralModelError() : bridgeError(status, message);
    }

    const std::string &artifactSha256() const override { return artifact_; }
    const std::string &runtimeVersion() const override { return runtime_; }
    const std::string &provider() const override { return provider_; }
    std::uint64_t workingBufferBytes() const override
    {
        return static_cast<std::uint64_t>(bytes_.size() + (XVEON_INPUT_FLOATS + XVEON_OUTPUT_FLOATS) * sizeof(float));
    }

private:
    std::vector<unsigned char> bytes_;
    std::string artifact_;
    std::string runtime_;
    std::string provider_;
    RtXveonOrtSession *session_ = nullptr;
    std::mutex runMutex_;
};

std::mutex cacheMutex;
std::map<std::string, std::shared_ptr<XVeonXTransRunner>> cache;

std::string canonicalPath(const std::string &path)
{
    std::unique_ptr<char, decltype(&g_free)> canonical(g_canonicalize_filename(path.c_str(), nullptr), g_free);
    return canonical ? canonical.get() : path;
}

} // namespace

XVeonXTransLoadResult loadCachedXVeonXTransRunner(const Glib::ustring &path)
{
    if (path.empty()) return {nullptr, NeuralModelError(NeuralModelErrorCode::IO, "RT_XVEON_XTRANS_MODEL is unset or empty")};
    const std::string key = canonicalPath(path.raw());
    std::lock_guard<std::mutex> lock(cacheMutex);
    const auto found = cache.find(key);
    if (found != cache.end()) return {found->second, NeuralModelError()};
    try {
        std::shared_ptr<XVeonXTransRunner> runner(new OrtXVeonRunner(readAuthenticatedModel(key)));
        cache.emplace(key, runner);
        return {runner, NeuralModelError()};
    } catch (const NeuralModelError &error) {
        return {nullptr, error};
    } catch (const std::bad_alloc &) {
        return {nullptr, NeuralModelError(NeuralModelErrorCode::ALLOCATION, "cannot allocate X-veon model/session storage")};
    } catch (const std::exception &error) {
        return {nullptr, NeuralModelError(NeuralModelErrorCode::RUNTIME, error.what())};
    }
}

} // namespace neural

} // namespace rtengine
