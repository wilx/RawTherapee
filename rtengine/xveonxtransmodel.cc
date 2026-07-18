#include "xveonxtransmodel.h"

#ifdef RT_WITH_ONNXRUNTIME
#include "xveon_ort_bridge.h"
#endif
#ifdef RT_WITH_MIGRAPHX
#include "xveon_migraphx_bridge.h"
#endif
#ifdef RT_WITH_TVM_VULKAN
#include "tvm_vulkan_bridge.h"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <new>
#include <stdexcept>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <glib.h>
#include <glib/gstdio.h>
#include <glibmm/checksum.h>

namespace rtengine
{

namespace neural
{

namespace
{

constexpr const char *MODEL_SHA256 = "45b1fa22b0027868fd5c20ec7b59234ed5aeb35de89fbc0950a4bec67f328500";
#ifdef RT_WITH_TVM_VULKAN
constexpr std::size_t TVM_MODULE_BYTES = 32594896;
constexpr const char *TVM_MODULE_SHA256 = "8648e3741a98345c8bc76b9e1c853a3b4ef58155b65ed726f9c2fda0226c206d";
#endif
#if defined(RT_WITH_ONNXRUNTIME) || defined(RT_WITH_MIGRAPHX)
constexpr std::size_t MODEL_BYTES = 15536134;
constexpr std::uint64_t MAX_MODEL_BYTES = 64u * 1024u * 1024u;
#endif

#if defined(RT_WITH_ONNXRUNTIME) || defined(RT_WITH_MIGRAPHX) || defined(RT_WITH_TVM_VULKAN)
std::string sha256(const std::vector<unsigned char> &bytes)
{
    Glib::Checksum checksum(Glib::Checksum::CHECKSUM_SHA256);
    if (!checksum) throw std::runtime_error("SHA-256 is unavailable");
    checksum.update(bytes.data(), bytes.size());
    return checksum.get_string();
}
#endif

#if defined(RT_WITH_ONNXRUNTIME) || defined(RT_WITH_MIGRAPHX)
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
#endif

#ifdef RT_WITH_TVM_VULKAN
std::vector<unsigned char> readAuthenticatedTvmModule(const std::string &path)
{
    struct stat pathStatus = {};
    if (lstat(path.c_str(), &pathStatus) != 0) {
        throw NeuralModelError(NeuralModelErrorCode::IO, "cannot stat X-veon TVM module");
    }
    if (!S_ISREG(pathStatus.st_mode) || S_ISLNK(pathStatus.st_mode)) {
        throw NeuralModelError(NeuralModelErrorCode::IO, "X-veon TVM module must be a regular non-symlink file");
    }
    const int descriptor = open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    if (descriptor < 0) {
        throw NeuralModelError(NeuralModelErrorCode::IO, "cannot open X-veon TVM module");
    }
    std::unique_ptr<int, void (*)(int *)> owner(new int(descriptor), [](int *value) {
        close(*value);
        delete value;
    });
    struct stat status = {};
    if (fstat(descriptor, &status) != 0 || !S_ISREG(status.st_mode)) {
        throw NeuralModelError(NeuralModelErrorCode::IO, "cannot validate X-veon TVM module");
    }
    if (status.st_dev != pathStatus.st_dev || status.st_ino != pathStatus.st_ino) {
        throw NeuralModelError(NeuralModelErrorCode::IO, "X-veon TVM module changed before opening");
    }
    if (status.st_size < 0 || static_cast<std::size_t>(status.st_size) != TVM_MODULE_BYTES) {
        throw NeuralModelError(NeuralModelErrorCode::SIZE, "X-veon TVM module size differs");
    }
    std::vector<unsigned char> bytes(TVM_MODULE_BYTES);
    std::size_t offset = 0;
    while (offset < bytes.size()) {
        const ssize_t count = read(descriptor, bytes.data() + offset, bytes.size() - offset);
        if (count < 0 && errno == EINTR) {
            continue;
        }
        if (count <= 0) {
            throw NeuralModelError(NeuralModelErrorCode::SIZE, "X-veon TVM module was truncated while reading");
        }
        offset += static_cast<std::size_t>(count);
    }
    unsigned char extra = 0;
    ssize_t extraCount = 0;
    do {
        extraCount = read(descriptor, &extra, 1);
    } while (extraCount < 0 && errno == EINTR);
    if (extraCount < 0) {
        throw NeuralModelError(NeuralModelErrorCode::IO, "cannot finish reading X-veon TVM module");
    }
    if (extraCount != 0) {
        throw NeuralModelError(NeuralModelErrorCode::SIZE, "X-veon TVM module changed while reading");
    }
    if (sha256(bytes) != TVM_MODULE_SHA256) {
        throw NeuralModelError(NeuralModelErrorCode::DIGEST, "X-veon TVM module SHA-256 differs");
    }
    return bytes;
}
#endif

#ifdef RT_WITH_ONNXRUNTIME
NeuralModelError ortError(int code, const char *message)
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
        bytes_(std::move(bytes)), artifact_(MODEL_SHA256), runtime_(rt_xveon_ort_version()),
        provider_("CPUExecutionProvider"), compileSource_("session-create")
    {
        const auto started = std::chrono::steady_clock::now();
        char message[1024] = {};
        const int status = rt_xveon_ort_create(bytes_.data(), bytes_.size(), &session_, message, sizeof(message));
        compilationUs_ = elapsedMicroseconds(started);
        if (status != RT_XVEON_ORT_OK) throw ortError(status, message);
    }

    ~OrtXVeonRunner() override { rt_xveon_ort_release(session_); }

    NeuralModelError run(const float *input, std::size_t inputCount, float *output, std::size_t outputCount) override
    {
        if (!input || !output) return NeuralModelError(NeuralModelErrorCode::RANGE, "X-veon input and output must be non-null");
        std::lock_guard<std::mutex> lock(runMutex_);
        char message[1024] = {};
        const auto started = std::chrono::steady_clock::now();
        const int status = rt_xveon_ort_run(session_, input, inputCount, output, outputCount, message, sizeof(message));
        inferenceUs_ = elapsedMicroseconds(started);
        return status == RT_XVEON_ORT_OK ? NeuralModelError() : ortError(status, message);
    }

    const std::string &artifactSha256() const override { return artifact_; }
    const std::string &runtimeVersion() const override { return runtime_; }
    const std::string &provider() const override { return provider_; }
    const std::string &compileSource() const override { return compileSource_; }
    std::uint64_t compilationMicroseconds() const override { return compilationUs_; }
    std::uint64_t lastInferenceMicroseconds() const override { return inferenceUs_; }
    std::uint64_t workingBufferBytes() const override
    {
        return static_cast<std::uint64_t>(bytes_.size() + (XVEON_INPUT_FLOATS + XVEON_OUTPUT_FLOATS) * sizeof(float));
    }

private:
    static std::uint64_t elapsedMicroseconds(std::chrono::steady_clock::time_point started)
    {
        return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - started).count());
    }
    std::vector<unsigned char> bytes_;
    std::string artifact_;
    std::string runtime_;
    std::string provider_;
    std::string compileSource_;
    RtXveonOrtSession *session_ = nullptr;
    std::mutex runMutex_;
    std::uint64_t compilationUs_ = 0;
    std::uint64_t inferenceUs_ = 0;
};
#endif

#ifdef RT_WITH_TVM_VULKAN
NeuralModelError tvmError(int code, const char *message)
{
    NeuralModelErrorCode mapped = NeuralModelErrorCode::RUNTIME;
    if (code == RT_TVM_VULKAN_SCHEMA) mapped = NeuralModelErrorCode::SCHEMA;
    else if (code == RT_TVM_VULKAN_ALLOCATION) mapped = NeuralModelErrorCode::ALLOCATION;
    else if (code == RT_TVM_VULKAN_IO) mapped = NeuralModelErrorCode::IO;
    else if (code == RT_TVM_VULKAN_NONFINITE) mapped = NeuralModelErrorCode::NONFINITE;
    else if (code == RT_TVM_VULKAN_VERSION) mapped = NeuralModelErrorCode::VERSION;
    return NeuralModelError(mapped, message ? message : "unknown TVM Vulkan error");
}

class TvmXVeonRunner final : public XVeonXTransRunner
{
public:
    explicit TvmXVeonRunner(std::vector<unsigned char> bytes) :
        bytes_(std::move(bytes)), artifact_(TVM_MODULE_SHA256), runtime_(rt_tvm_vulkan_version()),
        provider_("TVM-Vulkan"), compileSource_("ahead-of-time")
    {
        static const int64_t inputShape[] = {1, 4, 288, 288};
        static const int64_t outputShape[] = {1, 3, 288, 288};
        const RtTvmVulkanContract contract = {
            inputShape, 4, outputShape, 4, XVEON_INPUT_FLOATS, XVEON_OUTPUT_FLOATS
        };
        char message[1024] = {};
        const auto started = std::chrono::steady_clock::now();
        const int status = rt_tvm_vulkan_create(
            bytes_.data(), bytes_.size(), &contract, &session_, message, sizeof(message));
        compilationUs_ = elapsedMicroseconds(started);
        if (status != RT_TVM_VULKAN_OK) throw tvmError(status, message);
        provider_ = rt_tvm_vulkan_device(session_);
    }

    ~TvmXVeonRunner() override { rt_tvm_vulkan_release(session_); }

    NeuralModelError run(
        const float *input, std::size_t inputCount, float *output, std::size_t outputCount) override
    {
        std::lock_guard<std::mutex> lock(runMutex_);
        char message[1024] = {};
        const auto started = std::chrono::steady_clock::now();
        const int status = rt_tvm_vulkan_run(
            session_, input, inputCount, output, outputCount, message, sizeof(message));
        inferenceUs_ = elapsedMicroseconds(started);
        return status == RT_TVM_VULKAN_OK ? NeuralModelError() : tvmError(status, message);
    }

    const std::string &artifactSha256() const override { return artifact_; }
    const std::string &runtimeVersion() const override { return runtime_; }
    const std::string &provider() const override { return provider_; }
    const std::string &compileSource() const override { return compileSource_; }
    std::uint64_t compilationMicroseconds() const override { return compilationUs_; }
    std::uint64_t lastInferenceMicroseconds() const override { return inferenceUs_; }
    std::uint64_t workingBufferBytes() const override { return rt_tvm_vulkan_working_bytes(session_); }

private:
    static std::uint64_t elapsedMicroseconds(std::chrono::steady_clock::time_point started)
    {
        return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - started).count());
    }
    std::vector<unsigned char> bytes_;
    std::string artifact_;
    std::string runtime_;
    std::string provider_;
    std::string compileSource_;
    RtTvmVulkanSession *session_ = nullptr;
    std::mutex runMutex_;
    std::uint64_t compilationUs_ = 0;
    std::uint64_t inferenceUs_ = 0;
};
#endif

#ifdef RT_WITH_MIGRAPHX
NeuralModelError migraphxError(int code, const char *message)
{
    NeuralModelErrorCode mapped = NeuralModelErrorCode::RUNTIME;
    if (code == RT_XVEON_MIGRAPHX_VERSION) mapped = NeuralModelErrorCode::VERSION;
    else if (code == RT_XVEON_MIGRAPHX_SCHEMA) mapped = NeuralModelErrorCode::SCHEMA;
    else if (code == RT_XVEON_MIGRAPHX_ALLOCATION) mapped = NeuralModelErrorCode::ALLOCATION;
    else if (code == RT_XVEON_MIGRAPHX_IO) mapped = NeuralModelErrorCode::IO;
    else if (code == RT_XVEON_MIGRAPHX_NONFINITE) mapped = NeuralModelErrorCode::NONFINITE;
    return NeuralModelError(mapped, message ? message : "unknown MIGraphX error");
}

struct SecureCacheFile final {
    int descriptor = -1;
    std::vector<unsigned char> bytes;

    SecureCacheFile() = default;
    SecureCacheFile(const SecureCacheFile &) = delete;
    SecureCacheFile &operator=(const SecureCacheFile &) = delete;
    ~SecureCacheFile()
    {
        if (descriptor >= 0) close(descriptor);
    }

    std::string descriptorPath() const
    {
        return "/proc/self/fd/" + std::to_string(descriptor);
    }
};

bool secureDirectory(const std::string &path)
{
    struct stat status = {};
    if (g_mkdir_with_parents(path.c_str(), 0700) != 0 || lstat(path.c_str(), &status) != 0) return false;
    if (!S_ISDIR(status.st_mode) || S_ISLNK(status.st_mode) || status.st_uid != getuid() ||
        (status.st_mode & (S_IWGRP | S_IWOTH)) != 0) return false;
    return chmod(path.c_str(), 0700) == 0;
}

bool readSecureCacheFile(const std::string &path, std::size_t limit, SecureCacheFile &result)
{
    result.descriptor = open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    if (result.descriptor < 0) return false;
    struct stat status = {};
    if (fstat(result.descriptor, &status) != 0 || !S_ISREG(status.st_mode) || status.st_uid != getuid() ||
        (status.st_mode & (S_IWGRP | S_IWOTH)) != 0 || status.st_size < 0 ||
        static_cast<unsigned long long>(status.st_size) > limit) return false;
    result.bytes.resize(static_cast<std::size_t>(status.st_size));
    std::size_t offset = 0;
    while (offset < result.bytes.size()) {
        const ssize_t count = read(result.descriptor, result.bytes.data() + offset, result.bytes.size() - offset);
        if (count <= 0) return false;
        offset += static_cast<std::size_t>(count);
    }
    return lseek(result.descriptor, 0, SEEK_SET) == 0;
}

std::string cacheIdentity(bool fastMath, std::size_t payloadBytes, const std::string &payloadSha)
{
    return std::string("{\n") +
        "  \"artifact_sha256\": \"" + MODEL_SHA256 + "\",\n" +
        "  \"cache_format_revision\": 1,\n" +
        "  \"exhaustive_tune\": false,\n" +
        "  \"fast_math\": " + (fastMath ? "true" : "false") + ",\n" +
        "  \"gpu_isa\": \"gfx1101\",\n" +
        "  \"migraphx_version\": \"" + rt_xveon_migraphx_version() + "\",\n" +
        "  \"offload_copy\": true,\n" +
        "  \"payload_bytes\": " + std::to_string(payloadBytes) + ",\n" +
        "  \"payload_sha256\": \"" + payloadSha + "\"\n" +
        "}\n";
}

bool writeFile(const std::string &path, const std::string &data)
{
    const int descriptor = open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0600);
    if (descriptor < 0) return false;
    std::size_t offset = 0;
    while (offset < data.size()) {
        const ssize_t count = write(descriptor, data.data() + offset, data.size() - offset);
        if (count <= 0) {
            close(descriptor);
            return false;
        }
        offset += static_cast<std::size_t>(count);
    }
    const bool ok = fsync(descriptor) == 0 && close(descriptor) == 0;
    return ok;
}

struct CachePaths final {
    bool enabled = false;
    std::string program;
    std::string manifest;

    CachePaths() = default;
    CachePaths(bool enabled, std::string program, std::string manifest) :
        enabled(enabled), program(std::move(program)), manifest(std::move(manifest))
    {
    }
};

CachePaths cachePaths(bool fastMath)
{
    const char *directory = std::getenv("RT_XVEON_MIGRAPHX_CACHE_DIR");
    if (!directory || !*directory || !secureDirectory(directory)) return {};
    const std::string stem = std::string(directory) + "/xveon-gfx1101-" + (fastMath ? "fast" : "strict") + "-v1";
    return {true, stem + ".mxr", stem + ".json"};
}

bool authenticatedCache(const CachePaths &paths, bool fastMath, SecureCacheFile &program)
{
    if (!paths.enabled) return false;
    SecureCacheFile manifest;
    if (!readSecureCacheFile(paths.manifest, 4096, manifest) ||
        !readSecureCacheFile(paths.program, 256u * 1024u * 1024u, program)) return false;
    const std::string payloadSha = sha256(program.bytes);
    const std::string expected = cacheIdentity(fastMath, program.bytes.size(), payloadSha);
    return std::string(manifest.bytes.begin(), manifest.bytes.end()) == expected;
}

void publishCache(
    const CachePaths &paths,
    bool fastMath,
    RtXveonMigraphxSession *session)
{
    if (!paths.enabled) return;
    const std::string suffix = ".tmp-" + std::to_string(static_cast<unsigned long long>(getpid()));
    const std::string programTemporary = paths.program + suffix;
    const std::string manifestTemporary = paths.manifest + suffix;
    g_remove(programTemporary.c_str());
    g_remove(manifestTemporary.c_str());
    char message[1024] = {};
    if (rt_xveon_migraphx_save(session, programTemporary.c_str(), message, sizeof(message)) != RT_XVEON_MIGRAPHX_OK ||
        chmod(programTemporary.c_str(), 0600) != 0) {
        g_remove(programTemporary.c_str());
        return;
    }
    SecureCacheFile saved;
    if (!readSecureCacheFile(programTemporary, 256u * 1024u * 1024u, saved) ||
        fsync(saved.descriptor) != 0) {
        g_remove(programTemporary.c_str());
        return;
    }
    const std::string manifest = cacheIdentity(fastMath, saved.bytes.size(), sha256(saved.bytes));
    if (!writeFile(manifestTemporary, manifest)) {
        g_remove(programTemporary.c_str());
        g_remove(manifestTemporary.c_str());
        return;
    }
    // The compiled program is the completion marker.  A crash before its
    // rename leaves a harmless manifest which cannot authenticate a program.
    if (g_rename(manifestTemporary.c_str(), paths.manifest.c_str()) != 0 ||
        g_rename(programTemporary.c_str(), paths.program.c_str()) != 0) {
        g_remove(programTemporary.c_str());
        g_remove(manifestTemporary.c_str());
    }
}

class MigraphxXVeonRunner final : public XVeonXTransRunner
{
public:
    MigraphxXVeonRunner(std::vector<unsigned char> bytes, bool fastMath) :
        bytes_(std::move(bytes)), artifact_(MODEL_SHA256), runtime_(rt_xveon_migraphx_version()),
        provider_(fastMath ? "MIGraphX-gpu-fast-math" : "MIGraphX-gpu"), compileSource_("fresh"), fastMath_(fastMath)
    {
        if (access("/dev/kfd", R_OK | W_OK) != 0) {
            throw NeuralModelError(
                NeuralModelErrorCode::RUNTIME,
                "MIGraphX GPU access requires readable and writable /dev/kfd (restart the session after joining render)");
        }
        const auto started = std::chrono::steady_clock::now();
        char message[1024] = {};
        const CachePaths paths = cachePaths(fastMath_);
        SecureCacheFile cached;
        int status = RT_XVEON_MIGRAPHX_RUNTIME;
        if (authenticatedCache(paths, fastMath_, cached)) {
            const std::string descriptor = cached.descriptorPath();
            status = rt_xveon_migraphx_create(
                bytes_.data(), bytes_.size(), fastMath_ ? 1 : 0, descriptor.c_str(),
                &session_, message, sizeof(message));
            if (status == RT_XVEON_MIGRAPHX_OK) {
                std::vector<float> input(XVEON_INPUT_FLOATS, 0.f);
                std::vector<float> output(XVEON_OUTPUT_FLOATS);
                status = rt_xveon_migraphx_run(
                    session_, input.data(), input.size(), output.data(), output.size(), message, sizeof(message));
                const bool finite = status == RT_XVEON_MIGRAPHX_OK &&
                    std::all_of(output.begin(), output.end(), [](float value) { return std::isfinite(value); });
                if (finite) {
                    compileSource_ = "cache";
                } else {
                    rt_xveon_migraphx_release(session_);
                    session_ = nullptr;
                }
            }
        }
        if (!session_) {
            status = rt_xveon_migraphx_create(
                bytes_.data(), bytes_.size(), fastMath_ ? 1 : 0, nullptr, &session_, message, sizeof(message));
            if (status == RT_XVEON_MIGRAPHX_OK) {
                compileSource_ = "fresh";
                publishCache(paths, fastMath_, session_);
            }
        }
        compilationUs_ = elapsedMicroseconds(started);
        if (status != RT_XVEON_MIGRAPHX_OK) throw migraphxError(status, message);
    }

    ~MigraphxXVeonRunner() override { rt_xveon_migraphx_release(session_); }

    NeuralModelError run(const float *input, std::size_t inputCount, float *output, std::size_t outputCount) override
    {
        if (!input || !output) return NeuralModelError(NeuralModelErrorCode::RANGE, "X-veon input and output must be non-null");
        std::lock_guard<std::mutex> lock(runMutex_);
        char message[1024] = {};
        const auto started = std::chrono::steady_clock::now();
        const int status = rt_xveon_migraphx_run(session_, input, inputCount, output, outputCount, message, sizeof(message));
        inferenceUs_ = elapsedMicroseconds(started);
        if (status != RT_XVEON_MIGRAPHX_OK) return migraphxError(status, message);
        for (std::size_t i = 0; i < outputCount; ++i) {
            if (!std::isfinite(output[i])) return NeuralModelError(NeuralModelErrorCode::NONFINITE, "MIGraphX output contains NaN or infinity");
        }
        return NeuralModelError();
    }

    const std::string &artifactSha256() const override { return artifact_; }
    const std::string &runtimeVersion() const override { return runtime_; }
    const std::string &provider() const override { return provider_; }
    const std::string &compileSource() const override { return compileSource_; }
    std::uint64_t compilationMicroseconds() const override { return compilationUs_; }
    std::uint64_t lastInferenceMicroseconds() const override { return inferenceUs_; }
    std::uint64_t workingBufferBytes() const override
    {
        return static_cast<std::uint64_t>(bytes_.size() + (XVEON_INPUT_FLOATS + XVEON_OUTPUT_FLOATS) * sizeof(float));
    }

private:
    static std::uint64_t elapsedMicroseconds(std::chrono::steady_clock::time_point started)
    {
        return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - started).count());
    }
    std::vector<unsigned char> bytes_;
    std::string artifact_;
    std::string runtime_;
    std::string provider_;
    std::string compileSource_;
    bool fastMath_ = false;
    RtXveonMigraphxSession *session_ = nullptr;
    std::mutex runMutex_;
    std::uint64_t compilationUs_ = 0;
    std::uint64_t inferenceUs_ = 0;
};
#endif

// Successful sessions deliberately live until process termination.  In
// addition to matching the cache contract, avoiding static destruction is
// required by ROCm 7.2: destroying a compiled GPU program after the HIP
// runtime's own exit handlers have run can call hipFree on a torn-down device.
std::mutex &runnerCacheMutex()
{
    static std::mutex *mutex = new std::mutex;
    return *mutex;
}

std::map<std::string, std::shared_ptr<XVeonXTransRunner>> &runnerCache()
{
    static auto *cache = new std::map<std::string, std::shared_ptr<XVeonXTransRunner>>;
    return *cache;
}

std::string canonicalPath(const std::string &path)
{
    std::unique_ptr<char, decltype(&g_free)> canonical(g_canonicalize_filename(path.c_str(), nullptr), g_free);
    return canonical ? canonical.get() : path;
}

std::string selectedBackend()
{
    const char *requested = std::getenv("RT_XVEON_XTRANS_BACKEND");
    if (requested && *requested) return requested;
#ifdef RT_WITH_ONNXRUNTIME
    return "onnxruntime-cpu";
#elif defined(RT_WITH_MIGRAPHX)
    return "migraphx";
#elif defined(RT_WITH_TVM_VULKAN)
    return "tvm-vulkan";
#else
    return std::string();
#endif
}

bool fastMathEnabled()
{
    const char *value = std::getenv("RT_XVEON_MIGRAPHX_FAST_MATH");
    return value && std::string(value) == "1";
}

} // namespace

XVeonXTransLoadResult loadCachedXVeonXTransRunner(const Glib::ustring &path)
{
    if (path.empty()) return {nullptr, NeuralModelError(NeuralModelErrorCode::IO, "X-veon model or TVM module path is unset or empty")};
    const std::string backend = selectedBackend();
    if (backend != "onnxruntime-cpu" && backend != "migraphx" && backend != "tvm-vulkan") {
        return {nullptr, NeuralModelError(NeuralModelErrorCode::ENUM, "RT_XVEON_XTRANS_BACKEND must be onnxruntime-cpu, migraphx, or tvm-vulkan")};
    }
#ifndef RT_WITH_ONNXRUNTIME
    if (backend == "onnxruntime-cpu") {
        return {nullptr, NeuralModelError(NeuralModelErrorCode::RUNTIME, "RawTherapee was built without WITH_ONNXRUNTIME")};
    }
#endif
#ifndef RT_WITH_TVM_VULKAN
    if (backend == "tvm-vulkan") {
        return {nullptr, NeuralModelError(NeuralModelErrorCode::RUNTIME, "RawTherapee was built without WITH_TVM_VULKAN")};
    }
#endif
#ifndef RT_WITH_MIGRAPHX
    if (backend == "migraphx") {
        return {nullptr, NeuralModelError(NeuralModelErrorCode::RUNTIME, "RawTherapee was built without WITH_MIGRAPHX")};
    }
#endif
    const std::string modelPath = canonicalPath(path.raw());
    const bool fastMath = backend == "migraphx" && fastMathEnabled();
    std::string runtimeIdentity;
    std::string artifactIdentity = MODEL_SHA256;
#ifdef RT_WITH_ONNXRUNTIME
    if (backend == "onnxruntime-cpu") runtimeIdentity = rt_xveon_ort_version();
#endif
#ifdef RT_WITH_MIGRAPHX
    if (backend == "migraphx") runtimeIdentity = rt_xveon_migraphx_version() + std::string("\ngfx1101");
#endif
#ifdef RT_WITH_TVM_VULKAN
    if (backend == "tvm-vulkan") {
        runtimeIdentity = rt_tvm_vulkan_version();
        artifactIdentity = TVM_MODULE_SHA256;
    }
#endif
    const std::string key = modelPath + "\n" + artifactIdentity + "\n" + backend + "\n" +
        runtimeIdentity + (fastMath ? "\nfast" : "\nstrict");
    std::lock_guard<std::mutex> lock(runnerCacheMutex());
    auto &cache = runnerCache();
    const auto found = cache.find(key);
    if (found != cache.end()) return {found->second, NeuralModelError()};
    try {
        std::shared_ptr<XVeonXTransRunner> runner;
#ifdef RT_WITH_ONNXRUNTIME
        if (backend == "onnxruntime-cpu") runner.reset(new OrtXVeonRunner(readAuthenticatedModel(modelPath)));
#endif
#ifdef RT_WITH_MIGRAPHX
        if (backend == "migraphx") runner.reset(new MigraphxXVeonRunner(readAuthenticatedModel(modelPath), fastMath));
#endif
#ifdef RT_WITH_TVM_VULKAN
        if (backend == "tvm-vulkan") runner.reset(new TvmXVeonRunner(readAuthenticatedTvmModule(modelPath)));
#endif
        if (!runner) return {nullptr, NeuralModelError(NeuralModelErrorCode::RUNTIME, "selected X-veon backend is unavailable")};
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
