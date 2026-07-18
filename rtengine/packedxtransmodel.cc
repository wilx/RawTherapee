#include "packedxtransmodel.h"

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
#include <cerrno>
#include <cmath>
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

#include <glib/gstdio.h>
#include <glibmm/checksum.h>

namespace rtengine
{
namespace neural
{
namespace
{
#ifdef RT_WITH_TVM_VULKAN
constexpr std::size_t TVM_MODULE_BYTES = 3097488;
constexpr const char *TVM_MODULE_SHA256 = "2975665b5f36ffb29e9b0c9dec62f69ffc916605189f41ae11484e19d18cfc6e";
#endif
#if defined(RT_WITH_ONNXRUNTIME) || defined(RT_WITH_MIGRAPHX)
constexpr std::size_t MODEL_BYTES = 1673648;
constexpr std::uint64_t MAX_MODEL_BYTES = 64u * 1024u * 1024u;
#endif
#if defined(RT_WITH_ONNXRUNTIME) || defined(RT_WITH_MIGRAPHX)
const RtFixedOnnxContract CONTRACT = {
    "PackedXTransNet", "input", "output", {1, 1, 288, 288}, {1, 3, 288, 288},
    PACKED_XTRANS_INPUT_FLOATS, PACKED_XTRANS_OUTPUT_FLOATS, 0
};
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
std::vector<unsigned char> readModel(const std::string &path)
{
    std::unique_ptr<std::FILE, int (*)(std::FILE *)> file(g_fopen(path.c_str(), "rb"), std::fclose);
    if (!file) throw NeuralModelError(NeuralModelErrorCode::IO, "cannot open PackedXTransNet ONNX model");
    if (std::fseek(file.get(), 0, SEEK_END) != 0) throw NeuralModelError(NeuralModelErrorCode::IO, "cannot size PackedXTransNet ONNX model");
    const long size = std::ftell(file.get());
    if (size < 0 || std::fseek(file.get(), 0, SEEK_SET) != 0) throw NeuralModelError(NeuralModelErrorCode::IO, "cannot rewind PackedXTransNet ONNX model");
    if (static_cast<unsigned long long>(size) > MAX_MODEL_BYTES) throw NeuralModelError(NeuralModelErrorCode::LIMIT, "PackedXTransNet ONNX model exceeds 64 MiB");
    if (static_cast<std::size_t>(size) != MODEL_BYTES) throw NeuralModelError(NeuralModelErrorCode::SIZE, "PackedXTransNet ONNX model size differs");
    std::vector<unsigned char> bytes(MODEL_BYTES);
    if (std::fread(bytes.data(), 1, bytes.size(), file.get()) != bytes.size()) {
        throw NeuralModelError(std::ferror(file.get()) ? NeuralModelErrorCode::IO : NeuralModelErrorCode::SIZE,
            "PackedXTransNet ONNX model was truncated while reading");
    }
    if (std::fgetc(file.get()) != EOF) throw NeuralModelError(NeuralModelErrorCode::SIZE, "PackedXTransNet ONNX model changed while reading");
    if (sha256(bytes) != PACKED_XTRANS_ONNX_SHA256) throw NeuralModelError(NeuralModelErrorCode::DIGEST, "PackedXTransNet ONNX model SHA-256 differs");
    return bytes;
}
#endif

#ifdef RT_WITH_TVM_VULKAN
std::vector<unsigned char> readTvmModule(const std::string &path)
{
    struct stat pathStatus = {};
    if (lstat(path.c_str(), &pathStatus) != 0) {
        throw NeuralModelError(NeuralModelErrorCode::IO, "cannot stat PackedXTransNet TVM module");
    }
    if (!S_ISREG(pathStatus.st_mode) || S_ISLNK(pathStatus.st_mode)) {
        throw NeuralModelError(
            NeuralModelErrorCode::IO, "PackedXTransNet TVM module must be a regular non-symlink file");
    }
    const int descriptor = open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    if (descriptor < 0) {
        throw NeuralModelError(NeuralModelErrorCode::IO, "cannot open PackedXTransNet TVM module");
    }
    std::unique_ptr<int, void (*)(int *)> owner(new int(descriptor), [](int *value) {
        close(*value);
        delete value;
    });
    struct stat status = {};
    if (fstat(descriptor, &status) != 0 || !S_ISREG(status.st_mode)) {
        throw NeuralModelError(NeuralModelErrorCode::IO, "cannot validate PackedXTransNet TVM module");
    }
    if (status.st_dev != pathStatus.st_dev || status.st_ino != pathStatus.st_ino) {
        throw NeuralModelError(NeuralModelErrorCode::IO, "PackedXTransNet TVM module changed before opening");
    }
    if (status.st_size < 0 || static_cast<std::size_t>(status.st_size) != TVM_MODULE_BYTES) {
        throw NeuralModelError(NeuralModelErrorCode::SIZE, "PackedXTransNet TVM module size differs");
    }
    std::vector<unsigned char> bytes(TVM_MODULE_BYTES);
    std::size_t offset = 0;
    while (offset < bytes.size()) {
        const ssize_t count = read(descriptor, bytes.data() + offset, bytes.size() - offset);
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) {
            throw NeuralModelError(
                NeuralModelErrorCode::SIZE, "PackedXTransNet TVM module was truncated while reading");
        }
        offset += static_cast<std::size_t>(count);
    }
    unsigned char extra = 0;
    ssize_t extraCount = 0;
    do {
        extraCount = read(descriptor, &extra, 1);
    } while (extraCount < 0 && errno == EINTR);
    if (extraCount < 0) {
        throw NeuralModelError(NeuralModelErrorCode::IO, "cannot finish reading PackedXTransNet TVM module");
    }
    if (extraCount != 0) {
        throw NeuralModelError(NeuralModelErrorCode::SIZE, "PackedXTransNet TVM module changed while reading");
    }
    if (sha256(bytes) != TVM_MODULE_SHA256) {
        throw NeuralModelError(NeuralModelErrorCode::DIGEST, "PackedXTransNet TVM module SHA-256 differs");
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
    return NeuralModelError(mapped, message ? message : "unknown neural runtime error");
}
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
#endif

class TimedRunner : public PackedXTransRunner
{
public:
    const std::string &artifactSha256() const override { return artifact_; }
    const std::string &runtimeVersion() const override { return runtime_; }
    const std::string &provider() const override { return provider_; }
    const std::string &precision() const override { return precision_; }
    const std::string &compileSource() const override { return compileSource_; }
    std::uint64_t compilationMicroseconds() const override { return compilationUs_; }
    std::uint64_t lastInferenceMicroseconds() const override { return inferenceUs_; }
protected:
    static std::uint64_t elapsed(std::chrono::steady_clock::time_point start)
    {
        return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - start).count());
    }
    std::string artifact_ = PACKED_XTRANS_ONNX_SHA256;
    std::string runtime_;
    std::string provider_;
    std::string precision_ = "fp32";
    std::string compileSource_ = "session-create";
    std::uint64_t compilationUs_ = 0;
    std::uint64_t inferenceUs_ = 0;
    std::mutex mutex_;
};

#ifdef RT_WITH_TVM_VULKAN
class TvmRunner final : public TimedRunner
{
public:
    explicit TvmRunner(std::vector<unsigned char> bytes) : bytes_(std::move(bytes))
    {
        artifact_ = TVM_MODULE_SHA256;
        runtime_ = rt_tvm_vulkan_version();
        provider_ = "TVM-Vulkan";
        precision_ = "fp32";
        compileSource_ = "ahead-of-time";
        static const int64_t inputShape[] = {1, 1, 288, 288};
        static const int64_t outputShape[] = {1, 3, 288, 288};
        const RtTvmVulkanContract contract = {
            inputShape, 4, outputShape, 4, PACKED_XTRANS_INPUT_FLOATS, PACKED_XTRANS_OUTPUT_FLOATS
        };
        char message[1024] = {};
        const auto start = std::chrono::steady_clock::now();
        const int status = rt_tvm_vulkan_create(
            bytes_.data(), bytes_.size(), &contract, &session_, message, sizeof(message));
        compilationUs_ = elapsed(start);
        if (status != RT_TVM_VULKAN_OK) throw tvmError(status, message);
        provider_ = rt_tvm_vulkan_device(session_);
    }

    ~TvmRunner() override { rt_tvm_vulkan_release(session_); }

    NeuralModelError run(
        const float *input, std::size_t inputs, float *output, std::size_t outputs) override
    {
        std::lock_guard<std::mutex> lock(mutex_);
        char message[1024] = {};
        const auto start = std::chrono::steady_clock::now();
        const int status = rt_tvm_vulkan_run(
            session_, input, inputs, output, outputs, message, sizeof(message));
        inferenceUs_ = elapsed(start);
        return status == RT_TVM_VULKAN_OK ? NeuralModelError() : tvmError(status, message);
    }

    std::uint64_t workingBufferBytes() const override { return rt_tvm_vulkan_working_bytes(session_); }

private:
    std::vector<unsigned char> bytes_;
    RtTvmVulkanSession *session_ = nullptr;
};
#endif

#ifdef RT_WITH_ONNXRUNTIME
class OrtRunner final : public TimedRunner
{
public:
    explicit OrtRunner(std::vector<unsigned char> bytes) : bytes_(std::move(bytes))
    {
        runtime_ = rt_xveon_ort_version();
        provider_ = "CPUExecutionProvider";
        char message[1024] = {};
        const auto start = std::chrono::steady_clock::now();
        const int status = rt_xveon_ort_create_contract(bytes_.data(), bytes_.size(), &CONTRACT,
            &session_, message, sizeof(message));
        compilationUs_ = elapsed(start);
        if (status != RT_XVEON_ORT_OK) throw ortError(status, message);
    }
    ~OrtRunner() override { rt_xveon_ort_release(session_); }
    NeuralModelError run(const float *input, std::size_t inputs, float *output, std::size_t outputs) override
    {
        std::lock_guard<std::mutex> lock(mutex_);
        char message[1024] = {};
        const auto start = std::chrono::steady_clock::now();
        const int status = rt_xveon_ort_run(session_, input, inputs, output, outputs, message, sizeof(message));
        inferenceUs_ = elapsed(start);
        return status == RT_XVEON_ORT_OK ? NeuralModelError() : ortError(status, message);
    }
    std::uint64_t workingBufferBytes() const override { return bytes_.size() + (PACKED_XTRANS_INPUT_FLOATS + PACKED_XTRANS_OUTPUT_FLOATS) * sizeof(float); }
private:
    std::vector<unsigned char> bytes_;
    RtXveonOrtSession *session_ = nullptr;
};
#endif

#ifdef RT_WITH_MIGRAPHX
struct SecureCacheFile final {
    int descriptor = -1;
    std::vector<unsigned char> bytes;
    SecureCacheFile() = default;
    SecureCacheFile(const SecureCacheFile &) = delete;
    SecureCacheFile &operator=(const SecureCacheFile &) = delete;
    ~SecureCacheFile() { if (descriptor >= 0) close(descriptor); }
    std::string descriptorPath() const { return "/proc/self/fd/" + std::to_string(descriptor); }
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

std::string cacheIdentity(bool fp16, std::size_t payloadBytes, const std::string &payloadSha)
{
    return std::string("{\n") +
        "  \"artifact_sha256\": \"" + PACKED_XTRANS_ONNX_SHA256 + "\",\n" +
        "  \"cache_format_revision\": 1,\n" +
        "  \"exhaustive_tune\": false,\n" +
        "  \"fast_math\": false,\n" +
        "  \"gpu_isa\": \"gfx1101\",\n" +
        "  \"migraphx_version\": \"" + rt_xveon_migraphx_version() + "\",\n" +
        "  \"offload_copy\": true,\n" +
        "  \"precision\": \"" + (fp16 ? "fp16" : "fp32") + "\",\n" +
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
        if (count <= 0) { close(descriptor); return false; }
        offset += static_cast<std::size_t>(count);
    }
    const bool synchronized = fsync(descriptor) == 0;
    const bool closed = close(descriptor) == 0;
    return synchronized && closed;
}

struct CachePaths final {
    bool enabled = false;
    std::string program;
    std::string manifest;
    CachePaths() = default;
    CachePaths(bool enabled, std::string program, std::string manifest) :
        enabled(enabled), program(std::move(program)), manifest(std::move(manifest)) {}
};

CachePaths cachePaths(bool fp16)
{
    const char *directory = std::getenv("RT_PACKED_XTRANS_MIGRAPHX_CACHE_DIR");
    if (!directory || !*directory || !secureDirectory(directory)) return {};
    const std::string stem = std::string(directory) + "/packedxtrans-gfx1101-" + (fp16 ? "fp16" : "fp32") + "-v1";
    return {true, stem + ".mxr", stem + ".json"};
}

bool authenticatedCache(const CachePaths &paths, bool fp16, SecureCacheFile &program)
{
    if (!paths.enabled) return false;
    SecureCacheFile manifest;
    if (!readSecureCacheFile(paths.manifest, 4096, manifest) ||
        !readSecureCacheFile(paths.program, 256u * 1024u * 1024u, program)) return false;
    return std::string(manifest.bytes.begin(), manifest.bytes.end()) ==
        cacheIdentity(fp16, program.bytes.size(), sha256(program.bytes));
}

void publishCache(const CachePaths &paths, bool fp16, RtXveonMigraphxSession *session)
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
    if (!readSecureCacheFile(programTemporary, 256u * 1024u * 1024u, saved) || fsync(saved.descriptor) != 0) {
        g_remove(programTemporary.c_str());
        return;
    }
    const std::string manifest = cacheIdentity(fp16, saved.bytes.size(), sha256(saved.bytes));
    if (!writeFile(manifestTemporary, manifest)) {
        g_remove(programTemporary.c_str()); g_remove(manifestTemporary.c_str()); return;
    }
    if (g_rename(manifestTemporary.c_str(), paths.manifest.c_str()) != 0 ||
        g_rename(programTemporary.c_str(), paths.program.c_str()) != 0) {
        g_remove(programTemporary.c_str()); g_remove(manifestTemporary.c_str());
    }
}

class MigraphxRunner final : public TimedRunner
{
public:
    MigraphxRunner(std::vector<unsigned char> bytes, bool fp16) : bytes_(std::move(bytes))
    {
        runtime_ = rt_xveon_migraphx_version();
        provider_ = "MIGraphX-gpu";
        precision_ = fp16 ? "fp16" : "fp32";
        compileSource_ = "fresh";
        if (access("/dev/kfd", R_OK | W_OK) != 0) {
            throw NeuralModelError(NeuralModelErrorCode::RUNTIME,
                "MIGraphX GPU access requires readable and writable /dev/kfd");
        }
        char message[1024] = {};
        const auto start = std::chrono::steady_clock::now();
        const CachePaths paths = cachePaths(fp16);
        SecureCacheFile cached;
        int status = RT_XVEON_MIGRAPHX_RUNTIME;
        if (authenticatedCache(paths, fp16, cached)) {
            const std::string descriptor = cached.descriptorPath();
            status = rt_xveon_migraphx_create_contract(bytes_.data(), bytes_.size(), &CONTRACT,
                0, fp16 ? 1 : 0, descriptor.c_str(), &session_, message, sizeof(message));
            if (status == RT_XVEON_MIGRAPHX_OK) {
                std::vector<float> input(PACKED_XTRANS_INPUT_FLOATS, 0.f);
                std::vector<float> output(PACKED_XTRANS_OUTPUT_FLOATS);
                status = rt_xveon_migraphx_run(session_, input.data(), input.size(), output.data(), output.size(),
                    message, sizeof(message));
                if (status == RT_XVEON_MIGRAPHX_OK &&
                    std::all_of(output.begin(), output.end(), [](float value) { return std::isfinite(value); })) {
                    compileSource_ = "cache";
                } else {
                    rt_xveon_migraphx_release(session_); session_ = nullptr;
                }
            }
        }
        if (!session_) {
            status = rt_xveon_migraphx_create_contract(bytes_.data(), bytes_.size(), &CONTRACT,
                0, fp16 ? 1 : 0, nullptr, &session_, message, sizeof(message));
            if (status == RT_XVEON_MIGRAPHX_OK) publishCache(paths, fp16, session_);
        }
        compilationUs_ = elapsed(start);
        if (status != RT_XVEON_MIGRAPHX_OK) throw migraphxError(status, message);
    }
    ~MigraphxRunner() override { rt_xveon_migraphx_release(session_); }
    NeuralModelError run(const float *input, std::size_t inputs, float *output, std::size_t outputs) override
    {
        std::lock_guard<std::mutex> lock(mutex_);
        char message[1024] = {};
        const auto start = std::chrono::steady_clock::now();
        const int status = rt_xveon_migraphx_run(session_, input, inputs, output, outputs, message, sizeof(message));
        inferenceUs_ = elapsed(start);
        return status == RT_XVEON_MIGRAPHX_OK ? NeuralModelError() : migraphxError(status, message);
    }
    std::uint64_t workingBufferBytes() const override { return bytes_.size() + (PACKED_XTRANS_INPUT_FLOATS + PACKED_XTRANS_OUTPUT_FLOATS) * sizeof(float); }
private:
    std::vector<unsigned char> bytes_;
    RtXveonMigraphxSession *session_ = nullptr;
};
#endif

std::mutex &cacheMutex() { static std::mutex *value = new std::mutex; return *value; }
std::map<std::string, std::shared_ptr<PackedXTransRunner>> &cache()
{
    static auto *value = new std::map<std::string, std::shared_ptr<PackedXTransRunner>>;
    return *value;
}
std::string selectedBackend()
{
    const char *value = std::getenv("RT_PACKED_XTRANS_BACKEND");
    if (value && *value) return value;
#ifdef RT_WITH_ONNXRUNTIME
    return "onnxruntime-cpu";
#elif defined(RT_WITH_MIGRAPHX)
    return "migraphx";
#elif defined(RT_WITH_TVM_VULKAN)
    return "tvm-vulkan";
#else
    return {};
#endif
}
bool selectedFp16()
{
    const char *value = std::getenv("RT_PACKED_XTRANS_PRECISION");
    if (!value || !*value || std::string(value) == "fp32") return false;
    if (std::string(value) == "fp16") return true;
    throw NeuralModelError(NeuralModelErrorCode::ENUM, "RT_PACKED_XTRANS_PRECISION must be fp32 or fp16");
}
} // namespace

PackedXTransLoadResult loadCachedPackedXTransRunner(const Glib::ustring &path)
{
    if (path.empty()) return {nullptr, NeuralModelError(NeuralModelErrorCode::IO, "PackedXTransNet model or TVM module path is unset or empty")};
    try {
        const std::string backend = selectedBackend();
        if (backend != "onnxruntime-cpu" && backend != "migraphx" && backend != "tvm-vulkan") {
            return {nullptr, NeuralModelError(NeuralModelErrorCode::ENUM, "RT_PACKED_XTRANS_BACKEND must be onnxruntime-cpu, migraphx, or tvm-vulkan")};
        }
        const bool fp16 = selectedFp16();
        if (backend != "migraphx" && fp16) return {nullptr, NeuralModelError(NeuralModelErrorCode::ENUM, "FP16 is available only with MIGraphX")};
#ifndef RT_WITH_ONNXRUNTIME
        if (backend == "onnxruntime-cpu") return {nullptr, NeuralModelError(NeuralModelErrorCode::RUNTIME, "RawTherapee was built without WITH_ONNXRUNTIME")};
#endif
#ifndef RT_WITH_TVM_VULKAN
        if (backend == "tvm-vulkan") return {nullptr, NeuralModelError(NeuralModelErrorCode::RUNTIME, "RawTherapee was built without WITH_TVM_VULKAN")};
#endif
#ifndef RT_WITH_MIGRAPHX
        if (backend == "migraphx") return {nullptr, NeuralModelError(NeuralModelErrorCode::RUNTIME, "RawTherapee was built without WITH_MIGRAPHX")};
#endif
        std::unique_ptr<char, decltype(&g_free)> canonical(g_canonicalize_filename(path.c_str(), nullptr), g_free);
        const std::string modelPath = canonical ? canonical.get() : path.raw();
        std::string runtimeIdentity;
        std::string artifactIdentity = PACKED_XTRANS_ONNX_SHA256;
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
            runtimeIdentity + (fp16 ? "\nfp16" : "\nfp32");
        std::lock_guard<std::mutex> lock(cacheMutex());
        auto found = cache().find(key);
        if (found != cache().end()) return {found->second, NeuralModelError()};
        std::shared_ptr<PackedXTransRunner> runner;
#ifdef RT_WITH_ONNXRUNTIME
        if (backend == "onnxruntime-cpu") runner.reset(new OrtRunner(readModel(modelPath)));
#endif
#ifdef RT_WITH_MIGRAPHX
        if (backend == "migraphx") runner.reset(new MigraphxRunner(readModel(modelPath), fp16));
#endif
#ifdef RT_WITH_TVM_VULKAN
        if (backend == "tvm-vulkan") runner.reset(new TvmRunner(readTvmModule(modelPath)));
#endif
        if (!runner) return {nullptr, NeuralModelError(NeuralModelErrorCode::RUNTIME, "selected PackedXTransNet backend is unavailable")};
        cache().emplace(key, runner);
        return {runner, NeuralModelError()};
    } catch (const NeuralModelError &error) {
        return {nullptr, error};
    } catch (const std::bad_alloc &) {
        return {nullptr, NeuralModelError(NeuralModelErrorCode::ALLOCATION, "cannot allocate PackedXTransNet model/session storage")};
    } catch (const std::exception &error) {
        return {nullptr, NeuralModelError(NeuralModelErrorCode::RUNTIME, error.what())};
    }
}

} // namespace neural
} // namespace rtengine
