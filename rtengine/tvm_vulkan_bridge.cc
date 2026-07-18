#include "tvm_vulkan_bridge.h"

#include <tvm/ffi/container/shape.h>
#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/function.h>
#include <tvm/runtime/data_type.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/tensor.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <memory>
#include <new>
#include <optional>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>

#ifndef MFD_CLOEXEC
#define MFD_CLOEXEC 0x0001U
#endif
#ifndef MFD_ALLOW_SEALING
#define MFD_ALLOW_SEALING 0x0002U
#endif

namespace
{

void setMessage(char *message, std::size_t size, const std::string &value)
{
    if (!message || !size) {
        return;
    }
    const std::size_t count = std::min(size - 1, value.size());
    std::memcpy(message, value.data(), count);
    message[count] = '\0';
}

int makeMemfd()
{
#ifdef SYS_memfd_create
    return static_cast<int>(syscall(SYS_memfd_create, "rawtherapee-tvm-module.so", MFD_CLOEXEC | MFD_ALLOW_SEALING));
#else
    errno = ENOSYS;
    return -1;
#endif
}

bool writeAll(int descriptor, const unsigned char *data, std::size_t size)
{
    std::size_t offset = 0;
    while (offset < size) {
        const ssize_t count = write(descriptor, data + offset, size - offset);
        if (count < 0 && errno == EINTR) {
            continue;
        }
        if (count <= 0) {
            return false;
        }
        offset += static_cast<std::size_t>(count);
    }
    return true;
}

std::size_t elementCount(const int64_t *shape, std::size_t rank)
{
    if (!shape || rank == 0 || rank > 4) {
        throw std::runtime_error("TVM tensor rank must be between one and four");
    }
    std::size_t count = 1;
    for (std::size_t index = 0; index < rank; ++index) {
        if (shape[index] <= 0 || static_cast<std::uint64_t>(shape[index]) > SIZE_MAX / count) {
            throw std::runtime_error("TVM tensor shape is invalid or overflows");
        }
        count *= static_cast<std::size_t>(shape[index]);
    }
    return count;
}

bool tensorContract(const tvm::runtime::Tensor &tensor, const int64_t *shape, std::size_t rank)
{
    if (tensor->ndim != static_cast<int>(rank) || tensor->dtype.code != kDLFloat ||
        tensor->dtype.bits != 32 || tensor->dtype.lanes != 1) {
        return false;
    }
    for (std::size_t index = 0; index < rank; ++index) {
        if (tensor->shape[index] != shape[index]) {
            return false;
        }
    }
    return true;
}

} // namespace

struct RtTvmVulkanSession {
    int descriptor = -1;
    std::optional<tvm::ffi::Module> library;
    std::optional<tvm::ffi::Module> vm;
    std::optional<tvm::ffi::Function> main;
    tvm::runtime::Tensor input;
    std::vector<int64_t> inputShape;
    std::vector<int64_t> outputShape;
    std::size_t inputFloats = 0;
    std::size_t outputFloats = 0;
    std::uint64_t workingBytes = 0;
    std::string device = "Vulkan device 0";

    ~RtTvmVulkanSession()
    {
        if (descriptor >= 0) {
            close(descriptor);
        }
    }
};

extern "C" void rt_tvm_vulkan_release(RtTvmVulkanSession *session)
{
    delete session;
}

extern "C" int rt_tvm_vulkan_create(
    const void *moduleData,
    std::size_t moduleSize,
    const RtTvmVulkanContract *contract,
    RtTvmVulkanSession **out,
    char *message,
    std::size_t messageSize)
{
    if (out) {
        *out = nullptr;
    }
    if (!moduleData || !moduleSize || !contract || !out) {
        setMessage(message, messageSize, "TVM module, contract, and output are required");
        return RT_TVM_VULKAN_SCHEMA;
    }
    try {
        const std::size_t inputFloats = elementCount(contract->input_shape, contract->input_rank);
        const std::size_t outputFloats = elementCount(contract->output_shape, contract->output_rank);
        if (inputFloats != contract->input_floats || outputFloats != contract->output_floats) {
            throw std::runtime_error("TVM contract element counts differ from shapes");
        }
        std::unique_ptr<RtTvmVulkanSession> state(new RtTvmVulkanSession);
        state->descriptor = makeMemfd();
        if (state->descriptor < 0 || !writeAll(
                state->descriptor,
                static_cast<const unsigned char *>(moduleData),
                moduleSize)) {
            setMessage(message, messageSize, "cannot create authenticated sealed TVM module");
            return RT_TVM_VULKAN_IO;
        }
        const int seals = F_SEAL_WRITE | F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_SEAL;
        if (fcntl(state->descriptor, F_ADD_SEALS, seals) != 0) {
            setMessage(message, messageSize, "cannot seal authenticated TVM module");
            return RT_TVM_VULKAN_IO;
        }
        const std::string descriptorPath = "/proc/self/fd/" + std::to_string(state->descriptor);
        const auto loader = tvm::ffi::Function::GetGlobal("ffi.Module.load_from_file.so");
        if (!loader.has_value()) {
            throw std::runtime_error("TVM shared-library module loader is not registered");
        }
        state->library.emplace((*loader)(descriptorPath, "so").cast<tvm::ffi::Module>());
        const auto executableLoader = (*state->library)->GetFunction("vm_load_executable", true);
        if (!executableLoader.has_value()) {
            throw std::runtime_error("TVM module lacks vm_load_executable");
        }
        state->vm.emplace((*executableLoader)().cast<tvm::ffi::Module>());
        const auto initialize = (*state->vm)->GetFunction("vm_initialization");
        const auto main = (*state->vm)->GetFunction("main");
        if (!initialize.has_value() || !main.has_value()) {
            throw std::runtime_error("TVM module lacks VM initialization or main entry point");
        }
        (*initialize)(static_cast<int>(kDLVulkan), 0, 2, static_cast<int>(kDLCPU), 0, 2);
        state->main.emplace(*main);
        state->inputShape.assign(contract->input_shape, contract->input_shape + contract->input_rank);
        state->outputShape.assign(contract->output_shape, contract->output_shape + contract->output_rank);
        state->inputFloats = inputFloats;
        state->outputFloats = outputFloats;
        const DLDevice vulkan{static_cast<DLDeviceType>(kDLVulkan), 0};
        tvm::runtime::DeviceAPI *const deviceApi = tvm::runtime::DeviceAPI::Get(vulkan);
        tvm::ffi::Any exists;
        deviceApi->GetAttr(vulkan, tvm::runtime::kExist, &exists);
        if (!exists.cast<int>()) {
            throw std::runtime_error("TVM Vulkan device 0 is unavailable");
        }
        tvm::ffi::Any deviceName;
        tvm::ffi::Any driverVersion;
        deviceApi->GetAttr(vulkan, tvm::runtime::kDeviceName, &deviceName);
        deviceApi->GetAttr(vulkan, tvm::runtime::kDriverVersion, &driverVersion);
        state->device = "TVM-Vulkan/" + std::string(deviceName.cast<tvm::ffi::String>());
        if (driverVersion.type_index() != tvm::ffi::TypeIndex::kTVMFFINone) {
            state->device += "/driver-" + std::string(driverVersion.cast<tvm::ffi::String>());
        }
        const DLDataType float32{static_cast<std::uint8_t>(kDLFloat), 32, 1};
        state->input = tvm::runtime::Tensor::Empty(
            tvm::ffi::Shape(state->inputShape.begin(), state->inputShape.end()), float32, vulkan);
        state->workingBytes = moduleSize + (inputFloats + outputFloats) * sizeof(float);
        *out = state.release();
        setMessage(message, messageSize, "");
        return RT_TVM_VULKAN_OK;
    } catch (const std::bad_alloc &) {
        setMessage(message, messageSize, "cannot allocate TVM Vulkan session storage");
        return RT_TVM_VULKAN_ALLOCATION;
    } catch (const std::exception &error) {
        setMessage(message, messageSize, error.what());
        return RT_TVM_VULKAN_RUNTIME;
    }
}

extern "C" int rt_tvm_vulkan_run(
    RtTvmVulkanSession *session,
    const float *input,
    std::size_t inputCount,
    float *output,
    std::size_t outputCount,
    char *message,
    std::size_t messageSize)
{
    if (!session || !input || !output || inputCount != session->inputFloats ||
        outputCount != session->outputFloats) {
        setMessage(message, messageSize, "TVM input or output contract differs");
        return RT_TVM_VULKAN_SCHEMA;
    }
    if (!std::all_of(input, input + inputCount, [](float value) { return std::isfinite(value); })) {
        setMessage(message, messageSize, "TVM input contains NaN or infinity");
        return RT_TVM_VULKAN_NONFINITE;
    }
    try {
        session->input.CopyFromBytes(input, inputCount * sizeof(float));
        tvm::ffi::Any result = (*session->main)(session->input);
        const tvm::runtime::Tensor tensor = result.cast<tvm::runtime::Tensor>();
        if (!tensorContract(tensor, session->outputShape.data(), session->outputShape.size())) {
            setMessage(message, messageSize, "TVM output tensor contract differs");
            return RT_TVM_VULKAN_SCHEMA;
        }
        const DLDevice vulkan{static_cast<DLDeviceType>(kDLVulkan), 0};
        tvm::runtime::DeviceAPI::Get(vulkan)->StreamSync(vulkan, nullptr);
        tensor.CopyToBytes(output, outputCount * sizeof(float));
        if (!std::all_of(output, output + outputCount, [](float value) { return std::isfinite(value); })) {
            setMessage(message, messageSize, "TVM output contains NaN or infinity");
            return RT_TVM_VULKAN_NONFINITE;
        }
        setMessage(message, messageSize, "");
        return RT_TVM_VULKAN_OK;
    } catch (const std::bad_alloc &) {
        setMessage(message, messageSize, "cannot allocate TVM inference storage");
        return RT_TVM_VULKAN_ALLOCATION;
    } catch (const std::exception &error) {
        setMessage(message, messageSize, error.what());
        return RT_TVM_VULKAN_RUNTIME;
    }
}

extern "C" const char *rt_tvm_vulkan_version(void)
{
    return "Apache TVM 0.25.0 Vulkan runtime; TVM-FFI 0.1.12";
}

extern "C" const char *rt_tvm_vulkan_device(RtTvmVulkanSession *session)
{
    return session ? session->device.c_str() : "unavailable";
}

extern "C" std::uint64_t rt_tvm_vulkan_working_bytes(RtTvmVulkanSession *session)
{
    return session ? session->workingBytes : 0;
}
