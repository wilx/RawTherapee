/* C ABI for the optional C++17 Apache TVM Vulkan runtime bridge. */
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct RtTvmVulkanSession RtTvmVulkanSession;

typedef struct RtTvmVulkanContract {
    const int64_t *input_shape;
    size_t input_rank;
    const int64_t *output_shape;
    size_t output_rank;
    size_t input_floats;
    size_t output_floats;
} RtTvmVulkanContract;

enum RtTvmVulkanStatus {
    RT_TVM_VULKAN_OK = 0,
    RT_TVM_VULKAN_RUNTIME = 1,
    RT_TVM_VULKAN_SCHEMA = 2,
    RT_TVM_VULKAN_ALLOCATION = 3,
    RT_TVM_VULKAN_IO = 4,
    RT_TVM_VULKAN_NONFINITE = 5,
    RT_TVM_VULKAN_VERSION = 6
};

int rt_tvm_vulkan_create(
    const void *module_data,
    size_t module_size,
    const RtTvmVulkanContract *contract,
    RtTvmVulkanSession **out,
    char *message,
    size_t message_size);

int rt_tvm_vulkan_run(
    RtTvmVulkanSession *session,
    const float *input,
    size_t input_count,
    float *output,
    size_t output_count,
    char *message,
    size_t message_size);

const char *rt_tvm_vulkan_version(void);
const char *rt_tvm_vulkan_device(RtTvmVulkanSession *session);
uint64_t rt_tvm_vulkan_working_bytes(RtTvmVulkanSession *session);
void rt_tvm_vulkan_release(RtTvmVulkanSession *session);

#ifdef __cplusplus
}
#endif
