#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct RtXveonOrtSession RtXveonOrtSession;

typedef struct RtFixedOnnxContract {
    const char *label;
    const char *input_name;
    const char *output_name;
    int64_t input_shape[4];
    int64_t output_shape[4];
    size_t input_count;
    size_t output_count;
    int require_xveon_metadata;
} RtFixedOnnxContract;

enum RtXveonOrtError {
    RT_XVEON_ORT_OK = 0,
    RT_XVEON_ORT_VERSION = 1,
    RT_XVEON_ORT_SCHEMA = 2,
    RT_XVEON_ORT_RUNTIME = 3,
    RT_XVEON_ORT_ALLOCATION = 4
};

int rt_xveon_ort_create(
    const void *model_data,
    size_t model_size,
    RtXveonOrtSession **out,
    char *message,
    size_t message_size);

int rt_xveon_ort_create_contract(
    const void *model_data,
    size_t model_size,
    const RtFixedOnnxContract *contract,
    RtXveonOrtSession **out,
    char *message,
    size_t message_size);

int rt_xveon_ort_run(
    RtXveonOrtSession *session,
    const float *input,
    size_t input_count,
    float *output,
    size_t output_count,
    char *message,
    size_t message_size);

const char *rt_xveon_ort_version(void);
void rt_xveon_ort_release(RtXveonOrtSession *session);

#ifdef __cplusplus
}
#endif
