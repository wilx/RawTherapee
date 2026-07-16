#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct RtXveonOrtSession RtXveonOrtSession;

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
