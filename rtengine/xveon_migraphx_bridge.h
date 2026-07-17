/* C-only bridge for the optional developer MIGraphX backend. */
#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct RtXveonMigraphxSession RtXveonMigraphxSession;

enum RtXveonMigraphxError {
    RT_XVEON_MIGRAPHX_OK = 0,
    RT_XVEON_MIGRAPHX_VERSION = 1,
    RT_XVEON_MIGRAPHX_SCHEMA = 2,
    RT_XVEON_MIGRAPHX_RUNTIME = 3,
    RT_XVEON_MIGRAPHX_ALLOCATION = 4,
    RT_XVEON_MIGRAPHX_IO = 5,
    RT_XVEON_MIGRAPHX_NONFINITE = 6
};

int rt_xveon_migraphx_create(
    const void *model,
    size_t model_size,
    int fast_math,
    const char *load_program,
    RtXveonMigraphxSession **out,
    char *message,
    size_t message_size);

int rt_xveon_migraphx_run(
    RtXveonMigraphxSession *session,
    const float *input,
    size_t input_count,
    float *output,
    size_t output_count,
    char *message,
    size_t message_size);

int rt_xveon_migraphx_save(
    RtXveonMigraphxSession *session,
    const char *path,
    char *message,
    size_t message_size);

const char *rt_xveon_migraphx_version(void);
void rt_xveon_migraphx_release(RtXveonMigraphxSession *session);

#ifdef __cplusplus
}
#endif
