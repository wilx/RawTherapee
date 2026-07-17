/*
 * Direct MIGraphX C API bridge for the developer-only X-veon experiment.
 * No MIGraphX C++ wrapper types cross RawTherapee's C++11 boundary.
 */
#include "xveon_migraphx_bridge.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <migraphx/migraphx.h>
#include <migraphx/version.h>

#if MIGRAPHX_VERSION_MAJOR != 2 || MIGRAPHX_VERSION_MINOR != 15 || MIGRAPHX_VERSION_PATCH != 0
#error "The X-veon experiment requires MIGraphX 2.15.0"
#endif

#define INPUT_COUNT ((size_t)4 * 288 * 288)
#define OUTPUT_COUNT ((size_t)3 * 288 * 288)

struct RtXveonMigraphxSession {
    migraphx_program_t program;
    migraphx_program_parameters_t parameters;
    migraphx_argument_t input_argument;
    float *input;
};

static void set_message(char *message, size_t size, const char *operation, migraphx_status status)
{
    if (message && size) {
        snprintf(message, size, "%s failed (MIGraphX status %d)", operation, (int)status);
    }
}

static int check_status(
    migraphx_status status, const char *operation, char *message, size_t message_size)
{
    if (status == migraphx_status_success) {
        return 1;
    }
    set_message(message, message_size, operation, status);
    return 0;
}

static int validate_shape(
    const_migraphx_shape_t shape,
    size_t channels,
    char *message,
    size_t message_size)
{
    migraphx_shape_datatype_t type;
    size_t rank = 0;
    size_t elements = 0;
    const size_t *lengths = NULL;
    size_t lengths_size = 0;
    if (!check_status(migraphx_shape_type(&type, shape), "read tensor type", message, message_size) ||
        !check_status(migraphx_shape_ndim(&rank, shape), "read tensor rank", message, message_size) ||
        !check_status(migraphx_shape_elements(&elements, shape), "read tensor elements", message, message_size) ||
        !check_status(migraphx_shape_lengths(&lengths, &lengths_size, shape), "read tensor shape", message, message_size)) {
        return RT_XVEON_MIGRAPHX_RUNTIME;
    }
    if (type != migraphx_shape_float_type || rank != 4 || lengths_size != 4 ||
        lengths[0] != 1 || lengths[1] != channels || lengths[2] != 288 || lengths[3] != 288 ||
        elements != channels * 288 * 288) {
        snprintf(message, message_size, "X-veon MIGraphX tensor contract differs");
        return RT_XVEON_MIGRAPHX_SCHEMA;
    }
    return RT_XVEON_MIGRAPHX_OK;
}

static int validate_program(migraphx_program_t program, char *message, size_t message_size)
{
    migraphx_program_parameter_shapes_t parameters = NULL;
    migraphx_shapes_t outputs = NULL;
    const_migraphx_shape_t input_shape = NULL;
    const_migraphx_shape_t output_shape = NULL;
    const char *names[1] = {NULL};
    size_t count = 0;
    int result = RT_XVEON_MIGRAPHX_RUNTIME;
    if (!check_status(migraphx_program_get_parameter_shapes(&parameters, program),
                      "query program parameters", message, message_size) ||
        !check_status(migraphx_program_parameter_shapes_size(&count, parameters),
                      "query parameter count", message, message_size)) {
        goto done;
    }
    if (count != 1 ||
        !check_status(migraphx_program_parameter_shapes_names(names, parameters),
                      "query parameter names", message, message_size)) {
        snprintf(message, message_size, "X-veon MIGraphX model must have one input");
        result = RT_XVEON_MIGRAPHX_SCHEMA;
        goto done;
    }
    if (!names[0] || strcmp(names[0], "input") != 0) {
        snprintf(message, message_size, "X-veon MIGraphX input name differs");
        result = RT_XVEON_MIGRAPHX_SCHEMA;
        goto done;
    }
    if (!check_status(migraphx_program_parameter_shapes_get(&input_shape, parameters, "input"),
                      "query input shape", message, message_size)) {
        goto done;
    }
    result = validate_shape(input_shape, 4, message, message_size);
    if (result != RT_XVEON_MIGRAPHX_OK) goto done;
    if (!check_status(migraphx_program_get_output_shapes(&outputs, program),
                      "query output shapes", message, message_size) ||
        !check_status(migraphx_shapes_size(&count, outputs),
                      "query output count", message, message_size)) {
        result = RT_XVEON_MIGRAPHX_RUNTIME;
        goto done;
    }
    if (count != 1 ||
        !check_status(migraphx_shapes_get(&output_shape, outputs, 0),
                      "query output shape", message, message_size)) {
        snprintf(message, message_size, "X-veon MIGraphX model must have one output");
        result = RT_XVEON_MIGRAPHX_SCHEMA;
        goto done;
    }
    result = validate_shape(output_shape, 3, message, message_size);
done:
    if (outputs) migraphx_shapes_destroy(outputs);
    if (parameters) migraphx_program_parameter_shapes_destroy(parameters);
    return result;
}

void rt_xveon_migraphx_release(RtXveonMigraphxSession *state)
{
    if (!state) return;
    if (state->input_argument) migraphx_argument_destroy(state->input_argument);
    if (state->parameters) migraphx_program_parameters_destroy(state->parameters);
    if (state->program) migraphx_program_destroy(state->program);
    free(state->input);
    free(state);
}

int rt_xveon_migraphx_create(
    const void *model,
    size_t model_size,
    int fast_math,
    const char *load_program,
    RtXveonMigraphxSession **out,
    char *message,
    size_t message_size)
{
    RtXveonMigraphxSession *state = NULL;
    migraphx_onnx_options_t onnx = NULL;
    migraphx_compile_options_t compile = NULL;
    migraphx_target_t target = NULL;
    migraphx_file_options_t file = NULL;
    migraphx_program_parameter_shapes_t shapes = NULL;
    const_migraphx_shape_t input_shape = NULL;
    int result = RT_XVEON_MIGRAPHX_RUNTIME;
    if (!out || (!model && !load_program)) return RT_XVEON_MIGRAPHX_RUNTIME;
    *out = NULL;
    state = (RtXveonMigraphxSession *)calloc(1, sizeof(*state));
    if (!state) return RT_XVEON_MIGRAPHX_ALLOCATION;
    state->input = (float *)malloc(INPUT_COUNT * sizeof(float));
    if (!state->input) {
        result = RT_XVEON_MIGRAPHX_ALLOCATION;
        goto fail;
    }
    if (load_program) {
        if (!check_status(migraphx_file_options_create(&file), "create cache file options", message, message_size) ||
            !check_status(migraphx_file_options_set_file_format(file, "msgpack"), "set cache format", message, message_size) ||
            !check_status(migraphx_load(&state->program, load_program, file), "load compiled program", message, message_size)) {
            result = RT_XVEON_MIGRAPHX_IO;
            goto fail;
        }
    } else {
        if (!check_status(migraphx_onnx_options_create(&onnx), "create ONNX options", message, message_size) ||
            !check_status(migraphx_parse_onnx_buffer(&state->program, model, model_size, onnx),
                          "parse authenticated ONNX buffer", message, message_size)) goto fail;
        result = validate_program(state->program, message, message_size);
        if (result != RT_XVEON_MIGRAPHX_OK) goto fail;
        result = RT_XVEON_MIGRAPHX_RUNTIME;
        if (!check_status(migraphx_target_create(&target, "gpu"), "create GPU target", message, message_size) ||
            !check_status(migraphx_compile_options_create(&compile), "create compile options", message, message_size) ||
            !check_status(migraphx_compile_options_set_offload_copy(compile, true), "enable offload copies", message, message_size) ||
            !check_status(migraphx_compile_options_set_fast_math(compile, fast_math != 0), "set fast math", message, message_size) ||
            !check_status(migraphx_compile_options_set_exhaustive_tune_flag(compile, false), "disable exhaustive tuning", message, message_size) ||
            !check_status(migraphx_program_compile(state->program, target, compile), "compile GPU program", message, message_size)) goto fail;
    }
    result = validate_program(state->program, message, message_size);
    if (result != RT_XVEON_MIGRAPHX_OK) goto fail;
    result = RT_XVEON_MIGRAPHX_RUNTIME;
    if (!check_status(migraphx_program_get_parameter_shapes(&shapes, state->program),
                      "query compiled input shape", message, message_size) ||
        !check_status(migraphx_program_parameter_shapes_get(&input_shape, shapes, "input"),
                      "get compiled input shape", message, message_size) ||
        !check_status(migraphx_argument_create(&state->input_argument, input_shape, (char *)state->input),
                      "create input argument", message, message_size) ||
        !check_status(migraphx_program_parameters_create(&state->parameters),
                      "create program parameters", message, message_size) ||
        !check_status(migraphx_program_parameters_add(state->parameters, "input", state->input_argument),
                      "bind input argument", message, message_size)) goto fail;
    if (shapes) migraphx_program_parameter_shapes_destroy(shapes);
    if (file) migraphx_file_options_destroy(file);
    if (target) migraphx_target_destroy(target);
    if (compile) migraphx_compile_options_destroy(compile);
    if (onnx) migraphx_onnx_options_destroy(onnx);
    *out = state;
    return RT_XVEON_MIGRAPHX_OK;
fail:
    if (shapes) migraphx_program_parameter_shapes_destroy(shapes);
    if (file) migraphx_file_options_destroy(file);
    if (target) migraphx_target_destroy(target);
    if (compile) migraphx_compile_options_destroy(compile);
    if (onnx) migraphx_onnx_options_destroy(onnx);
    rt_xveon_migraphx_release(state);
    return result;
}

int rt_xveon_migraphx_run(
    RtXveonMigraphxSession *state,
    const float *input,
    size_t input_count,
    float *output,
    size_t output_count,
    char *message,
    size_t message_size)
{
    migraphx_arguments_t outputs = NULL;
    const_migraphx_argument_t result = NULL;
    const_migraphx_shape_t shape = NULL;
    char *buffer = NULL;
    int status = RT_XVEON_MIGRAPHX_RUNTIME;
    if (!state || !input || !output || input_count != INPUT_COUNT || output_count != OUTPUT_COUNT) {
        snprintf(message, message_size, "X-veon MIGraphX inference tensor size differs");
        return RT_XVEON_MIGRAPHX_SCHEMA;
    }
    memcpy(state->input, input, INPUT_COUNT * sizeof(float));
    if (!check_status(migraphx_program_run(&outputs, state->program, state->parameters),
                      "run GPU program", message, message_size) ||
        !check_status(migraphx_arguments_size(&input_count, outputs),
                      "query inference outputs", message, message_size) || input_count != 1 ||
        !check_status(migraphx_arguments_get(&result, outputs, 0),
                      "get inference output", message, message_size) ||
        !check_status(migraphx_argument_shape(&shape, result),
                      "get inference output shape", message, message_size)) goto done;
    status = validate_shape(shape, 3, message, message_size);
    /* offload_copy appends copy_from_gpu and sync_stream to the returned
     * graph.  Calling context_finish again is both redundant and triggers a
     * ROCm 7.2 device-id assertion for this compiled program. */
    if (!check_status(migraphx_argument_buffer(&buffer, result),
                      "read inference output", message, message_size)) {
        status = RT_XVEON_MIGRAPHX_RUNTIME;
        goto done;
    }
    for (size_t i = 0; i < OUTPUT_COUNT; ++i) {
        if (!isfinite(((const float *)buffer)[i])) {
            snprintf(message, message_size, "X-veon MIGraphX output contains NaN or infinity");
            status = RT_XVEON_MIGRAPHX_NONFINITE;
            goto done;
        }
    }
    memcpy(output, buffer, OUTPUT_COUNT * sizeof(float));
    status = RT_XVEON_MIGRAPHX_OK;
done:
    if (outputs) migraphx_arguments_destroy(outputs);
    return status;
}

int rt_xveon_migraphx_save(
    RtXveonMigraphxSession *state,
    const char *path,
    char *message,
    size_t message_size)
{
    migraphx_file_options_t file = NULL;
    int result = RT_XVEON_MIGRAPHX_RUNTIME;
    if (!state || !path) return RT_XVEON_MIGRAPHX_RUNTIME;
    if (!check_status(migraphx_file_options_create(&file), "create cache file options", message, message_size) ||
        !check_status(migraphx_file_options_set_file_format(file, "msgpack"), "set cache format", message, message_size) ||
        !check_status(migraphx_save(state->program, path, file), "save compiled program", message, message_size)) goto done;
    result = RT_XVEON_MIGRAPHX_OK;
done:
    if (file) migraphx_file_options_destroy(file);
    return result;
}

const char *rt_xveon_migraphx_version(void)
{
    return "2.15.0-" MIGRAPHX_VERSION_TWEAK;
}
