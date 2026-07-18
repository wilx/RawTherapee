/*
 * C-only ONNX Runtime bridge. ONNX Runtime 1.27's C header contains callback
 * declarations that are not accepted in a C++11 translation unit. Keeping the
 * vendor header here lets the rest of RawTherapee retain its C++11 baseline.
 */
#include "xveon_ort_bridge.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <onnxruntime_c_api.h>

#if ORT_API_VERSION != 27
#error "The X-veon experiment requires ONNX Runtime API version 27"
#endif

#define INPUT_COUNT ((size_t)4 * 288 * 288)
#define OUTPUT_COUNT ((size_t)3 * 288 * 288)

struct RtXveonOrtSession {
    const OrtApi *api;
    OrtEnv *env;
    OrtSession *session;
    OrtMemoryInfo *memory;
    OrtValue *input_value;
    OrtValue *output_value;
    float *input;
    float *output;
    RtFixedOnnxContract contract;
};

static void set_message(char *message, size_t size, const char *operation, const char *detail)
{
    if (!message || size == 0) {
        return;
    }
    if (operation && detail) {
        snprintf(message, size, "%s: %s", operation, detail);
    } else {
        snprintf(message, size, "%s", operation ? operation : (detail ? detail : "unknown ONNX Runtime error"));
    }
}

static int status_ok(
    const OrtApi *api,
    OrtStatus *status,
    const char *operation,
    char *message,
    size_t message_size)
{
    if (!status) {
        return 1;
    }
    set_message(message, message_size, operation, api->GetErrorMessage(status));
    api->ReleaseStatus(status);
    return 0;
}

static void release_allocated(const OrtApi *api, OrtAllocator *allocator, void *value)
{
    OrtStatus *status;
    if (!value) {
        return;
    }
    status = api->AllocatorFree(allocator, value);
    if (status) {
        api->ReleaseStatus(status);
    }
}

static int validate_value_contract(
    RtXveonOrtSession *state,
    int input,
    char *message,
    size_t message_size)
{
    const OrtApi *api = state->api;
    size_t count = 0;
    OrtAllocator *allocator = NULL;
    char *name = NULL;
    OrtTypeInfo *type = NULL;
    const OrtTensorTypeAndShapeInfo *tensor = NULL;
    ONNXTensorElementDataType scalar = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    size_t rank = 0;
    int64_t dimensions[4] = {0, 0, 0, 0};
    const char *expected_name = input ? state->contract.input_name : state->contract.output_name;
    const int64_t *expected_dimensions = input ? state->contract.input_shape : state->contract.output_shape;

    if (!status_ok(api, input ? api->SessionGetInputCount(state->session, &count)
                              : api->SessionGetOutputCount(state->session, &count),
                   input ? "query input count" : "query output count", message, message_size)) {
        return RT_XVEON_ORT_RUNTIME;
    }
    if (count != 1) {
        set_message(message, message_size, input ? "model must have one input" : "model must have one output", NULL);
        return RT_XVEON_ORT_SCHEMA;
    }
    if (!status_ok(api, api->GetAllocatorWithDefaultOptions(&allocator), "get ONNX Runtime allocator", message, message_size)) {
        return RT_XVEON_ORT_RUNTIME;
    }
    if (!status_ok(api, input ? api->SessionGetInputName(state->session, 0, allocator, &name)
                              : api->SessionGetOutputName(state->session, 0, allocator, &name),
                   input ? "query input name" : "query output name", message, message_size)) {
        return RT_XVEON_ORT_RUNTIME;
    }
    if (!name || strcmp(name, expected_name) != 0) {
        release_allocated(api, allocator, name);
        set_message(message, message_size, input ? "model input name differs" : "model output name differs", NULL);
        return RT_XVEON_ORT_SCHEMA;
    }
    release_allocated(api, allocator, name);

    if (!status_ok(api, input ? api->SessionGetInputTypeInfo(state->session, 0, &type)
                              : api->SessionGetOutputTypeInfo(state->session, 0, &type),
                   input ? "query input type" : "query output type", message, message_size)) {
        return RT_XVEON_ORT_RUNTIME;
    }
    if (!status_ok(api, api->CastTypeInfoToTensorInfo(type, &tensor), "read tensor contract", message, message_size)) {
        api->ReleaseTypeInfo(type);
        return RT_XVEON_ORT_RUNTIME;
    }
    if (!tensor) {
        api->ReleaseTypeInfo(type);
        set_message(message, message_size, "model input/output is not a tensor", NULL);
        return RT_XVEON_ORT_SCHEMA;
    }
    if (!status_ok(api, api->GetTensorElementType(tensor, &scalar), "read tensor scalar type", message, message_size) ||
        !status_ok(api, api->GetDimensionsCount(tensor, &rank), "read tensor rank", message, message_size)) {
        api->ReleaseTypeInfo(type);
        return RT_XVEON_ORT_RUNTIME;
    }
    if (scalar != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || rank != 4) {
        api->ReleaseTypeInfo(type);
        set_message(message, message_size, "model input/output type or rank differs", NULL);
        return RT_XVEON_ORT_SCHEMA;
    }
    if (!status_ok(api, api->GetDimensions(tensor, dimensions, 4), "read tensor dimensions", message, message_size)) {
        api->ReleaseTypeInfo(type);
        return RT_XVEON_ORT_RUNTIME;
    }
    api->ReleaseTypeInfo(type);
    for (size_t index = 0; index < 4; ++index) {
        if (dimensions[index] != expected_dimensions[index]) {
            set_message(message, message_size, "model input/output shape differs", NULL);
            return RT_XVEON_ORT_SCHEMA;
        }
    }
    return RT_XVEON_ORT_OK;
}

static int validate_metadata(RtXveonOrtSession *state, char *message, size_t message_size)
{
    static const char *keys[3] = {"epoch", "base_width", "best_val_psnr"};
    static const char *values[3] = {"399", "32", "45.78"};
    const OrtApi *api = state->api;
    OrtModelMetadata *metadata = NULL;
    OrtAllocator *allocator = NULL;
    int index;
    if (!status_ok(api, api->SessionGetModelMetadata(state->session, &metadata), "read X-veon metadata", message, message_size) ||
        !status_ok(api, api->GetAllocatorWithDefaultOptions(&allocator), "get ONNX Runtime allocator", message, message_size)) {
        if (metadata) api->ReleaseModelMetadata(metadata);
        return RT_XVEON_ORT_RUNTIME;
    }
    for (index = 0; index < 3; ++index) {
        char *wire = NULL;
        if (!status_ok(api, api->ModelMetadataLookupCustomMetadataMap(metadata, allocator, keys[index], &wire),
                       "read X-veon custom metadata", message, message_size)) {
            api->ReleaseModelMetadata(metadata);
            return RT_XVEON_ORT_RUNTIME;
        }
        if (!wire || strcmp(wire, values[index]) != 0) {
            release_allocated(api, allocator, wire);
            api->ReleaseModelMetadata(metadata);
            set_message(message, message_size, "X-veon custom metadata differs", keys[index]);
            return RT_XVEON_ORT_SCHEMA;
        }
        release_allocated(api, allocator, wire);
    }
    api->ReleaseModelMetadata(metadata);
    return RT_XVEON_ORT_OK;
}

const char *rt_xveon_ort_version(void)
{
    const OrtApiBase *base = OrtGetApiBase();
    return base ? base->GetVersionString() : "unavailable";
}

void rt_xveon_ort_release(RtXveonOrtSession *state)
{
    if (!state) return;
    if (state->api) {
        if (state->output_value) state->api->ReleaseValue(state->output_value);
        if (state->input_value) state->api->ReleaseValue(state->input_value);
        if (state->memory) state->api->ReleaseMemoryInfo(state->memory);
        if (state->session) state->api->ReleaseSession(state->session);
        if (state->env) state->api->ReleaseEnv(state->env);
    }
    free(state->output);
    free(state->input);
    free(state);
}

int rt_xveon_ort_create(
    const void *model_data,
    size_t model_size,
    RtXveonOrtSession **out,
    char *message,
    size_t message_size)
{
    static const RtFixedOnnxContract contract = {
        "X-veon", "input", "output", {1, 4, 288, 288}, {1, 3, 288, 288},
        INPUT_COUNT, OUTPUT_COUNT, 1
    };
    return rt_xveon_ort_create_contract(model_data, model_size, &contract, out, message, message_size);
}

int rt_xveon_ort_create_contract(
    const void *model_data,
    size_t model_size,
    const RtFixedOnnxContract *requested_contract,
    RtXveonOrtSession **out,
    char *message,
    size_t message_size)
{
    const OrtApiBase *base = OrtGetApiBase();
    const OrtApi *api;
    RtXveonOrtSession *state;
    OrtSessionOptions *options = NULL;
    int contract;
    if (out) *out = NULL;
    if (!base || !out || !model_data || !requested_contract || !requested_contract->label ||
        !requested_contract->input_name || !requested_contract->output_name ||
        !requested_contract->input_count || !requested_contract->output_count) {
        set_message(message, message_size, "ONNX Runtime API is unavailable", NULL);
        return RT_XVEON_ORT_RUNTIME;
    }
    if (strcmp(base->GetVersionString(), "1.27.0") != 0) {
        set_message(message, message_size, "ONNX Runtime version differs from required 1.27.0", NULL);
        return RT_XVEON_ORT_VERSION;
    }
    api = base->GetApi(ORT_API_VERSION);
    if (!api) {
        set_message(message, message_size, "ONNX Runtime API version 27 is unavailable", NULL);
        return RT_XVEON_ORT_VERSION;
    }
    state = (RtXveonOrtSession *)calloc(1, sizeof(*state));
    if (!state) {
        set_message(message, message_size, "cannot allocate ONNX Runtime session wrapper", NULL);
        return RT_XVEON_ORT_ALLOCATION;
    }
    state->api = api;
    state->contract = *requested_contract;
    state->input = (float *)malloc(state->contract.input_count * sizeof(float));
    state->output = (float *)malloc(state->contract.output_count * sizeof(float));
    if (!state->input || !state->output) {
        rt_xveon_ort_release(state);
        set_message(message, message_size, "cannot allocate ONNX Runtime tensor buffers", NULL);
        return RT_XVEON_ORT_ALLOCATION;
    }
    if (!status_ok(api, api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "RawTherapee-X-veon", &state->env),
                   "create ONNX Runtime environment", message, message_size) ||
        !status_ok(api, api->DisableTelemetryEvents(state->env), "disable ONNX Runtime telemetry", message, message_size) ||
        !status_ok(api, api->CreateSessionOptions(&options), "create ONNX Runtime session options", message, message_size) ||
        !status_ok(api, api->SetSessionExecutionMode(options, ORT_SEQUENTIAL), "select sequential graph execution", message, message_size) ||
        !status_ok(api, api->SetSessionGraphOptimizationLevel(options, ORT_ENABLE_ALL), "enable graph optimizations", message, message_size) ||
        !status_ok(api, api->SetInterOpNumThreads(options, 1), "set inter-op threads", message, message_size) ||
        !status_ok(api, api->CreateSessionFromArray(state->env, model_data, model_size, options, &state->session),
                   "create X-veon ONNX session", message, message_size)) {
        if (options) api->ReleaseSessionOptions(options);
        rt_xveon_ort_release(state);
        return RT_XVEON_ORT_RUNTIME;
    }
    api->ReleaseSessionOptions(options);
    contract = validate_value_contract(state, 1, message, message_size);
    if (contract == RT_XVEON_ORT_OK) contract = validate_value_contract(state, 0, message, message_size);
    if (contract == RT_XVEON_ORT_OK && state->contract.require_xveon_metadata) contract = validate_metadata(state, message, message_size);
    if (contract != RT_XVEON_ORT_OK) {
        rt_xveon_ort_release(state);
        return contract;
    }
    if (!status_ok(api, api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &state->memory),
                   "create CPU tensor memory info", message, message_size) ||
        !status_ok(api, api->CreateTensorWithDataAsOrtValue(state->memory, state->input, state->contract.input_count * sizeof(float),
                   state->contract.input_shape, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &state->input_value),
                   "create X-veon input tensor", message, message_size) ||
        !status_ok(api, api->CreateTensorWithDataAsOrtValue(state->memory, state->output, state->contract.output_count * sizeof(float),
                   state->contract.output_shape, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &state->output_value),
                   "create X-veon output tensor", message, message_size)) {
        rt_xveon_ort_release(state);
        return RT_XVEON_ORT_RUNTIME;
    }
    *out = state;
    return RT_XVEON_ORT_OK;
}

int rt_xveon_ort_run(
    RtXveonOrtSession *state,
    const float *input,
    size_t input_count,
    float *output,
    size_t output_count,
    char *message,
    size_t message_size)
{
    const char *input_names[1];
    const char *output_names[1];
    const OrtValue *inputs[1];
    OrtValue *outputs[1];
    if (!state || !input || !output || input_count != state->contract.input_count || output_count != state->contract.output_count) {
        set_message(message, message_size, "inference tensor size differs", NULL);
        return RT_XVEON_ORT_SCHEMA;
    }
    input_names[0] = state->contract.input_name;
    output_names[0] = state->contract.output_name;
    memcpy(state->input, input, input_count * sizeof(float));
    inputs[0] = state->input_value;
    outputs[0] = state->output_value;
    if (!status_ok(state->api, state->api->Run(state->session, NULL, input_names, inputs, 1,
                   output_names, 1, outputs), "run X-veon inference", message, message_size)) {
        return RT_XVEON_ORT_RUNTIME;
    }
    if (outputs[0] != state->output_value) {
        if (outputs[0]) state->api->ReleaseValue(outputs[0]);
        set_message(message, message_size, "ONNX Runtime replaced the preallocated X-veon output", NULL);
        return RT_XVEON_ORT_RUNTIME;
    }
    memcpy(output, state->output, output_count * sizeof(float));
    return RT_XVEON_ORT_OK;
}
