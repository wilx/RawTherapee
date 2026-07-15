/*
 *  This file is part of RawTherapee.
 *
 *  RawTherapee is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 */
#pragma once

#include <array>
#include <cstdint>
#include <string>

namespace rtengine
{

namespace neural
{

using Sha256Digest = std::array<std::uint8_t, 32>;

enum class NeuralModelErrorCode {
    NONE,
    IO,
    SIZE,
    MAGIC,
    VERSION,
    ENDIAN,
    FLAGS,
    ENUM,
    RESERVED,
    LIMIT,
    RANGE,
    ALIGNMENT,
    ORDER,
    SCHEMA,
    DIGEST,
    NONFINITE,
    ALLOCATION
};

const char *neuralModelErrorCodeName(NeuralModelErrorCode code);

struct NeuralModelError final {
    NeuralModelErrorCode code = NeuralModelErrorCode::NONE;
    std::string message;

    NeuralModelError() = default;
    NeuralModelError(NeuralModelErrorCode code, std::string message);

    explicit operator bool() const
    {
        return code != NeuralModelErrorCode::NONE;
    }
};

enum class TensorLayout : std::uint16_t {
    VECTOR = 1,
    OIHW = 2
};

struct NeuralTensorView final {
    std::uint32_t id = 0;
    std::uint16_t rank = 0;
    TensorLayout layout = TensorLayout::VECTOR;
    std::array<std::uint32_t, 4> dimensions{{0, 0, 0, 0}};
    std::uint64_t elementCount = 0;
    std::uint64_t payloadOffset = 0;
    std::uint64_t byteLength = 0;
    Sha256Digest sha256{{}};
    const float *data = nullptr;
};

} // namespace neural

} // namespace rtengine
