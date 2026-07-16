/*
 *  This file is part of RawTherapee.
 *
 *  RawTherapee is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 */

#include "neuralmodel.h"

#include <utility>

namespace rtengine
{

namespace neural
{

NeuralModelError::NeuralModelError(NeuralModelErrorCode code, std::string message) :
    code(code), message(std::move(message))
{
}

const char *neuralModelErrorCodeName(NeuralModelErrorCode code)
{
    switch (code) {
        case NeuralModelErrorCode::NONE:
            return "NONE";
        case NeuralModelErrorCode::IO:
            return "IO";
        case NeuralModelErrorCode::SIZE:
            return "SIZE";
        case NeuralModelErrorCode::MAGIC:
            return "MAGIC";
        case NeuralModelErrorCode::VERSION:
            return "VERSION";
        case NeuralModelErrorCode::ENDIAN:
            return "ENDIAN";
        case NeuralModelErrorCode::FLAGS:
            return "FLAGS";
        case NeuralModelErrorCode::ENUM:
            return "ENUM";
        case NeuralModelErrorCode::RESERVED:
            return "RESERVED";
        case NeuralModelErrorCode::LIMIT:
            return "LIMIT";
        case NeuralModelErrorCode::RANGE:
            return "RANGE";
        case NeuralModelErrorCode::ALIGNMENT:
            return "ALIGNMENT";
        case NeuralModelErrorCode::ORDER:
            return "ORDER";
        case NeuralModelErrorCode::SCHEMA:
            return "SCHEMA";
        case NeuralModelErrorCode::DIGEST:
            return "DIGEST";
        case NeuralModelErrorCode::NONFINITE:
            return "NONFINITE";
        case NeuralModelErrorCode::ALLOCATION:
            return "ALLOCATION";
        case NeuralModelErrorCode::RUNTIME:
            return "RUNTIME";
    }

    return "UNKNOWN";
}

} // namespace neural

} // namespace rtengine
