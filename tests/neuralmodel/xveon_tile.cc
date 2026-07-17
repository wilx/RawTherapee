#include "rtengine/xtrans_xveon.h"

#include <cmath>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <memory>
#include <vector>

#include <glib/gstdio.h>

int main(int argc, char **argv)
{
    if (argc != 3 && argc != 4) {
        std::cerr << "usage: rawtherapee-xveon-tile MODEL.onnx OUTPUT.f32le [ITERATIONS]\n";
        return 2;
    }
    const int iterations = argc == 4 ? std::atoi(argv[3]) : 1;
    if (iterations < 1 || iterations > 1000) {
        std::cerr << "iterations must be in 1..1000\n";
        return 2;
    }
    const auto loaded = rtengine::neural::loadCachedXVeonXTransRunner(argv[1]);
    if (!loaded) {
        std::cerr << "error [" << rtengine::neural::neuralModelErrorCodeName(loaded.error.code)
                  << "]: " << loaded.error.message << '\n';
        return 2;
    }
    std::vector<float> input(rtengine::neural::XVEON_INPUT_FLOATS, 0.f);
    std::vector<float> output(rtengine::neural::XVEON_OUTPUT_FLOATS);
    constexpr std::size_t pixels = 288u * 288u;
    for (int y = 0; y < 288; ++y) {
        for (int x = 0; x < 288; ++x) {
            const std::size_t pixel = static_cast<std::size_t>(y) * 288 + x;
            const int channel = rtengine::XVEON_XTRANS_CFA[y % 6][x % 6];
            input[pixel] = static_cast<float>((y * 31 + x * 17) % 1024) / 1023.f;
            input[(static_cast<std::size_t>(channel) + 1) * pixels + pixel] = 1.f;
        }
    }
    std::uint64_t inferenceTotal = 0;
    for (int iteration = 0; iteration < iterations; ++iteration) {
        const auto started = std::chrono::steady_clock::now();
        const auto error = loaded.runner->run(input.data(), input.size(), output.data(), output.size());
        inferenceTotal += static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - started).count());
        if (error) {
            std::cerr << "error [" << rtengine::neural::neuralModelErrorCodeName(error.code)
                      << "]: " << error.message << '\n';
            return 2;
        }
    }
    std::cerr << "backend=" << loaded.runner->provider()
              << " runtime=" << loaded.runner->runtimeVersion()
              << " compile_source=" << loaded.runner->compileSource()
              << " compile_us=" << loaded.runner->compilationMicroseconds()
              << " inference_us=" << loaded.runner->lastInferenceMicroseconds()
              << " iterations=" << iterations
              << " inference_average_us=" << inferenceTotal / static_cast<std::uint64_t>(iterations) << '\n';
    std::unique_ptr<std::FILE, int (*)(std::FILE *)> file(g_fopen(argv[2], "wb"), std::fclose);
    if (!file) {
        std::cerr << "cannot open output\n";
        return 2;
    }
    for (const float value : output) {
        std::uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        const unsigned char bytes[4] = {
            static_cast<unsigned char>(bits),
            static_cast<unsigned char>(bits >> 8),
            static_cast<unsigned char>(bits >> 16),
            static_cast<unsigned char>(bits >> 24)
        };
        if (std::fwrite(bytes, 1, sizeof(bytes), file.get()) != sizeof(bytes)) {
            std::cerr << "cannot write output\n";
            return 2;
        }
    }
    return 0;
}
