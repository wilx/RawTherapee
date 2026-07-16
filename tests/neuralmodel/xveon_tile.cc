#include "rtengine/xtrans_xveon.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <memory>
#include <vector>

#include <glib/gstdio.h>

int main(int argc, char **argv)
{
    if (argc != 3) {
        std::cerr << "usage: rawtherapee-xveon-tile MODEL.onnx OUTPUT.f32le\n";
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
    const auto error = loaded.runner->run(input.data(), input.size(), output.data(), output.size());
    if (error) {
        std::cerr << "error [" << rtengine::neural::neuralModelErrorCodeName(error.code)
                  << "]: " << error.message << '\n';
        return 2;
    }
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
