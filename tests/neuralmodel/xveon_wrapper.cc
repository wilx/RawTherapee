#include "rtengine/xtrans_xveon.h"
#include "rtengine/xtrans_cfa.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <memory>

#include <glib/gstdio.h>

namespace
{

void makeCfa(int result[6][6])
{
    // transformed_cfa((0, -1, 1, 0), 2, 3) in xveon_reference.py.
    for (int y = 0; y < 6; ++y) {
        for (int x = 0; x < 6; ++x) {
            result[y][x] = rtengine::XVEON_XTRANS_CFA[
                rtengine::positiveModulo(x + 3, 6)
            ][rtengine::positiveModulo(-y + 2, 6)];
        }
    }
}

bool writeFloat(std::FILE *file, float value)
{
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const unsigned char bytes[4] = {
        static_cast<unsigned char>(bits),
        static_cast<unsigned char>(bits >> 8),
        static_cast<unsigned char>(bits >> 16),
        static_cast<unsigned char>(bits >> 24)
    };
    return std::fwrite(bytes, 1, sizeof(bytes), file) == sizeof(bytes);
}

} // namespace

int main(int argc, char **argv)
{
    if (argc != 3) {
        std::cerr << "usage: rawtherapee-xveon-wrapper MODEL.onnx OUTPUT.f32le\n";
        return 2;
    }
    const auto loaded = rtengine::neural::loadCachedXVeonXTransRunner(argv[1]);
    if (!loaded) {
        std::cerr << "error [" << rtengine::neural::neuralModelErrorCodeName(loaded.error.code)
                  << "]: " << loaded.error.message << '\n';
        return 2;
    }

    constexpr int width = 173;
    constexpr int height = 31;
    array2D<float> raw(width, height), red(width, height), green(width, height), blue(width, height);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            raw[y][x] = static_cast<float>((y * 197 + x * 113) % 65536);
        }
    }
    int cfa[6][6];
    makeCfa(cfa);
    const auto result = rtengine::demosaicXVeonXTrans(
        raw, red, green, blue, width, height, cfa, loaded.runner);
    if (!result) {
        std::cerr << "error [" << rtengine::neural::neuralModelErrorCodeName(result.error.code)
                  << "]: " << result.error.message << '\n';
        return 2;
    }

    std::unique_ptr<std::FILE, int (*)(std::FILE *)> file(g_fopen(argv[2], "wb"), std::fclose);
    if (!file) {
        std::cerr << "cannot open output\n";
        return 2;
    }
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            if (!writeFloat(file.get(), red[y][x] / 65535.f) ||
                !writeFloat(file.get(), green[y][x] / 65535.f) ||
                !writeFloat(file.get(), blue[y][x] / 65535.f)) {
                std::cerr << "cannot write output\n";
                return 2;
            }
        }
    }
    return 0;
}
