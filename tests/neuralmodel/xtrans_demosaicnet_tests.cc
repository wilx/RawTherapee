#include "xtrans_demosaicnet_tests.h"

#include "rtengine/demosaicnetxtransinference.h"
#include "rtengine/rtnnreader_p.h"
#include "rtengine/xtrans_cfa.h"
#include "rtengine/xtrans_demosaicnet.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <glib/gstdio.h>

namespace
{

using rtengine::neural::DemosaicNetXTransModel;
using rtengine::neural::NeuralTensorView;
using rtengine::neural::TensorLayout;
using rtengine::neural::detail::DemosaicNetXTransModelAccess;
using rtengine::neural::detail::ParsedRtnn;

constexpr std::size_t ALIGNMENT = 64;
constexpr std::size_t TENSOR_COUNT = 26;

struct TensorSpec final {
    std::uint16_t rank;
    TensorLayout layout;
    std::array<std::uint32_t, 4> dimensions;
    std::uint64_t count;
};

const std::array<TensorSpec, TENSOR_COUNT> SPECS{{
    {4, TensorLayout::OIHW, {{64, 3, 3, 3}}, 1728},
    {1, TensorLayout::VECTOR, {{64, 0, 0, 0}}, 64},
    {4, TensorLayout::OIHW, {{64, 64, 3, 3}}, 36864}, {1, TensorLayout::VECTOR, {{64, 0, 0, 0}}, 64},
    {4, TensorLayout::OIHW, {{64, 64, 3, 3}}, 36864}, {1, TensorLayout::VECTOR, {{64, 0, 0, 0}}, 64},
    {4, TensorLayout::OIHW, {{64, 64, 3, 3}}, 36864}, {1, TensorLayout::VECTOR, {{64, 0, 0, 0}}, 64},
    {4, TensorLayout::OIHW, {{64, 64, 3, 3}}, 36864}, {1, TensorLayout::VECTOR, {{64, 0, 0, 0}}, 64},
    {4, TensorLayout::OIHW, {{64, 64, 3, 3}}, 36864}, {1, TensorLayout::VECTOR, {{64, 0, 0, 0}}, 64},
    {4, TensorLayout::OIHW, {{64, 64, 3, 3}}, 36864}, {1, TensorLayout::VECTOR, {{64, 0, 0, 0}}, 64},
    {4, TensorLayout::OIHW, {{64, 64, 3, 3}}, 36864}, {1, TensorLayout::VECTOR, {{64, 0, 0, 0}}, 64},
    {4, TensorLayout::OIHW, {{64, 64, 3, 3}}, 36864}, {1, TensorLayout::VECTOR, {{64, 0, 0, 0}}, 64},
    {4, TensorLayout::OIHW, {{64, 64, 3, 3}}, 36864}, {1, TensorLayout::VECTOR, {{64, 0, 0, 0}}, 64},
    {4, TensorLayout::OIHW, {{64, 64, 3, 3}}, 36864}, {1, TensorLayout::VECTOR, {{64, 0, 0, 0}}, 64},
    {4, TensorLayout::OIHW, {{64, 67, 3, 3}}, 38592},
    {1, TensorLayout::VECTOR, {{64, 0, 0, 0}}, 64},
    {4, TensorLayout::OIHW, {{3, 64, 1, 1}}, 192},
    {1, TensorLayout::VECTOR, {{3, 0, 0, 0}}, 3},
}};

void require(bool condition, const std::string &message)
{
    if (!condition) {
        throw std::runtime_error(message);
    }
}

struct SyntheticModel final {
    std::shared_ptr<const DemosaicNetXTransModel> model;
    std::array<float *, TENSOR_COUNT> tensors{{}};
};

SyntheticModel syntheticModel()
{
    std::unique_ptr<ParsedRtnn> parsed(new ParsedRtnn());
    parsed->architectureId = 1;
    parsed->modelRevision = 1;
    parsed->parameterCount = 409923;
    parsed->tensorPayloadBytes = 1639692;
    parsed->allocation.reset(new float[409923 + ALIGNMENT / sizeof(float)]);
    const std::uintptr_t address = reinterpret_cast<std::uintptr_t>(parsed->allocation.get());
    parsed->alignedData = reinterpret_cast<float *>((address + ALIGNMENT - 1) & ~(ALIGNMENT - 1));
    std::fill(parsed->alignedData, parsed->alignedData + 409923, 0.f);

    SyntheticModel result;
    std::uint64_t cursor = 0;
    for (std::size_t index = 0; index < SPECS.size(); ++index) {
        const TensorSpec &spec = SPECS[index];
        NeuralTensorView view;
        view.id = static_cast<std::uint32_t>(index + 1);
        view.rank = spec.rank;
        view.layout = spec.layout;
        view.dimensions = spec.dimensions;
        view.elementCount = spec.count;
        view.byteLength = spec.count * sizeof(float);
        view.data = parsed->alignedData + cursor;
        parsed->tensors.push_back(view);
        result.tensors[index] = parsed->alignedData + cursor;
        cursor += spec.count;
    }
    require(cursor == 409923, "synthetic model parameter count differs");
    result.model = DemosaicNetXTransModelAccess::create(std::move(parsed));
    return result;
}

void cfaFor(
    int a, int b, int c, int d, int ox, int oy,
    int result[6][6])
{
    for (int y = 0; y < 6; ++y) {
        for (int x = 0; x < 6; ++x) {
            result[y][x] = rtengine::CANONICAL_XTRANS_CFA[
                rtengine::positiveModulo(c * x + d * y + oy, 6)
            ][rtengine::positiveModulo(a * x + b * y + ox, 6)];
        }
    }
}

std::string cfaKey(const int cfa[6][6])
{
    std::string result;
    for (int y = 0; y < 6; ++y) {
        for (int x = 0; x < 6; ++x) {
            result.push_back(static_cast<char>('0' + cfa[y][x]));
        }
    }
    return result;
}

std::vector<float> readFloats(const std::string &path, std::size_t count)
{
    std::unique_ptr<std::FILE, int (*)(std::FILE *)> file(g_fopen(path.c_str(), "rb"), std::fclose);
    require(static_cast<bool>(file), "cannot open raw-wrapper fixture: " + path);
    std::vector<unsigned char> bytes(count * sizeof(float));
    require(std::fread(bytes.data(), 1, bytes.size(), file.get()) == bytes.size(),
        "cannot read raw-wrapper fixture: " + path);
    require(std::fgetc(file.get()) == EOF, "raw-wrapper fixture has trailing bytes: " + path);
    std::vector<float> result(count);
    for (std::size_t index = 0; index < count; ++index) {
        const unsigned char *source = bytes.data() + index * 4;
        const std::uint32_t bits = static_cast<std::uint32_t>(source[0]) |
            static_cast<std::uint32_t>(source[1]) << 8 |
            static_cast<std::uint32_t>(source[2]) << 16 |
            static_cast<std::uint32_t>(source[3]) << 24;
        std::memcpy(&result[index], &bits, sizeof(bits));
    }
    return result;
}

struct CorpusCase final {
    const char *id;
    int width;
    int height;
    int a, b, c, d, ox, oy;
    rtengine::DemosaicNetXTransDomain domain;
};

const std::array<CorpusCase, 7> CORPUS_CASES{{
    {"canonical-linear-43x41", 43, 41, 1, 0, 0, 1, 0, 0, rtengine::DemosaicNetXTransDomain::LINEAR},
    {"canonical-gamma22-43x41", 43, 41, 1, 0, 0, 1, 0, 0, rtengine::DemosaicNetXTransDomain::GAMMA22},
    {"translated-linear-47x37", 47, 37, 1, 0, 0, 1, 2, 5, rtengine::DemosaicNetXTransDomain::LINEAR},
    {"rotated-gamma22-45x39", 45, 39, 0, -1, 1, 0, 1, 4, rtengine::DemosaicNetXTransDomain::GAMMA22},
    {"reflected-linear-49x35", 49, 35, -1, 0, 0, 1, 3, 2, rtengine::DemosaicNetXTransDomain::LINEAR},
    {"horizontal-seam-linear-173x31", 173, 31, 1, 0, 0, 1, 4, 1, rtengine::DemosaicNetXTransDomain::LINEAR},
    {"vertical-seam-gamma22-31x173", 31, 173, 0, 1, 1, 0, 5, 3, rtengine::DemosaicNetXTransDomain::GAMMA22},
}};

} // namespace

namespace xtrans_demosaicnet_test
{

int cfaAndContract()
{
    constexpr int matrices[8][4] = {
        { 1,  0,  0,  1}, { 0, -1,  1,  0}, {-1,  0,  0, -1}, { 0,  1, -1,  0},
        {-1,  0,  0,  1}, { 1,  0,  0, -1}, { 0,  1,  1,  0}, { 0, -1, -1,  0}
    };
    std::set<std::string> unique;
    for (const auto &matrix : matrices) {
        for (int oy = 0; oy < 6; ++oy) {
            for (int ox = 0; ox < 6; ++ox) {
                int cfa[6][6];
                cfaFor(matrix[0], matrix[1], matrix[2], matrix[3], ox, oy, cfa);
                unique.insert(cfaKey(cfa));
                rtengine::XTransCfaTransform found {};
                require(rtengine::findCanonicalXTransTransform(cfa, found), "valid X-Trans CFA was rejected");
                for (const auto &size : std::array<std::array<int, 2>, 3>{{{{1, 1}}, {{17, 9}}, {{9, 17}}}}) {
                    rtengine::XTransCfaView view(found, size[0], size[1]);
                    require(view.valid(), "canonical finite view is invalid");
                    for (int y = 0; y < size[1]; ++y) {
                        for (int x = 0; x < size[0]; ++x) {
                            int u, v, actualX, actualY;
                            view.actualToCanonical(x, y, u, v);
                            view.canonicalToActual(u, v, actualX, actualY);
                            require(actualX == x && actualY == y, "CFA coordinate round trip differs");
                            require(view.colorAtCanonical(u, v) == cfa[y % 6][x % 6],
                                "canonical CFA colour differs from the actual matrix");
                        }
                    }
                }
            }
        }
    }
    require(unique.size() == 18, "X-Trans transforms do not produce exactly 18 unique matrices");

    SyntheticModel synthetic = syntheticModel();
    synthetic.tensors[25][0] = -0.25f;
    synthetic.tensors[25][1] = 0.5f;
    synthetic.tensors[25][2] = 1.25f;
    int canonical[6][6];
    cfaFor(1, 0, 0, 1, 0, 0, canonical);
    array2D<float> raw(13, 7);
    array2D<float> red(13, 7), green(13, 7), blue(13, 7);
    raw.fill(32768.f);
    auto run = rtengine::demosaicDemosaicNetXTrans(
        raw, red, green, blue, 13, 7, canonical,
        rtengine::DemosaicNetXTransDomain::LINEAR, synthetic.model);
    require(static_cast<bool>(run), "synthetic raw wrapper failed: " + run.error.message);
    require(run.tileCount == 1 && run.workerCount == 1 && run.workspaceBytesPerWorker > 0,
        "synthetic raw wrapper accounting differs");
    for (int y = 0; y < 7; ++y) {
        for (int x = 0; x < 13; ++x) {
            require(red[y][x] == 0.f, "signed output was not clamped");
            require(green[y][x] == 32767.5f, "linear output differs");
            require(blue[y][x] == 65535.f, "over-range output was not clamped");
        }
    }

    run = rtengine::demosaicDemosaicNetXTrans(
        raw, red, green, blue, 13, 7, canonical,
        rtengine::DemosaicNetXTransDomain::GAMMA22, synthetic.model);
    require(static_cast<bool>(run), "synthetic gamma raw wrapper failed");
    const float expectedGamma = std::pow(0.5f, 2.2f) * 65535.f;
    require(std::abs(green[3][6] - expectedGamma) < 0.01f, "gamma-2.2 inverse differs");

    raw[0][0] = std::numeric_limits<float>::quiet_NaN();
    run = rtengine::demosaicDemosaicNetXTrans(
        raw, red, green, blue, 13, 7, canonical,
        rtengine::DemosaicNetXTransDomain::LINEAR, synthetic.model);
    require(!run && run.error.code == rtengine::neural::NeuralModelErrorCode::NONFINITE,
        "non-finite raw input did not return NONFINITE");
    int invalid[6][6] = {};
    run = rtengine::demosaicDemosaicNetXTrans(
        raw, red, green, blue, 13, 7, invalid,
        rtengine::DemosaicNetXTransDomain::LINEAR, synthetic.model);
    require(!run && run.error.code == rtengine::neural::NeuralModelErrorCode::SCHEMA,
        "unsupported CFA did not return SCHEMA");
    raw[0][0] = 32768.f;
    array2D<float> wrongSize(12, 7);
    run = rtengine::demosaicDemosaicNetXTrans(
        raw, wrongSize, green, blue, 13, 7, canonical,
        rtengine::DemosaicNetXTransDomain::LINEAR, synthetic.model);
    require(!run && run.error.code == rtengine::neural::NeuralModelErrorCode::SIZE,
        "mismatched RGB storage did not return SIZE");
    run = rtengine::demosaicDemosaicNetXTrans(
        raw, red, green, blue, 13, 7, canonical,
        static_cast<rtengine::DemosaicNetXTransDomain>(99), synthetic.model);
    require(!run && run.error.code == rtengine::neural::NeuralModelErrorCode::ENUM,
        "unknown raw domain did not return ENUM");
    return 0;
}

int reviewedRawWrapper()
{
    const char *path = std::getenv("GHARBI_XTRANS_RTNN");
    if (!path || !*path) {
        std::cout << "SKIP: GHARBI_XTRANS_RTNN is not set\n";
        return 77;
    }
    const auto first = rtengine::loadCachedDemosaicNetXTransModel(path);
    require(static_cast<bool>(first), "cannot load reviewed RTNN: " + first.error.message);

    std::array<std::shared_ptr<const DemosaicNetXTransModel>, 8> cached;
    std::array<std::thread, 8> threads;
    for (std::size_t index = 0; index < threads.size(); ++index) {
        threads[index] = std::thread([&, index]() {
            cached[index] = rtengine::loadCachedDemosaicNetXTransModel(path).model;
        });
    }
    for (auto &thread : threads) {
        thread.join();
    }
    for (const auto &model : cached) {
        require(model == first.model, "concurrent cache lookup did not share model identity");
    }

    for (const CorpusCase &test : CORPUS_CASES) {
        const std::string root = RT_NEURAL_RAW_WRAPPER_DIR;
        const std::string prefix = root + "/cases/" + test.id;
        const std::size_t pixels = static_cast<std::size_t>(test.width) * test.height;
        const std::vector<float> input = readFloats(prefix + ".raw.f32le", pixels);
        const std::vector<float> expected = readFloats(prefix + ".rgb.f32le", pixels * 3);
        array2D<float> raw(test.width, test.height);
        array2D<float> red(test.width, test.height);
        array2D<float> green(test.width, test.height);
        array2D<float> blue(test.width, test.height);
        std::copy(input.begin(), input.end(), static_cast<float *>(raw));
        int cfa[6][6];
        cfaFor(test.a, test.b, test.c, test.d, test.ox, test.oy, cfa);
        const auto run = rtengine::demosaicDemosaicNetXTrans(
            raw, red, green, blue, test.width, test.height, cfa, test.domain, first.model);
        require(static_cast<bool>(run), std::string(test.id) + " wrapper failed: " + run.error.message);
        const std::array<const float *, 3> actual{{red, green, blue}};
        for (std::size_t channel = 0; channel < 3; ++channel) {
            for (std::size_t pixel = 0; pixel < pixels; ++pixel) {
                const float normalized = actual[channel][pixel] / 65535.f;
                const float reference = expected[channel * pixels + pixel];
                const float tolerance = 5e-6f + 1e-5f * std::abs(reference);
                if (std::abs(normalized - reference) > tolerance) {
                    std::ostringstream message;
                    message << test.id << " differs in channel " << channel
                            << " pixel " << pixel << ": actual=" << normalized
                            << " expected=" << reference;
                    throw std::runtime_error(message.str());
                }
            }
        }
    }
    return 0;
}

} // namespace xtrans_demosaicnet_test
