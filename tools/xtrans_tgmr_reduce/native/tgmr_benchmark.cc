// Research-only native executor for the frozen TGMR32/S9/q8 experiment.
//
// This is intentionally not part of rtengine or the RawTherapee build.  It
// measures the selected statistical contract without adding a demosaicing
// method, model search, profile setting, or user-visible runtime dependency.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <sys/resource.h>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

constexpr unsigned PATCH = 7;
constexpr unsigned AREA = 49;
constexpr unsigned PHASES = 18;
constexpr unsigned COMPONENTS = 32;
constexpr unsigned SUPPORT_AREA = 9;
constexpr unsigned SHORTLIST = 8;
constexpr float NU = 3.f;
constexpr float TEMPERATURE = 4.f;

struct Reader {
    explicit Reader(const std::string& path)
    {
        std::ifstream stream(path.c_str(), std::ios::binary);
        if (!stream) {
            throw std::runtime_error("cannot open model");
        }
        stream.seekg(0, std::ios::end);
        const std::streamoff size = stream.tellg();
        stream.seekg(0, std::ios::beg);
        if (size <= 0) {
            throw std::runtime_error("empty model");
        }
        bytes.resize(static_cast<std::size_t>(size));
        stream.read(reinterpret_cast<char*>(bytes.data()), size);
        if (!stream) {
            throw std::runtime_error("short model read");
        }
    }

    uint32_t u32()
    {
        require(4);
        const uint32_t value =
            static_cast<uint32_t>(bytes[offset])
            | (static_cast<uint32_t>(bytes[offset + 1]) << 8)
            | (static_cast<uint32_t>(bytes[offset + 2]) << 16)
            | (static_cast<uint32_t>(bytes[offset + 3]) << 24);
        offset += 4;
        return value;
    }

    float f32()
    {
        const uint32_t bits = u32();
        float value = 0.f;
        std::memcpy(&value, &bits, sizeof(value));
        if (!std::isfinite(value)) {
            throw std::runtime_error("non-finite model value");
        }
        return value;
    }

    void require(std::size_t count) const
    {
        if (count > bytes.size() - offset) {
            throw std::runtime_error("truncated model");
        }
    }

    std::vector<unsigned char> bytes;
    std::size_t offset = 0;
};

struct Phase {
    std::array<uint32_t, AREA> observed;
    uint32_t sampled = 0;
    std::array<uint32_t, 2> targets;
    std::array<float, COMPONENTS> logWeights;
    std::vector<float> meansObserved;
    std::vector<float> meansTarget;
    std::vector<float> fullCholesky;
    std::array<float, COMPONENTS> fullLogdet;
    std::vector<float> gains;
    std::array<uint32_t, SUPPORT_AREA> coarsePositions;
    std::vector<float> coarseCholesky;
    std::array<float, COMPONENTS> coarseLogdet;
};

struct Model {
    std::array<Phase, PHASES> phases;
};

template <typename Container>
void readFloats(Reader& reader, Container& values)
{
    for (auto& value : values) {
        value = reader.f32();
    }
}

Model loadModel(const std::string& path)
{
    Reader reader(path);
    const char expected[8] = {'X', 'T', 'G', 'R', 'R', 'E', 'D', '1'};
    reader.require(8);
    if (std::memcmp(reader.bytes.data(), expected, 8) != 0) {
        throw std::runtime_error("wrong model magic");
    }
    reader.offset = 8;
    const uint32_t version = reader.u32();
    const uint32_t patch = reader.u32();
    const uint32_t phases = reader.u32();
    const uint32_t components = reader.u32();
    const uint32_t support = reader.u32();
    const uint32_t shortlist = reader.u32();
    const uint32_t reserved = reader.u32();
    if (version != 1 || patch != PATCH || phases != PHASES
        || components != COMPONENTS || support != 3
        || shortlist != SHORTLIST || reserved != 0) {
        throw std::runtime_error("unsupported model contract");
    }
    Model model;
    for (Phase& phase : model.phases) {
        for (uint32_t& value : phase.observed) {
            value = reader.u32();
            if (value >= 3 * AREA) {
                throw std::runtime_error("invalid observed index");
            }
        }
        phase.sampled = reader.u32();
        phase.targets[0] = reader.u32();
        phase.targets[1] = reader.u32();
        if (phase.sampled > 2 || phase.targets[0] > 2 || phase.targets[1] > 2) {
            throw std::runtime_error("invalid channel contract");
        }
        readFloats(reader, phase.logWeights);
        phase.meansObserved.resize(COMPONENTS * AREA);
        phase.meansTarget.resize(COMPONENTS * 2);
        phase.fullCholesky.resize(COMPONENTS * AREA * AREA);
        phase.gains.resize(COMPONENTS * 2 * AREA);
        readFloats(reader, phase.meansObserved);
        readFloats(reader, phase.meansTarget);
        readFloats(reader, phase.fullCholesky);
        readFloats(reader, phase.fullLogdet);
        readFloats(reader, phase.gains);
        for (uint32_t& value : phase.coarsePositions) {
            value = reader.u32();
            if (value >= AREA) {
                throw std::runtime_error("invalid coarse index");
            }
        }
        phase.coarseCholesky.resize(COMPONENTS * SUPPORT_AREA * SUPPORT_AREA);
        readFloats(reader, phase.coarseCholesky);
        readFloats(reader, phase.coarseLogdet);
    }
    if (reader.offset != reader.bytes.size()) {
        throw std::runtime_error("noncanonical trailing model bytes");
    }
    return model;
}

float logDensity(float quadratic, float logdet, unsigned dimension)
{
    const double common =
        std::lgamma(0.5 * (NU + dimension)) - std::lgamma(0.5 * NU)
        - 0.5 * dimension * std::log(NU * 3.14159265358979323846);
    return static_cast<float>(
        common - 0.5 * logdet
        - 0.5 * (NU + dimension) * std::log1p(quadratic / NU));
}

template <unsigned N>
float quadratic(
    const float* lower,
    const std::array<float, N>& residual,
    std::array<float, N>& solved)
{
    float result = 0.f;
    for (unsigned row = 0; row < N; ++row) {
        float value = residual[row];
        const float* matrixRow = lower + row * N;
        for (unsigned column = 0; column < row; ++column) {
            value -= matrixRow[column] * solved[column];
        }
        solved[row] = value / matrixRow[row];
        result += solved[row] * solved[row];
    }
    return result;
}

std::array<float, 3> predict(
    const Phase& phase,
    const std::array<float, AREA>& observed)
{
    std::array<float, 3> channelSums = {{0.f, 0.f, 0.f}};
    std::array<unsigned, 3> channelCounts = {{0, 0, 0}};
    unsigned measuredPosition = 0;
    for (unsigned position = 0; position < AREA; ++position) {
        const unsigned channel = phase.observed[position] / AREA;
        channelSums[channel] += observed[position];
        ++channelCounts[channel];
        if (phase.observed[position] % AREA == AREA / 2) {
            measuredPosition = position;
        }
    }
    float dc = 0.f;
    for (unsigned channel = 0; channel < 3; ++channel) {
        dc += channelSums[channel] / channelCounts[channel];
    }
    dc /= 3.f;
    std::array<float, AREA> centered;
    for (unsigned position = 0; position < AREA; ++position) {
        centered[position] = observed[position] - dc;
    }

    std::array<std::pair<float, unsigned>, COMPONENTS> coarseScores;
    for (unsigned component = 0; component < COMPONENTS; ++component) {
        std::array<float, SUPPORT_AREA> residual;
        for (unsigned index = 0; index < SUPPORT_AREA; ++index) {
            const unsigned position = phase.coarsePositions[index];
            residual[index] = centered[position]
                - phase.meansObserved[component * AREA + position];
        }
        std::array<float, SUPPORT_AREA> solved;
        const float q = quadratic<SUPPORT_AREA>(
            phase.coarseCholesky.data()
                + component * SUPPORT_AREA * SUPPORT_AREA,
            residual, solved);
        coarseScores[component] = std::make_pair(
            phase.logWeights[component]
                + logDensity(q, phase.coarseLogdet[component], SUPPORT_AREA),
            component);
    }
    std::partial_sort(
        coarseScores.begin(), coarseScores.begin() + SHORTLIST, coarseScores.end(),
        [](const std::pair<float, unsigned>& left,
           const std::pair<float, unsigned>& right) {
            return left.first != right.first ? left.first > right.first
                                             : left.second < right.second;
        });

    std::array<float, SHORTLIST> scores;
    std::array<std::array<float, 2>, SHORTLIST> predictions;
    float maximum = -std::numeric_limits<float>::infinity();
    for (unsigned slot = 0; slot < SHORTLIST; ++slot) {
        const unsigned component = coarseScores[slot].second;
        std::array<float, AREA> residual;
        for (unsigned position = 0; position < AREA; ++position) {
            residual[position] = centered[position]
                - phase.meansObserved[component * AREA + position];
        }
        std::array<float, AREA> solved;
        const float q = quadratic<AREA>(
            phase.fullCholesky.data() + component * AREA * AREA,
            residual, solved);
        scores[slot] = (
            phase.logWeights[component]
            + logDensity(q, phase.fullLogdet[component], AREA)) / TEMPERATURE;
        maximum = std::max(maximum, scores[slot]);
        for (unsigned target = 0; target < 2; ++target) {
            float value = phase.meansTarget[component * 2 + target] + dc;
            const float* gain = phase.gains.data()
                + (component * 2 + target) * AREA;
            for (unsigned position = 0; position < AREA; ++position) {
                value += gain[position] * residual[position];
            }
            predictions[slot][target] = value;
        }
    }
    std::array<float, SHORTLIST> weights;
    float total = 0.f;
    for (unsigned slot = 0; slot < SHORTLIST; ++slot) {
        weights[slot] = std::exp(scores[slot] - maximum);
        total += weights[slot];
    }
    std::array<float, 3> output = {{0.f, 0.f, 0.f}};
    for (unsigned target = 0; target < 2; ++target) {
        float value = 0.f;
        for (unsigned slot = 0; slot < SHORTLIST; ++slot) {
            value += predictions[slot][target] * (weights[slot] / total);
        }
        output[phase.targets[target]] = value;
    }
    output[phase.sampled] = observed[measuredPosition];
    return output;
}

uint32_t mix(uint32_t value)
{
    value ^= value >> 16;
    value *= 0x7feb352dU;
    value ^= value >> 15;
    value *= 0x846ca68bU;
    return value ^ (value >> 16);
}

std::array<float, AREA> makeInput(std::size_t pixel, unsigned phase)
{
    std::array<float, AREA> result;
    for (unsigned index = 0; index < AREA; ++index) {
        const uint32_t bits = mix(
            static_cast<uint32_t>(pixel) ^ (index * 0x9e3779b9U)
            ^ (phase * 0x85ebca6bU));
        result[index] = 0.05f + 0.9f * static_cast<float>(bits & 0x00ffffffU)
            / static_cast<float>(0x01000000U);
    }
    return result;
}

double run(
    const Model& model,
    const std::vector<std::array<float, AREA>>& inputs,
    std::vector<std::array<float, 3>>& outputs,
    unsigned threads)
{
    std::array<std::vector<std::size_t>, PHASES> rows;
    for (std::size_t index = 0; index < inputs.size(); ++index) {
        rows[index % PHASES].push_back(index);
    }
#ifdef _OPENMP
    omp_set_num_threads(static_cast<int>(threads));
#else
    (void)threads;
#endif
    const auto start = std::chrono::steady_clock::now();
#pragma omp parallel for schedule(static)
    for (int phase = 0; phase < static_cast<int>(PHASES); ++phase) {
        for (const std::size_t index : rows[phase]) {
            outputs[index] = predict(model.phases[phase], inputs[index]);
        }
    }
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(end - start).count();
}

unsigned parseUnsigned(const char* value)
{
    const unsigned long parsed = std::stoul(value);
    if (parsed == 0 || parsed > std::numeric_limits<unsigned>::max()) {
        throw std::runtime_error("invalid numeric argument");
    }
    return static_cast<unsigned>(parsed);
}

} // namespace

int main(int argc, char** argv)
{
    try {
        if (argc != 6 && argc != 7) {
            std::cerr << "usage: tgmr_benchmark MODEL WIDTH HEIGHT THREADS REPETITIONS [OUTPUT.f32]\n";
            return 2;
        }
        const unsigned width = parseUnsigned(argv[2]);
        const unsigned height = parseUnsigned(argv[3]);
        const unsigned threads = parseUnsigned(argv[4]);
        const unsigned repetitions = parseUnsigned(argv[5]);
        const std::size_t count = static_cast<std::size_t>(width) * height;
        if (count / width != height) {
            throw std::runtime_error("image size overflow");
        }
        const Model model = loadModel(argv[1]);
        std::vector<std::array<float, AREA>> inputs(count);
        std::vector<std::array<float, 3>> outputs(count);
        for (std::size_t index = 0; index < count; ++index) {
            inputs[index] = makeInput(index, static_cast<unsigned>(index % PHASES));
        }
        run(model, inputs, outputs, threads); // warm-up
        std::vector<double> timings;
        for (unsigned repetition = 0; repetition < repetitions; ++repetition) {
            timings.push_back(run(model, inputs, outputs, threads));
        }
        std::sort(timings.begin(), timings.end());
        const double median = timings[timings.size() / 2];
        double checksum = 0.0;
        for (const auto& rgb : outputs) {
            checksum += rgb[0] + rgb[1] + rgb[2];
        }
        if (argc == 7) {
            std::ofstream dump(argv[6], std::ios::binary | std::ios::trunc);
            if (!dump) {
                throw std::runtime_error("cannot create output dump");
            }
            dump.write(
                reinterpret_cast<const char*>(outputs.data()),
                static_cast<std::streamsize>(outputs.size() * sizeof(outputs[0])));
            if (!dump) {
                throw std::runtime_error("cannot write output dump");
            }
        }
        const rusage usage = [] {
            rusage value{};
            getrusage(RUSAGE_SELF, &value);
            return value;
        }();
        const double mpps = count / median / 1e6;
        std::cout << std::setprecision(12)
            << "{\"checksum\":" << checksum
            << ",\"height\":" << height
            << ",\"median_seconds\":" << median
            << ",\"megapixels_per_second\":" << mpps
            << ",\"peak_rss_kib\":" << usage.ru_maxrss
            << ",\"pixels\":" << count
            << ",\"threads\":" << threads
            << ",\"width\":" << width << "}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 2;
    }
}
