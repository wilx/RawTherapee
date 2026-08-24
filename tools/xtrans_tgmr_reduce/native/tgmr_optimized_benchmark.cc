// Research-only optimized executor for the frozen TGMR32/S9/q8 model.
//
// The scalar implementation remains the correctness oracle.  This translation
// unit deliberately includes it so the benchmark cannot drift into a second
// model loader or a subtly different reference inference contract.

#define main tgmr_reference_embedded_main
#include "tgmr_benchmark.cc"
#undef main

#include <cerrno>
#include <cstdlib>
#include <immintrin.h>
#include <numeric>
#include <sstream>

namespace {

constexpr unsigned GROUPS = COMPONENTS / 8;

struct PreparedPhase {
    const Phase* source = nullptr;
    unsigned measuredPosition = 0;
    std::array<unsigned char, AREA> channels{};
    std::array<unsigned char, 3> channelCounts{};
    std::array<float, COMPONENTS> coarseScale{};
    std::array<float, COMPONENTS> fullScale{};

    // coefficient-major, eight component lanes contiguous
    std::vector<float> coarseMeans;
    std::vector<float> coarseLower;
    std::vector<float> coarseInvDiagonal;
    std::vector<float> coarseScaleAosoa;
    std::vector<float> fullInvDiagonal;
};

struct PreparedModel {
    const Model* source = nullptr;
    std::array<PreparedPhase, PHASES> phases;
    std::array<unsigned char, 36> residueToPhase{};
    std::size_t cacheBytes = 0;
};

struct PixelWork {
    std::array<float, AREA> centered{};
    std::array<unsigned char, SHORTLIST> ids{};
    std::array<float, SHORTLIST> weights{};
    std::array<std::array<float, 2>, SHORTLIST> predictions{};
    float dc = 0.f;
    float measured = 0.f;
};

struct Request {
    uint32_t pixel = 0;
    unsigned char slot = 0;

    Request() = default;
    Request(uint32_t pixelValue, unsigned char slotValue)
        : pixel(pixelValue), slot(slotValue)
    {
    }
};

struct StageTimes {
    double gather = 0.0;
    double dc = 0.0;
    double coarse = 0.0;
    double selection = 0.0;
    double fullSolve = 0.0;
    double prediction = 0.0;
    double normalize = 0.0;
    uint64_t pixels = 0;
};

inline double secondsSince(const std::chrono::steady_clock::time_point& start)
{
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
}

void stableInsert(
    std::array<float, SHORTLIST>& scores,
    std::array<unsigned char, SHORTLIST>& ids,
    float score,
    unsigned component)
{
    unsigned position = SHORTLIST;
    for (unsigned slot = 0; slot < SHORTLIST; ++slot) {
        if (score > scores[slot]
            || (score == scores[slot] && component < ids[slot])) {
            position = slot;
            break;
        }
    }
    if (position == SHORTLIST) {
        return;
    }
    for (unsigned slot = SHORTLIST - 1; slot > position; --slot) {
        scores[slot] = scores[slot - 1];
        ids[slot] = ids[slot - 1];
    }
    scores[position] = score;
    ids[position] = static_cast<unsigned char>(component);
}

void deriveResidueMap(PreparedModel& prepared)
{
    static const unsigned char cfa[6][6] = {
        {1, 2, 1, 1, 0, 1},
        {0, 1, 0, 2, 1, 2},
        {1, 2, 1, 1, 0, 1},
        {1, 0, 1, 1, 2, 1},
        {2, 1, 2, 0, 1, 0},
        {1, 0, 1, 1, 2, 1},
    };
    for (unsigned ry = 0; ry < 6; ++ry) {
        for (unsigned rx = 0; rx < 6; ++rx) {
            int match = -1;
            for (unsigned phase = 0; phase < PHASES; ++phase) {
                bool equal = true;
                for (unsigned position = 0; position < AREA; ++position) {
                    const int dy = static_cast<int>(position / PATCH) - 3;
                    const int dx = static_cast<int>(position % PATCH) - 3;
                    const unsigned yy = static_cast<unsigned>(
                        (static_cast<int>(ry) + dy + 12) % 6);
                    const unsigned xx = static_cast<unsigned>(
                        (static_cast<int>(rx) + dx + 12) % 6);
                    const unsigned observedChannel =
                        prepared.source->phases[phase].observed[position] / AREA;
                    if (observedChannel != cfa[yy][xx]) {
                        equal = false;
                        break;
                    }
                }
                if (equal) {
                    match = static_cast<int>(phase);
                    break;
                }
            }
            if (match < 0) {
                throw std::runtime_error("cannot derive X-Trans residue-to-phase map");
            }
            prepared.residueToPhase[ry * 6 + rx] =
                static_cast<unsigned char>(match);
        }
    }
}

PreparedModel prepareModel(const Model& model)
{
    PreparedModel result;
    result.source = &model;
    for (unsigned phaseIndex = 0; phaseIndex < PHASES; ++phaseIndex) {
        const Phase& phase = model.phases[phaseIndex];
        PreparedPhase& out = result.phases[phaseIndex];
        out.source = &phase;
        for (unsigned position = 0; position < AREA; ++position) {
            const unsigned channel = phase.observed[position] / AREA;
            out.channels[position] = static_cast<unsigned char>(channel);
            ++out.channelCounts[channel];
            if (phase.observed[position] % AREA == AREA / 2) {
                out.measuredPosition = position;
            }
        }

        float coarseMaximum = -std::numeric_limits<float>::infinity();
        float fullMaximum = -std::numeric_limits<float>::infinity();
        for (unsigned component = 0; component < COMPONENTS; ++component) {
            coarseMaximum = std::max(
                coarseMaximum,
                phase.logWeights[component] - 0.5f * phase.coarseLogdet[component]);
            fullMaximum = std::max(
                fullMaximum,
                (phase.logWeights[component] - 0.5f * phase.fullLogdet[component])
                    / TEMPERATURE);
        }
        for (unsigned component = 0; component < COMPONENTS; ++component) {
            out.coarseScale[component] = std::exp((
                phase.logWeights[component] - 0.5f * phase.coarseLogdet[component]
                - coarseMaximum) / 6.f);
            out.fullScale[component] = std::exp(
                (phase.logWeights[component] - 0.5f * phase.fullLogdet[component])
                    / TEMPERATURE
                - fullMaximum);
        }

        out.coarseMeans.resize(GROUPS * SUPPORT_AREA * 8);
        out.coarseLower.resize(GROUPS * SUPPORT_AREA * SUPPORT_AREA * 8);
        out.coarseInvDiagonal.resize(GROUPS * SUPPORT_AREA * 8);
        out.coarseScaleAosoa.resize(GROUPS * 8);
        out.fullInvDiagonal.resize(COMPONENTS * AREA);
        for (unsigned group = 0; group < GROUPS; ++group) {
            for (unsigned lane = 0; lane < 8; ++lane) {
                const unsigned component = group * 8 + lane;
                out.coarseScaleAosoa[group * 8 + lane] =
                    out.coarseScale[component];
                for (unsigned position = 0; position < AREA; ++position) {
                    out.fullInvDiagonal[component * AREA + position] = 1.f
                        / phase.fullCholesky[
                            (component * AREA + position) * AREA + position];
                }
                for (unsigned row = 0; row < SUPPORT_AREA; ++row) {
                    const unsigned position = phase.coarsePositions[row];
                    out.coarseMeans[(group * SUPPORT_AREA + row) * 8 + lane] =
                        phase.meansObserved[component * AREA + position];
                    for (unsigned column = 0; column < SUPPORT_AREA; ++column) {
                        out.coarseLower[
                            ((group * SUPPORT_AREA + row) * SUPPORT_AREA + column)
                                * 8
                            + lane] = phase.coarseCholesky[
                            (component * SUPPORT_AREA + row) * SUPPORT_AREA
                            + column];
                    }
                    out.coarseInvDiagonal[
                        (group * SUPPORT_AREA + row) * 8 + lane] = 1.f
                        / phase.coarseCholesky[
                            (component * SUPPORT_AREA + row) * SUPPORT_AREA + row];
                }
            }
        }
        result.cacheBytes +=
            (out.coarseMeans.size() + out.coarseLower.size()
             + out.coarseInvDiagonal.size() + out.coarseScaleAosoa.size()
             + out.fullInvDiagonal.size())
            * sizeof(float);
    }
    deriveResidueMap(result);
    return result;
}

void centerObservation(
    const PreparedPhase& phase,
    const std::array<float, AREA>& observed,
    PixelWork& work)
{
    std::array<float, 3> sums = {{0.f, 0.f, 0.f}};
    for (unsigned position = 0; position < AREA; ++position) {
        sums[phase.channels[position]] += observed[position];
    }
    work.dc = 0.f;
    for (unsigned channel = 0; channel < 3; ++channel) {
        work.dc += sums[channel] / phase.channelCounts[channel];
    }
    work.dc /= 3.f;
    for (unsigned position = 0; position < AREA; ++position) {
        work.centered[position] = observed[position] - work.dc;
    }
    work.measured = observed[phase.measuredPosition];
}

void coarseScalar(const PreparedPhase& prepared, PixelWork& work)
{
    const Phase& phase = *prepared.source;
    std::array<float, SHORTLIST> bestScores;
    bestScores.fill(-std::numeric_limits<float>::infinity());
    work.ids.fill(255);
    for (unsigned component = 0; component < COMPONENTS; ++component) {
        std::array<float, SUPPORT_AREA> residual{};
        for (unsigned row = 0; row < SUPPORT_AREA; ++row) {
            const unsigned position = phase.coarsePositions[row];
            residual[row] = work.centered[position]
                - phase.meansObserved[component * AREA + position];
        }
        std::array<float, SUPPORT_AREA> solved{};
        const float q = quadratic<SUPPORT_AREA>(
            phase.coarseCholesky.data()
                + component * SUPPORT_AREA * SUPPORT_AREA,
            residual, solved);
        const float score = prepared.coarseScale[component] / (1.f + q / 3.f);
        stableInsert(bestScores, work.ids, score, component);
    }
}

#if defined(__AVX2__)
void coarseAvx2(const PreparedPhase& prepared, PixelWork& work)
{
    const Phase& phase = *prepared.source;
    alignas(32) float allScores[COMPONENTS];
    for (unsigned group = 0; group < GROUPS; ++group) {
        __m256 solved[SUPPORT_AREA];
        __m256 q = _mm256_setzero_ps();
        for (unsigned row = 0; row < SUPPORT_AREA; ++row) {
            const unsigned position = phase.coarsePositions[row];
            __m256 value = _mm256_sub_ps(
                _mm256_set1_ps(work.centered[position]),
                _mm256_loadu_ps(
                    prepared.coarseMeans.data()
                    + (group * SUPPORT_AREA + row) * 8));
            for (unsigned column = 0; column < row; ++column) {
                const __m256 coefficient = _mm256_loadu_ps(
                    prepared.coarseLower.data()
                    + ((group * SUPPORT_AREA + row) * SUPPORT_AREA + column) * 8);
                value = _mm256_fnmadd_ps(coefficient, solved[column], value);
            }
            solved[row] = _mm256_mul_ps(
                value,
                _mm256_loadu_ps(
                    prepared.coarseInvDiagonal.data()
                    + (group * SUPPORT_AREA + row) * 8));
            q = _mm256_fmadd_ps(solved[row], solved[row], q);
        }
        const __m256 denominator = _mm256_fmadd_ps(
            q, _mm256_set1_ps(1.f / 3.f), _mm256_set1_ps(1.f));
        const __m256 score = _mm256_div_ps(
            _mm256_loadu_ps(prepared.coarseScaleAosoa.data() + group * 8),
            denominator);
        _mm256_store_ps(allScores + group * 8, score);
    }
    std::array<float, SHORTLIST> bestScores;
    bestScores.fill(-std::numeric_limits<float>::infinity());
    work.ids.fill(255);
    for (unsigned component = 0; component < COMPONENTS; ++component) {
        stableInsert(bestScores, work.ids, allScores[component], component);
    }
}
#else
void coarseAvx2(const PreparedPhase& prepared, PixelWork& work)
{
    coarseScalar(prepared, work);
}
#endif

inline float positiveWeight(float scale, float quadraticValue)
{
    const float s = 1.f + quadraticValue / 3.f;
    const float s2 = s * s;
    const float s4 = s2 * s2;
    return scale / (s4 * s2 * std::sqrt(s));
}

void fullScalarSlot(
    const PreparedPhase& prepared,
    PixelWork& work,
    unsigned slot)
{
    const Phase& phase = *prepared.source;
    const unsigned component = work.ids[slot];
    std::array<float, AREA> residual{};
    for (unsigned position = 0; position < AREA; ++position) {
        residual[position] = work.centered[position]
            - phase.meansObserved[component * AREA + position];
    }
    std::array<float, AREA> solved{};
    const float q = quadratic<AREA>(
        phase.fullCholesky.data() + component * AREA * AREA,
        residual, solved);
    work.weights[slot] = positiveWeight(prepared.fullScale[component], q);
    for (unsigned target = 0; target < 2; ++target) {
        float value = phase.meansTarget[component * 2 + target] + work.dc;
        const float* gain = phase.gains.data()
            + (component * 2 + target) * AREA;
        for (unsigned position = 0; position < AREA; ++position) {
            value += gain[position] * residual[position];
        }
        work.predictions[slot][target] = value;
    }
}

std::array<float, 3> finishPixel(
    const PreparedPhase& prepared,
    const PixelWork& work)
{
    float total = 0.f;
    for (unsigned slot = 0; slot < SHORTLIST; ++slot) {
        total += work.weights[slot];
    }
    if (!(total > 0.f) || !std::isfinite(total)) {
        throw std::runtime_error("specialized posterior has zero/non-finite weight");
    }
    std::array<float, 3> output = {{0.f, 0.f, 0.f}};
    for (unsigned target = 0; target < 2; ++target) {
        float value = 0.f;
        for (unsigned slot = 0; slot < SHORTLIST; ++slot) {
            value += work.predictions[slot][target] * (work.weights[slot] / total);
        }
        output[prepared.source->targets[target]] = value;
    }
    output[prepared.source->sampled] = work.measured;
    return output;
}

std::array<float, 3> predictSpecialized(
    const PreparedPhase& phase,
    const std::array<float, AREA>& observed,
    bool avxCoarse)
{
    PixelWork work;
    centerObservation(phase, observed, work);
    if (avxCoarse) {
        coarseAvx2(phase, work);
    } else {
        coarseScalar(phase, work);
    }
    for (unsigned slot = 0; slot < SHORTLIST; ++slot) {
        fullScalarSlot(phase, work, slot);
    }
    return finishPixel(phase, work);
}

struct ShortlistAudit {
    unsigned scalarMismatches = 0;
    unsigned avxMismatches = 0;
    unsigned cases = 0;
};

std::array<unsigned char, SHORTLIST> referenceShortlist(
    const PreparedPhase& prepared,
    const PixelWork& work)
{
    const Phase& phase = *prepared.source;
    std::array<std::pair<float, unsigned>, COMPONENTS> scores;
    for (unsigned component = 0; component < COMPONENTS; ++component) {
        std::array<float, SUPPORT_AREA> residual{};
        for (unsigned row = 0; row < SUPPORT_AREA; ++row) {
            const unsigned position = phase.coarsePositions[row];
            residual[row] = work.centered[position]
                - phase.meansObserved[component * AREA + position];
        }
        std::array<float, SUPPORT_AREA> solved{};
        const float q = quadratic<SUPPORT_AREA>(
            phase.coarseCholesky.data()
                + component * SUPPORT_AREA * SUPPORT_AREA,
            residual, solved);
        scores[component] = std::make_pair(
            phase.logWeights[component]
                + logDensity(q, phase.coarseLogdet[component], SUPPORT_AREA),
            component);
    }
    std::partial_sort(
        scores.begin(), scores.begin() + SHORTLIST, scores.end(),
        [](const std::pair<float, unsigned>& left,
           const std::pair<float, unsigned>& right) {
            return left.first != right.first ? left.first > right.first
                                             : left.second < right.second;
        });
    std::array<unsigned char, SHORTLIST> result{};
    for (unsigned slot = 0; slot < SHORTLIST; ++slot) {
        result[slot] = static_cast<unsigned char>(scores[slot].second);
    }
    return result;
}

ShortlistAudit auditShortlists(const PreparedModel& model)
{
    ShortlistAudit result;
    for (unsigned phase = 0; phase < PHASES; ++phase) {
        for (unsigned probe = 0; probe < 64; ++probe) {
            const std::array<float, AREA> input = makeInput(
                static_cast<std::size_t>(phase) * 64 + probe, phase);
            PixelWork referenceWork;
            centerObservation(model.phases[phase], input, referenceWork);
            const auto referenceIds = referenceShortlist(
                model.phases[phase], referenceWork);
            PixelWork scalarWork = referenceWork;
            coarseScalar(model.phases[phase], scalarWork);
            PixelWork avxWork = referenceWork;
            coarseAvx2(model.phases[phase], avxWork);
            result.scalarMismatches += scalarWork.ids != referenceIds;
            result.avxMismatches += avxWork.ids != referenceIds;
            ++result.cases;
        }
    }
    return result;
}

#if defined(__AVX2__)
void fullAvx2Group(
    const PreparedPhase& prepared,
    std::vector<PixelWork>& work,
    const Request* requests,
    unsigned count,
    unsigned component)
{
    const Phase& phase = *prepared.source;
    unsigned begin = 0;
    for (; begin + 8 <= count; begin += 8) {
        __m256 residual[AREA];
        __m256 solved[AREA];
        __m256 q = _mm256_setzero_ps();
        alignas(32) float lanes[8];
        for (unsigned position = 0; position < AREA; ++position) {
            for (unsigned lane = 0; lane < 8; ++lane) {
                lanes[lane] = work[requests[begin + lane].pixel].centered[position];
            }
            residual[position] = _mm256_sub_ps(
                _mm256_load_ps(lanes),
                _mm256_set1_ps(
                    phase.meansObserved[component * AREA + position]));
            __m256 value = residual[position];
            const float* matrixRow = phase.fullCholesky.data()
                + (component * AREA + position) * AREA;
            for (unsigned column = 0; column < position; ++column) {
                value = _mm256_fnmadd_ps(
                    _mm256_set1_ps(matrixRow[column]), solved[column], value);
            }
            solved[position] = _mm256_mul_ps(
                value,
                _mm256_set1_ps(
                    prepared.fullInvDiagonal[component * AREA + position]));
            q = _mm256_fmadd_ps(solved[position], solved[position], q);
        }
        const __m256 s = _mm256_fmadd_ps(
            q, _mm256_set1_ps(1.f / 3.f), _mm256_set1_ps(1.f));
        const __m256 s2 = _mm256_mul_ps(s, s);
        const __m256 s4 = _mm256_mul_ps(s2, s2);
        const __m256 denominator = _mm256_mul_ps(
            _mm256_mul_ps(s4, s2), _mm256_sqrt_ps(s));
        const __m256 weights = _mm256_div_ps(
            _mm256_set1_ps(prepared.fullScale[component]), denominator);
        _mm256_store_ps(lanes, weights);
        for (unsigned lane = 0; lane < 8; ++lane) {
            const Request& request = requests[begin + lane];
            work[request.pixel].weights[request.slot] = lanes[lane];
        }
        for (unsigned target = 0; target < 2; ++target) {
            for (unsigned lane = 0; lane < 8; ++lane) {
                lanes[lane] = phase.meansTarget[component * 2 + target]
                    + work[requests[begin + lane].pixel].dc;
            }
            __m256 value = _mm256_load_ps(lanes);
            const float* gain = phase.gains.data()
                + (component * 2 + target) * AREA;
            for (unsigned position = 0; position < AREA; ++position) {
                value = _mm256_fmadd_ps(
                    _mm256_set1_ps(gain[position]), residual[position], value);
            }
            _mm256_store_ps(lanes, value);
            for (unsigned lane = 0; lane < 8; ++lane) {
                const Request& request = requests[begin + lane];
                work[request.pixel].predictions[request.slot][target] = lanes[lane];
            }
        }
    }
    for (; begin < count; ++begin) {
        const Request& request = requests[begin];
        fullScalarSlot(prepared, work[request.pixel], request.slot);
    }
}
#else
void fullAvx2Group(
    const PreparedPhase& prepared,
    std::vector<PixelWork>& work,
    const Request* requests,
    unsigned count,
    unsigned)
{
    for (unsigned index = 0; index < count; ++index) {
        const Request& request = requests[index];
        fullScalarSlot(prepared, work[request.pixel], request.slot);
    }
}
#endif

template <typename IndexAccessor, typename InputAccessor>
void processBucketedChunk(
    const PreparedPhase& phase,
    unsigned count,
    IndexAccessor indexAt,
    InputAccessor inputAt,
    std::vector<std::array<float, 3>>& outputs)
{
    // One cache per OpenMP worker/instantiation; capacity is reused between
    // phase/chunk jobs, so no allocation occurs in the per-pixel hot path.
    static thread_local std::vector<PixelWork> work;
    work.resize(count);
    std::array<unsigned, COMPONENTS> counts{};
    for (unsigned local = 0; local < count; ++local) {
        centerObservation(phase, inputAt(local), work[local]);
        coarseAvx2(phase, work[local]);
        for (unsigned slot = 0; slot < SHORTLIST; ++slot) {
            ++counts[work[local].ids[slot]];
        }
    }
    std::array<unsigned, COMPONENTS + 1> offsets{};
    for (unsigned component = 0; component < COMPONENTS; ++component) {
        offsets[component + 1] = offsets[component] + counts[component];
    }
    static thread_local std::vector<Request> requests;
    requests.resize(count * SHORTLIST);
    std::array<unsigned, COMPONENTS> cursor{};
    for (unsigned component = 0; component < COMPONENTS; ++component) {
        cursor[component] = offsets[component];
    }
    for (unsigned local = 0; local < count; ++local) {
        for (unsigned slot = 0; slot < SHORTLIST; ++slot) {
            const unsigned component = work[local].ids[slot];
            requests[cursor[component]++] = Request(
                static_cast<uint32_t>(local), static_cast<unsigned char>(slot));
        }
    }
    for (unsigned component = 0; component < COMPONENTS; ++component) {
        fullAvx2Group(
            phase, work, requests.data() + offsets[component],
            counts[component], component);
    }
    for (unsigned local = 0; local < count; ++local) {
        outputs[indexAt(local)] = finishPixel(phase, work[local]);
    }
}

std::array<std::vector<std::size_t>, PHASES> makeRows(
    std::size_t count,
    const std::vector<unsigned char>* phaseIds)
{
    std::array<std::vector<std::size_t>, PHASES> rows;
    for (std::size_t index = 0; index < count; ++index) {
        const unsigned phase = phaseIds ? (*phaseIds)[index]
                                        : static_cast<unsigned>(index % PHASES);
        rows[phase].push_back(index);
    }
    return rows;
}

struct Job {
    unsigned phase = 0;
    unsigned begin = 0;
    unsigned end = 0;

    Job() = default;
    Job(unsigned phaseValue, unsigned beginValue, unsigned endValue)
        : phase(phaseValue), begin(beginValue), end(endValue)
    {
    }
};

std::vector<Job> makeJobs(
    const std::array<std::vector<std::size_t>, PHASES>& rows,
    unsigned chunk)
{
    std::vector<Job> jobs;
    for (unsigned phase = 0; phase < PHASES; ++phase) {
        for (unsigned begin = 0; begin < rows[phase].size(); begin += chunk) {
            jobs.push_back(Job(
                phase, begin,
                static_cast<unsigned>(std::min<std::size_t>(
                    rows[phase].size(), static_cast<std::size_t>(begin) + chunk))));
        }
    }
    return jobs;
}

double runSpecialized(
    const PreparedModel& model,
    const std::vector<std::array<float, AREA>>& inputs,
    std::vector<std::array<float, 3>>& outputs,
    unsigned threads,
    unsigned chunk,
    bool avx,
    bool bucketed,
    bool persistent,
    const std::vector<unsigned char>* phaseIds)
{
    const auto rows = makeRows(inputs.size(), phaseIds);
#ifdef _OPENMP
    omp_set_num_threads(static_cast<int>(threads));
#else
    (void)threads;
#endif
    const auto start = std::chrono::steady_clock::now();
    if (persistent) {
#pragma omp parallel
        {
            for (int phase = 0; phase < static_cast<int>(PHASES); ++phase) {
#pragma omp for schedule(static)
                for (int row = 0; row < static_cast<int>(rows[phase].size()); ++row) {
                    const std::size_t index = rows[phase][row];
                    outputs[index] = predictSpecialized(
                        model.phases[phase], inputs[index], avx);
                }
            }
        }
    } else {
        const std::vector<Job> jobs = makeJobs(rows, chunk);
#pragma omp parallel for schedule(static)
        for (int jobIndex = 0; jobIndex < static_cast<int>(jobs.size()); ++jobIndex) {
            const Job& job = jobs[jobIndex];
            const PreparedPhase& phase = model.phases[job.phase];
            if (bucketed) {
                const unsigned count = job.end - job.begin;
                processBucketedChunk(
                    phase, count,
                    [&](unsigned local) { return rows[job.phase][job.begin + local]; },
                    [&](unsigned local) -> const std::array<float, AREA>& {
                        return inputs[rows[job.phase][job.begin + local]];
                    },
                    outputs);
            } else {
                for (unsigned row = job.begin; row < job.end; ++row) {
                    const std::size_t index = rows[job.phase][row];
                    outputs[index] = predictSpecialized(phase, inputs[index], avx);
                }
            }
        }
    }
    return secondsSince(start);
}

unsigned mirrorIndex(int coordinate, unsigned extent)
{
    if (extent == 1) {
        return 0;
    }
    while (coordinate < 0 || coordinate >= static_cast<int>(extent)) {
        if (coordinate < 0) {
            coordinate = -coordinate;
        } else {
            coordinate = 2 * static_cast<int>(extent) - 2 - coordinate;
        }
    }
    return static_cast<unsigned>(coordinate);
}

float mosaicValue(unsigned x, unsigned y, unsigned width)
{
    const uint32_t bits = mix(
        static_cast<uint32_t>(static_cast<std::size_t>(y) * width + x)
        ^ 0x517cc1b7U);
    return 0.05f + 0.9f * static_cast<float>(bits & 0x00ffffffU)
        / static_cast<float>(0x01000000U);
}

std::array<float, AREA> gatherObservation(
    const PreparedPhase& phase,
    const std::vector<float>& mosaic,
    unsigned width,
    unsigned height,
    unsigned x,
    unsigned y)
{
    std::array<float, AREA> observed{};
    for (unsigned observedPosition = 0; observedPosition < AREA; ++observedPosition) {
        const unsigned spatial = phase.source->observed[observedPosition] % AREA;
        const int dx = static_cast<int>(spatial % PATCH) - 3;
        const int dy = static_cast<int>(spatial / PATCH) - 3;
        const unsigned xx = mirrorIndex(static_cast<int>(x) + dx, width);
        const unsigned yy = mirrorIndex(static_cast<int>(y) + dy, height);
        observed[observedPosition] = mosaic[static_cast<std::size_t>(yy) * width + xx];
    }
    return observed;
}

double runStreaming(
    const PreparedModel& model,
    const std::vector<float>& mosaic,
    unsigned width,
    unsigned height,
    std::vector<std::array<float, 3>>& outputs,
    unsigned threads,
    unsigned tile,
    unsigned chunk,
    StageTimes* profile)
{
    struct TileJob {
        unsigned x0, y0, x1, y1;
    };
    std::vector<TileJob> tiles;
    for (unsigned y = 0; y < height; y += tile) {
        for (unsigned x = 0; x < width; x += tile) {
            tiles.push_back({
                x, y, std::min(width, x + tile), std::min(height, y + tile)});
        }
    }
#ifdef _OPENMP
    omp_set_num_threads(static_cast<int>(threads));
#else
    (void)threads;
#endif
    const auto start = std::chrono::steady_clock::now();
#pragma omp parallel for schedule(static)
    for (int tileIndex = 0; tileIndex < static_cast<int>(tiles.size()); ++tileIndex) {
        const TileJob& tileJob = tiles[tileIndex];
        struct Coordinate {
            unsigned x;
            unsigned y;
            std::size_t output;
        };
        std::array<std::vector<Coordinate>, PHASES> rows;
        for (unsigned y = tileJob.y0; y < tileJob.y1; ++y) {
            for (unsigned x = tileJob.x0; x < tileJob.x1; ++x) {
                const unsigned phase = model.residueToPhase[(y % 6) * 6 + x % 6];
                rows[phase].push_back(
                    {x, y, static_cast<std::size_t>(y) * width + x});
            }
        }
        double gatherTime = 0.0;
        double inferenceTime = 0.0;
        for (unsigned phase = 0; phase < PHASES; ++phase) {
            for (unsigned begin = 0; begin < rows[phase].size(); begin += chunk) {
                const unsigned end = static_cast<unsigned>(std::min<std::size_t>(
                    rows[phase].size(), static_cast<std::size_t>(begin) + chunk));
                std::vector<std::array<float, AREA>> observations(end - begin);
                const auto gatherStart = std::chrono::steady_clock::now();
                for (unsigned local = 0; local < end - begin; ++local) {
                    const Coordinate& coordinate = rows[phase][begin + local];
                    observations[local] = gatherObservation(
                        model.phases[phase], mosaic, width, height,
                        coordinate.x, coordinate.y);
                }
                gatherTime += secondsSince(gatherStart);
                const auto inferenceStart = std::chrono::steady_clock::now();
                processBucketedChunk(
                    model.phases[phase], end - begin,
                    [&](unsigned local) { return rows[phase][begin + local].output; },
                    [&](unsigned local) -> const std::array<float, AREA>& {
                        return observations[local];
                    },
                    outputs);
                inferenceTime += secondsSince(inferenceStart);
            }
        }
        if (profile) {
#pragma omp atomic
            profile->gather += gatherTime;
#pragma omp atomic
            profile->fullSolve += inferenceTime;
        }
    }
    return secondsSince(start);
}

double runStreamingReference(
    const PreparedModel& model,
    const std::vector<float>& mosaic,
    unsigned width,
    unsigned height,
    std::vector<std::array<float, 3>>& outputs,
    unsigned threads,
    unsigned tile)
{
    struct TileJob {
        unsigned x0, y0, x1, y1;
    };
    std::vector<TileJob> tiles;
    for (unsigned y = 0; y < height; y += tile) {
        for (unsigned x = 0; x < width; x += tile) {
            tiles.push_back({
                x, y, std::min(width, x + tile), std::min(height, y + tile)});
        }
    }
#ifdef _OPENMP
    omp_set_num_threads(static_cast<int>(threads));
#else
    (void)threads;
#endif
    const auto start = std::chrono::steady_clock::now();
#pragma omp parallel for schedule(static)
    for (int tileIndex = 0; tileIndex < static_cast<int>(tiles.size()); ++tileIndex) {
        const TileJob& tileJob = tiles[tileIndex];
        for (unsigned y = tileJob.y0; y < tileJob.y1; ++y) {
            for (unsigned x = tileJob.x0; x < tileJob.x1; ++x) {
                const unsigned phase = model.residueToPhase[(y % 6) * 6 + x % 6];
                const std::array<float, AREA> observed = gatherObservation(
                    model.phases[phase], mosaic, width, height, x, y);
                outputs[static_cast<std::size_t>(y) * width + x] = predict(
                    *model.phases[phase].source, observed);
            }
        }
    }
    return secondsSince(start);
}

void writeOutputs(
    const std::string& path,
    const std::vector<std::array<float, 3>>& outputs)
{
    if (path.empty()) {
        return;
    }
    std::ofstream stream(path.c_str(), std::ios::binary | std::ios::trunc);
    if (!stream) {
        throw std::runtime_error("cannot create output dump");
    }
    stream.write(
        reinterpret_cast<const char*>(outputs.data()),
        static_cast<std::streamsize>(outputs.size() * sizeof(outputs[0])));
    if (!stream) {
        throw std::runtime_error("cannot write output dump");
    }
}

double checksum(const std::vector<std::array<float, 3>>& outputs)
{
    double result = 0.0;
    for (const auto& value : outputs) {
        result += value[0] + value[1] + value[2];
    }
    return result;
}

const char* compilerIdentity()
{
#if defined(__GNUC__)
    return __VERSION__;
#else
    return "unknown";
#endif
}

struct Arguments {
    std::string model;
    std::string variant;
    unsigned width = 0;
    unsigned height = 0;
    unsigned threads = 1;
    unsigned repetitions = 5;
    unsigned chunk = 512;
    unsigned tile = 256;
    std::string output;
    std::string input;
    bool profile = false;
};

Arguments parseArguments(int argc, char** argv)
{
    if (argc < 7) {
        throw std::runtime_error(
            "usage: tgmr_optimized_benchmark MODEL VARIANT WIDTH HEIGHT THREADS REPETITIONS [--chunk N] [--tile N] [--output FILE]");
    }
    Arguments result;
    result.model = argv[1];
    result.variant = argv[2];
    result.width = parseUnsigned(argv[3]);
    result.height = parseUnsigned(argv[4]);
    result.threads = parseUnsigned(argv[5]);
    result.repetitions = parseUnsigned(argv[6]);
    for (int index = 7; index < argc; ++index) {
        const std::string option = argv[index];
        if ((option == "--chunk" || option == "--tile" || option == "--output"
             || option == "--input")
            && index + 1 >= argc) {
            throw std::runtime_error("missing option value");
        }
        if (option == "--chunk") {
            result.chunk = parseUnsigned(argv[++index]);
        } else if (option == "--tile") {
            result.tile = parseUnsigned(argv[++index]);
        } else if (option == "--output") {
            result.output = argv[++index];
        } else if (option == "--input") {
            result.input = argv[++index];
        } else if (option == "--profile") {
            result.profile = true;
        } else {
            throw std::runtime_error("unknown option: " + option);
        }
    }
    return result;
}

void loadInputFile(
    const std::string& path,
    std::size_t expectedCount,
    std::vector<std::array<float, AREA>>& inputs,
    std::vector<unsigned char>& phases)
{
    Reader reader(path);
    static const char expectedMagic[8] = {'X', 'T', 'G', 'R', 'O', 'B', 'S', '1'};
    reader.require(8);
    if (std::memcmp(reader.bytes.data(), expectedMagic, 8) != 0) {
        throw std::runtime_error("wrong observation-file magic");
    }
    reader.offset = 8;
    const uint32_t count = reader.u32();
    if (count != expectedCount) {
        throw std::runtime_error("observation-file count does not match geometry");
    }
    inputs.resize(count);
    phases.resize(count);
    for (unsigned index = 0; index < count; ++index) {
        const uint32_t phase = reader.u32();
        if (phase >= PHASES) {
            throw std::runtime_error("invalid observation-file phase");
        }
        phases[index] = static_cast<unsigned char>(phase);
        readFloats(reader, inputs[index]);
    }
    if (reader.offset != reader.bytes.size()) {
        throw std::runtime_error("noncanonical observation-file trailing bytes");
    }
}

} // namespace

int main(int argc, char** argv)
{
    try {
        const Arguments arguments = parseArguments(argc, argv);
        const std::size_t count =
            static_cast<std::size_t>(arguments.width) * arguments.height;
        if (count / arguments.width != arguments.height) {
            throw std::runtime_error("image size overflow");
        }
        const Model model = loadModel(arguments.model);
        const PreparedModel prepared = prepareModel(model);
        const ShortlistAudit shortlistAudit = auditShortlists(prepared);
        std::vector<std::array<float, 3>> outputs(count);
        std::vector<std::array<float, AREA>> inputs;
        std::vector<unsigned char> inputPhases;
        std::vector<float> mosaic;
        const bool streaming = arguments.variant == "o5-stream"
            || arguments.variant == "o5-reference";
        if (streaming) {
            mosaic.resize(count);
            for (unsigned y = 0; y < arguments.height; ++y) {
                for (unsigned x = 0; x < arguments.width; ++x) {
                    mosaic[static_cast<std::size_t>(y) * arguments.width + x] =
                        mosaicValue(x, y, arguments.width);
                }
            }
        } else {
            if (!arguments.input.empty()) {
                loadInputFile(arguments.input, count, inputs, inputPhases);
            } else {
                inputs.resize(count);
                for (std::size_t index = 0; index < count; ++index) {
                    inputs[index] = makeInput(
                        index, static_cast<unsigned>(index % PHASES));
                }
            }
        }
        const auto execute = [&]() {
            if (arguments.variant == "o0-reference") {
                if (!inputPhases.empty()) {
                    throw std::runtime_error(
                        "the embedded O0 reference accepts generated phase order only");
                }
                return run(model, inputs, outputs, arguments.threads);
            }
            if (arguments.variant == "o1-specialized") {
                return runSpecialized(
                    prepared, inputs, outputs, 1, arguments.chunk,
                    false, false, false,
                    inputPhases.empty() ? nullptr : &inputPhases);
            }
            if (arguments.variant == "o2-chunk") {
                return runSpecialized(
                    prepared, inputs, outputs, arguments.threads, arguments.chunk,
                    false, false, false,
                    inputPhases.empty() ? nullptr : &inputPhases);
            }
            if (arguments.variant == "o2-persistent") {
                return runSpecialized(
                    prepared, inputs, outputs, arguments.threads, arguments.chunk,
                    false, false, true,
                    inputPhases.empty() ? nullptr : &inputPhases);
            }
            if (arguments.variant == "o3-avx-f1") {
                return runSpecialized(
                    prepared, inputs, outputs, 1, arguments.chunk,
                    true, false, false,
                    inputPhases.empty() ? nullptr : &inputPhases);
            }
            if (arguments.variant == "o4-avx-f1") {
                return runSpecialized(
                    prepared, inputs, outputs, arguments.threads, arguments.chunk,
                    true, false, false,
                    inputPhases.empty() ? nullptr : &inputPhases);
            }
            if (arguments.variant == "o4-avx-f2") {
                return runSpecialized(
                    prepared, inputs, outputs, arguments.threads, arguments.chunk,
                    true, true, false,
                    inputPhases.empty() ? nullptr : &inputPhases);
            }
            if (arguments.variant == "o5-stream") {
                return runStreaming(
                    prepared, mosaic, arguments.width, arguments.height, outputs,
                    arguments.threads, arguments.tile, arguments.chunk, nullptr);
            }
            if (arguments.variant == "o5-reference") {
                return runStreamingReference(
                    prepared, mosaic, arguments.width, arguments.height, outputs,
                    arguments.threads, arguments.tile);
            }
            throw std::runtime_error("unknown variant");
        };
        execute();
        std::vector<double> timings;
        for (unsigned repetition = 0; repetition < arguments.repetitions; ++repetition) {
            timings.push_back(execute());
        }
        std::sort(timings.begin(), timings.end());
        const double median = timings[timings.size() / 2];
        StageTimes profile;
        if (arguments.profile && arguments.variant == "o5-stream") {
            runStreaming(
                prepared, mosaic, arguments.width, arguments.height, outputs,
                arguments.threads, arguments.tile, arguments.chunk, &profile);
        }
        writeOutputs(arguments.output, outputs);
        rusage usage{};
        getrusage(RUSAGE_SELF, &usage);
        std::cout << std::setprecision(12)
            << "{\"avx2\":"
#if defined(__AVX2__)
            << "true"
#else
            << "false"
#endif
            << ",\"checksum\":" << checksum(outputs)
            << ",\"chunk\":" << arguments.chunk
            << ",\"compiler\":\"" << compilerIdentity() << "\""
            << ",\"height\":" << arguments.height
            << ",\"inference_worker_seconds\":" << profile.fullSolve
            << ",\"matrix_bytes\":" << inputs.size() * sizeof(inputs[0])
            << ",\"median_seconds\":" << median
            << ",\"megapixels_per_second\":" << count / median / 1e6
            << ",\"model_aosoa_bytes\":" << prepared.cacheBytes
            << ",\"peak_rss_kib\":" << usage.ru_maxrss
            << ",\"pixels\":" << count
            << ",\"repetition_max_seconds\":" << timings.back()
            << ",\"repetition_min_seconds\":" << timings.front()
            << ",\"gather_worker_seconds\":" << profile.gather
            << ",\"shortlist_avx_mismatches\":" << shortlistAudit.avxMismatches
            << ",\"shortlist_cases\":" << shortlistAudit.cases
            << ",\"shortlist_scalar_mismatches\":"
            << shortlistAudit.scalarMismatches
            << ",\"threads\":" << arguments.threads
            << ",\"tile\":" << arguments.tile
            << ",\"variant\":\"" << arguments.variant << "\""
            << ",\"width\":" << arguments.width << "}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 2;
    }
}
