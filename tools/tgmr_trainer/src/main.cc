#include "tgmr/camera_matrices.h"
#include "tgmr/corpus.h"
#include "tgmr/corpus_analysis.h"
#include "tgmr/manifest.h"
#include "tgmr/model_v2.h"
#include "tgmr/patch_selection.h"
#include "tgmr/sha256.h"
#include "tgmr/training.h"
#include "tgmr/validation.h"
#include "tgmr/xtrans_training.h"
#include "cJSON.h"

#include <array>
#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

void usage(std::ostream &output)
{
    output
        << "rt-tgmr-train - RawTherapee X-Trans Student-t GMR tooling\n\n"
        << "Usage:\n"
        << "  rt-tgmr-train corpus verify-sources MANIFEST CACHE\n"
        << "  rt-tgmr-train corpus validate-manifest MANIFEST\n"
        << "  rt-tgmr-train corpus matrices\n"
        << "  rt-tgmr-train corpus classify-file SOURCE-ID IMAGE\n"
        << "  rt-tgmr-train corpus classify INPUT CACHE OUTPUT.jsonl"
           " [--candidates|--open-images-cvdf SPLIT|--all]"
           " [--proxy] [--allow-failures] [--jobs N] [--work-dir DIR]"
           " [--checkpoint-images N] [--checkpoint-seconds N]"
           " [--progress-seconds N] [--retries N] [--force]\n"
        << "  rt-tgmr-train corpus select CANDIDATES.jsonl RECIPE.json OUTPUT.jsonl"
           " [--force]\n"
        << "  rt-tgmr-train corpus propose-patches MANIFEST CACHE OUTPUT.jsonl [--force]\n"
        << "  rt-tgmr-train corpus finalize MANIFEST CACHE OUTPUT.jsonl"
           " [--work-dir DIR] [--progress-seconds N] [--force]\n"
        << "  rt-tgmr-train corpus pack MANIFEST CACHE OUTPUT.tgpc"
           " [--training-augmentation production-v1|identity-only]"
           " [--noise none|sensor-v1] [--work-dir DIR]"
           " [--checkpoint-records N] [--progress-seconds N] [--force]\n"
        << "  rt-tgmr-train corpus release-files MANIFEST ATTRIBUTION.txt"
           " RIGHTS.json RECONSTRUCTION.tsv [--force]\n"
        << "  rt-tgmr-train corpus inspect FILE\n"
        << "  rt-tgmr-train corpus balance FILE [--training-min N] [--eval-min N]"
           " [--fixed-v1-thresholds]\n"
        << "  rt-tgmr-train corpus report FILE [--sources]"
           " [--fixed-v1-thresholds] [--json P] [--csv P] [--html P] [--force]\n"
        << "  rt-tgmr-train corpus gzip INPUT.tgpc OUTPUT.tgpc.gz [--level N] [--force]\n"
        << "  rt-tgmr-train export --legacy-v1 INPUT OUTPUT --corpus-sha256 HEX\n"
        << "      --configuration-sha256 HEX --attribution-sha256 HEX\n"
        << "      --trainer-revision-sha256 HEX [--model-revision N] [--force]\n"
        << "  rt-tgmr-train verify MODEL.tgmr\n"
        << "  rt-tgmr-train validate MODEL.tgmr CORPUS.tgpc[.gz]"
           " [--split validation|test|train] [--limit N]\n"
        << "  rt-tgmr-train benchmark CORPUS.tgpc[.gz] [--source-limit N] [options]\n"
        << "  rt-tgmr-train train CORPUS.tgpc[.gz] OUTPUT-DIR [--source-limit N] [options]\n"
        << "  rt-tgmr-train resume CORPUS CHECKPOINT OUTPUT [options]\n"
        << "  rt-tgmr-train export --checkpoints DIR OUTPUT [identity options]"
           " [--validation-report FILE]\n"
        << "  rt-tgmr-train --version\n\n"
        << "The production commands corpus verify-sources, classify, finalize, pack,\n"
        << "balance, report, release-files, train, resume, export, verify, and benchmark are\n"
        << "part of the same standalone executable.\n";
}

void writeTextAtomic(const std::string &path, const std::string &contents, bool force)
{
    if (!force && std::filesystem::exists(path)) {
        throw std::runtime_error("refusing to replace output: " + path);
    }
    const std::string temporary = path + ".tmp";
    {
        std::ofstream output(temporary, std::ios::binary);
        output.write(contents.data(), static_cast<std::streamsize>(contents.size()));
        output.flush();
        if (!output) {
            output.close();
            std::filesystem::remove(temporary);
            throw std::runtime_error("cannot write output: " + path);
        }
    }
    if (force) {
        std::error_code ignored;
        std::filesystem::remove(path, ignored);
    }
    std::filesystem::rename(temporary, path);
}

tgmr::TrainingBackend parseBackend(const std::string &value)
{
    if (value == "canonical") return tgmr::TrainingBackend::CANONICAL;
    if (value == "cpu") return tgmr::TrainingBackend::CPU;
    if (value == "omp-target") return tgmr::TrainingBackend::OMP_TARGET;
    throw std::runtime_error("backend must be canonical, cpu, or omp-target");
}

const char *backendName(tgmr::TrainingBackend value)
{
    switch (value) {
        case tgmr::TrainingBackend::CANONICAL: return "canonical";
        case tgmr::TrainingBackend::CPU: return "cpu";
        case tgmr::TrainingBackend::OMP_TARGET: return "omp-target";
    }
    return "unknown";
}

std::string phasePath(
    const std::filesystem::path &directory,
    std::size_t phase,
    const std::string &suffix)
{
    std::ostringstream name;
    name << "phase-" << std::setfill('0') << std::setw(2) << phase << '-' << suffix << ".tgmrc";
    return (directory / name.str()).string();
}

tgmr::FitConfiguration parseFitOptions(
    int argc, char **argv, int begin, bool &force, int &selectedPhase,
    std::uint32_t *additionalGaussian = nullptr,
    std::uint32_t *additionalStudent = nullptr)
{
    tgmr::FitConfiguration configuration;
    force = false;
    selectedPhase = -1;
    for (int index = begin; index < argc; ++index) {
        const std::string option = argv[index];
        auto value = [&]() -> std::string {
            if (++index >= argc) throw std::runtime_error(option + " requires a value");
            return argv[index];
        };
        if (option == "--backend") {
            configuration.backend = parseBackend(value());
        } else if (option == "--components") {
            configuration.components = std::stoul(value());
        } else if (option == "--gaussian-iterations") {
            configuration.gaussianIterations = std::stoul(value());
        } else if (option == "--student-iterations") {
            configuration.studentIterations = std::stoul(value());
        } else if (option == "--additional-gaussian" && additionalGaussian) {
            *additionalGaussian = std::stoul(value());
        } else if (option == "--additional-student" && additionalStudent) {
            *additionalStudent = std::stoul(value());
        } else if (option == "--covariance-floor") {
            configuration.covarianceFloor = std::stod(value());
        } else if (option == "--degrees-of-freedom") {
            configuration.degreesOfFreedom = std::stod(value());
        } else if (option == "--seed") {
            configuration.seed = std::stoull(value(), nullptr, 0);
        } else if (option == "--batch-size") {
            configuration.batchSize = std::stoull(value());
        } else if (option == "--maximum-memory-mib") {
            configuration.maximumBytes = std::stoull(value()) * 1024ULL * 1024ULL;
        } else if (option == "--source-limit") {
            configuration.sourceLimit = std::stoull(value());
        } else if (option == "--phase") {
            selectedPhase = std::stoi(value());
        } else if (option == "--device") {
            configuration.device = std::stoi(value());
        } else if (option == "--gpu-yield-ms") {
            configuration.gpuYieldMilliseconds = std::stoul(value());
        } else if (option == "--force") {
            force = true;
        } else {
            throw std::runtime_error("unknown fitting option: " + option);
        }
    }
    return configuration;
}

int trainCommand(int argc, char **argv)
{
    if (argc < 4) {
        throw std::runtime_error("train requires a corpus and output directory");
    }
    bool force = false;
    int selectedPhase = -1;
    const tgmr::FitConfiguration configuration = parseFitOptions(
        argc, argv, 4, force, selectedPhase);
    if (selectedPhase < -1 || selectedPhase >= 18) {
        throw std::runtime_error("--phase must be between 0 and 17");
    }
    const std::filesystem::path directory(argv[3]);
    std::filesystem::create_directories(directory);
    const auto phases = tgmr::xtransPhaseContracts();
    for (std::size_t phase = 0; phase < phases.size(); ++phase) {
        if (selectedPhase >= 0 && phase != static_cast<std::size_t>(selectedPhase)) {
            continue;
        }
        std::uint64_t sampleCount = 0;
        std::array<std::uint8_t, 32> corpusDigest{};
        std::vector<double> matrix = tgmr::loadPhaseTrainingMatrix(
            argv[2], phases[phase], sampleCount, corpusDigest,
            configuration.sourceLimit);
        auto checkpoint = [&](const tgmr::MixtureModel &model,
                              const char *stage, std::uint32_t iteration) {
            std::ostringstream suffix;
            suffix << (std::string(stage) == "gaussian-em" ? "gaussian" : "student")
                   << '-' << std::setfill('0') << std::setw(3) << iteration;
            tgmr::PhaseCheckpoint state;
            state.phase = phases[phase];
            state.sampleCount = sampleCount;
            state.configuration = configuration;
            state.model = model;
            state.corpusPayloadSha256 = corpusDigest;
            tgmr::writePhaseCheckpoint(
                phasePath(directory, phase, suffix.str()), state, force);
            std::cerr << "phase=" << phase << " stage=" << stage
                      << " iteration=" << iteration
                      << " mean_log_likelihood=" << model.meanLogLikelihood << '\n';
        };
        const tgmr::MixtureModel model = tgmr::fitStudentTMixture(
            matrix, sampleCount, 51, configuration, checkpoint);
        tgmr::PhaseCheckpoint final;
        final.phase = phases[phase];
        final.sampleCount = sampleCount;
        final.configuration = configuration;
        final.model = model;
        final.corpusPayloadSha256 = corpusDigest;
        tgmr::writePhaseCheckpoint(
            phasePath(directory, phase, "final"), final, force);
    }
    std::cout << "training complete: backend=" << backendName(configuration.backend)
              << " output=" << directory.string() << '\n';
    return 0;
}

int resumeCommand(int argc, char **argv)
{
    if (argc < 5) {
        throw std::runtime_error("resume requires CORPUS CHECKPOINT OUTPUT");
    }
    tgmr::PhaseCheckpoint checkpoint = tgmr::readPhaseCheckpoint(argv[3]);
    std::uint32_t additionalGaussian = 0;
    std::uint32_t additionalStudent = 0;
    bool force = false;
    int selectedPhase = -1;
    tgmr::FitConfiguration configuration = parseFitOptions(
        argc, argv, 5, force, selectedPhase, &additionalGaussian, &additionalStudent);
    if (selectedPhase >= 0 && selectedPhase != static_cast<int>(checkpoint.phase.index)) {
        throw std::runtime_error("resume --phase differs from checkpoint phase");
    }
    configuration.components = checkpoint.model.components;
    configuration.covarianceFloor = checkpoint.configuration.covarianceFloor;
    configuration.degreesOfFreedom = checkpoint.configuration.degreesOfFreedom;
    configuration.seed = checkpoint.configuration.seed;
    if (configuration.sourceLimit != 0
        && configuration.sourceLimit != checkpoint.configuration.sourceLimit) {
        throw std::runtime_error("resume --source-limit differs from checkpoint");
    }
    configuration.sourceLimit = checkpoint.configuration.sourceLimit;
    std::uint64_t sampleCount = 0;
    std::array<std::uint8_t, 32> corpusDigest{};
    const auto matrix = tgmr::loadPhaseTrainingMatrix(
        argv[2], checkpoint.phase, sampleCount, corpusDigest,
        configuration.sourceLimit);
    if (sampleCount != checkpoint.sampleCount
        || corpusDigest != checkpoint.corpusPayloadSha256) {
        throw std::runtime_error("resume corpus identity differs from checkpoint");
    }
    const tgmr::PhaseContract phaseContract = checkpoint.phase;
    const std::uint64_t checkpointSamples = checkpoint.sampleCount;
    const std::array<std::uint8_t, 32> checkpointCorpus =
        checkpoint.corpusPayloadSha256;
    const std::string progressPath = std::string(argv[4]) + ".progress.tgmrc";
    auto progress = [&](const tgmr::MixtureModel &model,
                        const char *stage, std::uint32_t iteration) {
        tgmr::PhaseCheckpoint state;
        state.phase = phaseContract;
        state.sampleCount = checkpointSamples;
        state.configuration = configuration;
        state.model = model;
        state.corpusPayloadSha256 = checkpointCorpus;
        tgmr::writePhaseCheckpoint(progressPath, state, true);
        std::cerr << "resume phase=" << phaseContract.index << " stage=" << stage
                  << " iteration=" << iteration
                  << " mean_log_likelihood=" << model.meanLogLikelihood << '\n';
    };
    checkpoint.configuration = configuration;
    checkpoint.model = tgmr::continueStudentTMixture(
        matrix, sampleCount, configuration, std::move(checkpoint.model),
        additionalGaussian, additionalStudent, progress);
    tgmr::writePhaseCheckpoint(argv[4], checkpoint, force);
    std::cout << "resume complete: gaussian_iterations="
              << checkpoint.model.gaussianIterations
              << " student_iterations=" << checkpoint.model.studentIterations << '\n';
    return 0;
}

int benchmarkCommand(int argc, char **argv)
{
    if (argc < 3) throw std::runtime_error("benchmark requires a TGPC corpus");
    tgmr::FitConfiguration configuration;
    configuration.backend = tgmr::TrainingBackend::CPU;
    configuration.gaussianIterations = 1;
    configuration.studentIterations = 1;
    std::size_t requestedSamples = 0;
    int phase = 0;
    for (int index = 3; index < argc; ++index) {
        const std::string option = argv[index];
        auto value = [&]() -> std::string {
            if (++index >= argc) throw std::runtime_error(option + " requires a value");
            return argv[index];
        };
        if (option == "--backend") configuration.backend = parseBackend(value());
        else if (option == "--samples") requestedSamples = std::stoull(value());
        else if (option == "--phase") phase = std::stoi(value());
        else if (option == "--components") configuration.components = std::stoull(value());
        else if (option == "--gaussian-iterations") configuration.gaussianIterations = std::stoul(value());
        else if (option == "--student-iterations") configuration.studentIterations = std::stoul(value());
        else if (option == "--batch-size") configuration.batchSize = std::stoull(value());
        else if (option == "--device") configuration.device = std::stoi(value());
        else if (option == "--gpu-yield-ms") configuration.gpuYieldMilliseconds = std::stoul(value());
        else if (option == "--maximum-memory-mib") {
            configuration.maximumBytes = std::stoull(value()) * 1024ULL * 1024ULL;
        } else if (option == "--source-limit") {
            configuration.sourceLimit = std::stoull(value());
        } else if (option == "--seed") configuration.seed = std::stoull(value(), nullptr, 0);
        else throw std::runtime_error("unknown benchmark option: " + option);
    }
    if (phase < 0 || phase >= 18) throw std::runtime_error("benchmark phase must be 0..17");
    std::uint64_t available = 0;
    std::array<std::uint8_t, 32> corpusDigest{};
    auto matrix = tgmr::loadPhaseTrainingMatrix(
        argv[2], tgmr::xtransPhaseContracts()[phase], available, corpusDigest,
        configuration.sourceLimit);
    const std::size_t samples = requestedSamples == 0
        ? static_cast<std::size_t>(available)
        : std::min<std::size_t>(requestedSamples, available);
    if (samples < configuration.components) {
        throw std::runtime_error("benchmark corpus has fewer samples than components");
    }
    matrix.resize(samples * 51);
    const auto start = std::chrono::steady_clock::now();
    const auto model = tgmr::fitStudentTMixture(
        matrix, samples, 51, configuration);
    const double seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    std::cout << std::fixed << std::setprecision(6)
        << "{\n"
        << "  \"backend\": \"" << backendName(configuration.backend) << "\",\n"
        << "  \"components\": " << configuration.components << ",\n"
        << "  \"corpus_payload_sha256\": \"" << tgmr::hex(corpusDigest) << "\",\n"
        << "  \"dimension\": 51,\n"
        << "  \"elapsed_seconds\": " << seconds << ",\n"
        << "  \"format\": \"rawtherapee-tgmr-training-benchmark-v1\",\n"
        << "  \"gaussian_iterations\": " << configuration.gaussianIterations << ",\n"
        << "  \"mean_log_likelihood\": " << model.meanLogLikelihood << ",\n"
        << "  \"phase\": " << phase << ",\n"
        << "  \"samples\": " << samples << ",\n"
        << "  \"source_limit\": " << configuration.sourceLimit << ",\n"
        << "  \"student_iterations\": " << configuration.studentIterations << "\n"
        << "}\n";
    return 0;
}

std::vector<std::uint8_t> readFile(const std::string &path)
{
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("cannot open input: " + path);
    }
    stream.seekg(0, std::ios::end);
    const std::streamoff size = stream.tellg();
    stream.seekg(0, std::ios::beg);
    if (size < 0 || static_cast<std::uint64_t>(size) > 64ULL * 1024 * 1024) {
        throw std::runtime_error("input size is invalid or excessive");
    }
    std::vector<std::uint8_t> result(static_cast<std::size_t>(size));
    stream.read(reinterpret_cast<char *>(result.data()), size);
    if (!stream) {
        throw std::runtime_error("short input read");
    }
    return result;
}

int verifyCommand(int argc, char **argv)
{
    if (argc != 3) {
        throw std::runtime_error("verify requires one TGMR v2 model path");
    }
    std::cout << tgmr::canonicalModelV2Json(tgmr::inspectModelV2(readFile(argv[2])));
    return 0;
}

int validateCommand(int argc, char **argv)
{
    if (argc < 4) {
        throw std::runtime_error("validate requires MODEL and CORPUS");
    }
    tgmr::CorpusSplit split = tgmr::CorpusSplit::VALIDATION;
    std::uint64_t limit = 0;
    for (int index = 4; index < argc; ++index) {
        const std::string option = argv[index];
        if (option == "--limit" && index + 1 < argc) {
            limit = std::stoull(argv[++index]);
        } else if (option == "--split" && index + 1 < argc) {
            const std::string value = argv[++index];
            if (value == "train") split = tgmr::CorpusSplit::TRAIN;
            else if (value == "validation") split = tgmr::CorpusSplit::VALIDATION;
            else if (value == "test") split = tgmr::CorpusSplit::TEST;
            else throw std::runtime_error("validate --split must be train, validation, or test");
        } else {
            throw std::runtime_error("unknown validate option: " + option);
        }
    }
    std::cout << tgmr::canonicalValidationJson(
        tgmr::validateModelOnCorpus(argv[2], argv[3], split, limit));
    return 0;
}

std::string canonicalTrainingManifest(
    const tgmr::ModelV2Inspection &inspection,
    const std::array<tgmr::PhaseCheckpoint, 18> *phases,
    const std::string *validation)
{
    std::ostringstream output;
    output.imbue(std::locale::classic());
    output << std::fixed << std::setprecision(12)
        << "{\n"
        << "  \"architecture\": \"XTRANS_TGMR_K32_S9_Q8\",\n"
        << "  \"artifact_bytes\": " << inspection.fileBytes << ",\n"
        << "  \"artifact_sha256\": \"" << tgmr::hex(inspection.fileSha256) << "\",\n"
        << "  \"attribution_sha256\": \""
        << tgmr::hex(inspection.identity.attributionSha256) << "\",\n"
        << "  \"corpus_sha256\": \"" << tgmr::hex(inspection.identity.corpusSha256) << "\",\n"
        << "  \"format\": \"rawtherapee-tgmr-training-manifest-v1\",\n"
        << "  \"model_revision\": " << inspection.identity.modelRevision << ",\n"
        << "  \"payload_sha256\": \"" << tgmr::hex(inspection.payloadSha256) << "\",\n"
        << "  \"phases\": [";
    if (phases) {
        output << '\n';
        for (std::size_t phase = 0; phase < phases->size(); ++phase) {
            const auto &checkpoint = (*phases)[phase];
            output << "    {\"backend\": \"" << backendName(checkpoint.configuration.backend)
                << "\", \"component_populations\": [";
            for (std::size_t component = 0;
                 component < checkpoint.model.effectiveCounts.size(); ++component) {
                if (component) output << ", ";
                output << checkpoint.model.effectiveCounts[component];
            }
            output << "], \"gaussian_iterations\": " << checkpoint.model.gaussianIterations
                << ", \"index\": " << phase
                << ", \"mean_log_likelihood\": " << checkpoint.model.meanLogLikelihood
                << ", \"samples\": " << checkpoint.sampleCount
                << ", \"source_limit\": " << checkpoint.configuration.sourceLimit
                << ", \"student_iterations\": " << checkpoint.model.studentIterations
                << '}' << (phase + 1 == phases->size() ? "\n" : ",\n");
        }
        output << "  ";
    }
    output << "],\n"
        << "  \"trainer_configuration_sha256\": \""
        << tgmr::hex(inspection.identity.trainerConfigurationSha256) << "\",\n"
        << "  \"trainer_revision_sha256\": \""
        << tgmr::hex(inspection.identity.trainerRevisionSha256) << "\",\n"
        << "  \"validation\": ";
    if (!validation) {
        output << "null\n";
    } else {
        std::string value = *validation;
        if (!value.empty() && value.back() == '\n') value.pop_back();
        for (std::size_t index = 0; index < value.size(); ++index) {
            output << value[index];
            if (value[index] == '\n') output << "  ";
        }
        output << '\n';
    }
    output
        << "}\n";
    return output.str();
}

std::string checkedValidationReport(
    const std::string &path,
    const tgmr::ModelV2Inspection &inspection)
{
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("cannot open TGMR validation report");
    std::string text{std::istreambuf_iterator<char>(stream),
                     std::istreambuf_iterator<char>()};
    if (text.empty() || text.size() > 1024 * 1024 || text.back() != '\n') {
        throw std::runtime_error("TGMR validation report encoding is noncanonical");
    }
    const char *parseEnd = nullptr;
    cJSON *root = cJSON_ParseWithOpts(text.c_str(), &parseEnd, true);
    if (!root || !cJSON_IsObject(root)) {
        cJSON_Delete(root);
        throw std::runtime_error("TGMR validation report is not JSON object data");
    }
    auto exactText = [&](const char *key, const std::string &expected) {
        const cJSON *value = cJSON_GetObjectItemCaseSensitive(root, key);
        return cJSON_IsString(value) && value->valuestring
            && expected == value->valuestring;
    };
    bool finiteNumbers = true;
    auto number = [&](const char *key) {
        const cJSON *value = cJSON_GetObjectItemCaseSensitive(root, key);
        if (!cJSON_IsNumber(value) || !std::isfinite(value->valuedouble)) {
            finiteNumbers = false;
            return 0.0;
        }
        return value->valuedouble;
    };
    const double medianSourcePsnr = number("median_source_psnr");
    const double mse = number("mse");
    const double patchRmsP99 = number("patch_rms_p99");
    const double patches = number("patches");
    const double psnr = number("psnr");
    const double requestedLimit = number("requested_limit");
    const double scalarValues = number("scalar_values");
    const double worstPatchRms = number("worst_patch_rms");
    const cJSON *phasePsnr = cJSON_GetObjectItemCaseSensitive(root, "phase_psnr");
    std::array<double, 18> phaseValues{};
    bool finitePhases = cJSON_IsArray(phasePsnr)
        && cJSON_GetArraySize(phasePsnr) == 18;
    if (finitePhases) {
        for (int phase = 0; phase < 18; ++phase) {
            const cJSON *value = cJSON_GetArrayItem(phasePsnr, phase);
            finitePhases = finitePhases && cJSON_IsNumber(value)
                && std::isfinite(value->valuedouble);
            if (finitePhases) phaseValues[phase] = value->valuedouble;
        }
    }
    const bool valid = exactText("format", "rawtherapee-tgmr-validation-report-v1")
        && exactText("model_sha256", tgmr::hex(inspection.fileSha256))
        && exactText("corpus_payload_sha256",
                     tgmr::hex(inspection.identity.corpusSha256))
        && exactText("split", "validation")
        && finiteNumbers && requestedLimit == 0.0
        && patches > 0.0 && patches == std::floor(patches)
        && scalarValues > 0.0 && scalarValues == std::floor(scalarValues)
        && mse >= 0.0 && patchRmsP99 >= 0.0 && worstPatchRms >= 0.0
        && finitePhases;
    if (!valid) {
        cJSON_Delete(root);
        throw std::runtime_error(
            "TGMR validation report is not bound to the exported model and corpus");
    }
    std::ostringstream canonical;
    canonical.imbue(std::locale::classic());
    canonical << std::fixed << std::setprecision(12)
        << "{\n"
        << "  \"corpus_payload_sha256\": \""
        << tgmr::hex(inspection.identity.corpusSha256) << "\",\n"
        << "  \"format\": \"rawtherapee-tgmr-validation-report-v1\",\n"
        << "  \"median_source_psnr\": " << medianSourcePsnr << ",\n"
        << "  \"model_sha256\": \"" << tgmr::hex(inspection.fileSha256) << "\",\n"
        << "  \"mse\": " << mse << ",\n"
        << "  \"patch_rms_p99\": " << patchRmsP99 << ",\n"
        << "  \"patches\": " << static_cast<std::uint64_t>(patches) << ",\n"
        << "  \"phase_psnr\": [";
    for (unsigned phase = 0; phase < phaseValues.size(); ++phase) {
        if (phase) canonical << ", ";
        canonical << phaseValues[phase];
    }
    canonical << "],\n"
        << "  \"psnr\": " << psnr << ",\n"
        << "  \"requested_limit\": 0,\n"
        << "  \"scalar_values\": " << static_cast<std::uint64_t>(scalarValues) << ",\n"
        << "  \"split\": \"validation\",\n"
        << "  \"worst_patch_rms\": " << worstPatchRms << "\n"
        << "}\n";
    cJSON_Delete(root);
    return canonical.str();
}

int exportCommand(int argc, char **argv)
{
    if (argc < 5 || (std::string(argv[2]) != "--legacy-v1"
                     && std::string(argv[2]) != "--checkpoints")) {
        throw std::runtime_error("export requires --legacy-v1 INPUT or --checkpoints DIR, then OUTPUT");
    }
    const bool checkpoints = std::string(argv[2]) == "--checkpoints";
    const std::string input = argv[3];
    const std::string output = argv[4];
    tgmr::ModelV2Identity identity;
    bool force = false;
    bool corpus = false;
    bool configuration = false;
    bool attribution = false;
    bool revision = false;
    std::string validationReportPath;
    for (int index = 5; index < argc; ++index) {
        const std::string option = argv[index];
        auto digest = [&](std::array<std::uint8_t, 32> &target, bool &seen) {
            if (++index >= argc) {
                throw std::runtime_error(option + " requires a SHA-256 value");
            }
            target = tgmr::parseSha256(argv[index]);
            seen = true;
        };
        if (option == "--corpus-sha256") {
            digest(identity.corpusSha256, corpus);
        } else if (option == "--configuration-sha256") {
            digest(identity.trainerConfigurationSha256, configuration);
        } else if (option == "--attribution-sha256") {
            digest(identity.attributionSha256, attribution);
        } else if (option == "--trainer-revision-sha256") {
            digest(identity.trainerRevisionSha256, revision);
        } else if (option == "--model-revision" && index + 1 < argc) {
            identity.modelRevision = static_cast<std::uint32_t>(std::stoul(argv[++index]));
        } else if (option == "--validation-report" && index + 1 < argc) {
            validationReportPath = argv[++index];
        } else if (option == "--force") {
            force = true;
        } else {
            throw std::runtime_error("unknown export option: " + option);
        }
    }
    if (!corpus || !configuration || !attribution || !revision) {
        throw std::runtime_error("export requires all four identity SHA-256 values");
    }
    std::vector<std::uint8_t> payload;
    std::array<tgmr::PhaseCheckpoint, 18> phaseCheckpoints;
    bool havePhaseCheckpoints = false;
    if (checkpoints) {
        for (std::size_t phase = 0; phase < phaseCheckpoints.size(); ++phase) {
            phaseCheckpoints[phase] = tgmr::readPhaseCheckpoint(
                phasePath(input, phase, "final"));
            if (phase != 0
                && phaseCheckpoints[phase].corpusPayloadSha256
                    != phaseCheckpoints[0].corpusPayloadSha256) {
                throw std::runtime_error("phase checkpoints bind different corpora");
            }
        }
        if (phaseCheckpoints[0].corpusPayloadSha256 != identity.corpusSha256) {
            throw std::runtime_error("--corpus-sha256 differs from phase checkpoints");
        }
        payload = tgmr::exportPhasePayload(phaseCheckpoints);
        havePhaseCheckpoints = true;
    } else {
        const auto legacy = readFile(input);
        static const char magic[8] = {'X', 'T', 'G', 'R', 'R', 'E', 'D', '1'};
        static const char reviewed[] =
            "6279b6a593ef4b595b1aff246182682b60c7eea701373ee2eb9f80cfcb50485c";
        if (legacy.size() != 6073164 || std::memcmp(legacy.data(), magic, 8) != 0
            || tgmr::hex(tgmr::sha256(legacy.data(), legacy.size())) != reviewed) {
            throw std::runtime_error("legacy input is not the authenticated research TGMR v1 artifact");
        }
        payload.assign(legacy.begin() + 36, legacy.end());
    }
    const std::string manifest = output + ".json";
    if ((!force && std::filesystem::exists(output))
        || (!force && std::filesystem::exists(manifest))) {
        throw std::runtime_error("refusing to replace TGMR model or companion manifest");
    }
    const std::string candidate = output + ".tmp-model";
    const std::string temporaryManifest = manifest + ".tmp";
    std::string json;
    try {
        tgmr::writeModelV2(candidate, identity, payload, true);
        const auto inspection = tgmr::inspectModelV2(readFile(candidate));
        const std::string validation = validationReportPath.empty()
            ? std::string() : checkedValidationReport(validationReportPath, inspection);
        json = canonicalTrainingManifest(
            inspection, havePhaseCheckpoints ? &phaseCheckpoints : nullptr,
            validationReportPath.empty() ? nullptr : &validation);
        {
            std::ofstream stream(temporaryManifest, std::ios::binary);
            stream.write(json.data(), json.size());
            if (!stream) throw std::runtime_error("cannot write TGMR companion manifest");
        }
        if (force) {
            std::error_code ignored;
            std::filesystem::remove(manifest, ignored);
            std::filesystem::remove(output, ignored);
        }
        std::filesystem::rename(temporaryManifest, manifest);
        std::filesystem::rename(candidate, output);
    } catch (...) {
        std::error_code ignored;
        std::filesystem::remove(temporaryManifest, ignored);
        std::filesystem::remove(candidate, ignored);
        throw;
    }
    std::cout << json;
    return 0;
}

int corpusCommand(int argc, char **argv)
{
    if (argc < 3) {
        usage(std::cerr);
        return 2;
    }
    const std::string command = argv[2];
    if (command == "matrices") {
        if (argc != 3) throw std::runtime_error("corpus matrices takes no arguments");
        std::cout << tgmr::canonicalCameraMatrixJson();
        return 0;
    }
    if (command == "classify-file") {
        if (argc != 5) {
            throw std::runtime_error("corpus classify-file requires SOURCE-ID and IMAGE");
        }
        const std::string sourceId = argv[3];
        if (sourceId.empty()
            || !std::all_of(sourceId.begin(), sourceId.end(), [](unsigned char value) {
                return std::isalnum(value) || value == '.' || value == '_'
                    || value == ':' || value == '-';
            })) {
            throw std::runtime_error("classify-file SOURCE-ID contains nonportable characters");
        }
        const tgmr::LinearImage image = tgmr::loadLinearImage(argv[4]);
        const tgmr::ImageClassification classification = tgmr::classifyImage(image);
        std::cout << tgmr::canonicalClassificationJson(image, classification, sourceId);
        return 0;
    }
    if (command == "verify-sources") {
        if (argc != 5) {
            throw std::runtime_error("corpus verify-sources requires MANIFEST and CACHE");
        }
        const auto result = tgmr::verifySources(tgmr::readSourceManifest(argv[3]), argv[4]);
        std::cout << "{\n"
            << "  \"authenticated\": " << result.authenticated << ",\n"
            << "  \"changed\": " << result.changed << ",\n"
            << "  \"format\": \"rawtherapee-tgmr-source-verification-v1\",\n"
            << "  \"missing\": " << result.missing << ",\n"
            << "  \"selected\": " << result.selected << "\n}\n";
        return result.authenticated == result.selected ? 0 : 1;
    }
    if (command == "validate-manifest") {
        if (argc != 4) throw std::runtime_error("corpus validate-manifest requires MANIFEST");
        const auto records = tgmr::readSourceManifest(argv[3]);
        tgmr::validateProductionManifest(records);
        std::cout << "{\"format\":\"rawtherapee-tgmr-production-manifest-validation-v1\","
                     "\"selected_sources\":5000,\"status\":\"valid\"}\n";
        return 0;
    }
    if (command == "classify") {
        if (argc < 6) {
            throw std::runtime_error(
                "corpus classify requires INPUT CACHE OUTPUT"
                " [--candidates|--all] [classifier options] [--force]");
        }
        bool force = false;
        bool includeUnselected = false;
        bool candidates = false;
        bool fetchedCandidates = false;
        tgmr::ClassificationOptions classification;
        auto unsignedOption = [&](int &index, const char *name) {
            if (++index >= argc) {
                throw std::runtime_error(std::string(name) + " requires a value");
            }
            std::size_t consumed = 0;
            const std::string encoded = argv[index];
            const std::uint64_t value = std::stoull(encoded, &consumed);
            if (consumed != encoded.size() || value == 0
                || value > std::numeric_limits<std::uint32_t>::max()) {
                throw std::runtime_error(std::string(name) + " is out of range");
            }
            return static_cast<std::uint32_t>(value);
        };
        for (int index = 6; index < argc; ++index) {
            const std::string option = argv[index];
            if (option == "--force") force = true;
            else if (option == "--all") includeUnselected = true;
            else if (option == "--candidates") {
                candidates = true;
                fetchedCandidates = true;
            }
            else if (option == "--open-images-cvdf") {
                if (++index >= argc) {
                    throw std::runtime_error("--open-images-cvdf requires a split");
                }
                candidates = true;
                classification.openImagesCvdfSplit = argv[index];
            }
            else if (option == "--proxy") classification.proxy = true;
            else if (option == "--allow-failures") classification.allowFailures = true;
            else if (option == "--jobs") classification.jobs = unsignedOption(index, "--jobs");
            else if (option == "--retries") {
                if (++index >= argc) throw std::runtime_error("--retries requires a value");
                std::size_t consumed = 0;
                const std::string encoded = argv[index];
                const std::uint64_t value = std::stoull(encoded, &consumed);
                if (consumed != encoded.size()
                    || value > std::numeric_limits<std::uint32_t>::max()) {
                    throw std::runtime_error("--retries is out of range");
                }
                classification.retries = static_cast<std::uint32_t>(value);
            } else if (option == "--checkpoint-images") {
                classification.checkpointImages = unsignedOption(index, "--checkpoint-images");
            } else if (option == "--checkpoint-seconds") {
                classification.checkpointSeconds = unsignedOption(index, "--checkpoint-seconds");
            } else if (option == "--progress-seconds") {
                classification.progressSeconds = unsignedOption(index, "--progress-seconds");
            } else if (option == "--work-dir") {
                if (++index >= argc) throw std::runtime_error("--work-dir requires a value");
                classification.workDirectory = argv[index];
            }
            else throw std::runtime_error("unknown corpus classify option: " + option);
        }
        if (candidates) {
            if (includeUnselected || (fetchedCandidates
                && !classification.openImagesCvdfSplit.empty())) {
                throw std::runtime_error("classification input modes are mutually exclusive");
            }
            tgmr::classifyFetchedCandidates(
                argv[3], argv[4], argv[5], force, classification);
        } else {
            tgmr::classifySources(tgmr::readSourceManifest(argv[3]), argv[4], argv[5],
                                  force, includeUnselected, classification);
        }
        std::cout << "classification complete: " << argv[5] << '\n';
        return 0;
    }
    if (command == "select") {
        if (argc < 6 || argc > 7) {
            throw std::runtime_error(
                "corpus select requires CANDIDATES RECIPE OUTPUT [--force]");
        }
        const bool force = argc == 7 && std::string(argv[6]) == "--force";
        if (argc == 7 && !force) throw std::runtime_error("unknown corpus select option");
        const auto records = tgmr::readSourceManifest(argv[3]);
        std::cout << tgmr::selectProductionSources(
            records, argv[4], argv[5], force);
        return 0;
    }
    if (command == "propose-patches") {
        if (argc < 6 || argc > 7) {
            throw std::runtime_error(
                "corpus propose-patches requires MANIFEST CACHE OUTPUT [--force]");
        }
        const bool force = argc == 7 && std::string(argv[6]) == "--force";
        if (argc == 7 && !force) throw std::runtime_error("unknown propose-patches option");
        const auto records = tgmr::readSourceManifest(argv[3]);
        std::ostringstream output;
        output.imbue(std::locale::classic());
        output << std::setprecision(12);
        static const double exposures[] = {-2.0,-1.5,-1.0,-0.5,0.5,1.0,1.5,2.0};
        static const std::array<std::array<double, 3>, 6> whiteBalances{{
            {{2.0,1.0,0.5}}, {{0.5,1.0,2.0}},
            {{1.5,0.75,0.888888888889}}, {{0.666666666667,1.333333333333,1.125}},
            {{1.25,0.8,1.0}}, {{0.8,1.25,1.0}},
        }};
        for (const auto &record : records) {
            if (!record.selected) continue;
            const std::filesystem::path path = std::filesystem::path(argv[4])
                / record.cacheFilename;
            if (tgmr::hex(tgmr::sha256File(path.string())) != record.sha256) {
                throw std::runtime_error("source changed before patch proposal: " + record.sourceId);
            }
            const auto image = tgmr::loadLinearImage(path.string());
            const std::size_t count = record.split == tgmr::CorpusSplit::TRAIN ? 256 : 128;
            const auto proposals = tgmr::proposePatches(
                image, record.patchSamplingSeed, count);
            output << "{\"format\":\"rawtherapee-tgmr-patch-proposal-v1\","
                << "\"patch_coordinates\":[";
            for (std::size_t index = 0; index < proposals.size(); ++index) {
                if (index) output << ',';
                const auto &patch = proposals[index];
                const bool identity = index % 4 == 0;
                const auto selector = tgmr::sha256(
                    record.sourceId.data(), record.sourceId.size());
                const unsigned choice = (selector[index % selector.size()] + index) & 255U;
                const double exposure = identity ? 0.0 : exposures[choice % 8];
                const auto whiteBalance = identity
                    ? std::array<double, 3>{{1.0,1.0,1.0}}
                    : whiteBalances[(choice / 8) % whiteBalances.size()];
                const unsigned matrix = identity ? 0
                    : record.split == tgmr::CorpusSplit::TRAIN ? 1 + choice % 4
                    : 101 + choice % 2;
                output << "{\"augmentation\":{\"exposure_stops\":" << exposure
                    << ",\"kind\":" << (identity ? 0 : 1)
                    << ",\"matrix_id\":" << matrix << ",\"sequence\":" << index
                    << ",\"white_balance\":[" << whiteBalance[0] << ','
                    << whiteBalance[1] << ',' << whiteBalance[2] << "]},"
                    << "\"coverage\":" << (patch.coverage ? "true" : "false")
                    << ",\"coverage_class\":" << unsigned(patch.coverageClass)
                    << ",\"x\":" << patch.x << ",\"y\":" << patch.y << '}';
            }
            output << "],\"source_id\":\"" << record.sourceId << "\"}\n";
        }
        writeTextAtomic(argv[5], output.str(), force);
        return 0;
    }
    if (command == "finalize") {
        if (argc < 6) {
            throw std::runtime_error(
                "corpus finalize requires MANIFEST CACHE OUTPUT"
                " [--work-dir DIR] [--progress-seconds N] [--force]");
        }
        bool force = false;
        tgmr::FinalizationOptions options;
        for (int index = 6; index < argc; ++index) {
            const std::string option = argv[index];
            if (option == "--force") {
                force = true;
            } else if (option == "--work-dir" && index + 1 < argc) {
                options.workDirectory = argv[++index];
            } else if (option == "--progress-seconds" && index + 1 < argc) {
                const std::string value = argv[++index];
                std::size_t consumed = 0;
                const std::uint64_t parsed = std::stoull(value, &consumed);
                if (consumed != value.size() || parsed == 0
                    || parsed > std::numeric_limits<std::uint32_t>::max()) {
                    throw std::runtime_error("--progress-seconds is out of range");
                }
                options.progressSeconds = static_cast<std::uint32_t>(parsed);
            } else {
                throw std::runtime_error("unknown corpus finalize option: " + option);
            }
        }
        const auto records = tgmr::readSourceManifest(argv[3]);
        tgmr::finalizeSources(records, argv[3], argv[4], argv[5], options, force);
        // Re-read the emitted bytes before applying the complete production
        // contract, so serialization omissions cannot escape the gate.
        tgmr::validateProductionManifest(tgmr::readSourceManifest(argv[5]));
        std::cout << "finalized production manifest: " << argv[5] << '\n';
        return 0;
    }
    if (command == "pack") {
        if (argc < 6) {
            throw std::runtime_error(
                "corpus pack requires MANIFEST CACHE OUTPUT [--noise RECIPE] [--force]");
        }
        bool force = false;
        tgmr::PackOptions options;
        for (int index = 6; index < argc; ++index) {
            const std::string option = argv[index];
            if (option == "--force") {
                force = true;
            } else if (option == "--noise" && index + 1 < argc) {
                const std::string value = argv[++index];
                if (value == "none") options.noise = tgmr::PackNoiseRecipe::NONE;
                else if (value == "sensor-v1") {
                    options.noise = tgmr::PackNoiseRecipe::SENSOR_V1;
                } else {
                    throw std::runtime_error(
                        "corpus pack --noise must be none or sensor-v1");
                }
            } else if (option == "--training-augmentation" && index + 1 < argc) {
                const std::string value = argv[++index];
                if (value == "production-v1") {
                    options.trainingAugmentation =
                        tgmr::TrainingAugmentationRecipe::PRODUCTION_V1;
                } else if (value == "identity-only") {
                    options.trainingAugmentation =
                        tgmr::TrainingAugmentationRecipe::IDENTITY_ONLY;
                } else {
                    throw std::runtime_error(
                        "--training-augmentation must be production-v1 or identity-only");
                }
            } else if (option == "--work-dir" && index + 1 < argc) {
                options.work.workDirectory = argv[++index];
            } else if ((option == "--checkpoint-records"
                        || option == "--progress-seconds")
                       && index + 1 < argc) {
                const std::string value = argv[++index];
                std::size_t consumed = 0;
                const std::uint64_t parsed = std::stoull(value, &consumed);
                if (consumed != value.size() || parsed == 0) {
                    throw std::runtime_error(option + " is out of range");
                }
                if (option == "--checkpoint-records") {
                    options.work.checkpointRecords = parsed;
                } else {
                    if (parsed > std::numeric_limits<std::uint32_t>::max()) {
                        throw std::runtime_error(option + " is out of range");
                    }
                    options.work.progressSeconds = static_cast<std::uint32_t>(parsed);
                }
            } else {
                throw std::runtime_error("unknown corpus pack option: " + option);
            }
        }
        const auto records = tgmr::readSourceManifest(argv[3]);
        tgmr::validateProductionManifest(records);
        tgmr::packSourcesWithOptions(records, argv[3], argv[4], argv[5], options, force);
        std::cout << tgmr::canonicalInspectionJson(tgmr::inspectCorpus(argv[5]));
        return 0;
    }
    if (command == "release-files") {
        if (argc < 7 || argc > 8) {
            throw std::runtime_error(
                "corpus release-files requires MANIFEST ATTRIBUTION RIGHTS RECONSTRUCTION"
                " [--force]");
        }
        const bool force = argc == 8 && std::string(argv[7]) == "--force";
        if (argc == 8 && !force) {
            throw std::runtime_error("unknown corpus release-files option");
        }
        const auto records = tgmr::readSourceManifest(argv[3]);
        tgmr::validateProductionManifest(records);
        // Prepare and validate every byte string before publishing any output.
        const std::string attribution = tgmr::canonicalAttributionNotice(records);
        const std::string rights = tgmr::canonicalRightsReportJson(records);
        const std::string reconstruction = tgmr::reconstructionListTsv(records);
        if (!force && (std::filesystem::exists(argv[4])
            || std::filesystem::exists(argv[5]) || std::filesystem::exists(argv[6]))) {
            throw std::runtime_error("refusing to replace a corpus release file");
        }
        writeTextAtomic(argv[4], attribution, force);
        writeTextAtomic(argv[5], rights, force);
        writeTextAtomic(argv[6], reconstruction, force);
        std::cout << "generated corpus release files\n";
        return 0;
    }
    if (command == "inspect") {
        if (argc != 4) {
            throw std::runtime_error("corpus inspect requires one TGPC path");
        }
        std::cout << tgmr::canonicalInspectionJson(tgmr::inspectCorpus(argv[3]));
        return 0;
    }
    if (command == "gzip") {
        if (argc < 5) {
            throw std::runtime_error("corpus gzip requires input and output paths");
        }
        int level = 9;
        bool force = false;
        for (int index = 5; index < argc; ++index) {
            const std::string option = argv[index];
            if (option == "--force") {
                force = true;
            } else if (option == "--level" && index + 1 < argc) {
                level = std::stoi(argv[++index]);
            } else {
                throw std::runtime_error("unknown corpus gzip option: " + option);
            }
        }
        tgmr::deterministicGzip(argv[3], argv[4], level, force);
        std::cout << tgmr::canonicalInspectionJson(tgmr::inspectCorpus(argv[4]));
        return 0;
    }
    if (command == "balance") {
        if (argc < 4) throw std::runtime_error("corpus balance requires one TGPC path");
        std::uint64_t training = 10'000;
        std::uint64_t evaluation = 1'000;
        bool fixedThresholds = false;
        for (int index = 4; index < argc; ++index) {
            const std::string option = argv[index];
            if (option == "--training-min" && index + 1 < argc) {
                training = std::stoull(argv[++index]);
            } else if (option == "--eval-min" && index + 1 < argc) {
                evaluation = std::stoull(argv[++index]);
            } else if (option == "--fixed-v1-thresholds") {
                fixedThresholds = true;
            } else {
                throw std::runtime_error("unknown corpus balance option: " + option);
            }
        }
        const auto statistics = fixedThresholds
            ? tgmr::analyzeCorpus(argv[3])
            : tgmr::analyzeCorpusTrainingTertiles(argv[3]);
        std::cout << tgmr::canonicalBalanceJson(statistics, training, evaluation);
        return 0;
    }
    if (command == "report") {
        if (argc < 4) throw std::runtime_error("corpus report requires one TGPC path");
        std::string jsonPath;
        std::string csvPath;
        std::string htmlPath;
        bool force = false;
        bool sources = false;
        bool fixedThresholds = false;
        for (int index = 4; index < argc; ++index) {
            const std::string option = argv[index];
            auto value = [&]() -> std::string {
                if (++index >= argc) throw std::runtime_error(option + " requires a path");
                return argv[index];
            };
            if (option == "--json") jsonPath = value();
            else if (option == "--csv") csvPath = value();
            else if (option == "--html") htmlPath = value();
            else if (option == "--sources") sources = true;
            else if (option == "--fixed-v1-thresholds") fixedThresholds = true;
            else if (option == "--force") force = true;
            else throw std::runtime_error("unknown corpus report option: " + option);
        }
        std::string json;
        std::string csv;
        std::string html;
        if (sources) {
            const auto records = tgmr::readSourceManifest(argv[3]);
            json = tgmr::canonicalSourceReportJson(records);
            csv = tgmr::sourceReportCsv(records);
            html = tgmr::sourceReportHtml(records);
        } else {
            const auto statistics = fixedThresholds
                ? tgmr::analyzeCorpus(argv[3])
                : tgmr::analyzeCorpusTrainingTertiles(argv[3]);
            json = tgmr::canonicalCorpusReportJson(statistics);
            csv = tgmr::corpusReportCsv(statistics);
            html = tgmr::corpusReportHtml(statistics);
        }
        if (jsonPath.empty() && csvPath.empty() && htmlPath.empty()) {
            std::cout << json;
        } else {
            if (!jsonPath.empty()) writeTextAtomic(jsonPath, json, force);
            if (!csvPath.empty()) writeTextAtomic(csvPath, csv, force);
            if (!htmlPath.empty()) writeTextAtomic(htmlPath, html, force);
        }
        return 0;
    }
    throw std::runtime_error("unsupported corpus command in this build: " + command);
}

} // namespace

int main(int argc, char **argv)
{
    try {
        std::locale::global(std::locale::classic());
        std::cout.imbue(std::locale::classic());
        std::cerr.imbue(std::locale::classic());
        if (argc == 2 && std::string(argv[1]) == "--version") {
            std::cout << "rt-tgmr-train 0.1.0 corpus-abi=1 model-abi=2\n";
            return 0;
        }
        if (argc >= 2 && std::string(argv[1]) == "corpus") {
            return corpusCommand(argc, argv);
        }
        if (argc >= 2 && std::string(argv[1]) == "verify") {
            return verifyCommand(argc, argv);
        }
        if (argc >= 2 && std::string(argv[1]) == "validate") {
            return validateCommand(argc, argv);
        }
        if (argc >= 2 && std::string(argv[1]) == "export") {
            return exportCommand(argc, argv);
        }
        if (argc >= 2 && std::string(argv[1]) == "train") {
            return trainCommand(argc, argv);
        }
        if (argc >= 2 && std::string(argv[1]) == "resume") {
            return resumeCommand(argc, argv);
        }
        if (argc >= 2 && std::string(argv[1]) == "benchmark") {
            return benchmarkCommand(argc, argv);
        }
        usage(argc == 1 ? std::cout : std::cerr);
        return argc == 1 ? 0 : 2;
    } catch (const std::exception &error) {
        std::cerr << "rt-tgmr-train error: " << error.what() << '\n';
        return 2;
    }
}
