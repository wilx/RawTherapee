#include "tgmr/corpus.h"
#include "tgmr/camera_matrices.h"
#include "tgmr/corpus_analysis.h"
#include "tgmr/image.h"
#include "tgmr/manifest.h"
#include "tgmr/model_v2.h"
#include "tgmr/patch_selection.h"
#include "tgmr/sha256.h"
#include "tgmr/training.h"
#include "tgmr/validation.h"
#include "tgmr/xtrans_training.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <unistd.h>
#include <png.h>
#include <tiffio.h>

namespace
{

void require(bool condition, const std::string &message)
{
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::string temporary(const char *suffix)
{
    char value[] = "/tmp/rt-tgmr-tooling-XXXXXX";
    const int descriptor = mkstemp(value);
    require(descriptor >= 0, "cannot create temporary path");
    close(descriptor);
    std::remove(value);
    return std::string(value) + suffix;
}

std::vector<unsigned char> read(const std::string &path)
{
    std::ifstream stream(path, std::ios::binary);
    require(static_cast<bool>(stream), "cannot read test artifact");
    return std::vector<unsigned char>(
        std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());
}

void appendU32(std::vector<std::uint8_t> &output, std::uint32_t value)
{
    for (unsigned byte = 0; byte < 4; ++byte) {
        output.push_back(static_cast<std::uint8_t>(value >> (8 * byte)));
    }
}

void appendFloat(std::vector<std::uint8_t> &output, float value)
{
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    appendU32(output, bits);
}

std::vector<std::uint8_t> constantGrayValidationPayload()
{
    std::vector<std::uint8_t> output;
    const auto phases = tgmr::xtransPhaseContracts();
    const std::array<std::uint32_t, 9> coarse{{16,17,18,23,24,25,30,31,32}};
    for (const auto &phase : phases) {
        for (std::uint32_t observed : phase.observedIndices) appendU32(output, observed);
        appendU32(output, phase.sampledCenterChannel);
        appendU32(output, phase.targetChannels[0]);
        appendU32(output, phase.targetChannels[1]);
        for (unsigned component = 0; component < 32; ++component) {
            appendFloat(output, -std::log(32.f));
        }
        for (unsigned i = 0; i < 32 * 49 + 32 * 2; ++i) appendFloat(output, 0.f);
        for (unsigned component = 0; component < 32; ++component) {
            for (unsigned row = 0; row < 49; ++row) {
                for (unsigned column = 0; column < 49; ++column) {
                    appendFloat(output, row == column ? 1.f : 0.f);
                }
            }
        }
        for (unsigned component = 0; component < 32; ++component) appendFloat(output, 0.f);
        for (unsigned i = 0; i < 32 * 2 * 49; ++i) appendFloat(output, 0.f);
        for (std::uint32_t position : coarse) appendU32(output, position);
        for (unsigned component = 0; component < 32; ++component) {
            for (unsigned row = 0; row < 9; ++row) {
                for (unsigned column = 0; column < 9; ++column) {
                    appendFloat(output, row == column ? 1.f : 0.f);
                }
            }
        }
        for (unsigned component = 0; component < 32; ++component) appendFloat(output, 0.f);
    }
    return output;
}

void writePngFixture(const std::string &path, unsigned width, unsigned height)
{
    png_image image{};
    image.version = PNG_IMAGE_VERSION;
    image.width = width;
    image.height = height;
    image.format = PNG_FORMAT_RGB;
    std::vector<unsigned char> pixels(static_cast<std::size_t>(width) * height * 3);
    for (unsigned y = 0; y < height; ++y) {
        for (unsigned x = 0; x < width; ++x) {
            const std::size_t index = (static_cast<std::size_t>(y) * width + x) * 3;
            pixels[index] = static_cast<unsigned char>(17 * x);
            pixels[index + 1] = static_cast<unsigned char>(13 * y);
            pixels[index + 2] = static_cast<unsigned char>(7 * (x + y));
        }
    }
    require(png_image_write_to_file(&image, path.c_str(), 0, pixels.data(), 0, nullptr) != 0,
            "cannot write PNG fixture");
}

void writeTiffFixture(const std::string &path, unsigned width, unsigned height)
{
    TIFF *tiff = TIFFOpen(path.c_str(), "w");
    require(tiff != nullptr, "cannot create TIFF fixture");
    TIFFSetField(tiff, TIFFTAG_IMAGEWIDTH, width);
    TIFFSetField(tiff, TIFFTAG_IMAGELENGTH, height);
    TIFFSetField(tiff, TIFFTAG_SAMPLESPERPIXEL, 3);
    TIFFSetField(tiff, TIFFTAG_BITSPERSAMPLE, 16);
    TIFFSetField(tiff, TIFFTAG_ORIENTATION, ORIENTATION_RIGHTTOP);
    TIFFSetField(tiff, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
    TIFFSetField(tiff, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_RGB);
    TIFFSetField(tiff, TIFFTAG_ROWSPERSTRIP, height);
    std::vector<std::uint16_t> row(static_cast<std::size_t>(width) * 3);
    for (unsigned y = 0; y < height; ++y) {
        for (unsigned x = 0; x < width; ++x) {
            row[x * 3] = static_cast<std::uint16_t>(1000 + 100 * x);
            row[x * 3 + 1] = static_cast<std::uint16_t>(2000 + 100 * y);
            row[x * 3 + 2] = static_cast<std::uint16_t>(3000 + 10 * (x + y));
        }
        require(TIFFWriteScanline(tiff, row.data(), y, 0) >= 0,
                "cannot write TIFF fixture scanline");
    }
    TIFFClose(tiff);
}

void testSha256()
{
    require(tgmr::hex(tgmr::sha256("", 0))
        == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "empty SHA-256 vector failed");
    require(tgmr::hex(tgmr::sha256("abc", 3))
        == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        "abc SHA-256 vector failed");
    const auto parsed = tgmr::parseSha256(
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    require(tgmr::hex(parsed)
        == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        "SHA-256 hexadecimal round trip failed");
}

void testCameraMatrices()
{
    const auto &identity = tgmr::cameraMatrix(0);
    require(identity.linearSrgbToCamera == std::array<double, 9>{{1,0,0,0,1,0,0,0,1}},
            "identity camera matrix changed");
    for (std::uint16_t id : {1, 2, 3, 4, 101, 102}) {
        const auto &matrix = tgmr::cameraMatrix(id);
        for (unsigned row = 0; row < 3; ++row) {
            const double sum = matrix.linearSrgbToCamera[row * 3]
                + matrix.linearSrgbToCamera[row * 3 + 1]
                + matrix.linearSrgbToCamera[row * 3 + 2];
            require(std::abs(sum - 1.0) < 1e-12,
                    "camera matrix does not preserve neutral linear RGB");
        }
    }
    require(tgmr::cameraMatrix(4).heldOut == false
        && tgmr::cameraMatrix(101).heldOut,
        "camera matrix train/evaluation partition changed");
    require(tgmr::canonicalCameraMatrixJson().find("FUJIFILM X-H2S") != std::string::npos,
            "camera matrix manifest changed");
}

void testCorpus()
{
    std::vector<tgmr::PatchRecord> records(3);
    for (std::size_t index = 0; index < records.size(); ++index) {
        auto &record = records[index];
        record.sourceIdSha256 = tgmr::sha256(&index, sizeof(index));
        record.sourceOrdinal = static_cast<std::uint32_t>(index + 10);
        record.x = static_cast<std::uint32_t>(17 + index);
        record.y = static_cast<std::uint32_t>(31 + index);
        record.split = static_cast<tgmr::CorpusSplit>(index + 1);
        record.augmentationKind = static_cast<std::uint8_t>(index);
        record.exposureStopsQ8 = static_cast<std::int16_t>(index * 128 - 128);
        record.patchSeed = 0x525454474d520000ULL + index;
        for (std::size_t value = 0; value < record.rgb.size(); ++value) {
            record.rgb[value] = static_cast<std::uint16_t>(value * 17 + index);
        }
    }
    const auto manifest = tgmr::sha256("manifest", 8);
    const auto config = tgmr::sha256("configuration", 13);
    const std::string corpus = temporary(".tgpc");
    const std::string gzipA = temporary("-a.tgpc.gz");
    const std::string gzipB = temporary("-b.tgpc.gz");
    tgmr::writeCorpus(corpus, manifest, config, records);
    std::size_t visited = 0;
    const auto plain = tgmr::inspectCorpus(corpus,
        [&](const tgmr::PatchRecord &record, std::uint64_t index) {
            require(index == visited, "visitor order changed");
            require(record.rgb[20] == static_cast<std::uint16_t>(20 * 17 + index),
                    "record pixels changed");
            ++visited;
        });
    require(visited == records.size(), "not all records were visited");
    require(plain.header.recordCount == 3
        && plain.header.splitCounts == std::array<std::uint64_t, 3>{{1, 1, 1}},
        "corpus counts changed");
    require(plain.fileBytes == tgmr::TGPC_HEADER_BYTES
        + records.size() * tgmr::TGPC_RECORD_BYTES,
        "corpus size changed");
    tgmr::deterministicGzip(corpus, gzipA, 9);
    tgmr::deterministicGzip(corpus, gzipB, 9);
    require(read(gzipA) == read(gzipB), "deterministic gzip output changed");
    const auto compressed = tgmr::inspectCorpus(gzipA);
    require(compressed.compressed && compressed.header.payloadSha256 == plain.header.payloadSha256,
            "compressed/uncompressed corpus parity failed");
    const auto statistics = tgmr::analyzeCorpus(corpus);
    require(statistics.uniqueSources == std::array<std::uint64_t, 3>{{1, 1, 1}},
            "corpus source counts changed");
    require(tgmr::canonicalCorpusReportJson(statistics).find(
        "rawtherapee-tgmr-corpus-statistics-v1") != std::string::npos,
        "corpus JSON report contract changed");
    require(tgmr::corpusReportCsv(statistics).find("brightness,train") != std::string::npos,
        "corpus CSV report contract changed");
    require(tgmr::corpusReportHtml(statistics).find("<!doctype html>") == 0,
        "corpus HTML report contract changed");
    require(tgmr::canonicalBalanceJson(statistics, 1, 1).find(
        "\"status\": \"deficient\"") != std::string::npos,
        "corpus balance calculation changed");

    auto corrupt = read(corpus);
    corrupt.back() ^= 1;
    const std::string corruptPath = temporary("-corrupt.tgpc");
    {
        std::ofstream stream(corruptPath, std::ios::binary);
        stream.write(reinterpret_cast<const char *>(corrupt.data()), corrupt.size());
    }
    bool rejected = false;
    try {
        tgmr::inspectCorpus(corruptPath);
    } catch (const std::exception &) {
        rejected = true;
    }
    require(rejected, "corrupt corpus was accepted");
    std::remove(corpus.c_str());
    std::remove(gzipA.c_str());
    std::remove(gzipB.c_str());
    std::remove(corruptPath.c_str());
}

void testSourceLimitedTrainingMatrix()
{
    std::vector<tgmr::PatchRecord> records(3);
    for (std::size_t source = 0; source < records.size(); ++source) {
        auto &record = records[source];
        record.sourceIdSha256 = tgmr::sha256(&source, sizeof(source));
        record.sourceOrdinal = static_cast<std::uint32_t>(source);
        record.split = tgmr::CorpusSplit::TRAIN;
        for (std::size_t value = 0; value < record.rgb.size(); ++value) {
            record.rgb[value] = static_cast<std::uint16_t>(1000 * source + value);
        }
    }
    const std::string corpus = temporary("-source-limit.tgpc");
    tgmr::writeCorpus(corpus, tgmr::sha256("manifest", 8),
                      tgmr::sha256("configuration", 13), records);
    std::uint64_t samples = 0;
    std::array<std::uint8_t, 32> digest{};
    const auto phase = tgmr::xtransPhaseContracts()[0];
    const auto limited = tgmr::loadPhaseTrainingMatrix(
        corpus, phase, samples, digest, 2);
    require(samples == 2 && limited.size() == 2 * 51,
            "source-limited matrix did not select the first two sources");
    const auto complete = tgmr::loadPhaseTrainingMatrix(
        corpus, phase, samples, digest);
    require(samples == 3 && complete.size() == 3 * 51,
            "unlimited matrix did not retain every training source");
    bool rejected = false;
    try {
        tgmr::loadPhaseTrainingMatrix(corpus, phase, samples, digest, 4);
    } catch (const std::exception &) {
        rejected = true;
    }
    require(rejected, "source limit larger than the corpus was accepted");
    std::remove(corpus.c_str());
}

void testImageManifestAndPack()
{
    const std::string imagePath = temporary(".png");
    const std::string manifestPath = temporary(".jsonl");
    const std::string classificationPath = temporary("-classification.jsonl");
    const std::string corpusPath = temporary(".tgpc");
    const std::string noisyCorpusA = temporary("-noise-a.tgpc");
    const std::string noisyCorpusB = temporary("-noise-b.tgpc");
    writePngFixture(imagePath, 10, 10);
    const auto image = tgmr::loadLinearImage(imagePath);
    const auto classification = tgmr::classifyImage(image);
    require(image.width == 10 && image.height == 10 && image.fileType == "png",
            "PNG fixture decoded incorrectly");
    require(classification.decodedPixelSha256.size() == 64
        && classification.perceptualHash.size() == 16,
        "image classification identities are malformed");
    const auto proposedA = tgmr::proposePatches(image, 1234, 12);
    const auto proposedB = tgmr::proposePatches(image, 1234, 12);
    require(proposedA.size() == 12 && proposedB.size() == 12,
            "patch proposal count changed");
    std::set<std::pair<std::uint32_t, std::uint32_t>> proposedPositions;
    std::size_t coverage = 0;
    for (std::size_t index = 0; index < proposedA.size(); ++index) {
        require(proposedA[index].x == proposedB[index].x
            && proposedA[index].y == proposedB[index].y
            && proposedA[index].coverage == proposedB[index].coverage,
            "patch proposal is not deterministic");
        proposedPositions.emplace(proposedA[index].x, proposedA[index].y);
        coverage += proposedA[index].coverage;
    }
    require(proposedPositions.size() == 12 && coverage == 3,
            "patch proposal is not 75% uniform/25% coverage or contains duplicates");
    const std::filesystem::path cache = std::filesystem::path(imagePath).parent_path();
    const std::string cacheName = std::filesystem::path(imagePath).filename().string();
    std::ofstream manifest(manifestPath, std::ios::binary);
    manifest << '{'
        << "\"advertised_checksum\":null,"
        << "\"author\":\"TGMR test author\","
        << "\"cache_filename\":\"" << cacheName << "\","
        << "\"classification\":{"
        << "\"channel_means\":[0,0,0],"
        << "\"chroma_ratio_mean\":0,"
        << "\"clipped_black_fraction\":0,"
        << "\"clipped_white_fraction\":0,"
        << "\"gradient_rms\":0,"
        << "\"hue_degrees\":0,"
        << "\"jpeg_blockiness\":0,"
        << "\"laplacian_rms\":0,"
        << "\"local_contrast\":0,"
        << "\"luminance_mean\":0,"
        << "\"luminance_p01\":0,"
        << "\"luminance_p99\":0,"
        << "\"luminance_stddev\":0,"
        << "\"perceptual_hash\":\"" << classification.perceptualHash << "\","
        << "\"saturation_mean\":0},"
        << "\"decoded_pixel_sha256\":\"" << classification.decodedPixelSha256 << "\","
        << "\"fallback_urls\":[],"
        << "\"file_type\":\"png\","
        << "\"format\":\"rawtherapee-tgmr-corpus-source-manifest-v1\","
        << "\"height\":10,"
        << "\"icc_identity\":\"assumed-srgb\","
        << "\"landing_page\":\"https://example.invalid/fixture\","
        << "\"license\":\"CC0-1.0\","
        << "\"license_url\":\"https://creativecommons.org/publicdomain/zero/1.0/\","
        << "\"orientation\":1,"
        << "\"original_url\":\"file://" << imagePath << "\","
        << "\"patch_coordinates\":[{\"augmentation\":{\"exposure_stops\":0,"
           "\"kind\":0,\"matrix_id\":0,\"sequence\":0,"
           "\"white_balance\":[1,1,1]},\"coverage\":false,\"x\":1,\"y\":1},"
           "{\"augmentation\":{\"exposure_stops\":1,\"kind\":1,"
           "\"matrix_id\":1,\"sequence\":1,\"white_balance\":[1.25,1,0.8]},"
           "\"coverage\":true,\"coverage_class\":3,\"x\":2,\"y\":2}],"
        << "\"patch_sampling_seed\":\"0x52545447\","
        << "\"selected\":true,"
        << "\"selection_status\":\"accepted-test-fixture\","
        << "\"sha256\":\"" << tgmr::hex(tgmr::sha256File(imagePath)) << "\","
        << "\"source_id\":\"fixture-png-1\","
        << "\"split\":\"train\","
        << "\"title\":\"fixture\","
        << "\"width\":10}\n";
    manifest.close();
    const auto records = tgmr::readSourceManifest(manifestPath);
    require(records.size() == 1 && records[0].patchSamplingSeed == 0x52545447,
            "source manifest did not preserve the reviewed contract");
    const auto verified = tgmr::verifySources(records, cache.string());
    require(verified.authenticated == 1 && verified.selected == 1,
            "source verification failed");
    tgmr::classifySources(records, cache.string(), classificationPath);
    require(read(classificationPath).size() > 100,
            "source classification output is unexpectedly empty");
    tgmr::packSources(records, manifestPath, cache.string(), corpusPath);
    const auto packed = tgmr::inspectCorpus(corpusPath);
    require(packed.header.recordCount == 2 && packed.header.splitCounts[0] == 2,
            "manifest packing produced the wrong corpus counts");
    tgmr::packSources(records, manifestPath, cache.string(), noisyCorpusA,
                      tgmr::PackNoiseRecipe::SENSOR_V1);
    tgmr::packSources(records, manifestPath, cache.string(), noisyCorpusB,
                      tgmr::PackNoiseRecipe::SENSOR_V1);
    require(read(noisyCorpusA) == read(noisyCorpusB),
            "sensor-v1 augmentation is not byte deterministic");
    // The only fixture patch is identity.  Noise must not alter its samples,
    // although the corpus configuration identity still distinguishes recipes.
    std::vector<tgmr::PatchRecord> plainRecords;
    std::vector<tgmr::PatchRecord> noisyRecords;
    tgmr::inspectCorpus(corpusPath, [&](const tgmr::PatchRecord &value, std::uint64_t) {
        plainRecords.push_back(value);
    });
    const auto noisyInspection = tgmr::inspectCorpus(
        noisyCorpusA, [&](const tgmr::PatchRecord &value, std::uint64_t) {
            noisyRecords.push_back(value);
        });
    require(plainRecords.size() == 2 && noisyRecords.size() == 2
        && plainRecords[0].rgb == noisyRecords[0].rgb
        && noisyRecords[0].augmentationKind == 0
        && plainRecords[1].rgb != noisyRecords[1].rgb
        && plainRecords[1].augmentationKind == 1
        && noisyRecords[1].augmentationKind == 2
        && packed.header.configurationSha256 != noisyInspection.header.configurationSha256,
        "noise recipe did not preserve identity, perturb augmentation, or bind configuration");
    std::remove(imagePath.c_str());
    std::remove(manifestPath.c_str());
    std::remove(classificationPath.c_str());
    std::remove(corpusPath.c_str());
    std::remove(noisyCorpusA.c_str());
    std::remove(noisyCorpusB.c_str());
}

void testTiff16Orientation()
{
    const std::string path = temporary(".tiff");
    writeTiffFixture(path, 9, 7);
    const auto image = tgmr::loadLinearImage(path);
    require(image.fileType == "tiff" && image.orientation == ORIENTATION_RIGHTTOP,
            "TIFF orientation metadata was not preserved");
    require(image.width == 7 && image.height == 9 && image.rgb.size() == 7 * 9 * 3,
            "TIFF orientation was not applied to RGB16 pixels");
    require(std::isfinite(image.rgb.front()) && std::isfinite(image.rgb.back()),
            "TIFF RGB16 conversion produced non-finite values");
    std::remove(path.c_str());
}

void testManifestNearDuplicateLeakage()
{
    const std::string path = temporary("-near-duplicate.jsonl");
    auto line = [](const char *id, const char *split, const char *author,
                   char decodedDigit, const char *perceptual) {
        std::ostringstream output;
        output << "{\"advertised_checksum\":null,\"author\":\"" << author
            << "\",\"cache_filename\":\"" << id << ".png\","
            << "\"classification\":{\"channel_means\":[0,0,0],"
            << "\"chroma_ratio_mean\":0,\"clipped_black_fraction\":0,"
            << "\"clipped_white_fraction\":0,\"gradient_rms\":0,"
            << "\"hue_degrees\":0,\"jpeg_blockiness\":0,"
            << "\"laplacian_rms\":0,\"local_contrast\":0,"
            << "\"luminance_mean\":0,\"luminance_p01\":0,"
            << "\"luminance_p99\":0,\"luminance_stddev\":0,"
            << "\"perceptual_hash\":\"" << perceptual << "\","
            << "\"saturation_mean\":0},\"decoded_pixel_sha256\":\""
            << std::string(64, decodedDigit) << "\",\"fallback_urls\":[],"
            << "\"file_type\":\"png\","
            << "\"format\":\"rawtherapee-tgmr-corpus-source-manifest-v1\","
            << "\"height\":7,\"icc_identity\":\"assumed-srgb\","
            << "\"landing_page\":\"https://example.invalid/" << id << "\","
            << "\"license\":\"CC0-1.0\","
            << "\"license_url\":\"https://creativecommons.org/publicdomain/zero/1.0/\","
            << "\"orientation\":1,\"original_url\":\"https://example.invalid/"
            << id << ".png\",\"patch_coordinates\":[],"
            << "\"patch_sampling_seed\":1,\"selected\":true,"
            << "\"selection_status\":\"accepted-test-fixture\",\"sha256\":\""
            << std::string(64, decodedDigit) << "\",\"source_id\":\"" << id
            << "\",\"split\":\"" << split << "\",\"title\":\"fixture\","
            << "\"width\":7}\n";
        return output.str();
    };
    {
        std::ofstream output(path, std::ios::binary);
        output << line("source-a", "train", "author-a", '1', "0123456789abcdef")
               << line("source-b", "test", "author-b", '2', "0123456789abcdee");
    }
    bool rejected = false;
    try {
        (void)tgmr::readSourceManifest(path);
    } catch (const std::exception &) {
        rejected = true;
    }
    require(rejected, "C++ manifest parser accepted a cross-split perceptual near-duplicate");
    std::remove(path.c_str());
}

void testModelV2()
{
    tgmr::ModelV2Identity identity;
    identity.modelRevision = 7;
    identity.corpusSha256 = tgmr::sha256("corpus", 6);
    identity.trainerConfigurationSha256 = tgmr::sha256("config", 6);
    identity.attributionSha256 = tgmr::sha256("attribution", 11);
    identity.trainerRevisionSha256 = tgmr::sha256("revision", 8);
    const std::vector<std::uint8_t> payload = constantGrayValidationPayload();
    const auto first = tgmr::serializeModelV2(identity, payload);
    const auto second = tgmr::serializeModelV2(identity, payload);
    require(first == second, "TGMR v2 serialization is not deterministic");
    std::vector<std::uint8_t> extracted;
    const auto inspection = tgmr::inspectModelV2(first, &extracted);
    require(extracted == payload, "TGMR v2 payload changed");
    require(inspection.identity.modelRevision == 7
        && inspection.fileBytes == tgmr::TGMR_V2_PAYLOAD_OFFSET + payload.size(),
        "TGMR v2 identity or size changed");
    auto corrupt = first;
    corrupt.back() ^= 0x80;
    bool rejected = false;
    try {
        tgmr::inspectModelV2(corrupt);
    } catch (const std::exception &) {
        rejected = true;
    }
    require(rejected, "corrupt TGMR v2 payload was accepted");
    corrupt = first;
    corrupt[129] ^= 1;
    rejected = false;
    try {
        tgmr::inspectModelV2(corrupt);
    } catch (const std::exception &) {
        rejected = true;
    }
    require(rejected, "unauthenticated TGMR v2 metadata was accepted");
}

void testValidation()
{
    const std::string corpusPath = temporary("-validation.tgpc");
    const std::string modelPath = temporary("-validation.tgmr");
    tgmr::PatchRecord patch;
    patch.split = tgmr::CorpusSplit::VALIDATION;
    patch.sourceOrdinal = 7;
    patch.rgb.fill(16384);
    tgmr::writeCorpus(corpusPath, tgmr::sha256("manifest", 8),
                      tgmr::sha256("configuration", 13), {patch});
    tgmr::ModelV2Identity identity;
    identity.corpusSha256 = tgmr::inspectCorpus(corpusPath).header.payloadSha256;
    identity.trainerConfigurationSha256 = tgmr::sha256("config", 6);
    identity.attributionSha256 = tgmr::sha256("attribution", 11);
    identity.trainerRevisionSha256 = tgmr::sha256("revision", 8);
    const auto payload = constantGrayValidationPayload();
    require(payload.size() == 6073128, "synthetic validation payload size changed");
    tgmr::writeModelV2(modelPath, identity, payload);
    const auto report = tgmr::validateModelOnCorpus(
        modelPath, corpusPath, tgmr::CorpusSplit::VALIDATION);
    require(report.metrics.patches == 1 && report.metrics.scalarValues == 54,
            "TGMR validation did not exercise every phase/channel");
    require(report.metrics.mse < 1e-14 && report.metrics.patchRmsP99 < 1e-7,
            "constant-gray TGMR validation fixture was not reconstructed exactly");
    require(tgmr::canonicalValidationJson(report).find(
        "rawtherapee-tgmr-validation-report-v1") != std::string::npos,
        "TGMR validation JSON contract changed");
    const std::string wrongModelPath = temporary("-wrong-corpus.tgmr");
    identity.corpusSha256 = tgmr::sha256("different-corpus", 16);
    tgmr::writeModelV2(wrongModelPath, identity, payload);
    bool rejected = false;
    try {
        tgmr::validateModelOnCorpus(
            wrongModelPath, corpusPath, tgmr::CorpusSplit::VALIDATION);
    } catch (const std::exception &) {
        rejected = true;
    }
    require(rejected, "TGMR validation accepted a corpus other than the model training corpus");
    std::remove(corpusPath.c_str());
    std::remove(modelPath.c_str());
    std::remove(wrongModelPath.c_str());
}

void testTraining()
{
    const double matrix[] = {4.0, 2.0, 2.0, 3.0};
    double lower[4];
    require(tgmr::choleskyLower(matrix, lower, 2), "Cholesky factorization failed");
    require(std::abs(lower[0] - 2.0) < 1e-14
        && std::abs(lower[2] - 1.0) < 1e-14
        && std::abs(lower[3] - std::sqrt(2.0)) < 1e-14,
        "Cholesky factor is incorrect");

    std::vector<double> samples;
    for (int index = 0; index < 200; ++index) {
        const double cluster = index < 100 ? -1.0 : 1.0;
        const double x = cluster + 0.08 * ((index * 17) % 13 - 6);
        const double y = 0.5 * cluster + 0.05 * ((index * 23) % 11 - 5);
        samples.push_back(x);
        samples.push_back(y);
    }
    tgmr::FitConfiguration configuration;
    configuration.components = 2;
    configuration.gaussianIterations = 3;
    configuration.studentIterations = 4;
    configuration.covarianceFloor = 1e-5;
    configuration.backend = tgmr::TrainingBackend::CANONICAL;
    const auto first = tgmr::fitStudentTMixture(samples, 200, 2, configuration);
    const auto second = tgmr::fitStudentTMixture(samples, 200, 2, configuration);
    require(first.weights == second.weights && first.means == second.means
        && first.scales == second.scales,
        "canonical Student-t fitting is not deterministic");
    require(first.weights.size() == 2 && std::abs(first.weights[0] - 0.5) < 0.05
        && std::abs(first.weights[1] - 0.5) < 0.05,
        "Student-t fitting did not recover cluster populations");
    require((first.means[0] > 0.5 && first.means[2] < -0.5)
        || (first.means[2] > 0.5 && first.means[0] < -0.5),
        "Student-t fitting did not retain both cluster centers");
    require(std::isfinite(first.meanLogLikelihood),
        "Student-t fitting produced a non-finite likelihood");

    tgmr::FitConfiguration partialConfiguration = configuration;
    partialConfiguration.studentIterations = 2;
    auto resumed = tgmr::fitStudentTMixture(
        samples, 200, 2, partialConfiguration);
    resumed = tgmr::continueStudentTMixture(
        samples, 200, configuration, std::move(resumed), 0, 2);
    require(resumed.weights == first.weights && resumed.means == first.means
        && resumed.scales == first.scales
        && resumed.meanLogLikelihood == first.meanLogLikelihood,
        "canonical checkpoint-style continuation differs from uninterrupted fitting");

    configuration.backend = tgmr::TrainingBackend::CPU;
    const auto parallel = tgmr::fitStudentTMixture(samples, 200, 2, configuration);
    require(std::abs(parallel.meanLogLikelihood - first.meanLogLikelihood) < 1e-10,
        "CPU and canonical fitting diverged on the synthetic mixture");

    tgmr::FitConfiguration one;
    one.components = 1;
    one.gaussianIterations = 1;
    one.studentIterations = 0;
    one.covarianceFloor = 1e-6;
    one.backend = tgmr::TrainingBackend::CANONICAL;
    const std::vector<double> gaussianSamples{1.0, 2.0, 3.0, 4.0, 5.0, 8.0};
    one.batchSize = 1;
    const auto oneAtATime = tgmr::fitStudentTMixture(gaussianSamples, 3, 2, one);
    one.batchSize = 4096;
    const auto oneBatch = tgmr::fitStudentTMixture(gaussianSamples, 3, 2, one);
    require(oneAtATime.weights == oneBatch.weights
        && oneAtATime.means == oneBatch.means
        && oneAtATime.scales == oneBatch.scales,
        "canonical fitting changed with batch size");
    require(std::abs(oneBatch.means[0] - 3.0) < 1e-14
        && std::abs(oneBatch.means[1] - 14.0 / 3.0) < 1e-14,
        "K=1 Gaussian conditional mean parity failed");
    require(std::abs(oneBatch.scales[0] - (8.0 / 3.0 + 1e-6)) < 1e-12
        && std::abs(oneBatch.scales[1] - 4.0) < 1e-12
        && std::abs(oneBatch.scales[2] - 4.0) < 1e-12
        && std::abs(oneBatch.scales[3] - (56.0 / 9.0 + 1e-6)) < 1e-12,
        "K=1 Gaussian covariance parity failed");

    // Frozen independent NumPy evaluation of one fixed-nu Student-t ECM step:
    // q=(x-mu)' Sigma^-1 (x-mu), u=(nu+d)/(nu+q), followed by the
    // weighted mean and scale updates. This checks the robust latent-weight
    // equation separately from C++ self-consistency and OpenMP parity.
    one.gaussianIterations = 0;
    one.studentIterations = 1;
    const auto oneStudent = tgmr::fitStudentTMixture(
        gaussianSamples, 3, 2, one);
    require(std::abs(oneStudent.means[0] - 2.999999100013366) < 1e-12
        && std::abs(oneStudent.means[1] - 4.666664366696863) < 1e-12,
        "one-step Student-t mean differs from the independent reference");
    const std::array<double, 4> expectedStudentScale{{
        2.6666718666124924, 4.000005999921439,
        4.000005999921439, 6.22223275543155,
    }};
    for (std::size_t value = 0; value < expectedStudentScale.size(); ++value) {
        require(std::abs(oneStudent.scales[value] - expectedStudentScale[value]) < 1e-11,
            "one-step Student-t covariance differs from the independent reference");
    }
}

void testXTransTrainingContract()
{
    const auto phases = tgmr::xtransPhaseContracts();
    std::set<std::array<std::uint32_t, 49>> patterns;
    for (std::size_t phase = 0; phase < phases.size(); ++phase) {
        const auto &contract = phases[phase];
        require(contract.index == phase, "X-Trans phase order changed");
        require(patterns.insert(contract.observedIndices).second,
                "X-Trans phase contract is duplicated");
        require(contract.observedIndices[24] / 49 == contract.sampledCenterChannel,
                "X-Trans center channel is inconsistent");
        require(contract.targetChannels[0] < contract.targetChannels[1]
            && contract.targetChannels[0] != contract.sampledCenterChannel
            && contract.targetChannels[1] != contract.sampledCenterChannel,
            "X-Trans target channels are inconsistent");
    }

    tgmr::PhaseCheckpoint source;
    source.phase = phases[3];
    source.sampleCount = 200;
    source.configuration.components = 2;
    source.configuration.covarianceFloor = 1e-6;
    source.configuration.degreesOfFreedom = 3.0;
    source.configuration.seed = 123;
    source.configuration.device = 2;
    source.configuration.gpuYieldMilliseconds = 7;
    source.configuration.batchSize = 1234;
    source.configuration.maximumBytes = 12345678;
    source.configuration.sourceLimit = 250;
    source.model.dimension = 51;
    source.model.components = 2;
    source.model.gaussianIterations = 4;
    source.model.studentIterations = 7;
    source.model.meanLogLikelihood = -12.5;
    source.model.weights = {0.6, 0.4};
    source.model.means.resize(2 * 51);
    source.model.scales.assign(2 * 51 * 51, 0.0);
    source.model.effectiveCounts = {120, 80};
    source.corpusPayloadSha256 = tgmr::sha256("tgpc", 4);
    for (std::size_t component = 0; component < 2; ++component) {
        for (std::size_t row = 0; row < 51; ++row) {
            source.model.means[component * 51 + row] = 0.001 * (component * 51 + row);
            source.model.scales[(component * 51 + row) * 51 + row] = 0.01 + 0.001 * component;
        }
    }
    const std::string path = temporary(".tgmrc");
    tgmr::writePhaseCheckpoint(path, source);
    const auto loaded = tgmr::readPhaseCheckpoint(path);
    require(loaded.phase.index == source.phase.index
        && loaded.sampleCount == source.sampleCount
        && loaded.model.weights == source.model.weights
        && loaded.model.means == source.model.means
        && loaded.model.scales == source.model.scales
        && loaded.configuration.device == source.configuration.device
        && loaded.configuration.gpuYieldMilliseconds
            == source.configuration.gpuYieldMilliseconds
        && loaded.configuration.batchSize == source.configuration.batchSize
        && loaded.configuration.maximumBytes == source.configuration.maximumBytes
        && loaded.configuration.sourceLimit == source.configuration.sourceLimit,
        "phase checkpoint round trip changed model state");
    std::remove(path.c_str());

    std::array<tgmr::PhaseCheckpoint, 18> exportModels;
    for (std::size_t phase = 0; phase < exportModels.size(); ++phase) {
        auto &checkpoint = exportModels[phase];
        checkpoint.phase = phases[phase];
        checkpoint.sampleCount = 1000;
        checkpoint.configuration.components = 32;
        checkpoint.configuration.degreesOfFreedom = 3.0;
        checkpoint.model.dimension = 51;
        checkpoint.model.components = 32;
        checkpoint.model.weights.assign(32, 1.0 / 32.0);
        checkpoint.model.means.assign(32 * 51, 0.0);
        checkpoint.model.scales.assign(32 * 51 * 51, 0.0);
        checkpoint.model.effectiveCounts.assign(32, 31.25);
        for (std::size_t component = 0; component < 32; ++component) {
            for (std::size_t row = 0; row < 51; ++row) {
                checkpoint.model.scales[(component * 51 + row) * 51 + row]
                    = 0.01 + component * 1e-5;
            }
        }
    }
    const auto payload = tgmr::exportPhasePayload(exportModels);
    require(payload.size() == 6073164 - 36,
            "exported K32/S9/q8 phase payload size changed");

    exportModels[7].configuration.sourceLimit = 500;
    bool rejectedMixedTraining = false;
    try {
        (void)tgmr::exportPhasePayload(exportModels);
    } catch (const std::runtime_error &) {
        rejectedMixedTraining = true;
    }
    require(rejectedMixedTraining,
            "export accepted phase checkpoints with different training settings");
    exportModels[7].configuration.sourceLimit = 0;

    exportModels[11].model.means[0] =
        std::numeric_limits<double>::quiet_NaN();
    bool rejectedNonfinite = false;
    try {
        (void)tgmr::exportPhasePayload(exportModels);
    } catch (const std::runtime_error &) {
        rejectedNonfinite = true;
    }
    require(rejectedNonfinite,
            "export accepted a non-finite phase checkpoint");
}

} // namespace

int main()
{
    try {
        testSha256();
        testCameraMatrices();
        testCorpus();
        testSourceLimitedTrainingMatrix();
        testImageManifestAndPack();
        testTiff16Orientation();
        testManifestNearDuplicateLeakage();
        testModelV2();
        testValidation();
        testTraining();
        testXTransTrainingContract();
        std::cout << "TGMR tooling tests passed\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "TGMR tooling test failure: " << error.what() << '\n';
        return 1;
    }
}
