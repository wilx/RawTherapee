#pragma once

#include "tgmr/corpus.h"
#include "tgmr/image.h"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace tgmr
{

struct ArchiveFallback final {
    std::string url;
    std::string sha256;
    std::string member;
    std::string memberSha256;
};

struct PatchSelection final {
    std::uint32_t x = 0;
    std::uint32_t y = 0;
    bool coverage = false;
    std::uint8_t coverageClass = 0;
    std::uint8_t augmentationKind = 0;
    std::int16_t exposureStopsQ8 = 0;
    std::array<std::uint16_t, 3> whiteBalanceQ12{{4096,4096,4096}};
    std::uint16_t matrixId = 0;
    std::uint16_t sequence = 0;
};

struct SourceRecord final {
    bool manifestV2 = false;
    std::string sourceId;
    CorpusSplit split = CorpusSplit::TRAIN;
    bool splitAssigned = true;
    bool selected = false;
    std::string selectionStatus;
    std::string advertisedChecksum;
    std::string cacheFilename;
    std::string originalUrl;
    std::vector<std::string> fallbackUrls;
    std::string landingPage;
    std::string sha256;
    std::string decodedPixelSha256;
    std::string author;
    std::string authorId;
    std::string authorUrl;
    std::string title;
    std::string license;
    std::string licenseUrl;
    std::string fileType;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint16_t orientation = 1;
    std::string iccIdentity;
    std::string perceptualHash;
    std::string pHash;
    ImageClassification classification;
    std::string catalogName;
    std::string catalogRevision;
    std::string catalogSnapshotSha256;
    std::string upstreamSourceId;
    std::string upstreamFlickrId;
    std::string rightsEvidenceUrl;
    std::string rightsEvidenceRevision;
    std::string rightsEvidenceSha256;
    std::string rightsReviewStatus;
    std::string peopleReviewStatus;
    std::vector<std::string> contentTags;
    std::vector<ArchiveFallback> archiveFallbacks;
    std::uint64_t patchSamplingSeed = 0;
    std::vector<PatchSelection> patches;
};

std::vector<SourceRecord> readSourceManifest(const std::string &path);
void validateProductionManifest(const std::vector<SourceRecord> &records);

struct SourceVerification final {
    std::uint64_t selected = 0;
    std::uint64_t authenticated = 0;
    std::uint64_t missing = 0;
    std::uint64_t changed = 0;
};

// Corpus-v1 deliberately freezes augmentation recipes by name.  NONE is the
// simpler clipping-only candidate.  SENSOR_V1 adds deterministic bounded read
// and signal-dependent perturbations to non-identity patches; its use must be
// selected on validation before the production corpus identity is frozen.
enum class PackNoiseRecipe : std::uint8_t {
    NONE = 0,
    SENSOR_V1 = 1,
};

SourceVerification verifySources(
    const std::vector<SourceRecord> &records,
    const std::string &cacheDirectory);

void classifySources(
    const std::vector<SourceRecord> &records,
    const std::string &cacheDirectory,
    const std::string &outputJsonl,
    bool force = false,
    bool includeUnselected = false);

// Classify the authenticated output of prepare_corpus.py fetch before a full
// source-manifest record exists.  This path reads only the minimal fetched
// candidate identity and never performs network I/O.
void classifyFetchedCandidates(
    const std::string &fetchedCandidateJsonl,
    const std::string &cacheDirectory,
    const std::string &outputJsonl,
    bool force = false);

// Select an exact production source population from reviewed v2 candidates.
// The returned string is a canonical decision report; the selected v2 JSONL
// manifest is published atomically at outputManifest.
std::string selectProductionSources(
    const std::vector<SourceRecord> &records,
    const std::string &selectionRecipe,
    const std::string &outputManifest,
    bool force = false);

void writeSourceManifestV2(
    const std::vector<SourceRecord> &records,
    const std::string &outputManifest,
    bool force = false);

std::string canonicalSourceReportJson(const std::vector<SourceRecord> &records);
std::string sourceReportCsv(const std::vector<SourceRecord> &records);
std::string sourceReportHtml(const std::vector<SourceRecord> &records);

void packSources(
    const std::vector<SourceRecord> &records,
    const std::string &manifestPath,
    const std::string &cacheDirectory,
    const std::string &outputTgpc,
    PackNoiseRecipe noiseRecipe = PackNoiseRecipe::NONE,
    bool force = false);

} // namespace tgmr
