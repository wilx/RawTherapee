#include "tgmr/manifest.h"

#include "tgmr/camera_matrices.h"
#include "tgmr/sha256.h"

#include "cJSON.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>

namespace tgmr
{
namespace
{

const cJSON *field(const cJSON *object, const char *name)
{
    const cJSON *value = cJSON_GetObjectItemCaseSensitive(object, name);
    if (!value) throw std::runtime_error(std::string("manifest field is missing: ") + name);
    return value;
}

std::string text(const cJSON *object, const char *name)
{
    const cJSON *value = field(object, name);
    if (!cJSON_IsString(value) || !value->valuestring || !*value->valuestring) {
        throw std::runtime_error(std::string("manifest field is not a non-empty string: ") + name);
    }
    return value->valuestring;
}

std::uint64_t unsignedInteger(const cJSON *object, const char *name)
{
    const cJSON *value = field(object, name);
    if (cJSON_IsString(value) && value->valuestring && *value->valuestring) {
        std::size_t consumed = 0;
        const std::string encoded(value->valuestring);
        const std::uint64_t result = std::stoull(encoded, &consumed, 0);
        if (consumed != encoded.size()) {
            throw std::runtime_error(std::string("manifest integer string is malformed: ") + name);
        }
        return result;
    }
    if (!cJSON_IsNumber(value) || value->valuedouble < 0.0
        || value->valuedouble > 9007199254740991.0
        || std::floor(value->valuedouble) != value->valuedouble) {
        throw std::runtime_error(std::string("manifest field is not an exact non-negative integer: ") + name);
    }
    return static_cast<std::uint64_t>(value->valuedouble);
}

double finiteNumber(const cJSON *object, const char *name)
{
    const cJSON *value = field(object, name);
    if (!cJSON_IsNumber(value) || !std::isfinite(value->valuedouble)) {
        throw std::runtime_error(std::string("manifest field is not a finite number: ") + name);
    }
    return value->valuedouble;
}

template<typename T>
T boundedUnsignedInteger(const cJSON *object, const char *name)
{
    const std::uint64_t value = unsignedInteger(object, name);
    if (value > std::numeric_limits<T>::max()) {
        throw std::runtime_error(std::string("manifest integer is out of range: ") + name);
    }
    return static_cast<T>(value);
}

void requireFields(
    const cJSON *object,
    const std::set<std::string> &allowed,
    const char *context)
{
    for (const cJSON *item = object->child; item; item = item->next) {
        if (!item->string || allowed.find(item->string) == allowed.end()) {
            throw std::runtime_error(std::string(context) + " contains an unknown field: "
                                     + (item->string ? item->string : "<unnamed>"));
        }
    }
}

bool canonicalSha256(const std::string &value)
{
    return value.size() == 64
        && std::all_of(value.begin(), value.end(), [](unsigned char character) {
            return std::isdigit(character) || (character >= 'a' && character <= 'f');
        });
}

unsigned hammingDistance64(std::uint64_t value)
{
    unsigned count = 0;
    while (value != 0) {
        value &= value - 1;
        ++count;
    }
    return count;
}

class NoiseRandom final
{
public:
    explicit NoiseRandom(std::uint64_t seed) : state_(seed ? seed : 1) {}

    std::uint32_t next()
    {
        std::uint64_t value = state_;
        value ^= value >> 12;
        value ^= value << 25;
        value ^= value >> 27;
        state_ = value;
        return static_cast<std::uint32_t>(
            (value * 0x2545f4914f6cdd1dULL) >> 32);
    }

private:
    std::uint64_t state_;
};

std::int32_t approximateGaussianQ8(NoiseRandom &random)
{
    // Twelve independent eight-bit uniforms form a bounded, integer-only
    // Irwin-Hall approximation having zero mean and sigma approximately 256.
    // Keeping this step entirely integral makes TGPC output reproducible on
    // every supported host rather than depending on libm's Gaussian sampler.
    std::int32_t sum = 0;
    for (unsigned sample = 0; sample < 12; ++sample) {
        sum += static_cast<std::int32_t>(random.next() & 255U);
    }
    return sum - 1530;
}

std::uint32_t integerSquareRoot(std::uint32_t value)
{
    std::uint32_t root = 0;
    std::uint32_t bit = 1U << 30;
    while (bit > value) bit >>= 2;
    while (bit != 0) {
        if (value >= root + bit) {
            value -= root + bit;
            root = (root >> 1) + bit;
        } else {
            root >>= 1;
        }
        bit >>= 2;
    }
    return root;
}

std::uint16_t addSensorNoise(
    std::uint16_t value,
    const std::array<std::uint8_t, 32> &sourceIdentity,
    const PatchSelection &selection,
    std::size_t sampleIndex)
{
    std::uint64_t seed = selection.sequence
        ^ (static_cast<std::uint64_t>(selection.x) << 16)
        ^ (static_cast<std::uint64_t>(selection.y) << 40)
        ^ (static_cast<std::uint64_t>(sampleIndex) * 0x9e3779b97f4a7c15ULL);
    for (unsigned byte = 0; byte < 8; ++byte) {
        seed ^= static_cast<std::uint64_t>(sourceIdentity[byte]) << (8 * byte);
    }
    NoiseRandom random(seed);
    const std::int64_t read = approximateGaussianQ8(random) / 32;
    const std::int64_t signal = static_cast<std::int64_t>(approximateGaussianQ8(random))
        * integerSquareRoot(value) / 1024;
    const std::int64_t noisy = static_cast<std::int64_t>(value) + read + signal;
    return static_cast<std::uint16_t>(std::max<std::int64_t>(
        0, std::min<std::int64_t>(65535, noisy)));
}

bool supportedUrl(const std::string &value)
{
    return value.rfind("https://", 0) == 0 || value.rfind("http://", 0) == 0
        || value.rfind("file://", 0) == 0;
}

CorpusSplit split(const std::string &value)
{
    if (value == "train") return CorpusSplit::TRAIN;
    if (value == "validation") return CorpusSplit::VALIDATION;
    if (value == "test") return CorpusSplit::TEST;
    throw std::runtime_error("manifest split is invalid: " + value);
}

std::filesystem::path sourcePath(const std::string &cache, const SourceRecord &record)
{
    const std::filesystem::path name(record.cacheFilename);
    if (name.is_absolute() || name.has_parent_path()) {
        throw std::runtime_error("manifest cache filename is unsafe");
    }
    return std::filesystem::path(cache) / name;
}

double gainFromQ12(std::uint16_t value)
{
    return value / 4096.0;
}

void verifyDecodedMetadata(
    const SourceRecord &record,
    const LinearImage &image,
    const ImageClassification &classification,
    bool requireDecodedDigest)
{
    if (image.width != record.width || image.height != record.height
        || image.orientation != record.orientation || image.fileType != record.fileType
        || image.iccIdentity != record.iccIdentity) {
        throw std::runtime_error("decoded metadata differs from manifest: " + record.sourceId);
    }
    if (requireDecodedDigest
        && classification.decodedPixelSha256 != record.decodedPixelSha256) {
        throw std::runtime_error("decoded pixels differ from manifest: " + record.sourceId);
    }
}

} // namespace

std::vector<SourceRecord> readSourceManifest(const std::string &path)
{
    std::ifstream stream(path);
    if (!stream) throw std::runtime_error("cannot open source manifest: " + path);
    std::vector<SourceRecord> output;
    std::set<std::string> ids;
    std::set<std::string> filenames;
    std::map<std::string, CorpusSplit> authorSplits;
    std::map<std::string, CorpusSplit> contentSplits;
    std::map<std::string, CorpusSplit> perceptualSplits;
    std::map<std::string, CorpusSplit> pHashSplits;
    std::string line;
    std::uint64_t lineNumber = 0;
    while (std::getline(stream, line)) {
        ++lineNumber;
        if (line.empty()) throw std::runtime_error("source manifest contains a blank line");
        cJSON *root = cJSON_Parse(line.c_str());
        if (!root || !cJSON_IsObject(root)) {
            cJSON_Delete(root);
            throw std::runtime_error("source manifest JSON parse failure on line "
                                     + std::to_string(lineNumber));
        }
        try {
            const std::string format = text(root, "format");
            const bool version2 = format == "rawtherapee-tgmr-corpus-source-manifest-v2";
            if (!version2 && format != "rawtherapee-tgmr-corpus-source-manifest-v1") {
                throw std::runtime_error("wrong source manifest format");
            }
            std::set<std::string> recordFields{
                "format", "source_id", "split", "selected", "selection_status",
                "original_url", "fallback_urls", "landing_page", "author", "title",
                "license", "license_url", "advertised_checksum", "sha256",
                "decoded_pixel_sha256", "cache_filename", "file_type", "width",
                "height", "orientation", "icc_identity", "classification",
                "patch_sampling_seed", "patch_coordinates",
            };
            if (version2) {
                recordFields.insert({
                    "archive_fallbacks", "author_id", "author_url", "catalog",
                    "content_tags", "people_review_status", "rights",
                    "upstream_flickr_id", "upstream_source_id",
                });
            }
            requireFields(root, recordFields, "source manifest record");
            SourceRecord record;
            record.manifestV2 = version2;
            record.sourceId = text(root, "source_id");
            if (!std::all_of(record.sourceId.begin(), record.sourceId.end(),
                    [](unsigned char value) {
                        return std::isalnum(value) || value == '.' || value == '_'
                            || value == ':' || value == '-';
                    })) {
                throw std::runtime_error("source_id contains nonportable characters");
            }
            const std::string splitName = text(root, "split");
            record.splitAssigned = splitName != "unassigned";
            if (!record.splitAssigned && !version2) {
                throw std::runtime_error("v1 source cannot have an unassigned split");
            }
            record.split = record.splitAssigned ? split(splitName) : CorpusSplit::TRAIN;
            const cJSON *selected = field(root, "selected");
            if (!cJSON_IsBool(selected)) throw std::runtime_error("selected must be Boolean");
            record.selected = cJSON_IsTrue(selected);
            if (record.selected && !record.splitAssigned) {
                throw std::runtime_error("selected source cannot have an unassigned split");
            }
            record.selectionStatus = text(root, "selection_status");
            const cJSON *advertised = field(root, "advertised_checksum");
            if (cJSON_IsString(advertised) && advertised->valuestring
                && *advertised->valuestring) {
                record.advertisedChecksum = advertised->valuestring;
            } else if (!cJSON_IsNull(advertised)) {
                throw std::runtime_error("advertised_checksum must be a string or null");
            }
            record.cacheFilename = text(root, "cache_filename");
            record.originalUrl = text(root, "original_url");
            record.landingPage = text(root, "landing_page");
            record.sha256 = text(root, "sha256");
            record.decodedPixelSha256 = text(root, "decoded_pixel_sha256");
            record.classification.decodedPixelSha256 = record.decodedPixelSha256;
            record.author = text(root, "author");
            record.title = text(root, "title");
            record.license = text(root, "license");
            record.licenseUrl = text(root, "license_url");
            record.fileType = text(root, "file_type");
            record.width = boundedUnsignedInteger<std::uint32_t>(root, "width");
            record.height = boundedUnsignedInteger<std::uint32_t>(root, "height");
            record.orientation = boundedUnsignedInteger<std::uint16_t>(root, "orientation");
            record.iccIdentity = text(root, "icc_identity");
            const cJSON *classification = field(root, "classification");
            if (!cJSON_IsObject(classification)) {
                throw std::runtime_error("classification must be an object");
            }
            std::set<std::string> classificationFields{
                "channel_means", "chroma_ratio_mean", "clipped_black_fraction",
                "clipped_white_fraction", "gradient_rms", "hue_degrees",
                "jpeg_blockiness", "laplacian_rms", "local_contrast",
                "luminance_mean", "luminance_p01", "luminance_p99",
                "luminance_stddev", "perceptual_hash", "saturation_mean",
            };
            if (version2) {
                classificationFields.insert({
                    "dhash", "phash", "luminance_histogram", "hue_histogram",
                    "saturation_histogram",
                });
            }
            requireFields(classification, classificationFields, "classification");
            const cJSON *channelMeans = field(classification, "channel_means");
            if (!cJSON_IsArray(channelMeans) || cJSON_GetArraySize(channelMeans) != 3) {
                throw std::runtime_error("classification channel_means must have three values");
            }
            for (int channel = 0; channel < 3; ++channel) {
                const cJSON *mean = cJSON_GetArrayItem(channelMeans, channel);
                if (!cJSON_IsNumber(mean) || !std::isfinite(mean->valuedouble)) {
                    throw std::runtime_error("classification channel_means contains a non-finite value");
                }
                record.classification.channelMeans[channel] = mean->valuedouble;
            }
            record.classification.chromaRatioMean = finiteNumber(classification, "chroma_ratio_mean");
            record.classification.clippedBlackFraction = finiteNumber(classification, "clipped_black_fraction");
            record.classification.clippedWhiteFraction = finiteNumber(classification, "clipped_white_fraction");
            record.classification.gradientRms = finiteNumber(classification, "gradient_rms");
            record.classification.hueDegrees = finiteNumber(classification, "hue_degrees");
            record.classification.jpegBlockiness = finiteNumber(classification, "jpeg_blockiness");
            record.classification.laplacianRms = finiteNumber(classification, "laplacian_rms");
            record.classification.localContrast = finiteNumber(classification, "local_contrast");
            record.classification.luminanceMean = finiteNumber(classification, "luminance_mean");
            record.classification.luminanceP01 = finiteNumber(classification, "luminance_p01");
            record.classification.luminanceP99 = finiteNumber(classification, "luminance_p99");
            record.classification.luminanceStddev = finiteNumber(classification, "luminance_stddev");
            record.classification.saturationMean = finiteNumber(classification, "saturation_mean");
            const cJSON *perceptual = cJSON_GetObjectItemCaseSensitive(
                classification, "perceptual_hash");
            if (!cJSON_IsString(perceptual) || !perceptual->valuestring
                || std::strlen(perceptual->valuestring) != 16
                || !std::all_of(perceptual->valuestring,
                                perceptual->valuestring + 16,
                                [](unsigned char character) {
                                    return std::isdigit(character)
                                        || (character >= 'a' && character <= 'f');
                                })) {
                throw std::runtime_error("classification perceptual_hash is malformed");
            }
            record.perceptualHash = perceptual->valuestring;
            record.classification.perceptualHash = record.perceptualHash;
            if (version2) {
                auto signature = [&](const char *name) {
                    const cJSON *value = field(classification, name);
                    if (!cJSON_IsString(value) || !value->valuestring
                        || std::strlen(value->valuestring) != 16
                        || !std::all_of(value->valuestring, value->valuestring + 16,
                            [](unsigned char character) {
                                return std::isdigit(character)
                                    || (character >= 'a' && character <= 'f');
                            })) {
                        throw std::runtime_error(std::string("classification ") + name
                                                 + " is malformed");
                    }
                    return std::string(value->valuestring);
                };
                if (signature("dhash") != record.perceptualHash) {
                    throw std::runtime_error("v2 dhash and compatibility perceptual_hash differ");
                }
                record.pHash = signature("phash");
                record.classification.pHash = record.pHash;
                // Array items do not have object names, so parse them directly.
                auto histogramItems = [&](const char *name, auto &target) {
                    const cJSON *array = field(classification, name);
                    if (!cJSON_IsArray(array)
                        || cJSON_GetArraySize(array) != static_cast<int>(target.size())) {
                        throw std::runtime_error(std::string("classification ") + name
                                                 + " has the wrong size");
                    }
                    for (std::size_t index = 0; index < target.size(); ++index) {
                        const cJSON *item = cJSON_GetArrayItem(array, static_cast<int>(index));
                        if (!cJSON_IsNumber(item) || item->valuedouble < 0.0
                            || std::floor(item->valuedouble) != item->valuedouble
                            || item->valuedouble > 9007199254740991.0) {
                            throw std::runtime_error(std::string("classification ") + name
                                                     + " contains a non-integer");
                        }
                        target[index] = static_cast<std::uint64_t>(item->valuedouble);
                    }
                };
                histogramItems("luminance_histogram", record.classification.luminanceHistogram);
                histogramItems("hue_histogram", record.classification.hueHistogram);
                histogramItems("saturation_histogram", record.classification.saturationHistogram);
            } else {
                record.pHash = record.perceptualHash;
                record.classification.pHash = record.perceptualHash;
            }
            record.patchSamplingSeed = unsignedInteger(root, "patch_sampling_seed");
            if (!canonicalSha256(record.sha256)
                || !canonicalSha256(record.decodedPixelSha256)) {
                throw std::runtime_error("manifest SHA-256 field is malformed");
            }
            if (record.width < 7 || record.height < 7
                || record.orientation < 1 || record.orientation > 8) {
                throw std::runtime_error("manifest decoded image contract is invalid");
            }
            static const std::set<std::string> permittedLicenses{
                "CC0-1.0", "PDM-1.0", "CC-BY-2.0", "CC-BY-3.0", "CC-BY-4.0",
            };
            if (record.selected && permittedLicenses.find(record.license) == permittedLicenses.end()) {
                throw std::runtime_error("selected manifest source has a prohibited license");
            }
            if (!supportedUrl(record.originalUrl) || !supportedUrl(record.landingPage)
                || !supportedUrl(record.licenseUrl)) {
                throw std::runtime_error("manifest URL has an unsupported scheme");
            }
            const cJSON *fallbacks = field(root, "fallback_urls");
            if (!cJSON_IsArray(fallbacks)) throw std::runtime_error("fallback_urls must be an array");
            for (int index = 0; index < cJSON_GetArraySize(fallbacks); ++index) {
                const cJSON *url = cJSON_GetArrayItem(fallbacks, index);
                if (!cJSON_IsString(url) || !url->valuestring || !supportedUrl(url->valuestring)) {
                    throw std::runtime_error("fallback_urls contains an invalid URL");
                }
                record.fallbackUrls.emplace_back(url->valuestring);
            }
            if (version2) {
                record.authorId = text(root, "author_id");
                record.authorUrl = text(root, "author_url");
                if (!supportedUrl(record.authorUrl)) {
                    throw std::runtime_error("v2 author URL has an unsupported scheme");
                }
                const cJSON *catalog = field(root, "catalog");
                if (!cJSON_IsObject(catalog)) throw std::runtime_error("catalog must be an object");
                requireFields(catalog, {"name", "revision", "snapshot_sha256"}, "catalog");
                record.catalogName = text(catalog, "name");
                record.catalogRevision = text(catalog, "revision");
                record.catalogSnapshotSha256 = text(catalog, "snapshot_sha256");
                static const std::set<std::string> catalogs{
                    "openimages-v7", "pass-v3", "wikimedia-commons",
                    "smithsonian-open-access",
                };
                if (catalogs.find(record.catalogName) == catalogs.end()
                    || !canonicalSha256(record.catalogSnapshotSha256)) {
                    throw std::runtime_error("v2 catalog identity is invalid");
                }
                record.upstreamSourceId = text(root, "upstream_source_id");
                const cJSON *flickr = cJSON_GetObjectItemCaseSensitive(
                    root, "upstream_flickr_id");
                if (cJSON_IsString(flickr) && flickr->valuestring && *flickr->valuestring) {
                    record.upstreamFlickrId = flickr->valuestring;
                } else if (!cJSON_IsNull(flickr)) {
                    throw std::runtime_error("upstream_flickr_id must be a string or null");
                }
                const cJSON *rights = field(root, "rights");
                if (!cJSON_IsObject(rights)) throw std::runtime_error("rights must be an object");
                requireFields(rights, {
                    "evidence_revision", "evidence_sha256", "evidence_url", "review_status",
                }, "rights");
                record.rightsEvidenceRevision = text(rights, "evidence_revision");
                record.rightsEvidenceSha256 = text(rights, "evidence_sha256");
                record.rightsEvidenceUrl = text(rights, "evidence_url");
                record.rightsReviewStatus = text(rights, "review_status");
                if (!canonicalSha256(record.rightsEvidenceSha256)
                    || !supportedUrl(record.rightsEvidenceUrl)
                    || (record.rightsReviewStatus != "approved"
                        && record.rightsReviewStatus != "rejected"
                        && record.rightsReviewStatus != "pending")) {
                    throw std::runtime_error("v2 rights evidence is invalid");
                }
                if (record.selected && record.rightsReviewStatus != "approved") {
                    throw std::runtime_error("selected v2 source lacks approved rights review");
                }
                record.peopleReviewStatus = text(root, "people_review_status");
                if (record.peopleReviewStatus != "not-applicable"
                    && record.peopleReviewStatus != "approved-no-minors-or-sensitive-content"
                    && record.peopleReviewStatus != "rejected"
                    && record.peopleReviewStatus != "pending") {
                    throw std::runtime_error("v2 people review status is invalid");
                }
                const cJSON *tags = field(root, "content_tags");
                if (!cJSON_IsArray(tags)) throw std::runtime_error("content_tags must be an array");
                static const std::set<std::string> permittedTags{
                    "people", "skin-hair-clothing", "foliage", "fur-feathers",
                    "architecture-brick", "textile-print", "metal-specular-jewelry",
                    "food", "water-sky", "low-light", "macro-specimen",
                };
                std::set<std::string> uniqueTags;
                for (int index = 0; index < cJSON_GetArraySize(tags); ++index) {
                    const cJSON *tag = cJSON_GetArrayItem(tags, index);
                    if (!cJSON_IsString(tag) || !tag->valuestring
                        || permittedTags.find(tag->valuestring) == permittedTags.end()
                        || !uniqueTags.insert(tag->valuestring).second) {
                        throw std::runtime_error("content_tags contains an invalid value");
                    }
                    record.contentTags.emplace_back(tag->valuestring);
                }
                if (record.selected
                    && std::find(record.contentTags.begin(), record.contentTags.end(), "people")
                        != record.contentTags.end()
                    && record.peopleReviewStatus != "approved-no-minors-or-sensitive-content") {
                    throw std::runtime_error("selected people image lacks explicit review");
                }
                const cJSON *archives = field(root, "archive_fallbacks");
                if (!cJSON_IsArray(archives)) {
                    throw std::runtime_error("archive_fallbacks must be an array");
                }
                for (int index = 0; index < cJSON_GetArraySize(archives); ++index) {
                    const cJSON *archive = cJSON_GetArrayItem(archives, index);
                    if (!cJSON_IsObject(archive)) {
                        throw std::runtime_error("archive fallback must be an object");
                    }
                    requireFields(archive, {"url", "sha256", "member", "member_sha256"},
                                  "archive fallback");
                    ArchiveFallback value;
                    value.url = text(archive, "url");
                    value.sha256 = text(archive, "sha256");
                    value.member = text(archive, "member");
                    value.memberSha256 = text(archive, "member_sha256");
                    const std::filesystem::path member(value.member);
                    if (!supportedUrl(value.url) || !canonicalSha256(value.sha256)
                        || !canonicalSha256(value.memberSha256) || member.is_absolute()) {
                        throw std::runtime_error("archive fallback is invalid");
                    }
                    for (const auto &component : member) {
                        if (component == "..") {
                            throw std::runtime_error("archive fallback member is unsafe");
                        }
                    }
                    record.archiveFallbacks.push_back(std::move(value));
                }
            } else {
                record.authorId = record.author;
                record.authorUrl = record.landingPage;
                record.rightsReviewStatus = record.selected ? "approved" : "pending";
                record.peopleReviewStatus = "not-applicable";
            }
            const std::filesystem::path cacheName(record.cacheFilename);
            if (cacheName.is_absolute() || cacheName.has_parent_path()
                || record.cacheFilename == "." || record.cacheFilename == "..") {
                throw std::runtime_error("manifest cache filename is unsafe");
            }
            const cJSON *patches = field(root, "patch_coordinates");
            if (!cJSON_IsArray(patches)) throw std::runtime_error("patch_coordinates must be an array");
            const int patchCount = cJSON_GetArraySize(patches);
            for (int index = 0; index < patchCount; ++index) {
                const cJSON *value = cJSON_GetArrayItem(patches, index);
                if (!cJSON_IsObject(value)) throw std::runtime_error("patch coordinate must be an object");
                requireFields(value,
                    {"x", "y", "coverage", "coverage_class", "augmentation"},
                    "patch coordinate");
                PatchSelection patch;
                patch.x = boundedUnsignedInteger<std::uint32_t>(value, "x");
                patch.y = boundedUnsignedInteger<std::uint32_t>(value, "y");
                const cJSON *coverage = cJSON_GetObjectItemCaseSensitive(value, "coverage");
                if (coverage) {
                    if (!cJSON_IsBool(coverage)) throw std::runtime_error("coverage must be Boolean");
                    patch.coverage = cJSON_IsTrue(coverage);
                }
                const cJSON *coverageClass = cJSON_GetObjectItemCaseSensitive(
                    value, "coverage_class");
                if (coverageClass) {
                    patch.coverageClass = boundedUnsignedInteger<std::uint8_t>(
                        value, "coverage_class");
                    if (patch.coverageClass > 16) {
                        throw std::runtime_error("coverage_class is outside 0..16");
                    }
                }
                const cJSON *augmentation = cJSON_GetObjectItemCaseSensitive(value, "augmentation");
                if (augmentation) {
                    if (!cJSON_IsObject(augmentation)) throw std::runtime_error("augmentation must be an object");
                    requireFields(augmentation,
                        {"kind", "exposure_stops", "white_balance", "matrix_id", "sequence"},
                        "patch augmentation");
                    patch.augmentationKind = boundedUnsignedInteger<std::uint8_t>(
                        augmentation, "kind");
                    const cJSON *exposure = field(augmentation, "exposure_stops");
                    if (!cJSON_IsNumber(exposure) || exposure->valuedouble < -2.0
                        || exposure->valuedouble > 2.0) {
                        throw std::runtime_error("augmentation exposure is outside -2..2");
                    }
                    patch.exposureStopsQ8 = static_cast<std::int16_t>(
                        std::llround(exposure->valuedouble * 256.0));
                    const cJSON *whiteBalance = field(augmentation, "white_balance");
                    if (!cJSON_IsArray(whiteBalance) || cJSON_GetArraySize(whiteBalance) != 3) {
                        throw std::runtime_error("augmentation white_balance must have three values");
                    }
                    for (unsigned channel = 0; channel < 3; ++channel) {
                        const cJSON *gain = cJSON_GetArrayItem(whiteBalance, channel);
                        if (!cJSON_IsNumber(gain) || gain->valuedouble < 0.5
                            || gain->valuedouble > 2.0) {
                            throw std::runtime_error("augmentation white-balance gain is outside 0.5..2");
                        }
                        patch.whiteBalanceQ12[channel] = static_cast<std::uint16_t>(
                            std::llround(gain->valuedouble * 4096.0));
                    }
                    patch.matrixId = boundedUnsignedInteger<std::uint16_t>(
                        augmentation, "matrix_id");
                    patch.sequence = boundedUnsignedInteger<std::uint16_t>(
                        augmentation, "sequence");
                }
                record.patches.push_back(patch);
            }
            if (!ids.insert(record.sourceId).second
                || !filenames.insert(record.cacheFilename).second) {
                throw std::runtime_error("duplicate manifest source ID or cache filename");
            }
            if (record.selected) {
                const auto author = authorSplits.emplace(record.authorId, record.split);
                if (!author.second && author.first->second != record.split) {
                    throw std::runtime_error("author occurs in more than one corpus split");
                }
                const auto content = contentSplits.emplace(record.decodedPixelSha256, record.split);
                if (!content.second) {
                    throw std::runtime_error("selected manifest contains an exact decoded-image duplicate");
                }
                if (!record.perceptualHash.empty()) {
                    const auto perceptualEntry = perceptualSplits.emplace(
                        record.perceptualHash, record.split);
                    if (!perceptualEntry.second
                        && perceptualEntry.first->second != record.split) {
                        throw std::runtime_error(
                            "perceptual duplicate occurs in more than one corpus split");
                    }
                }
                if (!record.pHash.empty()) {
                    const auto signature = pHashSplits.emplace(record.pHash, record.split);
                    if (!signature.second && signature.first->second != record.split) {
                        throw std::runtime_error(
                            "DCT perceptual duplicate occurs in more than one corpus split");
                    }
                }
            }
            output.push_back(std::move(record));
        } catch (...) {
            cJSON_Delete(root);
            throw;
        }
        cJSON_Delete(root);
    }
    // A 64-bit dHash equality check is insufficient for split-leakage
    // prevention: ordinary rescaling or recompression can change a few bits.
    // Five bits is the frozen corpus-v1 review threshold.  Candidate pairs at
    // or below it are rejected across author-independent splits and must be
    // resolved by source review rather than silently accepted.
    for (std::size_t left = 0; left < output.size(); ++left) {
        if (!output[left].selected) continue;
        const std::uint64_t leftHash = std::stoull(output[left].perceptualHash, nullptr, 16);
        for (std::size_t right = left + 1; right < output.size(); ++right) {
            if (!output[right].selected || output[left].split == output[right].split) continue;
            const std::uint64_t rightHash = std::stoull(
                output[right].perceptualHash, nullptr, 16);
            if (hammingDistance64(leftHash ^ rightHash) <= 5) {
                throw std::runtime_error(
                    "perceptual near-duplicate occurs in more than one corpus split");
            }
            const std::uint64_t leftPHash = std::stoull(output[left].pHash, nullptr, 16);
            const std::uint64_t rightPHash = std::stoull(output[right].pHash, nullptr, 16);
            if (hammingDistance64(leftPHash ^ rightPHash) <= 8) {
                throw std::runtime_error(
                    "DCT perceptual near-duplicate occurs in more than one corpus split");
            }
        }
    }
    return output;
}

void validateProductionManifest(const std::vector<SourceRecord> &records)
{
    std::array<std::uint64_t, 3> sourceCounts{};
    std::array<std::array<std::uint64_t, 3>, 4> catalogCounts{};
    std::array<std::uint64_t, 3> peopleCounts{};
    std::map<std::string, std::size_t> authorCounts;
    static const std::array<std::string, 4> catalogs{{
        "openimages-v7", "pass-v3", "wikimedia-commons", "smithsonian-open-access",
    }};
    for (const SourceRecord &record : records) {
        if (!record.selected) continue;
        const unsigned splitIndex = static_cast<unsigned>(record.split) - 1;
        ++sourceCounts[splitIndex];
        if (!record.manifestV2 || record.rightsReviewStatus != "approved") {
            throw std::runtime_error(
                "production source must use v2 provenance with approved rights: "
                + record.sourceId);
        }
        const auto catalog = std::find(catalogs.begin(), catalogs.end(), record.catalogName);
        if (catalog == catalogs.end()) {
            throw std::runtime_error("production source has an unsupported catalog");
        }
        ++catalogCounts[static_cast<std::size_t>(catalog - catalogs.begin())][splitIndex];
        if (++authorCounts[record.authorId] > 5) {
            throw std::runtime_error("production manifest exceeds the five-image author cap");
        }
        if (std::find(record.contentTags.begin(), record.contentTags.end(), "people")
            != record.contentTags.end()) {
            if (record.peopleReviewStatus != "approved-no-minors-or-sensitive-content") {
                throw std::runtime_error("production people source lacks explicit approval");
            }
            ++peopleCounts[splitIndex];
        }
        const std::size_t required = record.split == CorpusSplit::TRAIN ? 256 : 128;
        if (record.patches.size() != required) {
            throw std::runtime_error("selected source has the wrong frozen patch count: "
                                     + record.sourceId);
        }
        std::set<std::pair<std::uint32_t, std::uint32_t>> positions;
        std::size_t coverage = 0;
        std::size_t identity = 0;
        for (std::size_t index = 0; index < record.patches.size(); ++index) {
            const PatchSelection &patch = record.patches[index];
            if (!positions.emplace(patch.x, patch.y).second || patch.sequence != index) {
                throw std::runtime_error("source has duplicate patches or noncanonical sequence: "
                                         + record.sourceId);
            }
            coverage += patch.coverage;
            const bool isIdentity = patch.augmentationKind == 0;
            identity += isIdentity;
            if (isIdentity && (patch.matrixId != 0 || patch.exposureStopsQ8 != 0
                || patch.whiteBalanceQ12 != std::array<std::uint16_t, 3>{{4096,4096,4096}})) {
                throw std::runtime_error("identity augmentation changes a patch: " + record.sourceId);
            }
            if (!isIdentity && patch.matrixId == 0) {
                throw std::runtime_error("augmented patch lacks a camera matrix: " + record.sourceId);
            }
        }
        if (coverage != required / 4 || identity != required / 4) {
            throw std::runtime_error("source does not have frozen 75/25 coverage and identity ratios: "
                                     + record.sourceId);
        }
    }
    if (sourceCounts != std::array<std::uint64_t, 3>{{4000,500,500}}) {
        throw std::runtime_error(
            "production manifest must contain exactly 4000/500/500 selected sources");
    }
    const std::array<std::array<std::uint64_t, 3>, 4> expectedCatalogs{{
        {{2000,250,250}}, {{1200,150,150}}, {{480,60,60}}, {{320,40,40}},
    }};
    if (catalogCounts != expectedCatalogs) {
        throw std::runtime_error("production manifest does not match frozen source quotas");
    }
    if (peopleCounts[0] < 600 || peopleCounts[1] < 75 || peopleCounts[2] < 75) {
        throw std::runtime_error("production manifest lacks the controlled people share");
    }
}

SourceVerification verifySources(
    const std::vector<SourceRecord> &records,
    const std::string &cacheDirectory)
{
    SourceVerification result;
    for (const SourceRecord &record : records) {
        if (!record.selected) continue;
        ++result.selected;
        const auto path = sourcePath(cacheDirectory, record);
        if (!std::filesystem::is_regular_file(path)) {
            ++result.missing;
            continue;
        }
        if (hex(sha256File(path.string())) == record.sha256) ++result.authenticated;
        else ++result.changed;
    }
    return result;
}

void classifySources(
    const std::vector<SourceRecord> &records,
    const std::string &cacheDirectory,
    const std::string &outputJsonl,
    bool force,
    bool includeUnselected)
{
    if (!force && std::filesystem::exists(outputJsonl)) {
        throw std::runtime_error("refusing to replace classification output");
    }
    const std::string temporary = outputJsonl + ".tmp";
    std::ofstream output(temporary, std::ios::binary);
    if (!output) throw std::runtime_error("cannot create classification output");
    for (const SourceRecord &record : records) {
        if (!record.selected && !includeUnselected) continue;
        const auto path = sourcePath(cacheDirectory, record);
        if (hex(sha256File(path.string())) != record.sha256) {
            throw std::runtime_error("source changed before classification: " + record.sourceId);
        }
        const LinearImage image = loadLinearImage(path.string());
        const ImageClassification classification = classifyImage(image);
        verifyDecodedMetadata(record, image, classification, false);
        output << canonicalClassificationJson(
            image, classification, record.sourceId, record.cacheFilename);
    }
    output.flush();
    if (!output) {
        output.close();
        std::filesystem::remove(temporary);
        throw std::runtime_error("classification output write failed");
    }
    output.close();
    if (force) std::filesystem::remove(outputJsonl);
    std::filesystem::rename(temporary, outputJsonl);
}

void classifyFetchedCandidates(
    const std::string &fetchedCandidateJsonl,
    const std::string &cacheDirectory,
    const std::string &outputJsonl,
    bool force)
{
    if (!force && std::filesystem::exists(outputJsonl)) {
        throw std::runtime_error("refusing to replace classification output");
    }
    std::ifstream input(fetchedCandidateJsonl);
    if (!input) throw std::runtime_error("cannot open fetched candidate JSONL");
    const std::string temporary = outputJsonl + ".tmp";
    std::ofstream output(temporary, std::ios::binary);
    if (!output) throw std::runtime_error("cannot create classification output");
    std::set<std::string> identities;
    std::string line;
    std::uint64_t lineNumber = 0;
    try {
        while (std::getline(input, line)) {
            ++lineNumber;
            if (line.empty()) throw std::runtime_error("fetched candidates contain a blank line");
            cJSON *root = cJSON_Parse(line.c_str());
            if (!root || !cJSON_IsObject(root)) {
                cJSON_Delete(root);
                throw std::runtime_error("fetched candidate JSON parse failure on line "
                                         + std::to_string(lineNumber));
            }
            try {
                if (text(root, "format") != "rawtherapee-tgmr-fetched-candidate-v1") {
                    throw std::runtime_error("wrong fetched candidate format");
                }
                const std::string catalog = text(root, "catalog");
                const std::string upstream = text(root, "upstream_source_id");
                std::string sourceId = catalog + ':' + upstream;
                for (char &value : sourceId) {
                    const unsigned char byte = static_cast<unsigned char>(value);
                    if (std::isalnum(byte) || value == '.' || value == '_'
                        || value == ':' || value == '-') {
                    } else {
                        value = '-';
                    }
                }
                sourceId.erase(std::unique(sourceId.begin(), sourceId.end(),
                    [](char left, char right) { return left == '-' && right == '-'; }),
                    sourceId.end());
                while (!sourceId.empty() && sourceId.front() == '-') sourceId.erase(sourceId.begin());
                while (!sourceId.empty() && sourceId.back() == '-') sourceId.pop_back();
                if (sourceId.empty() || !identities.insert(sourceId).second) {
                    throw std::runtime_error("duplicate or empty fetched candidate identity");
                }
                const std::string cacheFilename = text(root, "cache_filename");
                const std::filesystem::path name(cacheFilename);
                if (name.is_absolute() || name.has_parent_path()) {
                    throw std::runtime_error("fetched candidate cache filename is unsafe");
                }
                const std::string expected = text(root, "sha256");
                if (!canonicalSha256(expected)) {
                    throw std::runtime_error("fetched candidate SHA-256 is malformed");
                }
                const std::filesystem::path path = std::filesystem::path(cacheDirectory) / name;
                if (hex(sha256File(path.string())) != expected) {
                    throw std::runtime_error("fetched candidate changed before classification: "
                                             + sourceId);
                }
                const LinearImage image = loadLinearImage(path.string());
                const ImageClassification classification = classifyImage(image);
                output << canonicalClassificationJson(
                    image, classification, sourceId, cacheFilename);
                cJSON_Delete(root);
            } catch (...) {
                cJSON_Delete(root);
                throw;
            }
        }
        output.flush();
        if (!input.eof() || !output) {
            throw std::runtime_error("fetched candidate classification I/O failed");
        }
        output.close();
        if (force) std::filesystem::remove(outputJsonl);
        std::filesystem::rename(temporary, outputJsonl);
    } catch (...) {
        output.close();
        std::filesystem::remove(temporary);
        throw;
    }
}

void packSources(
    const std::vector<SourceRecord> &records,
    const std::string &manifestPath,
    const std::string &cacheDirectory,
    const std::string &outputTgpc,
    PackNoiseRecipe noiseRecipe,
    bool force)
{
    const auto manifestDigest = sha256File(manifestPath);
    const char *configuration = noiseRecipe == PackNoiseRecipe::NONE
        ? "tgpc-v1:linear-srgb:7x7:chw:uint16:clip:camera-matrix:exposure-wb:matrix-set-v1:noise-none"
        : "tgpc-v1:linear-srgb:7x7:chw:uint16:clip:camera-matrix:exposure-wb:matrix-set-v1:noise-sensor-v1-read8-shot64";
    writeCorpusStream(outputTgpc, manifestDigest,
        sha256(configuration, std::strlen(configuration)),
        [&](const CorpusRecordSink &sink) {
            std::uint32_t sourceOrdinal = 0;
            for (const SourceRecord &record : records) {
                if (!record.selected) continue;
                const auto path = sourcePath(cacheDirectory, record);
                if (hex(sha256File(path.string())) != record.sha256) {
                    throw std::runtime_error("source changed before packing: " + record.sourceId);
                }
                const LinearImage image = loadLinearImage(path.string());
                const ImageClassification classification = classifyImage(image);
                verifyDecodedMetadata(record, image, classification, true);
                for (const PatchSelection &selection : record.patches) {
                    if (selection.x + 7 > image.width || selection.y + 7 > image.height) {
                        throw std::runtime_error(
                            "manifest patch is outside decoded image: " + record.sourceId);
                    }
                    const CameraMatrix &matrix = cameraMatrix(selection.matrixId);
                    if (selection.matrixId != 0
                        && matrix.heldOut == (record.split == CorpusSplit::TRAIN)) {
                        throw std::runtime_error(
                            "camera-matrix augmentation crosses the train/evaluation boundary");
                    }
                    PatchRecord patch;
                    patch.sourceIdSha256 = sha256(record.sourceId.data(), record.sourceId.size());
                    patch.sourceOrdinal = sourceOrdinal;
                    patch.x = selection.x;
                    patch.y = selection.y;
                    patch.split = record.split;
                    patch.augmentationKind = selection.augmentationKind == 0 ? 0
                        : noiseRecipe == PackNoiseRecipe::SENSOR_V1 ? 2 : 1;
                    patch.orientation = static_cast<std::uint8_t>(image.orientation);
                    patch.exposureStopsQ8 = selection.exposureStopsQ8;
                    patch.whiteBalanceQ12 = selection.whiteBalanceQ12;
                    patch.matrixId = selection.matrixId;
                    patch.augmentationSequence = selection.sequence;
                    patch.patchSeed = record.patchSamplingSeed;
                    const double exposure = std::exp2(selection.exposureStopsQ8 / 256.0);
                    for (unsigned y = 0; y < 7; ++y) {
                        for (unsigned x = 0; x < 7; ++x) {
                            const std::size_t input = ((selection.y + y) * image.width
                                + selection.x + x) * 3;
                            for (unsigned channel = 0; channel < 3; ++channel) {
                                double transformed = 0.0;
                                for (unsigned source = 0; source < 3; ++source) {
                                    transformed += matrix.linearSrgbToCamera[channel * 3 + source]
                                        * image.rgb[input + source];
                                }
                                const double value = std::max(0.0, std::min(1.0,
                                    transformed * exposure
                                    * gainFromQ12(selection.whiteBalanceQ12[channel])));
                                const std::size_t sampleIndex = channel * 49 + y * 7 + x;
                                std::uint16_t quantized = static_cast<std::uint16_t>(
                                    std::llround(value * 65535.0));
                                if (noiseRecipe == PackNoiseRecipe::SENSOR_V1
                                    && selection.augmentationKind != 0) {
                                    quantized = addSensorNoise(
                                        quantized, patch.sourceIdSha256, selection, sampleIndex);
                                }
                                patch.rgb[sampleIndex] = quantized;
                            }
                        }
                    }
                    sink(patch);
                }
                ++sourceOrdinal;
            }
        }, force);
}

} // namespace tgmr
