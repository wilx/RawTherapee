#include "rtnn_inspection.h"

#include "rtengine/demosaicnetxtransmodel.h"

#include <array>
#include <iomanip>
#include <sstream>

namespace rtnn_test
{

namespace
{

struct TensorName final {
    const char *symbol;
    const char *sourceName;
};

const std::array<TensorName, 26> TENSOR_NAMES{{
    {"MAIN_CONV1_WEIGHT", "main_processor.conv1.weight"},
    {"MAIN_CONV1_BIAS", "main_processor.conv1.bias"},
    {"MAIN_CONV2_WEIGHT", "main_processor.conv2.weight"},
    {"MAIN_CONV2_BIAS", "main_processor.conv2.bias"},
    {"MAIN_CONV3_WEIGHT", "main_processor.conv3.weight"},
    {"MAIN_CONV3_BIAS", "main_processor.conv3.bias"},
    {"MAIN_CONV4_WEIGHT", "main_processor.conv4.weight"},
    {"MAIN_CONV4_BIAS", "main_processor.conv4.bias"},
    {"MAIN_CONV5_WEIGHT", "main_processor.conv5.weight"},
    {"MAIN_CONV5_BIAS", "main_processor.conv5.bias"},
    {"MAIN_CONV6_WEIGHT", "main_processor.conv6.weight"},
    {"MAIN_CONV6_BIAS", "main_processor.conv6.bias"},
    {"MAIN_CONV7_WEIGHT", "main_processor.conv7.weight"},
    {"MAIN_CONV7_BIAS", "main_processor.conv7.bias"},
    {"MAIN_CONV8_WEIGHT", "main_processor.conv8.weight"},
    {"MAIN_CONV8_BIAS", "main_processor.conv8.bias"},
    {"MAIN_CONV9_WEIGHT", "main_processor.conv9.weight"},
    {"MAIN_CONV9_BIAS", "main_processor.conv9.bias"},
    {"MAIN_CONV10_WEIGHT", "main_processor.conv10.weight"},
    {"MAIN_CONV10_BIAS", "main_processor.conv10.bias"},
    {"MAIN_CONV11_WEIGHT", "main_processor.conv11.weight"},
    {"MAIN_CONV11_BIAS", "main_processor.conv11.bias"},
    {"POST_CONV_WEIGHT", "fullres_processor.post_conv.weight"},
    {"POST_CONV_BIAS", "fullres_processor.post_conv.bias"},
    {"OUTPUT_WEIGHT", "fullres_processor.output.weight"},
    {"OUTPUT_BIAS", "fullres_processor.output.bias"},
}};

std::string hexDigest(const rtengine::neural::Sha256Digest &digest)
{
    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for (const std::uint8_t byte : digest) {
        output << std::setw(2) << static_cast<unsigned>(byte);
    }
    return output.str();
}

} // namespace

std::string canonicalInspectionJson(
    const rtengine::neural::DemosaicNetXTransModel &model)
{
    using rtengine::neural::TensorLayout;

    std::ostringstream output;
    output << "{\n"
           << "  \"artifact\": {\n"
           << "    \"directory_bytes\": 2496,\n"
           << "    \"file_size\": " << model.fileSize() << ",\n"
           << "    \"format\": {\n"
           << "      \"magic_hex\": \"52544e4e0d0a1a0a\",\n"
           << "      \"major\": " << model.formatMajor() << ",\n"
           << "      \"minor\": " << model.formatMinor() << "\n"
           << "    },\n"
           << "    \"header_bytes\": 192,\n"
           << "    \"payload_region_bytes\": " << model.payloadRegionBytes() << ",\n"
           << "    \"payload_sha256\": \"" << hexDigest(model.payloadSha256()) << "\",\n"
           << "    \"sha256\": \"" << hexDigest(model.artifactSha256()) << "\",\n"
           << "    \"tensor_payload_bytes\": " << model.tensorPayloadBytes() << "\n"
           << "  },\n"
           << "  \"format\": \"rawtherapee-rtnn-inspection-v1\",\n"
           << "  \"model\": {\n"
           << "    \"architecture\": {\n"
           << "      \"id\": " << model.architectureId() << ",\n"
           << "      \"symbol\": \"DEMOSAICNET_XTRANS_V1\"\n"
           << "    },\n"
           << "    \"id\": \"demosaicnet-xtrans-v1\",\n"
           << "    \"revision\": " << model.modelRevision() << ",\n"
           << "    \"semantic_schema_sha256\": \"" << hexDigest(model.semanticSchemaSha256()) << "\"\n"
           << "  },\n"
           << "  \"summary\": {\n"
           << "    \"parameter_count\": " << model.parameterCount() << ",\n"
           << "    \"scalar_type\": \"float32\",\n"
           << "    \"tensor_count\": " << model.tensors().size() << "\n"
           << "  },\n"
           << "  \"tensors\": [\n";

    const std::vector<rtengine::neural::NeuralTensorView> &tensors = model.tensors();
    for (std::size_t index = 0; index < tensors.size(); ++index) {
        const rtengine::neural::NeuralTensorView &tensor = tensors[index];
        const TensorName &name = TENSOR_NAMES[index];
        output << "    {\n"
               << "      \"byte_length\": " << tensor.byteLength << ",\n"
               << "      \"element_count\": " << tensor.elementCount << ",\n"
               << "      \"id\": " << tensor.id << ",\n"
               << "      \"layout\": \"" << (tensor.layout == TensorLayout::OIHW ? "OIHW" : "vector") << "\",\n"
               << "      \"payload_offset\": " << tensor.payloadOffset << ",\n"
               << "      \"sha256\": \"" << hexDigest(tensor.sha256) << "\",\n"
               << "      \"shape\": [\n";
        for (std::size_t dimension = 0; dimension < tensor.rank; ++dimension) {
            output << "        " << tensor.dimensions[dimension]
                   << (dimension + 1 == tensor.rank ? "\n" : ",\n");
        }
        output << "      ],\n"
               << "      \"source_name\": \"" << name.sourceName << "\",\n"
               << "      \"symbol\": \"" << name.symbol << "\"\n"
               << "    }" << (index + 1 == tensors.size() ? "\n" : ",\n");
    }
    output << "  ]\n}\n";
    return output.str();
}

} // namespace rtnn_test
