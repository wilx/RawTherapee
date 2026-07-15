#include "rtnn_inspection.h"

#include "rtengine/demosaicnetxtransmodel.h"
#include "rtengine/neuralmodel.h"

#include <iostream>

int main(int argc, char **argv)
{
    if (argc != 2) {
        std::cerr << "usage: rawtherapee-rtnn-inspect PATH\n";
        return 2;
    }

    const rtengine::neural::DemosaicNetXTransLoadResult loaded =
        rtengine::neural::loadDemosaicNetXTransModel(argv[1]);
    if (!loaded) {
        std::cerr << "error ["
                  << rtengine::neural::neuralModelErrorCodeName(loaded.error.code)
                  << "]: " << loaded.error.message << '\n';
        return 2;
    }

    std::cout << rtnn_test::canonicalInspectionJson(*loaded.model);
    return 0;
}
