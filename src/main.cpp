// ============================================================================
// main.cpp — Virtual Wind Tunnel v2 entry point
// ============================================================================

#include "app.h"
#include "validate.h"

#include <cstdio>
#include <cstring>
#include <exception>
#include <string>

namespace {

void printUsage(const char* prog) {
    std::printf(
        "Virtual Wind Tunnel v2 — real-time GPU LBM aerodynamics\n\n"
        "Usage: %s [options]\n\n"
        "  --mesh <path>        load a model file on startup\n"
        "  --grid <X> <Y> <Z>   lattice resolution (default 128 80 80)\n"
        "  --no-les             disable the Smagorinsky subgrid model\n"
        "  --headless           run without a window and self-validate\n"
        "  --steps <N>          headless: number of LBM steps (default 240)\n"
        "  --shape <name>       headless: sphere | cube | cylinder | wing\n"
        "  --aoa <deg>          headless: angle of attack (default 0)\n"
        "  --validate           run the CFD validation suite (Reynolds sweep)\n"
        "  --cyl-mesh <path>    validation: provided cylinder model to test\n"
        "  --cube-mesh <path>   validation: provided cube model to test\n"
        "  --help               this message\n",
        prog);
}

int parseShape(const std::string& s) {
    if (s == "sphere")   return 0;
    if (s == "cube")     return 1;
    if (s == "cylinder") return 2;
    if (s == "wing")     return 3;
    return -1;
}

} // namespace

int main(int argc, char* argv[]) {
    vwt::StartOptions opts;
    vwt::ValidateOptions vopts;
    bool validate = false;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help") {
            printUsage(argv[0]);
            return 0;
        } else if (arg == "--validate") {
            validate = true;
        } else if (arg == "--cyl-mesh" && i + 1 < argc) {
            vopts.cylinderMesh = argv[++i];
        } else if (arg == "--cube-mesh" && i + 1 < argc) {
            vopts.cubeMesh = argv[++i];
        } else if (arg == "--headless") {
            opts.headless = true;
        } else if (arg == "--no-les") {
            opts.lesOff = true;
        } else if (arg == "--steps" && i + 1 < argc) {
            opts.steps = uint32_t(std::strtoul(argv[++i], nullptr, 10));
        } else if (arg == "--aoa" && i + 1 < argc) {
            opts.aoaDeg = std::strtof(argv[++i], nullptr);
        } else if (arg == "--shape" && i + 1 < argc) {
            const int s = parseShape(argv[++i]);
            if (s < 0) {
                std::fprintf(stderr, "Unknown shape '%s'\n", argv[i]);
                return 2;
            }
            opts.shape = s;
        } else if (arg == "--grid" && i + 3 < argc) {
            opts.gx = uint32_t(std::strtoul(argv[++i], nullptr, 10));
            opts.gy = uint32_t(std::strtoul(argv[++i], nullptr, 10));
            opts.gz = uint32_t(std::strtoul(argv[++i], nullptr, 10));
        } else if (arg == "--mesh" && i + 1 < argc) {
            opts.meshPath = argv[++i];
        } else {
            std::fprintf(stderr, "Unknown option '%s'\n", arg.c_str());
            printUsage(argv[0]);
            return 2;
        }
    }

    try {
        if (validate) return vwt::runValidation(vopts);
        vwt::App app;
        return app.run(opts);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "\n[FATAL] %s\n", e.what());
        return 1;
    }
}
