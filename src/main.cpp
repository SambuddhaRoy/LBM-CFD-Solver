// ============================================================================
// main.cpp — command line entry point
// ============================================================================

#include "app.hpp"
#include "geometry.hpp"
#include "tests.hpp"
#include "vk.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>

namespace {

void usage() {
    std::printf(
        "Wind Tunnel v3: real-time GPU lattice Boltzmann aerodynamics\n\n"
        "  (no arguments)          interactive wind tunnel\n"
        "  --selftest              GPU solver vs CPU reference, all precisions\n"
        "  --bench                 throughput benchmark (MLUPS)\n"
        "  --validate [SUITE]      CFD validation against exact solutions and experiments:\n"
        "                          standard (default: laminar exact solutions + bluff\n"
        "                          bodies, ~15 min), laminar, turbulent (cylinder Re 3900,\n"
        "                          3D LES, ~15 min at D = 40), all\n"
        "  --diameter D            turbulent suite: cells across the cylinder (default 40)\n"
        "  --grid X Y Z            benchmark grid (default 256 256 256)\n"
        "  --steps N               benchmark steps per timing (default 200)\n"
        "  --precision P           fp32 | fp16s | fp16c | all (default: all for --bench,\n"
        "                          fp16c otherwise)\n"
        "  --peak GBs              device peak bandwidth, adds a %% of peak column\n"
        "  --capture FILE [N]      run the interactive app for N frames (default 300),\n"
        "                          save the window as a PNG and exit\n\n"
        "Initial state of the interactive app:\n"
        "  --preset N              grid preset index (default 1)\n"
        "  --mesh FILE             load a model (STL, OBJ, glTF, FBX, PLY, ...)\n"
        "  --realtime              start in real-time mode\n"
        "  --mesh-info FILE        list a model's parts, check it is watertight, and exit\n"
        "  --shape S               sphere | cube | cylinder | wing\n"
        "  --pitch DEG             body pitch (angle of attack)\n"
        "  --view 2d|3d            slice or 3D camera\n"
        "  --field F               speed | pressure | vorticity | q\n"
        "  --zoom X                magnification relative to the fitted view\n");
}

bool parsePrecision(const std::string& s, std::vector<wt::Precision>& out) {
    if (s == "fp32")  { out = {wt::Precision::FP32};  return true; }
    if (s == "fp16s") { out = {wt::Precision::FP16S}; return true; }
    if (s == "fp16c") { out = {wt::Precision::FP16C}; return true; }
    if (s == "all")   { out = {wt::Precision::FP32, wt::Precision::FP16S, wt::Precision::FP16C}; return true; }
    return false;
}

} // namespace

int main(int argc, char** argv) {
    std::string mode;
    wt::BenchOptions bench;
    std::vector<wt::Precision> precs;
    wt::StartSetup setup;
    std::string suite = "standard";
    float diameter = 40.f;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--mesh-info" && i + 1 < argc) return wt::printMeshInfo(argv[++i]);
        if (a == "--selftest" || a == "--bench") mode = a;
        else if (a == "--validate") {
            mode = a;
            if (i + 1 < argc && argv[i + 1][0] != '-') suite = argv[++i];
        }
        else if (a == "--grid" && i + 3 < argc) {
            bench.nx = uint32_t(std::strtoul(argv[++i], nullptr, 10));
            bench.ny = uint32_t(std::strtoul(argv[++i], nullptr, 10));
            bench.nz = uint32_t(std::strtoul(argv[++i], nullptr, 10));
        } else if (a == "--steps" && i + 1 < argc) bench.steps = uint32_t(std::strtoul(argv[++i], nullptr, 10));
        else if (a == "--peak" && i + 1 < argc)  bench.peakGBs = std::strtod(argv[++i], nullptr);
        else if (a == "--capture" && i + 1 < argc) {
            setup.capturePath = argv[++i];
            if (i + 1 < argc && argv[i + 1][0] != '-')
                setup.captureFrames = uint32_t(std::strtoul(argv[++i], nullptr, 10));
        }
        else if (a == "--preset" && i + 1 < argc) setup.preset = std::atoi(argv[++i]);
        else if (a == "--mesh" && i + 1 < argc)   setup.meshPath = argv[++i];
        else if (a == "--realtime")               setup.realtime = true;
        else if (a == "--diameter" && i + 1 < argc) diameter = float(std::atof(argv[++i]));
        else if (a == "--pitch" && i + 1 < argc)  setup.pitch = float(std::atof(argv[++i]));
        else if (a == "--zoom" && i + 1 < argc)   setup.zoom = std::max(1e-3f, float(std::atof(argv[++i])));
        else if (a == "--view" && i + 1 < argc)   setup.view3d = std::string(argv[++i]) == "3d" ? 1 : 0;
        else if (a == "--shape" && i + 1 < argc) {
            const std::string s = argv[++i];
            const char* names[] = {"sphere", "cube", "cylinder", "wing"};
            for (int k = 0; k < 4; ++k) if (s == names[k]) setup.shape = wt::Shape(k);
        }
        else if (a == "--field" && i + 1 < argc) {
            const std::string s = argv[++i];
            const char* names[] = {"speed", "pressure", "vorticity", "q"};
            for (int k = 0; k < 4; ++k) if (s == names[k]) setup.field = wt::Field(k);
        }
        else if (a == "--precision" && i + 1 < argc) {
            if (!parsePrecision(argv[++i], precs)) { usage(); return 2; }
        } else { usage(); return a == "--help" ? 0 : 2; }
    }

    try {
        if (mode.empty()) {
            wt::App app;
            if (!precs.empty()) setup.precision = precs.front();
            return app.run(setup);
        }
        gpu::Context ctx;
        ctx.init(nullptr);
        std::printf("GPU: %s\n", ctx.deviceName.c_str());
        int rc = 0;
        if (mode == "--selftest") rc = wt::runSelfTest(ctx);
        else if (mode == "--bench") {
            if (!precs.empty()) bench.precisions = precs;
            rc = wt::runBenchmark(ctx, bench);
        } else rc = wt::runValidation(ctx, precs.empty() ? wt::Precision::FP16C : precs.front(), suite, diameter);
        ctx.destroy();
        return rc;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "\nerror: %s\n", e.what());
        return 1;
    }
}
