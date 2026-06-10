// ============================================================================
// main.cpp — Virtual Wind Tunnel Entry Point
// ============================================================================

#include "vk_engine.h"
#include "logger.h"
#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <csignal>

#ifdef _WIN32
#include <windows.h>
BOOL WINAPI consoleCtrlHandler(DWORD dwCtrlType) {
    if (dwCtrlType == CTRL_C_EVENT || dwCtrlType == CTRL_BREAK_EVENT) {
        std::cout << "\n[INFO] Received Ctrl+C. Shutting down...\n";
        vwt::Logger::log("Shutdown requested via Ctrl+C");
        exit(0);  // does not return
    }
    return FALSE;
}
#else
#include <unistd.h>
#include <signal.h>
void signalHandler(int sig) {
    std::cout << "\n[INFO] Received signal " << sig << ". Shutting down...\n";
    vwt::Logger::log("Shutdown requested via signal");
    exit(0);
}
#endif

void printUsage(const char* progName) {
    std::cout << "Usage: " << progName << " [options]\n";
    std::cout << "Options:\n";
    std::cout << "  --help            Show this help message\n";
    std::cout << "  --grid <X> <Y> <Z>  Set grid resolution (default: 128 64 64)\n";
    std::cout << "  --resolution <W> <H>  Set window resolution (default: 1600 900)\n";
    std::cout << "  --mesh <path>     Load a mesh file on startup\n";
    std::cout << "  --no-gui          Run in headless mode (no window)\n";
}

int main(int argc, char* argv[]) {
    // Parse CLI arguments
    bool headless = false;
    uint32_t gridX = 128, gridY = 64, gridZ = 64;
    uint32_t resW = 1600, resH = 900;
    std::string meshPath;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help") {
            printUsage(argv[0]);
            return 0;
        } else if (arg == "--grid" && i + 3 < argc) {
            gridX = std::stoi(argv[++i]);
            gridY = std::stoi(argv[++i]);
            gridZ = std::stoi(argv[++i]);
        } else if (arg == "--resolution" && i + 2 < argc) {
            resW = std::stoi(argv[++i]);
            resH = std::stoi(argv[++i]);
        } else if (arg == "--mesh" && i + 1 < argc) {
            meshPath = argv[++i];
        } else if (arg == "--no-gui") {
            headless = true;
        } else {
            std::cerr << "[ERROR] Unknown option: " << arg << "\n";
            printUsage(argv[0]);
            return EXIT_FAILURE;
        }
    }

    // Set up signal handling
#ifdef _WIN32
    SetConsoleCtrlHandler(consoleCtrlHandler, TRUE);
#else
    std::signal(SIGINT, signalHandler);
    std::signal(SIGTERM, signalHandler);
#endif

    vwt::Logger::init("vwt_session.log");
    vwt::Logger::log("Virtual Wind Tunnel starting up...");

    std::cout << "╔═══════════════════════════════════════════════════╗\n";
    std::cout << "║   Virtual Wind Tunnel — D3Q19 LBM Simulation    ║\n";
    std::cout << "║   GPU-Accelerated via Vulkan Compute Shaders    ║\n";
    std::cout << "╚═══════════════════════════════════════════════════╝\n\n";

    vwt::VulkanEngine engine;

    try {
        engine.init();
        engine.run();
        engine.cleanup();
    } catch (const std::exception& e) {
        std::cerr << "\n[FATAL] " << e.what() << "\n";
        if (engine.isInitialized()) {
            engine.cleanup();
        }
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
