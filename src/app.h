#pragma once
// ============================================================================
// app.h — application state and orchestration
// ============================================================================

#include "gpu.h"
#include "mesh.h"
#include "sim.h"
#include "viz.h"

#include <array>
#include <string>

struct GLFWwindow;
struct ImFont;

namespace vwt {

constexpr int kFramesInFlight = 2;
constexpr int kHist           = 160;   // plot history length

struct StartOptions {
    bool        headless  = false;
    bool        bench     = false;     // pure-throughput benchmark mode
    uint32_t    steps     = 240;       // headless step count
    int         shape     = 0;         // headless: Shape enum value
    float       aoaDeg    = 0.f;       // headless: angle of attack
    uint32_t    gx = 0, gy = 0, gz = 0;   // 0 → preset default
    std::string meshPath;
    bool        lesOff    = false;
    int         collision = -1;        // -1 = default; 0 BGK, 1 regularised, 2 TRT
};

struct FrameData {
    VkCommandPool   pool       = VK_NULL_HANDLE;
    VkCommandBuffer cmd        = VK_NULL_HANDLE;
    VkSemaphore     acquireSem = VK_NULL_HANDLE;
    VkSemaphore     renderSem  = VK_NULL_HANDLE;
    VkFence         fence      = VK_NULL_HANDLE;
    bool            ranAnalysis = false;
    uint32_t        batchSteps  = 0;
};

struct Fonts {
    ImFont* body = nullptr;
    ImFont* mono = nullptr;
    ImFont* big  = nullptr;
};

// Grid presets: coarse / medium / fine
struct GridPreset { const char* name; uint32_t gx, gy, gz; };
inline constexpr GridPreset kGrids[] = {
    { "Coarse — 96 x 64 x 64",    96, 64, 64 },
    { "Medium — 128 x 80 x 80",  128, 80, 80 },
    { "Fine — 176 x 104 x 104",  176, 104, 104 },
};
inline constexpr int kGridCount = int(sizeof(kGrids) / sizeof(kGrids[0]));

class App {
public:
    int run(const StartOptions& opts);

    // ── actions (invoked by UI / shortcuts) ─────────────────────────────────
    void loadPrimitive(Shape s);
    void loadMeshFile(const std::string& path);
    void revoxelize();              // re-rasterize after AoA / yaw edits
    void applyGridPreset(int preset);
    void resetSim();
    void snapshot();
    void toggleFullscreen();
    void setStatus(const std::string& msg);
    void openFileDialog();

    // ── shared state (UI binds directly) ────────────────────────────────────
    GpuContext gpu;
    Solver     solver;
    SliceView  slice;
    SimParams  params;
    UnitScale  units;

    // model
    bool        modelLoaded = false;
    VoxelModel  model;
    std::vector<Tri> meshTris;      // non-empty when loaded from file
    int         primShape = -1;     // >= 0 when model is a primitive
    float       aoaDeg = 0.f, yawDeg = 0.f;

    // run state
    bool     running        = false;
    bool     runSingleBatch = false;   // Step button: one batch, then pause
    uint64_t totalSteps     = 0;
    int      stepsPerFrame  = 8;
    int      gridPreset     = 1;

    // results & stats
    Analysis an;
    float cd = 0.f, cl = 0.f, cdPrev = 0.f, clPrev = 0.f;
    std::array<float, kHist> cdHist{}, clHist{}, resHist{}, fpsHist{};
    int   histIdx = 0;
    float fps = 0.f, mlups = 0.f, frameMs = 0.f;
    GpuTimings tim;
    uint64_t vramUse = 0, vramBudget = 0;

    // view
    int   visMode    = 0;           // 0 vel, 1 Cp, 2 vorticity, 3 Q
    int   axis       = 1;           // 0 XY, 1 XZ, 2 ZY
    int   sliceIndex = -1;          // -1 → centre of current axis
    float scaleVel = 0.12f, maxVort = 0.05f;
    float zoom = 1.f, panX = 0.f, panY = 0.f;
    bool  leftOpen = true, rightOpen = true, showHelp = false;
    std::string statusMsg;
    float statusTtl = 0.f;

    Fonts       fonts;
    GLFWwindow* window = nullptr;

    uint32_t sliceMaxIndex() const;   // depends on axis

private:
    int  runGui(const StartOptions& opts);
    int  runHeadless(const StartOptions& opts);
    int  runBench(const StartOptions& opts);
    void initWindow();
    void initImGui();
    void frame();
    void handleShortcuts();
    void uploadModel();
    void rebuildSliceView();
    void loadConfig();
    void saveConfig();
    static void dropCallback(GLFWwindow* win, int count, const char** paths);

    Swapchain swap_;
    std::array<FrameData, kFramesInFlight> frames_{};
    uint32_t frameIdx_ = 0;
    VkDescriptorPool imguiPool_ = VK_NULL_HANDLE;
    std::string pendingDrop_;
    bool fullscreen_ = false;
    int  savedX_ = 0, savedY_ = 0, savedW_ = 0, savedH_ = 0;
    uint32_t winW_ = 1600, winH_ = 900;
};

} // namespace vwt
