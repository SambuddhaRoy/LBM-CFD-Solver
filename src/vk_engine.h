#pragma once
// ============================================================================
// vk_engine.h
// ============================================================================

#include "vk_types.h"
#include "mesh_loader.h"
#include "fluid_solver.h"
#include "renderer.h"
#include <imgui.h>

struct GLFWwindow;

namespace vwt {
namespace benchmark { class AutoBenchmark; }

class VulkanEngine {
public:
    void init();
    void run();
    void runAutoBenchmark();
    void stepBenchmark(uint32_t steps);
    void cleanup();
    bool isInitialized() const { return initialized_; }

    friend class benchmark::AutoBenchmark;

private:
    // ── Init helpers ──────────────────────────────────────────────────────
    void initWindow();
    void initVulkan();
    void initPipelineCache();
    void savePipelineCache();
    void initSwapchain();
    void initFrameData();
    void initRenderPass();
    void initFramebuffers();
    void initImGui();
    void initSimulation();
    void recreateSwapchain();
    void cleanupSwapchain();

    // ── Per-frame ─────────────────────────────────────────────────────────
    void drawFrame();
    void buildCommandBuffer(VkCommandBuffer cmd, uint32_t imageIndex);
    void drawImGui();

    // ── UI panels ─────────────────────────────────────────────────────────
    void drawUI_TopBar();
    void drawUI_Rail();
    void drawUI_Left();
    void drawUI_Viewport();
    void drawUI_Right();
    void drawUI_StatusBar();
    void drawUI_HotkeyOverlay();

    // ── UI sub-helpers ────────────────────────────────────────────────────
    void drawCard_Aero();
    void drawCard_Convergence();
    void drawCard_FlowStats();
    void drawCard_GPU();
    void drawViewportColorbar(ImDrawList* dl, ImVec2 vpMin, ImVec2 vpMax);
    void drawViewportToolbar(float vpX, float vpW, float toolbarY, float toolbarH);
    void drawWelcomeOverlay(ImVec2 vpPos, ImVec2 vpSize);

    // Layout helper — converts an unscaled "design" pixel value to scaled pixels.
    float s(float v) const { return v * dpiScale_; }

    // ── App logic ─────────────────────────────────────────────────────────
    void loadMesh(const std::string& path);
    void loadConfig();
    void saveConfig();
    void processKeyboard();

    static void dropCallback(GLFWwindow* w, int count, const char** paths);
    static void keyCallback(GLFWwindow* w, int key, int scancode, int action, int mods);

    // ── Window ────────────────────────────────────────────────────────────
    GLFWwindow* window_      = nullptr;
    VkExtent2D  windowExtent_= { 1600, 900 };
    bool        initialized_ = false;
    bool        fullscreen_  = false;

    // ── Vulkan core ───────────────────────────────────────────────────────
    VkInstance               instance_       = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT debugMessenger_ = VK_NULL_HANDLE;
    VkPhysicalDevice         physDevice_     = VK_NULL_HANDLE;
    VkDevice                 device_         = VK_NULL_HANDLE;
    VkSurfaceKHR             surface_        = VK_NULL_HANDLE;
    VmaAllocator             allocator_      = VK_NULL_HANDLE;
    VkPipelineCache          pipelineCache_  = VK_NULL_HANDLE;

    // ── Queues ────────────────────────────────────────────────────────────
    VkQueue  graphicsQueue_       = VK_NULL_HANDLE;
    uint32_t graphicsQueueFamily_ = 0;
    VkQueue  computeQueue_        = VK_NULL_HANDLE;
    uint32_t computeQueueFamily_  = 0;
    bool     hasAsyncCompute_     = false;

    // ── Swapchain ─────────────────────────────────────────────────────────
    VkSwapchainKHR           swapchain_    = VK_NULL_HANDLE;
    VkFormat                 swapchainFmt_ = VK_FORMAT_UNDEFINED;
    std::vector<VkImage>     swapImages_;
    std::vector<VkImageView> swapViews_;
    VkRenderPass               renderPass_ = VK_NULL_HANDLE;
    std::vector<VkFramebuffer> framebuffers_;

    // ── Frames in flight ──────────────────────────────────────────────────
    std::array<FrameData, FRAMES_IN_FLIGHT> frames_;
    uint32_t currentFrame_ = 0;
    FrameData& frame() { return frames_[currentFrame_]; }

    // ── ImGui ─────────────────────────────────────────────────────────────
    VkDescriptorPool imguiPool_ = VK_NULL_HANDLE;
    ImFont* fontBody_ = nullptr;   // 15px default
    ImFont* fontMono_ = nullptr;   // 12px monospaced
    float   dpiScale_ = 1.f;       // window content scale (set during initImGui)

    // ── Application modules ───────────────────────────────────────────────
    MeshLoader  meshLoader_;
    FluidSolver fluidSolver_;
    Renderer    renderer_;
    SimParams   simParams_;

    // ── Sim state ─────────────────────────────────────────────────────────
    bool     simRunning_    = false;
    bool     meshLoaded_    = false;
    int      stepsPerFrame_ = 4;
    uint64_t totalSteps_    = 0;
    char     meshPath_[512] = "";
    bool     resizePending_ = false;

    // ── Aero forces ───────────────────────────────────────────────────────
    AeroForces aeroForces_;
    AeroForces aeroPrev_;          // previous sample for delta calc
    float      aeroCD_     = 0.f;
    float      aeroCL_     = 0.f;
    float      aeroCDPrev_ = 0.f;
    float      aeroCLPrev_ = 0.f;
    GpuTimings gpuTimings_;
    uint64_t   aeroUpdateInterval_    = 30;
    bool       aeroDispatchThisFrame_ = false;

    // ── Viewport ─────────────────────────────────────────────────────────
    float    zoomLevel_  = 1.0f;
    float    panX_       = 0.0f;
    float    panY_       = 0.0f;
    int      activeTool_ = 0;  // 0=orbit 1=pan 2=zoom

    // ── Flow controls ─────────────────────────────────────────────────────
    int velocityUnit_ = 0;  // 0=m/s 1=km/h 2=mph 3=knots
    int speedMode_    = 0;  // 0=subsonic 1=supersonic

    // ── Grid / quality ────────────────────────────────────────────────────
    float    gridQuality_ = 1.0f;
    uint32_t baseGridX_   = 128;
    uint32_t baseGridY_   = 64;
    uint32_t baseGridZ_   = 64;

    // ── Navigation rail mode ──────────────────────────────────────────────
    // 0=Simulation 1=Mesh 2=Probes 3=Compare 4=Library 5=Settings
    int  railMode_         = 0;
    bool leftPanelOpen_    = true;
    bool rightPanelOpen_   = true;
    bool showHotkeys_      = false;
    bool showAeroCard_     = true;
    bool showConvCard_     = true;
    bool showFlowCard_     = true;
    bool showGpuCard_      = true;

    // ── Performance history ───────────────────────────────────────────────
    static constexpr int kHist = 120;
    float    fpsHistory_[kHist]      = {};
    float    residualHistory_[kHist] = {};
    int      fpsHistIdx_             = 0;
    float    simResidual_            = 1.0f;
    float    avgFrameMs_             = 16.6f;
    float    lastResidualLog_        = 0.f;

    // ── VRAM ──────────────────────────────────────────────────────────────
    uint64_t vramBudget_ = 0;
    uint64_t vramUsage_  = 0;

    // ── GPU info ──────────────────────────────────────────────────────────
    char gpuName_[256] = "Unknown";

    bool benchmarkMode_ = false;

    DeletionQueue mainDQ_;
};

} // namespace vwt
