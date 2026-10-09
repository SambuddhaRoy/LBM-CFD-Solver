#pragma once
// ============================================================================
// app.hpp — the interactive wind tunnel: window, swapchain, frame loop, UI.
// ============================================================================

#include "fluid.hpp"
#include "geometry.hpp"
#include "render.hpp"
#include "solver.hpp"
#include "vk.hpp"

#include <array>
#include <deque>
#include <string>
#include <vector>

struct GLFWwindow;

namespace wt {

struct GridPreset { const char* name; uint32_t nx, ny, nz; };
inline constexpr GridPreset kGrids[] = {
    {"256 x 128 x 128",   256, 128, 128},
    {"384 x 192 x 192",   384, 192, 192},
    {"512 x 256 x 256",   512, 256, 256},
    {"640 x 320 x 320",   640, 320, 320},
    {"768 x 384 x 384",   768, 384, 384},
    {"1024 x 448 x 448", 1024, 448, 448},
    {"1024 x 512 x 512", 1024, 512, 512},
};

// Initial state from the command line. With a capture path the app saves the
// whole window as a PNG after `captureFrames` frames and exits: reproducible
// screenshots, and an end-to-end check of the interactive path.
struct StartSetup {
    Precision   precision = Precision::FP16C;
    int         preset = 1;              // index into kGrids
    Shape       shape = Shape::Sphere;
    float       pitch = 0.f;
    int         view3d = 0;
    Field       field = Field::Speed;
    float       zoom = 1.f;              // relative to the fitted view
    std::string meshPath;                // model to load at start
    bool        realtime = false;        // start in real-time mode
    std::string capturePath;
    uint32_t    captureFrames = 300;
};

class App {
public:
    int run(const StartSetup& setup);

private:
    struct Frame {
        VkCommandPool   pool = VK_NULL_HANDLE;
        VkCommandBuffer cmd  = VK_NULL_HANDLE;
        VkFence         fence = VK_NULL_HANDLE;
        VkSemaphore     acquired = VK_NULL_HANDLE;
        bool            stepped = false;     // this frame's submission ran steps
        uint32_t        steps = 0;
    };

    void initWindow();
    void initSwapchain();
    void destroySwapchain();
    void initImGui();
    void rebuildSolver();                    // grid or precision changed
    void applyGeometry();                    // body changed: keeps the flow
    void resetFlow();
    void updateScaling();                    // physical units -> tau
    void fitView();
    glm::vec3 bodyBoundsHalf() const;
    double referenceArea() const;

    void frame();
    void collectResults(Frame& f, uint32_t slot);
    void drawUi();
    void drawControls(float width, float height);
    void drawStats(float x, float width, float height);
    void drawViewport(float x, float width, float height);
    void handleViewportInput(float w, float h);
    void handleKeys();
    void loadMeshFile(const std::string& path);
    void snapshot();
    void saveCapture(uint32_t width, uint32_t height);

    GLFWwindow*  window_ = nullptr;
    gpu::Context ctx_;
    Solver       solver_;
    Geometry     geometry_;
    Renderer     renderer_;

    VkSwapchainKHR           swapchain_ = VK_NULL_HANDLE;
    VkFormat                 swapFormat_ = VK_FORMAT_UNDEFINED;
    VkExtent2D               swapExtent_{};
    std::vector<VkImage>     swapImages_;
    std::vector<VkImageView> swapViews_;
    std::vector<VkSemaphore> renderDone_;     // one per swapchain image
    bool                     swapDirty_ = false;
    std::array<Frame, Solver::kSlots> frames_;
    uint64_t frameCount_ = 0;

    // Simulation settings.
    int       gridIndex_ = 1;
    Precision precision_ = Precision::FP16C;
    Body      body_;
    std::string meshName_, meshNote_;
    bool      running_ = true;
    bool      geometryDirty_ = false, resetRequested_ = false, rebuildRequested_ = false;
    Ambient    ambient_;                     // fluid, temperature, pressure
    Properties props_;                       // derived from ambient_
    float      altitude_ = 0.f;              // m, standard-atmosphere helper
    bool       absolutePressure_ = false;    // colour bar in absolute pressure
    int        flowModel_ = 0;               // 0 auto, 1 laminar (no model), 2 turbulent (LES)
    float      reResolved_ = 0.f;            // highest Re the grid resolves without a model
    float     speed_ = 30.f;                 // m/s
    float     length_ = 1.f;                 // m, body reference length
    float     reRequested_ = 0.f, reSimulated_ = 0.f, mach_ = 0.f;
    float     dx_ = 0.f, dt_ = 0.f;          // m, s per cell / step
    bool      tauClamped_ = false;

    // Real-time mode: the simulated clock keeps pace with the wall clock, so
    // the air crosses the model at its actual speed. The tunnel is fitted
    // around the body and the resolution chosen so the GPU can keep up.
    bool      realtime_ = false;
    int       savedGridIndex_ = 1;          // preset to return to
    Body      savedBody_;                   // size and place in that preset
    bool      restoreBody_ = false;         // leaving real time: put them back
    uint32_t  rtGrid_[3] = {0, 0, 0};       // fitted tunnel, cells
    float     rtLength_ = 0.f;              // planned cells along the body
    glm::vec3 rtCenter_{0.f};
    double    rtDebt_ = 0, rtLastWall_ = 0; // simulated seconds owed; last frame time
    float     rtRatio_ = 0.f;               // achieved simulated / wall time
    double    rtCheckAt_ = 0;               // next pacing check (wall clock)
    double    rtThroughput_ = 0;            // throughput the current fit assumed, LUPS
    void planRealtime(double throughput);

    // Measurements.
    uint32_t stepsPerFrame_ = 4;
    double   mlups_ = 0, simMs_ = 0;
    std::array<double, 3> force_{};
    Stats    stats_;
    std::deque<float> histResidual_, histCd_, histCl_;

    // View.
    View  view_;
    float viewportW_ = 1, viewportH_ = 1;
    bool  autoRange_ = true;
    bool  fitPending_ = true;      // fit once the viewport knows its size
    float startZoom_ = 1.f;        // applied to the first fit only
    float rangeScale_ = 1.f;
    std::string capturePath_;
    uint32_t    captureFrames_ = 0;
    gpu::Buffer captureBuf_;
    std::string status_;
    double statusUntil_ = 0;
};

} // namespace wt
