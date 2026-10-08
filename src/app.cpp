// ============================================================================
// app.cpp — the interactive wind tunnel (see app.hpp)
// ============================================================================

#include "app.hpp"

#include <GLFW/glfw3.h>
#include <VkBootstrap.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <cstring>
#include <filesystem>
#include <stb_image_write.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <commdlg.h>
#endif

namespace wt {

namespace {

constexpr size_t kHistory = 240;

void push(std::deque<float>& h, float v) {
    h.push_back(v);
    if (h.size() > kHistory) h.pop_front();
}

// Same colour maps as render.comp, for the colour bar.
ImU32 turboColour(float t) {
    t = std::clamp(t, 0.f, 1.f);
    const float t2 = t*t, t3 = t2*t, t4 = t2*t2, t5 = t4*t;
    const float r = 0.13572138f + 4.61539260f*t - 42.66032258f*t2 + 132.13108234f*t3 - 152.94239396f*t4 + 59.28637943f*t5;
    const float g = 0.09140261f + 2.19418839f*t + 4.84296658f*t2 - 14.18503333f*t3 + 4.27729857f*t4 + 2.82956604f*t5;
    const float b = 0.10667330f + 12.64194608f*t - 60.58204836f*t2 + 110.36276771f*t3 - 89.90310912f*t4 + 27.34824973f*t5;
    return ImGui::ColorConvertFloat4ToU32({std::clamp(r, 0.f, 1.f), std::clamp(g, 0.f, 1.f), std::clamp(b, 0.f, 1.f), 1.f});
}
ImU32 divergingColour(float t) {
    t = std::clamp(t, 0.f, 1.f);
    const ImVec4 lo{0.23f, 0.30f, 0.75f, 1}, mid{0.87f, 0.87f, 0.87f, 1}, hi{0.71f, 0.02f, 0.15f, 1};
    auto mix = [](ImVec4 a, ImVec4 b, float s) {
        return ImVec4{a.x + (b.x - a.x)*s, a.y + (b.y - a.y)*s, a.z + (b.z - a.z)*s, 1};
    };
    return ImGui::ColorConvertFloat4ToU32(t < 0.5f ? mix(lo, mid, t*2) : mix(mid, hi, t*2 - 1));
}

std::string formatSi(double v, const char* unit) {
    char buf[64];
    const double a = std::abs(v);
    if (a != 0 && (a < 1e-2 || a >= 1e5)) std::snprintf(buf, sizeof buf, "%.2e %s", v, unit);
    else std::snprintf(buf, sizeof buf, "%.3g %s", v, unit);
    return buf;
}

#ifdef _WIN32
std::string openModelDialog(GLFWwindow*) {
    wchar_t file[MAX_PATH] = L"";
    OPENFILENAMEW ofn{};
    ofn.lStructSize = sizeof ofn;
    ofn.lpstrFilter = L"3D models\0*.stl;*.obj;*.gltf;*.glb;*.fbx;*.ply;*.3mf;*.dae\0All files\0*.*\0";
    ofn.lpstrFile = file;
    ofn.nMaxFile = MAX_PATH;
    ofn.Flags = OFN_FILEMUSTEXIST | OFN_NOCHANGEDIR;
    if (!GetOpenFileNameW(&ofn)) return {};
    const int n = WideCharToMultiByte(CP_UTF8, 0, file, -1, nullptr, 0, nullptr, nullptr);
    std::string out(size_t(std::max(n - 1, 0)), '\0');
    WideCharToMultiByte(CP_UTF8, 0, file, -1, out.data(), n, nullptr, nullptr);
    return out;
}
#endif

} // namespace

// ─── Setup ──────────────────────────────────────────────────────────────────

int App::run(const StartSetup& setup) {
    precision_     = setup.precision;
    gridIndex_     = std::clamp(setup.preset, 0, int(std::size(kGrids)) - 1);
    capturePath_   = setup.capturePath;
    captureFrames_ = setup.captureFrames;
    body_.shape    = setup.shape;
    body_.pitch    = setup.pitch;
    view_.mode3d   = setup.view3d;
    view_.field    = setup.field;
    startZoom_     = setup.zoom;
    initWindow();
    ctx_.init(window_);
    initSwapchain();
    for (auto& f : frames_) {
        VkCommandPoolCreateInfo cp{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        cp.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        cp.queueFamilyIndex = ctx_.family;
        gpu::check(vkCreateCommandPool(ctx_.device, &cp, nullptr, &f.pool), "frame pool");
        VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        ai.commandPool = f.pool;
        ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 1;
        gpu::check(vkAllocateCommandBuffers(ctx_.device, &ai, &f.cmd), "frame command buffer");
        VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        fi.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        gpu::check(vkCreateFence(ctx_.device, &fi, nullptr, &f.fence), "frame fence");
        VkSemaphoreCreateInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        gpu::check(vkCreateSemaphore(ctx_.device, &si, nullptr, &f.acquired), "frame semaphore");
    }
    initImGui();
    geometry_.create(ctx_);
    renderer_.create(ctx_);
    rebuildSolver();
    if (!setup.meshPath.empty()) loadMeshFile(setup.meshPath);

    while (!glfwWindowShouldClose(window_)) frame();

    vkDeviceWaitIdle(ctx_.device);
    ctx_.destroyBuffer(captureBuf_);
    renderer_.destroy();
    geometry_.destroy();
    solver_.destroy();
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    for (auto& f : frames_) {
        vkDestroySemaphore(ctx_.device, f.acquired, nullptr);
        vkDestroyFence(ctx_.device, f.fence, nullptr);
        vkDestroyCommandPool(ctx_.device, f.pool, nullptr);
    }
    destroySwapchain();
    ctx_.destroy();
    glfwDestroyWindow(window_);
    glfwTerminate();
    return 0;
}

void App::initWindow() {
    if (!glfwInit()) throw std::runtime_error("GLFW init failed");
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_MAXIMIZED, GLFW_TRUE);
    window_ = glfwCreateWindow(1600, 960, "Wind Tunnel v3", nullptr, nullptr);
    if (!window_) throw std::runtime_error("window creation failed");
    glfwSetWindowUserPointer(window_, this);
    glfwSetDropCallback(window_, [](GLFWwindow* w, int count, const char** paths) {
        if (count > 0) static_cast<App*>(glfwGetWindowUserPointer(w))->loadMeshFile(paths[0]);
    });
    glfwSetFramebufferSizeCallback(window_, [](GLFWwindow* w, int, int) {
        static_cast<App*>(glfwGetWindowUserPointer(w))->swapDirty_ = true;
    });
}

void App::initSwapchain() {
    int w = 0, h = 0;
    glfwGetFramebufferSize(window_, &w, &h);
    vkb::SwapchainBuilder sb(ctx_.phys, ctx_.device, ctx_.surface);
    auto sc = sb.set_desired_format({VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR})
                .set_desired_present_mode(VK_PRESENT_MODE_MAILBOX_KHR)
                .add_fallback_present_mode(VK_PRESENT_MODE_FIFO_KHR)
                .set_desired_extent(uint32_t(w), uint32_t(h))
                .set_image_usage_flags(VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT)
                .set_old_swapchain(swapchain_)
                .build();
    if (!sc) throw std::runtime_error("swapchain: " + sc.error().message());
    if (swapchain_) destroySwapchain();
    swapchain_  = sc->swapchain;
    swapFormat_ = sc->image_format;
    swapExtent_ = sc->extent;
    swapImages_ = sc->get_images().value();
    swapViews_  = sc->get_image_views().value();
    renderDone_.resize(swapImages_.size());
    for (auto& s : renderDone_) {
        VkSemaphoreCreateInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        gpu::check(vkCreateSemaphore(ctx_.device, &si, nullptr, &s), "present semaphore");
    }
}

void App::destroySwapchain() {
    for (auto v : swapViews_) vkDestroyImageView(ctx_.device, v, nullptr);
    for (auto s : renderDone_) vkDestroySemaphore(ctx_.device, s, nullptr);
    swapViews_.clear(); renderDone_.clear(); swapImages_.clear();
    vkDestroySwapchainKHR(ctx_.device, swapchain_, nullptr);
    swapchain_ = VK_NULL_HANDLE;
}

void App::initImGui() {
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.IniFilename = nullptr;
    float sx = 1, sy = 1;
    glfwGetWindowContentScale(window_, &sx, &sy);
    const float scale = std::max(sx, 1.f);

    ImGui::StyleColorsDark();
    ImGuiStyle& st = ImGui::GetStyle();
    st.WindowRounding = 0; st.FrameRounding = 3; st.GrabRounding = 3;
    st.WindowBorderSize = 0; st.FramePadding = {6, 4}; st.ItemSpacing = {8, 6};
    st.Colors[ImGuiCol_WindowBg] = {0.09f, 0.10f, 0.12f, 1};
    st.Colors[ImGuiCol_Header]   = {0.18f, 0.32f, 0.48f, 0.8f};
    st.ScaleAllSizes(scale);

    const char* fonts[] = {"C:/Windows/Fonts/segoeui.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                           "/usr/share/fonts/TTF/DejaVuSans.ttf"};
    for (const char* f : fonts)
        if (std::filesystem::exists(f)) { io.Fonts->AddFontFromFileTTF(f, 16.f * scale); break; }

    ImGui_ImplGlfw_InitForVulkan(window_, true);
    ImGui_ImplVulkan_InitInfo ii{};
    ii.ApiVersion         = VK_API_VERSION_1_3;
    ii.Instance           = ctx_.instance;
    ii.PhysicalDevice     = ctx_.phys;
    ii.Device             = ctx_.device;
    ii.QueueFamily        = ctx_.family;
    ii.Queue              = ctx_.queue;
    ii.DescriptorPoolSize = 64;
    ii.MinImageCount      = 2;
    ii.ImageCount         = uint32_t(swapImages_.size());
    ii.PipelineCache      = ctx_.pipelineCache;
    ii.UseDynamicRendering = true;
    ii.PipelineInfoMain.PipelineRenderingCreateInfo = {VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR};
    ii.PipelineInfoMain.PipelineRenderingCreateInfo.colorAttachmentCount = 1;
    ii.PipelineInfoMain.PipelineRenderingCreateInfo.pColorAttachmentFormats = &swapFormat_;
    ImGui_ImplVulkan_Init(&ii);
}

// ─── Simulation management ──────────────────────────────────────────────────

void App::rebuildSolver() {
    vkDeviceWaitIdle(ctx_.device);
    const GridConfig old = solver_.grid();
    const bool had = solver_.created();
    solver_.destroy();

    const GridPreset& p = kGrids[gridIndex_];
    GridConfig g;
    g.nx = p.nx; g.ny = p.ny; g.nz = p.nz; g.precision = precision_;
    try {
        solver_.create(ctx_, g);
    } catch (const std::exception& e) {
        status_ = std::string("Grid not created: ") + e.what();
        statusUntil_ = glfwGetTime() + 6;
        gridIndex_ = 0;
        g.nx = kGrids[0].nx; g.ny = kGrids[0].ny; g.nz = kGrids[0].nz;
        solver_.create(ctx_, g);
    }

    // Keep the body where it was relative to the tunnel, scaled with it.
    if (had && old.ny > 0) {
        const glm::vec3 frac = body_.center / glm::vec3(float(old.nx), float(old.ny), float(old.nz));
        body_.center = frac * glm::vec3(float(g.nx), float(g.ny), float(g.nz));
        const float s = float(g.ny) / float(old.ny);
        body_.length *= s;
        body_.span *= s;
    } else {
        body_.center = {0.3f * float(g.nx), 0.5f * float(g.ny) + 0.25f, 0.5f * float(g.nz) + 0.25f};
        body_.length = 0.25f * float(std::min(g.ny, g.nz));
    }
    solver_.flow.uIn = 0.08f;          // LES is decided in updateScaling()
    for (auto& f : frames_) f.stepped = false;
    applyGeometry();
    resetFlow();
    view_.slice = body_.center.z;
    fitPending_ = true;
}

void App::applyGeometry() {
    vkDeviceWaitIdle(ctx_.device);
    if (body_.shape == Shape::Mesh && !geometry_.hasMesh()) body_.shape = Shape::Sphere;
    geometry_.apply(solver_, body_);
    updateScaling();
}

void App::resetFlow() {
    vkDeviceWaitIdle(ctx_.device);
    updateScaling();
    ctx_.submitNow([&](VkCommandBuffer cmd) { solver_.reset(cmd); });
    histResidual_.clear(); histCd_.clear(); histCl_.clear();
    for (auto& f : frames_) f.stepped = false;
}

void App::updateScaling() {
    props_ = properties(ambient_);
    reRequested_ = float(speed_ * length_ / props_.nu);
    mach_ = float(speed_ / props_.sound);
    const float cells = std::max(body_.length, 1.f);
    const float u = solver_.flow.uIn;
    // Viscosity follows from the Reynolds number the body should see. The
    // floor keeps tau representable in FP32 with a usable margin above 1/2;
    // below it the requested Re is out of reach and the UI says so.
    const float tauMin = 0.5f + 2e-5f;
    float tau = 0.5f + 3.f * u * cells / std::max(reRequested_, 1e-3f);
    tauClamped_ = tau < tauMin;
    tau = std::max(tau, tauMin);
    solver_.flow.tau = tau;
    reSimulated_ = 3.f * u * cells / (tau - 0.5f);

    // Turbulence model. A subgrid model is only needed once the grid can no
    // longer resolve the smallest eddies: with the Kolmogorov scale
    // eta ~ L Re^(-3/4) and a resolved simulation needing dx <~ 2 eta, that
    // is Re > (2 L / dx)^(4/3). Below it the flow is computed directly; the
    // Smagorinsky model would only add spurious viscosity (measured: +4% at
    // tau = 0.515, +10% at 0.505 in laminar channel flow).
    reResolved_ = std::pow(2.f * cells, 4.f / 3.f);
    const bool les = flowModel_ == 2 || (flowModel_ == 0 && reSimulated_ > reResolved_);
    solver_.flow.smagorinsky = les ? 0.12f : 0.f;
    dx_ = length_ / cells;
    dt_ = speed_ > 0 ? u * dx_ / speed_ : 0.f;
}

// Force coefficients use the convention of the body type: planform area
// (chord x span) for a wing, projected frontal area for everything else.
// Normalising a pitched wing by its frontal area would inflate C_L severalfold.
double App::referenceArea() const {
    if (body_.shape == Shape::Wing)
        return double(body_.length) * (body_.span > 0 ? body_.span : double(solver_.grid().nz));
    return std::max<uint32_t>(geometry_.frontalCells(), 1);
}

glm::vec3 App::bodyBoundsHalf() const {
    const auto& g = solver_.grid();
    const float r = 0.5f * body_.length;
    const bool through = body_.span <= 0 && (body_.shape == Shape::Cylinder || body_.shape == Shape::Wing);
    if (through) return glm::vec3(float(std::max({g.nx, g.ny, g.nz})));
    const float ext = std::max(r, 0.5f * body_.span);
    return glm::vec3(ext * 1.75f + 10.f);    // covers any rotation, plus the SDF band
}

void App::fitView() {
    const auto& g = solver_.grid();
    const glm::vec3 n(float(g.nx), float(g.ny), float(g.nz));
    glm::vec2 plane = view_.axis == 2 ? glm::vec2(n.x, n.y) : view_.axis == 1 ? glm::vec2(n.x, n.z)
                                                                              : glm::vec2(n.z, n.y);
    view_.center = 0.5f * (plane - 1.f);
    view_.cellsPerPixel = 1.04f * std::max(plane.x / std::max(viewportW_, 1.f), plane.y / std::max(viewportH_, 1.f));
    view_.target = 0.5f * (n - 1.f);
    view_.distance = 1.4f * std::max({n.x, n.y, n.z});
    view_.yaw = -55.f; view_.pitch = 28.f;
}

// ─── Frame loop ─────────────────────────────────────────────────────────────

void App::collectResults(Frame& f, uint32_t slot) {
    if (!f.stepped) return;
    f.stepped = false;
    simMs_ = solver_.lastBatchMs(slot);
    if (simMs_ > 0) mlups_ = double(solver_.cells()) * f.steps / (simMs_ * 1e3);
    force_ = solver_.forces(slot);
    stats_ = solver_.stats(slot);
    const double area = referenceArea();
    const double q = 0.5 * solver_.flow.uIn * solver_.flow.uIn;
    push(histCd_, float(force_[0] / (q * area)));
    push(histCl_, float(force_[1] / (q * area)));
    push(histResidual_, float(std::log10(std::max(stats_.residual, 1e-12f))));
    // Fill ~14 ms of GPU time per frame with steps, so the solver runs as
    // fast as the display allows whatever the grid size.
    if (simMs_ > 0) {
        const double want = f.steps * 14.0 / simMs_;
        stepsPerFrame_ = uint32_t(std::clamp(0.7 * stepsPerFrame_ + 0.3 * want, 1.0, 4000.0));
    }
}

void App::frame() {
    glfwPollEvents();
    int fw = 0, fh = 0;
    glfwGetFramebufferSize(window_, &fw, &fh);
    if (fw == 0 || fh == 0) { glfwWaitEvents(); return; }
    if (swapDirty_) {
        vkDeviceWaitIdle(ctx_.device);
        initSwapchain();
        ImGui_ImplVulkan_SetMinImageCount(2);
        swapDirty_ = false;
    }

    const uint32_t slot = uint32_t(frameCount_ % Solver::kSlots);
    Frame& f = frames_[slot];
    vkWaitForFences(ctx_.device, 1, &f.fence, VK_TRUE, UINT64_MAX);
    collectResults(f, slot);
    renderer_.resize(slot, uint32_t(viewportW_), uint32_t(viewportH_));

    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    drawUi();
    handleKeys();
    ImGui::Render();

    if (rebuildRequested_) { rebuildRequested_ = false; geometryDirty_ = false; rebuildSolver(); }
    if (geometryDirty_)    { geometryDirty_ = false; applyGeometry(); }
    if (resetRequested_)   { resetRequested_ = false; resetFlow(); }

    uint32_t image = 0;
    const VkResult acq = vkAcquireNextImageKHR(ctx_.device, swapchain_, UINT64_MAX, f.acquired,
                                               VK_NULL_HANDLE, &image);
    if (acq == VK_ERROR_OUT_OF_DATE_KHR) { swapDirty_ = true; return; }
    vkResetFences(ctx_.device, 1, &f.fence);

    VkCommandBuffer cmd = f.cmd;
    vkResetCommandBuffer(cmd, 0);
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bi);
    gpu::Context::computeBarrier(cmd);
    if (running_) {
        solver_.recordSteps(cmd, stepsPerFrame_, true, true, slot, true);
        solver_.recordStats(cmd, slot);
        f.stepped = true;
        f.steps = stepsPerFrame_;
    }
    const glm::vec3 half = bodyBoundsHalf();
    renderer_.record(cmd, slot, solver_, view_, body_.center - half, body_.center + half);

    VkImageMemoryBarrier2 ib{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    ib.srcStageMask  = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    ib.dstStageMask  = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    ib.dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
    ib.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    ib.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    ib.image = swapImages_[image];
    ib.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    di.imageMemoryBarrierCount = 1;
    di.pImageMemoryBarriers = &ib;
    vkCmdPipelineBarrier2(cmd, &di);

    VkRenderingAttachmentInfo ca{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    ca.imageView   = swapViews_[image];
    ca.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    ca.loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR;
    ca.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;
    ca.clearValue.color = {{0.05f, 0.05f, 0.06f, 1.f}};
    VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
    ri.renderArea = {{0, 0}, swapExtent_};
    ri.layerCount = 1;
    ri.colorAttachmentCount = 1;
    ri.pColorAttachments = &ca;
    vkCmdBeginRendering(cmd, &ri);
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);
    vkCmdEndRendering(cmd);

    const bool capture = !capturePath_.empty() && frameCount_ == captureFrames_;
    ib.srcStageMask  = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    ib.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
    ib.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    if (capture) {
        // Copy the finished window image out before presenting it.
        const VkDeviceSize bytes = VkDeviceSize(swapExtent_.width) * swapExtent_.height * 4;
        if (captureBuf_.size < bytes) {
            ctx_.destroyBuffer(captureBuf_);
            captureBuf_ = ctx_.createBuffer(bytes, 0, gpu::Mem::Readback);
        }
        ib.dstStageMask  = VK_PIPELINE_STAGE_2_COPY_BIT;
        ib.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
        ib.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        vkCmdPipelineBarrier2(cmd, &di);
        VkBufferImageCopy c{};
        c.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        c.imageExtent = {swapExtent_.width, swapExtent_.height, 1};
        vkCmdCopyImageToBuffer(cmd, swapImages_[image], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, captureBuf_.buf, 1, &c);
        ib.srcStageMask  = VK_PIPELINE_STAGE_2_COPY_BIT;
        ib.srcAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
        ib.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    }
    ib.dstStageMask  = VK_PIPELINE_STAGE_2_NONE;
    ib.dstAccessMask = 0;
    ib.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    vkCmdPipelineBarrier2(cmd, &di);
    vkEndCommandBuffer(cmd);

    const VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.waitSemaphoreCount   = 1;
    si.pWaitSemaphores      = &f.acquired;
    si.pWaitDstStageMask    = &waitStage;
    si.commandBufferCount   = 1;
    si.pCommandBuffers      = &cmd;
    si.signalSemaphoreCount = 1;
    si.pSignalSemaphores    = &renderDone_[image];
    gpu::check(vkQueueSubmit(ctx_.queue, 1, &si, f.fence), "frame submit");

    VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    pi.waitSemaphoreCount = 1;
    pi.pWaitSemaphores    = &renderDone_[image];
    pi.swapchainCount     = 1;
    pi.pSwapchains        = &swapchain_;
    pi.pImageIndices      = &image;
    const VkResult pr = vkQueuePresentKHR(ctx_.queue, &pi);
    if (pr == VK_ERROR_OUT_OF_DATE_KHR || pr == VK_SUBOPTIMAL_KHR) swapDirty_ = true;
    if (capture) {
        vkDeviceWaitIdle(ctx_.device);
        saveCapture(swapExtent_.width, swapExtent_.height);
        glfwSetWindowShouldClose(window_, GLFW_TRUE);
    }
    ++frameCount_;
}

// ─── UI ─────────────────────────────────────────────────────────────────────

void App::drawUi() {
    const ImGuiIO& io = ImGui::GetIO();
    const float left = 340.f * ImGui::GetStyle().FramePadding.x / 6.f;
    const float right = 320.f * ImGui::GetStyle().FramePadding.x / 6.f;
    const float w = io.DisplaySize.x, h = io.DisplaySize.y;
    drawControls(left, h);
    drawViewport(left, std::max(w - left - right, 50.f), h);
    drawStats(w - right, right, h);
}

void App::drawControls(float width, float height) {
    ImGui::SetNextWindowPos({0, 0});
    ImGui::SetNextWindowSize({width, height});
    ImGui::Begin("Controls", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove);
    ImGui::PushItemWidth(-110.f * ImGui::GetStyle().FramePadding.x / 6.f);

    ImGui::TextDisabled("WIND TUNNEL v3");
    if (ImGui::Button(running_ ? "Pause (Space)" : "Run (Space)", {ImGui::GetContentRegionAvail().x * 0.5f - 4, 0}))
        running_ = !running_;
    ImGui::SameLine();
    if (ImGui::Button("Reset flow (R)", {-1, 0})) resetRequested_ = true;

    if (ImGui::CollapsingHeader("Body", ImGuiTreeNodeFlags_DefaultOpen)) {
        const char* shapes[] = {"Sphere", "Cube", "Cylinder", "NACA 0012 wing", "Imported model"};
        int s = int(body_.shape);
        if (ImGui::Combo("Shape", &s, shapes, 5)) {
            if (Shape(s) == Shape::Mesh && !geometry_.hasMesh()) {
#ifdef _WIN32
                const std::string p = openModelDialog(window_);
                if (!p.empty()) loadMeshFile(p);
#else
                status_ = "Drop a model file onto the window to load it";
                statusUntil_ = glfwGetTime() + 5;
#endif
            } else {
                body_.shape = Shape(s);
                geometryDirty_ = true;
            }
        }
#ifdef _WIN32
        if (ImGui::Button("Load model file...", {-1, 0})) {
            const std::string p = openModelDialog(window_);
            if (!p.empty()) loadMeshFile(p);
        }
#endif
        if (body_.shape == Shape::Mesh) ImGui::TextDisabled("%s", meshName_.c_str());
        else ImGui::TextDisabled("Or drop an STL/OBJ/glTF/FBX/PLY file on the window");

        const auto& g = solver_.grid();
        const float maxLen = 0.9f * float(std::max({g.nx, g.ny, g.nz}));
        bool ch = false;
        ch |= ImGui::SliderFloat("Size (cells)", &body_.length, 4.f, maxLen, "%.0f");
        if (body_.shape == Shape::Cylinder || body_.shape == Shape::Wing)
            ch |= ImGui::SliderFloat("Span (cells)", &body_.span, 0.f, float(g.nz), body_.span <= 0 ? "full" : "%.0f");
        ch |= ImGui::SliderFloat("Pitch", &body_.pitch, -90.f, 90.f, "%.1f deg");
        ch |= ImGui::SliderFloat("Yaw", &body_.yaw, -180.f, 180.f, "%.1f deg");
        ch |= ImGui::SliderFloat("Roll", &body_.roll, -180.f, 180.f, "%.1f deg");
        ch |= ImGui::SliderFloat("Position x", &body_.center.x, 0.f, float(g.nx), "%.0f");
        ch |= ImGui::SliderFloat("Position y", &body_.center.y, 0.f, float(g.ny), "%.0f");
        ch |= ImGui::SliderFloat("Position z", &body_.center.z, 0.f, float(g.nz), "%.0f");
        if (ImGui::Button("Zero angles", {-1, 0})) { body_.pitch = body_.yaw = body_.roll = 0; ch = true; }
        // The geometry is re-voxelized on the GPU while the flow keeps
        // running, so rotating the body visibly reorganises the wake.
        geometryDirty_ |= ch;
    }

    if (ImGui::CollapsingHeader("Flow", ImGuiTreeNodeFlags_DefaultOpen)) {
        bool ch = false;
        int kind = int(ambient_.kind);
        const char* kinds[] = {"Air", "Water", "Carbon dioxide"};
        if (ImGui::Combo("Fluid", &kind, kinds, 3)) {
            ambient_ = {FluidKind(kind), 293.15, 101325.0};
            ch = true;
        }
        // Ambient state. Temperature and pressure set the fluid's density,
        // viscosity and speed of sound, hence Re, Mach and every
        // dimensional result. The flow itself is isothermal.
        float tC = float(ambient_.T - 273.15);
        const bool liquid = ambient_.kind == FluidKind::Water;
        if (ImGui::SliderFloat("Temperature", &tC, liquid ? 0.5f : -90.f, liquid ? 99.5f : 400.f, "%.1f C")) {
            ambient_.T = double(tC) + 273.15;
            ch = true;
        }
        float pKpa = float(ambient_.p / 1000.0);
        if (ImGui::SliderFloat("Pressure", &pKpa, 0.1f, 10000.f, "%.2f kPa", ImGuiSliderFlags_Logarithmic)) {
            ambient_.p = double(pKpa) * 1000.0;
            ch = true;
        }
        if (ambient_.kind == FluidKind::Air &&
            ImGui::SliderFloat("ISA altitude", &altitude_, 0.f, 20000.f, "%.0f m")) {
            standardAtmosphere(altitude_, ambient_.T, ambient_.p);
            ch = true;
        }
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("International Standard Atmosphere: sets temperature and pressure\n"
                              "for the altitude.");
        if (ImGui::Button("Sea level", {ImGui::GetContentRegionAvail().x / 3 - 4, 0})) {
            ambient_ = {FluidKind::Air, 288.15, 101325.0}; altitude_ = 0; ch = true;
        }
        ImGui::SameLine();
        if (ImGui::Button("Mars", {ImGui::GetContentRegionAvail().x / 2 - 4, 0})) {
            ambient_ = {FluidKind::CO2, 210.0, 610.0}; ch = true;   // mean surface conditions
        }
        ImGui::SameLine();
        if (ImGui::Button("Water", {-1, 0})) { ambient_ = {FluidKind::Water, 293.15, 101325.0}; ch = true; }
        ImGui::TextDisabled("rho %.4g kg/m3   mu %.3g Pa s\nnu %.3g m2/s   c %.0f m/s",
                            props_.rho, props_.mu, props_.nu, props_.sound);
        ch |= ImGui::SliderFloat("Wind speed", &speed_, 0.5f, 340.f, "%.1f m/s", ImGuiSliderFlags_Logarithmic);
        ch |= ImGui::SliderFloat("Body length", &length_, 0.005f, 50.f, "%.3f m", ImGuiSliderFlags_Logarithmic);
        ch |= ImGui::SliderFloat("Lattice speed", &solver_.flow.uIn, 0.02f, 0.15f, "%.3f");
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Inlet speed in lattice units. Higher develops the flow faster per step;\n"
                              "above ~0.1 compressibility error grows (lattice Mach = %.2f).",
                              solver_.flow.uIn * std::sqrt(3.f));
        if (ch) updateScaling();
        const char* models[] = {"Auto", "Laminar (resolved)", "Turbulent (LES)"};
        if (ImGui::Combo("Flow model", &flowModel_, models, 3)) updateScaling();
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Laminar: the Navier-Stokes equations are solved directly, no turbulence model.\n"
                              "Turbulent: Smagorinsky LES for eddies smaller than a cell.\n"
                              "Auto: LES only above Re %.0f, where this body's resolution stops\n"
                              "resolving the smallest eddies.", reResolved_);
        ImGui::TextDisabled("Now: %s", solver_.flow.smagorinsky > 0 ? "turbulent, LES on" : "laminar, resolved");
        ImGui::Checkbox("Interpolated walls", &solver_.flow.bouzidi);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Bouzidi bounce-back: the wall sits at its true sub-cell position\n"
                              "instead of on the voxel staircase.");
        ImGui::SliderFloat("Inlet turbulence", &solver_.flow.turbulence, 0.f, 0.1f, "%.3f");
    }

    if (ImGui::CollapsingHeader("Grid", ImGuiTreeNodeFlags_DefaultOpen)) {
        auto label = [&](int i) {
            const auto& p = kGrids[i];
            const double cells = double(p.nx) * p.ny * p.nz;
            static char buf[96];
            std::snprintf(buf, sizeof buf, "%s  (%.0fM, %.1f GiB)", p.name, cells / 1e6,
                          cells * bytesPerCell(precision_) / double(1ull << 30));
            return buf;
        };
        if (ImGui::BeginCombo("Cells", label(gridIndex_))) {
            for (int i = 0; i < int(std::size(kGrids)); ++i) {
                const double need = double(kGrids[i].nx) * kGrids[i].ny * kGrids[i].nz * bytesPerCell(precision_);
                const bool fits = need < 0.92 * double(ctx_.vramBytes);
                if (ImGui::Selectable(label(i), i == gridIndex_, fits ? 0 : ImGuiSelectableFlags_Disabled)) {
                    gridIndex_ = i;
                    rebuildRequested_ = true;
                }
            }
            ImGui::EndCombo();
        }
        const char* precs[] = {"FP32", "FP16S", "FP16C"};
        int p = int(precision_);
        if (ImGui::Combo("Storage", &p, precs, 3)) { precision_ = Precision(p); rebuildRequested_ = true; }
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Memory format of the distribution functions; arithmetic is always FP32.\n"
                              "FP16C halves memory and roughly doubles speed. Against FP32 it moves bluff-body\n"
                              "drag and shedding frequency by under 1%%, but thin laminar boundary layers at\n"
                              "low viscosity come out 2-6%% thinner: use FP32 for skin friction.");
    }

    if (ImGui::CollapsingHeader("View", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::RadioButton("2D slice", &view_.mode3d, 0); ImGui::SameLine();
        ImGui::RadioButton("3D (V)", &view_.mode3d, 1);
        const char* fields[] = {"Velocity", "Pressure", "Vorticity", "Q-criterion"};
        int fi = int(view_.field);
        if (ImGui::Combo("Field (1-4)", &fi, fields, 4)) view_.field = Field(fi);
        const int oldAxis = view_.axis;
        ImGui::Text("Slice normal"); ImGui::SameLine();
        ImGui::RadioButton("X", &view_.axis, 0); ImGui::SameLine();
        ImGui::RadioButton("Y", &view_.axis, 1); ImGui::SameLine();
        ImGui::RadioButton("Z", &view_.axis, 2);
        const auto& g = solver_.grid();
        const float axisLen = float(view_.axis == 0 ? g.nx : view_.axis == 1 ? g.ny : g.nz);
        if (view_.axis != oldAxis) {
            view_.slice = view_.axis == 0 ? body_.center.x : view_.axis == 1 ? body_.center.y : body_.center.z;
            fitView();
        }
        ImGui::SliderFloat("Slice ([ ])", &view_.slice, 0.f, axisLen - 1.f, "%.1f");
        ImGui::Checkbox("Auto colour range", &autoRange_);
        if (autoRange_) ImGui::SliderFloat("Range scale", &rangeScale_, 0.05f, 20.f, "%.2fx", ImGuiSliderFlags_Logarithmic);
        else { ImGui::DragFloat("Low", &view_.lo, 1e-4f, 0, 0, "%.5f"); ImGui::DragFloat("High", &view_.hi, 1e-4f, 0, 0, "%.5f"); }
        ImGui::Checkbox("Show lattice when zoomed", &view_.grid);
        if (view_.field == Field::Pressure) ImGui::Checkbox("Absolute pressure", &absolutePressure_);
        if (ImGui::Button("Fit view (H)", {ImGui::GetContentRegionAvail().x * 0.5f - 4, 0})) fitView();
        ImGui::SameLine();
        if (ImGui::Button("Snapshot (S)", {-1, 0})) snapshot();
    }
    ImGui::PopItemWidth();

    ImGui::Separator();
    ImGui::TextDisabled("Wheel: zoom   Drag: pan (2D) / orbit (3D)\nRight drag: pan (3D)   1-4: field   V: 2D/3D");
    ImGui::End();
}

void App::drawStats(float x, float width, float height) {
    ImGui::SetNextWindowPos({x, 0});
    ImGui::SetNextWindowSize({width, height});
    ImGui::Begin("Stats", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove);
    const auto& g = solver_.grid();
    const float u = solver_.flow.uIn;

    ImGui::TextDisabled("PERFORMANCE");
    ImGui::Text("%.0f MLUPS", mlups_);
    ImGui::Text("%u steps / frame, %.1f ms GPU", running_ ? stepsPerFrame_ : 0u, simMs_);
    ImGui::Text("%.1fM cells, %s", double(solver_.cells()) / 1e6, precisionName(precision_));
    ImGui::Text("%.0f FPS", ImGui::GetIO().Framerate);
    ImGui::Text("t = %s (%llu steps)", formatSi(double(solver_.t) * dt_, "s").c_str(),
                (unsigned long long)solver_.t);

    ImGui::Spacing();
    ImGui::TextDisabled("AERODYNAMICS");
    const double area = referenceArea();
    const double frontal = std::max<uint32_t>(geometry_.frontalCells(), 1);
    const double q = 0.5 * u * u;
    const double cd = force_[0] / (q * area), cl = force_[1] / (q * area), cs = force_[2] / (q * area);
    const double qPhys = 0.5 * props_.rho * speed_ * speed_, aPhys = area * dx_ * dx_;
    ImGui::Text("C_D  %7.3f   drag %s", cd, formatSi(cd * qPhys * aPhys, "N").c_str());
    ImGui::Text("C_L  %7.3f   lift %s", cl, formatSi(cl * qPhys * aPhys, "N").c_str());
    ImGui::Text("C_S  %7.3f", cs);
    const double blockage = frontal / (double(g.ny) * g.nz);
    ImGui::Text("Reference: %s, %.0f cells", body_.shape == Shape::Wing ? "planform" : "frontal area", area);
    ImGui::Text("Blockage %.1f%%", 100 * blockage);
    if (!histCd_.empty()) {
        std::vector<float> v(histCd_.begin(), histCd_.end());
        ImGui::PlotLines("##cd", v.data(), int(v.size()), 0, "C_D history", FLT_MAX, FLT_MAX, {-1, 60});
        std::vector<float> l(histCl_.begin(), histCl_.end());
        ImGui::PlotLines("##cl", l.data(), int(l.size()), 0, "C_L history", FLT_MAX, FLT_MAX, {-1, 60});
    }

    ImGui::Spacing();
    ImGui::TextDisabled("FLOW");
    ImGui::Text("Re requested  %s", formatSi(reRequested_, "").c_str());
    ImGui::Text("Re simulated  %s", formatSi(reSimulated_, "").c_str());
    ImGui::Text("Mach  %.3f    tau  %.6f", mach_, solver_.flow.tau);
    // Dynamic pressure, and the temperature the flow reaches where it is
    // brought to rest (stagnation): T0 = T + V^2 / (2 cp).
    ImGui::Text("q %s   T0 %.1f C", formatSi(0.5 * props_.rho * speed_ * speed_, "Pa").c_str(),
                ambient_.T + double(speed_) * speed_ / (2.0 * props_.cp) - 273.15);
    ImGui::Text("dx %s   dt %s", formatSi(dx_, "m").c_str(), formatSi(dt_, "s").c_str());
    ImGui::Text("Peak |u| = %.2f x inlet", stats_.maxU / std::max(u, 1e-6f));
    ImGui::Text("Mean density %.4f", stats_.rhoMean);
    if (!histResidual_.empty()) {
        std::vector<float> v(histResidual_.begin(), histResidual_.end());
        ImGui::PlotLines("##res", v.data(), int(v.size()), 0, "log10 residual", FLT_MAX, FLT_MAX, {-1, 60});
    }

    ImGui::Spacing();
    ImGui::TextDisabled("ACCURACY");
    const ImVec4 warn{1.f, 0.72f, 0.3f, 1.f};
    if (body_.length < 32) ImGui::TextColored(warn, "%.0f cells across the body: forces\nneed >= 32 to be quantitative", body_.length);
    else ImGui::Text("%.0f cells across the body", body_.length);
    if (tauClamped_) ImGui::TextColored(warn, "Requested Re is beyond this grid:\nsimulating Re %s", formatSi(reSimulated_, "").c_str());
    if (reSimulated_ > 2e4) ImGui::TextColored(warn, "High Re: boundary layers are\nunresolved (no wall model); trust\ntrends and wakes, not absolute C_D");
    if (mach_ > 0.3f) ImGui::TextColored(warn, "Mach %.2f: compressibility is not\nmodelled; results are incompressible", mach_);
    if (!props_.warning.empty()) ImGui::TextColored(warn, "Fluid: %s", props_.warning.c_str());
    if (flowModel_ == 1 && reSimulated_ > reResolved_)
        ImGui::TextColored(warn, "Laminar forced above Re %.0f: the grid\ncannot resolve the smallest eddies and\n"
                                 "the run may diverge (use Auto or LES)", reResolved_);
    if (blockage > 0.1) ImGui::TextColored(warn, "Blockage %.0f%%: tunnel walls\ninflate the forces", 100 * blockage);
    if (stats_.maxU > 0.45f) ImGui::TextColored({1, 0.35f, 0.3f, 1}, "Peak lattice speed %.2f: unstable,\nlower the lattice speed", stats_.maxU);

    if (glfwGetTime() < statusUntil_) {
        ImGui::Spacing();
        ImGui::TextWrapped("%s", status_.c_str());
    }
    ImGui::End();
}

void App::drawViewport(float x, float width, float height) {
    ImGui::SetNextWindowPos({x, 0});
    ImGui::SetNextWindowSize({width, height});
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {0, 0});
    ImGui::Begin("Viewport", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove
                                      | ImGuiWindowFlags_NoScrollWithMouse | ImGuiWindowFlags_NoBringToFrontOnFocus);
    const ImVec2 avail = ImGui::GetContentRegionAvail();
    const ImVec2 fbScale = ImGui::GetIO().DisplayFramebufferScale;
    // Render at the panel's real pixel size: the viewport has no fixed
    // resolution, it is always exactly as sharp as the screen.
    viewportW_ = std::max(1.f, avail.x * fbScale.x);
    viewportH_ = std::max(1.f, avail.y * fbScale.y);
    if (fitPending_) {
        fitView();
        if (startZoom_ != 1.f) {
            // A start-up zoom looks at the body, not the middle of the tunnel.
            const glm::vec3 c = body_.center;
            view_.center = view_.axis == 2 ? glm::vec2(c.x, c.y) : view_.axis == 1 ? glm::vec2(c.x, c.z)
                                                                                  : glm::vec2(c.z, c.y);
            view_.target = c;
        }
        view_.cellsPerPixel /= startZoom_;
        view_.distance /= startZoom_;
        startZoom_ = 1.f;
        fitPending_ = false;
    }
    const ImVec2 origin = ImGui::GetCursorScreenPos();
    const uint32_t slot = uint32_t(frameCount_ % Solver::kSlots);
    if (renderer_.texture(slot))
        ImGui::Image(ImTextureRef(ImTextureID(renderer_.texture(slot))), avail);
    else
        ImGui::Dummy(avail);
    handleViewportInput(avail.x, avail.y);

    // Colour range in lattice units, from the inlet speed and body size.
    const float u = solver_.flow.uIn, L = std::max(body_.length, 1.f);
    if (autoRange_) {
        switch (view_.field) {
        case Field::Speed:      view_.lo = 0; view_.hi = 1.5f * u * std::min(rangeScale_, 4.f); break;
        case Field::Pressure:   view_.hi = 0.5f * u * u * rangeScale_; view_.lo = -view_.hi; break;
        case Field::Vorticity:  view_.hi = 12.f * u / L * rangeScale_; view_.lo = -view_.hi; break;
        case Field::QCriterion: view_.hi = 40.f * (u / L) * (u / L) * rangeScale_; view_.lo = -view_.hi; break;
        }
    }

    // Colour bar with physical units.
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const ImVec2 bar0{origin.x + 20, origin.y + avail.y - 46}, bar1{bar0.x + 260, bar0.y + 14};
    dl->AddRectFilled({bar0.x - 10, bar0.y - 26}, {bar1.x + 10, bar1.y + 22}, IM_COL32(10, 12, 16, 190), 4);
    const int n = 64;
    for (int i = 0; i < n; ++i) {
        const float t0 = float(i) / n, t1 = float(i + 1) / n;
        const ImU32 c = view_.field == Field::Speed ? turboColour(t0) : divergingColour(t0);
        dl->AddRectFilled({bar0.x + (bar1.x - bar0.x) * t0, bar0.y}, {bar0.x + (bar1.x - bar0.x) * t1, bar1.y}, c);
    }
    auto phys = [&](float v) -> std::string {
        // Lattice pressure p - p0 = (rho - 1)/3 scales with the dynamic
        // pressure: p - p_inf = (v / (u^2/2)) * rho V^2 / 2.
        const double gauge = v / (0.5 * u * u) * 0.5 * props_.rho * speed_ * speed_;
        switch (view_.field) {
        case Field::Speed:     return formatSi(v / u * speed_, "m/s");
        case Field::Pressure:  return absolutePressure_ ? formatSi((ambient_.p + gauge) / 1000.0, "kPa")
                                                       : formatSi(gauge, "Pa");
        case Field::Vorticity: return formatSi(dt_ > 0 ? v / dt_ : 0, "1/s");
        default:               return formatSi(dt_ > 0 ? v / (dt_ * dt_) : 0, "1/s^2");
        }
    };
    const char* names[] = {"Speed", absolutePressure_ ? "Pressure (absolute)" : "Pressure p - p_inf",
                           "Vorticity (slice normal)", "Q-criterion"};
    dl->AddText({bar0.x, bar0.y - 22}, IM_COL32(220, 222, 228, 255), names[int(view_.field)]);
    dl->AddText({bar0.x, bar1.y + 3}, IM_COL32(200, 202, 208, 255), phys(view_.lo).c_str());
    const std::string hi = phys(view_.hi);
    dl->AddText({bar1.x - ImGui::CalcTextSize(hi.c_str()).x, bar1.y + 3}, IM_COL32(200, 202, 208, 255), hi.c_str());

    // Scale bar in the slice view: a round number of cells and metres.
    if (view_.mode3d == 0 && dx_ > 0) {
        const float cellsPerPoint = view_.cellsPerPixel * fbScale.x;
        const float target = 120.f * cellsPerPoint;                // ~120 points long
        const float p10 = std::pow(10.f, std::floor(std::log10(target)));
        const float cells = target / p10 >= 5 ? 5 * p10 : target / p10 >= 2 ? 2 * p10 : p10;
        const float len = cells / cellsPerPoint;
        const ImVec2 s0{origin.x + avail.x - len - 24, origin.y + avail.y - 30};
        dl->AddLine(s0, {s0.x + len, s0.y}, IM_COL32(230, 230, 235, 255), 2);
        char buf[64];
        std::snprintf(buf, sizeof buf, "%g cells = %s", cells, formatSi(cells * dx_, "m").c_str());
        dl->AddText({s0.x + len - ImGui::CalcTextSize(buf).x, s0.y - 20}, IM_COL32(230, 230, 235, 255), buf);
        char zoom[48];
        std::snprintf(zoom, sizeof zoom, "%.2f px / cell", 1.f / view_.cellsPerPixel);
        dl->AddText({origin.x + avail.x - ImGui::CalcTextSize(zoom).x - 24, origin.y + 12},
                    IM_COL32(180, 182, 190, 255), zoom);
    }
    ImGui::End();
    ImGui::PopStyleVar();
}

void App::handleViewportInput(float w, float h) {
    if (!ImGui::IsItemHovered()) return;
    const ImGuiIO& io = ImGui::GetIO();
    const ImVec2 fb = io.DisplayFramebufferScale;
    const ImVec2 itemMin = ImGui::GetItemRectMin();
    // Cursor relative to the viewport centre, in rendered pixels.
    const float cx = (io.MousePos.x - itemMin.x - 0.5f * w) * fb.x;
    const float cy = (io.MousePos.y - itemMin.y - 0.5f * h) * fb.y;

    if (view_.mode3d == 0) {
        if (io.MouseWheel != 0) {
            // Zoom about the cursor: the plane point under it stays put.
            const glm::vec2 under = view_.center + glm::vec2(cx, -cy) * view_.cellsPerPixel;
            view_.cellsPerPixel = std::clamp(view_.cellsPerPixel * std::pow(0.85f, io.MouseWheel), 1e-3f, 64.f);
            view_.center = under - glm::vec2(cx, -cy) * view_.cellsPerPixel;
        }
        if (ImGui::IsMouseDragging(ImGuiMouseButton_Left, 0) || ImGui::IsMouseDragging(ImGuiMouseButton_Right, 0))
            view_.center -= glm::vec2(io.MouseDelta.x * fb.x, -io.MouseDelta.y * fb.y) * view_.cellsPerPixel;
    } else {
        if (io.MouseWheel != 0) view_.distance = std::clamp(view_.distance * std::pow(0.88f, io.MouseWheel), 2.f, 1e5f);
        if (ImGui::IsMouseDragging(ImGuiMouseButton_Left, 0)) {
            view_.yaw += io.MouseDelta.x * 0.3f;
            view_.pitch = std::clamp(view_.pitch + io.MouseDelta.y * 0.3f, -89.f, 89.f);
        }
        if (ImGui::IsMouseDragging(ImGuiMouseButton_Right, 0)) {
            const float yaw = glm::radians(view_.yaw), pitch = glm::radians(view_.pitch);
            const glm::vec3 fwd = -glm::vec3(std::cos(pitch) * std::cos(yaw), std::sin(pitch), std::cos(pitch) * std::sin(yaw));
            const glm::vec3 r = glm::normalize(glm::cross(fwd, glm::vec3(0, 1, 0)));
            const glm::vec3 up = glm::cross(r, fwd);
            const float k = view_.distance * 0.0015f;
            view_.target += (-io.MouseDelta.x * r + io.MouseDelta.y * up) * k;
        }
    }
}

void App::handleKeys() {
    const ImGuiIO& io = ImGui::GetIO();
    if (io.WantTextInput) return;
    if (ImGui::IsKeyPressed(ImGuiKey_Space, false)) running_ = !running_;
    if (ImGui::IsKeyPressed(ImGuiKey_R, false)) resetRequested_ = true;
    if (ImGui::IsKeyPressed(ImGuiKey_V, false)) view_.mode3d ^= 1;
    if (ImGui::IsKeyPressed(ImGuiKey_H, false) || ImGui::IsKeyPressed(ImGuiKey_Home, false)) fitView();
    if (ImGui::IsKeyPressed(ImGuiKey_S, false)) snapshot();
    const ImGuiKey fieldKeys[] = {ImGuiKey_1, ImGuiKey_2, ImGuiKey_3, ImGuiKey_4};
    for (int i = 0; i < 4; ++i) if (ImGui::IsKeyPressed(fieldKeys[i], false)) view_.field = Field(i);
    const float stepSize = io.KeyShift ? 10.f : 1.f;
    const auto& g = solver_.grid();
    const float axisLen = float(view_.axis == 0 ? g.nx : view_.axis == 1 ? g.ny : g.nz);
    if (ImGui::IsKeyPressed(ImGuiKey_LeftBracket))  view_.slice = std::max(0.f, view_.slice - stepSize);
    if (ImGui::IsKeyPressed(ImGuiKey_RightBracket)) view_.slice = std::min(axisLen - 1, view_.slice + stepSize);
}

void App::loadMeshFile(const std::string& path) {
    std::vector<Triangle> tris;
    std::string err;
    if (!loadMesh(path, tris, err)) {
        status_ = "Could not load " + path + ": " + err;
        statusUntil_ = glfwGetTime() + 6;
        return;
    }
    meshName_ = std::filesystem::path(path).filename().string() + "  (" + std::to_string(tris.size()) + " triangles)";
    geometry_.setMesh(std::move(tris));
    body_.shape = Shape::Mesh;
    body_.pitch = body_.yaw = body_.roll = 0;
    geometryDirty_ = true;
    status_ = "Loaded " + meshName_;
    statusUntil_ = glfwGetTime() + 4;
}

void App::saveCapture(uint32_t width, uint32_t height) {
    vmaInvalidateAllocation(ctx_.vma, captureBuf_.alloc, 0, VK_WHOLE_SIZE);
    std::vector<uint8_t> px(size_t(width) * height * 4);
    std::memcpy(px.data(), captureBuf_.map, px.size());
    for (size_t i = 0; i < px.size(); i += 4) std::swap(px[i], px[i + 2]);   // BGRA -> RGBA
    const bool ok = stbi_write_png(capturePath_.c_str(), int(width), int(height), 4, px.data(), int(width) * 4) != 0;
    std::printf("%s %s (%u x %u)\n", ok ? "captured" : "capture failed:", capturePath_.c_str(), width, height);
}

void App::snapshot() {
    std::filesystem::create_directories("snapshots");
    const std::time_t now = std::time(nullptr);
    char name[64];
    std::strftime(name, sizeof name, "snapshots/tunnel_%Y%m%d_%H%M%S.png", std::localtime(&now));
    const uint32_t last = uint32_t((frameCount_ + Solver::kSlots - 1) % Solver::kSlots);
    const bool ok = renderer_.savePng(last, name);
    status_ = ok ? std::string("Saved ") + name : "Snapshot failed";
    statusUntil_ = glfwGetTime() + 4;
}

} // namespace wt
