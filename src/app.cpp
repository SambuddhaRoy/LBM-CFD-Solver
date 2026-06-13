// ============================================================================
// app.cpp — application orchestration: frame loop, actions, headless mode
// ============================================================================

#include "app.h"
#include "ui.h"

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <commdlg.h>
#endif

namespace vwt {

namespace {

// ─── Native open-file dialog ─────────────────────────────────────────────────
std::string nativeFileDialog() {
#ifdef _WIN32
    OPENFILENAMEA ofn{};
    char file[512] = {};
    ofn.lStructSize = sizeof(ofn);
    ofn.lpstrFile   = file;
    ofn.nMaxFile    = sizeof(file);
    ofn.lpstrFilter = "3D models (*.stl;*.obj;*.glb;*.gltf;*.fbx;*.ply)\0"
                      "*.stl;*.obj;*.glb;*.gltf;*.fbx;*.ply\0All files\0*.*\0";
    ofn.nFilterIndex = 1;
    ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST | OFN_NOCHANGEDIR;
    if (GetOpenFileNameA(&ofn)) return file;
    return "";
#else
    // zenity → kdialog fallback chain
    for (const char* cmd : {
        "zenity --file-selection --title='Open model' "
        "--file-filter='3D models | *.stl *.obj *.glb *.gltf *.fbx *.ply' 2>/dev/null",
        "kdialog --getopenfilename . 2>/dev/null" }) {
        if (FILE* p = popen(cmd, "r")) {
            char buf[512] = {};
            const bool got = fgets(buf, sizeof(buf), p) != nullptr;
            const int rc = pclose(p);
            if (got && rc == 0) {
                std::string s(buf);
                while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
                if (!s.empty()) return s;
            }
        }
    }
    return "";
#endif
}

// ─── Minimal BMP writer (24-bit, bottom-up) ──────────────────────────────────
bool writeBmp(const std::filesystem::path& path,
              const std::vector<uint8_t>& rgba, uint32_t w, uint32_t h) {
    const uint32_t rowBytes = (w * 3 + 3) & ~3u;
    const uint32_t dataSize = rowBytes * h;
    const uint32_t fileSize = 54 + dataSize;

    std::ofstream f(path, std::ios::binary);
    if (!f.is_open()) return false;

    uint8_t hdr[54] = {};
    hdr[0] = 'B'; hdr[1] = 'M';
    std::memcpy(hdr + 2,  &fileSize, 4);
    const uint32_t offset = 54;        std::memcpy(hdr + 10, &offset, 4);
    const uint32_t ihSize = 40;        std::memcpy(hdr + 14, &ihSize, 4);
    std::memcpy(hdr + 18, &w, 4);
    std::memcpy(hdr + 22, &h, 4);
    const uint16_t planes = 1;         std::memcpy(hdr + 26, &planes, 2);
    const uint16_t bpp    = 24;        std::memcpy(hdr + 28, &bpp, 2);
    std::memcpy(hdr + 34, &dataSize, 4);
    f.write(reinterpret_cast<char*>(hdr), 54);

    std::vector<uint8_t> row(rowBytes, 0);
    for (uint32_t y = 0; y < h; ++y) {
        const uint8_t* src = rgba.data() + size_t(y) * w * 4;
        for (uint32_t x = 0; x < w; ++x) {
            row[x*3+0] = src[x*4+2];   // B
            row[x*3+1] = src[x*4+1];   // G
            row[x*3+2] = src[x*4+0];   // R
        }
        f.write(reinterpret_cast<char*>(row.data()), rowBytes);
    }
    return true;
}

} // namespace

// ════════════════════════════════════════════════════════════════════════════
// Entry
// ════════════════════════════════════════════════════════════════════════════

int App::run(const StartOptions& opts) {
    if (opts.bench)    return runBench(opts);
    if (opts.headless) return runHeadless(opts);
    return runGui(opts);
}

// Pure throughput benchmark: time a single submission of N collide-stream steps
// (no analysis, no per-step CPU synchronisation), which isolates the kernel's
// sustained throughput from command-submission and readback overhead.
int App::runBench(const StartOptions& opts) {
    params.gx = opts.gx ? opts.gx : 128;
    params.gy = opts.gy ? opts.gy : 64;
    params.gz = opts.gz ? opts.gz : 64;
    if (opts.lesOff)         params.les = false;
    if (opts.collision >= 0) params.collision = opts.collision;
    const uint32_t steps = std::max(200u, opts.steps);

    gpu.init(nullptr);
    solver.init(gpu, params);
    model = mesh::makePrimitive(Shape::Sphere, params.gx, params.gy, params.gz, 0.f, 0.f);
    solver.uploadObstacles(model.occupancy);
    solver.reset();

    // Warm up (driver/pipeline) then time one submission of `steps` steps.
    gpu.oneShot([&](VkCommandBuffer cmd) {
        solver.recordSteps(cmd, params, 200, 0, false);
    });
    const auto t0 = std::chrono::high_resolution_clock::now();
    gpu.oneShot([&](VkCommandBuffer cmd) {
        solver.recordSteps(cmd, params, steps, 0, false);
    });
    const auto t1 = std::chrono::high_resolution_clock::now();

    const double secs    = std::chrono::duration<double>(t1 - t0).count();
    const double mlupsVal = double(solver.cells()) * steps / secs / 1e6;
    const char* opName = params.collision == 0 ? "BGK"
                       : params.collision == 2 ? "TRT" : "Regularised";
    std::printf("%ux%ux%u  %s%s  %u steps in %.4f s  ->  %.1f MLUPS\n",
                params.gx, params.gy, params.gz, opName,
                params.les ? "+LES" : "", steps, secs, mlupsVal);

    solver.destroy();
    gpu.destroy();
    return 0;
}

uint32_t App::sliceMaxIndex() const {
    switch (axis) {
    case 0:  return params.gz - 1;   // XY plane stacks along Z
    case 1:  return params.gy - 1;   // XZ plane stacks along Y
    default: return params.gx - 1;   // ZY plane stacks along X
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Actions
// ════════════════════════════════════════════════════════════════════════════

void App::setStatus(const std::string& msg) {
    statusMsg = msg;
    statusTtl = 3.f;
    logMsg(msg);
}

void App::loadPrimitive(Shape s) {
    primShape = int(s);
    meshTris.clear();
    model = mesh::makePrimitive(s, params.gx, params.gy, params.gz, aoaDeg, yawDeg);
    uploadModel();
    setStatus(std::string("Loaded ") + shapeName(s));
}

void App::loadMeshFile(const std::string& path) {
    std::vector<Tri> tris;
    std::string err;
    if (!mesh::loadTriangles(path, tris, err)) {
        setStatus("Import failed: " + err);
        return;
    }
    meshTris  = std::move(tris);
    primShape = -1;
    model = mesh::voxelizeTriangles(meshTris, params.gx, params.gy, params.gz,
                                    aoaDeg, yawDeg,
                                    std::filesystem::path(path).filename().string());
    uploadModel();
    setStatus("Loaded " + model.name + " (" +
              std::to_string(model.triCount) + " triangles)");
}

void App::revoxelize() {
    if (primShape >= 0) {
        model = mesh::makePrimitive(Shape(primShape), params.gx, params.gy,
                                    params.gz, aoaDeg, yawDeg);
    } else if (!meshTris.empty()) {
        model = mesh::voxelizeTriangles(meshTris, params.gx, params.gy, params.gz,
                                        aoaDeg, yawDeg, model.name);
    } else {
        return;
    }
    uploadModel();
}

void App::uploadModel() {
    if (window) vkDeviceWaitIdle(gpu.device());
    solver.uploadObstacles(model.occupancy);
    modelLoaded = true;
    units.compute(model.spanCellsX, params);
    resetSim();
}

void App::resetSim() {
    if (!modelLoaded) return;
    if (window) vkDeviceWaitIdle(gpu.device());
    solver.reset();
    totalSteps = 0;
    an = {};
    cd = cl = cdPrev = clPrev = 0.f;
    cdHist.fill(0.f); clHist.fill(0.f);
    resHist.fill(0.f); fpsHist.fill(0.f);
    for (auto& fr : frames_) fr.ranAnalysis = false;
}

void App::applyGridPreset(int preset) {
    preset = std::clamp(preset, 0, kGridCount - 1);
    gridPreset = preset;
    params.gx = kGrids[preset].gx;
    params.gy = kGrids[preset].gy;
    params.gz = kGrids[preset].gz;

    vkDeviceWaitIdle(gpu.device());
    slice.destroy();
    solver.destroy();
    solver.init(gpu, params);
    sliceIndex = -1;
    rebuildSliceView();
    if (primShape >= 0 || !meshTris.empty()) revoxelize();
    setStatus(std::string("Grid: ") + kGrids[preset].name);
}

void App::rebuildSliceView() {
    slice.rebuild(params.gx, params.gy, params.gz, uint32_t(axis),
                  solver.macroBuffer(), solver.obstacleBuffer());
}

void App::snapshot() {
    std::vector<uint8_t> px;
    uint32_t w = 0, h = 0;
    if (!slice.readPixels(px, w, h)) {
        setStatus("Nothing to snapshot yet");
        return;
    }
    // Flip vertically: the viewport displays the image V-flipped (+Y up)
    std::vector<uint8_t> flipped(px.size());
    for (uint32_t y = 0; y < h; ++y)
        std::memcpy(flipped.data() + size_t(y) * w * 4,
                    px.data() + size_t(h - 1 - y) * w * 4, size_t(w) * 4);

    const auto dir = exeDir() / "snapshots";
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);

    const auto now = std::chrono::system_clock::now();
    const auto t   = std::chrono::system_clock::to_time_t(now);
    std::tm tmBuf{};
#ifdef _WIN32
    localtime_s(&tmBuf, &t);
#else
    localtime_r(&t, &tmBuf);
#endif
    char name[64];
    std::snprintf(name, sizeof(name), "vwt_%04d%02d%02d_%02d%02d%02d.bmp",
                  tmBuf.tm_year + 1900, tmBuf.tm_mon + 1, tmBuf.tm_mday,
                  tmBuf.tm_hour, tmBuf.tm_min, tmBuf.tm_sec);

    if (writeBmp(dir / name, flipped, w, h))
        setStatus(std::string("Snapshot saved: snapshots/") + name);
    else
        setStatus("Snapshot failed (could not write file)");
}

void App::openFileDialog() {
    const std::string path = nativeFileDialog();
    if (!path.empty()) loadMeshFile(path);
}

void App::toggleFullscreen() {
    if (!window) return;
    if (!fullscreen_) {
        glfwGetWindowPos(window, &savedX_, &savedY_);
        glfwGetWindowSize(window, &savedW_, &savedH_);
        GLFWmonitor* mon = glfwGetPrimaryMonitor();
        const GLFWvidmode* vm = glfwGetVideoMode(mon);
        glfwSetWindowMonitor(window, mon, 0, 0, vm->width, vm->height, vm->refreshRate);
    } else {
        glfwSetWindowMonitor(window, nullptr, savedX_, savedY_, savedW_, savedH_, 0);
    }
    fullscreen_ = !fullscreen_;
}

// ════════════════════════════════════════════════════════════════════════════
// Config persistence
// ════════════════════════════════════════════════════════════════════════════

void App::loadConfig() {
    std::ifstream f(exeDir() / "vwt2.ini");
    if (!f.is_open()) return;
    std::map<std::string, std::string> kv;
    std::string line;
    while (std::getline(f, line)) {
        const auto eq = line.find('=');
        if (eq != std::string::npos)
            kv[line.substr(0, eq)] = line.substr(eq + 1);
    }
    auto getF = [&](const char* k, float& v) {
        if (auto it = kv.find(k); it != kv.end()) v = std::stof(it->second);
    };
    auto getI = [&](const char* k, int& v) {
        if (auto it = kv.find(k); it != kv.end()) v = std::stoi(it->second);
    };
    int les = params.les ? 1 : 0;
    getI("gridPreset", gridPreset);
    getF("tau", params.tau);
    getF("uIn", params.uIn);
    getF("turb", params.turb);
    getI("collision", params.collision);
    getI("les", les);
    getF("csSmago", params.csSmago);
    getI("stepsPerFrame", stepsPerFrame);
    getI("visMode", visMode);
    getI("axis", axis);
    getF("scaleVel", scaleVel);
    getF("maxVort", maxVort);
    getI("fluid", units.fluidIdx);
    getF("physSpeed", units.physSpeed);
    getF("physLength", units.physLength);
    int w = int(winW_), h = int(winH_);
    getI("winW", w); getI("winH", h);
    winW_ = uint32_t(std::max(640, w));
    winH_ = uint32_t(std::max(480, h));

    params.les = (les != 0);
    gridPreset = std::clamp(gridPreset, 0, kGridCount - 1);
    units.fluidIdx = std::clamp(units.fluidIdx, 0, kFluidCount - 1);
    visMode = std::clamp(visMode, 0, 3);
    axis    = std::clamp(axis, 0, 2);
    stepsPerFrame = std::clamp(stepsPerFrame, 1, 40);
    params.gx = kGrids[gridPreset].gx;
    params.gy = kGrids[gridPreset].gy;
    params.gz = kGrids[gridPreset].gz;
}

void App::saveConfig() {
    std::ofstream f(exeDir() / "vwt2.ini");
    if (!f.is_open()) return;
    f << "gridPreset=" << gridPreset << "\n"
      << "tau=" << params.tau << "\n"
      << "uIn=" << params.uIn << "\n"
      << "turb=" << params.turb << "\n"
      << "collision=" << params.collision << "\n"
      << "les=" << (params.les ? 1 : 0) << "\n"
      << "csSmago=" << params.csSmago << "\n"
      << "stepsPerFrame=" << stepsPerFrame << "\n"
      << "visMode=" << visMode << "\n"
      << "axis=" << axis << "\n"
      << "scaleVel=" << scaleVel << "\n"
      << "maxVort=" << maxVort << "\n"
      << "fluid=" << units.fluidIdx << "\n"
      << "physSpeed=" << units.physSpeed << "\n"
      << "physLength=" << units.physLength << "\n"
      << "winW=" << winW_ << "\n"
      << "winH=" << winH_ << "\n";
}

// ════════════════════════════════════════════════════════════════════════════
// GUI mode
// ════════════════════════════════════════════════════════════════════════════

void App::dropCallback(GLFWwindow* win, int count, const char** paths) {
    if (count <= 0) return;
    auto* app = static_cast<App*>(glfwGetWindowUserPointer(win));
    if (app) app->pendingDrop_ = paths[0];
}

void App::initWindow() {
    if (!glfwInit()) throw std::runtime_error("glfwInit failed");
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
    window = glfwCreateWindow(int(winW_), int(winH_),
                              "Virtual Wind Tunnel", nullptr, nullptr);
    if (!window) throw std::runtime_error("Window creation failed");
    glfwSetWindowUserPointer(window, this);
    glfwSetDropCallback(window, dropCallback);
}

void App::initImGui() {
    VkDescriptorPoolSize sizes[] = {
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 64 },
        { VK_DESCRIPTOR_TYPE_SAMPLER,                16 },
    };
    VkDescriptorPoolCreateInfo pi{};
    pi.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pi.flags         = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    pi.maxSets       = 80;
    pi.poolSizeCount = 2;
    pi.pPoolSizes    = sizes;
    VK_CHECK(vkCreateDescriptorPool(gpu.device(), &pi, nullptr, &imguiPool_));

    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.IniFilename = nullptr;   // layout is fixed; no imgui.ini churn

    float sx = 1.f, sy = 1.f;
    glfwGetWindowContentScale(window, &sx, &sy);
    const float dpi = std::max(1.f, std::max(sx, sy));

    ui::loadFonts(*this, dpi);
    ui::applyTheme(dpi);

    ImGui_ImplGlfw_InitForVulkan(window, true);
    ImGui_ImplVulkan_InitInfo ii{};
    ii.ApiVersion     = VK_API_VERSION_1_3;
    ii.Instance       = gpu.instance();
    ii.PhysicalDevice = gpu.physDevice();
    ii.Device         = gpu.device();
    ii.QueueFamily    = gpu.queueFamily();
    ii.Queue          = gpu.queue();
    ii.DescriptorPool = imguiPool_;
    ii.MinImageCount  = 2;
    ii.ImageCount     = swap_.imageCount();
    ii.PipelineInfoMain.RenderPass = swap_.renderPass();
    ImGui_ImplVulkan_Init(&ii);
}

int App::runGui(const StartOptions& opts) {
    loadConfig();
    if (opts.gx && opts.gy && opts.gz) {
        params.gx = opts.gx; params.gy = opts.gy; params.gz = opts.gz;
    }
    if (opts.lesOff) params.les = false;

    initWindow();
    gpu.init(window);

    int fbW = 0, fbH = 0;
    glfwGetFramebufferSize(window, &fbW, &fbH);
    swap_.init(gpu, uint32_t(fbW), uint32_t(fbH));

    for (auto& fr : frames_) {
        VkCommandPoolCreateInfo pci{};
        pci.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pci.queueFamilyIndex = gpu.queueFamily();
        pci.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        VK_CHECK(vkCreateCommandPool(gpu.device(), &pci, nullptr, &fr.pool));

        VkCommandBufferAllocateInfo cai{};
        cai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cai.commandPool        = fr.pool;
        cai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cai.commandBufferCount = 1;
        VK_CHECK(vkAllocateCommandBuffers(gpu.device(), &cai, &fr.cmd));

        VkFenceCreateInfo fci{};
        fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        VK_CHECK(vkCreateFence(gpu.device(), &fci, nullptr, &fr.fence));

        VkSemaphoreCreateInfo sci{};
        sci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        VK_CHECK(vkCreateSemaphore(gpu.device(), &sci, nullptr, &fr.acquireSem));
        VK_CHECK(vkCreateSemaphore(gpu.device(), &sci, nullptr, &fr.renderSem));
    }

    initImGui();
    solver.init(gpu, params);
    slice.init(gpu, params.gx, params.gy, params.gz, uint32_t(axis),
               solver.macroBuffer(), solver.obstacleBuffer());

    if (!opts.meshPath.empty()) loadMeshFile(opts.meshPath);

    logMsg("Ready.");

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        glfwGetFramebufferSize(window, &fbW, &fbH);
        if (fbW == 0 || fbH == 0) { glfwWaitEvents(); continue; }
        if (uint32_t(fbW) != swap_.extent().width ||
            uint32_t(fbH) != swap_.extent().height)
            swap_.recreate(uint32_t(fbW), uint32_t(fbH));
        if (!fullscreen_) { winW_ = uint32_t(fbW); winH_ = uint32_t(fbH); }

        if (!pendingDrop_.empty()) {
            loadMeshFile(pendingDrop_);
            pendingDrop_.clear();
        }
        if (uint32_t(axis) != slice.axis()) rebuildSliceView();

        frame();
    }

    vkDeviceWaitIdle(gpu.device());
    saveConfig();

    slice.destroy();
    solver.destroy();
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    for (auto& fr : frames_) {
        vkDestroyCommandPool(gpu.device(), fr.pool, nullptr);
        vkDestroyFence(gpu.device(), fr.fence, nullptr);
        vkDestroySemaphore(gpu.device(), fr.acquireSem, nullptr);
        vkDestroySemaphore(gpu.device(), fr.renderSem, nullptr);
    }
    vkDestroyDescriptorPool(gpu.device(), imguiPool_, nullptr);
    swap_.destroy();
    gpu.destroy();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}

void App::handleShortcuts() {
    ImGuiIO& io = ImGui::GetIO();
    if (io.WantTextInput) return;

    if (ImGui::IsKeyPressed(ImGuiKey_Space, false) && modelLoaded)
        running = !running;
    if (ImGui::IsKeyPressed(ImGuiKey_R, false)) resetSim();
    if (ImGui::IsKeyPressed(ImGuiKey_1, false)) visMode = 0;
    if (ImGui::IsKeyPressed(ImGuiKey_2, false)) visMode = 1;
    if (ImGui::IsKeyPressed(ImGuiKey_3, false)) visMode = 2;
    if (ImGui::IsKeyPressed(ImGuiKey_4, false)) visMode = 3;
    if (ImGui::IsKeyPressed(ImGuiKey_LeftBracket))
        sliceIndex = std::max(0, (sliceIndex < 0 ? int(sliceMaxIndex()) / 2 : sliceIndex) - 1);
    if (ImGui::IsKeyPressed(ImGuiKey_RightBracket))
        sliceIndex = std::min(int(sliceMaxIndex()),
                              (sliceIndex < 0 ? int(sliceMaxIndex()) / 2 : sliceIndex) + 1);
    if (ImGui::IsKeyPressed(ImGuiKey_Tab, false))   leftOpen = !leftOpen;
    if (ImGui::IsKeyPressed(ImGuiKey_F, false))     rightOpen = !rightOpen;
    if (ImGui::IsKeyPressed(ImGuiKey_S, false))     snapshot();
    if (ImGui::IsKeyPressed(ImGuiKey_F11, false))   toggleFullscreen();
    if (ImGui::IsKeyPressed(ImGuiKey_Slash, false)) showHelp = !showHelp;   // '?'
    if (ImGui::IsKeyPressed(ImGuiKey_Escape, false)) {
        if (showHelp) showHelp = false;
        else { zoom = 1.f; panX = panY = 0.f; }
    }
}

void App::frame() {
    static auto lastT = std::chrono::high_resolution_clock::now();
    const auto t0 = std::chrono::high_resolution_clock::now();
    const float dtMs = std::chrono::duration<float, std::milli>(t0 - lastT).count();
    lastT = t0;
    frameMs = frameMs * 0.92f + dtMs * 0.08f;
    fps     = (frameMs > 0.01f) ? 1000.f / frameMs : 0.f;
    if (statusTtl > 0.f) statusTtl -= dtMs / 1000.f;

    auto& fr = frames_[frameIdx_];

    // ── Build UI ────────────────────────────────────────────────────────────
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    handleShortcuts();
    ui::draw(*this);
    ImGui::Render();

    // ── Wait for this slot's previous submission, then read its results ────
    VK_CHECK(vkWaitForFences(gpu.device(), 1, &fr.fence, VK_TRUE, 1'000'000'000));

    if (fr.ranAnalysis) {
        an = solver.readAnalysis(frameIdx_);
        const float lbm = solver.readTimestampMs(0);
        const float ana = solver.readTimestampMs(2);
        const float slc = solver.readTimestampMs(4);
        if (lbm > 0.f) tim.lbmMs      = tim.lbmMs * 0.9f + lbm * 0.1f;
        if (ana > 0.f) tim.analysisMs = tim.analysisMs * 0.9f + ana * 0.1f;
        if (slc > 0.f) tim.sliceMs    = tim.sliceMs * 0.9f + slc * 0.1f;
        if (an.valid) {
            const float q = 0.5f * params.uIn * params.uIn;
            const float A = model.frontalCells > 0 ? float(model.frontalCells) : 1.f;
            cdPrev = cd; clPrev = cl;
            cd = an.drag / (q * A);
            cl = an.lift / (q * A);
            mlups = (frameMs > 0.01f)
                ? float(double(solver.cells()) * fr.batchSteps / (frameMs * 1000.0))
                : 0.f;
        }
        fr.ranAnalysis = false;
    }

    cdHist[size_t(histIdx)]  = cd;
    clHist[size_t(histIdx)]  = cl;
    resHist[size_t(histIdx)] = std::log10(std::max(an.residual, 1e-9f));
    fpsHist[size_t(histIdx)] = fps;
    histIdx = (histIdx + 1) % kHist;

    if (totalSteps % 32 == 0) gpu.queryVram(vramUse, vramBudget);

    // ── Acquire ─────────────────────────────────────────────────────────────
    uint32_t imageIndex = 0;
    const VkResult acq = vkAcquireNextImageKHR(
        gpu.device(), swap_.handle(), 1'000'000'000,
        fr.acquireSem, VK_NULL_HANDLE, &imageIndex);
    if (acq == VK_ERROR_OUT_OF_DATE_KHR) {
        int w, h;
        glfwGetFramebufferSize(window, &w, &h);
        if (w > 0 && h > 0) swap_.recreate(uint32_t(w), uint32_t(h));
        return;
    }

    VK_CHECK(vkResetFences(gpu.device(), 1, &fr.fence));
    VK_CHECK(vkResetCommandBuffer(fr.cmd, 0));

    VkCommandBufferBeginInfo cbi{};
    cbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    cbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(fr.cmd, &cbi));

    // ── Simulation ──────────────────────────────────────────────────────────
    const bool doStep = (running || runSingleBatch) && modelLoaded;
    if (doStep) {
        const uint32_t n = uint32_t(stepsPerFrame);
        solver.recordSteps(fr.cmd, params, n, uint32_t(totalSteps), true);
        totalSteps += n;
        solver.recordAnalysis(fr.cmd, params, frameIdx_);
        fr.ranAnalysis = true;
        fr.batchSteps  = n;
        runSingleBatch = false;
    }

    // ── Visualization ───────────────────────────────────────────────────────
    if (modelLoaded) {
        const uint32_t maxIdx = sliceMaxIndex();
        const uint32_t idx = (sliceIndex < 0)
            ? maxIdx / 2 : std::min(uint32_t(sliceIndex), maxIdx);
        SlicePush sp{};
        sp.gx = params.gx; sp.gy = params.gy; sp.gz = params.gz;
        sp.axis     = uint32_t(axis);
        sp.index    = idx;
        sp.mode     = uint32_t(visMode);
        sp.scaleVel = scaleVel;
        sp.uIn      = params.uIn;
        sp.maxVort  = maxVort;
        slice.record(fr.cmd, sp, solver.queryPool());
    }

    // ── Render pass (clear + ImGui) ─────────────────────────────────────────
    VkClearValue clear{};
    clear.color = { {0.043f, 0.047f, 0.062f, 1.f} };
    VkRenderPassBeginInfo rbi{};
    rbi.sType             = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rbi.renderPass        = swap_.renderPass();
    rbi.framebuffer       = swap_.framebuffer(imageIndex);
    rbi.renderArea.extent = swap_.extent();
    rbi.clearValueCount   = 1;
    rbi.pClearValues      = &clear;
    vkCmdBeginRenderPass(fr.cmd, &rbi, VK_SUBPASS_CONTENTS_INLINE);
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), fr.cmd);
    vkCmdEndRenderPass(fr.cmd);
    VK_CHECK(vkEndCommandBuffer(fr.cmd));

    // ── Submit + present ────────────────────────────────────────────────────
    const VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkSubmitInfo si{};
    si.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.waitSemaphoreCount   = 1;
    si.pWaitSemaphores      = &fr.acquireSem;
    si.pWaitDstStageMask    = &waitStage;
    si.commandBufferCount   = 1;
    si.pCommandBuffers      = &fr.cmd;
    si.signalSemaphoreCount = 1;
    si.pSignalSemaphores    = &fr.renderSem;
    VK_CHECK(vkQueueSubmit(gpu.queue(), 1, &si, fr.fence));

    VkPresentInfoKHR pres{};
    pres.sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    pres.waitSemaphoreCount = 1;
    pres.pWaitSemaphores    = &fr.renderSem;
    pres.swapchainCount     = 1;
    VkSwapchainKHR sc = swap_.handle();
    pres.pSwapchains    = &sc;
    pres.pImageIndices  = &imageIndex;
    const VkResult pr = vkQueuePresentKHR(gpu.queue(), &pres);
    if (pr == VK_ERROR_OUT_OF_DATE_KHR || pr == VK_SUBOPTIMAL_KHR) {
        int w, h;
        glfwGetFramebufferSize(window, &w, &h);
        if (w > 0 && h > 0) swap_.recreate(uint32_t(w), uint32_t(h));
    }

    frameIdx_ = (frameIdx_ + 1) % kFramesInFlight;
}

// ════════════════════════════════════════════════════════════════════════════
// Headless mode — runs the solver without a window and self-validates.
// Exit code 0 = physics sane, 1 = failed checks. Suitable for CI.
// ════════════════════════════════════════════════════════════════════════════

int App::runHeadless(const StartOptions& opts) {
    params.gx = opts.gx ? opts.gx : 96;
    params.gy = opts.gy ? opts.gy : 64;
    params.gz = opts.gz ? opts.gz : 64;
    if (opts.lesOff) params.les = false;
    if (opts.collision >= 0) params.collision = opts.collision;

    gpu.init(nullptr);
    solver.init(gpu, params);

    if (!opts.meshPath.empty()) {
        std::vector<Tri> tris;
        std::string err;
        if (!mesh::loadTriangles(opts.meshPath, tris, err)) {
            logMsg("Import failed: " + err);
            gpu.destroy();
            return 1;
        }
        model = mesh::voxelizeTriangles(tris, params.gx, params.gy, params.gz,
                                        opts.aoaDeg, 0.f,
                                        std::filesystem::path(opts.meshPath)
                                            .filename().string());
    } else {
        const Shape s = Shape(std::clamp(opts.shape, 0, 3));
        model = mesh::makePrimitive(s, params.gx, params.gy, params.gz,
                                    opts.aoaDeg, 0.f);
    }
    solver.uploadObstacles(model.occupancy);
    modelLoaded = true;
    units.compute(model.spanCellsX, params);

    const char* opName = params.collision == 0 ? "BGK"
                       : params.collision == 2 ? "TRT" : "Regularized";
    std::printf("\n  Virtual Wind Tunnel v2 — headless validation\n");
    std::printf("  model: %-18s grid: %ux%ux%u   collision: %s%s   AoA: %.1f deg\n",
                model.name.c_str(), params.gx, params.gy, params.gz,
                opName, params.les ? " + LES" : "", opts.aoaDeg);
    std::printf("  frontal area: %u cells   fill: %.2f%%\n\n",
                model.frontalCells, model.fillPct);
    std::printf("  %8s  %12s  %10s  %10s  %8s  %8s\n",
                "step", "residual", "C_D", "C_L", "mass", "max|u|");

    const uint32_t total   = std::max(40u, opts.steps);
    const uint32_t batch   = 40;
    const uint32_t reports = 10;
    float firstRes = -1.f, lastRes = 1.f;
    uint64_t done = 0;

    const auto t0 = std::chrono::high_resolution_clock::now();
    while (done < total) {
        const uint32_t n = uint32_t(std::min<uint64_t>(batch, total - done));
        gpu.oneShot([&](VkCommandBuffer cmd) {
            solver.recordSteps(cmd, params, n, uint32_t(done), false);
            solver.recordAnalysis(cmd, params, 0);
        });
        done += n;

        an = solver.readAnalysis(0);
        const float q = 0.5f * params.uIn * params.uIn;
        const float A = model.frontalCells > 0 ? float(model.frontalCells) : 1.f;
        cd = an.drag / (q * A);
        cl = an.lift / (q * A);
        if (firstRes < 0.f && an.valid && done > batch) firstRes = an.residual;
        lastRes = an.residual;

        if (done % std::max(1u, (total / reports) / batch * batch) == 0 || done >= total)
            std::printf("  %8llu  %12.4e  %10.4f  %10.4f  %8.4f  %8.4f\n",
                        static_cast<unsigned long long>(done), an.residual,
                        cd, cl, an.massAvg, an.maxU);
    }
    const auto t1 = std::chrono::high_resolution_clock::now();
    const double secs = std::chrono::duration<double>(t1 - t0).count();
    const double mlupsAvg = double(solver.cells()) * double(total) / secs / 1e6;

    std::printf("\n  %llu steps in %.2f s  →  %.1f MLUPS on %s\n",
                static_cast<unsigned long long>(total), secs, mlupsAvg, gpu.gpuName());

    // ── Validation ──────────────────────────────────────────────────────────
    struct Check { const char* name; bool ok; };
    const bool finiteAll = an.valid && std::isfinite(cd) && std::isfinite(cl) &&
                           std::isfinite(an.massAvg) && std::isfinite(an.maxU);
    const Check checks[] = {
        { "all quantities finite",            finiteAll },
        { "mass conserved (1.00 +/- 0.10)",   an.massAvg > 0.90f && an.massAvg < 1.10f },
        { "velocity bounded (max < 0.60)",    an.maxU < 0.60f },
        { "residual decaying",                lastRes < firstRes || lastRes < 5e-2f },
        { "drag positive",                    opts.meshPath.empty() ? cd > 0.f : true },
    };
    bool pass = true;
    std::printf("\n");
    for (const Check& c : checks) {
        std::printf("  [%s] %s\n", c.ok ? "PASS" : "FAIL", c.name);
        pass = pass && c.ok;
    }
    std::printf("\n  %s\n\n", pass ? "VALIDATION PASSED" : "VALIDATION FAILED");

    solver.destroy();
    gpu.destroy();
    return pass ? 0 : 1;
}

} // namespace vwt
