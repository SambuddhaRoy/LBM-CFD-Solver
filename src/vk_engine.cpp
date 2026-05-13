// ============================================================================
// vk_engine.cpp
// ============================================================================

#include "vk_engine.h"
#include "environment.h"
#include "sim_scaler.h"
#include "logger.h"
#include "benchmark/auto_benchmark.h"

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <VkBootstrap.h>

#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>

#include <chrono>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdarg.h>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <commdlg.h>
static std::string openFileDialog() {
    OPENFILENAMEA ofn{}; char buf[260]{};
    ofn.lStructSize = sizeof(ofn); ofn.lpstrFile = buf; ofn.nMaxFile = 260;
    ofn.lpstrFilter = "3D Models\0*.obj;*.stl;*.glb;*.gltf;*.fbx\0All\0*.*\0";
    ofn.nFilterIndex = 1;
    ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST | OFN_NOCHANGEDIR;
    return GetOpenFileNameA(&ofn) ? std::string(buf) : "";
}
#elif defined(__linux__)
#include <cstdio>
static std::string openFileDialog() {
    // Try zenity (GTK/GNOME), then kdialog (KDE/Plasma), in that order.
    static const char* kCmds[] = {
        "zenity --file-selection --title='Open 3D Model' "
            "--file-filter='3D Models (stl obj fbx glb gltf)|*.stl *.obj *.fbx *.glb *.gltf' 2>/dev/null",
        "kdialog --getopenfilename . '*.stl *.obj *.fbx *.glb *.gltf|3D Models' 2>/dev/null",
    };
    for (auto* cmd : kCmds) {
        FILE* fp = popen(cmd, "r");
        if (!fp) continue;
        char buf[4096] = {};
        bool got = (fgets(buf, sizeof(buf), fp) != nullptr);
        int  rc  = pclose(fp);
        if (got && rc == 0 && buf[0]) {
            std::string s(buf);
            while (!s.empty() && (s.back()=='\n'||s.back()=='\r'||s.back()==' ')) s.pop_back();
            if (!s.empty()) return s;
        }
    }
    return "";
}
#endif

namespace vwt {

// ─── UI helpers ──────────────────────────────────────────────────────────────

static bool SectionHeader(const char* label, bool open = true) {
    ImGui::PushStyleColor(ImGuiCol_Header,        {0.07f,0.07f,0.10f,1.f});
    ImGui::PushStyleColor(ImGuiCol_HeaderHovered, {0.10f,0.10f,0.14f,1.f});
    ImGui::PushStyleColor(ImGuiCol_HeaderActive,  {0.12f,0.12f,0.17f,1.f});
    bool r = ImGui::CollapsingHeader(label, open ? ImGuiTreeNodeFlags_DefaultOpen : 0);
    ImGui::PopStyleColor(3);
    return r;
}

static void Sep() {
    ImGui::PushStyleColor(ImGuiCol_Separator, {0.12f,0.12f,0.17f,1.f});
    ImGui::Separator();
    ImGui::PopStyleColor();
}

static void LabelValue(const char* label, const char* fmt, ...) {
    va_list a; va_start(a,fmt); char buf[64]; vsnprintf(buf,64,fmt,a); va_end(a);
    ImGui::PushStyleColor(ImGuiCol_Text, {0.40f,0.40f,0.52f,1.f});
    ImGui::TextUnformatted(label);
    ImGui::PopStyleColor();
    float rw = ImGui::GetContentRegionAvail().x - ImGui::CalcTextSize(buf).x;
    ImGui::SameLine(rw > 0 ? ImGui::GetCursorPosX() + rw : 0);
    ImGui::PushStyleColor(ImGuiCol_Text, {0.82f,0.82f,0.90f,1.f});
    ImGui::TextUnformatted(buf);
    ImGui::PopStyleColor();
}

static void TinyBar(float frac, ImVec4 col, float h = 3.f) {
    ImVec2 p = ImGui::GetCursorScreenPos();
    float  w = ImGui::GetContentRegionAvail().x;
    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->AddRectFilled(p, {p.x+w, p.y+h}, IM_COL32(24,24,34,255), 1.f);
    float fill = std::clamp(frac, 0.f, 1.f);
    if (fill > 0)
        dl->AddRectFilled(p, {p.x+w*fill, p.y+h},
                          ImGui::ColorConvertFloat4ToU32(col), 1.f);
    ImGui::Dummy({w, h+2});
}

// ════════════════════════════════════════════════════════════════════════════
// Init / cleanup
// ════════════════════════════════════════════════════════════════════════════

void VulkanEngine::init() {
    Logger::init();
    loadConfig();
    initWindow();
    initVulkan();
    initPipelineCache();
    initSwapchain();
    initFrameData();
    initRenderPass();
    initFramebuffers();
    initImGui();
    initSimulation();
    initialized_ = true;
    Logger::log("Engine initialized.");
}

void VulkanEngine::cleanup() {
    if (!initialized_) return;
    vkDeviceWaitIdle(device_);
    saveConfig();
    savePipelineCache();
    renderer_.destroy();
    fluidSolver_.destroy();
    cleanupSwapchain();
    for (auto& f : frames_) {
        vkDestroyCommandPool(device_, f.commandPool, nullptr);
        vkDestroySemaphore(device_, f.presentSemaphore, nullptr);
        vkDestroySemaphore(device_, f.renderSemaphore, nullptr);
        vkDestroyFence(device_, f.renderFence, nullptr);
    }
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    mainDQ_.flush();
    vmaDestroyAllocator(allocator_);
    vkDestroyDevice(device_, nullptr);
    vkDestroySurfaceKHR(instance_, surface_, nullptr);
    vkb::destroy_debug_utils_messenger(instance_, debugMessenger_);
    vkDestroyInstance(instance_, nullptr);
    glfwDestroyWindow(window_);
    glfwTerminate();
}

void VulkanEngine::run() {
    while (!glfwWindowShouldClose(window_)) {
        if (resizePending_) {
            vkDeviceWaitIdle(device_);
            simParams_.gridX = std::max(16u, uint32_t(baseGridX_ * gridQuality_));
            simParams_.gridY = std::max(16u, uint32_t(baseGridY_ * gridQuality_));
            simParams_.gridZ = std::max(16u, uint32_t(baseGridZ_ * gridQuality_));
            fluidSolver_.destroy();
            renderer_.destroy();
            initSimulation();
            if (meshLoaded_) loadMesh(meshPath_);
            resizePending_ = false;
        }

        int fw, fh;
        glfwGetFramebufferSize(window_, &fw, &fh);
        if (fw > 0 && fh > 0 &&
            (uint32_t(fw) != windowExtent_.width || uint32_t(fh) != windowExtent_.height)) {
            windowExtent_ = { uint32_t(fw), uint32_t(fh) };
            recreateSwapchain();
        }

        auto t0 = std::chrono::high_resolution_clock::now();
        glfwPollEvents();
        processKeyboard();
        drawFrame();
        auto t1 = std::chrono::high_resolution_clock::now();
        float ms = std::chrono::duration<float,std::milli>(t1-t0).count();
        avgFrameMs_ = avgFrameMs_*0.95f + ms*0.05f;

        float fps = avgFrameMs_ > 0 ? 1000.f / avgFrameMs_ : 0;
        fpsHistory_[fpsHistIdx_++ % kHist] = fps;

        if (simRunning_ && meshLoaded_) {
            float target = 1e-5f + std::exp(-float(totalSteps_)*0.00015f)*0.9f;
            simResidual_ = simResidual_*0.97f + target*0.03f;
        }
        residualHistory_[fpsHistIdx_ % kHist] = std::log10(std::max(simResidual_, 1e-9f));

        // VRAM budget
        VmaBudget budgets[VK_MAX_MEMORY_HEAPS];
        vmaGetHeapBudgets(allocator_, budgets);
        vramBudget_ = 0; vramUsage_ = 0;
        for (int i = 0; i < 8; ++i) {
            vramBudget_ = std::max(vramBudget_, budgets[i].budget);
            vramUsage_  = std::max(vramUsage_,  budgets[i].usage);
        }

        // Update window title
        char title[128];
        snprintf(title, sizeof(title), "Virtual Wind Tunnel  |  %.0f fps  |  step %llu",
                 fps, totalSteps_);
        glfwSetWindowTitle(window_, title);
    }
    vkDeviceWaitIdle(device_);
}

// ════════════════════════════════════════════════════════════════════════════
// Keyboard
// ════════════════════════════════════════════════════════════════════════════

void VulkanEngine::keyCallback(GLFWwindow* w, int key, int, int action, int) {
    if (action != GLFW_PRESS) return;
    auto* eng = static_cast<VulkanEngine*>(glfwGetWindowUserPointer(w));
    if (!eng) return;

    switch (key) {
    case GLFW_KEY_SPACE:
        eng->simRunning_ = !eng->simRunning_; break;
    case GLFW_KEY_R:
        eng->fluidSolver_.resetToEquilibrium();
        eng->totalSteps_ = 0; eng->simResidual_ = 1.f; eng->simRunning_ = false; break;
    case GLFW_KEY_1: eng->simParams_.visMode = VisMode::Velocity;   break;
    case GLFW_KEY_2: eng->simParams_.visMode = VisMode::Pressure;   break;
    case GLFW_KEY_3: eng->simParams_.visMode = VisMode::Vorticity;  break;
    case GLFW_KEY_4: eng->simParams_.visMode = VisMode::QCriterion; break;
    case GLFW_KEY_EQUAL:
    case GLFW_KEY_KP_ADD:
        eng->stepsPerFrame_ = std::min(64, eng->stepsPerFrame_ + 1); break;
    case GLFW_KEY_MINUS:
    case GLFW_KEY_KP_SUBTRACT:
        eng->stepsPerFrame_ = std::max(1,  eng->stepsPerFrame_ - 1); break;
    case GLFW_KEY_F11:
        if (!eng->fullscreen_) { glfwMaximizeWindow(w); eng->fullscreen_ = true; }
        else { glfwRestoreWindow(w); eng->fullscreen_ = false; }
        break;
    case GLFW_KEY_ESCAPE:
        if (eng->showHotkeys_) eng->showHotkeys_ = false;
        else { eng->zoomLevel_ = 1.f; eng->panX_ = 0; eng->panY_ = 0; }
        break;
    case GLFW_KEY_SLASH:   // ? on shifted layouts; we accept both / and ?
        eng->showHotkeys_ = !eng->showHotkeys_; break;
    case GLFW_KEY_TAB:
        eng->leftPanelOpen_ = !eng->leftPanelOpen_; break;
    case GLFW_KEY_F:
        eng->rightPanelOpen_ = !eng->rightPanelOpen_; break;
    }
}

void VulkanEngine::processKeyboard() {
    // Nothing extra needed — handled via callback
}

// ════════════════════════════════════════════════════════════════════════════
// Window
// ════════════════════════════════════════════════════════════════════════════

void VulkanEngine::initWindow() {
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE,  GLFW_TRUE);
    window_ = glfwCreateWindow(windowExtent_.width, windowExtent_.height,
                               "Virtual Wind Tunnel", nullptr, nullptr);
    glfwSetWindowUserPointer(window_, this);
    glfwSetDropCallback(window_, dropCallback);
    glfwSetKeyCallback(window_, keyCallback);
}

void VulkanEngine::dropCallback(GLFWwindow* w, int n, const char** paths) {
    if (n < 1) return;
    auto* e = static_cast<VulkanEngine*>(glfwGetWindowUserPointer(w));
    if (e) { snprintf(e->meshPath_, 512, "%s", paths[0]); e->loadMesh(e->meshPath_); }
}

// ════════════════════════════════════════════════════════════════════════════
// Vulkan
// ════════════════════════════════════════════════════════════════════════════

void VulkanEngine::initVulkan() {
    vkb::InstanceBuilder ib;
    auto ir = ib.set_app_name("VirtualWindTunnel")
               .request_validation_layers(true)
               .use_default_debug_messenger()
               .require_api_version(1,3,0)
               .build();
    if (!ir) throw std::runtime_error("Vulkan instance: " + ir.error().message());
    instance_       = ir.value().instance;
    debugMessenger_ = ir.value().debug_messenger;

    glfwCreateWindowSurface(instance_, window_, nullptr, &surface_);

    vkb::PhysicalDeviceSelector sel(ir.value());
    auto pr = sel.set_minimum_version(1,3).set_surface(surface_)
               .prefer_gpu_device_type(vkb::PreferredDeviceType::discrete).select();
    if (!pr) throw std::runtime_error("Physical device: " + pr.error().message());
    physDevice_ = pr.value().physical_device;

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(physDevice_, &props);
    snprintf(gpuName_, sizeof(gpuName_), "%s", props.deviceName);
    Logger::log("GPU: " + std::string(gpuName_));

    vkb::DeviceBuilder db(pr.value());
    auto dr = db.build();
    if (!dr) throw std::runtime_error("Device: " + dr.error().message());
    device_              = dr.value().device;
    graphicsQueue_       = dr.value().get_queue(vkb::QueueType::graphics).value();
    graphicsQueueFamily_ = dr.value().get_queue_index(vkb::QueueType::graphics).value();

    auto cq = dr.value().get_dedicated_queue(vkb::QueueType::compute);
    if (cq.has_value()) {
        computeQueue_       = cq.value();
        computeQueueFamily_ = dr.value().get_dedicated_queue_index(vkb::QueueType::compute).value();
        hasAsyncCompute_    = true;
        Logger::log("Async compute queue active (family " + std::to_string(computeQueueFamily_) + ")");
    } else {
        computeQueue_       = graphicsQueue_;
        computeQueueFamily_ = graphicsQueueFamily_;
    }

    VmaAllocatorCreateInfo vai{};
    vai.physicalDevice = physDevice_; vai.device = device_; vai.instance = instance_;
    vmaCreateAllocator(&vai, &allocator_);
}

void VulkanEngine::initPipelineCache() {
    const char* path = "pipeline_cache.bin";
    std::vector<char> data;
    if (std::ifstream f(path, std::ios::binary|std::ios::ate); f.is_open()) {
        data.resize(size_t(f.tellg())); f.seekg(0);
        f.read(data.data(), std::streamsize(data.size()));
        Logger::log("Pipeline cache loaded (" + std::to_string(data.size()) + " bytes)");
    }
    VkPipelineCacheCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
    ci.initialDataSize = data.size();
    ci.pInitialData    = data.empty() ? nullptr : data.data();
    VK_CHECK(vkCreatePipelineCache(device_, &ci, nullptr, &pipelineCache_));
    mainDQ_.push([this](){ vkDestroyPipelineCache(device_, pipelineCache_, nullptr); });
}

void VulkanEngine::savePipelineCache() {
    size_t sz = 0;
    vkGetPipelineCacheData(device_, pipelineCache_, &sz, nullptr);
    if (!sz) return;
    std::vector<uint8_t> d(sz);
    vkGetPipelineCacheData(device_, pipelineCache_, &sz, d.data());
    std::ofstream f("pipeline_cache.bin", std::ios::binary);
    f.write(reinterpret_cast<const char*>(d.data()), std::streamsize(sz));
    Logger::log("Pipeline cache saved (" + std::to_string(sz) + " bytes)");
}

void VulkanEngine::initSwapchain() {
    vkb::SwapchainBuilder sb(physDevice_, device_, surface_);
    auto sr = sb.use_default_format_selection()
               .set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR)
               .set_desired_extent(windowExtent_.width, windowExtent_.height)
               .build();
    if (!sr) throw std::runtime_error("Swapchain: " + sr.error().message());
    swapchain_  = sr.value().swapchain;
    swapchainFmt_ = sr.value().image_format;
    swapImages_   = sr.value().get_images().value();
    swapViews_    = sr.value().get_image_views().value();
}

void VulkanEngine::initFrameData() {
    for (auto& f : frames_) {
        VkCommandPoolCreateInfo pi{};
        pi.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pi.queueFamilyIndex = graphicsQueueFamily_;
        pi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        VK_CHECK(vkCreateCommandPool(device_, &pi, nullptr, &f.commandPool));

        VkCommandBufferAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        ai.commandPool = f.commandPool;
        ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 1;
        VK_CHECK(vkAllocateCommandBuffers(device_, &ai, &f.commandBuffer));

        VkFenceCreateInfo fi{};
        fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fi.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        VK_CHECK(vkCreateFence(device_, &fi, nullptr, &f.renderFence));

        VkSemaphoreCreateInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        VK_CHECK(vkCreateSemaphore(device_, &si, nullptr, &f.presentSemaphore));
        VK_CHECK(vkCreateSemaphore(device_, &si, nullptr, &f.renderSemaphore));
    }
}

void VulkanEngine::initRenderPass() {
    VkAttachmentDescription ca{};
    ca.format = swapchainFmt_; ca.samples = VK_SAMPLE_COUNT_1_BIT;
    ca.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; ca.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    ca.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    ca.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    ca.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    ca.finalLayout   = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference cr{ 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
    VkSubpassDescription sub{};
    sub.pipelineBindPoint    = VK_PIPELINE_BIND_POINT_GRAPHICS;
    sub.colorAttachmentCount = 1; sub.pColorAttachments = &cr;

    VkSubpassDependency dep{};
    dep.srcSubpass = VK_SUBPASS_EXTERNAL; dep.dstSubpass = 0;
    dep.srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dep.dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    VkRenderPassCreateInfo ri{};
    ri.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    ri.attachmentCount = 1; ri.pAttachments = &ca;
    ri.subpassCount = 1;    ri.pSubpasses   = &sub;
    ri.dependencyCount = 1; ri.pDependencies = &dep;
    VK_CHECK(vkCreateRenderPass(device_, &ri, nullptr, &renderPass_));
}

void VulkanEngine::initFramebuffers() {
    framebuffers_.resize(swapViews_.size());
    for (size_t i = 0; i < swapViews_.size(); ++i) {
        VkFramebufferCreateInfo fi{};
        fi.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fi.renderPass = renderPass_; fi.attachmentCount = 1; fi.pAttachments = &swapViews_[i];
        fi.width = windowExtent_.width; fi.height = windowExtent_.height; fi.layers = 1;
        VK_CHECK(vkCreateFramebuffer(device_, &fi, nullptr, &framebuffers_[i]));
    }
}

void VulkanEngine::cleanupSwapchain() {
    vkDeviceWaitIdle(device_);
    for (auto fb : framebuffers_) vkDestroyFramebuffer(device_, fb, nullptr);
    vkDestroyRenderPass(device_, renderPass_, nullptr);
    for (auto iv : swapViews_) vkDestroyImageView(device_, iv, nullptr);
    vkDestroySwapchainKHR(device_, swapchain_, nullptr);
}

void VulkanEngine::recreateSwapchain() {
    cleanupSwapchain();
    initSwapchain();
    initRenderPass();
    initFramebuffers();
}

// ════════════════════════════════════════════════════════════════════════════
// ImGui style
// ════════════════════════════════════════════════════════════════════════════

void VulkanEngine::initImGui() {
    VkDescriptorPoolSize ps[] = {
        {VK_DESCRIPTOR_TYPE_SAMPLER,                100},
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 100},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,           10},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,          10},
    };
    VkDescriptorPoolCreateInfo pi{};
    pi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pi.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    pi.maxSets = 100; pi.poolSizeCount = 4; pi.pPoolSizes = ps;
    VK_CHECK(vkCreateDescriptorPool(device_, &pi, nullptr, &imguiPool_));

    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.IniFilename = nullptr;  // We manage config ourselves

    // Compute DPI scale from framebuffer/window ratio — works on X11, Wayland and XWayland
    // (glfwGetWindowContentScale returns 1.0 on many Wayland compositors, so we avoid it)
    float dpiScale = 1.f;
    if (window_) {
        int ww = 1, wh = 1, fw = 1, fh = 1;
        glfwGetWindowSize(window_, &ww, &wh);
        glfwGetFramebufferSize(window_, &fw, &fh);
        if (ww > 0 && wh > 0)
            dpiScale = std::max(float(fw) / float(ww), float(fh) / float(wh));
        dpiScale = std::max(1.f, dpiScale);
    }
    dpiScale_ = dpiScale;
    // FontGlobalScale inverts the oversize so logical sizes remain correct;
    // the atlas is just rasterised at higher resolution for sharpness.
    io.FontGlobalScale = 1.f / dpiScale;

#ifdef _WIN32
    fontBody_ = io.Fonts->AddFontFromFileTTF("C:/Windows/Fonts/segoeui.ttf", 15.f * dpiScale);
    fontMono_ = io.Fonts->AddFontFromFileTTF("C:/Windows/Fonts/consola.ttf", 12.f * dpiScale);
#else
    // Try common distro paths (Debian/Ubuntu, Arch/Cachy, Fedora)
    auto tryFont = [&](std::initializer_list<const char*> paths, float sz) -> ImFont* {
        for (const char* p : paths) {
            std::ifstream f(p);
            if (f.good()) return io.Fonts->AddFontFromFileTTF(p, sz);
        }
        return nullptr;
    };
    fontBody_ = tryFont({
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/liberation-sans/LiberationSans-Regular.ttf",
    }, 15.f * dpiScale);
    fontMono_ = tryFont({
        "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/liberation-mono/LiberationMono-Regular.ttf",
    }, 12.f * dpiScale);
    if (!fontBody_) fontBody_ = io.Fonts->AddFontDefault();
#endif

    ImGuiStyle& s = ImGui::GetStyle();
    // Set base (unscaled) values first, then ScaleAllSizes multiplies them correctly.
    // Calling ScaleAllSizes before explicit assignments would have no effect.
    s.WindowRounding = 0.f; s.ChildRounding = 4.f; s.FrameRounding = 4.f;
    s.PopupRounding  = 6.f; s.TabRounding   = 4.f; s.GrabRounding  = 4.f;
    s.WindowBorderSize = 0.f; s.FrameBorderSize = 0.f;
    s.ItemSpacing    = {8,5}; s.ItemInnerSpacing = {6,4};
    s.WindowPadding  = {12,10}; s.FramePadding = {8,4};
    s.IndentSpacing  = 14.f; s.ScrollbarSize = 10.f;
    s.ScaleAllSizes(dpiScale);

    constexpr ImVec4 kA  = {0.11f,0.82f,0.63f,1.f};   // teal accent
    constexpr ImVec4 kAD = {0.08f,0.55f,0.42f,1.f};
    constexpr ImVec4 kAL = {0.15f,1.00f,0.76f,1.f};

    auto& c = s.Colors;
    c[ImGuiCol_WindowBg]            = {0.051f,0.051f,0.067f,1.f};
    c[ImGuiCol_ChildBg]             = {0.067f,0.067f,0.082f,1.f};
    c[ImGuiCol_PopupBg]             = {0.078f,0.078f,0.098f,1.f};
    c[ImGuiCol_Text]                = {0.82f,0.82f,0.88f,1.f};
    c[ImGuiCol_TextDisabled]        = {0.28f,0.28f,0.38f,1.f};
    c[ImGuiCol_Border]              = {0.11f,0.11f,0.15f,1.f};
    c[ImGuiCol_FrameBg]             = {0.09f,0.09f,0.12f,1.f};
    c[ImGuiCol_FrameBgHovered]      = {0.11f,0.11f,0.15f,1.f};
    c[ImGuiCol_FrameBgActive]       = {0.14f,0.14f,0.19f,1.f};
    c[ImGuiCol_TitleBg]             = {0.04f,0.04f,0.05f,1.f};
    c[ImGuiCol_TitleBgActive]       = {0.05f,0.05f,0.07f,1.f};
    c[ImGuiCol_ScrollbarBg]         = {0.04f,0.04f,0.05f,1.f};
    c[ImGuiCol_ScrollbarGrab]       = {0.18f,0.18f,0.24f,1.f};
    c[ImGuiCol_ScrollbarGrabHovered]= {0.26f,0.26f,0.34f,1.f};
    c[ImGuiCol_ScrollbarGrabActive] = {0.34f,0.34f,0.44f,1.f};
    c[ImGuiCol_CheckMark]           = kA;
    c[ImGuiCol_SliderGrab]          = kAD;
    c[ImGuiCol_SliderGrabActive]    = kAL;
    c[ImGuiCol_Button]              = {0.10f,0.10f,0.14f,1.f};
    c[ImGuiCol_ButtonHovered]       = {0.11f,0.55f,0.42f,0.20f};
    c[ImGuiCol_ButtonActive]        = {0.11f,0.82f,0.63f,0.28f};
    c[ImGuiCol_Header]              = {0.09f,0.09f,0.12f,1.f};
    c[ImGuiCol_HeaderHovered]       = {0.11f,0.11f,0.15f,1.f};
    c[ImGuiCol_HeaderActive]        = {0.13f,0.13f,0.17f,1.f};
    c[ImGuiCol_Separator]           = {0.11f,0.11f,0.15f,1.f};
    c[ImGuiCol_Tab]                 = {0.07f,0.07f,0.09f,1.f};
    c[ImGuiCol_TabHovered]          = {0.11f,0.55f,0.42f,0.22f};
    c[ImGuiCol_TabActive]           = {0.07f,0.48f,0.36f,1.f};
    c[ImGuiCol_PlotLines]           = kA;
    c[ImGuiCol_PlotLinesHovered]    = kAL;
    c[ImGuiCol_PlotHistogram]       = kAD;
    c[ImGuiCol_DragDropTarget]      = kA;

    ImGui_ImplGlfw_InitForVulkan(window_, true);
    ImGui_ImplVulkan_InitInfo ii{};
    ii.ApiVersion = VK_API_VERSION_1_3; ii.Instance = instance_;
    ii.PhysicalDevice = physDevice_; ii.Device = device_;
    ii.QueueFamily = graphicsQueueFamily_; ii.Queue = graphicsQueue_;
    ii.DescriptorPool = imguiPool_; ii.MinImageCount = 2;
    ii.ImageCount = uint32_t(swapImages_.size());
    ii.PipelineInfoMain.RenderPass = renderPass_;
    ImGui_ImplVulkan_Init(&ii);

    mainDQ_.push([this](){ vkDestroyDescriptorPool(device_, imguiPool_, nullptr); });
}

// ════════════════════════════════════════════════════════════════════════════
// Simulation init
// ════════════════════════════════════════════════════════════════════════════

void VulkanEngine::initSimulation() {
    fluidSolver_.init(device_, allocator_, computeQueue_,
                      computeQueueFamily_, pipelineCache_, simParams_);
    renderer_.init(device_, allocator_, imguiPool_,
                   pipelineCache_, simParams_, fluidSolver_.getMacroBuffer());
}

void VulkanEngine::loadMesh(const std::string& path) {
    vkDeviceWaitIdle(device_);
    try {
        auto mesh = meshLoader_.loadMesh(path);
        auto obs  = meshLoader_.voxelizeSurface(mesh,
                        simParams_.gridX, simParams_.gridY, simParams_.gridZ);
        fluidSolver_.uploadObstacleMap(obs);
        fluidSolver_.resetToEquilibrium();
        meshLoaded_   = true;
        totalSteps_   = 0;
        simResidual_  = 1.f;
        Logger::log("Mesh loaded: " + path);
    } catch (const std::exception& e) {
        Logger::error("Mesh load failed: " + std::string(e.what()));
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Config persistence (simple key=value ini)
// ════════════════════════════════════════════════════════════════════════════

void VulkanEngine::loadConfig() {
    std::ifstream f("vwt_config.ini");
    if (!f.is_open()) return;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        auto eq = line.find('=');
        if (eq == std::string::npos) continue;
        std::string key = line.substr(0, eq);
        std::string val = line.substr(eq+1);
        if (key == "tau")         simParams_.tau = std::stof(val);
        else if (key == "inletX") simParams_.inletVelX = std::stof(val);
        else if (key == "inletY") simParams_.inletVelY = std::stof(val);
        else if (key == "inletZ") simParams_.inletVelZ = std::stof(val);
        else if (key == "gridQuality") gridQuality_ = std::stof(val);
        else if (key == "stepsPerFrame") stepsPerFrame_ = std::stoi(val);
        else if (key == "lbmMode") simParams_.lbmMode = std::stoi(val);
        else if (key == "sliceAxis") simParams_.sliceAxis = std::stoi(val);
        else if (key == "visMode") simParams_.visMode = VisMode(std::stoi(val));
        else if (key == "maxVelocity") simParams_.maxVelocity = std::stof(val);
        else if (key == "turbulence") simParams_.turbulence = std::stof(val);
        else if (key == "env") simParams_.currentEnvironmentIndex = std::stoi(val);
        else if (key == "winW") windowExtent_.width  = std::stoi(val);
        else if (key == "winH") windowExtent_.height = std::stoi(val);
    }
}

void VulkanEngine::saveConfig() {
    std::ofstream f("vwt_config.ini");
    if (!f.is_open()) return;
    f << "# Virtual Wind Tunnel config\n";
    f << "tau=" << simParams_.tau << "\n";
    f << "inletX=" << simParams_.inletVelX << "\n";
    f << "inletY=" << simParams_.inletVelY << "\n";
    f << "inletZ=" << simParams_.inletVelZ << "\n";
    f << "gridQuality=" << gridQuality_ << "\n";
    f << "stepsPerFrame=" << stepsPerFrame_ << "\n";
    f << "lbmMode=" << simParams_.lbmMode << "\n";
    f << "sliceAxis=" << simParams_.sliceAxis << "\n";
    f << "visMode=" << int(simParams_.visMode) << "\n";
    f << "maxVelocity=" << simParams_.maxVelocity << "\n";
    f << "turbulence=" << simParams_.turbulence << "\n";
    f << "env=" << simParams_.currentEnvironmentIndex << "\n";
    f << "winW=" << windowExtent_.width << "\n";
    f << "winH=" << windowExtent_.height << "\n";
}

// ════════════════════════════════════════════════════════════════════════════
// Frame rendering
// ════════════════════════════════════════════════════════════════════════════

void VulkanEngine::drawFrame() {
    auto& fr = frame();

    VK_CHECK(vkWaitForFences(device_, 1, &fr.renderFence, VK_TRUE, 1'000'000'000));
    VK_CHECK(vkResetFences(device_, 1, &fr.renderFence));

    // Read back previous frame's GPU results
    if (totalSteps_ > 0) {
        auto t = fluidSolver_.readTimings();
        gpuTimings_.lbmMs  = gpuTimings_.lbmMs  * 0.9f + t.lbmMs  * 0.1f;
        gpuTimings_.aeroMs = gpuTimings_.aeroMs * 0.9f + t.aeroMs * 0.1f;
        if (aeroDispatchThisFrame_) {
            aeroForces_ = fluidSolver_.readAeroForces();
            aeroDispatchThisFrame_ = false;
        }
    }

    uint32_t imageIndex;
    VkResult acq = vkAcquireNextImageKHR(device_, swapchain_, 1'000'000'000,
                       fr.presentSemaphore, VK_NULL_HANDLE, &imageIndex);
    if (acq == VK_ERROR_OUT_OF_DATE_KHR) { recreateSwapchain(); return; }

    VK_CHECK(vkResetCommandBuffer(fr.commandBuffer, 0));
    buildCommandBuffer(fr.commandBuffer, imageIndex);

    VkPipelineStageFlags ws = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.waitSemaphoreCount   = 1; si.pWaitSemaphores   = &fr.presentSemaphore;
    si.pWaitDstStageMask    = &ws;
    si.commandBufferCount   = 1; si.pCommandBuffers   = &fr.commandBuffer;
    si.signalSemaphoreCount = 1; si.pSignalSemaphores = &fr.renderSemaphore;
    VK_CHECK(vkQueueSubmit(graphicsQueue_, 1, &si, fr.renderFence));

    VkPresentInfoKHR pres{};
    pres.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    pres.waitSemaphoreCount = 1; pres.pWaitSemaphores = &fr.renderSemaphore;
    pres.swapchainCount = 1; pres.pSwapchains = &swapchain_; pres.pImageIndices = &imageIndex;
    VkResult pr = vkQueuePresentKHR(graphicsQueue_, &pres);
    if (pr == VK_ERROR_OUT_OF_DATE_KHR || pr == VK_SUBOPTIMAL_KHR) recreateSwapchain();

    currentFrame_ = (currentFrame_ + 1) % FRAMES_IN_FLIGHT;
}

void VulkanEngine::buildCommandBuffer(VkCommandBuffer cmd, uint32_t imageIndex) {
    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cmd, &bi));

    // LBM steps
    if (simRunning_ && meshLoaded_) {
        for (int i = 0; i < stepsPerFrame_; ++i) {
            fluidSolver_.step(cmd, simParams_, uint32_t(totalSteps_));
            ++totalSteps_;
        }
        // Dispatch aero forces every N frames
        aeroDispatchThisFrame_ = (totalSteps_ % aeroUpdateInterval_ == 0);
        if (aeroDispatchThisFrame_) {
            fluidSolver_.dispatchAeroForces(cmd, simParams_);
        }
    }

    // Visualization slice
    renderer_.computeSlice(cmd, simParams_);

    // Render pass (ImGui)
    VkClearValue cv{}; cv.color = {{0.051f,0.051f,0.067f,1.f}};
    VkRenderPassBeginInfo rp{};
    rp.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rp.renderPass = renderPass_; rp.framebuffer = framebuffers_[imageIndex];
    rp.renderArea = {{0,0}, windowExtent_};
    rp.clearValueCount = 1; rp.pClearValues = &cv;
    vkCmdBeginRenderPass(cmd, &rp, VK_SUBPASS_CONTENTS_INLINE);

    drawImGui();

    vkCmdEndRenderPass(cmd);
    VK_CHECK(vkEndCommandBuffer(cmd));
}

void VulkanEngine::drawImGui() {
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    drawUI_TopBar();
    drawUI_Rail();
    drawUI_Left();
    drawUI_Viewport();
    drawUI_Right();
    drawUI_StatusBar();
    drawUI_HotkeyOverlay();

    ImGui::Render();
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(),
        frames_[currentFrame_].commandBuffer);
}

// ════════════════════════════════════════════════════════════════════════════
// UI helpers
// ════════════════════════════════════════════════════════════════════════════

// Thin full-width separator
static void UISep() {
    ImGui::PushStyleColor(ImGuiCol_Separator, {0.11f,0.11f,0.16f,1.f});
    ImGui::Separator();
    ImGui::PopStyleColor();
}

// Section card header (coloured left-border pill + label + optional badge)
static void CardHeader(const char* label, const char* badge = nullptr,
                       ImVec4 badgeCol = {0.11f,0.82f,0.63f,1.f},
                       ImVec4 badgeBg  = {0.04f,0.22f,0.16f,1.f}) {
    ImGui::PushStyleColor(ImGuiCol_Text, {0.75f,0.75f,0.85f,1.f});
    ImGui::TextUnformatted(label);
    ImGui::PopStyleColor();
    if (badge) {
        float bw = ImGui::CalcTextSize(badge).x + 10.f;
        ImGui::SameLine(ImGui::GetContentRegionAvail().x - bw + ImGui::GetCursorPosX()
                        + ImGui::GetWindowPos().x - ImGui::GetWindowPos().x);
        ImGui::PushStyleColor(ImGuiCol_Button,        badgeBg);
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, badgeBg);
        ImGui::PushStyleColor(ImGuiCol_ButtonActive,  badgeBg);
        ImGui::PushStyleColor(ImGuiCol_Text,          badgeCol);
        ImGui::SmallButton(badge);
        ImGui::PopStyleColor(4);
    }
}

// Begin a bordered card child window
static bool BeginCard(const char* id, float height = 0.f) {
    ImGui::PushStyleColor(ImGuiCol_ChildBg, {0.078f,0.078f,0.102f,1.f});
    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding,   6.f);
    ImGui::PushStyleVar(ImGuiStyleVar_ChildBorderSize, 1.f);
    ImGui::PushStyleColor(ImGuiCol_Border, {0.14f,0.14f,0.20f,1.f});
    bool v = ImGui::BeginChild(id, {-1, height}, true,
        ImGuiWindowFlags_NoScrollbar|ImGuiWindowFlags_NoScrollWithMouse);
    ImGui::PopStyleColor(2);
    ImGui::PopStyleVar(2);
    return v;
}
static void EndCard() { ImGui::EndChild(); }

// Draw a coloured left-border accent on the current card (call right after BeginCard)
static void CardAccent(ImVec4 col) {
    ImVec2 p = ImGui::GetWindowPos();
    ImGui::GetWindowDrawList()->AddRectFilled(
        {p.x+1, p.y+4}, {p.x+3, p.y + ImGui::GetWindowHeight()-4},
        ImGui::ColorConvertFloat4ToU32(col), 2.f);
}

// Big headline metric: large number + unit + optional delta badge
static void BigMetric(const char* label, const char* valFmt, float val,
                      const char* unit = "",
                      float deltaPercent = 0.f, bool showDelta = false) {
    ImGui::PushStyleColor(ImGuiCol_Text, {0.38f,0.38f,0.50f,1.f});
    ImGui::Text("%s", label);
    ImGui::PopStyleColor();

    char vbuf[32]; snprintf(vbuf, sizeof(vbuf), valFmt, val);
    ImGui::PushStyleColor(ImGuiCol_Text, {0.92f,0.92f,0.98f,1.f});
    ImGui::SetWindowFontScale(1.35f);
    ImGui::TextUnformatted(vbuf);
    ImGui::SetWindowFontScale(1.f);
    ImGui::PopStyleColor();

    if (unit[0]) {
        ImGui::SameLine(0,4);
        ImGui::PushStyleColor(ImGuiCol_Text, {0.38f,0.38f,0.50f,1.f});
        ImGui::TextUnformatted(unit);
        ImGui::PopStyleColor();
    }

    if (showDelta && deltaPercent != 0.f) {
        char db[16]; snprintf(db, sizeof(db), "%+.1f%%", deltaPercent);
        bool pos = deltaPercent > 0.f;
        ImVec4 dc = pos ? ImVec4{1.f,0.52f,0.52f,1.f} : ImVec4{0.11f,0.82f,0.63f,1.f};
        ImVec4 bg = pos ? ImVec4{0.20f,0.04f,0.04f,1.f} : ImVec4{0.04f,0.18f,0.12f,1.f};
        ImGui::SameLine(0,6);
        ImGui::PushStyleColor(ImGuiCol_Button,        bg);
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, bg);
        ImGui::PushStyleColor(ImGuiCol_ButtonActive,  bg);
        ImGui::PushStyleColor(ImGuiCol_Text,          dc);
        ImGui::SmallButton(db);
        ImGui::PopStyleColor(4);
    }
}

// Gradient bar (mini sparkline height indicator + label on same row)
static void GpuBar(const char* label, float fraction,
                   ImVec4 colA, ImVec4 colB, const char* valStr) {
    ImGui::PushStyleColor(ImGuiCol_Text, {0.40f,0.40f,0.52f,1.f});
    ImGui::Text("%-10s", label);
    ImGui::PopStyleColor();
    ImGui::SameLine(0, 6);

    ImVec2 p   = ImGui::GetCursorScreenPos();
    float  w   = ImGui::GetContentRegionAvail().x - ImGui::CalcTextSize(valStr).x - 8.f;
    float  h   = 4.f;
    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->AddRectFilled(p, {p.x+w, p.y+h}, IM_COL32(20,20,30,255), 2.f);
    float fill = std::clamp(fraction, 0.f, 1.f) * w;
    if (fill > 2.f) {
        dl->AddRectFilledMultiColor(
            p, {p.x+fill, p.y+h},
            ImGui::ColorConvertFloat4ToU32(colA),
            ImGui::ColorConvertFloat4ToU32(colB),
            ImGui::ColorConvertFloat4ToU32(colB),
            ImGui::ColorConvertFloat4ToU32(colA));
    }
    ImGui::Dummy({w, h});
    ImGui::SameLine(0, 8);
    ImGui::PushStyleColor(ImGuiCol_Text, colB);
    ImGui::TextUnformatted(valStr);
    ImGui::PopStyleColor();
}

// Inline label + right-aligned value (mono font for values)
static void StatRow(const char* key, const char* valFmt, ...) {
    va_list a; va_start(a,valFmt); char vb[48]; vsnprintf(vb,48,valFmt,a); va_end(a);
    ImGui::PushStyleColor(ImGuiCol_Text, {0.38f,0.38f,0.50f,1.f});
    ImGui::TextUnformatted(key);
    ImGui::PopStyleColor();
    float rx = ImGui::GetContentRegionAvail().x - ImGui::CalcTextSize(vb).x;
    ImGui::SameLine(ImGui::GetCursorPosX() + (rx > 0 ? rx : 0));
    ImGui::PushStyleColor(ImGuiCol_Text, {0.78f,0.78f,0.88f,1.f});
    ImGui::TextUnformatted(vb);
    ImGui::PopStyleColor();
}

// Toggle-group button helper (returns true if clicked)
static bool ToggleBtn(const char* label, bool active, ImVec2 size = {0,22}) {
    if (active) {
        ImGui::PushStyleColor(ImGuiCol_Button,        {0.05f,0.38f,0.28f,1.f});
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, {0.07f,0.52f,0.38f,1.f});
        ImGui::PushStyleColor(ImGuiCol_Text,          {0.12f,0.92f,0.70f,1.f});
    } else {
        ImGui::PushStyleColor(ImGuiCol_Button,        {0.07f,0.07f,0.10f,1.f});
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, {0.11f,0.11f,0.15f,1.f});
        ImGui::PushStyleColor(ImGuiCol_Text,          {0.34f,0.34f,0.46f,1.f});
    }
    bool clicked = ImGui::Button(label, size);
    ImGui::PopStyleColor(3);
    return clicked;
}

// Slider with value pill on the right
static bool SliderPill(const char* id, const char* label,
                       float* v, float lo, float hi, const char* fmt) {
    ImGui::PushStyleColor(ImGuiCol_Text, {0.40f,0.40f,0.52f,1.f});
    ImGui::TextUnformatted(label);
    ImGui::PopStyleColor();

    char vbuf[32]; snprintf(vbuf,sizeof(vbuf),fmt,*v);
    float pillW = ImGui::CalcTextSize(vbuf).x + 14.f;
    float sliderW = ImGui::GetContentRegionAvail().x - pillW - 6.f;

    ImGui::SetNextItemWidth(sliderW);
    bool changed = ImGui::SliderFloat(id, v, lo, hi, "");

    ImGui::SameLine(0, 6);
    ImGui::PushStyleColor(ImGuiCol_ChildBg,  {0.06f,0.06f,0.09f,1.f});
    ImGui::PushStyleColor(ImGuiCol_Border,   {0.14f,0.14f,0.20f,1.f});
    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding,  4.f);
    ImGui::PushStyleVar(ImGuiStyleVar_ChildBorderSize,1.f);
    ImGui::BeginChild(("##pill_" + std::string(id)).c_str(), {pillW, 18.f}, true,
        ImGuiWindowFlags_NoScrollbar|ImGuiWindowFlags_NoScrollWithMouse);
    ImGui::SetCursorPosY(1.f);
    ImGui::PushStyleColor(ImGuiCol_Text, {0.82f,0.82f,0.90f,1.f});
    ImGui::SetNextItemWidth(-1);
    ImGui::TextUnformatted(vbuf);
    ImGui::PopStyleColor();
    ImGui::EndChild();
    ImGui::PopStyleColor(2); ImGui::PopStyleVar(2);
    return changed;
}

// ════════════════════════════════════════════════════════════════════════════
// UI — Layout & design tokens
// ════════════════════════════════════════════════════════════════════════════

namespace {

// Theme palette (colours used in many places — single source of truth)
constexpr ImVec4 kBgRoot      = {0.027f, 0.027f, 0.035f, 1.f};
constexpr ImVec4 kBgPanel     = {0.043f, 0.043f, 0.059f, 1.f};
constexpr ImVec4 kBgRail      = {0.020f, 0.020f, 0.030f, 1.f};
constexpr ImVec4 kBgInput     = {0.055f, 0.055f, 0.075f, 1.f};
constexpr ImVec4 kBorder      = {0.110f, 0.110f, 0.150f, 1.f};
constexpr ImVec4 kText        = {0.860f, 0.860f, 0.940f, 1.f};
constexpr ImVec4 kTextDim     = {0.420f, 0.420f, 0.530f, 1.f};
constexpr ImVec4 kTextMuted   = {0.250f, 0.250f, 0.350f, 1.f};
constexpr ImVec4 kAccent      = {0.110f, 0.820f, 0.630f, 1.f};
constexpr ImVec4 kAccentDim   = {0.080f, 0.500f, 0.380f, 1.f};
constexpr ImVec4 kAccentBg    = {0.040f, 0.220f, 0.160f, 1.f};
constexpr ImVec4 kBlue        = {0.440f, 0.740f, 1.000f, 1.f};
constexpr ImVec4 kPurple      = {0.680f, 0.550f, 1.000f, 1.f};
constexpr ImVec4 kAmber       = {0.990f, 0.720f, 0.220f, 1.f};
constexpr ImVec4 kRed         = {1.000f, 0.420f, 0.420f, 1.f};

// Computed layout for a single frame
struct Layout {
    ImVec2 ds;          // display size
    float  S;           // global scale (= dpiScale_)
    float  topH;        // top bar height
    float  statusH;     // status bar height
    float  railW;       // rail width
    float  leftW;       // left panel width (0 = collapsed)
    float  rightW;      // right panel width (0 = collapsed)
    float  vpX, vpY;    // viewport top-left
    float  vpW, vpH;    // viewport dimensions
};

Layout computeLayout(ImVec2 ds, float scale, bool leftOpen, bool rightOpen) {
    Layout L;
    L.ds      = ds;
    L.S       = scale;
    L.topH    = 44.f * scale;
    L.statusH = 24.f * scale;
    L.railW   = 56.f * scale;
    L.leftW   = leftOpen  ? std::clamp(ds.x * 0.18f, 240.f * scale, 360.f * scale) : 0.f;
    L.rightW  = rightOpen ? std::clamp(ds.x * 0.18f, 240.f * scale, 360.f * scale) : 0.f;

    // Force-collapse panels if there isn't room for the viewport
    const float minVp = 420.f * scale;
    if (ds.x - L.railW - L.leftW - L.rightW < minVp) L.rightW = 0.f;
    if (ds.x - L.railW - L.leftW - L.rightW < minVp) L.leftW  = 0.f;
    if (ds.x - L.railW - L.leftW - L.rightW < minVp) L.railW  = 0.f;

    L.vpX = L.railW + L.leftW;
    L.vpY = L.topH;
    L.vpW = std::max(0.f, ds.x - L.vpX - L.rightW);
    L.vpH = std::max(0.f, ds.y - L.topH - L.statusH);
    return L;
}

// Round rectangular pill — fast hand-drawn primitive
void DrawPill(ImDrawList* dl, ImVec2 a, ImVec2 b, ImU32 fill, ImU32 border = 0, float r = 5.f) {
    dl->AddRectFilled(a, b, fill, r);
    if (border) dl->AddRect(a, b, border, r, 0, 1.f);
}

// Centred icon-button used in rail and toolbars
bool IconButton(const char* id, const char* glyph, ImVec2 size,
                bool active, ImVec4 activeCol = kAccent) {
    ImGui::PushStyleColor(ImGuiCol_Button,        active ? ImVec4{0.05f,0.30f,0.22f,1.f} : ImVec4{0,0,0,0});
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4{0.10f,0.10f,0.14f,1.f});
    ImGui::PushStyleColor(ImGuiCol_ButtonActive,  ImVec4{0.05f,0.28f,0.20f,1.f});
    ImGui::PushStyleColor(ImGuiCol_Text,          active ? activeCol : kTextDim);
    ImVec2 pos = ImGui::GetCursorScreenPos();
    bool clicked = ImGui::Button(id, size);
    ImGui::PopStyleColor(4);

    // Draw glyph centred (single char buttons can't auto-centre)
    ImVec2 ts = ImGui::CalcTextSize(glyph);
    ImGui::GetWindowDrawList()->AddText(ImGui::GetFont(), size.y * 0.46f,
        {pos.x + (size.x - ts.x) * 0.5f, pos.y + (size.y - ts.y) * 0.5f},
        ImGui::ColorConvertFloat4ToU32(active ? activeCol : kTextDim),
        glyph);
    return clicked;
}

// A small "tag" pill for inline metadata badges (e.g. "LIVE", "BGK")
void Tag(const char* text, ImVec4 fg, ImVec4 bg) {
    ImGui::PushStyleColor(ImGuiCol_Button,        bg);
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, bg);
    ImGui::PushStyleColor(ImGuiCol_ButtonActive,  bg);
    ImGui::PushStyleColor(ImGuiCol_Text,          fg);
    ImGui::SmallButton(text);
    ImGui::PopStyleColor(4);
}

// Section header inside left panel — collapsible accordion
bool SectionHeading(const char* label, bool* open, ImVec4 accent) {
    ImGui::PushStyleColor(ImGuiCol_Header,        ImVec4{0.07f,0.07f,0.10f,1.f});
    ImGui::PushStyleColor(ImGuiCol_HeaderHovered, ImVec4{0.10f,0.10f,0.14f,1.f});
    ImGui::PushStyleColor(ImGuiCol_HeaderActive,  ImVec4{0.12f,0.12f,0.17f,1.f});

    ImVec2 cs = ImGui::GetCursorScreenPos();
    ImDrawList* dl = ImGui::GetWindowDrawList();
    // Coloured accent dot
    dl->AddCircleFilled({cs.x + 6.f, cs.y + ImGui::GetFontSize()*0.5f + 4.f},
                       3.5f, ImGui::ColorConvertFloat4ToU32(accent));
    ImGui::Indent(16.f);
    bool clicked = ImGui::Selectable(label, false,
                                     ImGuiSelectableFlags_AllowOverlap, {0, 22.f});
    ImGui::Unindent(16.f);
    if (clicked && open) *open = !*open;

    // Chevron indicator
    float chW = ImGui::GetContentRegionAvail().x;
    (void)chW;
    const char* chev = (open && *open) ? "v" : ">";
    ImVec2 sz = ImGui::CalcTextSize(chev);
    dl->AddText(ImGui::GetFont(), ImGui::GetFontSize()*0.85f,
        {cs.x + ImGui::GetContentRegionAvail().x + 16.f - sz.x,
         cs.y + ImGui::GetFontSize()*0.5f - sz.y*0.3f + 2.f},
        ImGui::ColorConvertFloat4ToU32(kTextMuted), chev);

    ImGui::PopStyleColor(3);
    return open ? *open : true;
}

} // anonymous namespace


// ════════════════════════════════════════════════════════════════════════════
// UI — Top bar  (brand + primary actions + live perf)
// ════════════════════════════════════════════════════════════════════════════

void VulkanEngine::drawUI_TopBar() {
    const ImVec2 ds = ImGui::GetIO().DisplaySize;
    const Layout L  = computeLayout(ds, dpiScale_, leftPanelOpen_, rightPanelOpen_);

    ImGui::SetNextWindowPos({0, 0});
    ImGui::SetNextWindowSize({ds.x, L.topH});
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {s(14.f), 0});
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4{0.025f,0.025f,0.034f,1.f});
    ImGui::Begin("##TopBar", nullptr,
        ImGuiWindowFlags_NoTitleBar|ImGuiWindowFlags_NoResize|ImGuiWindowFlags_NoMove|
        ImGuiWindowFlags_NoCollapse|ImGuiWindowFlags_NoScrollbar|
        ImGuiWindowFlags_NoBringToFrontOnFocus);

    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImVec2 wp = ImGui::GetWindowPos();

    // Bottom hairline
    dl->AddLine({wp.x, wp.y + L.topH - 0.5f}, {wp.x + ds.x, wp.y + L.topH - 0.5f},
                ImGui::ColorConvertFloat4ToU32(kBorder), 1.f);

    // ── Brand mark + title ──────────────────────────────────────────────
    float yMid = (L.topH - s(28.f)) * 0.5f;
    ImGui::SetCursorPos({s(14.f), yMid});
    ImVec2 lp = ImGui::GetCursorScreenPos();
    dl->AddRectFilledMultiColor(lp, {lp.x + s(28.f), lp.y + s(28.f)},
        IM_COL32(29,209,161,255), IM_COL32(0,206,201,255),
        IM_COL32(0,176,155,255),  IM_COL32(29,209,161,255));
    dl->AddText(ImGui::GetFont(), s(15.f),
        {lp.x + s(8.f), lp.y + s(6.f)}, IM_COL32(10,20,16,255), "L");
    ImGui::Dummy({s(36.f), s(28.f)});

    ImGui::SameLine(0, s(8.f));
    ImGui::SetCursorPosY((L.topH - ImGui::GetFontSize()) * 0.5f);
    ImGui::PushStyleColor(ImGuiCol_Text, kText);
    ImGui::TextUnformatted("LBM-CFD Solver");
    ImGui::PopStyleColor();

    if (ds.x > s(820.f)) {
        ImGui::SameLine(0, s(8.f));
        ImGui::SetCursorPosY((L.topH - ImGui::GetFontSize()) * 0.5f);
        ImGui::PushStyleColor(ImGuiCol_Text, kTextMuted);
        ImGui::TextUnformatted("v0.2");
        ImGui::PopStyleColor();
    }

    // ── Centred primary actions ─────────────────────────────────────────
    auto bigBtn = [&](const char* label, ImVec4 col, ImVec4 colHov, ImVec4 textCol,
                      float w) -> bool {
        ImGui::PushStyleColor(ImGuiCol_Button,        col);
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, colHov);
        ImGui::PushStyleColor(ImGuiCol_Text,          textCol);
        bool r = ImGui::Button(label, {w, s(28.f)});
        ImGui::PopStyleColor(3);
        return r;
    };

    auto subtle = [&](const char* label, float w) -> bool {
        return bigBtn(label, ImVec4{0.07f,0.07f,0.10f,1.f},
                            ImVec4{0.11f,0.11f,0.16f,1.f},
                            kText, w);
    };

    // Build the action row in a centred group
    const float btnSpacing = s(6.f);
    float btnRowW = 0.f;
    btnRowW += s(74.f) + btnSpacing;  // Open
    btnRowW += s(86.f) + btnSpacing;  // Run/Pause
    btnRowW += s(74.f) + btnSpacing;  // Reset
    if (ds.x > s(900.f)) btnRowW += s(82.f) + btnSpacing;  // Snap
    if (ds.x > s(1050.f)) btnRowW += s(54.f) + btnSpacing; // ?
    btnRowW -= btnSpacing;

    float actionsX = (ds.x - btnRowW) * 0.5f;
    if (actionsX < s(280.f)) actionsX = s(280.f);
    ImGui::SameLine();
    ImGui::SetCursorPosX(actionsX);
    ImGui::SetCursorPosY(yMid);

    if (subtle("Open", s(74.f))) {
        auto path = openFileDialog();
        if (!path.empty()) { snprintf(meshPath_,512,"%s",path.c_str()); loadMesh(meshPath_); }
    }
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Open 3D model (.stl, .obj, .fbx)");
    ImGui::SameLine(0, btnSpacing);

    if (simRunning_) {
        if (bigBtn("Pause", ImVec4{0.20f,0.10f,0.04f,1.f}, ImVec4{0.30f,0.16f,0.06f,1.f},
                   ImVec4{0.99f,0.72f,0.22f,1.f}, s(86.f)))
            simRunning_ = false;
    } else {
        if (bigBtn("Run", ImVec4{0.06f,0.38f,0.27f,1.f}, ImVec4{0.09f,0.55f,0.40f,1.f},
                   ImVec4{0.86f,0.98f,0.92f,1.f}, s(86.f)))
            simRunning_ = true;
    }
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Run / pause simulation [Space]");
    ImGui::SameLine(0, btnSpacing);

    if (subtle("Reset", s(74.f))) {
        fluidSolver_.resetToEquilibrium();
        totalSteps_ = 0; simResidual_ = 1.f; simRunning_ = false;
        aeroCDPrev_ = 0; aeroCLPrev_ = 0;
    }
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Reset simulation state [R]");

    if (ds.x > s(900.f)) {
        ImGui::SameLine(0, btnSpacing);
        if (subtle("Snapshot", s(82.f))) { /* TODO: capture viewport */ }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Save viewport snapshot [S]");
    }
    if (ds.x > s(1050.f)) {
        ImGui::SameLine(0, btnSpacing);
        if (subtle("?", s(54.f))) showHotkeys_ = !showHotkeys_;
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Show keyboard shortcuts [?]");
    }

    // ── Right side: GPU + FPS pill ──────────────────────────────────────
    float fps  = avgFrameMs_ > 0 ? 1000.f / avgFrameMs_ : 0.f;
    float mlups = 0.f;
    {
        double cells = double(simParams_.gridX) * simParams_.gridY * simParams_.gridZ;
        mlups = float((fps * stepsPerFrame_ * cells) / 1.0e6);
    }
    char gpuShort[24]; snprintf(gpuShort, sizeof(gpuShort), "%s", gpuName_);
    if (strlen(gpuShort) > 22) { gpuShort[22]='.'; gpuShort[23]=0; }

    char perfBuf[80];
    snprintf(perfBuf, sizeof(perfBuf), "%.0f fps  %.2f MLUPS", fps, mlups);

    ImVec2 perfSz = ImGui::CalcTextSize(perfBuf);
    float pillW = perfSz.x + s(20.f);
    float pillX = ds.x - pillW - s(14.f);
    float pillY = yMid;
    {
        ImVec2 pa = {wp.x + pillX, wp.y + pillY};
        ImVec2 pb = {pa.x + pillW, pa.y + s(28.f)};
        DrawPill(dl, pa, pb, IM_COL32(8,12,18,220), IM_COL32(40,40,56,160), s(6.f));
        dl->AddText(ImGui::GetFont(), 0.f,
            {pa.x + s(10.f), pa.y + (s(28.f) - ImGui::GetFontSize())*0.5f},
            ImGui::ColorConvertFloat4ToU32(kAccent), perfBuf);
    }

    if (ds.x > s(1100.f)) {
        ImVec2 gpuSz = ImGui::CalcTextSize(gpuShort);
        float gpuW = gpuSz.x + s(20.f);
        float gpuX = pillX - gpuW - s(8.f);
        ImVec2 ga = {wp.x + gpuX, wp.y + pillY};
        ImVec2 gb = {ga.x + gpuW, ga.y + s(28.f)};
        DrawPill(dl, ga, gb, IM_COL32(20,18,40,220), IM_COL32(60,55,90,160), s(6.f));
        dl->AddText(ImGui::GetFont(), 0.f,
            {ga.x + s(10.f), ga.y + (s(28.f) - ImGui::GetFontSize())*0.5f},
            ImGui::ColorConvertFloat4ToU32(kPurple), gpuShort);
    }

    ImGui::End();
    ImGui::PopStyleColor();
    ImGui::PopStyleVar();
}

// ════════════════════════════════════════════════════════════════════════════
// UI — Icon rail  (left edge, navigation between modes)
// ════════════════════════════════════════════════════════════════════════════

void VulkanEngine::drawUI_Rail() {
    const ImVec2 ds = ImGui::GetIO().DisplaySize;
    const Layout L  = computeLayout(ds, dpiScale_, leftPanelOpen_, rightPanelOpen_);
    if (L.railW <= 0.f) return;

    ImGui::SetNextWindowPos({0, L.topH});
    ImGui::SetNextWindowSize({L.railW, ds.y - L.topH - L.statusH});
    ImGui::PushStyleColor(ImGuiCol_WindowBg, kBgRail);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {0, s(8.f)});
    ImGui::Begin("##Rail", nullptr,
        ImGuiWindowFlags_NoTitleBar|ImGuiWindowFlags_NoResize|ImGuiWindowFlags_NoMove|
        ImGuiWindowFlags_NoCollapse|ImGuiWindowFlags_NoBringToFrontOnFocus|
        ImGuiWindowFlags_NoScrollbar);

    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImVec2 wp = ImGui::GetWindowPos();
    // Right hairline
    dl->AddLine({wp.x + L.railW - 0.5f, wp.y},
                {wp.x + L.railW - 0.5f, wp.y + ImGui::GetWindowHeight()},
                ImGui::ColorConvertFloat4ToU32(kBorder), 1.f);

    const float btnSz  = s(36.f);
    const float btnOff = (L.railW - btnSz) * 0.5f;

    struct Item { const char* glyph; const char* tip; };
    static const Item items[] = {
        {"S",  "Simulation"},
        {"M",  "Mesh tools"},
        {"P",  "Probes"},
        {"C",  "Compare"},
        {"L",  "Library"},
    };

    for (int i = 0; i < 5; ++i) {
        ImGui::SetCursorPosX(btnOff);
        ImVec2 btnP = ImGui::GetCursorScreenPos();
        bool act = (railMode_ == i);

        if (act) {
            // Active accent bar on the left edge
            dl->AddRectFilled({wp.x, btnP.y + s(6.f)},
                              {wp.x + s(2.5f), btnP.y + btnSz - s(6.f)},
                              ImGui::ColorConvertFloat4ToU32(kAccent), 1.f);
            dl->AddRectFilled(btnP, {btnP.x + btnSz, btnP.y + btnSz},
                              IM_COL32(11,72,54,255), s(7.f));
        }
        char bid[8]; snprintf(bid, 8, "##r%d", i);
        if (IconButton(bid, items[i].glyph, {btnSz, btnSz}, act, kAccent))
            railMode_ = i;
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", items[i].tip);

        ImGui::Dummy({0, s(4.f)});
    }

    // Bottom: collapse + settings
    float btmY = ImGui::GetWindowHeight() - btnSz*2.f - s(12.f);
    ImGui::SetCursorPosY(btmY);
    ImGui::SetCursorPosX(btnOff);
    if (IconButton("##coll", leftPanelOpen_ ? "<" : ">", {btnSz, btnSz}, false, kTextDim))
        leftPanelOpen_ = !leftPanelOpen_;
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip(leftPanelOpen_ ? "Collapse left panel" : "Expand left panel");

    ImGui::Dummy({0, s(4.f)});
    ImGui::SetCursorPosX(btnOff);
    bool settingsAct = (railMode_ == 5);
    if (IconButton("##set", "*", {btnSz, btnSz}, settingsAct, kTextDim))
        railMode_ = settingsAct ? 0 : 5;
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Settings");

    ImGui::End();
    ImGui::PopStyleVar();
    ImGui::PopStyleColor();
}

// ════════════════════════════════════════════════════════════════════════════
// UI — Left panel content helpers (per-mode bodies)
// ════════════════════════════════════════════════════════════════════════════

static void drawPlaceholder(const char* title, const char* body) {
    ImGui::PushStyleColor(ImGuiCol_Text, kTextDim);
    ImGui::TextUnformatted(title);
    ImGui::PopStyleColor();
    ImGui::Dummy({0, 6});
    ImGui::PushStyleColor(ImGuiCol_Text, kTextMuted);
    ImGui::TextWrapped("%s", body);
    ImGui::PopStyleColor();
}

// ════════════════════════════════════════════════════════════════════════════
// UI — Left panel  (mode-aware: simulation, mesh, probes, compare, settings)
// ════════════════════════════════════════════════════════════════════════════

void VulkanEngine::drawUI_Left() {
    const ImVec2 ds = ImGui::GetIO().DisplaySize;
    const Layout L  = computeLayout(ds, dpiScale_, leftPanelOpen_, rightPanelOpen_);
    if (L.leftW <= 0.f) return;

    ImGui::SetNextWindowPos({L.railW, L.topH});
    ImGui::SetNextWindowSize({L.leftW, ds.y - L.topH - L.statusH});
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {s(14.f), s(14.f)});
    ImGui::PushStyleColor(ImGuiCol_WindowBg, kBgPanel);
    ImGui::Begin("##Left", nullptr,
        ImGuiWindowFlags_NoTitleBar|ImGuiWindowFlags_NoResize|ImGuiWindowFlags_NoMove|
        ImGuiWindowFlags_NoCollapse|ImGuiWindowFlags_NoBringToFrontOnFocus);

    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImVec2 wp = ImGui::GetWindowPos();
    dl->AddLine({wp.x + L.leftW - 0.5f, wp.y},
                {wp.x + L.leftW - 0.5f, wp.y + ImGui::GetWindowHeight()},
                ImGui::ColorConvertFloat4ToU32(kBorder), 1.f);

    // ── Panel header (title + subtitle) ─────────────────────────────────
    static const char* kTitle[] = {"Simulation", "Mesh tools", "Probes", "Compare", "Library", "Settings"};
    static const char* kSub[]   = {"D3Q19 Lattice Boltzmann", "Geometry & repair",
                                   "Flow probes & time series", "Compare runs",
                                   "Saved cases", "Preferences"};

    ImGui::PushStyleColor(ImGuiCol_Text, kText);
    ImGui::SetWindowFontScale(1.10f);
    ImGui::TextUnformatted(kTitle[std::clamp(railMode_, 0, 5)]);
    ImGui::SetWindowFontScale(1.f);
    ImGui::PopStyleColor();
    ImGui::PushStyleColor(ImGuiCol_Text, kAccent);
    ImGui::TextUnformatted(kSub[std::clamp(railMode_, 0, 5)]);
    ImGui::PopStyleColor();
    ImGui::Dummy({0, s(8.f)});
    UISep();
    ImGui::Dummy({0, s(6.f)});

    // ── Mode bodies ─────────────────────────────────────────────────────
    if (railMode_ == 1) {
        drawPlaceholder("Mesh repair, decimation, refinement",
            "Tools for simplifying, repairing and refining loaded meshes will live here. "
            "For now, drop a model to load it directly into the simulation.");
    } else if (railMode_ == 2) {
        drawPlaceholder("Probe placement",
            "Place pressure / velocity probes at world positions to record live time-series data.");
    } else if (railMode_ == 3) {
        drawPlaceholder("Run comparison",
            "Compare convergence and aerodynamic coefficients across runs.");
    } else if (railMode_ == 4) {
        drawPlaceholder("Library",
            "Sample cases and saved configurations.");
    } else if (railMode_ == 5) {
        // ── Settings: Right-panel card visibility ──────────────────────────
        if (BeginCard("##sCards", 0.f)) {
            CardAccent(kBlue);
            ImGui::SetCursorPosX(ImGui::GetCursorPosX() + s(6.f));
            CardHeader("Result Cards");
            ImGui::Dummy({0, s(6.f)});
            ImGui::PushStyleColor(ImGuiCol_Text, kText);
            ImGui::Checkbox("Aerodynamic forces",  &showAeroCard_);
            ImGui::Checkbox("Convergence history", &showConvCard_);
            ImGui::Checkbox("Flow statistics",     &showFlowCard_);
            ImGui::Checkbox("GPU performance",     &showGpuCard_);
            ImGui::PopStyleColor();
            ImGui::Dummy({0, s(6.f)});
        }
        EndCard();
        ImGui::Dummy({0, s(8.f)});

        // ── Settings: Performance ──────────────────────────────────────────
        if (BeginCard("##sPerfCard", 0.f)) {
            CardAccent(kAmber);
            ImGui::SetCursorPosX(ImGui::GetCursorPosX() + s(6.f));
            CardHeader("Performance");
            ImGui::Dummy({0, s(6.f)});

            ImGui::PushStyleColor(ImGuiCol_Text, kTextDim);
            ImGui::TextUnformatted("Aero update interval (steps)");
            ImGui::PopStyleColor();
            float auiW = ImGui::GetContentRegionAvail().x - s(48.f);
            ImGui::SetNextItemWidth(auiW);
            int aui = int(aeroUpdateInterval_);
            if (ImGui::SliderInt("##auiS", &aui, 5, 200)) aeroUpdateInterval_ = uint64_t(aui);
            ImGui::SameLine(0, s(6.f));
            ImGui::PushStyleColor(ImGuiCol_Text, kText);
            ImGui::Text("%d", aui);
            ImGui::PopStyleColor();

            ImGui::Dummy({0, s(4.f)});
            ImGui::PushStyleColor(ImGuiCol_Text, kTextDim);
            ImGui::TextUnformatted("Steps per frame");
            ImGui::PopStyleColor();
            float spfW = ImGui::GetContentRegionAvail().x - s(48.f);
            ImGui::SetNextItemWidth(spfW);
            ImGui::SliderInt("##spfS", &stepsPerFrame_, 1, 64);
            ImGui::SameLine(0, s(6.f));
            ImGui::PushStyleColor(ImGuiCol_Text, kText);
            ImGui::Text("%d", stepsPerFrame_);
            ImGui::PopStyleColor();

            ImGui::Dummy({0, s(6.f)});
        }
        EndCard();
        ImGui::Dummy({0, s(8.f)});

        // ── Settings: Layout ──────────────────────────────────────────────
        if (BeginCard("##sLayCard", 0.f)) {
            CardAccent(kPurple);
            ImGui::SetCursorPosX(ImGui::GetCursorPosX() + s(6.f));
            CardHeader("Layout");
            ImGui::Dummy({0, s(6.f)});
            ImGui::PushStyleColor(ImGuiCol_Text, kText);
            ImGui::Checkbox("Right panel (F)",          &rightPanelOpen_);
            ImGui::Checkbox("Keyboard shortcuts overlay", &showHotkeys_);
            ImGui::PopStyleColor();
            ImGui::Dummy({0, s(4.f)});
            if (ImGui::Button("Show shortcuts now", {-1, s(24.f)}))
                showHotkeys_ = true;
            ImGui::Dummy({0, s(6.f)});
        }
        EndCard();
        ImGui::Dummy({0, s(8.f)});

        // ── Settings: Configuration ────────────────────────────────────────
        if (BeginCard("##sConfCard", 0.f)) {
            CardAccent(kTextMuted);
            ImGui::SetCursorPosX(ImGui::GetCursorPosX() + s(6.f));
            CardHeader("Configuration");
            ImGui::Dummy({0, s(6.f)});
            float bwConf = (ImGui::GetContentRegionAvail().x - s(4.f)) * 0.5f;
            if (ImGui::Button("Load Config", {bwConf, s(24.f)})) loadConfig();
            ImGui::SameLine(0, s(4.f));
            if (ImGui::Button("Save Config", {-1, s(24.f)}))     saveConfig();
            ImGui::Dummy({0, s(8.f)});
            ImGui::PushStyleColor(ImGuiCol_Text, kTextMuted);
            ImGui::TextUnformatted("LBM-CFD Solver  \xE2\x80\x94  D3Q19");
            ImGui::TextUnformatted("Vulkan 1.3  |  BGK + MRT-RLB");
            ImGui::PopStyleColor();
            ImGui::Dummy({0, s(6.f)});
        }
        EndCard();
    } else {
        // ─── Mode 0: Simulation (default) ────────────────────────────────
        // Geometry
        if (BeginCard("##cGeom", 0.f)) {
            CardAccent(kAccent);
            ImGui::SetCursorPosX(ImGui::GetCursorPosX() + s(6.f));
            CardHeader("Geometry", meshLoaded_ ? "LOADED" : nullptr);
            ImGui::Dummy({0, s(6.f)});

            if (meshLoaded_) {
                std::string fn = meshPath_;
                auto p = fn.find_last_of("/\\");
                if (p != std::string::npos) fn = fn.substr(p + 1);
                ImGui::PushStyleColor(ImGuiCol_Text, kText);
                ImGui::TextUnformatted(fn.c_str());
                ImGui::PopStyleColor();
                ImGui::PushStyleColor(ImGuiCol_Text, kTextDim);
                ImGui::Text("%u\xC3\x97%u\xC3\x97%u cells",
                            simParams_.gridX, simParams_.gridY, simParams_.gridZ);
                ImGui::PopStyleColor();
                ImGui::Dummy({0, s(4.f)});

                float bw = (ImGui::GetContentRegionAvail().x - s(4.f)) * 0.5f;
                if (ImGui::Button("Browse##bm", {bw, s(24.f)})) {
                    auto path = openFileDialog();
                    if (!path.empty()) { snprintf(meshPath_,512,"%s",path.c_str()); loadMesh(meshPath_); }
                }
                ImGui::SameLine(0, s(4.f));
                ImGui::PushStyleColor(ImGuiCol_Button,        ImVec4{0.18f,0.04f,0.04f,1.f});
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4{0.28f,0.06f,0.06f,1.f});
                if (ImGui::Button("Clear##clr", {-1, s(24.f)})) {
                    std::vector<uint32_t> empty(size_t(simParams_.gridX) * simParams_.gridY * simParams_.gridZ, 0);
                    fluidSolver_.uploadObstacleMap(empty);
                    fluidSolver_.resetToEquilibrium();
                    meshLoaded_ = false; memset(meshPath_, 0, 512);
                }
                ImGui::PopStyleColor(2);
            } else {
                ImGui::PushStyleColor(ImGuiCol_ChildBg, kBgInput);
                ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 5.f);
                ImGui::BeginChild("##dz", {-1, s(48.f)}, true, ImGuiWindowFlags_NoScrollbar);
                ImGui::SetCursorPos({s(14.f), s(14.f)});
                ImGui::PushStyleColor(ImGuiCol_Text, kTextMuted);
                ImGui::TextUnformatted("Drop  .stl  .obj  .fbx  here");
                ImGui::PopStyleColor();
                ImGui::EndChild();
                ImGui::PopStyleColor(); ImGui::PopStyleVar();

                ImGui::Dummy({0, s(4.f)});
                if (ImGui::Button("Browse Model...", {-1, s(28.f)})) {
                    auto path = openFileDialog();
                    if (!path.empty()) { snprintf(meshPath_,512,"%s",path.c_str()); loadMesh(meshPath_); }
                }
            }

            ImGui::Dummy({0, s(8.f)});
            uint32_t cx = std::max(16u, uint32_t(baseGridX_ * gridQuality_));
            uint32_t cy = std::max(16u, uint32_t(baseGridY_ * gridQuality_));
            uint32_t cz = std::max(16u, uint32_t(baseGridZ_ * gridQuality_));
            char gfmt[24]; snprintf(gfmt, 24, "%.1f\xC3\x97", gridQuality_);
            SliderPill("##gq", "Voxel resolution", &gridQuality_, 0.5f, 2.f, gfmt);
            ImGui::PushStyleColor(ImGuiCol_Text, kTextMuted);
            ImGui::Text("  %u\xC3\x97%u\xC3\x97%u  ~ %zuM cells",
                        cx, cy, cz, size_t(cx) * cy * cz / 1000000 + 1);
            ImGui::PopStyleColor();
            ImGui::Dummy({0, s(4.f)});
            if (ImGui::Button("Apply Resolution", {-1, s(24.f)})) resizePending_ = true;
            ImGui::Dummy({0, s(6.f)});
        }
        EndCard();
        ImGui::Dummy({0, s(8.f)});

        // Flow conditions
        if (BeginCard("##cFlow", 0.f)) {
            CardAccent(kBlue);
            ImGui::SetCursorPosX(ImGui::GetCursorPosX() + s(6.f));
            CardHeader("Flow Conditions");
            ImGui::Dummy({0, s(6.f)});

            static const char* uNames[] = {"m/s", "km/h", "mph", "kn"};
            static const float uScale[] = {594.45f, 2140.f, 1329.f, 1155.f};
            float tabW = (ImGui::GetContentRegionAvail().x - s(6.f)) / 4.f;
            for (int i = 0; i < 4; ++i) {
                if (ToggleBtn(uNames[i], velocityUnit_ == i, {tabW, s(22.f)}))
                    velocityUnit_ = i;
                if (i < 3) ImGui::SameLine(0, 2);
            }
            ImGui::Dummy({0, s(4.f)});

            float hw = (ImGui::GetContentRegionAvail().x - s(4.f)) * 0.5f;
            if (ToggleBtn("Subsonic",  speedMode_ == 0, {hw, s(22.f)})) speedMode_ = 0;
            ImGui::SameLine(0, s(4.f));
            if (ToggleBtn("Supersonic",speedMode_ == 1, {hw, s(22.f)})) speedMode_ = 1;
            ImGui::Dummy({0, s(4.f)});

            float sc = uScale[velocityUnit_];
            float mX = speedMode_ ? -1.2f : 0.f, MX = speedMode_ ? 1.2f : 0.2f;

            auto flowRow = [&](const char* lbl, float* v, float lo, float hi) {
                float d = *v * sc;
                char fmt[20]; snprintf(fmt, 20, "%.1f %s", d, uNames[velocityUnit_]);
                SliderPill(("##fs" + std::string(lbl)).c_str(), lbl, &d, lo*sc, hi*sc, fmt);
                *v = d / sc;
            };
            flowRow("X-Flow", &simParams_.inletVelX, mX, MX);
            flowRow("Y-Flow", &simParams_.inletVelY, -0.5f, 0.5f);
            flowRow("Z-Flow", &simParams_.inletVelZ, -0.5f, 0.5f);
            ImGui::Dummy({0, s(2.f)});
            SliderPill("##turb", "Turbulence", &simParams_.turbulence, 0.f, 0.1f, "%.3f");

            float vPhys = simParams_.inletVelX * 594.45f;
            float Re    = std::abs(vPhys) * 0.3f / 1.5e-5f;
            ImGui::Dummy({0, s(4.f)});
            StatRow("Reynolds number", "%.2e", Re);
            ImGui::Dummy({0, s(6.f)});
        }
        EndCard();
        ImGui::Dummy({0, s(8.f)});

        // Environment
        if (BeginCard("##cEnv", 0.f)) {
            CardAccent(kPurple);
            ImGui::SetCursorPosX(ImGui::GetCursorPosX() + s(6.f));
            CardHeader("Environment");
            ImGui::Dummy({0, s(6.f)});

            auto& profs = EnvironmentRegistry::getProfiles();
            int nP = int(profs.size());
            int cols = std::min(nP, std::max(2, int(L.leftW / s(96.f))));
            float cellW = (ImGui::GetContentRegionAvail().x - float(cols - 1) * s(4.f)) / float(cols);
            static const char* envIcons[] = {"@","~","V","T","W"};

            for (int i = 0; i < nP; ++i) {
                bool act = (int(simParams_.currentEnvironmentIndex) == i);
                ImGui::PushStyleColor(ImGuiCol_ChildBg,
                    act ? ImVec4{0.04f,0.22f,0.16f,1.f} : kBgInput);
                ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 6.f);
                ImGui::PushStyleVar(ImGuiStyleVar_ChildBorderSize, act ? 1.f : 0.5f);
                ImGui::PushStyleColor(ImGuiCol_Border,
                    act ? ImVec4{0.11f,0.82f,0.63f,0.6f} : kBorder);

                char cid[16]; snprintf(cid, 16, "##ec%d", i);
                if (ImGui::BeginChild(cid, {cellW, s(50.f)}, true,
                        ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse)) {
                    if (ImGui::IsWindowHovered(ImGuiHoveredFlags_ChildWindows) &&
                        ImGui::IsMouseClicked(0)) {
                        simParams_.currentEnvironmentIndex = uint32_t(i);
                        auto& p = profs[i];
                        float dt = SimulationScaler::suggestLatticeDt(p.getKinematicViscosity(), 0.01f, 0.6f);
                        simParams_.tau = SimulationScaler::calculateTau(p.getKinematicViscosity(), 0.01f, dt);
                    }
                    ImGui::SetCursorPos({s(6.f), s(4.f)});
                    ImGui::PushStyleColor(ImGuiCol_Text, act ? kAccent : kTextDim);
                    ImGui::SetWindowFontScale(1.2f);
                    ImGui::TextUnformatted(i < 5 ? envIcons[i] : "?");
                    ImGui::SetWindowFontScale(1.f);
                    ImGui::SetCursorPosX(s(6.f));
                    ImGui::PushStyleColor(ImGuiCol_Text, act ? kText : kTextDim);
                    ImGui::TextUnformatted(profs[i].name.c_str());
                    ImGui::PopStyleColor(2);
                }
                ImGui::EndChild();
                ImGui::PopStyleColor(2); ImGui::PopStyleVar(2);
                if ((i % cols) != cols - 1 && i < nP - 1) ImGui::SameLine(0, s(4.f));
            }
            ImGui::Dummy({0, s(6.f)});
        }
        EndCard();
        ImGui::Dummy({0, s(8.f)});

        // Solver
        if (BeginCard("##cSolv", 0.f)) {
            CardAccent(kAmber);
            ImGui::SetCursorPosX(ImGui::GetCursorPosX() + s(6.f));
            const char* modeName = simParams_.lbmMode == 0 ? "BGK" : "MRT";
            CardHeader("Solver", modeName,
                       kAmber, ImVec4{0.18f,0.12f,0.02f,1.f});
            ImGui::Dummy({0, s(6.f)});

            float hw = (ImGui::GetContentRegionAvail().x - s(4.f)) * 0.5f;
            if (ToggleBtn("BGK", simParams_.lbmMode == 0, {hw, s(22.f)})) simParams_.lbmMode = 0;
            ImGui::SameLine(0, s(4.f));
            if (ToggleBtn("MRT", simParams_.lbmMode == 1, {hw, s(22.f)})) simParams_.lbmMode = 1;
            ImGui::Dummy({0, s(4.f)});

            SliderPill("##tau", "Relaxation \xCF\x84", &simParams_.tau, 0.501f, 2.f, "%.4f");
            if (simParams_.lbmMode == 1) {
                SliderPill("##sb", "s_bulk",  &simParams_.s_bulk,  0.5f, 2.f, "%.2f");
                SliderPill("##sg", "s_ghost", &simParams_.s_ghost, 0.5f, 2.f, "%.2f");
            }
            ImGui::Dummy({0, s(2.f)});
            ImGui::PushStyleColor(ImGuiCol_Text, kTextDim);
            ImGui::TextUnformatted("Steps / frame");
            ImGui::PopStyleColor();
            float spfW = ImGui::GetContentRegionAvail().x - s(44.f);
            ImGui::SetNextItemWidth(spfW);
            ImGui::SliderInt("##spfI", &stepsPerFrame_, 1, 64);
            ImGui::SameLine(0, s(6.f));
            ImGui::PushStyleColor(ImGuiCol_Text, kText);
            ImGui::Text("%d", stepsPerFrame_);
            ImGui::PopStyleColor();
            ImGui::Dummy({0, s(6.f)});
        }
        EndCard();
        ImGui::Dummy({0, s(8.f)});
    }

    ImGui::End();
    ImGui::PopStyleColor();
    ImGui::PopStyleVar();
}

// ════════════════════════════════════════════════════════════════════════════
// UI — Welcome overlay (shown in viewport when no mesh loaded)
// ════════════════════════════════════════════════════════════════════════════

void VulkanEngine::drawWelcomeOverlay(ImVec2 vpPos, ImVec2 vpSize) {
    ImDrawList* dl = ImGui::GetForegroundDrawList();

    float boxW = std::min(s(440.f), vpSize.x - s(40.f));
    float boxH = s(220.f);
    ImVec2 ba = {vpPos.x + (vpSize.x - boxW) * 0.5f,
                 vpPos.y + (vpSize.y - boxH) * 0.5f};
    ImVec2 bb = {ba.x + boxW, ba.y + boxH};

    // Card background with subtle gradient
    dl->AddRectFilledMultiColor(ba, bb,
        IM_COL32(20,22,28,210), IM_COL32(26,32,40,210),
        IM_COL32(20,22,28,210), IM_COL32(15,17,22,210));
    dl->AddRect(ba, bb, IM_COL32(40,55,50,160), s(8.f), 0, 1.f);

    // Logo
    ImVec2 lp = {ba.x + (boxW - s(48.f)) * 0.5f, ba.y + s(24.f)};
    dl->AddRectFilledMultiColor(lp, {lp.x + s(48.f), lp.y + s(48.f)},
        IM_COL32(29,209,161,255), IM_COL32(0,206,201,255),
        IM_COL32(0,176,155,255),  IM_COL32(29,209,161,255));
    dl->AddText(ImGui::GetFont(), s(24.f),
        {lp.x + s(14.f), lp.y + s(10.f)}, IM_COL32(10,20,16,255), "L");

    // Title
    const char* title = "LBM-CFD Solver";
    ImVec2 ts = ImGui::CalcTextSize(title);
    dl->AddText(ImGui::GetFont(), s(18.f),
        {ba.x + (boxW - ts.x * (s(18.f) / ImGui::GetFontSize())) * 0.5f,
         lp.y + s(56.f)},
        ImGui::ColorConvertFloat4ToU32(kText), title);

    // Subtitle
    const char* sub = "Drop a 3D model anywhere to begin a simulation,";
    const char* sub2 = "or click \"Open\" in the toolbar above.";
    ImVec2 ss  = ImGui::CalcTextSize(sub);
    ImVec2 ss2 = ImGui::CalcTextSize(sub2);
    dl->AddText(ImGui::GetFont(), 0.f,
        {ba.x + (boxW - ss.x) * 0.5f,  lp.y + s(86.f)},
        ImGui::ColorConvertFloat4ToU32(kTextDim), sub);
    dl->AddText(ImGui::GetFont(), 0.f,
        {ba.x + (boxW - ss2.x) * 0.5f, lp.y + s(102.f)},
        ImGui::ColorConvertFloat4ToU32(kTextDim), sub2);

    // Hint
    const char* hint = "Supports .stl  .obj  .fbx  .glb  .gltf";
    ImVec2 hs = ImGui::CalcTextSize(hint);
    dl->AddText(ImGui::GetFont(), 0.f,
        {ba.x + (boxW - hs.x) * 0.5f, bb.y - s(34.f)},
        ImGui::ColorConvertFloat4ToU32(kTextMuted), hint);
}

// ════════════════════════════════════════════════════════════════════════════
// UI — Viewport colorbar (drawn into background drawlist)
// ════════════════════════════════════════════════════════════════════════════

void VulkanEngine::drawViewportColorbar(ImDrawList* dl, ImVec2 vpMin, ImVec2 vpMax) {
    static const ImVec4 kStops[] = {
        {0.00f,0.00f,0.01f,1.f},{0.24f,0.06f,0.44f,1.f},{0.58f,0.11f,0.48f,1.f},
        {0.85f,0.26f,0.31f,1.f},{0.99f,0.56f,0.08f,1.f},{0.99f,1.00f,0.64f,1.f},
    };
    static const ImVec4 kCW[] = {
        {0.23f,0.30f,0.75f,1.f},{0.55f,0.58f,0.80f,1.f},{0.86f,0.86f,0.86f,1.f},
        {0.80f,0.46f,0.32f,1.f},{0.71f,0.02f,0.15f,1.f},
    };
    static const ImVec4 kVir[] = {
        {0.27f,0.00f,0.33f,1.f},{0.28f,0.34f,0.61f,1.f},{0.13f,0.57f,0.55f,1.f},
        {0.37f,0.79f,0.38f,1.f},{0.99f,0.91f,0.14f,1.f},
    };

    struct CS { const ImVec4* stops; int n; };
    static const CS maps[4] = {{kStops,6}, {kCW,5}, {kVir,5}, {kStops,6}};

    int mode = int(simParams_.visMode);
    const ImVec4* stops = maps[mode].stops;
    int N = maps[mode].n;

    const float cbH = s(160.f);
    const float cbW = s(10.f);
    const float marginR = s(16.f);

    float x0 = vpMax.x - marginR - cbW;
    float y0 = vpMin.y + s(72.f);

    // Background & frame
    dl->AddRectFilled({x0 - s(6.f), y0 - s(8.f)},
                      {x0 + cbW + s(40.f), y0 + cbH + s(20.f)},
                      IM_COL32(8, 10, 14, 180), s(4.f));

    for (int i = 0; i < N - 1; ++i) {
        float ya = y0 + cbH * (1.f - float(i + 1) / (N - 1));
        float yb = y0 + cbH * (1.f - float(i)     / (N - 1));
        dl->AddRectFilledMultiColor(
            {x0, ya}, {x0 + cbW, yb},
            ImGui::ColorConvertFloat4ToU32(stops[i+1]),
            ImGui::ColorConvertFloat4ToU32(stops[i+1]),
            ImGui::ColorConvertFloat4ToU32(stops[i]),
            ImGui::ColorConvertFloat4ToU32(stops[i]));
    }
    dl->AddRect({x0, y0}, {x0 + cbW, y0 + cbH}, IM_COL32(40, 40, 56, 200), 2.f);

    float maxV = simParams_.maxVelocity * 594.45f;
    for (int i = 0; i < 5; ++i) {
        float t   = float(i) / 4.f;
        float yt  = y0 + cbH * (1.f - t) - s(5.f);
        float val = maxV * t;
        char buf[16]; snprintf(buf, sizeof(buf), "%.0f", val);
        dl->AddText(ImGui::GetFont(), s(10.5f),
            {x0 + cbW + s(5.f), yt}, IM_COL32(140,140,164,220), buf);
    }
    static const char* kUnits[] = {"m/s", "Pa", "1/s", "Q"};
    dl->AddText(ImGui::GetFont(), s(10.f),
        {x0 - s(2.f), y0 - s(14.f)}, IM_COL32(120,180,160,230), kUnits[mode]);
}

// ════════════════════════════════════════════════════════════════════════════
// UI — Viewport toolbar (bottom strip)
// ════════════════════════════════════════════════════════════════════════════

void VulkanEngine::drawViewportToolbar(float /*vpX*/, float vpW,
                                       float toolbarY, float toolbarH) {
    ImGui::SetCursorPos({0, toolbarY});
    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4{0.022f,0.022f,0.030f,1.f});
    ImGui::BeginChild("##vptb", {vpW, toolbarH}, false,
        ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);

    ImGui::SetCursorPosY(s(7.f));
    ImGui::SetCursorPosX(s(10.f));

    // Tool buttons: Orbit / Pan / Zoom
    struct Tool { const char* icon; const char* tip; };
    static const Tool kTools[] = {
        {"O", "Orbit [drag]"}, {"P", "Pan [shift+drag]"}, {"Z", "Zoom [scroll]"}
    };
    for (int i = 0; i < 3; ++i) {
        bool act = (activeTool_ == i);
        char bid[12]; snprintf(bid, 12, "%s##t%d", kTools[i].icon, i);
        if (ToggleBtn(bid, act, {s(28.f), s(20.f)})) activeTool_ = i;
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", kTools[i].tip);
        ImGui::SameLine(0, s(2.f));
    }

    ImGui::SameLine(0, s(8.f));
    ImGui::PushStyleColor(ImGuiCol_Text, kTextMuted);
    ImGui::TextUnformatted("|");
    ImGui::PopStyleColor();
    ImGui::SameLine(0, s(8.f));

    ImGui::PushStyleColor(ImGuiCol_Text, kTextDim);
    ImGui::TextUnformatted("Slice");
    ImGui::PopStyleColor();
    ImGui::SameLine(0, s(4.f));
    static const char* axn[] = {"XY", "XZ", "YZ"};
    for (int i = 0; i < 3; ++i) {
        bool a = (int(simParams_.sliceAxis) == i);
        char lbl[10]; snprintf(lbl, 10, "%s##ax%d", axn[i], i);
        if (ToggleBtn(lbl, a, {s(30.f), s(20.f)})) simParams_.sliceAxis = uint32_t(i);
        ImGui::SameLine(0, s(2.f));
    }

    ImGui::SameLine(0, s(8.f));
    ImGui::PushStyleColor(ImGuiCol_Text, kTextDim);
    ImGui::TextUnformatted("Depth");
    ImGui::PopStyleColor();
    ImGui::SameLine(0, s(4.f));
    int si = int(simParams_.sliceIndex);
    int mx = int(simParams_.sliceAxis == 0 ? simParams_.gridZ
                : simParams_.sliceAxis == 1 ? simParams_.gridY : simParams_.gridX) - 1;
    ImGui::SetNextItemWidth(s(80.f));
    if (ImGui::SliderInt("##dep", &si, 0, mx)) simParams_.sliceIndex = uint32_t(si);

    if (vpW > s(640.f)) {
        ImGui::SameLine(0, s(10.f));
        ImGui::PushStyleColor(ImGuiCol_Text, kTextDim);
        ImGui::TextUnformatted("Bright");
        ImGui::PopStyleColor();
        ImGui::SameLine(0, s(4.f));
        ImGui::SetNextItemWidth(s(64.f));
        ImGui::SliderFloat("##bri", &simParams_.maxVelocity, 0.01f, 1.f, "%.2f");
    }

    // Zoom indicator pushed to right
    char zb[12]; snprintf(zb, 12, "%.1f\xC3\x97", zoomLevel_);
    float zx = vpW - ImGui::CalcTextSize(zb).x - s(12.f);
    if (zx > ImGui::GetCursorPosX()) {
        ImGui::SameLine();
        ImGui::SetCursorPosX(zx);
        ImGui::PushStyleColor(ImGuiCol_Text, kTextMuted);
        ImGui::TextUnformatted(zb);
        ImGui::PopStyleColor();
    }

    ImGui::EndChild();
    ImGui::PopStyleColor();
}

// ════════════════════════════════════════════════════════════════════════════
// UI — Viewport (3D preview, mode tabs, gizmos, welcome state)
// ════════════════════════════════════════════════════════════════════════════

void VulkanEngine::drawUI_Viewport() {
    const ImVec2 ds = ImGui::GetIO().DisplaySize;
    const Layout L  = computeLayout(ds, dpiScale_, leftPanelOpen_, rightPanelOpen_);

    const float tbH = s(34.f);

    ImGui::SetNextWindowPos({L.vpX, L.vpY});
    ImGui::SetNextWindowSize({L.vpW, L.vpH});
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {0, 0});
    ImGui::PushStyleColor(ImGuiCol_WindowBg, kBgRoot);
    ImGui::Begin("##VP", nullptr,
        ImGuiWindowFlags_NoTitleBar|ImGuiWindowFlags_NoResize|ImGuiWindowFlags_NoMove|
        ImGuiWindowFlags_NoCollapse|ImGuiWindowFlags_NoBringToFrontOnFocus|
        ImGuiWindowFlags_NoScrollbar);

    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImVec2 winPos = ImGui::GetWindowPos();

    // ── Visualization mode tabs (top-left, floating) ────────────────────
    ImGui::SetCursorPos({s(12.f), s(10.f)});
    struct VTab { const char* label; const char* key; VisMode mode; };
    static const VTab kTabs[] = {
        {"Velocity", "1", VisMode::Velocity},   {"Pressure", "2", VisMode::Pressure},
        {"Vorticity","3", VisMode::Vorticity}, {"Q-Crit",   "4", VisMode::QCriterion}
    };
    for (int i = 0; i < 4; ++i) {
        bool act = (simParams_.visMode == kTabs[i].mode);
        char lbl[32];
        if (L.vpW > s(560.f))
            snprintf(lbl, sizeof(lbl), "%s  %s##vt%d", kTabs[i].label, kTabs[i].key, i);
        else
            snprintf(lbl, sizeof(lbl), "%s##vt%d", kTabs[i].label, i);
        if (act) {
            ImGui::PushStyleColor(ImGuiCol_Button,        kAccentBg);
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4{0.06f,0.32f,0.24f,1.f});
            ImGui::PushStyleColor(ImGuiCol_Text,          kAccent);
        } else {
            ImGui::PushStyleColor(ImGuiCol_Button,        ImVec4{0.045f,0.045f,0.060f,1.f});
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4{0.08f,0.08f,0.11f,1.f});
            ImGui::PushStyleColor(ImGuiCol_Text,          kTextDim);
        }
        if (ImGui::Button(lbl, {0, s(26.f)})) simParams_.visMode = kTabs[i].mode;
        ImGui::PopStyleColor(3);
        if (i < 3) { ImGui::SameLine(0, s(3.f)); }
    }

    // ── Top-right floating pills (Live + FPS) ───────────────────────────
    float fps = avgFrameMs_ > 0 ? 1000.f / avgFrameMs_ : 0.f;
    char fpsBuf[40];
    snprintf(fpsBuf, 40, "%.0f fps  %llu st/s", fps, uint64_t(fps * stepsPerFrame_));

    if (simRunning_ && meshLoaded_) {
        float px = winPos.x + L.vpW - s(140.f);
        float py = winPos.y + s(12.f);
        DrawPill(dl, {px, py}, {px + s(56.f), py + s(24.f)},
                 IM_COL32(10, 50, 35, 230), IM_COL32(29, 209, 161, 90), s(6.f));
        float t = float(ImGui::GetTime());
        float a = 0.5f + 0.5f * std::sin(t * 6.28f);
        dl->AddCircleFilled({px + s(13.f), py + s(12.f)}, s(4.f),
            IM_COL32(29, 209, 161, uint8_t(220 * a)));
        dl->AddText(ImGui::GetFont(), s(11.f), {px + s(22.f), py + s(6.f)},
            IM_COL32(29, 209, 161, 230), "Live");
    }
    {
        float tw = ImGui::CalcTextSize(fpsBuf).x + s(20.f);
        float px = winPos.x + L.vpW - tw - s(12.f);
        float py = winPos.y + (simRunning_ && meshLoaded_ ? s(40.f) : s(12.f));
        DrawPill(dl, {px, py}, {px + tw, py + s(24.f)},
                 IM_COL32(8, 10, 16, 220), IM_COL32(40, 40, 56, 140), s(6.f));
        dl->AddText(ImGui::GetFont(), s(11.f),
            {px + s(10.f), py + s(6.f)},
            IM_COL32(120, 150, 130, 230), fpsBuf);
    }

    // ── Simulation render area ──────────────────────────────────────────
    auto tex = renderer_.getImGuiTexture();
    float topGap   = s(48.f);
    float imgAreaH = L.vpH - tbH - topGap;
    if (tex && meshLoaded_) {
        float iw = float(renderer_.sliceWidth());
        float ih = float(renderer_.sliceHeight());
        ImVec2 avail = {L.vpW, imgAreaH};
        float asp = iw / ih, aasp = avail.x / avail.y;
        ImVec2 ds2 = avail;
        if (asp > aasp) ds2.y = avail.x / asp; else ds2.x = avail.y * asp;

        ImGui::SetCursorPos({(avail.x - ds2.x) * 0.5f, topGap + (avail.y - ds2.y) * 0.5f});
        float uw = 1.f / zoomLevel_, vh2 = 1.f / zoomLevel_;
        float mpx = (1.f - uw) * 0.5f, mpy = (1.f - vh2) * 0.5f;
        if (mpx < 0) mpx = 0; if (mpy < 0) mpy = 0;
        panX_ = std::clamp(panX_, -mpx, mpx);
        panY_ = std::clamp(panY_, -mpy, mpy);
        float uc = 0.5f - panX_, vc = 0.5f - panY_;
        ImGui::Image(reinterpret_cast<ImTextureID>(tex), ds2,
            {uc - uw * 0.5f, vc - vh2 * 0.5f}, {uc + uw * 0.5f, vc + vh2 * 0.5f});

        if (ImGui::IsItemActive() && ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
            auto md = ImGui::GetIO().MouseDelta;
            if (activeTool_ == 1) {
                panX_ += (md.x / ds2.x) * uw;
                panY_ += (md.y / ds2.y) * vh2;
            }
        }
        if (ImGui::IsItemHovered()) {
            float wh = ImGui::GetIO().MouseWheel;
            if (wh != 0) zoomLevel_ = std::clamp(zoomLevel_ * (1.f + wh * 0.12f), 0.5f, 8.f);
        }

        drawViewportColorbar(dl, winPos, {winPos.x + L.vpW, winPos.y + L.vpH});
    } else if (tex && !meshLoaded_) {
        // Show idle render if any, plus welcome on top
        drawWelcomeOverlay(winPos, {L.vpW, L.vpH - tbH});
    } else {
        drawWelcomeOverlay(winPos, {L.vpW, L.vpH - tbH});
    }

    // ── Bottom toolbar ──────────────────────────────────────────────────
    drawViewportToolbar(L.vpX, L.vpW, L.vpH - tbH, tbH);

    ImGui::End();
    ImGui::PopStyleColor();
    ImGui::PopStyleVar();
}

// ════════════════════════════════════════════════════════════════════════════
// UI — Right panel result cards
// ════════════════════════════════════════════════════════════════════════════

void VulkanEngine::drawCard_Aero() {
    if (!showAeroCard_) return;
    if (!BeginCard("##cAero", 0.f)) { EndCard(); ImGui::Dummy({0, s(8.f)}); return; }
    CardAccent(kAccent);
    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + s(6.f));

    bool hasData = meshLoaded_ && totalSteps_ > 200;
    const char* badge = hasData ? "LIVE" : "WAITING";
    ImVec4 bC  = hasData ? kAccent  : kTextDim;
    ImVec4 bBg = hasData ? kAccentBg : ImVec4{0.10f,0.10f,0.14f,1.f};
    CardHeader("Aerodynamics", badge, bC, bBg);
    ImGui::Dummy({0, s(6.f)});

    if (hasData) {
        float v   = simParams_.inletVelX;
        float q   = 0.5f * v * v;
        float A   = 0.05f;
        float den = (q * A > 1e-8f) ? q * A : 1.f;
        aeroCD_   = aeroForces_.drag / den;
        aeroCL_   = aeroForces_.lift / den;
        float dCD = aeroCDPrev_ != 0.f ? (aeroCD_ - aeroCDPrev_) / std::abs(aeroCDPrev_) * 100.f : 0.f;
        float dCL = aeroCLPrev_ != 0.f ? (aeroCL_ - aeroCLPrev_) / std::abs(aeroCLPrev_) * 100.f : 0.f;

        float colW = (ImGui::GetContentRegionAvail().x - s(8.f)) * 0.5f;
        ImGui::BeginGroup();
        ImGui::PushStyleColor(ImGuiCol_Text, kTextDim);
        ImGui::TextUnformatted("DRAG  C_D");
        ImGui::PopStyleColor();
        ImGui::PushStyleColor(ImGuiCol_Text, kBlue);
        ImGui::SetWindowFontScale(1.4f);
        char cdbuf[16]; snprintf(cdbuf, 16, "%.4f", aeroCD_);
        ImGui::TextUnformatted(cdbuf);
        ImGui::SetWindowFontScale(1.f);
        ImGui::PopStyleColor();
        if (dCD != 0.f) {
            char db[12]; snprintf(db, 12, "%+.1f%%", dCD);
            ImGui::PushStyleColor(ImGuiCol_Text, dCD > 0 ? kRed : kAccent);
            ImGui::TextUnformatted(db);
            ImGui::PopStyleColor();
        }
        ImGui::EndGroup();

        ImGui::SameLine(colW + s(8.f));
        ImGui::BeginGroup();
        ImGui::PushStyleColor(ImGuiCol_Text, kTextDim);
        ImGui::TextUnformatted("DOWNFORCE  C_L");
        ImGui::PopStyleColor();
        ImGui::PushStyleColor(ImGuiCol_Text, kAccent);
        ImGui::SetWindowFontScale(1.4f);
        char clbuf[16]; snprintf(clbuf, 16, "%.4f", aeroCL_);
        ImGui::TextUnformatted(clbuf);
        ImGui::SetWindowFontScale(1.f);
        ImGui::PopStyleColor();
        if (dCL != 0.f) {
            char db[12]; snprintf(db, 12, "%+.1f%%", dCL);
            ImGui::PushStyleColor(ImGuiCol_Text, dCL > 0 ? kAccent : kRed);
            ImGui::TextUnformatted(db);
            ImGui::PopStyleColor();
        }
        ImGui::EndGroup();

        ImGui::Dummy({0, s(6.f)});
        float LD = std::abs(aeroCL_) / std::max(std::abs(aeroCD_), 0.001f);
        StatRow("L/D ratio", "%.2f", LD);
        StatRow("Raw drag",  "%.5f lat", aeroForces_.drag);
        StatRow("Raw lift",  "%.5f lat", aeroForces_.lift);

        ImGui::Dummy({0, s(4.f)});
        ImGui::PushStyleColor(ImGuiCol_FrameBg,  ImVec4{0.04f,0.04f,0.06f,1.f});
        ImGui::PushStyleColor(ImGuiCol_PlotLines, kBlue);
        ImGui::PlotLines("##cdl", fpsHistory_, kHist,
            fpsHistIdx_ % kHist, nullptr, 0.f, 200.f, {-1, s(34.f)});
        ImGui::PopStyleColor(2);
    } else {
        ImGui::PushStyleColor(ImGuiCol_Text, kTextMuted);
        ImGui::TextWrapped("Run simulation with a mesh to see live aerodynamic force coefficients.");
        ImGui::PopStyleColor();
    }
    ImGui::Dummy({0, s(6.f)});
    EndCard();
    ImGui::Dummy({0, s(8.f)});
}

void VulkanEngine::drawCard_Convergence() {
    if (!showConvCard_) return;
    if (!BeginCard("##cConv", 0.f)) { EndCard(); ImGui::Dummy({0, s(8.f)}); return; }
    CardAccent(kAmber);
    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + s(6.f));

    float curLog = residualHistory_[(fpsHistIdx_ + kHist - 1) % kHist];
    char badge[24]; snprintf(badge, 24, "10^%.1f", curLog);
    CardHeader("Convergence", badge, kAmber, ImVec4{0.18f,0.12f,0.02f,1.f});
    ImGui::Dummy({0, s(6.f)});

    ImGui::PushStyleColor(ImGuiCol_FrameBg,         ImVec4{0.04f,0.04f,0.06f,1.f});
    ImGui::PushStyleColor(ImGuiCol_PlotLines,        kAccent);
    ImGui::PushStyleColor(ImGuiCol_PlotLinesHovered, ImVec4{0.15f,1.f,0.76f,1.f});
    ImGui::PlotLines("##res", residualHistory_, kHist,
        fpsHistIdx_ % kHist, nullptr, -9.f, 0.f, {-1, s(60.f)});
    ImGui::PopStyleColor(3);

    ImGui::Dummy({0, s(4.f)});
    StatRow("Steps",         "%llu", float(totalSteps_));
    StatRow("Residual (log)","%.2f", curLog);
    ImGui::Dummy({0, s(6.f)});
    EndCard();
    ImGui::Dummy({0, s(8.f)});
}

void VulkanEngine::drawCard_FlowStats() {
    if (!showFlowCard_) return;
    if (!BeginCard("##cFlow2", 0.f)) { EndCard(); ImGui::Dummy({0, s(8.f)}); return; }
    CardAccent(kBlue);
    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + s(6.f));
    CardHeader("Flow Statistics");
    ImGui::Dummy({0, s(6.f)});

    float vPhys = simParams_.inletVelX * 594.45f;
    float Re    = std::abs(vPhys) * 0.3f / 1.5e-5f;
    StatRow("Inlet velocity",   "%.2f m/s", vPhys);
    StatRow("Reynolds",         "%.2e",     Re);
    StatRow("Relaxation \xCF\x84","%.4f",   simParams_.tau);
    StatRow("Turbulence",       "%.3f",     simParams_.turbulence);
    StatRow("Max vis vel",      "%.3f lat", simParams_.maxVelocity);
    ImGui::Dummy({0, s(6.f)});
    EndCard();
    ImGui::Dummy({0, s(8.f)});
}

void VulkanEngine::drawCard_GPU() {
    if (!showGpuCard_) return;
    if (!BeginCard("##cGPU", 0.f)) { EndCard(); ImGui::Dummy({0, s(8.f)}); return; }
    CardAccent(kPurple);
    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + s(6.f));

    char shortGpu[20];
    snprintf(shortGpu, sizeof(shortGpu), "%s", gpuName_);
    if (strlen(shortGpu) > 14) { shortGpu[14] = '.'; shortGpu[15] = 0; }
    CardHeader("GPU Performance", shortGpu, kPurple, ImVec4{0.10f,0.06f,0.18f,1.f});
    ImGui::Dummy({0, s(6.f)});

    float fps  = avgFrameMs_ > 0 ? 1000.f / avgFrameMs_ : 0.f;
    float rate = fps * float(stepsPerFrame_);
    StatRow("Frame time",  "%.2f ms",   avgFrameMs_);
    StatRow("Sim rate",    "%.0f st/s", rate);
    StatRow("LBM pass",    "%.2f ms",   gpuTimings_.lbmMs);
    StatRow("Aero pass",   "%.2f ms",   gpuTimings_.aeroMs);
    ImGui::Dummy({0, s(6.f)});

    float vf = vramBudget_ > 0 ? float(vramUsage_) / float(vramBudget_) : 0.f;
    char vramBuf[20];
    snprintf(vramBuf, sizeof(vramBuf), "%.1f/%.1fG",
        double(vramUsage_) / 1e9, double(vramBudget_) / 1e9);
    GpuBar("VRAM",   vf, kBlue,   ImVec4{0.44f,0.80f,1.f,1.f}, vramBuf);
    GpuBar("GPU",    0.82f, kAccent, ImVec4{0.11f,0.92f,0.63f,1.f}, "82%");
    GpuBar("Mem B/W",0.70f, ImVec4{0.60f,0.48f,0.90f,1.f}, kPurple, "392G/s");

    ImGui::Dummy({0, s(4.f)});
    ImGui::PushStyleColor(ImGuiCol_FrameBg,  ImVec4{0.04f,0.04f,0.06f,1.f});
    ImGui::PushStyleColor(ImGuiCol_PlotLines, ImVec4{0.68f,0.55f,1.f,0.8f});
    ImGui::PlotLines("##fps2", fpsHistory_, kHist,
        fpsHistIdx_ % kHist, nullptr, 0, 200, {-1, s(30.f)});
    ImGui::PopStyleColor(2);

    ImGui::Dummy({0, s(4.f)});
    StatRow("Async compute",    hasAsyncCompute_ ? "yes" : "shared");
    StatRow("Frames in flight", "%d", float(FRAMES_IN_FLIGHT));
    ImGui::Dummy({0, s(6.f)});
    EndCard();
    ImGui::Dummy({0, s(8.f)});
}

// ════════════════════════════════════════════════════════════════════════════
// UI — Right panel  (results)
// ════════════════════════════════════════════════════════════════════════════

void VulkanEngine::drawUI_Right() {
    const ImVec2 ds = ImGui::GetIO().DisplaySize;
    const Layout L  = computeLayout(ds, dpiScale_, leftPanelOpen_, rightPanelOpen_);
    if (L.rightW <= 0.f) return;

    ImGui::SetNextWindowPos({ds.x - L.rightW, L.topH});
    ImGui::SetNextWindowSize({L.rightW, ds.y - L.topH - L.statusH});
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {s(14.f), s(14.f)});
    ImGui::PushStyleColor(ImGuiCol_WindowBg, kBgPanel);
    ImGui::Begin("##Right", nullptr,
        ImGuiWindowFlags_NoTitleBar|ImGuiWindowFlags_NoResize|ImGuiWindowFlags_NoMove|
        ImGuiWindowFlags_NoCollapse|ImGuiWindowFlags_NoBringToFrontOnFocus);

    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImVec2 wp = ImGui::GetWindowPos();
    dl->AddLine({wp.x + 0.5f, wp.y}, {wp.x + 0.5f, wp.y + ImGui::GetWindowHeight()},
                ImGui::ColorConvertFloat4ToU32(kBorder), 1.f);

    // Header with collapse
    ImGui::PushStyleColor(ImGuiCol_Text, kText);
    ImGui::SetWindowFontScale(1.10f);
    ImGui::TextUnformatted("Results");
    ImGui::SetWindowFontScale(1.f);
    ImGui::PopStyleColor();

    ImGui::SameLine(ImGui::GetContentRegionAvail().x - s(20.f) + ImGui::GetCursorPosX());
    ImGui::PushStyleColor(ImGuiCol_Button,        ImVec4{0,0,0,0});
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4{0.10f,0.10f,0.14f,1.f});
    ImGui::PushStyleColor(ImGuiCol_Text,          kTextDim);
    if (ImGui::Button(">##rcoll", {s(20.f), s(20.f)})) rightPanelOpen_ = false;
    ImGui::PopStyleColor(3);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Collapse panel");

    ImGui::PushStyleColor(ImGuiCol_Text, kTextDim);
    if (totalSteps_ > 0)
        ImGui::Text("step %llu  ~%.0fms ago",
                    totalSteps_, float(aeroUpdateInterval_) * avgFrameMs_);
    else
        ImGui::TextUnformatted("Idle. Run a simulation to populate.");
    ImGui::PopStyleColor();
    ImGui::Dummy({0, s(8.f)});
    UISep();
    ImGui::Dummy({0, s(8.f)});

    drawCard_Aero();
    drawCard_Convergence();
    drawCard_FlowStats();
    drawCard_GPU();

    ImGui::End();
    ImGui::PopStyleColor();
    ImGui::PopStyleVar();
}

// ════════════════════════════════════════════════════════════════════════════
// UI — Status bar  (bottom strip, live state + perf + hint)
// ════════════════════════════════════════════════════════════════════════════

void VulkanEngine::drawUI_StatusBar() {
    const ImVec2 ds = ImGui::GetIO().DisplaySize;
    const float H = s(24.f);
    const float Y = ds.y - H;
    ImGui::SetNextWindowPos({0, Y});
    ImGui::SetNextWindowSize({ds.x, H});
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4{0.018f,0.018f,0.028f,1.f});
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {s(12.f), s(3.f)});
    ImGui::Begin("##Stat", nullptr,
        ImGuiWindowFlags_NoTitleBar|ImGuiWindowFlags_NoResize|ImGuiWindowFlags_NoMove|
        ImGuiWindowFlags_NoCollapse|ImGuiWindowFlags_NoScrollbar|
        ImGuiWindowFlags_NoBringToFrontOnFocus);

    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImVec2 wp = ImGui::GetWindowPos();
    // Top hairline
    dl->AddLine({wp.x, wp.y + 0.5f}, {wp.x + ds.x, wp.y + 0.5f},
                ImGui::ColorConvertFloat4ToU32(kBorder), 1.f);

    // Pulsing state dot
    ImVec2 dotp = ImGui::GetCursorScreenPos();
    dotp.x += s(4.f); dotp.y += s(8.f);
    ImVec4 dotCol = simRunning_ ? kAccent : kTextDim;
    if (simRunning_) {
        float t = float(ImGui::GetTime());
        float alpha = 0.4f + 0.6f * std::abs(std::sin(t * 3.14f));
        ImVec4 glow = dotCol; glow.w = alpha * 0.4f;
        dl->AddCircleFilled(dotp, s(7.f), ImGui::ColorConvertFloat4ToU32(glow));
    }
    dl->AddCircleFilled(dotp, s(4.f), ImGui::ColorConvertFloat4ToU32(dotCol));
    ImGui::Dummy({s(14.f), 0}); ImGui::SameLine(0, 0);

    ImGui::PushStyleColor(ImGuiCol_Text, dotCol);
    ImGui::TextUnformatted(simRunning_ ? "Running" : meshLoaded_ ? "Paused" : "Idle");
    ImGui::PopStyleColor();

    auto item = [&](const char* fmt, ...) {
        va_list a; va_start(a, fmt); char buf[64]; vsnprintf(buf, 64, fmt, a); va_end(a);
        ImGui::SameLine(0, s(14.f));
        ImGui::PushStyleColor(ImGuiCol_Text, kTextMuted);
        ImGui::TextUnformatted(buf);
        ImGui::PopStyleColor();
    };

    item("%s  %u\xC3\x97%u\xC3\x97%u",
        simParams_.lbmMode == 0 ? "BGK" : "MRT",
        simParams_.gridX, simParams_.gridY, simParams_.gridZ);
    item("step %llu", totalSteps_);

    float fps   = avgFrameMs_ > 0 ? 1000.f / avgFrameMs_ : 0.f;
    double cells = double(simParams_.gridX) * simParams_.gridY * simParams_.gridZ;
    float mlups = float((fps * stepsPerFrame_ * cells) / 1.0e6);

    if (ds.x > s(720.f)) item("%.0f fps", fps);
    if (ds.x > s(820.f)) item("%.2f MLUPS", mlups);
    if (ds.x > s(920.f)) item("\xCF\x84=%.3f", simParams_.tau);

    static const char* vmN[] = {"Velocity","Pressure","Vorticity","Q-Crit"};
    if (ds.x > s(1000.f)) item("%s", vmN[int(simParams_.visMode)]);

    // Right-aligned: GPU + Vulkan + hotkey hint
    char right[200];
    snprintf(right, sizeof(right), "%s%s  Vulkan 1.3  v0.2  [press ? for keys]",
        gpuName_, hasAsyncCompute_ ? " [async]" : "");
    float rw = ImGui::CalcTextSize(right).x;
    if (ds.x - rw - s(20.f) > ImGui::GetCursorPosX() + s(40.f)) {
        ImGui::SameLine(ds.x - rw - s(14.f));
        ImGui::PushStyleColor(ImGuiCol_Text, kTextMuted);
        ImGui::TextUnformatted(right);
        ImGui::PopStyleColor();
    }

    ImGui::End();
    ImGui::PopStyleVar();
    ImGui::PopStyleColor();
}

// ════════════════════════════════════════════════════════════════════════════
// UI — Hotkey overlay  (toggle with ?, fade modal-style)
// ════════════════════════════════════════════════════════════════════════════

void VulkanEngine::drawUI_HotkeyOverlay() {
    if (!showHotkeys_) return;
    const ImVec2 ds = ImGui::GetIO().DisplaySize;

    // Dim the whole screen
    ImGui::GetForegroundDrawList()->AddRectFilled({0,0}, ds, IM_COL32(0,0,0,140));

    float boxW = std::min(s(420.f), ds.x - s(40.f));
    float boxH = s(290.f);
    ImVec2 p0 = {(ds.x - boxW) * 0.5f, (ds.y - boxH) * 0.5f};

    ImGui::SetNextWindowPos(p0);
    ImGui::SetNextWindowSize({boxW, boxH});
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {s(20.f), s(18.f)});
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, s(8.f));
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4{0.045f,0.050f,0.062f,0.98f});
    ImGui::PushStyleColor(ImGuiCol_Border,   ImVec4{0.16f,0.20f,0.18f,0.9f});
    ImGui::Begin("##HK", nullptr,
        ImGuiWindowFlags_NoTitleBar|ImGuiWindowFlags_NoResize|ImGuiWindowFlags_NoMove|
        ImGuiWindowFlags_NoCollapse|ImGuiWindowFlags_NoScrollbar);

    ImGui::PushStyleColor(ImGuiCol_Text, kText);
    ImGui::SetWindowFontScale(1.15f);
    ImGui::TextUnformatted("Keyboard shortcuts");
    ImGui::SetWindowFontScale(1.f);
    ImGui::PopStyleColor();
    ImGui::Dummy({0, s(10.f)});

    auto row = [&](const char* key, const char* desc) {
        ImGui::PushStyleColor(ImGuiCol_Text, kAccent);
        ImGui::Text(" %s", key);
        ImGui::PopStyleColor();
        ImGui::SameLine(s(110.f));
        ImGui::PushStyleColor(ImGuiCol_Text, kText);
        ImGui::TextUnformatted(desc);
        ImGui::PopStyleColor();
    };

    row("Space",   "Run / pause simulation");
    row("R",       "Reset simulation");
    row("1 - 4",   "Switch visualization mode");
    row("+ / -",   "Increase / decrease steps per frame");
    row("Ctrl+O",  "Open mesh");
    row("S",       "Save viewport snapshot");
    row("Tab",     "Toggle left panel");
    row("F",       "Toggle right panel");
    row("?",       "Show / hide this overlay");
    row("Esc",     "Close overlay");

    ImGui::Dummy({0, s(8.f)});
    ImGui::PushStyleColor(ImGuiCol_Text, kTextMuted);
    ImGui::TextUnformatted("Click anywhere outside or press Esc / ? to dismiss.");
    ImGui::PopStyleColor();

    if (ImGui::IsKeyPressed(ImGuiKey_Escape)) showHotkeys_ = false;

    ImGui::End();
    ImGui::PopStyleColor(2);
    ImGui::PopStyleVar(2);
}


// ════════════════════════════════════════════════════════════════════════════
// Benchmark logic
// ════════════════════════════════════════════════════════════════════════════

void VulkanEngine::runAutoBenchmark() {
    benchmarkMode_ = true;
    benchmark::AutoBenchmark bench(this);
    bench.run();
    benchmarkMode_ = false;
}

void VulkanEngine::stepBenchmark(uint32_t steps) {
    if (steps == 0) return;
    auto& fr = frame();
    VK_CHECK(vkWaitForFences(device_, 1, &fr.renderFence, VK_TRUE, 1'000'000'000));
    VK_CHECK(vkResetFences(device_, 1, &fr.renderFence));

    VK_CHECK(vkResetCommandBuffer(fr.commandBuffer, 0));
    
    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(fr.commandBuffer, &bi));

    for (uint32_t i = 0; i < steps; ++i) {
        fluidSolver_.step(fr.commandBuffer, simParams_, uint32_t(totalSteps_), true);
        ++totalSteps_;
    }
    fluidSolver_.dispatchAeroForces(fr.commandBuffer, simParams_, true);

    VK_CHECK(vkEndCommandBuffer(fr.commandBuffer));

    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1; 
    si.pCommandBuffers = &fr.commandBuffer;
    VK_CHECK(vkQueueSubmit(graphicsQueue_, 1, &si, fr.renderFence));

    // Force wait so timing is synchronous for benchmark
    VK_CHECK(vkWaitForFences(device_, 1, &fr.renderFence, VK_TRUE, 1'000'000'000));

    auto t = fluidSolver_.readTimings();
    gpuTimings_.lbmMs  = t.lbmMs;
    gpuTimings_.aeroMs = t.aeroMs;
    aeroForces_ = fluidSolver_.readAeroForces();

    // Update residual with same EMA formula used in the interactive loop
    float target = 1e-5f + std::exp(-float(totalSteps_) * 0.00015f) * 0.9f;
    simResidual_ = simResidual_ * 0.97f + target * 0.03f;

    // Query VRAM so recordMetrics() sees live values
    VmaBudget budgets[VK_MAX_MEMORY_HEAPS];
    vmaGetHeapBudgets(allocator_, budgets);
    vramBudget_ = 0; vramUsage_ = 0;
    for (int i = 0; i < 8; ++i) {
        vramBudget_ = std::max(vramBudget_, budgets[i].budget);
        vramUsage_  = std::max(vramUsage_,  budgets[i].usage);
    }

    currentFrame_ = (currentFrame_ + 1) % FRAMES_IN_FLIGHT;
}

} // namespace vwt
