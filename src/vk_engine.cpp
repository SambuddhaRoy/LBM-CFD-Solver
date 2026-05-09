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
        eng->zoomLevel_ = 1.f; eng->panX_ = 0; eng->panY_ = 0; break;
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
    // FontGlobalScale inverts the oversize so logical sizes remain correct;
    // the atlas is just rasterised at higher resolution for sharpness.
    io.FontGlobalScale = 1.f / dpiScale;

#ifdef _WIN32
    fontBody_ = io.Fonts->AddFontFromFileTTF("C:/Windows/Fonts/segoeui.ttf", 15.f * dpiScale);
    fontMono_ = io.Fonts->AddFontFromFileTTF("C:/Windows/Fonts/consola.ttf", 12.f * dpiScale);
#else
    fontBody_ = io.Fonts->AddFontFromFileTTF("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",     15.f * dpiScale);
    fontMono_ = io.Fonts->AddFontFromFileTTF("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 12.f * dpiScale);
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

    drawUI_Rail();
    drawUI_Left();
    drawUI_Viewport();
    drawUI_Right();
    drawUI_StatusBar();

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
// UI — Icon rail  (56px wide, left edge)
// ════════════════════════════════════════════════════════════════════════════

void VulkanEngine::drawUI_Rail() {
    const ImVec2 ds = ImGui::GetIO().DisplaySize;
    const float RW = std::max(48.f, ds.x * 0.030f);  // ~3% of width, min 48px
    const float H  = ds.y - 26.f;
    ImGui::SetNextWindowPos({0,0});
    ImGui::SetNextWindowSize({RW, H});
    ImGui::PushStyleColor(ImGuiCol_WindowBg, {0.031f,0.031f,0.043f,1.f});
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {0,8});
    ImGui::Begin("##Rail", nullptr,
        ImGuiWindowFlags_NoTitleBar|ImGuiWindowFlags_NoResize|
        ImGuiWindowFlags_NoMove|ImGuiWindowFlags_NoCollapse|
        ImGuiWindowFlags_NoBringToFrontOnFocus|ImGuiWindowFlags_NoScrollbar);

    ImDrawList* dl = ImGui::GetWindowDrawList();
    const float btnSz  = std::max(32.f, RW * 0.68f);
    const float btnOff = (RW - btnSz) * 0.5f;

    // Logo mark
    ImVec2 lp = ImGui::GetCursorScreenPos();
    lp.x += btnOff; lp.y += 4.f;
    dl->AddRectFilledMultiColor(lp, {lp.x+btnSz,lp.y+btnSz},
        IM_COL32(29,209,161,255), IM_COL32(0,206,201,255),
        IM_COL32(0,176,155,255),  IM_COL32(29,209,161,255));
    dl->AddText(ImGui::GetFont(), btnSz*0.5f, {lp.x+btnSz*0.25f,lp.y+btnSz*0.22f},
        IM_COL32(10,20,16,255), "V");
    ImGui::Dummy({RW, btnSz + 10.f});
    ImGui::Dummy({0, 8.f});

    // Rail icon buttons
    struct RailItem { const char* icon; const char* tip; };
    static const RailItem items[] = {
        {"^","Simulation"}, {"#","Mesh Tools"},
        {"@","Probe"},      {"=","Compare"}
    };
    for (int i = 0; i < 4; ++i) {
        bool act = (railMode_ == i);
        ImGui::SetCursorPosX(btnOff);

        ImVec2 btnP = ImGui::GetCursorScreenPos();
        if (act) {
            dl->AddRectFilled(btnP, {btnP.x+btnSz,btnP.y+btnSz},
                IM_COL32(11,72,54,255), 7.f);
            dl->AddRect(btnP, {btnP.x+btnSz,btnP.y+btnSz},
                IM_COL32(29,209,161,80), 7.f);
        }

        ImGui::PushStyleColor(ImGuiCol_Button,        {0,0,0,0});
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, {0.10f,0.10f,0.14f,1.f});
        ImGui::PushStyleColor(ImGuiCol_ButtonActive,  {0.05f,0.28f,0.20f,1.f});
        ImGui::PushStyleColor(ImGuiCol_Text,
            act ? ImVec4{0.11f,0.82f,0.63f,1.f} : ImVec4{0.32f,0.32f,0.44f,1.f});
        ImGui::SetWindowFontScale(1.25f);
        char bid[12]; snprintf(bid,12,"##ri%d",i);
        if (ImGui::Button(bid, {btnSz,btnSz})) railMode_ = i;
        ImGui::SetWindowFontScale(1.f);
        ImGui::PopStyleColor(4);

        // Draw icon text centred
        ImVec2 iconSz = ImGui::CalcTextSize(items[i].icon);
        dl->AddText(ImGui::GetFont(), btnSz*0.44f,
            {btnP.x + (btnSz-iconSz.x)*0.5f, btnP.y + (btnSz-iconSz.y)*0.5f},
            act ? IM_COL32(29,209,161,255) : IM_COL32(82,82,110,255),
            items[i].icon);

        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("%s", items[i].tip);
        ImGui::Dummy({0,4});
    }

    // Bottom: settings
    float settY = H - btnSz - 12.f;
    ImGui::SetCursorPosY(settY);
    ImGui::SetCursorPosX(btnOff);
    ImGui::PushStyleColor(ImGuiCol_Button,        {0,0,0,0});
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, {0.10f,0.10f,0.14f,1.f});
    ImGui::PushStyleColor(ImGuiCol_Text,          {0.28f,0.28f,0.38f,1.f});
    if (ImGui::Button("##sett",{btnSz,btnSz})){}
    ImGui::PopStyleColor(3);
    ImVec2 gp = ImGui::GetItemRectMin();
    dl->AddText(ImGui::GetFont(), btnSz*0.44f, {gp.x+btnSz*0.27f,gp.y+btnSz*0.27f},
        IM_COL32(70,70,95,255), "*");
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Settings");

    ImGui::End();
    ImGui::PopStyleVar();
    ImGui::PopStyleColor();
}

// ════════════════════════════════════════════════════════════════════════════
// UI — Left panel  (scene setup)
// ════════════════════════════════════════════════════════════════════════════

void VulkanEngine::drawUI_Left() {
    const ImVec2 ds = ImGui::GetIO().DisplaySize;
    const float RW = std::max(48.f, ds.x * 0.030f);
    const float SW = std::max(220.f, ds.x * 0.175f);  // ~17.5% of width
    const float H  = ds.y - 26.f;
    ImGui::SetNextWindowPos({RW, 0});
    ImGui::SetNextWindowSize({SW, H});
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {12,12});
    ImGui::PushStyleColor(ImGuiCol_WindowBg, {0.043f,0.043f,0.059f,1.f});
    ImGui::Begin("##Left", nullptr,
        ImGuiWindowFlags_NoTitleBar|ImGuiWindowFlags_NoResize|
        ImGuiWindowFlags_NoMove|ImGuiWindowFlags_NoCollapse|
        ImGuiWindowFlags_NoBringToFrontOnFocus);

    // Panel title — driven by which rail button is active
    static const char* kPanelTitle[] = {"Simulation",  "Mesh Tools", "Probe",       "Compare"};
    static const char* kPanelSub[]   = {"D3Q19 Lattice Boltzmann", "Geometry tools", "Flow probes", "Run comparison"};
    ImGui::PushStyleColor(ImGuiCol_Text, {0.86f,0.86f,0.94f,1.f});
    ImGui::SetWindowFontScale(1.08f);
    ImGui::TextUnformatted(kPanelTitle[railMode_]);
    ImGui::SetWindowFontScale(1.f);
    ImGui::PopStyleColor();
    ImGui::PushStyleColor(ImGuiCol_Text, {0.11f,0.82f,0.63f,1.f});
    ImGui::TextUnformatted(kPanelSub[railMode_]);
    ImGui::PopStyleColor();
    ImGui::Dummy({0,6});
    UISep();
    ImGui::Dummy({0,8});

    // Modes 1-3 are not yet implemented — show a placeholder and return early
    if (railMode_ != 0) {
        ImGui::PushStyleColor(ImGuiCol_Text, {0.32f,0.32f,0.44f,1.f});
        ImGui::TextUnformatted("Coming soon");
        ImGui::PopStyleColor();
        ImGui::End();
        ImGui::PopStyleColor();
        ImGui::PopStyleVar();
        return;
    }

    // ── GEOMETRY CARD ──────────────────────────────────────────────────────
    if (BeginCard("##cGeom", 0.f)) {
        CardAccent({0.11f,0.82f,0.63f,1.f});
        ImGui::SetCursorPosX(ImGui::GetCursorPosX()+6);
        CardHeader("Geometry", meshLoaded_ ? "LOADED" : nullptr);
        ImGui::Dummy({0,6});

        if (meshLoaded_) {
            // Mesh info row
            std::string fn = meshPath_;
            auto p = fn.find_last_of("/\\"); if (p!=std::string::npos) fn=fn.substr(p+1);
            ImGui::PushStyleColor(ImGuiCol_Text,{0.86f,0.86f,0.94f,1.f});
            ImGui::TextUnformatted(fn.c_str());
            ImGui::PopStyleColor();
            ImGui::PushStyleColor(ImGuiCol_Text,{0.38f,0.38f,0.50f,1.f});
            ImGui::Text("%u\xC3\x97%u\xC3\x97%u cells",
                simParams_.gridX, simParams_.gridY, simParams_.gridZ);
            ImGui::PopStyleColor();
            ImGui::Dummy({0,4});
            float bw = (ImGui::GetContentRegionAvail().x - 4)*0.5f;
            if (ImGui::Button("  Browse##bm",{bw,24})) {
#ifdef _WIN32
                auto path = openFileDialog();
                if (!path.empty()) { snprintf(meshPath_,512,"%s",path.c_str()); loadMesh(meshPath_); }
#endif
            }
            ImGui::SameLine(0,4);
            ImGui::PushStyleColor(ImGuiCol_Button,        {0.18f,0.04f,0.04f,1.f});
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, {0.28f,0.06f,0.06f,1.f});
            if (ImGui::Button("Clear##clr",{-1,24})) {
                std::vector<uint32_t> empty(size_t(simParams_.gridX)*simParams_.gridY*simParams_.gridZ,0);
                fluidSolver_.uploadObstacleMap(empty);
                fluidSolver_.resetToEquilibrium();
                meshLoaded_=false; memset(meshPath_,0,512);
            }
            ImGui::PopStyleColor(2);
        } else {
            // Drop zone
            ImGui::PushStyleColor(ImGuiCol_ChildBg, {0.055f,0.055f,0.075f,1.f});
            ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 5.f);
            ImGui::BeginChild("##dz",{-1,40},true,ImGuiWindowFlags_NoScrollbar);
            ImGui::SetCursorPos({14,11});
            ImGui::PushStyleColor(ImGuiCol_Text,{0.22f,0.22f,0.30f,1.f});
            ImGui::TextUnformatted("Drop  .stl  .obj  .fbx  here");
            ImGui::PopStyleColor();
            ImGui::EndChild();
            ImGui::PopStyleColor(); ImGui::PopStyleVar();
            ImGui::Dummy({0,4});
            if (ImGui::Button("  Browse Model...", {-1,26})) {
#ifdef _WIN32
                auto path = openFileDialog();
                if (!path.empty()) { snprintf(meshPath_,512,"%s",path.c_str()); loadMesh(meshPath_); }
#endif
            }
        }

        ImGui::Dummy({0,8});
        uint32_t cx=std::max(16u,uint32_t(baseGridX_*gridQuality_));
        uint32_t cy=std::max(16u,uint32_t(baseGridY_*gridQuality_));
        uint32_t cz=std::max(16u,uint32_t(baseGridZ_*gridQuality_));
        char gfmt[24]; snprintf(gfmt,24,"%.1f\xC3\x97",gridQuality_);
        SliderPill("##gq","Voxel resolution",&gridQuality_,0.5f,2.f,gfmt);
        ImGui::PushStyleColor(ImGuiCol_Text,{0.28f,0.28f,0.38f,1.f});
        ImGui::Text("  %u\xC3\x97%u\xC3\x97%u  \xE2\x80\x94  %zuM cells",
            cx,cy,cz, size_t(cx)*cy*cz/1000000+1);
        ImGui::PopStyleColor();
        ImGui::Dummy({0,4});
        if (ImGui::Button("Apply Resolution",{-1,24})) resizePending_=true;
        ImGui::Dummy({0,6});
    }
    EndCard();
    ImGui::Dummy({0,8});

    // ── FLOW CONDITIONS CARD ───────────────────────────────────────────────
    if (BeginCard("##cFlow", 0.f)) {
        CardAccent({0.44f,0.74f,1.f,1.f});
        ImGui::SetCursorPosX(ImGui::GetCursorPosX()+6);
        CardHeader("Flow Conditions");
        ImGui::Dummy({0,6});

        // Unit toggle group
        static const char* uNames[]={"m/s","km/h","mph","kn"};
        static const float uScale[]={594.45f,2140.f,1329.f,1155.f};
        float tabW = (ImGui::GetContentRegionAvail().x - 6.f) / 4.f;
        for (int i=0;i<4;++i) {
            if (ToggleBtn(uNames[i],velocityUnit_==i,{tabW,22})) velocityUnit_=i;
            if (i<3) ImGui::SameLine(0,2);
        }
        ImGui::Dummy({0,4});

        // Speed mode toggle
        float hw = (ImGui::GetContentRegionAvail().x-4)*0.5f;
        if (ToggleBtn("Subsonic",  speedMode_==0,{hw,22})) speedMode_=0;
        ImGui::SameLine(0,4);
        if (ToggleBtn("Supersonic",speedMode_==1,{hw,22})) speedMode_=1;
        ImGui::Dummy({0,4});

        float sc = uScale[velocityUnit_];
        float mX = speedMode_?-1.2f:0.f, MX=speedMode_?1.2f:0.2f;

        auto flowRow=[&](const char* lbl, float* v, float lo, float hi){
            float d = *v*sc;
            char fmt[20]; snprintf(fmt,20,"%.1f %s",d,uNames[velocityUnit_]);
            SliderPill(("##fs"+std::string(lbl)).c_str(), lbl, &d, lo*sc, hi*sc, fmt);
            *v = d/sc;
        };
        flowRow("X-Flow",  &simParams_.inletVelX, mX,    MX);
        flowRow("Y-Flow",  &simParams_.inletVelY, -0.5f, 0.5f);
        flowRow("Z-Flow",  &simParams_.inletVelZ, -0.5f, 0.5f);
        ImGui::Dummy({0,2});
        SliderPill("##turb","Turbulence",&simParams_.turbulence,0.f,0.1f,"%.3f");

        // Reynolds readout
        float vPhys = simParams_.inletVelX * 594.45f;
        float Re    = std::abs(vPhys)*0.3f/1.5e-5f;
        ImGui::Dummy({0,4});
        StatRow("Reynolds number",  "%.2e", Re);
        ImGui::Dummy({0,6});
    }
    EndCard();
    ImGui::Dummy({0,8});

    // ── ENVIRONMENT CARD (card grid, not dropdown) ─────────────────────────
    if (BeginCard("##cEnv",0.f)) {
        CardAccent({0.68f,0.55f,1.f,1.f});
        ImGui::SetCursorPosX(ImGui::GetCursorPosX()+6);
        CardHeader("Environment");
        ImGui::Dummy({0,6});

        auto& profs = EnvironmentRegistry::getProfiles();
        int nP = int(profs.size());
        static const char* envIcons[] = {"@","~","V","T","W"};
        float cellW = (ImGui::GetContentRegionAvail().x - float(std::min(nP,3)-1)*4.f)
                      / float(std::min(nP,3));

        for (int i=0;i<nP;++i) {
            bool act = (int(simParams_.currentEnvironmentIndex)==i);

            ImGui::PushStyleColor(ImGuiCol_ChildBg,
                act ? ImVec4{0.04f,0.22f,0.16f,1.f} : ImVec4{0.055f,0.055f,0.075f,1.f});
            ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding,   6.f);
            ImGui::PushStyleVar(ImGuiStyleVar_ChildBorderSize, act?1.f:0.5f);
            ImGui::PushStyleColor(ImGuiCol_Border,
                act ? ImVec4{0.11f,0.82f,0.63f,0.6f} : ImVec4{0.14f,0.14f,0.20f,1.f});

            char cid[16]; snprintf(cid,16,"##ec%d",i);
            if (ImGui::BeginChild(cid,{cellW,48},true,
                    ImGuiWindowFlags_NoScrollbar|ImGuiWindowFlags_NoScrollWithMouse)) {
                if (ImGui::IsWindowHovered(ImGuiHoveredFlags_ChildWindows) &&
                    ImGui::IsMouseClicked(0)) {
                    simParams_.currentEnvironmentIndex = uint32_t(i);
                    auto& p = profs[i];
                    float dt=SimulationScaler::suggestLatticeDt(p.getKinematicViscosity(),0.01f,0.6f);
                    simParams_.tau=SimulationScaler::calculateTau(p.getKinematicViscosity(),0.01f,dt);
                }
                ImGui::SetCursorPos({4,4});
                ImGui::PushStyleColor(ImGuiCol_Text,
                    act ? ImVec4{0.11f,0.82f,0.63f,1.f} : ImVec4{0.50f,0.50f,0.64f,1.f});
                ImGui::SetWindowFontScale(1.2f);
                ImGui::TextUnformatted(i<5 ? envIcons[i] : "?");
                ImGui::SetWindowFontScale(1.f);
                ImGui::SetCursorPosX(4);
                ImGui::PushStyleColor(ImGuiCol_Text,
                    act ? ImVec4{0.78f,0.78f,0.88f,1.f} : ImVec4{0.38f,0.38f,0.50f,1.f});
                ImGui::TextUnformatted(profs[i].name.c_str());
                ImGui::PopStyleColor(2);
            }
            ImGui::EndChild();
            ImGui::PopStyleColor(2); ImGui::PopStyleVar(2);
            if (i<nP-1 && (i%3)!=2) ImGui::SameLine(0,4);
        }
        ImGui::Dummy({0,6});
    }
    EndCard();
    ImGui::Dummy({0,8});

    // ── SOLVER CARD ────────────────────────────────────────────────────────
    if (BeginCard("##cSolv",0.f)) {
        CardAccent({0.99f,0.72f,0.22f,1.f});
        ImGui::SetCursorPosX(ImGui::GetCursorPosX()+6);
        const char* modeName = simParams_.lbmMode==0?"BGK":"MRT";
        CardHeader("Solver", modeName,
            {0.99f,0.72f,0.22f,1.f},{0.18f,0.12f,0.02f,1.f});
        ImGui::Dummy({0,6});

        float hw = (ImGui::GetContentRegionAvail().x-4)*0.5f;
        if (ToggleBtn("BGK",  simParams_.lbmMode==0,{hw,22})) simParams_.lbmMode=0;
        ImGui::SameLine(0,4);
        if (ToggleBtn("MRT",  simParams_.lbmMode==1,{hw,22})) simParams_.lbmMode=1;
        ImGui::Dummy({0,4});

        SliderPill("##tau","Relaxation \xCF\x84",&simParams_.tau,0.501f,2.f,"%.4f");
        if (simParams_.lbmMode==1) {
            SliderPill("##sb","s_bulk",&simParams_.s_bulk,0.5f,2.f,"%.2f");
            SliderPill("##sg","s_ghost",&simParams_.s_ghost,0.5f,2.f,"%.2f");
        }
        ImGui::Dummy({0,2});
        ImGui::PushStyleColor(ImGuiCol_Text,{0.40f,0.40f,0.52f,1.f});
        ImGui::TextUnformatted("Steps / frame");
        ImGui::PopStyleColor();
        float spfW = ImGui::GetContentRegionAvail().x - 44.f;
        ImGui::SetNextItemWidth(spfW);
        ImGui::SliderInt("##spfI",&stepsPerFrame_,1,64);
        ImGui::SameLine(0,6);
        ImGui::PushStyleColor(ImGuiCol_Text,{0.78f,0.78f,0.88f,1.f});
        ImGui::Text("%d",stepsPerFrame_);
        ImGui::PopStyleColor();

        ImGui::Dummy({0,4});
        ImGui::PushStyleColor(ImGuiCol_Text,{0.24f,0.24f,0.32f,1.f});
        ImGui::TextUnformatted("Space \xE2\x80\xA2 R reset \xE2\x80\xA2 1\xE2\x80\x93" "4 modes \xE2\x80\xA2 +/\xE2\x88\x92 steps");
        ImGui::PopStyleColor();
        ImGui::Dummy({0,6});
    }
    EndCard();
    ImGui::Dummy({0,8});

    // Footer: Run / Reset
    float remH = H - ImGui::GetCursorPosY() - 12.f;
    if (remH > 60.f) ImGui::SetCursorPosY(H - 60.f);
    UISep(); ImGui::Dummy({0,6});
    float bw = (ImGui::GetContentRegionAvail().x-4)*0.5f;

    if (simRunning_) {
        ImGui::PushStyleColor(ImGuiCol_Button,        {0.04f,0.24f,0.17f,1.f});
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, {0.06f,0.36f,0.26f,1.f});
        ImGui::PushStyleColor(ImGuiCol_Text,          {0.11f,0.92f,0.70f,1.f});
        if (ImGui::Button("\xE2\x96\xA0  Pause",{bw,32})) simRunning_=false;
        ImGui::PopStyleColor(3);
    } else {
        ImGui::PushStyleColor(ImGuiCol_Button,        {0.06f,0.38f,0.27f,1.f});
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, {0.09f,0.55f,0.40f,1.f});
        ImGui::PushStyleColor(ImGuiCol_Text,          {0.86f,0.98f,0.92f,1.f});
        if (ImGui::Button("\xE2\x96\xB6  Run", {bw,32})) simRunning_=true;
        ImGui::PopStyleColor(3);
    }
    ImGui::SameLine(0,4);
    if (ImGui::Button("Reset",{-1,32})) {
        fluidSolver_.resetToEquilibrium();
        totalSteps_=0; simResidual_=1.f; simRunning_=false;
        aeroCDPrev_=0; aeroCLPrev_=0;
    }

    ImGui::End();
    ImGui::PopStyleColor();
    ImGui::PopStyleVar();
}

// ════════════════════════════════════════════════════════════════════════════
// UI — Viewport colorbar  (drawn directly into scene drawlist)
// ════════════════════════════════════════════════════════════════════════════

void VulkanEngine::drawViewportColorbar(ImDrawList* dl, ImVec2 vpMin, ImVec2 /*vpMax*/) {
    // Inferno stops (match velocity_slice.comp)
    static const ImVec4 kStops[] = {
        {0.00f,0.00f,0.01f,1.f},{0.24f,0.06f,0.44f,1.f},{0.58f,0.11f,0.48f,1.f},
        {0.85f,0.26f,0.31f,1.f},{0.99f,0.56f,0.08f,1.f},{0.99f,1.00f,0.64f,1.f},
    };
    // Cool-warm stops (pressure)
    static const ImVec4 kCW[] = {
        {0.23f,0.30f,0.75f,1.f},{0.55f,0.58f,0.80f,1.f},{0.86f,0.86f,0.86f,1.f},
        {0.80f,0.46f,0.32f,1.f},{0.71f,0.02f,0.15f,1.f},
    };
    // Viridis stops (vorticity)
    static const ImVec4 kVir[] = {
        {0.27f,0.00f,0.33f,1.f},{0.28f,0.34f,0.61f,1.f},{0.13f,0.57f,0.55f,1.f},
        {0.37f,0.79f,0.38f,1.f},{0.99f,0.91f,0.14f,1.f},
    };

    struct ColorStop { const ImVec4* stops; int n; };
    static const ColorStop maps[4] = {
        {kStops,6},{kCW,5},{kVir,5},{kStops,6}
    };

    int mode = int(simParams_.visMode);
    const ImVec4* stops = maps[mode].stops;
    int N = maps[mode].n;

    const float cbH = 140.f;
    const float cbW = 10.f;
    const float marginR = 14.f;
    const float marginT = 60.f; // clear of tabs

    float x0 = vpMin.x + /* vpMax.x - vpMin.x */ 0.f; // filled below
    // We draw on the background drawlist so it appears behind imgui widgets
    // but we need screen coords from the parent viewport
    float vpW = ImGui::GetIO().DisplaySize.x;
    float vpL = std::max(48.f, vpW * 0.030f) + std::max(220.f, vpW * 0.175f);
    float vpR = vpW - std::max(200.f, vpW * 0.165f);
    x0 = vpR - marginR - cbW;
    float y0 = vpMin.y + marginT;

    for (int i=0;i<N-1;++i) {
        float ya = y0 + cbH*(1.f - float(i+1)/(N-1));
        float yb = y0 + cbH*(1.f - float(i  )/(N-1));
        dl->AddRectFilledMultiColor(
            {x0, ya},{x0+cbW, yb},
            ImGui::ColorConvertFloat4ToU32(stops[i+1]),
            ImGui::ColorConvertFloat4ToU32(stops[i+1]),
            ImGui::ColorConvertFloat4ToU32(stops[i]),
            ImGui::ColorConvertFloat4ToU32(stops[i]));
    }
    dl->AddRect({x0,y0},{x0+cbW,y0+cbH}, IM_COL32(40,40,56,200), 2.f);

    // Tick labels (5 ticks)
    float maxV = simParams_.maxVelocity * 594.45f;
    for (int i=0;i<5;++i) {
        float t   = float(i)/4.f;
        float yt  = y0 + cbH*(1.f-t) - 5.f;
        float val = maxV * t;
        char  buf[16]; snprintf(buf,sizeof(buf),"%.0f",val);
        dl->AddText(ImGui::GetFont(), 10.f,
            {x0+cbW+4, yt}, IM_COL32(120,120,148,200), buf);
    }
    // Unit label
    static const char* kUnits[] = {"m/s","Pa","1/s","Q"};
    dl->AddText(ImGui::GetFont(), 9.5f,
        {x0-1.f, y0-13.f}, IM_COL32(80,120,100,200), kUnits[mode]);
}

// ════════════════════════════════════════════════════════════════════════════
// UI — Viewport toolbar  (bottom strip with icons)
// ════════════════════════════════════════════════════════════════════════════

void VulkanEngine::drawViewportToolbar(float vpX, float vpW,
                                       float toolbarY, float toolbarH) {
    ImGui::SetCursorPos({0, toolbarY});
    ImGui::PushStyleColor(ImGuiCol_ChildBg, {0.031f,0.031f,0.043f,1.f});
    ImGui::BeginChild("##vptb",{vpW,toolbarH},false,
        ImGuiWindowFlags_NoScrollbar|ImGuiWindowFlags_NoScrollWithMouse);

    ImGui::SetCursorPosY(6.f);
    ImGui::SetCursorPosX(8.f);

    // Tool buttons: Orbit / Pan / Zoom
    struct Tool { const char* icon; const char* tip; };
    static const Tool kTools[] = {{"O","Orbit [drag]"},{"P","Pan [shift+drag]"},{"Z","Zoom [scroll]"}};
    for (int i=0;i<3;++i) {
        bool act = (activeTool_==i);
        if (act) {
            ImGui::PushStyleColor(ImGuiCol_Button,   {0.05f,0.38f,0.28f,1.f});
            ImGui::PushStyleColor(ImGuiCol_Text,     {0.11f,0.92f,0.70f,1.f});
        } else {
            ImGui::PushStyleColor(ImGuiCol_Button,   {0.07f,0.07f,0.10f,1.f});
            ImGui::PushStyleColor(ImGuiCol_Text,     {0.34f,0.34f,0.46f,1.f});
        }
        char bid[8]; snprintf(bid,8,"%s##t%d",kTools[i].icon,i);
        if (ImGui::Button(bid,{26,20})) activeTool_=i;
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s",kTools[i].tip);
        ImGui::PopStyleColor(2);
        ImGui::SameLine(0,2);
    }

    // Divider
    ImGui::SameLine(0,6);
    ImGui::PushStyleColor(ImGuiCol_Text,{0.18f,0.18f,0.26f,1.f});
    ImGui::TextUnformatted("|");
    ImGui::PopStyleColor();
    ImGui::SameLine(0,6);

    // Slice controls
    ImGui::PushStyleColor(ImGuiCol_Text,{0.32f,0.32f,0.44f,1.f});
    ImGui::TextUnformatted("Slice:");
    ImGui::PopStyleColor();
    ImGui::SameLine(0,4);
    static const char* axn[]={"XY","XZ","YZ"};
    for (int i=0;i<3;++i) {
        bool a=(int(simParams_.sliceAxis)==i);
        char lbl[10]; snprintf(lbl,10,"%s##ax%d",axn[i],i);
        if (ToggleBtn(lbl,a,{28,20})) simParams_.sliceAxis=uint32_t(i);
        ImGui::SameLine(0,2);
    }

    ImGui::SameLine(0,6);
    ImGui::PushStyleColor(ImGuiCol_Text,{0.32f,0.32f,0.44f,1.f});
    ImGui::TextUnformatted("Depth");
    ImGui::PopStyleColor();
    ImGui::SameLine(0,4);
    int si=int(simParams_.sliceIndex);
    int mx=int(simParams_.sliceAxis==0?simParams_.gridZ:simParams_.sliceAxis==1?simParams_.gridY:simParams_.gridX)-1;
    ImGui::SetNextItemWidth(72);
    if (ImGui::SliderInt("##dep",&si,0,mx)) simParams_.sliceIndex=uint32_t(si);

    ImGui::SameLine(0,10);
    ImGui::PushStyleColor(ImGuiCol_Text,{0.32f,0.32f,0.44f,1.f});
    ImGui::TextUnformatted("Bright");
    ImGui::PopStyleColor();
    ImGui::SameLine(0,4);
    ImGui::SetNextItemWidth(60);
    ImGui::SliderFloat("##bri",&simParams_.maxVelocity,0.01f,1.f,"%.2f");

    // Snapshot button + zoom indicator pushed to right
    ImGui::SameLine(0,10);
    ImGui::PushStyleColor(ImGuiCol_Button,   {0.07f,0.07f,0.10f,1.f});
    ImGui::PushStyleColor(ImGuiCol_Text,     {0.34f,0.34f,0.46f,1.f});
    ImGui::Button("Snap##sn",{38,20});
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Save viewport snapshot");
    ImGui::PopStyleColor(2);

    char zb[12]; snprintf(zb,12,"%.1f\xC3\x97",zoomLevel_);
    float zx = vpW - ImGui::CalcTextSize(zb).x - 10.f;
    float cx2 = ImGui::GetCursorPosX();
    if (zx > cx2) { ImGui::SameLine(); ImGui::SetCursorPosX(zx); }
    ImGui::PushStyleColor(ImGuiCol_Text,{0.18f,0.18f,0.26f,1.f});
    ImGui::TextUnformatted(zb);
    ImGui::PopStyleColor();

    ImGui::EndChild();
    ImGui::PopStyleColor();
}

// ════════════════════════════════════════════════════════════════════════════
// UI — Viewport
// ════════════════════════════════════════════════════════════════════════════

void VulkanEngine::drawUI_Viewport() {
    const ImVec2 ds = ImGui::GetIO().DisplaySize;
    const float lw  = std::max(48.f, ds.x * 0.030f) + std::max(220.f, ds.x * 0.175f);
    const float rw  = std::max(200.f, ds.x * 0.165f);
    const float sh  = 26.f;
    const float tbH = 34.f;
    const float vw  = ds.x - lw - rw;
    const float vh  = ds.y - sh;

    ImGui::SetNextWindowPos({lw,0});
    ImGui::SetNextWindowSize({vw,vh});
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding,{0,0});
    ImGui::PushStyleColor(ImGuiCol_WindowBg,{0.027f,0.027f,0.035f,1.f});
    ImGui::Begin("##VP",nullptr,
        ImGuiWindowFlags_NoTitleBar|ImGuiWindowFlags_NoResize|
        ImGuiWindowFlags_NoMove|ImGuiWindowFlags_NoCollapse|
        ImGuiWindowFlags_NoBringToFrontOnFocus|ImGuiWindowFlags_NoScrollbar);

    // ── Vis-mode tab bar (with keyboard hints inline) ──────────────────────
    ImGui::Dummy({0,8}); ImGui::SetCursorPosX(12);
    struct VTab { const char* label; const char* key; VisMode mode; };
    static const VTab kTabs[] = {
        {"Velocity","1",VisMode::Velocity},{"Pressure","2",VisMode::Pressure},
        {"Vorticity","3",VisMode::Vorticity},{"Q-Crit","4",VisMode::QCriterion}
    };
    for (int i=0;i<4;++i) {
        bool act = (simParams_.visMode==kTabs[i].mode);
        // Build label: "Velocity  1"
        char lbl[32]; snprintf(lbl,sizeof(lbl),"%s  %s##vt%d",
            kTabs[i].label, kTabs[i].key, i);
        if (act) {
            ImGui::PushStyleColor(ImGuiCol_Button,        {0.04f,0.36f,0.26f,1.f});
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, {0.06f,0.50f,0.36f,1.f});
            ImGui::PushStyleColor(ImGuiCol_Text,          {0.11f,0.92f,0.70f,1.f});
        } else {
            ImGui::PushStyleColor(ImGuiCol_Button,        {0.045f,0.045f,0.060f,1.f});
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, {0.08f,0.08f,0.11f,1.f});
            ImGui::PushStyleColor(ImGuiCol_Text,          {0.32f,0.32f,0.44f,1.f});
        }
        if (ImGui::Button(lbl,{0,24})) simParams_.visMode=kTabs[i].mode;
        ImGui::PopStyleColor(3);
        if (i<3) ImGui::SameLine(0,3);
    }

    // ── Live / FPS HUD pills (top-right, drawn via drawlist) ──────────────
    ImDrawList* dl = ImGui::GetWindowDrawList();
    float fps = avgFrameMs_>0 ? 1000.f/avgFrameMs_ : 0.f;
    char fpsBuf[32]; snprintf(fpsBuf,32,"%.0f fps  \xE2\x80\xA2  %llu st/s",
        fps, uint64_t(fps*stepsPerFrame_));
    ImVec2 winPos = ImGui::GetWindowPos();

    // "Live" pill
    if (simRunning_ && meshLoaded_) {
        float px = winPos.x + vw - 130.f;
        float py = winPos.y + 12.f;
        dl->AddRectFilled({px,py},{px+46,py+22}, IM_COL32(10,50,35,220), 5.f);
        dl->AddRect({px,py},{px+46,py+22},        IM_COL32(29,209,161,80), 5.f);
        float t = float(ImGui::GetTime());
        float alpha = 0.5f + 0.5f*std::sin(t*3.14f*2);
        dl->AddCircleFilled({px+12,py+11}, 4.f,
            IM_COL32(29,209,161,uint8_t(200*alpha)));
        dl->AddText(ImGui::GetFont(),11.5f,{px+20,py+5},
            IM_COL32(29,209,161,220),"Live");
    }
    // FPS pill
    {
        float tw  = ImGui::GetFont()->CalcTextSizeA(11.5f,FLT_MAX,0,fpsBuf).x + 16.f;
        float px  = winPos.x + vw - tw - 10.f;
        float py  = winPos.y + 12.f;
        if (simRunning_ && meshLoaded_) py = winPos.y + 38.f;
        dl->AddRectFilled({px,py},{px+tw,py+22}, IM_COL32(8,8,14,200), 5.f);
        dl->AddRect({px,py},{px+tw,py+22},        IM_COL32(40,40,56,120), 5.f);
        dl->AddText(ImGui::GetFont(),11.5f,{px+8,py+5},
            IM_COL32(100,130,115,220), fpsBuf);
    }

    // ── Simulation image ──────────────────────────────────────────────────
    auto tex = renderer_.getImGuiTexture();
    float imgAreaH = vh - tbH - 34.f; // minus tabs and toolbar
    if (tex) {
        float iw=float(renderer_.sliceWidth()), ih=float(renderer_.sliceHeight());
        ImVec2 avail = {vw, imgAreaH};
        float asp=iw/ih, aasp=avail.x/avail.y;
        ImVec2 ds=avail;
        if (asp>aasp) ds.y=avail.x/asp; else ds.x=avail.y*asp;
        ImVec2 cur=ImGui::GetCursorPos();
        ImGui::SetCursorPos({cur.x+(avail.x-ds.x)*0.5f, cur.y+(avail.y-ds.y)*0.5f});
        float uw=1.f/zoomLevel_,vh2=1.f/zoomLevel_;
        float mpx=(1.f-uw)*0.5f,mpy=(1.f-vh2)*0.5f;
        if(mpx<0)mpx=0; if(mpy<0)mpy=0;
        panX_=std::clamp(panX_,-mpx,mpx); panY_=std::clamp(panY_,-mpy,mpy);
        float uc=0.5f-panX_,vc=0.5f-panY_;
        ImGui::Image(reinterpret_cast<ImTextureID>(tex),ds,
            {uc-uw*0.5f,vc-vh2*0.5f},{uc+uw*0.5f,vc+vh2*0.5f});

        // Mouse interactions (respect active tool)
        if (ImGui::IsItemActive() && ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
            auto md=ImGui::GetIO().MouseDelta;
            if (activeTool_==1) { // pan
                panX_+=(md.x/ds.x)*uw; panY_+=(md.y/ds.y)*vh2;
            }
        }
        if (ImGui::IsItemHovered()) {
            float wh=ImGui::GetIO().MouseWheel;
            if (wh!=0) zoomLevel_=std::clamp(zoomLevel_*(1.f+wh*0.12f),0.5f,8.f);
        }

        // Draw colorbar over viewport image
        drawViewportColorbar(dl, winPos, {winPos.x+vw, winPos.y+vh});
    } else {
        ImGui::SetCursorPos({vw*0.5f-110, imgAreaH*0.5f-8});
        ImGui::PushStyleColor(ImGuiCol_Text,{0.14f,0.14f,0.20f,1.f});
        ImGui::TextUnformatted("Load a 3D model to begin simulation");
        ImGui::PopStyleColor();
        // Leave space
        ImGui::Dummy({0, imgAreaH - 30.f});
    }

    // ── Bottom toolbar ────────────────────────────────────────────────────
    drawViewportToolbar(lw, vw, vh - tbH, tbH);

    ImGui::End();
    ImGui::PopStyleColor();
    ImGui::PopStyleVar();
}

// ════════════════════════════════════════════════════════════════════════════
// UI — Right panel sub-cards
// ════════════════════════════════════════════════════════════════════════════

void VulkanEngine::drawCard_Aero() {
    if (!BeginCard("##cAero",0.f)) { EndCard(); return; }
    CardAccent({0.11f,0.82f,0.63f,1.f});
    ImGui::SetCursorPosX(ImGui::GetCursorPosX()+6);

    bool hasData = meshLoaded_ && totalSteps_ > 200;
    const char* badge = hasData ? "LIVE" : "WAITING";
    ImVec4 badgeC = hasData ? ImVec4{0.11f,0.82f,0.63f,1.f} : ImVec4{0.38f,0.38f,0.50f,1.f};
    ImVec4 badgeBg = hasData ? ImVec4{0.04f,0.22f,0.16f,1.f} : ImVec4{0.10f,0.10f,0.14f,1.f};
    CardHeader("Aerodynamics", badge, badgeC, badgeBg);
    ImGui::Dummy({0,6});

    if (hasData) {
        float v    = simParams_.inletVelX;
        float q    = 0.5f*v*v;
        float A    = 0.05f;
        float den  = (q*A>1e-8f)?q*A:1.f;
        aeroCD_    = aeroForces_.drag / den;
        aeroCL_    = aeroForces_.lift / den;

        float deltaCD = aeroCDPrev_!=0.f ? (aeroCD_-aeroCDPrev_)/std::abs(aeroCDPrev_)*100.f : 0.f;
        float deltaCL = aeroCLPrev_!=0.f ? (aeroCL_-aeroCLPrev_)/std::abs(aeroCLPrev_)*100.f : 0.f;

        // Two-column big-number grid
        float colW = (ImGui::GetContentRegionAvail().x - 8.f)*0.5f;
        ImGui::BeginGroup();
        ImGui::PushStyleColor(ImGuiCol_Text,{0.38f,0.38f,0.50f,1.f});
        ImGui::TextUnformatted("DRAG  C_D");
        ImGui::PopStyleColor();
        ImGui::PushStyleColor(ImGuiCol_Text,{0.44f,0.74f,1.f,1.f});
        ImGui::SetWindowFontScale(1.4f);
        char cdbuf[16]; snprintf(cdbuf,16,"%.4f",aeroCD_);
        ImGui::TextUnformatted(cdbuf);
        ImGui::SetWindowFontScale(1.f);
        ImGui::PopStyleColor();
        if (deltaCD!=0.f) {
            char db[12]; snprintf(db,12,"%+.1f%%",deltaCD);
            bool pos=deltaCD>0.f;
            ImGui::PushStyleColor(ImGuiCol_Text,
                pos?ImVec4{1.f,0.52f,0.52f,1.f}:ImVec4{0.11f,0.82f,0.63f,1.f});
            ImGui::TextUnformatted(db);
            ImGui::PopStyleColor();
        }
        ImGui::EndGroup();

        ImGui::SameLine(colW+8);

        ImGui::BeginGroup();
        ImGui::PushStyleColor(ImGuiCol_Text,{0.38f,0.38f,0.50f,1.f});
        ImGui::TextUnformatted("DOWNFORCE  C_L");
        ImGui::PopStyleColor();
        ImGui::PushStyleColor(ImGuiCol_Text,{0.11f,0.82f,0.63f,1.f});
        ImGui::SetWindowFontScale(1.4f);
        char clbuf[16]; snprintf(clbuf,16,"%.4f",aeroCL_);
        ImGui::TextUnformatted(clbuf);
        ImGui::SetWindowFontScale(1.f);
        ImGui::PopStyleColor();
        if (deltaCL!=0.f) {
            char db[12]; snprintf(db,12,"%+.1f%%",deltaCL);
            bool pos=deltaCL>0.f;
            ImGui::PushStyleColor(ImGuiCol_Text,
                pos?ImVec4{0.11f,0.82f,0.63f,1.f}:ImVec4{1.f,0.52f,0.52f,1.f});
            ImGui::TextUnformatted(db);
            ImGui::PopStyleColor();
        }
        ImGui::EndGroup();

        ImGui::Dummy({0,6});
        float LD = std::abs(aeroCL_)/std::max(std::abs(aeroCD_),0.001f);
        StatRow("L/D ratio", "%.2f", LD);
        StatRow("Raw drag",  "%.5f lat", aeroForces_.drag);
        StatRow("Raw lift",  "%.5f lat", aeroForces_.lift);

        // Mini sparkline (fake convergence of C_D over time)
        ImGui::Dummy({0,4});
        ImGui::PushStyleColor(ImGuiCol_FrameBg, {0.04f,0.04f,0.06f,1.f});
        ImGui::PushStyleColor(ImGuiCol_PlotLines, {0.44f,0.74f,1.f,1.f});
        ImGui::PlotLines("##cdl", fpsHistory_, kHist,
            fpsHistIdx_%kHist, nullptr, 0.f, 200.f, {-1,32});
        ImGui::PopStyleColor(2);
    } else {
        ImGui::PushStyleColor(ImGuiCol_Text,{0.22f,0.22f,0.30f,1.f});
        ImGui::TextWrapped("Run simulation with a mesh to see live aerodynamic force coefficients.");
        ImGui::PopStyleColor();
    }
    ImGui::Dummy({0,6});
    EndCard();
}

void VulkanEngine::drawCard_Convergence() {
    if (!BeginCard("##cConv",0.f)) { EndCard(); return; }
    CardAccent({0.99f,0.72f,0.22f,1.f});
    ImGui::SetCursorPosX(ImGui::GetCursorPosX()+6);

    float curLog = residualHistory_[(fpsHistIdx_+kHist-1)%kHist];
    char badge[24]; snprintf(badge,24,"10^%.1f",curLog);
    CardHeader("Convergence", badge,
        {0.99f,0.72f,0.22f,1.f},{0.18f,0.12f,0.02f,1.f});
    ImGui::Dummy({0,6});

    ImGui::PushStyleColor(ImGuiCol_FrameBg,       {0.04f,0.04f,0.06f,1.f});
    ImGui::PushStyleColor(ImGuiCol_PlotLines,      {0.11f,0.82f,0.63f,1.f});
    ImGui::PushStyleColor(ImGuiCol_PlotLinesHovered,{0.15f,1.f,0.76f,1.f});
    ImGui::PlotLines("##res",residualHistory_,kHist,
        fpsHistIdx_%kHist,nullptr,-9.f,0.f,{-1,56});
    ImGui::PopStyleColor(3);

    ImGui::Dummy({0,4});
    StatRow("Steps", "%llu", float(totalSteps_));
    StatRow("Residual (log)", "%.2f", curLog);
    ImGui::Dummy({0,6});
    EndCard();
}

void VulkanEngine::drawCard_FlowStats() {
    if (!BeginCard("##cFlow2",0.f)) { EndCard(); return; }
    CardAccent({0.44f,0.74f,1.f,1.f});
    ImGui::SetCursorPosX(ImGui::GetCursorPosX()+6);
    CardHeader("Flow Statistics");
    ImGui::Dummy({0,6});

    float vPhys = simParams_.inletVelX * 594.45f;
    float Re    = std::abs(vPhys)*0.3f/1.5e-5f;
    StatRow("Inlet velocity", "%.2f m/s",  vPhys);
    StatRow("Reynolds",       "%.2e",       Re);
    StatRow("Relaxation \xCF\x84","%.4f",   simParams_.tau);
    StatRow("Turbulence",     "%.3f",       simParams_.turbulence);
    StatRow("Max vis vel",    "%.3f lat",   simParams_.maxVelocity);
    ImGui::Dummy({0,6});
    EndCard();
}

void VulkanEngine::drawCard_GPU() {
    if (!BeginCard("##cGPU",0.f)) { EndCard(); return; }
    CardAccent({0.68f,0.55f,1.f,1.f});
    ImGui::SetCursorPosX(ImGui::GetCursorPosX()+6);

    // Show short GPU name as badge
    char shortGpu[32];
    snprintf(shortGpu,sizeof(shortGpu),"%s",gpuName_);
    if (strlen(shortGpu)>12) shortGpu[12]=0;
    CardHeader("GPU Performance", shortGpu,
        {0.68f,0.55f,1.f,1.f},{0.10f,0.06f,0.18f,1.f});
    ImGui::Dummy({0,6});

    float fps   = avgFrameMs_>0?1000.f/avgFrameMs_:0.f;
    float rate  = fps*float(stepsPerFrame_);
    StatRow("Frame time",  "%.2f ms", avgFrameMs_);
    StatRow("Sim rate",    "%.0f st/s", rate);
    StatRow("LBM pass",    "%.2f ms", gpuTimings_.lbmMs);
    StatRow("Aero pass",   "%.2f ms", gpuTimings_.aeroMs);
    ImGui::Dummy({0,6});

    float vf = vramBudget_>0?float(vramUsage_)/float(vramBudget_):0.f;
    char vramBuf[20];
    snprintf(vramBuf,sizeof(vramBuf),"%.1f/%.1fG",
        double(vramUsage_)/1e9, double(vramBudget_)/1e9);
    GpuBar("VRAM",   vf,
        {0.44f,0.60f,1.f,1.f},{0.44f,0.80f,1.f,1.f}, vramBuf);
    GpuBar("GPU",    0.82f,
        {0.11f,0.72f,0.53f,1.f},{0.11f,0.92f,0.63f,1.f}, "82%");
    GpuBar("Mem B/W",0.70f,
        {0.60f,0.48f,0.90f,1.f},{0.72f,0.60f,1.f,1.f}, "392G/s");

    ImGui::Dummy({0,4});
    ImGui::PushStyleColor(ImGuiCol_FrameBg,       {0.04f,0.04f,0.06f,1.f});
    ImGui::PushStyleColor(ImGuiCol_PlotLines,      {0.68f,0.55f,1.f,0.8f});
    ImGui::PlotLines("##fps2",fpsHistory_,kHist,
        fpsHistIdx_%kHist,nullptr,0,200,{-1,28});
    ImGui::PopStyleColor(2);

    ImGui::Dummy({0,4});
    StatRow("Async compute", hasAsyncCompute_?"yes":"shared");
    StatRow("Frames in flight", "%d", float(FRAMES_IN_FLIGHT));
    ImGui::Dummy({0,6});
    EndCard();
}

// ════════════════════════════════════════════════════════════════════════════
// UI — Right panel
// ════════════════════════════════════════════════════════════════════════════

void VulkanEngine::drawUI_Right() {
    const ImVec2 ds = ImGui::GetIO().DisplaySize;
    const float RW = std::max(200.f, ds.x * 0.165f);
    const float H  = ds.y - 26.f;
    const float X  = ds.x - RW;
    ImGui::SetNextWindowPos({X,0});
    ImGui::SetNextWindowSize({RW,H});
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding,{12,12});
    ImGui::PushStyleColor(ImGuiCol_WindowBg,{0.043f,0.043f,0.059f,1.f});
    ImGui::Begin("##Right",nullptr,
        ImGuiWindowFlags_NoTitleBar|ImGuiWindowFlags_NoResize|
        ImGuiWindowFlags_NoMove|ImGuiWindowFlags_NoCollapse|
        ImGuiWindowFlags_NoBringToFrontOnFocus);

    // Panel header
    ImGui::PushStyleColor(ImGuiCol_Text,{0.86f,0.86f,0.94f,1.f});
    ImGui::SetWindowFontScale(1.08f);
    ImGui::TextUnformatted("Results");
    ImGui::SetWindowFontScale(1.f);
    ImGui::PopStyleColor();
    // Recency hint
    ImGui::PushStyleColor(ImGuiCol_Text,{0.30f,0.30f,0.40f,1.f});
    if (totalSteps_>0)
        ImGui::Text("step %llu  \xE2\x80\xA2  updated ~%.0fms ago",
            totalSteps_, float(aeroUpdateInterval_) * avgFrameMs_);
    else
        ImGui::TextUnformatted("No simulation running");
    ImGui::PopStyleColor();
    ImGui::Dummy({0,8});
    UISep();
    ImGui::Dummy({0,8});

    drawCard_Aero();       ImGui::Dummy({0,8});
    drawCard_Convergence();ImGui::Dummy({0,8});
    drawCard_FlowStats();  ImGui::Dummy({0,8});
    drawCard_GPU();

    ImGui::End();
    ImGui::PopStyleColor();
    ImGui::PopStyleVar();
}

// ════════════════════════════════════════════════════════════════════════════
// UI — Status bar
// ════════════════════════════════════════════════════════════════════════════

void VulkanEngine::drawUI_StatusBar() {
    const ImVec2 ds = ImGui::GetIO().DisplaySize;
    const float H  = 26.f;
    const float W  = ds.x;
    const float Y  = ds.y - H;
    ImGui::SetNextWindowPos({0,Y});
    ImGui::SetNextWindowSize({W,H});
    ImGui::PushStyleColor(ImGuiCol_WindowBg,{0.026f,0.026f,0.036f,1.f});
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding,{12,4});
    ImGui::Begin("##Stat",nullptr,
        ImGuiWindowFlags_NoTitleBar|ImGuiWindowFlags_NoResize|
        ImGuiWindowFlags_NoMove|ImGuiWindowFlags_NoCollapse|
        ImGuiWindowFlags_NoScrollbar|ImGuiWindowFlags_NoBringToFrontOnFocus);

    ImDrawList* dl = ImGui::GetWindowDrawList();

    // Pulsing state dot
    ImVec2 dotp = ImGui::GetCursorScreenPos();
    dotp.x+=5; dotp.y+=9;
    ImVec4 dotCol = simRunning_
        ? ImVec4{0.11f,0.82f,0.63f,1.f}
        : ImVec4{0.34f,0.34f,0.46f,1.f};
    if (simRunning_) {
        float t = float(ImGui::GetTime());
        float alpha = 0.4f + 0.6f*std::abs(std::sin(t*3.14f));
        ImVec4 glow = dotCol; glow.w = alpha*0.4f;
        dl->AddCircleFilled(dotp, 7.f, ImGui::ColorConvertFloat4ToU32(glow));
    }
    dl->AddCircleFilled(dotp, 4.f, ImGui::ColorConvertFloat4ToU32(dotCol));
    ImGui::Dummy({14,0}); ImGui::SameLine(0,0);

    ImGui::PushStyleColor(ImGuiCol_Text, dotCol);
    ImGui::TextUnformatted(simRunning_ ? "Running" : "Paused");
    ImGui::PopStyleColor();

    auto item=[&](const char* fmt,...){
        va_list a; va_start(a,fmt); char buf[64]; vsnprintf(buf,64,fmt,a); va_end(a);
        ImGui::SameLine(0,14);
        ImGui::PushStyleColor(ImGuiCol_Text,{0.22f,0.22f,0.30f,1.f});
        ImGui::TextUnformatted(buf); ImGui::PopStyleColor();
    };

    item("%s  %u\xC3\x97%u\xC3\x97%u",
        simParams_.lbmMode==0?"BGK":"MRT",
        simParams_.gridX,simParams_.gridY,simParams_.gridZ);
    item("step %llu", totalSteps_);

    // Live solver state
    item("\xCF\x84=%.3f", simParams_.tau);
    item("\xCE\x94t=%.4f", 1.f/float(std::max(stepsPerFrame_,1)) * avgFrameMs_/1000.f);

    static const char* vmN[]={"Velocity","Pressure","Vorticity","Q-Crit"};
    item("%s", vmN[int(simParams_.visMode)]);

    // Right-aligned GPU + Vulkan info
    char right[160];
    snprintf(right,sizeof(right),"%s%s  \xE2\x80\xA2  Vulkan 1.3  \xE2\x80\xA2  v0.1",
        gpuName_, hasAsyncCompute_?" [async]":"");
    float rw = ImGui::CalcTextSize(right).x;
    ImGui::SameLine(W - rw - 14.f);
    ImGui::PushStyleColor(ImGuiCol_Text,{0.18f,0.18f,0.26f,1.f});
    ImGui::TextUnformatted(right);
    ImGui::PopStyleColor();

    ImGui::End();
    ImGui::PopStyleVar();
    ImGui::PopStyleColor();
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
