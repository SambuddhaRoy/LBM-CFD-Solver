// ============================================================================
// gpu.cpp — Vulkan context, swapchain, RAII helpers
// ============================================================================

#include "gpu.h"

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <VkBootstrap.h>

#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

#include <chrono>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <iostream>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace vwt {

// ─── Logging ─────────────────────────────────────────────────────────────────

void logMsg(const std::string& msg) {
    auto now  = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    auto ms   = std::chrono::duration_cast<std::chrono::milliseconds>(
                    now.time_since_epoch()) % 1000;
    std::tm tmBuf{};
#ifdef _WIN32
    localtime_s(&tmBuf, &time);
#else
    localtime_r(&time, &tmBuf);
#endif
    char stamp[16];
    std::snprintf(stamp, sizeof(stamp), "%02d:%02d:%02d.%03d",
                  tmBuf.tm_hour, tmBuf.tm_min, tmBuf.tm_sec, int(ms.count()));
    std::cout << "[" << stamp << "] " << msg << "\n";
}

// ─── Executable directory ────────────────────────────────────────────────────

const std::filesystem::path& exeDir() {
    static const std::filesystem::path dir = [] {
#ifdef _WIN32
        char buf[MAX_PATH] = {};
        GetModuleFileNameA(nullptr, buf, MAX_PATH);
        return std::filesystem::path(buf).parent_path();
#else
        std::error_code ec;
        auto p = std::filesystem::read_symlink("/proc/self/exe", ec);
        return ec ? std::filesystem::current_path() : p.parent_path();
#endif
    }();
    return dir;
}

// ─── Context ─────────────────────────────────────────────────────────────────

void GpuContext::init(GLFWwindow* window) {
    const bool headless = (window == nullptr);

    vkb::InstanceBuilder ib;
    ib.set_app_name("VirtualWindTunnel")
      .require_api_version(1, 3, 0);
    if (headless) ib.set_headless();
#ifndef NDEBUG
    ib.request_validation_layers(true).use_default_debug_messenger();
#endif
    auto ir = ib.build();
    if (!ir) throw std::runtime_error("Vulkan instance: " + ir.error().message());
    instance_  = ir.value().instance;
    debugMsgr_ = ir.value().debug_messenger;

    if (!headless)
        VK_CHECK(glfwCreateWindowSurface(instance_, window, nullptr, &surface_));

    vkb::PhysicalDeviceSelector sel(ir.value());
    sel.set_minimum_version(1, 3)
       .prefer_gpu_device_type(vkb::PreferredDeviceType::discrete);
    if (!headless) sel.set_surface(surface_);
    auto pr = sel.select();
    if (!pr) throw std::runtime_error("Physical device: " + pr.error().message());
    physDevice_ = pr.value().physical_device;

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(physDevice_, &props);
    std::snprintf(gpuName_, sizeof(gpuName_), "%s", props.deviceName);
    tsPeriodNs_ = props.limits.timestampPeriod;
    logMsg(std::string("GPU: ") + gpuName_);

    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(physDevice_, &memProps);
    memHeapCount_ = memProps.memoryHeapCount;

    vkb::DeviceBuilder db(pr.value());
    auto dr = db.build();
    if (!dr) throw std::runtime_error("Device: " + dr.error().message());
    device_ = dr.value().device;

    // One graphics+compute queue keeps the synchronization story simple.
    auto q = dr.value().get_queue(vkb::QueueType::graphics);
    if (q.has_value()) {
        queue_       = q.value();
        queueFamily_ = dr.value().get_queue_index(vkb::QueueType::graphics).value();
    } else {
        // Headless device may report no "graphics" queue — fall back to compute.
        queue_       = dr.value().get_queue(vkb::QueueType::compute).value();
        queueFamily_ = dr.value().get_queue_index(vkb::QueueType::compute).value();
    }

    VmaAllocatorCreateInfo vai{};
    vai.physicalDevice   = physDevice_;
    vai.device           = device_;
    vai.instance         = instance_;
    vai.vulkanApiVersion = VK_API_VERSION_1_3;
    VK_CHECK(vmaCreateAllocator(&vai, &allocator_));

    // Disk-backed pipeline cache
    std::vector<char> cacheData;
    const auto cachePath = exeDir() / "vwt_pipeline.bin";
    if (std::ifstream f(cachePath, std::ios::binary | std::ios::ate); f.is_open()) {
        cacheData.resize(size_t(f.tellg()));
        f.seekg(0);
        f.read(cacheData.data(), std::streamsize(cacheData.size()));
    }
    VkPipelineCacheCreateInfo ci{};
    ci.sType           = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
    ci.initialDataSize = cacheData.size();
    ci.pInitialData    = cacheData.empty() ? nullptr : cacheData.data();
    VK_CHECK(vkCreatePipelineCache(device_, &ci, nullptr, &cache_));

    VkCommandPoolCreateInfo pi{};
    pi.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pi.queueFamilyIndex = queueFamily_;
    pi.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VK_CHECK(vkCreateCommandPool(device_, &pi, nullptr, &oneShotPool_));
}

void GpuContext::destroy() {
    if (device_ == VK_NULL_HANDLE) return;
    vkDeviceWaitIdle(device_);
    savePipelineCache();
    vkDestroyCommandPool(device_, oneShotPool_, nullptr);
    vkDestroyPipelineCache(device_, cache_, nullptr);
    vmaDestroyAllocator(allocator_);
    vkDestroyDevice(device_, nullptr);
    if (surface_) vkDestroySurfaceKHR(instance_, surface_, nullptr);
    if (debugMsgr_) vkb::destroy_debug_utils_messenger(instance_, debugMsgr_);
    vkDestroyInstance(instance_, nullptr);
    device_ = VK_NULL_HANDLE;
}

GpuBuffer GpuContext::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, MemLoc loc) {
    VkBufferCreateInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size  = size;
    bi.usage = usage;

    VmaAllocationCreateInfo ai{};
    switch (loc) {
    case MemLoc::Device:
        ai.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        break;
    case MemLoc::HostWrite:
        ai.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
        ai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                   VMA_ALLOCATION_CREATE_MAPPED_BIT;
        break;
    case MemLoc::HostRead:
        ai.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
        ai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
                   VMA_ALLOCATION_CREATE_MAPPED_BIT;
        break;
    }

    GpuBuffer b;
    b.size = size;
    VmaAllocationInfo info{};
    VK_CHECK(vmaCreateBuffer(allocator_, &bi, &ai, &b.buffer, &b.alloc, &info));
    b.mapped = info.pMappedData;
    return b;
}

void GpuContext::destroyBuffer(GpuBuffer& b) {
    if (b.buffer) vmaDestroyBuffer(allocator_, b.buffer, b.alloc);
    b = {};
}

void GpuContext::oneShot(const std::function<void(VkCommandBuffer)>& record) {
    VkCommandBufferAllocateInfo ai{};
    ai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool        = oneShotPool_;
    ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VkCommandBuffer cmd;
    VK_CHECK(vkAllocateCommandBuffers(device_, &ai, &cmd));

    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cmd, &bi));
    record(cmd);
    VK_CHECK(vkEndCommandBuffer(cmd));

    VkSubmitInfo si{};
    si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers    = &cmd;
    VK_CHECK(vkQueueSubmit(queue_, 1, &si, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(queue_));
    vkFreeCommandBuffers(device_, oneShotPool_, 1, &cmd);
}

VkPipeline GpuContext::makeComputePipeline(const char* spvName,
                                           VkDescriptorSetLayout setLayout,
                                           uint32_t pushSize,
                                           VkPipelineLayout& outLayout) {
    const auto path = exeDir() / "shaders" / spvName;
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.is_open())
        throw std::runtime_error("Shader not found: " + path.string());
    std::vector<uint32_t> spirv(size_t(f.tellg()) / sizeof(uint32_t));
    f.seekg(0);
    f.read(reinterpret_cast<char*>(spirv.data()),
           std::streamsize(spirv.size() * sizeof(uint32_t)));

    VkShaderModuleCreateInfo smi{};
    smi.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smi.codeSize = spirv.size() * sizeof(uint32_t);
    smi.pCode    = spirv.data();
    VkShaderModule sm;
    VK_CHECK(vkCreateShaderModule(device_, &smi, nullptr, &sm));

    VkPushConstantRange pcr{ VK_SHADER_STAGE_COMPUTE_BIT, 0, pushSize };
    VkPipelineLayoutCreateInfo pli{};
    pli.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pli.setLayoutCount         = 1;
    pli.pSetLayouts            = &setLayout;
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges    = &pcr;
    VK_CHECK(vkCreatePipelineLayout(device_, &pli, nullptr, &outLayout));

    VkComputePipelineCreateInfo ci{};
    ci.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    ci.layout = outLayout;
    ci.stage  = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                  nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, sm, "main", nullptr };
    VkPipeline pipe;
    VK_CHECK(vkCreateComputePipelines(device_, cache_, 1, &ci, nullptr, &pipe));
    vkDestroyShaderModule(device_, sm, nullptr);
    return pipe;
}

void GpuContext::queryVram(uint64_t& usage, uint64_t& budget) const {
    VmaBudget budgets[VK_MAX_MEMORY_HEAPS] = {};
    vmaGetHeapBudgets(allocator_, budgets);
    usage = 0; budget = 0;
    for (uint32_t i = 0; i < memHeapCount_; ++i) {
        usage  = std::max<uint64_t>(usage,  budgets[i].usage);
        budget = std::max<uint64_t>(budget, budgets[i].budget);
    }
}

void GpuContext::savePipelineCache() {
    if (cache_ == VK_NULL_HANDLE) return;
    size_t sz = 0;
    vkGetPipelineCacheData(device_, cache_, &sz, nullptr);
    if (!sz) return;
    std::vector<uint8_t> data(sz);
    vkGetPipelineCacheData(device_, cache_, &sz, data.data());
    std::ofstream f(exeDir() / "vwt_pipeline.bin", std::ios::binary);
    f.write(reinterpret_cast<const char*>(data.data()), std::streamsize(sz));
}

// ─── Swapchain ───────────────────────────────────────────────────────────────

void Swapchain::init(GpuContext& ctx, uint32_t w, uint32_t h) {
    ctx_ = &ctx;
    create(w, h);
}

void Swapchain::create(uint32_t w, uint32_t h) {
    vkb::SwapchainBuilder sb(ctx_->physDevice(), ctx_->device(), ctx_->surface());
    auto sr = sb.use_default_format_selection()
                .set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR)
                .set_desired_extent(w, h)
                .build();
    if (!sr) throw std::runtime_error("Swapchain: " + sr.error().message());
    swapchain_ = sr.value().swapchain;
    format_    = sr.value().image_format;
    extent_    = sr.value().extent;
    images_    = sr.value().get_images().value();
    views_     = sr.value().get_image_views().value();

    VkAttachmentDescription color{};
    color.format         = format_;
    color.samples        = VK_SAMPLE_COUNT_1_BIT;
    color.loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
    color.storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
    color.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    color.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    color.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
    color.finalLayout    = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference ref{ 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
    VkSubpassDescription sub{};
    sub.pipelineBindPoint    = VK_PIPELINE_BIND_POINT_GRAPHICS;
    sub.colorAttachmentCount = 1;
    sub.pColorAttachments    = &ref;

    VkSubpassDependency dep{};
    dep.srcSubpass    = VK_SUBPASS_EXTERNAL;
    dep.dstSubpass    = 0;
    dep.srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dep.srcAccessMask = 0;
    dep.dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    VkRenderPassCreateInfo rp{};
    rp.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    rp.attachmentCount = 1;
    rp.pAttachments    = &color;
    rp.subpassCount    = 1;
    rp.pSubpasses      = &sub;
    rp.dependencyCount = 1;
    rp.pDependencies   = &dep;
    VK_CHECK(vkCreateRenderPass(ctx_->device(), &rp, nullptr, &renderPass_));

    framebuffers_.resize(views_.size());
    for (size_t i = 0; i < views_.size(); ++i) {
        VkFramebufferCreateInfo fb{};
        fb.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fb.renderPass      = renderPass_;
        fb.attachmentCount = 1;
        fb.pAttachments    = &views_[i];
        fb.width           = extent_.width;
        fb.height          = extent_.height;
        fb.layers          = 1;
        VK_CHECK(vkCreateFramebuffer(ctx_->device(), &fb, nullptr, &framebuffers_[i]));
    }
}

void Swapchain::destroyResources() {
    for (auto fb : framebuffers_) vkDestroyFramebuffer(ctx_->device(), fb, nullptr);
    framebuffers_.clear();
    vkDestroyRenderPass(ctx_->device(), renderPass_, nullptr);
    renderPass_ = VK_NULL_HANDLE;
    for (auto v : views_) vkDestroyImageView(ctx_->device(), v, nullptr);
    views_.clear();
    vkDestroySwapchainKHR(ctx_->device(), swapchain_, nullptr);
    swapchain_ = VK_NULL_HANDLE;
}

void Swapchain::recreate(uint32_t w, uint32_t h) {
    vkDeviceWaitIdle(ctx_->device());
    destroyResources();
    create(w, h);
}

void Swapchain::destroy() {
    if (ctx_ && swapchain_) destroyResources();
}

} // namespace vwt
