// ============================================================================
// vk.cpp — Vulkan context implementation (see vk.hpp)
// ============================================================================

#define VMA_IMPLEMENTATION
#define VMA_STATIC_VULKAN_FUNCTIONS 1
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 0
#include "vk.hpp"

#include <GLFW/glfw3.h>
#include <VkBootstrap.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

namespace gpu {

void check(VkResult r, const char* what) {
    if (r != VK_SUCCESS)
        throw std::runtime_error(std::string(what) + " failed (VkResult " + std::to_string(int(r)) + ")");
}

namespace {

std::filesystem::path exeDir() {
#ifdef _WIN32
    wchar_t buf[MAX_PATH];
    GetModuleFileNameW(nullptr, buf, MAX_PATH);
    return std::filesystem::path(buf).parent_path();
#else
    return std::filesystem::canonical("/proc/self/exe").parent_path();
#endif
}

std::vector<char> readFile(const std::filesystem::path& p) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f) return {};
    std::vector<char> data(size_t(f.tellg()));
    f.seekg(0);
    f.read(data.data(), std::streamsize(data.size()));
    return data;
}

} // namespace

void Context::init(GLFWwindow* window) {
    vkb::InstanceBuilder ib;
    ib.set_app_name("Wind Tunnel").require_api_version(1, 3, 0);
#ifndef NDEBUG
    ib.request_validation_layers(true).use_default_debug_messenger();
#endif
    if (!window) ib.set_headless(true);
    auto inst = ib.build();
    if (!inst) throw std::runtime_error("Vulkan instance: " + inst.error().message());
    instance   = inst->instance;
    messenger_ = inst->debug_messenger;

    if (window) check(glfwCreateWindowSurface(instance, window, nullptr, &surface), "surface");

    // Features the kernels need: 8/16-bit storage for flags, SDF and the FP16
    // distribution formats; the explicit arithmetic types those extensions
    // pull in; synchronization2 and dynamic rendering for the UI pass.
    VkPhysicalDeviceFeatures f10{};
    f10.shaderInt16 = VK_TRUE;
    VkPhysicalDeviceVulkan11Features f11{};
    f11.storageBuffer16BitAccess = VK_TRUE;
    VkPhysicalDeviceVulkan12Features f12{};
    f12.storageBuffer8BitAccess = VK_TRUE;
    f12.shaderInt8    = VK_TRUE;
    f12.shaderFloat16 = VK_TRUE;
    VkPhysicalDeviceVulkan13Features f13{};
    f13.synchronization2 = VK_TRUE;
    f13.dynamicRendering = VK_TRUE;

    vkb::PhysicalDeviceSelector sel(inst.value());
    sel.set_minimum_version(1, 3)
       .prefer_gpu_device_type(vkb::PreferredDeviceType::discrete)
       .set_required_features(f10)
       .set_required_features_11(f11)
       .set_required_features_12(f12)
       .set_required_features_13(f13)
       .add_required_extension(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME);
    // WT_SHADER_STATS=1 prints the driver's per-kernel statistics (registers,
    // instruction counts) when pipelines are built: the way to see why a
    // kernel is slower than its memory traffic says it should be.
    statsEnabled_ = std::getenv("WT_SHADER_STATS") != nullptr;
    VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR pexec{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR};
    pexec.pipelineExecutableInfo = VK_TRUE;
    if (statsEnabled_)
        sel.add_required_extension(VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME)
           .add_required_extension_features(pexec);
    if (surface) sel.set_surface(surface);
    else         sel.defer_surface_initialization();
    auto pd = sel.select();
    if (!pd) throw std::runtime_error("no suitable GPU: " + pd.error().message());
    phys       = pd->physical_device;
    deviceName = pd->name;

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(phys, &props);
    timestampPeriodNs = props.limits.timestampPeriod;
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(phys, &mp);
    for (uint32_t i = 0; i < mp.memoryHeapCount; ++i)
        if (mp.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
            vramBytes = std::max(vramBytes, mp.memoryHeaps[i].size);

    auto dev = vkb::DeviceBuilder(pd.value()).build();
    if (!dev) throw std::runtime_error("device: " + dev.error().message());
    device = dev->device;
    auto q = dev->get_queue(vkb::QueueType::graphics);
    auto qi = dev->get_queue_index(vkb::QueueType::graphics);
    if (!q || !qi) throw std::runtime_error("no graphics/compute queue");
    queue  = q.value();
    family = qi.value();

    pushDescriptorSet_ = reinterpret_cast<PFN_vkCmdPushDescriptorSetKHR>(
        vkGetDeviceProcAddr(device, "vkCmdPushDescriptorSetKHR"));

    VmaAllocatorCreateInfo ai{};
    ai.instance         = instance;
    ai.physicalDevice   = phys;
    ai.device           = device;
    ai.vulkanApiVersion = VK_API_VERSION_1_3;
    check(vmaCreateAllocator(&ai, &vma), "vmaCreateAllocator");

    VkCommandPoolCreateInfo cp{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cp.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cp.queueFamilyIndex = family;
    check(vkCreateCommandPool(device, &cp, nullptr, &pool_), "command pool");
    VkFenceCreateInfo fc{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    check(vkCreateFence(device, &fc, nullptr, &fence_), "fence");

    // Shared compute layout.
    VkDescriptorSetLayoutBinding b[6]{};
    for (uint32_t i = 0; i < 6; ++i) {
        b[i].binding         = i;
        b[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        b[i].descriptorCount = i == 0 ? 19u : 1u;
        b[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo sl{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    sl.flags        = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
    sl.bindingCount = 6;
    sl.pBindings    = b;
    check(vkCreateDescriptorSetLayout(device, &sl, nullptr, &setLayout), "set layout");
    VkPushConstantRange pr{VK_SHADER_STAGE_COMPUTE_BIT, 0, 128};
    VkPipelineLayoutCreateInfo pl{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pl.setLayoutCount         = 1;
    pl.pSetLayouts            = &setLayout;
    pl.pushConstantRangeCount = 1;
    pl.pPushConstantRanges    = &pr;
    check(vkCreatePipelineLayout(device, &pl, nullptr, &layout), "pipeline layout");

    const auto dir = exeDir();
    shaderDir = (dir / "shaders").string();
    const auto cacheData = readFile(dir / "pipeline_cache.bin");
    VkPipelineCacheCreateInfo pc{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
    pc.initialDataSize = cacheData.size();
    pc.pInitialData    = cacheData.empty() ? nullptr : cacheData.data();
    if (vkCreatePipelineCache(device, &pc, nullptr, &pipelineCache) != VK_SUCCESS) {
        pc.initialDataSize = 0; pc.pInitialData = nullptr;      // stale cache: start fresh
        check(vkCreatePipelineCache(device, &pc, nullptr, &pipelineCache), "pipeline cache");
    }

    dummy_ = createBuffer(256, 0, Mem::Device);
}

void Context::destroy() {
    if (!device) return;
    vkDeviceWaitIdle(device);
    if (pipelineCache) {
        size_t sz = 0;
        vkGetPipelineCacheData(device, pipelineCache, &sz, nullptr);
        std::vector<char> data(sz);
        if (sz && vkGetPipelineCacheData(device, pipelineCache, &sz, data.data()) == VK_SUCCESS) {
            std::ofstream f(exeDir() / "pipeline_cache.bin", std::ios::binary);
            f.write(data.data(), std::streamsize(sz));
        }
        vkDestroyPipelineCache(device, pipelineCache, nullptr);
    }
    destroyBuffer(dummy_);
    vkDestroyPipelineLayout(device, layout, nullptr);
    vkDestroyDescriptorSetLayout(device, setLayout, nullptr);
    vkDestroyFence(device, fence_, nullptr);
    vkDestroyCommandPool(device, pool_, nullptr);
    vmaDestroyAllocator(vma);
    vkDestroyDevice(device, nullptr);
    if (surface) vkDestroySurfaceKHR(instance, surface, nullptr);
    if (messenger_) vkb::destroy_debug_utils_messenger(instance, messenger_);
    vkDestroyInstance(instance, nullptr);
    device = VK_NULL_HANDLE;
}

Buffer Context::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, Mem mem) {
    Buffer b;
    b.size = std::max<VkDeviceSize>(size, 16);
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size  = b.size;
    bi.usage = usage | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
             | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VmaAllocationCreateInfo ac{};
    ac.usage = VMA_MEMORY_USAGE_AUTO;
    if (mem == Mem::Device) {
        ac.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT * (b.size > (256ull << 20));
        ac.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    } else {
        ac.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT
                 | (mem == Mem::Upload ? VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
                                       : VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT);
        ac.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
    }
    VmaAllocationInfo info{};
    check(vmaCreateBuffer(vma, &bi, &ac, &b.buf, &b.alloc, &info), "vmaCreateBuffer");
    b.map = info.pMappedData;
    return b;
}

void Context::destroyBuffer(Buffer& b) {
    if (b.buf) vmaDestroyBuffer(vma, b.buf, b.alloc);
    b = {};
}

void Context::upload(const Buffer& dst, const void* data, VkDeviceSize size, VkDeviceSize offset) {
    Buffer st = createBuffer(size, 0, Mem::Upload);
    std::memcpy(st.map, data, size_t(size));
    vmaFlushAllocation(vma, st.alloc, 0, VK_WHOLE_SIZE);
    submitNow([&](VkCommandBuffer cmd) {
        VkBufferCopy c{0, offset, size};
        vkCmdCopyBuffer(cmd, st.buf, dst.buf, 1, &c);
    });
    destroyBuffer(st);
}

void Context::download(const Buffer& src, void* data, VkDeviceSize size, VkDeviceSize offset) {
    Buffer st = createBuffer(size, 0, Mem::Readback);
    submitNow([&](VkCommandBuffer cmd) {
        computeBarrier(cmd);
        VkBufferCopy c{offset, 0, size};
        vkCmdCopyBuffer(cmd, src.buf, st.buf, 1, &c);
    });
    vmaInvalidateAllocation(vma, st.alloc, 0, VK_WHOLE_SIZE);
    std::memcpy(data, st.map, size_t(size));
    destroyBuffer(st);
}

void Context::fill(const Buffer& dst, uint32_t value) {
    submitNow([&](VkCommandBuffer cmd) { vkCmdFillBuffer(cmd, dst.buf, 0, VK_WHOLE_SIZE, value); });
}

Kernel Context::loadKernel(const std::string& spvName, std::vector<uint32_t> spec) {
    const auto path = std::filesystem::path(shaderDir) / (spvName + ".spv");
    const auto code = readFile(path);
    if (code.empty()) throw std::runtime_error("missing shader " + path.string());
    VkShaderModuleCreateInfo mi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    mi.codeSize = code.size();
    mi.pCode    = reinterpret_cast<const uint32_t*>(code.data());
    VkShaderModule mod;
    check(vkCreateShaderModule(device, &mi, nullptr, &mod), "shader module");
    VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    ci.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    ci.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    ci.stage.module = mod;
    ci.stage.pName  = "main";
    std::vector<VkSpecializationMapEntry> entries(spec.size());
    for (uint32_t i = 0; i < spec.size(); ++i) entries[i] = {i, i * 4u, 4u};
    const VkSpecializationInfo si{uint32_t(spec.size()), entries.data(), spec.size() * 4, spec.data()};
    ci.stage.pSpecializationInfo = spec.empty() ? nullptr : &si;
    ci.layout       = layout;
    if (statsEnabled_) ci.flags |= VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR;
    Kernel k;
    check(vkCreateComputePipelines(device, statsEnabled_ ? VK_NULL_HANDLE : pipelineCache, 1, &ci,
                                   nullptr, &k.pipe), spvName.c_str());
    vkDestroyShaderModule(device, mod, nullptr);
    if (statsEnabled_) printStats(k.pipe, spvName + (spec.empty() ? "" : "/" + std::to_string(spec[0])));
    return k;
}

void Context::printStats(VkPipeline pipe, const std::string& name) const {
    auto getProps = reinterpret_cast<PFN_vkGetPipelineExecutablePropertiesKHR>(
        vkGetDeviceProcAddr(device, "vkGetPipelineExecutablePropertiesKHR"));
    auto getStats = reinterpret_cast<PFN_vkGetPipelineExecutableStatisticsKHR>(
        vkGetDeviceProcAddr(device, "vkGetPipelineExecutableStatisticsKHR"));
    if (!getProps || !getStats) return;
    VkPipelineInfoKHR pi{VK_STRUCTURE_TYPE_PIPELINE_INFO_KHR};
    pi.pipeline = pipe;
    uint32_t ne = 0;
    getProps(device, &pi, &ne, nullptr);
    for (uint32_t e = 0; e < ne; ++e) {
        VkPipelineExecutableInfoKHR ei{VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INFO_KHR};
        ei.pipeline = pipe;
        ei.executableIndex = e;
        uint32_t ns = 0;
        getStats(device, &ei, &ns, nullptr);
        std::vector<VkPipelineExecutableStatisticKHR> st(ns, {VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_STATISTIC_KHR});
        getStats(device, &ei, &ns, st.data());
        std::printf("[stats] %-16s", name.c_str());
        for (const auto& x : st) {
            if (x.format == VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_UINT64_KHR)
                std::printf("  %s=%llu", x.name, (unsigned long long)x.value.u64);
            else if (x.format == VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_INT64_KHR)
                std::printf("  %s=%lld", x.name, (long long)x.value.i64);
        }
        std::printf("\n");
    }
}

void Context::destroyKernel(Kernel& k) {
    if (k.pipe) vkDestroyPipeline(device, k.pipe, nullptr);
    k = {};
}

void Context::submitNow(const std::function<void(VkCommandBuffer)>& record) {
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool        = pool_;
    ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VkCommandBuffer cmd;
    check(vkAllocateCommandBuffers(device, &ai, &cmd), "allocate command buffer");
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bi);
    record(cmd);
    vkEndCommandBuffer(cmd);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers    = &cmd;
    check(vkQueueSubmit(queue, 1, &si, fence_), "queue submit");
    check(vkWaitForFences(device, 1, &fence_, VK_TRUE, UINT64_MAX), "wait fence");
    vkResetFences(device, 1, &fence_);
    vkFreeCommandBuffers(device, pool_, 1, &cmd);
}

void Context::bind(VkCommandBuffer cmd, const Kernel& k, const Bindings& b,
                   const void* push, uint32_t pushSize) const {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, k.pipe);
    VkDescriptorBufferInfo info[kSlots];
    for (uint32_t i = 0; i < kSlots; ++i)
        info[i] = {b[i] ? b[i] : dummy_.buf, 0, VK_WHOLE_SIZE};
    VkWriteDescriptorSet w[6]{};
    for (uint32_t i = 0; i < 6; ++i) {
        w[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[i].dstBinding      = i;
        w[i].descriptorCount = i == 0 ? 19u : 1u;
        w[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w[i].pBufferInfo     = i == 0 ? &info[0] : &info[18 + i];
    }
    pushDescriptorSet_(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 6, w);
    if (pushSize) vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, pushSize, push);
}

uint32_t Context::groupCount(uint64_t threads) {
    const uint64_t groups = std::max<uint64_t>((threads + 255) / 256, 1);
    const uint64_t gx = std::min<uint64_t>(groups, 65535);
    const uint64_t gy = (groups + gx - 1) / gx;
    return uint32_t(gx * gy);
}

void Context::dispatchThreads(VkCommandBuffer cmd, uint64_t threads) {
    const uint64_t groups = std::max<uint64_t>((threads + 255) / 256, 1);
    const uint64_t gx = std::min<uint64_t>(groups, 65535);
    const uint64_t gy = (groups + gx - 1) / gx;
    vkCmdDispatch(cmd, uint32_t(gx), uint32_t(gy), 1);
}

void Context::computeBarrier(VkCommandBuffer cmd) {
    VkMemoryBarrier2 mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    mb.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    mb.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT;
    mb.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    mb.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT
                     | VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT;
    VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    di.memoryBarrierCount = 1;
    di.pMemoryBarriers    = &mb;
    vkCmdPipelineBarrier2(cmd, &di);
}

Bindings Context::emptyBindings() const {
    Bindings b;
    b.fill(VK_NULL_HANDLE);
    return b;
}

} // namespace gpu
