#pragma once
// ============================================================================
// gpu.h — Vulkan context, swapchain, and small RAII helpers
// ============================================================================

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include <cstdint>
#include <filesystem>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

struct GLFWwindow;

namespace vwt {

// ─── Error checking ──────────────────────────────────────────────────────────
inline void vkCheck(VkResult r, const char* what) {
    if (r != VK_SUCCESS)
        throw std::runtime_error(std::string(what) + " (VkResult=" + std::to_string(r) + ")");
}
#define VK_CHECK(call) ::vwt::vkCheck((call), #call)

// ─── Logging ─────────────────────────────────────────────────────────────────
void logMsg(const std::string& msg);

// ─── Filesystem anchors ──────────────────────────────────────────────────────
// Directory containing the running executable; shaders, config, cache, and
// snapshots all resolve relative to it so the app works from any CWD.
const std::filesystem::path& exeDir();

// ─── RAII deletion queue ─────────────────────────────────────────────────────
class DeletionQueue {
public:
    void push(std::function<void()>&& fn) { fns_.push_back(std::move(fn)); }
    void flush() {
        for (auto it = fns_.rbegin(); it != fns_.rend(); ++it) (*it)();
        fns_.clear();
    }
private:
    std::vector<std::function<void()>> fns_;
};

// ─── Buffer ──────────────────────────────────────────────────────────────────
enum class MemLoc { Device, HostWrite, HostRead };

struct GpuBuffer {
    VkBuffer      buffer = VK_NULL_HANDLE;
    VmaAllocation alloc  = VK_NULL_HANDLE;
    VkDeviceSize  size   = 0;
    void*         mapped = nullptr;   // non-null for Host* locations
};

// ─── GPU timing readback ─────────────────────────────────────────────────────
struct GpuTimings {
    float lbmMs      = 0.f;
    float analysisMs = 0.f;
    float sliceMs    = 0.f;
};

// ─── Context ─────────────────────────────────────────────────────────────────
class GpuContext {
public:
    // window == nullptr → headless (no surface; compute only)
    void init(GLFWwindow* window);
    void destroy();

    VkInstance       instance()    const { return instance_; }
    VkPhysicalDevice physDevice()  const { return physDevice_; }
    VkDevice         device()      const { return device_; }
    VkSurfaceKHR     surface()     const { return surface_; }
    VkQueue          queue()       const { return queue_; }
    uint32_t         queueFamily() const { return queueFamily_; }
    VmaAllocator     allocator()   const { return allocator_; }
    VkPipelineCache  cache()       const { return cache_; }
    const char*      gpuName()     const { return gpuName_; }
    float            timestampPeriodNs() const { return tsPeriodNs_; }
    uint32_t         memHeapCount() const { return memHeapCount_; }

    GpuBuffer createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, MemLoc loc);
    void      destroyBuffer(GpuBuffer& b);

    // Record + submit + wait. For setup transfers, not per-frame work.
    void oneShot(const std::function<void(VkCommandBuffer)>& record);

    // Compute pipeline from exeDir()/shaders/<spvName>
    VkPipeline makeComputePipeline(const char* spvName,
                                   VkDescriptorSetLayout setLayout,
                                   uint32_t pushSize,
                                   VkPipelineLayout& outLayout);

    void queryVram(uint64_t& usage, uint64_t& budget) const;
    void savePipelineCache();

private:
    VkInstance               instance_    = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT debugMsgr_   = VK_NULL_HANDLE;
    VkPhysicalDevice         physDevice_  = VK_NULL_HANDLE;
    VkDevice                 device_      = VK_NULL_HANDLE;
    VkSurfaceKHR             surface_     = VK_NULL_HANDLE;
    VkQueue                  queue_       = VK_NULL_HANDLE;
    uint32_t                 queueFamily_ = 0;
    VmaAllocator             allocator_   = VK_NULL_HANDLE;
    VkPipelineCache          cache_       = VK_NULL_HANDLE;
    VkCommandPool            oneShotPool_ = VK_NULL_HANDLE;
    char                     gpuName_[256] = "Unknown GPU";
    float                    tsPeriodNs_  = 1.f;
    uint32_t                 memHeapCount_ = 0;
};

// ─── Swapchain + render pass ─────────────────────────────────────────────────
class Swapchain {
public:
    void init(GpuContext& ctx, uint32_t w, uint32_t h);
    void recreate(uint32_t w, uint32_t h);
    void destroy();

    VkSwapchainKHR handle()     const { return swapchain_; }
    VkRenderPass   renderPass() const { return renderPass_; }
    VkFramebuffer  framebuffer(uint32_t i) const { return framebuffers_[i]; }
    VkExtent2D     extent()     const { return extent_; }
    uint32_t       imageCount() const { return uint32_t(images_.size()); }

private:
    void create(uint32_t w, uint32_t h);
    void destroyResources();

    GpuContext*              ctx_ = nullptr;
    VkSwapchainKHR           swapchain_  = VK_NULL_HANDLE;
    VkFormat                 format_     = VK_FORMAT_UNDEFINED;
    VkExtent2D               extent_     = {};
    VkRenderPass             renderPass_ = VK_NULL_HANDLE;
    std::vector<VkImage>     images_;
    std::vector<VkImageView> views_;
    std::vector<VkFramebuffer> framebuffers_;
};

} // namespace vwt
