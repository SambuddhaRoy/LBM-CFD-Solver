#pragma once
// ============================================================================
// vk.hpp — minimal Vulkan 1.3 context for a compute-heavy app.
//
// Every compute kernel shares one pipeline layout: a push-descriptor set with
//   binding 0  19 storage buffers (one per lattice direction)
//   binding 1-5 one storage buffer each
// plus 128 bytes of push constants. One layout means one bind helper and no
// descriptor pools to manage; unused slots are filled with a dummy buffer.
// ============================================================================

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include <array>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

struct GLFWwindow;

namespace gpu {

void check(VkResult r, const char* what);

enum class Mem { Device, Upload, Readback };

struct Buffer {
    VkBuffer      buf   = VK_NULL_HANDLE;
    VmaAllocation alloc = VK_NULL_HANDLE;
    void*         map   = nullptr;     // persistently mapped for Upload/Readback
    VkDeviceSize  size  = 0;
};

constexpr uint32_t kSlots = 24;        // 19 DDF + 5 single bindings
using Bindings = std::array<VkBuffer, kSlots>;

struct Kernel {
    VkPipeline pipe = VK_NULL_HANDLE;
};

class Context {
public:
    // window == nullptr: headless (no surface, no swapchain extension).
    void init(GLFWwindow* window);
    void destroy();

    Buffer createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, Mem mem);
    void   destroyBuffer(Buffer& b);
    void   upload(const Buffer& dst, const void* data, VkDeviceSize size, VkDeviceSize offset = 0);
    void   download(const Buffer& src, void* data, VkDeviceSize size, VkDeviceSize offset = 0);
    void   fill(const Buffer& dst, uint32_t value);

    // spec[i] sets specialization constant i (ignored by shaders without it).
    Kernel loadKernel(const std::string& spvName, std::vector<uint32_t> spec = {});
    void   destroyKernel(Kernel& k);

    // Record into a throwaway command buffer, submit, wait.
    void submitNow(const std::function<void(VkCommandBuffer)>& record);

    void bind(VkCommandBuffer cmd, const Kernel& k, const Bindings& b,
              const void* push, uint32_t pushSize) const;
    // Dispatch enough 256-wide groups for `threads`, folded into 2D so huge
    // grids stay under the 65535 groups-per-dimension portable limit.
    static void dispatchThreads(VkCommandBuffer cmd, uint64_t threads);
    static uint32_t groupCount(uint64_t threads);    // groups actually launched
    static void computeBarrier(VkCommandBuffer cmd);

    Bindings emptyBindings() const;

    VkInstance       instance = VK_NULL_HANDLE;
    VkPhysicalDevice phys     = VK_NULL_HANDLE;
    VkDevice         device   = VK_NULL_HANDLE;
    VkQueue          queue    = VK_NULL_HANDLE;
    uint32_t         family   = 0;
    VkSurfaceKHR     surface  = VK_NULL_HANDLE;
    VmaAllocator     vma      = VK_NULL_HANDLE;
    VkPipelineCache  pipelineCache = VK_NULL_HANDLE;
    VkPipelineLayout layout   = VK_NULL_HANDLE;
    VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
    std::string      deviceName;
    VkDeviceSize     vramBytes = 0;
    float            timestampPeriodNs = 1.f;
    std::string      shaderDir;

private:
    void printStats(VkPipeline pipe, const std::string& name) const;

    bool statsEnabled_ = false;
    VkDebugUtilsMessengerEXT messenger_ = VK_NULL_HANDLE;
    VkCommandPool pool_  = VK_NULL_HANDLE;
    VkFence       fence_ = VK_NULL_HANDLE;
    Buffer        dummy_;
    PFN_vkCmdPushDescriptorSetKHR pushDescriptorSet_ = nullptr;
};

} // namespace gpu
