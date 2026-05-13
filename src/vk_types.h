#pragma once
// ============================================================================
// vk_types.h — Core types, structs, and Vulkan utility helpers
// ============================================================================

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>
#include <glm/glm.hpp>

#include <cstdint>
#include <vector>
#include <string>
#include <array>
#include <span>
#include <memory>
#include <functional>
#include <optional>
#include <stdexcept>
#include <cassert>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <atomic>
#include <thread>
#include <mutex>

namespace vwt {

// ─── Constants ──────────────────────────────────────────────────────────────
static constexpr uint32_t FRAMES_IN_FLIGHT = 2;

// ─── RAII deletion queue ─────────────────────────────────────────────────────
class DeletionQueue {
public:
    void push(std::function<void()>&& fn) { deletors_.push_back(std::move(fn)); }
    void flush() {
        for (auto it = deletors_.rbegin(); it != deletors_.rend(); ++it) (*it)();
        deletors_.clear();
    }
    ~DeletionQueue() { flush(); }
private:
    std::vector<std::function<void()>> deletors_;
};

// ─── Allocated Buffer ────────────────────────────────────────────────────────
struct AllocatedBuffer {
    VkBuffer      buffer     = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    VkDeviceSize  size       = 0;
    void*         mappedPtr  = nullptr;  // non-null when persistently mapped
};

// ─── Allocated Image ─────────────────────────────────────────────────────────
struct AllocatedImage {
    VkImage       image      = VK_NULL_HANDLE;
    VkImageView   imageView  = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    VkExtent3D    extent     = {};
    VkFormat      format     = VK_FORMAT_UNDEFINED;
    VkImageLayout layout     = VK_IMAGE_LAYOUT_UNDEFINED;
};

// ─── Per-frame GPU resources ──────────────────────────────────────────────────
struct FrameData {
    VkCommandPool   commandPool    = VK_NULL_HANDLE;
    VkCommandBuffer commandBuffer  = VK_NULL_HANDLE;
    VkSemaphore     presentSemaphore = VK_NULL_HANDLE;
    VkSemaphore     renderSemaphore  = VK_NULL_HANDLE;
    VkFence         renderFence    = VK_NULL_HANDLE;
};

// ─── GPU timing readback ──────────────────────────────────────────────────────
struct GpuTimings {
    float lbmMs  = 0.f;
    float visMs  = 0.f;
    float aeroMs = 0.f;
};

// ─── Aerodynamic force readback ───────────────────────────────────────────────
struct AeroForces {
    float drag  = 0.f;   // along +X (flow direction)
    float lift  = 0.f;   // along +Y
    float side  = 0.f;   // along +Z
    float _pad  = 0.f;
};

// ─── Visualization mode ───────────────────────────────────────────────────────
enum class VisMode : uint32_t {
    Velocity   = 0,
    Pressure   = 1,
    Vorticity  = 2,
    QCriterion = 3,
    Count      = 4
};

// ─── Simulation parameters ───────────────────────────────────────────────────
struct SimParams {
    uint32_t gridX     = 128;
    uint32_t gridY     = 64;
    uint32_t gridZ     = 64;
    float    tau       = 0.6f;
    float    inletVelX = 0.05f;
    float    inletVelY = 0.0f;
    float    inletVelZ = 0.0f;
    uint32_t sliceAxis  = 1;      // 0=XY, 1=XZ, 2=YZ
    uint32_t sliceIndex = 32;
    float    maxVelocity   = 0.15f;
    float    maxVorticity  = 0.05f;  // for vorticity colormap scale
    float    turbulence    = 0.0f;
    uint32_t currentEnvironmentIndex = 0;
    int      lbmMode  = 0;           // 0=BGK, 1=MRT
    float    s_bulk   = 1.2f;
    float    s_ghost  = 1.5f;
    VisMode  visMode  = VisMode::Velocity;
};

// ─── LBM push constants (must match shader layout) ───────────────────────────
struct LBMPushConstants {
    uint32_t gridX;
    uint32_t gridY;
    uint32_t gridZ;
    float    tau;
    float    inletVelX;
    float    inletVelY;
    float    inletVelZ;
    float    time;
    float    turbulence;
    float    s_bulk;
    float    s_ghost;
    uint32_t lbmMode;
};

// ─── Visualization push constants ────────────────────────────────────────────
struct VisPushConstants {
    uint32_t gridX;
    uint32_t gridY;
    uint32_t gridZ;
    uint32_t sliceAxis;
    uint32_t sliceIndex;
    float    maxVelocity;
    uint32_t visMode;
    float    maxVorticity;
};

// ─── Aero forces push constants ──────────────────────────────────────────────
struct AeroPushConstants {
    uint32_t gridX;
    uint32_t gridY;
    uint32_t gridZ;
    float    inletVelX;
};

// ─── SPIR-V loader ────────────────────────────────────────────────────────────
inline std::vector<uint32_t> loadShaderModule(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open())
        throw std::runtime_error("Failed to open shader: " + path.string());
    size_t sz = static_cast<size_t>(file.tellg());
    std::vector<uint32_t> buf(sz / sizeof(uint32_t));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(buf.data()), sz);
    return buf;
}

// ─── VK_CHECK helper ─────────────────────────────────────────────────────────
inline void vkCheck(VkResult res, const char* msg) {
    if (res != VK_SUCCESS)
        throw std::runtime_error(std::string(msg) + " (VkResult=" + std::to_string(res) + ")");
}
#define VK_CHECK(call) vwt::vkCheck((call), #call)

} // namespace vwt
