#pragma once
// ============================================================================
// viz.h — GPU slice visualization → ImGui texture
// ============================================================================

#include "gpu.h"
#include "sim.h"

namespace vwt {

class SliceView {
public:
    // Builds the slice image for the given grid + axis and registers it with
    // ImGui. Call rebuild() whenever the grid dimensions or axis change.
    void init(GpuContext& ctx, uint32_t gx, uint32_t gy, uint32_t gz,
              uint32_t axis, VkBuffer macro, VkBuffer obstacle);
    void rebuild(uint32_t gx, uint32_t gy, uint32_t gz,
                 uint32_t axis, VkBuffer macro, VkBuffer obstacle);
    void destroy();

    // Records the slice compute pass; image ends in SHADER_READ_ONLY layout.
    // Timestamps go to slots 4/5 of the provided query pool (may be null).
    void record(VkCommandBuffer cmd, const SlicePush& push, VkQueryPool pool);

    // Blocking copy of the current image into RGBA8 pixels (for snapshots).
    bool readPixels(std::vector<uint8_t>& outRgba, uint32_t& w, uint32_t& h);

    void*    textureId() const { return (void*)imguiTex_; }
    uint32_t width()  const { return w_; }
    uint32_t height() const { return h_; }
    uint32_t axis()   const { return axis_; }

private:
    void createResources(VkBuffer macro, VkBuffer obstacle);
    void destroyResources();

    GpuContext* ctx_ = nullptr;
    uint32_t gx_ = 0, gy_ = 0, gz_ = 0;
    uint32_t w_ = 0, h_ = 0, axis_ = 1;

    VkImage         image_   = VK_NULL_HANDLE;
    VmaAllocation   alloc_   = VK_NULL_HANDLE;
    VkImageView     view_    = VK_NULL_HANDLE;
    VkSampler       sampler_ = VK_NULL_HANDLE;
    VkImageLayout   layout_  = VK_IMAGE_LAYOUT_UNDEFINED;

    VkDescriptorSetLayout setLayout_  = VK_NULL_HANDLE;
    VkDescriptorPool      descPool_   = VK_NULL_HANDLE;
    VkDescriptorSet       set_        = VK_NULL_HANDLE;
    VkPipelineLayout      pipeLayout_ = VK_NULL_HANDLE;
    VkPipeline            pipe_       = VK_NULL_HANDLE;

    VkDescriptorSet imguiTex_ = VK_NULL_HANDLE;
};

} // namespace vwt
