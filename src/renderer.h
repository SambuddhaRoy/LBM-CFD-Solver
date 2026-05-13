#pragma once
// ============================================================================
// renderer.h — Velocity Field Visualization Renderer
// ============================================================================

#include "vk_types.h"

namespace vwt {

class Renderer {
public:
    void init(VkDevice device, VmaAllocator allocator,
              VkDescriptorPool imguiPool, VkPipelineCache pipelineCache,
              const SimParams& params, VkBuffer macroBuffer);
    void destroy();

    // Record visualization compute shader into cmd
    void computeSlice(VkCommandBuffer cmd, const SimParams& params);

    // GPU timestamp for vis pass — slot 4 and 5 in the provided query pool
    void setTimestampPool(VkQueryPool pool) { externalTimestampPool_ = pool; }

    // ImGui texture handle for Image() call
    VkDescriptorSet getImGuiTexture() const { return imguiTexture_; }
    uint32_t sliceWidth()  const { return sliceW_; }
    uint32_t sliceHeight() const { return sliceH_; }

    void rebindMacroBuffer(VkBuffer newMacro, const SimParams& params);

private:
    void createSliceImage(const SimParams& params);
    void createSlicePipeline();
    void writeDescriptors();

    VkDevice        device_        = VK_NULL_HANDLE;
    VmaAllocator    allocator_     = VK_NULL_HANDLE;
    VkPipelineCache pipelineCache_ = VK_NULL_HANDLE;
    VkBuffer        macroBuffer_   = VK_NULL_HANDLE;
    VkQueryPool     externalTimestampPool_ = VK_NULL_HANDLE;

    AllocatedImage  sliceImage_;
    VkSampler       sliceSampler_  = VK_NULL_HANDLE;

    VkDescriptorSetLayout sliceLayout_ = VK_NULL_HANDLE;
    VkDescriptorPool      slicePool_   = VK_NULL_HANDLE;
    VkDescriptorSet       sliceSet_    = VK_NULL_HANDLE;
    VkPipelineLayout      pipeLayout_  = VK_NULL_HANDLE;
    VkPipeline            pipeline_    = VK_NULL_HANDLE;

    VkDescriptorSet imguiTexture_ = VK_NULL_HANDLE;

    uint32_t sliceW_ = 0, sliceH_ = 0;
    DeletionQueue deletionQueue_;
};

} // namespace vwt
