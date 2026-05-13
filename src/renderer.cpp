// ============================================================================
// renderer.cpp — Velocity Field Visualization Renderer
// ============================================================================

#include "renderer.h"
#include <imgui.h>
#include <imgui_impl_vulkan.h>
#include <array>
#include <iostream>

namespace vwt {

void Renderer::init(VkDevice device, VmaAllocator allocator,
                    VkDescriptorPool imguiPool, VkPipelineCache pipelineCache,
                    const SimParams& params, VkBuffer macroBuffer)
{
    device_        = device;
    allocator_     = allocator;
    pipelineCache_ = pipelineCache;
    macroBuffer_   = macroBuffer;

    createSliceImage(params);
    createSlicePipeline();
    writeDescriptors();

    // Register with ImGui — write-once
    imguiTexture_ = ImGui_ImplVulkan_AddTexture(
        sliceSampler_, sliceImage_.imageView,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    std::cout << "[Renderer] Initialized " << sliceW_ << "x" << sliceH_ << " slice.\n";
}

void Renderer::destroy() { deletionQueue_.flush(); }

void Renderer::createSliceImage(const SimParams& params) {
    uint32_t maxDim = std::max({params.gridX, params.gridY, params.gridZ});
    sliceW_ = maxDim; sliceH_ = maxDim;

    VkImageCreateInfo ii{};
    ii.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ii.imageType     = VK_IMAGE_TYPE_2D;
    ii.format        = VK_FORMAT_R8G8B8A8_UNORM;
    ii.extent        = { sliceW_, sliceH_, 1 };
    ii.mipLevels     = 1;
    ii.arrayLayers   = 1;
    ii.samples       = VK_SAMPLE_COUNT_1_BIT;
    ii.tiling        = VK_IMAGE_TILING_OPTIMAL;
    ii.usage         = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    ii.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo ai{};
    ai.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

    sliceImage_.format = VK_FORMAT_R8G8B8A8_UNORM;
    sliceImage_.extent = ii.extent;
    VK_CHECK(vmaCreateImage(allocator_, &ii, &ai,
        &sliceImage_.image, &sliceImage_.allocation, nullptr));
    sliceImage_.layout = VK_IMAGE_LAYOUT_UNDEFINED;

    VkImageViewCreateInfo vi{};
    vi.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vi.image    = sliceImage_.image;
    vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vi.format   = VK_FORMAT_R8G8B8A8_UNORM;
    vi.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    VK_CHECK(vkCreateImageView(device_, &vi, nullptr, &sliceImage_.imageView));

    VkSamplerCreateInfo si{};
    si.sType     = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    si.magFilter = VK_FILTER_LINEAR;
    si.minFilter = VK_FILTER_LINEAR;
    si.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    VK_CHECK(vkCreateSampler(device_, &si, nullptr, &sliceSampler_));

    deletionQueue_.push([this](){
        vkDestroySampler(device_, sliceSampler_, nullptr);
        vkDestroyImageView(device_, sliceImage_.imageView, nullptr);
        vmaDestroyImage(allocator_, sliceImage_.image, sliceImage_.allocation);
    });
}

void Renderer::createSlicePipeline() {
    // Layout: binding 0 = macro SSBO, binding 1 = storage image
    std::array<VkDescriptorSetLayoutBinding, 2> b{};
    b[0] = { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
    b[1] = { 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };

    VkDescriptorSetLayoutCreateInfo li{};
    li.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    li.bindingCount = 2; li.pBindings = b.data();
    VK_CHECK(vkCreateDescriptorSetLayout(device_, &li, nullptr, &sliceLayout_));

    std::array<VkDescriptorPoolSize, 2> ps{};
    ps[0] = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1 };
    ps[1] = { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  1 };
    VkDescriptorPoolCreateInfo pi{};
    pi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pi.maxSets = 1; pi.poolSizeCount = 2; pi.pPoolSizes = ps.data();
    VK_CHECK(vkCreateDescriptorPool(device_, &pi, nullptr, &slicePool_));

    VkDescriptorSetAllocateInfo dai{};
    dai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dai.descriptorPool = slicePool_; dai.descriptorSetCount = 1; dai.pSetLayouts = &sliceLayout_;
    VK_CHECK(vkAllocateDescriptorSets(device_, &dai, &sliceSet_));

    VkPushConstantRange pcr{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(VisPushConstants) };
    VkPipelineLayoutCreateInfo pli{};
    pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pli.setLayoutCount = 1; pli.pSetLayouts = &sliceLayout_;
    pli.pushConstantRangeCount = 1; pli.pPushConstantRanges = &pcr;
    VK_CHECK(vkCreatePipelineLayout(device_, &pli, nullptr, &pipeLayout_));

    auto spirv = loadShaderModule("shaders/velocity_slice.comp.spv");
    VkShaderModuleCreateInfo smi{};
    smi.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smi.codeSize = spirv.size() * sizeof(uint32_t); smi.pCode = spirv.data();
    VkShaderModule sm;
    VK_CHECK(vkCreateShaderModule(device_, &smi, nullptr, &sm));

    VkComputePipelineCreateInfo ci{};
    ci.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    ci.layout = pipeLayout_;
    ci.stage  = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                  nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, sm, "main", nullptr };
    VK_CHECK(vkCreateComputePipelines(device_, pipelineCache_, 1, &ci, nullptr, &pipeline_));
    vkDestroyShaderModule(device_, sm, nullptr);

    deletionQueue_.push([this](){
        vkDestroyPipeline(device_, pipeline_, nullptr);
        vkDestroyPipelineLayout(device_, pipeLayout_, nullptr);
        vkDestroyDescriptorPool(device_, slicePool_, nullptr);
        vkDestroyDescriptorSetLayout(device_, sliceLayout_, nullptr);
    });
}

void Renderer::writeDescriptors() {
    // Write-once: no per-frame descriptor updates
    VkDescriptorBufferInfo bi{ macroBuffer_, 0, VK_WHOLE_SIZE };
    VkDescriptorImageInfo  ii{ VK_NULL_HANDLE, sliceImage_.imageView, VK_IMAGE_LAYOUT_GENERAL };

    std::array<VkWriteDescriptorSet, 2> writes{};
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = sliceSet_; writes[0].dstBinding = 0;
    writes[0].descriptorCount = 1; writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[0].pBufferInfo = &bi;

    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet = sliceSet_; writes[1].dstBinding = 1;
    writes[1].descriptorCount = 1; writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[1].pImageInfo = &ii;

    vkUpdateDescriptorSets(device_, 2, writes.data(), 0, nullptr);
}

void Renderer::rebindMacroBuffer(VkBuffer newMacro, const SimParams& params) {
    macroBuffer_ = newMacro;
    writeDescriptors();
}

void Renderer::computeSlice(VkCommandBuffer cmd, const SimParams& params) {
    // Transition to GENERAL if needed (track layout to avoid redundant barriers)
    if (sliceImage_.layout != VK_IMAGE_LAYOUT_GENERAL) {
        VkImageMemoryBarrier b{};
        b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        b.oldLayout = sliceImage_.layout;
        b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        b.image     = sliceImage_.image;
        b.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        b.srcAccessMask = (sliceImage_.layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
            ? VK_ACCESS_SHADER_READ_BIT : 0;
        b.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

        vkCmdPipelineBarrier(cmd,
            (sliceImage_.layout == VK_IMAGE_LAYOUT_UNDEFINED)
                ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT
                : VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 0, nullptr, 0, nullptr, 1, &b);
        sliceImage_.layout = VK_IMAGE_LAYOUT_GENERAL;
    }

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipeLayout_, 0, 1, &sliceSet_, 0, nullptr);

    VisPushConstants vpc{};
    vpc.gridX        = params.gridX;
    vpc.gridY        = params.gridY;
    vpc.gridZ        = params.gridZ;
    vpc.sliceAxis    = params.sliceAxis;
    vpc.sliceIndex   = params.sliceIndex;
    vpc.maxVelocity  = params.maxVelocity;
    vpc.visMode      = static_cast<uint32_t>(params.visMode);
    vpc.maxVorticity = params.maxVorticity;
    vkCmdPushConstants(cmd, pipeLayout_, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(VisPushConstants), &vpc);

    uint32_t gx = (sliceW_ + 15) / 16;
    uint32_t gy = (sliceH_ + 15) / 16;
    vkCmdDispatch(cmd, gx, gy, 1);

    // Transition to SHADER_READ_ONLY for ImGui
    VkImageMemoryBarrier tr{};
    tr.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    tr.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    tr.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    tr.image     = sliceImage_.image;
    tr.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    tr.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    tr.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    tr.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    tr.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &tr);
    sliceImage_.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
}

} // namespace vwt
