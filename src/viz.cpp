// ============================================================================
// viz.cpp — slice compute pass, ImGui texture, snapshot readback
// ============================================================================

#include "viz.h"

#include <imgui.h>
#include <imgui_impl_vulkan.h>

#include <array>
#include <cstring>

namespace vwt {

void SliceView::init(GpuContext& ctx, uint32_t gx, uint32_t gy, uint32_t gz,
                     uint32_t axis, VkBuffer macro, VkBuffer obstacle) {
    ctx_ = &ctx;
    rebuild(gx, gy, gz, axis, macro, obstacle);
}

void SliceView::rebuild(uint32_t gx, uint32_t gy, uint32_t gz,
                        uint32_t axis, VkBuffer macro, VkBuffer obstacle) {
    if (image_) {
        vkDeviceWaitIdle(ctx_->device());
        destroyResources();
    }
    gx_ = gx; gy_ = gy; gz_ = gz; axis_ = axis;
    switch (axis) {
    case 0:  w_ = gx; h_ = gy; break;   // XY
    case 1:  w_ = gx; h_ = gz; break;   // XZ
    default: w_ = gz; h_ = gy; break;   // ZY
    }
    createResources(macro, obstacle);
}

void SliceView::createResources(VkBuffer macro, VkBuffer obstacle) {
    VkDevice dev = ctx_->device();

    VkImageCreateInfo ii{};
    ii.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ii.imageType     = VK_IMAGE_TYPE_2D;
    ii.format        = VK_FORMAT_R8G8B8A8_UNORM;
    ii.extent        = { w_, h_, 1 };
    ii.mipLevels     = 1;
    ii.arrayLayers   = 1;
    ii.samples       = VK_SAMPLE_COUNT_1_BIT;
    ii.tiling        = VK_IMAGE_TILING_OPTIMAL;
    ii.usage         = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                       VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    ii.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo ai{};
    ai.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    VK_CHECK(vmaCreateImage(ctx_->allocator(), &ii, &ai, &image_, &alloc_, nullptr));
    layout_ = VK_IMAGE_LAYOUT_UNDEFINED;

    VkImageViewCreateInfo vi{};
    vi.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vi.image    = image_;
    vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vi.format   = VK_FORMAT_R8G8B8A8_UNORM;
    vi.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    VK_CHECK(vkCreateImageView(dev, &vi, nullptr, &view_));

    VkSamplerCreateInfo si{};
    si.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    si.magFilter    = VK_FILTER_LINEAR;
    si.minFilter    = VK_FILTER_LINEAR;
    si.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    VK_CHECK(vkCreateSampler(dev, &si, nullptr, &sampler_));

    // Descriptors: macro SSBO, obstacle SSBO, storage image
    std::array<VkDescriptorSetLayoutBinding, 3> b{};
    b[0] = { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
    b[1] = { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
    b[2] = { 2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };

    VkDescriptorSetLayoutCreateInfo li{};
    li.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    li.bindingCount = 3;
    li.pBindings    = b.data();
    VK_CHECK(vkCreateDescriptorSetLayout(dev, &li, nullptr, &setLayout_));

    std::array<VkDescriptorPoolSize, 2> ps{};
    ps[0] = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2 };
    ps[1] = { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  1 };
    VkDescriptorPoolCreateInfo pi{};
    pi.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pi.maxSets       = 1;
    pi.poolSizeCount = 2;
    pi.pPoolSizes    = ps.data();
    VK_CHECK(vkCreateDescriptorPool(dev, &pi, nullptr, &descPool_));

    VkDescriptorSetAllocateInfo dai{};
    dai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dai.descriptorPool     = descPool_;
    dai.descriptorSetCount = 1;
    dai.pSetLayouts        = &setLayout_;
    VK_CHECK(vkAllocateDescriptorSets(dev, &dai, &set_));

    VkDescriptorBufferInfo macInfo{ macro, 0, VK_WHOLE_SIZE };
    VkDescriptorBufferInfo obsInfo{ obstacle, 0, VK_WHOLE_SIZE };
    VkDescriptorImageInfo  imgInfo{ VK_NULL_HANDLE, view_, VK_IMAGE_LAYOUT_GENERAL };

    std::array<VkWriteDescriptorSet, 3> writes{};
    for (auto& w : writes) w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = set_; writes[0].dstBinding = 0; writes[0].descriptorCount = 1;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[0].pBufferInfo = &macInfo;
    writes[1].dstSet = set_; writes[1].dstBinding = 1; writes[1].descriptorCount = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[1].pBufferInfo = &obsInfo;
    writes[2].dstSet = set_; writes[2].dstBinding = 2; writes[2].descriptorCount = 1;
    writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[2].pImageInfo = &imgInfo;
    vkUpdateDescriptorSets(dev, 3, writes.data(), 0, nullptr);

    pipe_ = ctx_->makeComputePipeline("slice.comp.spv", setLayout_,
                                      sizeof(SlicePush), pipeLayout_);

    imguiTex_ = ImGui_ImplVulkan_AddTexture(
        sampler_, view_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
}

void SliceView::destroyResources() {
    VkDevice dev = ctx_->device();
    if (imguiTex_) { ImGui_ImplVulkan_RemoveTexture(imguiTex_); imguiTex_ = VK_NULL_HANDLE; }
    vkDestroyPipeline(dev, pipe_, nullptr);
    vkDestroyPipelineLayout(dev, pipeLayout_, nullptr);
    vkDestroyDescriptorPool(dev, descPool_, nullptr);
    vkDestroyDescriptorSetLayout(dev, setLayout_, nullptr);
    vkDestroySampler(dev, sampler_, nullptr);
    vkDestroyImageView(dev, view_, nullptr);
    vmaDestroyImage(ctx_->allocator(), image_, alloc_);
    image_ = VK_NULL_HANDLE;
}

void SliceView::destroy() {
    if (ctx_ && image_) destroyResources();
}

void SliceView::record(VkCommandBuffer cmd, const SlicePush& push, VkQueryPool pool) {
    if (pool) {
        vkCmdResetQueryPool(cmd, pool, 4, 2);
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, pool, 4);
    }

    // → GENERAL for compute write
    VkImageMemoryBarrier toGeneral{};
    toGeneral.sType         = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    toGeneral.oldLayout     = layout_;
    toGeneral.newLayout     = VK_IMAGE_LAYOUT_GENERAL;
    toGeneral.image         = image_;
    toGeneral.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    toGeneral.srcAccessMask = (layout_ == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
                              ? VK_ACCESS_SHADER_READ_BIT : 0;
    toGeneral.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    toGeneral.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toGeneral.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    vkCmdPipelineBarrier(cmd,
        (layout_ == VK_IMAGE_LAYOUT_UNDEFINED)
            ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT
            : VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &toGeneral);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipeLayout_, 0, 1, &set_, 0, nullptr);
    vkCmdPushConstants(cmd, pipeLayout_, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(push), &push);
    vkCmdDispatch(cmd, (w_ + 15) / 16, (h_ + 15) / 16, 1);

    // → SHADER_READ_ONLY for ImGui sampling
    VkImageMemoryBarrier toRead = toGeneral;
    toRead.oldLayout     = VK_IMAGE_LAYOUT_GENERAL;
    toRead.newLayout     = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    toRead.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    toRead.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &toRead);
    layout_ = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    if (pool)
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, pool, 5);
}

bool SliceView::readPixels(std::vector<uint8_t>& outRgba, uint32_t& w, uint32_t& h) {
    if (!image_ || layout_ == VK_IMAGE_LAYOUT_UNDEFINED) return false;
    vkDeviceWaitIdle(ctx_->device());

    const VkDeviceSize sz = VkDeviceSize(w_) * h_ * 4;
    GpuBuffer host = ctx_->createBuffer(sz, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                        MemLoc::HostRead);

    ctx_->oneShot([&](VkCommandBuffer cmd) {
        VkImageMemoryBarrier toSrc{};
        toSrc.sType         = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        toSrc.oldLayout     = layout_;
        toSrc.newLayout     = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        toSrc.image         = image_;
        toSrc.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        toSrc.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
        toSrc.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        toSrc.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toSrc.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 0, nullptr, 0, nullptr, 1, &toSrc);

        VkBufferImageCopy region{};
        region.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
        region.imageExtent      = { w_, h_, 1 };
        vkCmdCopyImageToBuffer(cmd, image_, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               host.buffer, 1, &region);

        VkImageMemoryBarrier back = toSrc;
        back.oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        back.newLayout     = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        back.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        back.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            0, 0, nullptr, 0, nullptr, 1, &back);
    });

    vmaInvalidateAllocation(ctx_->allocator(), host.alloc, 0, VK_WHOLE_SIZE);
    outRgba.resize(sz);
    std::memcpy(outRgba.data(), host.mapped, sz);
    ctx_->destroyBuffer(host);
    w = w_; h = h_;
    return true;
}

} // namespace vwt
