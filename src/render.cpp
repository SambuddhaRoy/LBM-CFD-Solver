// ============================================================================
// render.cpp — viewport renderer (see render.hpp)
// ============================================================================

#include "render.hpp"

#include <imgui_impl_vulkan.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace wt {

namespace {

// Mirror of the Params block in render.comp.
struct RenderParams {
    uint32_t  dims[4];
    uint32_t  image[4];
    glm::vec4 origin, right, up, forward, slice, range, bodyLo, bodyHi;
};

} // namespace

const char* fieldName(Field f) {
    switch (f) {
    case Field::Speed:      return "Velocity";
    case Field::Pressure:   return "Pressure";
    case Field::Vorticity:  return "Vorticity";
    default:                return "Q-criterion";
    }
}

void Renderer::create(gpu::Context& ctx) {
    ctx_ = &ctx;
    kernel_ = ctx.loadKernel("render");
    VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    si.magFilter = si.minFilter = VK_FILTER_NEAREST;     // the image is drawn 1:1
    si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    gpu::check(vkCreateSampler(ctx.device, &si, nullptr, &sampler_), "sampler");
    for (auto& s : slots_) s.params = ctx.createBuffer(sizeof(RenderParams), 0, gpu::Mem::Upload);
}

void Renderer::release(Slot& s) {
    if (s.texture) ImGui_ImplVulkan_RemoveTexture(reinterpret_cast<VkDescriptorSet>(s.texture));
    if (s.imageView) vkDestroyImageView(ctx_->device, s.imageView, nullptr);
    if (s.image) vmaDestroyImage(ctx_->vma, s.image, s.alloc);
    ctx_->destroyBuffer(s.pixels);
    s.texture = 0; s.imageView = VK_NULL_HANDLE; s.image = VK_NULL_HANDLE; s.alloc = VK_NULL_HANDLE;
    s.width = s.height = 0;
}

void Renderer::destroy() {
    if (!ctx_) return;
    vkDeviceWaitIdle(ctx_->device);
    for (auto& s : slots_) { release(s); ctx_->destroyBuffer(s.params); }
    vkDestroySampler(ctx_->device, sampler_, nullptr);
    ctx_->destroyKernel(kernel_);
    ctx_ = nullptr;
}

void Renderer::resize(uint32_t slot, uint32_t w, uint32_t h) {
    Slot& s = slots_[slot];
    w = std::max(w, 1u); h = std::max(h, 1u);
    if (s.width == w && s.height == h) return;
    release(s);
    s.width = w; s.height = h;
    s.pixels = ctx_->createBuffer(VkDeviceSize(w) * h * 4, 0, gpu::Mem::Device);

    VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ii.imageType   = VK_IMAGE_TYPE_2D;
    ii.format      = VK_FORMAT_R8G8B8A8_UNORM;
    ii.extent      = {w, h, 1};
    ii.mipLevels   = 1;
    ii.arrayLayers = 1;
    ii.samples     = VK_SAMPLE_COUNT_1_BIT;
    ii.tiling      = VK_IMAGE_TILING_OPTIMAL;
    ii.usage       = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    VmaAllocationCreateInfo ac{};
    ac.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    gpu::check(vmaCreateImage(ctx_->vma, &ii, &ac, &s.image, &s.alloc, nullptr), "viewport image");
    VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vi.image    = s.image;
    vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vi.format   = ii.format;
    vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    gpu::check(vkCreateImageView(ctx_->device, &vi, nullptr, &s.imageView), "viewport view");

    // Start in the layout ImGui samples from, so the very first frame is valid.
    ctx_->submitNow([&](VkCommandBuffer cmd) {
        VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
        b.dstStageMask  = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        b.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
        b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        b.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        b.image = s.image;
        b.subresourceRange = vi.subresourceRange;
        VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        di.imageMemoryBarrierCount = 1;
        di.pImageMemoryBarriers = &b;
        vkCmdPipelineBarrier2(cmd, &di);
    });
    s.texture = reinterpret_cast<uint64_t>(
        ImGui_ImplVulkan_AddTexture(sampler_, s.imageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL));
}

void Renderer::record(VkCommandBuffer cmd, uint32_t slot, const Solver& solver, const View& v,
                      glm::vec3 bodyLo, glm::vec3 bodyHi) {
    Slot& s = slots_[slot];
    const auto& g = solver.grid();
    RenderParams p{};
    p.dims[0] = g.nx; p.dims[1] = g.ny; p.dims[2] = g.nz; p.dims[3] = uint32_t(v.field);
    p.image[0] = s.width; p.image[1] = s.height; p.image[2] = uint32_t(v.mode3d); p.image[3] = v.grid ? 1u : 0u;
    if (v.mode3d == 0) {
        p.origin = {v.center.x - 0.5f * float(s.width) * v.cellsPerPixel,
                    v.center.y + 0.5f * float(s.height) * v.cellsPerPixel, v.cellsPerPixel, 0.f};
    } else {
        const float yaw = glm::radians(v.yaw), pitch = glm::radians(v.pitch);
        const glm::vec3 offset = v.distance * glm::vec3(std::cos(pitch) * std::cos(yaw), std::sin(pitch),
                                                        std::cos(pitch) * std::sin(yaw));
        const glm::vec3 f = -glm::normalize(offset);
        const glm::vec3 r = glm::normalize(glm::cross(f, glm::vec3(0, 1, 0)));
        const glm::vec3 u = glm::cross(r, f);
        const float perPixel = 2.f * std::tan(glm::radians(v.fov) * 0.5f) / float(s.height);
        p.origin  = {v.target + offset, 0.f};
        p.right   = {r * perPixel, 0.f};
        p.up      = {u * perPixel, 0.f};
        p.forward = {f, 0.f};
    }
    p.slice  = {float(v.axis), v.slice, solver.flow.uIn, 0.f};
    p.range  = {v.lo, v.hi, 0.f, 0.f};
    p.bodyLo = {bodyLo, 0.f};
    p.bodyHi = {bodyHi, 0.f};
    std::memcpy(s.params.map, &p, sizeof p);
    vmaFlushAllocation(ctx_->vma, s.params.alloc, 0, VK_WHOLE_SIZE);

    gpu::Bindings b = solver.bindings();
    b[22] = s.params.buf;
    b[23] = s.pixels.buf;
    gpu::Context::computeBarrier(cmd);
    ctx_->bind(cmd, kernel_, b, nullptr, 0);
    vkCmdDispatch(cmd, (s.width + 15) / 16, (s.height + 15) / 16, 1);

    // pixels -> image, then back to the sampled layout for ImGui.
    VkImageMemoryBarrier2 ib{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    ib.srcStageMask  = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
    ib.srcAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
    ib.dstStageMask  = VK_PIPELINE_STAGE_2_COPY_BIT;
    ib.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    ib.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    ib.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    ib.image = s.image;
    ib.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VkMemoryBarrier2 mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    mb.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    mb.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
    mb.dstStageMask  = VK_PIPELINE_STAGE_2_COPY_BIT;
    mb.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
    VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    di.memoryBarrierCount = 1;      di.pMemoryBarriers = &mb;
    di.imageMemoryBarrierCount = 1; di.pImageMemoryBarriers = &ib;
    vkCmdPipelineBarrier2(cmd, &di);

    VkBufferImageCopy c{};
    c.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    c.imageExtent = {s.width, s.height, 1};
    vkCmdCopyBufferToImage(cmd, s.pixels.buf, s.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &c);

    ib.srcStageMask  = VK_PIPELINE_STAGE_2_COPY_BIT;
    ib.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    ib.dstStageMask  = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
    ib.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
    ib.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    ib.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    di.memoryBarrierCount = 0;
    vkCmdPipelineBarrier2(cmd, &di);
}

bool Renderer::savePng(uint32_t slot, const std::string& path) {
    const Slot& s = slots_[slot];
    if (!s.pixels.buf) return false;
    vkDeviceWaitIdle(ctx_->device);
    std::vector<uint32_t> px(size_t(s.width) * s.height);
    ctx_->download(s.pixels, px.data(), px.size() * 4);
    return stbi_write_png(path.c_str(), int(s.width), int(s.height), 4, px.data(), int(s.width) * 4) != 0;
}

} // namespace wt
