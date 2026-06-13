// ============================================================================
// sim.cpp — D3Q19 LBM solver: buffers, pipelines, dispatch, readback
// ============================================================================

#include "sim.h"

#include <array>
#include <cmath>
#include <cstring>

namespace vwt {

// ════════════════════════════════════════════════════════════════════════════
// Init / teardown
// ════════════════════════════════════════════════════════════════════════════

void Solver::init(GpuContext& ctx, const SimParams& p) {
    ctx_ = &ctx;
    gx_ = p.gx; gy_ = p.gy; gz_ = p.gz;

    const size_t       n     = cells();
    const VkDeviceSize fSz   = n * 19 * sizeof(float);
    const VkDeviceSize obSz  = n * sizeof(uint32_t);
    const VkDeviceSize macSz = n * 4 * sizeof(float);
    const VkDeviceSize parSz = kGroups * kVals * sizeof(float);

    const auto ssbo = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    fA_       = ctx.createBuffer(fSz,  ssbo | VK_BUFFER_USAGE_TRANSFER_DST_BIT, MemLoc::Device);
    fB_       = ctx.createBuffer(fSz,  ssbo | VK_BUFFER_USAGE_TRANSFER_DST_BIT, MemLoc::Device);
    obstacle_ = ctx.createBuffer(obSz, ssbo | VK_BUFFER_USAGE_TRANSFER_DST_BIT, MemLoc::Device);
    sdf_      = ctx.createBuffer(macSz / 4, ssbo | VK_BUFFER_USAGE_TRANSFER_DST_BIT, MemLoc::Device);
    macro_    = ctx.createBuffer(macSz, ssbo | VK_BUFFER_USAGE_TRANSFER_DST_BIT
                                            | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, MemLoc::Device);
    prev_     = ctx.createBuffer(macSz, ssbo | VK_BUFFER_USAGE_TRANSFER_DST_BIT, MemLoc::Device);
    staging_  = ctx.createBuffer(fSz,  VK_BUFFER_USAGE_TRANSFER_SRC_BIT, MemLoc::HostWrite);
    for (uint32_t s = 0; s < kSlots; ++s) {
        partial_[s]  = ctx.createBuffer(parSz, ssbo | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, MemLoc::Device);
        readback_[s] = ctx.createBuffer(parSz, VK_BUFFER_USAGE_TRANSFER_DST_BIT, MemLoc::HostRead);
    }

    dq_.push([this] {
        ctx_->destroyBuffer(fA_);       ctx_->destroyBuffer(fB_);
        ctx_->destroyBuffer(obstacle_); ctx_->destroyBuffer(macro_);
        ctx_->destroyBuffer(sdf_);
        ctx_->destroyBuffer(prev_);     ctx_->destroyBuffer(staging_);
        for (uint32_t s = 0; s < kSlots; ++s) {
            ctx_->destroyBuffer(partial_[s]);
            ctx_->destroyBuffer(readback_[s]);
        }
    });

    // ── Descriptor layouts: LBM (5 SSBOs), analysis (4 SSBOs) ──────────────
    auto makeLayout = [&](uint32_t count, VkDescriptorSetLayout& out) {
        std::array<VkDescriptorSetLayoutBinding, 5> b{};
        for (uint32_t i = 0; i < count; ++i)
            b[i] = { i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                     VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
        VkDescriptorSetLayoutCreateInfo li{};
        li.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        li.bindingCount = count;
        li.pBindings    = b.data();
        VK_CHECK(vkCreateDescriptorSetLayout(ctx_->device(), &li, nullptr, &out));
    };
    makeLayout(5, lbmLayout_);
    makeLayout(4, anaLayout_);

    VkDescriptorPoolSize ps{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 20 };
    VkDescriptorPoolCreateInfo pi{};
    pi.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pi.maxSets       = 4;
    pi.poolSizeCount = 1;
    pi.pPoolSizes    = &ps;
    VK_CHECK(vkCreateDescriptorPool(ctx_->device(), &pi, nullptr, &descPool_));

    {
        VkDescriptorSetLayout layouts[4] = { lbmLayout_, lbmLayout_, anaLayout_, anaLayout_ };
        VkDescriptorSetAllocateInfo ai{};
        ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorPool     = descPool_;
        ai.descriptorSetCount = 4;
        ai.pSetLayouts        = layouts;
        VkDescriptorSet sets[4];
        VK_CHECK(vkAllocateDescriptorSets(ctx_->device(), &ai, sets));
        lbmSetA_ = sets[0]; lbmSetB_ = sets[1];
        anaSet_[0] = sets[2]; anaSet_[1] = sets[3];
    }

    auto write = [&](VkDescriptorSet set, uint32_t binding, VkBuffer buf) {
        VkDescriptorBufferInfo bi{ buf, 0, VK_WHOLE_SIZE };
        VkWriteDescriptorSet w{};
        w.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet          = set;
        w.dstBinding      = binding;
        w.descriptorCount = 1;
        w.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w.pBufferInfo     = &bi;
        vkUpdateDescriptorSets(ctx_->device(), 1, &w, 0, nullptr);
    };
    // A→B
    write(lbmSetA_, 0, fA_.buffer); write(lbmSetA_, 1, fB_.buffer);
    write(lbmSetA_, 2, obstacle_.buffer); write(lbmSetA_, 3, macro_.buffer);
    write(lbmSetA_, 4, sdf_.buffer);
    // B→A
    write(lbmSetB_, 0, fB_.buffer); write(lbmSetB_, 1, fA_.buffer);
    write(lbmSetB_, 2, obstacle_.buffer); write(lbmSetB_, 3, macro_.buffer);
    write(lbmSetB_, 4, sdf_.buffer);
    // analysis, one set per in-flight slot (independent partial buffers)
    for (uint32_t s = 0; s < kSlots; ++s) {
        write(anaSet_[s], 0, macro_.buffer); write(anaSet_[s], 1, obstacle_.buffer);
        write(anaSet_[s], 2, prev_.buffer);  write(anaSet_[s], 3, partial_[s].buffer);
    }

    dq_.push([this] {
        vkDestroyDescriptorPool(ctx_->device(), descPool_, nullptr);
        vkDestroyDescriptorSetLayout(ctx_->device(), lbmLayout_, nullptr);
        vkDestroyDescriptorSetLayout(ctx_->device(), anaLayout_, nullptr);
    });

    // ── Pipelines ───────────────────────────────────────────────────────────
    lbmPipe_ = ctx.makeComputePipeline("lbm.comp.spv", lbmLayout_,
                                       sizeof(LbmPush), lbmPipeLayout_);
    anaPipe_ = ctx.makeComputePipeline("analysis.comp.spv", anaLayout_,
                                       sizeof(AnalysisPush), anaPipeLayout_);
    dq_.push([this] {
        vkDestroyPipeline(ctx_->device(), lbmPipe_, nullptr);
        vkDestroyPipelineLayout(ctx_->device(), lbmPipeLayout_, nullptr);
        vkDestroyPipeline(ctx_->device(), anaPipe_, nullptr);
        vkDestroyPipelineLayout(ctx_->device(), anaPipeLayout_, nullptr);
    });

    // ── Timestamp queries ───────────────────────────────────────────────────
    VkQueryPoolCreateInfo qi{};
    qi.sType      = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    qi.queryType  = VK_QUERY_TYPE_TIMESTAMP;
    qi.queryCount = 6;
    VK_CHECK(vkCreateQueryPool(ctx_->device(), &qi, nullptr, &queryPool_));
    dq_.push([this] { vkDestroyQueryPool(ctx_->device(), queryPool_, nullptr); });

    reset();

    const size_t mb = size_t(fSz * 2 + obSz + macSz * 2 + parSz) / (1024 * 1024);
    logMsg("Solver: grid " + std::to_string(gx_) + "x" + std::to_string(gy_) +
           "x" + std::to_string(gz_) + ", GPU memory ~" + std::to_string(mb) + " MB");
}

void Solver::destroy() {
    dq_.flush();
    ctx_ = nullptr;
}

// ════════════════════════════════════════════════════════════════════════════
// Data upload / reset
// ════════════════════════════════════════════════════════════════════════════

void Solver::uploadObstacles(const std::vector<uint32_t>& occ) {
    const VkDeviceSize sz = occ.size() * sizeof(uint32_t);
    std::memcpy(staging_.mapped, occ.data(), sz);
    vmaFlushAllocation(ctx_->allocator(), staging_.alloc, 0, sz);
    ctx_->oneShot([&](VkCommandBuffer cmd) {
        VkBufferCopy cr{ 0, 0, sz };
        vkCmdCopyBuffer(cmd, staging_.buffer, obstacle_.buffer, 1, &cr);
    });

    // Default signed-distance field: +/- half a cell from the occupancy sign.
    // At |phi|=0.5 the Bouzidi interpolation reduces to simple halfway
    // bounce-back, so this is a safe default for grid-aligned bodies. Callers
    // with an exact surface (analytic primitives) override via setSDF().
    std::vector<float> sdf(occ.size());
    for (size_t i = 0; i < occ.size(); ++i) sdf[i] = occ[i] ? -0.5f : 0.5f;
    setSDF(sdf);
}

void Solver::setSDF(const std::vector<float>& phi) {
    const VkDeviceSize sz = phi.size() * sizeof(float);
    std::memcpy(staging_.mapped, phi.data(), sz);
    vmaFlushAllocation(ctx_->allocator(), staging_.alloc, 0, sz);
    ctx_->oneShot([&](VkCommandBuffer cmd) {
        VkBufferCopy cr{ 0, 0, sz };
        vkCmdCopyBuffer(cmd, staging_.buffer, sdf_.buffer, 1, &cr);
    });
}

void Solver::reset() {
    const size_t       n   = cells();
    const VkDeviceSize fSz = n * 19 * sizeof(float);

    static constexpr float w[19] = {
        1.f/3.f,
        1.f/18.f,1.f/18.f,1.f/18.f,1.f/18.f,1.f/18.f,1.f/18.f,
        1.f/36.f,1.f/36.f,1.f/36.f,1.f/36.f,
        1.f/36.f,1.f/36.f,1.f/36.f,1.f/36.f,
        1.f/36.f,1.f/36.f,1.f/36.f,1.f/36.f,
    };

    // SoA: q-major so each plane is one memset-like fill of a single weight
    float* dst = static_cast<float*>(staging_.mapped);
    for (int q = 0; q < 19; ++q)
        for (size_t c = 0; c < n; ++c)
            dst[size_t(q) * n + c] = w[q];
    vmaFlushAllocation(ctx_->allocator(), staging_.alloc, 0, fSz);

    ctx_->oneShot([&](VkCommandBuffer cmd) {
        VkBufferCopy cr{ 0, 0, fSz };
        vkCmdCopyBuffer(cmd, staging_.buffer, fA_.buffer, 1, &cr);
        vkCmdCopyBuffer(cmd, staging_.buffer, fB_.buffer, 1, &cr);
        vkCmdFillBuffer(cmd, macro_.buffer, 0, VK_WHOLE_SIZE, 0);
        vkCmdFillBuffer(cmd, prev_.buffer,  0, VK_WHOLE_SIZE, 0);
    });
    pingPong_ = false;
}

// ════════════════════════════════════════════════════════════════════════════
// Recording
// ════════════════════════════════════════════════════════════════════════════

void Solver::recordSteps(VkCommandBuffer cmd, const SimParams& p,
                         uint32_t nSteps, uint32_t stepBase, bool timestamps) {
    if (timestamps) {
        vkCmdResetQueryPool(cmd, queryPool_, 0, 2);
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queryPool_, 0);
    }

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, lbmPipe_);

    LbmPush push{};
    push.gx = gx_; push.gy = gy_; push.gz = gz_;
    push.tau       = p.tau;
    push.uIn       = p.uIn;
    push.turb      = p.turb;
    push.collision = uint32_t(p.collision);
    push.les       = p.les ? 1u : 0u;
    push.csSmago   = p.csSmago;

    const uint32_t dx = (gx_ + 7) / 8, dy = (gy_ + 7) / 8, dz = (gz_ + 3) / 4;

    VkMemoryBarrier mb{};
    mb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    for (uint32_t i = 0; i < nSteps; ++i) {
        VkDescriptorSet set = pingPong_ ? lbmSetB_ : lbmSetA_;
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                lbmPipeLayout_, 0, 1, &set, 0, nullptr);
        push.time = float(stepBase + i);
        vkCmdPushConstants(cmd, lbmPipeLayout_, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(push), &push);
        vkCmdDispatch(cmd, dx, dy, dz);
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 1, &mb, 0, nullptr, 0, nullptr);
        pingPong_ = !pingPong_;
    }

    if (timestamps)
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPool_, 1);
}

void Solver::recordAnalysis(VkCommandBuffer cmd, const SimParams& p, uint32_t slot) {
    slot %= kSlots;
    vkCmdResetQueryPool(cmd, queryPool_, 2, 2);
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queryPool_, 2);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, anaPipe_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            anaPipeLayout_, 0, 1, &anaSet_[slot], 0, nullptr);

    AnalysisPush push{ p.gx, p.gy, p.gz, 0 };
    vkCmdPushConstants(cmd, anaPipeLayout_, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(push), &push);
    vkCmdDispatch(cmd, kGroups, 1, 1);

    VkMemoryBarrier mb{};
    mb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 1, &mb, 0, nullptr, 0, nullptr);

    VkBufferCopy cr{ 0, 0, kGroups * kVals * sizeof(float) };
    vkCmdCopyBuffer(cmd, partial_[slot].buffer, readback_[slot].buffer, 1, &cr);

    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPool_, 3);
}

// ════════════════════════════════════════════════════════════════════════════
// CPU readback
// ════════════════════════════════════════════════════════════════════════════

Analysis Solver::readAnalysis(uint32_t slot) const {
    slot %= kSlots;
    Analysis a;
    if (!readback_[slot].mapped) return a;
    vmaInvalidateAllocation(ctx_->allocator(), readback_[slot].alloc, 0, VK_WHOLE_SIZE);

    const float* part = static_cast<const float*>(readback_[slot].mapped);
    double acc[kVals] = {};
    for (uint32_t g = 0; g < kGroups; ++g) {
        for (uint32_t v = 0; v < kVals; ++v) {
            const double x = part[g * kVals + v];
            if (v == 6) acc[v] = std::max(acc[v], x);
            else        acc[v] += x;
        }
    }

    const double sumDelta = acc[0], sumU2 = acc[1];
    a.residual   = (sumU2 > 1e-12) ? float(std::sqrt(sumDelta / sumU2)) : 1.f;
    a.drag       = float(acc[2]);
    a.lift       = float(acc[3]);
    a.side       = float(acc[4]);
    a.fluidCells = float(acc[8]);
    a.massAvg    = (acc[8] > 0.5) ? float(acc[5] / acc[8]) : 0.f;
    a.maxU       = float(acc[6]);
    a.ke         = float(acc[7]);
    a.surfFaces  = float(acc[9]);
    a.valid      = std::isfinite(a.residual) && std::isfinite(a.drag);
    return a;
}

float Solver::readTimestampMs(uint32_t firstSlot) const {
    uint64_t ts[2] = {};
    if (vkGetQueryPoolResults(ctx_->device(), queryPool_, firstSlot, 2,
            sizeof(ts), ts, sizeof(uint64_t),
            VK_QUERY_RESULT_64_BIT) != VK_SUCCESS)
        return 0.f;
    return float(double(ts[1] - ts[0]) * ctx_->timestampPeriodNs() / 1e6);
}

} // namespace vwt
