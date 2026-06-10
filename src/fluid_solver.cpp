// ============================================================================
// fluid_solver.cpp — GPU D3Q19 LBM + Aero Force Integration
// ============================================================================

#include "fluid_solver.h"
#include <cstring>
#include <iostream>
#include <array>
#include <cmath>

namespace vwt {

// ════════════════════════════════════════════════════════════════════════════
// Initialization
// ════════════════════════════════════════════════════════════════════════════

void FluidSolver::init(VkDevice device, VmaAllocator allocator,
                       VkQueue computeQueue, uint32_t computeQueueFamily,
                       VkPipelineCache pipelineCache, const SimParams& params)
{
    device_        = device;
    allocator_     = allocator;
    queue_         = computeQueue;
    queueFamily_   = computeQueueFamily;
    pipelineCache_ = pipelineCache;
    gridX_         = params.gridX;
    gridY_         = params.gridY;
    gridZ_         = params.gridZ;

    // Query timestamp period
    VkPhysicalDevice physDev;
    {
        VmaAllocatorInfo info;
        vmaGetAllocatorInfo(allocator_, &info);
        physDev = info.physicalDevice;
    }
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(physDev, &props);
    timestampPeriodNs_ = props.limits.timestampPeriod;

    createCommandPool();
    createBuffers();
    createDescriptorSets();
    createLBMPipeline();
    createAeroPipeline();
    createResidualPipeline();
    createTimestampPool();
    resetToEquilibrium();

    size_t totalMB = (fBufferA_.size + fBufferB_.size +
                      obstacleBuffer_.size + macroBuffer_.size) / (1024*1024);
    std::cout << "[FluidSolver] Grid " << gridX_ << "x" << gridY_ << "x" << gridZ_
              << "  GPU mem ~" << totalMB << " MB\n";
}

void FluidSolver::destroy() { deletionQueue_.flush(); }

// ════════════════════════════════════════════════════════════════════════════
// One-shot command helper
// ════════════════════════════════════════════════════════════════════════════

void FluidSolver::submitOneShot(std::function<void(VkCommandBuffer)>&& record) {
    VkCommandBufferAllocateInfo ai{};
    ai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool        = transferPool_;
    ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(device_, &ai, &cmd);

    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bi);
    record(cmd);
    vkEndCommandBuffer(cmd);

    VkSubmitInfo si{};
    si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers    = &cmd;
    vkQueueSubmit(queue_, 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(queue_);
    vkFreeCommandBuffers(device_, transferPool_, 1, &cmd);
}

// ════════════════════════════════════════════════════════════════════════════
// Command pool
// ════════════════════════════════════════════════════════════════════════════

void FluidSolver::createCommandPool() {
    VkCommandPoolCreateInfo pi{};
    pi.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pi.queueFamilyIndex = queueFamily_;
    pi.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VK_CHECK(vkCreateCommandPool(device_, &pi, nullptr, &transferPool_));
    deletionQueue_.push([this](){ vkDestroyCommandPool(device_, transferPool_, nullptr); });
}

// ════════════════════════════════════════════════════════════════════════════
// Buffers
// ════════════════════════════════════════════════════════════════════════════

void FluidSolver::createBuffers() {
    size_t cells     = totalCells();
    VkDeviceSize fSz = cells * 19 * sizeof(float);   // SoA f[q][cell]
    VkDeviceSize obSz = cells * sizeof(uint32_t);
    VkDeviceSize macSz = cells * 4 * sizeof(float);  // [rho, ux, uy, uz]

    // GPU-local storage buffers
    auto makeGpu = [&](VkDeviceSize sz, VkBufferUsageFlags extra) -> AllocatedBuffer {
        VkBufferCreateInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bi.size  = sz;
        bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | extra;
        VmaAllocationCreateInfo ai{};
        ai.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        AllocatedBuffer buf; buf.size = sz;
        VK_CHECK(vmaCreateBuffer(allocator_, &bi, &ai, &buf.buffer, &buf.allocation, nullptr));
        return buf;
    };

    fBufferA_       = makeGpu(fSz,  0);
    fBufferB_       = makeGpu(fSz,  0);
    obstacleBuffer_ = makeGpu(obSz, 0);
    macroBuffer_    = makeGpu(macSz, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);

    // Aero partial sums — GPU local
    VkDeviceSize aeroPartSz = kAeroGroups * 4 * sizeof(float);
    aeroPartialBuffer_ = makeGpu(aeroPartSz, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);

    // Aero readback — persistently mapped CPU-visible
    {
        VkBufferCreateInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bi.size  = aeroPartSz;
        bi.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        VmaAllocationCreateInfo ai{};
        ai.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
        ai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
        VmaAllocationInfo allocInfo;
        aeroReadbackBuffer_.size = aeroPartSz;
        VK_CHECK(vmaCreateBuffer(allocator_, &bi, &ai,
            &aeroReadbackBuffer_.buffer, &aeroReadbackBuffer_.allocation, &allocInfo));
        aeroReadbackBuffer_.mappedPtr = allocInfo.pMappedData;
    }

    // Convergence-residual buffers
    //   prevVelBuffer_:    previous-sample velocity field (same layout as macro)
    //   residualPartial:   GPU partial sums (256 groups × 2 floats)
    //   residualReadback:  CPU-visible copy of the partial sums
    prevVelBuffer_ = makeGpu(macSz, 0);

    VkDeviceSize resPartSz = kResidualGroups * 2 * sizeof(float);
    residualPartialBuffer_ = makeGpu(resPartSz, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    {
        VkBufferCreateInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bi.size  = resPartSz;
        bi.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        VmaAllocationCreateInfo ai{};
        ai.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
        ai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
        VmaAllocationInfo allocInfo;
        residualReadbackBuffer_.size = resPartSz;
        VK_CHECK(vmaCreateBuffer(allocator_, &bi, &ai,
            &residualReadbackBuffer_.buffer, &residualReadbackBuffer_.allocation, &allocInfo));
        residualReadbackBuffer_.mappedPtr = allocInfo.pMappedData;
    }

    // Persistent-mapped staging buffer
    VkDeviceSize stgSz = std::max({fSz, obSz, macSz});
    {
        VkBufferCreateInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bi.size  = stgSz;
        bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        VmaAllocationCreateInfo ai{};
        ai.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
        ai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
        VmaAllocationInfo allocInfo;
        stagingBuffer_.size = stgSz;
        VK_CHECK(vmaCreateBuffer(allocator_, &bi, &ai,
            &stagingBuffer_.buffer, &stagingBuffer_.allocation, &allocInfo));
        stagingBuffer_.mappedPtr = allocInfo.pMappedData;
    }

    deletionQueue_.push([this](){
        vmaDestroyBuffer(allocator_, fBufferA_.buffer,          fBufferA_.allocation);
        vmaDestroyBuffer(allocator_, fBufferB_.buffer,          fBufferB_.allocation);
        vmaDestroyBuffer(allocator_, obstacleBuffer_.buffer,    obstacleBuffer_.allocation);
        vmaDestroyBuffer(allocator_, macroBuffer_.buffer,       macroBuffer_.allocation);
        vmaDestroyBuffer(allocator_, aeroPartialBuffer_.buffer, aeroPartialBuffer_.allocation);
        vmaDestroyBuffer(allocator_, aeroReadbackBuffer_.buffer,aeroReadbackBuffer_.allocation);
        vmaDestroyBuffer(allocator_, prevVelBuffer_.buffer,          prevVelBuffer_.allocation);
        vmaDestroyBuffer(allocator_, residualPartialBuffer_.buffer,  residualPartialBuffer_.allocation);
        vmaDestroyBuffer(allocator_, residualReadbackBuffer_.buffer, residualReadbackBuffer_.allocation);
        vmaDestroyBuffer(allocator_, stagingBuffer_.buffer,     stagingBuffer_.allocation);
    });
}

// ════════════════════════════════════════════════════════════════════════════
// Descriptor Sets
// ════════════════════════════════════════════════════════════════════════════

void FluidSolver::createDescriptorSets() {
    size_t cells = totalCells();
    VkDeviceSize fSz   = cells * 19 * sizeof(float);
    VkDeviceSize obSz  = cells * sizeof(uint32_t);
    VkDeviceSize macSz = cells * 4 * sizeof(float);
    VkDeviceSize aeroSz = kAeroGroups * 4 * sizeof(float);

    // ── LBM descriptor layout: 4 bindings (f_in, f_out, obstacle, macro) ──
    {
        std::array<VkDescriptorSetLayoutBinding, 4> b{};
        for (uint32_t i = 0; i < 4; ++i) {
            b[i].binding        = i;
            b[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            b[i].descriptorCount= 1;
            b[i].stageFlags     = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo li{};
        li.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        li.bindingCount = 4;
        li.pBindings    = b.data();
        VK_CHECK(vkCreateDescriptorSetLayout(device_, &li, nullptr, &lbmDescLayout_));
    }
    {
        VkDescriptorPoolSize ps{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 8 };
        VkDescriptorPoolCreateInfo pi{};
        pi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pi.maxSets = 2; pi.poolSizeCount = 1; pi.pPoolSizes = &ps;
        VK_CHECK(vkCreateDescriptorPool(device_, &pi, nullptr, &lbmDescPool_));
    }
    {
        VkDescriptorSetLayout layouts[2] = { lbmDescLayout_, lbmDescLayout_ };
        VkDescriptorSetAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorPool = lbmDescPool_; ai.descriptorSetCount = 2; ai.pSetLayouts = layouts;
        VkDescriptorSet sets[2];
        VK_CHECK(vkAllocateDescriptorSets(device_, &ai, sets));
        lbmSetA_ = sets[0]; lbmSetB_ = sets[1];
    }
    // Write LBM descriptor sets once (no per-frame updates)
    auto wb = [&](VkDescriptorSet set, uint32_t binding, VkBuffer buf, VkDeviceSize sz) {
        VkDescriptorBufferInfo bi{ buf, 0, sz };
        VkWriteDescriptorSet w{};
        w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet = set; w.dstBinding = binding;
        w.descriptorCount = 1; w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w.pBufferInfo = &bi;
        vkUpdateDescriptorSets(device_, 1, &w, 0, nullptr);
    };
    // SetA: f_in=A, f_out=B
    wb(lbmSetA_, 0, fBufferA_.buffer, fSz);
    wb(lbmSetA_, 1, fBufferB_.buffer, fSz);
    wb(lbmSetA_, 2, obstacleBuffer_.buffer, obSz);
    wb(lbmSetA_, 3, macroBuffer_.buffer, macSz);
    // SetB: f_in=B, f_out=A
    wb(lbmSetB_, 0, fBufferB_.buffer, fSz);
    wb(lbmSetB_, 1, fBufferA_.buffer, fSz);
    wb(lbmSetB_, 2, obstacleBuffer_.buffer, obSz);
    wb(lbmSetB_, 3, macroBuffer_.buffer, macSz);

    // ── Aero descriptor layout: 3 bindings (macro, obstacle, partial_out) ──
    {
        std::array<VkDescriptorSetLayoutBinding, 3> b{};
        for (uint32_t i = 0; i < 3; ++i) {
            b[i].binding = i; b[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            b[i].descriptorCount = 1; b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo li{};
        li.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        li.bindingCount = 3; li.pBindings = b.data();
        VK_CHECK(vkCreateDescriptorSetLayout(device_, &li, nullptr, &aeroDescLayout_));
    }
    {
        VkDescriptorPoolSize ps{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3 };
        VkDescriptorPoolCreateInfo pi{};
        pi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pi.maxSets = 1; pi.poolSizeCount = 1; pi.pPoolSizes = &ps;
        VK_CHECK(vkCreateDescriptorPool(device_, &pi, nullptr, &aeroDescPool_));
    }
    {
        VkDescriptorSetAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorPool = aeroDescPool_; ai.descriptorSetCount = 1; ai.pSetLayouts = &aeroDescLayout_;
        VK_CHECK(vkAllocateDescriptorSets(device_, &ai, &aeroDescSet_));
    }
    wb(aeroDescSet_, 0, macroBuffer_.buffer, macSz);
    wb(aeroDescSet_, 1, obstacleBuffer_.buffer, obSz);
    wb(aeroDescSet_, 2, aeroPartialBuffer_.buffer, aeroSz);

    // ── Residual descriptor layout: 4 bindings (macro, obstacle, prevVel, partial_out) ──
    VkDeviceSize resSz = kResidualGroups * 2 * sizeof(float);
    {
        std::array<VkDescriptorSetLayoutBinding, 4> b{};
        for (uint32_t i = 0; i < 4; ++i) {
            b[i].binding = i; b[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            b[i].descriptorCount = 1; b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo li{};
        li.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        li.bindingCount = 4; li.pBindings = b.data();
        VK_CHECK(vkCreateDescriptorSetLayout(device_, &li, nullptr, &residualDescLayout_));
    }
    {
        VkDescriptorPoolSize ps{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4 };
        VkDescriptorPoolCreateInfo pi{};
        pi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pi.maxSets = 1; pi.poolSizeCount = 1; pi.pPoolSizes = &ps;
        VK_CHECK(vkCreateDescriptorPool(device_, &pi, nullptr, &residualDescPool_));
    }
    {
        VkDescriptorSetAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorPool = residualDescPool_; ai.descriptorSetCount = 1; ai.pSetLayouts = &residualDescLayout_;
        VK_CHECK(vkAllocateDescriptorSets(device_, &ai, &residualDescSet_));
    }
    wb(residualDescSet_, 0, macroBuffer_.buffer,           macSz);
    wb(residualDescSet_, 1, obstacleBuffer_.buffer,        obSz);
    wb(residualDescSet_, 2, prevVelBuffer_.buffer,         macSz);
    wb(residualDescSet_, 3, residualPartialBuffer_.buffer, resSz);

    deletionQueue_.push([this](){
        vkDestroyDescriptorPool(device_, lbmDescPool_,  nullptr);
        vkDestroyDescriptorSetLayout(device_, lbmDescLayout_, nullptr);
        vkDestroyDescriptorPool(device_, aeroDescPool_, nullptr);
        vkDestroyDescriptorSetLayout(device_, aeroDescLayout_, nullptr);
        vkDestroyDescriptorPool(device_, residualDescPool_, nullptr);
        vkDestroyDescriptorSetLayout(device_, residualDescLayout_, nullptr);
    });
}

// ════════════════════════════════════════════════════════════════════════════
// Pipelines
// ════════════════════════════════════════════════════════════════════════════

void FluidSolver::createLBMPipeline() {
    auto spirv = loadShaderModule("shaders/fluid_lbm.comp.spv");
    VkShaderModuleCreateInfo smi{};
    smi.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smi.codeSize = spirv.size() * sizeof(uint32_t); smi.pCode = spirv.data();
    VkShaderModule sm;
    VK_CHECK(vkCreateShaderModule(device_, &smi, nullptr, &sm));

    VkPushConstantRange pcr{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(LBMPushConstants) };
    VkPipelineLayoutCreateInfo pli{};
    pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pli.setLayoutCount = 1; pli.pSetLayouts = &lbmDescLayout_;
    pli.pushConstantRangeCount = 1; pli.pPushConstantRanges = &pcr;
    VK_CHECK(vkCreatePipelineLayout(device_, &pli, nullptr, &lbmLayout_));

    VkComputePipelineCreateInfo ci{};
    ci.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    ci.layout = lbmLayout_;
    ci.stage  = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                  nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, sm, "main", nullptr };
    VK_CHECK(vkCreateComputePipelines(device_, pipelineCache_, 1, &ci, nullptr, &lbmPipeline_));
    vkDestroyShaderModule(device_, sm, nullptr);

    deletionQueue_.push([this](){
        vkDestroyPipeline(device_, lbmPipeline_, nullptr);
        vkDestroyPipelineLayout(device_, lbmLayout_, nullptr);
    });
}

void FluidSolver::createAeroPipeline() {
    auto spirv = loadShaderModule("shaders/aero_forces.comp.spv");
    VkShaderModuleCreateInfo smi{};
    smi.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smi.codeSize = spirv.size() * sizeof(uint32_t); smi.pCode = spirv.data();
    VkShaderModule sm;
    VK_CHECK(vkCreateShaderModule(device_, &smi, nullptr, &sm));

    VkPushConstantRange pcr{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(AeroPushConstants) };
    VkPipelineLayoutCreateInfo pli{};
    pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pli.setLayoutCount = 1; pli.pSetLayouts = &aeroDescLayout_;
    pli.pushConstantRangeCount = 1; pli.pPushConstantRanges = &pcr;
    VK_CHECK(vkCreatePipelineLayout(device_, &pli, nullptr, &aeroLayout_));

    VkComputePipelineCreateInfo ci{};
    ci.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    ci.layout = aeroLayout_;
    ci.stage  = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                  nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, sm, "main", nullptr };
    VK_CHECK(vkCreateComputePipelines(device_, pipelineCache_, 1, &ci, nullptr, &aeroPipeline_));
    vkDestroyShaderModule(device_, sm, nullptr);

    deletionQueue_.push([this](){
        vkDestroyPipeline(device_, aeroPipeline_, nullptr);
        vkDestroyPipelineLayout(device_, aeroLayout_, nullptr);
    });
}

void FluidSolver::createResidualPipeline() {
    auto spirv = loadShaderModule("shaders/residual.comp.spv");
    VkShaderModuleCreateInfo smi{};
    smi.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smi.codeSize = spirv.size() * sizeof(uint32_t); smi.pCode = spirv.data();
    VkShaderModule sm;
    VK_CHECK(vkCreateShaderModule(device_, &smi, nullptr, &sm));

    VkPushConstantRange pcr{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ResidualPushConstants) };
    VkPipelineLayoutCreateInfo pli{};
    pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pli.setLayoutCount = 1; pli.pSetLayouts = &residualDescLayout_;
    pli.pushConstantRangeCount = 1; pli.pPushConstantRanges = &pcr;
    VK_CHECK(vkCreatePipelineLayout(device_, &pli, nullptr, &residualLayout_));

    VkComputePipelineCreateInfo ci{};
    ci.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    ci.layout = residualLayout_;
    ci.stage  = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                  nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, sm, "main", nullptr };
    VK_CHECK(vkCreateComputePipelines(device_, pipelineCache_, 1, &ci, nullptr, &residualPipeline_));
    vkDestroyShaderModule(device_, sm, nullptr);

    deletionQueue_.push([this](){
        vkDestroyPipeline(device_, residualPipeline_, nullptr);
        vkDestroyPipelineLayout(device_, residualLayout_, nullptr);
    });
}

// ════════════════════════════════════════════════════════════════════════════
// Timestamp query pool
// ════════════════════════════════════════════════════════════════════════════

void FluidSolver::createTimestampPool() {
    VkQueryPoolCreateInfo qi{};
    qi.sType      = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    qi.queryType  = VK_QUERY_TYPE_TIMESTAMP;
    qi.queryCount = 4;  // 0=preLBM, 1=postLBM, 2=preAero, 3=postAero
    VK_CHECK(vkCreateQueryPool(device_, &qi, nullptr, &timestampPool_));
    deletionQueue_.push([this](){ vkDestroyQueryPool(device_, timestampPool_, nullptr); });
}

// ════════════════════════════════════════════════════════════════════════════
// Upload obstacle map
// ════════════════════════════════════════════════════════════════════════════

void FluidSolver::uploadObstacleMap(const std::vector<uint32_t>& data) {
    VkDeviceSize sz = data.size() * sizeof(uint32_t);
    assert(sz <= stagingBuffer_.size);
    std::memcpy(stagingBuffer_.mappedPtr, data.data(), sz);
    vmaFlushAllocation(allocator_, stagingBuffer_.allocation, 0, sz);

    submitOneShot([&](VkCommandBuffer cmd) {
        VkBufferCopy cr{ 0, 0, sz };
        vkCmdCopyBuffer(cmd, stagingBuffer_.buffer, obstacleBuffer_.buffer, 1, &cr);
    });
    std::cout << "[FluidSolver] Obstacle map uploaded.\n";
}

// ════════════════════════════════════════════════════════════════════════════
// Reset to equilibrium
// ════════════════════════════════════════════════════════════════════════════

void FluidSolver::resetToEquilibrium() {
    size_t cells = totalCells();
    VkDeviceSize fSz = cells * 19 * sizeof(float);

    // D3Q19 equilibrium weights at rho=1, u=0 → f_i = w_i
    static const float w[19] = {
        1.f/3.f,
        1.f/18.f,1.f/18.f,1.f/18.f,1.f/18.f,1.f/18.f,1.f/18.f,
        1.f/36.f,1.f/36.f,1.f/36.f,1.f/36.f,
        1.f/36.f,1.f/36.f,1.f/36.f,1.f/36.f,
        1.f/36.f,1.f/36.f,1.f/36.f,1.f/36.f,
    };

    // SoA: all f[q=0] first, then f[q=1], ..., f[q=18]
    std::vector<float> init(cells * 19);
    for (size_t c = 0; c < cells; ++c)
        for (int q = 0; q < 19; ++q)
            init[q * cells + c] = w[q];

    assert(fSz <= stagingBuffer_.size);
    std::memcpy(stagingBuffer_.mappedPtr, init.data(), fSz);
    vmaFlushAllocation(allocator_, stagingBuffer_.allocation, 0, fSz);

    submitOneShot([&](VkCommandBuffer cmd) {
        VkBufferCopy cr{ 0, 0, fSz };
        vkCmdCopyBuffer(cmd, stagingBuffer_.buffer, fBufferA_.buffer, 1, &cr);
        vkCmdCopyBuffer(cmd, stagingBuffer_.buffer, fBufferB_.buffer, 1, &cr);
        vkCmdFillBuffer(cmd, macroBuffer_.buffer, 0, VK_WHOLE_SIZE, 0);
        // Clear the previous-velocity history so the first residual sample is
        // well-defined (≈1.0) rather than comparing against stale data.
        vkCmdFillBuffer(cmd, prevVelBuffer_.buffer, 0, VK_WHOLE_SIZE, 0);
    });

    pingPong_ = false;
    std::cout << "[FluidSolver] Reset to equilibrium.\n";
}

// ════════════════════════════════════════════════════════════════════════════
// LBM step
// ════════════════════════════════════════════════════════════════════════════

void FluidSolver::step(VkCommandBuffer cmd, const SimParams& params, uint32_t timeStep, bool recordTimings) {
    // Timestamp: before LBM. Gated so multi-step batches only measure the last
    // step (the query pool has 2 LBM slots and successive writes would otherwise
    // race / overwrite each other).
    if (recordTimings) {
        vkCmdResetQueryPool(cmd, timestampPool_, 0, 2);
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, timestampPool_, 0);
    }

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, lbmPipeline_);
    VkDescriptorSet cur = pingPong_ ? lbmSetB_ : lbmSetA_;
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            lbmLayout_, 0, 1, &cur, 0, nullptr);

    LBMPushConstants pc{};
    pc.gridX      = params.gridX;
    pc.gridY      = params.gridY;
    pc.gridZ      = params.gridZ;
    pc.tau        = params.tau;
    pc.inletVelX  = params.inletVelX;
    pc.inletVelY  = params.inletVelY;
    pc.inletVelZ  = params.inletVelZ;
    pc.time       = static_cast<float>(timeStep);
    pc.turbulence = params.turbulence;
    pc.s_bulk     = params.s_bulk;
    pc.s_ghost    = params.s_ghost;
    pc.lbmMode    = static_cast<uint32_t>(params.lbmMode);
    vkCmdPushConstants(cmd, lbmLayout_, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(LBMPushConstants), &pc);

    uint32_t gx = (params.gridX + 7) / 8;
    uint32_t gy = (params.gridY + 7) / 8;
    uint32_t gz = (params.gridZ + 3) / 4;
    vkCmdDispatch(cmd, gx, gy, gz);

    // Barrier: ensure compute writes visible before next read
    VkMemoryBarrier barrier{};
    barrier.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 1, &barrier, 0, nullptr, 0, nullptr);

    if (recordTimings) {
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, timestampPool_, 1);
    }

    pingPong_ = !pingPong_;
}

// ════════════════════════════════════════════════════════════════════════════
// Aero force integration dispatch
// ════════════════════════════════════════════════════════════════════════════

void FluidSolver::dispatchAeroForces(VkCommandBuffer cmd, const SimParams& params, bool recordTimings) {
    if (recordTimings) {
        vkCmdResetQueryPool(cmd, timestampPool_, 2, 2);
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, timestampPool_, 2);
    }

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, aeroPipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            aeroLayout_, 0, 1, &aeroDescSet_, 0, nullptr);

    AeroPushConstants pc{};
    pc.gridX      = params.gridX;
    pc.gridY      = params.gridY;
    pc.gridZ      = params.gridZ;
    pc.inletVelX  = params.inletVelX;
    vkCmdPushConstants(cmd, aeroLayout_, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(AeroPushConstants), &pc);

    vkCmdDispatch(cmd, kAeroGroups, 1, 1);

    // Barrier: aero writes done, then copy partial sums to readback buffer
    VkMemoryBarrier mb{};
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 1, &mb, 0, nullptr, 0, nullptr);

    VkBufferCopy cr{ 0, 0, kAeroGroups * 4 * sizeof(float) };
    vkCmdCopyBuffer(cmd, aeroPartialBuffer_.buffer, aeroReadbackBuffer_.buffer, 1, &cr);

    if (recordTimings) {
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, timestampPool_, 3);
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Convergence residual dispatch
// ════════════════════════════════════════════════════════════════════════════

void FluidSolver::dispatchResidual(VkCommandBuffer cmd, const SimParams& params) {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, residualPipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            residualLayout_, 0, 1, &residualDescSet_, 0, nullptr);

    ResidualPushConstants pc{};
    pc.gridX = params.gridX;
    pc.gridY = params.gridY;
    pc.gridZ = params.gridZ;
    vkCmdPushConstants(cmd, residualLayout_, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(ResidualPushConstants), &pc);

    vkCmdDispatch(cmd, kResidualGroups, 1, 1);

    // Barrier: residual writes done, then copy partial sums to readback buffer
    VkMemoryBarrier mb{};
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 1, &mb, 0, nullptr, 0, nullptr);

    VkBufferCopy cr{ 0, 0, kResidualGroups * 2 * sizeof(float) };
    vkCmdCopyBuffer(cmd, residualPartialBuffer_.buffer, residualReadbackBuffer_.buffer, 1, &cr);
}

// ════════════════════════════════════════════════════════════════════════════
// CPU readbacks
// ════════════════════════════════════════════════════════════════════════════

AeroForces FluidSolver::readAeroForces() const {
    if (!aeroReadbackBuffer_.mappedPtr) return {};
    vmaInvalidateAllocation(allocator_, aeroReadbackBuffer_.allocation, 0, VK_WHOLE_SIZE);

    const float* partial = static_cast<const float*>(aeroReadbackBuffer_.mappedPtr);
    double drag = 0, lift = 0, side = 0;
    for (uint32_t i = 0; i < kAeroGroups; ++i) {
        drag += partial[i * 4 + 0];
        lift += partial[i * 4 + 1];
        side += partial[i * 4 + 2];
    }
    return { static_cast<float>(drag), static_cast<float>(lift), static_cast<float>(side), 0.f };
}

float FluidSolver::readResidual() const {
    if (!residualReadbackBuffer_.mappedPtr) return 1.f;
    vmaInvalidateAllocation(allocator_, residualReadbackBuffer_.allocation, 0, VK_WHOLE_SIZE);

    const float* partial = static_cast<const float*>(residualReadbackBuffer_.mappedPtr);
    double sumDelta = 0, sumVel = 0;
    for (uint32_t i = 0; i < kResidualGroups; ++i) {
        sumDelta += partial[i * 2 + 0];
        sumVel   += partial[i * 2 + 1];
    }
    // Normalised L2 residual. Guard the divisor for the first samples before
    // any flow has developed (sumVel ≈ 0).
    if (sumVel < 1e-12) return 1.f;
    return static_cast<float>(std::sqrt(sumDelta / sumVel));
}

GpuTimings FluidSolver::readTimings() const {
    // Query LBM slots (0-1) and aero slots (2-3) independently.
    // If aero was not dispatched this frame, slots 2-3 return VK_NOT_READY.
    // A single 4-slot query would return {} on that error and lose lbmMs too.
    float ns = timestampPeriodNs_;
    GpuTimings result{};
    uint64_t ts[2] = {};

    if (vkGetQueryPoolResults(device_, timestampPool_, 0, 2,
            sizeof(ts), ts, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT) == VK_SUCCESS) {
        result.lbmMs = static_cast<float>((ts[1] - ts[0]) * ns) / 1e6f;
    }
    if (vkGetQueryPoolResults(device_, timestampPool_, 2, 2,
            sizeof(ts), ts, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT) == VK_SUCCESS) {
        result.aeroMs = static_cast<float>((ts[1] - ts[0]) * ns) / 1e6f;
    }
    return result;
}

} // namespace vwt
