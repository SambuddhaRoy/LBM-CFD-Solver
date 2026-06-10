#pragma once
// ============================================================================
// fluid_solver.h — GPU D3Q19 LBM Fluid Solver + Aero Force Integration
// ============================================================================

#include "vk_types.h"

namespace vwt {

class FluidSolver {
public:
    void init(VkDevice device, VmaAllocator allocator,
              VkQueue computeQueue, uint32_t computeQueueFamily,
              VkPipelineCache pipelineCache, const SimParams& params);
    void destroy();

    // Upload obstacle map (blocks until transfer complete)
    void uploadObstacleMap(const std::vector<uint32_t>& obstacleData);

    // Reset distribution functions to equilibrium
    void resetToEquilibrium();

    // Record one LBM step into cmd
    void step(VkCommandBuffer cmd, const SimParams& params, uint32_t timeStep, bool recordTimings = true);

    // Record aero force integration dispatch into cmd
    // Results readable after GPU-CPU sync via readAeroForces()
    void dispatchAeroForces(VkCommandBuffer cmd, const SimParams& params, bool recordTimings = true);

    // Record convergence-residual reduction into cmd.
    // Compares the current velocity field against the previous sample.
    // Results readable after GPU-CPU sync via readResidual()
    void dispatchResidual(VkCommandBuffer cmd, const SimParams& params);

    // CPU readback of last aero force dispatch (call after fence wait)
    AeroForces readAeroForces() const;

    // CPU readback of last residual dispatch — normalised L2 velocity residual
    // sqrt(Σ|u_now-u_prev|² / Σ|u_now|²). Call after fence wait.
    float readResidual() const;

    // GPU timestamp queries — call after fence wait to get last frame timings
    GpuTimings readTimings() const;

    // Timestamp slots written by step() and dispatchAeroForces()
    // Slot 0: before LBM, 1: after LBM, 2: before aero, 3: after aero
    VkQueryPool getTimestampPool() const { return timestampPool_; }

    VkBuffer getMacroBuffer()    const { return macroBuffer_.buffer; }
    VkBuffer getObstacleBuffer() const { return obstacleBuffer_.buffer; }
    size_t   totalCells()        const { return size_t(gridX_) * gridY_ * gridZ_; }

private:
    void createCommandPool();
    void createBuffers();
    void createDescriptorSets();
    void createLBMPipeline();
    void createAeroPipeline();
    void createResidualPipeline();
    void createTimestampPool();
    void submitOneShot(std::function<void(VkCommandBuffer)>&& record);

    VkDevice        device_       = VK_NULL_HANDLE;
    VmaAllocator    allocator_    = VK_NULL_HANDLE;
    VkQueue         queue_        = VK_NULL_HANDLE;
    uint32_t        queueFamily_  = 0;
    VkPipelineCache pipelineCache_ = VK_NULL_HANDLE;

    uint32_t gridX_ = 0, gridY_ = 0, gridZ_ = 0;

    // Distribution function ping-pong buffers (SoA layout)
    AllocatedBuffer fBufferA_;
    AllocatedBuffer fBufferB_;
    bool            pingPong_ = false;

    AllocatedBuffer obstacleBuffer_;
    AllocatedBuffer macroBuffer_;
    AllocatedBuffer stagingBuffer_;

    // Aero forces: 256 partial-sum entries × 4 floats, then readback
    static constexpr uint32_t kAeroGroups = 256;
    AllocatedBuffer aeroPartialBuffer_;   // GPU: 256 × 16 bytes
    AllocatedBuffer aeroReadbackBuffer_;  // CPU-visible: 256 × 16 bytes

    // Convergence residual: 256 partial-sum entries × 2 floats (sumDelta, sumVel)
    static constexpr uint32_t kResidualGroups = 256;
    AllocatedBuffer prevVelBuffer_;           // GPU: previous-sample velocity field
    AllocatedBuffer residualPartialBuffer_;   // GPU: 256 × 8 bytes
    AllocatedBuffer residualReadbackBuffer_;  // CPU-visible: 256 × 8 bytes

    // LBM pipeline
    VkDescriptorPool      lbmDescPool_   = VK_NULL_HANDLE;
    VkDescriptorSetLayout lbmDescLayout_ = VK_NULL_HANDLE;
    VkDescriptorSet       lbmSetA_       = VK_NULL_HANDLE;  // A→B
    VkDescriptorSet       lbmSetB_       = VK_NULL_HANDLE;  // B→A
    VkPipelineLayout      lbmLayout_     = VK_NULL_HANDLE;
    VkPipeline            lbmPipeline_   = VK_NULL_HANDLE;

    // Aero force pipeline
    VkDescriptorPool      aeroDescPool_   = VK_NULL_HANDLE;
    VkDescriptorSetLayout aeroDescLayout_ = VK_NULL_HANDLE;
    VkDescriptorSet       aeroDescSet_    = VK_NULL_HANDLE;
    VkPipelineLayout      aeroLayout_     = VK_NULL_HANDLE;
    VkPipeline            aeroPipeline_   = VK_NULL_HANDLE;

    // Convergence residual pipeline
    VkDescriptorPool      residualDescPool_   = VK_NULL_HANDLE;
    VkDescriptorSetLayout residualDescLayout_ = VK_NULL_HANDLE;
    VkDescriptorSet       residualDescSet_    = VK_NULL_HANDLE;
    VkPipelineLayout      residualLayout_     = VK_NULL_HANDLE;
    VkPipeline            residualPipeline_   = VK_NULL_HANDLE;

    // GPU timestamps
    VkQueryPool  timestampPool_       = VK_NULL_HANDLE;
    float        timestampPeriodNs_   = 1.0f;  // nanoseconds per tick

    VkCommandPool transferPool_ = VK_NULL_HANDLE;
    DeletionQueue deletionQueue_;
};

} // namespace vwt
