#pragma once
// ============================================================================
// sim.h — simulation parameters, physical unit scaling, and the LBM solver
// ============================================================================

#include "gpu.h"

namespace vwt {

// ─── Lattice-level parameters ────────────────────────────────────────────────
struct SimParams {
    uint32_t gx = 128, gy = 80, gz = 80;
    float    tau       = 0.56f;   // base relaxation time (ν = (τ−½)/3)
    float    uIn       = 0.07f;   // inlet speed, lattice units
    float    turb      = 0.0f;    // inlet perturbation, fraction of uIn
    int      collision = 1;       // 0 = BGK, 1 = regularized (recommended)
    bool     les       = true;    // Smagorinsky subgrid model
    float    csSmago   = 0.14f;
};

// ─── Working fluids for physical scaling ─────────────────────────────────────
struct FluidPreset {
    const char* name;
    float rho;     // kg/m³
    float nu;      // m²/s  (kinematic viscosity)
    float sound;   // m/s
};

inline constexpr FluidPreset kFluids[] = {
    { "Air — sea level, 15 C",   1.225f,  1.461e-5f, 340.3f },
    { "Air — 11 km, -56 C",      0.364f,  3.930e-5f, 295.1f },
    { "Water — 20 C",            998.2f,  1.004e-6f, 1482.f },
    { "CO2 — Mars surface",      0.020f,  6.600e-4f, 244.0f },
};
inline constexpr int kFluidCount = int(sizeof(kFluids) / sizeof(kFluids[0]));

// Maps lattice quantities to physical ones. The model's physical length and
// its extent in lattice cells anchor dx; matching the inlet speeds anchors dt.
struct UnitScale {
    int   fluidIdx   = 0;
    float physSpeed  = 30.f;   // m/s
    float physLength = 1.f;    // m — model reference length (along flow)

    // Derived; call compute() after any input or grid/model change.
    float dx = 0, dt = 0;
    float rePhys = 0;          // Reynolds number at physical scale
    float reLat  = 0;          // Reynolds number the lattice actually resolves
    float mach   = 0;          // physical Mach number
    float velToMps = 0;        // lattice velocity → m/s

    void compute(uint32_t modelCellsX, const SimParams& p) {
        const FluidPreset& f = kFluids[fluidIdx];
        const float cellsX = float(modelCellsX > 0 ? modelCellsX : 1);
        dx       = physLength / cellsX;
        dt       = (physSpeed > 1e-6f) ? p.uIn * dx / physSpeed : 0.f;
        velToMps = (p.uIn > 1e-6f) ? physSpeed / p.uIn : 0.f;
        const float nuLat = (p.tau - 0.5f) / 3.f;
        rePhys = physSpeed * physLength / f.nu;
        reLat  = (nuLat > 1e-9f) ? p.uIn * cellsX / nuLat : 0.f;
        mach   = physSpeed / f.sound;
    }
};

// ─── Fused analysis results ──────────────────────────────────────────────────
struct Analysis {
    float residual   = 1.f;   // sqrt(Σ|Δu|² / Σ|u|²), measured
    float drag = 0.f, lift = 0.f, side = 0.f;   // momentum-exchange forces (lattice)
    float massAvg    = 0.f;   // mean ρ per fluid cell (1.0 = conserved)
    float maxU       = 0.f;   // peak |u|, lattice units
    float ke         = 0.f;   // total kinetic energy, lattice units
    float fluidCells = 0.f;
    float surfFaces  = 0.f;
    bool  valid      = false;
};

// ─── Push-constant mirrors (must match shader layouts) ───────────────────────
struct LbmPush {
    uint32_t gx, gy, gz;
    float    tau, uIn, time, turb;
    uint32_t collision, les;
    float    csSmago;
    uint32_t writeMacro;  // 1 = write rho/u to macro buffer (last step of batch only)
};

struct AnalysisPush { uint32_t gx, gy, gz, curF; };  // curF: 0=fA, 1=fB (last written)

struct SlicePush {
    uint32_t gx, gy, gz, axis, index, mode;
    float    scaleVel, uIn, maxVort;
};

// ─── Solver ──────────────────────────────────────────────────────────────────
// Query pool slots: 0/1 LBM batch, 2/3 analysis, 4/5 slice (used by SliceView).
class Solver {
public:
    void init(GpuContext& ctx, const SimParams& p);
    void destroy();

    void uploadObstacles(const std::vector<uint32_t>& occupancy);
    // Override the signed-distance field (lattice units, >0 in fluid) used by
    // the Bouzidi interpolated bounce-back. Call after uploadObstacles().
    void setSDF(const std::vector<float>& phi);
    void reset();   // distributions to equilibrium, history cleared

    // Records n collide-stream steps; timestamps bracket the whole batch.
    void recordSteps(VkCommandBuffer cmd, const SimParams& p,
                     uint32_t nSteps, uint32_t stepBase, bool timestamps);

    // Records the fused analysis reduction + readback copy (timestamps 2/3).
    // `slot` selects an independent partial/readback pair so two frames in
    // flight never race on the same readback memory.
    static constexpr uint32_t kSlots = 2;
    void recordAnalysis(VkCommandBuffer cmd, const SimParams& p, uint32_t slot);

    // CPU-side folds; call after the fence for the recorded work has signaled.
    Analysis readAnalysis(uint32_t slot) const;
    float    readTimestampMs(uint32_t firstSlot) const;   // 0, 2, or 4

    VkQueryPool queryPool()      const { return queryPool_; }
    VkBuffer    macroBuffer()    const { return macro_.buffer; }
    VkBuffer    obstacleBuffer() const { return obstacle_.buffer; }
    size_t      cells()          const { return size_t(gx_) * gy_ * gz_; }

private:
    static constexpr uint32_t kGroups = 256;
    static constexpr uint32_t kVals   = 12;

    GpuContext* ctx_ = nullptr;
    uint32_t gx_ = 0, gy_ = 0, gz_ = 0;
    bool pingPong_ = false;

    GpuBuffer fA_, fB_, obstacle_, sdf_, macro_, prev_, staging_;
    GpuBuffer partial_[kSlots], readback_[kSlots];

    VkDescriptorPool      descPool_   = VK_NULL_HANDLE;
    VkDescriptorSetLayout lbmLayout_  = VK_NULL_HANDLE;
    VkDescriptorSet       lbmSetA_    = VK_NULL_HANDLE;   // A→B
    VkDescriptorSet       lbmSetB_    = VK_NULL_HANDLE;   // B→A
    VkDescriptorSetLayout anaLayout_  = VK_NULL_HANDLE;
    VkDescriptorSet       anaSet_[kSlots] = {};

    VkPipelineLayout lbmPipeLayout_ = VK_NULL_HANDLE;
    VkPipeline       lbmPipe_       = VK_NULL_HANDLE;
    VkPipelineLayout anaPipeLayout_ = VK_NULL_HANDLE;
    VkPipeline       anaPipe_       = VK_NULL_HANDLE;

    VkQueryPool queryPool_ = VK_NULL_HANDLE;

    DeletionQueue dq_;
};

} // namespace vwt
