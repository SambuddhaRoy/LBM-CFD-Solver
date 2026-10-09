#pragma once
// ============================================================================
// solver.hpp — D3Q19 lattice Boltzmann solver on the GPU.
//
// Memory per cell: 19 distributions (2 B each in FP16, 4 B in FP32), a flag
// byte, a 16-bit signed distance and an 8-byte render field: 49 B in FP16,
// 87 B in FP32. The distributions are streamed in place (Esoteric-Pull), so
// there is a single copy of them.
// ============================================================================

#include "vk.hpp"

#include <array>
#include <cstdint>
#include <vector>

namespace wt {

enum class Precision : uint32_t { FP32 = 0, FP16S = 1, FP16C = 2 };
const char* precisionName(Precision p);
uint32_t    bytesPerCell(Precision p);

struct GridConfig {
    uint32_t  nx = 256, ny = 128, nz = 128;
    Precision precision = Precision::FP16C;
    bool      farFieldY = true;    // false: periodic in Y
    bool      farFieldZ = true;    // false: periodic in Z (2D runs use nz = 1)
    bool      periodicX = false;   // true: no inlet/outlet, periodic in X (test flows)
};

struct FlowParams {
    float tau         = 0.52f;     // molecular relaxation time, nu = (tau - 1/2)/3
    float smagorinsky = 0.12f;     // Smagorinsky constant; 0 disables LES
    float turbulence  = 0.f;       // inlet perturbation, fraction of the inlet speed
    float uIn         = 0.08f;     // inlet speed along +x, lattice units
    bool  bouzidi     = true;      // interpolated walls (else halfway bounce-back)
};

struct Stats {
    float    residual = 1.f;       // sqrt(sum|u - u_prev|^2 / sum|u|^2) between samples
    float    rhoMean  = 1.f;
    float    maxU     = 0.f;
    float    kinetic  = 0.f;       // sum of 1/2 rho u^2 over the sampled cells
    uint32_t samples  = 0;
};

// Relaxation time and turbulence model for a body `cells` long at lattice
// speed `u` that should see Reynolds number `re`. model: 0 auto, 1 laminar,
// 2 LES. Shared by the app and the stability suite, so the suite tests the
// policy the app runs.
struct FlowScaling {
    float tau = 0.6f, smagorinsky = 0.f;
    float reSimulated = 0.f;
    float reResolved = 0.f;    // highest Re the grid resolves without a model
    bool  limited = false;     // reSimulated < re
};
FlowScaling scaleFlow(double re, float cells, float u, int model);

// Mirror of `Params` in shaders/lattice.glsl.
struct StepPush {
    uint32_t nx, ny, nz, t;
    float    tau0, smag2, turb, bouzidi;
    float    uin[4];
    uint32_t writeField, faces, pad1, pad2;
};
static_assert(sizeof(StepPush) == 64);

class Solver {
public:
    static constexpr uint32_t kSlots = 2;   // readback slots for frames in flight

    void create(gpu::Context& ctx, const GridConfig& g);
    void destroy();

    // After writing the signed distance field (see geometry.hpp): rebuild cell
    // types and keep the running flow, reviving cells the body uncovered.
    void applyGeometry(VkCommandBuffer cmd);
    // Equilibrium at the inlet velocity everywhere; restarts the flow.
    void reset(VkCommandBuffer cmd);

    // Records n steps. The last one writes the render field if `field` and
    // sums the momentum-exchange force into readback slot `slot` if `forces`.
    void recordSteps(VkCommandBuffer cmd, uint32_t n, bool field, bool forces,
                     uint32_t slot = 0, bool timestamps = false);
    // Residual, mean density, peak speed from the render field (needs a step
    // with field = true before it).
    void recordStats(VkCommandBuffer cmd, uint32_t slot = 0);

    // Valid once the submission that recorded them has completed.
    std::array<double, 3> forces(uint32_t slot = 0) const;   // lattice units
    Stats  stats(uint32_t slot = 0) const;
    double lastBatchMs(uint32_t slot = 0) const;              // needs timestamps

    // Exact FP32 (rho, ux, uy, uz) of every cell; submits and waits.
    void probe(std::vector<float>& rhoU);

    bool created() const { return ctx_ != nullptr; }
    const GridConfig& grid() const { return g_; }
    uint64_t cells() const { return uint64_t(g_.nx) * g_.ny * g_.nz; }
    gpu::Bindings bindings() const;
    const gpu::Buffer& sdf()   const { return sdf_; }
    const gpu::Buffer& ddf(uint32_t q) const { return ddf_[q]; }   // tests: direct state upload
    const gpu::Buffer& flags() const { return flags_; }
    const gpu::Buffer& field() const { return field_; }

    FlowParams flow;
    uint64_t   t = 0;     // steps taken since reset; parity drives the streaming

private:
    StepPush push(bool writeField) const;
    uint32_t stepGroups() const;

    gpu::Context* ctx_ = nullptr;
    GridConfig    g_;
    std::array<gpu::Buffer, 19> ddf_;
    gpu::Buffer flags_, sdf_, field_, plane_, forcePartials_, statPrev_, statPartials_;
    std::array<gpu::Buffer, kSlots> forceOut_, statOut_;
    uint32_t statCount_ = 0, statStride_ = 1;
    // step.comp workgroup shape. A sweep on an RTX 5070 Ti (32x8, 32x4,
    // 32x2, 64x4, 64x1, 128x2, 256x1) put 32-wide rows clearly ahead: each
    // warp then covers exactly one row segment.
    static constexpr uint32_t wgx_ = 32, wgy_ = 8;

    gpu::Kernel step_[2], stepForce_[2];      // [parity]
    gpu::Kernel init_, classify_, wallmask_, probe_, stats_, reduce_;
    VkQueryPool queries_ = VK_NULL_HANDLE;
};

} // namespace wt
