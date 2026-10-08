// ============================================================================
// solver.cpp — buffer management and command recording for the LBM solver
// ============================================================================

#include "solver.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace wt {

const char* precisionName(Precision p) {
    switch (p) {
    case Precision::FP32:  return "FP32";
    case Precision::FP16S: return "FP16S";
    default:               return "FP16C";
    }
}

uint32_t bytesPerCell(Precision p) {
    const uint32_t ddf = p == Precision::FP32 ? 4u : 2u;
    return 19u * ddf + 1u /*flags*/ + 2u /*sdf*/ + 8u /*field*/;
}

void Solver::create(gpu::Context& ctx, const GridConfig& g) {
    ctx_ = &ctx;
    g_   = g;
    if (g.nx < 4 || g.ny < 1 || g.nz < 1)
        throw std::runtime_error("grid must be at least 4 cells long in x");
    const uint64_t n = cells();
    if (n >= (1ull << 32))
        throw std::runtime_error("grid exceeds 2^32 cells");

    const uint64_t need = n * bytesPerCell(g.precision);
    if (double(need) > 0.92 * double(ctx.vramBytes))
        throw std::runtime_error("grid needs " + std::to_string(need >> 20) + " MiB but the GPU has "
                                 + std::to_string(ctx.vramBytes >> 20) + " MiB");

    const VkDeviceSize ddfBytes = n * (g.precision == Precision::FP32 ? 4 : 2);
    for (auto& b : ddf_) b = ctx.createBuffer(ddfBytes, 0, gpu::Mem::Device);
    flags_ = ctx.createBuffer((n + 3) & ~3ull, 0, gpu::Mem::Device);
    sdf_   = ctx.createBuffer((n * 2 + 3) & ~3ull, 0, gpu::Mem::Device);
    field_ = ctx.createBuffer(n * 8, 0, gpu::Mem::Device);
    plane_ = ctx.createBuffer(VkDeviceSize(2) * g.ny * g.nz * 16, 0, gpu::Mem::Device);
    forcePartials_ = ctx.createBuffer(VkDeviceSize(stepGroups()) * 16, 0, gpu::Mem::Device);

    // Statistics on a strided subsample of at most ~4M cells. An odd stride
    // keeps the samples from lining up in columns of the grid.
    statStride_ = uint32_t(std::max<uint64_t>(1, n / (1u << 22)));
    if (statStride_ > 1 && statStride_ % 2 == 0) ++statStride_;
    statCount_ = uint32_t((n - 1) / statStride_ + 1);
    statPrev_     = ctx.createBuffer(VkDeviceSize(statCount_) * 16, 0, gpu::Mem::Device);
    statPartials_ = ctx.createBuffer(VkDeviceSize(gpu::Context::groupCount(statCount_)) * 32, 0,
                                     gpu::Mem::Device);
    for (uint32_t s = 0; s < kSlots; ++s) {
        forceOut_[s] = ctx.createBuffer(16, 0, gpu::Mem::Readback);
        statOut_[s]  = ctx.createBuffer(32, 0, gpu::Mem::Readback);
    }

    const std::string p = std::to_string(uint32_t(g.precision));
    for (uint32_t parity = 0; parity < 2; ++parity) {
        step_[parity]      = ctx.loadKernel("step_p" + p, {parity, wgx_, wgy_});
        stepForce_[parity] = ctx.loadKernel("step_force_p" + p, {parity, wgx_, wgy_});
    }
    init_      = ctx.loadKernel("init_p" + p);
    classify_  = ctx.loadKernel("classify_p" + p);
    probe_     = ctx.loadKernel("probe_p" + p);
    wallmask_  = ctx.loadKernel("wallmask");
    stats_     = ctx.loadKernel("stats");
    reduce_    = ctx.loadKernel("reduce");

    VkQueryPoolCreateInfo qi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    qi.queryType  = VK_QUERY_TYPE_TIMESTAMP;
    qi.queryCount = 2 * kSlots;
    gpu::check(vkCreateQueryPool(ctx.device, &qi, nullptr, &queries_), "query pool");

    // Everything starts as fluid with a far-away surface.
    ctx.fill(flags_, 0);
    ctx.fill(statPrev_, 0);
    const uint16_t farHalf = 0x4800;    // +8.0 in IEEE half
    ctx.fill(sdf_, uint32_t(farHalf) | uint32_t(farHalf) << 16);
}

void Solver::destroy() {
    if (!ctx_) return;
    auto& c = *ctx_;
    vkDeviceWaitIdle(c.device);
    for (auto& b : ddf_) c.destroyBuffer(b);
    for (auto* b : {&flags_, &sdf_, &field_, &plane_, &forcePartials_, &statPrev_, &statPartials_})
        c.destroyBuffer(*b);
    for (uint32_t s = 0; s < kSlots; ++s) { c.destroyBuffer(forceOut_[s]); c.destroyBuffer(statOut_[s]); }
    for (auto* k : {&step_[0], &step_[1], &stepForce_[0], &stepForce_[1], &init_, &classify_,
                    &wallmask_, &probe_, &stats_, &reduce_})
        c.destroyKernel(*k);
    vkDestroyQueryPool(c.device, queries_, nullptr);
    ctx_ = nullptr;
}

// step.comp runs wgx_ x wgy_ x 1 workgroups over the grid.
uint32_t Solver::stepGroups() const {
    return ((g_.nx + wgx_ - 1) / wgx_) * ((g_.ny + wgy_ - 1) / wgy_) * g_.nz;
}

gpu::Bindings Solver::bindings() const {
    gpu::Bindings b = ctx_->emptyBindings();
    for (uint32_t q = 0; q < 19; ++q) b[q] = ddf_[q].buf;
    b[19] = flags_.buf;
    b[20] = sdf_.buf;
    b[21] = field_.buf;
    return b;
}

StepPush Solver::push(bool writeField) const {
    StepPush p{};
    p.nx = g_.nx; p.ny = g_.ny; p.nz = g_.nz;
    p.t = uint32_t(t);
    p.tau0    = flow.tau;
    p.smag2   = flow.smagorinsky * flow.smagorinsky;
    p.turb    = flow.turbulence;
    p.bouzidi = flow.bouzidi ? 1.f : 0.f;
    p.uin[0]  = flow.uIn;
    p.writeField = writeField ? 1u : 0u;
    p.faces = (g_.farFieldY ? 1u : 0u) | (g_.farFieldZ ? 2u : 0u) | (g_.periodicX ? 4u : 0u);
    return p;
}

void Solver::applyGeometry(VkCommandBuffer cmd) {
    auto& c = *ctx_;
    const StepPush p = push(false);
    gpu::Context::computeBarrier(cmd);
    c.bind(cmd, classify_, bindings(), &p, sizeof p);
    gpu::Context::dispatchThreads(cmd, cells());
    gpu::Context::computeBarrier(cmd);
    c.bind(cmd, wallmask_, bindings(), &p, sizeof p);
    gpu::Context::dispatchThreads(cmd, cells());
    gpu::Context::computeBarrier(cmd);
}

void Solver::reset(VkCommandBuffer cmd) {
    t = 0;
    auto b = bindings();
    b[23] = plane_.buf;
    const StepPush p = push(true);
    gpu::Context::computeBarrier(cmd);
    ctx_->bind(cmd, init_, b, &p, sizeof p);
    gpu::Context::dispatchThreads(cmd, cells());
    vkCmdFillBuffer(cmd, statPrev_.buf, 0, VK_WHOLE_SIZE, 0);
    gpu::Context::computeBarrier(cmd);
}

void Solver::recordSteps(VkCommandBuffer cmd, uint32_t n, bool field, bool forces,
                         uint32_t slot, bool timestamps) {
    auto& c = *ctx_;
    auto b = bindings();
    b[22] = forcePartials_.buf;
    b[23] = plane_.buf;
    if (timestamps) {
        vkCmdResetQueryPool(cmd, queries_, 2 * slot, 2);
        vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_NONE, queries_, 2 * slot);
    }
    for (uint32_t i = 0; i < n; ++i) {
        const bool last = i + 1 == n;
        const uint32_t parity = uint32_t(t & 1);
        const gpu::Kernel& k = (last && forces) ? stepForce_[parity] : step_[parity];
        const StepPush p = push(last && field);
        c.bind(cmd, k, b, &p, sizeof p);
        vkCmdDispatch(cmd, (g_.nx + wgx_ - 1) / wgx_, (g_.ny + wgy_ - 1) / wgy_, g_.nz);
        gpu::Context::computeBarrier(cmd);
        ++t;
    }
    if (timestamps)
        vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, queries_, 2 * slot + 1);

    if (forces && n > 0) {
        struct { uint32_t count, K, maxMask, pad; } rp{stepGroups(), 1, 0, 0};
        auto rb = c.emptyBindings();
        rb[22] = forcePartials_.buf;
        rb[23] = forceOut_[slot].buf;
        c.bind(cmd, reduce_, rb, &rp, sizeof rp);
        vkCmdDispatch(cmd, 1, 1, 1);
        gpu::Context::computeBarrier(cmd);
    }
}

void Solver::recordStats(VkCommandBuffer cmd, uint32_t slot) {
    auto& c = *ctx_;
    auto b = bindings();
    b[22] = statPrev_.buf;
    b[23] = statPartials_.buf;
    struct { uint32_t N, stride, count, pad; } sp{uint32_t(cells()), statStride_, statCount_, 0};
    c.bind(cmd, stats_, b, &sp, sizeof sp);
    gpu::Context::dispatchThreads(cmd, statCount_);
    gpu::Context::computeBarrier(cmd);

    // Second vec4 of each partial is (max|u|, kinetic energy): bit 4 = max.
    struct { uint32_t count, K, maxMask, pad; } rp{gpu::Context::groupCount(statCount_), 2, 1u << 4, 0};
    auto rb = c.emptyBindings();
    rb[22] = statPartials_.buf;
    rb[23] = statOut_[slot].buf;
    c.bind(cmd, reduce_, rb, &rp, sizeof rp);
    vkCmdDispatch(cmd, 2, 1, 1);
    gpu::Context::computeBarrier(cmd);
}

std::array<double, 3> Solver::forces(uint32_t slot) const {
    vmaInvalidateAllocation(ctx_->vma, forceOut_[slot].alloc, 0, VK_WHOLE_SIZE);
    const float* f = static_cast<const float*>(forceOut_[slot].map);
    return {f[0], f[1], f[2]};
}

Stats Solver::stats(uint32_t slot) const {
    vmaInvalidateAllocation(ctx_->vma, statOut_[slot].alloc, 0, VK_WHOLE_SIZE);
    const float* v = static_cast<const float*>(statOut_[slot].map);
    Stats s;
    s.samples  = uint32_t(v[3]);
    s.residual = v[1] > 0.f ? std::sqrt(v[0] / v[1]) : 1.f;
    s.rhoMean  = v[3] > 0.f ? v[2] / v[3] : 1.f;
    s.maxU     = v[4];
    s.kinetic  = v[5];
    return s;
}

double Solver::lastBatchMs(uint32_t slot) const {
    uint64_t ts[2] = {};
    vkGetQueryPoolResults(ctx_->device, queries_, 2 * slot, 2, sizeof ts, ts, sizeof(uint64_t),
                          VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
    return double(ts[1] - ts[0]) * ctx_->timestampPeriodNs * 1e-6;
}

void Solver::probe(std::vector<float>& rhoU) {
    auto& c = *ctx_;
    gpu::Buffer out = c.createBuffer(cells() * 16, 0, gpu::Mem::Device);
    auto b = bindings();
    b[22] = out.buf;
    b[23] = plane_.buf;
    const StepPush p = push(false);
    c.submitNow([&](VkCommandBuffer cmd) {
        gpu::Context::computeBarrier(cmd);
        c.bind(cmd, probe_, b, &p, sizeof p);
        gpu::Context::dispatchThreads(cmd, cells());
    });
    rhoU.resize(cells() * 4);
    c.download(out, rhoU.data(), cells() * 16);
    c.destroyBuffer(out);
}

} // namespace wt
