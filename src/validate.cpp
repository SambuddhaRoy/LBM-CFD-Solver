// ============================================================================
// validate.cpp — Reynolds-resolved bluff-body validation
//
// For each case the harness:
//   1. sets nu (via tau) so that Re = U*D/nu hits the target,
//   2. builds a full-span (2D-equivalent) obstacle so free-slip walls make the
//      flow mathematically two-dimensional,
//   3. seeds the wake with a brief inlet perturbation, then removes it,
//   4. records the cross-flow force every batch to detect vortex shedding and
//      its Strouhal number, and
//   5. reads the velocity field back to measure the recirculation length and
//      locate the vortex core.
//
// Strouhal is the headline metric: it is purely kinematic (a wake frequency),
// so it validates the flow dynamics independently of any force-magnitude error.
// ============================================================================

#include "validate.h"

#include "gpu.h"
#include "mesh.h"
#include "sim.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>

namespace vwt {

namespace {

constexpr float kUin = 0.06f;   // inlet speed, lattice units (Mach ~0.10)

// ─── Geometry helpers ────────────────────────────────────────────────────────

struct Extent { int x0, x1, y0, y1, z0, z1; bool any; };

Extent measureExtent(const std::vector<uint32_t>& occ,
                     uint32_t gx, uint32_t gy, uint32_t gz) {
    Extent e{ int(gx), -1, int(gy), -1, int(gz), -1, false };
    for (uint32_t z = 0; z < gz; ++z)
        for (uint32_t y = 0; y < gy; ++y)
            for (uint32_t x = 0; x < gx; ++x)
                if (occ[(size_t(z)*gy + y)*gx + x]) {
                    e.any = true;
                    e.x0 = std::min(e.x0, int(x)); e.x1 = std::max(e.x1, int(x));
                    e.y0 = std::min(e.y0, int(y)); e.y1 = std::max(e.y1, int(y));
                    e.z0 = std::min(e.z0, int(z)); e.z1 = std::max(e.z1, int(z));
                }
    return e;
}

// Project the obstacle through the span axis so it becomes uniform there;
// with free-slip span walls this is an exact infinite-span (2D) body.
void extrudeAlong(std::vector<uint32_t>& occ, uint32_t gx, uint32_t gy,
                  uint32_t gz, int spanAxis /*1=Y, 2=Z*/) {
    if (spanAxis == 2) {
        for (uint32_t y = 0; y < gy; ++y)
            for (uint32_t x = 0; x < gx; ++x) {
                bool any = false;
                for (uint32_t z = 0; z < gz && !any; ++z)
                    any = occ[(size_t(z)*gy + y)*gx + x];
                if (any) for (uint32_t z = 0; z < gz; ++z)
                    occ[(size_t(z)*gy + y)*gx + x] = 1u;
            }
    } else {
        for (uint32_t z = 0; z < gz; ++z)
            for (uint32_t x = 0; x < gx; ++x) {
                bool any = false;
                for (uint32_t y = 0; y < gy && !any; ++y)
                    any = occ[(size_t(z)*gy + y)*gx + x];
                if (any) for (uint32_t y = 0; y < gy; ++y)
                    occ[(size_t(z)*gy + y)*gx + x] = 1u;
            }
    }
}

// Analytic infinite cylinder (axis = Z) of diameter D centred at (cx,cy).
std::vector<uint32_t> makeCylinder2D(uint32_t gx, uint32_t gy, uint32_t gz,
                                     float cx, float cy, float D) {
    std::vector<uint32_t> occ(size_t(gx)*gy*gz, 0u);
    const float r2 = (D*0.5f)*(D*0.5f);
    for (uint32_t z = 0; z < gz; ++z)
        for (uint32_t y = 0; y < gy; ++y)
            for (uint32_t x = 0; x < gx; ++x) {
                const float dx = float(x)+0.5f - cx, dy = float(y)+0.5f - cy;
                if (dx*dx + dy*dy <= r2)
                    occ[(size_t(z)*gy + y)*gx + x] = 1u;
            }
    return occ;
}

// Analytic infinite square prism (axis = Z), side D centred at (cx,cy).
std::vector<uint32_t> makeSquare2D(uint32_t gx, uint32_t gy, uint32_t gz,
                                   float cx, float cy, float D) {
    std::vector<uint32_t> occ(size_t(gx)*gy*gz, 0u);
    const float h = D*0.5f;
    for (uint32_t z = 0; z < gz; ++z)
        for (uint32_t y = 0; y < gy; ++y)
            for (uint32_t x = 0; x < gx; ++x) {
                const float dx = float(x)+0.5f - cx, dy = float(y)+0.5f - cy;
                if (std::abs(dx) <= h && std::abs(dy) <= h)
                    occ[(size_t(z)*gy + y)*gx + x] = 1u;
            }
    return occ;
}

// ─── Field readback ──────────────────────────────────────────────────────────

void downloadMacro(GpuContext& gpu, VkBuffer macro, size_t cells,
                   std::vector<float>& out) {
    const VkDeviceSize sz = VkDeviceSize(cells) * 4 * sizeof(float);
    GpuBuffer host = gpu.createBuffer(sz, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                      MemLoc::HostRead);
    gpu.oneShot([&](VkCommandBuffer cmd) {
        VkMemoryBarrier mb{};
        mb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
        VkBufferCopy cr{ 0, 0, sz };
        vkCmdCopyBuffer(cmd, macro, host.buffer, 1, &cr);
    });
    vmaInvalidateAllocation(gpu.allocator(), host.alloc, 0, VK_WHOLE_SIZE);
    out.resize(cells * 4);
    std::memcpy(out.data(), host.mapped, sz);
    gpu.destroyBuffer(host);
}

// ─── Vorticity BMP dump (viridis) ────────────────────────────────────────────

void viridis(float t, uint8_t& r, uint8_t& g, uint8_t& b) {
    t = std::clamp(t, 0.f, 1.f);
    auto P = [t](double a,double bb,double c,double d,double e,double f,double gg) {
        return float(a+t*(bb+t*(c+t*(d+t*(e+t*(f+t*gg))))));
    };
    auto cl = [](float v){ return uint8_t(std::clamp(v,0.f,1.f)*255.f); };
    r = cl(P( 0.2777, 0.1051,-0.3308,-4.6342, 6.2283, 4.7764,-5.4355));
    g = cl(P( 0.0054, 1.4046, 0.2148,-5.7991,14.1799,-13.7451, 4.6459));
    b = cl(P( 0.3341, 1.3846, 0.0951,-19.3324,56.6906,-65.3530,26.3124));
}

void dumpVorticityBmp(const std::filesystem::path& path,
                      const std::vector<float>& macro,
                      uint32_t gx, uint32_t gy, uint32_t gz,
                      uint32_t spanMid, int crossAxis, float scale) {
    // Render the flow plane (X × cross axis) at the mid-span slice.
    const uint32_t W = gx;
    const uint32_t H = (crossAxis == 1) ? gy : gz;
    const int crossComp = (crossAxis == 1) ? 2 : 3;   // uy or uz
    auto vel = [&](int x, int c, int comp) -> float {
        x = std::clamp(x, 0, int(gx)-1);
        c = std::clamp(c, 0, int(H)-1);
        const size_t cell = (crossAxis == 1)
            ? (size_t(spanMid)*gy + c)*gx + x          // span=Z: cross is Y
            : (size_t(c)*gy + spanMid)*gx + x;         // span=Y: cross is Z
        return macro[cell*4 + comp];
    };
    const uint32_t rowBytes = (W*3 + 3) & ~3u;
    std::vector<uint8_t> img(size_t(rowBytes)*H, 0);
    for (uint32_t c = 0; c < H; ++c)
        for (uint32_t x = 0; x < W; ++x) {
            // omega = d(u_cross)/dx - d(u_x)/d(cross)
            const float ducr = (vel(int(x)+1,int(c),crossComp) - vel(int(x)-1,int(c),crossComp))*0.5f;
            const float dux  = (vel(int(x),int(c)+1,1) - vel(int(x),int(c)-1,1))*0.5f;
            const float w = (ducr - dux);
            uint8_t r,g,b; viridis(0.5f + 0.5f*std::clamp(w/scale,-1.f,1.f), r,g,b);
            uint8_t* px = &img[size_t(c)*rowBytes + x*3];
            px[0]=b; px[1]=g; px[2]=r;
        }
    std::ofstream f(path, std::ios::binary);
    if (!f.is_open()) return;
    const uint32_t dataSize = rowBytes*H, fileSize = 54 + dataSize, off=54, ih=40;
    const uint16_t planes=1, bpp=24;
    uint8_t hdr[54]={}; hdr[0]='B'; hdr[1]='M';
    std::memcpy(hdr+2,&fileSize,4); std::memcpy(hdr+10,&off,4);
    std::memcpy(hdr+14,&ih,4); std::memcpy(hdr+18,&W,4); std::memcpy(hdr+22,&H,4);
    std::memcpy(hdr+26,&planes,2); std::memcpy(hdr+28,&bpp,2);
    std::memcpy(hdr+34,&dataSize,4);
    f.write(reinterpret_cast<char*>(hdr),54);
    f.write(reinterpret_cast<char*>(img.data()), std::streamsize(img.size()));
}

// ─── Case driver ─────────────────────────────────────────────────────────────

struct CaseResult {
    float Re=0, tau=0, D=0;
    float meanCd=0, clMean=0, clAmp=0;
    float strouhal=0;
    bool  shedding=false;
    bool  separated=false;
    float recircLD=0;
    float vortX=0, vortY=0;        // vortex core, units of D, relative to body rear/centre
    float residual=1;
    float maxU=0;                  // peak |u| (lattice) — freestream sanity check
    bool  valid=false;
};

float tauForRe(float Re, float D) {
    // Re = U D / nu, nu = (tau - 1/2)/3  →  tau = 1/2 + 3 U D / Re
    return 0.5f + 3.f * kUin * D / std::max(Re, 0.1f);
}

CaseResult runCase(GpuContext& gpu, uint32_t gx, uint32_t gy, uint32_t gz,
                   std::vector<uint32_t> occ, float D, float Re,
                   int spanAxis, uint32_t warmup, uint32_t window, uint32_t batch,
                   const std::filesystem::path& bmpPath,
                   const std::vector<float>* sdf = nullptr) {
    CaseResult R;
    R.Re = Re; R.D = D;
    R.tau = std::clamp(tauForRe(Re, D), 0.505f, 6.f);

    const int crossAxis = (spanAxis == 2) ? 1 : 2;   // 1=Y, 2=Z

    SimParams p;
    p.gx = gx; p.gy = gy; p.gz = gz;
    p.tau = R.tau; p.uIn = kUin;
    // TRT collision: the magic parameter Lambda=3/16 fixes the bounce-back
    // wall at the link midpoint independent of viscosity, so the effective
    // Reynolds number matches the nominal one (plain bounce-back shifts it).
    p.collision = 2;
    p.les = false;          // laminar shedding: no turbulence model
    p.turb = 0.f;

    Extent e = measureExtent(occ, gx, gy, gz);
    const int xRear = e.x1;
    const float crossC = (crossAxis == 1) ? 0.5f*(e.y0+e.y1) : 0.5f*(e.z0+e.z1);
    const uint32_t spanMid = (spanAxis == 2) ? gz/2 : gy/2;

    Solver solver;
    solver.init(gpu, p);
    solver.uploadObstacles(occ);
    if (sdf) solver.setSDF(*sdf);   // exact interpolated bounce-back
    solver.reset();

    // Reference area = frontal projection along the flow (X): the silhouette
    // the flow actually sees, = D x span. (Summing all solid cells and
    // dividing by span gives the cross-sectional AREA, ~pi/4 too large for a
    // cylinder — wrong for C_D/C_L.)
    const uint32_t frontal = [&]{
        uint32_t f = 0;
        for (uint32_t z = 0; z < gz; ++z)
            for (uint32_t y = 0; y < gy; ++y) {
                bool any = false;
                for (uint32_t x = 0; x < gx && !any; ++x)
                    any = occ[(size_t(z)*gy + y)*gx + x] != 0u;
                if (any) ++f;
            }
        return f;
    }();
    const float q = 0.5f * p.uIn * p.uIn;
    const float A = (frontal > 0) ? float(frontal) : 1.f;

    std::vector<float> perp;        // cross-flow force coefficient series
    perp.reserve(window / batch + 4);
    double cdAcc = 0; int cdN = 0;

    uint64_t step = 0;
    while (step < uint64_t(warmup) + window) {
        p.turb = (step < warmup/2) ? 0.15f : 0.f;   // seed, then let it self-sustain
        gpu.oneShot([&](VkCommandBuffer cmd) {
            solver.recordSteps(cmd, p, batch, uint32_t(step), false);
            solver.recordAnalysis(cmd, p, 0);
        });
        step += batch;
        const Analysis a = solver.readAnalysis(0);
        R.residual = a.residual;
        const float cd  = a.drag / (q * A);
        const float per = ((crossAxis == 1) ? a.lift : a.side) / (q * A);
        if (step > warmup) { perp.push_back(per); cdAcc += cd; ++cdN; }
        R.maxU = a.maxU;
        if (!a.valid) break;
    }

    R.meanCd = cdN ? float(cdAcc / cdN) : 0.f;

    // Cross-flow force statistics → shedding + Strouhal
    if (!perp.empty()) {
        double mean = 0; for (float v : perp) mean += v; mean /= perp.size();
        double var = 0, pk = 0;
        for (float v : perp) { var += (v-mean)*(v-mean); pk = std::max(pk, std::abs(v-mean)); }
        R.clMean = float(mean);
        R.clAmp  = float(pk);
        const float rms = float(std::sqrt(var / perp.size()));

        // up-crossings of the mean → shedding cycles
        int cross = 0;
        for (size_t i = 1; i < perp.size(); ++i)
            if (perp[i-1]-mean < 0 && perp[i]-mean >= 0) ++cross;
        R.shedding = (rms > 0.01f) && (cross >= 3);
        if (R.shedding) {
            const double stepsInWindow = double(perp.size()) * batch;
            const double f = double(cross) / stepsInWindow;   // cycles per step
            R.strouhal = float(f * D / p.uIn);
        }
    }

    // Field analysis: recirculation length + vortex core on the wake centreline
    std::vector<float> macro;
    downloadMacro(gpu, solver.macroBuffer(), solver.cells(), macro);

    auto cellAt = [&](int x, int cross) -> size_t {
        if (spanAxis == 2)   // span Z: vary X and Y(=cross), z = spanMid
            return (size_t(spanMid)*gy + cross)*gx + x;
        else                 // span Y: vary X and Z(=cross), y = spanMid
            return (size_t(cross)*gy + spanMid)*gx + x;
    };
    auto ux = [&](int x, int cross){ return macro[cellAt(x,cross)*4 + 1]; };
    auto ucr = [&](int x, int cross){
        return macro[cellAt(x,cross)*4 + (crossAxis==1 ? 2 : 3)];
    };

    const int cc = int(std::lround(crossC));
    int lastNeg = -1;
    for (int x = xRear + 1; x < int(gx) - 2; ++x) {
        if (ux(x, cc) < 0.f) lastNeg = x;
        else if (lastNeg >= 0 && ux(x, cc) >= 0.f && x > lastNeg + 1) break;
    }
    R.separated = (lastNeg >= 0);
    R.recircLD  = R.separated ? float(lastNeg - xRear) / D : 0.f;

    // Vortex core: max |omega_z| in the near wake (time snapshot)
    float bestW = 0; int bx = xRear, bcr = cc;
    const int crossMax = (crossAxis == 1) ? int(gy)-2 : int(gz)-2;
    for (int x = xRear + 1; x < std::min(int(gx)-2, xRear + int(3*D)); ++x)
        for (int c = std::max(1, cc - int(1.5f*D));
             c < std::min(crossMax, cc + int(1.5f*D)); ++c) {
            const float duc = (ucr(x+1,c) - ucr(x-1,c))*0.5f;
            const float dux = (ux(x,c+1) - ux(x,c-1))*0.5f;
            const float w = std::abs(duc - dux);
            if (w > bestW) { bestW = w; bx = x; bcr = c; }
        }
    R.vortX = float(bx - xRear) / D;
    R.vortY = float(bcr - cc) / D;

    if (!bmpPath.empty())
        dumpVorticityBmp(bmpPath, macro, gx, gy, gz, spanMid, crossAxis,
                         0.6f * p.uIn / D * 8.f);

    R.valid = std::isfinite(R.meanCd) && std::isfinite(R.strouhal);
    solver.destroy();
    return R;
}

const char* regimeName(const CaseResult& r) {
    if (r.shedding)      return "Von Karman vortex street (unsteady)";
    if (r.separated)     return "steady recirculation (twin vortices)";
    return "attached flow (no separation)";
}

void printRow(const CaseResult& r) {
    char st[16];
    if (r.shedding) std::snprintf(st, sizeof(st), "%.3f", r.strouhal);
    else            std::snprintf(st, sizeof(st), "  --  ");
    std::printf("  %5.0f  %6.3f  %7.3f  %8s  %7.2f  %7.3f  maxU=%.3f  %.1e  %s\n",
                r.Re, r.tau, r.meanCd, st, r.recircLD, r.clAmp, r.maxU, r.residual,
                regimeName(r));
}

} // namespace

// ════════════════════════════════════════════════════════════════════════════

int runValidation(const ValidateOptions& opts) {
    GpuContext gpu;
    gpu.init(nullptr);

    const auto outDir = exeDir() / "validation";
    std::error_code ec; std::filesystem::create_directories(outDir, ec);

    std::printf("\n");
    std::printf("==============================================================\n");
    std::printf("  Virtual Wind Tunnel v2 — CFD validation suite\n");
    std::printf("  GPU: %s\n", gpu.gpuName());
    std::printf("==============================================================\n");

    bool allPass = true;

    // ── Cylinder Reynolds sweep (analytic, infinite span) ───────────────────
    {
        const uint32_t gx=680, gy=340, gz=6;
        const float D = 40.f;
        const float cx = 0.25f*gx, cy = 0.5f*gy;   // blockage ~0.118

        std::printf("\n  CYLINDER — diameter %.0f cells, blockage %.2f, U=%.2f\n",
                    D, D/gy, kUin);
        std::printf("  ----------------------------------------------------------\n");
        std::printf("    Re     tau    C_D     St      L_r/D   |C_L|    regime\n");

        // Exact signed-distance field for the circle → interpolated bounce-back
        // that places the curved wall at its true sub-cell position.
        std::vector<float> sdf(size_t(gx)*gy*gz);
        for (uint32_t z = 0; z < gz; ++z)
            for (uint32_t yy = 0; yy < gy; ++yy)
                for (uint32_t xx = 0; xx < gx; ++xx) {
                    const float dx = float(xx)+0.5f - cx, dy = float(yy)+0.5f - cy;
                    sdf[(size_t(z)*gy + yy)*gx + xx] = std::sqrt(dx*dx + dy*dy) - 0.5f*D;
                }

        struct Spec { float Re; uint32_t warm, win; const char* img; };
        const Spec specs[] = {
            {   2.f,  16000,  4000, "cyl_Re2.bmp"   },
            {  40.f,  80000,  6000, "cyl_Re40.bmp"  },
            { 100.f, 130000, 70000, "cyl_Re100.bmp" },
            { 150.f, 150000, 80000, "cyl_Re150.bmp" },
        };
        std::vector<CaseResult> results;
        for (const Spec& s : specs) {
            auto occ = makeCylinder2D(gx, gy, gz, cx, cy, D);
            const auto img = s.img ? (outDir / s.img) : std::filesystem::path{};
            CaseResult r = runCase(gpu, gx, gy, gz, std::move(occ), D, s.Re,
                                   2 /*span Z*/, s.warm, s.win, 100, img, &sdf);
            printRow(r);
            results.push_back(r);
        }

        auto findRe = [&](float re)->const CaseResult&{
            for (auto& r : results) if (std::abs(r.Re-re)<1.f) return r; return results[0];
        };
        const CaseResult& re2   = findRe(2.f);
        const CaseResult& re40  = findRe(40.f);
        const CaseResult& re100 = findRe(100.f);
        const CaseResult& re150 = findRe(150.f);

        std::printf("\n  Literature: attached Re<5; steady recirc 5<Re<47 with\n");
        std::printf("  L_r/D ~ 0.05 Re (=> ~2.1 at Re=40); shedding Re>47 with\n");
        std::printf("  St ~ 0.164 (Re=100), 0.184 (Re=150).\n\n");

        struct Chk { const char* name; bool ok; };
        const Chk checks[] = {
            { "Re=2 attached (no separation)",      !re2.shedding && re2.recircLD < 0.3f },
            { "Re=40 steady recirculation",         !re40.shedding && re40.separated },
            { "Re=40 recirc length 1.7-2.5 D",      re40.recircLD > 1.7f && re40.recircLD < 2.5f },
            { "Re=100 vortex shedding",             re100.shedding },
            { "Re=100 Strouhal 0.14-0.19",          re100.strouhal > 0.14f && re100.strouhal < 0.19f },
            { "Re=150 Strouhal 0.16-0.21",          re150.strouhal > 0.16f && re150.strouhal < 0.21f },
            { "Strouhal rises with Re",             re150.strouhal > re100.strouhal },
        };
        for (const Chk& c : checks) {
            std::printf("    [%s] %s\n", c.ok ? "PASS":"FAIL", c.name);
            allPass = allPass && c.ok;
        }
    }

    // ── Square prism (analytic) ─────────────────────────────────────────────
    {
        const uint32_t gx=680, gy=340, gz=6;
        const float D = 36.f;
        const float cx = 0.25f*gx, cy = 0.5f*gy;
        std::printf("\n  SQUARE PRISM / CUBE CROSS-SECTION — side %.0f cells\n", D);
        std::printf("  ----------------------------------------------------------\n");
        std::printf("    Re     tau    C_D     St      L_r/D   |C_L|    regime\n");

        struct Spec { float Re; uint32_t warm, win; const char* img; };
        const Spec specs[] = {
            {  40.f,  80000,  6000, "square_Re40.bmp"  },
            { 200.f, 160000, 90000, "square_Re200.bmp" },
        };
        CaseResult hi{};
        for (const Spec& s : specs) {
            auto occ = makeSquare2D(gx, gy, gz, cx, cy, D);
            const auto img = s.img ? (outDir / s.img) : std::filesystem::path{};
            CaseResult r = runCase(gpu, gx, gy, gz, std::move(occ), D, s.Re,
                                   2, s.warm, s.win, 100, img);
            printRow(r);
            if (std::abs(s.Re-200.f)<1.f) hi = r;
        }
        std::printf("\n    Wake: separates at the leading edges, recirculation bubble\n");
        std::printf("    behind the body; vortex core (Re=200 snapshot) at x=%.2f D\n",
                    hi.vortX);
        std::printf("    downstream, y=%.2f D off the centreline.\n", hi.vortY);
        const bool ok = hi.shedding && hi.strouhal > 0.05f && hi.strouhal < 0.30f;
        std::printf("    [%s] square-body Von Karman shedding with finite Strouhal\n",
                    ok ? "PASS":"FAIL");
        allPass = allPass && ok;
    }

    // ── Provided models ─────────────────────────────────────────────────────
    auto runProvided = [&](const std::string& path, const char* label,
                           float Re, const char* img) {
        std::vector<Tri> tris; std::string err;
        if (!mesh::loadTriangles(path, tris, err)) {
            std::printf("\n  %s: import failed (%s)\n", label, err.c_str());
            return;
        }
        const uint32_t gx=300, gy=140, gz=140;
        VoxelModel m = mesh::voxelizeTriangles(tris, gx, gy, gz, 0.f, 0.f, label);
        Extent e = measureExtent(m.occupancy, gx, gy, gz);
        const int dy = e.y1-e.y0+1, dz = e.z1-e.z0+1;
        const int spanAxis = (dy >= dz) ? 1 : 2;     // longest cross-axis = span
        extrudeAlong(m.occupancy, gx, gy, gz, spanAxis);
        Extent e2 = measureExtent(m.occupancy, gx, gy, gz);
        const float D = (spanAxis == 2) ? float(e2.y1-e2.y0+1)
                                        : float(e2.z1-e2.z0+1);
        std::printf("\n  PROVIDED MODEL: %s  (D=%.0f cells, span axis %c)\n",
                    label, D, spanAxis==1?'Y':'Z');
        std::printf("    Re     tau    C_D     St      L_r/D   |C_L|    regime\n");
        const auto out = img ? (outDir / img) : std::filesystem::path{};
        CaseResult r = runCase(gpu, gx, gy, gz, std::move(m.occupancy), D, Re,
                               spanAxis, 70000, 45000, 100, out);
        printRow(r);
    };
    if (!opts.cylinderMesh.empty())
        runProvided(opts.cylinderMesh, "cylinder.glb", 220.f, "provided_cylinder.bmp");
    if (!opts.cubeMesh.empty())
        runProvided(opts.cubeMesh, "CUBE1.stl", 220.f, "provided_cube.bmp");

    std::printf("\n==============================================================\n");
    std::printf("  %s\n", allPass ? "VALIDATION PASSED — physics matches canonical results"
                                   : "VALIDATION INCOMPLETE — see failed checks above");
    if (opts.dumpImages)
        std::printf("  Vorticity field images written to: %s\n",
                    outDir.string().c_str());
    std::printf("==============================================================\n\n");

    gpu.destroy();
    return allPass ? 0 : 1;
}

} // namespace vwt
