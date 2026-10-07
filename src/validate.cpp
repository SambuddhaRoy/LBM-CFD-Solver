// ============================================================================
// validate.cpp — CFD validation against published bluff-body results.
//
//   Cylinder (2D)  Re 20, 40, 100, 150: drag coefficient, recirculation
//                  length, Strouhal number of vortex shedding
//   Sphere (3D)    Re 100: drag against the Schiller-Naumann correlation,
//                  wake length
//   Precision      the Re = 100 cylinder in FP32, FP16S and FP16C side by
//                  side: 16-bit storage must not move the answer
//
// The 2D cases run on a single cell in z with periodic z, which is an exact
// two-dimensional flow, so the domain can be large enough (5% blockage) that
// no blockage correction is applied: every coefficient reported here is raw.
// LES is off, these are laminar flows.
// ============================================================================

#include "geometry.hpp"
#include "solver.hpp"
#include "tests.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

namespace wt {

namespace {

constexpr float kU = 0.06f;                  // inlet speed, lattice units (Mach 0.10)

struct Measured {
    double cd = 0, cdLit = 0;
    double st = 0, stLit = 0;                // 0 = steady case
    double lr = 0, lrLit = 0;                // recirculation length / D, 0 = not measured
    double seconds = 0;
};

// Mean drag, and the Strouhal number from the up-crossings of the lift about
// its mean, with crossing times interpolated between samples.
struct ForceHistory {
    std::vector<double> t, drag, lift;

    double meanDrag() const {
        double s = 0; for (double d : drag) s += d;
        return drag.empty() ? 0 : s / double(drag.size());
    }
    // Shedding frequency in cycles per step, 0 if the lift does not oscillate.
    double frequency() const {
        if (lift.size() < 8) return 0;
        double mean = 0, amp = 0;
        for (double l : lift) mean += l;
        mean /= double(lift.size());
        for (double l : lift) amp = std::max(amp, std::abs(l - mean));
        if (amp < 1e-3 * std::max(std::abs(meanDrag()), 1e-9)) return 0;
        // Count a cycle only when the lift climbs through +band after having
        // been below -band: noise riding on the signal near its mean must
        // not register as extra cycles.
        const double band = 0.25 * amp;
        std::vector<double> up;
        bool armed = false;
        for (size_t i = 1; i < lift.size(); ++i) {
            const double a = lift[i - 1] - mean - band, b = lift[i] - mean - band;
            if (lift[i - 1] - mean < -band) armed = true;
            if (armed && a < 0 && b >= 0) {
                up.push_back(t[i - 1] + (t[i] - t[i - 1]) * (-a / (b - a)));
                armed = false;
            }
        }
        if (up.size() < 3) return 0;
        return double(up.size() - 1) / (up.back() - up.front());
    }
};

// Runs `warm` steps, then samples forces every `every` steps for `window`.
ForceHistory run(gpu::Context& ctx, Solver& s, uint32_t warm, uint32_t window, uint32_t every,
                 float kickSteps) {
    ForceHistory h;
    const uint64_t t0 = s.t;
    while (s.t - t0 < uint64_t(warm) + window) {
        // A brief inlet perturbation breaks the symmetry so shedding starts
        // promptly; it is switched off long before measurement.
        s.flow.turbulence = (s.t - t0) < uint64_t(kickSteps) ? 0.03f : 0.f;
        const bool sample = (s.t - t0) >= warm;
        ctx.submitNow([&](VkCommandBuffer cmd) { s.recordSteps(cmd, every, false, sample, 0); });
        if (sample) {
            const auto f = s.forces(0);
            h.t.push_back(double(s.t));
            h.drag.push_back(f[0]);
            h.lift.push_back(f[1]);
        }
    }
    return h;
}

// Length of the reversed-flow region behind the body along the wake centre
// line, in units of D, with the zero crossing interpolated.
double recirculation(const std::vector<float>& rhoU, const GridConfig& g, double xRear,
                     uint32_t y, uint32_t z, double D) {
    auto ux = [&](uint32_t x) { return double(rhoU[((size_t(z) * g.ny + y) * g.nx + x) * 4 + 1]); };
    uint32_t x = uint32_t(std::ceil(xRear));
    while (x < g.nx && ux(x) == 0.0) ++x;   // solid cells read exactly 0
    if (x + 1 >= g.nx || ux(x) >= 0) return 0;  // no reversed flow: attached
    while (x + 1 < g.nx && ux(x + 1) < 0) ++x;
    const double a = ux(x), b = ux(x + 1);
    const double x0 = double(x) + (-a / (b - a));
    return (x0 - xRear) / D;
}

Measured cylinder(gpu::Context& ctx, Precision prec, double Re, double cdLit, double lrLit,
                  double stLit) {
    const auto start = std::chrono::steady_clock::now();
    const float D = 40.f;
    GridConfig g;
    g.nx = 1600; g.ny = 800; g.nz = 1;
    g.precision = prec; g.farFieldY = true; g.farFieldZ = false;
    Solver s;
    s.create(ctx, g);
    s.flow.tau = 0.5f + 3.f * kU * D / float(Re);
    s.flow.smagorinsky = 0.f;
    s.flow.uIn = kU;
    Geometry geo;
    geo.create(ctx);
    Body b;
    b.shape = Shape::Cylinder;
    b.center = {400.f, 400.3f, 0.f};          // 10 D upstream, 30 D downstream
    b.length = D;
    geo.apply(s, b);
    ctx.submitNow([&](VkCommandBuffer cmd) { s.reset(cmd); });

    // Long enough to settle: ~20 flow-throughs of the body for the steady
    // cases, and ~15 shedding periods averaged for the unsteady ones.
    const bool shedding = stLit > 0;
    const uint32_t warm = shedding ? 90000 : 120000;
    const uint32_t window = shedding ? 70000 : 6000;
    const ForceHistory h = run(ctx, s, warm, window, 50, warm / 3.f);

    Measured m;
    const double q = 0.5 * kU * kU * D;       // per unit span, rho = 1
    m.cd = h.meanDrag() / q;  m.cdLit = cdLit;
    m.st = h.frequency() * D / kU; m.stLit = stLit;
    if (lrLit > 0) {
        std::vector<float> rhoU;
        s.probe(rhoU);
        m.lr = recirculation(rhoU, g, b.center.x + 0.5 * D, 400, 0, D);
        m.lrLit = lrLit;
    }
    geo.destroy();
    s.destroy();
    m.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return m;
}

Measured sphere(gpu::Context& ctx, Precision prec, double Re) {
    const auto start = std::chrono::steady_clock::now();
    const float D = 40.f;
    GridConfig g;
    g.nx = 480; g.ny = 240; g.nz = 240;       // blockage 2.2%
    g.precision = prec;
    Solver s;
    s.create(ctx, g);
    s.flow.tau = 0.5f + 3.f * kU * D / float(Re);
    s.flow.smagorinsky = 0.f;
    s.flow.uIn = kU;
    Geometry geo;
    geo.create(ctx);
    Body b;
    b.shape = Shape::Sphere;
    b.center = {140.f, 120.3f, 120.3f};
    b.length = D;
    geo.apply(s, b);
    ctx.submitNow([&](VkCommandBuffer cmd) { s.reset(cmd); });
    const ForceHistory h = run(ctx, s, 36000, 4000, 50, 0.f);

    Measured m;
    const double area = 3.14159265358979 * 0.25 * D * D;
    m.cd = h.meanDrag() / (0.5 * kU * kU * area);
    m.cdLit = 24.0 / Re * (1.0 + 0.15 * std::pow(Re, 0.687));    // Schiller-Naumann
    std::vector<float> rhoU;
    s.probe(rhoU);
    m.lr = recirculation(rhoU, g, b.center.x + 0.5 * D, 120, 120, D);
    m.lrLit = 0.88;                           // Taneda (1956), Re = 100
    geo.destroy();
    s.destroy();
    m.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return m;
}

bool within(double v, double lit, double tol) { return lit > 0 && std::abs(v - lit) <= tol * lit; }

void row(const char* label, const Measured& m, bool& ok, double tolCd, double tolSt, double tolLr) {
    char st[32] = "    --      ", lr[32] = "    --      ";
    if (m.stLit > 0) std::snprintf(st, sizeof st, "%.3f (%.3f)", m.st, m.stLit);
    if (m.lrLit > 0) std::snprintf(lr, sizeof lr, "%.2f (%.2f)", m.lr, m.lrLit);
    const bool pass = within(m.cd, m.cdLit, tolCd)
                   && (m.stLit == 0 || within(m.st, m.stLit, tolSt))
                   && (m.lrLit == 0 || within(m.lr, m.lrLit, tolLr));
    ok = ok && pass;
    std::printf("  %-14s %.3f (%.3f) %+6.1f%%   %-14s  %-12s  %5.0fs  %s\n", label, m.cd, m.cdLit,
                100.0 * (m.cd / m.cdLit - 1.0), st, lr, m.seconds, pass ? "PASS" : "FAIL");
}

} // namespace

int runValidation(gpu::Context& ctx, Precision prec) {
    bool ok = true;
    std::printf("\nCFD validation, %s storage. Measured (published) values.\n", precisionName(prec));
    std::printf("Tolerances: C_D 8%%, St 6%%, recirculation length 12%%.\n\n");

    // Published references: Dennis & Chang (1970) and Coutanceau & Bouard
    // (1977) for the steady wake; Park, Kwon & Choi (1998) and Williamson
    // (1996) for drag and Strouhal number in the shedding regime.
    std::printf("  CYLINDER, D = 40 cells, 1600 x 800, blockage 5%%\n");
    std::printf("  Re             C_D                      St              L_r / D        time\n");
    struct Case { double re, cd, lr, st; };
    const Case cyl[] = {
        { 20, 2.05, 0.94, 0     },
        { 40, 1.52, 2.24, 0     },   // L_r/D published 2.13-2.35: midpoint
        {100, 1.33, 0,    0.165 },
        {150, 1.32, 0,    0.184 },
    };
    for (const Case& c : cyl) {
        char label[16]; std::snprintf(label, sizeof label, "%.0f", c.re);
        row(label, cylinder(ctx, prec, c.re, c.cd, c.lr, c.st), ok, 0.08, 0.06, 0.12);
    }

    std::printf("\n  SPHERE, D = 40 cells, 480 x 240 x 240, blockage 2.2%%\n");
    row("100", sphere(ctx, prec, 100), ok, 0.08, 0.06, 0.12);

    std::printf("\n  STORAGE PRECISION, cylinder Re = 100\n");
    for (const Precision p : {Precision::FP32, Precision::FP16S, Precision::FP16C})
        row(precisionName(p), cylinder(ctx, p, 100, 1.33, 0, 0.165), ok, 0.08, 0.06, 0.12);

    std::printf("\n%s\n", ok ? "VALIDATION PASSED" : "VALIDATION FAILED");
    return ok ? 0 : 1;
}

} // namespace wt
