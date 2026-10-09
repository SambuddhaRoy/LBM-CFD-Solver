// ============================================================================
// tests.cpp — correctness tests against a CPU reference, and the throughput
// benchmark.
//
// The reference is the textbook two-buffer formulation: plain pull streaming
// from a separate post-collision array, bounce-back written out explicitly,
// the same collision and boundary faces, all in double precision. The GPU
// streams in place with Esoteric-Pull and gets bounce-back from slot parity,
// so agreement to round-off means the in-place bookkeeping, the race-free
// Bouzidi split, the outlet and the force reduction are all right.
// ============================================================================

#include "tests.hpp"

#include "fluid.hpp"
#include "geometry.hpp"
#include "lattice_cpu.hpp"
#include "solver.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace wt {

namespace {

using namespace cpu;

constexpr uint8_t SOLID = 1, EQ = 2, OUTLET = 4;

float halfToFloat(uint16_t h) {
    const uint32_t s = uint32_t(h & 0x8000) << 16, e = (h >> 10) & 0x1F, m = h & 0x3FF;
    uint32_t bits;
    if (e == 0) {
        if (m == 0) bits = s;
        else { float v = std::ldexp(float(m), -24); std::memcpy(&bits, &v, 4); bits |= s; }
    } else if (e == 31) bits = s | 0x7F800000 | (m << 13);
    else bits = s | ((e + 112) << 23) | (m << 13);
    float f; std::memcpy(&f, &bits, 4);
    return f;
}

// Recursive regularized collision + Smagorinsky (same algebra as step.comp).
void collide(const double f[19], double tau0, double smag2, double post[19], Vec3d& uOut, double& rhoOut) {
    double rho = 0; Vec3d u;
    for (int q = 0; q < 19; ++q) { rho += f[q]; u.x += CX[q]*f[q]; u.y += CY[q]*f[q]; u.z += CZ[q]*f[q]; }
    u.x /= rho; u.y /= rho; u.z /= rho;
    const double ul = std::sqrt(u.x*u.x + u.y*u.y + u.z*u.z);
    if (ul > 0.4) { u.x *= 0.4/ul; u.y *= 0.4/ul; u.z *= 0.4/ul; }
    double feq[19];
    equilibrium(rho, u, feq);
    double pxx = 0, pyy = 0, pzz = 0, pxy = 0, pyz = 0, pxz = 0;
    for (int q = 1; q < 19; ++q) {
        const double fn = f[q] - feq[q];
        pxx += fn*CX[q]*CX[q]; pyy += fn*CY[q]*CY[q]; pzz += fn*CZ[q]*CZ[q];
        pxy += fn*CX[q]*CY[q]; pyz += fn*CY[q]*CZ[q]; pxz += fn*CX[q]*CZ[q];
    }
    double tau = tau0;
    if (smag2 > 0) {
        const double Q = std::sqrt(pxx*pxx + pyy*pyy + pzz*pzz + 2*(pxy*pxy + pyz*pyz + pxz*pxz));
        tau = 0.5 * (tau0 + std::sqrt(tau0*tau0 + 18*std::sqrt(2.0)*smag2*Q/rho));
    }
    const double keep = 1 - 1/tau;
    const double a1[6] = {2*u.x*pxy + u.y*pxx, 2*u.z*pyz + u.y*pzz, 2*u.z*pxz + u.x*pzz,
                          2*u.y*pxy + u.x*pyy, 2*u.y*pyz + u.z*pyy, 2*u.x*pxz + u.z*pxx};
    const double bp[3] = {a1[0] + a1[1], a1[2] + a1[3], a1[4] + a1[5]};
    const double bm[3] = {a1[0] - a1[1], a1[2] - a1[3], a1[4] - a1[5]};
    for (int q = 0; q < 19; ++q) {
        const double cx = CX[q], cy = CY[q], cz = CZ[q];
        const double h2 = (cx*cx - 1.0/3)*pxx + (cy*cy - 1.0/3)*pyy + (cz*cz - 1.0/3)*pzz
                        + 2*(cx*cy*pxy + cy*cz*pyz + cx*cz*pxz);
        const double hxxy = cy*(cx*cx - 1.0/3), hyzz = cy*(cz*cz - 1.0/3);
        const double hxzz = cx*(cz*cz - 1.0/3), hxyy = cx*(cy*cy - 1.0/3);
        const double hyyz = cz*(cy*cy - 1.0/3), hxxz = cz*(cx*cx - 1.0/3);
        const double hp = (hxxy + hyzz)*bp[0] + (hxzz + hxyy)*bp[1] + (hyyz + hxxz)*bp[2];
        const double hm = (hxxy - hyzz)*bm[0] + (hxzz - hxyy)*bm[1] + (hyyz - hxxz)*bm[2];
        post[q] = feq[q] + keep * WQ[q] * (4.5*h2 + 13.5*hp + 4.5*hm);
    }
    uOut = u; rhoOut = rho;
}

class Reference {
public:
    Reference(int nx, int ny, int nz, std::vector<uint8_t> flags, std::vector<float> sdf,
              const FlowParams& fp)
        : nx_(nx), ny_(ny), nz_(nz), flags_(std::move(flags)), sdf_(std::move(sdf)), fp_(fp) {
        const size_t n = size_t(nx) * ny * nz;
        post_.assign(n * 19, 0);
        for (size_t c = 0; c < n; ++c) {
            const Vec3d u0 = (flags_[c] & SOLID) ? Vec3d{} : Vec3d{fp.uIn, 0, 0};
            equilibrium(1.0, u0, &post_[c * 19]);
        }
        plane_[0].assign(size_t(ny) * nz, Vec3d{fp.uIn, 0, 0});
        plane_[1] = plane_[0];
        for (int z = 0; z < nz; ++z) for (int y = 0; y < ny; ++y)
            if (flags_[idx(nx - 2, y, z)] & SOLID) plane_[0][size_t(z)*ny + y] = plane_[1][size_t(z)*ny + y] = {};
    }

    // Populations arriving at fluid cell c this step, with bounce-back.
    // Also returns, per direction, whether it arrived off a wall.
    void incoming(size_t c, double f[19], bool wallIn[19]) const {
        const int x = int(c % nx_), y = int((c / nx_) % ny_), z = int(c / (size_t(nx_) * ny_));
        f[0] = post_[c * 19];
        wallIn[0] = false;
        for (int q = 1; q < 19; ++q) {
            const size_t s = wrap(x - CX[q], y - CY[q], z - CZ[q]);
            wallIn[q] = (flags_[s] & SOLID) != 0;
            if (!wallIn[q]) { f[q] = post_[s * 19 + q]; continue; }
            const int d = OPP[q];                                // towards the wall
            const size_t other = wrap(x + CX[q], y + CY[q], z + CZ[q]);
            const bool otherSolid = (flags_[other] & SOLID) != 0;
            double qw = 0.5;
            if (fp_.bouzidi) {
                const double dd = double(sdf_[c]) - double(sdf_[s]);
                qw = dd > 1e-6 ? std::clamp(double(sdf_[c]) / dd, 0.0, 1.0) : 0.5;
            }
            // The GPU's first step finds unblended equilibrium in the wall
            // slots (initialisation stores no q >= 1/2 blend), so it is
            // halfway there; every later step sees the blend.
            if (otherSolid)       f[q] = post_[c * 19 + d];
            else if (qw < 0.5)    f[q] = 2*qw*post_[c * 19 + d] + (1 - 2*qw)*post_[other * 19 + d];
            else if (first_)      f[q] = post_[c * 19 + d];
            else { const double r = 0.5 / qw; f[q] = r*post_[c * 19 + d] + (1 - r)*post_[c * 19 + q]; }
        }
    }

    // One step; returns the momentum-exchange force taken during it.
    Vec3d step() {
        const size_t n = size_t(nx_) * ny_ * nz_;
        std::vector<double> next(post_);
        auto planeOut = plane_[t_ & 1];
        Vec3d force;
        for (size_t c = 0; c < n; ++c) {
            if (flags_[c] & SOLID) continue;
            const int x = int(c % nx_), y = int((c / nx_) % ny_), z = int(c / (size_t(nx_) * ny_));
            Vec3d u;
            double* out = &next[c * 19];
            if (flags_[c] & (EQ | OUTLET)) {
                u = (flags_[c] & OUTLET) ? plane_[(t_ + 1) & 1][size_t(z)*ny_ + y] : Vec3d{fp_.uIn, 0, 0};
                equilibrium(1.0, u, out);
            } else {
                double f[19]; bool wallIn[19];
                incoming(c, f, wallIn);
                double rho;
                collide(f, fp_.tau, double(fp_.smagorinsky) * fp_.smagorinsky, out, u, rho);
                for (int q = 1; q < 19; ++q) {
                    if (!wallIn[q]) continue;
                    const int d = OPP[q];
                    const double m = f[q] + out[d] + 2*WQ[d];
                    force.x += CX[d]*m; force.y += CY[d]*m; force.z += CZ[d]*m;
                }
            }
            if (x == nx_ - 2) planeOut[size_t(z)*ny_ + y] = u;
        }
        post_.swap(next);
        plane_[t_ & 1] = std::move(planeOut);
        ++t_;
        first_ = false;
        return force;
    }

    // rho, u of every cell as the GPU probe reports them.
    void macro(size_t c, double& rho, Vec3d& u) const {
        double f[19]; bool w[19];
        incoming(c, f, w);
        rho = 0; u = {};
        for (int q = 0; q < 19; ++q) { rho += f[q]; u.x += CX[q]*f[q]; u.y += CY[q]*f[q]; u.z += CZ[q]*f[q]; }
        u.x /= rho; u.y /= rho; u.z /= rho;
    }

    uint8_t flag(size_t c) const { return flags_[c]; }

private:
    size_t idx(int x, int y, int z) const { return (size_t(z) * ny_ + y) * nx_ + x; }
    size_t wrap(int x, int y, int z) const {
        return idx((x + nx_) % nx_, (y + ny_) % ny_, (z + nz_) % nz_);
    }

    int nx_, ny_, nz_;
    std::vector<uint8_t> flags_;
    std::vector<float> sdf_;
    FlowParams fp_;
    std::vector<double> post_;
    std::vector<Vec3d> plane_[2];
    uint64_t t_ = 0;
    bool first_ = true;
};

struct CaseResult { double maxDu, maxDrho, forceRel; };

CaseResult compareCase(gpu::Context& ctx, Precision prec, const Body& body, bool bouzidi,
                       uint32_t steps, bool farY, bool farZ) {
    GridConfig g;
    g.nx = 48; g.ny = 26; g.nz = 22;
    g.precision = prec; g.farFieldY = farY; g.farFieldZ = farZ;
    Solver s;
    s.create(ctx, g);
    s.flow.tau = 0.56f;
    s.flow.smagorinsky = 0.12f;
    s.flow.uIn = 0.08f;
    s.flow.bouzidi = bouzidi;
    Geometry geo;
    geo.create(ctx);
    geo.apply(s, body);
    ctx.submitNow([&](VkCommandBuffer cmd) { s.reset(cmd); });

    const size_t n = s.cells();
    std::vector<uint8_t> flags(n);
    ctx.download(s.flags(), flags.data(), n);
    std::vector<uint16_t> sdfHalf(n);
    ctx.download(s.sdf(), sdfHalf.data(), n * 2);
    std::vector<float> sdf(n);
    for (size_t i = 0; i < n; ++i) sdf[i] = halfToFloat(sdfHalf[i]);
    for (auto& fl : flags) fl &= 7;    // the reference derives wall cells itself

    Reference ref(int(g.nx), int(g.ny), int(g.nz), flags, sdf, s.flow);
    Vec3d refForce;
    for (uint32_t i = 0; i < steps; ++i) refForce = ref.step();
    ctx.submitNow([&](VkCommandBuffer cmd) { s.recordSteps(cmd, steps, false, true, 0); });
    const auto gpuForce = s.forces(0);

    std::vector<float> rhoU;
    s.probe(rhoU);
    CaseResult r{0, 0, 0};
    for (size_t c = 0; c < n; ++c) {
        if (ref.flag(c) & (SOLID | EQ | OUTLET)) continue;
        double rho; Vec3d u;
        ref.macro(c, rho, u);
        r.maxDrho = std::max(r.maxDrho, std::abs(rho - rhoU[c*4]));
        r.maxDu = std::max({r.maxDu, std::abs(u.x - rhoU[c*4 + 1]), std::abs(u.y - rhoU[c*4 + 2]),
                            std::abs(u.z - rhoU[c*4 + 3])});
    }
    const double fm = std::sqrt(refForce.x*refForce.x + refForce.y*refForce.y + refForce.z*refForce.z);
    const double df = std::sqrt(std::pow(refForce.x - gpuForce[0], 2) + std::pow(refForce.y - gpuForce[1], 2)
                                + std::pow(refForce.z - gpuForce[2], 2));
    r.forceRel = fm > 0 ? df / fm : df;
    geo.destroy();
    s.destroy();
    return r;
}

// Moments of the equilibrium: rho, rho u, rho uu + rho/3 I, and the six
// third-order combinations D3Q19 supports.
bool equilibriumMoments() {
    const double rho = 1.03;
    const Vec3d u{0.07, -0.05, 0.03};
    double f[19];
    equilibrium(rho, u, f);
    double m0 = 0, m1[3] = {}, m2[3][3] = {};
    for (int q = 0; q < 19; ++q) {
        const int c[3] = {CX[q], CY[q], CZ[q]};
        m0 += f[q];
        for (int a = 0; a < 3; ++a) { m1[a] += c[a]*f[q]; for (int b = 0; b < 3; ++b) m2[a][b] += c[a]*c[b]*f[q]; }
    }
    const double uv[3] = {u.x, u.y, u.z};
    double err = std::abs(m0 - rho);
    for (int a = 0; a < 3; ++a) {
        err = std::max(err, std::abs(m1[a] - rho*uv[a]));
        for (int b = 0; b < 3; ++b)
            err = std::max(err, std::abs(m2[a][b] - rho*uv[a]*uv[b] - (a == b ? rho/3 : 0.0)));
    }
    // Third order: sum f (H_xxy + H_yzz) must equal rho (ux^2 uy + uy uz^2), etc.
    double h3 = 0;
    for (int q = 0; q < 19; ++q) {
        const double cx = CX[q], cy = CY[q];
        h3 += f[q] * (cy*(cx*cx - 1.0/3) + cy*(CZ[q]*CZ[q] - 1.0/3));
    }
    err = std::max(err, std::abs(h3 - rho*(u.x*u.x*u.y + u.y*u.z*u.z)));
    std::printf("  equilibrium moments through 3rd order: max error %.2e  %s\n", err, err < 1e-12 ? "PASS" : "FAIL");
    return err < 1e-12;
}

// ─── Mesh voxelization ──────────────────────────────────────────────────────

// Unit-diameter icosphere (longest extent 1, like loadMesh output).
std::vector<Triangle> icosphere(int subdiv) {
    const float t = (1.f + std::sqrt(5.f)) / 2.f;
    std::vector<glm::vec3> v = {{-1, t, 0}, {1, t, 0}, {-1, -t, 0}, {1, -t, 0}, {0, -1, t}, {0, 1, t},
                                {0, -1, -t}, {0, 1, -t}, {t, 0, -1}, {t, 0, 1}, {-t, 0, -1}, {-t, 0, 1}};
    std::vector<std::array<int, 3>> f = {{0,11,5},{0,5,1},{0,1,7},{0,7,10},{0,10,11},{1,5,9},{5,11,4},
        {11,10,2},{10,7,6},{7,1,8},{3,9,4},{3,4,2},{3,2,6},{3,6,8},{3,8,9},{4,9,5},{2,4,11},{6,2,10},
        {8,6,7},{9,8,1}};
    for (int s = 0; s < subdiv; ++s) {
        std::vector<std::array<int, 3>> next;
        for (const auto& tri : f) {
            int m[3];
            for (int e = 0; e < 3; ++e) {
                v.push_back(0.5f * (v[tri[e]] + v[tri[(e + 1) % 3]]));
                m[e] = int(v.size()) - 1;
            }
            next.push_back({tri[0], m[0], m[2]}); next.push_back({tri[1], m[1], m[0]});
            next.push_back({tri[2], m[2], m[1]}); next.push_back({m[0], m[1], m[2]});
        }
        f = std::move(next);
    }
    std::vector<Triangle> out;
    for (const auto& tri : f)
        out.push_back({0.5f * glm::normalize(v[tri[0]]), 0.5f * glm::normalize(v[tri[1]]),
                       0.5f * glm::normalize(v[tri[2]])});
    return out;
}

// Torus around the z axis, ring radius R, tube radius r, with R + r = 1/2.
std::vector<Triangle> torusMesh(float R, float r, int nu, int nv) {
    auto p = [&](int i, int j) {
        const float a = 6.2831853f * float(i % nu) / float(nu), b = 6.2831853f * float(j % nv) / float(nv);
        return glm::vec3((R + r * std::cos(b)) * std::cos(a), (R + r * std::cos(b)) * std::sin(a), r * std::sin(b));
    };
    std::vector<Triangle> out;
    for (int i = 0; i < nu; ++i)
        for (int j = 0; j < nv; ++j) {
            out.push_back({p(i, j), p(i + 1, j), p(i + 1, j + 1)});
            out.push_back({p(i, j), p(i + 1, j + 1), p(i, j + 1)});
        }
    return out;
}

// Voxelizes `tris` as a rotated mesh body and compares every cell with the
// exact signed distance of the shape it approximates. A cell may only be
// misclassified within one cell of the true surface (faceting), and near the
// surface the stored distance must match the exact one.
bool meshCase(gpu::Context& ctx, const char* name, std::vector<Triangle> tris,
              float (*exact)(glm::vec3 unitBodyPoint)) {
    GridConfig g;
    g.nx = 72; g.ny = 64; g.nz = 64; g.precision = Precision::FP32;
    Solver s;
    s.create(ctx, g);
    Geometry geo;
    geo.create(ctx);
    const size_t nTris = tris.size();
    // Open meshes are shrink-wrapped: their surface moves out by ~0.866 cells.
    const bool closed = checkTopology(tris).watertight();
    const double offset = closed ? 0.0 : 0.866;
    geo.setMesh(std::move(tris), closed);
    Body b;
    b.shape = Shape::Mesh;
    b.center = {35.6f, 31.3f, 32.2f};
    b.length = 40.f;
    b.pitch = 20.f; b.yaw = 30.f; b.roll = -15.f;
    geo.apply(s, b);

    const size_t n = s.cells();
    std::vector<uint8_t> flags(n);
    std::vector<uint16_t> sdfHalf(n);
    ctx.download(s.flags(), flags.data(), n);
    ctx.download(s.sdf(), sdfHalf.data(), n * 2);
    const glm::mat3 toBody = bodyRotation(b);
    size_t solid = 0, wrong = 0;
    double worst = 0, sdfErr = 0;
    for (uint32_t z = 1; z + 1 < g.nz; ++z)
        for (uint32_t y = 1; y + 1 < g.ny; ++y)
            for (uint32_t x = 1; x + 1 < g.nx; ++x) {
                const size_t c = (size_t(z) * g.ny + y) * g.nx + x;
                const glm::vec3 p = toBody * (glm::vec3(float(x), float(y), float(z)) - b.center) / b.length;
                const double d = double(exact(p)) * b.length - offset;   // cells
                const bool gpuSolid = (flags[c] & SOLID) != 0;
                solid += gpuSolid;
                if (gpuSolid != (d < 0)) { ++wrong; worst = std::max(worst, std::abs(d)); }
                // Shrink-wrapped interiors measure distance to the nearest
                // triangle that exists, which overestimates next to a hole;
                // only the fluid side (what the walls use) is exact there.
                if (std::abs(d) < 3 && (closed || d > 0))
                    sdfErr = std::max(sdfErr, std::abs(double(halfToFloat(sdfHalf[c])) - d));
            }
    // Faceting moves the surface by at most the chord sagitta (well under a
    // cell here); fp16 storage rounds distances to ~0.002 cells.
    const bool pass = solid > 0 && worst < 1.0 && sdfErr < 0.35;
    std::printf("  mesh %-30s %-11s %6zu tris  %6zu solid  %3zu misclassified (all within %.2f cells of the surface)"
                "  max SDF error %.3f  %s\n", name, closed ? "exact" : "shrink-wrap", nTris, solid, wrong, worst,
                sdfErr, pass ? "PASS" : "FAIL");
    geo.destroy();
    s.destroy();
    return pass;
}

} // namespace

int runSelfTest(gpu::Context& ctx) {
    std::printf("\nSelf-test: GPU solver vs double-precision CPU reference\n");
    std::printf("  (48 x 26 x 22 grid, inlet + pressure outlet, LES on, 60 steps)\n\n");
    bool ok = equilibriumMoments();
    ok = fluidSelfTest() && ok;

    Body sphere;
    sphere.shape = Shape::Sphere; sphere.center = {16.3f, 12.6f, 10.8f}; sphere.length = 9.f;
    Body plate;                       // thin, pitched: diagonal links and one-cell gaps
    plate.shape = Shape::Cube; plate.center = {18.f, 13.f, 11.f}; plate.length = 7.f;
    plate.pitch = 28.f; plate.yaw = 17.f;
    Body wing;
    wing.shape = Shape::Wing; wing.center = {17.f, 13.f, 11.f}; wing.length = 16.f; wing.pitch = 12.f;

    struct Case { const char* name; Body body; bool bouzidi, farY, farZ; };
    const Case cases[] = {
        {"sphere, halfway walls",          sphere, false, true,  true },
        {"sphere, Bouzidi walls",          sphere, true,  true,  true },
        {"pitched cube, Bouzidi, periodic Z", plate, true, true,  false},
        {"NACA 0012 at 12 deg, Bouzidi",   wing,   true,  false, false},
    };
    for (const Precision prec : {Precision::FP32, Precision::FP16S, Precision::FP16C}) {
        // FP32 must agree to round-off; the 16-bit formats lose ~3-4 digits
        // per stored value, which bounds how closely they can follow.
        const double tolU = prec == Precision::FP32 ? 2e-6 : 2e-3;
        const double tolF = prec == Precision::FP32 ? 1e-4 : 5e-2;
        for (const Case& c : cases) {
            const CaseResult r = compareCase(ctx, prec, c.body, c.bouzidi, 60, c.farY, c.farZ);
            const bool pass = r.maxDu < tolU && r.maxDrho < tolU && r.forceRel < tolF;
            ok = ok && pass;
            std::printf("  %-6s %-36s max|du| %.2e  max|drho| %.2e  force rel %.2e  %s\n",
                        precisionName(prec), c.name, r.maxDu, r.maxDrho, r.forceRel, pass ? "PASS" : "FAIL");
        }
    }
    std::printf("\n");
    ok = meshCase(ctx, "icosphere, rotated", icosphere(4),
                  [](glm::vec3 p) { return glm::length(p) - 0.5f; }) && ok;
    // Non-convex with a through-hole: exercises the ray-parity sign.
    ok = meshCase(ctx, "torus, rotated", torusMesh(0.32f, 0.18f, 96, 48),
                  [](glm::vec3 p) {
                      const glm::vec2 q(glm::length(glm::vec2(p.x, p.y)) - 0.32f, p.z);
                      return glm::length(q) - 0.18f;
                  }) && ok;
    // Open meshes, as exported from visualisation models: a sphere with a
    // scattering of missing triangles (ray parity would streak through every
    // hole), and a single zero-thickness sheet (no inside at all).
    {
        auto holed = icosphere(4);
        for (size_t i = holed.size(); i-- > 0;) if (i % 37 == 0) holed.erase(holed.begin() + std::ptrdiff_t(i));
        ok = meshCase(ctx, "icosphere with holes, rotated", std::move(holed),
                      [](glm::vec3 p) { return glm::length(p) - 0.5f; }) && ok;
        const std::vector<Triangle> sheet = {{{-0.5f, -0.5f, 0}, {0.5f, -0.5f, 0}, {0.5f, 0.5f, 0}},
                                             {{-0.5f, -0.5f, 0}, {0.5f, 0.5f, 0}, {-0.5f, 0.5f, 0}}};
        ok = meshCase(ctx, "single sheet, rotated", sheet, [](glm::vec3 p) {
                          const glm::vec2 e = glm::max(glm::abs(glm::vec2(p.x, p.y)) - 0.5f, 0.f);
                          return std::sqrt(glm::dot(e, e) + p.z * p.z);
                      }) && ok;
    }

    std::printf("\n%s\n", ok ? "SELF-TEST PASSED" : "SELF-TEST FAILED");
    return ok ? 0 : 1;
}

int runBenchmark(gpu::Context& ctx, const BenchOptions& o) {
    std::printf("\nThroughput benchmark on %s (%.1f GiB)\n", ctx.deviceName.c_str(),
                double(ctx.vramBytes) / (1ull << 30));
    std::printf("  grid %u x %u x %u = %.1f M cells, %u timed steps, sphere obstacle\n\n",
                o.nx, o.ny, o.nz, double(o.nx) * o.ny * o.nz / 1e6, o.steps);
    std::printf("  precision   MLUPS    traffic GB/s   bytes/cell-step%s\n",
                o.peakGBs > 0 ? "   % of peak" : "");
    for (const Precision prec : o.precisions) {
        GridConfig g;
        g.nx = o.nx; g.ny = o.ny; g.nz = o.nz; g.precision = prec;
        Solver s;
        try { s.create(ctx, g); }
        catch (const std::exception& e) { std::printf("  %-8s  skipped: %s\n", precisionName(prec), e.what()); continue; }
        s.flow.tau = 0.51f;
        Geometry geo;
        geo.create(ctx);
        Body b;
        b.center = {0.3f * float(o.nx), 0.5f * float(o.ny), 0.5f * float(o.nz)};
        b.length = 0.2f * float(std::min(o.ny, o.nz));
        geo.apply(s, b);
        ctx.submitNow([&](VkCommandBuffer cmd) { s.reset(cmd); s.recordSteps(cmd, 20, false, false); });
        double best = 1e30;
        for (int rep = 0; rep < 3; ++rep) {
            ctx.submitNow([&](VkCommandBuffer cmd) { s.recordSteps(cmd, o.steps, false, false, 0, true); });
            best = std::min(best, s.lastBatchMs());
        }
        const double mlups = double(s.cells()) * o.steps / (best * 1e-3) / 1e6;
        // Per step every cell reads and writes 19 distributions and reads its flag byte.
        const double bytes = 19.0 * 2 * (prec == Precision::FP32 ? 4 : 2) + 1;
        const double gbs = mlups * 1e6 * bytes / 1e9;
        if (o.peakGBs > 0)
            std::printf("  %-8s  %7.0f   %10.0f       %5.0f          %5.1f%%\n", precisionName(prec), mlups, gbs,
                        bytes, 100.0 * gbs / o.peakGBs);
        else
            std::printf("  %-8s  %7.0f   %10.0f       %5.0f\n", precisionName(prec), mlups, gbs, bytes);
        geo.destroy();
        s.destroy();
    }
    return 0;
}

} // namespace wt
