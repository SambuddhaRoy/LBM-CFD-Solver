// ============================================================================
// geometry.cpp — analytic bodies, mesh import, GPU signed-distance voxelization
// ============================================================================

#include "geometry.hpp"

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>

namespace wt {

namespace {

constexpr float kBand = 8.f;     // SDF is exact within this many cells of the surface

// Push-constant mirror of sdf_shape.comp.
struct ShapePush {
    uint32_t  nx, ny, nz, shape;
    glm::vec4 center;            // xyz origin, w = band
    glm::vec4 size;
    glm::vec4 r0, r1, r2;
    glm::vec4 bounds;
    uint32_t  nPoly, pad0, pad1, pad2;
};
static_assert(sizeof(ShapePush) == 128);

// Push-constant mirror of sdf_mesh.comp / sdf_sign.comp.
struct MeshPush {
    int32_t  origin[4];
    uint32_t dims[4];
    uint32_t nTris, nbY, nbZ, binSize;
    float    band, pad[3];
};

// NACA 0012 with the closed trailing edge (last coefficient -0.1036), in the
// body frame: chord along x with the origin at the quarter chord (the usual
// pitch axis), thickness along y. Cosine spacing clusters points at the
// leading and trailing edges where curvature is highest.
std::vector<glm::vec2> naca0012(float chord, int perSide) {
    std::vector<glm::vec2> pts;
    auto yt = [](float x) {
        return 0.6f * (0.2969f*std::sqrt(x) - 0.1260f*x - 0.3516f*x*x
                       + 0.2843f*x*x*x - 0.1036f*x*x*x*x);
    };
    for (int i = perSide; i >= 0; --i) {            // upper surface, TE -> LE
        const float x = 0.5f * (1.f - std::cos(3.14159265f * float(i) / float(perSide)));
        pts.push_back({(x - 0.25f) * chord, yt(x) * chord});
    }
    for (int i = 1; i < perSide; ++i) {             // lower surface, LE -> TE
        const float x = 0.5f * (1.f - std::cos(3.14159265f * float(i) / float(perSide)));
        pts.push_back({(x - 0.25f) * chord, -yt(x) * chord});
    }
    return pts;
}

} // namespace

const char* shapeName(Shape s) {
    switch (s) {
    case Shape::Sphere:   return "Sphere";
    case Shape::Cube:     return "Cube";
    case Shape::Cylinder: return "Cylinder";
    case Shape::Wing:     return "NACA 0012 wing";
    default:              return "Mesh";
    }
}

glm::mat3 bodyRotation(const Body& b) {
    // Body -> world: roll about x, then yaw about y, then pitch about z.
    // Wind blows along +x, so a nose-up pitch turns the leading edge (-x)
    // towards +y, i.e. a negative rotation about +z.
    glm::mat4 m(1.f);
    m = glm::rotate(m, glm::radians(-b.pitch), {0, 0, 1});
    m = glm::rotate(m, glm::radians(b.yaw),    {0, 1, 0});
    m = glm::rotate(m, glm::radians(b.roll),   {1, 0, 0});
    return glm::transpose(glm::mat3(m));    // world -> body
}

bool loadMesh(const std::string& path, std::vector<Triangle>& tris, std::string& error) {
    Assimp::Importer imp;
    const aiScene* sc = imp.ReadFile(path, aiProcess_Triangulate | aiProcess_JoinIdenticalVertices
                                           | aiProcess_PreTransformVertices);
    if (!sc || !sc->mRootNode) { error = imp.GetErrorString(); return false; }
    tris.clear();
    glm::vec3 lo(std::numeric_limits<float>::max()), hi(-std::numeric_limits<float>::max());
    for (unsigned m = 0; m < sc->mNumMeshes; ++m) {
        const aiMesh* mesh = sc->mMeshes[m];
        for (unsigned f = 0; f < mesh->mNumFaces; ++f) {
            const aiFace& face = mesh->mFaces[f];
            if (face.mNumIndices != 3) continue;
            Triangle t;
            glm::vec3* v[3] = {&t.a, &t.b, &t.c};
            for (int k = 0; k < 3; ++k) {
                const aiVector3D& p = mesh->mVertices[face.mIndices[k]];
                *v[k] = {p.x, p.y, p.z};
                lo = glm::min(lo, *v[k]);
                hi = glm::max(hi, *v[k]);
            }
            tris.push_back(t);
        }
    }
    if (tris.empty()) { error = "no triangles in file"; return false; }
    const glm::vec3 c = 0.5f * (lo + hi);
    const float s = 1.f / std::max({hi.x - lo.x, hi.y - lo.y, hi.z - lo.z, 1e-9f});
    for (auto& t : tris) { t.a = (t.a - c) * s; t.b = (t.b - c) * s; t.c = (t.c - c) * s; }
    return true;
}

void Geometry::create(gpu::Context& ctx) {
    ctx_ = &ctx;
    sdfShape_ = ctx.loadKernel("sdf_shape");
    sdfSplat_ = ctx.loadKernel("sdf_mesh");
    sdfSign_  = ctx.loadKernel("sdf_sign");
    frontalK_ = ctx.loadKernel("frontal");
}

void Geometry::destroy() {
    if (!ctx_) return;
    for (auto* k : {&sdfShape_, &sdfSplat_, &sdfSign_, &frontalK_}) ctx_->destroyKernel(*k);
    for (auto* b : {&poly_, &tris_, &bins_, &scratch_}) ctx_->destroyBuffer(*b);
    ctx_ = nullptr;
}

void Geometry::setMesh(std::vector<Triangle> tris) { mesh_ = std::move(tris); }

void Geometry::apply(Solver& solver, const Body& body) {
    if (body.shape == Shape::Mesh)      prepareMesh(solver, body);
    else if (body.shape == Shape::Wing) prepareWing(body);
    ctx_->submitNow([&](VkCommandBuffer cmd) {
        if (body.shape == Shape::Mesh) writeMesh(cmd, solver);
        else                           writeAnalytic(cmd, solver, body);
        solver.applyGeometry(cmd);
    });
    frontal_ = measureFrontal(solver);
}

void Geometry::ensure(gpu::Buffer& buf, VkDeviceSize bytes) {
    if (buf.buf && buf.size >= bytes) return;
    ctx_->destroyBuffer(buf);
    buf = ctx_->createBuffer(bytes, 0, gpu::Mem::Device);
}

void Geometry::prepareWing(const Body& b) {
    const auto pts = naca0012(b.length, 96);
    glm::vec2 lo(1e30f), hi(-1e30f);
    for (const auto& q : pts) { lo = glm::min(lo, q); hi = glm::max(hi, q); }
    polyBounds_ = {lo, hi};
    polyCount_  = uint32_t(pts.size());
    ensure(poly_, pts.size() * sizeof(glm::vec2));
    ctx_->upload(poly_, pts.data(), pts.size() * sizeof(glm::vec2));
}

void Geometry::writeAnalytic(VkCommandBuffer cmd, Solver& s, const Body& b) {
    const auto& g = s.grid();
    const glm::mat3 r = bodyRotation(b);       // world -> body
    ShapePush p{};
    p.nx = g.nx; p.ny = g.ny; p.nz = g.nz;
    p.shape  = uint32_t(b.shape == Shape::Wing ? 3 : int(b.shape));
    p.center = {b.center, kBand};
    p.r0 = {r[0][0], r[1][0], r[2][0], 0};     // glm is column-major: rows of r
    p.r1 = {r[0][1], r[1][1], r[2][1], 0};
    p.r2 = {r[0][2], r[1][2], r[2][2], 0};
    // A zero span runs the body through the whole domain (a 2D section).
    const float through  = 4.f * float(std::max({g.nx, g.ny, g.nz}));
    const float halfSpan = b.span > 0.f ? 0.5f * b.span : through;
    switch (b.shape) {
    case Shape::Sphere:   p.size = {0.5f * b.length, 0, 0, 0}; break;
    case Shape::Cube:     p.size = glm::vec4(glm::vec3(0.5f * b.length), 0); break;
    case Shape::Cylinder: p.size = {0.5f * b.length, halfSpan, 0, 0}; break;
    default:              p.size = {halfSpan, 0, 0, 0}; p.bounds = polyBounds_; p.nPoly = polyCount_; break;
    }
    gpu::Bindings bnd = ctx_->emptyBindings();
    bnd[20] = s.sdf().buf;
    bnd[22] = poly_.buf;
    ctx_->bind(cmd, sdfShape_, bnd, &p, sizeof p);
    gpu::Context::dispatchThreads(cmd, s.cells());
}

void Geometry::prepareMesh(Solver& s, const Body& b) {
    const auto& g = s.grid();
    const glm::mat3 toWorld = glm::transpose(bodyRotation(b));
    std::vector<glm::vec4> verts;
    verts.reserve(mesh_.size() * 3);
    glm::vec3 lo(1e30f), hi(-1e30f);
    for (const auto& t : mesh_)
        for (const glm::vec3* v : {&t.a, &t.b, &t.c}) {
            const glm::vec3 w = b.center + toWorld * (*v * b.length);
            verts.push_back({w, 0.f});
            lo = glm::min(lo, w);
            hi = glm::max(hi, w);
        }
    nTris_ = uint32_t(mesh_.size());

    // Scratch box: the mesh bounds padded by the band, clipped to the grid.
    const glm::ivec3 gmax(int(g.nx) - 1, int(g.ny) - 1, int(g.nz) - 1);
    const glm::ivec3 o = glm::clamp(glm::ivec3(glm::floor(lo - kBand - 1.f)), glm::ivec3(0), gmax);
    const glm::ivec3 e = glm::clamp(glm::ivec3(glm::ceil(hi + kBand + 1.f)), glm::ivec3(0), gmax);
    boxOrigin_ = o;
    boxDims_   = glm::max(e - o + 1, glm::ivec3(1));

    // Bin triangles by the (y, z) columns their footprint covers, so each
    // parity ray only tests nearby triangles. CSR layout: starts, then ids.
    nbY_ = (uint32_t(boxDims_.y) + kBin - 1) / kBin;
    nbZ_ = (uint32_t(boxDims_.z) + kBin - 1) / kBin;
    const uint32_t nb = nbY_ * nbZ_;
    auto binRange = [&](const glm::vec4* v, glm::ivec2& b0, glm::ivec2& b1) {
        const glm::vec2 l(std::min({v[0].y, v[1].y, v[2].y}) - float(o.y),
                          std::min({v[0].z, v[1].z, v[2].z}) - float(o.z));
        const glm::vec2 h(std::max({v[0].y, v[1].y, v[2].y}) - float(o.y),
                          std::max({v[0].z, v[1].z, v[2].z}) - float(o.z));
        const glm::ivec2 top(int(nbY_) - 1, int(nbZ_) - 1);
        b0 = glm::clamp(glm::ivec2(glm::floor(l)) / int(kBin), glm::ivec2(0), top);
        b1 = glm::clamp(glm::ivec2(glm::ceil(h))  / int(kBin), glm::ivec2(0), top);
    };
    std::vector<uint32_t> bins(nb + 1, 0);
    for (uint32_t t = 0; t < nTris_; ++t) {
        glm::ivec2 b0, b1; binRange(&verts[3 * t], b0, b1);
        for (int z = b0.y; z <= b1.y; ++z)
            for (int y = b0.x; y <= b1.x; ++y) ++bins[size_t(z) * nbY_ + size_t(y) + 1];
    }
    for (uint32_t i = 0; i < nb; ++i) bins[i + 1] += bins[i];
    std::vector<uint32_t> cursor(bins.begin(), bins.end() - 1);
    bins.resize(size_t(nb) + 1 + bins[nb]);
    for (uint32_t t = 0; t < nTris_; ++t) {
        glm::ivec2 b0, b1; binRange(&verts[3 * t], b0, b1);
        for (int z = b0.y; z <= b1.y; ++z)
            for (int y = b0.x; y <= b1.x; ++y)
                bins[nb + 1 + cursor[size_t(z) * nbY_ + size_t(y)]++] = t;
    }

    ensure(tris_, verts.size() * sizeof(glm::vec4));
    ensure(bins_, bins.size() * 4);
    ensure(scratch_, VkDeviceSize(boxDims_.x) * boxDims_.y * boxDims_.z * 4);
    ctx_->upload(tris_, verts.data(), verts.size() * sizeof(glm::vec4));
    ctx_->upload(bins_, bins.data(), bins.size() * 4);
}

void Geometry::writeMesh(VkCommandBuffer cmd, Solver& s) {
    const auto& g = s.grid();
    MeshPush mp{};
    mp.origin[0] = boxOrigin_.x; mp.origin[1] = boxOrigin_.y; mp.origin[2] = boxOrigin_.z;
    mp.dims[0] = uint32_t(boxDims_.x); mp.dims[1] = uint32_t(boxDims_.y); mp.dims[2] = uint32_t(boxDims_.z);
    mp.nTris = nTris_; mp.nbY = nbY_; mp.nbZ = nbZ_; mp.binSize = kBin;
    mp.band  = kBand;

    // Unsigned distance (band-limited), then the sign by ray parity, then the
    // whole-grid pass that copies the box into the SDF.
    vkCmdFillBuffer(cmd, scratch_.buf, 0, VK_WHOLE_SIZE, std::bit_cast<uint32_t>(1e30f));
    gpu::Context::computeBarrier(cmd);
    gpu::Bindings bnd = ctx_->emptyBindings();
    bnd[19] = bins_.buf;
    bnd[22] = tris_.buf;
    bnd[23] = scratch_.buf;
    ctx_->bind(cmd, sdfSplat_, bnd, &mp, sizeof mp);
    gpu::Context::dispatchThreads(cmd, uint64_t(nTris_) * 256);     // one group per triangle
    gpu::Context::computeBarrier(cmd);
    ctx_->bind(cmd, sdfSign_, bnd, &mp, sizeof mp);
    gpu::Context::dispatchThreads(cmd, uint64_t(boxDims_.y) * boxDims_.z);
    gpu::Context::computeBarrier(cmd);

    ShapePush p{};
    p.nx = g.nx; p.ny = g.ny; p.nz = g.nz;
    p.shape  = 4;
    p.center = {0, 0, 0, kBand};
    p.size   = {glm::vec3(boxOrigin_), 0};
    p.bounds = {glm::vec3(boxDims_), 0};
    gpu::Bindings fb = ctx_->emptyBindings();
    fb[20] = s.sdf().buf;
    fb[23] = scratch_.buf;
    ctx_->bind(cmd, sdfShape_, fb, &p, sizeof p);
    gpu::Context::dispatchThreads(cmd, s.cells());
}

uint32_t Geometry::measureFrontal(Solver& s) {
    const auto& g = s.grid();
    const uint64_t cols = uint64_t(g.ny) * g.nz;
    gpu::Buffer bits = ctx_->createBuffer(((cols + 31) / 32) * 4, 0, gpu::Mem::Device);
    ctx_->submitNow([&](VkCommandBuffer cmd) {
        vkCmdFillBuffer(cmd, bits.buf, 0, VK_WHOLE_SIZE, 0);
        gpu::Context::computeBarrier(cmd);
        gpu::Bindings bnd = ctx_->emptyBindings();
        bnd[19] = s.flags().buf;
        bnd[22] = bits.buf;
        const uint32_t fp[4] = {g.nx, g.ny, g.nz, 0};
        ctx_->bind(cmd, frontalK_, bnd, fp, sizeof fp);
        gpu::Context::dispatchThreads(cmd, s.cells());
    });
    std::vector<uint32_t> host((cols + 31) / 32);
    ctx_->download(bits, host.data(), host.size() * 4);
    ctx_->destroyBuffer(bits);
    uint32_t n = 0;
    for (uint32_t w : host) n += uint32_t(std::popcount(w));
    return n;
}

} // namespace wt
