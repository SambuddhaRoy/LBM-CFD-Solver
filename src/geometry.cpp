// ============================================================================
// geometry.cpp — analytic bodies, mesh import, GPU signed-distance voxelization
// ============================================================================

#include "geometry.hpp"

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <limits>
#include <unordered_map>

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

MeshReport checkTopology(const std::vector<Triangle>& tris) {
    // In a closed surface every edge is shared by exactly two triangles.
    // Vertices are welded by position at 1e-6 of the model's size.
    struct KeyHash {
        size_t operator()(const std::array<int64_t, 3>& k) const {
            return size_t(k[0] * 73856093) ^ size_t(k[1] * 19349663) ^ size_t(k[2] * 83492791);
        }
    };
    std::unordered_map<std::array<int64_t, 3>, uint32_t, KeyHash> ids;
    auto id = [&](const glm::vec3& p) {
        const std::array<int64_t, 3> k{std::llround(p.x * 1e6), std::llround(p.y * 1e6), std::llround(p.z * 1e6)};
        return ids.emplace(k, uint32_t(ids.size())).first->second;
    };
    std::unordered_map<uint64_t, uint32_t> edges;
    edges.reserve(tris.size() * 2);
    for (const auto& t : tris) {
        const uint32_t v[3] = {id(t.a), id(t.b), id(t.c)};
        for (int e = 0; e < 3; ++e) {
            const auto [lo, hi] = std::minmax(v[e], v[(e + 1) % 3]);
            ++edges[uint64_t(lo) << 32 | hi];
        }
    }
    MeshReport r;
    r.edges = edges.size();
    for (const auto& [e, n] : edges) { r.openEdges += n == 1; r.nonManifoldEdges += n > 2; }
    return r;
}

bool loadMesh(const std::string& path, std::vector<Triangle>& tris, std::string& error, MeshReport* report) {
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

    // Orient the model in the tunnel (wind along +x, y up). Up follows the
    // file format's convention: Y for glTF, FBX, OBJ and Collada, Z for the
    // CAD formats (STL, PLY, 3MF). The longer of the two horizontal extents
    // is the length, aligned with the wind, with the front (+axis; glTF
    // defines +Z as forward) facing upstream (-x).
    std::string ext = std::filesystem::path(path).extension().string();
    for (auto& ch : ext) ch = char(std::tolower(static_cast<unsigned char>(ch)));
    const bool yUp = ext == ".gltf" || ext == ".glb" || ext == ".fbx" || ext == ".obj" || ext == ".dae";
    const int up = yUp ? 1 : 2;
    const glm::vec3 ext3 = hi - lo;
    const int a0 = 0, a1 = up == 1 ? 2 : 1;                     // the two horizontal axes
    const int fwd = ext3[a1] > ext3[a0] ? a1 : a0;
    glm::vec3 F(0), U(0);
    F[fwd] = 1; U[up] = 1;
    const glm::vec3 S = glm::cross(F, U);
    auto toTunnel = [&](const glm::vec3& p) { return glm::vec3(-glm::dot(p, F), glm::dot(p, U), -glm::dot(p, S)); };

    const glm::vec3 c = 0.5f * (lo + hi);
    const float s = 1.f / std::max({ext3.x, ext3.y, ext3.z, 1e-9f});
    for (auto& t : tris) {
        t.a = toTunnel((t.a - c) * s); t.b = toTunnel((t.b - c) * s); t.c = toTunnel((t.c - c) * s);
    }
    if (report) {
        *report = checkTopology(tris);
        const char* names = "xyz";
        report->alignment = std::string("file ") + names[up] + " up, front +" + names[fwd] + " facing the wind";
        report->extent = std::max({ext3.x, ext3.y, ext3.z});
        report->metres = ext == ".gltf" || ext == ".glb";
    }
    return true;
}

int printMeshInfo(const std::string& path) {
    Assimp::Importer imp;
    const aiScene* sc = imp.ReadFile(path, aiProcess_Triangulate | aiProcess_JoinIdenticalVertices
                                           | aiProcess_PreTransformVertices);
    if (!sc || !sc->mRootNode) { std::printf("load failed: %s\n", imp.GetErrorString()); return 1; }
    std::printf("%u meshes\n", sc->mNumMeshes);
    for (unsigned m = 0; m < sc->mNumMeshes; ++m) {
        const aiMesh* mesh = sc->mMeshes[m];
        glm::vec3 lo(1e30f), hi(-1e30f);
        for (unsigned v = 0; v < mesh->mNumVertices; ++v) {
            const glm::vec3 p(mesh->mVertices[v].x, mesh->mVertices[v].y, mesh->mVertices[v].z);
            lo = glm::min(lo, p); hi = glm::max(hi, p);
        }
        std::printf("  %3u %-32.32s %8u tris  x[%8.3f %8.3f] y[%8.3f %8.3f] z[%8.3f %8.3f]\n", m,
                    mesh->mName.C_Str(), mesh->mNumFaces, lo.x, hi.x, lo.y, hi.y, lo.z, hi.z);
    }
    std::vector<Triangle> tris;
    std::string err;
    MeshReport r;
    loadMesh(path, tris, err, &r);
    std::printf("%zu triangles, %zu edges: %zu open (one triangle), %zu shared by more than two\n",
                tris.size(), r.edges, r.openEdges, r.nonManifoldEdges);
    std::printf("%s\n", r.watertight() ? "watertight: exact inside/outside by ray parity"
                                       : "not watertight: voxelized by shrink-wrap (flood fill from outside)");
    std::printf("orientation: %s\n", r.alignment.c_str());
    return 0;
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

glm::vec3 Geometry::extent(const Body& b, bool& through) const {
    const glm::mat3 toWorld = glm::transpose(bodyRotation(b));
    glm::vec3 lo(1e30f), hi(-1e30f);
    auto add = [&](const glm::vec3& p) { const glm::vec3 w = toWorld * p; lo = glm::min(lo, w); hi = glm::max(hi, w); };
    through = (b.shape == Shape::Cylinder || b.shape == Shape::Wing) && b.span <= 0;
    if (b.shape == Shape::Mesh) {
        for (const auto& t : mesh_) { add(t.a); add(t.b); add(t.c); }
    } else if (b.shape == Shape::Sphere) {
        return glm::vec3(1.f);
    } else {
        // Oriented boxes: cube, cylinder (diameter 1 x span), wing (chord 1,
        // 12% thick, x span). A spanning body counts one length of span.
        glm::vec3 h(0.5f);
        const float span = through ? 1.f : b.span / std::max(b.length, 1.f);
        if (b.shape == Shape::Cylinder) h = {0.5f, 0.5f, 0.5f * span};
        if (b.shape == Shape::Wing)     h = {0.5f, 0.06f, 0.5f * span};
        for (int k = 0; k < 8; ++k)
            add({(k & 1) ? h.x : -h.x, (k & 2) ? h.y : -h.y, (k & 4) ? h.z : -h.z});
    }
    return hi - lo;
}

void Geometry::setMesh(std::vector<Triangle> tris, bool watertight) {
    mesh_ = std::move(tris);
    watertight_ = watertight;
}

void Geometry::apply(Solver& solver, const Body& body) {
    if (body.shape == Shape::Mesh)      prepareMesh(solver, body);
    else if (body.shape == Shape::Wing) prepareWing(body);
    if (body.shape == Shape::Mesh && !watertight_) {
        // Shrink-wrap needs a CPU flood fill between the distance pass and
        // the copy into the solver, so it runs as two submissions.
        ctx_->submitNow([&](VkCommandBuffer cmd) { writeMeshDistance(cmd, false); });
        shrinkWrap();
        ctx_->submitNow([&](VkCommandBuffer cmd) { writeMeshFinal(cmd, solver); solver.applyGeometry(cmd); });
    } else {
        ctx_->submitNow([&](VkCommandBuffer cmd) {
            if (body.shape == Shape::Mesh) { writeMeshDistance(cmd, true); writeMeshFinal(cmd, solver); }
            else writeAnalytic(cmd, solver, body);
            solver.applyGeometry(cmd);
        });
    }
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

// Band-limited unsigned distance in the scratch box, plus the inside/outside
// sign by ray parity when the mesh is closed (exact for closed surfaces).
void Geometry::writeMeshDistance(VkCommandBuffer cmd, bool parity) {
    MeshPush mp{};
    mp.origin[0] = boxOrigin_.x; mp.origin[1] = boxOrigin_.y; mp.origin[2] = boxOrigin_.z;
    mp.dims[0] = uint32_t(boxDims_.x); mp.dims[1] = uint32_t(boxDims_.y); mp.dims[2] = uint32_t(boxDims_.z);
    mp.nTris = nTris_; mp.nbY = nbY_; mp.nbZ = nbZ_; mp.binSize = kBin;
    mp.band  = kBand;
    vkCmdFillBuffer(cmd, scratch_.buf, 0, VK_WHOLE_SIZE, std::bit_cast<uint32_t>(1e30f));
    gpu::Context::computeBarrier(cmd);
    gpu::Bindings bnd = ctx_->emptyBindings();
    bnd[19] = bins_.buf;
    bnd[22] = tris_.buf;
    bnd[23] = scratch_.buf;
    ctx_->bind(cmd, sdfSplat_, bnd, &mp, sizeof mp);
    gpu::Context::dispatchThreads(cmd, uint64_t(nTris_) * 256);     // one group per triangle
    gpu::Context::computeBarrier(cmd);
    if (parity) {
        ctx_->bind(cmd, sdfSign_, bnd, &mp, sizeof mp);
        gpu::Context::dispatchThreads(cmd, uint64_t(boxDims_.y) * boxDims_.z);
        gpu::Context::computeBarrier(cmd);
    }
}

// Copies the scratch box into the solver's SDF (outside the box: distance to it).
void Geometry::writeMeshFinal(VkCommandBuffer cmd, Solver& s) {
    const auto& g = s.grid();
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

// Shrink-wrap for meshes that are not closed (typical of visualisation
// models: separate parts, gaps between panels, single-sheet surfaces), where
// ray parity flips inside/outside at every hole and leaves solid streaks.
// Every cell within kWrap of a triangle becomes a wall; a flood fill from the
// box boundary marks what the outside air can reach; the rest is interior.
// Gaps narrower than ~2 kWrap are sealed and single sheets become solid
// layers, at the price of offsetting surfaces outward by kWrap.
// kWrap = sqrt(3)/2 makes the wall layer around any surface at least one
// cell thick in every orientation, so neither the flood nor the flow leaks.
void Geometry::shrinkWrap() {
    constexpr float kWrap = 0.866f;
    const size_t nx = size_t(boxDims_.x), ny = size_t(boxDims_.y), nz = size_t(boxDims_.z);
    const size_t n = nx * ny * nz;
    std::vector<float> d(n);
    ctx_->download(scratch_, d.data(), n * 4);

    std::vector<uint8_t> state(n, 0);               // 0 unknown, 1 wall, 2 outside
    for (size_t c = 0; c < n; ++c) if (d[c] < kWrap) state[c] = 1;
    std::vector<size_t> queue;
    queue.reserve(n / 4);
    auto seed = [&](size_t c) { if (state[c] == 0) { state[c] = 2; queue.push_back(c); } };
    for (size_t z = 0; z < nz; ++z)
        for (size_t y = 0; y < ny; ++y)
            for (size_t x = 0; x < nx; ++x)
                if (x == 0 || y == 0 || z == 0 || x + 1 == nx || y + 1 == ny || z + 1 == nz)
                    seed((z * ny + y) * nx + x);
    for (size_t head = 0; head < queue.size(); ++head) {
        const size_t c = queue[head];
        const size_t x = c % nx, y = (c / nx) % ny, z = c / (nx * ny);
        if (x > 0)      seed(c - 1);
        if (x + 1 < nx) seed(c + 1);
        if (y > 0)      seed(c - nx);
        if (y + 1 < ny) seed(c + nx);
        if (z > 0)      seed(c - nx * ny);
        if (z + 1 < nz) seed(c + nx * ny);
    }
    // Walls and outside measure distance from the offset surface, so the
    // interpolated walls see a consistent surface at d = kWrap; the interior
    // is just negative.
    for (size_t c = 0; c < n; ++c) d[c] = state[c] == 0 ? -(d[c] + kWrap) : d[c] - kWrap;
    ctx_->upload(scratch_, d.data(), n * 4);
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
