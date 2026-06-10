// ============================================================================
// mesh.cpp — Assimp import, SAT voxelization, analytic primitives
// ============================================================================

#include "mesh.h"
#include "gpu.h"   // logMsg

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <cmath>

namespace vwt::mesh {

namespace {

constexpr float kPi = 3.14159265358979f;

// Model placement inside the tunnel: centred in the cross-section, slightly
// upstream so the wake has room to develop.
constexpr float kCenterX   = 0.34f;
constexpr float kTargetFit = 0.34f;   // model height as fraction of min(gy,gz)

glm::mat3 rotationYZ(float pitchDeg, float yawDeg) {
    // Aeronautical convention: positive angle of attack pitches the nose
    // (the -X end, facing the incoming flow) upward — clockwise in the XY
    // plane — hence the negated pitch angle.
    const float p = -pitchDeg * kPi / 180.f;
    const float yw = yawDeg * kPi / 180.f;
    const glm::mat3 rz{  std::cos(p), std::sin(p), 0.f,
                        -std::sin(p), std::cos(p), 0.f,
                         0.f,         0.f,         1.f };
    const glm::mat3 ry{  std::cos(yw), 0.f, -std::sin(yw),
                         0.f,          1.f,  0.f,
                         std::sin(yw), 0.f,  std::cos(yw) };
    return ry * rz;   // pitch (AoA) first, then yaw
}

// ── Separating-axis triangle/AABB overlap ───────────────────────────────────
bool triBoxOverlap(const glm::vec3& center, const glm::vec3& half,
                   glm::vec3 a, glm::vec3 b, glm::vec3 c) {
    a -= center; b -= center; c -= center;

    const glm::vec3 e0 = b - a, e1 = c - b, e2 = a - c;

    auto axisTest = [&](const glm::vec3& ax) {
        const float pa = glm::dot(a, ax);
        const float pb = glm::dot(b, ax);
        const float pc = glm::dot(c, ax);
        const float r  = half.x * std::abs(ax.x) +
                         half.y * std::abs(ax.y) +
                         half.z * std::abs(ax.z);
        const float mn = std::min({pa, pb, pc});
        const float mx = std::max({pa, pb, pc});
        return !(mn > r || mx < -r);
    };

    // 9 cross-product axes
    const glm::vec3 axes[3] = { {1,0,0}, {0,1,0}, {0,0,1} };
    for (const auto& e : { e0, e1, e2 })
        for (const auto& ax : axes) {
            const glm::vec3 cr = glm::cross(e, ax);
            if (glm::dot(cr, cr) > 1e-12f && !axisTest(cr)) return false;
        }

    // 3 box axes
    auto minMax = [](float x, float y, float z, float& mn, float& mx) {
        mn = std::min({x, y, z}); mx = std::max({x, y, z});
    };
    float mn, mx;
    minMax(a.x, b.x, c.x, mn, mx); if (mn > half.x || mx < -half.x) return false;
    minMax(a.y, b.y, c.y, mn, mx); if (mn > half.y || mx < -half.y) return false;
    minMax(a.z, b.z, c.z, mn, mx); if (mn > half.z || mx < -half.z) return false;

    // Triangle plane
    const glm::vec3 n = glm::cross(e0, e1);
    const float d = glm::dot(n, a);
    const float r = half.x * std::abs(n.x) + half.y * std::abs(n.y) + half.z * std::abs(n.z);
    return std::abs(d) <= r;
}

// Fill interior by parity counting along X: a cell is inside when an odd
// number of surface cells precede it... surface-only voxelization leaves
// hollow models, which is fine for the LBM (flow never reaches the interior
// because the shell bounces everything back), but interior fill makes the
// frontal-area count and slice rendering cleaner. We use a simple scanline
// flood: between the first and last surface voxel of each (y,z) column,
// cells bracketed by surface on both sides become solid.
void fillColumns(std::vector<uint32_t>& occ, uint32_t gx, uint32_t gy, uint32_t gz) {
    for (uint32_t z = 0; z < gz; ++z)
        for (uint32_t y = 0; y < gy; ++y) {
            const size_t row = (size_t(z) * gy + y) * gx;
            int first = -1, last = -1;
            for (uint32_t x = 0; x < gx; ++x)
                if (occ[row + x]) { if (first < 0) first = int(x); last = int(x); }
            if (first >= 0 && last > first)
                for (int x = first; x <= last; ++x) occ[row + x] = 1u;
        }
}

void finalize(VoxelModel& m, uint32_t gx, uint32_t gy, uint32_t gz) {
    uint32_t solid = 0, frontal = 0;
    int minX = int(gx), maxX = -1;
    for (uint32_t z = 0; z < gz; ++z)
        for (uint32_t y = 0; y < gy; ++y) {
            bool any = false;
            for (uint32_t x = 0; x < gx; ++x) {
                if (m.occupancy[(size_t(z) * gy + y) * gx + x]) {
                    ++solid; any = true;
                    minX = std::min(minX, int(x));
                    maxX = std::max(maxX, int(x));
                }
            }
            if (any) ++frontal;
        }
    m.frontalCells = frontal;
    m.spanCellsX   = (maxX >= minX) ? uint32_t(maxX - minX + 1) : 0;
    m.fillPct      = 100.f * float(solid) / float(size_t(gx) * gy * gz);
}

} // namespace

// ════════════════════════════════════════════════════════════════════════════
// Assimp import
// ════════════════════════════════════════════════════════════════════════════

bool loadTriangles(const std::filesystem::path& path,
                   std::vector<Tri>& out, std::string& error) {
    Assimp::Importer imp;
    const aiScene* scene = imp.ReadFile(path.string(),
        aiProcess_Triangulate | aiProcess_JoinIdenticalVertices |
        aiProcess_PreTransformVertices);
    if (!scene || !scene->mNumMeshes) {
        error = imp.GetErrorString();
        if (error.empty()) error = "No meshes in file";
        return false;
    }

    out.clear();
    for (unsigned m = 0; m < scene->mNumMeshes; ++m) {
        const aiMesh* mesh = scene->mMeshes[m];
        for (unsigned f = 0; f < mesh->mNumFaces; ++f) {
            const aiFace& face = mesh->mFaces[f];
            if (face.mNumIndices != 3) continue;
            auto v = [&](unsigned i) {
                const aiVector3D& p = mesh->mVertices[face.mIndices[i]];
                return glm::vec3(p.x, p.y, p.z);
            };
            out.push_back({ v(0), v(1), v(2) });
        }
    }
    if (out.empty()) { error = "File contained no triangles"; return false; }
    logMsg("Mesh: " + path.filename().string() + " — " +
           std::to_string(out.size()) + " triangles");
    return true;
}

// ════════════════════════════════════════════════════════════════════════════
// Triangle voxelization
// ════════════════════════════════════════════════════════════════════════════

VoxelModel voxelizeTriangles(const std::vector<Tri>& tris,
                             uint32_t gx, uint32_t gy, uint32_t gz,
                             float pitchDeg, float yawDeg,
                             const std::string& name) {
    VoxelModel m;
    m.occupancy.assign(size_t(gx) * gy * gz, 0u);
    m.name     = name;
    m.triCount = uint32_t(tris.size());
    if (tris.empty()) return m;

    // Rotate about the model centre, then fit to the tunnel.
    glm::vec3 lo(1e30f), hi(-1e30f);
    for (const Tri& t : tris)
        for (const glm::vec3& v : { t.a, t.b, t.c }) {
            lo = glm::min(lo, v); hi = glm::max(hi, v);
        }
    const glm::vec3 mid = 0.5f * (lo + hi);
    const glm::mat3 rot = rotationYZ(pitchDeg, yawDeg);

    glm::vec3 rlo(1e30f), rhi(-1e30f);
    std::vector<Tri> rt(tris.size());
    for (size_t i = 0; i < tris.size(); ++i) {
        rt[i].a = rot * (tris[i].a - mid);
        rt[i].b = rot * (tris[i].b - mid);
        rt[i].c = rot * (tris[i].c - mid);
        for (const glm::vec3& v : { rt[i].a, rt[i].b, rt[i].c }) {
            rlo = glm::min(rlo, v); rhi = glm::max(rhi, v);
        }
    }

    const glm::vec3 ext = glm::max(rhi - rlo, glm::vec3(1e-6f));
    const float fit   = kTargetFit * float(std::min(gy, gz));
    const float scale = fit / std::max({ ext.x, ext.y, ext.z });
    const glm::vec3 center{ kCenterX * float(gx), 0.5f * float(gy), 0.5f * float(gz) };

    for (Tri& t : rt) {
        t.a = t.a * scale + center;
        t.b = t.b * scale + center;
        t.c = t.c * scale + center;
    }

    // Conservative SAT rasterization over each triangle's cell bbox
    const glm::vec3 half{ 0.5f, 0.5f, 0.5f };
    for (const Tri& t : rt) {
        glm::vec3 tlo = glm::min(t.a, glm::min(t.b, t.c)) - 0.5f;
        glm::vec3 thi = glm::max(t.a, glm::max(t.b, t.c)) + 0.5f;
        const int x0 = std::max(0, int(tlo.x)), x1 = std::min(int(gx) - 1, int(thi.x));
        const int y0 = std::max(0, int(tlo.y)), y1 = std::min(int(gy) - 1, int(thi.y));
        const int z0 = std::max(0, int(tlo.z)), z1 = std::min(int(gz) - 1, int(thi.z));
        for (int z = z0; z <= z1; ++z)
            for (int y = y0; y <= y1; ++y)
                for (int x = x0; x <= x1; ++x) {
                    const size_t idx = (size_t(z) * gy + y) * gx + x;
                    if (m.occupancy[idx]) continue;
                    const glm::vec3 cc{ float(x) + 0.5f, float(y) + 0.5f, float(z) + 0.5f };
                    if (triBoxOverlap(cc, half, t.a, t.b, t.c))
                        m.occupancy[idx] = 1u;
                }
    }

    fillColumns(m.occupancy, gx, gy, gz);
    finalize(m, gx, gy, gz);
    return m;
}

// ════════════════════════════════════════════════════════════════════════════
// Analytic primitives
// ════════════════════════════════════════════════════════════════════════════

VoxelModel makePrimitive(Shape shape,
                         uint32_t gx, uint32_t gy, uint32_t gz,
                         float pitchDeg, float yawDeg) {
    VoxelModel m;
    m.occupancy.assign(size_t(gx) * gy * gz, 0u);
    m.name = shapeName(shape);

    const float s = float(std::min(gy, gz));
    const glm::vec3 center{ kCenterX * float(gx), 0.5f * float(gy), 0.5f * float(gz) };
    // Inverse rotation: test the cell centre in the primitive's local frame.
    const glm::mat3 invRot = glm::transpose(rotationYZ(pitchDeg, yawDeg));

    // NACA 0012 half-thickness (closed trailing edge), chord-normalized.
    auto naca = [](float xc) {
        xc = std::clamp(xc, 0.f, 1.f);
        return 5.f * 0.12f * (0.2969f * std::sqrt(xc) - 0.1260f * xc
                            - 0.3516f * xc * xc + 0.2843f * xc * xc * xc
                            - 0.1036f * xc * xc * xc * xc);
    };

    const float rSphere   = 0.17f * s;
    const float hCube     = 0.14f * s;
    const float rCyl      = 0.12f * s;
    const float spanCyl   = 0.36f * float(gz);
    const float chord     = 0.30f * float(gx);
    const float spanWing  = 0.34f * float(gz);

    auto inside = [&](glm::vec3 p) -> bool {
        switch (shape) {
        case Shape::Sphere:
            return glm::dot(p, p) <= rSphere * rSphere;
        case Shape::Cube:
            return std::abs(p.x) <= hCube && std::abs(p.y) <= hCube &&
                   std::abs(p.z) <= hCube;
        case Shape::Cylinder:   // axis spanwise (Z): classic vortex-street body
            return p.x * p.x + p.y * p.y <= rCyl * rCyl &&
                   std::abs(p.z) <= spanCyl;
        case Shape::Wing: {
            const float xc = (p.x + 0.5f * chord) / chord;
            if (xc < 0.f || xc > 1.f) return false;
            if (std::abs(p.z) > spanWing) return false;
            return std::abs(p.y) <= naca(xc) * chord;
        }
        }
        return false;
    };

    for (uint32_t z = 0; z < gz; ++z)
        for (uint32_t y = 0; y < gy; ++y)
            for (uint32_t x = 0; x < gx; ++x) {
                const glm::vec3 cc{ float(x) + 0.5f, float(y) + 0.5f, float(z) + 0.5f };
                if (inside(invRot * (cc - center)))
                    m.occupancy[(size_t(z) * gy + y) * gx + x] = 1u;
            }

    finalize(m, gx, gy, gz);
    return m;
}

} // namespace vwt::mesh
