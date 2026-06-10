// ============================================================================
// mesh_loader.cpp — 3D Mesh Loading and Surface Voxelization
// ============================================================================

#include "mesh_loader.h"

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include <glm/glm.hpp>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <cmath>

namespace vwt {

// ════════════════════════════════════════════════════════════════════════
// Mesh Loading via Assimp
// ════════════════════════════════════════════════════════════════════════

MeshData MeshLoader::loadMesh(const std::string& filepath) {
    Assimp::Importer importer;

    const aiScene* scene = importer.ReadFile(filepath,
        aiProcess_Triangulate       |
        aiProcess_JoinIdenticalVertices |
        aiProcess_GenNormals        |
        aiProcess_PreTransformVertices |
        aiProcess_OptimizeMeshes
    );

    if (!scene || !scene->mRootNode || (scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE)) {
        throw std::runtime_error("Assimp failed to load mesh: " +
            std::string(importer.GetErrorString()));
    }

    MeshData result;
    result.aabbMin = glm::vec3( std::numeric_limits<float>::max());
    result.aabbMax = glm::vec3(-std::numeric_limits<float>::max());

    // Iterate all meshes in the scene
    for (unsigned int m = 0; m < scene->mNumMeshes; ++m) {
        const aiMesh* mesh = scene->mMeshes[m];

        for (unsigned int f = 0; f < mesh->mNumFaces; ++f) {
            const aiFace& face = mesh->mFaces[f];
            if (face.mNumIndices != 3) continue; // skip non-triangles

            Triangle tri;
            for (int i = 0; i < 3; ++i) {
                const aiVector3D& v = mesh->mVertices[face.mIndices[i]];
                glm::vec3 vertex(v.x, v.y, v.z);

                if (i == 0) tri.v0 = vertex;
                else if (i == 1) tri.v1 = vertex;
                else tri.v2 = vertex;

                result.aabbMin = glm::min(result.aabbMin, vertex);
                result.aabbMax = glm::max(result.aabbMax, vertex);
            }
            result.triangles.push_back(tri);
        }
    }

    std::cout << "[MeshLoader] Loaded " << result.triangles.size()
              << " triangles from: " << filepath << "\n";
    std::cout << "[MeshLoader] AABB min: ("
              << result.aabbMin.x << ", " << result.aabbMin.y << ", " << result.aabbMin.z << ")\n";
    std::cout << "[MeshLoader] AABB max: ("
              << result.aabbMax.x << ", " << result.aabbMax.y << ", " << result.aabbMax.z << ")\n";

    return result;
}

// ════════════════════════════════════════════════════════════════════════
// Surface Voxelization
// ════════════════════════════════════════════════════════════════════════
//
// Algorithm: For each voxel cell in the grid, check if any triangle in the
// mesh intersects the voxel's axis-aligned bounding box (AABB).
// Uses the Separating Axis Theorem (SAT) for robust triangle-AABB overlap.
//
// The mesh is scaled and centered to fit within the grid, leaving a
// configurable margin for the wind tunnel walls.
// ════════════════════════════════════════════════════════════════════════

std::vector<uint32_t> MeshLoader::voxelizeSurface(
    const MeshData& mesh,
    uint32_t gridX, uint32_t gridY, uint32_t gridZ,
    float marginFraction)
{
    const size_t totalCells = static_cast<size_t>(gridX) * gridY * gridZ;
    std::vector<uint32_t> obstacleMap(totalCells, 0);

    if (mesh.triangles.empty()) return obstacleMap;

    // Compute mesh extents and the scaling transform
    glm::vec3 meshSize = mesh.aabbMax - mesh.aabbMin;
    glm::vec3 meshCenter = (mesh.aabbMax + mesh.aabbMin) * 0.5f;

    // Effective grid area after margin
    float margin = marginFraction;
    glm::vec3 gridDim(
        static_cast<float>(gridX) * (1.0f - 2.0f * margin),
        static_cast<float>(gridY) * (1.0f - 2.0f * margin),
        static_cast<float>(gridZ) * (1.0f - 2.0f * margin)
    );

    // Uniform scale to fit longest axis
    float maxMeshDim = std::max({meshSize.x, meshSize.y, meshSize.z});
    if (maxMeshDim < 1e-6f) {
        std::cerr << "[MeshLoader] Warning: Mesh has zero extent.\n";
        return obstacleMap;
    }
    float minGridDim = std::min({gridDim.x, gridDim.y, gridDim.z});
    float scale = minGridDim / maxMeshDim;

    // Grid center
    glm::vec3 gridCenter(
        static_cast<float>(gridX) * 0.5f,
        static_cast<float>(gridY) * 0.5f,
        static_cast<float>(gridZ) * 0.5f
    );

    // Transform a mesh vertex to grid coordinates
    auto transformVertex = [&](const glm::vec3& v) -> glm::vec3 {
        return (v - meshCenter) * scale + gridCenter;
    };

    // Voxel half-size (each voxel is 1x1x1 in grid units)
    glm::vec3 halfSize(0.5f);

    std::cout << "[Voxelizer] Grid: " << gridX << "x" << gridY << "x" << gridZ
              << ", Scale: " << scale << "\n";

    // Iterate over all triangles
    uint32_t solidCount = 0;
    for (const auto& tri : mesh.triangles) {
        // Transform triangle to grid space
        Triangle gridTri;
        gridTri.v0 = transformVertex(tri.v0);
        gridTri.v1 = transformVertex(tri.v1);
        gridTri.v2 = transformVertex(tri.v2);

        // Compute the AABB of the transformed triangle
        glm::vec3 triMin = glm::min(glm::min(gridTri.v0, gridTri.v1), gridTri.v2);
        glm::vec3 triMax = glm::max(glm::max(gridTri.v0, gridTri.v1), gridTri.v2);

        // Clamp to grid bounds
        int minX = std::max(0, static_cast<int>(std::floor(triMin.x - 0.5f)));
        int minY = std::max(0, static_cast<int>(std::floor(triMin.y - 0.5f)));
        int minZ = std::max(0, static_cast<int>(std::floor(triMin.z - 0.5f)));
        int maxX = std::min(static_cast<int>(gridX) - 1, static_cast<int>(std::ceil(triMax.x + 0.5f)));
        int maxY = std::min(static_cast<int>(gridY) - 1, static_cast<int>(std::ceil(triMax.y + 0.5f)));
        int maxZ = std::min(static_cast<int>(gridZ) - 1, static_cast<int>(std::ceil(triMax.z + 0.5f)));

        // Test each voxel in the triangle's AABB
        for (int z = minZ; z <= maxZ; ++z) {
            for (int y = minY; y <= maxY; ++y) {
                for (int x = minX; x <= maxX; ++x) {
                    size_t idx = static_cast<size_t>(z) * gridX * gridY
                               + static_cast<size_t>(y) * gridX
                               + static_cast<size_t>(x);

                    if (obstacleMap[idx] != 0) continue; // Already marked

                    glm::vec3 center(
                        static_cast<float>(x) + 0.5f,
                        static_cast<float>(y) + 0.5f,
                        static_cast<float>(z) + 0.5f
                    );

                    if (triangleAABBOverlap(center, halfSize, gridTri)) {
                        obstacleMap[idx] = 1;
                        ++solidCount;
                    }
                }
            }
        }
    }

    std::cout << "[Voxelizer] Marked " << solidCount << " / " << totalCells
              << " cells as solid ("
              << (100.0f * solidCount / totalCells) << "%)\n";

    return obstacleMap;
}

// ════════════════════════════════════════════════════════════════════════
// Frontal area projection (looking along +X — the flow direction)
// ════════════════════════════════════════════════════════════════════════
//
// Reference area for the drag/lift coefficients. We project the obstacle
// onto the YZ plane and count cells that are solid in at least one X-slice;
// that's the silhouette the flow "sees" head-on. Returned in lattice units²
// — physical dx cancels out of C_D / C_L so no further scaling is needed.
// ════════════════════════════════════════════════════════════════════════

uint32_t MeshLoader::computeFrontalArea(const std::vector<uint32_t>& obstacle,
                                        uint32_t gridX, uint32_t gridY, uint32_t gridZ)
{
    if (obstacle.empty()) return 0;
    uint32_t cells = 0;
    for (uint32_t z = 0; z < gridZ; ++z) {
        for (uint32_t y = 0; y < gridY; ++y) {
            const size_t row = size_t(z) * gridX * gridY + size_t(y) * gridX;
            for (uint32_t x = 0; x < gridX; ++x) {
                if (obstacle[row + x] != 0u) { ++cells; break; }
            }
        }
    }
    return cells;
}

// ════════════════════════════════════════════════════════════════════════
// Triangle-AABB Overlap Test (Tomas Akenine-Möller, 2001)
// ════════════════════════════════════════════════════════════════════════
// Uses the Separating Axis Theorem with 13 axes:
//   - 3 AABB face normals
//   - 1 triangle normal
//   - 9 cross products of AABB edges × triangle edges
// ════════════════════════════════════════════════════════════════════════

namespace {

inline void project(const glm::vec3& axis, const glm::vec3& v0,
                    const glm::vec3& v1, const glm::vec3& v2,
                    float& outMin, float& outMax)
{
    float p0 = glm::dot(axis, v0);
    float p1 = glm::dot(axis, v1);
    float p2 = glm::dot(axis, v2);
    outMin = std::min({p0, p1, p2});
    outMax = std::max({p0, p1, p2});
}

inline bool overlapOnAxis(const glm::vec3& axis, const glm::vec3& v0,
                          const glm::vec3& v1, const glm::vec3& v2,
                          const glm::vec3& halfSize)
{
    if (glm::dot(axis, axis) < 1e-10f) return true; // Degenerate axis

    float triMin, triMax;
    project(axis, v0, v1, v2, triMin, triMax);

    // AABB projection onto axis: radius = sum of half-extents projected
    float r = halfSize.x * std::abs(axis.x)
            + halfSize.y * std::abs(axis.y)
            + halfSize.z * std::abs(axis.z);

    // Check overlap: [-r, r] vs [triMin, triMax]
    return !(triMin > r || triMax < -r);
}

} // anonymous namespace

bool MeshLoader::triangleAABBOverlap(
    const glm::vec3& boxCenter,
    const glm::vec3& boxHalfSize,
    const Triangle& tri) const
{
    // Translate triangle so that AABB center is at origin
    glm::vec3 v0 = tri.v0 - boxCenter;
    glm::vec3 v1 = tri.v1 - boxCenter;
    glm::vec3 v2 = tri.v2 - boxCenter;

    // Triangle edges
    glm::vec3 f0 = v1 - v0;
    glm::vec3 f1 = v2 - v1;
    glm::vec3 f2 = v0 - v2;

    // AABB normals (standard basis)
    glm::vec3 u0(1, 0, 0), u1(0, 1, 0), u2(0, 0, 1);

    // Test 9 cross-product axes (AABB edges × triangle edges)
    glm::vec3 axes[9] = {
        glm::cross(u0, f0), glm::cross(u0, f1), glm::cross(u0, f2),
        glm::cross(u1, f0), glm::cross(u1, f1), glm::cross(u1, f2),
        glm::cross(u2, f0), glm::cross(u2, f1), glm::cross(u2, f2),
    };

    for (const auto& axis : axes) {
        if (!overlapOnAxis(axis, v0, v1, v2, boxHalfSize)) return false;
    }

    // Test 3 AABB face normals
    if (!overlapOnAxis(u0, v0, v1, v2, boxHalfSize)) return false;
    if (!overlapOnAxis(u1, v0, v1, v2, boxHalfSize)) return false;
    if (!overlapOnAxis(u2, v0, v1, v2, boxHalfSize)) return false;

    // Test triangle normal
    glm::vec3 triNormal = glm::cross(f0, f1);
    if (!overlapOnAxis(triNormal, v0, v1, v2, boxHalfSize)) return false;

    return true;
}

} // namespace vwt
