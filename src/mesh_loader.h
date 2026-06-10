#pragma once
// ============================================================================
// mesh_loader.h — 3D Mesh Loading and Surface Voxelization
// ============================================================================

#include "vk_types.h"
#include <glm/glm.hpp>
#include <string>
#include <vector>

namespace vwt {

struct Triangle {
    glm::vec3 v0, v1, v2;
};

struct MeshData {
    std::vector<Triangle>  triangles;
    glm::vec3              aabbMin;
    glm::vec3              aabbMax;
};

class MeshLoader {
public:
    /// Load a mesh from disk using Assimp. Supports .obj, .stl, .fbx, .glb
    MeshData loadMesh(const std::string& filepath);

    /// Perform surface voxelization of the mesh onto the LBM grid.
    /// Returns a flat array of uint32_t flags (0 = fluid, 1 = solid).
    /// The mesh is scaled and centered to fit within the grid with a margin.
    std::vector<uint32_t> voxelizeSurface(
        const MeshData&  mesh,
        uint32_t         gridX,
        uint32_t         gridY,
        uint32_t         gridZ,
        float            marginFraction = 0.1f   // 10% margin around the mesh
    );

    /// Count voxels in the projected YZ frontal plane (looking along +X — the
    /// flow direction). Used as the reference area for C_D / C_L
    /// computation; returns frontal area expressed in lattice units squared.
    static uint32_t computeFrontalArea(
        const std::vector<uint32_t>& obstacle,
        uint32_t gridX, uint32_t gridY, uint32_t gridZ);

private:
    /// Triangle-AABB overlap test (Separating Axis Theorem)
    bool triangleAABBOverlap(
        const glm::vec3& boxCenter,
        const glm::vec3& boxHalfSize,
        const Triangle&  tri
    ) const;
};

} // namespace vwt
