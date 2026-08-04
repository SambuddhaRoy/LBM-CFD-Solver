#pragma once
// ============================================================================
// mesh.h — triangle import, analytic primitives, and voxelization
// ============================================================================

#include <glm/glm.hpp>

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace vwt {

enum class Shape { Sphere = 0, Cube, Cylinder, Wing };
inline const char* shapeName(Shape s) {
    switch (s) {
    case Shape::Sphere:   return "Sphere";
    case Shape::Cube:     return "Cube";
    case Shape::Cylinder: return "Cylinder";
    case Shape::Wing:     return "NACA 0012 wing";
    }
    return "?";
}

struct Tri { glm::vec3 a, b, c; };

struct VoxelModel {
    std::vector<uint32_t> occupancy;     // gx*gy*gz, 1 = solid
    uint32_t frontalCells = 0;           // YZ-projected area, cells²
    uint32_t spanCellsX   = 0;           // model extent along flow, cells
    float    fillPct      = 0.f;
    uint32_t triCount     = 0;
    std::string name;
};

namespace mesh {

// Import any Assimp-supported format into a triangle soup.
bool loadTriangles(const std::filesystem::path& path,
                   std::vector<Tri>& out, std::string& error);

// Voxelize triangles into the grid. The mesh is recentred, rotated by pitch
// (angle of attack, about Z), yaw (about Y) and roll (about the flow axis X),
// then scaled to a wind-tunnel-appropriate fraction of the test section.
VoxelModel voxelizeTriangles(const std::vector<Tri>& tris,
                             uint32_t gx, uint32_t gy, uint32_t gz,
                             float pitchDeg, float yawDeg, float rollDeg,
                             const std::string& name);

// Analytic primitives — exact occupancy, no mesh required.
VoxelModel makePrimitive(Shape shape,
                         uint32_t gx, uint32_t gy, uint32_t gz,
                         float pitchDeg, float yawDeg, float rollDeg);

} // namespace mesh
} // namespace vwt
