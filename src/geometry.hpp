#pragma once
// ============================================================================
// geometry.hpp — bodies in the tunnel, voxelized on the GPU as a signed
// distance field (lattice units, > 0 in fluid). The SDF drives everything
// geometric: cell types, interpolated (Bouzidi) walls, and the rendered
// surface, which stays smooth at any zoom because it is interpolated rather
// than drawn as voxels.
// ============================================================================

#include "solver.hpp"

#include <glm/glm.hpp>

#include <string>
#include <vector>

namespace wt {

enum class Shape : int { Sphere = 0, Cube, Cylinder, Wing, Mesh };
const char* shapeName(Shape s);

struct Body {
    Shape     shape  = Shape::Sphere;
    glm::vec3 center = {0, 0, 0};    // lattice coordinates
    float     length = 32.f;         // sphere/cylinder diameter, cube edge, wing chord,
                                     // mesh: longest extent, all in cells
    float     span   = 0.f;          // cylinder/wing extent along body z; 0 = through the domain
    float     pitch = 0.f, yaw = 0.f, roll = 0.f;   // degrees; pitch > 0 = nose up
};

// World -> body rotation for the given angles.
glm::mat3 bodyRotation(const Body& b);

struct Triangle { glm::vec3 a, b, c; };

struct MeshReport {
    size_t edges = 0, openEdges = 0, nonManifoldEdges = 0;
    std::string alignment;              // how the model was oriented in the tunnel
    double extent = 0;                  // longest extent in file units
    bool   metres = false;              // file units are metres (glTF by specification)
    bool watertight() const { return openEdges == 0 && nonManifoldEdges == 0; }
};

// Edge topology of a triangle soup: closed surfaces share every edge between
// exactly two triangles.
MeshReport checkTopology(const std::vector<Triangle>& tris);

// Loads any Assimp-supported mesh (STL, OBJ, glTF, FBX, PLY, ...) into a flat
// triangle list, centred, scaled so its longest extent is 1, and oriented in
// the tunnel (see loadMesh). Returns false and sets `error` on failure.
bool loadMesh(const std::string& path, std::vector<Triangle>& tris, std::string& error,
              MeshReport* report = nullptr);

// Lists every mesh in a model file with its triangle count and bounds.
int printMeshInfo(const std::string& path);

class Geometry {
public:
    void create(gpu::Context& ctx);
    void destroy();

    // Mesh to use when body.shape == Shape::Mesh (unit-scaled, see loadMesh).
    // watertight selects exact ray-parity voxelization; otherwise shrink-wrap.
    void setMesh(std::vector<Triangle> tris, bool watertight);
    bool hasMesh() const { return !mesh_.empty(); }

    // Writes the solver's SDF for `body`, then classifies cells (keeping the
    // flow) and measures the frontal area. Submits and waits.
    void apply(Solver& solver, const Body& body);

    uint32_t frontalCells() const { return frontal_; }   // projected area, cells

    // Axis-aligned size of the body as placed in the tunnel, in units of its
    // length; `through` is set for cylinders and wings spanning the domain.
    glm::vec3 extent(const Body& b, bool& through) const;

private:
    static constexpr uint32_t kBin = 4;     // parity-ray bin size, cells

    void ensure(gpu::Buffer& buf, VkDeviceSize bytes);
    void prepareWing(const Body& b);
    void prepareMesh(Solver& s, const Body& b);
    void writeAnalytic(VkCommandBuffer cmd, Solver& s, const Body& b);
    void writeMeshDistance(VkCommandBuffer cmd, bool parity);
    void writeMeshFinal(VkCommandBuffer cmd, Solver& s);
    void shrinkWrap();
    uint32_t measureFrontal(Solver& s);

    gpu::Context* ctx_ = nullptr;
    gpu::Kernel sdfShape_, sdfSplat_, sdfSign_, frontalK_;
    gpu::Buffer poly_;                  // wing profile, lattice units
    uint32_t    polyCount_ = 0;
    glm::vec4   polyBounds_{0};
    std::vector<Triangle> mesh_;
    bool        watertight_ = true;
    gpu::Buffer tris_, bins_, scratch_; // mesh voxelization working set
    uint32_t    nTris_ = 0, nbY_ = 0, nbZ_ = 0;
    glm::ivec3  boxOrigin_{0}, boxDims_{1};
    uint32_t    frontal_ = 0;
};

} // namespace wt
