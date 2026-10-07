#pragma once
// ============================================================================
// render.hpp — viewport renderer. Runs render.comp at the viewport's exact
// pixel size and hands the result to ImGui as a texture.
// ============================================================================

#include "solver.hpp"

#include <glm/glm.hpp>

#include <cstdint>
#include <string>

namespace wt {

enum class Field : int { Speed = 0, Pressure, Vorticity, QCriterion };
const char* fieldName(Field f);

struct View {
    int   mode3d = 0;          // 0 = 2D slice, 1 = 3D camera
    Field field  = Field::Speed;
    int   axis   = 2;          // slice normal: 0 x, 1 y, 2 z
    float slice  = 0.f;        // slice position, cells
    bool  grid   = true;       // lattice overlay when zoomed in

    // 2D: plane coordinates at the viewport centre and zoom.
    glm::vec2 center{0.f};
    float     cellsPerPixel = 1.f;

    // 3D orbit camera around `target`.
    glm::vec3 target{0.f};
    float yaw = -35.f, pitch = 25.f, distance = 400.f, fov = 45.f;

    float lo = 0.f, hi = 1.f;  // colour range, display units of the field
};

class Renderer {
public:
    static constexpr uint32_t kSlots = 2;

    void create(gpu::Context& ctx);
    void destroy();

    // Sizes `slot` to the viewport. Call at the start of a frame, before the
    // UI references texture(slot), and only once that slot's previous use
    // has completed (a resize frees the old texture).
    void resize(uint32_t slot, uint32_t width, uint32_t height);
    // Records the render of `slot` at its current size.
    void record(VkCommandBuffer cmd, uint32_t slot, const Solver& solver, const View& view,
                glm::vec3 bodyLo, glm::vec3 bodyHi);

    uint64_t texture(uint32_t slot) const { return slots_[slot].texture; }   // ImTextureID
    // Writes the most recent image of `slot` as a PNG. Waits for the GPU.
    bool savePng(uint32_t slot, const std::string& path);

private:
    struct Slot {
        uint32_t      width = 0, height = 0;
        gpu::Buffer   pixels, params;
        VkImage       image = VK_NULL_HANDLE;
        VmaAllocation alloc = VK_NULL_HANDLE;
        VkImageView   imageView = VK_NULL_HANDLE;
        uint64_t      texture = 0;
    };
    void release(Slot& s);

    gpu::Context* ctx_ = nullptr;
    gpu::Kernel   kernel_;
    VkSampler     sampler_ = VK_NULL_HANDLE;
    Slot          slots_[kSlots];
};

} // namespace wt
