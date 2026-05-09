#pragma once
#include <vector>
#include <string>

namespace vwt::benchmark {

struct GridRes {
    uint32_t x, y, z;
    std::string toString() const {
        return std::to_string(x) + "x" + std::to_string(y) + "x" + std::to_string(z);
    }
};

// Benchmark configurations
constexpr int WARMUP_STEPS = 200;
constexpr int MEASUREMENT_STEPS = 500;
constexpr int CHECKPOINT_INTERVAL = 20;

const std::vector<GridRes> GRID_RESOLUTIONS = {
    {64, 32, 32},
    {128, 64, 64},
    {256, 128, 128}
};

const std::vector<std::string> COLLISION_OPERATORS = {
    "BGK",
    "MRT-RLB"
};

const std::vector<float> INLET_VELOCITIES = {
    0.03f,
    0.05f,
    0.10f
};

} // namespace vwt::benchmark
