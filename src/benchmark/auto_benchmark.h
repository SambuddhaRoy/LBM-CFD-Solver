#pragma once
#include <vector>
#include <string>
#include <chrono>
#include <xlsxwriter.h>

namespace vwt {
    class VulkanEngine;
}

namespace vwt::benchmark {

struct BenchmarkMetric {
    uint64_t step;
    double batchTimeMs;
    double mlups;
    float lbmMs;
    float aeroMs;
    float drag;
    float lift;
    float residual;
    uint64_t vramUsage;
    uint64_t vramBudget;
    std::string gpuName;
    std::string modelName;
    std::string gridRes;
    std::string operatorName;
    float inletVelocity;
};

class AutoBenchmark {
public:
    AutoBenchmark(VulkanEngine* engine);
    void run();

private:
    void recordMetrics(uint64_t step, double batchTimeMs, const std::string& model, const std::string& grid, const std::string& op, float vel);
    void writeExcel();

    VulkanEngine* engine_;
    std::vector<BenchmarkMetric> metrics_;
    std::chrono::steady_clock::time_point batchStart_;
};

} // namespace vwt::benchmark
