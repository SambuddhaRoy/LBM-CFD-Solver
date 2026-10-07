#pragma once
// tests.hpp — self-test against the CPU reference, throughput benchmark,
// and the CFD validation suite.

#include "solver.hpp"

#include <vector>

namespace wt {

struct BenchOptions {
    uint32_t nx = 256, ny = 256, nz = 256;
    uint32_t steps = 200;
    double   peakGBs = 0;          // device peak bandwidth, for the % column
    std::vector<Precision> precisions{Precision::FP32, Precision::FP16S, Precision::FP16C};
};

int runSelfTest(gpu::Context& ctx);
int runBenchmark(gpu::Context& ctx, const BenchOptions& o);
int runValidation(gpu::Context& ctx, Precision prec);

} // namespace wt
