#pragma once
// ============================================================================
// validate.h — Reynolds-resolved CFD validation harness
//
// Drives the solver through canonical bluff-body cases and measures the
// quantities a CFD solver is judged on: flow-regime transition, Strouhal
// number of vortex shedding, wake recirculation length, and vortex-core
// position. Runs windowless; prints a literature comparison.
// ============================================================================

#include <string>
#include <vector>

namespace vwt {

struct ValidateOptions {
    std::string cylinderMesh;   // optional: provided cylinder model
    std::string cubeMesh;       // optional: provided cube model
    bool        dumpImages = true;
};

int runValidation(const ValidateOptions& opts);

} // namespace vwt
