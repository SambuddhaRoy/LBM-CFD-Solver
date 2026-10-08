#pragma once
// ============================================================================
// fluid.hpp — physical properties of the working fluid from its temperature
// and pressure.
//
// The solver is isothermal and incompressible in the low-Mach sense: ambient
// temperature and pressure enter through the fluid's density, viscosity and
// speed of sound, which set the Reynolds and Mach numbers and every
// dimensional output. Heat transfer is not simulated.
//
//   Air, CO2  ideal gas, rho = p / (R T); viscosity from Sutherland's law
//             (pressure-independent for a dilute gas); c = sqrt(gamma R T)
//   Water     liquid; density from the Thiesen equation, viscosity from the
//             Vogel equation, speed of sound from Marczak (1997), with the
//             small compressibility of water for pressure; boiling point at
//             the given pressure from the Antoine equation
// ============================================================================

#include <string>

namespace wt {

enum class FluidKind : int { Air = 0, Water, CO2 };
const char* fluidName(FluidKind k);

struct Ambient {
    FluidKind kind = FluidKind::Air;
    double    T = 288.15;      // K
    double    p = 101325.0;    // Pa
};

struct Properties {
    double rho   = 0;   // kg/m^3
    double mu    = 0;   // Pa s
    double nu    = 0;   // m^2/s
    double sound = 0;   // m/s
    double gamma = 0;   // heat capacity ratio (gases); 0 for the liquid
    double cp    = 0;   // J/(kg K)
    std::string warning;   // empty when the state is inside the models' range
};

Properties properties(const Ambient& a);

// International Standard Atmosphere, 0 to 20 km: temperature and pressure.
void standardAtmosphere(double altitudeM, double& T, double& p);

// Checks the property models against tabulated reference values.
bool fluidSelfTest();

} // namespace wt
