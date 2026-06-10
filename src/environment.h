#pragma once
#include <string>
#include <vector>
#include <algorithm>
#include <cstdint>

namespace vwt {

struct EnvironmentProfile {
    std::string name;
    float density;         // kg/m^3
    float dynamicViscosity; // Pa*s
    float speedOfSound;    // m/s
    std::string description;

    float getKinematicViscosity() const {
        return dynamicViscosity / density;
    }
};

class EnvironmentRegistry {
public:
    static const std::vector<EnvironmentProfile>& getProfiles() {
        static const std::vector<EnvironmentProfile> profiles = {
            {"Earth (Air)", 1.225f, 1.81e-5f, 343.0f, "Standard Earth atmosphere at sea level."},
            {"Mars", 0.020f, 1.10e-5f, 240.0f, "Thin CO2 atmosphere with low density and pressure."},
            {"Venus", 65.0f, 3.30e-5f, 410.0f, "Super-critical CO2 atmosphere with high density."},
            {"Titan", 5.30f, 0.60e-5f, 194.0f, "Dense Nitrogen/Methane atmosphere at cryogenic temperatures."},
            {"Underwater", 1000.0f, 1.00e-3f, 1500.0f, "Standard liquid water conditions."}
        };
        return profiles;
    }

    // Safe lookup that clamps the index to the valid range.
    static const EnvironmentProfile& get(uint32_t idx) {
        const auto& p = getProfiles();
        uint32_t i = (p.empty() ? 0u : std::min(idx, uint32_t(p.size() - 1)));
        return p[i];
    }
};

// ─── Unit conversion ────────────────────────────────────────────────────────
// LBM lattice speed of sound c_s_lat = 1/√3. Holding Mach number constant
// between lattice and physical worlds gives  u_phys = u_lat · c_s_phys · √3.
// Earth: 343 m/s · 1.732 ≈ 594.05 m/s per lattice unit  (the magic number
// previously hard-coded throughout the UI).
inline float latticeToMps(const EnvironmentProfile& env) {
    constexpr float kSqrt3 = 1.7320508075688772f;
    return env.speedOfSound * kSqrt3;
}

} // namespace vwt
