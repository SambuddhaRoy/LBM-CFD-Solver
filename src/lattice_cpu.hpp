#pragma once
// ============================================================================
// lattice_cpu.hpp — the D3Q19 velocity set and equilibrium in double
// precision, for the CPU reference and test setups. Same ordering and algebra
// as shaders/lattice.glsl.
// ============================================================================

namespace wt::cpu {

inline constexpr int CX[19] = {0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 0, 0, 1,-1, 1,-1, 0, 0};
inline constexpr int CY[19] = {0, 0, 0, 1,-1, 0, 0, 1,-1, 0, 0, 1,-1,-1, 1, 0, 0, 1,-1};
inline constexpr int CZ[19] = {0, 0, 0, 0, 0, 1,-1, 0, 0, 1,-1, 1,-1, 0, 0,-1, 1,-1, 1};
inline constexpr double WQ[19] = {1.0/3,
    1.0/18, 1.0/18, 1.0/18, 1.0/18, 1.0/18, 1.0/18,
    1.0/36, 1.0/36, 1.0/36, 1.0/36, 1.0/36, 1.0/36,
    1.0/36, 1.0/36, 1.0/36, 1.0/36, 1.0/36, 1.0/36};
inline constexpr int OPP[19] = {0, 2,1, 4,3, 6,5, 8,7, 10,9, 12,11, 14,13, 16,15, 18,17};

struct Vec3d { double x = 0, y = 0, z = 0; };

// Equilibrium with the D3Q19 third-order terms, unshifted (same algebra as
// equilibrium() in lattice.glsl).
inline void equilibrium(double rho, Vec3d u, double feq[19]) {
    const double uu = u.x*u.x + u.y*u.y + u.z*u.z;
    const double xxy = rho*u.x*u.x*u.y, yzz = rho*u.y*u.z*u.z, xzz = rho*u.x*u.z*u.z;
    const double xyy = rho*u.x*u.y*u.y, yyz = rho*u.y*u.y*u.z, xxz = rho*u.x*u.x*u.z;
    const double ap[3] = {xxy + yzz, xzz + xyy, yyz + xxz};
    const double am[3] = {xxy - yzz, xzz - xyy, yyz - xxz};
    for (int q = 0; q < 19; ++q) {
        const double cx = CX[q], cy = CY[q], cz = CZ[q];
        const double cu = cx*u.x + cy*u.y + cz*u.z;
        const double hxxy = cy*(cx*cx - 1.0/3), hyzz = cy*(cz*cz - 1.0/3);
        const double hxzz = cx*(cz*cz - 1.0/3), hxyy = cx*(cy*cy - 1.0/3);
        const double hyyz = cz*(cy*cy - 1.0/3), hxxz = cz*(cx*cx - 1.0/3);
        const double hp = (hxxy + hyzz)*ap[0] + (hxzz + hxyy)*ap[1] + (hyyz + hxxz)*ap[2];
        const double hm = (hxxy - hyzz)*am[0] + (hxzz - hxyy)*am[1] + (hyyz - hxxz)*am[2];
        feq[q] = WQ[q] * (rho + rho*(3*cu + 4.5*cu*cu - 1.5*uu) + 13.5*hp + 4.5*hm);
    }
}


} // namespace wt::cpu
