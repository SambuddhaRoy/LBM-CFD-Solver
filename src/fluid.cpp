// ============================================================================
// fluid.cpp — fluid property models (see fluid.hpp)
// ============================================================================

#include "fluid.hpp"

#include <cmath>
#include <cstdio>

namespace wt {

namespace {

// Sutherland's law: mu = mu0 (T/T0)^1.5 (T0 + S) / (T + S).
double sutherland(double T, double mu0, double T0, double S) {
    return mu0 * std::pow(T / T0, 1.5) * (T0 + S) / (T + S);
}

Properties gas(const Ambient& a, double R, double gamma, double cp,
               double mu0, double T0, double S, double tMin, double tMax) {
    Properties p;
    p.rho   = a.p / (R * a.T);
    p.mu    = sutherland(a.T, mu0, T0, S);
    p.nu    = p.mu / p.rho;
    p.sound = std::sqrt(gamma * R * a.T);
    p.gamma = gamma;
    p.cp    = cp;
    if (a.T < tMin || a.T > tMax) p.warning = "temperature outside the viscosity model's range";
    else if (a.p > 5e6)           p.warning = "pressure high enough that the ideal-gas law is inaccurate";
    return p;
}

Properties water(const Ambient& a) {
    Properties p;
    const double t = a.T - 273.15;                                   // Celsius
    // Thiesen (1900), as used for the CIPM density table, 0-40 C to ~1 ppm
    // and within ~0.1% up to 100 C. Isothermal compressibility 4.6e-10 /Pa.
    const double rho1atm = 999.97495 * (1.0 - (t - 3.983035) * (t - 3.983035) * (t + 301.797)
                                              / (522528.9 * (t + 69.34881)));
    p.rho = rho1atm * (1.0 + 4.6e-10 * (a.p - 101325.0));
    // Vogel equation, within ~2.5% from 0 to 370 C.
    p.mu = 2.414e-5 * std::pow(10.0, 247.8 / (a.T - 140.0));
    p.nu = p.mu / p.rho;
    // Marczak (1997), 0-95 C.
    p.sound = 1.402385e3 + t * (5.038813 + t * (-5.799136e-2 + t * (3.287156e-4
              + t * (-1.398845e-6 + t * 2.787860e-9))));
    p.cp = 4182.0;
    // Antoine equation (1-100 C): boiling point at this pressure.
    const double pMmHg = a.p / 133.322;
    const double tBoil = 1730.63 / (8.07131 - std::log10(pMmHg)) - 233.426;
    if (t <= 0.0)          p.warning = "water is frozen at this temperature";
    else if (t >= tBoil)   p.warning = "water boils at this temperature and pressure";
    else if (t > 95.0)     p.warning = "temperature above the speed-of-sound fit (95 C)";
    return p;
}

} // namespace

const char* fluidName(FluidKind k) {
    switch (k) {
    case FluidKind::Air:   return "Air";
    case FluidKind::Water: return "Water";
    default:               return "Carbon dioxide";
    }
}

Properties properties(const Ambient& a) {
    switch (a.kind) {
    // Air: R = 287.058, Sutherland constants from the ISA (mu0 = 1.716e-5 at
    // 273.15 K, S = 110.4 K), valid ~100-1900 K.
    case FluidKind::Air: return gas(a, 287.058, 1.4, 1005.0, 1.716e-5, 273.15, 110.4, 100.0, 1900.0);
    // CO2: R = 188.92, gamma ~1.29 near room temperature, Sutherland
    // constants from White, Viscous Fluid Flow (mu0 = 1.370e-5 at 273 K,
    // S = 222 K, 190-1700 K within 2%).
    case FluidKind::CO2: return gas(a, 188.92, 1.29, 844.0, 1.370e-5, 273.0, 222.0, 190.0, 1700.0);
    default:             return water(a);
    }
}

void standardAtmosphere(double h, double& T, double& p) {
    if (h <= 11000.0) {
        T = 288.15 - 0.0065 * h;
        p = 101325.0 * std::pow(T / 288.15, 5.255877);
    } else {
        T = 216.65;
        p = 22632.06 * std::exp(-1.576885e-4 * (h - 11000.0));
    }
}

bool fluidSelfTest() {
    struct Ref { const char* what; FluidKind k; double T, p, rho, mu, sound; };
    // ISA tables (air), the CIPM / IAPWS water tables, NIST (CO2).
    const Ref refs[] = {
        {"air, ISA sea level",  FluidKind::Air,   288.15, 101325.0, 1.2250, 1.7894e-5, 340.29},
        {"air, ISA 11 km",      FluidKind::Air,   216.65,  22632.1, 0.36392, 1.4216e-5, 295.07},
        {"water, 20 C",         FluidKind::Water, 293.15, 101325.0, 998.21, 1.0016e-3, 1482.3},
        {"water, 80 C",         FluidKind::Water, 353.15, 101325.0, 971.79, 3.545e-4, 1554.0},
        {"CO2, 300 K, 1 bar",   FluidKind::CO2,   300.00, 100000.0, 1.7730, 1.5021e-5, 0.0},
    };
    bool ok = true;
    std::printf("\n  fluid properties against reference tables\n");
    for (const Ref& r : refs) {
        const Properties p = properties({r.k, r.T, r.p});
        const double eRho = std::abs(p.rho / r.rho - 1), eMu = std::abs(p.mu / r.mu - 1);
        const double eC = r.sound > 0 ? std::abs(p.sound / r.sound - 1) : 0.0;
        // Viscosity correlations are good to ~2.5%; density to 1% (the ideal-gas
        // law is 0.5% off for CO2 at 1 bar) and speed of sound to 0.5%.
        const bool pass = eRho < 0.01 && eMu < 0.025 && eC < 0.005;
        ok = ok && pass;
        std::printf("  %-22s rho %+.2f%%  mu %+.2f%%  c %+.2f%%  %s\n", r.what, 100 * (p.rho / r.rho - 1),
                    100 * (p.mu / r.mu - 1), r.sound > 0 ? 100 * (p.sound / r.sound - 1) : 0.0, pass ? "PASS" : "FAIL");
    }
    return ok;
}

} // namespace wt
