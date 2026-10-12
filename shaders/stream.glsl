// ============================================================================
// stream.glsl — the load half of a step: Esoteric-Pull streaming with
// halfway / Bouzidi bounce-back, folded straight into raw moments. Shared by
// step.comp and probe.comp so the readback used by tests and validation sees
// exactly what the solver sees. Requires bindings 1 (flags) and 2 (sdf).
// ============================================================================

layout(std430, set = 0, binding = 1) readonly buffer Flags { uint8_t flags[]; };
layout(std430, set = 0, binding = 2) readonly buffer Sdf { float16_t sdf[]; };

bool isSolid(uint idx) { return (uint(flags[idx]) & FLAG_SOLID) != 0u; }

// Fraction of the link from fluid cell n to solid cell s at which the wall
// sits, from the signed distance at both ends. 1/2 is halfway bounce-back.
// Recomputed where needed rather than cached: wall cells are rare, and a
// per-direction cache would cost every cell 19 registers.
float linkQ(uint n, uint s) {
    if (pc.bouzidi == 0.0) return 0.5;
    const float phiF = float(sdf[n]);
    const float d = phiF - float(sdf[s]);
    return d > 1e-6 ? clamp(phiF / d, 0.0, 1.0) : 0.5;
}

// Bit q of the result: the population arriving along c_q comes from a solid
// cell (n - c_q) and must be bounced back instead.
uint wallLinks(const Nbr b) {
    uint solid = 0u;
    [[unroll]] for (uint i = 1u; i < 19u; i += 2u) {
        if (isSolid(nb(b, i + 1u))) solid |= 1u << i;          // n - c_i
        if (isSolid(nb(b, i)))      solid |= 1u << (i + 1u);   // n + c_i
    }
    return solid;
}

// Raw moments of the shifted populations: everything the collision needs.
struct Moments {
    float rhom1;                                  // sum f~ = rho - 1
    vec3  mom;                                    // sum c f~ = rho u
    float sxx, syy, szz, sxy, syz, sxz;           // sum c c f~
};

void addPop(inout Moments m, uint q, float f) {
    const vec3 c = vec3(C[q]);
    m.rhom1 += f;
    m.mom   += c * f;
    m.sxx += c.x*c.x*f; m.syy += c.y*c.y*f; m.szz += c.z*c.z*f;
    m.sxy += c.x*c.y*f; m.syz += c.y*c.z*f; m.sxz += c.x*c.z*f;
}

// Streams in the populations arriving at fluid cell b.n one opposite pair at
// a time and folds them into raw moments, so no 19-entry array is ever live.
// `fin` collects the incoming half of the momentum exchange on wall links.
//
// `road`: this cell is in the first row above a moving ground (the solid row
// y = 0, wall halfway at y = 1/2, moving at the inlet velocity like a road
// under a car). Its links into the road bounce back with the moving-wall
// term 6 w_q (c_q . u_road) and are left out of the body's force.
Moments streamIn(const Nbr b, bool odd, uint solid, bool road, inout vec3 fin) {
    Moments m;
    m.rhom1 = 0.0; m.mom = vec3(0.0);
    m.sxx = 0.0; m.syy = 0.0; m.szz = 0.0; m.sxy = 0.0; m.syz = 0.0; m.sxz = 0.0;
    addPop(m, 0u, ddfLoad(0u, b.n));
    [[unroll]] for (uint i = 1u; i < 19u; i += 2u) {
        const bool sm = (solid & (1u << i)) != 0u;
        const bool sp = (solid & (1u << (i + 1u))) != 0u;
        // Bounce-back = the same location with the parity flipped: the slot
        // this cell wrote its wall-bound population into one step ago. Each
        // side of the branch names a fixed slot, so with the parity a
        // compile-time constant (step.comp) no load ever picks its buffer at
        // runtime.
        float a = sm ? ddfLoad(slotOwn(i, !odd), b.n)      : ddfLoad(slotOwn(i, odd), b.n);
        float c = sp ? ddfLoad(slotNbr(i, !odd), nb(b, i)) : ddfLoad(slotNbr(i, odd), nb(b, i));
        // a arrives along c_i from n - c_i, c along -c_i from n + c_i: below
        // this cell when c_i points up, respectively down.
        const bool rA = sm && road && C[i].y > 0;
        const bool rC = sp && road && C[i].y < 0;

        // Bouzidi with the wall closer than halfway (q < 1/2): interpolate
        // the bounced value with the population that arrived from the
        // opposite side; both are local. The q >= 1/2 branch needs this
        // cell's post-collision values, so step.comp applies it at store
        // time; that split is what keeps interpolated walls race-free under
        // in-place streaming. A one-cell gap (both sides solid) is halfway.
        if (sm != sp && pc.bouzidi != 0.0 && !rA && !rC) {
            const float a0 = a, c0 = c;
            if (sp) {
                const float q = linkQ(b.n, nb(b, i));
                if (q < 0.5) c = 2.0*q*c0 + (1.0 - 2.0*q)*a0;
            } else {
                const float q = linkQ(b.n, nb(b, i + 1u));
                if (q < 0.5) a = 2.0*q*a0 + (1.0 - 2.0*q)*c0;
            }
        }
        // Incoming half of the momentum exchange: for a population that came
        // back off a wall along c_q the wall-bound direction is -c_q, so its
        // share c_out * f_in is -c_q f_q. Shifted populations (f - w) are
        // used, which takes out the ambient pressure: on a closed body it
        // sums to zero anyway, but a body standing on the road has no fluid
        // under its contact patches, and the ambient pressure (rho/3, ~67x
        // the dynamic pressure at u = 0.1) on the rest of it would read as
        // tonnes of downforce.
        if (rA) a += 6.0 * W[i] * dot(vec3(C[i]), pc.uin.xyz);
        if (rC) c += 6.0 * W[i] * dot(vec3(C[i + 1u]), pc.uin.xyz);
        if (sm && !rA) fin -= vec3(C[i])      * a;
        if (sp && !rC) fin -= vec3(C[i + 1u]) * c;
        addPop(m, i, a);
        addPop(m, i + 1u, c);
    }
    return m;
}
