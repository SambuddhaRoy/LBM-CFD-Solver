// ============================================================================
// lattice.glsl — D3Q19 definitions shared by every kernel that touches the
// distribution functions (DDFs).
//
// Storage
//   One copy of the DDFs, streamed in place with Esoteric-Pull (Lehmann 2022).
//   Direction q lives in its own buffer F[q] (structure of arrays), so each
//   binding stays under Vulkan's 4 GiB storage-buffer range on huge grids.
//
//   DDFs are stored shifted, f~ = f - w, which keeps their magnitude far from
//   the density offset and buys precision in every format. Arithmetic is
//   always FP32; STORE picks the memory format:
//     0  FP32
//     1  FP16S  IEEE half, scaled by 2^15 so shifted DDFs use the normal range
//     2  FP16C  custom 1-4-11 half: one more mantissa bit, range +-2
//
// Velocity set ordering: direction i+1 is the opposite of i for odd i. The
// in-place streaming scheme relies on that pairing.
// ============================================================================

#extension GL_EXT_control_flow_attributes : require
#extension GL_EXT_shader_8bit_storage : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require

#ifndef STORE
#define STORE 2
#endif

const uint FLAG_SOLID  = 1u;    // obstacle cell; never updated
const uint FLAG_EQ     = 2u;    // fixed equilibrium: inlet and far-field faces
const uint FLAG_OUTLET = 4u;    // pressure outlet: rho = 1, velocity from x-1
const uint FLAG_WALL   = 8u;    // fluid cell with at least one solid neighbour

const ivec3 C[19] = ivec3[19](
    ivec3( 0, 0, 0),
    ivec3( 1, 0, 0), ivec3(-1, 0, 0), ivec3( 0, 1, 0), ivec3( 0,-1, 0),
    ivec3( 0, 0, 1), ivec3( 0, 0,-1), ivec3( 1, 1, 0), ivec3(-1,-1, 0),
    ivec3( 1, 0, 1), ivec3(-1, 0,-1), ivec3( 0, 1, 1), ivec3( 0,-1,-1),
    ivec3( 1,-1, 0), ivec3(-1, 1, 0), ivec3( 1, 0,-1), ivec3(-1, 0, 1),
    ivec3( 0, 1,-1), ivec3( 0,-1, 1));

const float W0 = 1.0/3.0, WS = 1.0/18.0, WE = 1.0/36.0;
const float W[19] = float[19](W0, WS,WS,WS,WS,WS,WS,
                              WE,WE,WE,WE,WE,WE,WE,WE,WE,WE,WE,WE);

layout(push_constant) uniform Params {
    uint  Nx, Ny, Nz;
    uint  t;           // time step; its parity selects the Esoteric-Pull slots
    float tau0;        // molecular relaxation time, nu = (tau0 - 1/2) / 3
    float smag2;       // Smagorinsky constant squared (0 disables LES)
    float turb;        // inlet perturbation amplitude, fraction of |u_in|
    float bouzidi;     // 1 = interpolated walls, 0 = halfway bounce-back
    vec4  uin;         // xyz: inlet / far-field velocity
    uint  writeField;  // 1 = write the render field this step
    uint  faces;       // bit 0: Y faces far field, bit 1: Z faces (else periodic)
    uint  pad1, pad2;
} pc;

// ─── DDF storage ────────────────────────────────────────────────────────────
#if STORE == 0
  #define DDF_T float
#elif STORE == 1
  #define DDF_T float16_t
#else
  #define DDF_T uint16_t
#endif

layout(std430, set = 0, binding = 0) buffer DDF { DDF_T v[]; } F[19];

#if STORE == 2
// Custom 16-bit float: 1 sign, 4 exponent, 11 mantissa bits, so magnitudes up
// to 2 with roughly 2x the resolution of IEEE half. Bit-exact port of the
// FluidX3D conversion, with denormal handling.
float h2f(uint x) {
    const uint e = (x & 0x7800u) >> 11;
    const uint m = (x & 0x07FFu) << 12;
    const uint v = floatBitsToUint(float(m)) >> 23;
    return uintBitsToFloat((x & 0x8000u) << 16
        | uint(e != 0u) * ((e + 112u) << 23 | m)
        | uint(e == 0u && m != 0u) * ((v - 37u) << 23 | ((m << (150u - v)) & 0x007FF000u)));
}
uint f2h(float x) {
    const uint b = floatBitsToUint(x) + 0x00000800u;
    const uint e = (b & 0x7F800000u) >> 23;
    const uint m = b & 0x007FFFFFu;
    return (b & 0x80000000u) >> 16
        | uint(e > 112u) * ((((e - 112u) << 11) & 0x7800u) | m >> 12)
        | uint(e < 113u && e > 100u) * ((((0x007FF800u + m) >> (124u - e)) + 1u) >> 1);
}
#endif

// Both 16-bit formats overflow just below 2; clamping keeps a diverging cell
// from wrapping into a plausible-looking finite value.
float ddfLoad(uint q, uint idx) {
#if STORE == 0
    return F[q].v[idx];
#elif STORE == 1
    return float(F[q].v[idx]) * (1.0 / 32768.0);
#else
    return h2f(uint(F[q].v[idx]));
#endif
}
void ddfStore(uint q, uint idx, float x) {
#if STORE == 0
    F[q].v[idx] = x;
#elif STORE == 1
    F[q].v[idx] = float16_t(clamp(x, -1.999, 1.999) * 32768.0);
#else
    F[q].v[idx] = uint16_t(f2h(clamp(x, -1.999, 1.999)));
#endif
}

// ─── Indexing ───────────────────────────────────────────────────────────────
// 2D dispatch of 256-wide groups so grids above 65535 * 256 cells still fit
// the portable workgroup-count limit.
uint cellIndex() {
    return (gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x) * 256u
         + gl_LocalInvocationID.x;
}
uint cellCount() { return pc.Nx * pc.Ny * pc.Nz; }

uvec3 coords(uint n) {
    const uint xy = pc.Nx * pc.Ny;
    const uint z = n / xy, r = n - z * xy;
    const uint y = r / pc.Nx;
    return uvec3(r - y * pc.Nx, y, z);
}

// Neighbourhood of a cell with periodic wrap: its index plus the index
// offsets one step down/up each axis. Domain faces that must not be periodic
// carry boundary flags, so the wrap only ever links boundary cells.
//
// Seven registers instead of a 19-entry index array: nb(q) recombines them,
// and with q a constant after unrolling that is one or two adds. Offsets are
// unsigned and rely on wrap-around arithmetic for the negative ones.
struct Nbr {
    uint n;
    uint xm, xp, ym, yp, zm, zp;
};

Nbr neighbourhood(uvec3 p) {
    const uint xy = pc.Nx * pc.Ny;
    Nbr b;
    b.n  = (p.z * pc.Ny + p.y) * pc.Nx + p.x;
    b.xm = p.x == 0u          ? pc.Nx - 1u               : 0xFFFFFFFFu;
    b.xp = p.x + 1u == pc.Nx  ? 1u - pc.Nx               : 1u;
    b.ym = p.y == 0u          ? (pc.Ny - 1u) * pc.Nx     : 0u - pc.Nx;
    b.yp = p.y + 1u == pc.Ny  ? 0u - (pc.Ny - 1u) * pc.Nx : pc.Nx;
    b.zm = p.z == 0u          ? (pc.Nz - 1u) * xy        : 0u - xy;
    b.zp = p.z + 1u == pc.Nz  ? 0u - (pc.Nz - 1u) * xy   : xy;
    return b;
}

uint nb(const Nbr b, uint q) {
    const ivec3 c = C[q];
    return b.n + (c.x < 0 ? b.xm : c.x > 0 ? b.xp : 0u)
               + (c.y < 0 ? b.ym : c.y > 0 ? b.yp : 0u)
               + (c.z < 0 ? b.zm : c.z > 0 ? b.zp : 0u);
}

// ─── Esoteric-Pull streaming ────────────────────────────────────────────────
// Each cell touches its own slots and the slots of its 9 "positive"
// neighbours j[i] (odd i); every slot is read and written by exactly one
// thread per step, which is what makes in-place streaming race-free.
//
// Slot of the population arriving at n along c_i:      (n,    odd ? i   : i+1)
// Slot of the population arriving at n along c_{i+1}:  (j[i], odd ? i+1 : i  )
// A population stored at step t is loaded at t+1 by the cell it streams to,
// from the same slot (the parity flip moves it one link).
//
// Bounce-back falls out of the same bookkeeping. A solid cell never runs, so
// nothing overwrites the slot a fluid cell stored its wall-bound population
// into; reading that slot with the parity flipped returns it one step later,
// reversed. That is exact halfway bounce-back (lag 1), unlike the implicit
// lag-2 variant obtained by simply skipping solid cells.
uint slotOwn(uint i, bool odd) { return odd ? i : i + 1u; }
uint slotNbr(uint i, bool odd) { return odd ? i + 1u : i; }

void storeAll(const Nbr b, bool odd, const float f[19]) {
    ddfStore(0u, b.n, f[0]);
    [[unroll]] for (uint i = 1u; i < 19u; i += 2u) {
        ddfStore(slotNbr(i, odd), nb(b, i), f[i]);
        ddfStore(slotOwn(i, odd), b.n, f[i + 1u]);
    }
}

// ─── Equilibrium ────────────────────────────────────────────────────────────
// Second-order Hermite equilibrium plus the six third-order terms D3Q19 can
// represent. On D3Q19 the individual H_xxy, H_yzz ... are not orthogonal, so
// they enter as the orthogonal combinations (H_xxy +- H_yzz) etc., whose
// inverse norms are 1/(2 cs^6) = 13.5 and 1/(6 cs^6) = 4.5 with cs^2 = 1/3.
// The third-order terms cancel the cubic velocity error of the plain
// second-order equilibrium in the off-diagonal stress.

// Hermite third-order combinations for direction q, (plus[3], minus[3]):
//   plus  = (H_xxy+H_yzz, H_xzz+H_xyy, H_yyz+H_xxz)
//   minus = (H_xxy-H_yzz, H_xzz-H_xyy, H_yyz-H_xxz)
void hermite3(uint q, out vec3 hp, out vec3 hm) {
    const vec3 c = vec3(C[q]);
    const vec3 c2 = c * c - 1.0/3.0;
    const float xxy = c.y * c2.x, yzz = c.y * c2.z;
    const float xzz = c.x * c2.z, xyy = c.x * c2.y;
    const float yyz = c.z * c2.y, xxz = c.z * c2.x;
    hp = vec3(xxy + yzz, xzz + xyy, yyz + xxz);
    hm = vec3(xxy - yzz, xzz - xyy, yyz - xxz);
}

// Same combinations of a symmetric third-order tensor given by its six
// distinct components a = (xxy, yzz, xzz, xyy, yyz, xxz).
void combos3(float xxy, float yzz, float xzz, float xyy, float yyz, float xxz,
             out vec3 ap, out vec3 am) {
    ap = vec3(xxy + yzz, xzz + xyy, yyz + xxz);
    am = vec3(xxy - yzz, xzz - xyy, yyz - xxz);
}

// Shifted equilibrium f~eq_q = feq_q - w_q for density 1 + rhom1.
// The -w_q is folded in analytically via rhom1, so no large cancellation.
void equilibrium(float rhom1, vec3 u, out float feq[19]) {
    const float rho = 1.0 + rhom1;
    const float uu = dot(u, u);
    vec3 ap, am;
    combos3(rho*u.x*u.x*u.y, rho*u.y*u.z*u.z, rho*u.x*u.z*u.z,
            rho*u.x*u.y*u.y, rho*u.y*u.y*u.z, rho*u.x*u.x*u.z, ap, am);
    [[unroll]] for (uint q = 0u; q < 19u; ++q) {
        const float cu = dot(vec3(C[q]), u);
        vec3 hp, hm; hermite3(q, hp, hm);
        feq[q] = W[q] * (rhom1 + rho * (3.0*cu + 4.5*cu*cu - 1.5*uu)
                         + 13.5 * dot(hp, ap) + 4.5 * dot(hm, am));
    }
}
