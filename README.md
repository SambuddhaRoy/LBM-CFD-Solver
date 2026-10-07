<div align="center">

<img src="https://img.shields.io/badge/version-v3.0.0-1dd1a1?style=for-the-badge" />
<img src="https://img.shields.io/badge/C%2B%2B-20-00599C?style=for-the-badge&logo=c%2B%2B" />
<img src="https://img.shields.io/badge/Vulkan-1.3-AD1F1F?style=for-the-badge&logo=vulkan" />
<img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" />

<br /><br />

# Wind Tunnel v3

**A real-time GPU wind tunnel: D3Q19 lattice Boltzmann in Vulkan compute**

</div>

---

Put a body in the tunnel (sphere, cube, cylinder, NACA 0012 wing, or any
STL/OBJ/glTF/FBX/PLY model), set the wind, and watch the flow develop live.
Rotate the body while the solver runs. Zoom into any part of the flow: the
viewport is drawn per screen pixel straight from the simulation, so zooming in
shows the lattice's real detail instead of magnified texels.

On an RTX 5070 Ti the solver sustains **9,200 to 9,300 million lattice updates
per second** with 16-bit storage, about 80% of the card's memory bandwidth, and
fits up to ~300 million cells in 16 GB.

v3 is a ground-up rewrite. Only the idea carries over from v2.

![Sphere at Re 5.8e5, velocity slice](docs/screenshots/sphere-velocity.png)

## Performance

`WindTunnel --bench --peak 896`, 256³ cells, sphere obstacle, RTX 5070 Ti:

| storage | MLUPS | effective bandwidth | % of 896 GB/s |
|---|---|---|---|
| FP32  | 4,780 | 731 GB/s | 82% |
| FP16S | 9,180 | 707 GB/s | 79% |
| FP16C | 9,257 | 713 GB/s | 80% |

Throughput is flat with grid size: FP16C runs at 9,215 MLUPS on
1024 x 448 x 448 (205M cells) and 9,241 on 1024 x 512 x 512 (268M cells).
In the interactive app, with rendering every frame, it sustains 9,200 to
9,500 MLUPS.

Bandwidth counts what each cell must move per step: its 19 distributions read
and written once, plus a flag byte (153 B in FP32, 77 B in FP16).

For reference, FluidX3D, the fastest open single-GPU LBM code, publishes
10,304 MLUPS (FP16C) for the RTX 5080. Scaled by memory bandwidth to the
5070 Ti that is about 9,600. That comparison is an estimate (FluidX3D was not
run on this machine), and FluidX3D's default collision is plain BGK, while
this solver runs a recursive-regularized collision with LES in the same
budget.

**Memory:** 49 bytes per cell in FP16 (87 in FP32), so the largest preset,
1024 x 512 x 512 = 268M cells, needs 12.2 GiB.

### Where the speed comes from

- **In-place streaming (Esoteric-Pull, Lehmann 2022).** One copy of the
  distributions instead of two: half the memory, the same traffic per step.
- **16-bit storage, 32-bit arithmetic.** Distributions are stored shifted
  (f - w) in IEEE half (FP16S) or a custom 1-4-11 format (FP16C), halving the
  bytes per step. Validation below shows the coefficients barely move.
- **Bounce-back for free.** A solid cell never runs, so the slot a fluid cell
  wrote its wall-bound population into is untouched; reading it with the step
  parity flipped returns it reversed one step later. That is exact halfway
  bounce-back, and the same bookkeeping supports interpolated (Bouzidi) walls
  with no races: the q < 1/2 blend happens at load time, the q >= 1/2 blend
  at store time, each using only local data.
- **Compile-time step parity.** Each direction lives in its own buffer (so
  grids can exceed Vulkan's 4 GiB per-binding limit), and the parity decides
  which buffer each access uses. Making the parity a specialization constant
  turned every access into a fixed binding: registers fell from 80 to 47.
- **No divergent boundary branch.** Inlet, far-field and outlet cells run the
  same code as every other cell, with density, velocity and stress overridden
  so the collision yields the equilibrium. A separate branch made any warp
  holding a face cell execute both paths; removing it took FP16 from 52% to
  80% of peak.

## Physics

- **Lattice:** D3Q19, single-step stream + collide kernel.
- **Collision:** recursive regularized BGK (Coreixas et al. 2017) with the six
  third-order Hermite terms D3Q19 supports. The non-equilibrium part is
  rebuilt from the stress tensor, which filters ghost modes and stays stable
  down to tau -> 1/2.
- **Turbulence:** Smagorinsky LES from the local non-equilibrium stress.
- **Walls:** halfway or Bouzidi interpolated bounce-back from a signed
  distance field, so curved walls sit at their true sub-cell position.
- **Tunnel:** equilibrium inlet (optional perturbation), far-field side faces
  or periodic, pressure outlet pinned at rho = 1.
- **Forces:** momentum exchange summed inside the step kernel on demand.
  Coefficients use planform area (chord x span) for the wing and projected
  frontal area for every other body.
- **Geometry:** voxelized on the GPU as a signed distance field. Analytic
  bodies are exact in any rotation; meshes use a narrow-band distance splat
  plus ray-parity inside/outside. Rotating the body re-voxelizes in
  milliseconds and keeps the running flow; uncovered cells are revived from
  their neighbours' populations.

## Viewport

- **2D slice** along any axis, pan and unlimited zoom. Every pixel samples
  the 3D field. Scalars are trilinearly interpolated; vorticity and the
  Q-criterion are computed at cell centres by central differences, then
  interpolated, so they stay smooth when magnified. The body outline comes
  from the signed distance field with per-pixel anti-aliasing and stays crisp
  at any zoom. Past ~12 px per cell the lattice grid fades in, so you can see
  the actual resolution you are looking at.
- **3D view:** orbit camera, the slice plane in place, the body sphere-traced
  from the SDF and shaded (coloured by surface pressure in pressure mode).
- **Fields:** velocity, pressure, vorticity, Q-criterion, with a colour bar
  and scale bar in physical units.

![NACA 0012 at 12 degrees, vorticity, zoomed to show the lattice](docs/screenshots/wing-vorticity-zoom.png)

![3D view, sphere coloured by surface pressure](docs/screenshots/sphere-3d-pressure.png)

## Validation

Two suites, both exit 0/1.

**`--selftest`** checks the GPU solver against a double-precision CPU
reference written the textbook way: two buffers, plain pull streaming,
bounce-back spelled out. Sphere (halfway and Bouzidi walls), a pitched cube
with one-cell gaps, and a pitched NACA 0012, with inlet, pressure outlet and
LES active. FP32 agrees to 6e-8 in velocity and 2e-6 in force, i.e. float
round-off; the 16-bit formats track within 7e-5. A mutation check (reverting
to lag-2 bounce-back) fails it by four orders of magnitude, so it does catch
bugs. It also voxelizes a rotated icosphere and a rotated torus mesh and
compares every cell to the exact shape: all misclassified cells lie within
0.02 cells of the true surface.

**`--validate`** runs canonical flows against published data. Every
coefficient is raw: the 2D cases use a domain big enough (5% blockage) that
no blockage correction is applied.

```
  CYLINDER (2D), D = 40 cells, 1600 x 800, blockage 5%, FP16C storage
  Re      C_D (published)          St (published)    L_r/D (published)
  20      2.141 (2.050)  +4.5%          --            0.94 (0.94)
  40      1.630 (1.520)  +7.2%          --            2.27 (2.13-2.35)
  100     1.396 (1.330)  +5.0%     0.171 (0.165)          --
  150     1.377 (1.320)  +4.3%     0.189 (0.184)          --

  SPHERE (3D), D = 40 cells, 480 x 240 x 240, blockage 2.2%
  100     1.141 (1.092)  +4.5%          --            0.86 (0.88)

  STORAGE PRECISION, cylinder Re = 100
  FP32    1.384  +4.0%             0.169
  FP16S   1.392  +4.7%             0.169
  FP16C   1.396  +5.0%             0.171
```

References: Dennis & Chang (1970) and Coutanceau & Bouard (1977) for the
steady wake, Park, Kwon & Choi (1998) and Williamson (1996) for drag and
Strouhal number, Schiller-Naumann for sphere drag, Taneda (1956) for the
sphere wake. Tolerances: C_D 8%, St 6%, L_r/D 12%.

The wake geometry and shedding frequency land within a few percent. Drag is
consistently 4-7% high: what remains of blockage plus the finite resolution
of 40 cells per diameter. Storing the distributions in 16 bits moves C_D by
under 1% and the Strouhal number by at most 0.002.

The whole suite takes about five minutes; the 3D sphere (28M cells, 40,000
steps) takes two of them.

## Accuracy: read before quoting a number

- **The lattice resolves its own Reynolds number.** Set a 1 m body at 30 m/s
  in air and the app asks for Re = 2e6. It simulates that Re when the
  relaxation time allows, otherwise the highest it can and says so. At high
  Re the boundary layer is far thinner than a cell and there is no wall
  model: wake structure, trends and comparisons between shapes are
  meaningful; absolute drag at Re 1e6 is not.
- **Resolution drives the remaining error.** The validated cases put 40 cells
  across the body. Below 32 the app warns.
- **Blockage:** the tunnel's side faces confine the flow. The blockage ratio
  is shown next to the forces; above 10% it warns. No correction is applied.
- Incompressible regime: physical Mach above 0.3 is flagged, not modelled.

## Building

Prerequisites: CMake 3.24+, a C++20 compiler (Visual Studio 2022+),
[vcpkg](https://github.com/microsoft/vcpkg), and a Vulkan 1.3
driver with 8/16-bit storage (any desktop GPU from the last several years).

```powershell
git clone https://github.com/SambuddhaRoy/LBM-CFD-Solver.git
cd LBM-CFD-Solver
cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE="<vcpkg>/scripts/buildsystems/vcpkg.cmake"
cmake --build build --config Release
```

Dependencies (installed by vcpkg on first configure): Vulkan loader, GLFW,
Dear ImGui, Assimp, GLM, vk-bootstrap, VMA, shaderc (for `glslc`), stb.

v3 has been built and tested on Windows only. The code avoids
platform-specific paths except the Windows file dialog (on Linux, drop a
model onto the window), but the Linux build is untested.

## Usage

```
WindTunnel                      interactive wind tunnel
WindTunnel --selftest           GPU solver vs CPU reference, all precisions
WindTunnel --bench [--grid X Y Z] [--precision fp32|fp16s|fp16c|all] [--peak GBs]
WindTunnel --validate [--precision P]
WindTunnel --capture out.png [frames]   run the app, save the window, exit
           --mesh FILE --shape S --pitch DEG --view 2d|3d --field F --zoom X --preset N
```

| Input | Action |
|---|---|
| `Space` | run / pause |
| `R` | reset flow |
| `1`-`4` | velocity / pressure / vorticity / Q-criterion |
| `V` | 2D slice / 3D view |
| `[` `]` | move the slice (Shift: 10 cells) |
| `H` | fit view |
| `S` | save the viewport as PNG |
| wheel | zoom about the cursor (2D) / dolly (3D) |
| drag | pan (2D) / orbit (3D); right drag pans in 3D |

Drop a model file anywhere on the window to load it.

## Code map

```
src/
  vk.*         Vulkan context: device, buffers, push-descriptor compute kernels
  solver.*     lattice buffers, step recording, forces, statistics, probe
  geometry.*   analytic bodies, mesh import, GPU signed-distance voxelization
  render.*     viewport renderer
  app.*        window, swapchain, frame loop, UI
  tests.cpp    CPU reference self-test, benchmark
  validate.cpp CFD validation suite
shaders/
  lattice.glsl   velocity set, storage formats, indexing, equilibrium
  stream.glsl    in-place streaming with bounce-back, moment accumulation
  step.comp      the solver kernel
  render.comp    per-pixel viewport
  ...            geometry, initialisation, reductions
```

## License

MIT.
