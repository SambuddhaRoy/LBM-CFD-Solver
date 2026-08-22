<div align="center">

<img src="https://img.shields.io/badge/version-v2.0.0--beta-1dd1a1?style=for-the-badge" />
<img src="https://img.shields.io/badge/C%2B%2B-23-00599C?style=for-the-badge&logo=c%2B%2B" />
<img src="https://img.shields.io/badge/Vulkan-1.3-AD1F1F?style=for-the-badge&logo=vulkan" />
<img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" />
<img src="https://img.shields.io/badge/Platform-Windows%20%7C%20Linux-0078D4?style=for-the-badge" />

<br /><br />

# Virtual Wind Tunnel v2

**Real-time GPU aerodynamics — a D3Q19 Lattice Boltzmann solver in Vulkan compute**

</div>

---

Drop in a model, or pick a built-in sphere, cylinder, cube, or NACA 0012 wing.
Set the wind speed, rotate the body on all three axes while the solver runs, and
watch velocity, pressure, vorticity, and vortex-core fields update live, with
measured drag and lift coefficients and a real convergence monitor.

Everything runs in Vulkan compute shaders. On an RTX 5070 Ti the solver sustains
**4161 to 4970 MLUPS** (million lattice updates per second) across grids from
3.9e5 to 2.6e7 cells, which is 71 to 84 percent of the card's peak memory
bandwidth under a 152 byte per cell traffic model. Throughput is independent of
the collision operator to within 1 percent, so the kernel is bandwidth-bound
rather than compute-bound.

v2 is a complete ground-up rewrite. Only the idea survives from v0/v1; every
line of engine, solver, and UI code is new.

**[Download the latest Windows build](https://github.com/SambuddhaRoy/LBM-CFD-Solver/releases/latest)**
(no install, unzip and run; needs a Vulkan 1.3 GPU driver). Or build from
source, below.

## Highlights

**Physics**
- D3Q19 LBM with **BGK**, **regularized**, and **TRT** collision operators
- **Smagorinsky LES** subgrid turbulence — effective relaxation time from the
  local non-equilibrium stress
- **Free-slip tunnel walls** (specular reflection) instead of periodic wrap —
  the wake can't re-enter the domain from the other side
- Equilibrium velocity inlet with optional perturbation; **pressure outlet**
  pinning rho = 1 (a floating zero-gradient outlet back-pressurises the domain
  and halves the effective Reynolds number); **Bouzidi interpolated
  bounce-back** at obstacles, so the wall sits at its true sub-cell position
  rather than on the voxel staircase
- **Measured, not modeled, diagnostics**: a single fused GPU reduction returns
  the L2 velocity residual, **momentum-exchange** drag/lift/side force, mass
  conservation, peak velocity, and kinetic energy every frame

**Aerodynamics workflow**
- Built-in analytic models — sphere, cube, spanwise cylinder, **NACA 0012 wing**
  — voxelized exactly, no mesh files needed
- **Pitch, yaw and roll sliders** rotate the body on all three axes and
  re-voxelize live — while the solver is running the existing flow field is
  kept, so the wake visibly reorganises instead of restarting from rest
- Mesh import (STL / OBJ / glTF / FBX / PLY) via Assimp with SAT voxelization
  and interior fill
- Reference area for C_D / C_L taken from the actual projected frontal area,
  with a **solid-blockage correction** applied (see Accuracy below)

**Physical unit scaling**
- Pick a working fluid (sea-level air, high-altitude air, water, Mars CO2),
  set the real wind speed and model size — the app derives dx, dt, the physical
  and lattice Reynolds numbers, and Mach, and warns when compressibility or
  resolution limits are hit
- Colorbar and field statistics are labeled in physical units (m/s)

**Engineering**
- `--headless` self-validates windowless (mass, residual decay, peak velocity
  relative to inlet, plausible C_D) and `--validate` runs the full Reynolds
  sweep against literature — both exit 0/1, ready for CI
- Frames-in-flight rendering with per-frame analysis readback slots (no
  CPU-GPU stalls, no readback races)
- Disk-backed pipeline cache; config persistence; BMP snapshot export
- Clean module split: `gpu` (context/swapchain), `sim` (solver), `mesh`
  (import/voxelize), `viz` (slice view), `ui` (panels), `app` (orchestration)

## Validation

Two tiers. `--headless` is a fast smoke test; `--validate` is the real one.

**`--validate`** runs a Reynolds sweep and asserts against published values —
including C_D, which is the number the tool exists to produce:

```
  CYLINDER — diameter 40 cells, blockage 0.12
    Re     tau    C_D     St      L_r/D    regime
     2   4.100  7.383      --      0.00    attached flow (no separation)
    40   0.680  1.589      --      2.15    steady recirculation (twin vortices)
   100   0.572  1.463   0.181      1.70    Von Karman vortex street
   150   0.548  1.502   0.192      1.40    Von Karman vortex street

  Literature: L_r/D ~ 2.1 at Re=40; St ~ 0.164 / 0.184 at Re=100 / 150;
              C_D ~ 1.50 / 1.35 / 1.33 at Re=40 / 100 / 150.
```

Recirculation length lands at 2.15 D against 2.1 published; Strouhal is 5-10%
high (blockage); C_D is within 6-13% and asserted to +/-25%. A square prism and
a **fully 3D sphere** checked against the Schiller-Naumann drag correlation
round it out — the sphere is the only case that exercises the same 3D code path
an imported model takes, since the 2D cases run at `gz = 6`.

**`--headless`** is the CI smoke test — exit code 0/1, thresholds stated
relative to the inlet speed rather than as loose absolutes:

```
  [PASS] all quantities finite
  [PASS] mass conserved (1.0447, want 1.00 +/- 0.05)
  [PASS] peak |u| = 1.30x inlet (want < 2.5x)
  [PASS] residual decayed (6.97e-01 -> 2.67e-01)
  [PASS] C_D plausible (0.754, want 0.05..8)
```

The `2.5x inlet` bound is calibrated, not guessed: healthy runs peak near 1.3x,
and a known-bad collision configuration peaks at 2.96x and is rejected.

## Building (Windows)

Prerequisites: Git, CMake 3.24+, Visual Studio 2022+ (C++ workload), and
[vcpkg](https://github.com/microsoft/vcpkg).

```powershell
git clone https://github.com/SambuddhaRoy/LBM-CFD-Solver.git
cd LBM-CFD-Solver

cmake -S . -B build `
    -DCMAKE_TOOLCHAIN_FILE="<vcpkg>/scripts/buildsystems/vcpkg.cmake" `
    -DVCPKG_TARGET_TRIPLET=x64-windows
cmake --build build --config Release
```

The first configure installs the dependencies (Vulkan loader, GLFW, ImGui,
Assimp, GLM, vk-bootstrap, VMA, shaderc) — this takes a while once.
The executable, compiled shaders, and DLLs land in `build/Release/`.

On Linux the same CMake flow works with `x64-linux`; GCC 13+ or Clang 17+.

## Controls

| Key | Action |
|---|---|
| `Space` | Run / pause |
| `R` | Reset flow |
| `1`–`4` | Velocity / Pressure / Vorticity / Q-criterion |
| `[` `]` | Move slice plane |
| `Tab` / `F` | Toggle left / right panel |
| `S` | Save snapshot (BMP) |
| `F11` | Fullscreen |
| `Esc` | Reset view / close help |
| Wheel / drag | Zoom / pan |

Drag-and-drop a mesh file anywhere on the window to load it.

## Architecture

```
src/
  main.cpp   CLI parsing, GUI/headless dispatch
  gpu.*      Vulkan context, swapchain, buffers, pipeline cache
  sim.*      D3Q19 solver, unit scaling, fused analysis reduction
  mesh.*     Assimp import, SAT voxelizer, analytic primitives
  viz.*      slice compute pass -> ImGui texture, snapshot readback
  ui.*       theme, panels, viewport, overlays
  app.*      frame loop, actions, config, headless validation
shaders/
  lbm.comp       collide-stream: BGK/regularized/TRT + Smagorinsky LES
  analysis.comp  residual + momentum-exchange forces + field stats, one pass
  slice.comp     4-mode field visualization, classic blue-to-red false colour
```

~5,000 lines of C++23 and GLSL. No engine middleware — Vulkan, GLFW, ImGui,
Assimp, GLM, VMA, vk-bootstrap via vcpkg.

## Accuracy — read this before quoting a number

This is a real-time solver. The flow structures it produces are trustworthy;
the absolute coefficients need caveats.

- **C_D is blockage-corrected.** A body of frontal area `A` in a tunnel of
  cross-section `C` accelerates the stream past it to roughly `U/(1-A/C)`, so
  normalising by the *inlet* dynamic pressure overstates C_D by `1/(1-beta)^2`
  — about +29% at 12% blockage, which was the bulk of this solver's former drag
  error. The continuity correction is now applied automatically and the
  blockage ratio is shown next to the forces.

  It is first order, and it is not equally right for every body. It takes the
  round-body cases from +36..+45% down to +6..+13%. For a sharp-edged prism,
  whose separation points are pinned at the corners rather than set by the
  local speed, it over-corrects: the square case reads 1.52 uncorrected against
  ~1.5 published and 1.22 corrected. Treat C_D on bluff, sharp-edged geometry as
  a lower bound.
- **The solver resolves the lattice Reynolds number, not the physical one.**
  Ask for a 1 m body at 30 m/s and the panel will report Re ~ 2x10^6 while the
  lattice is actually integrating Re ~ 10^3. Smagorinsky LES with no wall model
  does not bridge three orders of magnitude. Wake topology, trends, and
  comparisons between two shapes are meaningful; absolute forces at the
  physical Re are not. The UI says so explicitly when the gap exceeds 10x.
- **Resolution is the dominant remaining error, and it is measured.** On the
  sphere validation case at Re=100, against Schiller-Naumann:

  | cells across body | C_D error | wake length L/D (lit ~0.87) |
  |---|---|---|
  | 24 | +34% | 0.83 |
  | 32 | +13% | 0.88 |

  The wake geometry is already right at 24 cells; the *force* is what needs
  resolution. Want quantitative drag: keep at least 32 cells across the body —
  the app shows this figure and warns below it. The Coarse preset puts a
  default model at ~22 cells, Fine at ~35.
- Single precision throughout. No formal grid-convergence study ships, though
  the table above is a two-point version of one.

## Known limits

- No wall model or wall functions: boundary layers are unresolved at high Re
- Free-slip side walls, so the domain is a slip-walled duct rather than open air
- Single-phase, incompressible-regime flow (lattice Mach is clamped)
- No heat transfer, compressibility, or moving/deforming geometry

## License

MIT.
