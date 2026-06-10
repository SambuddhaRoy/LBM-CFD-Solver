<div align="center">

<img src="https://img.shields.io/badge/version-2.0--dev-1dd1a1?style=for-the-badge" />
<img src="https://img.shields.io/badge/C%2B%2B-23-00599C?style=for-the-badge&logo=c%2B%2B" />
<img src="https://img.shields.io/badge/Vulkan-1.3-AD1F1F?style=for-the-badge&logo=vulkan" />
<img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" />
<img src="https://img.shields.io/badge/Platform-Windows%20%7C%20Linux-0078D4?style=for-the-badge" />

<br /><br />

# Virtual Wind Tunnel v2

**Real-time GPU aerodynamics — a D3Q19 Lattice Boltzmann solver in Vulkan compute**

</div>

---

Drop in a model — or pick a built-in sphere, cylinder, cube, or NACA 0012 wing —
set the wind speed and angle of attack, and watch velocity, pressure, vorticity,
and vortex-core fields update live, with measured drag and lift coefficients and
a real convergence monitor. Everything runs in Vulkan compute shaders;
on a desktop GPU the solver sustains **thousands of MLUPS** (million lattice
updates per second).

This branch (`rewrite-v2`) is a complete ground-up rewrite. Only the idea
survives from v0/v1 — every line of engine, solver, and UI code is new.

## Highlights

**Physics**
- D3Q19 LBM with **BGK** and **regularized** collision operators
- **Smagorinsky LES** subgrid turbulence — effective relaxation time from the
  local non-equilibrium stress, enabling believable high-Re flow on coarse grids
- **Free-slip tunnel walls** (specular reflection) instead of periodic wrap —
  the wake can't re-enter the domain from the other side
- Equilibrium velocity inlet with optional spectrally-flat perturbation;
  zero-gradient outlet; halfway bounce-back obstacles
- **Measured, not modeled, diagnostics**: a single fused GPU reduction returns
  the L2 velocity residual, pressure drag/lift/side force, mass conservation,
  peak velocity, and kinetic energy every frame

**Aerodynamics workflow**
- Built-in analytic models — sphere, cube, spanwise cylinder, **NACA 0012 wing**
  — voxelized exactly, no mesh files needed
- **Angle-of-attack and yaw sliders**: the model re-voxelizes on release, so
  C_L vs alpha studies take seconds
- Mesh import (STL / OBJ / glTF / FBX / PLY) via Assimp with SAT voxelization
  and interior fill
- Reference area for C_D / C_L taken from the actual projected frontal area

**Physical unit scaling**
- Pick a working fluid (sea-level air, high-altitude air, water, Mars CO2),
  set the real wind speed and model size — the app derives dx, dt, the physical
  and lattice Reynolds numbers, and Mach, and warns when compressibility or
  resolution limits are hit
- Colorbar and field statistics are labeled in physical units (m/s)

**Engineering**
- `--headless` mode runs the solver windowless and **self-validates**
  (mass conservation, residual decay, bounded velocity, positive drag) —
  exit code 0/1, ready for CI
- Frames-in-flight rendering with per-frame analysis readback slots (no
  CPU-GPU stalls, no readback races)
- Disk-backed pipeline cache; config persistence; BMP snapshot export
- Clean module split: `gpu` (context/swapchain), `sim` (solver), `mesh`
  (import/voxelize), `viz` (slice view), `ui` (panels), `app` (orchestration)

## Validation

The headless mode doubles as a physics smoke test:

```
VirtualWindTunnel --headless --steps 4000 --shape sphere
```

```
    step      residual        C_D       C_L     mass   max|u|
     400    4.0178e-02     0.7642    0.0000   1.0934   0.0700
    1200    3.9506e-03     0.3102   -0.0000   1.0962   0.0700
    2800    2.9178e-04     0.3140   -0.0000   1.0962   0.0700
    4000    2.8610e-05     0.3136    0.0000   1.0962   0.0700

  4000 steps in 0.40 s  ->  3919 MLUPS on NVIDIA GeForce RTX 5070 Ti
  VALIDATION PASSED
```

The residual decays three orders of magnitude monotonically, C_L vanishes by
symmetry, mass is conserved at steady state, and the wing produces
C_L = +0.50 at 8 degrees angle of attack with the correct sign.

## Building (Windows)

Prerequisites: Git, CMake 3.24+, Visual Studio 2022+ (C++ workload), and
[vcpkg](https://github.com/microsoft/vcpkg).

```powershell
git clone https://github.com/SambuddhaRoy/LBM-CFD-Solver.git
cd LBM-CFD-Solver
git checkout rewrite-v2

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
  lbm.comp       collide-stream: BGK/regularized + Smagorinsky LES
  analysis.comp  residual + forces + field stats in one pass
  slice.comp     4-mode field visualization with physical colormaps
```

~4,300 lines of C++23 and GLSL. No engine middleware — Vulkan, GLFW, ImGui,
Assimp, GLM, VMA, vk-bootstrap via vcpkg.

## Known limits

- Drag/lift are the **pressure component** integrated over the voxel surface;
  viscous (friction) drag is not yet included — a momentum-exchange surface
  integral is the planned upgrade
- Voxel staircasing limits force accuracy on coarse grids; use the Fine preset
  for quantitative comparisons
- Single-phase, incompressible-regime flow (lattice Mach is clamped)

## License

MIT.
