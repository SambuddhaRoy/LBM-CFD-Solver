<div align="center">

<img src="https://img.shields.io/badge/version-0.1.0--dev-1dd1a1?style=for-the-badge" />
<img src="https://img.shields.io/badge/C%2B%2B-23-00599C?style=for-the-badge&logo=c%2B%2B" />
<img src="https://img.shields.io/badge/Vulkan-1.3-AD1F1F?style=for-the-badge&logo=vulkan" />
<img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" />
<img src="https://img.shields.io/badge/Platform-Windows-0078D4?style=for-the-badge&logo=windows" />

<br /><br />

# Virtual Wind Tunnel

**Real-time GPU-accelerated aerodynamic simulation using the Lattice Boltzmann Method**

[Report a Bug](https://github.com/SambuddhaRoy/VirtualWindTunnel/issues) · [Request a Feature](https://github.com/SambuddhaRoy/VirtualWindTunnel/issues) · [Discussions](https://github.com/SambuddhaRoy/VirtualWindTunnel/discussions)

</div>

---

Virtual Wind Tunnel runs a full **D3Q19 Lattice Boltzmann simulation** on the GPU in real time. Drop in any STL, OBJ, or FBX model, set your flow conditions, and watch velocity, pressure, vorticity, and Q-criterion fields update live at interactive frame rates — no cloud compute, no preprocessing pipeline, no waiting.

The entire solver and visualization pipeline lives in Vulkan compute shaders. The UI is designed to feel like a professional CFD tool rather than a research prototype.

<br />

## Screenshots

> *Coming soon — run the build and submit a screenshot PR!*

<br />

## What's new in v0.1.0-dev

This release is a complete rewrite of the engine and UI. The main branch (`main`) contains the stable v0.0.2-alpha. All active development is on the `rewrite-windows` branch.

**Engine & Performance**
- **Frames-in-flight (2)** — each frame has its own command pool, buffers, and sync primitives, eliminating CPU-GPU stalls and enabling true double-buffering
- **Async compute queue** — fluid solver dispatches on a dedicated compute-only queue family when available; falls back gracefully to the graphics queue
- **Disk-backed pipeline cache** — `pipeline_cache.bin` is loaded at startup and saved on exit, eliminating 100–300 ms shader recompilation on every subsequent launch
- **SoA buffer layout** — distribution functions stored as Structure-of-Arrays (`f[q * totalCells + cell]`) for coalesced memory access during the streaming step
- **GPU timestamp pool** — 4-slot query pool measures actual LBM and aero force dispatch times in milliseconds, shown live in the UI
- **VMA modernisation** — staging buffers use `AUTO_PREFER_HOST + HOST_ACCESS_SEQUENTIAL_WRITE + MAPPED_BIT` (persistent map, zero map/unmap overhead); GPU buffers use `AUTO_PREFER_DEVICE`
- **Descriptor sets written once** — removed the per-frame `vkUpdateDescriptorSets` call that was running on every rendered frame
- **Config persistence** — `vwt_config.ini` saves all solver and UI settings on exit and restores them on next launch

**Solver**
- **BGK and MRT-Regularized collision operators** — switch between fast BGK and the more numerically stable MRT-RLB scheme in the UI
- **Real aerodynamic force integration** — new `aero_forces.comp` shader performs a parallel GPU reduction over obstacle surface cells, accumulating pressure forces into 256 partial sums then summing on the CPU to produce actual C_D, C_L, and L/D readouts
- **Zou-He inlet boundary** — proper velocity inlet at x=0 using the Zou-He scheme; zero-gradient outflow at the outlet
- **Periodic Y/Z boundaries** — correct periodicity for wind tunnel walls
- **Hash-based turbulence injection** — replaced the old sinusoidal noise (which created visible spatial stripes) with a hash function for spectrally flat perturbations
- **Mach limiter** — lattice velocity clamped to |u| < 0.45 c_s to prevent blow-up at high inlet speeds

**Visualization**
- **4 visualization modes** — Velocity magnitude (inferno), Pressure/density (cool-warm diverging), Vorticity magnitude (viridis), Q-criterion (inferno on Q > 0, black on Q ≤ 0 for vortex core identification)
- **Colorbar in the viewport** — moved from the right panel into the scene; gradient updates automatically with vis mode; 5 tick marks with physical units
- **Mouse-wheel zoom + drag-to-pan** — interactive viewport navigation

**UI redesign**
- **Icon rail** — 56px left rail for switching between Simulation, Mesh, Probe, and Compare modes (future panels)
- **Card-based panels** — every section is a bordered card with a coloured left-edge accent. Geometry (teal), Flow (blue), Environment (purple), Solver (amber), Aerodynamics (teal), Convergence (amber), GPU (purple)
- **Big-number metrics** — C_D and C_L rendered at 1.4× font scale with live delta percentage badges (`+2.1%`) showing trend vs the previous sample
- **Vis-mode tabs with keyboard hints** — "Velocity 1", "Pressure 2" etc., always readable without hovering
- **Live/FPS HUD pills** — floating over the viewport, drawn on the scene's ImDrawList; the Live pill pulses while the simulation runs
- **Environment as a card grid** — Earth, Mars, Venus, Titan, Underwater as one-click cards instead of a hidden dropdown
- **Toggle groups** — unit selector (m/s / km/h / mph / kn), Subsonic/Supersonic, BGK/MRT, slice axis — all rendered as segmented button groups
- **SliderPill controls** — every slider shows its value in a bordered pill on the same row
- **Animated status bar** — state dot pulses when running; shows live τ, Δt, step count, vis mode, GPU name, and Vulkan version

<br />

## Feature overview

| Category | Feature | Status |
|---|---|---|
| **Solver** | D3Q19 LBM (BGK) | ✅ |
| | D3Q19 LBM (MRT-Regularized) | ✅ |
| | Zou-He velocity inlet | ✅ |
| | Zero-gradient outlet | ✅ |
| | Periodic Y/Z walls | ✅ |
| | Hash-based turbulence injection | ✅ |
| | Mach stability limiter | ✅ |
| **Forces** | GPU pressure-force integration | ✅ |
| | C_D, C_L, L/D readout | ✅ |
| | Delta indicators vs previous sample | ✅ |
| **Visualization** | Velocity magnitude (inferno) | ✅ |
| | Pressure / density (cool-warm) | ✅ |
| | Vorticity magnitude (viridis) | ✅ |
| | Q-criterion (vortex cores) | ✅ |
| | In-viewport colorbar with units | ✅ |
| | Mouse zoom / pan | ✅ |
| **Mesh** | STL import + voxelization | ✅ |
| | OBJ / FBX / glTF import | ✅ |
| | Drag-and-drop loading | ✅ |
| **Environments** | Earth, Mars, Venus, Titan, Water | ✅ |
| **Performance** | Async compute queue | ✅ |
| | Frames-in-flight (2) | ✅ |
| | Disk-backed pipeline cache | ✅ |
| | GPU timestamp readback | ✅ |
| | VRAM budget monitoring | ✅ |
| **UI** | Icon rail navigation | ✅ |
| | Card-based panel layout | ✅ |
| | Config persistence (.ini) | ✅ |
| | Keyboard shortcuts | ✅ |

<br />

## Keyboard shortcuts

| Key | Action |
|---|---|
| `Space` | Pause / Resume simulation |
| `R` | Reset simulation to equilibrium |
| `1` | Velocity magnitude view |
| `2` | Pressure / density view |
| `3` | Vorticity magnitude view |
| `4` | Q-criterion view |
| `+` / `-` | Increase / decrease steps per frame |
| `F11` | Toggle fullscreen |
| `Esc` | Reset viewport zoom and pan |
| Scroll wheel | Zoom viewport |
| Left drag | Pan viewport |

<br />

## Building from source (Windows)

### Prerequisites

| Tool | Minimum version | Download |
|---|---|---|
| **Git** | any | https://git-scm.com |
| **CMake** | 3.24 | https://cmake.org/download |
| **Visual Studio 2022** | 17.x with *Desktop development with C++* workload | https://visualstudio.microsoft.com |
| **Vulkan SDK** | 1.3 | https://vulkan.lunarg.com/sdk/home |

The Vulkan SDK installer sets the `VULKAN_SDK` environment variable automatically. Make sure CMake and Git are on your `PATH`.

### One-shot PowerShell build

Open **PowerShell** (no admin required) and paste this block:

```powershell
$buildRoot = "$env:USERPROFILE\Documents\VirtualWindTunnel-build"
New-Item -ItemType Directory -Force -Path $buildRoot | Out-Null

# Clone repo
git clone https://github.com/SambuddhaRoy/VirtualWindTunnel.git "$buildRoot\VirtualWindTunnel"

# Bootstrap vcpkg
git clone https://github.com/microsoft/vcpkg.git "$buildRoot\vcpkg"
& "$buildRoot\vcpkg\bootstrap-vcpkg.bat" -disableMetrics

# Install dependencies  (first run takes 10–20 min)
Set-Location "$buildRoot\VirtualWindTunnel"
& "$buildRoot\vcpkg\vcpkg.exe" install --triplet x64-windows

# Configure
cmake -S "$buildRoot\VirtualWindTunnel" -B "$buildRoot\cmake-build-release" `
    -DCMAKE_BUILD_TYPE=Release `
    -DCMAKE_TOOLCHAIN_FILE="$buildRoot\vcpkg\scripts\buildsystems\vcpkg.cmake" `
    -DVCPKG_TARGET_TRIPLET=x64-windows `
    -DCMAKE_INSTALL_PREFIX="$buildRoot\install"

# Build
cmake --build "$buildRoot\cmake-build-release" --config Release --parallel

# Install
cmake --install "$buildRoot\cmake-build-release" --config Release

Write-Host "`nDone. EXE at: $buildRoot\install" -ForegroundColor Green
```

The output is in `Documents\VirtualWindTunnel-build\install\` — everything needed to run is there including the compiled shaders and all DLLs.

### Building the rewrite branch

To build the latest development code:

```powershell
Set-Location "$env:USERPROFILE\Documents\VirtualWindTunnel-build\VirtualWindTunnel"
git fetch origin
git checkout rewrite-windows
```

Then re-run the cmake configure and build steps above.

### Switching branches

```powershell
git checkout main            # stable v0.0.2-alpha
git checkout rewrite-windows # active development
```

<br />

## Running

1. Launch `VirtualWindTunnel.exe` from the install folder
2. Drop an STL, OBJ, or FBX file onto the window — or click **Browse Model** in the Geometry card
3. Adjust inlet velocity, turbulence, and solver settings in the left panel
4. Click **▶ Run** or press `Space`
5. Switch visualization modes with `1`–`4` or the tab bar above the viewport

The first launch after a build compiles the SPIR-V shaders and writes `pipeline_cache.bin`. Every subsequent launch skips recompilation.

<br />

## Project structure

```
VirtualWindTunnel/
├── src/
│   ├── vk_engine.cpp/h     — Vulkan engine, UI panels, frame loop
│   ├── fluid_solver.cpp/h  — D3Q19 LBM + aero force integration
│   ├── renderer.cpp/h      — Visualization compute pass + ImGui texture
│   ├── mesh_loader.cpp/h   — Assimp import + SAT voxelization
│   ├── vk_types.h          — Shared structs, VK_CHECK, DeletionQueue
│   ├── environment.h       — Fluid environment profiles (Earth, Mars, ...)
│   ├── sim_scaler.h        — Lattice ↔ physical unit conversion
│   └── logger.h            — Simple file + console logger
├── shaders/
│   ├── fluid_lbm.comp      — D3Q19 collision-and-stream (BGK + MRT)
│   ├── velocity_slice.comp — 4-mode visualization (velocity/pressure/vorticity/Q)
│   └── aero_forces.comp    — Parallel pressure-force integration
├── CMakeLists.txt
└── vcpkg.json
```

<br />

## Technology stack

| Component | Library / API |
|---|---|
| Language | C++23 |
| Graphics & Compute | Vulkan 1.3 |
| Window & Input | GLFW 3 |
| UI | Dear ImGui (Vulkan + GLFW backend) |
| Memory | Vulkan Memory Allocator (VMA) |
| Device selection | vk-bootstrap |
| Mesh import | Assimp |
| Math | GLM |
| Dependencies | vcpkg |
| Shaders | GLSL compiled to SPIR-V via glslc |

<br />

## What's being worked on

These are actively in development on the `rewrite-windows` branch and will land in the next tagged release:

**Solver**
- Smagorinsky subgrid-scale turbulence model for Large Eddy Simulation (LES) quality flow at coarser resolutions
- Thermal LBM extension — temperature field coupled to the velocity solver for heat transfer and buoyancy-driven flow
- Higher-order boundary conditions at curved surfaces (interpolated bounce-back)
- Multi-resolution refinement zones — finer lattice in wake regions without scaling the whole domain

**Visualization**
- Streamline seeding and integration on the GPU — user-placeable seed lines with real-time Runge-Kutta integration
- Surface pressure coefficient (C_p) map rendered directly on the mesh geometry
- Time-averaged statistics accumulation — mean velocity, RMS fluctuations, TKE
- Snapshot export to PNG and EXR with full float precision

**UI**
- Probe tool (Mesh panel) — click any point in the viewport to read local velocity, pressure, and vorticity values
- Compare mode — run two configurations side by side with synchronized controls and a difference view
- Simulation recording and playback — save the macro field history and scrub through it offline
- Settings panel — colormap selection, font size, UI scale, keyboard binding remapping

**Build & platform**
- Native Linux build (no Wine) — Wayland and X11 targets via CMake option
- Automated GitHub Actions CI for Windows builds with artifact upload
- Pre-built release packages with bundled Vulkan loader for users without an SDK installed

<br />

## Contributing

Issues, pull requests, and discussions are all welcome.

For large changes — new solver features, UI overhauls, new visualization modes — open an issue first so we can align on the design before writing code.

<br />

## License

MIT. See [`LICENSE`](LICENSE) for the full text.

<div align="center">
<br />
Built for the engineering community.
</div>
