// ============================================================================
// ui.cpp — theme, panels, viewport, overlays
//
// Layout:  top bar / left "Setup" panel / viewport / right "Results" panel /
//          status bar, plus floating tool windows over the viewport.
//
// Two rules keep this UI honest:
//   1. Interactive controls that float over the viewport live in their OWN
//      ImGui windows. ImGui resolves hover per-window, so the viewport's
//      pan/zoom drag surface can never swallow their clicks. (Putting them in
//      the viewport window is exactly what broke every overlay button before:
//      a full-size InvisibleButton submitted first takes ActiveId on
//      mouse-down, and ItemHoverable() then rejects every later item.)
//   2. Nothing is positioned with hand-rolled absolute-x arithmetic. Widths
//      are measured, then laid out — so nothing overlaps at any DPI or size.
// ============================================================================

#include "ui.h"
#include "app.h"

#include <imgui.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <fstream>

namespace vwt::ui {

namespace {

// ─── Design tokens ───────────────────────────────────────────────────────────
float gScale = 1.f;
inline float S(float v) { return v * gScale; }

const ImVec4 kBg0    {0.043f, 0.047f, 0.062f, 1.f};   // app background
const ImVec4 kBg1    {0.067f, 0.073f, 0.094f, 1.f};   // panel
const ImVec4 kBg2    {0.090f, 0.098f, 0.125f, 1.f};   // widget
const ImVec4 kStroke {0.137f, 0.149f, 0.188f, 1.f};
const ImVec4 kText   {0.910f, 0.918f, 0.949f, 1.f};
const ImVec4 kDim    {0.541f, 0.569f, 0.639f, 1.f};
const ImVec4 kAccent {0.098f, 0.784f, 0.651f, 1.f};   // teal
const ImVec4 kBlue   {0.302f, 0.624f, 1.000f, 1.f};
const ImVec4 kAmber  {1.000f, 0.706f, 0.329f, 1.f};
const ImVec4 kRed    {1.000f, 0.365f, 0.365f, 1.f};
const ImVec4 kPurple {0.616f, 0.482f, 1.000f, 1.f};
const ImVec4 kGreen  {0.345f, 0.875f, 0.522f, 1.f};

ImU32 u32(const ImVec4& c, float aMul = 1.f) {
    return ImGui::ColorConvertFloat4ToU32({c.x, c.y, c.z, c.w * aMul});
}
ImVec4 fade(const ImVec4& c, float a) { return {c.x, c.y, c.z, a}; }

// ─── CPU colormaps (mirror slice.comp, for the colorbar) ────────────────────
ImVec4 cmJet(float t) {
    t = std::clamp(t, 0.f, 1.f);
    auto ch = [](float v) { return std::clamp(v, 0.f, 1.f); };
    return { ch(1.5f - std::fabs(4.f*t - 3.f)),
             ch(1.5f - std::fabs(4.f*t - 2.f)),
             ch(1.5f - std::fabs(4.f*t - 1.f)), 1.f };
}
ImVec4 cmCoolwarm(float t) {
    t = std::clamp(t, 0.f, 1.f);
    const ImVec4 cold{0.230f,0.299f,0.754f,1.f}, mid{0.865f,0.865f,0.865f,1.f},
                 warm{0.706f,0.016f,0.150f,1.f};
    auto mix = [](const ImVec4& a, const ImVec4& b, float u) {
        return ImVec4{a.x+(b.x-a.x)*u, a.y+(b.y-a.y)*u, a.z+(b.z-a.z)*u, 1.f};
    };
    return (t < 0.5f) ? mix(cold, mid, t*2.f) : mix(mid, warm, t*2.f-1.f);
}

// ─── Widgets ─────────────────────────────────────────────────────────────────

float buttonW(const char* label) {
    return ImGui::CalcTextSize(label).x + ImGui::GetStyle().FramePadding.x * 2.f;
}

void pushFontBig(App& app) {
    if (app.fonts.big) ImGui::PushFont(app.fonts.big, app.fonts.big->LegacySize);
}
void popFontBig(App& app) { if (app.fonts.big) ImGui::PopFont(); }

// Section heading: a coloured tick plus a title, then the body.
void SectionHeader(const char* title, ImVec4 accent) {
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const ImVec2 p = ImGui::GetCursorScreenPos();
    dl->AddRectFilled({p.x, p.y + S(3)}, {p.x + S(3), p.y + S(15)}, u32(accent), S(1.5f));
    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + S(10));
    ImGui::TextUnformatted(title);
    ImGui::Spacing();
}

void SectionEnd() {
    ImGui::Spacing();
    ImGui::PushStyleColor(ImGuiCol_Separator, kStroke);
    ImGui::Separator();
    ImGui::PopStyleColor();
    ImGui::Spacing();
}

// Dim caption above a full-width control, with an optional tooltip.
void FieldLabel(const char* text, const char* tip = nullptr) {
    ImGui::PushStyleColor(ImGuiCol_Text, kDim);
    ImGui::TextUnformatted(text);
    ImGui::PopStyleColor();
    if (tip && ImGui::IsItemHovered()) ImGui::SetTooltip("%s", tip);
    ImGui::SetNextItemWidth(-FLT_MIN);
}

void StatRow(const char* label, const char* fmt, ...) {
    char buf[96];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);
    ImGui::PushStyleColor(ImGuiCol_Text, kDim);
    ImGui::TextUnformatted(label);
    ImGui::PopStyleColor();
    const float avail = ImGui::GetContentRegionAvail().x;
    const float tw    = ImGui::CalcTextSize(buf).x;
    if (avail > tw) {
        ImGui::SameLine(ImGui::GetCursorPosX() + avail - tw);
    } else {
        ImGui::SameLine();
    }
    ImGui::TextUnformatted(buf);
}

// Equal-width exclusive choice. Returns true when the selection changed.
bool Segmented(const char* id, int* value, const char* const* labels, int count,
               ImVec4 accent) {
    bool changed = false;
    ImGui::PushID(id);
    const float spacing = ImGui::GetStyle().ItemSpacing.x;
    const float w = (ImGui::GetContentRegionAvail().x - spacing * float(count - 1))
                    / float(count);
    for (int i = 0; i < count; ++i) {
        if (i) ImGui::SameLine();
        const bool active = (*value == i);
        ImGui::PushStyleColor(ImGuiCol_Button, active ? fade(accent, 0.22f) : kBg2);
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, fade(accent, 0.30f));
        ImGui::PushStyleColor(ImGuiCol_Text, active ? accent : kDim);
        if (ImGui::Button(labels[i], {w, 0}) && *value != i) { *value = i; changed = true; }
        ImGui::PopStyleColor(3);
    }
    ImGui::PopID();
    return changed;
}

// A toggle styled like Segmented but standalone (used for overlay toolbars).
bool Pill(const char* label, bool active, ImVec4 accent, float w = 0.f) {
    ImGui::PushStyleColor(ImGuiCol_Button, active ? fade(accent, 0.25f) : fade(kBg1, 0.90f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, fade(accent, 0.34f));
    ImGui::PushStyleColor(ImGuiCol_Text, active ? accent : kDim);
    const bool hit = ImGui::Button(label, {w, 0});
    ImGui::PopStyleColor(3);
    return hit;
}

void BigValue(App& app, const char* label, float value, const char* fmt,
              float delta, ImVec4 accent) {
    ImGui::PushStyleColor(ImGuiCol_Text, kDim);
    ImGui::TextUnformatted(label);
    ImGui::PopStyleColor();

    char buf[32];
    std::snprintf(buf, sizeof(buf), fmt, value);
    pushFontBig(app);
    ImGui::PushStyleColor(ImGuiCol_Text, accent);
    ImGui::TextUnformatted(buf);
    ImGui::PopStyleColor();
    popFontBig(app);

    if (std::abs(delta) > 1e-6f) {
        char db[24];
        std::snprintf(db, sizeof(db), "%+.2f%%", delta);
        ImGui::SameLine();
        ImGui::PushStyleColor(ImGuiCol_Text, delta > 0 ? kRed : kGreen);
        ImGui::TextUnformatted(db);
        ImGui::PopStyleColor();
    }
}

void Sparkline(const char* id, const float* data, int count, int offset,
               float height, ImVec4 color) {
    ImGui::PushStyleColor(ImGuiCol_PlotLines, color);
    ImGui::PushStyleColor(ImGuiCol_FrameBg, kBg0);
    ImGui::PlotLines(id, data, count, offset, nullptr,
                     FLT_MAX, FLT_MAX, {ImGui::GetContentRegionAvail().x, height});
    ImGui::PopStyleColor(2);
}

// ─── Layout ──────────────────────────────────────────────────────────────────

// Panels scroll: NoDecoration would imply NoScrollbar and make tall content
// unreachable on short windows, so the flags are spelled out individually.
constexpr ImGuiWindowFlags kPanelFlags =
    ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
    ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoMove |
    ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoBringToFrontOnFocus;

constexpr ImGuiWindowFlags kBarFlags = kPanelFlags | ImGuiWindowFlags_NoScrollbar |
                                       ImGuiWindowFlags_NoScrollWithMouse;

// Floating tool windows sit above the viewport: no NoBringToFrontOnFocus, so
// they keep hover priority over the viewport's drag surface.
constexpr ImGuiWindowFlags kOverlayFlags =
    ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
    ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoMove |
    ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoScrollbar |
    ImGuiWindowFlags_NoScrollWithMouse | ImGuiWindowFlags_AlwaysAutoResize |
    ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNavInputs;

struct Layout {
    float  topH, statusH, leftW, rightW;
    ImVec2 display, vpPos, vpSize;
};

Layout computeLayout(const App& app) {
    Layout l;
    l.display = ImGui::GetIO().DisplaySize;
    l.topH    = S(48);
    l.statusH = S(28);
    l.leftW   = app.leftOpen  ? std::min(S(320), l.display.x * 0.28f) : 0.f;
    l.rightW  = app.rightOpen ? std::min(S(330), l.display.x * 0.28f) : 0.f;
    l.vpPos   = { l.leftW, l.topH };
    l.vpSize  = { std::max(l.display.x - l.leftW - l.rightW, S(80)),
                  std::max(l.display.y - l.topH - l.statusH, S(80)) };
    return l;
}

// ─── Top bar ─────────────────────────────────────────────────────────────────

void drawTopBar(App& app, const Layout& l) {
    ImGui::SetNextWindowPos({0, 0});
    ImGui::SetNextWindowSize({l.display.x, l.topH});
    ImGui::PushStyleColor(ImGuiCol_WindowBg, kBg1);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {S(12), S(9)});
    ImGui::Begin("##top", nullptr, kBarFlags);

    // Brand mark
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const ImVec2 p = ImGui::GetCursorScreenPos();
    dl->AddCircleFilled({p.x + S(9), p.y + S(13)}, S(7), u32(kAccent));
    dl->AddCircleFilled({p.x + S(9), p.y + S(13)}, S(3.2f), u32(kBg1));
    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + S(26));
    ImGui::AlignTextToFramePadding();
    ImGui::TextUnformatted("Virtual Wind Tunnel");

    ImGui::SameLine(0, S(20));

    // ── Model menu ──
    if (ImGui::Button("Model")) ImGui::OpenPopup("##loadmenu");
    ImGui::SetItemTooltip("Load a built-in shape or import a mesh file");
    if (ImGui::BeginPopup("##loadmenu")) {
        if (ImGui::MenuItem("Sphere"))         app.loadPrimitive(Shape::Sphere);
        if (ImGui::MenuItem("Cube"))           app.loadPrimitive(Shape::Cube);
        if (ImGui::MenuItem("Cylinder"))       app.loadPrimitive(Shape::Cylinder);
        if (ImGui::MenuItem("NACA 0012 wing")) app.loadPrimitive(Shape::Wing);
        ImGui::Separator();
        if (ImGui::MenuItem("Open file...")) app.openFileDialog();
        ImGui::EndPopup();
    }

    ImGui::SameLine(0, S(14));

    // ── Transport ──
    const bool canRun = app.modelLoaded;
    ImGui::BeginDisabled(!canRun);
    {
        const ImVec4 c = app.running ? kAmber : kAccent;
        ImGui::PushStyleColor(ImGuiCol_Button, fade(c, 0.25f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, fade(c, 0.36f));
        ImGui::PushStyleColor(ImGuiCol_Text, c);
        if (ImGui::Button(app.running ? "Pause" : "Run", {S(74), 0}))
            app.running = !app.running;
        ImGui::PopStyleColor(3);
        ImGui::SetItemTooltip("Space");
    }
    ImGui::SameLine();
    if (ImGui::Button("Step")) { app.running = false; app.runSingleBatch = true; }
    ImGui::SetItemTooltip("Advance one batch, then pause");
    ImGui::SameLine();
    if (ImGui::Button("Reset")) app.resetSim();
    ImGui::SetItemTooltip("Restart the flow from rest  (R)");
    ImGui::EndDisabled();

    // ── Right-aligned group ──
    // Measured and placed entirely in window-local coordinates: mixing
    // DisplaySize with SameLine() offsets (which are relative to the content
    // region, not the display) pushes the whole group off the right edge.
    const ImGuiStyle& st = ImGui::GetStyle();
    const float sp = st.ItemSpacing.x;

    char perfBuf[64];
    std::snprintf(perfBuf, sizeof(perfBuf), "%.0f fps   %.0f MLUPS", app.fps, app.mlups);
    const char* gpuName = app.gpu.gpuName();

    float need = ImGui::CalcTextSize(perfBuf).x + sp
               + buttonW("Snapshot") + sp
               + buttonW("Help") + sp
               + buttonW("Panels");
    const float gpuW   = ImGui::CalcTextSize(gpuName).x + sp * 2.f;

    ImGui::SameLine();
    const float cursorX  = ImGui::GetCursorPosX();
    const float rightEdge = cursorX + ImGui::GetContentRegionAvail().x;
    const bool  showGpu  = (rightEdge - cursorX) > (need + gpuW + sp * 4.f);
    if (showGpu) need += gpuW;
    if (rightEdge - need > cursorX) ImGui::SetCursorPosX(rightEdge - need);

    if (showGpu) {
        ImGui::PushStyleColor(ImGuiCol_Text, kDim);
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted(gpuName);
        ImGui::PopStyleColor();
        ImGui::SameLine(0, sp * 2.f);
    }
    ImGui::PushStyleColor(ImGuiCol_Text, kAccent);
    ImGui::AlignTextToFramePadding();
    ImGui::TextUnformatted(perfBuf);
    ImGui::PopStyleColor();
    ImGui::SameLine(0, sp * 2.f);

    ImGui::BeginDisabled(!app.modelLoaded);
    if (ImGui::Button("Snapshot")) app.snapshot();
    ImGui::EndDisabled();
    ImGui::SetItemTooltip("Save a PNG-style BMP of the current view  (S)");

    ImGui::SameLine();
    if (ImGui::Button("Help")) app.showHelp = !app.showHelp;
    ImGui::SetItemTooltip("Keyboard shortcuts  (?)");

    ImGui::SameLine();
    if (ImGui::Button("Panels")) ImGui::OpenPopup("##panelmenu");
    ImGui::SetItemTooltip("Show or hide the side panels");
    if (ImGui::BeginPopup("##panelmenu")) {
        ImGui::MenuItem("Setup panel",   "Tab", &app.leftOpen);
        ImGui::MenuItem("Results panel", "F",   &app.rightOpen);
        ImGui::Separator();
        if (ImGui::MenuItem("Fullscreen", "F11")) app.toggleFullscreen();
        ImGui::EndPopup();
    }

    ImGui::End();
    ImGui::PopStyleVar();
    ImGui::PopStyleColor();
}

// ─── Left panel: Setup ───────────────────────────────────────────────────────

void drawLeftPanel(App& app, const Layout& l) {
    if (!app.leftOpen) return;
    ImGui::SetNextWindowPos({0, l.topH});
    ImGui::SetNextWindowSize({l.leftW, l.display.y - l.topH - l.statusH});
    ImGui::PushStyleColor(ImGuiCol_WindowBg, kBg1);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {S(14), S(12)});
    ImGui::Begin("##left", nullptr, kPanelFlags);

    // ── Model ───────────────────────────────────────────────────────────────
    SectionHeader("Model", kAccent);
    if (app.modelLoaded) {
        StatRow("Name", "%s", app.model.name.c_str());
        if (app.model.triCount > 0) StatRow("Triangles", "%u", app.model.triCount);
        StatRow("Fill", "%.2f%% of domain", app.model.fillPct);
        StatRow("Frontal area", "%u cells", app.model.frontalCells);
    } else {
        ImGui::PushStyleColor(ImGuiCol_Text, kDim);
        ImGui::TextWrapped("No model loaded. Pick one from Model in the top bar, "
                           "or drop an STL / OBJ / glTF file onto the window.");
        ImGui::PopStyleColor();
    }
    SectionEnd();

    // ── Orientation ─────────────────────────────────────────────────────────
    SectionHeader("Orientation", kGreen);
    ImGui::BeginDisabled(!app.modelLoaded);
    {
        ImGui::PushStyleColor(ImGuiCol_Text, kDim);
        ImGui::TextWrapped("Drag while running — the wake reorganises live.");
        ImGui::PopStyleColor();
        ImGui::Spacing();

        // Live re-voxelisation, keeping the flow field (see App::uploadModel).
        FieldLabel("Pitch  /  angle of attack", "Rotation about Z, nose up positive");
        if (ImGui::SliderFloat("##aoa", &app.aoaDeg, -90.f, 90.f, "%.1f deg"))
            app.revoxelize(true);

        FieldLabel("Yaw  /  side to side", "Rotation about the vertical Y axis");
        if (ImGui::SliderFloat("##yaw", &app.yawDeg, -180.f, 180.f, "%.1f deg"))
            app.revoxelize(true);

        FieldLabel("Roll  /  about the flow axis", "Rotation about the streamwise X axis");
        if (ImGui::SliderFloat("##roll", &app.rollDeg, -180.f, 180.f, "%.1f deg"))
            app.revoxelize(true);

        ImGui::Spacing();
        const bool rotated = (app.aoaDeg != 0.f || app.yawDeg != 0.f || app.rollDeg != 0.f);
        ImGui::BeginDisabled(!rotated);
        if (ImGui::Button("Reset orientation", {-FLT_MIN, 0})) {
            app.aoaDeg = app.yawDeg = app.rollDeg = 0.f;
            app.revoxelize(true);
        }
        ImGui::EndDisabled();
    }
    ImGui::EndDisabled();
    SectionEnd();

    // ── Flow ────────────────────────────────────────────────────────────────
    SectionHeader("Flow", kBlue);
    FieldLabel("Wind speed");
    if (ImGui::SliderFloat("##uphys", &app.units.physSpeed, 1.f, 150.f, "%.1f m/s"))
        app.units.compute(app.model.spanCellsX, app.params);

    FieldLabel("Fluid");
    if (ImGui::BeginCombo("##fluid", kFluids[app.units.fluidIdx].name)) {
        for (int i = 0; i < kFluidCount; ++i)
            if (ImGui::Selectable(kFluids[i].name, i == app.units.fluidIdx)) {
                app.units.fluidIdx = i;
                app.units.compute(app.model.spanCellsX, app.params);
            }
        ImGui::EndCombo();
    }

    FieldLabel("Inlet turbulence", "Random perturbation added at the inlet");
    float turbPct = app.params.turb * 100.f;
    if (ImGui::SliderFloat("##turb", &turbPct, 0.f, 30.f, "%.0f%%"))
        app.params.turb = turbPct / 100.f;
    SectionEnd();

    // ── Solver ──────────────────────────────────────────────────────────────
    SectionHeader("Solver", kAmber);
    FieldLabel("Collision operator",
               "BGK is fastest; Regularized is more stable; "
               "TRT fixes the wall position independently of viscosity");
    {
        const char* ops[] = { "BGK", "Regular.", "TRT" };
        Segmented("##collision", &app.params.collision, ops, 3, kAmber);
    }
    ImGui::Spacing();

    ImGui::Checkbox("LES turbulence model", &app.params.les);
    ImGui::SetItemTooltip("Smagorinsky sub-grid model for under-resolved turbulence");
    if (app.params.les) {
        FieldLabel("Smagorinsky constant");
        ImGui::SliderFloat("##cs", &app.params.csSmago, 0.06f, 0.24f, "Cs = %.2f");
    }

    FieldLabel("Relaxation time (tau)", "Sets viscosity: nu = (tau - 1/2) / 3");
    if (ImGui::SliderFloat("##tau", &app.params.tau, 0.505f, 0.9f, "%.3f"))
        app.units.compute(app.model.spanCellsX, app.params);

    FieldLabel("Steps per frame", "More steps = faster simulated time, lower frame rate");
    ImGui::SliderInt("##spf", &app.stepsPerFrame, 1, 40);

    FieldLabel("Grid resolution", "Changing this rebuilds the solver and restarts the flow");
    if (ImGui::BeginCombo("##grid", kGrids[app.gridPreset].name)) {
        for (int i = 0; i < kGridCount; ++i) {
            const double gb = double(gridBytes(kGrids[i])) / (1024.0*1024.0*1024.0);
            // Don't offer a grid the card cannot hold — the allocation would
            // throw rather than fail gracefully.
            const bool fits = (app.vramBudget == 0) ||
                              (gridBytes(kGrids[i]) < app.vramBudget * 85 / 100);
            char lbl[128];
            std::snprintf(lbl, sizeof(lbl), "%s   %.2f GB", kGrids[i].name, gb);
            ImGui::BeginDisabled(!fits);
            if (ImGui::Selectable(lbl, i == app.gridPreset) && i != app.gridPreset)
                app.applyGridPreset(i);
            ImGui::EndDisabled();
            if (!fits && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
                ImGui::SetTooltip("Needs %.2f GB of %.2f GB VRAM", gb,
                                  double(app.vramBudget) / (1024.0*1024.0*1024.0));
        }
        ImGui::EndCombo();
    }
    SectionEnd();

    // ── Physical scale ──────────────────────────────────────────────────────
    SectionHeader("Physical scale", kPurple);
    FieldLabel("Model length", "Real-world size of the body along the flow");
    if (ImGui::SliderFloat("##plen", &app.units.physLength, 0.05f, 20.f, "%.2f m",
                           ImGuiSliderFlags_Logarithmic))
        app.units.compute(app.model.spanCellsX, app.params);

    StatRow("Reynolds (physical)", "%.3g", app.units.rePhys);
    StatRow("Reynolds (lattice)",  "%.3g", app.units.reLat);
    StatRow("Mach", "%.3f", app.units.mach);
    StatRow("Cell size dx", "%.4g m", app.units.dx);
    StatRow("Time step dt", "%.4g s", app.units.dt);

    if (app.units.mach > 0.3f) {
        ImGui::PushStyleColor(ImGuiCol_Text, kAmber);
        ImGui::TextWrapped("Mach > 0.3 — compressibility effects are not modelled.");
        ImGui::PopStyleColor();
    }
    if (app.params.les && app.units.rePhys > app.units.reLat * 2.f) {
        ImGui::PushStyleColor(ImGuiCol_Text, kDim);
        ImGui::TextWrapped("High-Re flow: sub-grid turbulence is LES-modelled.");
        ImGui::PopStyleColor();
    }
    ImGui::Dummy({0, S(6)});

    ImGui::End();
    ImGui::PopStyleVar();
    ImGui::PopStyleColor();
}

// ─── Right panel: Results ────────────────────────────────────────────────────

void drawRightPanel(App& app, const Layout& l) {
    if (!app.rightOpen) return;
    ImGui::SetNextWindowPos({l.display.x - l.rightW, l.topH});
    ImGui::SetNextWindowSize({l.rightW, l.display.y - l.topH - l.statusH});
    ImGui::PushStyleColor(ImGuiCol_WindowBg, kBg1);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {S(14), S(12)});
    ImGui::Begin("##right", nullptr, kPanelFlags);

    // ── Forces ──────────────────────────────────────────────────────────────
    SectionHeader("Aerodynamic forces", kAccent);
    const float dCd = (std::abs(app.cdPrev) > 1e-9f)
                      ? (app.cd - app.cdPrev) / std::abs(app.cdPrev) * 100.f : 0.f;
    const float dCl = (std::abs(app.clPrev) > 1e-9f)
                      ? (app.cl - app.clPrev) / std::abs(app.clPrev) * 100.f : 0.f;
    BigValue(app, "Drag coefficient  C_D", app.cd, "%.4f", dCd, kAccent);
    Sparkline("##cdspark", app.cdHist.data(), kHist, app.histIdx, S(30), kAccent);
    ImGui::Spacing();
    BigValue(app, "Lift coefficient  C_L", app.cl, "%.4f", dCl, kBlue);
    Sparkline("##clspark", app.clHist.data(), kHist, app.histIdx, S(30), kBlue);
    if (std::abs(app.cl) > 1e-9f && std::abs(app.cd) > 1e-9f)
        StatRow("L / D", "%.2f", app.cl / app.cd);
    ImGui::PushStyleColor(ImGuiCol_Text, kDim);
    ImGui::TextWrapped("Momentum exchange over the voxel surface.");
    ImGui::PopStyleColor();
    SectionEnd();

    // ── Convergence ─────────────────────────────────────────────────────────
    SectionHeader("Convergence", kAmber);
    Sparkline("##resplot", app.resHist.data(), kHist, app.histIdx, S(48), kAmber);
    StatRow("Residual", "%.3e", app.an.residual);
    const char* state = !app.modelLoaded ? "--"
        : (app.an.residual < 1e-4f) ? "Steady"
        : (app.an.residual < 5e-3f) ? "Converging" : "Developing";
    StatRow("State", "%s", state);
    StatRow("Steps", "%llu", static_cast<unsigned long long>(app.totalSteps));
    SectionEnd();

    // ── Field ───────────────────────────────────────────────────────────────
    SectionHeader("Field", kBlue);
    StatRow("Peak velocity", "%.1f m/s", app.an.maxU * app.units.velToMps);
    StatRow("  (lattice)", "%.3f", app.an.maxU);
    StatRow("Kinetic energy", "%.3g", app.an.ke);
    // massAvg is 0 until the first analysis lands; showing that raw reads as
    // a bogus -100% drift.
    if (app.an.valid) StatRow("Mass drift", "%+.3f%%", (app.an.massAvg - 1.f) * 100.f);
    else              StatRow("Mass drift", "--");
    StatRow("Surface faces", "%.0f", app.an.surfFaces);
    SectionEnd();

    // ── Display ranges ──────────────────────────────────────────────────────
    SectionHeader("Colour range", kGreen);
    if (app.visMode == 2 || app.visMode == 3) {
        FieldLabel("Vorticity max", "Upper end of the colour scale");
        ImGui::SliderFloat("##maxvort", &app.maxVort, 0.005f, 0.4f, "%.3f",
                           ImGuiSliderFlags_Logarithmic);
    } else {
        FieldLabel("Velocity max", "Upper end of the colour scale, lattice units");
        ImGui::SliderFloat("##scalevel", &app.scaleVel, 0.02f, 0.5f, "%.3f");
        ImGui::PushStyleColor(ImGuiCol_Text, kDim);
        ImGui::Text("= %.1f m/s", app.scaleVel * app.units.velToMps);
        ImGui::PopStyleColor();
    }
    SectionEnd();

    // ── Performance ─────────────────────────────────────────────────────────
    SectionHeader("Performance", kPurple);
    StatRow("Frame", "%.2f ms (%.0f fps)", app.frameMs, app.fps);
    StatRow("Throughput", "%.1f MLUPS", app.mlups);
    StatRow("LBM batch", "%.2f ms", app.tim.lbmMs);
    StatRow("Analysis", "%.3f ms", app.tim.analysisMs);
    StatRow("Slice", "%.3f ms", app.tim.sliceMs);
    {
        const double used = double(app.vramUse) / (1024.0*1024.0*1024.0);
        const double tot  = double(app.vramBudget) / (1024.0*1024.0*1024.0);
        StatRow("VRAM", "%.2f / %.2f GB", used, tot);
        const float frac = (tot > 0.0) ? float(used / tot) : 0.f;
        ImDrawList* dl = ImGui::GetWindowDrawList();
        const ImVec2 p = ImGui::GetCursorScreenPos();
        const float w = ImGui::GetContentRegionAvail().x;
        dl->AddRectFilled(p, {p.x + w, p.y + S(4)}, u32(kBg2), S(2));
        dl->AddRectFilled(p, {p.x + w*std::clamp(frac, 0.f, 1.f), p.y + S(4)},
                          u32(frac > 0.85f ? kRed : kPurple), S(2));
        ImGui::Dummy({w, S(10)});
    }

    ImGui::End();
    ImGui::PopStyleVar();
    ImGui::PopStyleColor();
}

// ─── Viewport ────────────────────────────────────────────────────────────────

void drawColorbar(App& app, ImVec2 vpPos, ImVec2 vpSize) {
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const float barW = std::min(S(300), vpSize.x * 0.55f);
    const float barH = S(10);
    const ImVec2 p{ vpPos.x + (vpSize.x - barW)*0.5f, vpPos.y + vpSize.y - S(40) };

    dl->AddRectFilled({p.x - S(12), p.y - S(24)},
                      {p.x + barW + S(12), p.y + barH + S(10)},
                      u32(kBg1, 0.88f), S(6));

    const int segs = 64;
    for (int i = 0; i < segs; ++i) {
        const float t0 = float(i) / segs, t1 = float(i+1) / segs;
        const ImVec4 c = (app.visMode == 1) ? cmCoolwarm(t0) : cmJet(t0);
        dl->AddRectFilled({p.x + barW*t0, p.y}, {p.x + barW*t1 + 1.f, p.y + barH}, u32(c));
    }

    char lo[32], hi[32];
    const char* title;
    switch (app.visMode) {
    case 1:
        title = "Pressure coefficient  C_p";
        std::snprintf(lo, sizeof(lo), "-2");
        std::snprintf(hi, sizeof(hi), "+2");
        break;
    case 2:
        title = "Vorticity  (lattice)";
        std::snprintf(lo, sizeof(lo), "0");
        std::snprintf(hi, sizeof(hi), "%.3f", app.maxVort);
        break;
    case 3:
        title = "Q-criterion  (vortex cores)";
        std::snprintf(lo, sizeof(lo), "0");
        std::snprintf(hi, sizeof(hi), "max");
        break;
    default:
        title = "Velocity  (m/s)";
        std::snprintf(lo, sizeof(lo), "0");
        std::snprintf(hi, sizeof(hi), "%.1f", app.scaleVel * app.units.velToMps);
        break;
    }
    dl->AddText({p.x, p.y - S(19)}, u32(kDim), title);
    dl->AddText({p.x + barW - ImGui::CalcTextSize(hi).x, p.y - S(19)}, u32(kText), hi);
    dl->AddText({p.x, p.y + barH + S(1)}, u32(kText), lo);
}

// The viewport window holds ONLY the image, the drag surface and non-interactive
// decoration. Every clickable control lives in an overlay window (see header).
void drawViewport(App& app, const Layout& l) {
    ImGui::SetNextWindowPos(l.vpPos);
    ImGui::SetNextWindowSize(l.vpSize);
    ImGui::PushStyleColor(ImGuiCol_WindowBg, kBg0);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {0, 0});
    ImGui::Begin("##viewport", nullptr, kBarFlags);

    ImDrawList* dl = ImGui::GetWindowDrawList();

    // Pan / zoom surface. AllowOverlap is belt-and-braces: the tool windows are
    // separate windows, so they already win the hit test.
    ImGui::SetCursorScreenPos(l.vpPos);
    ImGui::SetNextItemAllowOverlap();
    ImGui::InvisibleButton("##vpdrag", l.vpSize, ImGuiButtonFlags_MouseButtonLeft);
    if (ImGui::IsItemHovered()) {
        const float wheel = ImGui::GetIO().MouseWheel;
        if (wheel != 0.f)
            app.zoom = std::clamp(app.zoom * std::pow(1.12f, wheel), 0.25f, 12.f);
    }
    if (ImGui::IsItemActive() && ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
        app.panX += ImGui::GetIO().MouseDelta.x;
        app.panY += ImGui::GetIO().MouseDelta.y;
    }

    if (app.modelLoaded && app.slice.textureId()) {
        const float iw = float(app.slice.width());
        const float ih = float(app.slice.height());
        const float fit = std::min(l.vpSize.x / iw, l.vpSize.y / ih) * 0.88f * app.zoom;
        const ImVec2 isz{ iw * fit, ih * fit };
        const ImVec2 ip { l.vpPos.x + (l.vpSize.x - isz.x)*0.5f + app.panX,
                          l.vpPos.y + (l.vpSize.y - isz.y)*0.5f + app.panY };
        dl->AddRect({ip.x - 1, ip.y - 1}, {ip.x + isz.x + 1, ip.y + isz.y + 1},
                    u32(kStroke), 0.f);
        // Flip V so +Y (and +Z) point up on screen
        dl->AddImage(reinterpret_cast<ImTextureID>(app.slice.textureId()),
                     ip, {ip.x + isz.x, ip.y + isz.y}, {0, 1}, {1, 0});
        drawColorbar(app, l.vpPos, l.vpSize);

        // Run-state pill (decoration only)
        const char* txt = app.running ? "LIVE" : "PAUSED";
        const ImVec2 ts = ImGui::CalcTextSize(txt);
        const ImVec2 pp{ l.vpPos.x + l.vpSize.x - ts.x - S(34), l.vpPos.y + S(12) };
        dl->AddRectFilled(pp, {pp.x + ts.x + S(24), pp.y + ts.y + S(9)},
                          u32(kBg1, 0.9f), S(10));
        const float pulse = app.running
            ? 0.6f + 0.4f * std::sin(float(ImGui::GetTime()) * 4.f) : 0.35f;
        dl->AddCircleFilled({pp.x + S(11), pp.y + ts.y*0.5f + S(4)}, S(3.5f),
                            u32(app.running ? kAccent : kDim, pulse));
        dl->AddText({pp.x + S(19), pp.y + S(4)},
                    u32(app.running ? kAccent : kDim), txt);
    }

    ImGui::End();
    ImGui::PopStyleVar();
    ImGui::PopStyleColor();
}

// Floating tool windows. Separate windows === clicks always land.
void drawViewOverlays(App& app, const Layout& l) {
    // ── Welcome card (no model yet) ──
    if (!app.modelLoaded) {
        ImGui::SetNextWindowPos({ l.vpPos.x + l.vpSize.x*0.5f,
                                  l.vpPos.y + l.vpSize.y*0.45f },
                                ImGuiCond_Always, {0.5f, 0.5f});
        ImGui::PushStyleColor(ImGuiCol_WindowBg, fade(kBg1, 0.96f));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {S(22), S(18)});
        ImGui::Begin("##welcome", nullptr, kOverlayFlags);

        pushFontBig(app);
        ImGui::TextUnformatted("Virtual Wind Tunnel");
        popFontBig(app);
        ImGui::PushStyleColor(ImGuiCol_Text, kDim);
        ImGui::TextUnformatted("Pick a shape to get started, or drop in a mesh file.");
        ImGui::PopStyleColor();
        ImGui::Dummy({0, S(8)});

        const float bw = S(96);
        if (ImGui::Button("Sphere",   {bw, S(30)})) app.loadPrimitive(Shape::Sphere);
        ImGui::SameLine();
        if (ImGui::Button("Cube",     {bw, S(30)})) app.loadPrimitive(Shape::Cube);
        ImGui::SameLine();
        if (ImGui::Button("Cylinder", {bw, S(30)})) app.loadPrimitive(Shape::Cylinder);
        ImGui::SameLine();
        if (ImGui::Button("Wing",     {bw, S(30)})) app.loadPrimitive(Shape::Wing);
        ImGui::SameLine();
        if (ImGui::Button("Open...",  {bw, S(30)})) app.openFileDialog();

        ImGui::End();
        ImGui::PopStyleVar();
        ImGui::PopStyleColor();
        return;      // no field to configure yet
    }

    // ── View toolbar (top-left of the viewport) ──
    ImGui::SetNextWindowPos({ l.vpPos.x + S(12), l.vpPos.y + S(12) });
    ImGui::PushStyleColor(ImGuiCol_WindowBg, fade(kBg1, 0.90f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {S(8), S(8)});
    ImGui::Begin("##viewtools", nullptr, kOverlayFlags);

    // Row 1 — field
    {
        const char* modes[] = { "Velocity", "Pressure", "Vorticity", "Q-crit" };
        const char* tips[]  = { "Speed magnitude  (1)", "Pressure coefficient  (2)",
                                "Vorticity magnitude  (3)", "Vortex cores  (4)" };
        for (int i = 0; i < 4; ++i) {
            if (i) ImGui::SameLine();
            if (Pill(modes[i], app.visMode == i, kAccent, S(84))) app.visMode = i;
            ImGui::SetItemTooltip("%s", tips[i]);
        }
    }

    // Row 2 — slice plane
    {
        const char* axes[] = { "XY", "XZ", "ZY" };
        const char* atip[] = { "Plane through Z (top view)",
                               "Plane through Y (side view)",
                               "Plane through X (cross-section)" };
        for (int i = 0; i < 3; ++i) {
            if (i) ImGui::SameLine();
            if (Pill(axes[i], app.axis == i, kBlue, S(46)) && app.axis != i) {
                app.axis = i;
                app.sliceIndex = -1;   // recentre on the new axis
            }
            ImGui::SetItemTooltip("%s", atip[i]);
        }

        ImGui::SameLine(0, S(10));
        // sliceIndex is -1 while "centred"; resolve it so the slider never
        // shows an out-of-range value.
        const int maxIdx = int(app.sliceMaxIndex());
        int idx = (app.sliceIndex < 0) ? maxIdx / 2 : app.sliceIndex;
        idx = std::clamp(idx, 0, maxIdx);
        ImGui::SetNextItemWidth(S(150));
        if (ImGui::SliderInt("##sliceidx", &idx, 0, maxIdx, "slice %d"))
            app.sliceIndex = idx;
        ImGui::SetItemTooltip("Move the cutting plane  ( [ and ] )");

        ImGui::SameLine(0, S(10));
        const bool moved = (app.zoom != 1.f || app.panX != 0.f || app.panY != 0.f);
        ImGui::BeginDisabled(!moved);
        if (ImGui::Button("Reset view")) { app.zoom = 1.f; app.panX = app.panY = 0.f; }
        ImGui::EndDisabled();
        ImGui::SetItemTooltip("Recentre and unzoom  (Esc)");
    }

    ImGui::End();
    ImGui::PopStyleVar();
    ImGui::PopStyleColor();
}

// ─── Status bar ──────────────────────────────────────────────────────────────

void drawStatusBar(App& app, const Layout& l) {
    ImGui::SetNextWindowPos({0, l.display.y - l.statusH});
    ImGui::SetNextWindowSize({l.display.x, l.statusH});
    ImGui::PushStyleColor(ImGuiCol_WindowBg, kBg1);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {S(12), S(5)});
    ImGui::Begin("##status", nullptr, kBarFlags);

    ImDrawList* dl = ImGui::GetWindowDrawList();
    const ImVec2 p = ImGui::GetCursorScreenPos();
    const float pulse = app.running
        ? 0.6f + 0.4f * std::sin(float(ImGui::GetTime()) * 4.f) : 0.4f;
    dl->AddCircleFilled({p.x + S(5), p.y + S(8)}, S(4),
                        u32(app.running ? kAccent : kDim, pulse));
    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + S(16));

    const char* opName = app.params.collision == 0 ? "BGK"
                       : app.params.collision == 2 ? "TRT" : "Regularized";
    char buf[256];
    std::snprintf(buf, sizeof(buf),
        "%s   |   step %llu   |   %ux%ux%u   |   %s%s   |   tau %.3f   |   zoom %.0f%%",
        app.running ? "Running" : (app.modelLoaded ? "Paused" : "Idle"),
        static_cast<unsigned long long>(app.totalSteps),
        app.params.gx, app.params.gy, app.params.gz,
        opName, app.params.les ? " + LES" : "",
        app.params.tau, app.zoom * 100.f);
    ImGui::PushStyleColor(ImGuiCol_Text, kDim);
    ImGui::TextUnformatted(buf);
    ImGui::PopStyleColor();

    if (app.statusTtl > 0.f && !app.statusMsg.empty()) {
        const float w = ImGui::CalcTextSize(app.statusMsg.c_str()).x;
        ImGui::SameLine();
        const float cursorX   = ImGui::GetCursorPosX();
        const float rightEdge = cursorX + ImGui::GetContentRegionAvail().x;
        if (rightEdge - w > cursorX) {
            ImGui::SetCursorPosX(rightEdge - w);
            ImGui::PushStyleColor(ImGuiCol_Text, kAccent);
            ImGui::TextUnformatted(app.statusMsg.c_str());
            ImGui::PopStyleColor();
        }
    }

    ImGui::End();
    ImGui::PopStyleVar();
    ImGui::PopStyleColor();
}

// ─── Help ────────────────────────────────────────────────────────────────────

// A plain window, not a modal: the close button writes straight back to
// app.showHelp, so the button and the '?' / Esc shortcuts can never desync.
void drawHelp(App& app, const Layout& l) {
    if (!app.showHelp) return;
    ImGui::SetNextWindowPos({ l.vpPos.x + l.vpSize.x*0.5f, l.vpPos.y + l.vpSize.y*0.5f },
                            ImGuiCond_Appearing, {0.5f, 0.5f});
    ImGui::PushStyleColor(ImGuiCol_WindowBg, kBg1);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {S(18), S(14)});
    if (ImGui::Begin("Keyboard shortcuts", &app.showHelp,
                     ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings |
                     ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoScrollbar)) {
        struct Row { const char* key; const char* what; };
        static const Row rows[] = {
            { "Space",   "Run / pause" },
            { "R",       "Reset the flow" },
            { "1 - 4",   "Velocity / pressure / vorticity / Q" },
            { "[  ]",    "Move the slice plane" },
            { "Tab",     "Toggle the Setup panel" },
            { "F",       "Toggle the Results panel" },
            { "S",       "Save a snapshot" },
            { "F11",     "Fullscreen" },
            { "Esc",     "Reset the view / close this" },
            { "?",       "Toggle this window" },
        };
        if (ImGui::BeginTable("##keys", 2, ImGuiTableFlags_SizingFixedFit)) {
            for (const Row& r : rows) {
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                ImGui::PushStyleColor(ImGuiCol_Text, kAccent);
                ImGui::TextUnformatted(r.key);
                ImGui::PopStyleColor();
                ImGui::TableSetColumnIndex(1);
                ImGui::TextUnformatted(r.what);
            }
            ImGui::EndTable();
        }
        ImGui::Dummy({0, S(6)});
        ImGui::PushStyleColor(ImGuiCol_Text, kDim);
        ImGui::TextUnformatted("Drag the view to pan, scroll to zoom.");
        ImGui::PopStyleColor();
    }
    ImGui::End();
    ImGui::PopStyleVar();
    ImGui::PopStyleColor();
}

} // namespace

// ─── Public API ──────────────────────────────────────────────────────────────

void applyTheme(float dpiScale) {
    gScale = dpiScale;
    ImGuiStyle& s = ImGui::GetStyle();
    s.WindowRounding    = 0.f;
    s.ChildRounding     = 6.f;
    s.FrameRounding     = 5.f;
    s.PopupRounding     = 6.f;
    s.GrabRounding      = 5.f;
    s.TabRounding       = 5.f;
    s.ScrollbarRounding = 6.f;
    s.WindowBorderSize  = 0.f;
    s.ChildBorderSize   = 1.f;
    s.PopupBorderSize   = 1.f;
    s.FrameBorderSize   = 0.f;
    s.WindowPadding     = {12.f, 10.f};
    s.FramePadding      = {9.f, 5.f};
    s.ItemSpacing       = {8.f, 7.f};
    s.ItemInnerSpacing  = {6.f, 4.f};
    s.GrabMinSize       = 9.f;
    s.ScrollbarSize     = 11.f;

    ImVec4* c = s.Colors;
    c[ImGuiCol_WindowBg]            = kBg1;
    c[ImGuiCol_ChildBg]             = {0, 0, 0, 0};
    c[ImGuiCol_PopupBg]             = kBg1;
    c[ImGuiCol_Border]              = kStroke;
    c[ImGuiCol_Text]                = kText;
    c[ImGuiCol_TextDisabled]        = {0.38f, 0.40f, 0.46f, 1.f};
    c[ImGuiCol_FrameBg]             = kBg2;
    c[ImGuiCol_FrameBgHovered]      = {0.115f, 0.125f, 0.16f, 1.f};
    c[ImGuiCol_FrameBgActive]       = {0.14f, 0.15f, 0.19f, 1.f};
    c[ImGuiCol_TitleBg]             = kBg0;
    c[ImGuiCol_TitleBgActive]       = kBg0;
    c[ImGuiCol_ScrollbarBg]         = kBg0;
    c[ImGuiCol_ScrollbarGrab]       = kStroke;
    c[ImGuiCol_ScrollbarGrabHovered]= {0.22f, 0.24f, 0.30f, 1.f};
    c[ImGuiCol_ScrollbarGrabActive] = {0.28f, 0.30f, 0.38f, 1.f};
    c[ImGuiCol_CheckMark]           = kAccent;
    c[ImGuiCol_SliderGrab]          = {kAccent.x*0.85f, kAccent.y*0.85f, kAccent.z*0.85f, 1.f};
    c[ImGuiCol_SliderGrabActive]    = kAccent;
    c[ImGuiCol_Button]              = kBg2;
    c[ImGuiCol_ButtonHovered]       = fade(kAccent, 0.18f);
    c[ImGuiCol_ButtonActive]        = fade(kAccent, 0.30f);
    c[ImGuiCol_Header]              = kBg2;
    c[ImGuiCol_HeaderHovered]       = fade(kAccent, 0.16f);
    c[ImGuiCol_HeaderActive]        = fade(kAccent, 0.26f);
    c[ImGuiCol_Separator]           = kStroke;
    c[ImGuiCol_PlotLines]           = kAccent;
    c[ImGuiCol_PlotHistogram]       = kAccent;
    c[ImGuiCol_TableBorderLight]    = kStroke;
    c[ImGuiCol_TableBorderStrong]   = kStroke;

    s.ScaleAllSizes(dpiScale);
}

void loadFonts(App& app, float dpiScale) {
    ImGuiIO& io = ImGui::GetIO();
    auto tryLoad = [&](std::initializer_list<const char*> paths, float size) -> ImFont* {
        for (const char* p : paths) {
            if (std::ifstream f(p); f.good())
                return io.Fonts->AddFontFromFileTTF(p, size * dpiScale);
        }
        return nullptr;
    };

    app.fonts.body = tryLoad({
        "C:/Windows/Fonts/segoeui.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/liberation/LiberationSans-Regular.ttf",
    }, 16.5f);
    app.fonts.mono = tryLoad({
        "C:/Windows/Fonts/consola.ttf",
        "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    }, 13.f);
    app.fonts.big = tryLoad({
        "C:/Windows/Fonts/seguisb.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    }, 26.f);

    if (!app.fonts.body) app.fonts.body = io.Fonts->AddFontDefault();
    io.FontGlobalScale = 1.f;
    io.FontDefault = app.fonts.body;
}

void draw(App& app) {
    const Layout l = computeLayout(app);
    drawTopBar(app, l);
    drawLeftPanel(app, l);
    drawRightPanel(app, l);
    drawViewport(app, l);       // image + drag surface only
    drawViewOverlays(app, l);   // every clickable overlay, in its own window
    drawStatusBar(app, l);
    drawHelp(app, l);
}

} // namespace vwt::ui
