// ============================================================================
// ui.cpp — theme, panels, viewport, overlays
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

// ─── CPU colormaps (mirror slice.comp, for the colorbar) ────────────────────
ImVec4 cmInferno(float t) {
    t = std::clamp(t, 0.f, 1.f);
    auto P = [t](double a,double b,double c,double d,double e,double f,double g) {
        return float(a+t*(b+t*(c+t*(d+t*(e+t*(f+t*g))))));
    };
    return { P( 0.0002189403, 0.1065134195, 11.6024930825,-41.7039961314, 77.1629356994,-71.3194282450, 25.1311262248),
             P( 0.0016510046, 0.5639564368,-3.9728539657, 17.4363988821,-33.4023589421, 32.6260642640,-12.2426689524),
             P(-0.0194808984, 3.9327123889,-15.9423941063, 44.3541451987,-81.8073092574, 73.2095198580,-23.0703250029),
             1.f };
}
ImVec4 cmViridis(float t) {
    t = std::clamp(t, 0.f, 1.f);
    auto P = [t](double a,double b,double c,double d,double e,double f,double g) {
        return float(a+t*(b+t*(c+t*(d+t*(e+t*(f+t*g))))));
    };
    return { P( 0.2777273272, 0.1050930431,-0.3308618287,-4.6342304990, 6.2282699363, 4.7763849977,-5.4354558559),
             P( 0.0054073445, 1.4046135299, 0.2148475595,-5.7991009734,14.1799333668,-13.7451453777, 4.6458526122),
             P( 0.3340998053, 1.3845901626, 0.0950951630,-19.3324409563,56.6905526007,-65.3530326334,26.3124352495),
             1.f };
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

// ─── Small widgets ───────────────────────────────────────────────────────────

void CardHeader(const char* title, ImVec4 accent) {
    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImVec2 p = ImGui::GetCursorScreenPos();
    dl->AddRectFilled({p.x, p.y + S(3)}, {p.x + S(3), p.y + S(15)}, u32(accent), S(1.5f));
    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + S(10));
    ImGui::PushStyleColor(ImGuiCol_Text, kText);
    ImGui::TextUnformatted(title);
    ImGui::PopStyleColor();
    ImGui::Spacing();
}

void CardEnd() {
    ImGui::Spacing();
    ImGui::PushStyleColor(ImGuiCol_Separator, kStroke);
    ImGui::Separator();
    ImGui::PopStyleColor();
    ImGui::Spacing();
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
    const float w = ImGui::GetContentRegionAvail().x - ImGui::CalcTextSize(buf).x;
    ImGui::SameLine(std::max(w, 0.f) > 0.f ? ImGui::GetCursorPosX() + w : 0.f);
    ImGui::TextUnformatted(buf);
}

bool Segmented(const char* id, int* value, const char* const* labels, int count) {
    bool changed = false;
    ImGui::PushID(id);
    const float w = (ImGui::GetContentRegionAvail().x
                     - ImGui::GetStyle().ItemSpacing.x * float(count - 1)) / float(count);
    for (int i = 0; i < count; ++i) {
        if (i) ImGui::SameLine();
        const bool active = (*value == i);
        ImGui::PushStyleColor(ImGuiCol_Button, active ? ImVec4{kAccent.x, kAccent.y, kAccent.z, 0.22f} : kBg2);
        ImGui::PushStyleColor(ImGuiCol_Text, active ? kAccent : kDim);
        if (ImGui::Button(labels[i], {w, 0})) { *value = i; changed = true; }
        ImGui::PopStyleColor(2);
    }
    ImGui::PopID();
    return changed;
}

void BigValue(App& app, const char* label, float value, const char* fmt,
              float delta, ImVec4 accent) {
    ImGui::PushStyleColor(ImGuiCol_Text, kDim);
    ImGui::TextUnformatted(label);
    ImGui::PopStyleColor();

    char buf[32];
    std::snprintf(buf, sizeof(buf), fmt, value);
    if (app.fonts.big) ImGui::PushFont(app.fonts.big);
    ImGui::PushStyleColor(ImGuiCol_Text, accent);
    ImGui::TextUnformatted(buf);
    ImGui::PopStyleColor();
    if (app.fonts.big) ImGui::PopFont();

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

// ─── Panels ──────────────────────────────────────────────────────────────────

constexpr ImGuiWindowFlags kPanelFlags =
    ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
    ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoBringToFrontOnFocus;

struct Layout {
    float topH, statusH, leftW, rightW;
    ImVec2 display;
};

Layout computeLayout(const App& app) {
    Layout l;
    l.display = ImGui::GetIO().DisplaySize;
    l.topH    = S(46);
    l.statusH = S(27);
    l.leftW   = app.leftOpen  ? std::min(S(304), l.display.x * 0.26f) : 0.f;
    l.rightW  = app.rightOpen ? std::min(S(316), l.display.x * 0.26f) : 0.f;
    return l;
}

void drawTopBar(App& app, const Layout& l) {
    ImGui::SetNextWindowPos({0, 0});
    ImGui::SetNextWindowSize({l.display.x, l.topH});
    ImGui::PushStyleColor(ImGuiCol_WindowBg, kBg1);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {S(12), S(8)});
    ImGui::Begin("##top", nullptr, kPanelFlags);

    // Brand
    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImVec2 p = ImGui::GetCursorScreenPos();
    dl->AddCircleFilled({p.x + S(9), p.y + S(14)}, S(7), u32(kAccent));
    dl->AddCircleFilled({p.x + S(9), p.y + S(14)}, S(3.2f), u32(kBg1));
    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + S(26));
    ImGui::SetCursorPosY(ImGui::GetCursorPosY() + S(4));
    ImGui::TextUnformatted("Virtual Wind Tunnel");
    ImGui::SameLine();
    ImGui::PushStyleColor(ImGuiCol_Text, kDim);
    ImGui::TextUnformatted("v2");
    ImGui::PopStyleColor();

    ImGui::SameLine(0, S(24));

    // Load menu
    if (ImGui::Button("Load model")) ImGui::OpenPopup("##loadmenu");
    if (ImGui::BeginPopup("##loadmenu")) {
        if (ImGui::MenuItem("Sphere"))          app.loadPrimitive(Shape::Sphere);
        if (ImGui::MenuItem("Cube"))            app.loadPrimitive(Shape::Cube);
        if (ImGui::MenuItem("Cylinder"))        app.loadPrimitive(Shape::Cylinder);
        if (ImGui::MenuItem("NACA 0012 wing"))  app.loadPrimitive(Shape::Wing);
        ImGui::Separator();
        if (ImGui::MenuItem("Open file..."))    app.openFileDialog();
        ImGui::EndPopup();
    }

    ImGui::SameLine(0, S(16));

    // Transport
    const bool canRun = app.modelLoaded;
    if (!canRun) ImGui::BeginDisabled();
    ImGui::PushStyleColor(ImGuiCol_Button,
        app.running ? ImVec4{kAmber.x, kAmber.y, kAmber.z, 0.25f}
                    : ImVec4{kAccent.x, kAccent.y, kAccent.z, 0.25f});
    ImGui::PushStyleColor(ImGuiCol_Text, app.running ? kAmber : kAccent);
    if (ImGui::Button(app.running ? "Pause  [Space]" : "Run  [Space]"))
        app.running = !app.running;
    ImGui::PopStyleColor(2);
    ImGui::SameLine();
    if (ImGui::Button("Step")) {
        app.running = false;
        app.runSingleBatch = true;   // frame loop executes one batch then stops
    }
    ImGui::SameLine();
    if (ImGui::Button("Reset  [R]")) app.resetSim();
    if (!canRun) ImGui::EndDisabled();

    // Right side, laid out right-to-left
    const float right = ImGui::GetWindowWidth();
    float x = right - S(12);

    auto rightText = [&](const char* txt, ImVec4 col) {
        const float w = ImGui::CalcTextSize(txt).x;
        x -= w;
        ImGui::SameLine(x);
        ImGui::PushStyleColor(ImGuiCol_Text, col);
        ImGui::TextUnformatted(txt);
        ImGui::PopStyleColor();
        x -= S(16);
    };

    char fpsBuf[48];
    std::snprintf(fpsBuf, sizeof(fpsBuf), "%.0f fps  |  %.1f MLUPS", app.fps, app.mlups);
    if (l.display.x > S(900)) rightText(app.gpu.gpuName(), kDim);
    rightText(fpsBuf, kAccent);

    x -= S(70);
    ImGui::SameLine(x);
    if (ImGui::Button("Snapshot")) app.snapshot();
    x -= S(40);
    ImGui::SameLine(x);
    if (ImGui::Button("?")) app.showHelp = !app.showHelp;

    ImGui::End();
    ImGui::PopStyleVar();
    ImGui::PopStyleColor();
}

void drawLeftPanel(App& app, const Layout& l) {
    if (!app.leftOpen) return;
    ImGui::SetNextWindowPos({0, l.topH});
    ImGui::SetNextWindowSize({l.leftW, l.display.y - l.topH - l.statusH});
    ImGui::PushStyleColor(ImGuiCol_WindowBg, kBg1);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {S(14), S(12)});
    ImGui::Begin("##left", nullptr, kPanelFlags);

    // ── Model ───────────────────────────────────────────────────────────────
    CardHeader("Model", kAccent);
    if (app.modelLoaded) {
        StatRow("Name", "%s", app.model.name.c_str());
        if (app.model.triCount > 0)
            StatRow("Triangles", "%u", app.model.triCount);
        StatRow("Fill", "%.2f%% of domain", app.model.fillPct);
        StatRow("Frontal area", "%u cells", app.model.frontalCells);
        ImGui::Spacing();

        ImGui::PushStyleColor(ImGuiCol_Text, kDim);
        ImGui::TextUnformatted("Angle of attack");
        ImGui::PopStyleColor();
        ImGui::SetNextItemWidth(-1);
        ImGui::SliderFloat("##aoa", &app.aoaDeg, -20.f, 20.f, "%.1f deg");
        if (ImGui::IsItemDeactivatedAfterEdit()) app.revoxelize();

        ImGui::PushStyleColor(ImGuiCol_Text, kDim);
        ImGui::TextUnformatted("Yaw");
        ImGui::PopStyleColor();
        ImGui::SetNextItemWidth(-1);
        ImGui::SliderFloat("##yaw", &app.yawDeg, -30.f, 30.f, "%.1f deg");
        if (ImGui::IsItemDeactivatedAfterEdit()) app.revoxelize();
    } else {
        ImGui::PushStyleColor(ImGuiCol_Text, kDim);
        ImGui::TextWrapped("No model loaded. Use Load model in the top bar, "
                           "or drop a mesh file onto the window.");
        ImGui::PopStyleColor();
    }
    CardEnd();

    // ── Flow ────────────────────────────────────────────────────────────────
    CardHeader("Flow", kBlue);
    ImGui::PushStyleColor(ImGuiCol_Text, kDim);
    ImGui::TextUnformatted("Wind speed");
    ImGui::PopStyleColor();
    ImGui::SetNextItemWidth(-1);
    if (ImGui::SliderFloat("##uphys", &app.units.physSpeed, 1.f, 150.f, "%.1f m/s"))
        app.units.compute(app.model.spanCellsX, app.params);

    ImGui::PushStyleColor(ImGuiCol_Text, kDim);
    ImGui::TextUnformatted("Fluid");
    ImGui::PopStyleColor();
    ImGui::SetNextItemWidth(-1);
    if (ImGui::BeginCombo("##fluid", kFluids[app.units.fluidIdx].name)) {
        for (int i = 0; i < kFluidCount; ++i)
            if (ImGui::Selectable(kFluids[i].name, i == app.units.fluidIdx)) {
                app.units.fluidIdx = i;
                app.units.compute(app.model.spanCellsX, app.params);
            }
        ImGui::EndCombo();
    }

    ImGui::PushStyleColor(ImGuiCol_Text, kDim);
    ImGui::TextUnformatted("Inlet turbulence");
    ImGui::PopStyleColor();
    ImGui::SetNextItemWidth(-1);
    float turbPct = app.params.turb * 100.f;
    if (ImGui::SliderFloat("##turb", &turbPct, 0.f, 30.f, "%.0f%%"))
        app.params.turb = turbPct / 100.f;
    CardEnd();

    // ── Solver ──────────────────────────────────────────────────────────────
    CardHeader("Solver", kAmber);
    {
        const char* ops[] = { "BGK", "Regularized" };
        Segmented("##collision", &app.params.collision, ops, 2);
    }
    ImGui::Spacing();
    ImGui::Checkbox("LES turbulence model", &app.params.les);
    if (app.params.les) {
        ImGui::SetNextItemWidth(-1);
        ImGui::SliderFloat("##cs", &app.params.csSmago, 0.06f, 0.24f, "Cs = %.2f");
    }

    ImGui::PushStyleColor(ImGuiCol_Text, kDim);
    ImGui::TextUnformatted("Relaxation time (tau)");
    ImGui::PopStyleColor();
    ImGui::SetNextItemWidth(-1);
    if (ImGui::SliderFloat("##tau", &app.params.tau, 0.505f, 0.9f, "%.3f"))
        app.units.compute(app.model.spanCellsX, app.params);

    ImGui::PushStyleColor(ImGuiCol_Text, kDim);
    ImGui::TextUnformatted("Steps per frame");
    ImGui::PopStyleColor();
    ImGui::SetNextItemWidth(-1);
    ImGui::SliderInt("##spf", &app.stepsPerFrame, 1, 40);

    ImGui::PushStyleColor(ImGuiCol_Text, kDim);
    ImGui::TextUnformatted("Grid resolution");
    ImGui::PopStyleColor();
    ImGui::SetNextItemWidth(-1);
    if (ImGui::BeginCombo("##grid", kGrids[app.gridPreset].name)) {
        for (int i = 0; i < kGridCount; ++i)
            if (ImGui::Selectable(kGrids[i].name, i == app.gridPreset) && i != app.gridPreset)
                app.applyGridPreset(i);
        ImGui::EndCombo();
    }
    CardEnd();

    // ── Physical scale ──────────────────────────────────────────────────────
    CardHeader("Physical scale", kPurple);
    ImGui::PushStyleColor(ImGuiCol_Text, kDim);
    ImGui::TextUnformatted("Model length");
    ImGui::PopStyleColor();
    ImGui::SetNextItemWidth(-1);
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
        ImGui::TextWrapped("Mach > 0.3 — compressibility effects are not modeled.");
        ImGui::PopStyleColor();
    }
    if (app.params.les && app.units.rePhys > app.units.reLat * 2.f) {
        ImGui::PushStyleColor(ImGuiCol_Text, kDim);
        ImGui::TextWrapped("High-Re flow: sub-grid turbulence is LES-modeled.");
        ImGui::PopStyleColor();
    }

    ImGui::End();
    ImGui::PopStyleVar();
    ImGui::PopStyleColor();
}

void drawRightPanel(App& app, const Layout& l) {
    if (!app.rightOpen) return;
    ImGui::SetNextWindowPos({l.display.x - l.rightW, l.topH});
    ImGui::SetNextWindowSize({l.rightW, l.display.y - l.topH - l.statusH});
    ImGui::PushStyleColor(ImGuiCol_WindowBg, kBg1);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {S(14), S(12)});
    ImGui::Begin("##right", nullptr, kPanelFlags);

    // ── Forces ──────────────────────────────────────────────────────────────
    CardHeader("Aerodynamic forces", kAccent);
    const float dCd = (std::abs(app.cdPrev) > 1e-9f)
                      ? (app.cd - app.cdPrev) / std::abs(app.cdPrev) * 100.f : 0.f;
    const float dCl = (std::abs(app.clPrev) > 1e-9f)
                      ? (app.cl - app.clPrev) / std::abs(app.clPrev) * 100.f : 0.f;
    BigValue(app, "Drag coefficient C_D", app.cd, "%.4f", dCd, kAccent);
    Sparkline("##cdspark", app.cdHist.data(), kHist, app.histIdx, S(30), kAccent);
    ImGui::Spacing();
    BigValue(app, "Lift coefficient C_L", app.cl, "%.4f", dCl, kBlue);
    Sparkline("##clspark", app.clHist.data(), kHist, app.histIdx, S(30), kBlue);
    if (std::abs(app.cl) > 1e-9f && std::abs(app.cd) > 1e-9f)
        StatRow("L/D", "%.2f", app.cl / app.cd);
    ImGui::PushStyleColor(ImGuiCol_Text, kDim);
    ImGui::TextUnformatted("Pressure component, voxel surface");
    ImGui::PopStyleColor();
    CardEnd();

    // ── Convergence ─────────────────────────────────────────────────────────
    CardHeader("Convergence", kAmber);
    Sparkline("##resplot", app.resHist.data(), kHist, app.histIdx, S(48), kAmber);
    StatRow("Residual", "%.3e", app.an.residual);
    const char* state = !app.modelLoaded ? "--"
        : (app.an.residual < 1e-4f) ? "Steady"
        : (app.an.residual < 5e-3f) ? "Converging" : "Developing";
    StatRow("State", "%s", state);
    StatRow("Steps", "%llu", static_cast<unsigned long long>(app.totalSteps));
    CardEnd();

    // ── Field statistics ────────────────────────────────────────────────────
    CardHeader("Field", kBlue);
    StatRow("Peak velocity", "%.1f m/s", app.an.maxU * app.units.velToMps);
    StatRow("  (lattice)", "%.3f", app.an.maxU);
    StatRow("Kinetic energy", "%.3g", app.an.ke);
    StatRow("Mass drift", "%+.3f%%", (app.an.massAvg - 1.f) * 100.f);
    StatRow("Surface faces", "%.0f", app.an.surfFaces);
    CardEnd();

    // ── Performance ─────────────────────────────────────────────────────────
    CardHeader("Performance", kPurple);
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
        ImVec2 p = ImGui::GetCursorScreenPos();
        const float w = ImGui::GetContentRegionAvail().x;
        dl->AddRectFilled(p, {p.x + w, p.y + S(4)}, u32(kBg2), S(2));
        dl->AddRectFilled(p, {p.x + w*std::clamp(frac, 0.f, 1.f), p.y + S(4)},
                          u32(frac > 0.85f ? kRed : kPurple), S(2));
        ImGui::Dummy({w, S(7)});
    }

    ImGui::End();
    ImGui::PopStyleVar();
    ImGui::PopStyleColor();
}

void drawColorbar(App& app, ImVec2 vpPos, ImVec2 vpSize) {
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const float barW = std::min(S(280), vpSize.x * 0.5f);
    const float barH = S(10);
    const ImVec2 p{ vpPos.x + (vpSize.x - barW)*0.5f, vpPos.y + vpSize.y - S(38) };

    dl->AddRectFilled({p.x - S(10), p.y - S(20)},
                      {p.x + barW + S(10), p.y + barH + S(8)},
                      u32(kBg1, 0.85f), S(6));

    const int segs = 48;
    for (int i = 0; i < segs; ++i) {
        const float t0 = float(i) / segs, t1 = float(i+1) / segs;
        ImVec4 c;
        switch (app.visMode) {
        case 1:  c = cmCoolwarm(t0); break;
        case 2:  c = cmViridis(t0);  break;
        default: c = cmInferno(t0);  break;
        }
        dl->AddRectFilled({p.x + barW*t0, p.y}, {p.x + barW*t1 + 1.f, p.y + barH}, u32(c));
        (void)t1;
    }

    char lo[32], hi[32], title[64];
    switch (app.visMode) {
    case 0:
        std::snprintf(title, sizeof(title), "Velocity  (m/s)");
        std::snprintf(lo, sizeof(lo), "0");
        std::snprintf(hi, sizeof(hi), "%.1f", app.scaleVel * app.units.velToMps);
        break;
    case 1:
        std::snprintf(title, sizeof(title), "Pressure coefficient C_p");
        std::snprintf(lo, sizeof(lo), "-2");
        std::snprintf(hi, sizeof(hi), "+2");
        break;
    case 2:
        std::snprintf(title, sizeof(title), "Vorticity  (lattice)");
        std::snprintf(lo, sizeof(lo), "0");
        std::snprintf(hi, sizeof(hi), "%.3f", app.maxVort);
        break;
    default:
        std::snprintf(title, sizeof(title), "Q-criterion (vortex cores)");
        std::snprintf(lo, sizeof(lo), "0");
        std::snprintf(hi, sizeof(hi), "max");
        break;
    }
    dl->AddText({p.x, p.y - S(17)}, u32(kDim), title);
    dl->AddText({p.x + barW - ImGui::CalcTextSize(hi).x - S(34),
                 p.y - S(17)}, u32(kText), lo);
    dl->AddText({p.x + barW - ImGui::CalcTextSize(hi).x, p.y - S(17)},
                u32(kText), hi);
}

void drawViewport(App& app, const Layout& l) {
    const ImVec2 pos { l.leftW, l.topH };
    const ImVec2 size{ l.display.x - l.leftW - l.rightW,
                       l.display.y - l.topH - l.statusH };
    ImGui::SetNextWindowPos(pos);
    ImGui::SetNextWindowSize(size);
    ImGui::PushStyleColor(ImGuiCol_WindowBg, kBg0);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {0, 0});
    ImGui::Begin("##viewport", nullptr, kPanelFlags | ImGuiWindowFlags_NoScrollbar);

    ImDrawList* dl = ImGui::GetWindowDrawList();

    // Interaction surface
    ImGui::SetCursorScreenPos(pos);
    ImGui::InvisibleButton("##vpdrag", size,
                           ImGuiButtonFlags_MouseButtonLeft);
    const bool hovered = ImGui::IsItemHovered();
    if (hovered) {
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
        const float fit = std::min(size.x / iw, size.y / ih) * 0.90f * app.zoom;
        const ImVec2 isz{ iw * fit, ih * fit };
        const ImVec2 ip { pos.x + (size.x - isz.x)*0.5f + app.panX,
                          pos.y + (size.y - isz.y)*0.5f + app.panY };
        dl->AddRect({ip.x - 1, ip.y - 1}, {ip.x + isz.x + 1, ip.y + isz.y + 1},
                    u32(kStroke), 0.f);
        // Flip V so +Y (and +Z) point up on screen
        dl->AddImage(reinterpret_cast<ImTextureID>(app.slice.textureId()),
                     ip, {ip.x + isz.x, ip.y + isz.y}, {0, 1}, {1, 0});
        drawColorbar(app, pos, size);
    } else {
        // Welcome card
        const ImVec2 c{ pos.x + size.x*0.5f, pos.y + size.y*0.42f };
        const char* t1 = "Virtual Wind Tunnel";
        const char* t2 = "Load a sample shape or drop an STL / OBJ / glTF file";
        dl->AddText({c.x - ImGui::CalcTextSize(t1).x*0.5f, c.y - S(26)}, u32(kText), t1);
        dl->AddText({c.x - ImGui::CalcTextSize(t2).x*0.5f, c.y}, u32(kDim), t2);

        ImGui::SetCursorScreenPos({c.x - S(180), c.y + S(34)});
        if (ImGui::Button("Sphere",   {S(80), 0})) app.loadPrimitive(Shape::Sphere);
        ImGui::SameLine();
        if (ImGui::Button("Cylinder", {S(80), 0})) app.loadPrimitive(Shape::Cylinder);
        ImGui::SameLine();
        if (ImGui::Button("Wing",     {S(80), 0})) app.loadPrimitive(Shape::Wing);
        ImGui::SameLine();
        if (ImGui::Button("Open...",  {S(80), 0})) app.openFileDialog();
    }

    // ── Vis-mode tabs (floating, top-left) ─────────────────────────────────
    ImGui::SetCursorScreenPos({pos.x + S(12), pos.y + S(12)});
    {
        const char* modes[] = { "Velocity 1", "Pressure 2", "Vorticity 3", "Q-crit 4" };
        for (int i = 0; i < 4; ++i) {
            if (i) ImGui::SameLine();
            const bool active = (app.visMode == i);
            ImGui::PushStyleColor(ImGuiCol_Button,
                active ? ImVec4{kAccent.x, kAccent.y, kAccent.z, 0.25f}
                       : ImVec4{kBg1.x, kBg1.y, kBg1.z, 0.88f});
            ImGui::PushStyleColor(ImGuiCol_Text, active ? kAccent : kDim);
            if (ImGui::Button(modes[i])) app.visMode = i;
            ImGui::PopStyleColor(2);
        }
    }

    // ── Slice controls (floating, below tabs) ───────────────────────────────
    ImGui::SetCursorScreenPos({pos.x + S(12), pos.y + S(44)});
    {
        const char* axes[] = { "XY", "XZ", "ZY" };
        for (int i = 0; i < 3; ++i) {
            if (i) ImGui::SameLine();
            const bool active = (app.axis == i);
            ImGui::PushStyleColor(ImGuiCol_Button,
                active ? ImVec4{kBlue.x, kBlue.y, kBlue.z, 0.25f}
                       : ImVec4{kBg1.x, kBg1.y, kBg1.z, 0.88f});
            ImGui::PushStyleColor(ImGuiCol_Text, active ? kBlue : kDim);
            if (ImGui::Button(axes[i]) && app.axis != i) {
                app.axis = i;
                app.sliceIndex = -1;   // recentre
            }
            ImGui::PopStyleColor(2);
        }
        ImGui::SameLine();
        int idx = app.sliceIndex;
        const int maxIdx = int(app.sliceMaxIndex());
        ImGui::SetNextItemWidth(S(140));
        if (ImGui::SliderInt("##sliceidx", &idx, 0, maxIdx, "slice %d"))
            app.sliceIndex = idx;
    }

    // ── Live pill (top-right) ───────────────────────────────────────────────
    if (app.modelLoaded) {
        const char* txt = app.running ? "LIVE" : "PAUSED";
        const ImVec2 ts = ImGui::CalcTextSize(txt);
        const ImVec2 pp{ pos.x + size.x - ts.x - S(34), pos.y + S(12) };
        dl->AddRectFilled(pp, {pp.x + ts.x + S(22), pp.y + ts.y + S(8)},
                          u32(kBg1, 0.9f), S(10));
        const float pulse = app.running
            ? 0.6f + 0.4f * std::sin(float(ImGui::GetTime()) * 4.f) : 0.35f;
        dl->AddCircleFilled({pp.x + S(10), pp.y + ts.y*0.5f + S(4)}, S(3.5f),
                            u32(app.running ? kAccent : kDim, pulse));
        dl->AddText({pp.x + S(18), pp.y + S(4)},
                    u32(app.running ? kAccent : kDim), txt);
    }

    // ── Help overlay ────────────────────────────────────────────────────────
    if (app.showHelp) {
        const ImVec2 hc{ pos.x + size.x*0.5f, pos.y + size.y*0.5f };
        const ImVec2 hs{ S(330), S(252) };
        dl->AddRectFilled({hc.x - hs.x*0.5f, hc.y - hs.y*0.5f},
                          {hc.x + hs.x*0.5f, hc.y + hs.y*0.5f},
                          u32(kBg1, 0.97f), S(8));
        dl->AddRect({hc.x - hs.x*0.5f, hc.y - hs.y*0.5f},
                    {hc.x + hs.x*0.5f, hc.y + hs.y*0.5f}, u32(kStroke), S(8));
        float ty = hc.y - hs.y*0.5f + S(14);
        auto line = [&](const char* k, const char* v) {
            dl->AddText({hc.x - hs.x*0.5f + S(18), ty}, u32(kAccent), k);
            dl->AddText({hc.x - hs.x*0.5f + S(110), ty}, u32(kText), v);
            ty += S(21);
        };
        dl->AddText({hc.x - ImGui::CalcTextSize("Keyboard shortcuts").x*0.5f, ty},
                    u32(kText), "Keyboard shortcuts");
        ty += S(28);
        line("Space",  "Run / pause");
        line("R",      "Reset flow");
        line("1 - 4",  "Visualization mode");
        line("[ / ]",  "Move slice plane");
        line("Tab",    "Toggle left panel");
        line("F",      "Toggle right panel");
        line("S",      "Save snapshot");
        line("F11",    "Fullscreen");
        line("Esc",    "Reset view / close help");
        line("?",      "Toggle this help");
    }

    ImGui::End();
    ImGui::PopStyleVar();
    ImGui::PopStyleColor();
}

void drawStatusBar(App& app, const Layout& l) {
    ImGui::SetNextWindowPos({0, l.display.y - l.statusH});
    ImGui::SetNextWindowSize({l.display.x, l.statusH});
    ImGui::PushStyleColor(ImGuiCol_WindowBg, kBg1);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {S(12), S(5)});
    ImGui::Begin("##status", nullptr, kPanelFlags);

    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImVec2 p = ImGui::GetCursorScreenPos();
    const float pulse = app.running
        ? 0.6f + 0.4f * std::sin(float(ImGui::GetTime()) * 4.f) : 0.4f;
    dl->AddCircleFilled({p.x + S(5), p.y + S(8)}, S(4),
                        u32(app.running ? kAccent : kDim, pulse));
    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + S(16));

    ImGui::PushStyleColor(ImGuiCol_Text, kDim);
    char buf[256];
    std::snprintf(buf, sizeof(buf),
        "%s   |   step %llu   |   %ux%ux%u   |   %s%s   |   tau %.3f   |   zoom %.0f%%",
        app.running ? "Running" : (app.modelLoaded ? "Paused" : "Idle"),
        static_cast<unsigned long long>(app.totalSteps),
        app.params.gx, app.params.gy, app.params.gz,
        app.params.collision ? "Regularized" : "BGK",
        app.params.les ? " + LES" : "",
        app.params.tau, app.zoom * 100.f);
    ImGui::TextUnformatted(buf);
    ImGui::PopStyleColor();

    // Status toast (right side)
    if (app.statusTtl > 0.f && !app.statusMsg.empty()) {
        const float w = ImGui::CalcTextSize(app.statusMsg.c_str()).x;
        ImGui::SameLine(ImGui::GetWindowWidth() - w - S(16));
        ImGui::PushStyleColor(ImGuiCol_Text, kAccent);
        ImGui::TextUnformatted(app.statusMsg.c_str());
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
    s.FramePadding      = {8.f, 5.f};
    s.ItemSpacing       = {8.f, 6.f};
    s.ItemInnerSpacing  = {6.f, 4.f};
    s.GrabMinSize       = 8.f;
    s.ScrollbarSize     = 11.f;

    ImVec4* c = s.Colors;
    c[ImGuiCol_WindowBg]            = kBg1;
    c[ImGuiCol_ChildBg]             = {0, 0, 0, 0};
    c[ImGuiCol_PopupBg]             = kBg1;
    c[ImGuiCol_Border]              = kStroke;
    c[ImGuiCol_Text]                = kText;
    c[ImGuiCol_TextDisabled]        = kDim;
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
    c[ImGuiCol_ButtonHovered]       = {kAccent.x, kAccent.y, kAccent.z, 0.18f};
    c[ImGuiCol_ButtonActive]        = {kAccent.x, kAccent.y, kAccent.z, 0.30f};
    c[ImGuiCol_Header]              = kBg2;
    c[ImGuiCol_HeaderHovered]       = {kAccent.x, kAccent.y, kAccent.z, 0.16f};
    c[ImGuiCol_HeaderActive]        = {kAccent.x, kAccent.y, kAccent.z, 0.26f};
    c[ImGuiCol_Separator]           = kStroke;
    c[ImGuiCol_PlotLines]           = kAccent;
    c[ImGuiCol_PlotHistogram]       = kAccent;

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
    }, 27.f);

    if (!app.fonts.body) app.fonts.body = io.Fonts->AddFontDefault();
    io.FontGlobalScale = 1.f;
    io.FontDefault = app.fonts.body;
}

void draw(App& app) {
    const Layout l = computeLayout(app);
    drawTopBar(app, l);
    drawLeftPanel(app, l);
    drawRightPanel(app, l);
    drawViewport(app, l);
    drawStatusBar(app, l);
}

} // namespace vwt::ui
