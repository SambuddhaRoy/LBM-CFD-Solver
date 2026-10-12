// ============================================================================
// uitest.cpp — scripted GUI interaction test (see uitest.hpp)
// ============================================================================

#include "uitest.hpp"

#include "app.hpp"

#include <GLFW/glfw3.h>
#include <imgui_internal.h>

#include <cmath>
#include <cstdio>
#include <filesystem>

namespace wt {

UiTest::UiTest(std::string outDir) : outDir_(std::move(outDir)) {
    std::filesystem::create_directories(outDir_);
}

void UiTest::place(const char* name) {
    items_[name] = {ImGui::GetItemRectMin(), ImGui::GetItemRectMax()};
}

ImVec2 UiTest::at(const std::string& item, float fx, float fy) {
    const auto it = items_.find(item);
    if (it == items_.end()) {
        expect(false, "control not found: " + item);
        return mouse_;
    }
    const Rect& r = it->second;
    return {r.lo.x + (r.hi.x - r.lo.x) * fx, r.lo.y + (r.hi.y - r.lo.y) * fy};
}

// ─── Gestures ───────────────────────────────────────────────────────────────
// Positions are looked up when the gesture's frame runs, so a control that
// moved (a header collapsed, a row appeared) is still hit.

void UiTest::moveTo(const std::string& item, float fx, float fy) {
    frame([=, this](App&) {
        mouse_ = at(item, fx, fy);
        ImGui::GetIO().AddMousePosEvent(mouse_.x, mouse_.y);
    });
}

void UiTest::click(const std::string& item, float fx, ImGuiMouseButton b) {
    moveTo(item, fx);
    frame([b](App&) { ImGui::GetIO().AddMouseButtonEvent(b, true); });
    frame([b](App&) { ImGui::GetIO().AddMouseButtonEvent(b, false); });
    wait(2);
}

// Ctrl+click turns a slider into a text field with its value selected;
// typing replaces it and Enter commits, which ImGui reports as the slider
// being released after an edit, exactly as a drag ends.
void UiTest::typeInto(const std::string& item, const std::string& text) {
    moveTo(item);
    frame([](App&) {
        ImGui::GetIO().AddKeyEvent(ImGuiMod_Ctrl, true);
        ImGui::GetIO().AddKeyEvent(ImGuiKey_LeftCtrl, true);
    });
    frame([](App&) { ImGui::GetIO().AddMouseButtonEvent(0, true); });
    frame([](App&) {
        ImGui::GetIO().AddMouseButtonEvent(0, false);
        ImGui::GetIO().AddKeyEvent(ImGuiKey_LeftCtrl, false);
        ImGui::GetIO().AddKeyEvent(ImGuiMod_Ctrl, false);
    });
    wait(1);
    frame([text](App&) { for (char c : text) ImGui::GetIO().AddInputCharacter(c); });
    frame([](App&) { ImGui::GetIO().AddKeyEvent(ImGuiKey_Enter, true); });
    frame([](App&) { ImGui::GetIO().AddKeyEvent(ImGuiKey_Enter, false); });
    wait(2);
}

void UiTest::drag(const std::string& item, float fx0, float fx1, int frames) {
    moveTo(item, fx0);
    frame([](App&) { ImGui::GetIO().AddMouseButtonEvent(0, true); });
    for (int i = 1; i <= frames; ++i) {
        const float fx = fx0 + (fx1 - fx0) * float(i) / float(frames);
        moveTo(item, fx);
    }
    frame([](App&) { ImGui::GetIO().AddMouseButtonEvent(0, false); });
    wait(2);
}

// Opens the combo, then clicks entry `index` of its popup: entries are one
// font height apart plus the item spacing, from the popup's content start.
void UiTest::pickCombo(const std::string& item, int index) {
    click(item);
    frame([=, this](App&) {
        ImGuiContext& g = *ImGui::GetCurrentContext();
        if (g.OpenPopupStack.empty() || !g.OpenPopupStack.back().Window) {
            expect(false, "combo did not open: " + item);
            return;
        }
        const ImGuiWindow* w = g.OpenPopupStack.back().Window;
        const float row = ImGui::GetFontSize() + ImGui::GetStyle().ItemSpacing.y;
        mouse_ = {w->Pos.x + 0.5f * w->Size.x, w->DC.CursorStartPos.y + row * float(index) + 0.5f * ImGui::GetFontSize()};
        ImGui::GetIO().AddMousePosEvent(mouse_.x, mouse_.y);
    });
    frame([](App&) { ImGui::GetIO().AddMouseButtonEvent(0, true); });
    frame([](App&) { ImGui::GetIO().AddMouseButtonEvent(0, false); });
    wait(2);
}

void UiTest::key(ImGuiKey k) {
    frame([k](App&) { ImGui::GetIO().AddKeyEvent(k, true); });
    frame([k](App&) { ImGui::GetIO().AddKeyEvent(k, false); });
    wait(1);
}

void UiTest::wheel(float fx, float fy, float amount) {
    moveTo("Viewport", fx, fy);
    frame([amount](App&) { ImGui::GetIO().AddMouseWheelEvent(0, amount); });
    wait(1);
}

void UiTest::dragViewport(float fx0, float fy0, float fx1, float fy1, ImGuiMouseButton b) {
    moveTo("Viewport", fx0, fy0);
    frame([b](App&) { ImGui::GetIO().AddMouseButtonEvent(b, true); });
    const int n = 30;
    for (int i = 1; i <= n; ++i) {
        const float s = float(i) / n;
        moveTo("Viewport", fx0 + (fx1 - fx0) * s, fy0 + (fy1 - fy0) * s);
    }
    frame([b](App&) { ImGui::GetIO().AddMouseButtonEvent(b, false); });
    wait(1);
}

void UiTest::expect(bool ok, const std::string& what) {
    if (ok) return;
    ++stepFailures_;
    notes_.push_back(what);
}

// ─── Driver ─────────────────────────────────────────────────────────────────

// Runs after the GLFW backend has queued this frame's real events, so the
// scripted ones come last and win.
void UiTest::input(App& app) {
    if (mouse_.x >= 0) ImGui::GetIO().AddMousePosEvent(mouse_.x, mouse_.y);
    if (!started_ || done() || queue_.empty()) return;
    Action a = std::move(queue_.front());
    queue_.pop_front();
    a(app);
}

void UiTest::begin(App& app) {
    const Step& s = steps_[step_];
    rebuilds0_ = app.rebuildCount_;
    resets0_ = app.resetCount_;
    validation0_ = gpu::validationMessages;
    minViewportFrac_ = 1.f;
    stepFailures_ = 0;
    notes_.clear();
    settling_ = false;
    if (s.act) s.act(app);
}

void UiTest::afterFrame(App& app) {
    if (!started_) {
        if (ImGui::GetFrameCount() < 60) return;      // let the app settle in
        started_ = true;
        std::printf("\nGUI interaction test: %zu steps, screenshots in %s\n\n", steps_.size(), outDir_.c_str());
        begin(app);
        return;
    }
    if (done()) return;

    // Monitors, every frame. The viewport must fill the space between the
    // side panels at all times; a transient collapse would show here.
    ImGuiWindow* controls = ImGui::FindWindowByName("Controls");
    ImGuiWindow* stats = ImGui::FindWindowByName("Stats");
    const ImGuiIO& io = ImGui::GetIO();
    if (controls && stats) {
        const float w = (stats->Pos.x - (controls->Pos.x + controls->Size.x)) * io.DisplayFramebufferScale.x;
        const float h = io.DisplaySize.y * io.DisplayFramebufferScale.y;
        if (w > 1 && h > 1)
            minViewportFrac_ = std::min({minViewportFrac_, app.viewportW_ / w, app.viewportH_ / h});
    }

    // Gestures run one per frame (input()), then the step settles.
    if (!queue_.empty()) return;
    if (!settling_) {
        settling_ = true;
        settleUntil_ = glfwGetTime() + steps_[step_].settle;
        return;
    }
    if (glfwGetTime() < settleUntil_) return;
    finish(app);
    if (++step_ < steps_.size()) begin(app);
    else {
        std::printf("\n%s: %d of %zu steps failed\n", failures_ ? "GUI TEST FAILED" : "GUI TEST PASSED",
                    failures_, steps_.size());
        glfwSetWindowShouldClose(app.window_, GLFW_TRUE);
    }
}

void UiTest::finish(App& app) {
    const Step& s = steps_[step_];
    const int rebuilds = int(app.rebuildCount_ - rebuilds0_);
    const int resets = int(app.resetCount_ - resets0_) - rebuilds;      // every rebuild resets too
    if (s.rebuilds >= 0)
        expect(rebuilds == s.rebuilds, "solver rebuilt " + std::to_string(rebuilds) + "x, expected " +
                                           std::to_string(s.rebuilds));
    if (s.resets >= 0)
        expect(resets == s.resets, "flow reset " + std::to_string(resets) + "x, expected " +
                                       std::to_string(s.resets));
    expect(gpu::validationMessages == validation0_,
           std::to_string(gpu::validationMessages - validation0_) + " Vulkan validation messages");
    expect(minViewportFrac_ > 0.99f, "viewport shrank to " + std::to_string(int(100 * minViewportFrac_)) +
                                         "% of the space between the panels");

    // Flow and body sanity.
    const auto& g = app.solver_.grid();
    const Stats& st = app.stats_;
    if (app.running_ && app.solver_.t > 200) {
        expect(std::isfinite(st.maxU) && st.maxU < 0.39f, "flow diverged (peak lattice speed " +
                                                              std::to_string(st.maxU) + ")");
        expect(std::abs(st.rhoMean - 1.f) < 0.05f, "mean density drifted to " + std::to_string(st.rhoMean));
    }
    const glm::vec3 c = app.body_.center;
    expect(c.x >= 0 && c.y >= 0 && c.z >= 0 && c.x <= float(g.nx) && c.y <= float(g.ny) && c.z <= float(g.nz),
           "body centre outside the tunnel");
    expect(app.body_.length > 0 && app.geometry_.frontalCells() > 0, "no body in the tunnel");
    for (const char* bad : {"diverged", "Could not", "Grid not created", "failed"})
        if (app.status_.find(bad) != std::string::npos && glfwGetTime() < app.statusUntil_)
            expect(false, "status: " + app.status_);
    if (app.realtime_) {
        expect(g.nx == app.rtGrid_[0] && g.ny == app.rtGrid_[1] && g.nz == app.rtGrid_[2],
               "real time on but the grid is not the fitted tunnel");
        if (s.settle >= 5) expect(app.rtRatio_ >= 0.9f, "real time running at x" + std::to_string(app.rtRatio_));
    }
    if (s.check) s.check(app);

    // How much of the viewport the tunnel fills (2D view): a thin strip
    // means most of the screen shows nothing.
    char fill[96] = "";
    if (app.view_.mode3d == 0) {
        const glm::vec3 n(float(g.nx), float(g.ny), float(g.nz));
        const int a = app.view_.axis;
        const glm::vec2 plane = a == 2 ? glm::vec2(n.x, n.y) : a == 1 ? glm::vec2(n.x, n.z) : glm::vec2(n.z, n.y);
        const float fw = std::min(1.f, plane.x / app.view_.cellsPerPixel / app.viewportW_);
        const float fh = std::min(1.f, plane.y / app.view_.cellsPerPixel / app.viewportH_);
        std::snprintf(fill, sizeof fill, "  view fill %3.0f%% x %3.0f%%", 100 * fw, 100 * fh);
    }
    char rt[64] = "";
    if (app.realtime_) std::snprintf(rt, sizeof rt, "  real time x%.2f, %.0f cells", app.rtRatio_, app.body_.length);
    char shot[160];
    std::snprintf(shot, sizeof shot, "%s/%02zu.png", outDir_.c_str(), step_ + 1);
    shotPath = shot;

    if (stepFailures_) ++failures_;
    std::printf("%s %2zu  %-46s grid %ux%ux%u  rebuilds %d resets %d%s%s\n", stepFailures_ ? "FAIL" : "ok  ",
                step_ + 1, s.name.c_str(), g.nx, g.ny, g.nz, rebuilds, resets, fill, rt);
    for (const auto& n : notes_) std::printf("        - %s\n", n.c_str());
}

// ─── Scenario ───────────────────────────────────────────────────────────────

void UiTest::buildScenario(const std::string& meshPath) {
    auto add = [this](std::string name, Action act, double settle, int rebuilds, int resets, Action check) {
        steps_.push_back({std::move(name), std::move(act), settle, rebuilds, resets, std::move(check)});
    };
    auto near = [](double a, double b, double tol = 1e-3) { return std::abs(a - b) <= tol * std::max(1.0, std::abs(b)); };
    static float before = 0;   // a value captured by an action for its check

    // ── Non-real-time basics ──────────────────────────────────────────────
    add("start-up", nullptr, 2, 0, 0, nullptr);
    add("Pause button", [this](App&) { click("Pause"); }, 0.5, 0, 0,
        [this](App& a) { expect(!a.running_, "still running"); });
    add("Space runs again", [this](App&) { key(ImGuiKey_Space); }, 0.5, 0, 0,
        [this](App& a) { expect(a.running_, "not running"); });
    add("Reset flow button", [this](App&) { click("Reset flow"); }, 0.3, 0, 1, nullptr);
    add("R resets", [this](App&) { key(ImGuiKey_R); }, 0.3, 0, 1, nullptr);
    add("field keys 2, 3, 4", [this](App&) { key(ImGuiKey_2); key(ImGuiKey_3); key(ImGuiKey_4); }, 0.5, 0, 0,
        [this](App& a) { expect(a.view_.field == Field::QCriterion, "field not Q"); });
    add("Field combo: pressure", [this](App&) { pickCombo("Field", 1); }, 0.5, 0, 0,
        [this](App& a) { expect(a.view_.field == Field::Pressure, "field not pressure"); });
    add("Absolute pressure on", [this](App&) { click("Absolute pressure"); }, 0.5, 0, 0,
        [this](App& a) { expect(a.absolutePressure_, "absolute pressure off"); });
    add("Absolute pressure off, key 1", [this](App&) { click("Absolute pressure"); key(ImGuiKey_1); }, 0.5, 0, 0,
        [this](App& a) { expect(!a.absolutePressure_ && a.view_.field == Field::Speed, "field/absolute wrong"); });
    add("V: 3D view", [this](App&) { key(ImGuiKey_V); }, 1, 0, 0,
        [this](App& a) { expect(a.view_.mode3d == 1, "not 3D"); });
    add("3D orbit, pan, zoom",
        [this](App& a) {
            before = a.view_.yaw;
            dragViewport(0.5f, 0.5f, 0.7f, 0.4f);
            dragViewport(0.5f, 0.5f, 0.45f, 0.55f, 1);
            wheel(0.5f, 0.5f, 3);
        }, 1, 0, 0,
        [this](App& a) { expect(a.view_.yaw != before, "orbit did not turn the camera"); });
    add("2D slice radio", [this](App&) { click("2D slice"); }, 0.5, 0, 0,
        [this](App& a) { expect(a.view_.mode3d == 0, "not 2D"); });
    add("wheel zoom at cursor",
        [this](App& a) { before = a.view_.cellsPerPixel; wheel(0.35f, 0.5f, 5); }, 0.5, 0, 0,
        [this](App& a) { expect(a.view_.cellsPerPixel < 0.6f * before, "did not zoom in"); });
    add("drag pan", [this](App& a) { before = a.view_.center.x; dragViewport(0.5f, 0.5f, 0.3f, 0.6f); }, 0.5, 0, 0,
        [this](App& a) { expect(a.view_.center.x != before, "did not pan"); });
    add("H fits the view", [this](App&) { key(ImGuiKey_H); }, 0.5, 0, 0, nullptr);
    add("slice normal X", [this](App&) { click("X"); }, 0.5, 0, 0,
        [this](App& a) { expect(a.view_.axis == 0 && a.view_.slice == a.body_.center.x, "X slice wrong"); });
    add("slice normal Y", [this](App&) { click("Y"); }, 0.5, 0, 0,
        [this](App& a) { expect(a.view_.axis == 1, "Y slice wrong"); });
    add("slice normal Z", [this](App&) { click("Z"); }, 0.5, 0, 0,
        [this](App& a) { expect(a.view_.axis == 2, "Z slice wrong"); });
    add("Slice typed 50, then [ and ]",
        [this](App&) { typeInto("Slice", "50"); key(ImGuiKey_LeftBracket); key(ImGuiKey_LeftBracket); key(ImGuiKey_RightBracket); },
        0.5, 0, 0, [this](App& a) { expect(a.view_.slice == 49.f, "slice " + std::to_string(a.view_.slice)); });
    add("Shape: cube", [this](App&) { pickCombo("Shape", 1); }, 1, 0, 1,
        [this](App& a) { expect(a.body_.shape == Shape::Cube, "not a cube"); });
    add("Shape: cylinder, Span typed 40", [this](App&) { pickCombo("Shape", 2); typeInto("Span (cells)", "40"); }, 1, 0, 1,
        [this](App& a) { expect(a.body_.shape == Shape::Cylinder && a.body_.span == 40.f, "cylinder/span wrong"); });
    add("Shape: wing", [this](App&) { pickCombo("Shape", 3); }, 1, 0, 1,
        [this](App& a) { expect(a.body_.shape == Shape::Wing, "not a wing"); });
    add("Shape: sphere", [this](App&) { pickCombo("Shape", 0); }, 1, 0, 1,
        [this](App& a) { expect(a.body_.shape == Shape::Sphere, "not a sphere"); });
    add("Size, angles, position typed",
        [this](App&) {
            typeInto("Size (cells)", "40"); typeInto("Pitch", "15"); typeInto("Yaw", "10");
            typeInto("Roll", "5"); typeInto("Position x", "120");
        }, 1, 0, 0,
        [this](App& a) {
            expect(a.body_.length == 40.f && a.body_.pitch == 15.f && a.body_.yaw == 10.f && a.body_.roll == 5.f &&
                   a.body_.center.x == 120.f, "typed body values not applied");
        });
    add("Zero angles, Size dragged",
        [this](App&) { click("Zero angles"); drag("Size (cells)", 0.15f, 0.3f); }, 1, 0, 0,
        [this](App& a) {
            expect(a.body_.pitch == 0 && a.body_.yaw == 0 && a.body_.roll == 0, "angles not zero");
            expect(a.body_.length != 40.f, "drag did not change the size");
        });
    add("Size 48, Position x 115", [this](App&) { typeInto("Size (cells)", "48"); typeInto("Position x", "115"); }, 1, 0, 0,
        nullptr);
    add("Fluid combo: water, CO2, air",
        [this](App&) { pickCombo("Fluid", 1); pickCombo("Fluid", 2); pickCombo("Fluid", 0); }, 0.5, 0, 0,
        [this](App& a) { expect(a.ambient_.kind == FluidKind::Air, "not air"); });
    add("Temperature 40 C, pressure 50 kPa",
        [this](App&) { typeInto("Temperature", "40"); typeInto("Pressure", "50"); }, 0.5, 0, 0,
        [this, near](App& a) {
            expect(near(a.ambient_.T, 313.15) && near(a.ambient_.p, 50000), "T/p not applied");
            expect(near(a.props_.rho, 50000 / (287.05 * 313.15), 0.01), "density does not follow T/p");
        });
    add("ISA altitude 5000 m", [this](App&) { typeInto("ISA altitude", "5000"); }, 0.5, 0, 0,
        [this, near](App& a) { expect(near(a.ambient_.T, 255.65, 1e-3), "ISA T " + std::to_string(a.ambient_.T)); });
    add("Mars, Water, Sea level buttons",
        [this](App&) { click("Mars"); click("Water"); click("Sea level"); }, 0.5, 0, 0,
        [this, near](App& a) {
            expect(a.ambient_.kind == FluidKind::Air && near(a.ambient_.T, 288.15) && near(a.ambient_.p, 101325),
                   "sea level not restored");
        });
    add("Wind speed 60, body length 2 m",
        [this](App&) { typeInto("Wind speed", "60"); typeInto("Body length", "2"); }, 1, 0, 0,
        [this, near](App& a) {
            expect(a.speed_ == 60.f && a.length_ == 2.f, "speed/length not applied");
            expect(near(a.reRequested_, 60.0 * 2.0 / a.props_.nu, 1e-3), "Re does not follow");
        });
    add("Wind speed dragged", [this](App& a) { before = a.speed_; drag("Wind speed", 0.5f, 0.7f); }, 1, 0, 0,
        [this](App& a) { expect(a.speed_ != before, "drag did not change the speed"); });
    add("Wind speed 30, length 1", [this](App&) { typeInto("Wind speed", "30"); typeInto("Body length", "1"); }, 0.5, 0, 0,
        nullptr);
    add("Lattice speed 0.12", [this](App&) { typeInto("Lattice speed", "0.12"); }, 1, 0, 0,
        [this, near](App& a) { expect(near(a.solver_.flow.uIn, 0.12), "lattice speed not applied"); });
    add("Lattice speed 0.1", [this](App&) { typeInto("Lattice speed", "0.1"); }, 0.5, 0, 0, nullptr);
    add("Flow model: laminar", [this](App&) { pickCombo("Flow model", 1); }, 2, 0, 0,
        [this](App& a) { expect(a.solver_.flow.smagorinsky == 0 && a.solver_.flow.tau > 0.52f, "laminar not bounded"); });
    add("Flow model: LES", [this](App&) { pickCombo("Flow model", 2); }, 1, 0, 0,
        [this](App& a) { expect(a.solver_.flow.smagorinsky > 0, "LES off"); });
    add("Flow model: auto", [this](App&) { pickCombo("Flow model", 0); }, 1, 0, 0,
        [this](App& a) { expect(a.flowModel_ == 0, "not auto"); });
    add("Interpolated walls off/on, inlet turbulence",
        [this](App&) { click("Interpolated walls"); click("Interpolated walls"); typeInto("Inlet turbulence", "0.02"); }, 1, 0, 0,
        [this](App& a) { expect(a.solver_.flow.bouzidi && a.solver_.flow.turbulence > 0.019f, "walls/turbulence wrong"); });
    add("Inlet turbulence 0", [this](App&) { typeInto("Inlet turbulence", "0"); }, 0.5, 0, 0, nullptr);
    add("Cells combo: smallest grid", [this](App&) { pickCombo("Cells", 0); }, 1.5, 1, 0,
        [this](App& a) { expect(a.solver_.grid().nx == kGrids[0].nx && a.body_.length > 20, "grid/body wrong"); });
    add("Cells combo: default grid", [this](App&) { pickCombo("Cells", 1); }, 1.5, 1, 0,
        [this](App& a) { expect(a.solver_.grid().nx == kGrids[1].nx && a.body_.length == 48.f,
                                "body not scaled back: " + std::to_string(a.body_.length)); });
    add("Storage: FP32", [this](App&) { pickCombo("Storage", 0); }, 1.5, 1, 0,
        [this](App& a) { expect(a.precision_ == Precision::FP32, "not FP32"); });
    add("Storage: FP16S", [this](App&) { pickCombo("Storage", 1); }, 1, 1, 0, nullptr);
    add("Storage: FP16C", [this](App&) { pickCombo("Storage", 2); }, 1, 1, 0,
        [this](App& a) { expect(a.precision_ == Precision::FP16C, "not FP16C"); });
    add("Auto colour range off, Low/High, on",
        [this](App&) { click("Auto colour range"); drag("High", 0.5f, 0.6f, 10); click("Auto colour range"); }, 0.5, 0, 0,
        [this](App& a) { expect(a.autoRange_, "auto range off"); });
    add("Range scale 2, lattice overlay off/on",
        [this](App&) { typeInto("Range scale", "2"); click("Show lattice when zoomed"); click("Show lattice when zoomed"); }, 0.5, 0, 0,
        [this](App& a) { expect(a.rangeScale_ == 2.f && a.view_.grid, "range/overlay wrong"); });
    add("Fit view button", [this](App&) { typeInto("Range scale", "1"); wheel(0.5f, 0.5f, 4); click("Fit view"); }, 0.5, 0, 0,
        nullptr);
    add("Snapshot button", [this](App&) { click("Snapshot"); }, 0.5, 0, 0,
        [this](App& a) {
            const auto pos = a.status_.find("snapshots/");
            const bool ok = pos != std::string::npos && std::filesystem::exists(a.status_.substr(pos));
            expect(ok, "snapshot not written: " + a.status_);
            if (ok) std::filesystem::remove(a.status_.substr(pos));
        });
    add("collapse and reopen every header",
        [this](App&) {
            for (const char* h : {"Body", "Flow", "Grid", "View"}) { click(h); click(h); }
        }, 0.5, 0, 0, nullptr);
    add("window restored to 1280 x 760",
        [this](App& a) {
            frame([&a](App&) { glfwRestoreWindow(a.window_); glfwSetWindowSize(a.window_, 1280, 760); });
            wait(10);
        }, 1.5, 0, 0, [this](App& a) {
            int w = 0, h = 0;
            glfwGetFramebufferSize(a.window_, &w, &h);
            expect(a.swapExtent_.width == uint32_t(w) && a.swapExtent_.height == uint32_t(h), "swapchain not resized");
        });
    add("window maximized", [this](App& a) { frame([&a](App&) { glfwMaximizeWindow(a.window_); }); wait(10); }, 1.5, 0, 0,
        nullptr);
    if (!meshPath.empty())
        add("model dropped on the window", [meshPath](App& a) { a.loadMeshFile(meshPath); }, 3, 0, 1,
            [this](App& a) {
                expect(a.body_.shape == Shape::Mesh, "not the model");
                expect(std::abs(a.length_ - 5.612f) < 0.01f, "glTF length not used: " + std::to_string(a.length_));
            });

    // ── Real time ──────────────────────────────────────────────────────────
    static Body saved;
    static int savedGrid = 0;
    add("Real time on",
        [this](App& a) { saved = a.body_; savedGrid = a.gridIndex_; click("Real time"); }, 6, 1, 0,
        [this](App& a) { expect(a.realtime_ && a.body_.length == a.rtLength_, "real time body not planned"); });
    add("real time: idle 10 s (no refits)", nullptr, 10, 0, 0, nullptr);
    add("real time: 3D view, orbiting 4 s",
        [this](App&) {
            key(ImGuiKey_V);
            for (int i = 0; i < 6; ++i) dragViewport(0.4f, 0.5f, 0.6f, 0.45f);
        }, 6, 0, 0, nullptr);
    add("real time: back to 2D", [this](App&) { key(ImGuiKey_V); }, 2, 0, 0, nullptr);
    add("real time: zoom and pan stay put",
        [this](App& a) { wheel(0.4f, 0.5f, 4); dragViewport(0.5f, 0.5f, 0.6f, 0.5f); frame([&a](App&) { before = a.view_.cellsPerPixel; }); },
        6, 0, 0, [this](App& a) { expect(a.view_.cellsPerPixel == before, "the view was reset"); });
    add("real time: H fits the view", [this](App&) { key(ImGuiKey_H); }, 1, 0, 0, nullptr);
    add("real time: Size slider typed",
        [this](App& a) { before = a.body_.length; typeInto("Size (cells)", "30"); }, 8, 0, 0, nullptr);
    add("real time: pitch 5 deg", [this](App&) { typeInto("Pitch", "5"); }, 6, 0, 0,
        [this](App& a) { expect(a.body_.pitch == 5.f, "pitch not applied"); });
    add("real time: wind speed 60 (refit)", [this](App&) { typeInto("Pitch", "0"); typeInto("Wind speed", "60"); }, 6, 1, 0,
        [this](App& a) { expect(a.speed_ == 60.f, "speed not applied"); });
    add("real time: wind speed dragged 2 s",
        [this](App&) { drag("Wind speed", 0.6f, 0.65f, 60); drag("Wind speed", 0.65f, 0.6f, 60); }, 6, -1, 0, nullptr);
    add("real time: wind speed 30", [this](App&) { typeInto("Wind speed", "30"); }, 6, -1, 0, nullptr);
    add("real time: lattice speed is locked", [this](App&) { typeInto("Lattice speed", "0.05"); }, 1, 0, 0,
        [this](App& a) { expect(a.solver_.flow.uIn == 0.1f, "lattice speed changed in real time"); });
    add("real time: window resize",
        [this](App& a) {
            frame([&a](App&) { glfwRestoreWindow(a.window_); glfwSetWindowSize(a.window_, 1400, 800); });
            wait(10);
            frame([&a](App&) { glfwMaximizeWindow(a.window_); });
            wait(10);
        }, 6, 0, 0, nullptr);
    add("real time: shape sphere (refit)", [this](App&) { pickCombo("Shape", 0); }, 6, 1, 1, nullptr);
    add("real time: storage FP32 and back", [this](App&) { pickCombo("Storage", 0); wait(30); pickCombo("Storage", 2); }, 6, 2, 0,
        nullptr);
    add("Real time off",
        [this](App&) { click("Real time"); }, 2, 1, 0,
        [this](App& a) {
            expect(!a.realtime_ && a.gridIndex_ == savedGrid, "preset not restored");
            expect(a.body_.length == saved.length && a.body_.center == saved.center,
                   "body not restored: length " + std::to_string(a.body_.length) + " vs " + std::to_string(saved.length));
        });
    if (!meshPath.empty()) {
        add("Real time on, then a model dropped",
            [this, meshPath](App&) { click("Real time"); wait(30); frame([meshPath](App& b) { b.loadMeshFile(meshPath); }); },
            6, 2, -1, [this](App& a) { expect(a.body_.shape == Shape::Mesh && a.realtime_, "model/real time lost"); });
        add("real time: idle 10 s with the model", nullptr, 10, 0, 0, nullptr);
        add("Real time off again", [this](App&) { click("Real time"); }, 2, 1, 0,
            [this](App& a) { expect(!a.realtime_ && a.body_.shape == Shape::Mesh, "state lost"); });
    }
}

} // namespace wt
