#pragma once
// ============================================================================
// uitest.hpp — scripted GUI interaction test (--ui-test).
//
// Drives the real app through Dear ImGui's input queue: mouse moves, clicks
// and drags on controls found by name, typed slider values (Ctrl+click),
// combo picks, wheel, keys, window resizes and file drops. After every step
// it checks invariants (viewport geometry, view fit, rebuild/reset counts,
// flow sanity, real-time pacing, Vulkan validation errors) and saves a
// screenshot, so a run is both a pass/fail test and a visual record.
// ============================================================================

#include <imgui.h>

#include <deque>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

namespace wt {

class App;

struct UiTest {
    using Action = std::function<void(App&)>;
    struct Step {
        std::string name;
        Action      act;             // queues gestures
        double      settle = 1.0;    // seconds to run once the gestures are done
        int         rebuilds = 0;    // expected solver rebuilds (-1: any)
        int         resets = -1;     // expected flow resets beyond rebuilds (-1: any)
        Action      check;           // extra assertions, via expect()
    };

    explicit UiTest(std::string outDir);

    // Hooks called by App.
    void place(const char* name);    // records the last drawn item's rectangle
    void input(App& app);            // before ImGui::NewFrame: feeds this frame's events
    void afterFrame(App& app);       // after present: monitors, checks, advances
    bool done() const { return step_ >= steps_.size(); }
    int  result() const { return failures_ == 0 ? 0 : 1; }
    std::string shotPath;            // set when this frame should be saved

    // Gestures, queued one frame apart.
    void frame(Action a) { queue_.push_back(std::move(a)); }
    void wait(int frames) { for (int i = 0; i < frames; ++i) frame([](App&) {}); }
    void moveTo(const std::string& item, float fx = 0.5f, float fy = 0.5f);
    void click(const std::string& item, float fx = 0.5f, ImGuiMouseButton b = 0);
    void typeInto(const std::string& item, const std::string& text);   // Ctrl+click, type, Enter
    void drag(const std::string& item, float fx0, float fx1, int frames = 20);
    void pickCombo(const std::string& item, int index);
    void key(ImGuiKey k);
    void wheel(float fx, float fy, float amount);      // over the viewport, fractions of it
    void dragViewport(float fx0, float fy0, float fx1, float fy1, ImGuiMouseButton b = 0);

    void expect(bool ok, const std::string& what);

    // The scenario: every control, both modes, resizes, model loading.
    void buildScenario(const std::string& meshPath);
    std::vector<Step> steps_;

private:
    struct Rect { ImVec2 lo, hi; };
    ImVec2 at(const std::string& item, float fx, float fy);
    void   begin(App& app);
    void   finish(App& app);

    std::string outDir_;
    std::unordered_map<std::string, Rect> items_;
    std::deque<Action> queue_;
    ImVec2   mouse_{-1, -1};
    size_t   step_ = 0;
    bool     started_ = false, settling_ = false;
    double   settleUntil_ = 0;
    uint64_t rebuilds0_ = 0, resets0_ = 0, validation0_ = 0;
    float    minViewportFrac_ = 1.f;   // smallest viewport size seen this step, vs expected
    int      failures_ = 0, stepFailures_ = 0;
    std::vector<std::string> notes_;
};

} // namespace wt
