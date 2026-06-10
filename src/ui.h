#pragma once
// ============================================================================
// ui.h — theme and panel drawing (all ImGui code lives in ui.cpp)
// ============================================================================

namespace vwt {

class App;

namespace ui {

// Style + colors; call once after ImGui::CreateContext().
void applyTheme(float dpiScale);

// Font set with per-platform fallbacks; call before backend init.
void loadFonts(App& app, float dpiScale);

// Draws the complete frame UI (top bar, panels, viewport, status bar).
void draw(App& app);

} // namespace ui
} // namespace vwt
