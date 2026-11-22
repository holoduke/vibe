#pragma once

#include <cstdint>
#include <string>

struct GLFWwindow;
struct VkSurfaceKHR_T;
struct VkInstance_T;

namespace vibe {

struct WindowConfig {
    uint32_t width = 1280;
    uint32_t height = 720;
    std::string title = "Vibe";
    bool capture_cursor = false;
};

// Everything the player can do that has a key or a button. The order is the order of the
// list in the menu (Controls); the names in window.cpp (kActionInfo) follow it.
enum class Action : int {
    Fwd, Back, Left, Right, Jump, Sprint, Crouch, Fire, Aim, Use, Reload, Flashlight,
    Weapon1, Weapon2, Fly, Camera, Home, Screenshot, BrushPrev, BrushSmaller, BrushLarger, Wireframe,
    Count
};
inline constexpr int kActionCount = static_cast<int>(Action::Count);

// What is bound to what, and how the controller behaves. Kept by the engine (saved with the
// settings) and handed to Window::consume_input every frame.
//   key[]: a GLFW key code, or kMouseBase + mouse button, or -1 (nothing)
//   pad[]: a GLFW gamepad button 0..14, kPadLT / kPadRT for the triggers, or -1
struct InputBindings {
    static constexpr int kMouseBase = 1000;
    static constexpr int kPadLT = 15, kPadRT = 16;
    int key[kActionCount];
    int pad[kActionCount];
    bool  invert_mouse_y   = false;
    bool  pad_enabled      = true;
    float pad_look_speed   = 200.0f;  // degrees a second at full stick
    float pad_look_ratio_y = 0.65f;   // up / down, as a part of that
    bool  pad_invert_y     = false;
    float pad_deadzone     = 0.16f;   // of the stick's travel
    float pad_curve        = 1.8f;    // 1 = linear; higher = finer near the centre
    bool  pad_sprint_toggle = true;   // one press sprints until the stick is let go
    bool  pad_swap_sticks  = false;   // left-handed: look left, move right
    InputBindings() { reset(); }
    void reset();                      // the bindings only (not the controller's tuning)
};
// For the menu and the settings file: "Jump" / "jump" for an action; "Space", "Mouse 1",
// "A", "RT" for a binding ("-" for none).
const char* action_label(Action a);
const char* action_id(Action a);
std::string key_binding_name(int code);
const char* pad_binding_name(int code);

struct InputFrame {
    // Movement as an amount (controller's stick; 0 when the keys are used): x right, y forward.
    float move_x = 0.0f, move_y = 0.0f;
    // The look stick, after dead zone and curve: x right, y down (-1..1).
    float look_x = 0.0f, look_y = 0.0f;
    bool flashlight = false;    // F, level; the engine edge-detects
    bool reload = false;        // R, level
    bool pad_connected = false;
    bool using_pad = false;     // the controller was the last thing touched
    bool fwd = false, back = false, left = false, right = false;
    bool jump = false;
    bool sprint = false;    // Left Shift — 1.5× walk speed
    bool crawl = false;     // Left Ctrl — 0.4× walk speed (sprint wins if both)
    bool menu_key = false;  // ESC; engine edge-detects to toggle pause
    bool edit_key = false;  // E; engine edge-detects to toggle terrain edit mode
    bool brush_smaller = false; // `[` — level; engine shrinks brush while held
    bool brush_larger  = false; // `]` — level; engine grows brush while held
    bool brush_mode_prev = false; // Q; engine edge-detects to cycle brush mode
    bool brush_mode_next = false; // R; engine edge-detects to cycle brush mode
    bool wireframe_key = false;   // 9; engine edge-detects to toggle terrain wireframe
    bool home_key = false;        // H; engine edge-detects: back to the home spot
    bool cam_key = false;         // C; engine edge-detects: the rocket camera
    bool fly_key = false;         // V; engine edge-detects to toggle fly mode
    bool weapon1 = false, weapon2 = false;   // 1 / 2: laser gun / airstrike tablet
    bool fire = false;      // left mouse, edge-triggered (one click = one)
    bool fire_held = false; // left mouse, level (true while button is down)
    bool alt_fire_held = false; // right mouse, level (true while button is down)
                                // — used by sculpt brush as "lower" alt
    bool screenshot = false;// F12, edge-triggered
    double mouse_dx = 0.0;
    double mouse_dy = 0.0;
};

class Window {
public:
    explicit Window(const WindowConfig& cfg);
    ~Window();

    Window(const Window&) = delete;
    Window& operator=(const Window&) = delete;

    GLFWwindow* handle() const { return window_; }
    bool should_close() const;
    void poll_events();

    void framebuffer_size(uint32_t& w, uint32_t& h) const;
    VkSurfaceKHR_T* create_surface(VkInstance_T* instance) const;
    // Native window handle (HWND on Windows) - for creating the surface
    // through Streamline's proxy, which has to know the window.
    void* native_handle() const;

    // Read accumulated input since the last call, then clear edge-triggered bits.
    InputFrame consume_input(const InputBindings& b);
    // For "press a key" in the menu: the first key / mouse button (or controller button)
    // that is down now, or -1.
    int any_key_down() const;
    int any_pad_down() const;
    // The connected controller's name ("" if none), and whether it was the last thing used.
    const char* gamepad_name() const;
    bool using_gamepad() const { return using_pad_; }

    void set_cursor_captured(bool capture);
    bool cursor_captured() const { return cursor_captured_; }
    bool has_focus() const;

    void request_close();

private:
    GLFWwindow* window_ = nullptr;
    bool cursor_captured_ = false;
    double last_cursor_x_ = 0.0;
    double last_cursor_y_ = 0.0;
    bool first_cursor_sample_ = true;
    bool fire_edge_ = false;
    bool prev_fire_ = false;
    bool prev_screenshot_ = false;
    int  pad_jid_ = -1;          // the controller in use (a GLFW joystick id)
    bool using_pad_ = false;
    bool pad_sprint_latch_ = false;
    bool prev_pad_sprint_ = false;
};

} // namespace vibe
