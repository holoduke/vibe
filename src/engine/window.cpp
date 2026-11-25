#include "engine/window.h"

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#ifdef _WIN32
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>
#endif

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace vibe {

namespace {
int g_glfw_refcount = 0;
}

Window::Window(const WindowConfig& cfg) {
    if (g_glfw_refcount == 0) {
        if (!glfwInit()) throw std::runtime_error("glfwInit failed");
    }
    ++g_glfw_refcount;

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

    window_ = glfwCreateWindow(static_cast<int>(cfg.width),
                               static_cast<int>(cfg.height),
                               cfg.title.c_str(), nullptr, nullptr);
    if (!window_) throw std::runtime_error("glfwCreateWindow failed");

    glfwSetWindowUserPointer(window_, this);

    if (cfg.capture_cursor) set_cursor_captured(true);

    if (glfwRawMouseMotionSupported()) {
        glfwSetInputMode(window_, GLFW_RAW_MOUSE_MOTION, GLFW_TRUE);
    }
}

Window::~Window() {
    if (window_) glfwDestroyWindow(window_);
    if (--g_glfw_refcount == 0) glfwTerminate();
}

bool Window::should_close() const { return glfwWindowShouldClose(window_); }
void Window::poll_events() { glfwPollEvents(); }
void Window::request_close() { glfwSetWindowShouldClose(window_, GLFW_TRUE); }
bool Window::has_focus() const {
    return glfwGetWindowAttrib(window_, GLFW_FOCUSED) == GLFW_TRUE;
}

void Window::framebuffer_size(uint32_t& w, uint32_t& h) const {
    int iw = 0, ih = 0;
    glfwGetFramebufferSize(window_, &iw, &ih);
    w = static_cast<uint32_t>(iw);
    h = static_cast<uint32_t>(ih);
}

void* Window::native_handle() const {
#ifdef _WIN32
    return glfwGetWin32Window(window_);
#else
    return nullptr;
#endif
}

VkSurfaceKHR_T* Window::create_surface(VkInstance_T* instance) const {
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    if (glfwCreateWindowSurface(reinterpret_cast<VkInstance>(instance),
                                window_, nullptr, &surface) != VK_SUCCESS) {
        throw std::runtime_error("glfwCreateWindowSurface failed");
    }
    return reinterpret_cast<VkSurfaceKHR_T*>(surface);
}

void Window::set_cursor_captured(bool capture) {
    cursor_captured_ = capture;
    glfwSetInputMode(window_, GLFW_CURSOR,
                     capture ? GLFW_CURSOR_DISABLED : GLFW_CURSOR_NORMAL);
    first_cursor_sample_ = true;  // re-anchor delta tracking
}

// ---------------------------------------------------------------------
// Bindings
// ---------------------------------------------------------------------

namespace {
struct ActionInfo { const char* label; const char* id; int key; int pad; };
// (the order of enum Action)
const ActionInfo kActionInfo[kActionCount] = {
    { "Forward",                  "fwd",        GLFW_KEY_W,             -1 },
    { "Back",                     "back",       GLFW_KEY_S,             -1 },
    { "Left",                     "left",       GLFW_KEY_A,             -1 },
    { "Right",                    "right",      GLFW_KEY_D,             -1 },
    { "Jump / swim up",           "jump",       GLFW_KEY_SPACE,         GLFW_GAMEPAD_BUTTON_A },
    { "Sprint",                   "sprint",     GLFW_KEY_LEFT_SHIFT,    GLFW_GAMEPAD_BUTTON_LEFT_THUMB },
    { "Crouch / dive",            "crouch",     GLFW_KEY_LEFT_CONTROL,  GLFW_GAMEPAD_BUTTON_B },
    { "Fire",                     "fire",       InputBindings::kMouseBase + GLFW_MOUSE_BUTTON_LEFT,  InputBindings::kPadRT },
    { "Aim",                      "aim",        InputBindings::kMouseBase + GLFW_MOUSE_BUTTON_RIGHT, InputBindings::kPadLT },
    { "Use / edit mode",          "use",        GLFW_KEY_E,             GLFW_GAMEPAD_BUTTON_Y },
    { "Reload / next brush",      "reload",     GLFW_KEY_R,             GLFW_GAMEPAD_BUTTON_X },
    { "Flashlight",               "flashlight", GLFW_KEY_F,             GLFW_GAMEPAD_BUTTON_RIGHT_THUMB },
    { "Weapon 1: laser",          "weapon1",    GLFW_KEY_1,             GLFW_GAMEPAD_BUTTON_LEFT_BUMPER },
    { "Weapon 2: airstrike",      "weapon2",    GLFW_KEY_2,             GLFW_GAMEPAD_BUTTON_RIGHT_BUMPER },
    { "Weapon 3: rifle",          "weapon3",    GLFW_KEY_3,             GLFW_GAMEPAD_BUTTON_DPAD_RIGHT },
    { "Fly mode",                 "fly",        GLFW_KEY_V,             GLFW_GAMEPAD_BUTTON_DPAD_UP },
    { "Rocket camera",            "camera",     GLFW_KEY_C,             GLFW_GAMEPAD_BUTTON_BACK },
    { "Back to the start",        "home",       GLFW_KEY_H,             GLFW_GAMEPAD_BUTTON_DPAD_DOWN },
    { "Photo",                    "screenshot", GLFW_KEY_F12,           -1 },
    { "Edit: previous brush",     "brush_prev", GLFW_KEY_Q,             GLFW_GAMEPAD_BUTTON_DPAD_LEFT },
    { "Edit: smaller brush",      "brush_smaller", GLFW_KEY_LEFT_BRACKET,  -1 },
    { "Edit: larger brush",       "brush_larger",  GLFW_KEY_RIGHT_BRACKET, -1 },
    { "Terrain wireframe",        "wireframe",  GLFW_KEY_9,             -1 },
};

// Dead zone on the whole stick (not per axis: that would pull diagonals to the axes), then
// the response curve on what is left.
void shape_stick(float& x, float& y, float dead, float curve) {
    const float len = std::sqrt(x * x + y * y);
    if (len <= dead || len < 1e-5f) { x = y = 0.0f; return; }
    const float t = (std::min)((len - dead) / (std::max)(1.0f - dead, 1e-3f), 1.0f);
    const float k = std::pow(t, curve) / len;
    x *= k; y *= k;
}
} // namespace

void InputBindings::reset() {
    for (int i = 0; i < kActionCount; ++i) { key[i] = kActionInfo[i].key; pad[i] = kActionInfo[i].pad; }
}
const char* action_label(Action a) { return kActionInfo[static_cast<int>(a)].label; }
const char* action_id(Action a)    { return kActionInfo[static_cast<int>(a)].id; }

std::string key_binding_name(int code) {
    if (code < 0) return "-";
    if (code >= InputBindings::kMouseBase) {
        const int b = code - InputBindings::kMouseBase;
        if (b == 0) return "Left mouse";
        if (b == 1) return "Right mouse";
        if (b == 2) return "Middle mouse";
        return "Mouse " + std::to_string(b + 1);
    }
    switch (code) {
        case GLFW_KEY_SPACE: return "Space";
        case GLFW_KEY_LEFT_SHIFT: case GLFW_KEY_RIGHT_SHIFT: return "Shift";
        case GLFW_KEY_LEFT_CONTROL: case GLFW_KEY_RIGHT_CONTROL: return "Ctrl";
        case GLFW_KEY_LEFT_ALT: case GLFW_KEY_RIGHT_ALT: return "Alt";
        case GLFW_KEY_TAB: return "Tab";
        case GLFW_KEY_ENTER: return "Enter";
        case GLFW_KEY_BACKSPACE: return "Backspace";
        case GLFW_KEY_CAPS_LOCK: return "Caps Lock";
        case GLFW_KEY_UP: return "Up";
        case GLFW_KEY_DOWN: return "Down";
        case GLFW_KEY_LEFT: return "Left";
        case GLFW_KEY_RIGHT: return "Right";
        case GLFW_KEY_INSERT: return "Insert";
        case GLFW_KEY_DELETE: return "Delete";
        case GLFW_KEY_HOME: return "Home";
        case GLFW_KEY_END: return "End";
        case GLFW_KEY_PAGE_UP: return "Page Up";
        case GLFW_KEY_PAGE_DOWN: return "Page Down";
        default: break;
    }
    if (code >= GLFW_KEY_F1 && code <= GLFW_KEY_F25) return "F" + std::to_string(code - GLFW_KEY_F1 + 1);
    if (code >= GLFW_KEY_KP_0 && code <= GLFW_KEY_KP_9) return "Num " + std::to_string(code - GLFW_KEY_KP_0);
    // A printable key: by what it prints on this keyboard layout.
    if (const char* n = glfwGetKeyName(code, 0); n && *n) {
        std::string s(n);
        if (s.size() == 1 && s[0] >= 'a' && s[0] <= 'z') s[0] = static_cast<char>(s[0] - 'a' + 'A');
        return s;
    }
    return "Key " + std::to_string(code);
}

const char* pad_binding_name(int code) {
    static const char* kNames[17] = { "A", "B", "X", "Y", "LB", "RB", "Back", "Start", "Guide",
                                      "L3", "R3", "D-pad up", "D-pad right", "D-pad down", "D-pad left",
                                      "LT", "RT" };
    return (code >= 0 && code < 17) ? kNames[code] : "-";
}

int Window::any_key_down() const {
    for (int k = GLFW_KEY_SPACE; k <= GLFW_KEY_LAST; ++k) {
        if (k == GLFW_KEY_ESCAPE) continue;            // (Esc cancels, and is the menu's own)
        if (glfwGetKey(window_, k) == GLFW_PRESS) return k == GLFW_KEY_RIGHT_SHIFT ? GLFW_KEY_LEFT_SHIFT
                                                       : k == GLFW_KEY_RIGHT_CONTROL ? GLFW_KEY_LEFT_CONTROL
                                                       : k == GLFW_KEY_RIGHT_ALT ? GLFW_KEY_LEFT_ALT : k;
    }
    for (int b = 0; b <= GLFW_MOUSE_BUTTON_LAST; ++b)
        if (glfwGetMouseButton(window_, b) == GLFW_PRESS) return InputBindings::kMouseBase + b;
    return -1;
}

int Window::any_pad_down() const {
    GLFWgamepadstate st;
    if (pad_jid_ < 0 || !glfwGetGamepadState(pad_jid_, &st)) return -1;
    for (int b = 0; b <= GLFW_GAMEPAD_BUTTON_LAST; ++b) {
        if (b == GLFW_GAMEPAD_BUTTON_START) continue;  // (the menu's own)
        if (st.buttons[b] == GLFW_PRESS) return b;
    }
    if (st.axes[GLFW_GAMEPAD_AXIS_LEFT_TRIGGER]  > 0.0f) return InputBindings::kPadLT;
    if (st.axes[GLFW_GAMEPAD_AXIS_RIGHT_TRIGGER] > 0.0f) return InputBindings::kPadRT;
    return -1;
}

const char* Window::gamepad_name() const {
    if (pad_jid_ < 0 || !glfwJoystickIsGamepad(pad_jid_)) return "";
    const char* n = glfwGetGamepadName(pad_jid_);
    return n ? n : "";
}

// ---------------------------------------------------------------------
// One frame of input: keyboard and mouse, and the controller if there is one. Both always
// work; `using_pad` only says which was touched last (for the prompts on screen).
// ---------------------------------------------------------------------

InputFrame Window::consume_input(const InputBindings& b) {
    InputFrame f{};
    bool kb_used = false;
    // A key or a mouse button. Shift / Ctrl / Alt: either side counts.
    auto code_down = [&](int code) {
        if (code < 0) return false;
        if (code >= InputBindings::kMouseBase)
            return glfwGetMouseButton(window_, code - InputBindings::kMouseBase) == GLFW_PRESS;
        if (glfwGetKey(window_, code) == GLFW_PRESS) return true;
        const int twin = code == GLFW_KEY_LEFT_SHIFT ? GLFW_KEY_RIGHT_SHIFT
                       : code == GLFW_KEY_LEFT_CONTROL ? GLFW_KEY_RIGHT_CONTROL
                       : code == GLFW_KEY_LEFT_ALT ? GLFW_KEY_RIGHT_ALT : -1;
        return twin >= 0 && glfwGetKey(window_, twin) == GLFW_PRESS;
    };

    // ---- the controller ----
    // The one in use stays in use while it is connected; otherwise the first one found.
    GLFWgamepadstate st{};
    bool have_pad = false;
    if (b.pad_enabled) {
        if (pad_jid_ >= 0 && glfwGetGamepadState(pad_jid_, &st)) have_pad = true;
        else {
            pad_jid_ = -1;
            for (int j = GLFW_JOYSTICK_1; j <= GLFW_JOYSTICK_LAST && !have_pad; ++j)
                if (glfwJoystickIsGamepad(j) && glfwGetGamepadState(j, &st)) { pad_jid_ = j; have_pad = true; }
        }
    }
    if (!have_pad) {
        st = GLFWgamepadstate{};
        st.axes[GLFW_GAMEPAD_AXIS_LEFT_TRIGGER] = st.axes[GLFW_GAMEPAD_AXIS_RIGHT_TRIGGER] = -1.0f;
    }
    f.pad_connected = have_pad;
    bool pad_used = false;
    auto pad_down = [&](int code) {
        if (!have_pad || code < 0) return false;
        if (code == InputBindings::kPadLT) return st.axes[GLFW_GAMEPAD_AXIS_LEFT_TRIGGER]  > -0.2f;   // (-1 released .. +1 fully in)
        if (code == InputBindings::kPadRT) return st.axes[GLFW_GAMEPAD_AXIS_RIGHT_TRIGGER] > -0.2f;
        return code <= GLFW_GAMEPAD_BUTTON_LAST && st.buttons[code] == GLFW_PRESS;
    };
    auto act = [&](Action a) {
        const int i = static_cast<int>(a);
        const bool k = code_down(b.key[i]), p = pad_down(b.pad[i]);
        kb_used |= k; pad_used |= p;
        return k || p;
    };

    f.fwd   = act(Action::Fwd);
    f.back  = act(Action::Back);
    f.left  = act(Action::Left);
    f.right = act(Action::Right);
    f.jump     = act(Action::Jump);
    f.crawl    = act(Action::Crouch);
    f.edit_key = act(Action::Use);
    f.brush_smaller = act(Action::BrushSmaller);
    f.brush_larger  = act(Action::BrushLarger);
    f.brush_mode_prev = act(Action::BrushPrev);
    f.brush_mode_next = act(Action::Reload);
    f.reload          = f.brush_mode_next;
    f.wireframe_key   = act(Action::Wireframe);
    f.fly_key         = act(Action::Fly);
    f.cam_key         = act(Action::Camera);
    f.home_key        = act(Action::Home);
    f.weapon1         = act(Action::Weapon1);
    f.weapon2         = act(Action::Weapon2);
    f.weapon3         = act(Action::Weapon3);
    f.flashlight      = act(Action::Flashlight);
    // The menu: Esc, and the controller's Start (neither can be rebound).
    {
        const bool k = glfwGetKey(window_, GLFW_KEY_ESCAPE) == GLFW_PRESS;
        const bool p = have_pad && st.buttons[GLFW_GAMEPAD_BUTTON_START] == GLFW_PRESS;
        kb_used |= k; pad_used |= p;
        f.menu_key = k || p;
    }

    // The sticks.
    if (have_pad) {
        float mx = st.axes[GLFW_GAMEPAD_AXIS_LEFT_X],  my = st.axes[GLFW_GAMEPAD_AXIS_LEFT_Y];
        float lx = st.axes[GLFW_GAMEPAD_AXIS_RIGHT_X], ly = st.axes[GLFW_GAMEPAD_AXIS_RIGHT_Y];
        if (b.pad_swap_sticks) { std::swap(mx, lx); std::swap(my, ly); }
        shape_stick(mx, my, b.pad_deadzone, 1.0f);              // walking: linear
        shape_stick(lx, ly, b.pad_deadzone, b.pad_curve);
        f.move_x = mx; f.move_y = -my;                          // (up on the stick is -1)
        f.look_x = lx; f.look_y = ly;
        if (mx != 0.0f || my != 0.0f || lx != 0.0f || ly != 0.0f) pad_used = true;
        // The same as keys too, for everything that only knows keys (fly mode, the network).
        f.fwd   |= f.move_y > 0.35f;  f.back |= f.move_y < -0.35f;
        f.right |= f.move_x > 0.35f;  f.left |= f.move_x < -0.35f;
    }

    // Sprint. On the controller one press of the stick keeps it going until the stick is let
    // go (holding a stick down while steering with it is hard on the thumb).
    {
        const int si = static_cast<int>(Action::Sprint);
        const bool k = code_down(b.key[si]), p = pad_down(b.pad[si]);
        kb_used |= k; pad_used |= p;
        if (b.pad_sprint_toggle) {
            if (p && !prev_pad_sprint_) pad_sprint_latch_ = !pad_sprint_latch_;
            if (f.move_x * f.move_x + f.move_y * f.move_y < 0.04f) pad_sprint_latch_ = false;
        } else {
            pad_sprint_latch_ = p;
        }
        prev_pad_sprint_ = p;
        f.sprint = k || pad_sprint_latch_;
    }

    bool fire_now = act(Action::Fire);
    f.fire = fire_now && !prev_fire_;  // rising edge
    f.fire_held = fire_now;             // level (held)
    prev_fire_ = fire_now;
    // Aim - also the sculpt brush's alternate "lower" gesture (fire = raise / current mode).
    f.alt_fire_held = act(Action::Aim);

    // Photo: rising edge. The engine writes a numbered file in screenshots/.
    bool ss_now = act(Action::Screenshot);
    f.screenshot = ss_now && !prev_screenshot_;
    prev_screenshot_ = ss_now;

    double cx = 0.0, cy = 0.0;
    glfwGetCursorPos(window_, &cx, &cy);
    if (first_cursor_sample_) {
        last_cursor_x_ = cx;
        last_cursor_y_ = cy;
        first_cursor_sample_ = false;
    }
    f.mouse_dx = cx - last_cursor_x_;
    f.mouse_dy = cy - last_cursor_y_;
    last_cursor_x_ = cx;
    last_cursor_y_ = cy;
    if (f.mouse_dx * f.mouse_dx + f.mouse_dy * f.mouse_dy > 4.0) kb_used = true;   // (not a nudge of the desk)
    if (b.invert_mouse_y) f.mouse_dy = -f.mouse_dy;

    // Which one is in the hands of the player: the last one touched.
    if (pad_used) using_pad_ = true;
    else if (kb_used || !have_pad) using_pad_ = false;
    f.using_pad = using_pad_;

    return f;
}

} // namespace vibe
