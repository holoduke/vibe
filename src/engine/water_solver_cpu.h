// water_solver_cpu.h - the CPU side of the local shallow-water solver (vk_engine/water_solver.cpp,
// shaders/water_solver.comp) that needs no GPU: the window's placement, the fixed-step clock, the
// read-back's decoding and sampling, the foam rule's obstacle term (the shader's, for tests), and
// the spray emitters' rates (vk_engine/water_spray.cpp). Unit tests: tests/test_water_solver.cpp.
#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/packing.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace vibe::sw {

constexpr int   kN = 512;            // cells a side
constexpr float kCell = 0.2f;        // m (the reference's numerics want >= ~0.15 m)
constexpr float kStep = 0.01f;       // s, fixed (the reference's)
constexpr int   kMaxSteps = 4;       // a frame at most (at 25 fps and below the water runs slow)

// The window's corner for an eye: whole cells, the eye in the middle (so a scroll is a whole
// number of cells and the resampling is an exact copy).
inline glm::ivec2 corner_cells(glm::vec2 eye, int n = kN, float cell = kCell) {
    return glm::ivec2(glm::floor(eye / cell)) - glm::ivec2(n / 2);
}

// The fixed-step clock: how many 0.01 s steps this frame takes, at most kMaxSteps (what is left
// over beyond the cap is dropped: below 25 fps the water runs slow rather than taking ever more steps).
inline int take_steps(double& acc, double dt, int cap = kMaxSteps) {
    acc += std::clamp(dt, 0.0, 0.25);
    int n = int(std::floor(acc / double(kStep) + 1e-9));
    if (n > cap) { n = cap; acc = 0.0; }
    else acc -= double(n) * double(kStep);
    return n;
}

// The edge weight the shader blends the state to the authored one with (the outer 10 % each side).
inline float edge_weight(glm::vec2 uv) {
    const float m = std::max(std::abs(uv.x - 0.5f), std::abs(uv.y - 0.5f));
    return std::clamp(10.0f * m - 4.0f, 0.0f, 1.0f);
}

// The read-back: per cell f32 height + 2 x f16 velocity (0, 0 = no water there), and the
// window it was made in.
struct Readback {
    glm::ivec2 corner{ 0 };     // cells
    int n = kN;
    float cell = kCell;
    std::vector<uint32_t> data;  // 2 a cell
    bool valid = false;

    bool cell_at(int i, int j, float& h, glm::vec2& v) const {
        if (i < 0 || j < 0 || i >= n || j >= n) return false;
        const size_t k = (size_t(j) * size_t(n) + size_t(i)) * 2;
        const uint32_t hb = data[k];
        if (hb == 0u) return false;
        std::memcpy(&h, &hb, 4);
        v = glm::unpackHalf2x16(data[k + 1]);
        return true;
    }
    // The surface and the current at (x, z), bilinear over the four cells round it; false where
    // any of them has no water or it lies outside the window. `fade`: 1 inside, falling to 0 over
    // the window's outer 10 % (the solver blends to the authored state there).
    bool sample(float x, float z, float& h, glm::vec2& v, float* fade = nullptr) const {
        if (!valid) return false;
        const float gx = x / cell - float(corner.x) - 0.5f, gz = z / cell - float(corner.y) - 0.5f;
        const float fx = std::floor(gx), fz = std::floor(gz);
        const int i = int(fx), j = int(fz);
        const float tx = gx - fx, tz = gz - fz;
        float h00, h10, h01, h11;
        glm::vec2 v00, v10, v01, v11;
        if (!cell_at(i, j, h00, v00) || !cell_at(i + 1, j, h10, v10) || !cell_at(i, j + 1, h01, v01) || !cell_at(i + 1, j + 1, h11, v11)) return false;
        h = (h00 * (1.0f - tx) + h10 * tx) * (1.0f - tz) + (h01 * (1.0f - tx) + h11 * tx) * tz;
        v = (v00 * (1.0f - tx) + v10 * tx) * (1.0f - tz) + (v01 * (1.0f - tx) + v11 * tx) * tz;
        if (fade) {
            const glm::vec2 uv((gx + 0.5f) / float(n), (gz + 0.5f) / float(n));
            *fade = 1.0f - edge_weight(uv);
        }
        return true;
    }
};

// Packs one cell as the shader's read-back pass does (for tests).
inline void pack_cell(std::vector<uint32_t>& data, int n, int i, int j, float h, glm::vec2 v) {
    const size_t k = (size_t(j) * size_t(n) + size_t(i)) * 2;
    std::memcpy(&data[k], &h, 4);
    data[k + 1] = glm::packHalf2x16(v);
}

// The foam rule's obstacle term (water_solver.comp mode 2, O): fast (> 0.65 m/s), shallow (< 0.8 m)
// water over a rough bed that runs into higher ground downstream - the white water in front of
// and round a stone. `step_max`: the largest bed step to a neighbour (m), `rise_down`: how much
// higher the bed is downstream (m), `depth` (m), `speed` (m/s). Returns what it adds a step.
inline float foam_obstacle(float speed, float depth, float step_max, float rise_down) {
    auto S = [](float x) { x = std::clamp(x, 0.0f, 1.0f); return x * x * (3.0f - 2.0f * x); };
    const float rough = std::clamp((step_max - 0.02f) / 0.28f, 0.0f, 1.0f);
    const float fast = std::clamp((speed - 0.65f) / 0.35f, 0.0f, 1.0f);
    const float rise = std::clamp((rise_down - 0.03f) / 0.12f, 0.0f, 1.0f);
    const float deep = std::clamp((depth - 0.15f) / 0.65f, 0.0f, 1.0f);
    const float rfr = rough * fast * rise;
    const float O = 0.8f * speed * S(rough) * S(fast) * S(rise) * (1.0f - S(deep)) * rfr * rfr;
    return std::min(0.008f, O / (O + 1.0f));
}

// Spray rates (droplets per second) of the emitters (vk_engine/water_spray.cpp).
// A fall: by its height and the width that pours over it.
inline float spray_rate_fall(float drop_m, float width_m) {
    if (drop_m < 0.25f) return 0.0f;
    return 26.0f * std::min(drop_m, 3.0f) * std::clamp(width_m, 0.5f, 6.0f);
}
// A stone the water runs into: from 1 m/s, by the speed above that and the stone's width.
inline float spray_rate_rock(float speed, float radius) {
    if (speed < 1.0f) return 0.0f;
    return 22.0f * std::min(speed - 1.0f, 3.0f) * std::clamp(radius, 0.2f, 2.0f);
}
// Whitecaps on the open sea: bursts a second within the spray's reach, from 12 m/s (a near gale).
inline float spray_rate_sea(float wind) {
    const float k = std::clamp((wind - 12.0f) / 13.0f, 0.0f, 1.0f);
    return 14.0f * k * k * (3.0f - 2.0f * k);
}

}  // namespace vibe::sw
