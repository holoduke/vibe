#pragma once
// Animation level of detail for characters other than the local player.
//
//   0  full: matching at full rate, foot locks and ground, both arms, fingers
//   1  matching at a third of the rate, stride warp only on the feet, arms, fingers
//   2  matching twice a second, no foot work, arms without fingers
//   3  not seen: no pose at all; timers and the playback position keep running
//
// Levels are chosen by how tall the character is on screen, not by a fixed distance, with a
// margin against flicker at a boundary. Someone the local player is fighting stays at 0.
#include <algorithm>

namespace vibe::anim {

struct LodParams {
    float full_px = 110.0f;      // at least this tall on screen: level 0
    float mid_px = 45.0f;        // ... level 1
    float low_px = 14.0f;        // ... level 2; smaller is level 3
    float hysteresis = 0.15f;    // a level is left only when the size is this share past its boundary
};

struct LodInput {
    float distance = 0.0f;       // metres from the camera
    float screen_height = 1080.0f;       // pixels
    float tan_half_fov_y = 0.7f;
    float character_height = 1.8f;
    bool  in_view = true;        // frustum (and occlusion, where known)
    bool  combat = false;        // shooting at, or being shot by, the local player
    int   previous = 0;
};

inline float projected_height_px(const LodInput& in) {
    const float d = std::max(in.distance, 0.05f);
    return in.character_height / (d * std::max(in.tan_half_fov_y, 1e-3f)) * 0.5f * in.screen_height;
}

inline int select_lod(const LodInput& in, const LodParams& p = {}) {
    if (!in.in_view) return in.combat ? 2 : 3;        // off screen: still cheaply posed if it matters (shadows, quick turns)
    if (in.combat) return 0;
    const float px = projected_height_px(in);
    const float bounds[3] = { p.full_px, p.mid_px, p.low_px };
    int lod = 3;
    for (int i = 0; i < 3; ++i)
        if (px >= bounds[i]) { lod = i; break; }
    // hysteresis: keep the previous level while the size is within the margin of the boundary crossed
    const int prev = std::clamp(in.previous, 0, 3);
    if (lod > prev && prev < 3 && px >= bounds[prev] * (1.0f - p.hysteresis)) return prev;
    if (lod < prev && px < bounds[prev - 1] * (1.0f + p.hysteresis)) return prev;
    return lod;
}

}  // namespace vibe::anim
