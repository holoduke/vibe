#pragma once
// WHO SOMEBODY IS, for the animation: what makes two people who play the same clip not look the
// same. All of it follows from one number (the character's id), so it is the same every time
// that person is shown and on every machine, and costs nothing to keep.
//
// Used by the animator's life layers (life_layers.cpp, docs/animation/actions.md "Life layers")
// and by whoever chooses the clips (where in a loop this person starts, how fast it is played,
// whether it is played mirrored).
#include <cstdint>

namespace vibe::anim {

struct PersonStyle {
    // ---- for whoever plays the clips ----
    float phase = 0.0f;          // 0 .. 1: where in a loop this person begins
    float rate = 1.0f;           // playback speed, 0.93 .. 1.07
    bool  mirror = false;        // plays a clip left for right, where the clip allows it
    // ---- posture: small constant offsets on top of whatever is played (radians, metres) ----
    float slouch = 0.0f;         // + a rounded back, the head forward; - the chest out
    float lean = 0.0f;           // the trunk to a side (+ the person's right)
    float shoulders = 0.0f;      // both shoulders up (+) or dropped
    float shoulder_tilt = 0.0f;  // the right one higher (+) than the left
    float head_tilt = 0.0f;      // the head on a side (+ to the right shoulder)
    float head_turn = 0.0f;      // carried a little to a side (+ right)
    float head_nod = 0.0f;       // the chin up (+) or down
    float arms_out = 0.0f;       // how far the upper arms are held from the body
    float feet_fwd = 0.0f;       // on a seat: the feet out in front (+) or tucked under it
    float feet_apart = 0.0f;     // on a seat: the feet wider (+) or closer
    // ---- the layers that run on their own clocks ----
    float breath_rate = 1.6f;    // radians a second (a breath every 3.3 - 5 s)
    float breath_depth = 1.0f;   // 0.7 .. 1.4
    float fidget = 1.0f;         // how restless: 0.6 .. 1.6 times as often
    float curious = 1.0f;        // how readily the head goes to what passes: 0.7 .. 1.4
    float clock = 0.0f;          // seconds this person's clocks are ahead of everybody else's
};

// id 0 is nobody in particular: every offset zero.
inline PersonStyle person_style(uint32_t id) {
    PersonStyle s;
    if (id == 0) return s;
    // (one well-mixed number a trait: the id's own bits are far too regular - ids are 1, 2, 3 ...)
    auto u = [id](uint32_t k) {
        uint32_t x = id * 0x9E3779B1u + k * 0x85EBCA77u + 0x165667B1u;
        x ^= x >> 15; x *= 0x2C1B3C6Du; x ^= x >> 12; x *= 0x297A2D39u; x ^= x >> 15;
        return float(x & 0xFFFFFFu) / float(0x1000000);            // 0 .. 1
    };
    auto pm = [&](uint32_t k) { return u(k) * 2.0f - 1.0f; };      // -1 .. 1
    s.phase = u(1);
    s.rate = 1.0f + 0.07f * pm(2);
    s.mirror = u(3) < 0.5f;
    // (most people slouch a little; a few stand very straight)
    s.slouch = -0.05f + 0.19f * u(4) * u(4);
    s.lean = 0.035f * pm(5);
    s.shoulders = 0.07f * pm(6);
    s.shoulder_tilt = 0.06f * pm(7);
    s.head_tilt = 0.075f * pm(8);
    s.head_turn = 0.07f * pm(9);
    s.head_nod = 0.06f * pm(10);
    s.arms_out = 0.02f + 0.07f * u(11);
    s.feet_fwd = -0.07f + 0.17f * u(12);
    s.feet_apart = 0.045f * pm(13);
    s.breath_rate = 1.25f + 0.65f * u(14);
    s.breath_depth = 0.7f + 0.7f * u(15);
    s.fidget = 0.6f + 1.0f * u(16);
    s.curious = 0.7f + 0.7f * u(17);
    s.clock = 40.0f * u(18);
    return s;
}

// Which joints a layer has a say over (a weight a joint, feathered where the body carries on).
enum class LayerMask : uint8_t {
    UpperBody,     // the spine upward: the trunk, both arms, the head
    Head,          // the neck and the head
    ArmLeft, ArmRight, Arms,     // from the shoulder down, fingers and all
    Count
};

// Things done with a hand or a shift of the seat, for a few seconds, over whatever is played.
// A fidget is one the animator starts by itself (CharacterAnimationInput::fidgets); any of them
// can be asked for (gesture + gesture_seq).
enum class Gesture : uint8_t {
    None,
    ScratchHead,   // a hand up to the side of the head, the fingers working
    RubNeck,       // a hand to the back of the neck
    CheckWrist,    // the forearm up in front of the chest, the head down to it
    Shift,         // the weight moved on the seat (or from foot to foot), the back straightened
    Wave,          // a hand up beside the head, to and fro
    Point,         // an arm out at something (CharacterAnimationInput::gesture_at), the first finger straight
    Talk,          // a forearm up in front of the body, the open hand moving with what is said
    Count
};

}  // namespace vibe::anim
