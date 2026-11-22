#pragma once
// Motion matching for grounded locomotion: keeps a play position in the database, advances
// it, and several times a second asks the database for a better frame - one that continues
// the pose the character is in and goes where the controller is about to take it.
//
// The game moves far faster than a person in a capture studio (8 - 12 m/s against 1 - 2), so
// the request is made at the capture's scale: speeds are mapped into the captured range and
// the predicted path is shrunk by the same factor. What is left of the difference is covered
// downstream by playback rate, stride warping and lean (never by distorting the controller).
#include "anim/motion_db.h"
#include "anim/trajectory.h"

namespace vibe::anim {

struct MatcherParams {
    float search_interval = 0.10f;      // seconds between searches while nothing changes
    float change_interval = 0.04f;      // ... when the request changed (new direction, speed, stance)
    float change_speed = 0.35f;         // m/s (capture scale) of request change that counts as "changed"
    float min_switch_interval = 0.07f;  // never switch clips more often than this (no thrashing)
    float keep_ratio = 0.90f;           // stay unless the best cost is below this share of the current one ...
    float keep_margin = 0.02f;          // ... by at least this much
    int   min_hop_frames = 8;           // a jump of fewer frames within the same clip is not worth a transition
    // Speed mapping game -> capture: 1:1 up to walk_speed, then easing toward run_speed.
    float walk_speed = 1.0f;
    float run_speed = 2.1f;
    float speed_knee = 2.5f;
    // High-speed split: above fast_speed (game m/s) the path counts for more against the pose,
    // so reversals at speed pick the new direction at once.
    float fast_speed = 5.0f;
    float fast_pose_weight = 0.55f;
    // A best match whose path cost is above this did not really fit: the request is outside
    // what was captured. The search is then repeated by path alone (directional fallback).
    float fallback_traj_cost = 9.0f;
};

struct MatcherInput {
    Trajectory traj;                    // game scale, character frame
    glm::vec2  velocity{0.0f};          // game scale, character frame (x right, z back)
    uint32_t   require_tags = kTagLocomotion;
    uint32_t   exclude_tags = 0;
    float      rate = 1.0f;             // playback rate
    bool       force_search = false;    // e.g. after landing
};

struct MatcherState {
    int       frame = -1;               // database frame; -1 = nothing playing yet
    float     frac = 0.0f;
    bool      mirrored = false;
    float     since_search = 1e9f, since_switch = 1e9f;
    glm::vec2 last_request{0.0f};
    uint32_t  last_tags = 0;
    // debug / statistics
    MotionMatch last_match;             // the candidate the last search found
    float       last_current_cost = 0.0f;
    bool        last_fallback = false;
    uint32_t    searches = 0, switches = 0;
};

struct MatcherOutput {
    int       clip = -1;
    float     time = 0.0f;              // seconds into the clip
    bool      mirrored = false;
    bool      switched = false;         // jumped to another place this update: inertialise
    bool      searched = false;
    glm::vec2 root_vel{0.0f};           // the animation's own velocity (character frame, capture scale)
    float     yaw_rate = 0.0f;
    float     hips_y = 0.0f;
    bool      contact[2] = { false, false };    // left, right (after mirroring)
    float     speed_scale = 1.0f;       // capture speed / game speed of the request (<= 1)
};

// Game speed -> the capture's equivalent.
float capture_speed(float game_speed, const MatcherParams& p);

MatcherOutput update_matcher(const MotionDatabase& db, MatcherState& st, const MatcherInput& in, const MatcherParams& p, float dt);
// Jump straight to a database frame (used by the airborne model); reports it as a switch.
MatcherOutput matcher_goto(const MotionDatabase& db, MatcherState& st, int frame, bool mirrored);
// The output for the state as it is (no advance, no search).
MatcherOutput matcher_output(const MotionDatabase& db, const MatcherState& st);

}  // namespace vibe::anim
