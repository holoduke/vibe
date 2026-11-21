#pragma once
// Jumps, falls and landings. Position, velocity and "on the ground" come from the character
// controller; this only decides which captured frame shows that state.
//
// A captured jump lasts a third of a second; the game's lasts as long as the physics says
// (double jumps, low gravity, a fall from a tower). So the flight is not played, it is
// scrubbed: the frame follows the vertical speed while rising (push-off -> top), holds around
// the top for as long as the fall lasts, and reaches for the ground only when contact is
// predicted within `reach_time`. No clip can run out, whatever the airtime.
//
// Landing: the capture's own landing plays for a moment only when the character lands hard
// and is not asked to move; a landing into a run or another jump goes straight on.
#include "anim/motion_db.h"

namespace qlike::anim {

enum class AirPhase : uint8_t { Ground, Rise, Apex, Fall, Reach, Land };
const char* air_phase_name(AirPhase p);

struct AirborneParams {
    float min_air_time = 0.05f;       // shorter than this is a bump over a kerb, not a flight
    float reach_time = 0.30f;         // seconds before predicted contact the legs reach down
    float fall_speed_ref = 9.0f;      // m/s of fall at which the "falling" pose is fully in
    float max_scrub_rate = 3.0f;      // frames advance at most this many times the capture's pace
    float land_min_impact = 4.0f;     // m/s: softer landings show no landing at all
    float land_max_move = 2.5f;       // m/s horizontal: faster than this goes straight into the run
    float land_time = 0.26f;          // seconds of captured landing at most
    float land_rate = 1.25f;
};

struct AirborneInput {
    float     dt = 0.0f;
    bool      airborne = false;       // controller: not on the ground (and not swimming)
    float     vertical_speed = 0.0f;  // m/s, + up
    float     ground_distance = 1e9f; // metres from the feet down to the ground; huge if unknown
    float     gravity = 25.0f;
    glm::vec2 velocity{0.0f};         // horizontal, character frame (x right, z back)
    bool      wants_move = false;     // movement or jump input held
};

struct AirborneState {
    AirPhase phase = AirPhase::Ground;
    int      jump = -1;               // index into MotionDatabase::jumps()
    bool     mirrored = false;
    float    frame = 0.0f;            // database frame (fractional) while not on the ground
    float    air_time = 0.0f, land_time = 0.0f;
    float    peak_up = 0.0f;          // largest upward speed seen since the last push-off
    float    last_vertical = 0.0f;
    float    impact = 0.0f;           // m/s at the last touch-down
    float    time_to_contact = 1e9f;
    uint32_t flights = 0;
};

struct AirborneOutput {
    bool  active = false;             // the airborne model owns the lower body this frame
    bool  started = false;            // took over this frame (inertialise)
    bool  finished = false;           // gave the lower body back this frame (search again)
    bool  landed = false;             // touched down this frame (with `impact`)
    float impact = 0.0f;
    int   frame = 0;                  // database frame + fraction to show
    float frac = 0.0f;
    bool  mirrored = false;
    float stand_hips = 0.0f;          // hips height of standing in the chosen capture
};

// Seconds until the feet reach the ground, by the controller's gravity. Huge if never.
float time_to_contact(float ground_distance, float vertical_speed, float gravity);
// The captured flight that leaves the ground most like `velocity`. -1 if the database has none.
int find_jump(const MotionDatabase& db, glm::vec2 velocity_capture_scale, bool& mirrored);

AirborneOutput update_airborne(const MotionDatabase& db, AirborneState& st, const AirborneInput& in, const AirborneParams& p,
                               float capture_speed_scale);

}  // namespace qlike::anim
