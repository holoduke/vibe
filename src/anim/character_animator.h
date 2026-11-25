#pragma once
// One character's animation: controller state in, pose out.
//
//   CharacterAnimationInput (const)  ->  CharacterAnimator::update  ->  model-space joint matrices
//
// The animator never writes to the controller and has no pointer to it: position, velocity,
// collisions, jumps, aim and fire timing are decided elsewhere and arrive here as numbers.
// For another player they are the interpolated network state; the code path is the same.
//
// Order of evaluation (later steps see the earlier ones' result; IK is last so nothing fights it):
//   1  movement mode from the controller (ground / air / swim)
//   2  lower body: motion matching on the ground, the airborne model in the air
//   3  clip sampling, mirroring, inertialisation of any change of source
//   4  pelvis: crouch depth, landing dip, lean
//   5  orientation warp: legs toward the travel direction, chest stays on the aim
//   6  upper body: aim pitch spread over spine / neck / head, recoil
//   7  feet: stride warp, ground height, world locks, pelvis compensation, two-bone fit
//   8  weapon frame (from gameplay aim), reload / equip offsets, both arms fitted to the grips, fingers
// On top of whatever made the pose - locomotion, an action clip, the seat - come the LIFE LAYERS
// (life_layers.cpp; docs/animation/actions.md "Life layers"), always in this order:
//   a  the person's own posture (PersonStyle: constant offsets)           additive
//   b  a clip over part of the body (layer_clip + LayerMask)               override, masked
//   c  breathing, the weight from one side to the other                    additive, own clocks
//   d  a gesture or a fidget (a hand to the head, a look at the wrist)     override, one arm
//   e  the look: head, neck, chest and eyes to what is worth one           within limits
// and only then what fits the body to the world (feet, the seat's hips, hands on a weapon).
#include "anim/airborne.h"
#include "anim/inertializer.h"
#include "anim/motion_matcher.h"
#include "anim/person_style.h"
#include "anim/runtime.h"
#include "game/physics.h"

#include <memory>

namespace vibe::anim {

// Everything characters share: the runtime (skeleton + clips), the motion database, joint lookups.
struct AnimationSet {
    std::unique_ptr<IAnimationRuntime> runtime;
    MotionDatabase                     db;
    struct Hand {
        int       finger[4][3], thumb[3];     // index, middle, ring, pinky; -1 if absent
        glm::vec3 a0{0.0f}, n0{0.0f}, b0{0.0f};  // at rest: wrist -> knuckles, out of the palm, a0 x n0
        float     palm = 0.08f;               // wrist to knuckles, metres
        float     index_side = 1.0f;          // +1 / -1: the turn about n0 that takes a finger towards the index finger's side
        bool      ok = false;
    };
    Hand   hand[2]{};                    // left, right
    // The face, if the skeleton has one (Jaw, Left/RightEye, ...UpperLid, ...LowerLid, ...Brow, all
    // on the head; the lids turn about the eye's centre). -1 = absent; `ok` = eyes and upper lids.
    struct Face {
        int  jaw = -1, eye[2] = { -1, -1 }, lid_up[2] = { -1, -1 }, lid_lo[2] = { -1, -1 }, brow[2] = { -1, -1 };
        bool ok = false;
    };
    Face   face{};
    float  stand_hips = 1.0f;            // hips height of the standing idle, metres
    // A weight a joint for every LayerMask (life_layers.cpp build_layer_masks).
    std::vector<float> mask[int(LayerMask::Count)];
    size_t asset_bytes = 0;

    bool load(const char* path, std::string* why = nullptr);
    bool init(const AnimAsset& asset, std::string* why = nullptr);
    void build_masks();
    bool ok() const { return runtime != nullptr && db.ok(); }
};

enum class HeldItem : uint8_t { None, Rifle, Tablet, Torch };

// ONE HAND'S FINGERS ON WHAT IT HOLDS (docs/animation/hands.md). Every finger has its own bend:
// a hand whose four fingers all bend alike is a mitten.
//   curl[finger][joint]: radians each joint is bent towards the palm - index, middle, ring,
//                        pinky; knuckle, middle joint, tip
//   spread[finger]:      radians the finger is fanned away from the middle finger's line
//                        (the middle finger's own: towards the index)
//   thumb:               [0] swung across the palm, [1] bent at its base, [2] at its outer joints
struct HandPose {
    float curl[4][3];
    float spread[4];
    float thumb[3];
};

// Where the hands go on what is carried, in view space (x right, y up, z toward the viewer),
// and how it kicks and reloads. Gameplay aim places the view frame; the hands follow it.
struct WeaponDef {
    glm::vec3 grip_r{0.165f, -0.300f, -0.120f}, grip_l{0.178f, -0.243f, -0.430f};   // palm centres
    glm::vec3 r_fingers{-0.10f, 0.25f, -1.0f}, r_palm{-1.0f, 0.0f, 0.10f};          // direction of the fingers, palm normal
    glm::vec3 l_fingers{0.95f, 0.20f, -0.35f}, l_palm{0.05f, 1.0f, 0.15f};
    // The right hand on a pistol grip: three fingers wrapped, tighter towards the little finger
    // (it has the least to go round); the index finger laid on the trigger, hardly bent; the
    // thumb round the other side, pointing forward.
    HandPose  pose_r{ { { 0.30f, 0.62f, 0.30f }, { 1.02f, 1.32f, 0.80f }, { 1.10f, 1.38f, 0.85f }, { 1.18f, 1.42f, 0.88f } },
                      { 0.10f, 0.0f, 0.03f, 0.07f }, { 0.85f, 0.30f, 0.45f } };
    // The left hand under the fore-end: the palm carries it, the fingers come up round its far
    // side, a little fanned along it; the thumb lies along the near side.
    HandPose  pose_l{ { { 0.50f, 1.20f, 0.70f }, { 0.58f, 1.28f, 0.78f }, { 0.64f, 1.30f, 0.80f }, { 0.72f, 1.34f, 0.84f } },
                      { 0.13f, 0.02f, 0.07f, 0.16f }, { 0.22f, 0.10f, 0.22f } };
    glm::vec3 muzzle{0.18f, -0.18f, -0.72f};
    glm::vec3 butt{0.172f, -0.175f, 0.03f};                // the end of the stock (weapon frame)
    glm::vec3 shoulder_offset{-0.035f, 0.020f, -0.07f};    // where the butt sits from the right shoulder joint (aim frame: +x right, -z forward)
    float     recoil_stroke = 0.05f;     // metres the weapon comes back
    float     recoil_duration = 0.14f;
    float     recoil_pitch = 0.035f;     // radians the muzzle climbs
    float     reload_time = 1.5f;        // seconds
    glm::vec3 reload_hand{0.10f, -0.42f, -0.16f};      // where the support hand fetches the magazine
    // At the ready (not aiming) the weapon points where the view does, like the aimed one: only
    // lower. Seen from outside the butt comes down from the shoulder to the ribs; in the player's
    // own view it sits lower and a little canted. (ready_pitch / ready_yaw let the muzzle hang
    // while nothing is fired; 0 = it never does. It read as a weapon pointing the wrong way.)
    glm::vec3 ready_shift{0.015f, -0.135f, 0.015f};     // aim frame, added to shoulder_offset
    float     ready_pitch = 0.0f, ready_yaw = 0.0f;     // radians: muzzle down, and to the left
    glm::vec3 ready_shift_fp{-0.045f, 0.050f, 0.020f};      // (was 0.020, -0.045, 0.030: only the barrel's end showed in the corner)
    float     ready_pitch_fp = 0.0f, ready_yaw_fp = 0.0f, ready_roll_fp = -0.07f;
    // Aiming, in the player's own view only: the weapon comes to the middle of the screen, its
    // sights under the crosshair. Nobody else sees this: for them it is in the shoulder.
    // (LR-9: brings its holo sight, at (0.162, -0.119) in the weapon frame, onto the crosshair.)
    glm::vec3 ads_shift_fp{-0.162f, 0.119f, -0.120f};
    float     hip_fire_hold = 0.7f;     // seconds a weapon fired from the hip stays level before it sinks again
    glm::vec3 tablet_centre{0.155f, -0.165f, -0.40f};   // the tablet in the weapon frame (as drawn)
    glm::vec3 tablet_chest{0.10f, -0.20f, -0.30f};      // ... and from the neck joint when seen from outside (model axes)
};
const WeaponDef& default_rifle();

struct AnimTuning {
    MatcherParams  matcher;
    AirborneParams air;
    physics::MoveParams move;            // the controller's movement model (set from game::kPlayerMove)
    float inertial_halflife = 0.085f;    // seconds; shorter = snappier, more visible changes
    float min_rate = 0.75f, max_rate = 1.5f;           // playback rate fitted to the real speed
    float min_stride = 0.7f, max_stride = 4.5f;        // then stride length
    float stride_reach = 0.72f;                        // a striding foot goes this share of the leg's length from the hip at most
    float stride_leg = 0.90f;                          // ... and is lifted off the ground where the leg would be longer than this share
    float stride_drop = 0.04f, max_stride_drop = 0.10f;// metres the hips come down per unit of extra stride, and at most
    float max_orient_warp = 1.45f;       // radians the legs may turn away from the capture's travel direction
    float strafe_run_speed = 5.5f;       // m/s: faster than this sideways, the legs run forward turned to the side (no sidestep)
    float lean_strength = 1.0f;          // 0 = none
    float accel_lean = 0.012f;           // radians per m/s^2 (turning at speed is an acceleration too)
    float speed_lean = 0.010f;           // radians per m/s, into the direction of travel
    float max_lean = 0.22f;
    float flinch = 0.22f;                // radians the torso gives under an ordinary hit (0 = none)
    // The face (seen from outside, full detail only). The three *_hold values are for looking at
    // it: they force the lids shut / the jaw open / the eyes to a side, whatever else is going on.
    float face = 1.0f;                   // 0 = a still face
    float blink_hold = 0.0f, jaw_hold = 0.0f, eyes_hold = 0.0f;
    float idle_look = 1.0f;              // not aiming: the head looks about, at others, or nowhere for a while (0 = never)
    float idle_sway = 1.0f;              // standing, not aiming: breathing and a shift of weight now and then (0 = none)
    // How far a look may turn the head from the way the chest faces (radians; the eyes do the
    // rest), what is added by the look included to what the clip already has.
    float look_yaw_max = 1.05f, look_pitch_up = 0.45f, look_pitch_down = 0.60f;
    float layer_halflife = 0.16f;        // seconds: a clip over part of the body comes and goes this fast
    float head_to_sights = 1.0f;         // seen from outside: the head leans to the stock (0 = stays upright)
    float chest_keep = 0.35f;            // share of the torso's lean away from its rest place that is kept (1 = all of it)
    float spine_pitch_share = 0.30f;     // share of the aim pitch taken by the spine (third person)
    float spine_pitch_share_fp = 0.10f;  // ... when seen from the character's own eyes
    float foot_ik_weight = 1.0f;
    float foot_lock_max_speed = 4.0f;    // m/s: above it feet are never held
    float foot_lock_max_accel = 30.0f;   // m/s^2: a hard start or stop lets go at once
    float foot_lock_break = 0.22f;       // metres a held foot may lag before it is let go
    float foot_ground_range = 0.40f;     // metres up / down a foot follows the ground
    float land_dip = 0.022f;             // metres of pelvis dip per m/s of impact
    float max_land_dip = 0.28f;
    float airborne_pose_strength = 1.0f; // 0 = keep standing pose in the air (comparison)
    float air_sway = 1.0f;               // movement of the limbs through a long fall
    bool  shoulder_weapon = true;        // seen from outside, the rifle's butt is in the shoulder (off: where the player's own view has it)
    bool  stride_warp = true, orient_warp = true, foot_ik = true, foot_lock = true, hand_ik = true;
};

struct CharacterAnimationInput {
    float     dt = 0.0f;
    glm::vec3 position{0.0f};            // world: the model's origin (on the ground under the body)
    glm::vec3 velocity{0.0f};            // world
    glm::vec3 desired_velocity{0.0f};    // world, horizontal: wish direction x wish speed
    glm::vec3 ground_velocity{0.0f};     // world: what the character stands on moves like this
    float     facing_yaw = 0.0f;         // the body's and the aim's yaw (0 looks down -Z, + turns right)
    float     aim_pitch = 0.0f;          // + up
    bool      grounded = true;
    bool      swimming = false;
    bool      wants_jump = false;
    float     crouch = 0.0f;             // 0 standing .. 1 lowest
    float     gravity = 25.0f;
    glm::vec3 eye_local{0.0f, 1.6f, 0.0f};             // the eye in model space (the weapon hangs on it)
    float     stand_eye_height = 1.6f;
    HeldItem  held = HeldItem::Rifle;
    glm::vec3 torch_pos{0.0f}, torch_fwd{0.0f, 0.0f, -1.0f}, torch_right{1.0f, 0.0f, 0.0f};   // model space, HeldItem::Torch
    uint32_t  shot_seq = 0;              // counts shots; a change kicks the recoil
    uint32_t  reload_seq = 0;            // counts reload starts
    uint32_t  hit_seq = 0;               // counts hits taken; a change makes the body flinch
    // Killed: 0 -> 1 over the moment before the body is the physics' (a ragdoll). The legs give,
    // the torso goes the way the last shot went (hit_dir), the head drops, the weapon sinks.
    float     dying = 0.0f;
    glm::vec3 hit_dir{0.0f, 0.0f, 1.0f};  // model space: the way the shot travelled
    float     hit_power = 1.0f;          // 1 = an ordinary hit
    bool      reloading = false;         // false while a reload runs = it was interrupted
    // Aiming (the player holds the aim button): the rifle is up in the shoulder, the head on the
    // sights. Not aiming it is carried at the ready - lower, canted, the muzzle down until a
    // shot is fired from the hip. Default true: the stance everything had before there were two.
    bool      aiming = true;
    // At the wheel of a vehicle: the pose is a driver's - the model's hips where they are at
    // rest (the caller puts them on the seat), the legs forward to the pedals (seat_legs: 0 an
    // upright seat, 1 a low one with the legs stretched out), the hands on the wheel's rim.
    bool      seated = false;
    glm::vec3 seat_hand_l{0.0f}, seat_hand_r{0.0f};      // model space: where the wrists go
    float     seat_legs = 0.0f;
    // What the car itself knows of its seat (SeatExtras, below; take_seat() fills these). Model space.
    //   seat_wheel: the hands HOLD a wheel. seat_hand_* are then points on its rim (not
    //     wrists), seat_hub is its middle and seat_axis the column, pointing at the driver: the
    //     palm lies on the outside of the rim, the knuckles along it, the fingers round it.
    //   seat_feet: the feet have places. seat_heel_* on the floor, seat_ball_* where the ball
    //     of the foot presses (the accelerator's pad; the floor or the foot rest).
    //   seat_back: degrees the backrest leans back (negative: as seat_legs says, as before).
    //   seat_steer: -1 .. 1, how far the wheel is turned to the right (the head goes into the bend).
    bool      seat_wheel = false, seat_feet = false;
    glm::vec3 seat_hub{0.0f}, seat_axis{0.0f, 0.0f, 1.0f};
    glm::vec3 seat_heel_l{0.0f}, seat_ball_l{0.0f}, seat_heel_r{0.0f}, seat_ball_r{0.0f};
    float     seat_back = -1.0f, seat_steer = 0.0f;
    // (the same in the caller's space - the world - brought into the model's)
    struct SeatExtras {
        bool      wheel = false, feet = false;
        glm::vec3 hub{0.0f}, axis{0.0f, 0.0f, 1.0f};
        glm::vec3 heel_l{0.0f}, ball_l{0.0f}, heel_r{0.0f}, ball_r{0.0f};
        float     back = -1.0f, steer = 0.0f;
    };
    void take_seat(const SeatExtras& e, const glm::mat4& to_model) {
        auto pt = [&](glm::vec3 p) { return glm::vec3(to_model * glm::vec4(p, 1.0f)); };
        seat_wheel = e.wheel; seat_feet = e.feet;
        seat_hub = pt(e.hub); seat_axis = glm::vec3(to_model * glm::vec4(e.axis, 0.0f));
        seat_heel_l = pt(e.heel_l); seat_ball_l = pt(e.ball_l); seat_heel_r = pt(e.heel_r); seat_ball_r = pt(e.ball_r);
        seat_back = e.back; seat_steer = e.steer;
    }
    // AN ACTION: something done once, from a clip played by name (IAnimationRuntime::find_clip;
    // clips tagged kTagAction) instead of locomotion - sitting down, getting up, getting into
    // a car. The caller owns its time (action_time, seconds into the clip) and where the model
    // stands; the clip's own travel from its first frame is kept (the body really steps back
    // on to the seat). The change into it, between two of them and out of it is inertialised.
    //   Motion warping: with action_warp the hips are carried, over the clip, to
    //   action_hips (model space) - the seat this chair really has, not the one the capture
    //   had. action_warp_from_start: the clip BEGINS there instead (getting up).
    //   Full-body follow-through: feet that stand on the ground stay where the clip put
    //   them while the hips are moved (the legs are solved again).
    ClipId    action_clip = kNoClip;
    float     action_time = 0.0f;
    bool      action_warp = false, action_warp_from_start = false;
    // The clip's own travel is taken out again over its length, so that it ENDS where the
    // model stands (getting up from a seat: the caller has the character standing at one
    // known place afterwards, wherever the capture walked to).
    bool      action_end_at_origin = false;
    // LOOPS. action_in_place: the clip's travel over the ground is left out altogether - the
    // caller moves the model (a way of walking played round and round while the character is
    // walked along its path). action_warp_full: the hips are at action_hips for the whole
    // clip, not carried there over it (somebody who sits, eating: every loop on the seat).
    bool      action_in_place = false, action_warp_full = false;
    glm::vec3 action_hips{0.0f};
    // Something worth a look (another player), model space; the head turns to it now and then
    // while the character is not aiming. `variation` makes two characters not move in step.
    bool      has_look_at = false;
    glm::vec3 look_at{0.0f, 1.6f, -5.0f};
    uint32_t  variation = 0;
    // ---- LIFE LAYERS (see the top of this file; PersonStyle in person_style.h) ----
    // individuality: how much of this person's own posture (person_style(variation)) is put on
    //   top of what is played. 0 = nobody in particular (the players' soldiers).
    // life_on_clip: an action clip or the seat pose also breathes, shifts its weight, looks
    //   and fidgets, as locomotion always does. Off for something done once (sitting down).
    // look_weight: how much of the time the head goes to look_at (1 = now and then, 2.5 = most
    //   of the time: who is talked to; 4 and more = fixed on it). look_id: what is looked at -
    //   a change is something new that catches the eye soon.
    // fidgets: how often a hand goes to the head or the wrist by itself (0 never, 1 every
    //   quarter of a minute or so). fidget_arms: which hands are free for it (1 left, 2 right).
    // gesture + gesture_seq: one asked for (a change of gesture_seq starts it).
    // layer_clip: a clip over part of the body (layer_mask) while the rest goes on as it was -
    //   an upper body that talks on legs that sit, a head that looks about on a body that
    //   walks. The animator owns its time (layer_rate) and its coming and going; it is played
    //   round and round and let go for a moment where it wraps.
    // action_mirror: the action clip is played left for right.
    float     individuality = 0.0f;
    bool      life_on_clip = false;
    float     look_weight = 1.0f;
    uint32_t  look_id = 0;
    float     fidgets = 0.0f;
    uint8_t   fidget_arms = 3;
    Gesture   gesture = Gesture::None;
    uint32_t  gesture_seq = 0;
    ClipId    layer_clip = kNoClip;
    LayerMask layer_mask = LayerMask::UpperBody;
    float     layer_rate = 1.0f, layer_weight = 1.0f;
    bool      action_mirror = false;
    bool      first_person = false;
    bool      teleported = false;        // position jumped: drop every world-space memory
    int       lod = 0;                   // 0 full .. 3 no pose (see anim_lod.h)
    // Height of the ground at a world (x, z); nullptr = flat at position.y.
    float   (*ground_height)(void* user, float x, float z) = nullptr;
    void*     ground_user = nullptr;
};

struct AnimDebug {
    const char* mode = "ground";
    int   clip = -1;  float clip_time = 0.0f;  bool mirrored = false;
    MotionMatch match;  float current_cost = 0.0f;  bool fallback = false;
    uint32_t searches = 0, switches = 0, flights = 0, invalid_poses = 0;
    Trajectory traj;
    AirPhase air_phase = AirPhase::Ground;  float time_to_contact = 0.0f, impact = 0.0f;
    float rate = 1.0f, stride = 1.0f, warp = 0.0f, lean_roll = 0.0f, lean_pitch = 0.0f;
    float anim_speed = 0.0f, speed = 0.0f, inertial_energy = 0.0f, pelvis_offset = 0.0f;
    bool  contact[2]{}, locked[2]{};
    glm::vec3 foot_target[2]{}, hand_target[2]{};      // model space
    float recoil = 0.0f, reload = 0.0f, equip = 0.0f;
    int   lod = 0;
    float update_ms = 0.0f, search_ms = 0.0f;
};

class CharacterAnimator {
public:
    explicit CharacterAnimator(const AnimationSet& set);

    AnimTuning tuning;
    WeaponDef  weapon = default_rifle();

    void reset();
    // An action clip's travel from its first frame to `t` seconds (see CharacterAnimationInput::action_clip).
    Transform action_root(ClipId clip, float t) const;
    void update(const CharacterAnimationInput& in);

    std::span<const glm::mat4> model() const { return model_; }       // joint matrices, model space
    std::span<const Transform> local() const { return pose_; }
    // What is carried is drawn at  view frame * weapon_offset()  (recoil, reload, equip).
    const glm::mat4& weapon_offset() const { return weapon_offset_; }
    // Where the weapon is the moment a shot leaves it: as weapon_offset, but level (a weapon at
    // the ready comes level to fire) and before the recoil. What a tracer starts from.
    const glm::mat4& weapon_fire_offset() const { return weapon_fire_offset_; }
    // What the head is doing (debug views, tests): 0 at rest, 1 looking to a side, 2 looking at
    // look_at, 3 a slow sweep.
    int   gaze_mode() const { return gaze_mode_; }
    float gaze_yaw() const { return gaze_yaw_; }       // radians, + right
    float gaze_pitch() const { return gaze_pitch_; }   // radians, + up
    // The face: how far the lids are shut (0 open, 1 shut), the eyes off straight ahead, the jaw open.
    float lids() const { return lid_; }
    glm::vec2 eyes() const { return glm::vec2(eye_yaw_, eye_pitch_); }      // radians: + right, + up
    float jaw() const { return jaw_; }                                      // radians
    // The life layers (debug views, tests): the gesture in hand and how far it is in (0 .. 1),
    // how far the clip over part of the body is in, this person's style.
    Gesture gesture() const { return gest_; }
    float   gesture_weight() const { return gest_w_; }
    float   layer_weight() const { return layer_w_; }
    const PersonStyle& style() const { return style_; }
    int   blinks() const { return blinks_; }
    // Model-space muzzle position and direction (effects; never hit registration).
    glm::vec3 muzzle_position() const { return muzzle_pos_; }
    // The magazine in the support hand through the middle of a reload: its place in the weapon
    // frame (drawn at view * weapon_offset * this), and whether there is one to draw.
    bool      magazine_in_hand() const { return mag_visible_; }
    glm::vec3 magazine_position() const { return mag_pos_; }
    glm::vec3 muzzle_direction() const { return muzzle_dir_; }
    const AnimDebug& debug() const { return dbg_; }
    const AnimationSet& set() const { return set_; }

private:
    void lower_body(const CharacterAnimationInput& in, glm::vec2 v_local, float speed, float yaw_rate, float ground_dist);
    void pelvis_and_warp(const CharacterAnimationInput& in, glm::vec2 v_local, float speed, float yaw_rate, glm::vec2 accel_local);
    void upper_body(const CharacterAnimationInput& in);
    void feet(const CharacterAnimationInput& in, glm::vec2 v_local, float speed, float accel);
    void arms(const CharacterAnimationInput& in);
    void seated(const CharacterAnimationInput& in);
    void action(const CharacterAnimationInput& in);
    // The life layers: `base` 0 locomotion, 1 the seat pose, 2 an action clip.
    void life_layers(const CharacterAnimationInput& in, int base);
    void posture_layer(const CharacterAnimationInput& in, bool sitting);
    void clip_layer(const CharacterAnimationInput& in, float dt);
    void gesture_layer(const CharacterAnimationInput& in, float dt, bool alive, bool sitting, bool moving);
    void look_layer(const CharacterAnimationInput& in, float dt, bool busy, bool moving, bool sitting);
    void rotate_in_model(int joint, const glm::quat& q);              // turns a joint about model axes (local pose)
    glm::quat model_rotation(int joint) const;                        // from pose_, by walking the parents
    void fit_limb(int a, int b, int c, glm::vec3 target, const glm::vec3* pole, bool from_rest = false);
    float ground_at(const CharacterAnimationInput& in, glm::vec3 model_point) const;   // metres above the model's origin plane

    const AnimationSet& set_;
    std::unique_ptr<EvalContext> ctx_;
    std::vector<Transform> pose_, src_, next_, scratch_, good_;
    std::vector<glm::mat4> model_, good_model_;
    Inertializer inert_;
    int       last_mode_ = 0;                  // what made the last pose: 0 locomotion, 1 the seat, 2 an action
    ClipId    action_clip_ = kNoClip;          // the action playing
    float     action_last_t_ = 0.0f;           // its time last frame (a jump back = it loops: blended)
    glm::vec3 action_ref_hips_{0.0f};          // where that clip has the hips at its end (or start), model space
    MatcherState match_;
    AirborneState air_;
    MatcherOutput cur_;                   // what the lower body is playing
    bool   started_ = false, have_good_ = false;
    float  prev_yaw_ = 0.0f, yaw_rate_ = 0.0f;
    glm::vec3 prev_vel_{0.0f}, accel_{0.0f};
    float  rate_ = 1.0f, stride_ = 1.0f, warp_ = 0.0f;
    float  aim_w_ = 1.0f;                // 1 = aiming, 0 = at the ready
    float  since_shot_ = 1e9f;           // seconds: a weapon fired from the hip stays level for a moment
    float  snap_axis_ = 0.0f;            // the captured direction (a multiple of 90 degrees) a diagonal request is turned onto
    bool   snap_on_ = false, strafe_run_ = false;
    float  lean_roll_ = 0.0f, lean_roll_v_ = 0.0f, lean_pitch_ = 0.0f, lean_pitch_v_ = 0.0f;
    float  dip_ = 0.0f, dip_v_ = 0.0f, pelvis_ik_ = 0.0f, pelvis_ik_v_ = 0.0f;
    float  air_blend_ = 0.0f, time_ = 0.0f, stand_hips_ = 0.0f;
    struct Lock { bool on = false, broken = false; glm::vec3 world{0.0f}, off{0.0f}, off_v{0.0f}; float ground = 0.0f, ground_v = 0.0f; } lock_[2];
    uint32_t shot_seq_ = 0, reload_seq_ = 0;
    float  recoil_t_ = 0.0f, reload_t_ = -1.0f, reload_u_ = 0.0f, reload_w_ = 0.0f, equip_t_ = 1.0f;
    HeldItem held_ = HeldItem::Rifle;
    glm::mat4 weapon_offset_{1.0f};
    glm::vec3 muzzle_pos_{0.0f}, muzzle_dir_{0.0f, 0.0f, -1.0f};
    glm::vec3 mag_pos_{0.0f};
    glm::mat4 weapon_fire_offset_{1.0f};
    // idle life: where the head looks, and the weight on the feet
    uint32_t  rng_ = 0x9E3779B9u;
    bool      rng_seeded_ = false;
    int       gaze_mode_ = 0;
    float     gaze_left_ = 0.0f;                       // seconds until the next choice
    float     gaze_yaw_t_ = 0.0f, gaze_pitch_t_ = 0.0f;   // what a glance looks at
    float     gaze_half_ = 0.45f;                      // how quickly this turn of the head is made (spring halflife, s)
    float     gaze_sweep_ = 0.0f;                      // a sweep: radians a second, toward gaze_sweep_to_
    float     gaze_sweep_to_ = 0.0f;
    float     gaze_yaw_ = 0.0f, gaze_yaw_v_ = 0.0f, gaze_pitch_ = 0.0f, gaze_pitch_v_ = 0.0f, gaze_w_ = 0.0f;
    float     shift_left_ = 3.0f, shift_t_ = 0.0f, shift_ = 0.0f, shift_v_ = 0.0f;
    float     rnd();                                   // 0 .. 1, this animator's own sequence
    // the face
    float     blink_left_ = 2.0f, blink_t_ = -1.0f, lid_ = 0.0f;
    int       blinks_ = 0;
    float     eye_yaw_ = 0.0f, eye_pitch_ = 0.0f, dart_yaw_ = 0.0f, dart_pitch_ = 0.0f, dart_left_ = 1.0f;
    float     jaw_ = 0.0f, jaw_v_ = 0.0f, brow_ = 0.0f, brow_v_ = 0.0f;
    float     gaze_want_yaw_ = 0.0f, gaze_want_pitch_ = 0.0f;     // where the head is being turned to (the eyes are there first)
    void      face_layer(const CharacterAnimationInput& in);
    bool      mag_visible_ = false;
    // the life layers
    PersonStyle style_{};
    uint32_t  style_of_ = 0;
    bool      action_mirror_ = false;
    ClipId    layer_clip_ = kNoClip;
    LayerMask layer_mask_ = LayerMask::UpperBody;
    float     layer_t_ = 0.0f, layer_w_ = 0.0f, layer_wv_ = 0.0f;
    Gesture   gest_ = Gesture::None;
    int       gest_side_ = 1;                          // 0 left, 1 right
    float     gest_t_ = 0.0f, gest_dur_ = 0.0f, gest_w_ = 0.0f, fidget_left_ = 6.0f;
    uint32_t  gesture_seq_ = 0, look_id_ = 0;
    float     gest_look_yaw_ = 0.0f, gest_look_pitch_ = 0.0f;     // where a gesture takes the head
    // (a blend of two poses of a joint keeps the way round it chose when it began: see blend_kept)
    std::vector<glm::quat> layer_pa_, layer_pb_;
    glm::quat gest_pa_[3]{}, gest_pb_[3]{};
    float     layer_prev_eff_ = 0.0f, gest_prev_w_ = 0.0f;
    float     layer_eff_ = 0.0f;                       // the weight of the clip layer this frame (0: not there)
    float     hips_shift_ = 0.0f;                      // metres the layers moved the hips sideways this frame
    uint32_t  hit_seq_ = 0;
    float     flinch_pitch_ = 0.0f, flinch_pitch_v_ = 0.0f, flinch_roll_ = 0.0f, flinch_roll_v_ = 0.0f;
    AnimDebug dbg_;
};

}  // namespace vibe::anim
