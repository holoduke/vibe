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
#include "anim/airborne.h"
#include "anim/inertializer.h"
#include "anim/motion_matcher.h"
#include "anim/runtime.h"
#include "game/physics.h"

#include <memory>

namespace qlike::anim {

// Everything characters share: the runtime (skeleton + clips), the motion database, joint lookups.
struct AnimationSet {
    std::unique_ptr<IAnimationRuntime> runtime;
    MotionDatabase                     db;
    struct Hand {
        int       finger[4][3], thumb[3];     // index, middle, ring, pinky; -1 if absent
        glm::vec3 a0{0.0f}, n0{0.0f}, b0{0.0f};  // at rest: wrist -> knuckles, out of the palm, a0 x n0
        float     palm = 0.08f;               // wrist to knuckles, metres
        bool      ok = false;
    };
    Hand   hand[2]{};                    // left, right
    float  stand_hips = 1.0f;            // hips height of the standing idle, metres
    size_t asset_bytes = 0;

    bool load(const char* path, std::string* why = nullptr);
    bool init(const AnimAsset& asset, std::string* why = nullptr);
    bool ok() const { return runtime != nullptr && db.ok(); }
};

enum class HeldItem : uint8_t { None, Rifle, Tablet, Torch };

// Where the hands go on what is carried, in view space (x right, y up, z toward the viewer),
// and how it kicks and reloads. Gameplay aim places the view frame; the hands follow it.
struct WeaponDef {
    glm::vec3 grip_r{0.165f, -0.300f, -0.120f}, grip_l{0.178f, -0.243f, -0.430f};   // palm centres
    glm::vec3 r_fingers{-0.10f, 0.25f, -1.0f}, r_palm{-1.0f, 0.0f, 0.10f};          // direction of the fingers, palm normal
    glm::vec3 l_fingers{0.95f, 0.20f, -0.35f}, l_palm{0.05f, 1.0f, 0.15f};
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
    glm::vec3 ready_shift_fp{0.020f, -0.045f, 0.030f};
    float     ready_pitch_fp = 0.0f, ready_yaw_fp = 0.0f, ready_roll_fp = -0.07f;
    // Aiming, in the player's own view only: the weapon comes to the middle of the screen, its
    // sights under the crosshair. Nobody else sees this: for them it is in the shoulder.
    glm::vec3 ads_shift_fp{-0.158f, 0.052f, -0.120f};
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
    float idle_look = 1.0f;              // not aiming: the head looks about, at others, or nowhere for a while (0 = never)
    float idle_sway = 1.0f;              // standing, not aiming: breathing and a shift of weight now and then (0 = none)
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
    glm::vec3 hit_dir{0.0f, 0.0f, 1.0f};  // model space: the way the shot travelled
    float     hit_power = 1.0f;          // 1 = an ordinary hit
    bool      reloading = false;         // false while a reload runs = it was interrupted
    // Aiming (the player holds the aim button): the rifle is up in the shoulder, the head on the
    // sights. Not aiming it is carried at the ready - lower, canted, the muzzle down until a
    // shot is fired from the hip. Default true: the stance everything had before there were two.
    bool      aiming = true;
    // Something worth a look (another player), model space; the head turns to it now and then
    // while the character is not aiming. `variation` makes two characters not move in step.
    bool      has_look_at = false;
    glm::vec3 look_at{0.0f, 1.6f, -5.0f};
    uint32_t  variation = 0;
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
    void rotate_in_model(int joint, const glm::quat& q);              // turns a joint about model axes (local pose)
    glm::quat model_rotation(int joint) const;                        // from pose_, by walking the parents
    void fit_limb(int a, int b, int c, glm::vec3 target, const glm::vec3* pole, bool from_rest = false);
    float ground_at(const CharacterAnimationInput& in, glm::vec3 model_point) const;   // metres above the model's origin plane

    const AnimationSet& set_;
    std::unique_ptr<EvalContext> ctx_;
    std::vector<Transform> pose_, src_, next_, scratch_, good_;
    std::vector<glm::mat4> model_, good_model_;
    Inertializer inert_;
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
    bool      mag_visible_ = false;
    uint32_t  hit_seq_ = 0;
    float     flinch_pitch_ = 0.0f, flinch_pitch_v_ = 0.0f, flinch_roll_ = 0.0f, flinch_roll_v_ = 0.0f;
    AnimDebug dbg_;
};

}  // namespace qlike::anim
