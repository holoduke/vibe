#pragma once
// The motion-matching database: one entry per sampled frame of every locomotion clip, with
// the features the search compares and the metadata the animator needs.
//
// Feature vector (all in the character's root frame: x right, y up, z back; metres, m/s).
// The schema is stored in the file and checked at load:
//
//   group        dims  what
//   foot_pos       6   left, right ankle position
//   foot_vel       6   left, right ankle velocity
//   hip_vel        3   hips velocity
//   traj_pos       6   root position on the ground (x, z) at each horizon in the future
//   traj_dir       6   root facing (x, z) at each horizon
//
// "Root facing" is where the body faces, not where it travels: a sideways run has the path
// going to the side while the facing stays ahead. The game sets the facing from the aim, so
// strafing selects the sideways captures.
//
// Stored features are normalised: (raw - offset) / scale, with one scale per group (so a
// group's dimensions keep their relative size) divided by the group's weight. Offsets and
// scales are left/right symmetric, which lets the mirrored half of the data be searched by
// mirroring the query instead of storing every frame twice.
//
// Rows are padded to kStride floats so four AVX registers cover a row.
#include "anim/types.h"

namespace qlike::anim {

enum FeatureGroup : int { kFootPos, kFootVel, kHipVel, kTrajPos, kTrajDir, kFeatureGroups };

struct MotionSchema {
    static constexpr uint32_t kVersion = 3;
    static constexpr int kHorizons = 3;
    static constexpr int kFeat = 27;
    static constexpr int kStride = 32;
    // first dimension of each group, and one past the last
    static constexpr int kGroupFirst[kFeatureGroups + 1] = { 0, 6, 12, 15, 21, 27 };
    static const char* group_name(int g);
};

struct MotionFrameInfo {
    float   root_vel_x, root_vel_z;     // the root's own velocity, root frame, m/s
    float   yaw_rate;                   // rad/s, positive = turning right (the sign of the player's yaw)
    float   hips_y;                     // hips height above the ground, metres
    uint8_t contact;                    // bit 0 left foot, bit 1 right foot on the ground
    uint8_t pad[3];
};

struct MotionRange {                    // one clip's frames
    int32_t  clip;                      // index into the asset's clips
    uint32_t first, count;              // database frames [first, first + count)
    uint32_t tags;                      // ClipTag bits
    uint32_t head, tail;                // frames at the start / end that may be played but not jumped to
};

struct MotionJump {                     // one captured flight inside a clip tagged kTagJump
    uint32_t takeoff;                   // first frame with both feet off the ground
    uint32_t apex;                      // highest hips
    uint32_t land;                      // first frame with a foot back down
    float    rise;                      // metres the hips rise above standing
    float    vel_x, vel_z;              // root velocity at take-off (root frame)
};

struct MotionQuery {
    float    raw[MotionSchema::kFeat] = {};             // not normalised
    float    group_weight[kFeatureGroups] = { 1, 1, 1, 1, 1 };
    uint32_t require_tags = kTagLocomotion;             // a range must have all of these ...
    uint32_t exclude_tags = 0;                          // ... and none of these
};

struct MotionMatch {
    int   frame = -1;                   // database frame
    bool  mirrored = false;
    float cost = 1e30f;
    float group_cost[kFeatureGroups] = {};
};

class MotionDatabase {
public:
    bool load(const uint8_t* data, size_t size, uint64_t rig_id, std::string* why = nullptr);
    bool ok() const { return frames_ > 0; }

    uint32_t frames() const { return frames_; }
    float    sample_rate() const { return rate_; }
    float    horizon(int i) const { return horizons_[i]; }
    const std::vector<MotionRange>& ranges() const { return ranges_; }
    const std::vector<MotionJump>&  jumps() const { return jumps_; }
    const MotionFrameInfo& info(uint32_t frame) const { return info_[frame]; }
    const MotionRange&     range_of(uint32_t frame) const { return ranges_[range_of_[frame]]; }
    float time_in_clip(uint32_t frame, float frac = 0.0f) const;
    // The stored (normalised) row, and the same de-normalised into `raw` (kFeat floats).
    const float* row(uint32_t frame) const { return &feat_[size_t(frame) * MotionSchema::kStride]; }
    void raw_features(uint32_t frame, bool mirrored, float* raw) const;
    void normalise(const float* raw, float* out) const;                 // out: kStride floats
    static void mirror_raw(const float* raw, float* out);               // left <-> right
    size_t memory_bytes() const;
    // The pace the capture keeps up in a direction (0 forward, 1 sideways, 2 backward), standing
    // or crouched: a high percentile of the root speed of the frames going that way. A request
    // for more than this finds only single moments, not whole steps. 0 = nothing captured.
    float cruise_speed(bool crouched, int axis) const { return cruise_[crouched ? 1 : 0][axis < 0 ? 0 : axis > 2 ? 2 : axis]; }

    // Exhaustive search over every frame a range allows, both as captured and mirrored.
    MotionMatch search(const MotionQuery& q) const;
    // Cost of one frame for a query, with the per-group breakdown (debug views, tests).
    MotionMatch cost(const MotionQuery& q, uint32_t frame, bool mirrored) const;
    // Slow reference implementation of search(), used by the tests to verify the fast path.
    MotionMatch search_reference(const MotionQuery& q) const;

private:
    friend struct MotionDatabaseBuilder;
    uint32_t frames_ = 0;
    float    rate_ = 30.0f;
    float    horizons_[MotionSchema::kHorizons] = {};
    float    offset_[MotionSchema::kStride] = {}, scale_[MotionSchema::kStride] = {};
    std::vector<float>           feat_;        // frames x kStride, normalised, padding zero
    std::vector<MotionFrameInfo> info_;
    std::vector<MotionRange>     ranges_;
    std::vector<uint32_t>        range_of_;
    std::vector<MotionJump>      jumps_;
    float    cruise_[2][3] = {};
};

// What the offline build hands over; serialised into the asset's motion block.
struct MotionDatabaseBuilder {
    uint64_t rig_id = 0;
    float    rate = 30.0f;
    float    horizons[MotionSchema::kHorizons] = { 1.0f / 3.0f, 2.0f / 3.0f, 1.0f };
    float    group_weight[kFeatureGroups] = { 0.75f, 1.0f, 1.0f, 1.0f, 1.5f };
    std::vector<float>           raw;          // frames x kFeat, not normalised
    std::vector<MotionFrameInfo> info;
    std::vector<MotionRange>     ranges;
    std::vector<MotionJump>      jumps;
    // Computes the symmetric normalisation and writes the block.
    std::vector<uint8_t> serialize() const;
};

}  // namespace qlike::anim
