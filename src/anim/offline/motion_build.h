#pragma once
// Builds the motion-matching database from raw clips (offline only): features, frame
// metadata, clip ranges with their exclusion zones, and the flights found in jump clips.
#include "anim/motion_db.h"
#include "anim/offline/raw_clip.h"

#include <functional>

namespace vibe::anim::offline {

struct MotionBuildParams {
    float horizons[MotionSchema::kHorizons] = { 1.0f / 3.0f, 2.0f / 3.0f, 1.0f };   // seconds ahead
    float group_weight[kFeatureGroups] = { 0.75f, 1.0f, 1.0f, 1.0f, 1.5f };
    int   head_frames = 2;              // frames at a clip's start that are never jumped to
    float min_jump_rise = 0.055f;       // metres the hips must rise for a flight to count
    int   min_flight_frames = 4;
    float landing_frames_s = 0.47f;     // frames kept after a landing (its recovery)
};

// The builder with everything filled in (the tests look at it directly).
bool build_motion_features(const SkeletonDef& skel, const std::vector<RawClip>& clips, const MotionBuildParams& p,
                           MotionDatabaseBuilder& out, std::string* why = nullptr);
// ... and serialised into an asset's motion block.
bool build_motion_database(const SkeletonDef& skel, const std::vector<RawClip>& clips, const MotionBuildParams& p,
                           std::vector<uint8_t>& blob, const std::function<void(const std::string&)>& log,
                           std::string* why = nullptr);

}  // namespace vibe::anim::offline
