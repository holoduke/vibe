#pragma once
// Cooking: raw clips on the rig -> the runtime's compressed skeleton and clips (offline only).
#include "anim/asset.h"
#include "anim/offline/raw_clip.h"
#include "anim/offline/retarget.h"

#include <functional>

namespace vibe::anim::offline {

bool build_runtime_skeleton(const SkeletonDef& s, std::vector<uint8_t>& blob, std::string* why = nullptr);

struct CompressParams {
    float tolerance = 0.001f;     // metres of error allowed at `distance` from a joint
    float distance = 0.10f;
    bool  optimize = true;        // remove keys the tolerance does not need
};
bool build_runtime_clip(const SkeletonDef& s, const RawClip& c, const CompressParams& p, std::vector<uint8_t>& blob,
                        std::string* why = nullptr);

// One capture take to import: <dir>/<style>/<style>_<kind>.bvh, cut by Frame_Cuts.csv.
struct Style100Source {
    std::string style;
    std::string kinds;            // comma separated: FW,FR,BW,BR,SW,SR,ID,TR1,TR2,TR3
    uint32_t    tags = 0;
};
// The curated set for the shooter: neutral locomotion, crouch, starts and stops, two-foot jumps.
std::vector<Style100Source> default_style100_sources();

struct CookedClips {
    std::vector<RawClip>    raw;        // kept for the motion-database build
    std::vector<ClipReport> reports;
};
using CookLog = std::function<void(const std::string&)>;
// Imports, retargets, validates and compresses every source into `asset` (skeleton + clips).
// Fails if any clip fails validation.
bool cook_style100(const std::string& dataset_dir, const SkeletonDef& skel, const std::vector<Style100Source>& sources,
                   const RetargetParams& rp, const CompressParams& cp, AnimAsset& asset, CookedClips& cooked,
                   const CookLog& log, std::string* why = nullptr);

// One whole BVH file to import as a clip: <dir>/<file>.
struct TakeSource {
    std::string file, name;
    uint32_t    tags = 0;
    bool        reversed = false;     // played backward: a forward run becomes a run going backward
};
// The runs of the CMU Graphics Lab Motion Capture Database (subjects 9, 16, 35): 2.3 - 4.1 m/s,
// a second or two each. Free to use, also in products (mocap.cs.cmu.edu).
// with_reversed: also the steady runs played backward, as the fast back-pedal nobody captured.
std::vector<TakeSource> default_cmu_sources(bool with_reversed = true);
// Adds whole-file takes to an asset that already has its skeleton (after cook_style100). A file
// that is not there is skipped and said so; a clip that fails validation fails the cook.
bool cook_takes(const std::string& dir, const std::vector<TakeSource>& sources, const SkeletonDef& skel, const RetargetParams& rp,
                const CompressParams& cp, const std::string& provenance, const std::string& licence, AnimAsset& asset,
                CookedClips& cooked, const CookLog& log, std::string* why = nullptr);

}  // namespace vibe::anim::offline

