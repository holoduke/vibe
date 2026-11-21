// vibe_animcook: builds the soldier's animation asset from motion capture.
//
//   vibe_animcook <100STYLE dir> <soldier_fp.qchar> <out.qanim> [--rate 30] [--no-optimize]
//
// Imports the curated takes (anim/offline/cook.cpp, default_style100_sources), retargets them
// to the soldier, validates every clip, compresses them for the runtime and builds the
// motion-matching database. The raw capture files are not part of the repository: download
// 100STYLE (CC BY 4.0) from https://www.ianxmason.com/100style/.
#include "anim/offline/cook.h"
#include "anim/offline/motion_build.h"
#include "game/player.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

using namespace qlike;

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: vibe_animcook <100STYLE dir> <soldier_fp.qchar> <out.qanim> [--cmu <dir of CMU .bvh>] [--no-reversed] [--rate HZ] [--no-optimize]\n");
        return 2;
    }
    const std::string dir = argv[1], qchar = argv[2], out = argv[3];
    anim::offline::RetargetParams rp = anim::offline::style100_params();
    anim::offline::CompressParams cp;
    std::string cmu_dir;
    bool reversed = true;
    for (int i = 4; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--rate") && i + 1 < argc) rp.out_rate = float(std::atof(argv[++i]));
        else if (!std::strcmp(argv[i], "--no-optimize")) cp.optimize = false;
        else if (!std::strcmp(argv[i], "--cmu") && i + 1 < argc) cmu_dir = argv[++i];
        else if (!std::strcmp(argv[i], "--no-reversed")) reversed = false;
        else { std::fprintf(stderr, "unknown option %s\n", argv[i]); return 2; }
    }
    std::string why;
    anim::SkeletonDef skel;
    if (!anim::load_qchar_skeleton(qchar.c_str(), skel, &why)) { std::fprintf(stderr, "%s: %s\n", qchar.c_str(), why.c_str()); return 1; }
    // Metres: the same scale the engine gives the mesh (eyes at the player's standing eye height).
    const float stand_eye = game::Player::kHalfExtents.y + game::Player::kEyeOffsetStand;
    anim::scale_skeleton(skel, anim::eye_height_scale(skel, stand_eye));
    std::printf("rig: %d joints, scale %.4f, id %016llx\n", skel.size(), skel.scale, static_cast<unsigned long long>(skel.rig_id()));

    anim::AnimAsset asset;
    anim::offline::CookedClips cooked;
    const auto log = [](const std::string& s) { std::printf("%s\n", s.c_str()); };
    if (!anim::offline::cook_style100(dir, skel, anim::offline::default_style100_sources(), rp, cp, asset, cooked, log, &why)) {
        std::fprintf(stderr, "cook failed: %s\n", why.c_str());
        return 1;
    }
    if (!cmu_dir.empty()) {
        anim::offline::RetargetParams crp = anim::offline::cmu_params();
        crp.out_rate = rp.out_rate;
        if (!anim::offline::cook_takes(cmu_dir, anim::offline::default_cmu_sources(reversed), skel, crp, cp, "CMU Graphics Lab Motion Capture Database",
                                       "free to use and to include in products (mocap.cs.cmu.edu; created with funding from NSF EIA-0196217)",
                                       asset, cooked, log, &why)) {
            std::fprintf(stderr, "cook failed: %s\n", why.c_str());
            return 1;
        }
    }
    size_t clip_bytes = 0, frames = 0;
    for (const auto& b : asset.clip_blobs) clip_bytes += b.size();
    for (const auto& c : cooked.raw) frames += size_t(c.frames);
    std::printf("clips: %zu, %zu frames, %.2f MB compressed (%.1f bytes per frame)\n", asset.clips.size(), frames,
                double(clip_bytes) / 1048576.0, double(clip_bytes) / double(frames));

    if (!anim::offline::build_motion_database(skel, cooked.raw, anim::offline::MotionBuildParams{}, asset.motion_blob, log, &why)) {
        std::fprintf(stderr, "motion database failed: %s\n", why.c_str());
        return 1;
    }
    if (!asset.save(out.c_str(), &why)) { std::fprintf(stderr, "%s: %s\n", out.c_str(), why.c_str()); return 1; }
    std::printf("wrote %s (%.2f MB)\n", out.c_str(), double(asset.serialize().size()) / 1048576.0);
    return 0;
}
