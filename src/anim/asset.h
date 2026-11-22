#pragma once
// The cooked animation asset of one character (assets/character/soldier.qanim), written by
// tools/anim (vibe_animcook) and read at start-up.
//
//   "QANM", u32 version
//   skeleton      names, parents, rest positions (metres), scale, rig id
//   runtime       name of the runtime the binary blocks are for ("ozz-animation 0.17"),
//                 skeleton block, one block per clip
//   clips         name, duration, rate, loop, tags, provenance, licence
//   motion        the motion-matching database (motion_db.h), its own version inside
//
// The binary blocks are opaque here; only the runtime that wrote them reads them.
#include "anim/skeleton_def.h"

namespace vibe::anim {

struct AnimAsset {
    static constexpr uint32_t kVersion = 2;

    SkeletonDef                        skeleton;
    std::string                        runtime_format;
    std::vector<uint8_t>               skeleton_blob;
    std::vector<ClipInfo>              clips;
    std::vector<std::vector<uint8_t>>  clip_blobs;
    std::vector<uint8_t>               motion_blob;

    std::vector<uint8_t> serialize() const;
    bool deserialize(const uint8_t* data, size_t size, std::string* why = nullptr);
    bool save(const char* path, std::string* why = nullptr) const;
    bool load(const char* path, std::string* why = nullptr);
    int  find_clip(std::string_view name) const;
};

}  // namespace vibe::anim
