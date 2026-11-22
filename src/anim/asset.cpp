#include "anim/asset.h"

#include "anim/binary_io.h"

#include <fstream>

namespace vibe::anim {

std::vector<uint8_t> AnimAsset::serialize() const {
    ByteWriter w;
    w.raw("QANM", 4);
    w.pod(kVersion);
    w.pod(uint32_t(skeleton.size()));
    for (int i = 0; i < skeleton.size(); ++i) {
        w.str(skeleton.names[i]);
        w.pod(int32_t(skeleton.parents[i]));
        w.pod(skeleton.rest_pos[i]);
        w.pod(skeleton.rest_tail[i]);
    }
    w.pod(skeleton.scale);
    w.pod(skeleton.rig_id());
    w.str(runtime_format);
    w.blob(skeleton_blob);
    w.pod(uint32_t(clips.size()));
    for (size_t i = 0; i < clips.size(); ++i) {
        const ClipInfo& c = clips[i];
        w.str(c.name);
        w.pod(c.duration);
        w.pod(c.sample_rate);
        w.pod(uint8_t(c.loop ? 1 : 0));
        w.pod(c.tags);
        w.str(c.provenance);
        w.str(c.licence);
        w.blob(clip_blobs[i]);
    }
    w.blob(motion_blob);
    return std::move(w.bytes);
}

bool AnimAsset::deserialize(const uint8_t* data, size_t size, std::string* why) {
    *this = AnimAsset{};
    auto fail = [&](const char* s) { if (why) *why = s; *this = AnimAsset{}; return false; };
    ByteReader r(data, size);
    char magic[4];
    r.raw(magic, 4);
    if (!r.ok || std::memcmp(magic, "QANM", 4) != 0) return fail("not an animation asset (QANM)");
    const uint32_t ver = r.pod<uint32_t>();
    if (ver != kVersion) return fail("animation asset has another version: cook it again (tools/anim)");
    const uint32_t nj = r.pod<uint32_t>();
    if (!r.ok || nj == 0 || nj > 1024) return fail("bad joint count");
    for (uint32_t i = 0; i < nj; ++i) {
        skeleton.names.push_back(r.str());
        skeleton.parents.push_back(r.pod<int32_t>());
        skeleton.rest_pos.push_back(r.pod<glm::vec3>());
        skeleton.rest_tail.push_back(r.pod<glm::vec3>());
    }
    skeleton.scale = r.pod<float>();
    const uint64_t rig = r.pod<uint64_t>();
    std::string v;
    if (!r.ok || !skeleton.validate(&v)) return fail("bad skeleton block");
    if (rig != skeleton.rig_id()) return fail("skeleton block does not match its rig id");
    runtime_format = r.str();
    skeleton_blob = r.blob();
    const uint32_t nc = r.pod<uint32_t>();
    if (!r.ok || nc > 65536) return fail("bad clip count");
    for (uint32_t i = 0; i < nc; ++i) {
        ClipInfo c;
        c.name = r.str();
        c.duration = r.pod<float>();
        c.sample_rate = r.pod<float>();
        c.loop = r.pod<uint8_t>() != 0;
        c.tags = r.pod<uint32_t>();
        c.provenance = r.str();
        c.licence = r.str();
        clip_blobs.push_back(r.blob());
        if (!r.ok) return fail("animation asset truncated (clips)");
        clips.push_back(std::move(c));
    }
    motion_blob = r.blob();
    if (!r.ok) return fail("animation asset truncated");
    return true;
}

bool AnimAsset::save(const char* path, std::string* why) const {
    const std::vector<uint8_t> b = serialize();
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (f.good()) f.write(reinterpret_cast<const char*>(b.data()), std::streamsize(b.size()));
    if (!f.good()) { if (why) *why = "cannot write the animation asset"; return false; }
    return true;
}

bool AnimAsset::load(const char* path, std::string* why) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.good()) { if (why) *why = "cannot open the animation asset"; *this = AnimAsset{}; return false; }
    const std::streamsize n = f.tellg();
    f.seekg(0);
    std::vector<uint8_t> b(static_cast<size_t>(n));
    f.read(reinterpret_cast<char*>(b.data()), n);
    if (!f.good()) { if (why) *why = "cannot read the animation asset"; *this = AnimAsset{}; return false; }
    return deserialize(b.data(), b.size(), why);
}

int AnimAsset::find_clip(std::string_view name) const {
    for (size_t i = 0; i < clips.size(); ++i)
        if (clips[i].name == name) return int(i);
    return -1;
}

}  // namespace vibe::anim
