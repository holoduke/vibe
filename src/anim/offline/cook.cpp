#include "anim/offline/cook.h"

#include "anim/runtime.h"

#include <algorithm>
#include <cstdlib>
#include <ozz/animation/offline/animation_builder.h>
#include <ozz/animation/offline/animation_optimizer.h>
#include <ozz/animation/offline/raw_animation.h>
#include <ozz/animation/offline/raw_skeleton.h>
#include <ozz/animation/offline/skeleton_builder.h>
#include <ozz/animation/runtime/animation.h>
#include <ozz/animation/runtime/skeleton.h>
#include <ozz/base/io/archive.h>
#include <ozz/base/io/stream.h>
#include <ozz/base/memory/unique_ptr.h>

#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>

namespace qlike::anim::offline {

namespace {

template <typename T>
bool write_blob(const T& obj, std::vector<uint8_t>& blob) {
    ozz::io::MemoryStream ms;
    {
        ozz::io::OArchive ar(&ms);
        ar << obj;
    }
    blob.resize(ms.Size());
    ms.Seek(0, ozz::io::Stream::kSet);
    return ms.Read(blob.data(), blob.size()) == blob.size();
}

void add_children(const SkeletonDef& s, int parent, ozz::animation::offline::RawSkeleton::Joint::Children& dst) {
    for (int j = 0; j < s.size(); ++j) {
        if (s.parents[size_t(j)] != parent) continue;
        ozz::animation::offline::RawSkeleton::Joint jt;
        jt.name = s.names[size_t(j)].c_str();
        const glm::vec3 t = s.rest_local(j);
        jt.transform.translation = ozz::math::Float3(t.x, t.y, t.z);
        jt.transform.rotation = ozz::math::Quaternion::identity();
        jt.transform.scale = ozz::math::Float3::one();
        add_children(s, j, jt.children);
        dst.push_back(std::move(jt));
    }
}

std::unique_ptr<ozz::animation::Skeleton, void (*)(ozz::animation::Skeleton*)> make_skeleton(const SkeletonDef& s, std::string* why) {
    using Ptr = std::unique_ptr<ozz::animation::Skeleton, void (*)(ozz::animation::Skeleton*)>;
    auto del = [](ozz::animation::Skeleton* p) { ozz::Delete(p); };
    ozz::animation::offline::RawSkeleton raw;
    add_children(s, -1, raw.roots);
    if (!raw.Validate()) { if (why) *why = "the skeleton is not valid for the runtime"; return Ptr(nullptr, del); }
    ozz::animation::offline::SkeletonBuilder builder;
    ozz::unique_ptr<ozz::animation::Skeleton> sk = builder(raw);
    if (!sk) { if (why) *why = "the runtime could not build the skeleton"; return Ptr(nullptr, del); }
    // The runtime orders joints depth first; so must the character file, or mesh bone indices would not match.
    if (sk->num_joints() != s.size()) { if (why) *why = "joint count changed in the skeleton build"; return Ptr(nullptr, del); }
    for (int i = 0; i < s.size(); ++i) {
        if (s.names[size_t(i)] != sk->joint_names()[i] || s.parents[size_t(i)] != int(sk->joint_parents()[i])) {
            if (why) *why = "the character file's joints are not in depth-first order (joint " + s.names[size_t(i)] + ")";
            return Ptr(nullptr, del);
        }
    }
    return Ptr(sk.release(), del);
}

}  // namespace

bool build_runtime_skeleton(const SkeletonDef& s, std::vector<uint8_t>& blob, std::string* why) {
    auto sk = make_skeleton(s, why);
    if (!sk) return false;
    if (!write_blob(*sk, blob)) { if (why) *why = "cannot serialize the skeleton"; return false; }
    return true;
}

bool build_runtime_clip(const SkeletonDef& s, const RawClip& c, const CompressParams& p, std::vector<uint8_t>& blob,
                        std::string* why) {
    auto fail = [&](const std::string& m) { if (why) *why = m; return false; };
    if (c.joints != s.size() || c.frames < 2) return fail("clip does not fit the skeleton");
    const int hips = s.find("Hips");
    ozz::animation::offline::RawAnimation raw;
    raw.duration = c.info.duration;
    raw.name = c.info.name.c_str();
    raw.tracks.resize(size_t(c.joints));
    for (int j = 0; j < c.joints; ++j) {
        auto& tr = raw.tracks[size_t(j)];
        const bool moves = j == 0 || j == hips;
        if (!moves) {
            const glm::vec3 t = s.rest_local(j);
            tr.translations.push_back({ 0.0f, ozz::math::Float3(t.x, t.y, t.z) });
        }
        tr.rotations.reserve(size_t(c.frames));
        for (int f = 0; f < c.frames; ++f) {
            const float time = std::min(c.time_of(f), raw.duration);
            const Transform& t = c.at(f, j);
            if (moves) tr.translations.push_back({ time, ozz::math::Float3(t.t.x, t.t.y, t.t.z) });
            tr.rotations.push_back({ time, ozz::math::Quaternion(t.r.x, t.r.y, t.r.z, t.r.w) });
        }
    }
    if (!raw.Validate()) return fail("clip is not valid for the runtime: " + c.info.name);
    auto sk = make_skeleton(s, why);
    if (!sk) return false;
    ozz::animation::offline::RawAnimation opt;
    const ozz::animation::offline::RawAnimation* src = &raw;
    if (p.optimize) {
        ozz::animation::offline::AnimationOptimizer optimizer;
        optimizer.setting.tolerance = p.tolerance;
        optimizer.setting.distance = p.distance;
        if (!optimizer(raw, *sk, &opt)) return fail("clip optimisation failed: " + c.info.name);
        src = &opt;
    }
    ozz::animation::offline::AnimationBuilder builder;
    ozz::unique_ptr<ozz::animation::Animation> anim = builder(*src);
    if (!anim) return fail("the runtime could not build clip " + c.info.name);
    if (!write_blob(*anim, blob)) return fail("cannot serialize clip " + c.info.name);
    return true;
}

std::vector<Style100Source> default_style100_sources() {
    return {
        { "Neutral",     "FW,FR,BW,BR,SW,SR,ID,TR1,TR2,TR3", kTagLocomotion },
        { "StartStop",   "FW,FR,BW,BR,SW,SR",                kTagLocomotion },
        { "Rushed",      "FW,FR,BW,BR,SW,SR",                kTagLocomotion },      // the fastest takes of the set (to 2.6 m/s)
        { "Crouched",    "FW,FR,BW,BR,SW,SR,ID,TR1,TR2,TR3", kTagCrouch },
        { "TwoFootJump", "FW,FR,BW,SW,ID",                   kTagJump },
    };
}

bool cook_style100(const std::string& dir, const SkeletonDef& skel, const std::vector<Style100Source>& sources,
                   const RetargetParams& rp, const CompressParams& cp, AnimAsset& asset, CookedClips& cooked,
                   const CookLog& log, std::string* why) {
    auto fail = [&](const std::string& m) { if (why) *why = m; return false; };
    // Frame_Cuts.csv: per style, the usable frame range of each take.
    std::map<std::string, std::map<std::string, std::string>> cuts;
    {
        std::ifstream f(dir + "/Frame_Cuts.csv");
        if (!f.good()) return fail("cannot open " + dir + "/Frame_Cuts.csv");
        std::string line;
        std::vector<std::string> head;
        auto split = [](const std::string& l) {
            std::vector<std::string> v;
            std::stringstream ss(l);
            std::string c;
            while (std::getline(ss, c, ',')) {
                while (!c.empty() && (c.back() == '\r' || c.back() == ' ')) c.pop_back();
                v.push_back(c);
            }
            return v;
        };
        if (std::getline(f, line)) head = split(line);
        if (!head.empty() && head[0].size() >= 3 && static_cast<unsigned char>(head[0][0]) == 0xEF) head[0] = head[0].substr(3);   // BOM
        while (std::getline(f, line)) {
            const std::vector<std::string> v = split(line);
            if (v.empty()) continue;
            for (size_t i = 1; i < v.size() && i < head.size(); ++i) cuts[v[0]][head[i]] = v[i];
        }
    }
    asset = AnimAsset{};
    asset.skeleton = skel;
    asset.runtime_format = runtime_format_name();
    if (!build_runtime_skeleton(skel, asset.skeleton_blob, why)) return false;
    cooked = CookedClips{};
    for (const Style100Source& src : sources) {
        if (!cuts.count(src.style)) return fail("Frame_Cuts.csv has no style " + src.style);
        std::stringstream ks(src.kinds);
        std::string kind;
        while (std::getline(ks, kind, ',')) {
            const std::string c0 = cuts[src.style][kind + "_START"], c1 = cuts[src.style][kind + "_STOP"];
            if (c0.empty() || c1.empty() || c0 == "N/A" || c1 == "N/A") continue;
            const std::string file = src.style + "/" + src.style + "_" + kind + ".bvh";
            Bvh bvh;
            std::string e;
            if (!load_bvh(dir + "/" + file, bvh, &e)) { if (log) log("skipped " + file + ": " + e); continue; }
            RawClip clip;
            const int first = std::atoi(c0.c_str()), end = std::atoi(c1.c_str());
            if (!retarget_bvh(bvh, first, end, skel, rp, clip, &e)) return fail(file + ": " + e);
            clip.info.name = src.style + "_" + kind;
            clip.info.loop = false;
            clip.info.tags = src.tags | (kind == "ID" && (src.tags & kTagJump) == 0 ? uint32_t(kTagIdle) : 0u);
            clip.info.provenance = "100STYLE " + file + " frames " + c0 + ".." + c1;
            clip.info.licence = "CC BY 4.0 - 100STYLE dataset, Ian Mason, Sebastian Starke, Taku Komura";
            const ClipReport rep = validate_clip(skel, clip);
            if (!rep.ok) return fail(clip.info.name + ": " + rep.errors);
            std::vector<uint8_t> blob;
            if (!build_runtime_clip(skel, clip, cp, blob, why)) return false;
            if (log) {
                char line[256];
                std::snprintf(line, sizeof(line), "%-16s %6d frames %6.1f s  speed mean %.2f max %.2f m/s  contact %2.0f%%/%2.0f%%  %7zu bytes%s%s",
                              clip.info.name.c_str(), clip.frames, clip.info.duration, rep.mean_speed, rep.max_speed,
                              100.0f * rep.contact_left, 100.0f * rep.contact_right, blob.size(),
                              rep.warnings.empty() ? "" : "  WARN: ", rep.warnings.c_str());
                log(line);
            }
            asset.clips.push_back(clip.info);
            asset.clip_blobs.push_back(std::move(blob));
            cooked.raw.push_back(std::move(clip));
            cooked.reports.push_back(rep);
        }
    }
    if (asset.clips.empty()) return fail("no clips were imported");
    return true;
}

std::vector<TakeSource> default_cmu_sources(bool with_reversed) {
    const uint32_t run = kTagLocomotion | kTagShortTake;
    std::vector<TakeSource> v;
    auto add = [&](const char* id) { v.push_back({ std::string(id) + ".bvh", std::string("cmu_") + id, run }); };
    for (const char* id : { "09_01", "09_02", "09_03", "09_04", "09_05", "09_06", "09_07", "09_08", "09_09", "09_10", "09_11" }) add(id);   // run, 3.2 - 3.4 m/s
    for (const char* id : { "16_35", "16_36", "16_45", "16_46", "16_55", "16_56" }) add(id);                                                // jog 2.6, run 3.4 - 4.0
    for (const char* id : { "16_37", "16_39", "16_48", "16_49" }) add(id);                                                                    // veering left / right
    for (const char* id : { "16_08", "16_57" }) add(id);                                                                                      // run to a sudden stop
    for (const char* id : { "35_17", "35_18", "35_19", "35_20", "35_21", "35_22", "35_23", "35_24", "35_25", "35_26" }) add(id);            // run / jog, 3.0 - 3.4
    // Nobody captured a fast back-pedal. The steady runs played backward stand in for one: the
    // same steps, the body travelling the other way (a stretched 1.5 m/s back-pedal is the alternative).
    if (with_reversed)
        for (const char* id : { "09_01", "09_02", "09_03", "09_04", "09_05", "09_06", "09_07", "09_08", "09_09", "09_10", "09_11", "16_35", "16_36", "16_45", "16_46",
                                "35_17", "35_18", "35_19", "35_20", "35_21", "35_22", "35_23", "35_24", "35_25", "35_26" })
            v.push_back({ std::string(id) + ".bvh", std::string("cmu_") + id + "_back", run, true });
    return v;
}

bool cook_takes(const std::string& dir, const std::vector<TakeSource>& sources, const SkeletonDef& skel, const RetargetParams& rp,
                const CompressParams& cp, const std::string& provenance, const std::string& licence, AnimAsset& asset,
                CookedClips& cooked, const CookLog& log, std::string* why) {
    auto fail = [&](const std::string& m) { if (why) *why = m; return false; };
    int added = 0;
    for (const TakeSource& src : sources) {
        Bvh bvh;
        std::string e;
        if (!load_bvh(dir + "/" + src.file, bvh, &e)) { if (log) log("skipped " + src.file + ": " + e); continue; }
        RawClip clip;
        if (!retarget_bvh(bvh, 0, bvh.frames, skel, rp, clip, &e)) return fail(src.file + ": " + e);
        if (src.reversed) {
            // The frames in the opposite order: the body travels backward on the same steps.
            const size_t J = size_t(clip.joints);
            for (int a = 0, b = clip.frames - 1; a < b; ++a, --b) {
                std::swap_ranges(clip.local.begin() + ptrdiff_t(size_t(a) * J), clip.local.begin() + ptrdiff_t(size_t(a + 1) * J), clip.local.begin() + ptrdiff_t(size_t(b) * J));
                std::swap(clip.contact[size_t(a) * 2], clip.contact[size_t(b) * 2]);
                std::swap(clip.contact[size_t(a) * 2 + 1], clip.contact[size_t(b) * 2 + 1]);
            }
        }
        clip.info.name = src.name;
        clip.info.loop = false;
        clip.info.tags = src.tags;
        clip.info.provenance = provenance + " " + src.file + (src.reversed ? " (played backward)" : "");
        clip.info.licence = licence;
        const ClipReport rep = validate_clip(skel, clip);
        if (!rep.ok) return fail(clip.info.name + ": " + rep.errors);
        std::vector<uint8_t> blob;
        if (!build_runtime_clip(skel, clip, cp, blob, why)) return false;
        if (log) {
            char line[256];
            std::snprintf(line, sizeof(line), "%-16s %6d frames %6.1f s  speed mean %.2f max %.2f m/s  contact %2.0f%%/%2.0f%%  %7zu bytes%s%s",
                          clip.info.name.c_str(), clip.frames, clip.info.duration, rep.mean_speed, rep.max_speed,
                          100.0f * rep.contact_left, 100.0f * rep.contact_right, blob.size(),
                          rep.warnings.empty() ? "" : "  WARN: ", rep.warnings.c_str());
            log(line);
        }
        asset.clips.push_back(clip.info);
        asset.clip_blobs.push_back(std::move(blob));
        cooked.raw.push_back(std::move(clip));
        cooked.reports.push_back(rep);
        ++added;
    }
    if (log) log("added " + std::to_string(added) + " of " + std::to_string(sources.size()) + " takes from " + dir);
    return true;
}

}  // namespace qlike::anim::offline
