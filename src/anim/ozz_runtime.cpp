// IAnimationRuntime on ozz-animation. The only runtime file that includes ozz.
#include "anim/runtime.h"

#include <ozz/animation/runtime/animation.h>
#include <ozz/animation/runtime/blending_job.h>
#include <ozz/animation/runtime/ik_aim_job.h>
#include <ozz/animation/runtime/ik_two_bone_job.h>
#include <ozz/animation/runtime/local_to_model_job.h>
#include <ozz/animation/runtime/sampling_job.h>
#include <ozz/animation/runtime/skeleton.h>
#include <ozz/base/containers/vector.h>
#include <ozz/base/io/archive.h>
#include <ozz/base/io/stream.h>
#include <ozz/base/maths/simd_math.h>
#include <ozz/base/memory/unique_ptr.h>
#include <ozz/base/maths/simd_quaternion.h>
#include <ozz/base/maths/soa_transform.h>
#include <ozz/base/span.h>

#include <algorithm>
#include <cmath>
#include <cstring>

namespace vibe::anim {

namespace {

using ozz::math::Float4x4;
using ozz::math::SimdFloat4;
using ozz::math::SoaTransform;

void to_soa(std::span<const Transform> in, ozz::vector<SoaTransform>& out) {
    const size_t n = in.size();
    for (size_t s = 0; s < out.size(); ++s) {
        float t[3][4], r[4][4];
        for (size_t k = 0; k < 4; ++k) {
            const size_t j = s * 4 + k;
            const Transform tr = j < n ? in[j] : Transform{};
            t[0][k] = tr.t.x; t[1][k] = tr.t.y; t[2][k] = tr.t.z;
            r[0][k] = tr.r.x; r[1][k] = tr.r.y; r[2][k] = tr.r.z; r[3][k] = tr.r.w;
        }
        SoaTransform& o = out[s];
        o.translation.x = ozz::math::simd_float4::LoadPtrU(t[0]);
        o.translation.y = ozz::math::simd_float4::LoadPtrU(t[1]);
        o.translation.z = ozz::math::simd_float4::LoadPtrU(t[2]);
        o.rotation.x = ozz::math::simd_float4::LoadPtrU(r[0]);
        o.rotation.y = ozz::math::simd_float4::LoadPtrU(r[1]);
        o.rotation.z = ozz::math::simd_float4::LoadPtrU(r[2]);
        o.rotation.w = ozz::math::simd_float4::LoadPtrU(r[3]);
        o.scale.x = o.scale.y = o.scale.z = ozz::math::simd_float4::one();
    }
}

void from_soa(const ozz::vector<SoaTransform>& in, std::span<Transform> out) {
    const size_t n = out.size();
    for (size_t s = 0; s < in.size(); ++s) {
        float t[3][4], r[4][4];
        const SoaTransform& o = in[s];
        ozz::math::StorePtrU(o.translation.x, t[0]);
        ozz::math::StorePtrU(o.translation.y, t[1]);
        ozz::math::StorePtrU(o.translation.z, t[2]);
        ozz::math::StorePtrU(o.rotation.x, r[0]);
        ozz::math::StorePtrU(o.rotation.y, r[1]);
        ozz::math::StorePtrU(o.rotation.z, r[2]);
        ozz::math::StorePtrU(o.rotation.w, r[3]);
        for (size_t k = 0; k < 4; ++k) {
            const size_t j = s * 4 + k;
            if (j >= n) break;
            out[j].t = glm::vec3(t[0][k], t[1][k], t[2][k]);
            out[j].r = glm::quat(r[3][k], r[0][k], r[1][k], r[2][k]);
        }
    }
}

Float4x4 to_ozz(const glm::mat4& m) {
    Float4x4 o;
    for (int c = 0; c < 4; ++c) o.cols[c] = ozz::math::simd_float4::LoadPtrU(&m[c][0]);
    return o;
}

glm::mat4 to_glm(const Float4x4& m) {
    glm::mat4 o;
    for (int c = 0; c < 4; ++c) ozz::math::StorePtrU(m.cols[c], &o[c][0]);
    return o;
}

SimdFloat4 simd3(glm::vec3 v) { return ozz::math::simd_float4::Load(v.x, v.y, v.z, 0.0f); }

glm::quat to_glm(const ozz::math::SimdQuaternion& q) {
    float f[4];
    ozz::math::StorePtrU(q.xyzw, f);
    return glm::quat(f[3], f[0], f[1], f[2]);
}

template <typename T>
bool read_blob(const std::vector<uint8_t>& blob, T& out) {
    if (blob.empty()) return false;
    ozz::io::MemoryStream ms;
    if (ms.Write(blob.data(), blob.size()) != blob.size()) return false;
    ms.Seek(0, ozz::io::Stream::kSet);
    ozz::io::IArchive ar(&ms);
    if (!ar.TestTag<T>()) return false;
    ar >> out;
    return true;
}

class OzzContext final : public EvalContext {
public:
    OzzContext(int joints, int slots) {
        const int soa = (joints + 3) / 4;
        for (int i = 0; i < slots + 1; ++i) samplers.push_back(ozz::make_unique<ozz::animation::SamplingJob::Context>(joints));
        sampled.resize(size_t(soa));
        blended.resize(size_t(soa));
        models.resize(size_t(joints));
        layer_soa.resize(8);
        for (auto& l : layer_soa) l.resize(size_t(soa));
        mask_soa.resize(8);
        for (auto& m : mask_soa) m.resize(size_t(soa));
        blend_layers.reserve(8);
        add_layers.reserve(8);
    }
    ozz::vector<ozz::unique_ptr<ozz::animation::SamplingJob::Context>> samplers;   // the last one is for root_motion
    ozz::vector<SoaTransform> sampled, blended;
    ozz::vector<Float4x4>     models;
    ozz::vector<ozz::vector<SoaTransform>> layer_soa;
    ozz::vector<ozz::vector<SimdFloat4>>   mask_soa;
    ozz::vector<ozz::animation::BlendingJob::Layer> blend_layers, add_layers;
};

class OzzRuntime final : public IAnimationRuntime {
public:
    bool init(const AnimAsset& asset, std::string* why) {
        auto fail = [&](const std::string& s) { if (why) *why = s; return false; };
        if (asset.runtime_format != runtime_format_name())
            return fail("animation asset was cooked for '" + asset.runtime_format + "', this build reads '" + runtime_format_name() + "'");
        if (!read_blob(asset.skeleton_blob, skeleton_)) return fail("skeleton block is not readable");
        def_ = asset.skeleton;
        if (skeleton_.num_joints() != def_.size()) return fail("skeleton block has another joint count than the asset");
        for (int i = 0; i < def_.size(); ++i) {
            if (def_.names[i] != skeleton_.joint_names()[i] || def_.parents[i] != int(skeleton_.joint_parents()[i]))
                return fail("skeleton block and asset disagree on joint " + def_.names[i]);
        }
        rig_ = make_rig(def_);
        clips_ = asset.clips;
        anims_.resize(clips_.size());
        for (size_t i = 0; i < clips_.size(); ++i) {
            if (!read_blob(asset.clip_blobs[i], anims_[i])) return fail("clip block is not readable: " + clips_[i].name);
            if (anims_[i].num_tracks() != def_.size()) return fail("clip has another track count than the skeleton: " + clips_[i].name);
            clip_bytes_ += anims_[i].size();
        }
        rest_.resize(size_t(def_.size()));
        ozz::vector<SoaTransform> soa(skeleton_.joint_rest_poses().begin(), skeleton_.joint_rest_poses().end());
        from_soa(soa, rest_);
        inv_bind_.resize(size_t(def_.size()));
        for (int i = 0; i < def_.size(); ++i) inv_bind_[size_t(i)] = -def_.rest_pos[size_t(i)];
        return true;
    }

    const SkeletonDef& skeleton() const override { return def_; }
    const Rig&         rig() const override { return rig_; }
    int                num_clips() const override { return int(clips_.size()); }
    const ClipInfo&    clip(ClipId id) const override { return clips_[size_t(id)]; }
    ClipId find_clip(std::string_view name) const override {
        for (size_t i = 0; i < clips_.size(); ++i)
            if (clips_[i].name == name) return ClipId(i);
        return kNoClip;
    }
    std::unique_ptr<EvalContext> create_context(int sample_slots) const override {
        return std::make_unique<OzzContext>(def_.size(), std::max(sample_slots, 1));
    }
    void rest_pose(std::span<Transform> out) const override { std::copy(rest_.begin(), rest_.end(), out.begin()); }

    float ratio_of(ClipId id, float time) const {
        const ClipInfo& c = clips_[size_t(id)];
        if (c.duration <= 1e-6f) return 0.0f;
        float r = time / c.duration;
        if (c.loop) r -= std::floor(r);
        return std::clamp(r, 0.0f, 1.0f);
    }

    void sample(EvalContext& ctx_, int slot, ClipId id, float time, std::span<Transform> out) const override {
        OzzContext& ctx = static_cast<OzzContext&>(ctx_);
        ozz::animation::SamplingJob job;
        job.animation = &anims_[size_t(id)];
        job.context = ctx.samplers[size_t(std::clamp(slot, 0, int(ctx.samplers.size()) - 2))].get();
        job.ratio = ratio_of(id, time);
        job.output = ozz::make_span(ctx.sampled);
        if (!job.Run()) { rest_pose(out); return; }
        from_soa(ctx.sampled, out);
    }

    void blend(EvalContext& ctx_, std::span<const PoseLayer> layers, std::span<const PoseLayer> additive,
               std::span<Transform> out) const override {
        OzzContext& ctx = static_cast<OzzContext&>(ctx_);
        ctx.blend_layers.clear();
        ctx.add_layers.clear();
        size_t used = 0;
        auto add = [&](const PoseLayer& l, ozz::vector<ozz::animation::BlendingJob::Layer>& dst) {
            if (used >= ctx.layer_soa.size()) return;                    // more than 8 layers: the rest is dropped
            to_soa(l.pose, ctx.layer_soa[used]);
            ozz::animation::BlendingJob::Layer bl;
            bl.weight = l.weight;
            bl.transform = ozz::make_span(ctx.layer_soa[used]);
            if (!l.joint_weights.empty()) {
                ozz::vector<SimdFloat4>& m = ctx.mask_soa[used];
                for (size_t s = 0; s < m.size(); ++s) {
                    float w[4];
                    for (size_t k = 0; k < 4; ++k) w[k] = s * 4 + k < l.joint_weights.size() ? l.joint_weights[s * 4 + k] : 0.0f;
                    m[s] = ozz::math::simd_float4::LoadPtrU(w);
                }
                bl.joint_weights = ozz::make_span(m);
            }
            dst.push_back(bl);
            ++used;
        };
        for (const PoseLayer& l : layers) add(l, ctx.blend_layers);
        for (const PoseLayer& l : additive) add(l, ctx.add_layers);
        ozz::animation::BlendingJob job;
        job.threshold = 0.1f;
        job.layers = ozz::make_span(ctx.blend_layers);
        job.additive_layers = ozz::make_span(ctx.add_layers);
        job.rest_pose = skeleton_.joint_rest_poses();
        job.output = ozz::make_span(ctx.blended);
        if (!job.Run()) { rest_pose(out); return; }
        from_soa(ctx.blended, out);
    }

    void local_to_model(EvalContext& ctx_, std::span<const Transform> local, std::span<glm::mat4> model, int from) const override {
        OzzContext& ctx = static_cast<OzzContext&>(ctx_);
        to_soa(local, ctx.sampled);
        const int n = def_.size();
        if (from >= 0)
            for (int i = 0; i < n; ++i) ctx.models[size_t(i)] = to_ozz(model[size_t(i)]);      // the job reads the parents it does not write
        ozz::animation::LocalToModelJob job;
        job.skeleton = &skeleton_;
        job.input = ozz::make_span(ctx.sampled);
        job.output = ozz::make_span(ctx.models);
        if (from >= 0) job.from = from;
        if (!job.Run()) return;
        for (int i = 0; i < n; ++i) model[size_t(i)] = to_glm(ctx.models[size_t(i)]);
    }

    bool two_bone_ik(EvalContext& ctx, const TwoBoneIk& ik, std::span<Transform> local, std::span<glm::mat4> model) const override {
        if (ik.start < 0 || ik.mid < 0 || ik.end < 0) return false;
        const Float4x4 ms = to_ozz(model[size_t(ik.start)]), mm = to_ozz(model[size_t(ik.mid)]), me = to_ozz(model[size_t(ik.end)]);
        ozz::math::SimdQuaternion cs, cm;
        bool reached = false;
        ozz::animation::IKTwoBoneJob job;
        job.target = simd3(ik.target);
        job.pole_vector = simd3(ik.pole);
        {
            // The hinge axis: perpendicular to the two bones as they are now, seen from the middle
            // joint (a positive turn about it opens the limb). Nearly straight limbs have no such
            // perpendicular; then, or if the limb is bent the wrong way, the caller's axis is used.
            const glm::vec3 ps(model[size_t(ik.start)][3]), pm(model[size_t(ik.mid)][3]), pe(model[size_t(ik.end)][3]);
            const glm::vec3 a = ps - pm, b = pe - pm;
            const glm::vec3 c = glm::cross(a, b);
            glm::vec3 axis = glm::normalize(ik.mid_axis);
            const float sin_bend = glm::length(c) / std::max(glm::length(a) * glm::length(b), 1e-9f);
            if (sin_bend > 0.035f) {
                const glm::vec3 local_axis = glm::transpose(glm::mat3(model[size_t(ik.mid)])) * (c / glm::length(c));
                if (glm::dot(local_axis, axis) > 0.0f) axis = glm::normalize(local_axis);
            }
            job.mid_axis = simd3(axis);
        }
        job.soften = ik.soften;
        job.weight = ik.weight;
        job.start_joint = &ms;
        job.mid_joint = &mm;
        job.end_joint = &me;
        job.start_joint_correction = &cs;
        job.mid_joint_correction = &cm;
        job.reached = &reached;
        if (!job.Run()) return false;
        local[size_t(ik.start)].r = glm::normalize(local[size_t(ik.start)].r * to_glm(cs));
        local[size_t(ik.mid)].r = glm::normalize(local[size_t(ik.mid)].r * to_glm(cm));
        local_to_model(ctx, local, model, ik.start);
        return reached;
    }

    void aim_ik(EvalContext& ctx, const AimIk& ik, std::span<Transform> local, std::span<glm::mat4> model) const override {
        if (ik.joint < 0) return;
        const Float4x4 mj = to_ozz(model[size_t(ik.joint)]);
        ozz::math::SimdQuaternion c;
        ozz::animation::IKAimJob job;
        job.target = simd3(ik.target);
        job.forward = simd3(ik.forward);
        job.up = simd3(ik.up);
        job.pole_vector = simd3(ik.pole);
        job.weight = ik.weight;
        job.joint = &mj;
        job.joint_correction = &c;
        if (!job.Run()) return;
        local[size_t(ik.joint)].r = glm::normalize(local[size_t(ik.joint)].r * to_glm(c));
        local_to_model(ctx, local, model, ik.joint);
    }

    void skinning_matrices(std::span<const glm::mat4> model, std::span<glm::mat4> out) const override {
        // The bind pose has no rotation: its inverse is a translation by -rest position.
        for (size_t i = 0; i < inv_bind_.size(); ++i) {
            glm::mat4 m = model[i];
            m[3] = m * glm::vec4(inv_bind_[i], 1.0f);
            out[i] = m;
        }
    }

    Transform root_motion(EvalContext& ctx_, ClipId id, float t0, float t1) const override {
        OzzContext& ctx = static_cast<OzzContext&>(ctx_);
        auto root_at = [&](float t) {
            ozz::animation::SamplingJob job;
            job.animation = &anims_[size_t(id)];
            job.context = ctx.samplers.back().get();
            job.ratio = std::clamp(clips_[size_t(id)].duration > 1e-6f ? t / clips_[size_t(id)].duration : 0.0f, 0.0f, 1.0f);
            job.output = ozz::make_span(ctx.sampled);
            Transform r;
            if (!job.Run()) return r;
            float tx[4], ty[4], tz[4], rx[4], ry[4], rz[4], rw[4];
            const SoaTransform& o = ctx.sampled[0];
            ozz::math::StorePtrU(o.translation.x, tx); ozz::math::StorePtrU(o.translation.y, ty); ozz::math::StorePtrU(o.translation.z, tz);
            ozz::math::StorePtrU(o.rotation.x, rx); ozz::math::StorePtrU(o.rotation.y, ry);
            ozz::math::StorePtrU(o.rotation.z, rz); ozz::math::StorePtrU(o.rotation.w, rw);
            r.t = glm::vec3(tx[0], ty[0], tz[0]);
            r.r = glm::normalize(glm::quat(rw[0], rx[0], ry[0], rz[0]));
            return r;
        };
        const Transform a = root_at(t0), b = root_at(t1);
        Transform d;
        const glm::quat ia = glm::inverse(a.r);
        d.t = ia * (b.t - a.t);
        d.r = glm::normalize(ia * b.r);
        return d;
    }

    size_t clip_bytes() const override { return clip_bytes_; }
    const char* name() const override { return runtime_format_name(); }

private:
    SkeletonDef def_;
    Rig         rig_{};
    ozz::animation::Skeleton skeleton_;
    std::vector<ClipInfo> clips_;
    std::vector<ozz::animation::Animation> anims_;
    std::vector<Transform> rest_;
    std::vector<glm::vec3> inv_bind_;
    size_t clip_bytes_ = 0;
};

}  // namespace

const char* runtime_format_name() { return "ozz-animation 0.17 (skeleton v2, animation v7)"; }

std::unique_ptr<IAnimationRuntime> create_runtime(const AnimAsset& asset, std::string* why) {
    auto rt = std::make_unique<OzzRuntime>();
    if (!rt->init(asset, why)) return nullptr;
    return rt;
}

void mirror_pose(const Rig& rig, std::span<Transform> pose, std::span<Transform> scratch) {
    // Reflection in the plane x = 0: a rotation (x, y, z, w) becomes (x, -y, -z, w), a
    // translation's x changes sign, and each joint takes what its partner of the other side had.
    const size_t n = pose.size();
    for (size_t i = 0; i < n; ++i) {
        const Transform& s = pose[size_t(rig.mirror[i])];
        scratch[i].t = glm::vec3(-s.t.x, s.t.y, s.t.z);
        scratch[i].r = glm::quat(s.r.w, s.r.x, -s.r.y, -s.r.z);
    }
    std::copy(scratch.begin(), scratch.begin() + ptrdiff_t(n), pose.begin());
}

bool pose_is_valid(std::span<const Transform> pose) {
    for (const Transform& t : pose) {
        if (!is_finite(t.t) || !is_finite(t.r)) return false;
        if (std::abs(glm::dot(t.r, t.r) - 1.0f) > 1e-3f) return false;
    }
    return true;
}

bool pose_is_valid(std::span<const glm::mat4> pose) {
    for (const glm::mat4& m : pose)
        if (!is_finite(m)) return false;
    return true;
}

}  // namespace vibe::anim
