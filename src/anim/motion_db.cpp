#include "anim/motion_db.h"

#include "anim/binary_io.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace qlike::anim {

namespace {
constexpr char kConvention[] = "right-handed, +Y up, root frame x right / z back, metres, seconds";
// Dimensions that hold an x coordinate (sign flips in the mirror).
constexpr int kXDims[] = { 0, 3, 6, 9, 12, 15, 17, 19, 21, 23, 25 };

int group_of(int dim) {
    for (int g = 0; g < kFeatureGroups; ++g)
        if (dim < MotionSchema::kGroupFirst[g + 1]) return g;
    return kFeatureGroups - 1;
}
}  // namespace

const char* MotionSchema::group_name(int g) {
    static const char* const names[kFeatureGroups] = { "foot_pos", "foot_vel", "hip_vel", "traj_pos", "traj_dir" };
    return g >= 0 && g < kFeatureGroups ? names[g] : "?";
}

void MotionDatabase::mirror_raw(const float* raw, float* out) {
    float m[MotionSchema::kFeat];
    for (int k = 0; k < 3; ++k) {
        m[k] = raw[3 + k]; m[3 + k] = raw[k];          // foot positions swap sides
        m[6 + k] = raw[9 + k]; m[9 + k] = raw[6 + k];  // foot velocities too
    }
    for (int k = 12; k < MotionSchema::kFeat; ++k) m[k] = raw[k];
    for (int k : kXDims) m[k] = -m[k];
    std::memcpy(out, m, sizeof(m));
}

std::vector<uint8_t> MotionDatabaseBuilder::serialize() const {
    const size_t n = info.size();
    const int F = MotionSchema::kFeat, S = MotionSchema::kStride;
    // Normalisation over the data and its mirror image together: left/right symmetric by construction.
    double mean[MotionSchema::kFeat] = {}, var[MotionSchema::kFeat] = {};
    float mr[MotionSchema::kFeat];
    for (size_t i = 0; i < n; ++i) {
        const float* r = &raw[i * size_t(F)];
        MotionDatabase::mirror_raw(r, mr);
        for (int k = 0; k < F; ++k) mean[k] += double(r[k]) + double(mr[k]);
    }
    for (int k = 0; k < F; ++k) mean[k] /= double(std::max<size_t>(2 * n, 1));
    for (size_t i = 0; i < n; ++i) {
        const float* r = &raw[i * size_t(F)];
        MotionDatabase::mirror_raw(r, mr);
        for (int k = 0; k < F; ++k) {
            const double a = double(r[k]) - mean[k], b = double(mr[k]) - mean[k];
            var[k] += a * a + b * b;
        }
    }
    float offset[MotionSchema::kStride] = {}, scale[MotionSchema::kStride];
    for (int k = 0; k < S; ++k) scale[k] = 1.0f;
    for (int g = 0; g < kFeatureGroups; ++g) {
        const int a = MotionSchema::kGroupFirst[g], b = MotionSchema::kGroupFirst[g + 1];
        double sd = 0.0;
        for (int k = a; k < b; ++k) sd += std::sqrt(var[k] / double(std::max<size_t>(2 * n, 1)));
        sd /= double(b - a);
        const float s = float(std::max(sd, 1e-6)) / std::max(group_weight[g], 1e-6f);
        for (int k = a; k < b; ++k) { scale[k] = s; offset[k] = float(mean[k]); }
    }
    for (int k : kXDims) offset[k] = 0.0f;              // (exactly zero, not "zero up to rounding")

    ByteWriter w;
    w.raw("QMDB", 4);
    w.pod(MotionSchema::kVersion);
    w.pod(rig_id);
    w.str(kConvention);
    w.pod(rate);
    w.pod(uint32_t(F));
    w.pod(uint32_t(S));
    w.pod(uint32_t(MotionSchema::kHorizons));
    for (float h : horizons) w.pod(h);
    w.pod(uint32_t(kFeatureGroups));
    for (int g = 0; g < kFeatureGroups; ++g) {
        w.str(MotionSchema::group_name(g));
        w.pod(uint32_t(MotionSchema::kGroupFirst[g]));
        w.pod(uint32_t(MotionSchema::kGroupFirst[g + 1] - MotionSchema::kGroupFirst[g]));
        w.pod(group_weight[g]);
    }
    w.raw(offset, sizeof(offset));
    w.raw(scale, sizeof(scale));
    w.array(ranges);
    w.array(jumps);
    w.array(info);
    std::vector<float> feat(n * size_t(S), 0.0f);
    for (size_t i = 0; i < n; ++i)
        for (int k = 0; k < F; ++k) feat[i * size_t(S) + size_t(k)] = (raw[i * size_t(F) + size_t(k)] - offset[k]) / scale[k];
    w.array(feat);
    return std::move(w.bytes);
}

// (of the frames going one way, the share that is slower than the "cruise" pace)
static const double kCruisePercentile = 0.95;

bool MotionDatabase::load(const uint8_t* data, size_t size, uint64_t rig_id, std::string* why) {
    *this = MotionDatabase{};
    auto fail = [&](const char* s) { if (why) *why = s; *this = MotionDatabase{}; return false; };
    ByteReader r(data, size);
    char magic[4];
    r.raw(magic, 4);
    if (!r.ok || std::memcmp(magic, "QMDB", 4) != 0) return fail("no motion database in the asset");
    if (r.pod<uint32_t>() != MotionSchema::kVersion) return fail("motion database has another version: cook the asset again");
    if (r.pod<uint64_t>() != rig_id) return fail("motion database was built for another rig");
    if (r.str() != kConvention) return fail("motion database uses another coordinate convention");
    rate_ = r.pod<float>();
    const uint32_t F = r.pod<uint32_t>(), S = r.pod<uint32_t>(), H = r.pod<uint32_t>();
    if (!r.ok || F != uint32_t(MotionSchema::kFeat) || S != uint32_t(MotionSchema::kStride) || H != uint32_t(MotionSchema::kHorizons))
        return fail("motion database has another feature layout");
    for (float& h : horizons_) h = r.pod<float>();
    if (r.pod<uint32_t>() != uint32_t(kFeatureGroups)) return fail("motion database has another feature layout");
    for (int g = 0; g < kFeatureGroups; ++g) {
        const std::string name = r.str();
        const uint32_t first = r.pod<uint32_t>(), count = r.pod<uint32_t>();
        (void)r.pod<float>();
        if (name != MotionSchema::group_name(g) || first != uint32_t(MotionSchema::kGroupFirst[g]) ||
            count != uint32_t(MotionSchema::kGroupFirst[g + 1] - MotionSchema::kGroupFirst[g]))
            return fail("motion database has another feature layout");
    }
    r.raw(offset_, sizeof(offset_));
    r.raw(scale_, sizeof(scale_));
    ranges_ = r.array<MotionRange>();
    jumps_ = r.array<MotionJump>();
    info_ = r.array<MotionFrameInfo>();
    feat_ = r.array<float>();
    if (!r.ok) return fail("motion database truncated");
    if (info_.empty() || info_.size() > 16000000 || feat_.size() != info_.size() * size_t(S)) return fail("motion database arrays do not fit together");
    if (!(rate_ >= 10.0f && rate_ <= 240.0f)) return fail("motion database: bad sample rate");
    for (float v : feat_) if (!std::isfinite(v)) return fail("motion database holds a value that is not finite");
    for (int k = 0; k < MotionSchema::kFeat; ++k) if (!(scale_[k] > 0.0f) || !std::isfinite(offset_[k])) return fail("motion database: bad normalisation");
    frames_ = uint32_t(info_.size());
    range_of_.assign(frames_, 0);
    uint32_t covered = 0;
    for (size_t i = 0; i < ranges_.size(); ++i) {
        const MotionRange& rg = ranges_[i];
        if (rg.first != covered || rg.count == 0 || rg.first + rg.count > frames_) return fail("motion database: ranges do not tile the frames");
        for (uint32_t f = rg.first; f < rg.first + rg.count; ++f) range_of_[f] = uint32_t(i);
        covered += rg.count;
    }
    if (covered != frames_) return fail("motion database: ranges do not tile the frames");
    for (const MotionJump& j : jumps_)
        if (!(j.takeoff < j.land && j.land < frames_ && j.apex >= j.takeoff && j.apex <= j.land && range_of_[j.takeoff] == range_of_[j.land]))
            return fail("motion database: bad jump entry");
    // The boxes the search prunes with.
    {
        const uint32_t S = uint32_t(MotionSchema::kStride), blocks = (frames_ + kBlock - 1) / kBlock;
        block_lo_.assign(size_t(blocks) * S, 1e30f);
        block_hi_.assign(size_t(blocks) * S, -1e30f);
        for (uint32_t f = 0; f < frames_; ++f) {
            const float* r = row(f);
            float* lo = &block_lo_[size_t(f / kBlock) * S];
            float* hi = &block_hi_[size_t(f / kBlock) * S];
            for (uint32_t k = 0; k < S; ++k) { lo[k] = std::min(lo[k], r[k]); hi[k] = std::max(hi[k], r[k]); }
        }
    }
    // What pace each direction of the capture sustains (see cruise_speed).
    for (int stance = 0; stance < 2; ++stance) {
        std::vector<float> sp[3];
        for (const MotionRange& rg : ranges_) {
            if (rg.tags & kTagJump) continue;
            if (stance == 0 ? !(rg.tags & kTagLocomotion) : !(rg.tags & kTagCrouch)) continue;
            for (uint32_t f = rg.first; f < rg.first + rg.count; ++f) {
                const float x = std::abs(info_[f].root_vel_x), z = info_[f].root_vel_z, v = std::sqrt(x * x + z * z);
                if (v < 0.4f) continue;
                const int axis = -z > 0.82f * v ? 0 : z > 0.82f * v ? 2 : x > 0.82f * v ? 1 : -1;      // within 35 degrees
                if (axis >= 0) sp[axis].push_back(v);
            }
        }
        for (int a = 0; a < 3; ++a) {
            if (sp[a].size() < 30) continue;
            const size_t nth = size_t(double(sp[a].size() - 1) * kCruisePercentile);
            std::nth_element(sp[a].begin(), sp[a].begin() + std::ptrdiff_t(nth), sp[a].end());
            cruise_[stance][a] = sp[a][nth];
        }
    }
    return true;
}

float MotionDatabase::time_in_clip(uint32_t frame, float frac) const {
    return (float(frame - range_of(frame).first) + frac) / rate_;
}

void MotionDatabase::raw_features(uint32_t frame, bool mirrored, float* raw) const {
    const float* f = row(frame);
    for (int k = 0; k < MotionSchema::kFeat; ++k) raw[k] = f[k] * scale_[k] + offset_[k];
    if (mirrored) mirror_raw(raw, raw);
}

void MotionDatabase::normalise(const float* raw, float* out) const {
    for (int k = 0; k < MotionSchema::kFeat; ++k) out[k] = (raw[k] - offset_[k]) / scale_[k];
    for (int k = MotionSchema::kFeat; k < MotionSchema::kStride; ++k) out[k] = 0.0f;
}

size_t MotionDatabase::memory_bytes() const {
    return feat_.size() * sizeof(float) + info_.size() * sizeof(MotionFrameInfo) + range_of_.size() * sizeof(uint32_t) +
           ranges_.size() * sizeof(MotionRange) + jumps_.size() * sizeof(MotionJump);
}

namespace {
struct Prepared {
    float q[2][MotionSchema::kStride];      // the query, and the query mirrored (both normalised)
    float w[MotionSchema::kStride];         // weight per dimension, squared
};

Prepared prepare(const MotionDatabase& db, const MotionQuery& q) {
    Prepared p;
    float mr[MotionSchema::kFeat];
    db.normalise(q.raw, p.q[0]);
    MotionDatabase::mirror_raw(q.raw, mr);
    db.normalise(mr, p.q[1]);
    for (int k = 0; k < MotionSchema::kStride; ++k) {
        const float gw = k < MotionSchema::kFeat ? q.group_weight[group_of(k)] : 0.0f;
        p.w[k] = gw * gw;
    }
    return p;
}

inline float row_cost(const float* f, const float* q, const float* w) {
#if defined(__AVX2__)
    __m256 acc = _mm256_setzero_ps();
    for (int k = 0; k < MotionSchema::kStride; k += 8) {
        const __m256 d = _mm256_sub_ps(_mm256_loadu_ps(f + k), _mm256_loadu_ps(q + k));
        acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_mul_ps(d, d), _mm256_loadu_ps(w + k)));
    }
    const __m128 s = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
    const __m128 h = _mm_add_ps(s, _mm_movehl_ps(s, s));
    return _mm_cvtss_f32(_mm_add_ss(h, _mm_shuffle_ps(h, h, 1)));
#else
    float c = 0.0f;
    for (int k = 0; k < MotionSchema::kStride; ++k) { const float d = f[k] - q[k]; c += d * d * w[k]; }
    return c;
#endif
}

// The least a row inside the box [lo, hi] can cost.
inline float box_cost(const float* lo, const float* hi, const float* q, const float* w) {
#if defined(__AVX2__)
    __m256 acc = _mm256_setzero_ps();
    const __m256 zero = _mm256_setzero_ps();
    for (int k = 0; k < MotionSchema::kStride; k += 8) {
        const __m256 qq = _mm256_loadu_ps(q + k);
        const __m256 d = _mm256_max_ps(zero, _mm256_max_ps(_mm256_sub_ps(_mm256_loadu_ps(lo + k), qq), _mm256_sub_ps(qq, _mm256_loadu_ps(hi + k))));
        acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_mul_ps(d, d), _mm256_loadu_ps(w + k)));
    }
    const __m128 s = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
    const __m128 h = _mm_add_ps(s, _mm_movehl_ps(s, s));
    return _mm_cvtss_f32(_mm_add_ss(h, _mm_shuffle_ps(h, h, 1)));
#else
    float c = 0.0f;
    for (int k = 0; k < MotionSchema::kStride; ++k) { const float d = std::max(0.0f, std::max(lo[k] - q[k], q[k] - hi[k])); c += d * d * w[k]; }
    return c;
#endif
}

bool range_allowed(const MotionRange& r, const MotionQuery& q) {
    return (r.tags & q.require_tags) == q.require_tags && (r.tags & q.exclude_tags) == 0 && r.count > r.head + r.tail;
}
}  // namespace

MotionMatch MotionDatabase::cost(const MotionQuery& q, uint32_t frame, bool mirrored) const {
    MotionMatch m;
    if (frame >= frames_) return m;
    const Prepared p = prepare(*this, q);
    const float* f = row(frame);
    const float* qq = p.q[mirrored ? 1 : 0];
    m.frame = int(frame);
    m.mirrored = mirrored;
    m.cost = 0.0f;
    for (int g = 0; g < kFeatureGroups; ++g) {
        float c = 0.0f;
        for (int k = MotionSchema::kGroupFirst[g]; k < MotionSchema::kGroupFirst[g + 1]; ++k) { const float d = f[k] - qq[k]; c += d * d * p.w[k]; }
        m.group_cost[g] = c;
        m.cost += c;
    }
    return m;
}

MotionMatch MotionDatabase::search(const MotionQuery& q) const {
    const Prepared p = prepare(*this, q);
    float best = 1e30f;
    int best_frame = -1, best_mirror = 0;
    uint64_t looked = 0, skipped = 0;
    for (const MotionRange& r : ranges_) {
        if (!range_allowed(r, q)) continue;
        const uint32_t a = r.first + r.head, b = r.first + r.count - r.tail;
        for (int m = 0; m < 2; ++m) {
            // block by block: a block whose box cannot beat the best so far is skipped whole
            // (with a hair of margin for rounding, so nothing that could win is ever skipped)
            for (uint32_t i0 = a; i0 < b;) {
                const uint32_t blk = i0 / kBlock, i1 = std::min(b, (blk + 1) * kBlock);
                const size_t at = size_t(blk) * size_t(MotionSchema::kStride);
                if (box_cost(&block_lo_[at], &block_hi_[at], p.q[m], p.w) * 0.9999f >= best) { skipped += i1 - i0; i0 = i1; continue; }
                const float* f = row(i0);
                for (uint32_t i = i0; i < i1; ++i, f += MotionSchema::kStride) {
                    const float c = row_cost(f, p.q[m], p.w);
                    if (c < best) { best = c; best_frame = int(i); best_mirror = m; }
                }
                looked += i1 - i0;
                i0 = i1;
            }
        }
    }
    rows_looked_at_.v.fetch_add(looked, std::memory_order_relaxed);
    rows_skipped_.v.fetch_add(skipped, std::memory_order_relaxed);
    if (best_frame < 0) return MotionMatch{};
    return cost(q, uint32_t(best_frame), best_mirror != 0);
}

MotionMatch MotionDatabase::search_reference(const MotionQuery& q) const {
    MotionMatch best;
    float raw[MotionSchema::kFeat], qn[MotionSchema::kStride], fn[MotionSchema::kStride];
    normalise(q.raw, qn);
    for (const MotionRange& r : ranges_) {
        if (!range_allowed(r, q)) continue;
        for (uint32_t i = r.first + r.head; i < r.first + r.count - r.tail; ++i) {
            for (int m = 0; m < 2; ++m) {
                // The candidate's own features, mirrored for real, against the query as given.
                raw_features(i, m != 0, raw);
                normalise(raw, fn);
                double c = 0.0;
                for (int k = 0; k < MotionSchema::kFeat; ++k) {
                    const double d = double(fn[k]) - double(qn[k]), gw = double(q.group_weight[group_of(k)]);
                    c += d * d * gw * gw;
                }
                if (c < double(best.cost)) { best.cost = float(c); best.frame = int(i); best.mirrored = m != 0; }
            }
        }
    }
    return best;
}

}  // namespace qlike::anim
