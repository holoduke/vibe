#pragma once
// SOUND IN SPACE, the part that runs on the audio thread (audio.cpp wraps each in a node of the
// mixer's graph; sound_space.h has the curves the game thread sets these by).
//
//   SpatialVoice  one per source. Mono in; out: the two ears, and what goes to the reverb.
//                 gain per ear - low-pass (two one-pole stages) - the far ear later (a delay of
//                 at most 0.66 ms, read between samples) and duller (a one-pole blend).
//   SpaceReverb   ONE for everything: all sources send to it. A pre-delay line with three early
//                 reflections and a single slap-back echo (a wall, a cliff) tapped off it, and a
//                 tail of four delay lines fed back through a Householder matrix with a low-pass
//                 in the loop.
//
// RULES OF THIS FILE. process() runs in the device's callback: no allocation, no lock, no
// system call, no branch a sample (per block is fine). The game thread talks to it through
// atomics only (set()); every parameter is FOLLOWED a sample at a time (a one-pole, 20 ms for
// a voice, 60 ms for the reverb), so whatever the game thread writes once a frame arrives as
// a curve, never as a step. Buffers are sized in prepare(), before the node is in the graph.
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace vibe::sound_dsp {

constexpr int kChunk = 512;       // frames worked at a time (the mixer asks for at most 480)

// A one-pole low-pass's coefficient for a cut-off: y += a (x - y).
inline float onepole(float hz, float sample_rate) {
    return 1.0f - std::exp(-6.2831853f * (std::max)(hz, 1.0f) / sample_rate);
}

class SpatialVoice {
public:
    enum Param { GainL, GainR, Lowpass, ShadowL, ShadowR, DelayL, DelayR, Send, ParamCount };

    void prepare(float sample_rate) {
        sr_ = sample_rate;
        follow_ = 1.0f - std::exp(-1.0f / (0.020f * sr_));
        shadow_a_ = onepole(1500.0f, sr_);
        const float init[ParamCount] = { 0.0f, 0.0f, onepole(4000.0f, sr_), 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
        for (int i = 0; i < ParamCount; ++i) { tgt_[i].store(init[i], std::memory_order_relaxed); cur_[i] = init[i]; }
    }
    float sample_rate() const { return sr_; }
    // (game thread) DelayL / DelayR in samples; Lowpass as onepole(hz).
    void set(Param p, float v) { tgt_[p].store(v, std::memory_order_relaxed); }

    // (audio thread) in: `in_ch` interleaved channels (the first two are averaged). dry, send:
    // interleaved; left and right go to their first two channels, the rest are silent.
    void process(const float* in, int in_ch, float* dry, int dry_ch, float* send, int send_ch, int frames) {
        float tgt[ParamCount];
        for (int i = 0; i < ParamCount; ++i) tgt[i] = tgt_[i].load(std::memory_order_relaxed);
        tgt[DelayL] = std::clamp(tgt[DelayL], 0.0f, float(kRing - 4));
        tgt[DelayR] = std::clamp(tgt[DelayR], 0.0f, float(kRing - 4));
        const int i1 = in_ch >= 2 ? 1 : 0;
        float L[kChunk], R[kChunk], S[kChunk];
        for (int done = 0; done < frames;) {
            const int n = (std::min)(frames - done, kChunk);
            const float* src = in + size_t(done) * size_t(in_ch);
            float c[ParamCount];
            for (int i = 0; i < ParamCount; ++i) c[i] = cur_[i];
            float lp1 = lp1_, lp2 = lp2_, shl = shl_, shr = shr_;
            uint32_t w = w_;
            const float k = follow_, sa = shadow_a_;
            for (int f = 0; f < n; ++f) {
                for (int i = 0; i < ParamCount; ++i) c[i] += (tgt[i] - c[i]) * k;
                const float x = 0.5f * (src[f * in_ch] + src[f * in_ch + i1]);
                lp1 += c[Lowpass] * (x - lp1);
                lp2 += c[Lowpass] * (lp1 - lp2);
                ring_[w & (kRing - 1)] = lp2;
                // each ear: its own lateness (between two samples), then its loss of highs
                const int dl = int(c[DelayL]), dr = int(c[DelayR]);
                const float fl = c[DelayL] - float(dl), fr = c[DelayR] - float(dr);
                const float l0 = ring_[(w - uint32_t(dl)) & (kRing - 1)], l1 = ring_[(w - uint32_t(dl) - 1u) & (kRing - 1)];
                const float r0 = ring_[(w - uint32_t(dr)) & (kRing - 1)], r1 = ring_[(w - uint32_t(dr) - 1u) & (kRing - 1)];
                const float l = l0 + fl * (l1 - l0), r = r0 + fr * (r1 - r0);
                shl += sa * (l - shl);
                shr += sa * (r - shr);
                L[f] = (l + c[ShadowL] * (shl - l)) * c[GainL];
                R[f] = (r + c[ShadowR] * (shr - r)) * c[GainR];
                S[f] = lp2 * c[Send];
                ++w;
            }
            for (int i = 0; i < ParamCount; ++i) cur_[i] = c[i];
            lp1_ = lp1; lp2_ = lp2; shl_ = shl; shr_ = shr; w_ = w;
            write(dry + size_t(done) * size_t(dry_ch), dry_ch, L, R, n);
            write(send + size_t(done) * size_t(send_ch), send_ch, S, S, n);
            done += n;
        }
    }

    // Left and right into an interleaved block of any width.
    static void write(float* out, int ch, const float* L, const float* R, int n) {
        if (ch == 2) { for (int f = 0; f < n; ++f) { out[2 * f] = L[f]; out[2 * f + 1] = R[f]; } return; }
        if (ch == 1) { for (int f = 0; f < n; ++f) out[f] = 0.7071f * (L[f] + R[f]); return; }
        std::memset(out, 0, sizeof(float) * size_t(n) * size_t(ch));
        for (int f = 0; f < n; ++f) { out[ch * f] = L[f]; out[ch * f + 1] = R[f]; }
    }

private:
    static constexpr int kRing = 256;      // (0.66 ms is 32 samples at 48 kHz)
    std::atomic<float> tgt_[ParamCount];
    float cur_[ParamCount]{};
    float sr_ = 48000.0f, follow_ = 0.001f, shadow_a_ = 0.2f;
    float lp1_ = 0.0f, lp2_ = 0.0f, shl_ = 0.0f, shr_ = 0.0f;
    float ring_[kRing]{};
    uint32_t w_ = 0;
};

// The reverb's own constants (set by numbers; sound_space.h has the rest).
constexpr float kTailLineMs[4] = { 32.5f, 40.6f, 50.8f, 60.8f };   // the four lines of the tail: unlike lengths, so no one pitch rings
constexpr float kEarlyMs[3]    = { 17.0f, 31.0f, 47.0f };          // the first reflections (left, right, both)
constexpr float kPreDelayMs    = 12.0f;                            // the tail begins this long after the sound
constexpr float kEarlyLevel    = 0.70f;
constexpr float kTailLevel     = 1.00f;
constexpr float kSlapHz        = 2600.0f;                          // an echo off a far wall has lost its highs
constexpr float kSlapMaxS      = 1.2f;

class SpaceReverb {
public:
    enum Param { Fb0, Fb1, Fb2, Fb3, Damp, Early, Tail, SlapL, SlapR, ParamCount };

    void prepare(float sample_rate) {
        sr_ = sample_rate;
        follow_ = 1.0f - std::exp(-1.0f / (0.060f * sr_));
        size_t pre = 1; while (float(pre) < sr_ * (kSlapMaxS + 0.1f)) pre <<= 1;
        pre_.assign(pre, 0.0f); pre_mask_ = uint32_t(pre - 1);
        size_t ln = 1; while (float(ln) < sr_ * kTailLineMs[3] * 0.001f + 8.0f) ln <<= 1;
        for (int i = 0; i < 4; ++i) { line_[i].assign(ln, 0.0f); len_[i] = uint32_t(sr_ * kTailLineMs[i] * 0.001f) | 1u; }
        line_mask_ = uint32_t(ln - 1);
        for (int i = 0; i < 3; ++i) early_[i] = uint32_t(sr_ * kEarlyMs[i] * 0.001f);
        predelay_ = uint32_t(sr_ * kPreDelayMs * 0.001f);
        slap_a_ = onepole(kSlapHz, sr_);
        tap_[0] = tap_[1] = uint32_t(sr_ * 0.25f);
        slap_delay_.store(sr_ * 0.25f, std::memory_order_relaxed);
        xf_step_ = 1.0f / (0.30f * sr_);
        for (int i = 0; i < ParamCount; ++i) { tgt_[i].store(0.0f, std::memory_order_relaxed); cur_[i] = 0.0f; }
        set_space(0.5f, 3500.0f, 0.3f, 0.0f, 0.0f, 0.0f);
    }

    // (game thread) The surroundings: how long the tail rings (s to -60 dB), its loss of highs,
    // the early reflections' share, and the one echo (after how long, how loud, from which side).
    void set_space(float decay_s, float damp_hz, float early, float slap_s, float slap, float slap_pan) {
        decay_s = std::clamp(decay_s, 0.1f, 4.0f);
        float g2 = 0.0f;
        for (int i = 0; i < 4; ++i) {
            const float g = std::pow(10.0f, -3.0f * float(len_[i]) / (decay_s * sr_));
            tgt_[Fb0 + i].store(g, std::memory_order_relaxed);
            g2 += 0.25f * g * g;
        }
        tgt_[Damp].store(onepole(std::clamp(damp_hz, 300.0f, 16000.0f), sr_), std::memory_order_relaxed);
        tgt_[Early].store(kEarlyLevel * std::clamp(early, 0.0f, 1.0f), std::memory_order_relaxed);
        // (the longer it rings the more it piles up: levelled, so that `decay` is length and not loudness)
        tgt_[Tail].store(kTailLevel * std::sqrt((std::max)(1.0f - g2, 0.0f)), std::memory_order_relaxed);
        const float th = (std::clamp(slap_pan, -1.0f, 1.0f) * 0.8f + 1.0f) * 0.78539816f;
        slap = std::clamp(slap, 0.0f, 1.0f);
        tgt_[SlapL].store(slap * std::cos(th) * 1.4142f, std::memory_order_relaxed);
        tgt_[SlapR].store(slap * std::sin(th) * 1.4142f, std::memory_order_relaxed);
        if (slap > 0.0f) slap_delay_.store(std::clamp(slap_s, 0.03f, kSlapMaxS) * sr_, std::memory_order_relaxed);
    }

    // (audio thread) in may be null (nothing sends): the tail still rings out.
    void process(const float* in, int in_ch, float* out, int out_ch, int frames) {
        // Nothing has come in for longer than any echo and the tail is under hearing: nothing to do.
        float in_peak = 0.0f;
        if (in) for (int i = 0, n = frames * in_ch; i < n; ++i) in_peak = (std::max)(in_peak, std::abs(in[i]));
        if (in_peak < 1.0e-7f) silent_ = (std::min)(silent_ + uint64_t(frames), uint64_t(1) << 40); else silent_ = 0;
        if (silent_ > uint64_t(pre_.size()) + uint64_t(sr_ * 4.0f) && out_peak_ < 1.0e-6f) {
            std::memset(out, 0, sizeof(float) * size_t(frames) * size_t(out_ch));
            return;
        }
        float tgt[ParamCount];
        for (int i = 0; i < ParamCount; ++i) tgt[i] = tgt_[i].load(std::memory_order_relaxed);
        // The echo's delay never slides (that would bend its pitch): a second tap is set to the
        // new delay and the two are cross-faded.
        if (xf_ <= 0.0f || xf_ >= 1.0f) {
            const int active = xf_ >= 1.0f ? 1 : 0;
            const uint32_t want = (std::min)(uint32_t(slap_delay_.load(std::memory_order_relaxed)), pre_mask_ - 2u);
            const uint32_t diff = want > tap_[active] ? want - tap_[active] : tap_[active] - want;
            if (float(diff) > 0.004f * sr_) { tap_[1 - active] = want; xf_dir_ = active ? -xf_step_ : xf_step_; }
            else xf_dir_ = 0.0f;
        }
        static const float kNothing[kChunk] = {};
        if (!in) in_ch = 1;
        const int i1 = in_ch >= 2 ? 1 : 0;
        float L[kChunk], R[kChunk];
        float peak = 0.0f;
        float* pre = pre_.data();
        float* ln[4] = { line_[0].data(), line_[1].data(), line_[2].data(), line_[3].data() };
        const uint32_t pm = pre_mask_, lm = line_mask_;
        for (int done = 0; done < frames;) {
            const int n = (std::min)(frames - done, kChunk);
            const float* src = in ? in + size_t(done) * size_t(in_ch) : kNothing;
            float c[ParamCount];
            for (int i = 0; i < ParamCount; ++i) c[i] = cur_[i];
            float s[4] = { s_[0], s_[1], s_[2], s_[3] };
            float slp = slap_lp_, xf = xf_;
            uint32_t w = w_;
            const float k = follow_;
            for (int f = 0; f < n; ++f) {
                for (int i = 0; i < ParamCount; ++i) c[i] += (tgt[i] - c[i]) * k;
                pre[w & pm] = 0.5f * (src[f * in_ch] + src[f * in_ch + i1]);
                const float e1 = pre[(w - early_[0]) & pm], e2 = pre[(w - early_[1]) & pm], e3 = pre[(w - early_[2]) & pm];
                const float xin = pre[(w - predelay_) & pm];
                xf = std::clamp(xf + xf_dir_, 0.0f, 1.0f);
                const float ta = pre[(w - tap_[0]) & pm], tb = pre[(w - tap_[1]) & pm];
                slp += slap_a_ * (ta + xf * (tb - ta) - slp);
                for (int i = 0; i < 4; ++i) s[i] += c[Damp] * (ln[i][(w - len_[i]) & lm] - s[i]);
                const float h = 0.5f * (s[0] + s[1] + s[2] + s[3]);
                const float inj = 0.5f * (xin + 0.5f * (e1 + e2));
                for (int i = 0; i < 4; ++i) ln[i][w & lm] = (s[i] - h) * c[Fb0 + i] + inj;
                L[f] = c[Early] * (e1 + 0.5f * e3) + c[Tail] * (s[0] + s[2]) + slp * c[SlapL];
                R[f] = c[Early] * (e2 + 0.5f * e3) + c[Tail] * (s[1] + s[3]) + slp * c[SlapR];
                peak = (std::max)(peak, (std::max)(std::abs(L[f]), std::abs(R[f])));
                ++w;
            }
            for (int i = 0; i < ParamCount; ++i) cur_[i] = c[i];
            for (int i = 0; i < 4; ++i) s_[i] = s[i];
            slap_lp_ = slp; xf_ = xf; w_ = w;
            SpatialVoice::write(out + size_t(done) * size_t(out_ch), out_ch, L, R, n);
            done += n;
        }
        out_peak_ = peak;
    }

private:
    std::atomic<float> tgt_[ParamCount];
    std::atomic<float> slap_delay_{ 12000.0f };      // samples
    float cur_[ParamCount]{};
    float sr_ = 48000.0f, follow_ = 0.001f, slap_a_ = 0.3f;
    std::vector<float> pre_, line_[4];
    uint32_t pre_mask_ = 0, line_mask_ = 0, len_[4]{}, early_[3]{}, predelay_ = 0, tap_[2]{};
    float s_[4]{}, slap_lp_ = 0.0f, xf_ = 0.0f, xf_dir_ = 0.0f, xf_step_ = 0.0f, out_peak_ = 0.0f;
    uint64_t silent_ = 0;
    uint32_t w_ = 0;
};

}  // namespace vibe::sound_dsp
