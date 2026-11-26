#pragma once
// The sounds of breaking glass, made here: no recordings.
//
//   crack(variant)  a quarter of a second: the snap of the impact (a burst of noise a few
//                   milliseconds long), the pane ringing on a handful of high partials that die
//                   in tens of milliseconds, and the ticks of the cracks running on after it.
//   shatter()       two seconds: a pane coming down - the break itself, then a cascade of
//                   tinkles (each a short ringing pair of partials between 2.5 and 11 kHz),
//                   dense at first and thinning out, and the duller knocks of the larger pieces
//                   landing.
//
// write_wav() writes one as a 16-bit mono WAV at 44.1 kHz. glass.cpp writes
// assets/sounds/glass_crack_1.wav, glass_crack_2.wav and glass_shatter.wav the first time glass
// is hit and they are not there, and plays them through the audio engine (AudioEngine::play_at).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace vibe::glass_synth {

constexpr int kRate = 44100;

struct Rng {
    uint32_t s;
    float next() { s ^= s << 13; s ^= s >> 17; s ^= s << 5; return float(s & 0xFFFFFF) / float(0x1000000); }      // 0 .. 1
    float pm() { return next() * 2.0f - 1.0f; }
    float range(float a, float b) { return a + (b - a) * next(); }
};

// A struck partial: a sine that starts at `t0` and dies with the time constant `tau`.
inline void ring(std::vector<float>& out, float t0, float hz, float tau, float amp, float phase = 0.0f) {
    const int i0 = std::max(0, int(t0 * float(kRate)));
    const int n = int(tau * 7.0f * float(kRate));
    const float w = 6.2831853f * hz / float(kRate);
    for (int i = 0; i < n && size_t(i0 + i) < out.size(); ++i) {
        const float t = float(i) / float(kRate);
        const float attack = std::min(1.0f, float(i) / 12.0f);            // (a quarter of a millisecond: no click)
        out[size_t(i0 + i)] += amp * attack * std::exp(-t / tau) * std::sin(w * float(i) + phase);
    }
}

// A burst of noise from `t0`, dying with `tau`; `bright` 0 .. 1 takes the low end out of it.
inline void burst(std::vector<float>& out, Rng& r, float t0, float tau, float amp, float bright) {
    const int i0 = std::max(0, int(t0 * float(kRate)));
    const int n = int(tau * 7.0f * float(kRate));
    float prev = 0.0f, low = 0.0f;
    for (int i = 0; i < n && size_t(i0 + i) < out.size(); ++i) {
        const float t = float(i) / float(kRate);
        const float x = r.pm();
        low += 0.18f * (x - low);                                        // (what a high-pass leaves out)
        const float hp = x - low * bright;
        const float y = 0.6f * hp + 0.4f * (hp - prev);
        prev = hp;
        out[size_t(i0 + i)] += amp * std::exp(-t / tau) * y;
    }
}

inline std::vector<int16_t> to_pcm(std::vector<float>& f, float peak_out) {
    float peak = 1e-6f;
    for (float x : f) peak = std::max(peak, std::abs(x));
    std::vector<int16_t> out(f.size());
    const size_t fade = size_t(kRate / 100);
    for (size_t i = 0; i < f.size(); ++i) {
        const float edge = std::min(1.0f, float(f.size() - 1 - i) / float(fade));
        out[i] = int16_t(std::lround(std::tanh(f[i] / peak * 1.4f) / std::tanh(1.4f) * edge * peak_out * 32000.0f));
    }
    return out;
}

inline std::vector<int16_t> crack(int variant) {
    std::vector<float> f(size_t(kRate * 0.30f), 0.0f);
    Rng r{ 0x51ED270Bu + uint32_t(variant) * 0x9E3779B9u };
    burst(f, r, 0.0f, 0.0035f, 1.0f, 0.9f);                              // the snap
    burst(f, r, 0.0f, 0.016f, 0.25f, 0.6f);
    const int partials = 5 + variant;
    for (int k = 0; k < partials; ++k)                                   // the pane rings
        ring(f, 0.0005f, r.range(2600.0f, 10500.0f), r.range(0.012f, 0.045f), r.range(0.12f, 0.34f), r.range(0.0f, 6.28f));
    ring(f, 0.0f, r.range(900.0f, 1500.0f), 0.02f, 0.22f);               // (a little body)
    const int ticks = 7 + variant * 3;                                   // the cracks running on
    for (int k = 0; k < ticks; ++k) {
        const float t = 0.004f + 0.14f * r.next() * r.next();
        const float a = 0.5f * (1.0f - t / 0.16f) * r.range(0.35f, 1.0f);
        burst(f, r, t, 0.0012f, a, 1.0f);
        ring(f, t, r.range(4200.0f, 12000.0f), r.range(0.004f, 0.012f), a * 0.35f);
    }
    return to_pcm(f, 0.92f);
}

inline std::vector<int16_t> shatter() {
    std::vector<float> f(size_t(kRate * 2.1f), 0.0f);
    Rng r{ 0xC0FFEE11u };
    burst(f, r, 0.0f, 0.006f, 1.0f, 0.8f);                               // the break
    burst(f, r, 0.0f, 0.045f, 0.55f, 0.7f);
    burst(f, r, 0.02f, 0.16f, 0.22f, 0.95f);                             // the hiss of a thousand small pieces
    for (int k = 0; k < 9; ++k)
        ring(f, 0.0f, r.range(1800.0f, 9000.0f), r.range(0.02f, 0.07f), r.range(0.10f, 0.25f), r.range(0.0f, 6.28f));
    // the cascade: tinkles, most of them early
    for (int k = 0; k < 260; ++k) {
        const float u = r.next();
        const float t = 0.015f + 1.55f * u * u;                           // (dense at first, thinning)
        const float a = (0.34f - 0.16f * t) * r.range(0.25f, 1.0f) * (r.next() < 0.12f ? 1.8f : 1.0f);
        const float hz = r.range(2500.0f, 11000.0f);
        const float tau = r.range(0.010f, 0.050f);
        ring(f, t, hz, tau, a, r.range(0.0f, 6.28f));
        ring(f, t, hz * r.range(1.35f, 1.9f), tau * 0.7f, a * 0.5f);
        if (r.next() < 0.5f) burst(f, r, t, 0.0010f, a * 0.8f, 1.0f);    // (its click as it lands)
    }
    // the larger pieces landing: duller, later
    for (int k = 0; k < 14; ++k) {
        const float t = r.range(0.28f, 1.15f);
        const float a = r.range(0.10f, 0.26f);
        burst(f, r, t, 0.004f, a, 0.5f);
        ring(f, t, r.range(700.0f, 1900.0f), r.range(0.015f, 0.035f), a * 0.8f);
        ring(f, t, r.range(3000.0f, 6500.0f), r.range(0.02f, 0.06f), a * 0.6f);
    }
    return to_pcm(f, 0.95f);
}

inline bool write_wav(const std::string& path, const std::vector<int16_t>& pcm) {
    std::FILE* f = nullptr;
#ifdef _MSC_VER
    if (fopen_s(&f, path.c_str(), "wb") != 0) f = nullptr;
#else
    f = std::fopen(path.c_str(), "wb");
#endif
    if (!f) return false;
    const uint32_t data = uint32_t(pcm.size() * 2), riff = 36 + data, fmt = 16, rate = uint32_t(kRate), bytes = rate * 2;
    const uint16_t pcm_tag = 1, chans = 1, align = 2, bits = 16;
    std::fwrite("RIFF", 1, 4, f); std::fwrite(&riff, 4, 1, f); std::fwrite("WAVEfmt ", 1, 8, f); std::fwrite(&fmt, 4, 1, f);
    std::fwrite(&pcm_tag, 2, 1, f); std::fwrite(&chans, 2, 1, f); std::fwrite(&rate, 4, 1, f); std::fwrite(&bytes, 4, 1, f);
    std::fwrite(&align, 2, 1, f); std::fwrite(&bits, 2, 1, f); std::fwrite("data", 1, 4, f); std::fwrite(&data, 4, 1, f);
    const bool ok = std::fwrite(pcm.data(), 2, pcm.size(), f) == pcm.size();
    std::fclose(f);
    return ok;
}

}  // namespace vibe::glass_synth
