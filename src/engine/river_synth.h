#pragma once
// Running water, made here: no recording. Eight seconds that loop - a bed of broadband rush
// (noise through a gentle band-pass, its loudness wandering slowly as the water's surface
// does) and the babble on top: small bubbles closing, each a short sine that rises as it rings
// down (a bubble's own note, 500 - 2500 Hz), forty or so a second at random. The loop's last
// half second is cross-faded into its start, so played end to end there is no seam.
//
// write_wav() writes it as a 16-bit mono WAV at 44.1 kHz. water_fx.cpp (tick_river_sound) writes
// assets/sounds/river_loop.wav the first time the river is to be heard and the file is not
// there, and plays it as one source of the spatialiser on the course nearest the listener.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace vibe::river_synth {

constexpr int   kRate = 44100;
constexpr float kSeconds = 8.0f;
constexpr float kFade = 0.5f;

inline std::vector<int16_t> make() {
    const int n = int(kSeconds * float(kRate)), nf = int(kFade * float(kRate)), nt = n + nf;
    std::vector<float> s(static_cast<size_t>(nt), 0.0f);
    uint32_t rng = 0x9E3779B9u;
    auto rnd = [&]() { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return float(rng >> 8) / 16777216.0f; };
    // The rush: white noise, high-passed at ~120 Hz and low-passed at ~1.8 kHz (one pole each),
    // its level wandering with two slow sines and a little noise.
    float lp = 0.0f, hp_in = 0.0f, hp = 0.0f, lp2 = 0.0f;
    const float a_lp = 1.0f - std::exp(-6.2831853f * 1800.0f / float(kRate));
    const float a_lp2 = 1.0f - std::exp(-6.2831853f * 900.0f / float(kRate));
    const float a_hp = std::exp(-6.2831853f * 120.0f / float(kRate));
    for (int i = 0; i < nt; ++i) {
        const float t = float(i) / float(kRate);
        const float w = rnd() * 2.0f - 1.0f;
        lp += a_lp * (w - lp);
        lp2 += a_lp2 * (lp - lp2);
        hp = a_hp * (hp + lp2 - hp_in); hp_in = lp2;
        const float lvl = 0.75f + 0.15f * std::sin(t * 1.7f) + 0.10f * std::sin(t * 4.3f + 1.0f);
        s[size_t(i)] = hp * 2.2f * lvl;
    }
    // The babble: bubbles.
    const int nb = int(kSeconds * 42.0f);
    for (int b = 0; b < nb; ++b) {
        const int at = int(rnd() * float(n));
        const float f0 = 500.0f * std::pow(5.0f, rnd());              // 500 .. 2500 Hz
        const float amp = 0.05f + 0.22f * rnd() * rnd();
        const float dur = 0.012f + 0.03f * rnd();                       // rings down in tens of ms
        const int len = int(dur * 3.0f * float(kRate));
        double ph = 0.0;
        for (int i = 0; i < len && at + i < nt; ++i) {
            const float tt = float(i) / float(kRate);
            const float f = f0 * (1.0f + 1.6f * tt / dur * 0.25f);        // (the note rises as the bubble closes)
            ph += 6.283185307179586 * double(f) / double(kRate);
            const float env = std::exp(-tt / dur) * (std::min)(1.0f, tt / 0.0015f);
            s[size_t(at + i)] += amp * env * float(std::sin(ph));
        }
    }
    // The seam: the tail beyond kSeconds faded into the start.
    for (int i = 0; i < nf; ++i) {
        const float k = float(i) / float(nf);
        s[size_t(i)] = s[size_t(i)] * k + s[size_t(n + i)] * (1.0f - k);
    }
    float peak = 1e-6f;
    for (int i = 0; i < n; ++i) peak = (std::max)(peak, std::abs(s[size_t(i)]));
    std::vector<int16_t> out(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) out[size_t(i)] = int16_t(std::lround(std::tanh(s[size_t(i)] / peak * 1.2f) * 26000.0f));
    return out;
}

inline bool write_wav(const std::string& path) {
    const std::vector<int16_t> pcm = make();
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

}  // namespace vibe::river_synth
