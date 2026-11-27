#pragma once
// Tyres sliding on asphalt, made here: no recording. A squeal is the rubber sticking and
// slipping many hundred times a second - a rough tone, not a clean one: its pitch wanders a
// little, it chatters (the loudness flickers a few to a dozen times a second) and a hiss of the
// scrub lies under it. So: noise through two narrow resonances (about 1050 Hz and its octave,
// each drifting slowly by a few per cent), a broad band of hiss from 3 to 6 kHz, the whole
// shaken by a slow random chatter. Six seconds that loop: the last half second is faded into
// the first, so played end to end there is no seam.
//
// write_squeal_wav() writes it as a 16-bit mono WAV at 44.1 kHz. jeep.cpp (car_tyre_audio)
// writes assets/sounds/tyre_squeal.wav the first time a squeal is to be heard and the file is
// not there, and plays it as a source of the spatialiser (AudioEngine::source_start, looping).
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace vibe::tyre {

constexpr int   kRate = 44100;
constexpr float kSeconds = 6.0f;
constexpr float kFade = 0.5f;

// A two-pole band-pass (the RBJ cookbook's, constant peak gain), its centre set every sample.
struct Band {
    double x1 = 0, x2 = 0, y1 = 0, y2 = 0;
    float run(float x, float hz, float q) {
        const double w = 6.283185307179586 * double(hz) / double(kRate), al = std::sin(w) / (2.0 * double(q)), cw = std::cos(w);
        const double a0 = 1.0 + al;
        const double y = (al * double(x) - al * x2 - (-2.0 * cw) * y1 - (1.0 - al) * y2) / a0;
        x2 = x1; x1 = double(x); y2 = y1; y1 = y;
        return float(y);
    }
};

inline std::vector<int16_t> make() {
    const int n_all = int((kSeconds + kFade) * float(kRate));
    std::vector<float> s(static_cast<size_t>(n_all));
    uint32_t r = 0x2545F491u;
    auto noise = [&]() { r ^= r << 13; r ^= r >> 17; r ^= r << 5; return float(r) / 2147483648.0f - 1.0f; };
    Band b1, b2, b3, b4;
    float chat = 0.6f, chat_to = 0.6f, drift = 0.0f, drift_to = 0.0f;
    int next_chat = 0, next_drift = 0;
    for (int i = 0; i < n_all; ++i) {
        // (the chatter: a new loudness every 40 - 160 ms, eased toward; the drift every 0.3 - 0.8 s)
        if (i >= next_chat) { chat_to = 0.35f + 0.65f * (0.5f + 0.5f * noise()); next_chat = i + int((0.04f + 0.12f * (0.5f + 0.5f * noise())) * float(kRate)); }
        if (i >= next_drift) { drift_to = 0.035f * noise(); next_drift = i + int((0.3f + 0.5f * (0.5f + 0.5f * noise())) * float(kRate)); }
        chat += (chat_to - chat) * 0.0016f;
        drift += (drift_to - drift) * 0.00012f;
        const float t = float(i) / float(kRate);
        const float f0 = 1050.0f * (1.0f + drift + 0.012f * std::sin(t * 6.283f * 5.3f));
        const float w = noise();
        float v = 1.00f * b1.run(w, f0, 38.0f) + 0.55f * b2.run(w, f0 * 2.01f, 30.0f) + 0.22f * b3.run(w, f0 * 3.02f, 22.0f);
        v = std::tanh(v * 3.2f) * chat;                         // (rough: the tone driven a little)
        v += 0.10f * b4.run(w, 4300.0f, 0.9f);                  // (the scrub's hiss)
        s[static_cast<size_t>(i)] = v;
    }
    // (the loop: the last kFade seconds laid over the first, equal power)
    const int n = int(kSeconds * float(kRate)), nf = n_all - n;
    for (int i = 0; i < nf; ++i) {
        const float a = float(i) / float(nf);
        s[static_cast<size_t>(i)] = s[static_cast<size_t>(i)] * std::sin(a * 1.5707963f) + s[static_cast<size_t>(n + i)] * std::cos(a * 1.5707963f);
    }
    float peak = 1e-6f;
    for (int i = 0; i < n; ++i) peak = std::fmax(peak, std::fabs(s[static_cast<size_t>(i)]));
    std::vector<int16_t> out(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) out[static_cast<size_t>(i)] = int16_t(std::lround(s[static_cast<size_t>(i)] / peak * 26000.0f));
    return out;
}

inline bool write_squeal_wav(const std::string& path) {
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

}  // namespace vibe::tyre
