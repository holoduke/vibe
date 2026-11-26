#pragma once
// A police siren, made here: no recording. Sixteen seconds that loop - three slow wails (the
// pitch climbs from 650 Hz to 1500 Hz in a second and a half and falls back over two and a
// half) and four seconds of yelp (the same climb and fall three times a second) - as a horn
// sounds: a tone with its overtones, a little rough. It begins and ends at the bottom of a
// wail, faded over 15 ms, so that played end to end there is no click.
//
// write_siren_wav() writes it as a 16-bit mono WAV at 44.1 kHz. jeep.cpp (siren_audio_tick)
// writes assets/sounds/siren_wail.wav the first time a siren is to be heard and the file is not
// there, and plays it as a source of the spatialiser (AudioEngine::source_start, looping;
// how far it carries: sound_space::kSirenShape).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace vibe::siren {

constexpr int   kRate = 44100;
constexpr float kSeconds = 16.0f;

// The pitch (Hz) `t` seconds into the loop.
inline float pitch_at(float t) {
    const float lo = 650.0f, hi = 1500.0f;
    float k;                                            // 0 at the bottom .. 1 at the top
    if (t < 12.0f) {                                    // the wail: 4 s each, 1.5 up and 2.5 down
        const float u = std::fmod(t, 4.0f);
        k = u < 1.5f ? u / 1.5f : 1.0f - (u - 1.5f) / 2.5f;
        k = k * k * (3.0f - 2.0f * k);                  // (eased at both ends, as the rotor of a siren spins up and down)
    } else {                                            // the yelp: a third of a second each
        const float u = std::fmod(t - 12.0f, 1.0f / 3.0f) * 3.0f;
        k = u < 0.45f ? u / 0.45f : 1.0f - (u - 0.45f) / 0.55f;
    }
    return lo * std::pow(hi / lo, k);
}

inline std::vector<int16_t> make() {
    const int n = int(kSeconds * float(kRate));
    std::vector<int16_t> out(static_cast<size_t>(n));
    double phase = 0.0;
    for (int i = 0; i < n; ++i) {
        const float t = float(i) / float(kRate);
        phase += 6.283185307179586 * double(pitch_at(t)) / double(kRate);
        if (phase > 6.283185307179586 * 64.0) phase -= 6.283185307179586 * 64.0;
        const float p = float(phase);
        // (a horn: strong second and third overtones, the higher ones falling away)
        float s = std::sin(p) + 0.62f * std::sin(2.0f * p + 0.4f) + 0.44f * std::sin(3.0f * p + 1.1f) + 0.21f * std::sin(4.0f * p) + 0.12f * std::sin(5.0f * p + 0.7f) + 0.06f * std::sin(7.0f * p);
        s = std::tanh(s * 0.9f) * 0.62f;                // (driven a little: the edge a siren has)
        const float edge = std::min(std::min(t, kSeconds - t) / 0.015f, 1.0f);
        out[static_cast<size_t>(i)] = int16_t(std::lround(s * edge * 30000.0f));
    }
    return out;
}

inline bool write_siren_wav(const std::string& path) {
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

}  // namespace vibe::siren
