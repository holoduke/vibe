// Sound in space (engine/sound_space.h, engine/sound_dsp.h): the curves have no step, and the
// audio thread's half turns a stepped parameter into a curve.
#include "engine/sound_dsp.h"
#include "engine/sound_space.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <cmath>
#include <vector>

namespace ss = vibe::sound_space;
namespace dsp = vibe::sound_dsp;
using Catch::Approx;

TEST_CASE("a source's loudness falls without a step from beside it to silence", "[sound]") {
    const float ref = 6.0f;
    float prev = ss::gain(0.0f, ref);
    REQUIRE(prev == Approx(1.0f));
    for (float d = 0.25f; d <= ss::kHear + 20.0f; d += 0.25f) {
        const float g = ss::gain(d, ref);
        REQUIRE(g <= prev);                        // never louder further off
        REQUIRE(prev - g < 0.02f);                 // a quarter of a metre is never a step (steepest beside the source)
        if (d > 50.0f) REQUIRE(prev - g < 0.001f);
        prev = g;
    }
    REQUIRE(ss::gain(ss::kHear, ref) == 0.0f);
    // where the window to nothing begins the source is far down already: the end is not heard
    REQUIRE(ss::gain(ss::kFadeFrom, ref) < 0.004f);
    // twice as far, about half as loud (the inverse of the distance) plus what the air takes
    REQUIRE(ss::gain(100.0f, ref) / ss::gain(50.0f, ref) == Approx(0.5f * std::pow(10.0f, -ss::kExcessDbPerM * 50.0f / 20.0f)).margin(0.01f));
}

TEST_CASE("the further the duller, and what goes to the reverb falls slower than the sound itself", "[sound]") {
    float prev = ss::cutoff_hz(0.0f, 4500.0f);
    REQUIRE(prev == Approx(4500.0f));
    for (float d = 1.0f; d <= ss::kHear; d += 1.0f) {
        const float hz = ss::cutoff_hz(d, 4500.0f);
        REQUIRE(hz <= prev);
        REQUIRE(hz >= ss::kMinCutoffHz);
        prev = hz;
    }
    const float g_near = ss::gain(10.0f, 6.0f), g_far = ss::gain(200.0f, 6.0f);
    REQUIRE(ss::send(g_far) / g_far > ss::send(g_near) / g_near);
    REQUIRE(ss::send(0.0f) == 0.0f);
    REQUIRE(ss::send(1.0f) == Approx(ss::kSendLevel));
}

TEST_CASE("Doppler: higher coming, lower going, nothing beside the source, clamped", "[sound]") {
    REQUIRE(ss::doppler(30.0f, 0.0f, 80.0f) == Approx(343.0f / 313.0f));
    REQUIRE(ss::doppler(-30.0f, 0.0f, 80.0f) == Approx(343.0f / 373.0f));
    REQUIRE(ss::doppler(0.0f, 30.0f, 80.0f) == Approx(313.0f / 343.0f));       // the listener flees
    REQUIRE(ss::doppler(30.0f, 30.0f, 80.0f) == Approx(1.0f));                 // side by side at one speed
    REQUIRE(ss::doppler(30.0f, 0.0f, 0.5f) == Approx(1.0f));
    REQUIRE(ss::doppler(1000.0f, 0.0f, 80.0f) <= ss::kDopplerMax);
    REQUIRE(ss::doppler(-1000.0f, 0.0f, 80.0f) >= ss::kDopplerMin);
}

TEST_CASE("the two ears: equal power, the far ear later and duller, the middle at the listener's feet", "[sound]") {
    for (float side = -1.0f; side <= 1.0f; side += 0.05f) {
        const ss::Ears e = ss::ears(side, 0.0f, 30.0f);
        REQUIRE(e.gain_l * e.gain_l + e.gain_r * e.gain_r == Approx(1.0f));
    }
    const ss::Ears right = ss::ears(1.0f, 0.0f, 30.0f);
    REQUIRE(right.gain_r > right.gain_l);
    REQUIRE(right.delay_l_s > 0.0f); REQUIRE(right.delay_r_s == 0.0f);
    REQUIRE(right.shadow_l > 0.0f);  REQUIRE(right.shadow_r == 0.0f);
    const ss::Ears here = ss::ears(1.0f, 0.0f, 0.0f);
    REQUIRE(here.gain_l == Approx(here.gain_r));
    REQUIRE(ss::ears(0.0f, -1.0f, 30.0f).rear > 0.9f);
    REQUIRE(ss::ears(0.0f, 1.0f, 30.0f).rear == 0.0f);
}

TEST_CASE("a voice turns a stepped loudness into a curve", "[sound]") {
    dsp::SpatialVoice v;
    v.prepare(48000.0f);
    using P = dsp::SpatialVoice;
    v.set(P::Lowpass, 1.0f);       // (wide open: only the gain is looked at)
    const int n = 4800;
    std::vector<float> in(n, 1.0f), dry(size_t(n) * 2), send(size_t(n) * 2);
    v.process(in.data(), 1, dry.data(), 2, send.data(), 2, n);       // silent so far
    REQUIRE(std::abs(dry[size_t(n) * 2 - 2]) < 1.0e-6f);
    v.set(P::GainL, 1.0f); v.set(P::GainR, 1.0f); v.set(P::Send, 0.5f);       // the game thread's step
    float last = dry[size_t(n) * 2 - 2], worst = 0.0f;
    for (int block = 0; block < 4; ++block) {
        v.process(in.data(), 1, dry.data(), 2, send.data(), 2, n);
        for (int f = 0; f < n; ++f) { worst = (std::max)(worst, std::abs(dry[size_t(f) * 2] - last)); last = dry[size_t(f) * 2]; }
    }
    REQUIRE(worst < 0.002f);                    // no sample jumps (the step is spread over some 20 ms)
    REQUIRE(last == Approx(1.0f).margin(0.001f));
    REQUIRE(send[size_t(n) * 2 - 1] == Approx(0.5f).margin(0.001f));
}

TEST_CASE("the far ear hears it later", "[sound]") {
    dsp::SpatialVoice v;
    v.prepare(48000.0f);
    using P = dsp::SpatialVoice;
    v.set(P::Lowpass, 1.0f); v.set(P::GainL, 1.0f); v.set(P::GainR, 1.0f); v.set(P::DelayL, 24.0f);
    const int n = 9600;
    std::vector<float> in(n, 0.0f), dry(size_t(n) * 2), send(size_t(n) * 2);
    v.process(in.data(), 1, dry.data(), 2, send.data(), 2, n);       // (the parameters arrive)
    in[100] = 1.0f;
    v.process(in.data(), 1, dry.data(), 2, send.data(), 2, n);
    int at_l = -1, at_r = -1;
    for (int f = 0; f < n && (at_l < 0 || at_r < 0); ++f) {
        if (at_r < 0 && std::abs(dry[size_t(f) * 2 + 1]) > 0.5f) at_r = f;
        if (at_l < 0 && std::abs(dry[size_t(f) * 2]) > 0.5f) at_l = f;
    }
    REQUIRE(at_r == 100);
    REQUIRE(at_l == 124);
}

TEST_CASE("the reverb rings out, longer in a closed place, and falls silent", "[sound]") {
    auto energy_after = [](float decay_s) {
        dsp::SpaceReverb r;
        r.prepare(48000.0f);
        r.set_space(decay_s, 4000.0f, 0.3f, 0.0f, 0.0f, 0.0f);
        const int n = 480;
        std::vector<float> in(size_t(n) * 2, 0.0f), out(size_t(n) * 2);
        for (int b = 0; b < 50; ++b) r.process(in.data(), 2, out.data(), 2, n);       // (the parameters arrive)
        in[0] = in[1] = 1.0f;
        r.process(in.data(), 2, out.data(), 2, n);
        in[0] = in[1] = 0.0f;
        double e = 0.0;
        for (int b = 1; b < 100; ++b) {       // the second after the click
            r.process(in.data(), 2, out.data(), 2, n);
            for (float x : out) { REQUIRE(std::isfinite(x)); if (b >= 40) e += double(x) * double(x); }
        }
        return e;
    };
    const double open = energy_after(0.35), closed = energy_after(1.8);
    REQUIRE(closed > 20.0 * open);
    REQUIRE(closed < 1.0);          // (and it does not build up)

    dsp::SpaceReverb r;
    r.prepare(48000.0f);
    const int n = 480;
    std::vector<float> in(size_t(n) * 2, 0.0f), out(size_t(n) * 2, 1.0f);
    in[0] = 1.0f;
    r.process(in.data(), 2, out.data(), 2, n);
    for (int b = 0; b < 100 * 12; ++b) r.process(nullptr, 2, out.data(), 2, n);       // 12 s of nothing coming in
    for (float x : out) REQUIRE(x == 0.0f);
}

TEST_CASE("the one echo comes back after its time, from its side", "[sound]") {
    dsp::SpaceReverb r;
    r.prepare(48000.0f);
    r.set_space(0.3f, 4000.0f, 0.0f, 0.5f, 1.0f, 1.0f);       // a wall on the right, half a second there and back
    const int n = 480;
    std::vector<float> in(size_t(n) * 2, 0.0f), out(size_t(n) * 2);
    for (int b = 0; b < 200; ++b) r.process(in.data(), 2, out.data(), 2, n);          // (the parameters arrive, the tap moves)
    in[0] = in[1] = 1.0f;
    double peak_l = 0.0, peak_r = 0.0; int at = -1;
    for (int b = 0; b < 100; ++b) {
        r.process(in.data(), 2, out.data(), 2, n);
        in[0] = in[1] = 0.0f;
        for (int f = 0; f < n; ++f) {
            if (b * n + f < 20000) continue;         // (the tail of a short room is over by then)
            if (std::abs(out[size_t(f) * 2 + 1]) > peak_r) { peak_r = std::abs(out[size_t(f) * 2 + 1]); at = b * n + f; }
            peak_l = (std::max)(peak_l, double(std::abs(out[size_t(f) * 2])));
        }
    }
    REQUIRE(at > 23900); REQUIRE(at < 24200);          // 0.5 s at 48 kHz, and the low-pass's few samples
    REQUIRE(peak_r > 4.0 * peak_l);
}
