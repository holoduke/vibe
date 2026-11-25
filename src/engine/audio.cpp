#include "engine/audio.h"
#include "engine/log.h"
#include "engine/sound_dsp.h"

#include <glm/glm.hpp>

// miniaudio is a single-header library. Define the implementation in
// exactly this TU; every other audio.cpp consumer just sees the small
// AudioEngine wrapper.
//
// Vorbis (.ogg) isn't supported by the core miniaudio decoders — you
// have to bring your own. miniaudio ships extras/stb_vorbis.c for
// exactly this purpose; the documented dance is "header-only declare
// stb_vorbis BEFORE miniaudio's implementation, then include
// stb_vorbis again WITHOUT the header-only flag for its definitions".
// That way miniaudio's internal Vorbis path links cleanly.
#define STB_VORBIS_HEADER_ONLY
#include "extras/stb_vorbis.c"

#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"

#undef STB_VORBIS_HEADER_ONLY
#include "extras/stb_vorbis.c"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <list>
#include <cmath>
#include <fstream>
#include <vector>
#include <random>
#include <string>
#include <string_view>
#include <unordered_map>

namespace vibe {

namespace {

float frand_pm(std::default_random_engine& rng, float amt) {
    if (amt <= 0.0f) return 0.0f;
    std::uniform_real_distribution<float> d(-amt, amt);
    return d(rng);
}

// The device's callback: miniaudio's own (mix the engine's graph into the buffer), timed -
// AudioEngine::mixer_cost_take. Runs on the audio thread: two clock reads and three atomic adds
// a block, nothing else.
std::atomic<uint64_t> g_mix_ns{ 0 }, g_mix_frames{ 0 }, g_mix_blocks{ 0 };
void timed_data_callback(ma_device* dev, void* out, const void*, ma_uint32 frames) {
    const auto t0 = std::chrono::steady_clock::now();
    ma_engine_read_pcm_frames(static_cast<ma_engine*>(dev->pUserData), out, frames, nullptr);
    const auto t1 = std::chrono::steady_clock::now();
    g_mix_ns.fetch_add(uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()), std::memory_order_relaxed);
    g_mix_frames.fetch_add(frames, std::memory_order_relaxed);
    g_mix_blocks.fetch_add(1, std::memory_order_relaxed);
}

// SOUND IN SPACE: the two kinds of node in the mixer's graph (sound_dsp.h does the work).
//   SpatialNode: one a source. In: the source's sounds (unspatialised). Out 0: the two ears,
//     to the endpoint. Out 1: its share for the reverb.
//   ReverbNode: one. In: every source's share. Out: to the endpoint. It is worked on also when
//     nothing comes in (the tail rings out), and tells itself when there is nothing to do.
struct SpatialNode { ma_node_base base{}; sound_dsp::SpatialVoice dsp; int ch = 2; };
struct ReverbNode  { ma_node_base base{}; sound_dsp::SpaceReverb  dsp; int ch = 2; };

void spatial_node_process(ma_node* node, const float** in, ma_uint32*, float** out, ma_uint32* frames_out) {
    auto* s = reinterpret_cast<SpatialNode*>(node);
    const size_t bytes = sizeof(float) * size_t(*frames_out) * size_t(s->ch);
    if (!in || !in[0]) { std::memset(out[0], 0, bytes); std::memset(out[1], 0, bytes); return; }
    s->dsp.process(in[0], s->ch, out[0], s->ch, out[1], s->ch, int(*frames_out));
}
void reverb_node_process(ma_node* node, const float** in, ma_uint32* frames_in, float** out, ma_uint32* frames_out) {
    auto* r = reinterpret_cast<ReverbNode*>(node);
    const bool have = in && in[0] && frames_in && *frames_in >= *frames_out;
    r->dsp.process(have ? in[0] : nullptr, r->ch, out[0], r->ch, int(*frames_out));
}
ma_node_vtable g_spatial_vtable = { spatial_node_process, nullptr, 1, 2, 0 };
ma_node_vtable g_reverb_vtable  = { reverb_node_process, nullptr, 1, 1, MA_NODE_FLAG_CONTINUOUS_PROCESSING | MA_NODE_FLAG_ALLOW_NULL_INPUT };

// How fast the game thread's own following is (s): the loudness up and down, what is in the
// way, the Doppler shift, the surroundings.
constexpr float kRiseS = 0.10f, kFallS = 0.25f, kOccS = 0.30f, kDopplerS = 0.15f, kSpaceS = 0.60f;
constexpr float kOccEveryS = 0.20f;         // a source's line to the ear is looked at this often
constexpr float kRiddenSend = 0.5f;         // the car the listener rides: this share of the surroundings' echo

inline float follow(float dt, float tau) { return 1.0f - std::exp(-dt / tau); }

} // namespace

struct AudioEngine::Impl {
    ma_engine engine{};
    bool paused = false;
    bool ok = false;

    // Clips indexed by ClipID — was unordered_map<string,Clip>, but
    // every play call allocated a std::string for the lookup. Now O(1)
    // array indexed by enum + zero allocations on the hot path.
    std::string clip_paths[static_cast<size_t>(ClipID::Count)];
    // One resident (never played) decoded instance per clip: miniaudio's
    // resource manager ref-counts decoded data by path, so while this holds
    // a reference every play call reuses the PCM instead of decoding the
    // file again on the main thread (15 s MP3 explosions / flybys).
    ma_sound* resident[static_cast<size_t>(ClipID::Count)]{};

    // Active one-shot voices. We can't free a ma_sound until it's
    // finished playing, so we keep them in a list and reap drained ones
    // at each set_listener call.
    std::list<ma_sound> oneshots;

    // Effect voices (explosions): sound -> optional low-pass node ->
    // endpoint. std::list keeps addresses stable for the mixer thread.
    struct Effect {
        ma_sound    sound;
        ma_lpf_node lpf;
        bool        has_lpf = false;
    };
    std::list<Effect> effects;

    // Tracked voices (jet flybys): heap-allocated so the ma_sound never
    // moves while miniaudio's mixer thread references it.
    std::unordered_map<uint32_t, ma_sound*> tracked;
    uint32_t next_handle = 1;

    // (loops removed — see comment in start_loop deletion below.)

    // Engines: see audio.h.
    struct EngineSet { std::vector<float> rpm; std::vector<std::string> on, off; };
    std::vector<EngineSet> engine_sets;
    // A source's place in space (audio.h "SOUND IN SPACE"): its node, and what the game thread
    // follows for it.
    struct Spatial {
        SpatialNode node;
        sound_space::Shape shape;
        float heard = 0.0f, send = 0.0f;      // followed: the loudness and the reverb's share
        float occ = 0.0f, occ_target = 0.0f, occ_wait = 0.0f;
        float doppler = 1.0f;
        bool  fresh = true, occ_known = false;
    };
    Spatial* spatial_new(sound_space::Shape shape);
    void     spatial_free(Spatial* s);
    Heard    spatial_update(Spatial& s, glm::vec3 pos, glm::vec3 vel, float level, float ridden_hz);
    bool     sound_into(Spatial& s, const std::string& path, ma_sound* out);

    struct EngineVoice {
        int set = -1;
        std::vector<ma_sound>  snd;        // per band: on, off
        std::vector<uint8_t>   ok, playing;
        Spatial* sp = nullptr;
    };
    std::unordered_map<uint32_t, EngineVoice*> engines;

    // Sources (audio.h "A SOURCE").
    struct Source {
        ma_sound snd{};
        Spatial* sp = nullptr;
        glm::vec3 pos{0.0f};
        bool loop = true, playing = false, stopping = false, updated = false;
    };
    std::unordered_map<uint32_t, Source*> sources;

    // The listener, the frame's length by the clock, the surroundings (wanted, and as followed).
    glm::vec3 ear{0.0f}, ear_fwd{0.0f, 0.0f, -1.0f}, ear_right{1.0f, 0.0f, 0.0f}, ear_front{0.0f, 0.0f, -1.0f}, ear_vel{0.0f};
    std::chrono::steady_clock::time_point last_tick{};
    float dt = 0.016f;
    float sr = 48000.0f;
    int   ch = 2;
    ReverbNode reverb;
    bool  has_reverb = false;
    sound_space::Space space_want, space_now;
    std::function<float(glm::vec3, glm::vec3)> occlusion;
    uint32_t stagger = 0;

    std::default_random_engine rng{ std::random_device{}() };
};

AudioEngine::AudioEngine() : impl_(std::make_unique<Impl>()) {
    ma_engine_config ecfg = ma_engine_config_init();
    ecfg.dataCallback = timed_data_callback;       // (what miniaudio's own does, with a clock round it)
    ma_result r = ma_engine_init(&ecfg, &impl_->engine);
    if (r != MA_SUCCESS) {
        log::warnf("[audio] ma_engine_init failed (%d) — sound disabled", r);
        impl_->ok = false;
        return;
    }
    impl_->ok = true;
    impl_->sr = float(ma_engine_get_sample_rate(&impl_->engine));
    impl_->ch = int(ma_engine_get_channels(&impl_->engine));
    {   // the one reverb
        impl_->reverb.ch = impl_->ch;
        impl_->reverb.dsp.prepare(impl_->sr);
        ma_node_config c = ma_node_config_init();
        const ma_uint32 chn[1] = { ma_uint32(impl_->ch) };
        c.vtable = &g_reverb_vtable; c.pInputChannels = chn; c.pOutputChannels = chn;
        if (ma_node_init(ma_engine_get_node_graph(&impl_->engine), &c, nullptr, &impl_->reverb.base) == MA_SUCCESS) {
            impl_->has_reverb = true;
            ma_node_attach_output_bus(&impl_->reverb.base, 0, ma_engine_get_endpoint(&impl_->engine), 0);
        }
    }
    impl_->last_tick = std::chrono::steady_clock::now();
    log::infof("[audio] miniaudio engine initialized (%.0f Hz, %d channels)", double(impl_->sr), impl_->ch);
}

AudioEngine::~AudioEngine() {
    if (!impl_->ok) return;
    for (auto& s : impl_->oneshots) ma_sound_uninit(&s);
    for (auto& kv : impl_->tracked) { ma_sound_uninit(kv.second); delete kv.second; }
    for (auto*& s : impl_->resident) if (s) { ma_sound_uninit(s); delete s; s = nullptr; }
    for (auto& e : impl_->effects) {
        ma_sound_uninit(&e.sound);
        if (e.has_lpf) ma_lpf_node_uninit(&e.lpf, nullptr);
    }
    while (!impl_->engines.empty()) engine_stop(impl_->engines.begin()->first);
    for (auto& kv : impl_->sources) { ma_sound_uninit(&kv.second->snd); impl_->spatial_free(kv.second->sp); delete kv.second; }
    impl_->sources.clear();
    if (impl_->has_reverb) ma_node_uninit(&impl_->reverb.base, nullptr);
    ma_engine_uninit(&impl_->engine);
}

// ---------------------------------------------------------------------------------------
// Engine voices (audio.h "ENGINES").
int AudioEngine::engine_set_load(std::string_view list_file) {
    if (!impl_->ok) return -1;
    const std::string path(list_file);
    std::ifstream f(path);
    if (!f) { log::warnf("[audio] engine set %s missing - that car is silent", path.c_str()); return -1; }
    const std::string dir = path.substr(0, path.find_last_of("/\\") + 1);
    Impl::EngineSet es;
    float rpm; std::string on, off;
    while (f >> rpm >> on >> off) { es.rpm.push_back(rpm); es.on.push_back(dir + on); es.off.push_back(dir + off); }
    if (es.rpm.size() < 2) return -1;
    impl_->engine_sets.push_back(std::move(es));
    return int(impl_->engine_sets.size()) - 1;
}

// ---------------------------------------------------------------------------------------
// SOUND IN SPACE (audio.h). The game thread's half: once a frame for each source, the curves of
// sound_space.h from where the source and the listener are, followed over time, written to the
// source's node as targets (atomics; the node follows them a sample at a time - sound_dsp.h).
AudioEngine::Impl::Spatial* AudioEngine::Impl::spatial_new(sound_space::Shape shape) {
    auto* s = new Spatial();
    s->shape = shape;
    s->node.ch = ch;
    s->node.dsp.prepare(sr);
    ma_node_config c = ma_node_config_init();
    const ma_uint32 in_ch[1] = { ma_uint32(ch) }, out_ch[2] = { ma_uint32(ch), ma_uint32(ch) };
    c.vtable = &g_spatial_vtable; c.pInputChannels = in_ch; c.pOutputChannels = out_ch;
    if (ma_node_init(ma_engine_get_node_graph(&engine), &c, nullptr, &s->node.base) != MA_SUCCESS) { delete s; return nullptr; }
    ma_node_attach_output_bus(&s->node.base, 0, ma_engine_get_endpoint(&engine), 0);
    if (has_reverb) ma_node_attach_output_bus(&s->node.base, 1, &reverb.base, 0);
    // (each source looks at its line to the ear at its own moment, not all in one frame)
    s->occ_wait = kOccEveryS * float(stagger++ % 8u) / 8.0f;
    return s;
}

void AudioEngine::Impl::spatial_free(Spatial* s) {
    if (!s) return;
    ma_node_uninit(&s->node.base, nullptr);
    delete s;
}

// A file's sound, unspatialised (the node does that), into a source's node.
bool AudioEngine::Impl::sound_into(Spatial& s, const std::string& path, ma_sound* out) {
    ma_sound_config c = ma_sound_config_init_2(&engine);
    c.pFilePath = path.c_str();
    c.flags = MA_SOUND_FLAG_DECODE | MA_SOUND_FLAG_NO_SPATIALIZATION;
    c.pInitialAttachment = &s.node.base;
    c.volumeSmoothTimeInPCMFrames = ma_uint32(sr * 0.02f);       // (a loop's share of a cross-fade is set once a frame: followed over 20 ms)
    return ma_sound_init_ex(&engine, &c, out) == MA_SUCCESS;
}

AudioEngine::Heard AudioEngine::Impl::spatial_update(Spatial& s, glm::vec3 pos, glm::vec3 vel, float level, float ridden_hz) {
    namespace ss = sound_space;
    Heard h;
    const glm::vec3 to = ear - pos;
    const float d = glm::length(to);
    const glm::vec3 n = d > 1.0e-3f ? to / d : glm::vec3(0.0f, 0.0f, 1.0f);       // from the source to the ear
    level = (std::max)(level, 0.0f);
    float g = 0.0f, hz = 0.0f, send = 0.0f, dop = 1.0f;
    ss::Ears e;
    if (ridden_hz > 0.0f) {
        g = level; hz = ridden_hz;
        send = level * kRiddenSend * space_now.amount;
        s.occ = s.occ_target = 0.0f;
    } else {
        const float dg = ss::gain(d, s.shape.ref_m);
        s.occ_wait -= dt;
        if (s.occ_wait <= 0.0f) {
            s.occ_wait = kOccEveryS;
            s.occ_target = (occlusion && dg > 0.0f && level > 0.0f) ? std::clamp(occlusion(ear, pos), 0.0f, 1.0f) : 0.0f;
            if (!s.occ_known) { s.occ = s.occ_target; s.occ_known = true; }       // (the first look: taken as found)
        }
        s.occ += (s.occ_target - s.occ) * follow(dt, kOccS);
        e = ss::ears(glm::dot(-n, ear_right), glm::dot(-n, ear_front), d);
        g = level * dg * ss::occluded_gain(s.occ) * space_now.outside_gain;
        hz = (std::min)(ss::cutoff_hz(d, s.shape.top_hz) * ss::occluded_cutoff(s.occ) * (1.0f - ss::kRearDull * e.rear), space_now.outside_hz);
        send = level * ss::send(dg) * space_now.amount * space_now.outside_gain;
        dop = ss::doppler(glm::dot(vel, n), glm::dot(ear_vel, n), d);
    }
    hz = std::clamp(hz, 150.0f, 20000.0f);
    if (s.fresh) { s.doppler = dop; s.fresh = false; }
    s.doppler += (dop - s.doppler) * follow(dt, kDopplerS);
    s.heard += (g - s.heard) * follow(dt, g > s.heard ? kRiseS : kFallS);
    s.send  += (send - s.send) * follow(dt, send > s.send ? kRiseS : kFallS);
    if (g <= 0.0f && s.heard < 1.0e-5f) { s.heard = 0.0f; s.send = 0.0f; }

    using P = sound_dsp::SpatialVoice;
    // (x sqrt 2: a source straight ahead is as loud in each ear as it is at its source)
    s.node.dsp.set(P::GainL, s.heard * e.gain_l * 1.41421356f);
    s.node.dsp.set(P::GainR, s.heard * e.gain_r * 1.41421356f);
    // (two one-pole stages are 3 dB down at 0.644 of their corner: `hz` is where the pair is)
    s.node.dsp.set(P::Lowpass, sound_dsp::onepole((std::min)(hz * 1.554f, 0.45f * sr), sr));
    s.node.dsp.set(P::ShadowL, e.shadow_l);
    s.node.dsp.set(P::ShadowR, e.shadow_r);
    s.node.dsp.set(P::DelayL, e.delay_l_s * sr);
    s.node.dsp.set(P::DelayR, e.delay_r_s * sr);
    s.node.dsp.set(P::Send, s.send);

    h.gain = s.heard; h.distance = d; h.cutoff_hz = hz; h.doppler = s.doppler; h.pan = e.pan; h.send = s.send; h.occlusion = s.occ;
    return h;
}

uint32_t AudioEngine::engine_start(int set, sound_space::Shape shape) {
    if (!impl_->ok || set < 0 || size_t(set) >= impl_->engine_sets.size()) return 0;
    const Impl::EngineSet& es = impl_->engine_sets[size_t(set)];
    Impl::Spatial* sp = impl_->spatial_new(shape);
    if (!sp) return 0;
    auto* v = new Impl::EngineVoice();
    v->set = set;
    v->sp = sp;
    const size_t n = es.rpm.size() * 2;
    v->snd.resize(n); v->ok.assign(n, 0); v->playing.assign(n, 0);
    for (size_t i = 0; i < n; ++i) {
        const std::string& p = (i & 1) ? es.off[i / 2] : es.on[i / 2];
        if (!impl_->sound_into(*sp, p, &v->snd[i])) continue;
        v->ok[i] = 1;
        ma_sound_set_looping(&v->snd[i], MA_TRUE);
        ma_sound_set_volume(&v->snd[i], 0.0f);
    }
    const uint32_t h = impl_->next_handle++;
    impl_->engines[h] = v;
    return h;
}

AudioEngine::Heard AudioEngine::engine_update(uint32_t h, glm::vec3 pos, glm::vec3 vel, float rpm, float load, float level, float ridden_hz, bool simple) {
    if (!impl_->ok) return {};
    const auto it = impl_->engines.find(h);
    if (it == impl_->engines.end()) return {};
    Impl::EngineVoice& v = *it->second;
    const Heard heard = impl_->spatial_update(*v.sp, pos, vel, level, ridden_hz);
    const Impl::EngineSet& es = impl_->engine_sets[size_t(v.set)];
    const size_t K = es.rpm.size();
    rpm = std::clamp(rpm, es.rpm.front() * 0.7f, es.rpm.back() * 1.08f);
    load = std::clamp(load, 0.0f, 1.0f);
    size_t b = 0;
    while (b + 2 < K && rpm > es.rpm[b + 1]) ++b;
    const float w = std::clamp((rpm - es.rpm[b]) / (std::max)(es.rpm[b + 1] - es.rpm[b], 1.0f), 0.0f, 1.0f);
    // (equal power: the sum of two unlike loops keeps its loudness through the fade)
    const float g_lo0 = std::cos(w * 1.5707963f), g_hi0 = std::sin(w * 1.5707963f);
    float g_on = std::sqrt(load), g_off = std::sqrt(1.0f - load);
    float g_lo = g_lo0, g_hi = g_hi0;
    if (simple) {       // one loop instead of four: a quarter of the mixing and resampling (nobody hears the blend at that distance)
        if (w < 0.5f) { g_lo = 1.0f; g_hi = 0.0f; } else { g_lo = 0.0f; g_hi = 1.0f; }
        if (load > 0.35f) { g_on = 1.0f; g_off = 0.0f; } else { g_on = 0.0f; g_off = 1.0f; }
    }
    // Under the floor (-72 dB) nothing of it is mixed. (The loops carry only their share of
    // the cross-fade; how loud the engine is heard is the node's - `heard`.)
    if (heard.gain < sound_space::kFloor) g_lo = g_hi = 0.0f;
    for (size_t i = 0; i < v.snd.size(); ++i) {
        if (!v.ok[i]) continue;
        const size_t band = i / 2;
        const float g = (band == b ? g_lo : band == b + 1 ? g_hi : 0.0f) * ((i & 1) ? g_off : g_on);
        if (g > 0.003f) {
            // (the engine's speed, and the Doppler shift: one resampling does both)
            ma_sound_set_pitch(&v.snd[i], std::clamp(rpm / es.rpm[band], 0.35f, 2.6f) * heard.doppler);
            ma_sound_set_volume(&v.snd[i], g);
            if (!v.playing[i]) { ma_sound_start(&v.snd[i]); v.playing[i] = 1; }
        } else if (v.playing[i]) {
            ma_sound_stop(&v.snd[i]);
            v.playing[i] = 0;
        }
    }
    return heard;
}

void AudioEngine::engine_stop(uint32_t h) {
    if (!impl_->ok) return;
    const auto it = impl_->engines.find(h);
    if (it == impl_->engines.end()) return;
    Impl::EngineVoice* v = it->second;
    for (size_t i = 0; i < v->snd.size(); ++i) if (v->ok[i]) ma_sound_uninit(&v->snd[i]);
    impl_->spatial_free(v->sp);
    delete v;
    impl_->engines.erase(it);
}

// Sources (audio.h "A SOURCE").
uint32_t AudioEngine::source_start(ClipID id, sound_space::Shape shape, bool loop) {
    if (!impl_->ok) return 0;
    const auto idx = static_cast<size_t>(id);
    if (idx >= static_cast<size_t>(ClipID::Count) || impl_->clip_paths[idx].empty()) return 0;
    Impl::Spatial* sp = impl_->spatial_new(shape);
    if (!sp) return 0;
    auto* s = new Impl::Source();
    s->sp = sp; s->loop = loop;
    if (!impl_->sound_into(*sp, impl_->clip_paths[idx], &s->snd)) { impl_->spatial_free(sp); delete s; return 0; }
    ma_sound_set_looping(&s->snd, loop ? MA_TRUE : MA_FALSE);
    const uint32_t h = impl_->next_handle++;
    if (impl_->next_handle == 0) impl_->next_handle = 1;
    impl_->sources[h] = s;
    return h;
}

AudioEngine::Heard AudioEngine::source_update(uint32_t h, glm::vec3 pos, glm::vec3 vel, float level, float pitch) {
    if (!impl_->ok) return {};
    const auto it = impl_->sources.find(h);
    if (it == impl_->sources.end() || it->second->stopping) return {};
    Impl::Source& s = *it->second;
    s.pos = pos;
    const Heard heard = impl_->spatial_update(*s.sp, pos, vel, level, 0.0f);
    ma_sound_set_pitch(&s.snd, std::clamp(pitch, 0.1f, 4.0f) * heard.doppler);
    // A loop under the floor is not mixed (it goes on from where it was when it is heard
    // again); a clip that plays once is started once and runs to its end.
    const bool audible = heard.gain >= sound_space::kFloor;
    if (s.loop) {
        if (audible && !s.playing) { ma_sound_start(&s.snd); s.playing = true; }
        else if (!audible && s.playing) { ma_sound_stop(&s.snd); s.playing = false; }
    } else if (!s.playing) { ma_sound_start(&s.snd); s.playing = true; }
    return heard;
}

void AudioEngine::source_stop(uint32_t h) {
    if (!impl_->ok) return;
    const auto it = impl_->sources.find(h);
    if (it != impl_->sources.end()) it->second->stopping = true;       // (faded and freed in set_listener)
}

void AudioEngine::set_space(const sound_space::Space& s) { if (impl_) impl_->space_want = s; }
sound_space::Space AudioEngine::space() const { return impl_ ? impl_->space_now : sound_space::Space{}; }
void AudioEngine::set_occlusion_probe(std::function<float(glm::vec3, glm::vec3)> fn) { if (impl_) impl_->occlusion = std::move(fn); }

void AudioEngine::set_listener_velocity(glm::vec3 v) {
    if (!impl_->ok) return;
    impl_->ear_vel = v;
    ma_engine_listener_set_velocity(&impl_->engine, 0, v.x, v.y, v.z);
}

void AudioEngine::engine_counts(int& voices, int& loops_playing) const {
    voices = 0; loops_playing = 0;
    if (!impl_ || !impl_->ok) return;
    for (const auto& kv : impl_->engines) {
        ++voices;
        for (uint8_t p : kv.second->playing) loops_playing += p ? 1 : 0;
    }
}

AudioEngine::MixerCost AudioEngine::mixer_cost_take() {
    MixerCost c;
    if (!impl_ || !impl_->ok) return c;
    c.mix_ms = double(g_mix_ns.exchange(0, std::memory_order_relaxed)) * 1e-6;
    c.audio_ms = double(g_mix_frames.exchange(0, std::memory_order_relaxed)) * 1000.0 / double((std::max)(ma_engine_get_sample_rate(&impl_->engine), ma_uint32(1)));
    c.blocks = g_mix_blocks.exchange(0, std::memory_order_relaxed);
    return c;
}

bool AudioEngine::ok() const { return impl_ && impl_->ok; }
size_t AudioEngine::active_effects() const { return impl_ ? impl_->effects.size() : 0; }

void AudioEngine::set_listener(glm::vec3 pos, glm::vec3 forward) {
    if (!impl_->ok) return;
    ma_engine_listener_set_position(&impl_->engine, 0, pos.x, pos.y, pos.z);
    ma_engine_listener_set_direction(&impl_->engine, 0,
                                       forward.x, forward.y, forward.z);
    {   // SOUND IN SPACE: the ear, the frame's length, the surroundings followed, sources that fade out.
        Impl& I = *impl_;
        I.ear = pos; I.ear_fwd = forward;
        const glm::vec3 flat(forward.x, 0.0f, forward.z);
        if (glm::dot(flat, flat) > 0.0025f) {          // (looking straight up or down: the ears stay where they were)
            I.ear_front = glm::normalize(flat);
            I.ear_right = glm::vec3(-I.ear_front.z, 0.0f, I.ear_front.x);
        }
        const auto now = std::chrono::steady_clock::now();
        I.dt = std::clamp(std::chrono::duration<float>(now - I.last_tick).count(), 0.001f, 0.1f);
        I.last_tick = now;
        const float k = follow(I.dt, kSpaceS);
        sound_space::Space& c = I.space_now; const sound_space::Space& w = I.space_want;
        c.amount += (w.amount - c.amount) * k;   c.decay_s += (w.decay_s - c.decay_s) * k;
        c.damp_hz += (w.damp_hz - c.damp_hz) * k; c.early += (w.early - c.early) * k;
        c.slap += (w.slap - c.slap) * k;
        if (w.slap > 0.0f) { c.slap_s = w.slap_s; c.slap_pan = w.slap_pan; }       // (the reverb cross-fades to a new echo itself)
        c.outside_gain += (w.outside_gain - c.outside_gain) * k * 2.0f;
        c.outside_hz += (w.outside_hz - c.outside_hz) * k * 2.0f;
        if (I.has_reverb) I.reverb.dsp.set_space(c.decay_s, c.damp_hz, c.early, c.slap_s, c.slap, c.slap_pan);
        for (auto it = I.sources.begin(); it != I.sources.end();) {
            Impl::Source& s = *it->second;
            if (!s.stopping) { ++it; continue; }
            if (I.spatial_update(*s.sp, s.pos, glm::vec3(0.0f), 0.0f, 0.0f).gain > 0.0f) { ++it; continue; }
            ma_sound_uninit(&s.snd);
            I.spatial_free(s.sp);
            delete it->second;
            it = I.sources.erase(it);
        }
    }
    // Reap finished effects. A scheduled (delayed) sound reports NOT
    // playing until its start time, so the at_end test is what keeps
    // delayed explosions alive - do not drop it.
    for (auto it = impl_->effects.begin(); it != impl_->effects.end();) {
        if (ma_sound_is_playing(&it->sound) == MA_FALSE && ma_sound_at_end(&it->sound)) {
            ma_sound_uninit(&it->sound);
            if (it->has_lpf) ma_lpf_node_uninit(&it->lpf, nullptr);
            it = impl_->effects.erase(it);
        } else {
            ++it;
        }
    }
    // Reap finished tracked voices (handle then reads as not playing).
    for (auto it = impl_->tracked.begin(); it != impl_->tracked.end();) {
        if (ma_sound_is_playing(it->second) == MA_FALSE) {
            ma_sound_uninit(it->second);
            delete it->second;
            it = impl_->tracked.erase(it);
        } else {
            ++it;
        }
    }
    // Reap drained one-shots.
    for (auto it = impl_->oneshots.begin(); it != impl_->oneshots.end();) {
        if (ma_sound_is_playing(&*it) == MA_FALSE) {
            ma_sound_uninit(&*it);
            it = impl_->oneshots.erase(it);
        } else {
            ++it;
        }
    }
}

void AudioEngine::load_clip(ClipID id, std::string_view path) {
    if (!impl_->ok) return;
    auto idx = static_cast<size_t>(id);
    if (idx >= static_cast<size_t>(ClipID::Count)) return;
    impl_->clip_paths[idx].assign(path);
    if (impl_->resident[idx]) { ma_sound_uninit(impl_->resident[idx]); delete impl_->resident[idx]; impl_->resident[idx] = nullptr; }
    auto* s = new ma_sound{};
    if (ma_sound_init_from_file(&impl_->engine, impl_->clip_paths[idx].c_str(), MA_SOUND_FLAG_DECODE,
                                nullptr, nullptr, s) == MA_SUCCESS) {
        impl_->resident[idx] = s;
    } else {
        delete s;
        log::warnf("[audio] could not load %s", impl_->clip_paths[idx].c_str());
    }
}

// Reap drained one-shots so the list stays bounded even if
// set_listener() stops being called (paused loop, menu open).
// Without this, every fired shot leaks an ma_sound between
// listener updates.
static void reap_oneshots(std::list<ma_sound>& oneshots) {
    for (auto it = oneshots.begin(); it != oneshots.end();) {
        if (ma_sound_is_playing(&*it) == MA_FALSE) {
            ma_sound_uninit(&*it);
            it = oneshots.erase(it);
        } else {
            ++it;
        }
    }
}

void AudioEngine::play_at(ClipID id, glm::vec3 pos,
                          float volume, float pitch_jitter, float volume_jitter) {
    if (!impl_->ok) return;
    auto idx = static_cast<size_t>(id);
    if (idx >= static_cast<size_t>(ClipID::Count)) return;
    const std::string& path = impl_->clip_paths[idx];
    if (path.empty()) return;
    reap_oneshots(impl_->oneshots);
    impl_->oneshots.emplace_back();
    ma_sound& s = impl_->oneshots.back();
    if (ma_sound_init_from_file(&impl_->engine, path.c_str(),
                                 MA_SOUND_FLAG_DECODE,
                                 nullptr, nullptr, &s) != MA_SUCCESS) {
        impl_->oneshots.pop_back();
        return;
    }
    ma_sound_set_position(&s, pos.x, pos.y, pos.z);
    ma_sound_set_volume(&s, volume * (1.0f + frand_pm(impl_->rng, volume_jitter)));
    ma_sound_set_pitch (&s, 1.0f + frand_pm(impl_->rng, pitch_jitter));
    ma_sound_start(&s);
}

void AudioEngine::play_local(ClipID id,
                              float volume, float pitch_jitter, float volume_jitter) {
    if (!impl_->ok) return;
    auto idx = static_cast<size_t>(id);
    if (idx >= static_cast<size_t>(ClipID::Count)) return;
    const std::string& path = impl_->clip_paths[idx];
    if (path.empty()) return;
    reap_oneshots(impl_->oneshots);
    impl_->oneshots.emplace_back();
    ma_sound& s = impl_->oneshots.back();
    if (ma_sound_init_from_file(&impl_->engine, path.c_str(),
                                 MA_SOUND_FLAG_DECODE | MA_SOUND_FLAG_NO_SPATIALIZATION,
                                 nullptr, nullptr, &s) != MA_SUCCESS) {
        impl_->oneshots.pop_back();
        return;
    }
    ma_sound_set_volume(&s, volume * (1.0f + frand_pm(impl_->rng, volume_jitter)));
    ma_sound_set_pitch (&s, 1.0f + frand_pm(impl_->rng, pitch_jitter));
    ma_sound_start(&s);
}

// start_loop / stop_loop / is_loop_playing removed — no call sites.
// If we ever need looping audio (ambience, footstep loop), add an
// enum-keyed version that mirrors play_at / play_local rather than
// resurrecting the string-keyed map.

uint32_t AudioEngine::play_tracked(ClipID id, glm::vec3 pos, float volume,
                                   float rate, float start_s) {
    if (!impl_->ok) return 0;
    auto idx = static_cast<size_t>(id);
    if (idx >= static_cast<size_t>(ClipID::Count)) return 0;
    const std::string& path = impl_->clip_paths[idx];
    if (path.empty()) return 0;
    ma_sound* s = new ma_sound;
    if (ma_sound_init_from_file(&impl_->engine, path.c_str(), MA_SOUND_FLAG_DECODE,
                                nullptr, nullptr, s) != MA_SUCCESS) {
        delete s;
        return 0;
    }
    ma_sound_set_attenuation_model(s, ma_attenuation_model_none);
    ma_sound_set_doppler_factor(s, 0.0f);
    ma_sound_set_position(s, pos.x, pos.y, pos.z);
    ma_sound_set_volume(s, volume);
    ma_sound_set_pitch(s, rate);
    if (start_s > 0.0f) {
        ma_uint32 sr = 0;
        ma_sound_get_data_format(s, nullptr, nullptr, &sr, nullptr, 0);
        if (sr) ma_sound_seek_to_pcm_frame(s, ma_uint64(double(start_s) * sr));
        ma_sound_set_fade_in_milliseconds(s, 0.0f, 1.0f, 700);   // mid-clip start: no hard onset
    }
    ma_sound_start(s);
    const uint32_t h = impl_->next_handle++;
    if (impl_->next_handle == 0) impl_->next_handle = 1;
    impl_->tracked[h] = s;
    return h;
}

void AudioEngine::set_voice(uint32_t h, glm::vec3 pos, float volume) {
    if (!impl_->ok) return;
    auto it = impl_->tracked.find(h);
    if (it == impl_->tracked.end()) return;
    ma_sound_set_position(it->second, pos.x, pos.y, pos.z);
    ma_sound_set_volume(it->second, volume);
}

void AudioEngine::play_effect(ClipID id, glm::vec3 pos, float volume, float pitch,
                              float delay_s, float lpf_hz, int lpf_order, float start_s) {
    if (!impl_->ok) return;
    auto idx = static_cast<size_t>(id);
    if (idx >= static_cast<size_t>(ClipID::Count)) return;
    const std::string& path = impl_->clip_paths[idx];
    if (path.empty()) return;
    if (impl_->effects.size() > 64) return;        // runaway guard (bomb carpets)
    impl_->effects.emplace_back();
    Impl::Effect& e = impl_->effects.back();
    if (ma_sound_init_from_file(&impl_->engine, path.c_str(), MA_SOUND_FLAG_DECODE,
                                nullptr, nullptr, &e.sound) != MA_SUCCESS) {
        impl_->effects.pop_back();
        return;
    }
    if (lpf_hz > 0.0f && lpf_hz < 18000.0f) {
        ma_lpf_node_config lc = ma_lpf_node_config_init(
            ma_engine_get_channels(&impl_->engine), ma_engine_get_sample_rate(&impl_->engine),
            lpf_hz, ma_uint32(lpf_order < 1 ? 1 : (lpf_order > 8 ? 8 : lpf_order)));
        if (ma_lpf_node_init(ma_engine_get_node_graph(&impl_->engine), &lc, nullptr, &e.lpf) == MA_SUCCESS) {
            e.has_lpf = true;
            ma_node_attach_output_bus(&e.lpf, 0, ma_engine_get_endpoint(&impl_->engine), 0);
            ma_node_attach_output_bus(&e.sound, 0, &e.lpf, 0);
        }
    }
    ma_sound_set_attenuation_model(&e.sound, ma_attenuation_model_none);
    ma_sound_set_doppler_factor(&e.sound, 0.0f);
    ma_sound_set_position(&e.sound, pos.x, pos.y, pos.z);
    ma_sound_set_volume(&e.sound, volume);
    if (start_s > 0.0f) {
        ma_uint32 sr = 0;
        if (ma_sound_get_data_format(&e.sound, nullptr, nullptr, &sr, nullptr, 0) == MA_SUCCESS && sr > 0)
            ma_sound_seek_to_pcm_frame(&e.sound, ma_uint64(double(start_s) * double(sr)));
    }
    ma_sound_set_pitch(&e.sound, pitch);
    if (delay_s > 0.0f)
        ma_sound_set_start_time_in_milliseconds(
            &e.sound, ma_engine_get_time_in_milliseconds(&impl_->engine) + ma_uint64(delay_s * 1000.0f));
    ma_sound_start(&e.sound);
}

void AudioEngine::set_voice_rate(uint32_t h, float rate) {
    if (!impl_->ok) return;
    auto it = impl_->tracked.find(h);
    if (it != impl_->tracked.end()) ma_sound_set_pitch(it->second, rate);
}

float AudioEngine::voice_time(uint32_t h) const {
    if (!impl_->ok) return 0.0f;
    auto it = impl_->tracked.find(h);
    if (it == impl_->tracked.end()) return 0.0f;
    float t = 0.0f;
    ma_sound_get_cursor_in_seconds(it->second, &t);
    return t;
}

void AudioEngine::stop_voice(uint32_t h, float fade_s) {
    if (!impl_->ok) return;
    auto it = impl_->tracked.find(h);
    if (it == impl_->tracked.end()) return;
    ma_uint64 ms = ma_uint64((std::max)(0.0f, fade_s) * 1000.0f);
    // Fade from the current volume, then stop (reaped in set_listener).
    ma_sound_set_stop_time_with_fade_in_milliseconds(
        it->second, ma_engine_get_time_in_milliseconds(&impl_->engine) + ms, ms);
}

bool AudioEngine::voice_playing(uint32_t h) const {
    if (!impl_->ok) return false;
    auto it = impl_->tracked.find(h);
    return it != impl_->tracked.end() && ma_sound_is_playing(it->second) == MA_TRUE;
}

void AudioEngine::set_master_volume(float v) {
    if (!impl_->ok) return;
    ma_engine_set_volume(&impl_->engine, v);
}

} // namespace vibe

namespace vibe {

void AudioEngine::set_paused(bool paused) {
    if (!impl_ || !impl_->ok || impl_->paused == paused) return;
    impl_->paused = paused;
    if (paused) ma_engine_stop(&impl_->engine);
    else        ma_engine_start(&impl_->engine);
}

} // namespace vibe
