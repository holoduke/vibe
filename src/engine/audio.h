#pragma once

#include "engine/sound_space.h"

#include <glm/vec3.hpp>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <string_view>

namespace vibe {

// Compile-time clip handles. Maps to a small array inside the engine —
// no string allocation per play call. Add new entries as the game
// grows; the count is the array size.
enum class ClipID : uint8_t {
    Shot = 0,
    Impact,
    Jump,
    Laser,      // player laser gun (assets/sounds/laser_shot.wav)
    AaBurst,    // anti-aircraft burst, the unmodified recording (assets/sounds/aa_burst.wav)
    Rain,       // rain bed, 8 s with a Hann envelope (assets/sounds/rain_bed.wav, synthesised)
    Splash,     // water splash (assets/sounds/water_splash.wav, synthesised)
    SamLaunch,  // missile launch: bang + rocket roar (assets/sounds/sam_launch.wav, synthesised)
    JetFlyby1,  // jet flybys (15 s recordings, one pass each)
    JetFlyby2,
    JetFlyby3,
    ExplClose1, ExplClose2, ExplClose3, ExplClose4, ExplClose5,   // war explosions, close-up
    ExplFar1, ExplFar2, ExplFar3, ExplFar4, ExplFar5,              // distant rumbling explosions
    Thunder1, Thunder2, Thunder3,   // thunder: close clap, distant roll, overhead crackle (assets/sounds/thunder_*.mp3, generated)
    RocketRoar, // the launch: 8 s of engine roar with a Hann envelope, overlapped like the rain (assets/sounds/rocket_roar.wav, synthesised)
    Beep,       // the count's tick (assets/sounds/beep.wav, synthesised)
    Land,       // landing after a fall (assets/sounds/land.ogg)
    Siren,      // a police siren: wail and yelp, 16 s that loop (assets/sounds/siren_wail.wav, synthesised: engine/siren_synth.h)
    GlassCrack1, GlassCrack2,   // a bolt through a pane (assets/sounds/glass_crack_*.wav, synthesised: engine/glass_synth.h)
    GlassShatter,               // a pane coming down (assets/sounds/glass_shatter.wav, synthesised: engine/glass_synth.h)
    River,                      // running water, 8 s that loop (assets/sounds/river_loop.wav, synthesised: engine/river_synth.h)
    MgShot1, MgShot2, MgShot3,  // the rifle as a machine gun: one round (assets/sounds/mg_shot_v1_*.wav, synthesised: engine/gun_synth.h)
    MgTail,                     // ... and its tail from the surroundings (mg_tail_v1.wav), played through the spatialiser
    TyreSqueal,                 // tyres sliding on asphalt, 6 s that loop (assets/sounds/tyre_squeal.wav, synthesised: engine/tyre_synth.h)
    Count
};

// Thin wrapper around miniaudio's high-level engine. Hides the C-style
// pointer dance and exposes only what the game needs: load a clip once
// from disk, then trigger one-shots at world positions, or start/stop a
// 2D loop (footsteps, ambient).
//
// Lifecycle: construct AudioEngine once at engine init (after window
// creation, before draw); call set_listener every frame from the camera;
// destruct on shutdown (the dtor stops everything cleanly).
class AudioEngine {
public:
    AudioEngine();
    ~AudioEngine();

    AudioEngine(const AudioEngine&) = delete;
    AudioEngine& operator=(const AudioEngine&) = delete;

    // True if miniaudio came up successfully. False = no audio device or
    // init failure; trigger calls become silent no-ops.
    bool ok() const;

    // Update the 3D listener (camera position + forward). Call once per
    // render frame.
    void set_listener(glm::vec3 pos, glm::vec3 forward);

    // Pre-load a clip and bind it to an enum slot. Subsequent trigger
    // calls reference the ClipID directly (no string lookup). File
    // path is relative to CWD; falls back silently if missing.
    void load_clip(ClipID id, std::string_view path);

    // One-shot at a world position with optional pitch/volume jitter.
    // pitch_jitter and volume_jitter are ± randomised offsets (e.g.
    // 0.05 = ±5%). Returns immediately, plays for the clip's duration.
    void play_at(ClipID id, glm::vec3 pos,
                 float volume = 1.0f,
                 float pitch_jitter = 0.05f,
                 float volume_jitter = 0.05f);

    // 2D one-shot (no spatialisation). Used for player-local sounds
    // like the jump grunt where head position == listener.
    void play_local(ClipID id,
                     float volume = 1.0f,
                     float pitch_jitter = 0.05f,
                     float volume_jitter = 0.05f);

    // Tracked spatial voice: panned by position, NO distance attenuation
    // (the caller drives volume - e.g. a flyby recording already carries
    // its own approach / recede loudness) and no Doppler. `rate` scales
    // playback speed (and pitch); `start_s` seeks into the clip.
    // Returns a handle (0 = failed).
    uint32_t play_tracked(ClipID id, glm::vec3 pos, float volume,
                          float rate, float start_s);
    // Fire-and-forget spatial effect: panned, no distance attenuation
    // (caller sets volume), starts after `delay_s` (speed of sound) and
    // runs through its own low-pass at `lpf_hz` (air absorption; <= 0 or
    // >= 18 kHz = no filter).
    void     play_effect(ClipID id, glm::vec3 pos, float volume, float pitch,
                         float delay_s, float lpf_hz, int lpf_order = 2, float start_s = 0.0f);
    size_t   active_effects() const;          // scheduled + playing play_effect voices
    void     set_voice(uint32_t h, glm::vec3 pos, float volume);
    void     set_voice_rate(uint32_t h, float rate);
    float    voice_time(uint32_t h) const;   // playback position in the clip (s)
    void     stop_voice(uint32_t h, float fade_s);
    bool     voice_playing(uint32_t h) const;

    // ENGINES (a car's sound). A set is a list of loops recorded at steady engine speeds, each
    // twice: under load and coasting (assets/sounds/engines/<name>.txt: `rpm on.wav off.wav`
    // a line - rendered from an engine simulation by C:/Users/.../engine-sim/vibe, see
    // docs/ARCHITECTURE.md "Car sound"). A voice plays the two loops either side of the
    // engine's speed, each pitched to it, cross-faded by speed and by load, through its own
    // spatialiser (below): the caller gives where it is, how fast it moves and how loud it is
    // at its source; distance, tone, Doppler, direction, what is in the way and the echo of
    // the surroundings are the spatialiser's.
    //   level: the engine's loudness beside it (its load and speed, the kind of car).
    //   ridden_hz > 0: the listener rides in or with this car - no distance law, no Doppler,
    //     in the middle, through this low-pass (the cabin's, the chase view's).
    //   simple: one loop only (a car far off).
    // Returns what is heard of it now (Heard::gain is what to judge a voice's worth by: a
    // voice may be let go once it is 0).
    int      engine_set_load(std::string_view list_file);        // -1: not there
    uint32_t engine_start(int set, sound_space::Shape shape = {});   // 0: failed
    struct Heard;
    Heard    engine_update(uint32_t h, glm::vec3 pos, glm::vec3 vel, float rpm, float load, float level, float ridden_hz = 0.0f, bool simple = false);
    void     engine_stop(uint32_t h);

    // SOUND IN SPACE: the spatialiser every moving source goes through (the curves:
    // sound_space.h; the filters, the ears and the reverb: sound_dsp.h; docs/ARCHITECTURE.md
    // "Sound in space"). Per source, from where it is and where the listener is: loudness by
    // the distance down to silence, the air's low-pass, Doppler, equal-power panning with the
    // far ear later and duller, a hill or a building in the way, and its share of the one
    // reverb. Everything is followed over time - a source comes out of silence and goes back
    // into it; nothing steps.
    struct Heard {
        float gain = 0.0f;          // the loudness it has now (source level x distance x what is in the way), followed; 0 = silent
        float distance = 0.0f;      // m
        float cutoff_hz = 0.0f;     // its low-pass
        float doppler = 1.0f;       // the pitch ratio
        float pan = 0.0f;           // -1 left .. +1 right
        float send = 0.0f;          // what goes to the reverb
        float occlusion = 0.0f;     // 0 clear .. 1 behind a hill or a building
    };
    // A SOURCE: any clip, looping or not, at a place in the world (a siren, a generator, a
    // radio). The whole interface:
    //     h = source_start(ClipID::X, { ref_m, top_hz });       // once (0: failed)
    //     source_update(h, pos, vel, level, pitch);             // every frame it sounds
    //     source_stop(h);                                       // fades to nothing, then frees itself
    // level 1 = the clip as recorded, heard from within Shape::ref_m; a level of 0 fades it
    // out without letting it go. Not updated for a frame: it keeps its last place.
    // Shape also says how far it carries (hear_m: an engine 450 m, sound_space::kSirenShape
    // 1 km). start_s: where in the clip it begins.
    uint32_t source_start(ClipID id, sound_space::Shape shape = {}, bool loop = true, float start_s = 0.0f);
    Heard    source_update(uint32_t h, glm::vec3 pos, glm::vec3 vel, float level = 1.0f, float pitch = 1.0f);
    void     source_stop(uint32_t h);
    // The surroundings (how much echo, the listener under a roof or in a cabin): from the game
    // a few times a second, followed smoothly.
    void     set_space(const sound_space::Space& s);
    sound_space::Space space() const;       // as it is now (followed)
    // What is between the listener's ear and a source, 0 clear .. 1 blocked. Asked on the
    // game thread, about five times a second for each source.
    void     set_occlusion_probe(std::function<float(glm::vec3 ear, glm::vec3 source)> fn);
    void     set_listener_velocity(glm::vec3 v);
    void     engine_counts(int& voices, int& loops_playing) const;       // engine voices held, and how many of their loops are being mixed

    // What the mixer costs: the time the audio thread spent filling the device's buffer since the
    // last call, and how much sound that was (ms / ms = the share of one core). Two clock reads a
    // block (about every 10 ms); everything the engine plays is in it, not only the cars.
    struct MixerCost { double mix_ms = 0.0, audio_ms = 0.0; uint64_t blocks = 0; };
    MixerCost mixer_cost_take();

    // Master volume (0..1). Persisted via settings.cpp.
    void set_master_volume(float v);

    // Freeze / resume every voice (pause menu). Scheduled starts keep their
    // remaining delay because the engine clock stops with the device.
    void set_paused(bool paused);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace vibe
