// What is replicated about a player, how it is packed for the wire, how received states are
// buffered and sampled, and how action events are de-duplicated and released in step with the
// (delayed) picture. No sockets and no engine types in here: all of it is tested directly.
//
// Animation is reconstructed from these fields on every machine; no bones, clip names or
// pose-search results are ever sent.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <vector>

namespace vibe::net {

constexpr uint8_t  kProtocolVersion = 9;       // 2: packed state with weapon / input / shot counter, stamped events. 3: cars. 4: the weapon byte also says which car a player rides in. 5: the NPC world (kMsgNpc). 6: its wire format 4 (styles, runners). 7: shots at NPCs and their cars taken, as events. 8: NpcHit may name somebody who sits in a car, or a car (its driver); a car with people in it can be taken. 9: the NPC world's wire format 5 (a car's speed in thirds of a m/s, its kind and siren in the style bits), NpcCarPush
constexpr uint32_t kInterpDelayMs = 100;       // remote players are shown at least this far in the past
constexpr uint32_t kMaxInterpDelayMs = 500;    // ... and at most this far, however slow the link
constexpr uint32_t kMaxExtrapolateMs = 250;    // nothing newer: carry on along the last velocity this long at most
constexpr float    kTeleportDistance = 6.0f;   // metres between two states that is a jump, not a movement

struct PlayerState {
    float px = 0, py = 0, pz = 0;              // AABB centre (Player::position)
    float vx = 0, vy = 0, vz = 0;
    float yaw = 0, pitch = 0;                  // the view: body yaw and aim are the same in this game
    float crouch = 0;                          // 0 standing .. 1 crawling
    uint8_t flags = 0;                         // kFlag*
    uint8_t health = 100;                      // 0..100
    uint8_t weapon = 0;                        // 0 laser gun, 1 airstrike tablet, 2 rifle
    uint8_t move = 0;                          // kMove*: the movement keys held (what the player wants, for prediction)
    uint16_t shot_seq = 0;                     // shots fired so far (wraps): recoil survives a lost event
};
enum : uint8_t { kFlagGround = 1, kFlagSwim = 2, kFlagTorch = 4, kFlagReload = 8, kFlagTeleport = 16, kFlagAim = 32 };
enum : uint8_t { kMoveFwd = 1, kMoveBack = 2, kMoveLeft = 4, kMoveRight = 8, kMoveSprint = 16, kMoveJump = 32, kMoveCrawl = 64 };

// The wire form. Positions and velocities stay 32-bit floats: the world is a planet (positions
// to 100 km per chart) and free flight reaches kilometres per second. Angles and the crouch
// are quantised: yaw to 2 pi / 65536 (0.0055 degrees), pitch likewise over +-pi/2.
#pragma pack(push, 1)
struct PackedState {
    float    p[3];
    float    v[3];
    uint16_t yaw;
    int16_t  pitch;
    uint8_t  crouch;
    uint8_t  flags, health, weapon, move;
    uint16_t shot_seq;
};
struct Event {
    enum Kind : uint8_t { Fire = 1, Hit = 2, Death = 3, Leave = 4, Airstrike = 5,      // Airstrike: a = target, b = run direction
                          NpcHit = 6,        // somebody of the NPC world was shot: to = which (a person - on foot or in a car - or a car: its driver), a = where, b = the shot's direction
                          NpcCarTake = 7,    // a client asks the host for an NPC car that stands: to = which
                          NpcCarTaken = 8,   // that car is a players' car now: from = who gets in, to/a/b = pack_taken()
                          NpcCarPush = 9 };  // a client's car struck an NPC car: to = which, a = the change of its velocity (m/s, world), b[0] = of its turning (rad/s); the host's to apply
    uint8_t  kind;
    uint32_t from;             // Fire: shooter. Hit: shooter. Death: the one who died. Leave: who left
    uint32_t to;               // Hit: victim. Death: killer
    float    a[3];             // Fire: origin. Hit: hit position
    float    b[3];             // Fire: direction
    float    v;                // Hit: damage
    uint32_t t;                // session clock (ms) when it happened; stamped by the sender
    uint16_t seq;              // per sender, counts up: duplicates and stale repeats are dropped
};
#pragma pack(pop)
static_assert(sizeof(PackedState) == 35, "wire size of a player state");

// An NPC car that became a players' car (Event::NpcCarTaken): which agent it was, the number it
// has from now on in every machine's list of cars, its kind and paint, and where it stands.
struct NpcCarTaken {
    uint8_t  agent = 0, car = 0, kind = 0, look = 0;
    float    p[3]{ 0, 0, 0 };
    float    yaw = 0;
    uint32_t taker = 0;                        // the player who gets in (0: nobody - told again for somebody who joined)
};
inline Event pack_taken(const NpcCarTaken& t) {
    Event e{};
    e.kind = Event::NpcCarTaken; e.from = t.taker;
    e.to = uint32_t(t.agent) | (uint32_t(t.car) << 8) | (uint32_t(t.kind) << 16) | (uint32_t(t.look) << 24);
    e.a[0] = t.p[0]; e.a[1] = t.p[1]; e.a[2] = t.p[2]; e.b[0] = t.yaw;
    return e;
}
inline NpcCarTaken unpack_taken(const Event& e) {
    NpcCarTaken t;
    t.agent = uint8_t(e.to); t.car = uint8_t(e.to >> 8); t.kind = uint8_t(e.to >> 16); t.look = uint8_t(e.to >> 24);
    t.p[0] = e.a[0]; t.p[1] = e.a[1]; t.p[2] = e.a[2]; t.yaw = e.b[0]; t.taker = e.from;
    return t;
}
inline bool taken_is_sane(const NpcCarTaken& t, int max_cars) {
    for (float f : { t.p[0], t.p[1], t.p[2], t.yaw }) if (!std::isfinite(f) || std::abs(f) > 2.0e6f) return false;
    return int(t.car) < max_cars && t.kind < 3 && t.agent < 100;
}

inline PackedState pack_state(const PlayerState& s) {
    constexpr float kTwoPi = 6.28318530718f, kHalfPi = 1.57079632679f;
    PackedState o{};
    o.p[0] = s.px; o.p[1] = s.py; o.p[2] = s.pz;
    o.v[0] = s.vx; o.v[1] = s.vy; o.v[2] = s.vz;
    float y = std::fmod(s.yaw, kTwoPi);
    if (y < 0.0f) y += kTwoPi;
    o.yaw = uint16_t(uint32_t(std::lround(y / kTwoPi * 65536.0f)) & 0xFFFFu);
    o.pitch = int16_t(std::lround(std::clamp(s.pitch / kHalfPi, -1.0f, 1.0f) * 32767.0f));
    o.crouch = uint8_t(std::lround(std::clamp(s.crouch, 0.0f, 1.0f) * 255.0f));
    o.flags = s.flags; o.health = s.health; o.weapon = s.weapon; o.move = s.move; o.shot_seq = s.shot_seq;
    return o;
}

inline PlayerState unpack_state(const PackedState& p) {
    constexpr float kTwoPi = 6.28318530718f, kHalfPi = 1.57079632679f;
    PlayerState s;
    s.px = p.p[0]; s.py = p.p[1]; s.pz = p.p[2];
    s.vx = p.v[0]; s.vy = p.v[1]; s.vz = p.v[2];
    s.yaw = float(p.yaw) / 65536.0f * kTwoPi;
    s.pitch = float(p.pitch) / 32767.0f * kHalfPi;
    s.crouch = float(p.crouch) / 255.0f;
    s.flags = p.flags; s.health = p.health; s.weapon = p.weapon; s.move = p.move; s.shot_seq = p.shot_seq;
    return s;
}

// Values that are not finite or absurd never reach the game (a damaged or hostile packet).
inline bool state_is_sane(const PlayerState& s) {
    for (float f : { s.px, s.py, s.pz, s.vx, s.vy, s.vz, s.yaw, s.pitch, s.crouch })
        if (!std::isfinite(f)) return false;
    return std::abs(s.px) < 2.0e6f && std::abs(s.py) < 2.0e6f && std::abs(s.pz) < 2.0e6f && s.health <= 100;
}

inline float lerp_angle(float a, float b, float t) {
    float d = std::fmod(b - a, 6.2831853f);
    if (d > 3.14159265f) d -= 6.2831853f;
    if (d < -3.14159265f) d += 6.2831853f;
    return a + d * t;
}

inline PlayerState mix_state(const PlayerState& a, const PlayerState& b, float t) {
    PlayerState o = t < 0.5f ? a : b;          // flags, health, weapon, keys, shot counter: the nearer one
    o.px = a.px + (b.px - a.px) * t; o.py = a.py + (b.py - a.py) * t; o.pz = a.pz + (b.pz - a.pz) * t;
    o.vx = a.vx + (b.vx - a.vx) * t; o.vy = a.vy + (b.vy - a.vy) * t; o.vz = a.vz + (b.vz - a.vz) * t;
    o.yaw = lerp_angle(a.yaw, b.yaw, t);
    o.pitch = a.pitch + (b.pitch - a.pitch) * t;
    o.crouch = a.crouch + (b.crouch - a.crouch) * t;
    // a shot or a jump between the two states is not lost by taking the older one
    o.shot_seq = int16_t(uint16_t(b.shot_seq - a.shot_seq)) > 0 && t > 0.0f ? b.shot_seq : o.shot_seq;
    return o;
}

// The states received from one player, in time order.
struct Snap { uint32_t t; PlayerState s; };
struct SnapBuffer {
    std::deque<Snap> snaps;
    uint32_t dropped = 0;                      // late, duplicated or insane states thrown away
    float    lag_ms = 0.0f;                    // how old states are when they arrive (rises at once, falls slowly)

    // How far in the past this player is shown: far enough that two states normally lie on
    // either side of the moment (so it is interpolated, not guessed), on a slow link too.
    uint32_t delay_ms() const {
        return std::clamp(uint32_t(lag_ms) + 50u, kInterpDelayMs, kMaxInterpDelayMs);
    }

    // `now`: the session clock at arrival. false: not stored (out of order, duplicate, or not sane).
    bool add(uint32_t t, const PlayerState& s, uint32_t now) {
        if (!state_is_sane(s)) { ++dropped; return false; }
        {
            const float lag = float(std::min(now > t ? now - t : 0u, 2000u));
            lag_ms = lag > lag_ms ? lag : lag_ms + (lag - lag_ms) * 0.01f;
        }
        if (!snaps.empty()) {
            const uint32_t last = snaps.back().t;
            if (t <= last) {
                if (last - t < 5000u) { ++dropped; return false; }      // duplicate or out of order
                snaps.clear();                                          // the clock restarted (new host)
            }
        }
        snaps.push_back({ t, s });
        while (snaps.size() > 64) snaps.pop_front();
        return true;
    }

    struct Sample { PlayerState s; bool teleported = false; bool extrapolated = false; bool valid = false; };
    // The player at session time `rt`. Between two states: interpolated (or the later one, flagged,
    // if they are a teleport apart). Past the newest: along its velocity for kMaxExtrapolateMs at
    // most, then standing where that leads - a stall does not send the player off into the distance.
    Sample sample(uint32_t rt) const {
        Sample o;
        if (snaps.empty()) return o;
        o.valid = true;
        if (rt <= snaps.front().t) { o.s = snaps.front().s; return o; }
        if (rt >= snaps.back().t) {
            const float dt = float(std::min(rt - snaps.back().t, kMaxExtrapolateMs)) * 0.001f;
            o.s = snaps.back().s;
            o.s.px += o.s.vx * dt; o.s.py += o.s.vy * dt; o.s.pz += o.s.vz * dt;
            o.extrapolated = rt > snaps.back().t;
            if (rt - snaps.back().t > kMaxExtrapolateMs) { o.s.vx = o.s.vy = o.s.vz = 0.0f; o.s.move = 0; }
            return o;
        }
        for (size_t i = snaps.size() - 1; i > 0; --i) {
            if (snaps[i - 1].t > rt) continue;
            const Snap& a = snaps[i - 1];
            const Snap& b = snaps[i];
            const float dx = b.s.px - a.s.px, dy = b.s.py - a.s.py, dz = b.s.pz - a.s.pz;
            const float gap = float(b.t - a.t) * 0.001f;
            const float sp = std::sqrt(b.s.vx * b.s.vx + b.s.vy * b.s.vy + b.s.vz * b.s.vz);
            // further apart than the movement between them explains: a teleport / respawn, no slide
            const float allow = kTeleportDistance + 1.5f * sp * gap;
            if ((b.s.flags & kFlagTeleport) || dx * dx + dy * dy + dz * dz > allow * allow) { o.s = b.s; o.teleported = true; return o; }
            o.s = mix_state(a.s, b.s, float(rt - a.t) / float(std::max<uint32_t>(b.t - a.t, 1u)));
            return o;
        }
        o.s = snaps.back().s;
        return o;
    }
};

// Action events from the network: each (sender, kind) stream is de-duplicated by its sequence
// number, and events are handed to the game when the delayed picture reaches the moment they
// happened - so a remote muzzle flash and recoil coincide with the pose that fired.
class EventGate {
public:
    // false: a duplicate or an older repeat, dropped.
    bool push(const Event& e) {
        if (e.kind == Event::Fire || e.kind == Event::Airstrike) {
            for (Last& l : last_) {
                if (l.from != e.from || l.kind != e.kind) continue;
                if (int16_t(uint16_t(e.seq - l.seq)) <= 0) { ++duplicates; return false; }
                l.seq = e.seq;
                queue_.push_back(e);
                return true;
            }
            last_.push_back({ e.from, e.kind, e.seq });
        }
        queue_.push_back(e);
        return true;
    }
    // Events whose time has come at render time `rt` (session ms). Only shots wait for the
    // picture; hits, deaths and leaves are gameplay and pass at once. Nothing waits longer than
    // `max_hold_ms` (clocks that disagree must not hold an event back).
    // `rt_of(sender id)`: the session time that sender is shown at.
    template <typename RtOf>
    void release(RtOf&& rt_of, std::vector<Event>& out, uint32_t max_hold_ms = kMaxInterpDelayMs + 100) {
        size_t keep = 0;
        for (size_t i = 0; i < queue_.size(); ++i) {
            const Event& e = queue_[i];
            const bool timed = e.kind == Event::Fire;
            const uint32_t rt = timed ? uint32_t(rt_of(e.from)) : 0u;
            const bool due = !timed || e.t <= rt || e.t - rt > max_hold_ms;
            if (due) out.push_back(e);
            else queue_[keep++] = e;
        }
        queue_.resize(keep);
    }
    void forget(uint32_t from) {
        last_.erase(std::remove_if(last_.begin(), last_.end(), [&](const Last& l) { return l.from == from; }), last_.end());
    }
    void clear() { queue_.clear(); last_.clear(); }
    size_t   waiting() const { return queue_.size(); }
    uint32_t duplicates = 0;

private:
    struct Last { uint32_t from; uint8_t kind; uint16_t seq; };
    std::vector<Last>  last_;
    std::vector<Event> queue_;
};

// ---------------------------------------------------------------------------------------------
// Cars. A car is simulated on ONE machine: its driver's (and, once the driver has got out,
// still theirs, until somebody else gets in). That machine sends where the car is; everybody
// else draws it from those states, a moment in the past, as they draw the players. Nothing is
// predicted and nothing is corrected: the driver's own car answers at once, and what others
// see is what happened.
//
// A car is named by its number in the list every machine builds the same way at start-up.
constexpr int kMaxCars = 32;
struct CarState {
    uint8_t car = 0;
    uint8_t flags = 0;                         // kCar*
    float p[3]{ 0, 0, 0 };                     // the body's origin
    float q[4]{ 0, 0, 0, 1 };                  // ... and how it is turned (x y z w)
    float v[3]{ 0, 0, 0 }, w[3]{ 0, 0, 0 };    // moving and turning (world; m/s, rad/s)
    float steer = 0;                           // the front wheels' angle (rad)
    float susp[4]{ 0, 0, 0, 0 };               // each wheel: how far above its place at rest (m)
    float spin[4]{ 0, 0, 0, 0 };               // each wheel: how fast it turns (rad/s)
};
// kCarDriven: the sender sits at its wheel (otherwise it stands, or rolls on, where it was left).
// Bits 4..7: wheel 0..3 touches the ground.
enum : uint8_t { kCarDriven = 1, kCarLights = 2, kCarBrake = 4, kCarTeleport = 8, kCarContact0 = 16 };

#pragma pack(push, 1)
struct PackedCar {
    uint8_t  car, flags;
    float    p[3];
    int16_t  q[4];             // x 32767
    int16_t  v[3];             // cm/s (+-327 m/s)
    int16_t  w[3];             // mrad/s
    int8_t   steer;            // 1/100 rad
    int8_t   susp[4];          // 4 mm
    int16_t  spin[4];          // 1/100 rad/s (+-327: a 0.35 m wheel at 400 km/h)
};
#pragma pack(pop)
static_assert(sizeof(PackedCar) == 47, "wire size of a car state");

inline PackedCar pack_car(const CarState& s) {
    auto i16 = [](float f, float k) { return int16_t(std::lround(std::clamp(f * k, -32767.0f, 32767.0f))); };
    auto i8 = [](float f, float k) { return int8_t(std::lround(std::clamp(f * k, -127.0f, 127.0f))); };
    PackedCar o{};
    o.car = s.car; o.flags = s.flags;
    for (int i = 0; i < 3; ++i) { o.p[i] = s.p[i]; o.v[i] = i16(s.v[i], 100.0f); o.w[i] = i16(s.w[i], 1000.0f); }
    for (int i = 0; i < 4; ++i) { o.q[i] = i16(s.q[i], 32767.0f); o.susp[i] = i8(s.susp[i], 250.0f); o.spin[i] = i16(s.spin[i], 100.0f); }
    o.steer = i8(s.steer, 100.0f);
    return o;
}

inline CarState unpack_car(const PackedCar& p) {
    CarState s;
    s.car = p.car; s.flags = p.flags;
    for (int i = 0; i < 3; ++i) { s.p[i] = p.p[i]; s.v[i] = float(p.v[i]) / 100.0f; s.w[i] = float(p.w[i]) / 1000.0f; }
    float n = 0.0f;
    for (int i = 0; i < 4; ++i) { s.q[i] = float(p.q[i]) / 32767.0f; n += s.q[i] * s.q[i]; s.susp[i] = float(p.susp[i]) / 250.0f; s.spin[i] = float(p.spin[i]) / 100.0f; }
    n = std::sqrt(n);
    if (n > 0.5f) { for (float& f : s.q) f /= n; } else { s.q[0] = s.q[1] = s.q[2] = 0.0f; s.q[3] = 1.0f; }
    s.steer = float(p.steer) / 100.0f;
    return s;
}

inline bool car_is_sane(const CarState& s) {
    for (float f : { s.p[0], s.p[1], s.p[2] })
        if (!std::isfinite(f) || std::abs(f) > 2.0e6f) return false;
    return s.car < kMaxCars;
}

// Between two states `gap` seconds apart. With a gap the place follows the curve that leaves a
// with a's velocity and arrives at b with b's (a straight line between states 33 ms apart shows
// as a car going round a bend in short straight pieces); gap 0: a straight line. The turn goes
// the short way.
inline CarState mix_car(const CarState& a, const CarState& b, float t, float gap) {
    CarState o = t < 0.5f ? a : b;
    const float t2 = t * t, t3 = t2 * t;
    const float h00 = 2.0f * t3 - 3.0f * t2 + 1.0f, h10 = t3 - 2.0f * t2 + t, h01 = -2.0f * t3 + 3.0f * t2, h11 = t3 - t2;
    for (int i = 0; i < 3; ++i) {
        o.p[i] = gap > 0.0f ? h00 * a.p[i] + h10 * gap * a.v[i] + h01 * b.p[i] + h11 * gap * b.v[i]
                            : a.p[i] + (b.p[i] - a.p[i]) * t;
        o.v[i] = a.v[i] + (b.v[i] - a.v[i]) * t;
        o.w[i] = a.w[i] + (b.w[i] - a.w[i]) * t;
    }
    float d = 0.0f, n = 0.0f;
    for (int i = 0; i < 4; ++i) d += a.q[i] * b.q[i];
    for (int i = 0; i < 4; ++i) { o.q[i] = a.q[i] * (1.0f - t) + (d < 0.0f ? -b.q[i] : b.q[i]) * t; n += o.q[i] * o.q[i]; }
    n = std::sqrt(std::max(n, 1e-12f));
    for (float& f : o.q) f /= n;
    o.steer = a.steer + (b.steer - a.steer) * t;
    for (int i = 0; i < 4; ++i) { o.susp[i] = a.susp[i] + (b.susp[i] - a.susp[i]) * t; o.spin[i] = a.spin[i] + (b.spin[i] - a.spin[i]) * t; }
    return o;
}

// The host's clock as a client knows it: host time - local time, from the time stamps of the
// host's snapshots. A stamp is only ever seen LATE (the network, and the frame in which the
// packet is noticed), never early - so each sample is the true offset or less, and the
// largest of the last few seconds is the best there is. The estimate moves toward that at a
// limited rate (half a millisecond a second): everything a client stamps and everything it
// draws from others' stamps runs on this clock, and a clock that follows every sample's
// noise moves a fast car back and forth on the other screens.
struct ClockOffset {
    double value = 0.0;
    bool   valid = false;
    void reset() { valid = false; n_ = 0; }
    // `sample`: host stamp + half the round trip - local time at arrival (ms). `dt_ms`: since the last one.
    void add(double sample, double dt_ms) {
        ring_[n_++ % kWindow] = sample;
        double best = -1e300;
        for (int i = 0; i < std::min(n_, kWindow); ++i) best = std::max(best, ring_[i]);
        if (!valid || std::abs(best - value) > 250.0) { value = best; valid = true; return; }
        const double step = 0.0005 * std::clamp(dt_ms, 0.0, 200.0);
        value += std::clamp(best - value, -step, step);
    }
private:
    static constexpr int kWindow = 180;        // 3 s of snapshots
    double ring_[kWindow]{};
    int    n_ = 0;
};

// The states received for one car, in time order (the players' SnapBuffer, for a car).
// Times are session milliseconds WITH their fraction: at 100 km/h a whole millisecond is 3 cm,
// and a car stamped and sampled in whole milliseconds moved unevenly by that much every frame.
struct CarBuffer {
    struct Snap { double t; CarState s; };
    std::deque<Snap> snaps;
    uint32_t dropped = 0;
    float    lag_ms = 0.0f;

    double delay_ms() const { return std::clamp(double(lag_ms) + 50.0, double(kInterpDelayMs), double(kMaxInterpDelayMs)); }
    // The delay in use: it follows delay_ms() at 1 % of the time that passes, so a late packet
    // makes the car run 1 % slow for a moment instead of jumping back (and forward again later).
    double delay_now = double(kInterpDelayMs);
    double advance_delay(double passed_ms) {
        const double step = 0.01 * std::clamp(passed_ms, 0.0, 200.0);
        delay_now += std::clamp(delay_ms() - delay_now, -step, step);
        return delay_now;
    }

    bool add(double t, const CarState& s, double now) {
        if (!car_is_sane(s)) { ++dropped; return false; }
        {
            const float lag = float(std::clamp(now - t, 0.0, 2000.0));
            lag_ms = lag > lag_ms ? lag : lag_ms + (lag - lag_ms) * 0.01f;
        }
        if (!snaps.empty()) {
            const double last = snaps.back().t;
            if (t <= last) {
                if (last - t < 5000.0) { ++dropped; return false; }
                snaps.clear();
            }
        }
        snaps.push_back({ t, s });
        while (snaps.size() > 64) snaps.pop_front();
        return true;
    }

    struct Sample { CarState s; bool teleported = false; bool extrapolated = false; bool valid = false; };
    Sample sample(double rt) const {
        Sample o;
        if (snaps.empty()) return o;
        o.valid = true;
        if (rt <= snaps.front().t) { o.s = snaps.front().s; return o; }
        if (rt >= snaps.back().t) {
            // Nothing newer: on along its velocity for a quarter of a second at most, then it stands.
            // (a parked car is told once a second: it has no velocity, so it simply stays)
            const float dt = float(std::min(rt - snaps.back().t, double(kMaxExtrapolateMs))) * 0.001f;
            o.s = snaps.back().s;
            for (int i = 0; i < 3; ++i) o.s.p[i] += o.s.v[i] * dt;
            o.extrapolated = rt > snaps.back().t;
            if (rt - snaps.back().t > kMaxExtrapolateMs) { for (int i = 0; i < 3; ++i) o.s.v[i] = o.s.w[i] = 0.0f; for (float& f : o.s.spin) f = 0.0f; }
            return o;
        }
        for (size_t i = snaps.size() - 1; i > 0; --i) {
            if (snaps[i - 1].t > rt) continue;
            const Snap& a = snaps[i - 1];
            const Snap& b = snaps[i];
            const float gap = float(b.t - a.t) * 0.001f;
            float d2 = 0.0f, sp2 = 0.0f;
            for (int k = 0; k < 3; ++k) { const float d = b.s.p[k] - a.s.p[k]; d2 += d * d; sp2 += std::max(a.s.v[k] * a.s.v[k], b.s.v[k] * b.s.v[k]); }
            const float allow = kTeleportDistance + 1.5f * std::sqrt(sp2) * gap;
            if ((b.s.flags & kCarTeleport) || d2 > allow * allow) { o.s = b.s; o.teleported = true; return o; }
            const float t = float((rt - a.t) / std::max(b.t - a.t, 1e-3));
            // (states far apart in time - a parked car's, a second apart - are joined by a straight
            //  line: the curve through two velocities is only right while they describe the way between)
            o.s = mix_car(a.s, b.s, t, gap < 0.25f ? gap : 0.0f);
            return o;
        }
        o.s = snaps.back().s;
        return o;
    }
};

}  // namespace vibe::net
