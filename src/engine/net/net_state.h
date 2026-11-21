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

namespace qlike::net {

constexpr uint8_t  kProtocolVersion = 2;       // 2: packed state with weapon / input / shot counter, stamped events
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
    uint8_t weapon = 0;                        // 0 laser gun, 1 airstrike tablet
    uint8_t move = 0;                          // kMove*: the movement keys held (what the player wants, for prediction)
    uint16_t shot_seq = 0;                     // shots fired so far (wraps): recoil survives a lost event
};
enum : uint8_t { kFlagGround = 1, kFlagSwim = 2, kFlagTorch = 4, kFlagReload = 8, kFlagTeleport = 16 };
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
    enum Kind : uint8_t { Fire = 1, Hit = 2, Death = 3, Leave = 4, Airstrike = 5 };   // Airstrike: a = target, b = run direction
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

}  // namespace qlike::net
