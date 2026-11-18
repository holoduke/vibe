// LAN multiplayer: discovery, automatic host election, state replication.
//
// How it works (the model of Half-Life / Source, sized for a LAN):
//  - Every instance starts as a host with nobody connected ("solo") and
//    broadcasts a small beacon on the local network twice a second.
//  - Two solo instances that see each other: the one with the LOWER random
//    id stays host, the other connects to it. A solo instance that sees a
//    host with players joins that host. When the host goes away the clients
//    fall back to solo and the same rule elects a new host.
//  - Each client simulates its own player (so its own movement is instant:
//    "client-side prediction" with nothing to correct) and sends its state
//    30 times a second, stamped with the shared session clock. The host
//    relays everyone's latest state as snapshots.
//  - Remote players are drawn ~100 ms in the past, interpolated between two
//    received states ("entity interpolation"), so motion is smooth even
//    when packets arrive unevenly or one is lost; short gaps are bridged by
//    extrapolating along the last velocity.
//  - Shots and hits are reliable events. A hit is decided on the shooter's
//    machine against what the shooter sees ("favour the shooter", the result
//    lag compensation gives).
//
// Transport: ENet (reliable + unreliable-sequenced channels over UDP).
// No Vulkan / engine types in here: see vk_engine/multiplayer.cpp for the
// game side.
#pragma once

#include <glm/vec3.hpp>

#include <cstdint>
#include <deque>
#include <string>
#include <vector>

namespace qlike::net {

constexpr uint16_t kDiscoveryPort = 47777;
constexpr uint16_t kGamePort      = 47778;   // first choice; the next free one of 8 is taken
constexpr int      kMaxPlayers    = 8;
constexpr double   kSendInterval  = 1.0 / 30.0;
constexpr uint32_t kInterpDelayMs = 100;     // remote players are shown this far in the past

#pragma pack(push, 1)
struct PlayerState {
    float px, py, pz;          // AABB centre (Player::position)
    float vx, vy, vz;
    float yaw, pitch;
    float crouch;              // 0 standing .. 1 crawling
    uint8_t flags;             // bit 0 on ground, bit 1 swimming, bit 2 flashlight
    uint8_t health;            // 0..100
};
struct Event {
    enum Kind : uint8_t { Fire = 1, Hit = 2, Death = 3, Leave = 4, Airstrike = 5 };   // Airstrike: a = target, b = run direction
    uint8_t  kind;
    uint32_t from;             // Fire: shooter. Hit: shooter. Death: the one who died. Leave: who left
    uint32_t to;               // Hit: victim. Death: killer
    float    a[3];             // Fire: origin. Hit: hit position
    float    b[3];             // Fire: direction
    float    v;                // Hit: damage
};
// Shared world state: weather + clock. A client that changes it sends a
// request to the host; the host applies it and sends it to everybody (and
// again once a second, so a late joiner and any drift are brought in line).
struct WorldState {
    uint8_t request;           // 1 = a client asks the host to apply this
    uint8_t weather_on, day_night;
    int8_t  preset;            // weather type (-1 = none)
    int8_t  season;            // 0 spring .. 3 winter
    float   transition_s;      // seconds a change takes
    float   manual, time_of_day, day_len_min, change_min;
    float   tgt[9];            // weather target amounts (host -> clients)
    float   cur[9];            // weather current amounts (host -> clients)
};
#pragma pack(pop)

struct RemotePlayer {
    uint32_t    id = 0;
    std::string name;
    PlayerState state{};       // interpolated to the render time (see Session::update)
    bool        fresh = false; // has at least one state and is not stale
    // internal
    struct Snap { uint32_t t; PlayerState s; };
    std::deque<Snap> snaps;
    double last_rx = 0.0;
};

enum class Role { Off, Solo, Connecting, Client, Host };

class Session {
public:
    Session();
    ~Session();
    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;

    bool start(const std::string& player_name);
    void stop();

    // Once per frame. `now` = monotonic seconds. Sends the local state at
    // kSendInterval, receives, runs discovery / election, and interpolates
    // every remote player to the current render time.
    void update(double now, const PlayerState& local);

    void send_event(const Event& e);          // reliable, to everybody else
    std::vector<Event> take_events();         // events received since the last call
    void send_world(const WorldState& w);     // host: to all clients; client: request to the host
    bool take_world(WorldState& out);         // the newest world state received since the last call

    const std::vector<RemotePlayer>& players() const { return players_; }
    Role        role() const { return role_; }
    uint32_t    id() const { return id_; }
    const char* role_name() const;
    std::string name_of(uint32_t id) const;
    uint32_t    ping_ms() const;              // client: round trip to the host

private:
    struct Impl;
    Impl* d_ = nullptr;
    uint32_t id_ = 0;
    Role role_ = Role::Off;
    std::string name_;
    std::vector<RemotePlayer> players_;
    std::vector<Event> events_;
    WorldState world_{};
    bool world_new_ = false;

    RemotePlayer& player(uint32_t id);
    void remove_player(uint32_t id);
    void add_snap(uint32_t id, uint32_t t, const PlayerState& s, double now);
    void interpolate(uint32_t render_t, double now);
    void to_solo(const char* why);
};

}  // namespace qlike::net
