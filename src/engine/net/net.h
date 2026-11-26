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

#include "engine/net/net_state.h"

#include <glm/vec3.hpp>

#include <cstdint>
#include <deque>
#include <string>
#include <vector>

namespace vibe::net {

constexpr uint16_t kDiscoveryPort = 47777;
constexpr uint16_t kGamePort      = 47778;   // first choice; the next free one of 8 is taken
constexpr int      kMaxPlayers    = 8;
constexpr double   kSendInterval  = 1.0 / 30.0;

#pragma pack(push, 1)
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
    bool        teleported = false; // this frame's state is a jump from the last (respawn): no blending across it
    // internal
    SnapBuffer  buf;
    double last_rx = 0.0;
};

// A car somebody else tells us about (net_state.h, "Cars").
struct RemoteCar {
    uint8_t   car = 0;
    uint32_t  sender = 0;          // who simulates it (the last player to send a state for it)
    CarState  state{};             // at the render time
    bool      fresh = false;       // has a state
    bool      teleported = false;  // this frame's state is a jump from the last
    double    age = 0.0;           // seconds since a state for it last arrived
    double    shown_ms = 0.0;      // the session time it is shown at (diagnostics)
    bool      extrapolated = false; // ... which is past its newest state
    // internal
    CarBuffer buf;
    double    last_rx = 0.0;
    double    last_now_ms = 0.0;
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

    void send_event(Event e);                 // reliable, to everybody else; stamped with the session clock and a sequence number
    std::vector<Event> take_events();         // events whose moment the (delayed) picture has reached
    // Traffic since start(): payload bytes and packets, for bandwidth figures.
    struct Stats { uint64_t tx_bytes = 0, rx_bytes = 0, tx_packets = 0, rx_packets = 0; uint32_t duplicate_events = 0, dropped_states = 0; };
    const Stats& stats() const { return stats_; }
    uint32_t render_time_ms() const { return render_t_; }   // the session time remote players are shown at
    // A car this machine simulates, as it is at `now` (the same clock update() is given): sent
    // with the next update(), stamped with that moment - not with the moment it is sent.
    // The caller decides how often (30 times a second while it moves, once a second parked).
    void queue_car(const CarState& c, double now);
    const std::vector<RemoteCar>& cars() const { return cars_; }
    void send_world(const WorldState& w);     // host: to all clients; client: request to the host
    bool take_world(WorldState& out);         // the newest world state received since the last call
    // The people and cars that live their own life (src/npc): the host's world, as its own
    // bytes (npc::World::serialize) - the session only carries them. Host: to all clients, a few
    // times a second, unreliable (the next one replaces a lost one). Client: the newest received.
    static constexpr size_t kMaxNpcBytes = 4 + 160 * 27;      // (npc::kMaxAgents agents of 27 bytes)
    void send_npc(const uint8_t* data, size_t size);
    bool take_npc(std::vector<uint8_t>& out);

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
    std::vector<RemoteCar> cars_;
    struct QueuedCar { CarState s; double t_ms; };
    std::vector<QueuedCar> car_queue_;
    void add_car(uint32_t sender, double t, const CarState& s, double now);
    EventGate gate_;
    Stats stats_{};
    uint32_t render_t_ = 0;
    uint16_t event_seq_ = 0;
    WorldState world_{};
    bool world_new_ = false;
    std::vector<uint8_t> npc_;
    bool npc_new_ = false;

    RemotePlayer& player(uint32_t id);
    void remove_player(uint32_t id);
    void add_snap(uint32_t id, uint32_t t, const PlayerState& s, double now);
    void interpolate(double session_now, double now);
    void to_solo(const char* why);
};

}  // namespace vibe::net
