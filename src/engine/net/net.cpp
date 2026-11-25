// LAN multiplayer session - see net.h for the model.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <enet/enet.h>
#include <iphlpapi.h>

#include "engine/net/net.h"
#include "engine/log.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <random>

namespace vibe::net {

namespace {

#pragma pack(push, 1)
struct Beacon     { char magic[4]; uint8_t ver; uint8_t role; uint8_t nplayers; uint16_t port; uint32_t id; };
struct MsgState   { uint8_t type; uint32_t id; uint32_t t; PackedState s; };
struct MsgSnapHdr { uint8_t type; uint32_t server_t; uint8_t n; };
struct SnapEntry  { uint32_t id; uint32_t t; PackedState s; };
struct MsgEvent   { uint8_t type; Event e; };
struct MsgHello   { uint8_t type; uint32_t id; char name[16]; };
struct MsgWorld   { uint8_t type; WorldState w; };
struct MsgCar     { uint8_t type; uint32_t id; uint32_t t; uint8_t t_frac; PackedCar c; };      // (t_frac: 1/256 ms)
#pragma pack(pop)

enum : uint8_t { kMsgState = 1, kMsgSnap = 2, kMsgEvent = 3, kMsgHello = 4, kMsgWorld = 5, kMsgCar = 6, kMsgNpc = 7 };
constexpr uint8_t kVersion = kProtocolVersion;

// VIBE_NET_PORT=<n>: another discovery port (game ports follow it), so test instances on a
// machine or a network do not meet games that are being played.
uint16_t discovery_port() {
    static const uint16_t p = [] { const char* e = std::getenv("VIBE_NET_PORT"); const int v = e ? std::atoi(e) : 0; return uint16_t(v > 1024 && v < 65000 ? v : kDiscoveryPort); }();
    return p;
}
uint16_t game_port() { return uint16_t(discovery_port() == kDiscoveryPort ? kGamePort : discovery_port() + 1); }

}  // namespace

struct Session::Impl {
    ENetHost* host = nullptr;
    uint16_t  port = 0;
    SOCKET    disc = INVALID_SOCKET;
    ENetPeer* server = nullptr;            // client / connecting: the host
    bool      enet_ok = false;
    double    t0 = -1.0;                   // local origin of the millisecond clock
    double    offset_ms = 0.0;             // client: host clock - local clock (ClockOffset::value)
    bool      have_offset = false;
    ClockOffset clock;
    double    clock_last_ms = 0.0;
    double    next_send = 0.0, next_snap = 0.0, next_beacon = 0.0, next_addr = 0.0;
    double    connect_deadline = 0.0, retry_after = 0.0;
    std::vector<uint32_t> bcast;           // broadcast addresses (network order)
    struct Seen { uint32_t id, ip; uint16_t port; uint8_t role, n; double t; };
    std::vector<Seen> seen;
    struct PeerInfo { ENetPeer* peer; uint32_t id; char name[16]; };
    std::vector<PeerInfo> peers;           // host: the connected clients
    PlayerState local{};
    double    server_now_ms = 0.0;
    Session::Stats* stats = nullptr;

    void send(ENetPeer* p, const void* data, size_t len, bool reliable) {
        if (!p) return;
        ENetPacket* pk = enet_packet_create(data, len, reliable ? ENET_PACKET_FLAG_RELIABLE : 0);
        if (pk && enet_peer_send(p, reliable ? 1 : 0, pk) < 0) { enet_packet_destroy(pk); return; }
        if (pk && stats) { stats->tx_bytes += len; ++stats->tx_packets; }
    }
    void send_all(const void* data, size_t len, bool reliable, ENetPeer* except = nullptr) {
        for (auto& pi : peers) if (pi.peer != except) send(pi.peer, data, len, reliable);
    }
    void refresh_broadcast() {
        bcast.clear();
        bcast.push_back(0xFFFFFFFFu);
        ULONG sz = 16 * 1024;
        std::vector<uint8_t> buf(sz);
        const ULONG fl = GAA_FLAG_SKIP_ANYCAST | GAA_FLAG_SKIP_MULTICAST | GAA_FLAG_SKIP_DNS_SERVER;
        ULONG r = GetAdaptersAddresses(AF_INET, fl, nullptr, reinterpret_cast<IP_ADAPTER_ADDRESSES*>(buf.data()), &sz);
        if (r == ERROR_BUFFER_OVERFLOW) {
            buf.resize(sz);
            r = GetAdaptersAddresses(AF_INET, fl, nullptr, reinterpret_cast<IP_ADAPTER_ADDRESSES*>(buf.data()), &sz);
        }
        if (r != NO_ERROR) return;
        for (auto* a = reinterpret_cast<IP_ADAPTER_ADDRESSES*>(buf.data()); a; a = a->Next) {
            if (a->OperStatus != IfOperStatusUp || a->IfType == IF_TYPE_SOFTWARE_LOOPBACK) continue;
            for (auto* u = a->FirstUnicastAddress; u; u = u->Next) {
                if (!u->Address.lpSockaddr || u->Address.lpSockaddr->sa_family != AF_INET) continue;
                const uint32_t ip = reinterpret_cast<sockaddr_in*>(u->Address.lpSockaddr)->sin_addr.s_addr;
                const uint32_t pl = u->OnLinkPrefixLength;
                if (pl == 0 || pl >= 31) continue;
                const uint32_t mask = htonl(~((1u << (32 - pl)) - 1u));
                const uint32_t b = ip | ~mask;
                if (std::find(bcast.begin(), bcast.end(), b) == bcast.end()) bcast.push_back(b);
            }
        }
    }
};

Session::Session() : d_(new Impl) { d_->stats = &stats_; }
Session::~Session() { stop(); delete d_; }

const char* Session::role_name() const {
    switch (role_) {
    case Role::Solo:       return "searching";
    case Role::Connecting: return "joining";
    case Role::Client:     return "joined";
    case Role::Host:       return "hosting";
    default:               return "off";
    }
}

std::string Session::name_of(uint32_t id) const {
    if (id == id_) return name_;
    for (const auto& p : players_) if (p.id == id) return p.name.empty() ? "player" : p.name;
    return "player";
}

uint32_t Session::ping_ms() const {
    return (role_ == Role::Client && d_->server) ? d_->server->roundTripTime : 0u;
}

bool Session::start(const std::string& player_name) {
    stop();
    name_ = player_name.substr(0, 15);
    if (enet_initialize() != 0) { log::warn("[net] enet_initialize failed"); return false; }
    d_->enet_ok = true;
    std::random_device rd;
    do { id_ = (uint32_t(rd()) << 1) ^ uint32_t(rd()) ^ uint32_t(GetTickCount64()); } while (id_ == 0);

    for (uint16_t k = 0; k < 8 && !d_->host; ++k) {
        ENetAddress a; a.host = ENET_HOST_ANY; a.port = uint16_t(game_port() + k);
        d_->host = enet_host_create(&a, kMaxPlayers, 2, 0, 0);
        if (d_->host) d_->port = a.port;
    }
    if (!d_->host) { log::warn("[net] no free game port"); stop(); return false; }

    d_->disc = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (d_->disc != INVALID_SOCKET) {
        BOOL yes = TRUE;
        setsockopt(d_->disc, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char*>(&yes), sizeof(yes));
        setsockopt(d_->disc, SOL_SOCKET, SO_BROADCAST, reinterpret_cast<const char*>(&yes), sizeof(yes));
        sockaddr_in sa{}; sa.sin_family = AF_INET; sa.sin_addr.s_addr = INADDR_ANY; sa.sin_port = htons(discovery_port());
        if (bind(d_->disc, reinterpret_cast<sockaddr*>(&sa), sizeof(sa)) != 0) {
            log::warnf("[net] discovery bind failed (%d)", WSAGetLastError());
            closesocket(d_->disc); d_->disc = INVALID_SOCKET;
        } else {
            u_long nb = 1; ioctlsocket(d_->disc, FIONBIO, &nb);
        }
    }
    role_ = Role::Solo;
    d_->t0 = -1.0;
    log::infof("[net] started: id %08X, game port %u, discovery %s", id_, unsigned(d_->port),
               d_->disc != INVALID_SOCKET ? "on" : "OFF");
    return true;
}

void Session::stop() {
    if (d_->host) {
        if (d_->server) enet_peer_disconnect_now(d_->server, 0);
        for (auto& pi : d_->peers) enet_peer_disconnect_now(pi.peer, 0);
        enet_host_flush(d_->host);
        enet_host_destroy(d_->host);
        d_->host = nullptr;
    }
    if (d_->disc != INVALID_SOCKET) { closesocket(d_->disc); d_->disc = INVALID_SOCKET; }
    if (d_->enet_ok) { enet_deinitialize(); d_->enet_ok = false; }
    d_->server = nullptr; d_->peers.clear(); d_->seen.clear();
    players_.clear(); gate_.clear(); cars_.clear(); car_queue_.clear();
    role_ = Role::Off;
}

RemotePlayer& Session::player(uint32_t id) {
    for (auto& p : players_) if (p.id == id) return p;
    players_.emplace_back();
    players_.back().id = id;
    return players_.back();
}

void Session::remove_player(uint32_t id) {
    players_.erase(std::remove_if(players_.begin(), players_.end(), [&](const RemotePlayer& p) { return p.id == id; }),
                   players_.end());
}

void Session::add_snap(uint32_t id, uint32_t t, const PlayerState& s, double now) {
    if (id == 0 || id == id_) return;
    RemotePlayer& p = player(id);
    p.last_rx = now;
    if (!p.buf.add(t, s, uint32_t(std::max(0.0, d_->server_now_ms)))) ++stats_.dropped_states;
}

void Session::add_car(uint32_t sender, double t, const CarState& s, double now) {
    if (sender == 0 || sender == id_ || s.car >= kMaxCars) return;
    RemoteCar* rc = nullptr;
    for (auto& c : cars_) if (c.car == s.car) rc = &c;
    if (!rc) { cars_.emplace_back(); rc = &cars_.back(); rc->car = s.car; }
    // Another machine has taken the car over: its states are stamped by the same clock, but
    // nothing of the old sender's is to be blended into them.
    if (rc->sender != sender) { rc->buf.snaps.clear(); rc->sender = sender; }
    // VIBE_LOG_NET_CAR=1: is each state where the last one, its time stamp and the two velocities
    // say it should be? (the evidence when a car moves unevenly on another screen)
    if (static const bool lg = std::getenv("VIBE_LOG_NET_CAR") != nullptr; lg && !rc->buf.snaps.empty()) {
        const CarBuffer::Snap& a = rc->buf.snaps.back();
        const double gap = (t - a.t) * 0.001;
        double d2 = 0.0, va = 0.0, vb = 0.0;
        for (int k = 0; k < 3; ++k) { d2 += double(s.p[k] - a.s.p[k]) * double(s.p[k] - a.s.p[k]); va += double(a.s.v[k]) * a.s.v[k]; vb += double(s.v[k]) * s.v[k]; }
        const double off = std::sqrt(d2) - 0.5 * (std::sqrt(va) + std::sqrt(vb)) * gap;
        if (std::abs(off) > 0.01) log::infof("[net] car %d: state %.2f ms after the last is %.3f m off its velocity (%.2f ms at this speed)", int(s.car), gap * 1000.0, off, off / std::max(std::sqrt(vb), 0.1) * 1000.0);
    }
    if (static const char* lv = std::getenv("VIBE_LOG_NET_CAR"); lv && lv[0] == '2')
        log::infof("[net] RX %.3f %.4f %.4f %.4f %.3f %.3f %.3f", t, double(s.p[0]), double(s.p[1]), double(s.p[2]), double(s.v[0]), double(s.v[1]), double(s.v[2]));
    if (rc->buf.add(t, s, std::max(0.0, d_->server_now_ms))) rc->last_rx = now;
    else ++stats_.dropped_states;
}

void Session::interpolate(double session_now_ms, double now) {
    const uint32_t session_now = uint32_t(std::max(0.0, session_now_ms));
    for (auto& c : cars_) {
        c.fresh = !c.buf.snaps.empty();
        c.age = now - c.last_rx;
        const double passed = c.last_now_ms > 0.0 ? session_now_ms - c.last_now_ms : 0.0;
        c.last_now_ms = session_now_ms;
        c.shown_ms = std::max(0.0, session_now_ms - c.buf.advance_delay(passed));
        const CarBuffer::Sample s = c.buf.sample(c.shown_ms);
        c.extrapolated = s.extrapolated;
        if (!s.valid) continue;
        c.state = s.s;
        c.teleported = s.teleported;
    }
    for (auto& p : players_) {
        p.fresh = !p.buf.snaps.empty() && (now - p.last_rx) < 2.0;
        const uint32_t delay = p.buf.delay_ms();
        const SnapBuffer::Sample s = p.buf.sample(session_now > delay ? session_now - delay : 0u);
        if (!s.valid) continue;
        p.state = s.s;
        p.teleported = s.teleported;
    }
}

void Session::to_solo(const char* why) {
    if (role_ != Role::Solo) log::infof("[net] back to searching (%s)", why);
    d_->server = nullptr;
    d_->have_offset = false;
    players_.clear();
    gate_.clear();
    role_ = Role::Solo;
}

void Session::send_event(Event e) {
    if (!d_->host) return;
    e.t = uint32_t(std::max(0.0, d_->server_now_ms));
    e.seq = ++event_seq_;
    MsgEvent m{ kMsgEvent, e };
    if (role_ == Role::Host) d_->send_all(&m, sizeof(m), true);
    else if (role_ == Role::Client) d_->send(d_->server, &m, sizeof(m), true);
}

void Session::queue_car(const CarState& c, double now) {
    if (!d_->host || d_->t0 < 0.0) return;
    if (role_ == Role::Client && !d_->have_offset) return;
    const double local_ms = (now - d_->t0) * 1000.0;
    car_queue_.push_back({ c, std::max(0.0, role_ == Role::Client ? local_ms + d_->offset_ms : local_ms) });
}

void Session::send_world(const WorldState& w) {
    if (!d_->host) return;
    MsgWorld m{ kMsgWorld, w };
    if (role_ == Role::Host) { m.w.request = 0; d_->send_all(&m, sizeof(m), true); }
    else if (role_ == Role::Client) { m.w.request = 1; d_->send(d_->server, &m, sizeof(m), true); }
}

void Session::send_npc(const uint8_t* data, size_t size) {
    if (!d_->host || role_ != Role::Host || !data || size == 0 || size > kMaxNpcBytes) return;
    uint8_t buf[1 + kMaxNpcBytes];
    buf[0] = kMsgNpc;
    std::memcpy(buf + 1, data, size);
    d_->send_all(buf, 1 + size, false);
}

bool Session::take_npc(std::vector<uint8_t>& out) {
    if (!npc_new_) return false;
    npc_new_ = false;
    out = npc_;
    return true;
}

bool Session::take_world(WorldState& out) {
    if (!world_new_) return false;
    world_new_ = false;
    out = world_;
    return true;
}

std::vector<Event> Session::take_events() {
    std::vector<Event> out;
    const uint32_t now_ms = uint32_t(std::max(0.0, d_->server_now_ms));
    gate_.release([&](uint32_t from) {
        uint32_t delay = kInterpDelayMs;
        for (const auto& p : players_) if (p.id == from) delay = p.buf.delay_ms();
        return now_ms > delay ? now_ms - delay : 0u;
    }, out);
    return out;
}

void Session::update(double now, const PlayerState& local) {
    if (!d_->host) return;
    if (d_->t0 < 0.0) d_->t0 = now;
    d_->local = local;
    const double local_ms = (now - d_->t0) * 1000.0;
    d_->server_now_ms = role_ == Role::Client ? local_ms + d_->offset_ms : local_ms;

    // ---- transport ----
    ENetEvent ev;
    while (enet_host_service(d_->host, &ev, 0) > 0) {
        switch (ev.type) {
        case ENET_EVENT_TYPE_CONNECT:
            if (ev.peer == d_->server) {
                role_ = Role::Client;
                d_->have_offset = false;
                enet_peer_timeout(ev.peer, 0, 2000, 4000);
                MsgHello h{ kMsgHello, id_, {} };
                std::strncpy(h.name, name_.c_str(), sizeof(h.name) - 1);
                d_->send(ev.peer, &h, sizeof(h), true);
                log::info("[net] joined a host");
            } else if ((role_ == Role::Solo || role_ == Role::Host) && int(d_->peers.size()) < kMaxPlayers - 1) {
                enet_peer_timeout(ev.peer, 0, 2000, 4000);
                // Tell the newcomer who is here.
                MsgHello me{ kMsgHello, id_, {} };
                std::strncpy(me.name, name_.c_str(), sizeof(me.name) - 1);
                d_->send(ev.peer, &me, sizeof(me), true);
                for (auto& pi : d_->peers) {
                    if (!pi.id) continue;
                    MsgHello h{ kMsgHello, pi.id, {} };
                    std::memcpy(h.name, pi.name, sizeof(h.name));
                    d_->send(ev.peer, &h, sizeof(h), true);
                }
                d_->peers.push_back({ ev.peer, 0u, {} });
                if (role_ != Role::Host) log::info("[net] now hosting");
                role_ = Role::Host;
            } else {
                enet_peer_disconnect_now(ev.peer, 0);          // busy joining someone else: they try again
            }
            break;
        case ENET_EVENT_TYPE_RECEIVE: {
            const uint8_t* data = ev.packet->data;
            const size_t len = ev.packet->dataLength;
            const uint8_t type = len ? data[0] : 0;
            stats_.rx_bytes += len; ++stats_.rx_packets;
            if (type == kMsgState && len == sizeof(MsgState) && role_ == Role::Host) {
                MsgState m; std::memcpy(&m, data, sizeof(m));
                for (auto& pi : d_->peers) if (pi.peer == ev.peer && pi.id == m.id) add_snap(m.id, m.t, unpack_state(m.s), now);
            } else if (type == kMsgSnap && len >= sizeof(MsgSnapHdr) && ev.peer == d_->server) {
                MsgSnapHdr h; std::memcpy(&h, data, sizeof(h));
                if (len == sizeof(h) + size_t(h.n) * sizeof(SnapEntry)) {
                    const double cand = double(h.server_t) + 0.5 * double(ev.peer->roundTripTime) - local_ms;
                    if (!d_->have_offset) d_->clock.reset();
                    d_->clock.add(cand, d_->have_offset ? local_ms - d_->clock_last_ms : 0.0);      // (net_state.h ClockOffset: why not a running average)
                    d_->clock_last_ms = local_ms;
                    d_->offset_ms = d_->clock.value;
                    d_->have_offset = true;
                    for (uint8_t i = 0; i < h.n; ++i) {
                        SnapEntry e; std::memcpy(&e, data + sizeof(h) + size_t(i) * sizeof(e), sizeof(e));
                        add_snap(e.id, e.t, unpack_state(e.s), now);
                    }
                }
            } else if (type == kMsgCar && len == sizeof(MsgCar)) {
                MsgCar m; std::memcpy(&m, data, sizeof(m));
                if (role_ == Role::Host) {
                    // from a client, about a car it simulates: taken, and passed on to the others at once
                    for (auto& pi : d_->peers)
                        if (pi.peer == ev.peer && pi.id == m.id) { add_car(m.id, double(m.t) + double(m.t_frac) / 256.0, unpack_car(m.c), now); d_->send_all(&m, sizeof(m), false, ev.peer); }
                } else if (ev.peer == d_->server) {
                    add_car(m.id, double(m.t) + double(m.t_frac) / 256.0, unpack_car(m.c), now);
                }
            } else if (type == kMsgEvent && len == sizeof(MsgEvent)) {
                MsgEvent m; std::memcpy(&m, data, sizeof(m));
                if (m.e.kind == Event::Leave) { remove_player(m.e.from); gate_.forget(m.e.from); }
                // (a duplicate or a stale repeat is neither played nor passed on)
                if (gate_.push(m.e)) { if (role_ == Role::Host) d_->send_all(&m, sizeof(m), true, ev.peer); }
                else ++stats_.duplicate_events;
            } else if (type == kMsgNpc && len > 1 && len <= 1 + kMaxNpcBytes) {
                // (only the host's word counts; what is in it is npc::World::apply's to check)
                if (ev.peer == d_->server) { npc_.assign(data + 1, data + len); npc_new_ = true; }
            } else if (type == kMsgWorld && len == sizeof(MsgWorld)) {
                MsgWorld m; std::memcpy(&m, data, sizeof(m));
                // host: only requests from clients; client: only from the host
                if ((role_ == Role::Host && m.w.request == 1) || (ev.peer == d_->server && m.w.request == 0)) { world_ = m.w; world_new_ = true; }
            } else if (type == kMsgHello && len == sizeof(MsgHello)) {
                MsgHello m; std::memcpy(&m, data, sizeof(m));
                m.name[sizeof(m.name) - 1] = 0;
                if (m.id && m.id != id_) {
                    player(m.id).name = m.name;
                    if (role_ == Role::Host) {
                        for (auto& pi : d_->peers) if (pi.peer == ev.peer) { pi.id = m.id; std::memcpy(pi.name, m.name, sizeof(pi.name)); }
                        d_->send_all(&m, sizeof(m), true, ev.peer);
                        log::infof("[net] %s joined", m.name);
                    }
                }
            }
            enet_packet_destroy(ev.packet);
            break;
        }
        case ENET_EVENT_TYPE_DISCONNECT:
            if (ev.peer == d_->server) {
                to_solo(role_ == Role::Connecting ? "could not join" : "host left");
                d_->retry_after = now + 1.0;
            } else {
                for (size_t i = 0; i < d_->peers.size(); ++i) {
                    if (d_->peers[i].peer != ev.peer) continue;
                    const uint32_t pid = d_->peers[i].id;
                    d_->peers.erase(d_->peers.begin() + ptrdiff_t(i));
                    if (pid) {
                        MsgEvent m{ kMsgEvent, {} };
                        m.e.kind = Event::Leave; m.e.from = pid;
                        d_->send_all(&m, sizeof(m), true);
                        gate_.push(m.e);
                        gate_.forget(pid);
                        remove_player(pid);
                    }
                    break;
                }
                if (role_ == Role::Host && d_->peers.empty()) { role_ = Role::Solo; players_.clear(); log::info("[net] everybody left"); }
            }
            break;
        default: break;
        }
    }

    // ---- discovery ----
    if (d_->disc != INVALID_SOCKET) {
        for (;;) {
            Beacon b; sockaddr_in from{}; int fl = sizeof(from);
            const int n = recvfrom(d_->disc, reinterpret_cast<char*>(&b), sizeof(b), 0, reinterpret_cast<sockaddr*>(&from), &fl);
            if (n <= 0) break;
            if (n != sizeof(b) || std::memcmp(b.magic, "VIBE", 4) != 0 || b.ver != kVersion || b.id == id_ || b.id == 0) continue;
            bool found = false;
            for (auto& s : d_->seen) if (s.id == b.id) { s = { b.id, from.sin_addr.s_addr, b.port, b.role, b.nplayers, now }; found = true; }
            if (!found) {
                d_->seen.push_back({ b.id, from.sin_addr.s_addr, b.port, b.role, b.nplayers, now });
                log::infof("[net] found another game on the network (id %08X)", b.id);
            }
        }
        d_->seen.erase(std::remove_if(d_->seen.begin(), d_->seen.end(), [&](const Impl::Seen& s) { return now - s.t > 3.0; }),
                       d_->seen.end());
        if (now >= d_->next_addr) { d_->refresh_broadcast(); d_->next_addr = now + 5.0; }
        if (now >= d_->next_beacon) {
            d_->next_beacon = now + 0.5;
            Beacon b{ { 'V', 'I', 'B', 'E' }, kVersion,
                      uint8_t(role_ == Role::Solo ? 0 : (role_ == Role::Host ? 1 : 2)),
                      uint8_t(players_.size() + 1), d_->port, id_ };
            for (uint32_t addr : d_->bcast) {
                sockaddr_in to{}; to.sin_family = AF_INET; to.sin_addr.s_addr = addr; to.sin_port = htons(discovery_port());
                sendto(d_->disc, reinterpret_cast<const char*>(&b), sizeof(b), 0, reinterpret_cast<sockaddr*>(&to), sizeof(to));
            }
        }
    }

    // ---- election: who hosts ----
    if (role_ == Role::Solo && now >= d_->retry_after) {
        const Impl::Seen* best = nullptr;
        for (const auto& s : d_->seen) {
            if (s.role == 2) continue;                              // a client of someone
            if (s.role == 0 && s.id > id_) continue;                // they will come to us
            if (!best || s.role > best->role || (s.role == best->role && s.id < best->id)) best = &s;
        }
        if (best) {
            ENetAddress a; a.host = best->ip; a.port = best->port;
            d_->server = enet_host_connect(d_->host, &a, 2, 0);
            if (d_->server) {
                role_ = Role::Connecting;
                d_->connect_deadline = now + 3.0;
                log::infof("[net] joining %08X", best->id);
            }
        }
    } else if (role_ == Role::Connecting && now > d_->connect_deadline) {
        if (d_->server) enet_peer_reset(d_->server);
        to_solo("no answer");
        d_->retry_after = now + 1.0;
    }

    // ---- send ----
    if (role_ == Role::Client && d_->have_offset && now >= d_->next_send) {
        d_->next_send = std::max(d_->next_send + kSendInterval, now - kSendInterval);
        MsgState m{ kMsgState, id_, uint32_t(std::max(0.0, local_ms + d_->offset_ms)), pack_state(local) };
        d_->send(d_->server, &m, sizeof(m), false);
    }
    // The cars this machine simulates (queue_car).
    if ((role_ == Role::Client && d_->have_offset) || role_ == Role::Host) {
        for (const QueuedCar& qc : car_queue_) {
            const double ts = qc.t_ms;
            MsgCar m{ kMsgCar, id_, uint32_t(ts), uint8_t((ts - std::floor(ts)) * 256.0), pack_car(qc.s) };
            if (role_ == Role::Host) d_->send_all(&m, sizeof(m), false);
            else d_->send(d_->server, &m, sizeof(m), false);
        }
    }
    car_queue_.clear();
    if (role_ == Role::Host && now >= d_->next_snap) {
        d_->next_snap = std::max(d_->next_snap + 1.0 / 60.0, now - 1.0 / 60.0);
        uint8_t buf[sizeof(MsgSnapHdr) + (kMaxPlayers + 1) * sizeof(SnapEntry)];
        MsgSnapHdr h{ kMsgSnap, uint32_t(local_ms), 0 };
        size_t off = sizeof(h);
        auto put = [&](uint32_t pid, uint32_t t, const PlayerState& s) {
            if (h.n > kMaxPlayers) return;
            SnapEntry e{ pid, t, pack_state(s) };
            std::memcpy(buf + off, &e, sizeof(e)); off += sizeof(e); ++h.n;
        };
        put(id_, uint32_t(local_ms), local);
        for (const auto& p : players_) if (!p.buf.snaps.empty()) put(p.id, p.buf.snaps.back().t, p.buf.snaps.back().s);
        std::memcpy(buf, &h, sizeof(h));
        d_->send_all(buf, off, false);
    }
    enet_host_flush(d_->host);

    // ---- remote players at the render time ----
    players_.erase(std::remove_if(players_.begin(), players_.end(),
                                  [&](const RemotePlayer& p) { return p.last_rx > 0.0 && now - p.last_rx > 6.0; }),
                   players_.end());
    const double server_now = role_ == Role::Client ? local_ms + d_->offset_ms : local_ms;
    d_->server_now_ms = server_now;
    render_t_ = uint32_t(std::max(0.0, server_now - double(kInterpDelayMs)));
    interpolate(std::max(0.0, server_now), now);
}

}  // namespace vibe::net
