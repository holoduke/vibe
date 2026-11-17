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
#include <cstring>
#include <random>

namespace qlike::net {

namespace {

#pragma pack(push, 1)
struct Beacon     { char magic[4]; uint8_t ver; uint8_t role; uint8_t nplayers; uint16_t port; uint32_t id; };
struct MsgState   { uint8_t type; uint32_t id; uint32_t t; PlayerState s; };
struct MsgSnapHdr { uint8_t type; uint32_t server_t; uint8_t n; };
struct SnapEntry  { uint32_t id; uint32_t t; PlayerState s; };
struct MsgEvent   { uint8_t type; Event e; };
struct MsgHello   { uint8_t type; uint32_t id; char name[16]; };
struct MsgWorld   { uint8_t type; WorldState w; };
#pragma pack(pop)

enum : uint8_t { kMsgState = 1, kMsgSnap = 2, kMsgEvent = 3, kMsgHello = 4, kMsgWorld = 5 };
constexpr uint8_t kVersion = 1;

float lerp_angle(float a, float b, float t) {
    float d = std::fmod(b - a, 6.2831853f);
    if (d > 3.14159265f) d -= 6.2831853f;
    if (d < -3.14159265f) d += 6.2831853f;
    return a + d * t;
}

PlayerState mix_state(const PlayerState& a, const PlayerState& b, float t) {
    PlayerState o = t < 0.5f ? a : b;          // flags / health: nearest
    o.px = a.px + (b.px - a.px) * t; o.py = a.py + (b.py - a.py) * t; o.pz = a.pz + (b.pz - a.pz) * t;
    o.vx = a.vx + (b.vx - a.vx) * t; o.vy = a.vy + (b.vy - a.vy) * t; o.vz = a.vz + (b.vz - a.vz) * t;
    o.yaw = lerp_angle(a.yaw, b.yaw, t);
    o.pitch = a.pitch + (b.pitch - a.pitch) * t;
    o.crouch = a.crouch + (b.crouch - a.crouch) * t;
    return o;
}

}  // namespace

struct Session::Impl {
    ENetHost* host = nullptr;
    uint16_t  port = 0;
    SOCKET    disc = INVALID_SOCKET;
    ENetPeer* server = nullptr;            // client / connecting: the host
    bool      enet_ok = false;
    double    t0 = -1.0;                   // local origin of the millisecond clock
    double    offset_ms = 0.0;             // client: host clock - local clock
    bool      have_offset = false;
    double    next_send = 0.0, next_snap = 0.0, next_beacon = 0.0, next_addr = 0.0;
    double    connect_deadline = 0.0, retry_after = 0.0;
    std::vector<uint32_t> bcast;           // broadcast addresses (network order)
    struct Seen { uint32_t id, ip; uint16_t port; uint8_t role, n; double t; };
    std::vector<Seen> seen;
    struct PeerInfo { ENetPeer* peer; uint32_t id; char name[16]; };
    std::vector<PeerInfo> peers;           // host: the connected clients
    PlayerState local{};

    void send(ENetPeer* p, const void* data, size_t len, bool reliable) {
        if (!p) return;
        ENetPacket* pk = enet_packet_create(data, len, reliable ? ENET_PACKET_FLAG_RELIABLE : 0);
        if (pk && enet_peer_send(p, reliable ? 1 : 0, pk) < 0) enet_packet_destroy(pk);
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

Session::Session() : d_(new Impl) {}
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
        ENetAddress a; a.host = ENET_HOST_ANY; a.port = uint16_t(kGamePort + k);
        d_->host = enet_host_create(&a, kMaxPlayers, 2, 0, 0);
        if (d_->host) d_->port = a.port;
    }
    if (!d_->host) { log::warn("[net] no free game port"); stop(); return false; }

    d_->disc = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (d_->disc != INVALID_SOCKET) {
        BOOL yes = TRUE;
        setsockopt(d_->disc, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char*>(&yes), sizeof(yes));
        setsockopt(d_->disc, SOL_SOCKET, SO_BROADCAST, reinterpret_cast<const char*>(&yes), sizeof(yes));
        sockaddr_in sa{}; sa.sin_family = AF_INET; sa.sin_addr.s_addr = INADDR_ANY; sa.sin_port = htons(kDiscoveryPort);
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
    players_.clear(); events_.clear();
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
    if (!p.snaps.empty()) {
        const uint32_t last = p.snaps.back().t;
        if (t <= last) {
            if (last - t < 5000u) return;       // duplicate or out of order
            p.snaps.clear();                    // the clock restarted (new host)
        }
    }
    p.snaps.push_back({ t, s });
    while (p.snaps.size() > 64) p.snaps.pop_front();
}

void Session::interpolate(uint32_t rt, double now) {
    for (auto& p : players_) {
        p.fresh = !p.snaps.empty() && (now - p.last_rx) < 2.0;
        if (p.snaps.empty()) continue;
        const auto& q = p.snaps;
        if (rt <= q.front().t) { p.state = q.front().s; continue; }
        if (rt >= q.back().t) {
            // Nothing newer yet: carry on along the last velocity for a
            // quarter of a second at most, then stand still.
            const float dt = std::min(float(rt - q.back().t) * 0.001f, 0.25f);
            p.state = q.back().s;
            p.state.px += p.state.vx * dt; p.state.py += p.state.vy * dt; p.state.pz += p.state.vz * dt;
            continue;
        }
        for (size_t i = q.size() - 1; i > 0; --i) {
            if (q[i - 1].t > rt) continue;
            const auto& a = q[i - 1]; const auto& b = q[i];
            const float dx = b.s.px - a.s.px, dy = b.s.py - a.s.py, dz = b.s.pz - a.s.pz;
            if (dx * dx + dy * dy + dz * dz > 36.0f) { p.state = b.s; break; }      // teleport / respawn: no slide
            p.state = mix_state(a.s, b.s, float(rt - a.t) / float(std::max<uint32_t>(b.t - a.t, 1u)));
            break;
        }
    }
}

void Session::to_solo(const char* why) {
    if (role_ != Role::Solo) log::infof("[net] back to searching (%s)", why);
    d_->server = nullptr;
    d_->have_offset = false;
    players_.clear();
    role_ = Role::Solo;
}

void Session::send_event(const Event& e) {
    if (!d_->host) return;
    MsgEvent m{ kMsgEvent, e };
    if (role_ == Role::Host) d_->send_all(&m, sizeof(m), true);
    else if (role_ == Role::Client) d_->send(d_->server, &m, sizeof(m), true);
}

void Session::send_world(const WorldState& w) {
    if (!d_->host) return;
    MsgWorld m{ kMsgWorld, w };
    if (role_ == Role::Host) { m.w.request = 0; d_->send_all(&m, sizeof(m), true); }
    else if (role_ == Role::Client) { m.w.request = 1; d_->send(d_->server, &m, sizeof(m), true); }
}

bool Session::take_world(WorldState& out) {
    if (!world_new_) return false;
    world_new_ = false;
    out = world_;
    return true;
}

std::vector<Event> Session::take_events() {
    std::vector<Event> out;
    out.swap(events_);
    return out;
}

void Session::update(double now, const PlayerState& local) {
    if (!d_->host) return;
    if (d_->t0 < 0.0) d_->t0 = now;
    d_->local = local;
    const double local_ms = (now - d_->t0) * 1000.0;

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
            if (type == kMsgState && len == sizeof(MsgState) && role_ == Role::Host) {
                MsgState m; std::memcpy(&m, data, sizeof(m));
                for (auto& pi : d_->peers) if (pi.peer == ev.peer && pi.id == m.id) add_snap(m.id, m.t, m.s, now);
            } else if (type == kMsgSnap && len >= sizeof(MsgSnapHdr) && ev.peer == d_->server) {
                MsgSnapHdr h; std::memcpy(&h, data, sizeof(h));
                if (len == sizeof(h) + size_t(h.n) * sizeof(SnapEntry)) {
                    const double cand = double(h.server_t) + 0.5 * double(ev.peer->roundTripTime) - local_ms;
                    if (!d_->have_offset || std::abs(cand - d_->offset_ms) > 250.0) d_->offset_ms = cand;
                    else d_->offset_ms += (cand - d_->offset_ms) * 0.02;
                    d_->have_offset = true;
                    for (uint8_t i = 0; i < h.n; ++i) {
                        SnapEntry e; std::memcpy(&e, data + sizeof(h) + size_t(i) * sizeof(e), sizeof(e));
                        add_snap(e.id, e.t, e.s, now);
                    }
                }
            } else if (type == kMsgEvent && len == sizeof(MsgEvent)) {
                MsgEvent m; std::memcpy(&m, data, sizeof(m));
                if (m.e.kind == Event::Leave) remove_player(m.e.from);
                events_.push_back(m.e);
                if (role_ == Role::Host) d_->send_all(&m, sizeof(m), true, ev.peer);
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
                        events_.push_back(m.e);
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
                sockaddr_in to{}; to.sin_family = AF_INET; to.sin_addr.s_addr = addr; to.sin_port = htons(kDiscoveryPort);
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
        MsgState m{ kMsgState, id_, uint32_t(std::max(0.0, local_ms + d_->offset_ms)), local };
        d_->send(d_->server, &m, sizeof(m), false);
    }
    if (role_ == Role::Host && now >= d_->next_snap) {
        d_->next_snap = std::max(d_->next_snap + 1.0 / 60.0, now - 1.0 / 60.0);
        uint8_t buf[sizeof(MsgSnapHdr) + (kMaxPlayers + 1) * sizeof(SnapEntry)];
        MsgSnapHdr h{ kMsgSnap, uint32_t(local_ms), 0 };
        size_t off = sizeof(h);
        auto put = [&](uint32_t pid, uint32_t t, const PlayerState& s) {
            if (h.n > kMaxPlayers) return;
            SnapEntry e{ pid, t, s };
            std::memcpy(buf + off, &e, sizeof(e)); off += sizeof(e); ++h.n;
        };
        put(id_, uint32_t(local_ms), local);
        for (const auto& p : players_) if (!p.snaps.empty()) put(p.id, p.snaps.back().t, p.snaps.back().s);
        std::memcpy(buf, &h, sizeof(h));
        d_->send_all(buf, off, false);
    }
    enet_host_flush(d_->host);

    // ---- remote players at the render time ----
    players_.erase(std::remove_if(players_.begin(), players_.end(),
                                  [&](const RemotePlayer& p) { return p.last_rx > 0.0 && now - p.last_rx > 6.0; }),
                   players_.end());
    const double server_now = role_ == Role::Host ? local_ms : local_ms + d_->offset_ms;
    interpolate(uint32_t(std::max(0.0, server_now - double(kInterpDelayMs))), now);
}

}  // namespace qlike::net
