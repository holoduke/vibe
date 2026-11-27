// The NPC engine: see npc.h.
#include "npc/npc.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <map>
#include <queue>
#include <sstream>

namespace vibe::npc {
namespace {

constexpr float kPi = 3.14159265f;
constexpr float kWalk = 1.35f;              // m/s
constexpr float kLane = 1.9f;               // a lane's middle, from the centre line
constexpr float kCruise = 21.0f;            // m/s on the open road
constexpr float kSiteSpeed = 4.2f;          // m/s on the site's ways
constexpr float kBayRunUp = 8.8f;           // a bay can be turned into from the aisle when it lies this far along it (bay_curve)
constexpr float kDecide = 90.0f;            // a car makes up its mind this far before the way in

struct Rng {
    uint32_t s = 1;
    uint32_t next() { s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s; }
    float uni() { return float(next() >> 8) * (1.0f / 16777216.0f); }
    float range(float a, float b) { return a + (b - a) * uni(); }
    int   pick(int n) { return n <= 0 ? 0 : int(next() % uint32_t(n)); }
    bool  chance(float p) { return uni() < p; }
};

float wrap_pi(float a) { while (a > kPi) a -= 2.0f * kPi; while (a < -kPi) a += 2.0f * kPi; return a; }
// The game's yaw of a direction (x, z): 0 looks toward -z, + turns toward +x.
float yaw_of(glm::vec2 d) { return std::atan2(d.x, -d.y); }

struct Node { std::string name; glm::vec3 p; std::vector<int> adj; uint8_t zone = 0; };
// A seat, a spot, a bay or a stop. `aim`: what the hands or the eyes go to there (a desk's
// keyboard, the screen pointed at, the rocket) - for the renderer (View::aim). `zone`: which
// site it belongs to (0 the first one loaded, 1.. those added with add_site).
struct Place { std::string kind; glm::vec3 p; float yaw = 0.0f; int user = -1; int node = -1; glm::vec3 aim{0.0f}; bool has_aim = false; uint8_t zone = 0; };

enum Role : uint8_t { Cashier, Cook, Server, Local, Traveller, Runner,
                      // the people of an added site (THE ADDED SITES, below)
                      Desk, Floater, Visitor, Crew };
// What a person of a role wears (npc.h Dress, in bits 4..6 of `look`; the rest of `look` as drawn).
static uint8_t dressed(uint8_t look, Role role) {
    const uint32_t h = (uint32_t(look) * 2654435761u) >> 24;          // (0 .. 255, of the person's own byte)
    Dress d = Dress::Civilian;
    switch (role) {
    case Runner: d = Dress::Soldier; break;
    case Crew: d = h < 190 ? Dress::Technician : Dress::Garrison; break;
    case Desk: case Floater: d = h < 150 ? Dress::Office : Dress::Garrison; break;
    case Local: case Traveller: d = h < 50 ? Dress::Garrison : Dress::Civilian; break;
    default: d = Dress::Civilian; break;
    }
    return uint8_t((look & 0x8F) | (uint8_t(d) << 4));
}
enum TaskType : uint8_t { GoNode, GoPoint, Sit, Stand, EnterCar, DoAct };
struct Task { TaskType type; int a = -1; glm::vec3 p{0.0f}; float yaw = 0.0f; float time = 0.0f; Pose pose = Pose::Stand; uint8_t style = 0; glm::vec3 aim{0.0f}; bool has_aim = false; uint8_t step = 0; };

enum CarState : uint8_t { Cruise, Approach, OnPath, Waiting, Parked };
enum Cls : uint8_t { Slow, Normal, Fast, Speeder, Police };      // the kinds of driver (make_driver)

struct Agent {
    Kind kind = Kind::Person; Pose pose = Pose::Stand; uint8_t look = 0; bool hidden = false;
    glm::vec3 pos{0.0f}; float yaw = 0.0f; float speed = 0.0f;
    uint8_t style = 0;                   // View::style
    uint8_t item = 0, placed = 0;        // what it carries, and what it has set down where it is (npc.h Item, WHAT PEOPLE CARRY)
    float gone = 0.0f;                   // > 0: out of the world (shot, or a car taken) for so many seconds more
    float run_s = 0.0f; int run_dir = 1; float run_side = 0.0f, run_lat = 0.0f;      // a runner: metres along the run, which way, and how far beside its line
    // ---- a person
    Role role = Local;
    std::vector<Task> tasks;             // the next is at the back
    std::vector<int> route;              // nodes still to pass, the next at the back
    int at_node = -1;
    int seat = -1, spot = -1;            // what it holds
    int home_seat = -1;
    int home_spot = -1;                  // somebody of an added site: the spot it begins on (or home_seat)
    uint8_t zone = 0;                    // ... and which site that is (0: the first site, the road)
    bool asleep = false;                 // ... nobody is near that site: it is not there
    int car = -1;                        // a traveller's car (agent index)
    int car_seat = 0;                    // which seat of it: 0 the driver's, 1 beside, 2 behind
    float timer = 0.0f;
    float walk_v = 0.0f;                 // how fast it walks now
    float stall_t = 0.0f, stall_d = 1e9f;        // how long it has not come nearer its target, and the nearest it was
    uint8_t act = 0; float act_t = 0.0f; glm::vec3 aim{0.0f};      // an Act in progress (npc.h)
    glm::vec3 from{0.0f}; float settle = 1.0f;   // sitting down: where it stood, and how far down it is (0..1)
    // ---- a car
    CarState cs = Cruise;
    float s = 0.0f; int dir = 1; float v_want = kCruise;
    glm::vec3 rear{0.0f}; float steer = 0.0f;   // the middle of its rear axle; its front wheels' angle (+ = to the right)
    std::vector<glm::vec3> path; float along = 0.0f; float path_len = 0.0f; bool reverse = false;
    std::vector<std::vector<glm::vec3>> legs;   // what it still has to drive on the site, the next at the back
    std::vector<uint8_t> legs_rev;
    int bay = -1, stop = -1;
    int riders[3] = { -1, -1, -1 }; int out = 0;       // its people, and how many are out of it
    float wait = 0.0f; bool decided = false; bool leaving = false; float stuck = 0.0f; float visit = 0.0f;
    bool driverless = false; float abandoned = 0.0f;   // its driver was shot: it rolls to a stop, and how long it has stood since
    // ---- a car in the traffic (THE TRAFFIC, below)
    uint8_t cls = 1;                                   // what kind of driver: Cls
    float accel = 1.8f, side = 2.6f;                   // how hard he accelerates (m/s2), and how much sideways pull he takes through a bend
    float gap_t = 1.3f, patience = 3.0f;               // the time he keeps behind the car in front (s), and how long he is held up before he looks to pass
    float lat = 1.9f, lat_cmd = 1.9f;                  // metres right of the centre line, the way it drives: where it is, and where it is steering to
    uint8_t ov = 0;                                    // 0 in its lane, 1 out and passing, 2 pulling back in
    int ov_key = -1; float ov_lat = -1.9f;             // ... whom it has to get past (a Thing's key), and where it drives meanwhile
    float held = 0.0f, ov_cool = 0.0f, ov_t = 0.0f;
    bool siren = false; float siren_t = 0.0f;          // a police car: its siren, and the seconds until it is switched
    float yield = 0.0f;                                // it keeps to the side for a siren this long yet
    bool hard = false;                                 // it brakes as hard as it can
    glm::vec2 knock{0.0f}; float knock_w = 0.0f;       // struck (push_car): it slides (m/s, world) and turns (rad/s); both die away
    float shaken = 0.0f;                               // ... and its driver stands this long yet
    // ---- a person put out of a car out on the road: stands there so many seconds more, then is gone
    float leave = 0.0f;
};

float poly_len(const std::vector<glm::vec3>& p) {
    float L = 0.0f;
    for (size_t i = 1; i < p.size(); ++i) L += glm::distance(p[i - 1], p[i]);
    return L;
}
// The point `d` metres along a polyline and the direction there.
glm::vec3 poly_at(const std::vector<glm::vec3>& p, float d, glm::vec3& dir) {
    for (size_t i = 1; i < p.size(); ++i) {
        const float l = glm::distance(p[i - 1], p[i]);
        if (d <= l || i + 1 == p.size()) {
            dir = l > 1e-4f ? (p[i] - p[i - 1]) / l : glm::vec3(0.0f, 0.0f, 1.0f);
            return p[i - 1] + dir * std::clamp(d, 0.0f, l);
        }
        d -= l;
    }
    dir = glm::vec3(0.0f, 0.0f, 1.0f);
    return p.empty() ? glm::vec3(0.0f) : p.back();
}
// Rounds a polyline's corners (Chaikin, twice): a car does not turn on the spot.
std::vector<glm::vec3> smooth(std::vector<glm::vec3> p) {
    for (int pass = 0; pass < 2 && p.size() > 2; ++pass) {
        std::vector<glm::vec3> q; q.push_back(p.front());
        for (size_t i = 0; i + 1 < p.size(); ++i) {
            q.push_back(glm::mix(p[i], p[i + 1], 0.25f)); q.push_back(glm::mix(p[i], p[i + 1], 0.75f));
        }
        q.push_back(p.back());
        p.swap(q);
    }
    return p;
}

#pragma pack(push, 1)
struct Wire { uint8_t id; uint8_t bits; uint8_t look; uint8_t car; float x, y, z; int16_t yaw; int8_t speed; uint8_t act; uint8_t act_t; int16_t aim[3]; uint8_t item; };
#pragma pack(pop)
static_assert(sizeof(Wire) == 28, "an agent on the wire is 28 bytes");

}  // namespace

struct World::Impl {
    glm::vec3 origin{0.0f};
    std::vector<Node> nodes;
    std::map<std::string, int> node_of;
    std::vector<Place> seats, spots, bays, stops;
    std::map<std::string, std::vector<glm::vec3>> paths;
    Road road;
    float s_in = 0.0f, s_out = 0.0f;         // where the site's way in leaves the road, and where its way out joins it
    bool site = false;
    int home_nodes = 0;                      // how many of `nodes` are the first site's (the rest: added sites')
    // THE ADDED SITES (add_site): the pavilion by the launch pad, the pad itself. Their people
    // are only there while a player is near (set_eyes): see wake_zones.
    struct Zone { glm::vec3 centre{0.0f}; float radius = 0.0f; bool awake = false; };
    std::vector<Zone> zones;
    std::vector<glm::vec3> eyes;             // where the players are (set_eyes)
    Rng zrng;                                // the added sites' own chances (the first site's life does not change with them)

    std::vector<Agent> agents;
    Rng rng;
    std::vector<glm::vec3> run_path; float run_len = 0.0f;
    std::vector<glm::vec3> walkers;          // the players on foot (set_walkers): cars do not drive into them
    std::vector<Obstacle> obstacles;         // the players' cars (set_obstacles)
    struct ObsAt { glm::vec3 pos{0.0f}; float s = 0.0f, goff = 0.0f; bool near_road = false; };
    std::map<uint32_t, ObsAt> obs_at;        // ... and where along the road each was found last
    // Everything on the road, as the traffic sees it (build_things, once a step): metres along
    // the road, metres right of the centre line and speed along it FOR INCREASING s, half its
    // length along the road and half its width across.
    struct Thing { float s, goff, v, hl, hw; int key; bool siren; };
    std::vector<Thing> things;
    int n_overtakes = 0, n_given_up = 0;
    int steps = 0;
    float acc = 0.0f;
    std::vector<View> prev, cur;             // the last two states, for views()
    bool is_replica = false;
    float replica_gap = 0.25f;               // seconds between the last two states taken in

    // ---------------------------------------------------------------- the site
    int nearest_node(glm::vec3 p, int zone = -1) const {
        int best = -1; float bd = 1e30f;
        for (size_t i = 0; i < nodes.size(); ++i) {
            if (zone >= 0 && int(nodes[i].zone) != zone) continue;
            const float d = glm::distance(nodes[i].p, p);
            if (d < bd) { bd = d; best = int(i); }
        }
        return best;
    }
    // The nodes from `a` to `b`, `b` first (so that the next to walk to is at the back).
    std::vector<int> route(int a, int b) const {
        std::vector<int> out;
        if (a < 0 || b < 0) return out;
        if (a == b) { out.push_back(b); return out; }
        std::vector<float> dist(nodes.size(), 1e30f); std::vector<int> from(nodes.size(), -1);
        typedef std::pair<float, int> QE;
        std::priority_queue<QE, std::vector<QE>, std::greater<QE>> q;
        dist[size_t(a)] = 0.0f; q.push({ 0.0f, a });
        while (!q.empty()) {
            const QE e = q.top(); q.pop();
            if (e.first > dist[size_t(e.second)]) continue;
            if (e.second == b) break;
            for (int n : nodes[size_t(e.second)].adj) {
                const float nd = e.first + glm::distance(nodes[size_t(e.second)].p, nodes[size_t(n)].p);
                if (nd < dist[size_t(n)]) { dist[size_t(n)] = nd; from[size_t(n)] = e.second; q.push({ nd, n }); }
            }
        }
        if (from[size_t(b)] < 0) return out;
        for (int n = b; n != a; n = from[size_t(n)]) out.push_back(n);
        return out;
    }
    // `zone` 0: the site (everything there was is dropped). 1..: a site ADDED to it - its nodes,
    // seats and spots join the same lists (a link may name a node of a site read before it).
    bool parse(std::istream& in, uint8_t zone = 0) {
        if (zone == 0) { nodes.clear(); node_of.clear(); seats.clear(); spots.clear(); bays.clear(); stops.clear(); paths.clear(); zones.clear(); }
        const size_t node0 = nodes.size();
        std::string line;
        std::vector<std::pair<std::string, std::string>> links;
        while (std::getline(in, line)) {
            std::istringstream ls(line);
            std::string kw; ls >> kw;
            if (kw.empty() || kw[0] == '#') continue;
            if (kw == "node") {
                Node n; ls >> n.name >> n.p.x >> n.p.y >> n.p.z; n.p += origin; n.zone = zone;
                node_of[n.name] = int(nodes.size()); nodes.push_back(n);
            } else if (kw == "link") {
                std::string a, b; ls >> a >> b; links.push_back({ a, b });
            } else if (kw == "seat" || kw == "spot" || kw == "stop") {
                Place p; ls >> p.kind >> p.p.x >> p.p.y >> p.p.z >> p.yaw; p.p += origin; p.zone = zone;
                // (and, for some, the point the hands or the eyes go to there)
                if (glm::vec3 q; ls >> q.x >> q.y >> q.z) { p.aim = q + origin; p.has_aim = true; }
                (kw == "seat" ? seats : kw == "spot" ? spots : stops).push_back(p);
            } else if (kw == "bay") {
                Place p; p.kind = "bay"; ls >> p.p.x >> p.p.y >> p.p.z >> p.yaw; p.p += origin; bays.push_back(p);
            } else if (kw == "path") {
                std::string nm; glm::vec3 p; ls >> nm >> p.x >> p.y >> p.z; paths[nm].push_back(p + origin);
            }
        }
        for (const auto& l : links) {
            auto a = node_of.find(l.first), b = node_of.find(l.second);
            if (a == node_of.end() || b == node_of.end()) continue;
            nodes[size_t(a->second)].adj.push_back(b->second); nodes[size_t(b->second)].adj.push_back(a->second);
        }
        for (auto* v : { &seats, &spots, &bays, &stops })
            for (Place& p : *v) if (p.zone == zone) p.node = nearest_node(p.p, zone);
        // (a pump's stop and the spot of the same name: the driver's place beside the car)
        for (Place& p : stops) { auto it = node_of.find(p.kind); if (it != node_of.end()) p.node = it->second; }
        if (zone == 0) { site = !nodes.empty(); home_nodes = int(nodes.size()); return site; }
        // An added site. Which way somebody stands on a spot: toward what is looked at from
        // there, and in a group toward its middle (the spots of a group within 2.5 m).
        for (Place& p : spots) {
            if (p.zone != zone) continue;
            glm::vec2 to(0.0f);
            if (p.kind == "group") {
                glm::vec2 c(0.0f); int n = 0;
                for (const Place& o : spots) if (o.zone == zone && o.kind == "group" && glm::distance(o.p, p.p) < 2.5f) { c += glm::vec2(o.p.x, o.p.z); ++n; }
                if (n > 1) to = c / float(n) - glm::vec2(p.p.x, p.p.z);
            } else if (p.has_aim) to = glm::vec2(p.aim.x - p.p.x, p.aim.z - p.p.z);
            if (glm::dot(to, to) > 1e-4f) p.yaw = yaw_of(to);
        }
        // Where it is and how far it reaches: its people are there while a player is near.
        if (nodes.size() > node0) {
            Zone z;
            for (size_t i = node0; i < nodes.size(); ++i) z.centre += nodes[i].p;
            z.centre /= float(nodes.size() - node0);
            for (size_t i = node0; i < nodes.size(); ++i) z.radius = std::max(z.radius, glm::distance(nodes[i].p, z.centre));
            if (zones.size() < size_t(zone)) zones.resize(size_t(zone));
            zones[size_t(zone) - 1] = z;
        }
        return nodes.size() > node0;
    }
    // Where along the road a point of the site lies nearest.
    float road_s_of(glm::vec3 p) const {
        if (!road.at || road.length <= 0.0f) return 0.0f;
        float best = 0.0f, bd = 1e30f;
        for (float s = 0.0f; s < road.length; s += 4.0f) {
            glm::vec3 q; glm::vec2 d;
            if (!road.at(s, q, d)) continue;
            const float dd = glm::distance(glm::vec2(q.x, q.z), glm::vec2(p.x, p.z));
            if (dd < bd) { bd = dd; best = s; }
        }
        return best;
    }
    void link_road() {
        auto in = paths.find("in13"), out = paths.find("out13");
        if (in == paths.end() || out == paths.end() || in->second.empty() || out->second.empty()) return;
        s_in = road_s_of(in->second.front());
        s_out = road_s_of(out->second.back());
    }

    // ---------------------------------------------------------------- places
    int free_place(std::vector<Place>& v, const char* kind, bool random = true) {
        std::vector<int> c;
        for (size_t i = 0; i < v.size(); ++i) if (v[i].user < 0 && (!kind || v[i].kind == kind)) c.push_back(int(i));
        if (c.empty()) return -1;
        return c[size_t(random ? rng.pick(int(c.size())) : 0)];
    }
    void release(Agent& a) {
        const int me = int(&a - agents.data());
        if (a.seat >= 0 && seats[size_t(a.seat)].user == me) seats[size_t(a.seat)].user = -1;
        if (a.spot >= 0 && spots[size_t(a.spot)].user == me) spots[size_t(a.spot)].user = -1;
        a.seat = a.spot = -1;
    }

    // ---------------------------------------------------------------- people
    // Queues (in the order done) a walk to a seat or a spot and the stay there.
    void plan_stay(Agent& a, std::vector<Task>& plan, bool is_seat, int idx, float time, Pose pose) {
        const Place& pl = is_seat ? seats[size_t(idx)] : spots[size_t(idx)];
        Task go; go.type = GoNode; go.a = pl.node; plan.push_back(go);
        Task ap; ap.type = GoPoint; ap.p = pl.p; if (is_seat) ap.p = seat_stand_point(idx); plan.push_back(ap);
        Task st; st.type = is_seat ? Sit : Stand; st.a = idx; st.p = pl.p; st.yaw = pl.yaw; st.time = time; st.pose = pose; plan.push_back(st);
        (void)a;
    }
    // Where somebody stands to sit down on a seat, and gets up to: in front of it, toward the
    // way that leads there. A chair pushed in at a desk or a table has no room in front: beside
    // it, on the side the way comes from.
    glm::vec3 seat_stand_point(int idx) const {
        const Place& pl = seats[size_t(idx)];
        const glm::vec3 np = nodes[size_t(std::max(pl.node, 0))].p;
        if (pl.kind != "desk" && pl.kind != "meet") return glm::mix(np, pl.p, 0.55f);
        const glm::vec3 right(std::cos(pl.yaw), 0.0f, std::sin(pl.yaw)), fwd(std::sin(pl.yaw), 0.0f, -std::cos(pl.yaw));
        const float side = glm::dot(np - pl.p, right) < 0.0f ? -1.0f : 1.0f;
        glm::vec3 p = pl.p + right * (side * 0.56f) - fwd * 0.12f;
        p.y = np.y;
        return p;
    }
    // ---------------------------------------------------------------- THE ADDED SITES
    // The pavilion by the launch pad and the pad itself (add_site: assets/office/office.nav from
    // tools/office/make_office.py, assets/rocket/site.nav). Who lives there:
    //   Desk     somebody with a desk of his own (seat kind "desk", or a chair at the meeting
    //            table): works there most of the time; now and then gets up - to a colleague's
    //            desk to point at the screen ("point": only beside somebody who sits), to the
    //            kitchen ("coffee"), to the glass to look at the pad ("window"), to the wall of
    //            screens ("wall"), to a sofa - and comes back.
    //   Floater  somebody who is on the way most of the time: from one of those places to
    //            another, up the stair to the rail of the mezzanine ("rail"), out on the terrace.
    //   Visitor  outside: in a small group that talks ("group"), looking up at the rocket
    //            ("watch"), at the apron's edge ("edge"); moves on after a minute or two.
    //   Crew     the technicians: from one thing to see to ("tech": a tank, the pump house,
    //            the tower's foot, the mount, the control house) to the next, busy there a while.
    // THEY ARE ONLY THERE WHILE A PLAYER IS NEAR (wake_zones): nobody near, they are hidden and
    // not stepped; when a player comes within reach they begin again at their places. Their
    // chances come from a generator of their own (zrng), so the first site's life - the service
    // station, the traffic - is step for step what it is without them.
    int free_in_zone(std::vector<Place>& v, const char* kind, uint8_t zone) {
        std::vector<int> c;
        for (size_t i = 0; i < v.size(); ++i) if (v[i].user < 0 && v[i].zone == zone && v[i].kind == kind) c.push_back(int(i));
        return c.empty() ? -1 : c[size_t(zrng.pick(int(c.size())))];
    }
    // How somebody stands on a spot of that kind (View::style), and whether he is busy with his hands.
    static void spot_manner(const Place& pl, Task& st, Rng& r) {
        if (pl.kind == "tech") { st.pose = Pose::Work; st.style = 0; }
        else if (pl.kind == "point") st.style = 5;
        else if (pl.kind == "wall") st.style = r.chance(0.5f) ? 5 : 7;
        else if (pl.kind == "group" || pl.kind == "coffee") st.style = 6;
        else st.style = 7;                                   // (window, terrace, rail, watch, edge: looks at what is there)
        st.aim = pl.aim; st.has_aim = pl.has_aim;
    }
    bool zone_stand(Agent& a, std::vector<Task>& plan, const char* kind, float t0, float t1) {
        int i = -1;
        if (std::string(kind) == "point") {
            // (only beside a colleague who sits at that desk)
            std::vector<int> c;
            for (size_t k = 0; k < spots.size(); ++k) {
                if (spots[k].user >= 0 || spots[k].zone != a.zone || spots[k].kind != "point") continue;
                for (const Place& s : seats)
                    if (s.user >= 0 && s.kind == "desk" && glm::distance(s.p, spots[k].p) < 1.2f && agents[size_t(s.user)].pose == Pose::Sit) { c.push_back(int(k)); break; }
            }
            if (!c.empty()) i = c[size_t(zrng.pick(int(c.size())))];
        } else if (std::string(kind) == "group") {
            // (rather where somebody stands already: a group, not three people each alone)
            std::vector<int> c;
            for (size_t k = 0; k < spots.size(); ++k) {
                if (spots[k].user >= 0 || spots[k].zone != a.zone || spots[k].kind != "group") continue;
                for (const Place& o : spots)
                    if (o.user >= 0 && o.kind == "group" && glm::distance(o.p, spots[k].p) < 2.5f) { c.push_back(int(k)); break; }
            }
            i = c.empty() ? free_in_zone(spots, kind, a.zone) : c[size_t(zrng.pick(int(c.size())))];
        } else i = free_in_zone(spots, kind, a.zone);
        if (i < 0) return false;
        spots[size_t(i)].user = int(&a - agents.data()); a.spot = i;
        plan_stay(a, plan, false, i, zrng.range(t0, t1), Pose::Stand);
        spot_manner(spots[size_t(i)], plan.back(), zrng);
        return true;
    }
    bool zone_sit(Agent& a, std::vector<Task>& plan, int seat, float t0, float t1) {
        if (seat < 0 || seats[size_t(seat)].user >= 0) return false;
        seats[size_t(seat)].user = int(&a - agents.data()); a.seat = seat;
        plan_stay(a, plan, true, seat, zrng.range(t0, t1), Pose::Sit);
        return true;
    }
    void think_zone(Agent& a, std::vector<Task>& plan) {
        const float r = zrng.uni();
        switch (a.role) {
        case Desk: {
            const bool meeting = a.home_seat >= 0 && seats[size_t(a.home_seat)].kind == "meet";
            if (r < (meeting ? 0.86f : 0.66f) && zone_sit(a, plan, a.home_seat, 50.0f, 190.0f)) break;
            if (r < 0.80f && zone_stand(a, plan, "point", 14.0f, 36.0f)) break;
            if (r < 0.88f && zone_stand(a, plan, "coffee", 20.0f, 55.0f)) break;
            if (r < 0.93f && zone_stand(a, plan, "window", 15.0f, 40.0f)) break;
            if (r < 0.97f && zone_stand(a, plan, "wall", 12.0f, 30.0f)) break;
            if (zone_sit(a, plan, free_in_zone(seats, "sofa", a.zone), 40.0f, 120.0f)) break;
            zone_sit(a, plan, a.home_seat, 50.0f, 190.0f);
            break; }
        case Floater: {
            static const char* kinds[6] = { "rail", "window", "coffee", "wall", "terrace", "point" };
            for (int tries = 0; tries < 4 && plan.empty(); ++tries) zone_stand(a, plan, kinds[zrng.pick(6)], 7.0f, 20.0f);
            break; }
        case Visitor:
            if (r < 0.50f && zone_stand(a, plan, "group", 50.0f, 160.0f)) break;
            if (r < 0.82f && zone_stand(a, plan, "watch", 30.0f, 100.0f)) break;
            if (zone_stand(a, plan, "edge", 30.0f, 90.0f)) break;
            zone_stand(a, plan, "group", 50.0f, 160.0f);
            break;
        case Crew:
            if (r < 0.14f && zone_stand(a, plan, "group", 20.0f, 50.0f)) break;
            zone_stand(a, plan, "tech", 12.0f, 34.0f);
            break;
        default: break;
        }
        if (plan.empty()) { Task st; st.type = Stand; st.p = a.pos; st.yaw = a.yaw; st.time = zrng.range(3.0f, 8.0f); plan.push_back(st); }
    }
    // Somebody of an added site at the place it begins on (when its site wakes, and when
    // somebody new takes the place of one who was shot).
    void zone_start(Agent& a) {
        const int me = int(&a - agents.data());
        release(a);
        a.tasks.clear(); a.route.clear(); a.act = 0; a.act_t = 0.0f; a.hidden = false; a.asleep = false; a.gone = 0.0f;
        a.speed = a.walk_v = 0.0f; a.stall_t = 0.0f; a.stall_d = 1e9f; a.style = 0; a.pose = Pose::Stand; a.from = glm::vec3(0.0f);
        {   // (what they go about with: the crew a toolbox or a tablet, those on the way a clipboard, a tablet, a coffee or a phone, some visitors a phone)
            const uint32_t h = (uint32_t(a.look) * 2246822519u + uint32_t(me) * 3266489917u) >> 24;
            a.placed = 0;
            a.item = a.role == Crew ? uint8_t(h < 150 ? Item::Toolbox : Item::Tablet)
                   : a.role == Floater ? uint8_t(h < 70 ? Item::Clipboard : h < 130 ? Item::Tablet : h < 190 ? Item::Cup : Item::Phone)
                   : a.role == Visitor ? uint8_t(h < 90 ? Item::Phone : Item::None)
                   : a.role == Desk ? uint8_t(h < 40 ? Item::Clipboard : Item::None) : uint8_t(Item::None);
        }
        if (a.home_seat >= 0 && seats[size_t(a.home_seat)].user < 0) {
            const Place& s = seats[size_t(a.home_seat)];
            seats[size_t(a.home_seat)].user = me; a.seat = a.home_seat;
            a.pos = s.p; a.yaw = s.yaw; a.pose = Pose::Sit; a.at_node = s.node;
            Task st; st.type = Sit; st.a = a.home_seat; st.p = s.p; st.yaw = s.yaw; st.time = zrng.range(15.0f, 190.0f);
            a.tasks.push_back(st);
        } else if (a.home_spot >= 0 && spots[size_t(a.home_spot)].user < 0) {
            const Place& s = spots[size_t(a.home_spot)];
            spots[size_t(a.home_spot)].user = me; a.spot = a.home_spot;
            a.pos = s.p; a.yaw = s.yaw; a.at_node = s.node;
            Task st; st.type = Stand; st.a = a.home_spot; st.p = s.p; st.yaw = s.yaw; st.time = zrng.range(10.0f, 120.0f);
            spot_manner(s, st, zrng);
            if (st.pose == Pose::Work) st.time = zrng.range(6.0f, 30.0f);
            a.tasks.push_back(st);
        } else {
            std::vector<int> c;
            for (size_t i = 0; i < nodes.size(); ++i) if (nodes[i].zone == a.zone) c.push_back(int(i));
            if (!c.empty()) { a.at_node = c[size_t(zrng.pick(int(c.size())))]; a.pos = nodes[size_t(a.at_node)].p; }
        }
    }
    // A site's people are there while a player is within reach of it (430 m from its edge; they
    // go when the last player is 560 m away - bodies are drawn to 150 m, so nobody sees either).
    void wake_zones() {
        for (size_t z = 0; z < zones.size(); ++z) {
            Zone& zn = zones[z];
            bool near_ = false;
            for (const glm::vec3& e : eyes)
                if (glm::distance(glm::vec2(e.x, e.z), glm::vec2(zn.centre.x, zn.centre.z)) < zn.radius + (zn.awake ? 560.0f : 430.0f)) { near_ = true; break; }
            if (near_ == zn.awake) continue;
            zn.awake = near_;
            for (Agent& a : agents) {
                if (a.kind != Kind::Person || int(a.zone) != int(z) + 1) continue;
                if (near_) zone_start(a);
                else { release(a); a.tasks.clear(); a.route.clear(); a.act = 0; a.hidden = true; a.asleep = true; a.speed = a.walk_v = 0.0f; a.pose = Pose::Stand; }
            }
        }
    }
    void set_plan(Agent& a, std::vector<Task>& plan) {
        a.tasks.assign(plan.rbegin(), plan.rend());
    }
    bool hold_seat(Agent& a, const char* kind) {
        const int i = free_place(seats, kind);
        if (i < 0) return false;
        seats[size_t(i)].user = int(&a - agents.data()); a.seat = i; return true;
    }
    bool hold_spot(Agent& a, const char* kind) {
        const int i = free_place(spots, kind);
        if (i < 0) return false;
        spots[size_t(i)].user = int(&a - agents.data()); a.spot = i; return true;
    }
    // WHAT PEOPLE CARRY. Where a seat's sitter sets a thing down (npc.h item_place: the same rule the
    // renderer draws it by), and what somebody takes along from a place it stood at: a coffee from
    // the pavilion's kitchen, a bag or a coffee from the shop's till or shelves.
    glm::vec3 seat_place(int idx, glm::vec3 p, float yaw) const {
        if (idx < 0) return item_place(p, yaw, 3);
        const std::string& k = seats[size_t(idx)].kind;
        if (k == "desk") return item_place(p, yaw, 2, seats[size_t(idx)].aim);
        if (k == "booth" || k == "chair" || k == "picnic" || k == "stool" || k == "meet") return item_place(p, yaw, 1);
        return item_place(p, yaw, 3);
    }
    void fetched(Agent& a, const Task& t) {
        if (a.item || t.a < 0 || size_t(t.a) >= spots.size() || t.type != Stand) return;
        const std::string& k = spots[size_t(t.a)].kind;
        Rng& R = a.zone ? zrng : rng;
        if (k == "coffee" && (a.role == Desk || a.role == Floater || a.role == Visitor) && R.chance(0.8f)) a.item = uint8_t(Item::Cup);
        else if ((k == "pay" || k == "browse") && a.role == Traveller) { const float r = R.uni(); a.item = uint8_t(r < 0.55f ? Item::Bag : r < 0.75f ? Item::Cup : r < 0.85f ? Item::Box : Item::None); }
        else if (k == "browse" && a.role == Local && R.chance(0.35f)) a.item = uint8_t(Item::Cup);
    }
    // What a person does next, once it has nothing to do.
    void think(Agent& a) {
        std::vector<Task> plan;
        release(a);
        if (a.zone) { think_zone(a, plan); set_plan(a, plan); return; }      // (the pavilion's and the pad's people: THE ADDED SITES)
        switch (a.role) {
        default: break;
        case Cashier:
            if (hold_spot(a, "cashier")) plan_stay(a, plan, false, a.spot, rng.range(120.0f, 400.0f), Pose::Work);
            break;
        case Cook:
            if (hold_spot(a, "cook")) plan_stay(a, plan, false, a.spot, rng.range(15.0f, 45.0f), Pose::Work);
            break;
        case Server: {
            // to a table where somebody sits (or any), a word there, back behind the bar
            std::vector<int> busy;
            for (size_t i = 0; i < seats.size(); ++i) if (seats[i].user >= 0 && (seats[i].kind == "booth" || seats[i].kind == "chair" || seats[i].kind == "stool")) busy.push_back(int(i));
            if (!busy.empty() && rng.chance(0.7f)) {
                const Place& s = seats[size_t(busy[size_t(rng.pick(int(busy.size())))])];
                Task go; go.type = GoNode; go.a = s.node; plan.push_back(go);
                Task st; st.type = Stand; st.a = -1; st.p = nodes[size_t(s.node)].p; st.yaw = yaw_of(glm::vec2(s.p.x - nodes[size_t(s.node)].p.x, s.p.z - nodes[size_t(s.node)].p.z)); st.time = rng.range(4.0f, 9.0f); st.pose = Pose::Stand; plan.push_back(st);
            }
            if (hold_spot(a, "serve")) plan_stay(a, plan, false, a.spot, rng.range(8.0f, 22.0f), Pose::Work);
            break; }
        case Local: {
            const float r = rng.uni();
            if (r < 0.45f && a.home_seat >= 0 && seats[size_t(a.home_seat)].user < 0) {
                seats[size_t(a.home_seat)].user = int(&a - agents.data()); a.seat = a.home_seat;
                plan_stay(a, plan, true, a.seat, rng.range(60.0f, 240.0f), Pose::Sit);
            } else if (r < 0.70f && hold_spot(a, "browse")) {
                plan_stay(a, plan, false, a.spot, rng.range(10.0f, 28.0f), Pose::Stand);
            } else if (r < 0.85f && hold_seat(a, rng.chance(0.5f) ? "booth" : "stool")) {
                plan_stay(a, plan, true, a.seat, rng.range(90.0f, 260.0f), Pose::Sit);
            } else if (hold_seat(a, rng.chance(0.5f) ? "picnic" : "bench")) {
                plan_stay(a, plan, true, a.seat, rng.range(40.0f, 160.0f), Pose::Sit);
            } else {
                Task go; go.type = GoNode; go.a = rng.pick(home_nodes); plan.push_back(go);
                Task st; st.type = Stand; st.p = nodes[size_t(go.a)].p; st.yaw = rng.range(-kPi, kPi); st.time = rng.range(5.0f, 20.0f); plan.push_back(st);
            }
            break; }
        case Traveller: {
            // Its errand is done (or never began): back to the car.
            if (a.car < 0) break;
            const Agent& c = agents[size_t(a.car)];
            const int n = c.stop >= 0 ? stops[size_t(c.stop)].node : c.bay >= 0 ? bays[size_t(c.bay)].node : nearest_node(c.pos);
            Task go; go.type = GoNode; go.a = n; plan.push_back(go);
            if (a.item) {
                // (what it bought goes into the boot: behind the car, facing it)
                const glm::vec3 cf(std::sin(c.yaw), 0.0f, -std::cos(c.yaw));
                Task bt; bt.type = GoPoint; bt.p = c.pos - cf * 2.75f; plan.push_back(bt);
                Task pd; pd.type = DoAct; pd.a = int(Act::PutDown); pd.p = c.pos - cf * 2.15f + glm::vec3(0.0f, 0.72f, 0.0f); pd.yaw = c.yaw; plan.push_back(pd);
            }
            Task ap; ap.type = GoPoint; ap.p = c.pos + glm::vec3(std::cos(c.yaw), 0.0f, std::sin(c.yaw)) * -1.2f; plan.push_back(ap);
            { Task da; da.type = DoAct; da.a = int(Act::CarEnter); da.p = c.pos; da.yaw = c.yaw; plan.push_back(da); }
            Task en; en.type = EnterCar; plan.push_back(en);
            break; }
        }
        if (plan.empty()) { Task st; st.type = Stand; st.p = a.pos; st.yaw = a.yaw; st.time = rng.range(3.0f, 8.0f); plan.push_back(st); }
        // (how it stands there: a cashier is a cashier; who waits a while stands some way of his own)
        for (Task& t : plan) if (t.type == Stand) {
            if (t.pose == Pose::Work) t.style = a.role == Cashier ? 1 : 0;
            else if (t.time > 6.0f) t.style = uint8_t(rng.pick(5));
        }
        set_plan(a, plan);
    }
    // A traveller gets out: what it came for.
    void errand(Agent& a, bool at_pump, bool driver, int group_seat_kind) {
        std::vector<Task> plan;
        release(a);
        const Agent& c = agents[size_t(a.car)];
        { Task da; da.type = DoAct; da.a = int(Act::CarExit); da.p = c.pos; da.yaw = c.yaw; plan.push_back(da); }
        if (at_pump && driver) {
            // fill up, standing at the pump
            const Place& stp = stops[size_t(c.stop)];
            Task ap; ap.type = GoPoint; ap.p = nodes[size_t(stp.node)].p; plan.push_back(ap);
            Task st; st.type = Stand; st.p = ap.p; st.yaw = wrap_pi(c.yaw + kPi * 0.5f); st.time = rng.range(14.0f, 26.0f); st.pose = Pose::Work; st.style = 2; plan.push_back(st);
            if (rng.chance(0.75f)) {
                if (rng.chance(0.45f) && hold_spot(a, "browse")) { plan_stay(a, plan, false, a.spot, rng.range(8.0f, 20.0f), Pose::Stand); }
                const int pay = free_place(spots, "pay", false);
                if (pay >= 0) plan_stay(a, plan, false, pay, rng.range(5.0f, 10.0f), Pose::Stand);     // (not held: people queue on one spot)
            }
        } else {
            const char* kind = group_seat_kind == 0 ? "booth" : group_seat_kind == 1 ? "stool" : "picnic";
            if (rng.chance(0.85f) && hold_seat(a, kind)) plan_stay(a, plan, true, a.seat, rng.range(80.0f, 220.0f), Pose::Sit);
            else if (hold_spot(a, "browse")) plan_stay(a, plan, false, a.spot, rng.range(12.0f, 30.0f), Pose::Stand);
        }
        // (then think() sends it back to the car)
        if (plan.empty()) { Task st; st.type = Stand; st.p = a.pos; st.yaw = a.yaw; st.time = 2.0f; plan.push_back(st); }
        set_plan(a, plan);
    }
    // HOW A PERSON WALKS. It goes the way it faces: it turns toward where it is going no faster
    // than a person turns (and gently into the last of the turn), walks slower through a turn
    // and all but stands to turn right round, picks its speed up and lets it go over half a
    // second, and cuts a corner of its way instead of touching it. Until 2026-11-25 it moved
    // straight at its target at full speed from the first step while its body swung round
    // behind (it slid sideways), stopped dead, and turned its corners on a point.
    //   near: how near counts as there - a corner on the way (0.4 m) or a place to stand (0.1 m).
    // NOBODY WALKS THROUGH ANYTHING. Two things keep a walker out of a car and out of another
    // person. It steers: whoever or whatever is close and in the way pushes its heading away
    // and round (people pass each other on the right; a car is gone round by the side nearer
    // where the walker is going). And it cannot: after every step it is put outside every car's
    // outline and half a metre from everybody else - whatever the steering did. Its own car is
    // not in its way (it has to reach the door). (Until 2026-11-26 neither existed: a walker
    // went straight along the site's graph, through a car parked across it.)
    static constexpr float kCarHalfW = 1.02f, kCarHalfL = 2.35f, kBody = 0.34f;
    // From a car's outline to a point: the way out (unit, world) and how far outside it is
    // (negative: inside by so much).
    float car_gap(const Agent& car, glm::vec2 p, glm::vec2& out) const {
        const glm::vec2 f(std::sin(car.yaw), -std::cos(car.yaw)), r(std::cos(car.yaw), std::sin(car.yaw));
        const glm::vec2 d(p.x - car.pos.x, p.y - car.pos.z);
        const float lx = glm::dot(d, r), lz = glm::dot(d, f);
        const float cx = std::clamp(lx, -kCarHalfW, kCarHalfW), cz = std::clamp(lz, -kCarHalfL, kCarHalfL);
        const glm::vec2 q(lx - cx, lz - cz);
        const float dist = glm::length(q);
        if (dist > 1e-4f) { out = r * (q.x / dist) + f * (q.y / dist); return dist; }
        // inside: out through the nearer side
        const float px = kCarHalfW - std::abs(lx), pz = kCarHalfL - std::abs(lz);
        if (px < pz) { out = r * (lx < 0.0f ? -1.0f : 1.0f); return -px; }
        out = f * (lz < 0.0f ? -1.0f : 1.0f); return -pz;
    }
    glm::vec2 steer_clear(const Agent& a, glm::vec2 want) const {
        const int me = int(&a - agents.data());
        const glm::vec2 p(a.pos.x, a.pos.z), mine_right(-want.y, want.x);
        glm::vec2 push(0.0f);
        for (size_t i = 0; i < agents.size(); ++i) {
            const Agent& o = agents[i];
            if (int(i) == me) continue;
            if (o.kind == Kind::Person) {
                if (o.hidden) continue;
                const glm::vec2 d(p.x - o.pos.x, p.y - o.pos.z);
                const float dist = glm::length(d), R = 1.25f;
                if (dist >= R || dist < 1e-3f) continue;
                const float w = (R - dist) / R;
                const glm::vec2 n = d / dist;
                push += n * (w * 1.3f);
                if (glm::dot(-n, want) > 0.2f) push += mine_right * (w * 1.1f);       // (ahead of me: round it, to my right)
            } else {
                if (int(i) == a.car) continue;
                if (std::abs(o.pos.x - p.x) > 5.5f || std::abs(o.pos.z - p.y) > 5.5f) continue;
                glm::vec2 n;
                const float gap = car_gap(o, p, n), R = 1.3f;
                if (gap >= R) continue;
                const float w = std::clamp((R - gap) / R, 0.0f, 1.0f);
                push += n * (w * 1.8f);
                glm::vec2 t(-n.y, n.x);                                             // (along its side: the way that goes on toward where I am going)
                if (glm::dot(t, want) < 0.0f) t = -t;
                if (glm::dot(-n, want) > 0.0f) push += t * (w * 1.6f);
            }
        }
        const glm::vec2 dir = want + push;
        const float l = glm::length(dir);
        return l > 1e-3f ? dir / l : want;
    }
    // After a step: outside every car, clear of everybody.
    void keep_apart(Agent& a) {
        const int me = int(&a - agents.data());
        // (several rounds: stepping clear of one body can be a step into the next - three in a doorway)
        for (int pass = 0; pass < 6; ++pass)
            for (size_t i = 0; i < agents.size(); ++i) {
                const Agent& o = agents[i];
                if (int(i) == me || o.gone > 0.0f) continue;
                glm::vec2 p(a.pos.x, a.pos.z);
                if (o.kind == Kind::Person) {
                    if (o.hidden) continue;
                    const glm::vec2 d(p.x - o.pos.x, p.y - o.pos.z);
                    const float dist = glm::length(d), R = o.pose == Pose::Sit ? 0.36f : 2.0f * kBody - 0.14f;
                    if (dist >= R) continue;
                    const glm::vec2 n = dist > 1e-3f ? d / dist : glm::vec2(std::cos(float(me)), std::sin(float(me)));
                    p += n * (R - dist);                               // (whoever moves steps back out, the whole way)
                } else {
                    if (int(i) == a.car) continue;
                    if (std::abs(o.pos.x - p.x) > 4.5f || std::abs(o.pos.z - p.y) > 4.5f) continue;
                    glm::vec2 n;
                    const float gap = car_gap(o, p, n);
                    if (gap >= kBody) continue;
                    p += n * (kBody - gap);
                }
                a.pos.x = p.x; a.pos.z = p.y;
            }
    }
    void walk_toward(Agent& a, glm::vec3 target, float dt, bool& arrived, float near_) {
        const glm::vec2 d(target.x - a.pos.x, target.z - a.pos.z);
        const float dist = glm::length(d);
        const bool stop_there = near_ < 0.2f;
        if (dist <= near_) { arrived = true; a.stall_t = 0.0f; a.stall_d = 1e9f; if (stop_there) { a.walk_v = 0.0f; a.speed = 0.0f; } return; }
        arrived = false;
        // (A place it cannot get to - somebody stands on it, a car is parked over it: after four
        //  seconds without coming nearer, this is as near as it gets.)
        if (dist < a.stall_d - 0.03f) { a.stall_d = dist; a.stall_t = 0.0f; }
        else if ((a.stall_t += dt) > 4.0f) { arrived = true; a.stall_t = 0.0f; a.stall_d = 1e9f; a.walk_v = 0.0f; a.speed = 0.0f; return; }
        const glm::vec2 want = steer_clear(a, d / dist);
        const float err = wrap_pi(yaw_of(want) - a.yaw);
        a.yaw = wrap_pi(a.yaw + std::clamp(err * 5.0f, -3.0f, 3.0f) * dt);
        // (everybody has a pace of his own: npc.h kPace - the renderer plays the walk that goes with it)
        float v = kWalk * kPace[a.look % 8] * std::clamp(1.0f - (std::abs(err) - 0.25f) / 1.1f, 0.10f, 1.0f);
        if (stop_there) v = std::min(v, 0.22f + dist * 1.7f);
        a.walk_v += std::clamp(v - a.walk_v, -3.0f * dt, 2.2f * dt);
        // (the last steps: straight to the place - it need not circle it)
        const glm::vec2 dir = dist < 0.45f ? d / dist : glm::vec2(std::sin(a.yaw), -std::cos(a.yaw));
        const float stepd = std::min(a.walk_v * dt, dist);
        a.pos.x += dir.x * stepd; a.pos.z += dir.y * stepd;
        a.pos.y += (target.y - a.pos.y) * std::min(1.0f, stepd / std::max(dist, 1e-3f));
        keep_apart(a);
        a.speed = a.walk_v; a.pose = Pose::Walk;
    }
    // Standing: round to `yaw`, eased. True once it faces that way.
    bool turn_to(Agent& a, float yaw, float dt) {
        const float err = wrap_pi(yaw - a.yaw);
        a.yaw = wrap_pi(a.yaw + std::clamp(err * 3.5f, -2.4f, 2.4f) * dt);
        return std::abs(err) < 0.12f;
    }
    // A soldier on the run: along the path at a steady pace, in two files, there and back.
    void step_runner(Agent& a, float dt) {
        if (run_len <= 1.0f) return;
        const float v = 3.4f;
        a.run_s += float(a.run_dir) * v * dt;
        if (a.run_s > run_len - 2.0f) { a.run_s = run_len - 2.0f; a.run_dir = -1; }
        if (a.run_s < 2.0f) { a.run_s = 2.0f; a.run_dir = 1; }
        glm::vec3 dir, d2;
        const glm::vec3 p = poly_at(run_path, a.run_s, dir);
        // (the way the path goes here: over four metres of it - a corner of the line is a bend to them)
        { const glm::vec3 pa = poly_at(run_path, std::max(a.run_s - 2.0f, 0.0f), d2), pb = poly_at(run_path, std::min(a.run_s + 2.0f, run_len), d2); if (glm::distance(pa, pb) > 0.5f) dir = glm::normalize(pb - pa); }
        const glm::vec2 f = glm::normalize(glm::vec2(dir.x, dir.z) + glm::vec2(1e-6f, 0.0f)) * float(a.run_dir);
        // (keeps to the right of its line, the way it is going)
        // (beside the line, on the right of the way it runs; turning back at an end it crosses over
        //  at a walk's pace sideways - it is not put on the other side)
        const glm::vec2 pd = glm::normalize(glm::vec2(dir.x, dir.z) + glm::vec2(1e-6f, 0.0f));
        const float lat_want = a.run_side * float(a.run_dir);
        a.run_lat += std::clamp(lat_want - a.run_lat, -1.2f * dt, 1.2f * dt);
        if (dt <= 0.0f) a.run_lat = lat_want;
        a.pos = p + glm::vec3(-pd.y, 0.0f, pd.x) * a.run_lat;
        const float want = yaw_of(f);
        a.yaw = wrap_pi(a.yaw + std::clamp(wrap_pi(want - a.yaw), -3.0f * dt, 3.0f * dt));
        a.speed = v; a.pose = Pose::Run; a.style = 0;
    }
    void respawn(Agent& a) {
        // somebody new: a local, walking on to the site
        const uint8_t look = uint8_t(rng.pick(256));
        const Kind k = a.kind;
        const bool police = k == Kind::Car && a.cls == Police;      // (the road keeps its police)
        const bool speeder = k == Kind::Car && a.cls == Speeder;    // (... and its one who is far too fast)
        a = Agent();
        a.kind = k; a.look = look;
        if (k == Kind::Person) a.look = dressed(look, Local);
        if (k == Kind::Person) {
            a.role = Local;
            if (home_nodes > 0) { const int nd = rng.pick(home_nodes); a.pos = nodes[size_t(nd)].p; a.at_node = nd; }
        } else {
            a.cs = Cruise; a.pose = Pose::Drive; a.dir = rng.chance(0.5f) ? 1 : -1;
            a.s = rng.range(0.0f, std::max(road.length, 1.0f)); make_driver(a, police ? int(Police) : speeder ? int(Speeder) : -1); a.speed = a.v_want;
            // (not on top of the site, nor of another car)
            for (int tries = 0; tries < 8; ++tries) {
                bool clear = !(site && std::abs(wrap_s(a.s - s_in + 0.5f * road.length) - 0.5f * road.length) < 400.0f);
                for (const Agent& o : agents) if (&o != &a && o.kind == Kind::Car && o.gone <= 0.0f && o.cs == Cruise && std::abs(wrap_s(o.s - a.s + 0.5f * road.length) - 0.5f * road.length) < 40.0f) clear = false;
                if (clear) break;
                a.s = rng.range(0.0f, std::max(road.length, 1.0f));
            }
            car_on_road(a);
        }
    }
    void step_person(Agent& a, float dt) {
        if (a.asleep) return;                  // (of an added site nobody is near: not there)
        if (a.zone && a.gone > 0.0f) {         // (shot: after a while somebody new is at that place)
            a.gone -= dt;
            if (a.gone <= 0.0f) { a.look = dressed(uint8_t(zrng.pick(256)), a.role); a.pos.y += 20000.0f; zone_start(a); }
            return;
        }
        if (a.gone > 0.0f) { a.gone -= dt; if (a.gone <= 0.0f) respawn(a); return; }
        if (a.leave > 0.0f) {                  // (put out of a car out on the road - put_out: there is nowhere to walk to)
            a.leave -= dt; a.pose = Pose::Stand; a.speed = 0.0f;
            if (a.leave <= 0.0f) { a.hidden = true; a.gone = 45.0f; a.pos.y -= 20000.0f; }
            return;
        }
        if (a.role == Runner) { step_runner(a, dt); return; }
        if (a.hidden) { if (a.car >= 0) { a.pos = agents[size_t(a.car)].pos; a.yaw = agents[size_t(a.car)].yaw; } return; }
        if (a.tasks.empty()) think(a);
        if (a.tasks.empty()) return;
        Task& t = a.tasks.back();
        switch (t.type) {
        case GoNode: {
            if (a.route.empty()) {
                if (a.at_node < 0) a.at_node = nearest_node(a.pos);
                if (a.at_node == t.a || t.a < 0) { a.tasks.pop_back(); break; }
                a.route = route(a.at_node, t.a);
                if (a.route.empty()) { a.tasks.pop_back(); break; }      // (no way there: give it up)
                // (not standing on the graph: first to the node it is nearest)
                if (glm::distance(a.pos, nodes[size_t(a.at_node)].p) > 0.5f) a.route.push_back(a.at_node);
            }
            bool arrived = false;
            walk_toward(a, nodes[size_t(a.route.back())].p, dt, arrived, 0.4f);
            if (arrived) { a.at_node = a.route.back(); a.route.pop_back(); if (a.route.empty()) a.tasks.pop_back(); }
            break; }
        case GoPoint: {
            bool arrived = false;
            walk_toward(a, t.p, dt, arrived, 0.1f);
            if (arrived) a.tasks.pop_back();
            break; }
        case Sit: {
            // Sitting down: round to the way the seat faces, then the act of sitting down (the
            // renderer plays it: it stands where it stood and the body goes back and down on
            // to the seat), then it sits. Getting up: the act of standing up, back at the place
            // it sat down from.
            if (a.pose != Pose::Sit && a.act != uint8_t(Act::StandUp) && a.act != uint8_t(Act::PickUp)) {
                a.walk_v = 0.0f; a.speed = 0.0f; a.pose = Pose::Stand;
                if (a.act == 0 || a.act == uint8_t(Act::PutDown)) {
                    if (a.act == 0 && !turn_to(a, t.yaw, dt)) break;
                    // (what it carries first goes on the table, the desk or the bench beside it)
                    if (a.act == 0 && a.item) { a.act = uint8_t(Act::PutDown); a.act_t = 0.0f; a.aim = seat_place(t.a, t.p, t.yaw); }
                    if (a.act == uint8_t(Act::PutDown)) {
                        a.yaw = t.yaw; a.act_t += dt;
                        if (a.act_t < kActTime[int(Act::PutDown)]) break;
                        a.placed = a.item; a.item = 0;
                    }
                    a.act = uint8_t(Act::SitDown); a.act_t = 0.0f; a.from = a.pos; a.aim = t.p;
                }
                a.yaw = t.yaw;
                a.act_t += dt;
                if (a.act_t < kActTime[int(Act::SitDown)]) break;
                a.act = 0; a.act_t = 0.0f;
                a.pose = Pose::Sit; a.pos = t.p;
            }
            // (how it sits: at the bar on a stool; at a table eating or talking; plain on a bench)
            if (a.pose == Pose::Sit && t.a >= 0) {
                const std::string& k = seats[size_t(t.a)].kind;
                a.style = k == "stool" ? 1 : (k == "booth" || k == "chair" || k == "picnic") ? uint8_t(2 + ((a.look >> 3) & 1)) : k == "desk" ? 4 : k == "meet" ? 3 : 0;
                // (at a desk: where its keyboard is - the renderer puts the hands there)
                if (a.act == 0 && seats[size_t(t.a)].has_aim) a.aim = seats[size_t(t.a)].aim;
            }
            if (a.act == uint8_t(Act::PickUp)) {                   // (up, and what it set down taken again)
                a.pose = Pose::Stand; a.pos = a.from; a.yaw = t.yaw; a.speed = 0.0f;
                a.act_t += dt;
                if (a.act_t < kActTime[int(Act::PickUp)]) break;
                a.act = 0; a.act_t = 0.0f; a.item = a.placed; a.placed = 0;
                if (a.seat >= 0 && seats[size_t(a.seat)].user == int(&a - agents.data())) seats[size_t(a.seat)].user = -1;
                a.seat = -1;
                a.tasks.pop_back();
                if (a.tasks.empty()) release(a);
                break;
            }
            if (a.act == uint8_t(Act::StandUp)) {
                a.pose = Pose::Stand; a.pos = a.from; a.yaw = t.yaw; a.speed = 0.0f;
                a.act_t += dt;
                if (a.act_t < kActTime[int(Act::StandUp)]) break;
                a.act = 0; a.act_t = 0.0f;
                if (a.placed && (a.zone ? zrng : rng).chance(0.75f)) { a.act = uint8_t(Act::PickUp); a.act_t = 0.0f; a.aim = seat_place(t.a, t.p, t.yaw); break; }
                a.placed = 0;
                if (a.seat >= 0 && seats[size_t(a.seat)].user == int(&a - agents.data())) seats[size_t(a.seat)].user = -1;
                a.seat = -1;
                a.tasks.pop_back();
                if (a.tasks.empty()) release(a);
                break;
            }
            a.pos = t.p; a.yaw = t.yaw; a.speed = 0.0f;
            t.time -= dt;
            if (t.time <= 0.0f) {
                // (where it gets up to: where it sat down from - or, if it began the world seated, in front of the seat)
                if (t.a >= 0 && glm::distance(a.from, t.p) > 1.6f) {
                    const glm::vec3 np = nodes[size_t(std::max(seats[size_t(t.a)].node, 0))].p;
                    a.from = seat_stand_point(t.a); a.from.y = np.y;
                }
                if (t.a >= 0) a.at_node = seats[size_t(t.a)].node;
                a.act = uint8_t(Act::StandUp); a.act_t = 0.0f; a.aim = t.p;
            }
            break; }
        case DoAct: {
            // (getting out of a car, or into one: it stands by the door meanwhile)
            if (a.act != uint8_t(t.a)) { a.act = uint8_t(t.a); a.act_t = 0.0f; a.aim = t.p; }
            a.pose = Pose::Stand; a.walk_v = 0.0f; a.speed = 0.0f; a.yaw = t.yaw;
            a.act_t += dt;
            if (a.act_t >= kActTime[t.a]) { if (t.a == int(Act::PutDown)) a.item = 0; a.act = 0; a.act_t = 0.0f; a.tasks.pop_back(); }
            break; }
        case Stand: {
            // WHAT PEOPLE CARRY (npc.h Item): somebody busy at a machine sets the toolbox down
            // beside it first and takes it up again when done (the acts; the renderer plays them).
            if (a.act == uint8_t(Act::PutDown) || a.act == uint8_t(Act::PickUp)) {
                a.pose = Pose::Stand; a.speed = a.walk_v = 0.0f;
                a.act_t += dt;
                if (a.act_t < kActTime[a.act]) break;
                if (a.act == uint8_t(Act::PutDown)) { a.placed = a.item; a.item = 0; a.act = 0; a.act_t = 0.0f; t.step = 1; }
                else { a.item = a.placed; a.placed = 0; a.act = 0; a.act_t = 0.0f; a.tasks.pop_back(); if (a.tasks.empty()) release(a); break; }
            }
            if (t.step == 0 && t.pose == Pose::Work && (a.item == uint8_t(Item::Toolbox) || a.item == uint8_t(Item::Box))) {
                if (!turn_to(a, t.yaw, dt)) { a.pose = Pose::Stand; a.speed = a.walk_v = 0.0f; break; }
                a.act = uint8_t(Act::PutDown); a.act_t = 0.0f; a.aim = item_place(a.pos, t.yaw, 0); t.step = 1;
                a.pose = Pose::Stand; break;
            }
            a.pose = t.pose; a.style = t.style;
            if (t.has_aim) a.aim = t.aim;      // (what is pointed or looked at from here)
            turn_to(a, t.yaw, dt);
            a.walk_v = 0.0f; a.speed = 0.0f;
            t.time -= dt;
            if (t.time <= 0.0f) {
                if (a.placed && t.pose == Pose::Work) { a.act = uint8_t(Act::PickUp); a.act_t = 0.0f; a.aim = item_place(a.pos, t.yaw, 0); a.pose = Pose::Stand; break; }
                fetched(a, t);
                a.tasks.pop_back();
                if (a.tasks.empty()) release(a);
            }
            break; }
        case EnterCar: {
            a.tasks.pop_back();
            if (a.car >= 0) { a.hidden = true; a.pose = Pose::Stand; a.speed = 0.0f; a.item = a.placed = 0; release(a); --agents[size_t(a.car)].out; }
            break; }
        }
    }

    // ---------------------------------------------------------------- cars
    // HOW A CAR MOVES. It is a car, not a point on a line: it has a rear axle, a wheelbase and
    // front wheels that steer, and each step it goes where its wheels take it (the bicycle
    // model). It steers at a point of its line some way ahead (pure pursuit), the wheel turns
    // no faster than a driver turns it, the speed changes no faster than an engine and brakes
    // do, and it is no faster through a bend than the sideways pull allows. So it swings into
    // the station from its lane, straightens up at the pump, backs out of a bay on an arc and
    // pulls out on to the road - and nowhere is it put somewhere: it only ever drives there.
    // (Until 2026-11-25 it was a point slid along the line with its nose eased round: it
    // jumped two metres sideways where the way in left the lane, and turned on the spot.)
    static constexpr float kWheelbase = 2.7f, kMaxSteer = 0.60f, kSteerRate = 1.5f;
    static constexpr float kAccel = 1.8f, kBrake = 3.2f, kSideways = 2.6f;
    float wrap_s(float s) const { if (road.length <= 0.0f) return 0.0f; s = std::fmod(s, road.length); return s < 0.0f ? s + road.length : s; }
    // The middle of the car's lane `s` along the road, and the way it runs for this car.
    // (`lat`: so far right of the centre line instead - a car that passes another drives on the left)
    bool lane_at(const Agent& c, float s, glm::vec3& p, glm::vec2& fwd, float lat = kLane) const {
        glm::vec2 d;
        if (!road.at || !road.at(wrap_s(s), p, d)) return false;
        fwd = d * float(c.dir);
        p += glm::vec3(-fwd.y, 0.0f, fwd.x) * lat;
        return true;
    }

    // ---------------------------------------------------------------- THE TRAFFIC
    // Every driver is somebody (make_driver): a slow one in a loaded pickup, most at about the
    // same pace, some in a hurry, and now and then one who drives twice as fast as the rest.
    // The police patrol at the common pace and sometimes put the siren on and go.
    //
    // Each step everything on the road is put in one list (build_things): this world's cars
    // and the players' (set_obstacles) alike, each with its place along the road, how far right
    // of the centre line it is, and its speed along the road. From that list a car
    //   FOLLOWS   whatever is ahead in the strip of road it drives on: it comes no nearer than
    //             its driver's time gap, and is never faster than it can still brake from
    //             (2.4 m/s2 planned, 7 when it must). What stands - a car left on the road -
    //             it stops well short of.
    //   PASSES    when it has been held up for longer than its driver's patience: the cars to
    //             get past are counted (up to three nose to tail), the time that takes at the
    //             speed it would pass at is worked out, and it pulls out only if nothing comes
    //             the other way within that time and a margin, nobody is beside or coming up
    //             behind in the other lane, the road ahead is straight enough to have seen all
    //             that, and it is not by the station's ways in and out. Out, it drives on the
    //             left until it is clear of the last of them and the lane has room, and pulls
    //             back in.
    //   GIVES UP  when, out there, somebody comes after all: it drops back in behind if it is
    //             not yet alongside, pushes on if it is all but past, and otherwise brakes
    //             until there is room behind the car it is beside. Whoever meets a car in his
    //             own lane brakes too, and keeps right.
    //   MAKES WAY for a siren: to the edge of the road and slower, so that the police car can
    //             go down the middle.
    static constexpr float kShoulder = 3.15f;     // how far right of the centre line a car goes to make way
    float sdiff(float a, float b) const {         // a - b along the road, the short way round
        if (road.length <= 0.0f) return 0.0f;
        float d = std::fmod(a - b, road.length);
        if (d > 0.5f * road.length) d -= road.length;
        if (d < -0.5f * road.length) d += road.length;
        return d;
    }
    void make_driver(Agent& c, int cls = -1) {
        if (cls < 0) { const float r = rng.uni(); cls = r < 0.16f ? Slow : r < 0.74f ? Normal : r < 0.965f ? Fast : Speeder; }
        c.cls = uint8_t(cls);
        int kind = int(c.look % 3);               // (the renderer's: 0 a sports car, 1 a saloon, 2 a pickup)
        switch (cls) {
        case Slow:    c.v_want = kCruise * rng.range(0.60f, 0.74f); c.accel = 1.2f; c.side = 2.2f; c.gap_t = rng.range(1.5f, 2.0f); c.patience = rng.range(6.0f, 12.0f); kind = 2; break;
        case Fast:    c.v_want = kCruise * rng.range(1.14f, 1.34f); c.accel = 2.4f; c.side = 3.4f; c.gap_t = rng.range(0.9f, 1.2f); c.patience = rng.range(1.0f, 2.5f); break;
        case Speeder: c.v_want = kCruise * rng.range(1.75f, 2.0f);  c.accel = 3.6f; c.side = 6.0f; c.gap_t = rng.range(0.6f, 0.8f); c.patience = rng.range(0.3f, 0.8f); kind = 0; break;
        case Police:  c.v_want = kCruise * rng.range(0.92f, 1.0f);  c.accel = 2.2f; c.side = 3.0f; c.gap_t = 1.3f; c.patience = 3.0f; c.siren_t = rng.range(20.0f, 70.0f); kind = 1; break;
        default:      c.v_want = kCruise * rng.range(0.86f, 1.08f); c.accel = 1.8f; c.side = 2.6f; c.gap_t = rng.range(1.2f, 1.6f); c.patience = rng.range(2.5f, 6.0f); break;
        }
        c.look = uint8_t(std::min(int(c.look), 251) / 3 * 3 + kind);
        c.lat = c.lat_cmd = kLane;
    }
    // Where along the road a point is, and how far right of the centre line (for increasing s).
    // `hint`: near there if it is given (>= 0), else the whole road is looked through.
    bool locate(glm::vec3 p, float hint, float& s_out_, float& goff, float& dist) const {
        if (!road.at || road.length <= 0.0f) return false;
        float best = 0.0f, bd = 1e30f;
        auto probe = [&](float s) {
            glm::vec3 q; glm::vec2 d;
            if (!road.at(wrap_s(s), q, d)) return;
            const float dd = glm::distance(glm::vec2(q.x, q.z), glm::vec2(p.x, p.z));
            if (dd < bd) { bd = dd; best = s; }
        };
        if (hint >= 0.0f) for (float s = hint - 48.0f; s <= hint + 48.0f; s += 6.0f) probe(s);
        if (hint < 0.0f || bd > 40.0f) for (float s = 0.0f; s < road.length; s += 10.0f) probe(s);
        const float mid = best;
        for (float s = mid - 5.0f; s <= mid + 5.0f; s += 1.0f) probe(s);
        glm::vec3 q; glm::vec2 d;
        if (!road.at(wrap_s(best), q, d)) return false;
        const glm::vec2 rel(p.x - q.x, p.z - q.z);
        s_out_ = wrap_s(best + glm::dot(rel, d));
        goff = glm::dot(rel, glm::vec2(-d.y, d.x));
        dist = bd;
        return true;
    }
    void build_things() {
        things.clear();
        for (size_t i = 0; i < agents.size(); ++i) {
            const Agent& a = agents[i];
            if (a.kind != Kind::Car || a.gone > 0.0f || (a.cs != Cruise && a.cs != Approach)) continue;
            things.push_back({ a.s, a.lat * float(a.dir), a.speed * float(a.dir), kCarHalfL, kCarHalfW, int(i), a.siren });
        }
        // The players' cars. Finding one on a long road costs a look along all of it, so where
        // each was found is kept: one that has moved is looked for near there, one that stands
        // away from the road is not looked for again until it moves.
        std::map<uint32_t, ObsAt> seen;
        for (const Obstacle& o : obstacles) {
            ObsAt at; bool have = false;
            if (auto it = obs_at.find(o.id); it != obs_at.end()) { at = it->second; have = true; }
            const float moved = have ? glm::distance(at.pos, o.pos) : 1e9f;
            if (!have || moved > 60.0f || (at.near_road ? moved > 0.02f : moved > 3.0f)) {
                float s = 0.0f, g = 0.0f, d = 1e9f;
                if (!locate(o.pos, have && at.near_road && moved <= 60.0f ? at.s : -1.0f, s, g, d)) continue;
                at.pos = o.pos; at.s = s; at.goff = g; at.near_road = d < 30.0f;
            }
            seen[o.id] = at;
            if (!at.near_road || std::abs(at.goff) > 7.5f) continue;
            glm::vec3 q; glm::vec2 d;
            if (!road.at(wrap_s(at.s), q, d)) continue;
            // (as it lies on the road: across it a car is 4.7 m wide and 2 m long)
            const glm::vec2 f(std::sin(o.yaw), -std::cos(o.yaw));
            const float ca = std::abs(glm::dot(f, d)), sa = std::sqrt(std::max(0.0f, 1.0f - ca * ca));
            things.push_back({ at.s, at.goff, glm::dot(glm::vec2(o.vel.x, o.vel.z), d), ca * o.half_l + sa * o.half_w, ca * o.half_w + sa * o.half_l, 1000 + int(o.id % 100000u), false });
        }
        obs_at.swap(seen);
    }
    // What one of the things is to this car: metres ahead of it (centre to centre; behind: < 0),
    // metres right of the centre line and speed, both the way this car drives.
    struct Rel { float gc, lat, v, gap; };
    Rel rel_of(const Agent& c, const Thing& t) const {
        Rel r;
        r.gc = sdiff(t.s, c.s) * float(c.dir);
        r.lat = t.goff * float(c.dir); r.v = t.v * float(c.dir);
        r.gap = std::abs(r.gc) - t.hl - kCarHalfL;                 // bumper to bumper
        return r;
    }
    // How long it takes to gain `D` metres on something it is `rel0` m/s faster than now, speeding
    // up at `a` until it is `dv` m/s faster.
    static float pass_time(float rel0, float D, float dv, float a) {
        if (D <= 0.0f) return 0.0f;
        const float t_a = std::max(dv - rel0, 0.0f) / a;
        const float gained = rel0 * t_a + 0.5f * a * t_a * t_a;
        if (gained >= D) return (-rel0 + std::sqrt(std::max(rel0 * rel0 + 2.0f * a * D, 0.0f))) / a;
        return t_a + (D - gained) / std::max(dv, 0.5f);
    }
    // May this car pull out now to get past what holds it up? `want`: the speed it would
    // drive if it could. Gives whom it has to get past (`key`) and where it would drive (`lat_p`).
    bool can_pass(const Agent& c, float want, int& key, float& lat_p) const {
        const int me = int(&c - agents.data());
        const float me_v = std::abs(c.speed);
        // (not by the station: cars turn in and pull out there)
        if (site && sdiff(c.s, s_in) > -230.0f && sdiff(c.s, s_out) < 230.0f) return false;
        // ---- the cars to get past: the one in front and whoever is nose to tail with it
        const float w0 = kCarHalfW + 0.35f;
        const Thing* line[4] = { nullptr, nullptr, nullptr, nullptr }; Rel lr[4]{}; int n = 0;
        for (;;) {
            const Thing* next = nullptr; Rel nr{};
            for (const Thing& t : things) {
                if (t.key == me) continue;
                const Rel r = rel_of(c, t);
                if (r.gc <= 0.0f || r.v < -0.5f || std::abs(r.lat - kLane) >= t.hw + w0) continue;
                if (n > 0 && r.gc <= lr[n - 1].gc) continue;
                if (!next || r.gc < nr.gc) { next = &t; nr = r; }
            }
            if (!next) break;
            if (n > 0 && nr.gc - lr[n - 1].gc - next->hl - line[n - 1]->hl > 20.0f + 0.9f * std::max(nr.v, 0.0f)) break;      // (room to pull in behind that one)
            if (n == 3) return false;                              // (a longer queue than can be passed in one go)
            line[n] = next; lr[n] = nr; ++n;
        }
        if (n == 0) return false;
        float v_ref = 0.0f, lat_min = 1e9f, hw_max = 0.0f;
        for (int i = 0; i < n; ++i) { v_ref = std::max(v_ref, lr[i].v); lat_min = std::min(lat_min, lr[i].lat); hw_max = std::max(hw_max, line[i]->hw); }
        const float v_pass = std::min(want * 1.10f + 1.0f, want + 5.0f);
        const float dv = v_pass - v_ref;
        if (dv < 2.5f) return false;
        // ---- how long it takes: up to the passing speed, then at it, until clear of the last
        const float D = lr[n - 1].gc + line[n - 1]->hl + kCarHalfL + 6.0f + 0.5f * v_ref;
        const float a = c.siren ? 3.4f : std::max(c.accel, 2.6f);
        const float t_a = std::max(v_pass - me_v, 0.0f) / a;
        float secs = pass_time(std::min(me_v - v_ref, dv), D, dv, a);
        secs += 3.0f;                                              // (out, and back in)
        if (secs > 18.0f) return false;
        const float travel = 0.5f * (me_v + v_pass) * std::min(secs, t_a) + v_pass * std::max(secs - t_a, 0.0f);
        // ---- a straight enough road to have seen it all
        {
            glm::vec3 p0, p1; glm::vec2 d0, d1;
            if (!road.at(wrap_s(c.s), p0, d0) || !road.at(wrap_s(c.s + travel * float(c.dir)), p1, d1)) return false;
            if (glm::dot(d0, d1) < 0.90f) return false;
        }
        // ---- where it drives meanwhile: the other lane - past something that stands, only as
        //      far out as it must; with the siren on, down the middle if they have made way
        const float need = lat_min - (hw_max + kCarHalfW + 0.85f);
        lat_p = v_ref < 1.0f ? need : c.siren ? std::max(need, -kLane) : std::min(need, -kLane);
        if (lat_p < -3.4f) return false;
        // ---- and that strip free for as long as it takes
        for (const Thing& t : things) {
            if (t.key == me) continue;
            const Rel r = rel_of(c, t);
            if (std::abs(r.lat - lat_p) >= t.hw + w0) continue;
            if (r.v < -0.5f) {                                     // coming the other way
                if (r.gc > 0.0f ? r.gap < (v_pass - r.v) * (secs + 3.5f) : r.gc > -12.0f) return false;
            } else if (r.gc > 0.0f) {                              // in that lane ahead, going my way or standing
                if (r.gap < (r.v < 1.0f ? travel + 30.0f : D + 40.0f)) return false;
            } else if (r.gc > -(10.0f + std::max(r.v - me_v, 0.0f) * 5.0f + std::max(r.v - me_v, 0.0f) * std::max(r.v - me_v, 0.0f) / 5.0f)) return false;      // beside, or coming up behind
        }
        key = line[n - 1]->key;
        return true;
    }
    // One step of a car on the road among the others: the speed it may drive, and (lat_want)
    // how far right of the centre line it wants to be.
    float traffic(Agent& c, float dt, float& lat_want) {
        const int me = int(&c - agents.data());
        const float me_v = std::abs(c.speed);
        // (the police: the siren goes on and off)
        if (c.cls == Police && !c.driverless) {
            c.siren_t -= dt;
            if (c.siren_t <= 0.0f) { c.siren = !c.siren; c.siren_t = c.siren ? rng.range(45.0f, 100.0f) : rng.range(60.0f, 170.0f); }
        } else if (c.driverless) c.siren = false;
        const float want = c.siren ? kCruise * 1.75f : c.v_want;
        const float v_pass = std::min(want * 1.10f + 1.0f, want + 5.0f);
        float v = c.ov == 1 ? v_pass : want;
        const float w0 = kCarHalfW + 0.35f;
        // ---- what is ahead in the strip it drives on (from where it is to where it steers)
        const Thing* lead = nullptr; Rel lead_r{};                 // the nearest ahead in its own lane, going its way or standing
        const Thing* target = nullptr; Rel target_r{};             // whom it is getting past (ov_key)
        bool home_clear = true, danger = false, make_way = false, meets = false;
        float beside_v = 1e9f;
        const float range = 70.0f + me_v * me_v / (2.0f * 2.4f);
        for (const Thing& t : things) {
            if (t.key == me) continue;
            const Rel r = rel_of(c, t);
            if (c.ov && t.key == c.ov_key) { target = &t; target_r = r; }
            const float w = t.hw + w0;
            // (a siren: behind and going my way, or coming toward me)
            if (t.siren && !c.siren && ((r.v > 0.5f && r.gc < 0.0f && r.gc > -160.0f) || (r.v < -0.5f && r.gc > 0.0f && r.gc < 260.0f))) make_way = true;
            // (its own lane, about where it is: is there room to be in it?)
            if (std::abs(r.lat - kLane) < w) {
                // (room for whoever comes up behind to slow to my speed, room for me to slow to his in front)
                const float faster = std::max(r.v - me_v, 0.0f), slower = std::max(me_v - r.v, 0.0f);
                if (r.gc > -(t.hl + kCarHalfL + 4.0f + 0.4f * std::max(r.v, 0.0f) + faster * faster / 5.0f) && r.gc < t.hl + kCarHalfL + 3.0f + slower + slower * slower / 5.0f) { home_clear = false; beside_v = std::min(beside_v, r.v); }
                if (r.gc > 0.0f && r.v >= -0.5f && (!lead || r.gc < lead_r.gc)) { lead = &t; lead_r = r; }
            }
            if (r.gc <= 0.0f || r.gap > range) continue;
            // (where it will be across the road by the time it gets there)
            const float reach = c.ov ? std::clamp(0.35f * std::max(r.gap, 0.0f), 0.0f, std::abs(c.lat_cmd - c.lat)) : 0.0f;
            const float lat_then = c.lat + (c.lat_cmd > c.lat ? reach : -reach);
            if (r.lat < std::min(lat_then, c.lat_cmd) - w || r.lat > std::max(lat_then, c.lat_cmd) + w) continue;
            if (r.v >= -0.5f) {
                // going my way, or standing: no nearer than the time gap, never too fast to stop
                const float room = r.gap - (r.v < 1.0f ? 8.0f : 2.5f);
                float vl = room > 0.0f ? std::sqrt(r.v * r.v + 2.0f * 2.4f * room) : 0.0f;
                if (r.gap < 12.0f + 2.5f * me_v) vl = std::min(vl, std::max(0.0f, r.v + (room - (c.ov ? 0.5f : c.gap_t) * std::max(r.v, 0.0f)) * 0.5f));
                v = std::min(v, vl);
            } else {
                // coming at me in the strip I drive on
                const float ttc = r.gap / std::max(me_v - r.v, 0.1f);
                if (ttc < 5.0f) meets = true;
                if (ttc < 3.2f) v = 0.0f;
            }
        }
        c.yield = make_way ? 2.5f : std::max(0.0f, c.yield - dt);
        c.ov_cool = std::max(0.0f, c.ov_cool - dt);
        const bool may = c.cs == Cruise && !c.driverless && c.shaken <= 0.0f;
        if (c.ov == 0) {
            // ---- held up? then, patience gone, out if the way is free
            const bool slow = lead && lead_r.v < want - std::max(2.5f, 0.12f * want) &&
                              lead_r.gap < 20.0f + 1.6f * me_v + std::max(me_v * me_v - lead_r.v * lead_r.v, 0.0f) / 4.8f;      // (near enough to be braking for it)
            c.held = slow ? c.held + dt : std::max(0.0f, c.held - 2.0f * dt);
            int key = -1; float lat_p = -kLane;
            if (may && slow && c.held > (c.siren ? 0.3f : c.patience) && c.ov_cool <= 0.0f && can_pass(c, want, key, lat_p)) {
                c.ov = 1; c.ov_key = key; c.ov_lat = lat_p; c.ov_t = 0.0f; c.held = 0.0f;
                ++n_overtakes;
            }
        } else if (c.ov == 1) {
            c.ov_t += dt;
            const bool past = !target || target_r.gc < -(target->hl + kCarHalfL + 5.0f + 0.45f * std::max(target_r.v, 0.0f));
            // ---- does somebody come after all? (in the strip it passes on, within the time it still needs)
            const float d_rem = target ? target_r.gc + target->hl + kCarHalfL + 5.0f + 0.45f * std::max(target_r.v, 0.0f) : 0.0f;
            const float v_t = target ? std::max(target_r.v, 0.0f) : 0.0f;
            const float t_rem = pass_time(std::min(me_v - v_t, v_pass - v_t), d_rem, std::max(v_pass - v_t, 1.0f), c.siren ? 3.4f : std::max(c.accel, 2.6f)) + 1.5f;
            if (!past)
                for (const Thing& t : things) {
                    if (t.key == me) continue;
                    const Rel r = rel_of(c, t);
                    if (r.gc > 0.0f && r.v < -0.5f && std::abs(r.lat - c.ov_lat) < t.hw + w0 && r.gap < (me_v - r.v) * (t_rem + 1.0f)) danger = true;
                }
            if ((past || danger || c.ov_t > 45.0f || !may) && home_clear) {
                c.ov = 2;
                if (!past) ++n_given_up;
            } else if (danger && d_rem > 10.0f && beside_v < 1e8f) {
                v = std::min(v, std::max(0.0f, beside_v - 5.0f));  // (beside somebody, and no time to get past: back behind him)
            }
        }
        if (c.ov == 2) {
            if (std::abs(c.lat - kLane) < 0.45f) { c.ov = 0; c.ov_cool = 3.0f; c.held = 0.0f; }
            // (the room it was pulling in to has gone - somebody braked, somebody came up: out
            //  again while it is still mostly out, and back in when there is room)
            else if (!home_clear && c.lat_cmd < 0.4f) c.ov = 1;
        }
        lat_want = c.ov == 1 ? c.ov_lat : kLane;
        if (c.ov == 0 && !c.siren) {
            if (c.yield > 0.0f) { lat_want = kShoulder; v = std::min(v, want * 0.6f); }
            else if (meets) lat_want = kShoulder;
        }
        return v;
    }
    void car_on_road(Agent& c) {                   // (put there: only when the world is made)
        glm::vec3 p; glm::vec2 fwd;
        if (!lane_at(c, c.s, p, fwd)) return;
        c.pos = p; c.yaw = yaw_of(fwd); c.steer = 0.0f;
        c.rear = p - glm::vec3(fwd.x, 0.0f, fwd.y) * (kWheelbase * 0.5f);
    }
    // One step: steer at `target`, come to `v_want` (m/s, never negative; `reverse` says which way).
    // `straighten`: added to the wheel's angle (+ = to the right) - coming to rest on a line, a
    // driver also turns the car to lie along it, not only to reach it.
    void drive(Agent& c, glm::vec3 target, float v_want, float dt, float straighten = 0.0f) {
        const glm::vec2 right(std::cos(c.yaw), std::sin(c.yaw));
        const glm::vec2 to(target.x - c.rear.x, target.z - c.rear.z);
        float kappa = 2.0f * glm::dot(to, right) / std::max(glm::dot(to, to), 0.25f);             // of the circle through the target; + = to the right
        {   // (The target behind it - it overshot a turn it could not make: full lock toward
            //  it, at a crawl. Without this the circle through a point behind is nearly a
            //  straight line, and the car drove on across the country.)
            const glm::vec2 fw(std::sin(c.yaw), -std::cos(c.yaw));
            if (glm::dot(to, fw) * (c.reverse ? -1.0f : 1.0f) < 0.0f) {
                kappa = (glm::dot(to, right) >= 0.0f ? 1.0f : -1.0f) * std::tan(kMaxSteer) / kWheelbase * (c.reverse ? -1.0f : 1.0f);
                v_want = std::min(v_want, 1.6f);
            }
        }
        // (each driver his own: how hard he accelerates and how fast he takes a bend - make_driver)
        const float accel = c.siren ? 3.4f : c.ov == 1 ? std::max(c.accel, 2.6f) : c.accel;
        const float side = c.siren ? std::max(c.side, 5.0f) : c.side;
        const float brake = c.hard ? 7.0f : kBrake;
        if (std::abs(kappa) > 1e-3f) v_want = std::min(v_want, std::sqrt(side / std::abs(kappa)));
        float sp = std::abs(c.speed);
        sp = std::max(0.0f, sp + std::clamp(v_want - sp, -brake * dt, accel * dt));
        const float steer_want = std::clamp(std::atan(kWheelbase * kappa) + straighten, -kMaxSteer, kMaxSteer);
        c.steer += std::clamp(steer_want - c.steer, -kSteerRate * dt, kSteerRate * dt);
        const float v = c.reverse ? -sp : sp;
        c.yaw = wrap_pi(c.yaw + v / kWheelbase * std::tan(c.steer) * dt);
        const glm::vec2 f(std::sin(c.yaw), -std::cos(c.yaw));
        c.rear.x += f.x * v * dt; c.rear.z += f.y * v * dt;
        c.speed = v;
        c.pos.x = c.rear.x + f.x * kWheelbase * 0.5f; c.pos.z = c.rear.z + f.y * kWheelbase * 0.5f;
    }
    // A point `d` metres along a line; past its end, straight on the way it ended.
    glm::vec3 line_at(const std::vector<glm::vec3>& p, float d, float len) const {
        glm::vec3 dir;
        if (d <= len) return poly_at(p, d, dir);
        const glm::vec3 e = poly_at(p, len, dir);
        return e + dir * (d - len);
    }
    void begin_leg(Agent& c) {
        c.path = c.legs.back(); c.reverse = c.legs_rev.back() != 0;
        c.legs.pop_back(); c.legs_rev.pop_back();
        c.path_len = poly_len(c.path); c.along = 0.0f; c.cs = OnPath;
        if (c.reverse != (c.speed < 0.0f)) c.speed = 0.0f;
    }
    void push_leg(Agent& c, std::vector<glm::vec3> pts, bool rev = false) { c.legs.insert(c.legs.begin(), smooth(std::move(pts))); c.legs_rev.insert(c.legs_rev.begin(), rev ? 1 : 0); }
    // The curve from the aisle into a bay (nose in), as driven forward.
    std::vector<glm::vec3> bay_curve(const Place& b, const std::vector<glm::vec3>& aisle) const {
        const float ax = aisle.empty() ? b.p.x + 4.5f : aisle.front().x;
        const float sgn = aisle.size() > 1 && aisle.back().z < aisle.front().z ? -1.0f : 1.0f;     // the way the aisle is driven
        // (As a driver does it: out to the far side of the aisle first, then round and in, the
        //  last metre and a half straight. Turned straight off the aisle's line a car cannot
        //  come round in the room there is, and stood 35 degrees askew in its bay.)
        // A swing out, a quarter circle of 4.6 m (a car turns in 4), and the rest straight in.
        const float R = 4.6f, xs = ax + 1.8f, y = b.p.y, z = b.p.z;
        std::vector<glm::vec3> p = { glm::vec3(ax, y, z - sgn * (R + 4.1f)), glm::vec3(ax + 0.35f, y, z - sgn * (R + 2.7f)), glm::vec3(ax + 1.2f, y, z - sgn * (R + 1.2f)) };
        for (float a = 0.0f; a <= 90.1f; a += 15.0f) {
            const float r = a * 0.0174533f;
            p.push_back(glm::vec3(xs - R * (1.0f - std::cos(r)), y, z - sgn * (R - R * std::sin(r))));
        }
        if (xs - R > b.p.x + 0.3f) p.push_back(glm::vec3(0.5f * (xs - R + b.p.x), y, z));
        p.push_back(b.p);
        return p;
    }
    void plan_visit(Agent& c) {
        const int me = int(&c - agents.data());
        c.legs.clear(); c.legs_rev.clear(); c.bay = c.stop = -1;
        const bool want_pump = rng.chance(0.6f);
        int stop = -1;
        if (want_pump) {
            // The far stop of a lane first, so that a second car can pull up behind - but only
            // while the near one is free: the way to the far stop runs through it. (A car that
            // took the far stop while one stood at the near stop never got there, and the one
            // at the near stop waited for the far stop to clear: both stood for good.)
            auto taken = [&](const std::string& nm) { for (const Place& p : stops) if (p.kind == nm) return p.user >= 0; return true; };
            for (const char* lane : { "13", "20" }) {
                const std::string a = std::string("pump") + lane + "a", b = std::string("pump") + lane + "b";
                const std::string pickn = !taken(a) && !taken(b) ? b : !taken(a) ? a : std::string();
                if (pickn.empty()) continue;
                for (size_t i = 0; i < stops.size(); ++i) if (stops[i].kind == pickn) { stop = int(i); break; }
                if (stop >= 0) break;
            }
        }
        if (stop >= 0) {
            const std::string lane = stops[size_t(stop)].kind.substr(4, 2);
            const bool far = stops[size_t(stop)].kind.back() == 'b';
            // (the stop in front must be free to drive through to the far one - and to leave by)
            std::vector<glm::vec3> in = paths["in" + lane];
            if (far) { const auto& mid = paths["mid" + lane]; in.insert(in.end(), mid.begin() + 1, mid.end()); }
            std::vector<glm::vec3> out;
            if (!far) { const auto& mid = paths["mid" + lane]; out.insert(out.end(), mid.begin(), mid.end() - 1); }
            const auto& o = paths["out" + lane]; out.insert(out.end(), o.begin(), o.end());
            push_leg(c, in); push_leg(c, out);
            stops[size_t(stop)].user = me; c.stop = stop;
        } else {
            std::vector<int> free;
            const auto& aisle = paths["aisle"];
            for (size_t i = 0; i < bays.size(); ++i)
                if (bays[i].user < 0 && !aisle.empty() && bays[i].p.z > std::min(aisle.front().z, aisle.back().z) + kBayRunUp && bays[i].p.z < std::max(aisle.front().z, aisle.back().z) - 1.0f) free.push_back(int(i));
            if (free.empty()) { c.decided = true; return; }
            const int b = free[size_t(rng.pick(int(free.size())))];
            const Place& bay = bays[size_t(b)];
            std::vector<glm::vec3> in = paths["inpark"];
            const std::vector<glm::vec3> curve = bay_curve(bay, aisle);
            in.insert(in.end(), curve.begin(), curve.end());
            push_leg(c, in);
            std::vector<glm::vec3> back(curve.rbegin(), curve.rend());
            push_leg(c, back, true);                                   // out of the bay backwards, the way it came in
            std::vector<glm::vec3> out; out.push_back(curve.front());
            out.push_back(aisle.back());
            const auto& o = paths["outpark"]; out.insert(out.end(), o.begin(), o.end());
            push_leg(c, out);
            bays[size_t(b)].user = me; c.bay = b;
        }
        c.cs = Approach;
    }
    void arrive(Agent& c) {
        c.cs = Waiting; c.speed = 0.0f; c.pose = Pose::Park; c.wait = 3.0f; c.out = 0;
        const bool at_pump = c.stop >= 0;
        const int group = rng.pick(3);
        bool first = true;
        for (int r : c.riders) {
            if (r < 0) continue;
            Agent& p = agents[size_t(r)];
            if (at_pump && !first && rng.chance(0.6f)) { first = false; continue; }      // (passengers mostly stay in the car at a pump)
            p.hidden = false; p.car = int(&c - agents.data());
            const float side = first ? -1.0f : 1.0f;               // the driver's door is on the left
            const glm::vec2 f(std::sin(c.yaw), -std::cos(c.yaw));
            p.pos = c.pos + glm::vec3(f.y * side * -1.3f, 0.0f, -f.x * side * -1.3f) + glm::vec3(f.x, 0.0f, f.y) * (first ? 0.3f : -0.3f);
            p.yaw = c.yaw; p.at_node = -1; p.route.clear();
            errand(p, at_pump, first, group);
            ++c.out; first = false;
        }
    }
    // A CAR THAT IS ITS PEOPLE'S NO LONGER (a player took it, or its driver was shot). Whoever
    // sits in it is put out beside it - the driver on the left, the others on the right and
    // behind - and nobody belongs to it any more. At the site they go about as the locals do;
    // out on the road, far from anywhere one can walk, they stand there a while and are gone.
    void put_out(Agent& c) {
        const glm::vec3 right(std::cos(c.yaw), 0.0f, std::sin(c.yaw)), fwd(std::sin(c.yaw), 0.0f, -std::cos(c.yaw));
        for (int& r : c.riders) {
            if (r < 0) continue;
            Agent& p = agents[size_t(r)];
            if (p.hidden && p.gone <= 0.0f) {
                const float side = p.car_seat == 1 ? 1.0f : -1.0f;
                p.pos = c.pos + right * (side * 1.75f) + fwd * (p.car_seat == 2 ? -1.1f : 0.1f);
                p.yaw = wrap_pi(c.yaw - side * 1.5708f);          // (looking at the car that was theirs)
                p.hidden = false; p.pose = Pose::Stand; p.style = 0; p.speed = 0.0f; p.walk_v = 0.0f; p.act = 0; p.act_t = 0.0f;
                p.tasks.clear(); p.route.clear(); p.at_node = -1;
                const int nd = nodes.empty() ? -1 : nearest_node(p.pos);
                if (nd < 0 || glm::distance(p.pos, nodes[size_t(nd)].p) > 45.0f) p.leave = 40.0f;
            }
            p.car = -1; p.role = Local; r = -1;
        }
        c.out = 0;
    }
    // ... and the car itself leaves the world for `gone` seconds (then another joins the traffic).
    void car_leaves(Agent& c, float gone) {
        const int me = int(&c - agents.data());
        if (c.stop >= 0 && stops[size_t(c.stop)].user == me) stops[size_t(c.stop)].user = -1;
        if (c.bay >= 0 && bays[size_t(c.bay)].user == me) bays[size_t(c.bay)].user = -1;
        c.stop = c.bay = -1; c.out = 0; c.legs.clear(); c.legs_rev.clear(); c.path.clear();
        c.cs = Parked; c.pose = Pose::Park; c.speed = 0.0f; c.hidden = true; c.gone = gone;
        c.driverless = false; c.abandoned = 0.0f;
        c.pos.y -= 20000.0f; c.rear.y -= 20000.0f;
    }
    // How fast a car may be `ahead` metres before somebody on foot it must not drive into: it
    // stands six metres short, braking no harder than a driver does who saw it in time.
    static float walker_speed(float ahead) { return std::sqrt(2.0f * 2.6f * std::max(ahead - 6.0f, 0.0f)); }
    void step_car(Agent& c, float dt) {
        if (c.gone > 0.0f) { c.gone -= dt; if (c.gone <= 0.0f) { respawn(c); c.hidden = false; } return; }
        const int me = int(&c - agents.data());
        // Struck (push_car): it slides and turns, less and less.
        if (glm::dot(c.knock, c.knock) > 1e-4f || std::abs(c.knock_w) > 1e-3f) {
            c.pos.x += c.knock.x * dt; c.pos.z += c.knock.y * dt;
            c.yaw = wrap_pi(c.yaw + c.knock_w * dt);
            const glm::vec2 f(std::sin(c.yaw), -std::cos(c.yaw));
            c.rear = c.pos - glm::vec3(f.x, 0.0f, f.y) * (kWheelbase * 0.5f);
            const float k = glm::length(c.knock);
            c.knock = k > 5.0f * dt ? c.knock * ((k - 5.0f * dt) / k) : glm::vec2(0.0f);
            c.knock_w = std::abs(c.knock_w) > 3.0f * dt ? c.knock_w - (c.knock_w > 0.0f ? 3.0f : -3.0f) * dt : 0.0f;
        }
        // Its driver was shot (World::kill): it rolls to a stop where it is going (the branches
        // below, with no speed asked for); once it stands the others get out, and two minutes
        // later it is cleared away - nothing may stand for good.
        if (c.driverless && (std::abs(c.speed) < 0.05f || c.cs == Waiting || c.cs == Parked)) {
            c.speed = 0.0f; c.pose = Pose::Park;
            if (c.abandoned <= 0.0f) put_out(c);
            c.abandoned += dt;
            if (c.abandoned > 120.0f) car_leaves(c, 20.0f);
            return;
        }
        c.visit = (c.cs == OnPath || c.cs == Waiting) ? c.visit + dt : 0.0f;
        switch (c.cs) {
        case Parked: c.speed = 0.0f; c.pose = Pose::Park; break;
        case Cruise: case Approach: {
            // Where along the road it is: the place on its lane beside it (its own wheels took it
            // there - round the inside of a bend that is fewer metres than round the outside).
            // (and how far right of the centre line)
            glm::vec3 lp; glm::vec2 lf;
            if (!lane_at(c, c.s, lp, lf, 0.0f)) break;
            {
                const glm::vec2 rel(c.pos.x - lp.x, c.pos.z - lp.z);
                c.s = wrap_s(c.s + glm::dot(rel, lf) * float(c.dir));
                c.lat = glm::dot(rel, glm::vec2(-lf.y, lf.x));
            }
            // the others on the road: behind the one in front, past it when that can be done (THE TRAFFIC)
            float lat_want = kLane;
            float v = traffic(c, dt, lat_want);
            float to_in = (s_in - c.s) * float(c.dir);
            if (to_in < 0.0f) to_in += road.length;
            if (c.dir > 0 && site) {
                if (to_in > kDecide + 30.0f && to_in < road.length - 30.0f) c.decided = false;
                if (!c.decided && c.cs == Cruise && to_in < kDecide) {
                    c.decided = true;
                    // (not the police, and nobody who is out passing somebody)
                    if (c.cls != Police && c.ov == 0 && rng.chance(0.38f)) plan_visit(c);
                }
                // (off the throttle and on the brake in good time: 7 m/s where it turns in)
                if (c.cs == Approach) v = std::min(v, std::sqrt(49.0f + 2.0f * 1.9f * to_in));
            }
            if (c.cs == Approach && !c.driverless && (to_in <= std::abs(c.speed) * dt + 0.5f || to_in > road.length - 20.0f)) { begin_leg(c); break; }
            // (nor into a player who stands on its lane: it brakes in good time, and waits)
            for (const glm::vec3& wk : walkers) {
                if (std::abs(wk.x - c.pos.x) > 95.0f || std::abs(wk.z - c.pos.z) > 95.0f || std::abs(wk.y - c.pos.y) > 4.0f) continue;
                for (float a = 0.0f; a <= 90.0f; a += 3.0f) {
                    glm::vec3 q; glm::vec2 qf;
                    if (!lane_at(c, c.s + a * float(c.dir), q, qf, c.lat_cmd)) break;
                    if (glm::distance(glm::vec2(q.x, q.z), glm::vec2(wk.x, wk.z)) < 2.3f) { v = std::min(v, walker_speed(a)); break; }
                }
            }
            // (a bend ahead: down to the speed it can be taken at, in good time - the faster the
            //  driver, the further he has to look)
            {
                const float side = c.siren ? std::max(c.side, 5.0f) : c.side, sp = std::abs(c.speed);
                for (float d : { 25.0f, 50.0f, 80.0f, 120.0f, 170.0f, 230.0f, 300.0f }) {
                    if (d > 40.0f + sp * sp / 4.4f) break;
                    glm::vec3 p0, p1; glm::vec2 d0, d1;
                    if (!road.at(wrap_s(c.s + (d - 8.0f) * float(c.dir)), p0, d0) || !road.at(wrap_s(c.s + (d + 8.0f) * float(c.dir)), p1, d1)) break;
                    const float kappa = std::abs(d0.x * d1.y - d0.y * d1.x) / 16.0f;
                    if (kappa < 1e-4f) continue;
                    v = std::min(v, std::sqrt(side / kappa + 2.0f * 2.2f * std::max(d - 14.0f, 0.0f)));
                }
            }
            if (c.driverless || c.shaken > 0.0f) v = 0.0f;
            c.shaken = std::max(0.0f, c.shaken - dt);
            c.hard = !c.driverless && v < std::abs(c.speed) - 3.0f;                 // (far too fast for what is ahead: everything the brakes have)
            // (where it steers to moves across the road no faster than a driver changes lanes)
            {
                const float rate = c.ov ? std::clamp(0.6f + 0.07f * std::abs(c.speed), 1.0f, 1.7f) : std::clamp(0.3f + 0.06f * std::abs(c.speed), 0.4f, 1.3f);
                c.lat_cmd += std::clamp(lat_want - c.lat_cmd, -rate * dt, rate * dt);
            }
            // (it looks further ahead the faster it goes: at 40 m/s seventeen metres is under half a second)
            const float look = std::clamp(std::abs(c.speed) * 0.75f, 5.0f, 30.0f);
            glm::vec3 tp; glm::vec2 tf;
            if (!lane_at(c, c.s + look * float(c.dir), tp, tf, c.lat_cmd)) break;
            c.reverse = false;
            drive(c, tp, v, dt);
            c.hard = false;
            c.pos.y = lp.y + (tp.y - lp.y) * std::min(1.0f, std::abs(c.speed) * dt / look);
            c.pose = c.driverless ? Pose::Park : Pose::Drive;
            break; }
        case OnPath: {
            // How far along its line it is: the nearest point of it, never backwards.
            const glm::vec3 ref = c.reverse ? c.rear : c.pos;
            for (int k = 0; k < 60 && c.along < c.path_len; ++k) {
                glm::vec3 d0;
                const glm::vec3 a = poly_at(c.path, c.along, d0), b = poly_at(c.path, std::min(c.along + 0.2f, c.path_len), d0);
                const glm::vec2 da(a.x - ref.x, a.z - ref.z), db(b.x - ref.x, b.z - ref.z);
                if (glm::dot(db, db) >= glm::dot(da, da)) break;
                c.along = std::min(c.along + 0.2f, c.path_len);
            }
            const float left = c.path_len - c.along;
            const bool last = c.leaving && c.legs.empty();          // the way out to the road: it does not stop at its end
            float v = c.reverse ? 1.8f : kSiteSpeed;
            if (last) v = std::min(9.0f, kSiteSpeed + std::max(0.0f, 14.0f - left) * 0.35f);
            else v = std::min(v, std::sqrt(2.0f * 1.2f * std::max(left, 0.0f)) + 0.25f);
            // (nor into somebody on foot: it waits - a car's people excepted, they are at its doors)
            for (const Agent& o : agents) {
                if (o.kind != Kind::Person || o.hidden || o.car == me) continue;
                const glm::vec2 to(o.pos.x - c.pos.x, o.pos.z - c.pos.z);
                const glm::vec2 f(std::sin(c.yaw), -std::cos(c.yaw));
                const float ahead = glm::dot(to, f) * (c.reverse ? -1.0f : 1.0f);
                if (ahead > 0.0f && ahead < 8.0f && std::abs(to.x * f.y - to.y * f.x) < 1.55f && c.stuck < 20.0f)
                    v = std::min(v, std::max(0.0f, (ahead - 3.6f) * 0.9f));
            }
            // (nor into a player on foot)
            for (const glm::vec3& wk : walkers) {
                if (std::abs(wk.y - c.pos.y) > 4.0f) continue;
                const glm::vec2 to(wk.x - c.pos.x, wk.z - c.pos.z);
                const glm::vec2 f(std::sin(c.yaw), -std::cos(c.yaw));
                const float ahead = glm::dot(to, f) * (c.reverse ? -1.0f : 1.0f);
                if (ahead > 0.0f && ahead < 8.0f && std::abs(to.x * f.y - to.y * f.x) < 1.55f && c.stuck < 20.0f)
                    v = std::min(v, std::max(0.0f, (ahead - 3.6f) * 0.9f));
            }
            if (c.driverless) v = 0.0f;
            // (not into another car on the site)
            for (const Agent& o : agents) {
                if (&o == &c || o.kind != Kind::Car || o.cs == Cruise || o.cs == Approach || o.cs == Parked) continue;
                const glm::vec2 to(o.pos.x - c.pos.x, o.pos.z - c.pos.z);
                const glm::vec2 f(std::sin(c.yaw), -std::cos(c.yaw));
                const float ahead = glm::dot(to, f) * (c.reverse ? -1.0f : 1.0f);
                const float side = std::abs(to.x * f.y - to.y * f.x);
                const bool yield = o.cs == OnPath ? int(&o - agents.data()) < me && side < 1.7f : side < 1.2f;
                if (ahead > 0.5f && ahead < 8.0f && yield && c.stuck < 20.0f)
                    v = std::min(v, std::max(0.0f, (ahead - 5.5f) * 1.0f));
            }
            // (out on to the road: not in front of a car that is nearly there)
            if (last && left < 16.0f && left > 5.0f && c.stuck < 20.0f) {
                for (const Agent& o : agents) {
                    if (o.kind != Kind::Car || (o.cs != Cruise && o.cs != Approach) || o.dir <= 0) continue;
                    float gap = s_out - o.s;
                    if (gap < -25.0f) gap += road.length;
                    if (gap > -25.0f && gap < std::abs(o.speed) * 4.0f + 10.0f) v = std::min(v, std::max(0.0f, (left - 6.0f) * 0.8f));
                }
            }
            // (stood for twenty seconds: it goes, whoever is there - nothing may stand for good)
            c.stuck = std::abs(c.speed) < 0.15f && left > 0.5f ? c.stuck + dt : std::max(0.0f, c.stuck - dt * 4.0f);
            const float look = c.reverse ? 2.0f : std::clamp(std::abs(c.speed) * 0.7f, 1.9f, 8.0f);
            float straighten = 0.0f;
            if (!last && !c.reverse && left < 7.0f) {              // (the last metres to where it will stand)
                glm::vec3 d1;
                poly_at(c.path, std::min(c.along + 0.8f, c.path_len), d1);
                straighten = std::clamp(1.1f * wrap_pi(yaw_of(glm::vec2(d1.x, d1.z)) - c.yaw), -0.35f, 0.35f) * (1.0f - left / 7.0f);
            }
            drive(c, line_at(c.path, c.along + look, c.path_len), v, dt, straighten);
            { glm::vec3 d0; c.pos.y = poly_at(c.path, c.along, d0).y; }
            c.pose = c.driverless ? Pose::Park : Pose::Drive;
            if (!c.driverless && left <= (last ? 0.6f : 0.12f)) {
                if (!c.leaving) { arrive(c); }
                else if (!c.legs.empty()) { begin_leg(c); }
                else {                                   // back on the road
                    c.cs = Cruise; c.s = s_out; c.leaving = false; c.decided = true; c.reverse = false;
                    c.lat = c.lat_cmd = kLane; c.ov = 0; c.held = 0.0f;
                    if (c.stop >= 0) stops[size_t(c.stop)].user = -1;
                    if (c.bay >= 0) bays[size_t(c.bay)].user = -1;
                    c.stop = c.bay = -1;
                }
            }
            break; }
        case Waiting: {
            c.speed = 0.0f; c.pose = Pose::Park;
            c.wait -= dt;
            if (c.out <= 0 && c.wait <= 0.0f) {
                // (a car at the near stop of a lane waits for the far stop to clear)
                bool blocked = false;
                if (c.stop >= 0 && stops[size_t(c.stop)].kind.back() == 'a') {
                    std::string far = stops[size_t(c.stop)].kind; far.back() = 'b';
                    for (const Place& s : stops) if (s.kind == far && s.user >= 0 && s.user != me && agents[size_t(s.user)].cs == Waiting) blocked = true;
                }
                if (!blocked && !c.legs.empty()) { c.leaving = true; begin_leg(c); }
                else if (c.legs.empty()) { c.cs = Parked; }
            }
            break; }
        }
    }

    // ---------------------------------------------------------------- the step
    void snapshot(std::vector<View>& out) const {
        out.resize(agents.size());
        for (size_t i = 0; i < agents.size(); ++i) {
            const Agent& a = agents[i];
            View& v = out[i];
            v.id = uint8_t(i); v.kind = a.kind; v.pose = a.pose; v.look = a.look; v.hidden = a.hidden;
            v.pos = a.pos; v.yaw = a.yaw; v.speed = a.speed;
            v.car = (a.kind == Kind::Person && a.car >= 0) ? uint8_t(a.car) : uint8_t(255); v.seat = uint8_t(a.car_seat);
            v.act = Act(a.act); v.act_t = a.act_t; v.aim = a.aim; v.style = a.style;
            v.item = uint8_t((a.item & 15) | (a.placed << 4));
            if (a.kind == Kind::Car) v.style = uint8_t((a.cls == Police ? kCarPolice : 0) | (a.siren ? kCarSiren : 0) | (a.ov ? kCarOvertaking : 0));
        }
    }
    void step() {
        build_things();
        for (Agent& a : agents) if (a.kind == Kind::Car) step_car(a, kStep);
        wake_zones();
        for (Agent& a : agents) if (a.kind == Kind::Person) step_person(a, kStep);
        // (and once more when everybody has moved: somebody who got up from a seat, or out of a
        //  car, stands where a walker was passing - the walker steps clear)
        for (Agent& a : agents) if (a.kind == Kind::Person && !a.hidden && a.gone <= 0.0f && a.pose == Pose::Walk) keep_apart(a);
        ++steps;
    }
};

World::World() : d_(new Impl) {}
World::~World() { delete d_; }

bool World::load_site(const std::string& nav_file, glm::vec3 origin) {
    std::ifstream f(nav_file);
    if (!f) return false;
    d_->origin = origin;
    const bool ok = d_->parse(f);
    d_->link_road();
    return ok;
}
bool World::load_site_text(const std::string& text, glm::vec3 origin) {
    std::istringstream f(text);
    d_->origin = origin;
    const bool ok = d_->parse(f);
    d_->link_road();
    return ok;
}
bool World::add_site(const std::string& nav_file, glm::vec3 origin) {
    std::ifstream f(nav_file);
    if (!f || d_->zones.size() >= 250) return false;
    const glm::vec3 keep = d_->origin;
    d_->origin = origin;
    const bool ok = d_->parse(f, uint8_t(d_->zones.size() + 1));
    d_->origin = keep;
    return ok;
}
bool World::add_site_text(const std::string& text, glm::vec3 origin) {
    std::istringstream f(text);
    if (d_->zones.size() >= 250) return false;
    const glm::vec3 keep = d_->origin;
    d_->origin = origin;
    const bool ok = d_->parse(f, uint8_t(d_->zones.size() + 1));
    d_->origin = keep;
    return ok;
}
void World::set_eyes(const std::vector<glm::vec3>& at) { d_->eyes = at; }
void World::set_road(Road road) { d_->road = std::move(road); d_->link_road(); }

void World::populate(uint32_t seed, int traffic) {
    Impl& w = *d_;
    w.rng.s = seed ? seed : 1u;
    w.agents.clear(); w.steps = 0; w.acc = 0.0f; w.is_replica = false;
    w.n_overtakes = w.n_given_up = 0; w.obs_at.clear(); w.things.clear();
    for (auto* v : { &w.seats, &w.spots, &w.bays, &w.stops }) for (Place& p : *v) p.user = -1;
    auto person = [&](Role role) -> Agent* {
        if (int(w.agents.size()) >= kMaxAgents) return nullptr;
        Agent a; a.kind = Kind::Person; a.role = role; a.look = dressed(uint8_t(w.rng.pick(256)), role);
        w.agents.push_back(a); return &w.agents.back();
    };
    if (w.site) {
        // ---- the staff, at their places from the start
        for (Role r : { Cashier, Cook, Cook, Server }) {
            Agent* a = person(r); if (!a) break;
            const char* kind = r == Cashier ? "cashier" : r == Cook ? "cook" : "serve";
            const int s = w.free_place(w.spots, kind);
            if (s >= 0) { a->pos = w.spots[size_t(s)].p; a->yaw = w.spots[size_t(s)].yaw; a->at_node = w.spots[size_t(s)].node; }
        }
        // ---- the locals: each has a seat outside that is "his", and starts on a seat somewhere
        for (int i = 0; i < 14; ++i) {
            Agent* a = person(Local); if (!a) break;
            const int me = int(a - w.agents.data());
            a->home_seat = w.free_place(w.seats, i % 2 ? "picnic" : "bench");
            const char* kinds[4] = { "booth", "stool", "picnic", "chair" };
            const int s = w.free_place(w.seats, kinds[i % 4]);
            if (s >= 0) {
                w.seats[size_t(s)].user = me; a->seat = s;
                a->pos = w.seats[size_t(s)].p; a->yaw = w.seats[size_t(s)].yaw; a->pose = Pose::Sit; a->at_node = w.seats[size_t(s)].node;
                Task st; st.type = Sit; st.a = s; st.p = a->pos; st.yaw = a->yaw; st.time = w.rng.range(20.0f, 200.0f);
                a->tasks.push_back(st);
            } else if (!w.nodes.empty()) {
                const int n = w.rng.pick(w.home_nodes);
                a->pos = w.nodes[size_t(n)].p; a->at_node = n;
            }
        }
        // ---- cars that stand parked (in the bays the visiting cars do not use)
        const auto aisle = w.paths.find("aisle");
        int parked = 0;
        for (size_t i = 0; i < w.bays.size() && parked < 5; ++i) {
            const bool visitors = aisle != w.paths.end() && !aisle->second.empty() &&
                                  w.bays[i].p.z > std::min(aisle->second.front().z, aisle->second.back().z) + kBayRunUp &&
                                  w.bays[i].p.z < std::max(aisle->second.front().z, aisle->second.back().z) - 1.0f;
            if (visitors || i % 2 == 0) continue;          // (the bays visitors can turn into are left to them)
            if (int(w.agents.size()) >= kMaxAgents) break;
            Agent c; c.kind = Kind::Car; c.cs = Parked; c.pose = Pose::Park; c.look = uint8_t(w.rng.pick(256));
            c.pos = w.bays[i].p; c.yaw = w.bays[i].yaw; c.bay = int(i);
            c.rear = c.pos - glm::vec3(std::sin(c.yaw), 0.0f, -std::cos(c.yaw)) * 1.35f;
            w.bays[i].user = int(w.agents.size());
            w.agents.push_back(c); ++parked;
        }
    }
    // ---- the traffic: spread round the course, half each way; those that pass the site on
    //      their right carry people who may get out there
    if (w.road.at && w.road.length > 100.0f) {
        const int n = std::max(traffic, 0);
        for (int i = 0; i < n && int(w.agents.size()) < kMaxAgents; ++i) {
            Agent c; c.kind = Kind::Car; c.cs = Cruise; c.pose = Pose::Drive; c.look = uint8_t(w.rng.pick(256));
            c.dir = i % 2 ? -1 : 1;
            c.s = w.road.length * (float(i) + w.rng.range(0.1f, 0.9f)) / float(n);
            // (who drives it: two of them are the police, one is always somebody far too fast,
            //  the rest as it falls - make_driver)
            w.make_driver(c, i == 5 || i == 12 ? int(Police) : i == 8 ? int(Speeder) : -1); c.speed = c.v_want;
            w.agents.push_back(c);
            const int ci = int(w.agents.size()) - 1;
            if (c.dir > 0 && w.site && c.cls != Police) {
                const int riders = 1 + (w.rng.chance(0.5f) ? 1 : 0) + (w.rng.chance(0.2f) ? 1 : 0);
                for (int r = 0; r < riders; ++r) {
                    Agent* p = person(Traveller); if (!p) break;
                    p->hidden = true; p->car = ci; p->car_seat = r;
                    w.agents[size_t(ci)].riders[r] = int(p - w.agents.data());
                }
            }
            w.car_on_road(w.agents[size_t(ci)]);
        }
    }
    // ---- the soldiers on the run: two squads of six, two abreast, apart on the way
    if (w.run_len > 60.0f) {
        for (int q = 0; q < 2; ++q)
            for (int i = 0; i < 6; ++i) {
                Agent* a = person(Runner); if (!a) break;
                a->run_dir = q ? -1 : 1;
                a->run_s = std::clamp(w.run_len * (q ? 0.72f : 0.18f) - float(a->run_dir) * float(i / 2) * 1.9f, 3.0f, w.run_len - 3.0f);
                a->run_side = (i % 2) ? 1.25f : 0.35f;
                a->pose = Pose::Run;
                w.step_runner(*a, 0.0f);
                glm::vec3 dir; poly_at(w.run_path, a->run_s, dir);
                a->yaw = yaw_of(glm::vec2(dir.x, dir.z) * float(a->run_dir));
            }
    }
    // ---- the people of the added sites (THE ADDED SITES): made last, so that everybody else
    //      has the number he had without them; asleep until a player comes near (wake_zones).
    w.zrng.s = (seed ? seed : 1u) * 2654435761u + 0x51ED270Bu; if (!w.zrng.s) w.zrng.s = 1u;
    for (Impl::Zone& z : w.zones) z.awake = false;
    for (size_t zi = 0; zi < w.zones.size(); ++zi) {
        const uint8_t zone = uint8_t(zi + 1);
        auto of_kind = [&](std::vector<Place>& v, const char* kind) {
            std::vector<int> c;
            for (size_t i = 0; i < v.size(); ++i) if (v[i].zone == zone && v[i].kind == kind) c.push_back(int(i));
            return c;
        };
        auto shuffled = [&](std::vector<int> c) {
            for (size_t i = c.size(); i > 1; --i) std::swap(c[i - 1], c[size_t(w.zrng.pick(int(i)))]);
            return c;
        };
        auto add = [&](Role role, int seat, int spot) {
            if (int(w.agents.size()) >= kMaxAgents) return;
            Agent a; a.kind = Kind::Person; a.role = role; a.look = dressed(uint8_t(w.zrng.pick(256)), role);
            a.zone = zone; a.home_seat = seat; a.home_spot = spot; a.hidden = true; a.asleep = true;
            a.pos = w.zones[zi].centre;
            w.agents.push_back(a);
        };
        // at the desks: rather more than half of them taken; three round the meeting table; two on the way
        const std::vector<int> desks = shuffled(of_kind(w.seats, "desk")), meet = shuffled(of_kind(w.seats, "meet"));
        for (size_t i = 0; i < desks.size() && i < (desks.size() * 11 + 10) / 20; ++i) add(Desk, desks[i], -1);
        for (size_t i = 0; i < meet.size() && i < 3; ++i) add(Desk, meet[i], -1);
        if (!desks.empty()) for (int i = 0; i < 2; ++i) add(Floater, -1, -1);
        // outside: every group but one, most of those who look up, half of the apron's edge, half of the things to see to
        const std::vector<int> group = of_kind(w.spots, "group"), watch = shuffled(of_kind(w.spots, "watch")),
                               edge = shuffled(of_kind(w.spots, "edge")), tech = shuffled(of_kind(w.spots, "tech"));
        for (size_t i = 0; i < group.size(); ++i) if (i / 3 != 2) add(Visitor, -1, group[i]);
        for (size_t i = 0; i < watch.size() && i < (watch.size() * 2 + 2) / 3; ++i) add(Visitor, -1, watch[i]);
        for (size_t i = 0; i < edge.size() && i < edge.size() / 2; ++i) add(Visitor, -1, edge[i]);
        for (size_t i = 0; i < tech.size() && i < tech.size() / 2; ++i) add(Crew, -1, tech[i]);
    }
    w.snapshot(w.cur); w.prev = w.cur;
}

void World::set_run_path(std::vector<glm::vec3> pts) { d_->run_path = std::move(pts); d_->run_len = poly_len(d_->run_path); }

bool World::kill(int id) {
    Impl& w = *d_;
    if (w.is_replica || id < 0 || size_t(id) >= w.agents.size()) return false;
    Agent& a = w.agents[size_t(id)];
    if (a.gone > 0.0f) return false;
    if (a.kind == Kind::Car) {
        // The driver of a car that has nobody of this world at its wheel (the traffic the other
        // way). One with a driver of its own: that one is shot under his own number.
        if (a.driverless || a.pose != Pose::Drive) return false;
        for (int r : a.riders) if (r >= 0 && w.agents[size_t(r)].hidden && w.agents[size_t(r)].car_seat == 0) return false;
        a.driverless = true; a.abandoned = 0.0f; a.pose = Pose::Park;
        w.snapshot(w.cur);
        return true;
    }
    if (a.hidden && a.car < 0) return false;
    w.release(a);
    if (a.car >= 0) {
        Agent& c = w.agents[size_t(a.car)];
        if (a.hidden) {
            // (in the car: with its driver dead it rolls to a stop - step_car)
            for (int& r : c.riders) if (r == id) r = -1;
            if (a.car_seat == 0 && !c.driverless) { c.driverless = true; c.abandoned = 0.0f; c.pose = Pose::Park; }
        } else {
            // (its car does not wait for it)
            for (int& r : c.riders) if (r == id) { r = -1; if (c.out > 0) --c.out; }
        }
    }
    const bool runner = a.role == Runner;
    a.tasks.clear(); a.route.clear(); a.act = 0; a.car = -1; a.hidden = true; a.speed = 0.0f;
    a.gone = runner ? 1e9f : 75.0f;                // (a soldier who fell is not replaced)
    a.pos.y -= 20000.0f;                           // (out of everybody's way)
    w.snapshot(w.cur);
    return true;
}

bool World::can_take_car(int id) const {
    const Impl& w = *d_;
    if (w.is_replica || id < 0 || size_t(id) >= w.agents.size()) return false;
    const Agent& c = w.agents[size_t(id)];
    // (whatever it is doing and whoever sits in it: a car that stands, or barely rolls)
    return c.kind == Kind::Car && c.gone <= 0.0f && std::abs(c.speed) <= kTakeSpeed;
}

void World::set_walkers(const std::vector<glm::vec3>& feet) { d_->walkers = feet; }
void World::set_obstacles(const std::vector<Obstacle>& obs) { d_->obstacles = obs; }

bool World::push_car(int id, glm::vec3 dv, float spin) {
    Impl& w = *d_;
    if (w.is_replica || id < 0 || size_t(id) >= w.agents.size()) return false;
    Agent& c = w.agents[size_t(id)];
    if (c.kind != Kind::Car || c.gone > 0.0f || !std::isfinite(dv.x + dv.z + spin)) return false;
    glm::vec2 d(dv.x, dv.z);
    const float hit = glm::length(d);
    if (hit > 30.0f) d *= 30.0f / hit;
    const glm::vec2 f(std::sin(c.yaw), -std::cos(c.yaw));
    if (c.cs == Cruise || c.cs == Approach || c.cs == OnPath) {
        // (what goes along it is its speed; a car that reverses is not thrown forward)
        const float along = glm::dot(d, f);
        c.speed = c.reverse ? std::min(0.0f, c.speed + along) : std::max(0.0f, c.speed + along);
        d -= f * along;
    }
    c.knock += d;
    c.knock_w = std::clamp(c.knock_w + spin, -2.5f, 2.5f);
    if (hit > 3.0f && (c.cs == Cruise || c.cs == Approach)) c.shaken = std::max(c.shaken, std::min(2.0f + hit * 0.4f, 7.0f));
    if (c.ov == 1) c.ov = 2;
    return true;
}

void World::set_sirens(bool on) {
    for (Agent& a : d_->agents) if (a.kind == Kind::Car && a.cls == Police) { a.siren = on; a.siren_t = 1e9f; }
}

int World::add_car(float s, int dir, float cruise, uint8_t style) {
    Impl& w = *d_;
    if (w.is_replica || int(w.agents.size()) >= kMaxAgents || !w.road.at) return -1;
    Agent c; c.kind = Kind::Car; c.cs = Cruise; c.pose = Pose::Drive; c.look = uint8_t(w.agents.size() * 7u);
    c.dir = dir < 0 ? -1 : 1; c.s = w.wrap_s(s); c.decided = true;
    w.make_driver(c, (style & kCarPolice) ? int(Police) : int(Normal));
    c.v_want = cruise; c.speed = cruise; c.patience = 1.5f;
    if (cruise > kCruise * 1.6f) { c.cls = Speeder; c.accel = 3.6f; c.side = 6.0f; c.gap_t = 0.7f; c.patience = 0.5f; }
    if (style & kCarSiren) { c.siren = true; c.siren_t = 1e9f; } else if (c.cls == Police) c.siren_t = 1e9f;
    w.agents.push_back(c);
    w.car_on_road(w.agents.back());
    w.snapshot(w.cur); w.prev = w.cur;
    return int(w.agents.size()) - 1;
}

World::Traffic World::traffic() const {
    const Impl& w = *d_;
    Traffic t; t.overtakes = w.n_overtakes; t.given_up = w.n_given_up;
    for (const Agent& a : w.agents) {
        if (a.kind != Kind::Car || a.gone > 0.0f || (a.cs != Cruise && a.cs != Approach)) continue;
        t.overtaking += a.ov ? 1 : 0; t.slow += a.cls == Slow ? 1 : 0; t.speeders += a.cls == Speeder ? 1 : 0;
        t.police += a.cls == Police ? 1 : 0; t.sirens += a.siren ? 1 : 0;
        t.fastest = std::max(t.fastest, std::abs(a.speed));
    }
    return t;
}

bool World::take_car(int id) {
    if (!can_take_car(id)) return false;
    Impl& w = *d_;
    Agent& c = w.agents[size_t(id)];
    w.put_out(c);
    w.car_leaves(c, 150.0f);
    w.snapshot(w.cur);
    return true;
}

void World::update(float frame_dt) {
    Impl& w = *d_;
    w.acc += std::clamp(frame_dt, 0.0f, 0.5f);
    if (w.is_replica) return;
    while (w.acc >= kStep) {
        w.acc -= kStep;
        w.prev = w.cur;
        w.step();
        w.snapshot(w.cur);
    }
}

void World::views(std::vector<View>& out) const {
    const Impl& w = *d_;
    out = w.cur;
    const float gap = w.is_replica ? std::max(w.replica_gap, 0.05f) : kStep;
    const float t = std::clamp(w.acc / gap, 0.0f, 1.0f);
    for (size_t i = 0; i < out.size() && i < w.prev.size(); ++i) {
        const View& a = w.prev[i]; View& b = out[i];
        if (a.kind != b.kind || a.hidden != b.hidden || glm::distance(a.pos, b.pos) > 8.0f) continue;      // (appeared, or moved in one go: no blend)
        b.pos = glm::mix(a.pos, b.pos, t);
        b.yaw = wrap_pi(a.yaw + wrap_pi(b.yaw - a.yaw) * t);
        b.speed = a.speed + (b.speed - a.speed) * t;
        if (a.act == b.act && b.act != Act::None) b.act_t = a.act_t + (b.act_t - a.act_t) * t;
    }
}

void World::serialize(std::vector<uint8_t>& out) const {
    const Impl& w = *d_;
    out.resize(4 + w.cur.size() * sizeof(Wire));
    out[0] = 'N'; out[1] = 'P'; out[2] = 6;      // (wire format 6: + what a person carries) out[3] = uint8_t(w.cur.size());
    for (size_t i = 0; i < w.cur.size(); ++i) {
        const View& v = w.cur[i];
        Wire r;
        r.id = v.id; r.bits = uint8_t(uint8_t(v.kind) | (uint8_t(v.pose) << 1) | ((v.seat & 3) << 4) | (v.hidden ? 0x80 : 0)); r.look = v.look; r.car = v.car;
        r.act = uint8_t(uint8_t(v.act) | (v.style << 3)); r.act_t = uint8_t(std::clamp(std::lround(v.act_t * 20.0f), 0L, 255L));
        for (int k = 0; k < 3; ++k) r.aim[k] = int16_t(std::clamp(std::lround((v.aim[k] - v.pos[k]) * 100.0f), -32000L, 32000L));      // (centimetres from where it stands)
        r.x = v.pos.x; r.y = v.pos.y; r.z = v.pos.z;
        r.item = v.item;
        r.yaw = int16_t(std::lround(wrap_pi(v.yaw) * 10000.0f));
        r.speed = int8_t(std::clamp(std::lround(v.speed * 2.0f), -127L, 127L));      // (halves of a m/s: to 63 m/s - a speeder passing drives 47)
        std::memcpy(out.data() + 4 + i * sizeof(Wire), &r, sizeof(Wire));
    }
}

bool World::apply(const uint8_t* data, size_t size) {
    Impl& w = *d_;
    if (!data || size < 4 || data[0] != 'N' || data[1] != 'P' || data[2] != 6) return false;
    const size_t n = data[3];
    if (n > size_t(kMaxAgents) || size != 4 + n * sizeof(Wire)) return false;
    std::vector<View> in(n);
    for (size_t i = 0; i < n; ++i) {
        Wire r; std::memcpy(&r, data + 4 + i * sizeof(Wire), sizeof(Wire));
        View& v = in[i];
        v.id = r.id; v.kind = Kind(r.bits & 1); v.pose = Pose((r.bits >> 1) & 7); v.hidden = (r.bits & 0x80) != 0; v.look = r.look; v.car = r.car; v.seat = uint8_t((r.bits >> 4) & 3);
        v.act = Act((r.act & 7) <= uint8_t(Act::PickUp) ? (r.act & 7) : 0); v.item = r.item; v.style = uint8_t(r.act >> 3); v.act_t = float(r.act_t) * 0.05f;
        v.pos = glm::vec3(r.x, r.y, r.z); v.yaw = float(r.yaw) * 0.0001f; v.speed = float(r.speed) * 0.5f;
        v.aim = v.pos + glm::vec3(float(r.aim[0]), float(r.aim[1]), float(r.aim[2])) * 0.01f;
        if (!std::isfinite(v.pos.x) || !std::isfinite(v.pos.y) || !std::isfinite(v.pos.z) || uint8_t(v.pose) > uint8_t(Pose::Run)) return false;
    }
    if (!w.is_replica) { w.is_replica = true; w.agents.clear(); w.prev = in; }
    else {
        std::vector<View> now; views(now);          // (carry on from where the blend stands)
        w.prev = now.size() == in.size() ? now : in;
        w.replica_gap = std::clamp(w.acc, 0.05f, 1.0f);
    }
    w.cur = std::move(in);
    w.acc = 0.0f;
    return true;
}

const std::vector<View>& World::state() const { return d_->cur; }
bool World::replica() const { return d_->is_replica; }
int World::agent_count() const { return int(d_->cur.size()); }
int World::step_count() const { return d_->steps; }

World::Census World::census() const {
    Census c;
    for (const View& v : d_->cur) {
        if (v.kind == Kind::Person) {
            ++c.people;
            if (v.hidden) ++c.hidden;
            else if (v.pose == Pose::Sit) ++c.sitting;
            else if (v.pose == Pose::Walk) ++c.walking;
        } else {
            ++c.cars;
            if (v.pose == Pose::Drive) ++c.cars_moving;
        }
    }
    for (const Agent& a : d_->agents) if (a.kind == Kind::Car && (a.cs == OnPath || a.cs == Waiting || a.cs == Parked)) { ++c.cars_at_site; c.longest_visit = std::max(c.longest_visit, a.visit); }
    for (const Agent& a : d_->agents) if (a.kind == Kind::Person && a.asleep) { ++c.asleep; --c.hidden; }
    return c;
}

std::string World::describe_site() const {
    std::ostringstream o;
    const char* names[5] = { "cruise", "approach", "on-path", "waiting", "parked" };
    for (size_t i = 0; i < d_->agents.size(); ++i) {
        const Agent& a = d_->agents[i];
        if (a.kind != Kind::Car || a.cs == Cruise) continue;
        o << "car " << i << " " << names[a.cs] << " at (" << a.pos.x << ", " << a.pos.z << ") v " << a.speed << " along " << a.along << "/" << a.path_len
          << " legs " << a.legs.size() << " out " << a.out << " stop " << a.stop << " bay " << a.bay << " stuck " << a.stuck << (a.leaving ? " leaving" : "") << (a.reverse ? " reverse" : "") << '\n';
    }
    return o.str();
}

bool World::sane() const {
    const Impl& w = *d_;
    for (const View& v : w.cur)
        if (!std::isfinite(v.pos.x) || !std::isfinite(v.pos.y) || !std::isfinite(v.pos.z) || !std::isfinite(v.yaw) || !std::isfinite(v.speed)) return false;
    // (a held place is held by the agent that says so)
    for (size_t i = 0; i < w.agents.size(); ++i) {
        const Agent& a = w.agents[i];
        if (a.seat >= 0 && w.seats[size_t(a.seat)].user != int(i)) return false;
        if (a.kind == Kind::Car && a.stop >= 0 && w.stops[size_t(a.stop)].user != int(i)) return false;
        if (a.kind == Kind::Car && a.bay >= 0 && w.bays[size_t(a.bay)].user != int(i)) return false;
    }
    std::vector<int> seen;
    for (const Place& s : w.seats) if (s.user >= 0) { if (std::find(seen.begin(), seen.end(), s.user) != seen.end()) return false; seen.push_back(s.user); }
    return int(w.cur.size()) <= kMaxAgents;
}

}  // namespace vibe::npc
