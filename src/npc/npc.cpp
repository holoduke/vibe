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

struct Node { std::string name; glm::vec3 p; std::vector<int> adj; };
struct Place { std::string kind; glm::vec3 p; float yaw = 0.0f; int user = -1; int node = -1; };   // a seat, a spot, a bay or a stop

enum Role : uint8_t { Cashier, Cook, Server, Local, Traveller, Runner };
enum TaskType : uint8_t { GoNode, GoPoint, Sit, Stand, EnterCar, DoAct };
struct Task { TaskType type; int a = -1; glm::vec3 p{0.0f}; float yaw = 0.0f; float time = 0.0f; Pose pose = Pose::Stand; uint8_t style = 0; };

enum CarState : uint8_t { Cruise, Approach, OnPath, Waiting, Parked };

struct Agent {
    Kind kind = Kind::Person; Pose pose = Pose::Stand; uint8_t look = 0; bool hidden = false;
    glm::vec3 pos{0.0f}; float yaw = 0.0f; float speed = 0.0f;
    uint8_t style = 0;                   // View::style
    float gone = 0.0f;                   // > 0: out of the world (shot, or a car taken) for so many seconds more
    float run_s = 0.0f; int run_dir = 1; float run_side = 0.0f, run_lat = 0.0f;      // a runner: metres along the run, which way, and how far beside its line
    // ---- a person
    Role role = Local;
    std::vector<Task> tasks;             // the next is at the back
    std::vector<int> route;              // nodes still to pass, the next at the back
    int at_node = -1;
    int seat = -1, spot = -1;            // what it holds
    int home_seat = -1;
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
struct Wire { uint8_t id; uint8_t bits; uint8_t look; uint8_t car; float x, y, z; int16_t yaw; int8_t speed; uint8_t act; uint8_t act_t; int16_t aim[3]; };
#pragma pack(pop)
static_assert(sizeof(Wire) == 27, "an agent on the wire is 27 bytes");

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

    std::vector<Agent> agents;
    Rng rng;
    std::vector<glm::vec3> run_path; float run_len = 0.0f;
    int steps = 0;
    float acc = 0.0f;
    std::vector<View> prev, cur;             // the last two states, for views()
    bool is_replica = false;
    float replica_gap = 0.25f;               // seconds between the last two states taken in

    // ---------------------------------------------------------------- the site
    int nearest_node(glm::vec3 p) const {
        int best = -1; float bd = 1e30f;
        for (size_t i = 0; i < nodes.size(); ++i) {
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
    bool parse(std::istream& in) {
        nodes.clear(); node_of.clear(); seats.clear(); spots.clear(); bays.clear(); stops.clear(); paths.clear();
        std::string line;
        std::vector<std::pair<std::string, std::string>> links;
        while (std::getline(in, line)) {
            std::istringstream ls(line);
            std::string kw; ls >> kw;
            if (kw.empty() || kw[0] == '#') continue;
            if (kw == "node") {
                Node n; ls >> n.name >> n.p.x >> n.p.y >> n.p.z; n.p += origin;
                node_of[n.name] = int(nodes.size()); nodes.push_back(n);
            } else if (kw == "link") {
                std::string a, b; ls >> a >> b; links.push_back({ a, b });
            } else if (kw == "seat" || kw == "spot" || kw == "stop") {
                Place p; ls >> p.kind >> p.p.x >> p.p.y >> p.p.z >> p.yaw; p.p += origin;
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
            for (Place& p : *v) p.node = nearest_node(p.p);
        // (a pump's stop and the spot of the same name: the driver's place beside the car)
        for (Place& p : stops) { auto it = node_of.find(p.kind); if (it != node_of.end()) p.node = it->second; }
        site = !nodes.empty();
        return site;
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
        Task ap; ap.type = GoPoint; ap.p = pl.p; if (is_seat) ap.p = glm::mix(nodes[size_t(std::max(pl.node, 0))].p, pl.p, 0.55f); plan.push_back(ap);
        Task st; st.type = is_seat ? Sit : Stand; st.a = idx; st.p = pl.p; st.yaw = pl.yaw; st.time = time; st.pose = pose; plan.push_back(st);
        (void)a;
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
    // What a person does next, once it has nothing to do.
    void think(Agent& a) {
        std::vector<Task> plan;
        release(a);
        switch (a.role) {
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
                Task go; go.type = GoNode; go.a = rng.pick(int(nodes.size())); plan.push_back(go);
                Task st; st.type = Stand; st.p = nodes[size_t(go.a)].p; st.yaw = rng.range(-kPi, kPi); st.time = rng.range(5.0f, 20.0f); plan.push_back(st);
            }
            break; }
        case Traveller: {
            // Its errand is done (or never began): back to the car.
            if (a.car < 0) break;
            const Agent& c = agents[size_t(a.car)];
            const int n = c.stop >= 0 ? stops[size_t(c.stop)].node : c.bay >= 0 ? bays[size_t(c.bay)].node : nearest_node(c.pos);
            Task go; go.type = GoNode; go.a = n; plan.push_back(go);
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
        a = Agent();
        a.kind = k; a.look = look;
        if (k == Kind::Person) {
            a.role = Local;
            if (!nodes.empty()) { const int nd = rng.pick(int(nodes.size())); a.pos = nodes[size_t(nd)].p; a.at_node = nd; }
        } else {
            a.cs = Cruise; a.pose = Pose::Drive; a.dir = rng.chance(0.5f) ? 1 : -1;
            a.s = rng.range(0.0f, std::max(road.length, 1.0f)); a.v_want = kCruise * rng.range(0.8f, 1.15f); a.speed = a.v_want;
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
        if (a.gone > 0.0f) { a.gone -= dt; if (a.gone <= 0.0f) respawn(a); return; }
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
            if (a.pose != Pose::Sit && a.act != uint8_t(Act::StandUp)) {
                a.walk_v = 0.0f; a.speed = 0.0f; a.pose = Pose::Stand;
                if (a.act == 0) {
                    if (!turn_to(a, t.yaw, dt)) break;
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
                a.style = k == "stool" ? 1 : (k == "booth" || k == "chair" || k == "picnic") ? uint8_t(2 + ((a.look >> 3) & 1)) : 0;
            }
            if (a.act == uint8_t(Act::StandUp)) {
                a.pose = Pose::Stand; a.pos = a.from; a.yaw = t.yaw; a.speed = 0.0f;
                a.act_t += dt;
                if (a.act_t < kActTime[int(Act::StandUp)]) break;
                a.act = 0; a.act_t = 0.0f;
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
                    a.from = glm::mix(np, seats[size_t(t.a)].p, 0.55f); a.from.y = np.y;
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
            if (a.act_t >= kActTime[t.a]) { a.act = 0; a.act_t = 0.0f; a.tasks.pop_back(); }
            break; }
        case Stand: {
            a.pose = t.pose; a.style = t.style;
            turn_to(a, t.yaw, dt);
            a.walk_v = 0.0f; a.speed = 0.0f;
            t.time -= dt;
            if (t.time <= 0.0f) {
                a.tasks.pop_back();
                if (a.tasks.empty()) release(a);
            }
            break; }
        case EnterCar: {
            a.tasks.pop_back();
            if (a.car >= 0) { a.hidden = true; a.pose = Pose::Stand; a.speed = 0.0f; release(a); --agents[size_t(a.car)].out; }
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
    bool lane_at(const Agent& c, float s, glm::vec3& p, glm::vec2& fwd) const {
        glm::vec2 d;
        if (!road.at || !road.at(wrap_s(s), p, d)) return false;
        fwd = d * float(c.dir);
        p += glm::vec3(-fwd.y, 0.0f, fwd.x) * kLane;
        return true;
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
        if (std::abs(kappa) > 1e-3f) v_want = std::min(v_want, std::sqrt(kSideways / std::abs(kappa)));
        float sp = std::abs(c.speed);
        sp = std::max(0.0f, sp + std::clamp(v_want - sp, -kBrake * dt, kAccel * dt));
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
    void step_car(Agent& c, float dt) {
        if (c.gone > 0.0f) { c.gone -= dt; if (c.gone <= 0.0f) { respawn(c); c.hidden = false; } return; }
        const int me = int(&c - agents.data());
        c.visit = (c.cs == OnPath || c.cs == Waiting) ? c.visit + dt : 0.0f;
        switch (c.cs) {
        case Parked: c.speed = 0.0f; c.pose = Pose::Park; break;
        case Cruise: case Approach: {
            // Where along the road it is: the place on its lane beside it (its own wheels took it
            // there - round the inside of a bend that is fewer metres than round the outside).
            glm::vec3 lp; glm::vec2 lf;
            if (!lane_at(c, c.s, lp, lf)) break;
            c.s = wrap_s(c.s + glm::dot(glm::vec2(c.pos.x - lp.x, c.pos.z - lp.z), lf) * float(c.dir));
            // keep behind the car in front
            float v = c.v_want;
            for (const Agent& o : agents) {
                if (&o == &c || o.kind != Kind::Car || (o.cs != Cruise && o.cs != Approach) || o.dir != c.dir) continue;
                float gap = (o.s - c.s) * float(c.dir);
                if (gap < 0.0f) gap += road.length;
                if (gap < 45.0f) v = std::min(v, std::max(0.0f, std::abs(o.speed) + (gap - 14.0f) * 0.5f));
            }
            float to_in = (s_in - c.s) * float(c.dir);
            if (to_in < 0.0f) to_in += road.length;
            if (c.dir > 0 && site) {
                if (to_in > kDecide + 30.0f && to_in < road.length - 30.0f) c.decided = false;
                if (!c.decided && c.cs == Cruise && to_in < kDecide) {
                    c.decided = true;
                    if (rng.chance(0.38f)) plan_visit(c);
                }
                // (off the throttle and on the brake in good time: 7 m/s where it turns in)
                if (c.cs == Approach) v = std::min(v, std::sqrt(49.0f + 2.0f * 1.9f * to_in));
            }
            if (c.cs == Approach && (to_in <= std::abs(c.speed) * dt + 0.5f || to_in > road.length - 20.0f)) { begin_leg(c); break; }
            const float look = std::clamp(std::abs(c.speed) * 0.75f, 5.0f, 17.0f);
            glm::vec3 tp; glm::vec2 tf;
            if (!lane_at(c, c.s + look * float(c.dir), tp, tf)) break;
            c.reverse = false;
            drive(c, tp, v, dt);
            c.pos.y = lp.y + (tp.y - lp.y) * std::min(1.0f, std::abs(c.speed) * dt / look);
            c.pose = Pose::Drive;
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
            c.pose = Pose::Drive;
            if (left <= (last ? 0.6f : 0.12f)) {
                if (!c.leaving) { arrive(c); }
                else if (!c.legs.empty()) { begin_leg(c); }
                else {                                   // back on the road
                    c.cs = Cruise; c.s = s_out; c.leaving = false; c.decided = true; c.reverse = false;
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
        }
    }
    void step() {
        for (Agent& a : agents) if (a.kind == Kind::Car) step_car(a, kStep);
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
void World::set_road(Road road) { d_->road = std::move(road); d_->link_road(); }

void World::populate(uint32_t seed) {
    Impl& w = *d_;
    w.rng.s = seed ? seed : 1u;
    w.agents.clear(); w.steps = 0; w.acc = 0.0f; w.is_replica = false;
    for (auto* v : { &w.seats, &w.spots, &w.bays, &w.stops }) for (Place& p : *v) p.user = -1;
    auto person = [&](Role role) -> Agent* {
        if (int(w.agents.size()) >= kMaxAgents) return nullptr;
        Agent a; a.kind = Kind::Person; a.role = role; a.look = uint8_t(w.rng.pick(256));
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
                const int n = w.rng.pick(int(w.nodes.size()));
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
        const int n = 26;
        for (int i = 0; i < n && int(w.agents.size()) < kMaxAgents; ++i) {
            Agent c; c.kind = Kind::Car; c.cs = Cruise; c.pose = Pose::Drive; c.look = uint8_t(w.rng.pick(256));
            c.dir = i % 2 ? -1 : 1;
            c.s = w.road.length * (float(i) + w.rng.range(0.1f, 0.9f)) / float(n);
            c.v_want = kCruise * w.rng.range(0.8f, 1.15f); c.speed = c.v_want;
            w.agents.push_back(c);
            const int ci = int(w.agents.size()) - 1;
            if (c.dir > 0 && w.site) {
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
    w.snapshot(w.cur); w.prev = w.cur;
}

void World::set_run_path(std::vector<glm::vec3> pts) { d_->run_path = std::move(pts); d_->run_len = poly_len(d_->run_path); }

bool World::kill(int id) {
    Impl& w = *d_;
    if (w.is_replica || id < 0 || size_t(id) >= w.agents.size()) return false;
    Agent& a = w.agents[size_t(id)];
    if (a.kind != Kind::Person || a.hidden || a.gone > 0.0f) return false;
    w.release(a);
    if (a.car >= 0) {
        // (its car does not wait for it)
        Agent& c = w.agents[size_t(a.car)];
        for (int& r : c.riders) if (r == id) { r = -1; if (c.out > 0) --c.out; }
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
    if (c.kind != Kind::Car || c.gone > 0.0f || !(c.cs == Parked || c.cs == Waiting) || std::abs(c.speed) > 0.3f) return false;
    for (int r : c.riders) if (r >= 0 && w.agents[size_t(r)].hidden) return false;        // (somebody sits in it)
    return true;
}

bool World::take_car(int id) {
    if (!can_take_car(id)) return false;
    Impl& w = *d_;
    Agent& c = w.agents[size_t(id)];
    for (int& r : c.riders) if (r >= 0) { Agent& p = w.agents[size_t(r)]; p.car = -1; p.role = Local; r = -1; }
    if (c.stop >= 0 && w.stops[size_t(c.stop)].user == id) w.stops[size_t(c.stop)].user = -1;
    if (c.bay >= 0 && w.bays[size_t(c.bay)].user == id) w.bays[size_t(c.bay)].user = -1;
    c.stop = c.bay = -1; c.out = 0; c.legs.clear(); c.legs_rev.clear(); c.path.clear();
    c.cs = Parked; c.pose = Pose::Park; c.speed = 0.0f; c.hidden = true; c.gone = 150.0f;
    c.pos.y -= 20000.0f; c.rear.y -= 20000.0f;
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
    out[0] = 'N'; out[1] = 'P'; out[2] = 4; out[3] = uint8_t(w.cur.size());
    for (size_t i = 0; i < w.cur.size(); ++i) {
        const View& v = w.cur[i];
        Wire r;
        r.id = v.id; r.bits = uint8_t(uint8_t(v.kind) | (uint8_t(v.pose) << 1) | ((v.seat & 3) << 4) | (v.hidden ? 0x80 : 0)); r.look = v.look; r.car = v.car;
        r.act = uint8_t(uint8_t(v.act) | (v.style << 3)); r.act_t = uint8_t(std::clamp(std::lround(v.act_t * 20.0f), 0L, 255L));
        for (int k = 0; k < 3; ++k) r.aim[k] = int16_t(std::clamp(std::lround((v.aim[k] - v.pos[k]) * 100.0f), -32000L, 32000L));      // (centimetres from where it stands)
        r.x = v.pos.x; r.y = v.pos.y; r.z = v.pos.z;
        r.yaw = int16_t(std::lround(wrap_pi(v.yaw) * 10000.0f));
        r.speed = int8_t(std::clamp(std::lround(v.speed * 4.0f), -127L, 127L));
        std::memcpy(out.data() + 4 + i * sizeof(Wire), &r, sizeof(Wire));
    }
}

bool World::apply(const uint8_t* data, size_t size) {
    Impl& w = *d_;
    if (!data || size < 4 || data[0] != 'N' || data[1] != 'P' || data[2] != 4) return false;
    const size_t n = data[3];
    if (n > size_t(kMaxAgents) || size != 4 + n * sizeof(Wire)) return false;
    std::vector<View> in(n);
    for (size_t i = 0; i < n; ++i) {
        Wire r; std::memcpy(&r, data + 4 + i * sizeof(Wire), sizeof(Wire));
        View& v = in[i];
        v.id = r.id; v.kind = Kind(r.bits & 1); v.pose = Pose((r.bits >> 1) & 7); v.hidden = (r.bits & 0x80) != 0; v.look = r.look; v.car = r.car; v.seat = uint8_t((r.bits >> 4) & 3);
        v.act = Act((r.act & 7) <= uint8_t(Act::CarExit) ? (r.act & 7) : 0); v.style = uint8_t(r.act >> 3); v.act_t = float(r.act_t) * 0.05f;
        v.pos = glm::vec3(r.x, r.y, r.z); v.yaw = float(r.yaw) * 0.0001f; v.speed = float(r.speed) * 0.25f;
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
