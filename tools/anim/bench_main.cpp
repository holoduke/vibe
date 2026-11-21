// vibe_animbench: what animation costs on this machine, for 1 .. 1000 characters.
//
//   vibe_animbench <soldier.qanim> [seconds per case]
//
// Every character is a CharacterAnimator on the real asset, driven by the real character
// controller with its own scripted input (running, strafing, turning, jumping, firing). Measured:
// CPU time of the animation update per frame (not skinning, not drawing: those are the
// renderer's, see the [anim] line in vibe.log), at each level of detail, for a mix of levels as
// a client would have, on one thread and spread over all cores; memory; heap allocations made
// inside the update. Prints a table for docs/animation/performance.md.
#include "anim/anim_lod.h"
#include "anim/character_animator.h"
#include "game/player.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <random>
#include <thread>
#include <vector>

using namespace qlike;

static std::atomic<long long> g_allocs{0};
static std::atomic<bool> g_count{false};
void* operator new(std::size_t n) {
    if (g_count.load(std::memory_order_relaxed)) g_allocs.fetch_add(1, std::memory_order_relaxed);
    if (void* p = std::malloc(n ? n : 1)) return p;
    throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }

namespace {

float flat(void*, float, float) { return 0.0f; }

struct Actor {
    game::Player p;
    std::unique_ptr<anim::CharacterAnimator> anim;
    std::mt19937 rng;
    game::PlayerInput pin;
    float until = 0.0f, yaw_rate = 0.0f;
    uint32_t shots = 0;
    int lod = 0;
};

double now_ms() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

void drive(Actor& a, float t, float dt, const std::vector<collision::AABB>& world) {
    if (t >= a.until) {
        std::uniform_real_distribution<float> u(0.0f, 1.0f);
        a.pin = game::PlayerInput{};
        a.pin.fwd = u(a.rng) < 0.6f; a.pin.back = !a.pin.fwd && u(a.rng) < 0.3f;
        a.pin.left = u(a.rng) < 0.25f; a.pin.right = !a.pin.left && u(a.rng) < 0.25f;
        a.pin.sprint = u(a.rng) < 0.4f; a.pin.crawl = u(a.rng) < 0.08f;
        a.pin.jump = u(a.rng) < 0.25f;
        a.yaw_rate = (u(a.rng) - 0.5f) * 5.0f;
        a.until = t + 0.2f + u(a.rng) * 1.2f;
        if (u(a.rng) < 0.5f) a.shots += 3;
    }
    a.p.yaw += a.yaw_rate * dt;
    game::update_player(a.p, a.pin, world, dt);
}

void animate(Actor& a, float dt) {
    anim::CharacterAnimationInput in;
    in.dt = dt;
    in.position = a.p.position - glm::vec3(0.0f, 0.9f, 0.0f);
    in.velocity = a.p.velocity;
    const glm::vec3 f(std::sin(a.p.yaw), 0.0f, -std::cos(a.p.yaw)), r(std::cos(a.p.yaw), 0.0f, std::sin(a.p.yaw));
    glm::vec3 wish(0.0f);
    if (a.pin.fwd) wish += f;
    if (a.pin.back) wish -= f;
    if (a.pin.right) wish += r;
    if (a.pin.left) wish -= r;
    if (glm::dot(wish, wish) > 0.0f) in.desired_velocity = glm::normalize(wish) * 8.0f * (a.pin.sprint ? 1.5f : a.pin.crawl ? 0.4f : 1.0f);
    in.facing_yaw = a.p.yaw;
    in.grounded = a.p.on_ground;
    in.wants_jump = a.pin.jump;
    in.crouch = a.p.crouch_factor;
    in.eye_local = glm::vec3(0.0f, 0.9f + a.p.eye_offset_y(), -0.11f);
    in.shot_seq = a.shots;
    in.ground_height = flat;
    in.lod = a.lod;
    a.anim->update(in);
}

struct Stat { double p50, p95, p99, mean; long long allocs; };

// lod_of(i, n): the level of character i of n. threads: 1 = on the calling thread.
Stat run_case(const anim::AnimationSet& set, int n, float seconds, int (*lod_of)(int, int), unsigned threads) {
    std::vector<collision::AABB> world{ { glm::vec3(-1e5f, -1.0f, -1e5f), glm::vec3(1e5f, 0.0f, 1e5f) } };
    std::vector<Actor> actors(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        Actor& a = actors[size_t(i)];
        a.p.position = glm::vec3(float(i % 40) * 3.0f, 0.9f, float(i / 40) * 3.0f);
        a.p.on_ground = true;
        a.rng.seed(unsigned(i) * 7919u + 13u);
        a.anim = std::make_unique<anim::CharacterAnimator>(set);
        a.lod = lod_of(i, n);
    }
    const float dt = 1.0f / 60.0f;
    std::vector<double> ms;
    float t = 0.0f;
    long long allocs = 0;
    for (int frame = 0; t < seconds + 1.0f; ++frame, t += dt) {
        for (Actor& a : actors) { drive(a, t, dt * 0.5f, world); drive(a, t, dt * 0.5f, world); }      // the controller: 120 Hz
        const bool measured = t >= 1.0f;                                                                // first second: warm-up
        if (measured) { g_allocs = 0; g_count = true; }
        const double t0 = now_ms();
        if (threads <= 1) {
            for (Actor& a : actors) animate(a, dt);
        } else {
            g_count = false;                                                                            // (thread start-up allocates)
            std::vector<std::thread> pool;
            std::atomic<int> next{0};
            for (unsigned k = 0; k < threads; ++k)
                pool.emplace_back([&] { for (int i; (i = next.fetch_add(8)) < n;) for (int j = i; j < std::min(i + 8, n); ++j) animate(actors[size_t(j)], dt); });
            for (auto& th : pool) th.join();
        }
        const double t1 = now_ms();
        g_count = false;
        if (measured) { ms.push_back(t1 - t0); allocs += g_allocs.load(); }
    }
    std::sort(ms.begin(), ms.end());
    double sum = 0.0;
    for (double v : ms) sum += v;
    auto pct = [&](double q) { return ms[std::min(ms.size() - 1, size_t(q * double(ms.size())))]; };
    return { pct(0.50), pct(0.95), pct(0.99), sum / double(ms.size()), threads <= 1 ? allocs : -1 };
}

int all0(int, int) { return 0; }
int all1(int, int) { return 1; }
int all2(int, int) { return 2; }
int all3(int, int) { return 3; }
// What one client has in a crowd: a few close, some at mid range, many small, the rest not in view.
int mixed(int i, int) { return i < 8 ? 0 : i < 32 ? 1 : i < 96 ? 2 : 3; }

// How far the feet slide. The real controller is held on one key combination over flat ground
// until its speed is steady; then, whenever the capture has a foot on the ground (its contact
// flag), that foot moving over the ground is slide (0 = planted), and a foot the animator has
// lifted off the ground meanwhile is counted as not standing ("on the ground" < 100 %).
void slide_table(const anim::AnimationSet& set) {
    const anim::Rig& rig = set.runtime->rig();
    const int foot[2] = { rig[anim::Role::LeftFoot], rig[anim::Role::RightFoot] };
    struct Dir { const char* name; bool f, b, r; };
    const Dir dirs[] = { { "forward", true, false, false }, { "diagonal", true, false, true }, { "sideways", false, false, true }, { "backward", false, true, false } };
    struct Pace { const char* name; bool sprint, crawl; };
    const Pace paces[] = { { "crawl key", false, true }, { "walk", false, false }, { "sprint", true, false } };
    std::printf("pace the capture sustains (m/s), forward / sideways / backward: standing %.2f / %.2f / %.2f, crouched %.2f / %.2f / %.2f\n\n",
                double(set.db.cruise_speed(false, 0)), double(set.db.cruise_speed(false, 1)), double(set.db.cruise_speed(false, 2)),
                double(set.db.cruise_speed(true, 0)), double(set.db.cruise_speed(true, 1)), double(set.db.cruise_speed(true, 2)));
    std::printf("| pace | direction | speed m/s | clip | clip speed | rate | stride | slide m/s while down | share of speed | foot on the ground while down | locked | fastest joint rad/s |\n");
    std::printf("|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|\n");
    const float dt = 1.0f / 60.0f;
    std::vector<collision::AABB> world{ { glm::vec3(-1e5f, -1.0f, -1e5f), glm::vec3(1e5f, 0.0f, 1e5f) } };
    static const char* kTraceEnv = std::getenv("SLIDE_TRACE");
    static const float trace_from = std::getenv("SLIDE_TRACE_T") ? float(std::atof(std::getenv("SLIDE_TRACE_T"))) : 4.0f;
    for (const Pace& pc : paces) {
        for (const Dir& d : dirs) {
            Actor a;
            a.p.position = glm::vec3(0.0f, 0.9f, 0.0f);
            a.p.on_ground = true;
            a.anim = std::make_unique<anim::CharacterAnimator>(set);
            if (const char* off = std::getenv("SLIDE_OFF")) {            // comparison: switch layers off
                const std::string o(off);
                if (o.find("stride") != std::string::npos) a.anim->tuning.stride_warp = false;
                if (o.find("orient") != std::string::npos) a.anim->tuning.orient_warp = false;
                if (o.find("lock") != std::string::npos) a.anim->tuning.foot_lock = false;
                if (o.find("footik") != std::string::npos) a.anim->tuning.foot_ik = false;
                if (o.find("hand") != std::string::npos) a.anim->tuning.hand_ik = false;
                if (o.find("lean") != std::string::npos) a.anim->tuning.lean_strength = 0.0f;
                if (o.find("rate") != std::string::npos) a.anim->tuning.max_rate = 1.0f;
            }
            if (const char* e = std::getenv("SLIDE_STRIDE")) a.anim->tuning.max_stride = float(std::atof(e));
            if (const char* e = std::getenv("SLIDE_LEG")) a.anim->tuning.stride_leg = float(std::atof(e));
            if (const char* e = std::getenv("SLIDE_REACH")) a.anim->tuning.stride_reach = float(std::atof(e));
            if (const char* e = std::getenv("SLIDE_DROP")) a.anim->tuning.max_stride_drop = float(std::atof(e));
            a.pin.fwd = d.f; a.pin.back = d.b; a.pin.right = d.r;
            a.pin.sprint = pc.sprint; a.pin.crawl = pc.crawl;
            std::vector<glm::vec3> track[2];
            std::vector<char> on[2];
            std::string clip;
            double clip_speed = 0.0, rate = 0.0, stride = 0.0, locked = 0.0, speed = 0.0;
            float rest_y[2] = { 1e9f, 1e9f };
            std::vector<anim::Transform> prev;
            float worst_rate = 0.0f, worst_t = 0.0f;
            std::string worst_joint;
            int n = 0;
            for (float t = 0.0f; t < 9.0f; t += dt) {
                game::update_player(a.p, a.pin, world, dt * 0.5f);
                game::update_player(a.p, a.pin, world, dt * 0.5f);
                animate(a, dt);
                if (t < 2.0f) continue;
                {
                    const auto loc = a.anim->local();
                    if (!prev.empty())
                        for (size_t j = 1; j < loc.size(); ++j) {
                            const glm::quat dq = glm::inverse(prev[j].r) * loc[j].r;
                            const float ang = 2.0f * std::asin(std::min(glm::length(glm::vec3(dq.x, dq.y, dq.z)), 1.0f)) / dt;
                            const std::string& jn = set.runtime->skeleton().names[j];
                            if (jn == "LeftHand" || jn == "RightHand") continue;
                            if (ang > worst_rate) { worst_rate = ang; worst_t = t; worst_joint = set.runtime->skeleton().names[j]; }
                        }
                    prev.assign(loc.begin(), loc.end());
                }
                const anim::AnimDebug& g = a.anim->debug();
                clip_speed += g.anim_speed; rate += g.rate; stride += g.stride;
                speed += glm::length(glm::vec2(a.p.velocity.x, a.p.velocity.z));
                locked += 0.5 * (double(g.locked[0]) + double(g.locked[1]));
                ++n;
                for (int k = 0; k < 2; ++k) {
                    const glm::vec3 w = a.p.position - glm::vec3(0.0f, 0.9f, 0.0f) + glm::vec3(a.anim->model()[size_t(foot[k])][3]);     // yaw 0: model = world axes
                    track[k].push_back(w);
                    on[k].push_back(g.contact[k] ? 1 : 0);
                    rest_y[k] = std::min(rest_y[k], w.y);
                }
                if (g.clip >= 0) clip = set.runtime->clip(g.clip).name;
                if (kTraceEnv && std::string(kTraceEnv) == std::string(pc.name) + "/" + d.name && t > trace_from && t < trace_from + 0.5f) {
                    // per leg: how far the foot is from the hip joint as a share of the leg's length, its height, the knee's rate
                    static std::vector<anim::Transform> tp;
                    const auto loc = a.anim->local();
                    const auto m = a.anim->model();
                    std::printf("  t %.3f %s @ %.2f%s stride %.2f |", double(t), clip.c_str(), double(g.clip_time), g.mirrored ? " m" : "", double(g.stride));
                    for (int k = 0; k < 2; ++k) {
                        const int up = rig[k == 0 ? anim::Role::LeftUpLeg : anim::Role::RightUpLeg], lo = rig[k == 0 ? anim::Role::LeftLeg : anim::Role::RightLeg];
                        const glm::vec3 pu(m[size_t(up)][3]), pl(m[size_t(lo)][3]), pf(m[size_t(foot[k])][3]);
                        const float len = glm::length(pl - pu) + glm::length(pf - pl);
                        float kr = 0.0f;
                        if (!tp.empty()) {
                            const glm::quat dq = glm::inverse(tp[size_t(lo)].r) * loc[size_t(lo)].r;
                            kr = 2.0f * std::asin(std::min(glm::length(glm::vec3(dq.x, dq.y, dq.z)), 1.0f)) / dt;
                        }
                        std::printf(" %c c%d ext %.3f z %+.3f y %.3f knee %5.1f |", k == 0 ? 'L' : 'R', int(g.contact[k]), double(glm::length(pf - pu) / len),
                                    double(pf.z - pu.z), double(pf.y), double(kr));
                    }
                    {
                        // right arm: reach as a share of its length, rates of the shoulder, upper arm and forearm
                        const int sh = rig[anim::Role::RightShoulder], ua = rig[anim::Role::RightArm], fa = rig[anim::Role::RightForeArm], hd = rig[anim::Role::RightHand];
                        const glm::vec3 p0(m[size_t(ua)][3]), p1(m[size_t(fa)][3]), p2(m[size_t(hd)][3]);
                        auto rate = [&](int j) {
                            if (tp.empty()) return 0.0f;
                            const glm::quat dq = glm::inverse(tp[size_t(j)].r) * loc[size_t(j)].r;
                            return 2.0f * std::asin(std::min(glm::length(glm::vec3(dq.x, dq.y, dq.z)), 1.0f)) / dt;
                        };
                        std::printf(" arm ext %.3f sh %5.1f ua %5.1f fa %5.1f  warp %.2f", double(glm::length(p2 - p0) / (glm::length(p1 - p0) + glm::length(p2 - p1))),
                                    double(rate(sh)), double(rate(ua)), double(rate(fa)), double(g.warp));
                    }
                    std::printf("\n");
                    tp.assign(loc.begin(), loc.end());

                }
            }
            double slid = 0.0;
            int down = 0, standing = 0;
            for (int k = 0; k < 2; ++k) {
                for (size_t i = 1; i < track[k].size(); ++i) {
                    if (!on[k][i] || !on[k][i - 1]) continue;
                    ++down;
                    if (track[k][i].y > rest_y[k] + 0.03f) continue;          // lifted: in the air, not sliding
                    ++standing;
                    const glm::vec3 dv = track[k][i] - track[k][i - 1];
                    slid += double(std::sqrt(dv.x * dv.x + dv.z * dv.z)) / double(dt);
                }
            }
            const double s = standing ? slid / standing : 0.0;
            speed /= n;
            std::printf("| %s | %s | %.1f | %s | %.2f | %.2f | %.2f | %.2f | %.0f %% | %.0f %% | %.0f %% | %.0f %s at %.2f s |\n", pc.name, d.name, speed, clip.c_str(), clip_speed / n, rate / n,
                        stride / n, s, 100.0 * s / std::max(speed, 1e-3), down ? 100.0 * standing / down : 0.0, 100.0 * locked / n,
                        double(worst_rate), worst_joint.c_str(), double(worst_t));
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: vibe_animbench <soldier.qanim> [seconds per case | slide]\n"); return 2; }
    if (argc > 2 && std::string(argv[2]) == "slide") {
        anim::AnimationSet sset;
        std::string swhy;
        if (!sset.load(argv[1], &swhy)) { std::fprintf(stderr, "%s: %s\n", argv[1], swhy.c_str()); return 1; }
        slide_table(sset);
        return 0;
    }
    const float seconds = argc > 2 ? float(std::atof(argv[2])) : 4.0f;
    anim::AnimationSet set;
    std::string why;
    if (!set.load(argv[1], &why)) { std::fprintf(stderr, "%s: %s\n", argv[1], why.c_str()); return 1; }
    const unsigned cores = std::max(1u, std::thread::hardware_concurrency());
    std::printf("asset: %d clips, %.2f MB compressed clips, motion database %u frames %.2f MB; %u hardware threads\n",
                set.runtime->num_clips(), double(set.runtime->clip_bytes()) / 1048576.0, set.db.frames(),
                double(set.db.memory_bytes()) / 1048576.0, cores);
    // per-character memory: build one and count its buffers by allocation
    {
        const size_t joints = size_t(set.runtime->skeleton().size());
        const size_t pose = joints * (5 * sizeof(anim::Transform) + 2 * sizeof(glm::mat4) + 4 * sizeof(glm::vec3));
        std::printf("per character: about %.1f kB of pose buffers (%zu joints) plus the runtime's sampling caches\n", double(pose) / 1024.0, joints);
    }
    std::printf("\n| characters | levels | threads | ms / frame p50 | p95 | p99 | us / character (mean) | allocations in update |\n");
    std::printf("|---:|---|---:|---:|---:|---:|---:|---:|\n");
    struct Case { const char* name; int (*lod)(int, int); };
    const Case cases[] = { { "all 0 (full)", all0 }, { "all 1", all1 }, { "all 2", all2 }, { "all 3 (unseen)", all3 }, { "mix 8/24/64/rest", mixed } };
    for (int n : { 1, 8, 16, 32, 64, 128, 256, 512, 1000 }) {
        for (const Case& c : cases) {
            if (c.lod == mixed && n < 128) continue;
            if ((c.lod == all1 || c.lod == all2 || c.lod == all3) && n != 64 && n != 1000) continue;
            const Stat s = run_case(set, n, seconds, c.lod, 1);
            std::printf("| %d | %s | 1 | %.3f | %.3f | %.3f | %.2f | %lld |\n", n, c.name, s.p50, s.p95, s.p99, s.mean * 1000.0 / double(n), s.allocs);
            std::fflush(stdout);
        }
        if (n >= 64) {
            const Stat s = run_case(set, n, seconds, all0, cores);
            std::printf("| %d | all 0 (full) | %u | %.3f | %.3f | %.3f | %.2f | - |\n", n, cores, s.p50, s.p95, s.p99, s.mean * 1000.0 / double(n));
            std::fflush(stdout);
        }
    }
    return 0;
}
