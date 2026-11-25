#pragma once
// What grows along the river: one rule, used twice.
//
// The river is walked in stations kStep apart along its real middle (the course of
// terrain.h world_river_course, moved by the same wander the formula cuts it with). At each
// station and on each side the bank is looked for - the first place outward where the bed comes
// up to the water - and from it:
//
//   * water lilies on the shallows off the bank, in rafts: mostly on the INSIDE of a bend, where
//     the water is slow (nine stations in ten there, one in ten on the outside where the
//     current runs against the bank, a third on a straight reach); never further out than the
//     outer half of the stream (the middle is the fast water), never where a pad would touch
//     the bank, never under a bridge, and not in the river's mouth (below the coast road's bridge
//     it is the sea's water);
//   * reeds in beds at the waterline, three stations in four, a second stand reaching into
//     the water on the inside of a bend;
//   * up the bank, thinning out by 7 m from the water: ferns, long grass, and in drifts
//     buttercups (the wet foot of the bank), nettles, foxgloves and thistles (higher up).
//
// Everything about a plant comes from a hash of (station, side): the same river every run,
// whoever asks. scatter.cpp asks for the stations inside the home tile at start-up (their
// plants go through the small plants' GPU culling); vk_engine/river_plants.cpp asks for the
// rest as the player comes near and frees them again (streamed).

#include "engine/scatter.h"
#include "engine/terrain.h"

#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

namespace vibe::river_rule {

constexpr float kStep = 7.0f;             // metres of river a station
constexpr float kTidal = 1640.0f;         // east of this (the coast road's bridge) the river is the sea's: no lilies, and reeds only to kSea
constexpr float kSea = 1760.0f;
constexpr float kHome = 1000.0f;          // a station whose middle lies within this of the origin in x and z is the home tile's

struct Station { glm::vec2 mid, dir; float bend; };      // bend: > 0 the river turns toward +nrm (nrm = (-dir.y, dir.x)), about 1 for a 30 m radius

// The river's length in stations.
inline int station_count() {
    const glm::vec2* rp = nullptr;
    const int rn = world_river_course(rp);
    float len = 0.0f;
    for (int i = 1; i < rn; ++i) len += glm::distance(rp[i - 1], rp[i]);
    return int(len / kStep);
}
// The stream's middle `s` metres down the course.
inline glm::vec2 middle(float s) {
    const glm::vec2* rp = nullptr;
    const int rn = world_river_course(rp);
    glm::vec2 cp = rp[rn - 1];
    for (int i = 1; i < rn; ++i) {
        const float len = glm::distance(rp[i - 1], rp[i]);
        if (s <= len || i == rn - 1) { cp = glm::mix(rp[i - 1], rp[i], std::clamp(s / std::max(len, 1e-3f), 0.0f, 1.0f)); break; }
        s -= len;
    }
    return cp - world_river_wander(cp);
}
inline Station station(int k) {
    Station st;
    const float s = float(k) * kStep;
    st.mid = middle(s);
    const glm::vec2 a = middle(std::max(s - 12.0f, 0.0f)), b = middle(s + 12.0f);
    const glm::vec2 d0 = st.mid - a, d1 = b - st.mid;
    st.dir = glm::normalize(b - a + glm::vec2(1e-5f, 0.0f));
    const float l0 = glm::length(d0), l1 = glm::length(d1);
    st.bend = (l0 > 1.0f && l1 > 1.0f) ? (d0.x * d1.y - d0.y * d1.x) / (l0 * l1) * 30.0f / (0.5f * (l0 + l1)) : 0.0f;
    return st;
}
inline bool in_home(glm::vec2 mid) { return std::abs(mid.x) < kHome && std::abs(mid.y) < kHome; }

struct Counts { int lilies = 0, reeds = 0, bank = 0; };

inline uint32_t hash_u(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t h = a * 0x9E3779B1u ^ (b + 0x7F4A7C15u) * 0x85EBCA6Bu ^ (c + 0x165667B1u) * 0xC2B2AE35u;
    h ^= h >> 15; h *= 0x2C1B3C6Du; h ^= h >> 12; h *= 0x297A2D39u; h ^= h >> 15;
    return h;
}

// The plants of station k, appended to per[species] (ScatterSet::kPlantSpecies lists).
//   h_at(x, z)            the ground (the river's bed under the water)
//   blocked(x, z, water)  nothing here: a building, a road; for `water` also under a bridge
//   tufts                 long grass (species 15) on the bank; without it ferns take its place
template <class H, class Blocked>
void plants_at(int k, float wl, H&& h_at, Blocked&& blocked, bool tufts, std::vector<ScatterInstance>* per, Counts& n_out) {
    const Station st = station(k);
    const glm::vec2 dir = st.dir, nrm(-dir.y, dir.x);
    const float two_pi = glm::two_pi<float>();
    for (int side = 0; side < 2; ++side) {
        const float sd = side ? 1.0f : -1.0f;
        uint32_t rs = hash_u(uint32_t(k), uint32_t(side), 0x51u) | 1u;
        auto u = [&rs]() { rs ^= rs << 13; rs ^= rs >> 17; rs ^= rs << 5; return float(rs >> 8) / 16777216.0f; };
        // out from the middle to where the bed comes up to the water
        float edge = -1.0f;
        for (float o = 2.0f; o < 45.0f; o += 0.5f) {
            const glm::vec2 q = st.mid + nrm * (sd * o);
            if (h_at(q.x, q.y) > wl - 0.05f) { edge = o; break; }
        }
        if (edge < 3.0f) continue;                          // (no water here: a bridge's fill, the spring)
        const glm::vec2 e = st.mid + nrm * (sd * edge);
        if (blocked(e.x, e.y, false)) continue;
        // which side of a bend this is: 1 inside (slow water), -1 outside, 0 a straight reach
        const float turn = std::clamp(st.bend * sd, -1.0f, 1.0f);
        const float inside = glm::smoothstep(0.15f, 0.6f, turn), outside = glm::smoothstep(0.15f, 0.6f, -turn);
        // -- lilies --
        {
            const float chance = glm::mix(glm::mix(0.34f, 0.90f, inside), 0.10f, outside);
            const bool patch = u() < chance && st.mid.x < kTidal, pink = u() < 0.3f;
            const int n = patch ? int((10.0f + 22.0f * u()) * (1.0f + 0.7f * inside)) : 0;
            const float reach = glm::mix(3.2f, 6.0f, inside);            // how far off the bank the patch goes
            for (int i = 0; i < n; ++i) {
                const float off = 0.8f + reach * u(), al = (u() - 0.5f) * 8.0f, sc = 0.8f + 0.7f * u(), bare = u(), rot = u();
                if (edge - off < 0.5f * edge) continue;                  // (the middle of the stream: fast water)
                const float wx = e.x - nrm.x * sd * off + dir.x * al, wz = e.y - nrm.y * sd * off + dir.y * al;
                const float h = h_at(wx, wz);
                if (h > wl - 0.12f || h < wl - 1.15f) continue;          // 12 cm to a little over a metre of water
                if (blocked(wx, wz, true)) continue;
                bool afloat = true;                                      // (a raft reaches ~0.6 m x its size: no pad on the bank)
                for (int q = 0; q < 8 && afloat; ++q) {
                    const float qa = float(q) * 0.7853982f;
                    if (h_at(wx + std::cos(qa) * 0.72f * sc, wz + std::sin(qa) * 0.72f * sc) > wl - 0.06f) afloat = false;
                }
                if (!afloat) continue;
                ScatterInstance si{};
                si.pos_scale = glm::vec4(wx, wl, wz, sc);
                // (flower colour; a raft without a flower is marked by a dark tint - the shader hides its petals)
                glm::vec3 col = pink ? glm::vec3(0.92f, 0.55f, 0.66f) : glm::vec3(0.95f, 0.94f, 0.88f);
                if (bare < 0.55f) col = glm::vec3(0.02f);
                si.rot_tint = glm::vec4(rot * two_pi, col.x, col.y, col.z);
                per[16].push_back(si);
                ++n_out.lilies;
            }
        }
        // -- reeds --
        {
            const bool bed = float(hash_u(uint32_t(std::floor(e.x * 0.06f) + 4096.0f), uint32_t(std::floor(e.y * 0.06f) + 4096.0f), 53u) >> 8) / 16777216.0f > 0.25f;
            const int nb = bed ? 9 + int(u() * 14.0f) : 0, n = st.mid.x < kSea ? nb + int(inside * (8.0f + 10.0f * u())) : 0;
            for (int i = 0; i < n; ++i) {
                // (the bed itself round the waterline; the inside of a bend's second stand out in the shallows)
                const bool wade = i >= nb;
                const float out = wade ? -(0.6f + 2.2f * u()) : (u() * 1.9f - 1.1f), al = (u() - 0.5f) * 7.6f, sc = 0.75f + 0.6f * u(), cu = u(), rot = u();
                const float wx = e.x + nrm.x * sd * out + dir.x * al, wz = e.y + nrm.y * sd * out + dir.y * al;
                const float h = h_at(wx, wz);
                if (h < wl - (wade ? 0.55f : 0.40f) || h > wl + 0.30f) continue;
                if (blocked(wx, wz, true)) continue;
                ScatterInstance si{};
                si.pos_scale = glm::vec4(wx, h - 0.03f, wz, sc);
                const glm::vec3 col = glm::mix(glm::vec3(0.32f, 0.38f, 0.16f), glm::vec3(0.42f, 0.40f, 0.20f), cu);
                si.rot_tint = glm::vec4(rot * two_pi, col.x, col.y, col.z);
                per[7].push_back(si);
                ++n_out.reeds;
            }
        }
        // -- up the bank --
        {
            // (drifts: what flowers here is decided for some 25 m of bank at a time)
            const float drift = float(hash_u(uint32_t(k / 4), uint32_t(side), 0x77u) >> 8) / 16777216.0f;
            const int n = 9 + int(u() * 11.0f);
            for (int i = 0; i < n; ++i) {
                const float o2 = u() * u(), al = (u() - 0.5f) * 7.6f, pick = u(), s0 = u(), c0 = u(), rot = u();
                const float out = 0.5f + 6.5f * o2;
                const float wx = e.x + nrm.x * sd * out + dir.x * al, wz = e.y + nrm.y * sd * out + dir.y * al;
                if (blocked(wx, wz, false)) continue;
                const float h = h_at(wx, wz);
                if (h < wl + 0.12f || h > wl + 4.0f) continue;
                const float dx = h_at(wx + 1.0f, wz) - h_at(wx - 1.0f, wz), dz = h_at(wx, wz + 1.0f) - h_at(wx, wz - 1.0f);
                if (2.0f / std::sqrt(dx * dx + dz * dz + 4.0f) < 0.55f) continue;
                const float up = h - wl;
                int sp;
                if (pick < 0.40f) sp = 0;                                                    // fern
                else if (pick < 0.64f) sp = tufts ? 15 : 0;                                  // long grass
                else if (drift < 0.30f) sp = up < 1.4f ? 5 : 0;                              // buttercups at the wet foot
                else if (drift < 0.55f) sp = 11;                                             // nettles
                else if (drift < 0.72f) sp = up > 0.8f ? 12 : 0;                             // foxgloves higher up
                else if (drift < 0.82f) sp = up > 0.8f && pick > 0.88f ? 13 : 0;             // a thistle
                else sp = 0;
                ScatterInstance si{};
                glm::vec3 col; float sc;
                switch (sp) {
                case 0:  sc = 0.7f + 0.7f * s0; col = glm::vec3(0.10f, 0.22f, 0.06f) * (0.8f + 0.4f * c0); break;
                case 15: sc = 0.9f + 0.8f * s0; col = glm::mix(glm::vec3(0.085f, 0.20f, 0.045f), glm::vec3(0.20f, 0.30f, 0.07f), c0); break;
                case 5:  sc = 1.1f + 0.6f * s0; col = glm::vec3(0.95f, 0.76f, 0.06f) * (0.9f + 0.2f * c0); break;
                case 12: sc = 0.8f + 0.45f * s0; col = glm::mix(glm::vec3(0.80f, 0.32f, 0.62f), glm::vec3(0.62f, 0.25f, 0.70f), c0); break;
                case 13: sc = 0.85f + 0.4f * s0; col = glm::vec3(0.58f, 0.22f, 0.62f) * (0.85f + 0.3f * c0); break;
                default: sc = 0.8f + 0.45f * s0; col = glm::vec3(0.6f); break;               // (nettle: leaf only, tint unused)
                }
                si.pos_scale = glm::vec4(wx, h - 0.02f, wz, sc);
                si.rot_tint = glm::vec4(rot * two_pi, col.x, col.y, col.z);
                per[sp].push_back(si);
                ++n_out.bank;
            }
        }
    }
}

}  // namespace vibe::river_rule
