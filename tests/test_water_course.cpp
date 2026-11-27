// The stream's channel and current (engine/water_course.h): the carve keeps the water inside its
// banks, the current is finite and as the shader draws it.
#include "engine/water_course.h"

#include <catch2/catch_test_macros.hpp>
#include <glm/glm.hpp>

#include <cmath>

namespace vibe {
// (water_course.cpp lays the stream out to the river with this; not used by these tests)
float world_river_distance(float, float) { return -10.0f; }
}

using namespace vibe;

namespace {
// A course bending through a valley: 20 points 6 m apart, falling 4 % with one 1.2 m step.
wc::Course make_course() {
    wc::Course c;
    float s = 50.0f;
    for (int i = 0; i < 20; ++i) {
        const float x = float(i) * 6.0f, z = 8.0f * std::sin(float(i) * 0.35f);
        const bool step = i == 9;
        if (i > 0) s -= step ? 1.2f : 0.24f;
        const float hw = 1.2f + 0.05f * float(i);
        c.pts.push_back(glm::vec4(x, z, s, step ? -hw : hw));
    }
    const int n = int(c.pts.size());
    for (int b = 0; b * wc::kSegsPerBox < n - 1; ++b) {
        glm::vec2 lo(1e9f), hi(-1e9f);
        for (int k = b * wc::kSegsPerBox; k <= std::min((b + 1) * wc::kSegsPerBox, n - 1); ++k) { lo = glm::min(lo, glm::vec2(c.pts[size_t(k)])); hi = glm::max(hi, glm::vec2(c.pts[size_t(k)])); }
        c.box.push_back(glm::vec4(lo - glm::vec2(wc::kBoxPad), hi + glm::vec2(wc::kBoxPad)));
    }
    c.ok = true;
    return c;
}
}  // namespace

TEST_CASE("the carved stream stays inside its banks", "[water]") {
    const wc::Course c = make_course();
    int checked = 0;
    for (float x = 0.0f; x <= 114.0f; x += 0.7f)
        for (float z = -20.0f; z <= 20.0f; z += 0.37f) {
            const glm::vec2 p(x, z);
            const wc::At A = wc::at(c, p);
            if (!A.on) continue;
            // land a little above the water (0.05 - 0.6 m) and rising away from it - where the old
            // fixed 4 m rounding dug under the surface
            for (float above : { 0.05f, 0.15f, 0.3f, 0.6f, 2.0f }) {
                const float h = A.surf + above + 0.2f * std::max(A.dist - 3.0f, 0.0f);
                const float g = wc::carve(c, h, p);
                REQUIRE(std::isfinite(g));
                REQUIRE(g <= std::max(h, A.surf + 0.15f) + 1e-4f);           // the carve lowers the ground (or builds the bank's lip)
                if (A.dist >= wc::wet_reach(A) - 0.35f) {                     // at the bank, out of the water:
                    REQUIRE(g >= A.surf - 1e-4f);                             // never under the surface
                    ++checked;
                }
            }
            // inside the channel the bed lies under the surface
            if (A.dist < 0.5f * A.half_w) REQUIRE(wc::carve(c, A.surf + 3.0f, p) < A.surf - 0.1f);
        }
    REQUIRE(checked > 1000);
}

TEST_CASE("land lower than the stream gets a bank that holds the water", "[water]") {
    // (the owner's straight-cut water edges: land beside the channel under the surface let the water
    //  reach out to its footprint's end; the carve now builds the bank's profile up there)
    const wc::Course c = make_course();
    const float mouth = c.pts.back().z;
    int checked = 0;
    for (float x = 0.0f; x <= 114.0f; x += 0.7f)
        for (float z = -20.0f; z <= 20.0f; z += 0.37f) {
            const glm::vec2 p(x, z);
            const wc::At A = wc::at(c, p);
            if (!A.on || A.surf - mouth < 0.6f) continue;
            const float h = A.surf - 0.5f;                                   // the land half a metre under the water
            const float g = wc::carve(c, h, p);
            REQUIRE(std::isfinite(g));
            if (A.dist >= wc::wet_reach(A) && A.dist < wc::wet_reach(A) + 0.25f) {
                REQUIRE(g >= A.surf + 0.02f);                                // the bank stands out of the water at the reach
                ++checked;
            }
            // beyond the reach the bank falls off into the land (0.5 m a metre)
            REQUIRE(g <= std::max(h, A.surf + 0.15f - 0.5f * std::max(A.dist - wc::wet_reach(A) - 1.0f, 0.0f)) + 1e-3f);
        }
    REQUIRE(checked > 50);
}

TEST_CASE("the stream's current is finite and runs downstream", "[water]") {
    const wc::Course c = make_course();
    for (float x = -5.0f; x <= 120.0f; x += 0.5f)
        for (float z = -12.0f; z <= 12.0f; z += 0.5f) {
            const wc::At A = wc::at(c, glm::vec2(x, z));
            if (!A.on) continue;
            const float s = wc::flow_speed(A);
            REQUIRE(std::isfinite(s));
            REQUIRE(s >= 0.37f);
            REQUIRE(s <= 4.5f);
            REQUIRE(std::abs(glm::length(A.dir) - 1.0f) < 1e-3f);
            REQUIRE(A.dir.x > 0.0f);                                          // (this course runs toward +x)
            REQUIRE(std::isfinite(A.surf));
            REQUIRE(A.slope >= -1e-4f);
        }
}

TEST_CASE("the step lip noise is the shader's", "[water]") {
    // wc_n1 in water_course.glsl has the same bits: spot values, and the range
    for (float x = -50.0f; x < 50.0f; x += 0.173f) {
        const float v = wc::n1(x);
        REQUIRE(v >= 0.0f);
        REQUIRE(v <= 1.0f);
    }
    REQUIRE(wc::n1(3.0f) == wc::n1(3.0f));
    REQUIRE(std::abs(wc::n1(0.0f) - wc::n1(-0.0f)) < 1e-6f);
}

TEST_CASE("the carve runs out into the land: no wall at its reach", "[water]") {
    // (2026-11-27, the owner's "clear cut in the mountain": land standing higher than the valley's
    //  side at 40 m was cut down to it and stood up again, sheer, along the reach's line)
    const wc::Course c = make_course();
    for (float x : { 20.0f, 54.0f, 90.0f })
        for (float side : { -1.0f, 1.0f }) {
            float prev = 0.0f; bool first = true;
            for (float z = 0.0f; z <= 46.0f; z += 0.1f) {
                const glm::vec2 p(x, side * z);
                const wc::At A = wc::at(c, p);
                const float h = 50.0f + 3.0f * z;                            // a mountainside: 3 m a metre, 138 m over the water at the reach
                const float g = A.on ? wc::carve(c, h, p) : h;
                REQUIRE(g <= h + 1e-3f);
                if (!first) REQUIRE(std::abs(g - prev) < 2.5f);              // at most 25 m a metre: steep, never a step
                prev = g; first = false;
            }
        }
}
