#include "engine/terrain_keep.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <cmath>
#include <set>
#include <utility>

using vibe::terrain_keep_phase;
using vibe::terrain_keep_places;
using Catch::Approx;

// The temporal upscaler's sample places (shaders/terrain_rm_resolve.frag): a kept pixel is only
// ever filled by a sample that falls on it, so a round of frames has to reach every one.

TEST_CASE("terrain keep: places per low-resolution pixel", "[terrain_keep]") {
    // kept picture = twice the render size; the terrain at 50 %, 75 % and 25 % of the render size
    CHECK(terrain_keep_places(3440, 860) == 4);      // 4 kept pixels under each: 4 x 4 places
    CHECK(terrain_keep_places(3440, 1290) == 3);     // 2.67: three each way
    CHECK(terrain_keep_places(3440, 430) == 4);      // 8: no more than four (a round would take 64 frames)
    CHECK(terrain_keep_places(3440, 1720) == 2);
    CHECK(terrain_keep_places(3440, 3440) == 2);     // never fewer than two
    CHECK(terrain_keep_places(3440, 0) == 4);        // (no division by zero)
}

TEST_CASE("terrain keep: a round visits every place once", "[terrain_keep]") {
    for (int n = 2; n <= 4; ++n) {
        std::set<std::pair<int, int>> seen;
        for (uint64_t f = 0; f < uint64_t(n * n); ++f) {
            const glm::vec2 p = terrain_keep_phase(n, f);
            // inside the pixel, on the centre of a cell of the n x n grid
            REQUIRE(std::abs(p.x) < 0.5f);
            REQUIRE(std::abs(p.y) < 0.5f);
            const float cx = (p.x + 0.5f) * float(n) - 0.5f, cy = (p.y + 0.5f) * float(n) - 0.5f;
            REQUIRE(cx == Approx(std::round(cx)).margin(1e-4));
            REQUIRE(cy == Approx(std::round(cy)).margin(1e-4));
            seen.insert({ int(std::lround(cx)), int(std::lround(cy)) });
        }
        CHECK(int(seen.size()) == n * n);
        // ... and the next round is the same again, wherever the frame counter stands
        for (uint64_t f : { uint64_t(0), uint64_t(5), uint64_t(1) << 40, ~uint64_t(0) - 40 }) {
            const glm::vec2 a = terrain_keep_phase(n, f), b = terrain_keep_phase(n, f + uint64_t(n * n));
            if (f + uint64_t(n * n) < f) continue;       // (the counter wrapped)
            // (n = 3: 2^64 is not a multiple of 9, so only frames well below the wrap repeat)
            CHECK(a.x == Approx(b.x));
            CHECK(a.y == Approx(b.y));
        }
    }
}

TEST_CASE("terrain keep: the samples are centred on the pixel", "[terrain_keep]") {
    // The offsets of a round add up to nothing: the kept picture does not sit aside from the
    // low-resolution one (it would show as the ground shifting when the upscaler is switched).
    for (int n = 2; n <= 4; ++n) {
        glm::vec2 sum(0.0f);
        for (uint64_t f = 0; f < uint64_t(n * n); ++f) sum += terrain_keep_phase(n, f);
        CHECK(sum.x == Approx(0.0f).margin(1e-4));
        CHECK(sum.y == Approx(0.0f).margin(1e-4));
    }
}

TEST_CASE("terrain keep: the first frames of a round spread over the pixel", "[terrain_keep]") {
    // n = 4: frames 0 .. 3 take one place in each quarter - a view that has just come to rest
    // has the detail of a half-size round after four frames, not one corner of the pixel.
    for (uint64_t base : { uint64_t(0), uint64_t(4), uint64_t(8), uint64_t(12) }) {
        std::set<std::pair<int, int>> quarters;
        for (uint64_t f = base; f < base + 4; ++f) {
            const glm::vec2 p = terrain_keep_phase(4, f);
            quarters.insert({ p.x > 0.0f ? 1 : 0, p.y > 0.0f ? 1 : 0 });
        }
        CHECK(quarters.size() == 4);
    }
    // out-of-range n falls back to the nearest rule
    CHECK(terrain_keep_phase(7, 3).x == Approx(terrain_keep_phase(4, 3).x));
    CHECK(terrain_keep_phase(1, 3).y == Approx(terrain_keep_phase(2, 3).y));
}
