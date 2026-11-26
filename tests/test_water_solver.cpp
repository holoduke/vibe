// The shallow-water solver's CPU side (engine/water_solver_cpu.h): the window, the fixed-step
// clock, the read-back's decoding and sampling, the foam rule's obstacle term, the spray rates.
#include "engine/water_solver_cpu.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

using namespace vibe;
using Catch::Matchers::WithinAbs;

TEST_CASE("the solver's window is whole cells with the eye in the middle", "[water][solver]") {
    const glm::ivec2 c = sw::corner_cells(glm::vec2(100.03f, -50.19f));
    // the eye's cell is the window's middle one
    REQUIRE(c.x + sw::kN / 2 == int(std::floor(100.03f / sw::kCell)));
    REQUIRE(c.y + sw::kN / 2 == int(std::floor(-50.19f / sw::kCell)));
    // a move inside one cell does not move it
    REQUIRE(sw::corner_cells(glm::vec2(100.05f, -50.19f)) == c);
}

TEST_CASE("the fixed-step clock takes 0.01 s steps, at most four a frame", "[water][solver]") {
    double acc = 0.0;
    REQUIRE(sw::take_steps(acc, 1.0 / 60.0) == 1);
    REQUIRE(sw::take_steps(acc, 1.0 / 60.0) == 2);        // (0.0067 + 0.0167 left over)
    int total = 0;
    for (int i = 0; i < 600; ++i) total += sw::take_steps(acc, 1.0 / 60.0);
    REQUIRE(std::abs(total - 1000) <= 1);                  // 10 s of frames, 1000 steps
    acc = 0.0;
    REQUIRE(sw::take_steps(acc, 0.2) == sw::kMaxSteps);    // a hitch: capped, the rest dropped
    REQUIRE(acc == 0.0);
}

TEST_CASE("the read-back decodes and samples like the shader packs", "[water][solver]") {
    sw::Readback rb;
    rb.n = 8; rb.cell = 0.5f; rb.corner = glm::ivec2(10, 20);
    rb.data.assign(size_t(rb.n) * rb.n * 2, 0u);
    for (int j = 0; j < 8; ++j)
        for (int i = 0; i < 8; ++i)
            if (i != 6) sw::pack_cell(rb.data, rb.n, i, j, 30.0f + 0.1f * float(i), glm::vec2(1.0f, -0.5f));
    rb.valid = true;
    float h = 0.0f, fade = 0.0f;
    glm::vec2 v(0.0f);
    // the centre of cell (2, 3) and halfway to (3, 3)
    REQUIRE(rb.sample((10 + 2 + 0.5f) * 0.5f, (20 + 3 + 0.5f) * 0.5f, h, v, &fade));
    REQUIRE_THAT(h, WithinAbs(30.2, 1e-4));
    REQUIRE_THAT(v.x, WithinAbs(1.0, 1e-3));
    REQUIRE_THAT(v.y, WithinAbs(-0.5, 1e-3));
    REQUIRE(rb.sample((10 + 3.0f) * 0.5f, (20 + 3 + 0.5f) * 0.5f, h, v));
    REQUIRE_THAT(h, WithinAbs(30.25, 1e-4));
    // next to a cell without water (column 6): no answer
    REQUIRE_FALSE(rb.sample((10 + 6.0f) * 0.5f, (20 + 3 + 0.5f) * 0.5f, h, v));
    // outside the window: none
    REQUIRE_FALSE(rb.sample(0.0f, 0.0f, h, v));
}

TEST_CASE("the edge blend covers the outer tenth of the window", "[water][solver]") {
    REQUIRE(sw::edge_weight(glm::vec2(0.5f)) == 0.0f);
    REQUIRE(sw::edge_weight(glm::vec2(0.39f, 0.5f)) == 0.0f);
    REQUIRE(sw::edge_weight(glm::vec2(0.95f, 0.5f)) > 0.4f);
    REQUIRE(sw::edge_weight(glm::vec2(1.0f, 0.5f)) == 1.0f);
}

TEST_CASE("obstacle foam: only fast shallow water running into higher bed", "[water][solver]") {
    REQUIRE(sw::foam_obstacle(0.5f, 0.3f, 0.4f, 0.4f) == 0.0f);    // too slow (< 0.65 m/s)
    REQUIRE(sw::foam_obstacle(1.5f, 1.2f, 0.4f, 0.4f) == 0.0f);    // too deep (> 0.8 m)
    REQUIRE(sw::foam_obstacle(1.5f, 0.3f, 0.4f, 0.0f) == 0.0f);    // nothing higher downstream
    const float f = sw::foam_obstacle(1.5f, 0.3f, 0.4f, 0.4f);
    REQUIRE(f > 0.0f);
    REQUIRE(f <= 0.008f);                                           // (the shader's cap a step)
}

TEST_CASE("spray rates", "[water][spray]") {
    REQUIRE(sw::spray_rate_fall(0.1f, 2.0f) == 0.0f);
    REQUIRE(sw::spray_rate_fall(2.0f, 2.0f) > sw::spray_rate_fall(0.5f, 2.0f));
    REQUIRE(sw::spray_rate_rock(0.8f, 1.0f) == 0.0f);
    REQUIRE(sw::spray_rate_rock(2.5f, 1.0f) > 0.0f);
    REQUIRE(sw::spray_rate_sea(8.0f) == 0.0f);
    REQUIRE(sw::spray_rate_sea(25.0f) > sw::spray_rate_sea(15.0f));
}
