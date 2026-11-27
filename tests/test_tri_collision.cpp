#include "game/tri_collision.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

using namespace vibe::tricol;
using Catch::Approx;

namespace {
// A square (two triangles) through the four corners.
void quad(Mesh& m, glm::vec3 a, glm::vec3 b, glm::vec3 c, glm::vec3 d) { m.add(a, b, c); m.add(a, c, d); }
}

TEST_CASE("closest point on a triangle: inside, on an edge, at a corner") {
    const glm::vec3 a(0, 0, 0), b(2, 0, 0), c(0, 0, 2);
    REQUIRE(closest_on_triangle(glm::vec3(0.5f, 3, 0.5f), a, b, c).y == Approx(0.0f));
    REQUIRE(closest_on_triangle(glm::vec3(0.5f, 3, 0.5f), a, b, c).x == Approx(0.5f));
    const glm::vec3 e = closest_on_triangle(glm::vec3(1, 0, -1), a, b, c);
    REQUIRE(e.x == Approx(1.0f)); REQUIRE(e.z == Approx(0.0f));
    const glm::vec3 k = closest_on_triangle(glm::vec3(-1, 0, -1), a, b, c);
    REQUIRE(glm::length(k - a) == Approx(0.0f).margin(1e-6));
}

TEST_CASE("a segment through a triangle is at distance 0") {
    glm::vec3 s, t;
    const float d = segment_triangle(glm::vec3(0.5f, -1, 0.5f), glm::vec3(0.5f, 1, 0.5f),
                                     glm::vec3(0, 0, 0), glm::vec3(2, 0, 0), glm::vec3(0, 0, 2), s, t);
    REQUIRE(d == Approx(0.0f).margin(1e-6));
    REQUIRE(s.y == Approx(0.0f).margin(1e-5));
}

TEST_CASE("a walker sunk into a flat rock top is lifted onto it and stands") {
    Mesh m;
    quad(m, glm::vec3(-5, 2, -5), glm::vec3(5, 2, -5), glm::vec3(5, 2, 5), glm::vec3(-5, 2, 5));
    m.build();
    const float r = 0.3f;
    glm::vec3 a(0, 2.0f + r - 0.1f, 0), b = a + glm::vec3(0, 1.2f, 0), vel(1, -3, 0);
    const Contact c = resolve_capsule({ &m }, a, b, r, vel, 0.55f);
    REQUIRE(c.ground);
    REQUIRE(a.y == Approx(2.0f + r).margin(1e-3));
    REQUIRE(a.x == Approx(0.0f).margin(1e-5));      // (straight up, no sideways shove)
    REQUIRE(vel.y == Approx(0.0f));
    REQUIRE(vel.x == Approx(1.0f));
}

TEST_CASE("a walker on a walkable slope is lifted straight up, not slid down it") {
    Mesh m;   // 30 degrees, rising toward +x
    const float k = std::tan(glm::radians(30.0f));
    quad(m, glm::vec3(-5, -5 * k, -5), glm::vec3(5, 5 * k, -5), glm::vec3(5, 5 * k, 5), glm::vec3(-5, -5 * k, 5));
    m.build();
    const float r = 0.3f;
    glm::vec3 a(1, 1 * k + r * 0.8f, 0), b = a + glm::vec3(0, 1.2f, 0), vel(0);
    const Contact c = resolve_capsule({ &m }, a, b, r, vel, 0.55f);
    REQUIRE(c.ground);
    REQUIRE(a.x == Approx(1.0f).margin(1e-5));
    REQUIRE(a.z == Approx(0.0f).margin(1e-5));
}

TEST_CASE("a steep rock face pushes the walker out sideways and stops what runs into it") {
    Mesh m;   // a vertical face at x = 1, facing -x
    quad(m, glm::vec3(1, 0, -5), glm::vec3(1, 0, 5), glm::vec3(1, 5, 5), glm::vec3(1, 5, -5));
    m.build();
    const float r = 0.3f;
    glm::vec3 a(0.85f, 1.0f, 0), b = a + glm::vec3(0, 1.2f, 0), vel(4, 0, 2);
    const Contact c = resolve_capsule({ &m }, a, b, r, vel, 0.55f);
    REQUIRE_FALSE(c.ground);
    REQUIRE(a.x == Approx(1.0f - r).margin(1e-4));
    REQUIRE(a.y == Approx(1.0f).margin(1e-5));
    REQUIRE(vel.x == Approx(0.0f).margin(1e-5));     // (into the face: gone)
    REQUIRE(vel.z == Approx(2.0f));                  // (along it: kept)
}

TEST_CASE("an overhang stops a jump: the head is pushed down") {
    Mesh m;   // a ceiling at y = 3
    quad(m, glm::vec3(-5, 3, -5), glm::vec3(-5, 3, 5), glm::vec3(5, 3, 5), glm::vec3(5, 3, -5));
    m.build();
    const float r = 0.3f;
    glm::vec3 a(0, 1.6f, 0), b(0, 2.8f, 0), vel(0, 5, 0);
    resolve_capsule({ &m }, a, b, r, vel, 0.55f);
    REQUIRE(b.y == Approx(3.0f - r).margin(1e-3));
    REQUIRE(vel.y == Approx(0.0f).margin(1e-5));
}

TEST_CASE("floor_below finds the highest walkable face under a point, not a steep one") {
    Mesh m;
    quad(m, glm::vec3(-5, 1, -5), glm::vec3(5, 1, -5), glm::vec3(5, 1, 5), glm::vec3(-5, 1, 5));   // low shelf
    quad(m, glm::vec3(-1, 2, -1), glm::vec3(1, 2, -1), glm::vec3(1, 2, 1), glm::vec3(-1, 2, 1));   // a step on it
    quad(m, glm::vec3(3, 0, -5), glm::vec3(3, 0, 5), glm::vec3(3, 8, 5), glm::vec3(3, 8, -5));     // a wall (not a floor)
    m.build(1.0f);
    float y = 0.0f; glm::vec3 n;
    REQUIRE(floor_below({ &m }, 0.0f, 0.0f, 2.5f, 0.0f, 0.55f, y, n));
    REQUIRE(y == Approx(2.0f));
    REQUIRE(n.y == Approx(1.0f));
    REQUIRE(floor_below({ &m }, 3.0f, 0.0f, 9.0f, 0.0f, 0.55f, y, n));
    REQUIRE(y == Approx(1.0f));
    REQUIRE_FALSE(floor_below({ &m }, 0.0f, 0.0f, 0.5f, 0.0f, 0.55f, y, n));
}

TEST_CASE("a walker glides round a boulder's round body and stands on its top") {
    // a boulder: the box 2 m wide, 1 m high over the ground at 0 (world_props add_rock_ex)
    const Round o = round_in_box(glm::vec3(-1, -1, -1), glm::vec3(1, 1, 1));
    REQUIRE(o.r == Approx(1.0f));
    REQUIRE(o.b.y == Approx(0.0f));
    const std::vector<Round> rs{ o };
    // walking +x into it a little to the side of its middle: pushed out sideways, not stopped
    const float r = 0.3f;
    glm::vec3 a(-1.05f, 0.3f, 0.5f), b(-1.05f, 1.5f, 0.5f), vel(3, 0, 0);
    Contact c;
    resolve_capsule_rounds(rs, a, b, r, vel, 0.55f, c);
    REQUIRE(c.touched);
    REQUIRE_FALSE(c.ground);
    REQUIRE(vel.z != Approx(0.0f));                  // (it turns along the stone)
    REQUIRE(glm::length(glm::vec2(a.x, a.z)) > 1.2f);
    // its top: walkable near the middle, not at the side
    float y = 0.0f; glm::vec3 n;
    REQUIRE(floor_below_rounds(rs, 0.1f, 0.0f, 2.0f, 0.0f, 0.55f, y, n));
    REQUIRE(y == Approx(std::sqrt(1.0f - 0.01f)).margin(1e-4));
    REQUIRE_FALSE(floor_below_rounds(rs, 0.95f, 0.0f, 2.0f, 0.0f, 0.55f, y, n));
}
