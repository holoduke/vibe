// The planet's charts (engine/globe.h): the rules the rest of the engine leans on when it
// keeps its coordinates small - see "Small coordinates" there.
#include "engine/globe.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <cmath>

using namespace vibe;
using Catch::Approx;

TEST_CASE("globe: a chart point goes to the planet and back") {
    const globe::Chart home;
    const globe::Chart n = globe::near_chart(home, 15000.0, -22000.0);
    for (const glm::dvec2 p : { glm::dvec2(0.0, 12288.0), glm::dvec2(1500.0, 9000.0), glm::dvec2(-3000.0, 15000.0) }) {
        const glm::dvec2 q = globe::to_chart(n, globe::to_planet(n, p.x, p.y));
        CHECK(q.x == Approx(p.x).margin(1e-6));
        CHECK(q.y == Approx(p.y).margin(1e-6));
    }
}

TEST_CASE("globe: a near chart keeps home's coordinates, less whole tiles") {
    const globe::Chart home;
    // Where the player stands when the engine leaves home's chart.
    const glm::dvec2 at(12950.0, -3777.0);
    const globe::Chart n = globe::near_chart(home, at.x, at.y);
    CHECK_FALSE(n.home);
    CHECK_FALSE(n.high);
    const glm::dvec2 local = globe::to_chart(n, globe::to_planet(home, at.x, at.y));
    const glm::dvec2 shift = at - local;
    // A whole number of tiles each way, and the one the chart reports.
    CHECK(shift.x / globe::kTile == Approx(std::round(shift.x / globe::kTile)).margin(1e-9));
    CHECK(shift.y / globe::kTile == Approx(std::round(shift.y / globe::kTile)).margin(1e-9));
    CHECK(double(n.tile.x) == Approx(shift.x / globe::kTile).margin(1e-9));
    CHECK(double(n.tile.y) == Approx(shift.y / globe::kTile).margin(1e-9));
    // The player is back within half a tile of the anchor: small coordinates.
    CHECK(std::abs(local.x) <= globe::kTile * 0.5 + 1e-6);
    CHECK(std::abs(local.y - globe::kNearOffZ) <= globe::kTile * 0.5 + 1e-6);
    // And so is the land round him: a point 500 m off keeps its coordinates to a millimetre.
    const glm::dvec2 off(350.0, -360.0);
    const glm::dvec2 l2 = globe::to_chart(n, globe::to_planet(home, at.x + off.x, at.y + off.y));
    CHECK(l2.x == Approx(local.x + off.x).margin(1e-3));
    CHECK(l2.y == Approx(local.y + off.y).margin(1e-3));
}

TEST_CASE("globe: near chart to near chart is a shift by whole tiles too") {
    const globe::Chart home;
    const globe::Chart a = globe::near_chart(home, 20000.0, 30000.0);
    // The player has walked to the edge of its reach.
    const glm::dvec2 p(globe::kNearReach + 5.0, globe::kNearOffZ - 900.0);
    const globe::Chart b = globe::near_chart(a, p.x, p.y);
    const glm::dvec2 q = globe::to_chart(b, globe::to_planet(a, p.x, p.y));
    const glm::dvec2 shift = p - q;
    CHECK(shift.x == Approx(globe::kTile).margin(1e-6));           // one tile along x
    CHECK(shift.y == Approx(0.0).margin(1e-6));                     // none along z
    CHECK(b.tile.x == a.tile.x + 1);
    CHECK(b.tile.y == a.tile.y);
    CHECK(std::abs(q.x) < globe::kNearReach);
    // A tree 200 m away stands at the same coordinates, shifted.
    const glm::dvec2 t(p.x - 150.0, p.y + 130.0);
    const glm::dvec2 tq = globe::to_chart(b, globe::to_planet(a, t.x, t.y));
    CHECK(tq.x == Approx(t.x - shift.x).margin(1e-3));
    CHECK(tq.y == Approx(t.y - shift.y).margin(1e-3));
}

TEST_CASE("globe: home coordinates from a near chart (the ring's mapping)") {
    const globe::Chart home;
    const glm::dvec2 at(-41000.0, 27000.0);                         // 49 km from home
    const globe::Chart n = globe::near_chart(home, at.x, at.y);
    const globe::Ring rg = globe::ring_of(n);
    REQUIRE(rg.on);
    const glm::dvec2 local = globe::to_chart(n, globe::to_planet(home, at.x, at.y));
    // Exact against the linear mapping the shaders use, across the chart's reach.
    for (const glm::dvec2 d : { glm::dvec2(0.0), glm::dvec2(3000.0, 0.0), glm::dvec2(-2500.0, 3500.0), glm::dvec2(0.0, -3600.0) }) {
        const glm::dvec2 p = local + d;
        const glm::dvec2 exact = globe::home_az(globe::to_planet(n, p.x, p.y));
        const glm::dvec2 lin = rg.to_home(p);
        CHECK(glm::length(exact - lin) < 0.02);                     // under 2 cm
        const glm::dvec2 back = rg.to_chart(lin);
        CHECK(glm::length(back - p) < 1e-6);
    }
    // Beyond the flat formula's reach there is no ring; home's chart has none either.
    CHECK_FALSE(globe::ring_of(globe::near_chart(home, 400000.0, 0.0)).on);
    CHECK_FALSE(globe::ring_of(home).on);
}

TEST_CASE("globe: the flat formula's lattice is the home position, split") {
    // Octave i of the FBM samples at h0 * 0.003 * (2 m2)^i: cell + position in the cell.
    const glm::dvec2 h0(63211.37, -48817.9);
    float lat[globe::kFlatOctaves][4];
    globe::flat_lattice(h0, lat);
    glm::dvec2 q = h0 * 0.003;
    for (int i = 0; i < 12; ++i) {
        const glm::dvec2 cell = glm::floor(q);
        CHECK(double(lat[i][0]) == Approx(q.x - cell.x).margin(1e-5));
        CHECK(double(lat[i][1]) == Approx(q.y - cell.y).margin(1e-5));
        CHECK(lat[i][0] >= 0.0f);
        CHECK(lat[i][0] < 1.0f);
        // (cells this near home are well inside the wrap: stored as they are)
        CHECK(double(lat[i][2]) == Approx(cell.x).margin(1e-6));
        CHECK(double(lat[i][3]) == Approx(cell.y).margin(1e-6));
        q = glm::dvec2(0.8 * q.x + 0.6 * q.y, -0.6 * q.x + 0.8 * q.y) * 2.0;
    }
}

TEST_CASE("globe: a high chart and a site's chart keep their own origins") {
    const globe::Chart home;
    const globe::Chart h = globe::high_chart(home, 150000.0, 0.0);
    CHECK(h.high);
    const glm::dvec2 p = globe::to_chart(h, globe::to_planet(home, 150000.0, 0.0));
    CHECK(p.x == Approx(globe::kFarX).margin(1e-6));
    CHECK(p.y == Approx(0.0).margin(1e-6));
    // From a high chart a near chart starts afresh at its anchor.
    const globe::Chart n = globe::near_chart(h, p.x + 2000.0, p.y - 500.0);
    const glm::dvec2 q = globe::to_chart(n, globe::to_planet(h, p.x + 2000.0, p.y - 500.0));
    CHECK(q.x == Approx(0.0).margin(1e-6));
    CHECK(q.y == Approx(globe::kNearOffZ).margin(1e-6));
    CHECK(n.tile == glm::ivec2(0));
}
