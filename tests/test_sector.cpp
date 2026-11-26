// Sectors (engine/sector.h): the chart-independent address of a piece of the planet that
// height tiles, region content and - later - servers key on.
#include "engine/sector.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

using namespace vibe;
using Catch::Approx;

namespace {
// A few places: home, near home, 5000 km from home, near a face's edge, near a cube corner.
std::vector<glm::dvec3> places() {
    const globe::Chart home;
    return {
        glm::dvec3(0.0, 1.0, 0.0),
        globe::to_planet(home, 310.0, -120.0),
        globe::to_planet(home, 4.1e6, 2.9e6),               // 5022 km from home
        glm::normalize(glm::dvec3(1.0, 1.0002, 0.2)),       // just inside face 2's edge to face 0
        glm::normalize(glm::dvec3(1.0, 1.0001, 0.9999)),    // by the corner of faces 0, 2, 4
        glm::normalize(glm::dvec3(-0.3, -0.2, -1.0)),       // the far side
    };
}
bool has(const sector::Id* v, int n, sector::Id x) { return std::find(v, v + n, x) != v + n; }
}  // namespace

TEST_CASE("sector: an id holds its parts, and its name reads back") {
    const sector::Id id = sector::make(4, 14, 12345, 16383);
    REQUIRE(id.valid());
    CHECK(id.face() == 4);
    CHECK(id.level() == 14);
    CHECK(id.i() == 12345u);
    CHECK(id.j() == 16383u);
    CHECK(sector::name(id) == "4-14-12345-16383");
    sector::Id back;
    REQUIRE(sector::parse("4-14-12345-16383", back));
    CHECK(back == id);
    CHECK_FALSE(sector::make(6, 3, 0, 0).valid());
    CHECK_FALSE(sector::make(2, 3, 8, 0).valid());          // level 3 has 8 cells a side: 0..7
    CHECK_FALSE(sector::make(2, 29, 0, 0).valid());
    CHECK_FALSE(sector::parse("2-3-8-0", back));
    CHECK_FALSE(sector::parse("2-3-1", back));
    CHECK_FALSE(sector::parse("2-3-1-1x", back));
    CHECK_FALSE(sector::Id{}.valid());
}

TEST_CASE("sector: home is the middle of face 2, east is a, south is b") {
    const sector::FacePos p = sector::face_pos(glm::dvec3(0.0, 1.0, 0.0));
    CHECK(p.face == 2);
    CHECK(p.a == Approx(0.5));
    CHECK(p.b == Approx(0.5));
    const globe::Chart home;
    const sector::FacePos e = sector::face_pos(globe::to_planet(home, 1000.0, 0.0));
    const sector::FacePos s = sector::face_pos(globe::to_planet(home, 0.0, 1000.0));
    CHECK(e.a > 0.5); CHECK(e.b == Approx(0.5));
    CHECK(s.b > 0.5); CHECK(s.a == Approx(0.5));
    // A level-14 cell at home is 610 m across.
    const glm::dvec2 sz = sector::size_m(sector::cell_at(glm::dvec3(0.0, 1.0, 0.0), 14));
    CHECK(sz.x == Approx(sector::kFaceM / 16384.0).epsilon(1e-4));
    CHECK(sz.y == Approx(sz.x).epsilon(1e-6));
}

TEST_CASE("sector: a point goes to its cell and back, anywhere on the planet") {
    for (const glm::dvec3& G : places())
        for (int level : { 0, 8, 14, 20, 28 }) {
            glm::dvec2 fr;
            const sector::Id id = sector::cell_at(G, level, &fr);
            REQUIRE(id.valid());
            CHECK(id.level() == level);
            CHECK(fr.x >= 0.0); CHECK(fr.x <= 1.0); CHECK(fr.y >= 0.0); CHECK(fr.y <= 1.0);
            // The point of the cell at that fraction is the point we started from: to a
            // hundredth of a millimetre on the ground, 5000 km from home as at home.
            const glm::dvec3 back = sector::point(id, fr.x, fr.y);
            CHECK(globe::kR * glm::distance(back, G) < 1e-5);
            // ... and the cell's own middle lies in the cell.
            glm::dvec2 mid;
            CHECK(sector::cell_at(sector::centre(id), level, &mid) == id);
            if (level < 28) { CHECK(mid.x == Approx(0.5).margin(1e-6)); CHECK(mid.y == Approx(0.5).margin(1e-6)); }
        }
}

TEST_CASE("sector: the hierarchy") {
    for (const glm::dvec3& G : places()) {
        const sector::Id id = sector::cell_at(G, 14);
        const sector::Id up = sector::parent(id);
        CHECK(up == sector::cell_at(G, 13));
        CHECK(sector::ancestor(id, 8) == sector::cell_at(G, 8));
        CHECK(sector::ancestor(id, 14) == id);
        CHECK_FALSE(sector::ancestor(id, 15).valid());
        CHECK(sector::contains(sector::cell_at(G, 8), id));
        CHECK(sector::contains(id, id));
        CHECK_FALSE(sector::contains(id, up));
        bool found = false;
        for (int k = 0; k < 4; ++k) {
            const sector::Id c = sector::child(up, k);
            CHECK(sector::parent(c) == up);
            found = found || c == id;
        }
        CHECK(found);
    }
    CHECK_FALSE(sector::parent(sector::make(1, 0, 0, 0)).valid());
}

TEST_CASE("sector: neighbours, also across a face's edge and at a cube corner") {
    for (const glm::dvec3& G : places())
        for (int level : { 6, 14 }) {
            const sector::Id id = sector::cell_at(G, level);
            sector::Id nb[8];
            const int n = sector::neighbours(id, nb);
            CHECK(n >= 7);
            CHECK(n <= 8);
            const double r = sector::radius_m(id);
            for (int k = 0; k < n; ++k) {
                REQUIRE(nb[k].valid());
                CHECK(nb[k] != id);
                CHECK(nb[k].level() == level);
                // It touches: the middles are nearer than two bounding circles.
                CHECK(globe::ground_distance(sector::centre(id), sector::centre(nb[k])) < 2.0 * r * 1.3);
                // And it is mutual.
                sector::Id back[8];
                const int m = sector::neighbours(nb[k], back);
                CHECK(has(back, m, id));
            }
        }
    // The last cell of face 2 along a: its +a neighbour is on face 0, and steps back.
    const sector::Id edge = sector::make(2, 14, 16383, 9000);
    const sector::Id over = sector::neighbour(edge, 1, 0);
    REQUIRE(over.valid());
    CHECK(over.face() == 0);
    sector::Id back[8];
    const int m = sector::neighbours(over, back);
    CHECK(has(back, m, edge));
    // The two share an edge: the middle of ours' far edge lies on theirs.
    const glm::dvec3 mid = sector::point(edge, 1.0, 0.5);
    CHECK(globe::ground_distance(mid, sector::centre(over)) < sector::radius_m(over));
    // A cube corner: three cells meet, so seven neighbours.
    sector::Id cn[8];
    CHECK(sector::neighbours(sector::make(2, 6, 63, 63), cn) == 7);
    // Within a face: eight, the straight ones first.
    CHECK(sector::neighbours(sector::make(2, 6, 30, 30), cn) == 8);
    CHECK(cn[0] == sector::make(2, 6, 29, 30));
    CHECK(cn[1] == sector::make(2, 6, 31, 30));
    CHECK(cn[2] == sector::make(2, 6, 30, 29));
    CHECK(cn[3] == sector::make(2, 6, 30, 31));
}

TEST_CASE("sector: the sectors within reach hold every point within reach") {
    for (const glm::dvec3& G : places()) {
        const globe::Chart c = globe::far_chart(glm::dmat3(glm::normalize(glm::cross(glm::dvec3(0.3, 0.1, 0.9), G)), G,
                                                           glm::cross(glm::normalize(glm::cross(glm::dvec3(0.3, 0.1, 0.9), G)), G)));
        const double radius = 2300.0;
        std::vector<sector::Id> cells;
        sector::cover(G, radius, 14, cells);
        CHECK(cells.size() >= 16);
        CHECK(cells.size() <= 120);
        for (int k = 0; k < 96; ++k) {
            const double ang = 0.37 * k, d = radius * (0.999 - 0.01 * (k % 90));
            const glm::dvec3 P = globe::to_planet(c, c.org.x + d * std::cos(ang), c.org.y + d * std::sin(ang));
            const sector::Id id = sector::cell_at(P, 14);
            CHECK(std::find(cells.begin(), cells.end(), id) != cells.end());
        }
    }
}

TEST_CASE("sector: to a chart and back, 5000 km from home") {
    const globe::Chart home;
    const globe::Chart n = globe::near_chart(home, 4.1e6, 2.9e6);
    const glm::dvec2 player(0.0, globe::kNearOffZ);
    // (the player's coordinates are small there)
    const glm::dvec2 pl = globe::to_chart(n, globe::to_planet(home, 4.1e6, 2.9e6));
    CHECK(std::abs(pl.x - player.x) <= globe::kTile * 0.5 + 1e-6);
    for (const glm::dvec2 off : { glm::dvec2(0.0, 0.0), glm::dvec2(1234.5, -876.25), glm::dvec2(-3000.0, 2500.0) }) {
        const glm::dvec2 p = pl + off;
        glm::dvec2 fr;
        const sector::Id id = sector::from_chart(n, p.x, p.y, 14, &fr);
        REQUIRE(id.valid());
        const glm::dvec2 q = sector::to_chart(n, id, fr.x, fr.y);
        CHECK(q.x == Approx(p.x).margin(1e-5));
        CHECK(q.y == Approx(p.y).margin(1e-5));
        // The same sector from home's chart (the address does not depend on the chart).
        const glm::dvec2 ph = globe::to_chart(home, globe::to_planet(n, p.x, p.y));
        CHECK(sector::from_chart(home, ph.x, ph.y, 14) == id);
    }
}

TEST_CASE("sector: the series a shader evaluates is the exact mapping to 2 mm within 4 km") {
    const globe::Chart home;
    struct Case { globe::Chart c; glm::dvec2 anchor; };
    std::vector<Case> cases;
    cases.push_back({ home, glm::dvec2(300.0, -200.0) });
    cases.push_back({ home, glm::dvec2(-9000.0, 7000.0) });
    {
        const globe::Chart n = globe::near_chart(home, 4.1e6, 2.9e6);
        cases.push_back({ n, glm::round(globe::to_chart(n, globe::to_planet(home, 4.1e6, 2.9e6))) });
    }
    {
        // By a cube corner, where the cells are most skewed.
        const glm::dvec3 G = glm::normalize(glm::dvec3(1.0, 1.0001, 0.9999));
        const glm::dvec3 e = glm::normalize(glm::cross(glm::dvec3(0.0, 1.0, 0.0), G));
        globe::Chart c = globe::far_chart(glm::dmat3(e, G, glm::cross(e, G)));
        c.high = false; c.org = glm::dvec2(0.0, globe::kNearOffZ);
        cases.push_back({ c, c.org });
    }
    for (const Case& k : cases) {
        const sector::Id c0 = sector::from_chart(k.c, k.anchor.x, k.anchor.y, 14);
        const int64_t bi = int64_t(c0.i()) - 8, bj = int64_t(c0.j()) - 8;
        const sector::ChartMap m = sector::chart_map(k.c, k.anchor, c0.face(), 14, bi, bj);
        const double cell_m = std::min(sector::size_m(c0).x, sector::size_m(c0).y);
        double worst = 0.0;
        for (int q = 0; q < 200; ++q) {
            const double ang = 0.731 * q, d = 4000.0 * ((q % 20) + 1) / 20.0;
            const glm::dvec2 p = k.anchor + d * glm::dvec2(std::cos(ang), std::sin(ang));
            const glm::dvec2 a = m.eval(p), b = sector::chart_cell_exact(k.c, p, c0.face(), 14, bi, bj);
            worst = std::max(worst, glm::length(a - b) * sector::kFaceM / 16384.0);
        }
        INFO("cell " << sector::name(c0) << " (" << cell_m << " m): worst " << worst * 1000.0 << " mm");
        CHECK(worst < 0.002);
        // The cell coordinates stay small numbers (what a float needs).
        CHECK(std::abs(m.c.x - 8.5) <= 0.5);
        CHECK(std::abs(m.c.y - 8.5) <= 0.5);
    }
}
