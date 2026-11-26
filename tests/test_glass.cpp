// Panes (game/glass_panes.h): the sheets of a glass mesh are found from its triangles, and the
// bars that stand on a sheet cut it into panes.
#include "game/glass_panes.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <vector>

namespace gl = vibe::game::glass;
using Catch::Approx;

namespace {
struct Soup {
    std::vector<glm::vec3> p;
    std::vector<uint32_t> idx;
    // a quad with its own four corners, as the model files have it
    void quad(glm::vec3 a, glm::vec3 b, glm::vec3 c, glm::vec3 d) {
        const uint32_t i = uint32_t(p.size());
        p.insert(p.end(), { a, b, c, d });
        idx.insert(idx.end(), { i, i + 1, i + 2, i, i + 2, i + 3 });
    }
    void quad2(glm::vec3 a, glm::vec3 b, glm::vec3 c, glm::vec3 d) { quad(a, b, c, d); quad(d, c, b, a); }       // seen from both sides
    std::vector<gl::Sheet> sheets() const { return gl::find_sheets([&](uint32_t i) { return p[i]; }, idx.data(), idx.size()); }
};
}  // namespace

TEST_CASE("a wall of glass written from both sides is one rectangular sheet", "[glass]") {
    Soup s;
    // far from the origin, as a building is (the launch site stands at x 1600)
    const glm::vec3 o(1600.0f, 51.0f, 744.0f);
    s.quad2(o, o + glm::vec3(5.7f, 0.0f, 0.0f), o + glm::vec3(5.7f, 6.2f, 0.0f), o + glm::vec3(0.0f, 6.2f, 0.0f));
    const auto sh = s.sheets();
    REQUIRE(sh.size() == 1);
    REQUIRE(sh[0].rect);
    REQUIRE(sh[0].tris.size() == 4);
    REQUIRE(sh[0].size.x == Approx(5.7f).margin(0.01f));
    REQUIRE(sh[0].size.y == Approx(6.2f).margin(0.01f));
    REQUIRE(glm::distance(sh[0].origin, o) < 0.02f);
    REQUIRE(std::abs(sh[0].v.y) == Approx(1.0f).margin(1e-4f));       // v is up
}

TEST_CASE("sheets that only touch, or meet at a corner, stay apart", "[glass]") {
    Soup s;
    // a door bay: a sheet either side and one over the door - they share corners, not edges
    s.quad2({ 0, 0, 0 }, { 2, 0, 0 }, { 2, 6, 0 }, { 0, 6, 0 });
    s.quad2({ 4, 0, 0 }, { 6, 0, 0 }, { 6, 6, 0 }, { 4, 6, 0 });
    s.quad2({ 2, 3, 0 }, { 4, 3, 0 }, { 4, 6, 0 }, { 2, 6, 0 });
    // and a second wall at right angles, sharing the first one's edge
    s.quad2({ 0, 0, 0 }, { 0, 0, 3 }, { 0, 6, 3 }, { 0, 6, 0 });
    const auto sh = s.sheets();
    REQUIRE(sh.size() == 4);
    for (const auto& x : sh) REQUIRE(x.rect);
}

TEST_CASE("a sheet that is not a rectangle cannot break", "[glass]") {
    Soup s;
    // a stair's balustrade: a parallelogram
    s.quad2({ 0, 0, 0 }, { 3, 2, 0 }, { 3, 3, 0 }, { 0, 1, 0 });
    const auto sh = s.sheets();
    REQUIRE(sh.size() == 1);
    REQUIRE_FALSE(sh[0].rect);
}

TEST_CASE("mullions and a transom cut a sheet into panes", "[glass]") {
    Soup s;
    s.quad2({ 0, 0, 0 }, { 6, 0, 0 }, { 6, 6, 0 }, { 0, 6, 0 });
    const auto sh = s.sheets();
    REQUIRE(sh.size() == 1);
    gl::Bars bars;
    // the faces of three posts (7 cm wide, 5 cm proud of the glass) and of one transom, as two
    // triangles each; and things that are no bars: the frame along the sheet's edge, a wide
    // panel, a face far in front of the glass
    auto face = [&](glm::vec3 a, glm::vec3 b, glm::vec3 c, glm::vec3 d) { bars.add(sh[0], a, b, c); bars.add(sh[0], a, c, d); };
    for (float x : { 1.5f, 3.0f, 4.5f })
        face({ x - 0.035f, 0, 0.05f }, { x + 0.035f, 0, 0.05f }, { x + 0.035f, 6, 0.05f }, { x - 0.035f, 6, 0.05f });
    face({ 0, 2.465f, 0.05f }, { 6, 2.465f, 0.05f }, { 6, 2.535f, 0.05f }, { 0, 2.535f, 0.05f });
    face({ 0, 0, 0.05f }, { 6, 0, 0.05f }, { 6, 0.07f, 0.05f }, { 0, 0.07f, 0.05f });              // the sill
    face({ 2, 1, 0.05f }, { 4, 1, 0.05f }, { 4, 5, 0.05f }, { 2, 5, 0.05f });                      // a panel
    face({ 2.0f, 0, 0.6f }, { 2.07f, 0, 0.6f }, { 2.07f, 6, 0.6f }, { 2.0f, 6, 0.6f });            // a column in the room
    REQUIRE(bars.u_cuts.size() == 3);
    REQUIRE(bars.v_cuts.size() == 1);
    const auto panes = gl::split(sh[0], bars);
    REQUIRE(panes.size() == 8);
    float area = 0.0f;
    for (const auto& r : panes) area += (r.second.x - r.first.x) * (r.second.y - r.first.y);
    REQUIRE(area == Approx(36.0f).margin(0.01f));                    // the panes cover the sheet
    REQUIRE(gl::split(sh[0], gl::Bars{}).size() == 1);               // no bars: one pane
}
