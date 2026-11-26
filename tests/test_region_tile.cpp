// Height tiles (engine/region_tile.h): stored ground added to the formula's - the reading rule
// (bilinear, edge mask), the file, the store, and the loaded set the CPU and the shader share.
#include "engine/region_tile.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>

using namespace vibe;
using Catch::Approx;

namespace {
region::Store temp_store(const char* tag) {
    region::Store s;
    const auto root = std::filesystem::temp_directory_path() / (std::string("vibe_region_test_") + tag);
    std::error_code ec;
    std::filesystem::remove_all(root, ec);
    s.local_dir = (root / "assets").string();
    s.cache_dir = (root / "cache").string();
    return s;
}

// The loaded set a streamer would make of these tiles, for a camera at `cam` in `chart`.
std::shared_ptr<region::ActiveSet> set_of(const region::Store& st, const std::vector<sector::Id>& ids,
                                          const globe::Chart& chart, uint32_t serial, glm::dvec2 cam) {
    auto s = std::make_shared<region::ActiveSet>();
    int k = 0;
    for (sector::Id id : ids) {
        auto t = std::make_shared<region::Tile>();
        REQUIRE(st.load_tile(id, *t));
        s->slot[k++].tile = t;
    }
    s->update_fades();
    s->place_windows(chart, serial, cam);
    return s;
}

// shaders/region_tiles.glsl region_delta, line for line, in floats, from the table the shader gets.
float shader_delta(const region::GpuTable& T, const region::ActiveSet& S, glm::vec2 wp) {
    if (T.head[0] < 0.5f) return 0.0f;
    const glm::vec2 bd = wp - glm::vec2(T.head[1], T.head[2]);
    if (glm::dot(bd, bd) > T.head[3] * T.head[3]) return 0.0f;
    for (int k = 0; k < 2; ++k) {
        const region::GpuTable::Win& W = T.win[k];
        const glm::vec2 d = wp - glm::vec2(W.c_anchor[2], W.c_anchor[3]);
        const glm::vec2 cc = glm::vec2(W.c_anchor[0], W.c_anchor[1]) + glm::vec2(W.A[0] * d.x + W.A[1] * d.y, W.A[2] * d.x + W.A[3] * d.y) +
                             0.5f * glm::vec2(W.Hi[0] * d.x * d.x + 2.0f * W.Hi[1] * d.x * d.y + W.Hi[2] * d.y * d.y,
                                              W.Hj[0] * d.x * d.x + 2.0f * W.Hj[1] * d.x * d.y + W.Hj[2] * d.y * d.y);
        if (cc.x < W.lim[0] || cc.y < W.lim[1] || cc.x >= W.lim[2] || cc.y >= W.lim[3]) continue;
        const glm::ivec2 ci(int(cc.x), int(cc.y));
        const int c = ci.y * region::kWin + ci.x;
        const uint32_t s = (T.map[k][c >> 4][(c >> 2) & 3] >> ((c & 3) * 8)) & 255u;
        if (s == 0u) return 0.0f;
        const glm::vec2 ab = cc - glm::vec2(ci);
        // (the texture holds the tile's samples; the CPU's tile is that texture)
        region::Tile unit = *S.slot[s - 1].tile;
        unit.unit_m = T.slot[s - 1][0];
        return region::sample(unit, uint32_t(T.slot[s - 1][1] + 0.5f), ab.x, ab.y);
    }
    return 0.0f;
}
}  // namespace

TEST_CASE("height tile: read bilinear between its samples") {
    region::Tile t(sector::make(2, region::kTileLevel, 8000, 8000));
    for (int ib = 0; ib < region::kTileN; ++ib)
        for (int ia = 0; ia < region::kTileN; ++ia) t.at(ia, ib) = int16_t(3 * ia - 2 * ib + ((ia * 7 + ib * 13) % 11));
    const float u = t.unit_m, s = 1.0f / 512.0f;
    // On a sample: the sample. (No fading: fade = 0.)
    CHECK(region::sample(t, 0, 100 * s, 37 * s) == Approx(float(t.at(100, 37)) * u).margin(1e-5));
    CHECK(region::sample(t, 0, 0.0f, 0.0f) == Approx(float(t.at(0, 0)) * u).margin(1e-5));
    CHECK(region::sample(t, 0, 1.0f, 1.0f) == Approx(float(t.at(512, 512)) * u).margin(1e-5));
    CHECK(region::sample(t, 0, 1.0f, 0.25f) == Approx(float(t.at(512, 128)) * u).margin(1e-5));
    // Half way along a step: the mean of two; in the middle of a cell: the mean of four.
    CHECK(region::sample(t, 0, 100.5f * s, 37 * s) == Approx(0.5f * float(t.at(100, 37) + t.at(101, 37)) * u).margin(1e-4));
    CHECK(region::sample(t, 0, 100.5f * s, 37.5f * s) ==
          Approx(0.25f * float(t.at(100, 37) + t.at(101, 37) + t.at(100, 38) + t.at(101, 38)) * u).margin(1e-4));
    // A quarter of the way: three parts and one.
    CHECK(region::sample(t, 0, 100 * s, 37.25f * s) == Approx((0.75f * float(t.at(100, 37)) + 0.25f * float(t.at(100, 38))) * u).margin(1e-4));
    // Quantising metres.
    CHECK(region::to_sample(1.0f, 1.0f / 64.0f) == 64);
    CHECK(region::to_sample(-1e6f, 1.0f / 64.0f) == -32767);
    CHECK(region::to_sample(1e6f, 1.0f / 64.0f) == 32767);
}

TEST_CASE("height tile: the edge mask is 0 on a fading edge, 1 inside, and has no step") {
    using namespace region;
    const uint32_t all = kW | kE | kN | kS;
    CHECK(edge_mask(0.5f, 0.5f, all) == 1.0f);
    CHECK(edge_mask(0.0f, 0.5f, all) == 0.0f);
    CHECK(edge_mask(1.0f, 0.5f, all) == 0.0f);
    CHECK(edge_mask(0.5f, 0.0f, all) == 0.0f);
    CHECK(edge_mask(0.5f, 1.0f, all) == 0.0f);
    CHECK(edge_mask(kFade, 0.5f, all) == Approx(1.0f));
    CHECK(edge_mask(0.5f * kFade, 0.5f, all) == Approx(0.5f));
    // An edge that continues does not fade.
    CHECK(edge_mask(0.0f, 0.5f, all & ~kW) == 1.0f);
    CHECK(edge_mask(1.0f, 0.5f, kW) == 1.0f);
    CHECK(edge_mask(0.3f, 0.0f, 0) == 1.0f);
    // A corner alone: 0 at the point, 1 a fade's width away, in every direction the same.
    CHECK(edge_mask(0.0f, 0.0f, kNW) == 0.0f);
    CHECK(edge_mask(kFade, 0.0f, kNW) == Approx(1.0f));
    CHECK(edge_mask(0.0f, kFade, kNW) == Approx(1.0f));
    CHECK(edge_mask(1.0f, 1.0f, kSE) == 0.0f);
    CHECK(edge_mask(1.0f, 0.0f, kNE) == 0.0f);
    CHECK(edge_mask(0.0f, 1.0f, kSW) == 0.0f);
    CHECK(edge_mask(0.0f, 1.0f, kNW | kNE | kSE) == 1.0f);
    // Continuous and rising: no step anywhere on the way in.
    float prev = 0.0f;
    for (int k = 0; k <= 2000; ++k) {
        const float a = float(k) / 2000.0f * 0.1f, m = edge_mask(a, a * 0.7f, all | kNW);
        CHECK(m >= prev);
        CHECK(m - prev < 0.002f);
        prev = m;
    }
}

TEST_CASE("height tile: the file reads back, and a damaged or unknown one is refused") {
    region::Tile t(sector::make(2, region::kTileLevel, 8191, 8192));
    t.joined = region::kE | region::kS | region::kSE;
    for (size_t k = 0; k < t.h.size(); ++k) t.h[k] = int16_t((k * 2654435761u) >> 17) / 3;
    t.h[5] = -32767; t.h[6] = 32767;
    std::vector<uint8_t> bytes = region::encode(t);
    CHECK(bytes.size() == 32u + size_t(region::kTileN) * region::kTileN * 2u);
    region::Tile r;
    REQUIRE(region::decode(bytes.data(), bytes.size(), r));
    CHECK(r.id == t.id);
    CHECK(r.joined == t.joined);
    CHECK(r.unit_m == t.unit_m);
    CHECK(r.h == t.h);
    // One bit of the heights turned: the checksum catches it.
    { auto b = bytes; b[1000] ^= 4; CHECK_FALSE(region::decode(b.data(), b.size(), r)); }
    // Cut short; a later format version; a compression this build does not know; not a tile.
    CHECK_FALSE(region::decode(bytes.data(), bytes.size() - 2, r));
    { auto b = bytes; b[4] = 2; CHECK_FALSE(region::decode(b.data(), b.size(), r)); }
    { auto b = bytes; b[21] = 1; CHECK_FALSE(region::decode(b.data(), b.size(), r)); }
    { auto b = bytes; b[0] = 'X'; CHECK_FALSE(region::decode(b.data(), b.size(), r)); }
    CHECK_FALSE(region::decode(bytes.data(), 10, r));
}

TEST_CASE("region store: an index lists tiles; saving a tile lists it") {
    std::vector<sector::Id> ids;
    region::Store::parse_index("# the valley\n"
                               "tile 2-14-8191-8192\n"
                               "tile 2-14-8192-8192   # its neighbour\n"
                               "site harbour-7\n"                // another kind of content: not ours
                               "tile 2-13-1-1\n"                 // not a height tile's level
                               "tile nonsense\n"
                               "tile 2-14-8191-8192\n", ids);    // twice
    REQUIRE(ids.size() == 2);
    CHECK(ids[0] == sector::make(2, 14, 8191, 8192));
    CHECK(ids[1] == sector::make(2, 14, 8192, 8192));
    CHECK(region::Store::tile_file(ids[0]) == "h_2-14-8191-8192.vht");
    CHECK(region::Store::index_file(sector::ancestor(ids[0], region::kIndexLevel)) == "idx_2-8-127-128.txt");

    const region::Store st = temp_store("store");
    const sector::Id sec = sector::ancestor(ids[0], region::kIndexLevel);
    std::vector<sector::Id> listed;
    CHECK_FALSE(st.load_index(sec, listed));                  // nothing there: a server would be asked
    CHECK_FALSE(st.has_tile(ids[0]));
    region::Tile t(ids[0]);
    t.at(10, 20) = 640;
    REQUIRE(st.save_tile(t));
    REQUIRE(st.save_tile(t));                                 // again: listed once
    CHECK(st.has_tile(ids[0]));
    REQUIRE(st.load_index(sec, listed));
    REQUIRE(listed.size() == 1);
    CHECK(listed[0] == ids[0]);
    region::Tile r;
    REQUIRE(st.load_tile(ids[0], r));
    CHECK(r.at(10, 20) == 640);
    CHECK_FALSE(st.load_tile(ids[1], r));
    // A shipped tile and a fetched one in the same sector: both listed.
    region::Tile t2(sector::make(2, 14, 8190, 8192));
    REQUIRE(st.save_tile(t2, true));
    listed.clear();
    REQUIRE(st.load_index(sec, listed));
    CHECK(listed.size() == 2);
}

TEST_CASE("height tiles: a shape over four tiles is whole, and fades where a tile is missing") {
    const region::Store st = temp_store("stamp");
    // The corner four tiles share, near home.
    const sector::Id a = sector::make(2, region::kTileLevel, 8200, 8190);
    const glm::dvec3 G = sector::point(a, 1.0, 1.0);
    const std::vector<sector::Id> ids = region::stamp(st, G, region::Shape::Mesa, 150.0, 20.0);
    REQUIRE(ids.size() == 4);
    const globe::Chart home;
    const glm::dvec2 c = globe::to_chart(home, G);
    auto set = set_of(st, ids, home, 7, c + glm::dvec2(40.0, -25.0));
    CHECK(set->count == 4);
    // All four are joined round the shared corner: nothing fades there, the top is the mesa's.
    CHECK(set->delta_exact(home, c.x, c.y) == Approx(20.0f).margin(0.02));
    CHECK(set->delta_exact(home, c.x + 30.0, c.y - 20.0) == Approx(20.0f).margin(0.02));
    CHECK(set->delta_exact(home, c.x + 200.0, c.y) == 0.0f);
    // Walk across it in 5 cm steps, over both tile borders: no step (the mesa's side is 20 m
    // over 68 m, at its steepest 0.45: 2.3 cm a step).
    const auto walk = [&](const region::ActiveSet& s, glm::dvec2 from, glm::dvec2 to) {
        float worst = 0.0f, prev = s.delta_exact(home, from.x, from.y);
        const int n = int(glm::distance(from, to) / 0.05);
        for (int k = 1; k <= n; ++k) {
            const glm::dvec2 p = glm::mix(from, to, double(k) / n);
            const float v = s.delta_exact(home, p.x, p.y);
            worst = std::max(worst, std::abs(v - prev));
            prev = v;
        }
        return worst;
    };
    CHECK(walk(*set, c + glm::dvec2(-170.0, -13.0), c + glm::dvec2(170.0, 21.0)) < 0.04f);
    CHECK(walk(*set, c + glm::dvec2(9.0, -170.0), c + glm::dvec2(-17.0, 170.0)) < 0.04f);
    CHECK(walk(*set, c + glm::dvec2(-120.0, -120.0), c + glm::dvec2(120.0, 120.0)) < 0.04f);

    // One of the four is not loaded (not fetched yet, or out of slots): the other three fade
    // toward it and toward the corner - lower ground for now, but still no step anywhere.
    for (size_t gone = 0; gone < 4; ++gone) {
        std::vector<sector::Id> three;
        for (size_t k = 0; k < 4; ++k) if (k != gone) three.push_back(ids[k]);
        auto s3 = set_of(st, three, home, 7, c);
        CHECK(s3->count == 3);
        CHECK(s3->delta_exact(home, c.x, c.y) == Approx(0.0f).margin(1e-4));
        // (fading 20 m over 38 m: at its steepest 0.8 - 4 cm a step)
        CHECK(walk(*s3, c + glm::dvec2(-170.0, -13.0), c + glm::dvec2(170.0, 21.0)) < 0.07f);
        CHECK(walk(*s3, c + glm::dvec2(9.0, -170.0), c + glm::dvec2(-17.0, 170.0)) < 0.07f);
        CHECK(walk(*s3, c + glm::dvec2(-120.0, -120.0), c + glm::dvec2(120.0, 120.0)) < 0.07f);
        CHECK(walk(*s3, c + glm::dvec2(-120.0, 120.0), c + glm::dvec2(120.0, -120.0)) < 0.07f);
        // Along the borders themselves, a centimetre to either side.
        CHECK(walk(*s3, c + glm::dvec2(-150.0, 0.01), c + glm::dvec2(150.0, 0.01)) < 0.07f);
        CHECK(walk(*s3, c + glm::dvec2(-150.0, -0.01), c + glm::dvec2(150.0, -0.01)) < 0.07f);
        CHECK(walk(*s3, c + glm::dvec2(0.01, -150.0), c + glm::dvec2(0.01, 150.0)) < 0.07f);
        CHECK(walk(*s3, c + glm::dvec2(-0.01, -150.0), c + glm::dvec2(-0.01, 150.0)) < 0.07f);
    }
    // A lone tile fades on every side.
    auto s1 = set_of(st, { ids[0] }, home, 7, c);
    CHECK(s1->slot[0].fade == 0x0Fu);
    CHECK(walk(*s1, c + glm::dvec2(-170.0, -13.0), c + glm::dvec2(170.0, 21.0)) < 0.07f);
}

TEST_CASE("height tiles: the CPU's ground, the shader's series and the exact mapping agree") {
    const region::Store st = temp_store("series");
    const globe::Chart home;
    struct Place { globe::Chart chart; glm::dvec2 at; };
    std::vector<Place> places;
    places.push_back({ home, glm::dvec2(310.0, -140.0) });                    // by the castle
    places.push_back({ home, glm::dvec2(5200.0, 3100.0) });                   // outside the home tile
    {
        const globe::Chart n = globe::near_chart(home, 4.1e6, 2.9e6);         // 5000 km away
        places.push_back({ n, globe::to_chart(n, globe::to_planet(home, 4.1e6, 2.9e6)) + glm::dvec2(150.0, 90.0) });
    }
    uint32_t serial = 100;
    for (const Place& P : places) {
        const glm::dvec3 G = globe::to_planet(P.chart, P.at.x, P.at.y);
        const std::vector<sector::Id> ids = region::stamp(st, G, region::Shape::Crater, 120.0, 14.0);
        REQUIRE(!ids.empty());
        // The camera some way off, as it would be.
        auto set = set_of(st, ids, P.chart, ++serial, P.at + glm::dvec2(700.0, -450.0));
        region::GpuTable T;
        region::fill_gpu(*set, T);
        CHECK(T.head[0] == float(ids.size()));
        float worst_s = 0.0f, worst_g = 0.0f, deepest = 0.0f;
        for (int k = 0; k < 400; ++k) {
            const double ang = 0.61 * k, d = 140.0 * double(k % 40) / 39.0;
            const glm::dvec2 p = P.at + d * glm::dvec2(std::cos(ang), std::sin(ang));
            const float exact = set->delta_exact(P.chart, p.x, p.y);
            worst_s = std::max(worst_s, std::abs(set->delta_series(p.x, p.y) - exact));
            worst_g = std::max(worst_g, std::abs(shader_delta(T, *set, glm::vec2(p)) - exact));
            deepest = std::min(deepest, exact);
        }
        // (the crater's wall is as steep as 0.5: 2 mm of mapping is 1 mm of height; the shader's
        //  floats resolve a millimetre at these coordinates)
        CHECK(worst_s < 0.005f);
        CHECK(worst_g < 0.02f);
        CHECK(deepest == Approx(-14.0f).margin(0.5));
        // Outside every tile's reach: nothing, without looking.
        CHECK(set->delta_series(P.at.x + 5000.0, P.at.y) == 0.0f);
        CHECK(shader_delta(T, *set, glm::vec2(P.at) + glm::vec2(5000.0f, 0.0f)) == 0.0f);
    }
    // ground_delta(): what world_ground_height adds - through the published set.
    {
        const Place& P = places[0];
        CHECK(region::ground_delta(float(P.at.x), float(P.at.y)) == 0.0f);       // nothing loaded
        const glm::dvec3 G = globe::to_planet(P.chart, P.at.x, P.at.y);
        std::vector<sector::Id> ids;
        sector::cover(G, 120.0, region::kTileLevel, ids);
        ids.erase(std::remove_if(ids.begin(), ids.end(), [&](sector::Id id) { return !st.has_tile(id); }), ids.end());
        auto set = set_of(st, ids, P.chart, globe::chart_serial(), P.at);
        region::set_active(set);
        CHECK(region::ground_delta(float(P.at.x), float(P.at.y)) == Approx(-14.0f).margin(0.05));
        // A set made for another chart serial (the frame of a re-centre): the exact way, same answer.
        auto stale = std::make_shared<region::ActiveSet>(*set);
        stale->chart_serial = globe::chart_serial() + 1;
        region::set_active(stale);
        CHECK(region::ground_delta(float(P.at.x), float(P.at.y)) == Approx(-14.0f).margin(0.05));
        region::set_active(nullptr);
        CHECK(region::ground_delta(float(P.at.x), float(P.at.y)) == 0.0f);
    }
}

TEST_CASE("height tiles: the table the shader reads is the size its block declares") {
    // shaders/region_tiles.glsl: head (1 vec4) + 2 windows of 5 vec4 + 32 slots + 2 x 16 uvec4.
    CHECK(sizeof(region::GpuTable) == 16u * (1 + 2 * 5 + region::kSlots + 2 * 16));
    CHECK(region::kWin * region::kWin / 16 == 16);
}
