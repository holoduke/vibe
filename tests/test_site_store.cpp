#include "engine/site_store.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

using vibe::SiteData;
using vibe::SiteHeader;
using vibe::SiteStore;
using Catch::Approx;

TEST_CASE("site index: a line with coordinates") {
    SiteHeader h;
    REQUIRE(SiteStore::parse_index_line("seawatch 563.624 -205.739 300 1500 14 330 Seawatch on Sea", h));
    CHECK(h.id == "seawatch");
    CHECK(h.name == "Seawatch on Sea");
    CHECK_FALSE(h.auto_place);
    CHECK(h.east_m == Approx(563624.0));
    CHECK(h.south_m == Approx(-205739.0));
    CHECK(h.flat == Approx(300.0f));
    CHECK(h.slope == Approx(1500.0f));
    CHECK(h.rise == Approx(14.0f));
    CHECK(h.clearing == Approx(330.0f));
}

TEST_CASE("site index: auto placement, comments, and what is not a site") {
    SiteHeader h;
    REQUIRE(SiteStore::parse_index_line("isle auto - 200 900 10 250   # picked by the engine", h));
    CHECK(h.auto_place);
    CHECK(h.name == "isle");                               // no name given: the id
    CHECK_FALSE(SiteStore::parse_index_line("# only a comment", h));
    CHECK_FALSE(SiteStore::parse_index_line("", h));
    CHECK_FALSE(SiteStore::parse_index_line("bad 1 2", h));                   // no ground
    CHECK_FALSE(SiteStore::parse_index_line("bad x y 300 1500 14 330", h));   // not numbers
    CHECK_FALSE(SiteStore::parse_index_line("bad 1 2 300 200 14 330", h));    // the slope must lie outside the flat
}

TEST_CASE("site file: objects, a road in pieces, unknown kinds") {
    SiteData d;
    REQUIRE(SiteStore::parse_site(
        "# a small place\n"
        "tower 0 0 3 5EA7001\n"
        "house -15 70\n"
        "chapel 0 -178\n"
        "lamp 3.3 190 -1 0\n"
        "windmill 40 40 7\n"                // a kind this build does not know: left out
        "road 0 262 0 243\n"
        "road 0 224\n", d));
    REQUIRE(d.objects.size() == 4);
    CHECK(d.objects[0].kind == "tower");
    CHECK(d.objects[0].a == Approx(3.0f));
    CHECK(d.objects[0].seed == 0x5EA7001u);
    CHECK(d.objects[1].kind == "house");
    CHECK(d.objects[1].x == Approx(-15.0f));
    CHECK(d.objects[1].z == Approx(70.0f));
    CHECK(d.objects[3].kind == "lamp");
    CHECK(d.objects[3].a == Approx(-1.0f));
    REQUIRE(d.road.size() == 3);
    CHECK(d.road[2].y == Approx(224.0f));
}

TEST_CASE("site file: nothing in it is no site; an id is never a path") {
    SiteData d;
    CHECK_FALSE(SiteStore::parse_site("# empty\n\n", d));
    SiteStore s;
    CHECK_FALSE(s.load("../secrets", d));
    CHECK_FALSE(s.load("a/b", d));
    CHECK_FALSE(s.load("", d));
}
