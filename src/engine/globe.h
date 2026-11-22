#pragma once
// The planet as a real sphere.
//
// The engine works in a flat "chart": x east, z south, y up, in metres - floats, the home
// landscape at the origin. A chart is the planet unrolled round one point of it (the
// exponential map: distance and bearing from that point are kept), with a frame there:
// which way its east, up and south point in the planet's own frame. Home's chart is the
// identity (home is the planet's top: up = +y, east = +x, south = +z); the player is in it
// anywhere within 100 km of home. Further out the engine re-anchors: a new chart under the
// player, its origin at local (kFarX, 0), never nearer than 48 km, so that nothing of home's content - which lives at
// its home coordinates near the origin - is anywhere near.
//
// The world's shape is a function of the point on the sphere (shaders/world_map.glsl, mirrored
// in terrain.cpp + here), so it is the same from every chart, and flying on round the planet
// comes back home.

#include <glm/glm.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace vibe::globe {

constexpr double kR = 6360000.0;            // m (the atmosphere model's planet)
// (Coordinates stay under 100 km: a float is then fine to under a centimetre, which the
// shaders' close-up detail needs. A far chart's own origin is never nearer than 48 km to the
// player, further than anything of home's content is drawn.)
constexpr double kFarX = 64000.0;           // a far chart's origin in its local coordinates (x; z = 0)
constexpr double kHomeReach = 100000.0;     // the home chart is used out to here (m from home)
constexpr double kHomeBack = 92000.0;       // ... and taken up again from here
// Near the ground the home chart is used only this near home (its coordinates are home's:
// 12 km is a millimetre); further out a near chart takes over, back again from kHomeNearBack.
constexpr double kHomeNear = 12000.0, kHomeNearBack = 9000.0;
constexpr double kFarReach = 16000.0;       // a far chart is re-anchored this far from its origin
constexpr double kBlend0 = 40000.0, kBlend1 = 90000.0;      // where the flat formula's land hands over to the sphere's (m from home)
constexpr int    kOctaves = 12;

struct Chart {
    glm::dmat3 F{1.0};                      // columns: east, up, south at the chart's origin (planet frame)
    bool home = true;
    int  site = -1;                         // a site's own chart: its number (see Site below); else -1
    // A wandering chart is one of two kinds (see "Small coordinates" below): taken up near the
    // ground its origin is local (0, 0) and the player is kept round (0, kNearOffZ); taken up
    // high above it (`high`), its origin is local (kFarX, 0).
    bool high = false;
    // Where the chart's own origin (the point the planet is unrolled round) lies in its local
    // coordinates. Home's and a site's: (0, 0). A high chart's: (kFarX, 0). A near chart's: see
    // near_chart - whatever keeps the coordinates of the land under the player what they were,
    // less a whole number of tiles.
    glm::dvec2 org{0.0};
    // A near chart: how many tiles (kTile) its coordinates are short of the coordinates the
    // chain of charts started from - home's, when the chain started at home.
    glm::ivec2 tile{0};
    double ox() const { return org.x; }
};

// ---- Sites ----
// A site is a hand-made place: buildings, a road, shaped ground. Home is one - everything
// of it lies at its home coordinates near the engine's origin. Every other site has a chart
// of its own, fixed on the planet, whose origin is kSiteOffZ north of the site: so the
// site's content lies at fixed, SMALL engine coordinates (0, kSiteOffZ) - small matters:
// at the 64 km of a wandering far chart a float resolves 4 to 8 mm, too coarse for grass
// blades and close-up ground; at 12 km it is 1 mm.
// That spot is 12 km south of home in engine coordinates, and the home chart reaches there
// too (as another part of the planet altogether). So a site's content - mesh parts,
// colliders - carries its site's number and is drawn, traced and collided with only while
// the engine is in that site's chart (VulkanEngine::site_here). Home's own content needs no
// such test: a site's chart is left kSiteLeave from the site, 7 km short of home's
// coordinates, further than any of it is drawn.
// The engine is in a site's chart while the player is within kSiteReach of it; further out
// the site's ground (it is part of the landscape formula) is seen from a wandering chart.
constexpr double kSiteReach = 4600.0;       // the site's chart is entered this near it (m)
constexpr double kSiteLeave = 5400.0;       // ... and left again from here
constexpr double kSiteOffZ  = 12288.0;      // the site's place in its chart: this far south of the chart's origin
// A site's content is loaded this near it, and dropped again from kSiteUnload (the engine
// holds one site's content at a time: vk_engine/sites.cpp).
constexpr double kSiteLoad = 9000.0, kSiteUnload = 14000.0;
// What the landscape needs of a site: its ground is level at `rise` above the water within
// `flat` of its centre (what is built stands on that) and meets the natural ground - or the
// sea - at `slope`; no woods grow within `clearing`.
struct SiteShape {
    glm::dvec3 G{0.0, 1.0, 0.0};            // where on the planet (unit vector)
    float flat = 300.0f, slope = 1500.0f, rise = 14.0f, clearing = 330.0f;
};
struct Site : SiteShape {
    std::string id, name;
    Chart      chart;                       // its own chart (home: the home chart)
    glm::dvec2 at{0.0};                     // its centre in that chart (engine x, z)
};
// The sites known - there may be thousands; this is only where each is and the shape of its
// ground (the index: engine/site_store.h). Home is site 0. Filled at start-up, on the main
// thread, before anything reads it.
int  site_count();
const Site& site(int k);
int  add_site(const std::string& id, const std::string& name, const SiteShape& shape);
// The sites near the player (within some hundreds of km): what the landscape functions look
// at - on any thread, for every height they work out - instead of the whole list. Published
// by the main thread when the set changes (globe_update); an unchanging snapshot.
const std::vector<SiteShape>& sites_nearby();
void set_sites_nearby(std::vector<SiteShape> v);
// Metres over the ground between two points of the planet.
double ground_distance(const glm::dvec3& a, const glm::dvec3& b);

// The chart the engine is in. Set on the main thread (re-anchoring), read from anywhere.
const Chart& current();
void set_current(const Chart& c);

// The ground point of local (x, z) as a unit vector in the planet frame.
glm::dvec3 to_planet(const Chart& c, double x, double z);
// The chart's frame carried to that ground point (its east / up / south there).
glm::dmat3 frame_at(const Chart& c, double x, double z);
// Home coordinates (m east, m south of home) of the point G.
glm::dvec2 home_az(const glm::dvec3& G);
// Local (x, z) of the point G in the chart.
glm::dvec2 to_chart(const Chart& c, const glm::dvec3& G);
// ---- Small coordinates ----
// A float resolves 4 to 8 mm at 64 km and 1 mm at 12 km. Grass blades and close-up ground need
// the millimetre: at 64 km the blades tear into blocks and textures smear. So near the ground
// the engine's coordinates are kept small everywhere on the planet - the same few kilometres of
// numbers are used again and again, each time for another piece of it:
//   * near_chart(): a wandering chart unrolled round the player, whose coordinates are the
//     old chart's SHIFTED BY A WHOLE NUMBER OF TILES (kTile, 6400 m), by as many as bring the
//     player back to within half a tile of (0, kNearOffZ). It is re-anchored when the player is
//     kNearReach from there along either axis, so coordinates stay under 16 km (1 mm). (Not
//     round (0, 0): home's content lives at those coordinates in every chart - a site's chart
//     keeps clear of it the same way.)
//     The whole tiles are the point: the land round the player keeps its coordinates exactly,
//     up to that shift, from home's chart into a near chart and from one near chart to the
//     next. So whatever is worked out from coordinates and repeats once a tile - the grass's
//     blades, the ground's textures and fine noise: the shaders' tile_* functions in
//     scene_ubo.glsl - is the same before and after: nothing re-arranges when the engine
//     re-centres. (The trick of UE5's large-world coordinates: a tile and a small offset.)
//   * High above the ground the millimetre does not matter and re-anchoring every few
//     kilometres would (the streamed woods are laid out again each time): from kHighUp over
//     the water the old kind of chart is used - origin at local (kFarX, 0), re-anchored kFarReach
//     from it - and below kHighDown a near chart again.
constexpr double kNearOffZ  = 12288.0;      // (= kSiteOffZ)
constexpr double kTile = 6400.0;            // (shaders: kTile in scene_ubo.glsl)
constexpr double kNearReach = 3600.0;       // along either axis (more than half a tile: a shift then always follows)
constexpr double kHighUp = 1800.0, kHighDown = 900.0;       // m over the water
// ---- Home's landscape from another chart ----
// Within 90 km of home the land is the flat formula's (a function of home coordinates), so a
// chart that is not home's but lies in that region has to work it out too - without ever
// forming the large home coordinates in a float. `Ring` is what that takes: home coordinates
// are, near the chart's anchor, h0 + J (p - wp0) (J is very nearly a rotation; the error of
// taking the map as linear is under 2 cm within a near chart's reach). The formula's FBM then
// runs on J (p - wp0) - small numbers - and gets h0's share per octave as a lattice cell and
// a position in it (flat_lattice), the way the sphere's FBM does (lattice below).
struct Ring {
    bool on = false;                        // the chart is within kRingReach of home (and is not home's)
    glm::dvec2 wp0{0.0}, h0{0.0};
    glm::dmat2 J{1.0};
    glm::dvec2 to_home(const glm::dvec2& p) const { return h0 + J * (p - wp0); }
    glm::dvec2 to_chart(const glm::dvec2& h) const { return wp0 + glm::inverse(J) * (h - h0); }
};
constexpr double kRingReach = 130000.0;
constexpr int    kFlatOctaves = 24;
Ring ring_of(const Chart& c);
const Ring& ring_now();                     // the current chart's (set with it)
uint32_t chart_serial();                    // counts set_current: what is laid out in chart coordinates is made again when it changes
// Per octave of the flat formula's FBM at home coordinates h0: out[i] = position within the
// lattice cell (x, y), the cell (x, y; wrapped to +-2^22, exact in a float).
void flat_lattice(const glm::dvec2& h0, float out[kFlatOctaves][4]);

// A chart for the ground point at (x, z) of chart c: near the ground, or high above it.
Chart near_chart(const Chart& c, double x, double z);
Chart high_chart(const Chart& c, double x, double z);
// A chart with the given frame at its origin (a wandering chart of the `high` kind; the
// others are made from it: near_chart, a site's).
Chart far_chart(const glm::dmat3& frame);

// Per octave of the landscape's 3D FBM: the lattice cell of `origin` (planet frame, metres)
// and the position within it - what the shaders need to evaluate the FBM near that point.
void lattice(const glm::dvec3& origin, int32_t cells[kOctaves][4], float fracs[kOctaves][4]);

// CPU side of shaders/world_map.glsl's globe part (P: planet frame; G unit vector).
float noise3(const glm::dvec3& p);
float shape3(const glm::dvec3& P_m, float oct_f);       // wm_shape3
float continent3(const glm::dvec3& G);                  // wm_continent3
float ranges3(const glm::dvec3& G);                     // wm_ranges3

}  // namespace vibe::globe
