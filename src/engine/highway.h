#pragma once

// Highways: roads made from a line.
//
// A highway is a centre line (points a few metres apart, each with the height of the road's
// surface there). Everything else follows from the line by rule:
//
//   * the ground: a level shelf along it, cut into slopes and built up over hollows and
//     water (ground(): the physics and everything placed on the land read it; the terrain
//     shader does the same from the field below),
//   * the surface: asphalt, a gravel verge, lane markings (the terrain shader),
//   * what does not grow there: woods (terrain.cpp world_forest), grass (the near-field bake).
//
// Where the lines come from is a separate matter. build() makes the one the game has today:
// the course - a closed road round home, along the coast past the launch pad, round the
// castle's country and through the mountains, found by a search over the landscape (see
// build() in highway.cpp). Lines from data (a file, a server) would be added to the same
// table (add_line).
//
// For the shader the lines near the camera are baked into a field (bake()): for every 4 m of a
// 4096 m window, the signed distance to the nearest centre line, the road's height there and
// the distance along the road. Distance, height and length are all linear across a straight
// stretch, so the shader's bilinear read of the field is exact to the centimetre, and one
// texture fetch replaces any search through line segments.
//
// The lines are kept in home coordinates (x east, z south, metres). In home's chart those are
// the engine's own; in another chart near home (engine/globe.h Ring) they are carried over:
// the CPU's queries by the caller (terrain.cpp home_coords), the field and the roadside
// objects by the ring's mapping.

#include "engine/globe.h"

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

#include <cstdint>
#include <vector>

namespace vibe::highway {

// The cross-section (m from the centre line). Shaders: the same numbers in
// terrain_rm_noise_road.glsl (rm_highway) and terrain_rm_main_ground.glsl.
constexpr float kPaved   = 5.5f;      // asphalt to here: two 3.75 m lanes and a strip beyond each edge line
constexpr float kVerge   = 8.0f;      // gravel to here; the shelf is level this far
constexpr float kBlend   = 30.0f;     // the cut / fill has met the natural ground by here
// The field (bake()).
constexpr int   kFieldN    = 1024;    // texels a side
constexpr float kFieldCell = 4.0f;    // metres a texel
constexpr float kFieldSide = kFieldN * kFieldCell;      // 4096 m
constexpr float kFieldMaxD = 64.0f;   // distances are stored to this
constexpr float kFieldHLo  = -64.0f, kFieldHSpan = 512.0f;   // heights stored in [lo, lo + span)

// Makes the coast road and its spur (once; later calls do nothing). Needs the landscape
// formula (terrain.cpp), so call it when the water level is known and before any ground
// height is asked for that should include the road.
void build(float water_level);
bool ready();

// A line from data: points on the ground plan with the surface height (y). Thread-safe only
// before the first query.
// `width`: the road's size against a highway's (1 = the cross-section above). A road of width
// w is the same road w times as large across: asphalt to 5.5 w, verge to 8 w, the cut and
// fill met by 30 w. The castle lane (one lane, 4 m of asphalt) is 0.364.
void add_line(const std::vector<glm::vec3>& pts, float s0 = 0.0f, float width = 1.0f);

// The nearest centre line within `reach` of (x, z): false when there is none. d: signed
// distance (left of the line's direction is negative), h: surface height, s: metres along.
// DISTANCES ARE IN THE ROAD'S OWN MEASURE: metres divided by its width (above). For a highway
// that is metres; 5.5 is the asphalt's edge on any road, 8 its verge's. So whoever asks "is
// this on the asphalt" needs to know nothing of how wide the road is. `reach` likewise.
// "Nearest" is by that measure too (beside a junction a point belongs to the road whose
// cross-section it lies further inside).
bool nearest(float x, float z, float reach, float& d, float& h, float& s);
bool nearest(float x, float z, float reach, float& d, float& h, float& s, int& line);
float line_width(int line);
// Distance to the nearest centre line (1e9 beyond 64 m).
float distance(float x, float z);
// The ground with the shelf: `h` is the natural height at (x, z).
float ground(float h, float x, float z);
// How far the surface lies under the centre line's height, `ad` metres from it (the crown).
float cross_fall(float ad);
// The length of the first line (the course), metres.
float course_length();
// Bridges: the stretches of the course over ground that is under water. No shelf is made
// there (ground(), the baked field): the land and the water stay as they are, and a bridge
// carries the road (vk_engine/bridges.cpp). `bridge`: its ends, in point_along's metres.
int   bridge_count();
void  bridge(int i, float& s0, float& s1);
// Every road's bridges (the two above are the course's only): which line, and from where to
// where in that line's own metres (point_on).
int   span_count();
void  span(int i, int& line, float& s0, float& s1);
bool  point_on(int line, float s, glm::vec3& pos, glm::vec2& dir);
// On a bridge's deck: its surface height at (x, z).
bool  deck(float x, float z, float& h);

// The field around `center` (rounded by the caller): two RGBA8 images, kFieldN a side.
//   a: R,G = signed distance (16 bit over +-64 m), B,A = surface height (16 bit over 512 m)
//   b: R,G,B = metres along the road less s_ref, in millimetres + 2^23; A = the road's width x 255
//   (the distance is in the road's own measure - see nearest())
// bbox: min x, min z, max x, max z of where the shelf is, inside the window (the shader reads
// the field only there). Returns false when no road is in the window.
// `ring`: when the engine is in a chart that is not home's (engine/globe.h Ring, `on`), the
// field is laid out in that chart's coordinates: the lines are carried into it first.
// `jn`: the junction's record for the shaders, in the field's coordinates (below).
bool bake(glm::vec2 center, std::vector<uint8_t>& a, std::vector<uint8_t>& b, glm::vec4& bbox, const globe::Ring& ring, glm::vec4* jn = nullptr);

// THE JUNCTION: where the castle lane leaves the coast road.
// The field holds ONE road per texel (the nearest), read bilinearly - so where two roads meet,
// distance, height and metres along are mixed between unlike texels: torn markings, a seam in
// the shelf, the one road's lines running onto the other. A junction is therefore not left to
// the field. It is a record: the point J on the highway's centre line, the way the lane leaves
// (unit, level; straight for its first 48 m), and the lane's heights at 0, 8 .. 56 m. Round it
//   * the field holds the highway alone (bake: the lane's texels are left out for its first
//     35 m), so the highway's surface, markings and wear run through undisturbed;
//   * the lane is worked out from the record, laid OVER the highway's ground: nothing inside
//     the highway's asphalt, all of its own section by 8 m beyond it - one surface;
//   * the paving is the union of the two roads with the corners rounded (kJnFlare);
//   * 41..46 m out the lane is handed back to the field.
// ground() and the shader (terrain_rm_noise_road.glsl rm_jn_*) work by the same rule and the
// same numbers - keep them in step.
// u: metres from J along the lane, v: metres to the side of the lane's axis.
constexpr float kJnFlare = 4.0f;                      // the mouth's corner radius (m)
constexpr float kJnU0 = -8.0f, kJnU1 = 46.0f, kJnV = 24.0f;      // the junction's box
constexpr float kJnFieldU = 35.0f;                    // the field holds the highway alone to here
constexpr float kJnStopA = 4.80f, kJnStopB = 5.25f;   // the stop line: a bar from here to here from the highway's centre line (its edge line is at 3.75)
struct Junction {
    bool      on = false;
    int       line = 1;            // the side road
    glm::vec2 j{0.0f}, w{0.0f};    // J (x, z) and the way the lane leaves, home coordinates
    float     k = 1.0f;            // the lane's width class
    float     s0 = 0.0f;           // the lane's metres at J
    float     hs[8]{};             // the lane's centre line heights at u = 0, 8 .. 56
};
const Junction& junction();
// The lane's centre line height u metres from J (0..48), as its line has it.
float junction_lane_h(float u);
// "How far from the asphalt" at the junction, as a distance from a centre line in a road's
// measure (5.5 = the edge of the paving): the two roads and the rounded corners between them.
// xh: metres from the highway's centre line.
float junction_ad(float u, float v, float xh);

// The lines themselves (for what is built along them: vk_engine/highway_furniture.cpp).
int line_count();
const std::vector<glm::vec3>& line_points(int line);

// The castle lane's end (where its car park begins) and the way the road arrives there.
bool lane_end(glm::vec3& pos, glm::vec3& dir);

// The coast road's tightest bend: where (xyz) and its radius in metres (w).
glm::vec4 tightest_bend();

// For tests and captures: a point of the coast road `metres` along it from the spur's
// junction (negative: the other way), and the road's direction there.
bool point_along(float metres, glm::vec3& pos, glm::vec2& dir);

}  // namespace vibe::highway
