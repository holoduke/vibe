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
// the coast road - it follows the shoreline of the sea east of home (the landscape formula's
// own land / sea value, traced and smoothed) - and a spur from it to the launch pad. Lines
// from data (a file, a server) would be added to the same table (add_line).
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
void add_line(const std::vector<glm::vec3>& pts, float s0 = 0.0f);

// The nearest centre line within `reach` of (x, z): false when there is none. d: signed
// distance (left of the line's direction is negative), h: surface height, s: metres along.
bool nearest(float x, float z, float reach, float& d, float& h, float& s);
// Distance to the nearest centre line (1e9 beyond 64 m).
float distance(float x, float z);
// The ground with the shelf: `h` is the natural height at (x, z).
float ground(float h, float x, float z);

// The field around `center` (rounded by the caller): two RGBA8 images, kFieldN a side.
//   a: R,G = signed distance (16 bit over +-64 m), B,A = surface height (16 bit over 512 m)
//   b: R,G,B = metres along the road less s_ref, in millimetres + 2^23; A = 255
// bbox: min x, min z, max x, max z of where the shelf is, inside the window (the shader reads
// the field only there). Returns false when no road is in the window.
// `ring`: when the engine is in a chart that is not home's (engine/globe.h Ring, `on`), the
// field is laid out in that chart's coordinates: the lines are carried into it first.
bool bake(glm::vec2 center, std::vector<uint8_t>& a, std::vector<uint8_t>& b, glm::vec4& bbox, const globe::Ring& ring);

// The lines themselves (for what is built along them: vk_engine/highway_furniture.cpp).
int line_count();
const std::vector<glm::vec3>& line_points(int line);

// The coast road's tightest bend: where (xyz) and its radius in metres (w).
glm::vec4 tightest_bend();

// For tests and captures: a point of the coast road `metres` along it from the spur's
// junction (negative: the other way), and the road's direction there.
bool point_along(float metres, glm::vec3& pos, glm::vec2& dir);

}  // namespace vibe::highway
