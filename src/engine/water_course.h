#pragma once
// Water that stands above the sea: a COURSE is a polyline with a water-surface height at every
// point (falling downstream, in steps where the ground is steep), and a channel carved into the
// world formula under that surface. The stream from the hillside east of the castle is one
// (built once from the ground it runs over). The GPU twin is shaders/water_course.glsl: the
// points go to it through the scene block (stream_pts / stream_box / stream_info), so both sides
// read the same numbers.
//
// Along a segment from point k to k + 1 the surface is level and drops at its lower end when the
// drop is more than kFallDrop (a step: a small fall, with a pool above it), or runs evenly
// otherwise (a run / rapid). The reference game's speed rule: S = 0.75 + 3 |tangent . down| m/s.
#include <glm/glm.hpp>
#include <functional>
#include <vector>

namespace vibe::wc {

constexpr int   kMaxPts = 128;          // points in the scene block (keep: scene_ubo.glsl stream_pts)
constexpr int   kSegsPerBox = 8;        // segments per bounding box (stream_box[16])
constexpr float kFallDrop = 0.35f;      // m: a larger drop between two points is a step
constexpr float kBoxPad = 40.0f;        // m round each box: as far as the carve reaches

struct Course {
    std::vector<glm::vec4> pts;         // x, z, water surface, half width of the channel
    std::vector<glm::vec4> box;         // per kSegsPerBox segments: min x, min z, max x, max z (padded)
    float depth0 = 0.45f, depth1 = 1.3f;   // the channel's depth under the surface, spring .. mouth
    bool  ok = false;
};

// The stream: laid out once by build_stream over the ground the game stands on (world.cpp,
// after a saved sculpt is loaded), ending at the river's level; empty before that.
// `level`: the water's level at a place (the river falls down its course: water_fx.cpp
// water_level_off) - the stream ends at the level where it meets the river; empty: water_level.
void build_stream(float water_level, const std::function<float(float, float)>& ground,
                  const std::function<float(float, float)>& level = {});
const Course& stream(float water_level);

struct At {
    bool  on = false;                   // within the carve's reach of the course
    float dist = 1e9f;                  // m from the centre line
    float surf = 0.0f;                  // the water surface here (m)
    float half_w = 0.0f;                // the channel's half width here
    float depth = 0.0f;                 // its depth under the surface
    float slope = 0.0f;                 // the surface's fall per metre downstream (>= 0)
    float along = 0.0f;                 // metres down the course
    glm::vec2 dir{ 1.0f, 0.0f };        // the way it flows
    float lat = 0.0f;                   // m across from the centre line (left of the flow > 0)
    float seg = 0.0f;                   // which segment and how far along it (k + u)
    float foot = 0.0f;                  // 0..1: in the boiling pool at the foot of a step
    float drop = 0.0f;                  // the height of the step this segment ends in (0: a run)
    float sslope = 0.0f;                // the runs' slope smoothed across the points (water_course.glsl WcAt.sslope)
};
At at(const Course& c, glm::vec2 wp);
// How far from the centre line the stream's water can reach at A (its bank's foot + the bank up
// to the surface + a margin; water_course.glsl wc_wet_reach). The water's own edge is where the
// carved ground rises out of it, inside this.
float wet_reach(const At& A);
// The current's speed (m/s) at A, as the water shader draws it (water_course.glsl / the stream's
// block in terrain_rm_main_water.glsl).
float flow_speed(const At& A);

// The ground height h at wp with the course's channel cut into it (h itself away from it).
float carve(const Course& c, float h, glm::vec2 wp);
// The same for the stream, safe to call from inside the ground formula (returns h while the
// stream itself is being laid out from that formula).
float carve_stream(float h, glm::vec2 wp, float water_level);
// The stream's water surface at wp if wp is within its wet reach (wet_reach), else -1e9 (where
// the ground stands above it there is no water: water_at compares). And its current (m/s) there.
float stream_surface(glm::vec2 wp, float water_level);
glm::vec2 stream_flow(glm::vec2 wp, float water_level);
// 1D value noise (the GPU twin wc_n1 has the same bits).
float n1(float x);

// The point `along` metres down the course, `side` metres across it (left of the flow > 0), and
// the water's surface there (the test hook VIBE_TEST_AT_STREAM). False on an empty course.
bool point_along(const Course& c, float along, float side, glm::vec2& p, float& surf);
// Distance (m) from the stream's water edge: negative in the water. 1e9 far away.
float stream_distance(glm::vec2 wp, float water_level);

}  // namespace vibe::wc
