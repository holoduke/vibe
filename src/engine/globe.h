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

namespace qlike::globe {

constexpr double kR = 6360000.0;            // m (the atmosphere model's planet)
// (Coordinates stay under 100 km: a float is then fine to under a centimetre, which the
// shaders' close-up detail needs. A far chart's own origin is never nearer than 48 km to the
// player, further than anything of home's content is drawn.)
constexpr double kFarX = 64000.0;           // a far chart's origin in its local coordinates (x; z = 0)
constexpr double kHomeReach = 100000.0;     // the home chart is used out to here (m from home)
constexpr double kHomeBack = 92000.0;       // ... and taken up again from here
constexpr double kFarReach = 16000.0;       // a far chart is re-anchored this far from its origin
constexpr double kBlend0 = 40000.0, kBlend1 = 90000.0;      // where the flat formula's land hands over to the sphere's (m from home)
constexpr int    kOctaves = 12;

struct Chart {
    glm::dmat3 F{1.0};                      // columns: east, up, south at the chart's origin (planet frame)
    bool home = true;
    double ox() const { return home ? 0.0 : kFarX; }
};

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
// A chart with its origin at G and the given frame there.
Chart far_chart(const glm::dmat3& frame);

// Per octave of the landscape's 3D FBM: the lattice cell of `origin` (planet frame, metres)
// and the position within it - what the shaders need to evaluate the FBM near that point.
void lattice(const glm::dvec3& origin, int32_t cells[kOctaves][4], float fracs[kOctaves][4]);

// CPU side of shaders/world_map.glsl's globe part (P: planet frame; G unit vector).
float noise3(const glm::dvec3& p);
float shape3(const glm::dvec3& P_m, float oct_f);       // wm_shape3
float continent3(const glm::dvec3& G);                  // wm_continent3
float ranges3(const glm::dvec3& G);                     // wm_ranges3

}  // namespace qlike::globe
