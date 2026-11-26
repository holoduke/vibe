#pragma once
// Rock formations: large rock pieces with real gaps (an arch, a cleft wall, an overhang, a
// stack of slabs, leaning pillars, a boulder pile, a cliff buttress, a low outcrop) stood
// about the landscape by rule (rock_formations.cpp) - and a family of sheer CLIFF FACES that
// stand side by side as one wall (four variants, a headland, a sea stack): white chalk along
// the sea, the dark rock as crag bands inland.
//
// THE KIT is made in Blender by tools/rocks/make_formations.py (assets/rocks/formations.rockf):
// every piece at three levels of detail, with ambient occlusion and curvature per vertex and
// a coarse grid of its solid cells for collision.
//
// WHERE they stand is a list made once from the ground and the roads (place()): along every
// narrow road (the castle lane) and on the steepest cliffs of the home tile. A formation is
// a piece, a place, a turn about the vertical and a size - nothing else; the same on every
// PC. What is drawn, traced and collided with is chosen from that list around the player by
// the engine (rock_formations_gpu.cpp) and given back out of sight.
//
// A piece's own axes: x its width, y up (0 = where the ground is meant to be; it has a root
// below), z toward its FRONT. Its back is what is sunk into a slope.

#include "game/level.h"
#include "game/collision.h"

#include <glm/glm.hpp>

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace vibe::rockf {

constexpr int kLevels = 3;

struct Piece {
    std::string     name;
    game::LevelMesh lod[kLevels];           // uv = (ambient occlusion, curvature)
    glm::vec3       bmin{0.0f}, bmax{0.0f};
    // the solid cells: `cell` metres a side from `org`, x fastest, then y, then z
    glm::vec3       org{0.0f};
    float           cell = 1.0f;
    glm::ivec3      n{0};
    std::vector<uint8_t> solid;
};

struct Inst {
    int       piece = 0;
    glm::vec3 pos{0.0f};                    // the piece's origin (its ground line)
    float     yaw = 0.0f;                   // its front (+z) points along (sin yaw, 0, cos yaw)
    float     scale = 1.0f;
    glm::vec3 lo{0.0f}, hi{0.0f};           // its box in the world
    int       why = 0;                      // 0 a cut beside the road, 1 the foot of a slope, 2 beside the road, 3 a gate, 4 a cliff, 5 the test row, 6 the sea cliff, 7 a crag band
    int       look = 0;                     // 0 the landscape's dark rock, 1 white chalk (rock_form.glsl)
    float     reach = 1.0f;                 // its levels of detail are held this many times further out (tall pieces)
};

// The ground as it is drawn at (x, z), home coordinates.
using Ground = std::function<float(float, float)>;
struct Env {
    Ground ground;
    // true where nothing may stand: a circle of radius r round (x, z) touches a building, a
    // paved place, the castle (roads and water are asked for here).
    std::function<bool(float x, float z, float r)> blocked;
    float water = 0.0f;
};

// The kit (once). False when the file is missing or not understood.
bool load_kit(const char* path);
const std::vector<Piece>& kit();

// The list (once; needs highway::ready() and the kit).
void place(const Env& env);
bool placed();
const std::vector<Inst>& instances();

glm::mat4 transform(const Inst& in);
// The formation's solid cells as boxes in the world (cells under the ground are left out).
void boxes(const Inst& in, const Ground& ground, std::vector<collision::AABB>& out);

}  // namespace vibe::rockf
