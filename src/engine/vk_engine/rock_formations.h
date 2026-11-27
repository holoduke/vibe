#pragma once
// Rock formations: large rock pieces with real gaps (an arch, a cleft wall, an overhang, a
// stack of slabs, leaning pillars, a boulder pile, a cliff buttress, a low outcrop) stood
// about the landscape by rule (rock_formations.cpp) - and a family of sheer CLIFF FACES that
// stand side by side as one wall (four variants, a headland, a sea stack): white chalk along
// the sea, the dark rock as crag bands inland.
//
// THE KIT is made in Blender by tools/rocks/make_formations.py (assets/rocks/kit.rockf):
// every piece at three levels of detail, with ambient occlusion and curvature per vertex and
// a coarse grid of its solid cells (collision is the mesh itself: collision_triangles).
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
#include "game/tri_collision.h"

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
    float     tilt = 0.0f;                  // then turned about its own x: + brings its top forward (the cliff walls: from their 65-degree slope to the ground's)
    float     scale = 1.0f;
    glm::vec3 lo{0.0f}, hi{0.0f};           // its box in the world
    int       why = 0;                      // 0 a cut beside the road, 1 the foot of a slope, 2 beside the road, 3 a gate, 4 a cliff, 5 the test row, 6 the sea cliff, 7 a crag band, 8 in the water, 9 a river boulder (place_river_boulders), 10 the shore kit, 11 a cliff wall (rock_cliffs.cpp)
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
    // The water's level at (x, z) where it is not the datum `water` (the sea 1 m under it, the river
    // falling to it: water_fx.cpp water_level_off); empty: `water` everywhere.
    std::function<float(float x, float z)> level;
};

// The kit (once). False when the file is missing or not understood. `append`: a second kit
// (the shore kit, assets/shore/shore.rockf) whose pieces follow the first's - its bake's layers
// follow the first's the same way (rock_bake.cpp). Its first piece's index: shore_base().
bool load_kit(const char* path, bool append = false, int* base = nullptr);   // (base: where an appended kit's first piece is - else it is the shore kit's)
int  shore_base();                          // -1: no shore kit

// THE CLIFF WALLS (rock_cliffs.cpp): a third kit, assets/cliffs/cliffs.rockf (24 rock walls made for
// a 65-degree slope: tools/rocks/make_formations.py --set cliffs), laid over the ground wherever it
// is steeper than 55 degrees round the camera - in 64 m cells made within 420 m and given back
// beyond 560 m. Each formation (why 11, tilted: Inst::tilt) lives in a SLOT; an empty slot has
// piece -1; a slot's generation changes whenever it is emptied or filled again.
bool load_cliff_kit(const char* path);
int  cliff_base();                          // -1: no cliff kit
bool cliff_stream(const Env& env, glm::vec3 eye, int budget);      // true when the slots changed
const std::vector<Inst>& cliff_slots();
const std::vector<uint32_t>& cliff_slot_gen();
// The shore kit's piece `in.piece` is small enough to have no collider (cobbles s / m, sticks).
bool no_collider(const Inst& in);
const std::vector<Piece>& kit();

// The list (once; needs highway::ready() and the kit).
void place(const Env& env);
bool placed();
const std::vector<Inst>& instances();

glm::mat4 transform(const Inst& in);
// Collision is the rock's own surface: the triangles of its middle level of detail.
constexpr int kCollisionLevel = 1;
// A piece's collision triangles in its own axes, three corners each (the physics world's
// shared mesh: PhysicsWorld::mesh_pieces_build_async).
void collision_triangles(int piece, std::vector<glm::vec3>& out);
// A formation's collision triangles in the world, with their grid (the walker: tri_collision.h).
void collision_mesh(const Inst& in, tricol::Mesh& out);

}  // namespace vibe::rockf
