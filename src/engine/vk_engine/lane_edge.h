#pragma once
// The edge of a narrow road at a drop: a low dry-stone wall (lane_edge.cpp).
//
// Along every road line that is not the coast highway (highway.h: lines 1..; today the castle
// lane) a wall of stones stands on the verge wherever the ground beside the road falls away -
// decided from the ground itself (make()), so it stays right when the road is moved. The
// stations and where a wall is wanted are a few numbers a station, made once; the stones are
// made a 120 m stretch at a time by whoever streams roadside objects (highway_furniture.cpp)
// and freed again out of sight.

#include "game/level.h"
#include "game/collision.h"
#include "engine/physics_world.h"

#include <glm/vec3.hpp>

#include <cstddef>
#include <functional>
#include <vector>

namespace vibe::lane_edge {

// The ground as it is drawn at (x, z), home coordinates.
using Ground = std::function<float(float, float)>;

// Where the walls stand (once; needs highway::ready()). `no_wall(x, z)`: places that must stay
// open whatever the ground does (a car park's entrance).
void make(const Ground& ground, const std::function<bool(float, float)>& no_wall);
bool made();

// Streaming: the road cut into stretches.
size_t    stretch_count();
bool      stretch_has_wall(size_t c);
glm::vec3 stretch_middle(size_t c);
void      build_stretch(size_t c, const Ground& ground, std::vector<game::LevelMesh>& out);

// Collision: for a car (and bolts) boxes turned along the wall; for a walker small upright boxes.
void car_boxes(glm::vec3 at, float reach, std::vector<PhysicsWorld::OrientedBox>& out);
const std::vector<collision::AABB>& walker_boxes();

}  // namespace vibe::lane_edge
