#pragma once
// The river's plants beyond the home tile, streamed (river_stream.cpp; the rule: engine/river_plants.h).
//
// The small plants the engine places at start-up (scatter.cpp) stand on the home tile's height
// map. The river runs on to the sea 900 m beyond it. There its lilies, reeds and bank plants are
// made a 70 m stretch at a time, on a background thread, as the player comes within kLoad of the
// stretch, written into a slot of one small instance buffer, drawn with the small plants' own
// pipeline and meshes (world_render.cpp calls draw()), and given back beyond kFree.
// Home's chart only (their coordinates are home's).

#include "engine/scatter.h"

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>
#include <glm/vec3.hpp>

#include <functional>

namespace vibe::river_stream {

// Once a frame. `eye`: the camera (home coordinates), `here`: the engine is in home's chart,
// `octaves`: the ground formula's detail the terrain is drawn with.
void poll(VmaAllocator alloc, glm::vec3 eye, bool here, float water_level, int octaves);

// Inside the small plants' pass, their pipeline bound. `begin_species(sp)` pushes that species'
// constants and returns its draw distance; `cell_in(cell, distance)` is the pass's own test of
// a cell (distance, frustum, hidden behind a hill).
void draw(VkCommandBuffer cmd, const ScatterSet& set, glm::vec3 eye,
          const std::function<float(int)>& begin_species,
          const std::function<bool(const ScatterCell&, float)>& cell_in);

void destroy(VmaAllocator alloc);

}  // namespace vibe::river_stream
