#pragma once
// The mart's goods (vk_engine/shop_goods.cpp): what the engine holds of them.
//
// Private to the files that implement it - vk_engine.h only knows the name of this struct and
// holds a pointer to it (shop_). Created on first use and freed by shop_destroy().

#include "engine/vk_engine.h"
#include "game/level.h"

#include <glm/glm.hpp>

#include <string>
#include <utility>
#include <vector>

namespace vibe {

struct VulkanEngine::ShopState {
    bool tried = false, ok = false;
    // The goods kit (assets/shop/goods.kit): per product its three levels, in its own axes
    // (x its width, y up from what it stands on, z toward its front).
    struct Sku { std::string name, cat; glm::vec3 size{0.0f}; game::LevelMesh lod[3]; };
    std::vector<Sku> kit;
    // Every product on every shelf, once (the fill is deterministic): station-local place.
    struct Item {
        int       sku = 0;
        glm::mat4 m{1.0f};       // its axes and place in the station's coordinates (scale included)
        uint8_t   row = 0;       // 0 the front row, 1, 2 ... behind it
        bool      hidden = false;    // behind others: dropped from the far level
        int       bin = 0;       // which merged part it goes into
    };
    std::vector<Item> items;
    int bins = 0;
    glm::vec3 lo{1e9f}, hi{-1e9f};   // the goods' box in the world
    // The level drawn now (-1 none) and its parts; the next level's parts as they are made,
    // one a frame; parts given back, waiting out the frames in flight.
    int level = -1, building = -1;
    std::vector<CastleMeshGpu> parts, next;
    size_t next_bin = 0;
    uint32_t tris = 0, next_tris = 0;
    std::vector<std::pair<int, std::vector<CastleMeshGpu>>> retired;
};

} // namespace vibe
