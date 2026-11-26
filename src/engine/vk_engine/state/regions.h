#pragma once
// Regions: the stored ground (height tiles) streamed round the camera (vk_engine/regions.cpp;
// engine/region_tile.h; docs/ARCHITECTURE.md "Regions"): its state.
//
// Private to the files that implement it - vk_engine.h only knows the name of this struct
// and holds a pointer to it (rg_). Created and freed in vk_engine/private_state.cpp.

#include "engine/vk_engine.h"
#include "engine/region_tile.h"
#include "engine/sector.h"

#include <chrono>
#include <future>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace vibe {

struct VulkanEngine::RegionState {
    // A tile is loaded when the camera is this near its bounding circle (m over the ground),
    // and freed beyond kFreeM.
    static constexpr double kLoadM = 1800.0, kFreeM = 2400.0;

    region::Store store;
    bool off = false, log = false, started = false;

    // The tiles on the GPU: an array texture, a layer a slot (scene binding 50), and the
    // table that says which lies where (binding 49).
    VkImage        image = VK_NULL_HANDLE;
    VmaAllocation  alloc = nullptr;
    VkImageView    view = VK_NULL_HANDLE;
    VkSampler      sampler = VK_NULL_HANDLE;
    VkBuffer       ubo = VK_NULL_HANDLE;
    VmaAllocation  ubo_alloc = nullptr;
    void*          ubo_map = nullptr;
    // One tile is copied to its layer per frame, through a staging buffer per frame in flight.
    VkBuffer       stage[kFrameOverlap]{};
    VmaAllocation  stage_alloc[kFrameOverlap]{};
    void*          stage_map[kFrameOverlap]{};

    // What the index files say exists, per index sector (level region::kIndexLevel).
    struct Known { sector::Id id; glm::dvec3 c{0.0}; double r = 0.0; };
    struct Index {
        std::vector<Known> tiles;
        bool   have = false, asked = false;
        double next_look = 0.0;         // the file is read again from then (every 3 s)
    };
    std::unordered_map<uint64_t, Index> index;

    enum { kEmpty, kLoaded, kUploaded, kActive };
    struct Slot {
        std::shared_ptr<region::Tile> tile;
        int      state = kEmpty;
        uint64_t ready_frame = 0;       // kUploaded: part of the loaded set from this frame on
        bool     dirty = false;         // edited: the layer is copied again
        bool     unsaved = false;
        std::vector<float> edit;        // while it is being sculpted: its heights in metres, unrounded
        glm::dvec3 c{0.0};              // its middle on the planet, and its bounding circle (m)
        double   r = 0.0;
    };
    Slot slot[region::kSlots];
    // The loaded set as last published (region::set_active), and what it held: to tell which
    // tiles' ground changed when it is made again.
    region::ActiveSet cur;
    uint64_t prev_id[region::kSlots] = {};
    uint32_t prev_fade[region::kSlots] = {};
    std::vector<sector::Id> changed;    // tiles whose ground changed: told to what stands on it
    bool     rebuild = false;

    // One file is read at a time, on a worker.
    struct Read { std::shared_ptr<region::Tile> tile; double ms = 0.0; };
    std::future<Read> job;
    sector::Id job_id;
    bool       job_active = false;
    std::unordered_map<uint64_t, double> missing;       // a listed tile whose file is not there: looked for again from then
    std::unordered_set<uint64_t> asked;                 // ... and asked for (Store::request), once
    double     next_scan = 0.0;
    bool       budget_logged = false;

    // Editing (the terrain brush on region tiles).
    bool      edit = false, stroke = false;
    float     stamp_radius = 120.0f, stamp_height = 18.0f;
    // Home's fixed trees under a tile: what each was lifted by, and the height written then
    // (a tree that was planted again since has another height: its lift starts at nothing).
    std::vector<std::pair<float, float>> veg_lift;
    bool      home_dirty = false;

    // Frame times round the last tile that joined the set (VIBE_LOG_REGIONS).
    float     dt_ring[16] = {};
    std::chrono::steady_clock::time_point last_call{};
    uint64_t  ev_upload = ~0ull, ev_active = ~0ull;
    double    update_ms = 0.0;
};

} // namespace vibe
