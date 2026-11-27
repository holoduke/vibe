#pragma once
// Scree streamed round the camera (vk_engine/scree_stream.cpp). Private to that file and the
// two passes that draw it; vk_engine.h only knows the name (scree_).

#include "engine/vk_engine.h"
#include "engine/scatter.h"

#include <map>
#include <utility>
#include <vector>

namespace vibe {

struct VulkanEngine::ScreeState {
    ScatterParams params{};                  // the rule's inputs (build_scatter_from; trees not kept)
    bool          have_params = false;
    Mesh          mesh[kScreeMeshes]{};      // the angular scree meshes
    // A chunk = kChunkCells x kChunkCells of the rule's 5 m cells.
    static constexpr int   kChunkCells = 8;  // 40 m
    struct Chunk { std::vector<ScatterInstance> rocks[kScreeMeshes]; glm::vec3 mn{ 0.0f }, mx{ 0.0f }; };
    std::map<std::pair<int, int>, Chunk> chunks;
    // The instances on the GPU: kBuffers ring buffers (one is rewritten while the others may
    // still be read by the frames in flight), each sorted by mesh then chunk.
    static constexpr int      kBuffers = 3;
    static constexpr uint32_t kCapacity = 60000;
    VkBuffer      buf[kBuffers]{};
    VmaAllocation alloc[kBuffers]{};
    void*         mapped[kBuffers]{};
    int           cur = -1;                  // the buffer drawn from (-1 = none yet)
    std::vector<ScatterCell> cells[kScreeMeshes];        // per mesh: per chunk, a range in buf[cur]
    uint32_t      count = 0;
    bool          dirty = false;
    uint64_t      last_upload = 0;
};

} // namespace vibe
