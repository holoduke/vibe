// Scree and boulder fields streamed round the camera (2026-11-26).
//
// The rule is scatter.cpp scree_place (the reference game's rule for its loose rocks: steep rocky
// ground and the ground below a cliff, clumped, sunk, larger further from the foot). Here it is
// run per 40 m chunk near the eye - a few chunks a frame - and chunks out of the draw range are
// freed, so the fields reach as far as the clutter is drawn anywhere on the home tile (the old
// list was made at start-up inside a 220 m square). The rocks are 4 angular meshes of their own
// (scatter.cpp build_scree_mesh_data), drawn by the clutter's pipeline (scatter.vert: rot > 8 pi)
// and its shadow pass, toned by the baked ground under them (scatter.frag sc_ground_colour).
// VIBE_NO_SCREE=1: none. VIBE_LOG_ROCKS=1: the count each time the set changes.

#include "engine/vk_engine.h"
#include "engine/vk_engine/internal.h"
#include "engine/vk_engine/state/scree.h"
#include "engine/globe.h"
#include "engine/log.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>

namespace vibe {

void VulkanEngine::scree_set_params(const ScatterParams& p) {
    if (!scree_) scree_ = new ScreeState();
    ScreeState& S = *scree_;
    S.params = p;
    S.params.trees = nullptr;                 // (the caller's own list; the rule does not read it)
    S.have_params = true;
    S.chunks.clear();                         // (the ground or the rule's inputs changed: start again)
    S.dirty = true;
}

void VulkanEngine::scree_poll() {
    static const bool off = std::getenv("VIBE_NO_SCREE") != nullptr;
    if (off || !scree_ || !scree_->have_params) return;
    ScreeState& S = *scree_;
    if (!S.mesh[0].vertex_buffer) {
        for (int k = 0; k < kScreeMeshes; ++k) {
            std::vector<Vertex> v; std::vector<uint32_t> ix;
            build_scree_mesh_data(k, v, ix);
            S.mesh[k] = create_mesh_from_data(device_, allocator_, graphics_queue_, graphics_queue_family_, v.data(), uint32_t(v.size()), ix.data(), uint32_t(ix.size()));
        }
        for (int b = 0; b < ScreeState::kBuffers; ++b) {
            VkBufferCreateInfo bci{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
            bci.size = VkDeviceSize(sizeof(ScatterInstance)) * ScreeState::kCapacity;
            bci.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
            VmaAllocationCreateInfo aci{};
            aci.usage = VMA_MEMORY_USAGE_AUTO;
            aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
            VmaAllocationInfo ai{};
            vk_check(vmaCreateBuffer(allocator_, &bci, &aci, &S.buf[b], &S.alloc[b], &ai), "scree instances");
            S.mapped[b] = ai.pMappedData;
        }
    }
    const bool home = globe::current().home;
    const glm::vec3 eye = player_.eye_position();
    const float chunk = kScreeCell * float(ScreeState::kChunkCells);
    const float range = std::max(60.0f, rt_.grass_distance * 1.6f) * 0.85f + 10.0f;      // (scatter.vert: scree is seen to 85 % of the clutter's range)
    // free what is out of range (or everything away from home)
    for (auto it = S.chunks.begin(); it != S.chunks.end();) {
        const glm::vec2 c = (glm::vec2(float(it->first.first), float(it->first.second)) + 0.5f) * chunk;
        if (!home || glm::length(c - glm::vec2(eye.x, eye.z)) > range + chunk * 1.5f) { it = S.chunks.erase(it); S.dirty = true; }
        else ++it;
    }
    // make up to 6 missing chunks a frame, nearest first
    if (home) {
        const int r = int(std::ceil(range / chunk)) + 1;
        const int ex = int(std::floor(eye.x / chunk)), ez = int(std::floor(eye.z / chunk));
        std::vector<std::pair<float, std::pair<int, int>>> want;
        for (int dz = -r; dz <= r; ++dz)
            for (int dx = -r; dx <= r; ++dx) {
                const std::pair<int, int> key(ex + dx, ez + dz);
                const glm::vec2 c = (glm::vec2(float(key.first), float(key.second)) + 0.5f) * chunk;
                if (std::max(std::abs(c.x), std::abs(c.y)) > 1020.0f) continue;        // (the home tile's ground only)
                const float d = glm::length(c - glm::vec2(eye.x, eye.z));
                if (d > range + chunk * 0.75f || S.chunks.count(key)) continue;
                want.push_back({ d, key });
            }
        std::sort(want.begin(), want.end());
        const auto h_at = [this](float x, float z) { return sample_terrain_height(x, z); };
        for (size_t i = 0; i < want.size() && i < 6; ++i) {
            const auto key = want[i].second;
            ScreeState::Chunk& C = S.chunks[key];
            const int c0x = key.first * ScreeState::kChunkCells, c0z = key.second * ScreeState::kChunkCells;
            scree_place(S.params, h_at, c0x, c0x + ScreeState::kChunkCells, c0z, c0z + ScreeState::kChunkCells, C.rocks);
            C.mn = glm::vec3(1e9f); C.mx = glm::vec3(-1e9f);
            for (const auto& v : C.rocks)
                for (const ScatterInstance& si : v) {
                    const glm::vec3 p(si.pos_scale), e(si.pos_scale.w * 2.0f + 0.5f);
                    C.mn = glm::min(C.mn, p - e); C.mx = glm::max(C.mx, p + e);
                }
            S.dirty = true;
        }
    }
    // VIBE_LOG_SCREE=1: once (frame 60), every stone within 25 m of the eye: where, its size, the
    // ground at its middle, the lowest ground under its outline and how far its bed stands over
    // that (> 0: it floats on its downhill side).
    if (static const bool log_scree = std::getenv("VIBE_LOG_SCREE") != nullptr; log_scree && frame_number_ == 60) {
        int n = 0, floating = 0;
        for (const auto& kv : S.chunks)
            for (int k = 0; k < kScreeMeshes; ++k)
                for (const ScatterInstance& si : kv.second.rocks[k]) {
                    const glm::vec3 p(si.pos_scale);
                    if (glm::distance(glm::vec2(p.x, p.z), glm::vec2(eye.x, eye.z)) > 25.0f) continue;
                    const ScreeExtent ext = scree_mesh_extent(k);
                    const float sc = si.pos_scale.w, R = 0.8f * ext.radius * sc;
                    float gmin = sample_terrain_height(p.x, p.z);
                    for (int a = 0; a < 8; ++a) gmin = std::min(gmin, sample_terrain_height(p.x + R * std::cos(float(a) * 0.785398f), p.z + R * std::sin(float(a) * 0.785398f)));
                    const float gap = p.y + ext.bed * sc - gmin;
                    ++n; floating += gap > 0.02f ? 1 : 0;
                    log::infof("[scree] kind %d at (%.1f, %.2f, %.1f) size %.2f (%.1f m across): ground %.2f, lowest under it %.2f, bed over that %+.2f m",
                               k, double(p.x), double(p.y), double(p.z), double(sc), double(2.0f * ext.radius * sc), double(sample_terrain_height(p.x, p.z)), double(gmin), double(gap));
                }
        log::infof("[scree] %d stones within 25 m, %d with their bed over the ground", n, floating);
    }
    // the instances to the GPU: at most every 4th frame, into the ring buffer not in flight
    if (!S.dirty || frame_number_ < S.last_upload + 4) return;
    S.dirty = false;
    S.last_upload = frame_number_;
    const int b = (S.cur + 1) % ScreeState::kBuffers;
    ScatterInstance* dst = static_cast<ScatterInstance*>(S.mapped[b]);
    uint32_t n = 0;
    for (int k = 0; k < kScreeMeshes; ++k) {
        S.cells[k].clear();
        for (const auto& kv : S.chunks) {
            const auto& v = kv.second.rocks[k];
            if (v.empty()) continue;
            const uint32_t take = std::min<uint32_t>(uint32_t(v.size()), ScreeState::kCapacity - n);
            if (!take) break;
            std::memcpy(dst + n, v.data(), sizeof(ScatterInstance) * take);
            ScatterCell c; c.mn = kv.second.mn; c.mx = kv.second.mx; c.first = n; c.count = take;
            S.cells[k].push_back(c);
            n += take;
        }
    }
    vmaFlushAllocation(allocator_, S.alloc[b], 0, VK_WHOLE_SIZE);
    S.cur = b;
    S.count = n;
    static const bool verbose = std::getenv("VIBE_LOG_ROCKS") != nullptr;
    if (verbose) {
        static uint32_t last = ~0u;
        if (last == ~0u || n / 500 != last / 500) log::infof("[scree] %u loose rocks in %zu chunks round (%.0f, %.0f), range %.0f m", n, S.chunks.size(), double(eye.x), double(eye.z), double(range));
        last = n;
    }
}

// One instanced draw per scree mesh over the visible chunks. The caller has bound the pipeline
// and pushed its constants (the clutter's colour pass or its shadow pass).
void VulkanEngine::scree_draw(VkCommandBuffer cmd, const std::function<bool(const ScatterCell&)>& visible) {
    if (!scree_ || scree_->cur < 0 || !scree_->count || !globe::current().home) return;
    ScreeState& S = *scree_;
    for (int k = 0; k < kScreeMeshes; ++k) {
        if (S.cells[k].empty()) continue;
        VkBuffer b[2] = { S.mesh[k].vertex_buffer, S.buf[S.cur] };
        VkDeviceSize o[2] = { 0, 0 };
        vkCmdBindVertexBuffers(cmd, 0, 2, b, o);
        vkCmdBindIndexBuffer(cmd, S.mesh[k].index_buffer, 0, VK_INDEX_TYPE_UINT32);
        scatter_draw_cells(cmd, S.mesh[k].index_count, S.cells[k], visible);
    }
}

void VulkanEngine::scree_destroy() {
    if (!scree_) return;
    ScreeState& S = *scree_;
    for (int k = 0; k < kScreeMeshes; ++k) if (S.mesh[k].vertex_buffer) destroy_mesh(allocator_, S.mesh[k]);
    for (int b = 0; b < ScreeState::kBuffers; ++b) if (S.buf[b]) vmaDestroyBuffer(allocator_, S.buf[b], S.alloc[b]);
    delete scree_;
    scree_ = nullptr;
}

} // namespace vibe
