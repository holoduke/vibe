// World vegetation: trees and bushes everywhere beyond the home tile.
//
// The home tile (2048 m) keeps its own, hand-tuned planting (world_props.cpp
// place_vegetation, the paint map). Everything beyond it is planted by rule:
//
//   * The world is cut into 128 m cells. A cell's plants are a pure function of
//     its coordinates: a jittered grid of candidate spots, each with a random
//     stream of its own, kept where the world's forest rule (world_forest: the
//     same function the far landscape is coloured by) says woods grow. Coming
//     back to a place finds the same trees.
//   * Cells are generated on background threads, nearest first, and live while
//     the camera is in range. Their instances go into a fixed pool of slots
//     inside the vegetation instance buffer, so the existing GPU path draws them:
//     culling, levels of detail, impostors, shadows, ray tracing.
//   * A cell's trees are ordered by a rank. With distance only a leading part of
//     the list is live (all, half, a quarter, a twelfth; bushes only close by),
//     so the far woods are thinner, not different: every tree that stands far
//     away still stands when you get there.
//   * Trunk colliders exist for the cells around the player.
//
// Budget: kWvPool instances at 32 bytes, plus the visible lists. One compute
// thread per slot per frame in veg_cull.comp (an empty slot returns at once).
#include "engine/vk_engine/internal.h"
#include "engine/vk_initializers.h"
#include <cstring>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <future>
#include <cstdlib>
#include <mutex>
#include <thread>
#include <unordered_map>

namespace vibe {

namespace {

constexpr float kCell      = 128.0f;
constexpr float kHomeHalf  = 1000.0f;     // the home tile's own planting reaches this far
constexpr int   kTreeGrid  = 19;          // candidates per cell side: one per 6.7 m (~45 m2 a tree in closed forest)
constexpr int   kBushGrid  = 30;          // one per 4.3 m
constexpr int   kLevels    = 5;
// Live share of a cell's trees by level, and the distance (to the cell's centre) each level reaches.
constexpr float kShare[kLevels] = { 1.0f, 1.0f, 0.5f, 0.25f, 0.085f };
constexpr float kReach[kLevels] = { 260.0f, 650.0f, 1100.0f, 1600.0f, 1e9f };     // level 0 also has the bushes

struct CellKey {
    int x, z;
    bool operator==(const CellKey& o) const { return x == o.x && z == o.z; }
};
struct CellHash {
    size_t operator()(const CellKey& k) const { return size_t(uint32_t(k.x) * 73856093u) ^ size_t(uint32_t(k.z) * 19349663u); }
};

struct CellData {
    std::vector<VegInstance>     items;     // trees by rank (lowest first), then the bushes
    std::vector<collision::AABB> cols;      // one per tree, same order
    uint32_t                     trees = 0;
};

inline uint32_t hash3(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t h = a * 0x9E3779B1u ^ (b + 0x7F4A7C15u) * 0x85EBCA77u ^ (c + 0x165667B1u) * 0xC2B2AE3Du;
    h ^= h >> 15; h *= 0x2C1B3C6Du; h ^= h >> 12; h *= 0x297A2D39u; h ^= h >> 15;
    return h | 1u;
}

}  // namespace

struct VulkanEngine::WvState {
    struct Cell {
        std::shared_ptr<CellData> data;       // null: not generated yet
        std::vector<uint32_t>     slots;      // pool slot of each live item (UINT32_MAX: none was free)
        int  level = kLevels;                 // kLevels: nothing live
        bool requested = false;
        bool cols_live = false;
    };
    std::unordered_map<CellKey, Cell, CellHash> cells;
    std::vector<uint32_t> free_slots;
    std::vector<uint32_t> arch_used;
    glm::vec3 scan_eye{1e9f};
    float     scan_far = 0.0f;
    bool      dirty = true;                   // something is still to be applied: scan again next frame
    glm::vec3 last_eye{1e9f};                 // (camera speed)
    std::chrono::steady_clock::time_point last_t{}, scan_t{};
    bool      parked = false;                 // nothing live, nothing queued (too high / too fast to see trees)

    // Background generation.
    std::vector<std::thread> workers;
    std::mutex mu;
    std::condition_variable cv;
    std::vector<CellKey> jobs;                // nearest LAST (workers pop the back)
    std::vector<std::pair<CellKey, std::shared_ptr<CellData>>> done;
    bool quit = false;
    // The near field for the grass (wv_nf_poll).
    std::future<std::vector<unsigned char>> nf_job;
    bool      nf_active = false, nf_valid = false;
    glm::vec2 nf_job_center{0.0f}, nf_center{0.0f};
    const globe::Chart* nf_chart = nullptr;     // the chart the bake (or the job) was made in
    const globe::Chart* nf_job_chart = nullptr;
    // The bake's texture is made once; later bakes are copied into it at the start of a frame
    // (record_nf_upload), through one staging buffer per frame in flight - no waiting for the GPU.
    VkImage        nf_image = VK_NULL_HANDLE;
    VkBuffer       nf_stage[kFrameOverlap]{};
    VmaAllocation  nf_stage_alloc[kFrameOverlap]{};
    void*          nf_stage_map[kFrameOverlap]{};
    std::vector<unsigned char> nf_pending;
    bool           nf_upload_pending = false;
    // Read-only for the workers.
    std::vector<int> pines, broads, old_pines, old_broads, bushes;
    std::vector<int> river_shrubs;                 // river-bank scrub (world_props.h river_shrub_density)
    std::vector<float> arch_h;
    float water_level = 20.9f;
    int   octaves = 10;
    float tree_density = 1.0f, bush_density = 1.0f;
};

uint32_t VulkanEngine::wv_quota(size_t a) const {
    if (a >= veg_arch_.size() || veg_arch_[a].species > 2) return 0;
    size_t n = 0;
    for (const auto& A : veg_arch_) if (A.species <= 2) ++n;
    // Twice an even share: species are not used evenly (pines up high, broadleaf low).
    return uint32_t(std::max<size_t>(2 * size_t(kWvPool) / std::max<size_t>(n, 1), 3000));
}

// The pool's slots sit between the home tile's ordinary instances and the clustered rocks.
void VulkanEngine::wv_insert_pool() {
    const uint32_t at = veg_cluster_count_ ? veg_cluster_first_ : uint32_t(veg_inst_cpu_.size());
    veg_inst_cpu_.insert(veg_inst_cpu_.begin() + at, size_t(kWvPool), VegInstance{});
    veg_world_first_ = at;
    if (veg_cluster_count_) veg_cluster_first_ += kWvPool;
    else veg_cluster_first_ = uint32_t(veg_inst_cpu_.size());
}

// The near-field ground bake is there and belongs to the chart the engine is in (the shaders'
// nf_setup checks it: scene.globe_info.z).
bool VulkanEngine::wv_nf_ok() const { return wv_ && wv_->nf_valid && wv_->nf_chart == &globe::current(); }

void VulkanEngine::wv_reset() {
    if (!wv_) return;
    WvState& S = *wv_;
    for (auto& kv : S.cells) { kv.second.slots.clear(); kv.second.level = kLevels; kv.second.cols_live = false; }
    S.free_slots.resize(kWvPool);
    for (uint32_t i = 0; i < kWvPool; ++i) S.free_slots[i] = kWvPool - 1 - i;
    S.arch_used.assign(veg_arch_.size(), 0);
    wv_cols_.clear();
    S.scan_eye = glm::vec3(1e9f);
    S.dirty = true;
}

void VulkanEngine::wv_shutdown() {
    if (!wv_) return;
    {
        std::lock_guard<std::mutex> lk(wv_->mu);
        wv_->quit = true;
    }
    wv_->cv.notify_all();
    for (auto& t : wv_->workers) if (t.joinable()) t.join();
    if (wv_->nf_active && wv_->nf_job.valid()) wv_->nf_job.wait();
    for (uint32_t i = 0; i < kFrameOverlap; ++i)
        if (wv_->nf_stage[i]) vmaDestroyBuffer(allocator_, wv_->nf_stage[i], wv_->nf_stage_alloc[i]);
    delete wv_;
    wv_ = nullptr;
}

namespace {

// One cell's plants. Pure: depends on the cell and the world formula only.
std::shared_ptr<CellData> generate_cell(CellKey key, const VulkanEngine::WvState& S) {
    auto out = std::make_shared<CellData>();
    const float x0 = float(key.x) * kCell, z0 = float(key.z) * kCell;
    // Wholly inside the home tile's own planting: nothing.
    if (x0 >= -kHomeHalf && x0 + kCell <= kHomeHalf && z0 >= -kHomeHalf && z0 + kCell <= kHomeHalf) return out;
    const float wl = S.water_level;
    // The ground on a 4 m grid (the landscape formula is the cost of a cell): candidates read
    // height and slope from it; a plant that is kept gets its exact height.
    constexpr int G = 33;
    constexpr float gs = kCell / float(G - 1);
    float hg[G * G];
    float hmin = 1e9f, hmax = -1e9f;
    for (int j = 0; j < G; ++j)
        for (int i = 0; i < G; ++i) {
            const float h = world_ground_height(x0 + float(i) * gs, z0 + float(j) * gs, wl, S.octaves);
            hg[j * G + i] = h;
            hmin = std::min(hmin, h); hmax = std::max(hmax, h);
        }
    if (hmax < wl + 1.0f || hmin > wl + 64.0f) return out;       // open water, or all above the treeline
    auto h_at = [&](float lx, float lz) {
        const float fx = std::clamp(lx / gs, 0.0f, float(G - 1) - 0.001f), fz = std::clamp(lz / gs, 0.0f, float(G - 1) - 0.001f);
        const int ix = int(fx), iz = int(fz);
        const float tx = fx - float(ix), tz = fz - float(iz);
        const float* r0 = &hg[iz * G + ix];
        const float* r1 = r0 + G;
        return (r0[0] * (1.0f - tx) + r0[1] * tx) * (1.0f - tz) + (r1[0] * (1.0f - tx) + r1[1] * tx) * tz;
    };
    auto ny_at = [&](float lx, float lz) {
        const float e = 3.0f;
        const float dx = h_at(lx + e, lz) - h_at(lx - e, lz), dz = h_at(lx, lz + e) - h_at(lx, lz - e);
        return 2.0f * e / std::sqrt(dx * dx + dz * dz + 4.0f * e * e);
    };
    // (nothing is planted at home - it has its own vegetation - nor on the launch site)
    auto in_home = [](float x, float z) {
        const float dx = x - kLaunchSiteX, dz = z - kLaunchSiteZ;
        return (std::abs(x) < kHomeHalf && std::abs(z) < kHomeHalf) || dx * dx + dz * dz < 150.0f * 150.0f;
    };

    struct Ranked { float rank; VegInstance vi; collision::AABB col; };
    std::vector<Ranked> trees;
    const float ts = kCell / float(kTreeGrid);
    for (int j = 0; j < kTreeGrid; ++j)
        for (int i = 0; i < kTreeGrid; ++i) {
            uint32_t rs = hash3(uint32_t(key.x), uint32_t(key.z), uint32_t(j * kTreeGrid + i));
            auto u = [&rs]() { rs ^= rs << 13; rs ^= rs >> 17; rs ^= rs << 5; return float(rs >> 8) / 16777216.0f; };
            // (jitter kept off the cell's edges: neighbours never stand in each other's trunk)
            const float lx = (float(i) + 0.14f + 0.72f * u()) * ts, lz = (float(j) + 0.14f + 0.72f * u()) * ts;
            const float x = x0 + lx, z = z0 + lz;
            const float pick = u(), kind = u(), size = u(), oldp = u(), rot = u(), phase = u(), which = u(), rank = u();
            if (in_home(x, z)) continue;
            const float hc = h_at(lx, lz);
            if (hc < wl + 2.0f) continue;
            const float f = world_forest(x, z, hc, ny_at(lx, lz), wl);
            // Closed forest keeps nearly every spot; thin woods few; a rare lone tree in open country.
            const float dens = f * f * 0.96f + ((hc < wl + 46.0f && ny_at(lx, lz) > 0.8f) ? 0.004f : 0.0f);
            if (pick > dens * S.tree_density) continue;
            const float h = world_ground_height(x, z, wl, S.octaves);
            if (h < wl + 2.0f) continue;
            const bool pine = kind < std::clamp((h - 28.0f) / 30.0f, 0.15f, 0.95f);
            // Old-growth stands: patches a few hundred metres across where the veterans stand.
            const float og = world_forest(x * 0.37f + 5100.0f, z * 0.37f - 3300.0f, wl + 20.0f, 1.0f, wl);
            const auto& oldp_pool = pine ? S.old_pines : S.old_broads;
            const bool old_tree = og > 0.75f && f > 0.6f && !oldp_pool.empty() && oldp < 0.55f;
            const auto& pool = old_tree ? oldp_pool : (pine ? S.pines : S.broads);
            if (pool.empty()) continue;
            const float scale = old_tree ? 0.95f + 0.35f * size : 0.75f + 0.55f * size;
            const int ai = pool[size_t(which * float(pool.size())) % pool.size()];
            Ranked r;
            // Big trees outlast small ones as the wood thins with distance.
            r.rank = rank / (scale * (old_tree ? 1.8f : 1.0f));
            r.vi.pos_scale = glm::vec4(x, h - 0.05f, z, scale);
            r.vi.rot_arch_phase = glm::vec4(rot * 6.2831853f, float(ai), phase * 6.2831853f, 0.0f);
            const float tr = 0.38f * scale;
            r.col = { glm::vec3(x - tr, h - 0.5f, z - tr), glm::vec3(x + tr, h + S.arch_h[size_t(ai)] * scale * 0.6f, z + tr) };
            trees.push_back(r);
        }
    std::sort(trees.begin(), trees.end(), [](const Ranked& a, const Ranked& b) { return a.rank < b.rank; });
    out->trees = uint32_t(trees.size());
    out->items.reserve(trees.size() + 64);
    out->cols.reserve(trees.size());
    for (const auto& t : trees) { out->items.push_back(t.vi); out->cols.push_back(t.col); }

    // Bushes: undergrowth inside the woods and thickets along their edges.
    if (!S.bushes.empty()) {
        const float bs = kCell / float(kBushGrid);
        for (int j = 0; j < kBushGrid; ++j)
            for (int i = 0; i < kBushGrid; ++i) {
                uint32_t rs = hash3(uint32_t(key.x) + 0x51EDu, uint32_t(key.z), uint32_t(j * kBushGrid + i));
                auto u = [&rs]() { rs ^= rs << 13; rs ^= rs >> 17; rs ^= rs << 5; return float(rs >> 8) / 16777216.0f; };
                const float lx = (float(i) + u()) * bs, lz = (float(j) + u()) * bs;
                const float x = x0 + lx, z = z0 + lz;
                const float pick = u(), size = u(), rot = u(), phase = u(), which = u();
                if (in_home(x, z)) continue;
                const float hc = h_at(lx, lz);
                if (hc < wl + 1.2f) continue;
                const float ny = ny_at(lx, lz);
                if (ny < 0.75f) continue;
                const float f = world_forest(x, z, hc, ny, wl);
                const float dens = std::clamp(f * 0.45f + (1.0f - std::abs(f - 0.4f) * 3.0f) * 0.5f, 0.0f, 1.0f) * 0.42f * (f > 0.02f ? 1.0f : 0.0f);
                if (pick > dens * S.bush_density) continue;
                const float h = world_ground_height(x, z, wl, S.octaves);
                if (h < wl + 1.0f) continue;
                const int ai = S.bushes[size_t(which * float(S.bushes.size())) % S.bushes.size()];
                VegInstance vi{};
                vi.pos_scale = glm::vec4(x, h, z, 0.6f + 0.9f * size);
                vi.rot_arch_phase = glm::vec4(rot * 6.2831853f, float(ai), phase * 6.2831853f, 0.0f);
                out->items.push_back(vi);
            }
    }
    // River-bank scrub, where the river crosses this cell (the same rule as on the home tile).
    if (!S.river_shrubs.empty() && x0 + kCell > -85.0f && x0 < 2095.0f && z0 + kCell > -110.0f && z0 < 780.0f) {
        constexpr int kRiverGrid = 40;                              // 3.2 m cells: one try each
        const float rs_cell = kCell / float(kRiverGrid);
        for (int j = 0; j < kRiverGrid; ++j)
            for (int i = 0; i < kRiverGrid; ++i) {
                uint32_t rs = hash3(uint32_t(key.x) + 0xB1FEu, uint32_t(key.z), uint32_t(j * kRiverGrid + i));
                auto u = [&rs]() { rs ^= rs << 13; rs ^= rs >> 17; rs ^= rs << 5; return float(rs >> 8) / 16777216.0f; };
                const float lx = (float(i) + u()) * rs_cell, lz = (float(j) + u()) * rs_cell;
                const float x = x0 + lx, z = z0 + lz;
                const float pick = u(), size = u(), rot = u(), phase = u(), which = u();
                if (in_home(x, z)) continue;
                if (world_river_distance(x, z) > 40.0f) continue;
                const float h = world_ground_height(x, z, wl, S.octaves);
                if (pick > river_shrub_density(x, z, h, ny_at(lx, lz), wl) * S.bush_density * 0.55f) continue;
                const int ai = S.river_shrubs[size_t(which * float(S.river_shrubs.size())) % S.river_shrubs.size()];
                VegInstance vi{};
                vi.pos_scale = glm::vec4(x, h - 0.06f, z, 0.75f + 0.55f * size);
                vi.rot_arch_phase = glm::vec4(rot * 6.2831853f, float(ai), phase * 6.2831853f, 0.0f);
                out->items.push_back(vi);
            }
    }
    return out;
}

}  // namespace

// The ground around the camera for the world's grass (grass_raymarch.frag nf_*): 512 x 512 m at
// 1 m, re-baked on background threads when the camera has moved 72 m from the last centre, and
// swapped into normal-texture slot kTexVegShade (unused otherwise). Only near the ground and
// outside (or near the edge of) the home tile, where the grass shader has its own baked heights.
// A finished near-field bake goes into its texture here, with the frame's other uploads.
void VulkanEngine::record_nf_upload(VkCommandBuffer cmd, uint32_t slot) {
    if (!wv_ || !wv_->nf_upload_pending || !wv_->nf_image) return;
    WvState& S = *wv_;
    S.nf_upload_pending = false;
    constexpr uint32_t N = 512;
    if (S.nf_pending.size() != size_t(N) * N * 4) return;
    slot %= kFrameOverlap;
    if (!S.nf_stage[slot]) {
        VkBufferCreateInfo bci{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
        bci.size = VkDeviceSize(N) * N * 4;
        bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
        aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
        VmaAllocationInfo ai{};
        if (vmaCreateBuffer(allocator_, &bci, &aci, &S.nf_stage[slot], &S.nf_stage_alloc[slot], &ai) != VK_SUCCESS) return;
        S.nf_stage_map[slot] = ai.pMappedData;
    }
    std::memcpy(S.nf_stage_map[slot], S.nf_pending.data(), S.nf_pending.size());
    vmaFlushAllocation(allocator_, S.nf_stage_alloc[slot], 0, VK_WHOLE_SIZE);
    vkinit::transition_image(cmd, S.nf_image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    VkBufferImageCopy reg{};
    reg.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    reg.imageExtent = { N, N, 1 };
    vkCmdCopyBufferToImage(cmd, S.nf_stage[slot], S.nf_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &reg);
    vkinit::transition_image(cmd, S.nf_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
}

void VulkanEngine::wv_nf_poll() {
    if (!wv_ || !scene_desc_set_) return;
    WvState& S = *wv_;
    const glm::vec3 eye = S.scan_eye;
    if (eye.x > 1e8f) return;
    constexpr int N = 512;
    if (S.nf_active && S.nf_job.valid() && S.nf_job.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
        const std::vector<unsigned char> px = S.nf_job.get();
        S.nf_active = false;
        if (S.nf_job_chart != &globe::current()) return;      // (baked for another part of the planet: re-anchored meanwhile)
        if (S.nf_image) {
            // (every bake after the first: into the texture that is already bound)
            S.nf_pending = px;
            S.nf_upload_pending = true;
            S.nf_center = S.nf_job_center;
            S.nf_chart = S.nf_job_chart;
            S.nf_valid = true;
            return;
        }
        Texture2D t = upload_texture_from_pixels(device_, allocator_, graphics_queue_, graphics_queue_family_,
                                                 px.data(), N, N, VK_FORMAT_R8G8B8A8_UNORM, "near_ground");
        if (t.ok) {
            texture_upload_batch_sync(device_, allocator_, graphics_queue_);
            vkDeviceWaitIdle(device_);
            TextureSlot old = normal_textures_[kTexVegShade];
            normal_textures_[kTexVegShade] = { t.image, t.alloc, t.view };
            S.nf_image = t.image;
            VkDescriptorImageInfo ii{ texture_sampler_, t.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
            VkWriteDescriptorSet w{};
            w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w.dstSet = scene_desc_set_; w.dstBinding = 4; w.dstArrayElement = uint32_t(kTexVegShade); w.descriptorCount = 1;
            w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w.pImageInfo = &ii;
            vkUpdateDescriptorSets(device_, 1, &w, 0, nullptr);
            if (old.view) vkDestroyImageView(device_, old.view, nullptr);
            if (old.image) vmaDestroyImage(allocator_, old.image, old.alloc);
            S.nf_center = S.nf_job_center;
            S.nf_chart = S.nf_job_chart;
            S.nf_valid = true;
        }
    }
    const bool want = std::max(std::abs(eye.x), std::abs(eye.z)) > 1024.0f - 260.0f && eye.y - rt_.water_level < 450.0f;
    if (S.nf_active || !want) return;
    const glm::vec2 e2(eye.x, eye.z);
    if (S.nf_chart != &globe::current()) S.nf_valid = false;
    // VIBE_TEST_NF_REBAKE=1: bake again every few seconds (the in-frame upload path, without having to move).
    { static const bool rb = std::getenv("VIBE_TEST_NF_REBAKE") != nullptr; if (rb && frame_number_ % 200 == 0) S.nf_valid = false; }
    if (S.nf_valid && glm::distance(e2, S.nf_center) < 72.0f) return;
    const glm::vec2 c = glm::round(e2 / 64.0f) * 64.0f;
    S.nf_job_center = c;
    S.nf_job_chart = &globe::current();
    S.nf_active = true;
    const float wl = S.water_level;
    S.nf_job = std::async(std::launch::async, [c, wl]() {
        std::vector<unsigned char> px(size_t(N) * N * 4, 255);
        const float ox = c.x - 256.0f, oz = c.y - 256.0f;
        const unsigned nt = std::clamp(std::thread::hardware_concurrency() / 3u, 1u, 6u);
        std::vector<std::thread> th;
        for (unsigned t = 0; t < nt; ++t)
            th.emplace_back([&, t]() {
                for (int j = 1 + int(t); j < N; j += int(nt))
                    for (int i = 0; i < N; ++i) {
                        // (9 octaves: what the grass stands on at home too - the bake there is 9 octaves)
                        const float h = world_ground_height(ox + float(i), oz + float(j), wl, 9);
                        const uint32_t v = uint32_t(std::clamp(h + 64.0f, 0.0f, 255.99f) * 65536.0f);
                        unsigned char* p = &px[(size_t(j) * N + size_t(i)) * 4];
                        p[0] = (unsigned char)(v >> 16); p[1] = (unsigned char)(v >> 8); p[2] = (unsigned char)v;
                    }
            });
        for (auto& x : th) x.join();
        const uint32_t ix = uint32_t(int(std::lround(ox / 64.0f)) + 8388608), iz = uint32_t(int(std::lround(oz / 64.0f)) + 8388608);
        px[0] = (unsigned char)(ix >> 16); px[1] = (unsigned char)(ix >> 8); px[2] = (unsigned char)ix;
        px[4] = (unsigned char)(iz >> 16); px[5] = (unsigned char)(iz >> 8); px[6] = (unsigned char)iz;
        px[8] = 77; px[9] = 123; px[10] = 201;
        return px;
    });
}

void VulkanEngine::wv_update(glm::vec3 eye) {
    if (veg_world_first_ == UINT32_MAX || !veg_inst_mapped_ || veg_arch_.empty()) return;
    static const bool off = std::getenv("VIBE_NO_WORLD_VEG") != nullptr;
    if (off) return;
    if (!wv_) {
        wv_ = new WvState();
        WvState& S = *wv_;
        for (size_t a = 0; a < veg_arch_.size(); ++a) {
            const auto& A = veg_arch_[a];
            S.arch_h.push_back(A.height);
            if (A.species == 0) (A.old ? S.old_pines : S.pines).push_back(int(a));
            else if (A.species == 1) (A.old ? S.old_broads : S.broads).push_back(int(a));
            else if (A.species == 2) (A.riverside ? S.river_shrubs : S.bushes).push_back(int(a));
        }
        S.water_level  = rt_.water_level;
        S.octaves      = std::clamp(rt_.terrain_raymarch_octaves, 4, 24);
        S.tree_density = glm::clamp(rt_.forest_density, 0.0f, 2.0f);
        S.bush_density = glm::clamp(rt_.bush_density, 0.0f, 2.0f);
        wv_reset();
        const unsigned nt = std::clamp(std::thread::hardware_concurrency() / 3u, 1u, 4u);
        for (unsigned t = 0; t < nt; ++t)
            S.workers.emplace_back([this]() {
                WvState& W = *wv_;
                for (;;) {
                    CellKey k;
                    {
                        std::unique_lock<std::mutex> lk(W.mu);
                        W.cv.wait(lk, [&]() { return W.quit || !W.jobs.empty(); });
                        if (W.quit) return;
                        k = W.jobs.back();
                        W.jobs.pop_back();
                    }
                    auto d = generate_cell(k, W);
                    std::lock_guard<std::mutex> lk(W.mu);
                    W.done.emplace_back(k, std::move(d));
                }
            });
        log::infof("[wveg] world vegetation: %u slots, %u generator threads", unsigned(kWvPool), nt);
    }
    WvState& S = *wv_;
    VegInstance* gpu = static_cast<VegInstance*>(veg_inst_mapped_);

    // Take what the generators finished.
    {
        std::lock_guard<std::mutex> lk(S.mu);
        for (auto& d : S.done) {
            auto it = S.cells.find(d.first);
            if (it != S.cells.end()) { it->second.data = std::move(d.second); S.dirty = true; }
        }
        S.done.clear();
    }

    const float far_m = object_far_ + 0.75f * kCell;
    // Camera speed. Above the height from which no tree can be seen, and while travelling faster
    // than cells could ever be generated for (space flight covers kilometres a frame), there is
    // nothing to do: everything is given back once and the generators' queue is emptied. This
    // was the frame-rate drop in space - a full rescan and thousands of new requests every frame.
    const auto now = std::chrono::steady_clock::now();
    const float dt = std::max(std::chrono::duration<float>(now - S.last_t).count(), 1e-3f);
    const float speed = S.last_eye.x > 1e8f ? 0.0f : glm::distance(eye, S.last_eye) / dt;
    S.last_eye = eye; S.last_t = now;
    const float height = eye.y - rt_.water_level;
    if (height > far_m + 400.0f || speed > 900.0f) {
        if (!S.parked) {
            {
                std::lock_guard<std::mutex> lk(S.mu);
                S.jobs.clear();
            }
            bool any = false;
            for (auto& kv : S.cells) {
                WvState::Cell& c = kv.second;
                if (c.data && !c.slots.empty()) {
                    for (size_t i = 0; i < c.slots.size(); ++i) {
                        const uint32_t sl = c.slots[i];
                        if (sl == UINT32_MAX) continue;
                        gpu[veg_world_first_ + sl].pos_scale.w = 0.0f;
                        veg_inst_cpu_[veg_world_first_ + sl].pos_scale.w = 0.0f;
                        any = true;
                    }
                }
            }
            S.cells.clear();
            S.free_slots.resize(kWvPool);
            for (uint32_t i = 0; i < kWvPool; ++i) S.free_slots[i] = kWvPool - 1 - i;
            S.arch_used.assign(veg_arch_.size(), 0);
            if (!wv_cols_.empty()) { wv_cols_.clear(); near_static_src_[0] = SIZE_MAX; }
            if (any) {
                vmaFlushAllocation(allocator_, veg_inst_alloc_, sizeof(VegInstance) * veg_world_first_, sizeof(VegInstance) * kWvPool);
                veg_rt_valid_ = false;
            }
            S.parked = true;
        }
        S.scan_eye = eye;                       // (the near-field bake reads it)
        return;
    }
    const bool moved = S.parked || glm::distance(glm::vec2(eye.x, eye.z), glm::vec2(S.scan_eye.x, S.scan_eye.z)) > 24.0f ||
                       std::abs(far_m - S.scan_far) > 1.0f;
    if (!moved && !S.dirty) return;
    // At most ten scans a second (a scan walks every cell in range).
    if (!S.parked && std::chrono::duration<float>(now - S.scan_t).count() < 0.1f) { S.dirty = true; return; }
    S.parked = false;
    S.scan_t = now;
    S.scan_eye = eye; S.scan_far = far_m; S.dirty = false;

    auto set_live = [&](WvState::Cell& c, size_t want) {
        const CellData& D = *c.data;
        while (c.slots.size() > want) {                          // drop the tail
            const uint32_t s = c.slots.back();
            c.slots.pop_back();
            if (s == UINT32_MAX) continue;
            const size_t a = size_t(D.items[c.slots.size()].rot_arch_phase.y + 0.5f);
            gpu[veg_world_first_ + s].pos_scale.w = 0.0f;
            veg_inst_cpu_[veg_world_first_ + s].pos_scale.w = 0.0f;
            if (a < S.arch_used.size() && S.arch_used[a]) --S.arch_used[a];
            S.free_slots.push_back(s);
        }
        while (c.slots.size() < want) {                          // add to the tail
            const VegInstance& vi = D.items[c.slots.size()];
            const size_t a = size_t(vi.rot_arch_phase.y + 0.5f);
            uint32_t s = UINT32_MAX;
            if (!S.free_slots.empty() && a < S.arch_used.size() && S.arch_used[a] < wv_quota(a)) {
                s = S.free_slots.back();
                S.free_slots.pop_back();
                ++S.arch_used[a];
                // (the scale last: a slot the GPU reads half-written is still an empty one)
                VegInstance& g = gpu[veg_world_first_ + s];
                g.rot_arch_phase = vi.rot_arch_phase;
                g.pos_scale = vi.pos_scale;
                veg_inst_cpu_[veg_world_first_ + s] = vi;
            }
            c.slots.push_back(s);
        }
    };

    const int ccx = int(std::floor(eye.x / kCell)), ccz = int(std::floor(eye.z / kCell));
    const int rc = int(std::ceil(far_m / kCell)) + 1;
    std::vector<std::pair<float, CellKey>> want_jobs;
    bool cols_changed = false, any_change = false;
    // Camera height: from the air the forest thins out as a whole (no bushes under an aeroplane).
    // Not by the 3D distance to each cell, as it was: that kept the full forest in a disc under
    // the camera whose radius shrank while climbing - seen from above, a dark circle on the
    // ground that closed in on the spot below. Now a cell's level is the coarser of what its
    // distance along the ground and what the camera's height call for: the thinning by
    // height applies to everything in view at once.
    const float eye_h = std::max(eye.y - rt_.water_level - 60.0f, 0.0f);
    for (int dz = -rc; dz <= rc; ++dz)
        for (int dx = -rc; dx <= rc; ++dx) {
            const CellKey k{ ccx + dx, ccz + dz };
            const float cx = (float(k.x) + 0.5f) * kCell - eye.x, cz = (float(k.z) + 0.5f) * kCell - eye.z;
            const float d_h = std::sqrt(cx * cx + cz * cz);
            if (d_h > far_m) continue;
            // Wholly inside the home tile's planting: never anything there.
            {
                const float x0 = float(k.x) * kCell, z0 = float(k.z) * kCell;
                if (x0 >= -kHomeHalf && x0 + kCell <= kHomeHalf && z0 >= -kHomeHalf && z0 + kCell <= kHomeHalf) continue;
            }
            WvState::Cell& c = S.cells[k];
            const float d = std::sqrt(d_h * d_h + eye_h * eye_h);        // (the order cells are generated in)
            int lvl = kLevels - 1, lvl_a = kLevels - 1;
            for (int l = 0; l < kLevels; ++l) {
                // (a cell keeps its level until 8 % past the border: no flicker on a boundary)
                const float reach = kReach[l] * ((c.level <= l) ? 1.08f : 1.0f);
                if (d_h < reach) { lvl = l; break; }
            }
            for (int l = 0; l < kLevels; ++l) {
                const float reach = kReach[l] * ((c.level <= l) ? 1.08f : 1.0f);
                if (eye_h < reach) { lvl_a = l; break; }
            }
            lvl = std::max(lvl, lvl_a);
            if (!c.data) {
                c.requested = true;
                want_jobs.push_back({ d, k });
                continue;
            }
            if (lvl != c.level) {
                const CellData& D = *c.data;
                const size_t want = lvl == 0 ? D.items.size() : size_t(std::ceil(float(D.trees) * kShare[lvl]));
                set_live(c, want);
                c.level = lvl;
                any_change = true;
            }
            const bool cols = d_h < 200.0f && eye_h < 150.0f;
            if (cols != c.cols_live) { c.cols_live = cols; cols_changed = true; }
        }
    // Out of range: give the slots back; forget cells well outside (their data is regenerated on return).
    for (auto it = S.cells.begin(); it != S.cells.end();) {
        const float cx = (float(it->first.x) + 0.5f) * kCell - eye.x, cz = (float(it->first.z) + 0.5f) * kCell - eye.z;
        const float d = std::sqrt(cx * cx + cz * cz);
        if (d > far_m && it->second.level != kLevels && it->second.data) {
            set_live(it->second, 0);
            it->second.level = kLevels;
            any_change = true;
            if (it->second.cols_live) { it->second.cols_live = false; cols_changed = true; }
        }
        // (a cell that left the range before it was generated is forgotten too: its request goes with it)
        if ((d > far_m + 6.0f * kCell && it->second.data) || (d > far_m && !it->second.data)) it = S.cells.erase(it);
        else ++it;
    }
    if (any_change) {
        vmaFlushAllocation(allocator_, veg_inst_alloc_, sizeof(VegInstance) * veg_world_first_, sizeof(VegInstance) * kWvPool);
        veg_rt_valid_ = false;                  // the ray-traced tree set is re-picked
    }
    if (cols_changed) {
        wv_cols_.clear();
        for (const auto& kv : S.cells) {
            const auto& c = kv.second;
            if (!c.cols_live || !c.data) continue;
            const size_t n = std::min<size_t>(c.slots.size(), c.data->trees);
            for (size_t i = 0; i < n; ++i)
                if (c.slots[i] != UINT32_MAX) wv_cols_.push_back(c.data->cols[i]);
        }
        near_static_src_[0] = SIZE_MAX;         // the player's broadphase takes the new set
    }
    {
        // The queue is exactly what is wanted NOW, nearest last (workers take from the back).
        // (A cell a worker is busy with may be listed again: it is then made twice - harmless.)
        std::sort(want_jobs.begin(), want_jobs.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
        std::lock_guard<std::mutex> lk(S.mu);
        S.jobs.clear();
        for (const auto& j : want_jobs) S.jobs.push_back(j.second);
        if (!S.jobs.empty()) { S.dirty = true; S.cv.notify_all(); }
    }
}

}  // namespace vibe
