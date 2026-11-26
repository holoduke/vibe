// terrain_shadow_bake.cpp - baked terrain sun-shadow texture and its background worker
// (split out of terrain_resources.cpp; see docs/ARCHITECTURE.md for the map.)
// Terrain CPU/GPU resources: heightmap sculpting (FBM noise/erosion brush),
// grass-eligibility mask, height textures, quadtree CLOD instancing, the
// async sun-shadow bake worker, and the curvature/sky-vis aux + grass-mask
// + fog-wisp procedural textures. Split out of world.cpp (maintainability).

#include "engine/vk_engine/internal.h"
#include "engine/gltf_loader.h"
#include "engine/region_tile.h"
#include "engine/grass.h"
#include "engine/skybox.h"
#include "engine/terrain.h"
#include "engine/texture.h"

#include <GLFW/glfw3.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <future>
#include <limits>
#include <utility>
#include <vector>

namespace vibe {

void VulkanEngine::destroy_terrain_shadow_texture() {
    deferred_desc_sig_ = 0;   // handles may be recreated (even at the same address)
    if (terrain_shadow_view_) {
        vkDestroyImageView(device_, terrain_shadow_view_, nullptr);
        terrain_shadow_view_ = VK_NULL_HANDLE;
    }
    if (terrain_shadow_image_) {
        vmaDestroyImage(allocator_, terrain_shadow_image_, terrain_shadow_alloc_);
        terrain_shadow_image_ = VK_NULL_HANDLE;
        terrain_shadow_alloc_ = nullptr;
    }
    if (terrain_shadow_sampler_) {
        vkDestroySampler(device_, terrain_shadow_sampler_, nullptr);
        terrain_shadow_sampler_ = VK_NULL_HANDLE;
    }
    terrain_shadow_dim_ = 0;
}

namespace {
// First-time creation of the heightmap shadow image + view + sampler.
// Pulled out of rebuild_terrain_shadow_texture so the progressive tile
// path can rely on the resources existing without duplicating setup.
// Caller writes binding 6 separately; transitions the image into
// SHADER_READ_ONLY_OPTIMAL on first creation (so partial uploads can
// flip TRANSFER_DST → SHADER_READ_ONLY symmetrically).
void ensure_shadow_resources(VkDevice device, VmaAllocator alloc,
                             VkQueue queue, uint32_t qfam,
                             int W, int H,
                             VkImage& img, VmaAllocation& img_alloc,
                             VkImageView& view, VkSampler& sampler,
                             int& dim_out) {
    if (img) return;
    VkImageCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType = VK_IMAGE_TYPE_2D;
    // RG8: R = shadow (0/1), G = blocker distance for shader-side PCSS.
    ici.format = VK_FORMAT_R8G8_UNORM;
    ici.extent = { static_cast<uint32_t>(W),
                   static_cast<uint32_t>(H), 1 };
    ici.mipLevels = 1; ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_SAMPLED_BIT |
                VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    vk_check(vmaCreateImage(alloc, &ici, &aci, &img, &img_alloc, nullptr),
             "terrain shadow image");

    VkImageViewCreateInfo vci{};
    vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vci.image = img;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = VK_FORMAT_R8G8_UNORM;
    vci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    vci.subresourceRange.baseMipLevel = 0;
    vci.subresourceRange.levelCount = 1;
    vci.subresourceRange.baseArrayLayer = 0;
    vci.subresourceRange.layerCount = 1;
    vk_check(vkCreateImageView(device, &vci, nullptr, &view),
             "terrain shadow view");

    VkSamplerCreateInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    si.magFilter = VK_FILTER_LINEAR;
    si.minFilter = VK_FILTER_LINEAR;
    si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    si.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    vk_check(vkCreateSampler(device, &si, nullptr, &sampler),
             "terrain shadow sampler");

    // Initialise to SHADER_READ_ONLY so the very first tick's
    // tile transition (SHADER_READ_ONLY → TRANSFER_DST → SHADER_READ_ONLY)
    // is symmetric. Without this the layout starts UNDEFINED and the
    // first frame would have to special-case the source layout.
    vkinit::one_time_submit(device, queue, qfam, [&](VkCommandBuffer cb) {
        vkinit::transition_image(cb, img,
                                  VK_IMAGE_LAYOUT_UNDEFINED,
                                  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    });
    dim_out = W;
}
} // namespace

void VulkanEngine::rebuild_terrain_shadow_texture(bool defer_bake) {
    if (terrain_data_.heights.empty()) return;
    // Reconstruct a Heightmap view over the cached data.
    Heightmap hm{};
    hm.dim       = terrain_data_.dim;
    hm.cell      = terrain_data_.cell;
    hm.origin_x  = terrain_data_.origin_x;
    hm.origin_z  = terrain_data_.origin_z;
    hm.heights   = terrain_data_.heights;

    // Sun direction from the user settings (matches what cube.frag uses).
    float p = glm::radians(rt_.sun_pitch_deg);
    float y = glm::radians(rt_.sun_yaw_deg);
    glm::vec3 sun_dir(std::sin(y) * std::cos(p), std::sin(p),
                      std::cos(y) * std::cos(p));
    terrain_shadow_sun_dir_        = sun_dir;
    terrain_shadow_target_sun_dir_ = sun_dir;
    {
        std::lock_guard<std::mutex> lk(terrain_shadow_mutex_);
        terrain_shadow_jobs_.clear();
        terrain_shadow_results_.clear();
    }

    // Sync bake stays at SS=1 for fast level load (~150ms). When the
    // user has supersample > 1 the texture is created at SS×heightmap-dim,
    // and we replicate each SS=1 texel into an SS×SS block so the
    // texture is fully populated from frame 1 — the worker then
    // re-bakes properly at the user's SS in the background.
    int ss = std::max(1, rt_.terrain_bake_supersample);
    {   // The texture is (dim+1)*ss per side: stay within the GPU limit
        // (4097 * 4 = 16388 already exceeds 16384 on many GPUs).
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(physical_device_, &props);
        const uint32_t lim = props.limits.maxImageDimension2D;
        while (ss > 1 && uint32_t(std::max(hm.width(), hm.height()) * ss) > lim) ss /= 2;
        if (ss != rt_.terrain_bake_supersample) {
            log::warnf("[terrain] shadow bake supersample %d -> %d (max image dimension %u)",
                       rt_.terrain_bake_supersample, ss, lim);
            rt_.terrain_bake_supersample = ss;   // else the per-frame change check fires forever
        }
    }
    terrain_shadow_active_ss_ = ss;
    std::vector<uint8_t> base;
    if (defer_bake) {
        // R = 0 (lit), G = 255 (no occluder) - the worker streams in the
        // real bake right after startup, nearest tiles first.
        base.resize(static_cast<size_t>(hm.width()) * static_cast<size_t>(hm.height()) * 2);
        for (size_t i = 0; i < base.size(); i += 2) { base[i] = 0; base[i + 1] = 255; }
    } else {
        base = bake_heightmap_shadow(hm, sun_dir);
    }
    const int W = hm.width()  * ss;
    const int H = hm.height() * ss;
    std::vector<uint8_t> shadow;
    if (ss == 1) {
        shadow = std::move(base);
    } else {
        shadow.resize(static_cast<size_t>(W) * static_cast<size_t>(H) * 2);
        for (int z = 0; z < H; ++z) {
            for (int x = 0; x < W; ++x) {
                int sx = x / ss;
                int sz = z / ss;
                size_t dst = (static_cast<size_t>(z) * static_cast<size_t>(W) +
                              static_cast<size_t>(x)) * 2;
                size_t src = (static_cast<size_t>(sz) * static_cast<size_t>(hm.width()) +
                              static_cast<size_t>(sx)) * 2;
                shadow[dst]     = base[src];
                shadow[dst + 1] = base[src + 1];
            }
        }
    }

    const bool first_create = (terrain_shadow_image_ == VK_NULL_HANDLE);
    ensure_shadow_resources(device_, allocator_, graphics_queue_,
                             graphics_queue_family_, W, H,
                             terrain_shadow_image_, terrain_shadow_alloc_,
                             terrain_shadow_view_, terrain_shadow_sampler_,
                             terrain_shadow_dim_);

    // Stage upload via a scratch host-visible buffer (2 bytes/texel RG8).
    const VkDeviceSize bytes =
        static_cast<VkDeviceSize>(W) * static_cast<VkDeviceSize>(H) * 2;
    VkBuffer stage = VK_NULL_HANDLE;
    VmaAllocation stage_alloc = nullptr;
    {
        VkBufferCreateInfo bci{
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .pNext = nullptr, .flags = 0, .size = bytes,
            .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0, .pQueueFamilyIndices = nullptr,
        };
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
        aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                    VMA_ALLOCATION_CREATE_MAPPED_BIT;
        vmaCreateBuffer(allocator_, &bci, &aci, &stage, &stage_alloc, nullptr);
        VmaAllocationInfo ai{};
        vmaGetAllocationInfo(allocator_, stage_alloc, &ai);
        std::memcpy(ai.pMappedData, shadow.data(), static_cast<size_t>(bytes));
    }

    vkinit::one_time_submit(device_, graphics_queue_, graphics_queue_family_,
                            [&](VkCommandBuffer cb) {
        vkinit::transition_image(cb, terrain_shadow_image_,
                                  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        VkBufferImageCopy region{};
        region.bufferOffset = 0;
        region.bufferRowLength = 0;
        region.bufferImageHeight = 0;
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.mipLevel = 0;
        region.imageSubresource.baseArrayLayer = 0;
        region.imageSubresource.layerCount = 1;
        region.imageOffset = {0, 0, 0};
        region.imageExtent = { static_cast<uint32_t>(W),
                                static_cast<uint32_t>(H), 1 };
        vkCmdCopyBufferToImage(cb, stage, terrain_shadow_image_,
                                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                1, &region);
        vkinit::transition_image(cb, terrain_shadow_image_,
                                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    });

    vmaDestroyBuffer(allocator_, stage, stage_alloc);

    // Write the texture into descriptor binding 6 only on first
    // creation; the image handle doesn't change after that, so partial
    // tile uploads don't need to re-write the descriptor.
    if (first_create) {
        VkDescriptorImageInfo dii{};
        dii.sampler = terrain_shadow_sampler_;
        dii.imageView = terrain_shadow_view_;
        dii.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        VkWriteDescriptorSet w{};
        w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet = scene_desc_set_;
        w.dstBinding = 6;
        w.descriptorCount = 1;
        w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        w.pImageInfo = &dii;
        vkUpdateDescriptorSets(device_, 1, &w, 0, nullptr);
    }
}

// ---------------- Heightmap texture (binding 8) ----------------
//
// Uploads `terrain_data_.heights` as an R32_SFLOAT 2D image so
// terrain_raymarch.frag can sample the gameplay heightmap and produce
// a procedural-look surface that matches its shape exactly. One-shot
// upload at init; sculpt edits don't propagate (call this again to
// refresh if/when needed).

void VulkanEngine::start_terrain_shadow_worker() {
    if (terrain_data_.heights.empty()) return;
    if (terrain_shadow_worker_.joinable()) return;     // already running
    // Snapshot the heightmap into a Heightmap struct the worker can read
    // without locks. Future sculpt edits won't propagate; if/when we
    // care, call stop_terrain_shadow_worker / start_terrain_shadow_worker
    // again to rebuild the snapshot.
    terrain_shadow_baker_hm_.dim       = terrain_data_.dim;
    terrain_shadow_baker_hm_.cell      = terrain_data_.cell;
    terrain_shadow_baker_hm_.origin_x  = terrain_data_.origin_x;
    terrain_shadow_baker_hm_.origin_z  = terrain_data_.origin_z;
    terrain_shadow_baker_hm_.heights   = terrain_data_.heights;
    terrain_shadow_stop_.store(false);
    terrain_shadow_worker_ = std::thread([this]() { terrain_shadow_worker_loop(); });
    const unsigned hc = (std::max)(2u, std::thread::hardware_concurrency());
    for (unsigned i = 0; i + 2 < hc / 2; ++i)
        terrain_shadow_helpers_.emplace_back([this]() { terrain_shadow_worker_loop(); });
}

void VulkanEngine::stop_terrain_shadow_worker() {
    if (!terrain_shadow_worker_.joinable()) return;
    {
        std::lock_guard<std::mutex> lk(terrain_shadow_mutex_);
        terrain_shadow_stop_.store(true);
        terrain_shadow_jobs_.clear();
    }
    terrain_shadow_cv_.notify_all();
    terrain_shadow_worker_.join();
    for (auto& t : terrain_shadow_helpers_) if (t.joinable()) t.join();
    terrain_shadow_helpers_.clear();
}

void VulkanEngine::terrain_shadow_worker_loop() {
    // Background bake loop. Pops one job at a time, runs the tile bake
    // on the worker thread (zero impact on main FPS), and pushes the
    // result back. Main thread drains results in tick_terrain_shadow_progressive.
    while (true) {
        ShadowBakeJob job;
        {
            std::unique_lock<std::mutex> lk(terrain_shadow_mutex_);
            terrain_shadow_cv_.wait(lk, [&]{
                return !terrain_shadow_jobs_.empty() ||
                       terrain_shadow_stop_.load();
            });
            if (terrain_shadow_stop_.load()) return;
            job = terrain_shadow_jobs_.front();
            terrain_shadow_jobs_.pop_front();
        }
        // Bake outside the lock so other threads keep running.
        ShadowBakeResult r;
        r.ix = job.ix; r.iz = job.iz; r.w = job.w; r.h = job.h; r.ss = job.ss;
        r.sun_dir = job.sun_dir;
        r.data.resize(static_cast<size_t>(job.w) * static_cast<size_t>(job.h) * 2);
        if (job.ss <= 1) {
            bake_heightmap_shadow_tile(terrain_shadow_baker_hm_, job.sun_dir,
                                        job.ix, job.iz, job.w, job.h, r.data.data());
        } else {
            bake_heightmap_shadow_tile_ss(terrain_shadow_baker_hm_, job.sun_dir,
                                           job.ix, job.iz, job.w, job.h, job.ss,
                                           r.data.data());
        }
        {
            std::lock_guard<std::mutex> lk(terrain_shadow_mutex_);
            terrain_shadow_results_.push_back(std::move(r));
        }
    }
}

void VulkanEngine::enqueue_terrain_shadow_rebake(glm::vec3 target_sun) {
    if (terrain_data_.heights.empty()) return;
    terrain_shadow_target_sun_dir_ = target_sun;

    // Tile coordinates are in TEXTURE space (heightmap_dim+1 × ss).
    const int ss = std::max(1, terrain_shadow_active_ss_);
    const int W = (terrain_data_.dim + 1) * ss;
    const int H = (terrain_data_.dim + 1) * ss;
    const int ts = kShadowTileSize;
    const glm::vec3 cam = player_.eye_position();

    // Build the full tile list, sorted nearest-first so the worker
    // bakes near-camera tiles before far ones.
    std::vector<ShadowBakeTile> tiles;
    int n_tiles = ((W + ts - 1) / ts) * ((H + ts - 1) / ts);
    tiles.reserve(static_cast<size_t>(n_tiles));
    const float sub_cell = terrain_data_.cell / static_cast<float>(ss);
    for (int iz = 0; iz < H; iz += ts) {
        int th = std::min(ts, H - iz);
        for (int ix = 0; ix < W; ix += ts) {
            int tw = std::min(ts, W - ix);
            float wx = terrain_data_.origin_x +
                       (static_cast<float>(ix) + 0.5f * static_cast<float>(tw)) *
                       sub_cell;
            float wz = terrain_data_.origin_z +
                       (static_cast<float>(iz) + 0.5f * static_cast<float>(th)) *
                       sub_cell;
            float dx = wx - cam.x, dz = wz - cam.z;
            tiles.push_back({ix, iz, tw, th, dx * dx + dz * dz});
        }
    }
    std::sort(tiles.begin(), tiles.end(),
              [](const ShadowBakeTile& a, const ShadowBakeTile& b) {
                  return a.dist_sq < b.dist_sq;
              });

    // Replace the worker's job queue. Drop any older results too — they
    // were baked against a stale sun direction or a different ss.
    {
        std::lock_guard<std::mutex> lk(terrain_shadow_mutex_);
        terrain_shadow_jobs_.clear();
        terrain_shadow_results_.clear();
        for (const auto& t : tiles) {
            terrain_shadow_jobs_.push_back({t.ix, t.iz, t.w, t.h, ss, target_sun});
        }
    }
    terrain_shadow_cv_.notify_all();
}

void VulkanEngine::tick_terrain_shadow_progressive() {
    if (terrain_shadow_image_ == VK_NULL_HANDLE) return;
    // Async path: drain a few finished bake results from the worker
    // thread and upload them to the GPU image. Bake CPU work happens
    // off-thread so there's no impact on FPS even when many tiles are
    // baking. We early-return if no results are ready.
    std::vector<ShadowBakeResult> ready;
    bool jobs_remaining = false;
    {
        std::lock_guard<std::mutex> lk(terrain_shadow_mutex_);
        const int max_take = kShadowUploadsPerFrame;
        while (ready.size() < static_cast<size_t>(max_take) &&
               !terrain_shadow_results_.empty()) {
            auto& r = terrain_shadow_results_.front();
            // Drop stale results (sun has changed since this tile was
            // queued, or supersample doesn't match the live texture).
            if (glm::distance(r.sun_dir, terrain_shadow_target_sun_dir_) < 0.001f &&
                r.ss == terrain_shadow_active_ss_) {
                ready.push_back(std::move(r));
            }
            terrain_shadow_results_.pop_front();
        }
        jobs_remaining = !terrain_shadow_jobs_.empty() ||
                         !terrain_shadow_results_.empty();
    }
    if (ready.empty()) {
        if (!jobs_remaining) {
            terrain_shadow_sun_dir_ = terrain_shadow_target_sun_dir_;
        }
        return;
    }
    // Pack tiles into one staging buffer, upload via a single submit.
    size_t total_bytes = 0;
    for (const auto& r : ready) {
        total_bytes += static_cast<size_t>(r.w) * static_cast<size_t>(r.h) * 2;
    }
    VkBuffer stage = VK_NULL_HANDLE;
    VmaAllocation stage_alloc = nullptr;
    VkBufferCreateInfo bci{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = nullptr, .flags = 0, .size = total_bytes,
        .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0, .pQueueFamilyIndices = nullptr,
    };
    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
    aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                VMA_ALLOCATION_CREATE_MAPPED_BIT;
    vmaCreateBuffer(allocator_, &bci, &aci, &stage, &stage_alloc, nullptr);
    VmaAllocationInfo ai{};
    vmaGetAllocationInfo(allocator_, stage_alloc, &ai);
    uint8_t* mapped = static_cast<uint8_t*>(ai.pMappedData);

    std::vector<VkBufferImageCopy> regions;
    regions.reserve(ready.size());
    size_t off = 0;
    for (const auto& r : ready) {
        size_t bytes = static_cast<size_t>(r.w) * static_cast<size_t>(r.h) * 2;
        std::memcpy(mapped + off, r.data.data(), bytes);
        VkBufferImageCopy reg{};
        reg.bufferOffset = off;
        reg.bufferRowLength = 0;
        reg.bufferImageHeight = 0;
        reg.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        reg.imageSubresource.mipLevel = 0;
        reg.imageSubresource.baseArrayLayer = 0;
        reg.imageSubresource.layerCount = 1;
        reg.imageOffset = { r.ix, r.iz, 0 };
        reg.imageExtent = { static_cast<uint32_t>(r.w),
                            static_cast<uint32_t>(r.h), 1 };
        regions.push_back(reg);
        off += bytes;
    }
    vkinit::one_time_submit(device_, graphics_queue_, graphics_queue_family_,
                            [&](VkCommandBuffer cb) {
        vkinit::transition_image(cb, terrain_shadow_image_,
                                  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        vkCmdCopyBufferToImage(cb, stage, terrain_shadow_image_,
                                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                static_cast<uint32_t>(regions.size()),
                                regions.data());
        vkinit::transition_image(cb, terrain_shadow_image_,
                                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    });
    vmaDestroyBuffer(allocator_, stage, stage_alloc);
    if (!jobs_remaining) {
        terrain_shadow_sun_dir_ = terrain_shadow_target_sun_dir_;
    }
}


float VulkanEngine::sample_terrain_height(float x, float z) const {
    if (terrain_data_.heights.empty() || terrain_data_.dim <= 0) return 0.0f;
    const int W = terrain_data_.dim + 1;
    float fx = (x - terrain_data_.origin_x) / terrain_data_.cell;
    float fz = (z - terrain_data_.origin_z) / terrain_data_.cell;
    // Outside the baked home tile: the wider world's ground, from the same formula the
    // raymarcher draws (it was 0 out there - nothing to stand on).
    const auto outside = [&]() { return world_ground_height(x, z, rt_.water_level, std::clamp(rt_.terrain_raymarch_octaves, 4, 24)); };
    if (fx < 0.0f || fz < 0.0f) return outside();
    int ix = static_cast<int>(std::floor(fx));
    int iz = static_cast<int>(std::floor(fz));
    if (ix >= terrain_data_.dim || iz >= terrain_data_.dim) return outside();
    float tx = fx - static_cast<float>(ix);
    float tz = fz - static_cast<float>(iz);
    auto h = [&](int x_, int z_) {
        return terrain_data_.heights[static_cast<size_t>(z_) * static_cast<size_t>(W) +
                                     static_cast<size_t>(x_)];
    };
    float h00 = h(ix,     iz);
    float h10 = h(ix + 1, iz);
    float h01 = h(ix,     iz + 1);
    float h11 = h(ix + 1, iz + 1);
    float h0 = h00 * (1.0f - tx) + h10 * tx;
    float h1 = h01 * (1.0f - tx) + h11 * tx;
    // (a region's height tile may lie over home's: it adds to it - engine/region_tile.h)
    return h0 * (1.0f - tz) + h1 * tz + region::ground_delta(x, z);
}


// ---- Procedural terrain textures (curvature/sky-vis aux, grass mask, fog) ----

// Terrain aux bake (AAA Phases 4/5). RG8 1024² covering the 2048 m
// world: R = curvature (0.5 flat, <0.5 concave gullies — sediment +
// cavity darkening, >0.5 convex ridges — exposed rock), G = sky
// visibility from an 8-direction horizon scan (macro AO for valleys).
// CPU bake, multi-threaded; ~100 ms at load.

} // namespace vibe
