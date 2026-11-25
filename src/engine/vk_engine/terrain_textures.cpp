// terrain_textures.cpp - terrain GPU textures: height maps, CLOD instances, aux (curvature / sky-vis), grass mask, fog wisps
// (split out of terrain_resources.cpp; see docs/ARCHITECTURE.md for the map.)
// Terrain CPU/GPU resources: heightmap sculpting (FBM noise/erosion brush),
// grass-eligibility mask, height textures, quadtree CLOD instancing, the
// async sun-shadow bake worker, and the curvature/sky-vis aux + grass-mask
// + fog-wisp procedural textures. Split out of world.cpp (maintainability).

#include "engine/vk_engine/internal.h"
#include "engine/highway.h"
#include "engine/vk_engine/state/fog.h"
#include "engine/vk_engine/state/sun_shadow.h"
#include "engine/gltf_loader.h"
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

void VulkanEngine::init_terrain_height_texture() {
    if (terrain_data_.heights.empty()) {
        log::warn("init_terrain_height_texture: heightmap empty, skipping");
        return;
    }
    // Heightmap is stored as (dim+1) × (dim+1) — vertices, not cells —
    // so the texture must match that or the row-stride doesn't line up
    // and the upload scrambles the data (cells get shifted by 1 column
    // per row). Texture is dim+1 in each axis to match.
    const int W = terrain_data_.dim + 1;
    const int H = terrain_data_.dim + 1;

    // Cache the max for the raymarch's upper-bound clip. Compute once
    // here — sculpt edits don't update the texture anyway.
    float max_h = 0.0f;
    for (float h : terrain_data_.heights) if (h > max_h) max_h = h;
    terrain_height_max_ = max_h;

    VkImageCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = VK_FORMAT_R32_SFLOAT;
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
    vk_check(vmaCreateImage(allocator_, &ici, &aci,
                             &terrain_height_image_,
                             &terrain_height_alloc_, nullptr),
             "terrain height image");

    VkImageViewCreateInfo vci{};
    vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vci.image = terrain_height_image_;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = VK_FORMAT_R32_SFLOAT;
    vci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    vci.subresourceRange.baseMipLevel = 0;
    vci.subresourceRange.levelCount = 1;
    vci.subresourceRange.baseArrayLayer = 0;
    vci.subresourceRange.layerCount = 1;
    vk_check(vkCreateImageView(device_, &vci, nullptr, &terrain_height_view_),
             "terrain height view");

    VkSamplerCreateInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    si.magFilter = VK_FILTER_LINEAR;     // bilinear height interpolation
    si.minFilter = VK_FILTER_LINEAR;
    si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    si.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    vk_check(vkCreateSampler(device_, &si, nullptr, &terrain_height_sampler_),
             "terrain height sampler");

    upload_terrain_height_payload(VK_IMAGE_LAYOUT_UNDEFINED);

    // Write binding 8.
    VkDescriptorImageInfo dii{};
    dii.sampler = terrain_height_sampler_;
    dii.imageView = terrain_height_view_;
    dii.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkWriteDescriptorSet w{};
    w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w.dstSet = scene_desc_set_;
    w.dstBinding = 8;
    w.descriptorCount = 1;
    w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    w.pImageInfo = &dii;
    vkUpdateDescriptorSets(device_, 1, &w, 0, nullptr);

    log::infof("heightmap texture uploaded (%dx%d, %.1f MB)", W, H,
               static_cast<double>(W) * H * sizeof(float) / (1024.0 * 1024.0));
}

void VulkanEngine::refresh_terrain_height_texture() {
    if (!terrain_height_image_ || terrain_data_.heights.empty()) return;
    upload_terrain_height_payload(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    terrain_height_dirty_ = false;
}

// Shared upload helper for both init_ and refresh_terrain_height_texture.
// Allocates a host-mapped staging buffer, writes the delta-from-baseline
// (or raw heights if no baseline), runs a one-shot transition+copy+
// transition, frees the staging.
void VulkanEngine::upload_terrain_height_payload(VkImageLayout src_layout) {
    const int W = terrain_data_.dim + 1;
    const int H = terrain_data_.dim + 1;
    const VkDeviceSize bytes = static_cast<VkDeviceSize>(W) *
                                static_cast<VkDeviceSize>(H) * sizeof(float);

    VkBuffer stage = VK_NULL_HANDLE;
    VmaAllocation stage_alloc = nullptr;
    VkBufferCreateInfo bci{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = nullptr, .flags = 0, .size = bytes,
        .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0, .pQueueFamilyIndices = nullptr,
    };
    VmaAllocationCreateInfo bac{};
    bac.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
    bac.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                VMA_ALLOCATION_CREATE_MAPPED_BIT;
    vmaCreateBuffer(allocator_, &bci, &bac, &stage, &stage_alloc, nullptr);
    VmaAllocationInfo ai{};
    vmaGetAllocationInfo(allocator_, stage_alloc, &ai);
    // DELTA from baseline so the raymarch shader can layer sculpt edits on
    // top of its own GLSL FBM. On first launch baseline == current →
    // delta is all zeros → no change to the procedural look.
    const size_t n = static_cast<size_t>(W) * static_cast<size_t>(H);
    if (terrain_baseline_heights_.size() == n) {
        std::vector<float> delta(n);
        for (size_t i = 0; i < n; ++i) {
            delta[i] = terrain_data_.heights[i] - terrain_baseline_heights_[i];
        }
        // VIBE_DUMP_DELTA=<x>,<z>: the land's height above its baseline round that place, 3 m apart (cm), to the log.
        if (const char* dd = std::getenv("VIBE_DUMP_DELTA")) {
            float gx = 0.0f, gz = 0.0f;
            std::sscanf(dd, "%f,%f", &gx, &gz);
            const float cell = 2048.0f / float(W - 1);
            for (int j = -6; j <= 6; ++j) {
                std::string row;
                for (int i = -6; i <= 6; ++i) {
                    const int mx = std::clamp(int(std::lround((gx + float(i) * 1.0f + 0.5f * 2048.0f) / cell)), 0, int(W) - 1), mz = std::clamp(int(std::lround((gz + float(j) * 1.0f + 0.5f * 2048.0f) / cell)), 0, int(H) - 1);
                    char b[16]; std::snprintf(b, sizeof(b), "%6.0f", double(terrain_data_.heights[size_t(mz) * W + mx]) * 10.0);
                    row += b;
                }
                log::infof("[delta] z %+3d m:%s", j * 3, row.c_str());
            }
        }
        std::memcpy(ai.pMappedData, delta.data(), static_cast<size_t>(bytes));
    } else {
        std::memcpy(ai.pMappedData, terrain_data_.heights.data(),
                    static_cast<size_t>(bytes));
    }

    vkinit::one_time_submit(device_, graphics_queue_, graphics_queue_family_,
                            [&](VkCommandBuffer cb) {
        vkinit::transition_image(cb, terrain_height_image_,
                                  src_layout,
                                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        VkBufferImageCopy region{};
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.layerCount = 1;
        region.imageExtent = { static_cast<uint32_t>(W),
                                static_cast<uint32_t>(H), 1 };
        vkCmdCopyBufferToImage(cb, stage, terrain_height_image_,
                                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                1, &region);
        vkinit::transition_image(cb, terrain_height_image_,
                                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    });
    vmaDestroyBuffer(allocator_, stage, stage_alloc);
}


void VulkanEngine::destroy_terrain_height_texture() {
    if (terrain_height_view_) {
        vkDestroyImageView(device_, terrain_height_view_, nullptr);
        terrain_height_view_ = VK_NULL_HANDLE;
    }
    if (terrain_height_image_) {
        vmaDestroyImage(allocator_, terrain_height_image_,
                         terrain_height_alloc_);
        terrain_height_image_ = VK_NULL_HANDLE;
        terrain_height_alloc_ = nullptr;
    }
    if (terrain_height_sampler_) {
        vkDestroySampler(device_, terrain_height_sampler_, nullptr);
        terrain_height_sampler_ = VK_NULL_HANDLE;
    }
    if (terrain_height_full_view_) {
        vkDestroyImageView(device_, terrain_height_full_view_, nullptr);
        terrain_height_full_view_ = VK_NULL_HANDLE;
    }
    if (terrain_height_full_image_) {
        vmaDestroyImage(allocator_, terrain_height_full_image_,
                         terrain_height_full_alloc_);
        terrain_height_full_image_ = VK_NULL_HANDLE;
        terrain_height_full_alloc_ = nullptr;
    }
    if (terrain_height_full_sampler_) {
        vkDestroySampler(device_, terrain_height_full_sampler_, nullptr);
        terrain_height_full_sampler_ = VK_NULL_HANDLE;
    }
}

// Stages the FULL baked terrain heights (NOT the baseline delta) into
// terrain_height_full_image_. This is the exact terrain_data_.heights
// array the CDLOD mesh + physics are built from, so the water path can
// reference the real mesh surface without evaluating FBM noise.
void VulkanEngine::upload_terrain_height_full_payload(VkImageLayout src_layout) {
    const int W = terrain_data_.dim + 1;
    const int H = terrain_data_.dim + 1;
    const VkDeviceSize bytes = static_cast<VkDeviceSize>(W) *
                                static_cast<VkDeviceSize>(H) * sizeof(float);
    VkBuffer stage = VK_NULL_HANDLE;
    VmaAllocation stage_alloc = nullptr;
    VkBufferCreateInfo bci{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = nullptr, .flags = 0, .size = bytes,
        .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0, .pQueueFamilyIndices = nullptr,
    };
    VmaAllocationCreateInfo bac{};
    bac.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
    bac.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                VMA_ALLOCATION_CREATE_MAPPED_BIT;
    vmaCreateBuffer(allocator_, &bci, &bac, &stage, &stage_alloc, nullptr);
    VmaAllocationInfo ai{};
    vmaGetAllocationInfo(allocator_, stage_alloc, &ai);
    std::memcpy(ai.pMappedData, terrain_data_.heights.data(),
                static_cast<size_t>(bytes));
    vkinit::one_time_submit(device_, graphics_queue_, graphics_queue_family_,
                            [&](VkCommandBuffer cb) {
        vkinit::transition_image(cb, terrain_height_full_image_,
                                 src_layout,
                                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        VkBufferImageCopy region{};
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.layerCount = 1;
        region.imageExtent = { static_cast<uint32_t>(W),
                               static_cast<uint32_t>(H), 1 };
        vkCmdCopyBufferToImage(cb, stage, terrain_height_full_image_,
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                               1, &region);
        vkinit::transition_image(cb, terrain_height_full_image_,
                                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                 VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    });
    vmaDestroyBuffer(allocator_, stage, stage_alloc);
}

// ---- Quadtree CDLOD terrain (engine side) -------------------------------

void VulkanEngine::init_terrain_clod() {
    if (terrain_data_.heights.empty()) {
        log::warn("init_terrain_clod: heightmap empty, skipping");
        return;
    }
    terrain_clod_patch_ = build_clod_patch_mesh(device_, allocator_,
                                                graphics_queue_,
                                                graphics_queue_family_);
    terrain_clod_pyramid_ = build_clod_pyramid(terrain_data_.heights.data(),
                                               terrain_data_.dim,
                                               terrain_data_.cell, 16.0f);

    // Host-visible per-frame instance ring: kFrameOverlap slots × two
    // lists (camera, shadow) × kClodMaxInstances.
    const VkDeviceSize bytes = VkDeviceSize(kFrameOverlap) * 2 *
                               kClodMaxInstances * sizeof(ClodInstance);
    VkBufferCreateInfo bci{};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = bytes;
    bci.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
    aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                VMA_ALLOCATION_CREATE_MAPPED_BIT;
    VmaAllocationInfo ai{};
    vk_check(vmaCreateBuffer(allocator_, &bci, &aci,
                             &terrain_clod_inst_buf_,
                             &terrain_clod_inst_alloc_, &ai),
             "terrain clod instance buffer");
    terrain_clod_inst_map_ = ai.pMappedData;
    log::infof("[terrain] CLOD ready: patch %u verts / %u idx, pyramid %d "
               "levels, instance ring %.0f KB",
               terrain_clod_patch_.vertex_count,
               terrain_clod_patch_.index_count,
               terrain_clod_pyramid_.levels,
               double(bytes) / 1024.0);
}

void VulkanEngine::destroy_terrain_clod() {
    destroy_mesh(allocator_, terrain_clod_patch_);
    if (terrain_clod_inst_buf_) {
        vmaDestroyBuffer(allocator_, terrain_clod_inst_buf_,
                         terrain_clod_inst_alloc_);
        terrain_clod_inst_buf_ = VK_NULL_HANDLE;
        terrain_clod_inst_alloc_ = nullptr;
        terrain_clod_inst_map_ = nullptr;
    }
}

glm::vec4 VulkanEngine::terrain_clod_hm_params() const {
    const float cell = (terrain_data_.cell > 0.0f) ? terrain_data_.cell : 1.0f;
    return glm::vec4(terrain_data_.origin_x, terrain_data_.origin_z,
                     1.0f / cell, 1.0f / float(terrain_data_.dim + 1));
}

void VulkanEngine::update_terrain_clod_instances() {
    terrain_clod_cam_count_ = terrain_clod_shadow_count_ = 0;
    if (!terrain_clod_inst_map_ || terrain_data_.heights.empty() ||
        rt_.terrain_raymarch_enabled) {
        return;
    }
    ClodParams p;
    p.world_origin_x = terrain_data_.origin_x;
    p.world_origin_z = terrain_data_.origin_z;
    p.world_side     = float(terrain_data_.dim) * terrain_data_.cell;
    p.leaf_size_m    = 16.0f;
    p.levels = 1;
    while (p.leaf_size_m * float(1 << (p.levels - 1)) < p.world_side - 0.5f)
        ++p.levels;
    p.range0 = std::max(10.0f, rt_.terrain_clod_range0);

    select_clod_nodes(p, terrain_clod_pyramid_, current_frame_view_.eye_pos,
                      terrain_clod_nodes_);

    const uint32_t slot = uint32_t(frame_number_ % kFrameOverlap);
    const uint32_t list_bytes = kClodMaxInstances * uint32_t(sizeof(ClodInstance));
    terrain_clod_cam_offset_    = slot * 2 * list_bytes;
    terrain_clod_shadow_offset_ = terrain_clod_cam_offset_ + list_bytes;
    auto* cam_dst = reinterpret_cast<ClodInstance*>(
        static_cast<char*>(terrain_clod_inst_map_) + terrain_clod_cam_offset_);
    auto* sh_dst = reinterpret_cast<ClodInstance*>(
        static_cast<char*>(terrain_clod_inst_map_) + terrain_clod_shadow_offset_);

    const Frustum& camf = current_frame_view_.frustum;
    const Frustum lightf = extract_frustum(sun_->shadow_light_vp);
    for (const ClodNode& n : terrain_clod_nodes_) {
        if (terrain_clod_cam_count_ < kClodMaxInstances &&
            aabb_visible(camf, n.aabb_min, n.aabb_max)) {
            cam_dst[terrain_clod_cam_count_++] = n.inst;
        }
        if (terrain_clod_shadow_count_ < kClodMaxInstances &&
            aabb_visible(lightf, n.aabb_min, n.aabb_max)) {
            sh_dst[terrain_clod_shadow_count_++] = n.inst;
        }
    }
}

void VulkanEngine::init_terrain_height_full_texture() {
    if (terrain_data_.heights.empty()) {
        log::warn("init_terrain_height_full_texture: heightmap empty, skipping");
        return;
    }
    const int W = terrain_data_.dim + 1;
    const int H = terrain_data_.dim + 1;

    VkImageCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = VK_FORMAT_R32_SFLOAT;
    ici.extent = { static_cast<uint32_t>(W), static_cast<uint32_t>(H), 1 };
    ici.mipLevels = 1; ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    vk_check(vmaCreateImage(allocator_, &ici, &aci,
                             &terrain_height_full_image_,
                             &terrain_height_full_alloc_, nullptr),
             "terrain height full image");

    VkImageViewCreateInfo vci{};
    vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vci.image = terrain_height_full_image_;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = VK_FORMAT_R32_SFLOAT;
    vci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    vci.subresourceRange.baseMipLevel = 0;
    vci.subresourceRange.levelCount = 1;
    vci.subresourceRange.baseArrayLayer = 0;
    vci.subresourceRange.layerCount = 1;
    vk_check(vkCreateImageView(device_, &vci, nullptr,
                               &terrain_height_full_view_),
             "terrain height full view");

    VkSamplerCreateInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    si.magFilter = VK_FILTER_LINEAR;
    si.minFilter = VK_FILTER_LINEAR;
    si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    si.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    vk_check(vkCreateSampler(device_, &si, nullptr,
                             &terrain_height_full_sampler_),
             "terrain height full sampler");

    upload_terrain_height_full_payload(VK_IMAGE_LAYOUT_UNDEFINED);

    // Write binding 17.
    VkDescriptorImageInfo dii{};
    dii.sampler = terrain_height_full_sampler_;
    dii.imageView = terrain_height_full_view_;
    dii.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkWriteDescriptorSet w{};
    w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w.dstSet = scene_desc_set_;
    w.dstBinding = 17;
    w.descriptorCount = 1;
    w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    w.pImageInfo = &dii;
    vkUpdateDescriptorSets(device_, 1, &w, 0, nullptr);

    log::infof("full heightmap texture uploaded (%dx%d, %.1f MB)", W, H,
               static_cast<double>(W) * H * sizeof(float) / (1024.0 * 1024.0));
}

void VulkanEngine::refresh_terrain_height_full_texture() {
    if (!terrain_height_full_image_ || terrain_data_.heights.empty()) return;
    upload_terrain_height_full_payload(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
}

// ---------------- Sun shadow map (single-cascade ortho) ----------------
//
// Standard depth-only shadow pass: one 2048² D32 depth target written
// from the sun's POV each frame, sampled by grass.vert as a
// sampler2DShadow at descriptor binding 7. Cube.frag continues to use
// RT shadow rays — we don't move the whole engine to CSM yet, this is
// the cheap path that lets dynamic occluders (castle, dyn-props) cast
// shadow on grass without firing a ray per blade.
//
// init_sun_shadow_resources / destroy_sun_shadow_resources /
// update_sun_shadow_light_vp / render_sun_shadow_pass moved to
// vk_engine/sun_shadow.cpp.


void VulkanEngine::init_terrain_aux_texture() {
    if (terrain_data_.heights.empty()) return;
    const int dim = 1024;
    const float kWorldSide = 2048.0f;
    std::vector<uint8_t> data(size_t(dim) * dim * 2, 0);

    auto h_at = [&](float wx, float wz) {
        return sample_terrain_height(wx, wz);
    };
    const float curv_r = 5.0f;     // curvature stencil radius (m)
    const glm::vec2 dirs[8] = {
        { 1, 0 }, { 0.7071f, 0.7071f }, { 0, 1 }, { -0.7071f, 0.7071f },
        { -1, 0 }, { -0.7071f, -0.7071f }, { 0, -1 }, { 0.7071f, -0.7071f },
    };
    const float steps[6] = { 3.0f, 6.0f, 11.0f, 20.0f, 38.0f, 70.0f };

    unsigned n_threads = (std::max)(1u, std::thread::hardware_concurrency());
    std::vector<std::thread> workers;
    int rows_per = (dim + int(n_threads) - 1) / int(n_threads);
    for (unsigned t = 0; t < n_threads; ++t) {
        int z0 = int(t) * rows_per;
        int z1 = (std::min)(dim, z0 + rows_per);
        if (z0 >= z1) break;
        workers.emplace_back([&, z0, z1]() {
            for (int tz = z0; tz < z1; ++tz) {
                float wz = (float(tz) / dim - 0.5f) * kWorldSide;
                for (int tx = 0; tx < dim; ++tx) {
                    float wx = (float(tx) / dim - 0.5f) * kWorldSide;
                    float h0 = h_at(wx, wz);
                    // Curvature: 4-tap Laplacian at curv_r.
                    float lap = h_at(wx + curv_r, wz) + h_at(wx - curv_r, wz)
                              + h_at(wx, wz + curv_r) + h_at(wx, wz - curv_r)
                              - 4.0f * h0;
                    // lap > 0 = concave (valley floor). Map ±~6 m of
                    // Laplacian to the full byte range around 0.5.
                    float curv = std::clamp(0.5f - lap * 0.08f, 0.0f, 1.0f);
                    // Sky visibility: per direction take the max horizon
                    // slope over the step ladder; vis = 1 at open sky.
                    float vis_sum = 0.0f;
                    for (const glm::vec2& d : dirs) {
                        float max_slope = 0.0f;
                        for (float s : steps) {
                            float dh = h_at(wx + d.x * s, wz + d.y * s) - h0;
                            if (dh > 0.0f) {
                                max_slope = (std::max)(max_slope, dh / s);
                            }
                        }
                        float ang = std::atan(max_slope);   // 0..~pi/2
                        vis_sum += 1.0f - ang * (2.0f / 3.14159265f);
                    }
                    float vis = vis_sum / 8.0f;
                    size_t o = (size_t(tz) * dim + tx) * 2;
                    data[o]     = uint8_t(curv * 255.0f + 0.5f);
                    data[o + 1] = uint8_t(std::clamp(vis, 0.0f, 1.0f) * 255.0f + 0.5f);
                }
            }
        });
    }
    for (auto& th : workers) th.join();

    // Upload as RG8 + write binding 26.
    VkImageCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = VK_FORMAT_R8G8_UNORM;
    ici.extent = { uint32_t(dim), uint32_t(dim), 1 };
    ici.mipLevels = 1; ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    vk_check(vmaCreateImage(allocator_, &ici, &aci, &terrain_aux_image_,
                            &terrain_aux_alloc_, nullptr), "terrain aux image");
    VkImageViewCreateInfo vci{};
    vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vci.image = terrain_aux_image_;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = VK_FORMAT_R8G8_UNORM;
    vci.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    vk_check(vkCreateImageView(device_, &vci, nullptr, &terrain_aux_view_),
             "terrain aux view");
    VkSamplerCreateInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    si.magFilter = VK_FILTER_LINEAR;
    si.minFilter = VK_FILTER_LINEAR;
    si.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    vk_check(vkCreateSampler(device_, &si, nullptr, &terrain_aux_sampler_),
             "terrain aux sampler");

    // Staging upload.
    VkBuffer stage = VK_NULL_HANDLE; VmaAllocation stage_alloc = nullptr;
    VkBufferCreateInfo bci{};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = data.size();
    bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VmaAllocationCreateInfo bac{};
    bac.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
    bac.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                VMA_ALLOCATION_CREATE_MAPPED_BIT;
    VmaAllocationInfo ai{};
    vmaCreateBuffer(allocator_, &bci, &bac, &stage, &stage_alloc, &ai);
    std::memcpy(ai.pMappedData, data.data(), data.size());
    vkinit::one_time_submit(device_, graphics_queue_, graphics_queue_family_,
                            [&](VkCommandBuffer cb) {
        vkinit::transition_image(cb, terrain_aux_image_,
                                 VK_IMAGE_LAYOUT_UNDEFINED,
                                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        VkBufferImageCopy region{};
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.layerCount = 1;
        region.imageExtent = { uint32_t(dim), uint32_t(dim), 1 };
        vkCmdCopyBufferToImage(cb, stage, terrain_aux_image_,
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
        vkinit::transition_image(cb, terrain_aux_image_,
                                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                 VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    });
    vmaDestroyBuffer(allocator_, stage, stage_alloc);

    // Write binding 26 (runs after descriptors are created).
    VkDescriptorImageInfo dii{};
    dii.sampler = terrain_aux_sampler_;
    dii.imageView = terrain_aux_view_;
    dii.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkWriteDescriptorSet w{};
    w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w.dstSet = scene_desc_set_;
    w.dstBinding = 26;
    w.descriptorCount = 1;
    w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    w.pImageInfo = &dii;
    vkUpdateDescriptorSets(device_, 1, &w, 0, nullptr);
    log::infof("[terrain] aux bake (curvature + sky-vis) %dx%d uploaded", dim, dim);
}

void VulkanEngine::destroy_terrain_aux_texture() {
    deferred_desc_sig_ = 0;   // handles may be recreated (even at the same address)
    if (terrain_aux_view_)    vkDestroyImageView(device_, terrain_aux_view_, nullptr);
    if (terrain_aux_image_)   vmaDestroyImage(allocator_, terrain_aux_image_, terrain_aux_alloc_);
    if (terrain_aux_sampler_) vkDestroySampler(device_, terrain_aux_sampler_, nullptr);
    terrain_aux_view_ = VK_NULL_HANDLE;
    terrain_aux_image_ = VK_NULL_HANDLE;
    terrain_aux_alloc_ = nullptr;
    terrain_aux_sampler_ = VK_NULL_HANDLE;
}

void VulkanEngine::init_grass_mask_texture() {
    const int dim = kGrassMaskDim;
    const float kWorldSide = 2048.0f;     // matches kHeightmapSide
    // RG8 — 2 bytes per texel. The vector lives on the engine so the
    // GrassAdd/GrassRemove brush can paint into it without rebaking the
    // noise. We populate it here, upload from it once, and reupload via
    // reupload_grass_mask() after every paint stroke.
    grass_mask_data_.assign(static_cast<size_t>(dim) * dim * 2, 0);
    std::vector<uint8_t>& data = grass_mask_data_;

    auto smoothstep_f = [](float a, float b, float v) {
        float t = std::clamp((v - a) / (b - a), 0.0f, 1.0f);
        return t * t * (3.0f - 2.0f * t);
    };
    for (int z = 0; z < dim; ++z) {
        for (int x = 0; x < dim; ++x) {
            float wx = (static_cast<float>(x) + 0.5f) / dim * kWorldSide
                       - kWorldSide * 0.5f;
            float wz = (static_cast<float>(z) + 0.5f) / dim * kWorldSide
                       - kWorldSide * 0.5f;
            // R: presence — low-freq FBM, smoothstep gate.
            float n = fbm_noise2(wx * 0.02f, wz * 0.02f);
            float presence = smoothstep_f(0.40f, 0.55f, n);
            // Castle keep-out. The raymarched grass shader plants
            // blades at terrain Y (= plateau height inside castle), so
            // without this guard blade tips poke through the castle
            // floor brushes. The mask is 1024² over a 2048 m world
            // (в‰€2 m / texel) and the shader samples it with LINEAR
            // filtering — bilinear can leak presence ~2-4 m inward
            // from a hard boundary, so we use a wider keep-out band
            // (18 m vs the 11 m outer wall) to keep ALL bilinear
            // samples inside the floor's extent at exactly zero.
            const float kCastleHalf = 18.0f;
            if (std::abs(wx) < kCastleHalf && std::abs(wz) < kCastleHalf) {
                presence = 0.0f;
            }
            // G: slope mag — finite-diff on real terrain heights, so
            // cliffs/peaks naturally read as steep. Stored normalized
            // [0..1] = raw_slope / 2.0; shader un-normalizes.
            const float kEps = 2.0f;       // metres
            float h_c = sample_terrain_height(wx, wz);
            float h_r = sample_terrain_height(wx + kEps, wz);
            float h_u = sample_terrain_height(wx, wz + kEps);
            float dx = (h_r - h_c) / kEps;
            float dz = (h_u - h_c) / kEps;
            float slope_mag = std::sqrt(dx * dx + dz * dz);
            float slope_norm = std::clamp(slope_mag * 0.5f, 0.0f, 1.0f);
            const size_t idx = (static_cast<size_t>(z) * dim + x) * 2;
            data[idx + 0] =
                static_cast<uint8_t>(std::clamp(presence, 0.0f, 1.0f) * 255.0f);
            data[idx + 1] =
                static_cast<uint8_t>(slope_norm * 255.0f);
        }
    }

    // Overlay any persisted user paint on top of the freshly baked
    // presence/slope field. Mirrors the height-load path: bake the
    // procedural baseline first so a missing save file is a no-op,
    // then replace bytes if the file is present.
    // ONLY the presence is the user's (R). The slope (G) is the land's as it is NOW: the saved
    // file carries the slope of the day it was painted, and where the land has changed since
    // (the river's valley, a road's shelf) that left round bare patches in a meadow and grass
    // on faces too steep for it.
    {
        std::vector<uint8_t> slope_now(static_cast<size_t>(dim) * dim);
        for (size_t i = 0; i < slope_now.size(); ++i) slope_now[i] = data[i * 2 + 1];
        load_grass_mask();
        if (grass_mask_data_.size() == slope_now.size() * 2)
            for (size_t i = 0; i < slope_now.size(); ++i) grass_mask_data_[i * 2 + 1] = slope_now[i];
    }
    // Keep the road paving + verge clear of blades.
    clear_grass_mask(castle_road_, grass_mask_data_, dim, kWorldSide);
    // ... and of a highway's line that runs through the home tile (the castle lane).
    if (highway::ready() && !grass_mask_data_.empty()) {
        const float cell = kWorldSide / float(dim);
        for (int li = 0; li < highway::line_count(); ++li) {
            const std::vector<glm::vec3>& P = highway::line_points(li);
            glm::vec2 lo(1e9f), hi(-1e9f);
            for (const glm::vec3& p : P) if (std::abs(p.x) < 0.5f * kWorldSide + 10.0f && std::abs(p.z) < 0.5f * kWorldSide + 10.0f) { lo = glm::min(lo, glm::vec2(p.x, p.z)); hi = glm::max(hi, glm::vec2(p.x, p.z)); }
            if (hi.x < lo.x) continue;
            const int x0 = std::clamp(int((lo.x - 6.0f + 0.5f * kWorldSide) / cell), 0, dim - 1), x1 = std::clamp(int((hi.x + 6.0f + 0.5f * kWorldSide) / cell), 0, dim - 1);
            const int z0 = std::clamp(int((lo.y - 6.0f + 0.5f * kWorldSide) / cell), 0, dim - 1), z1 = std::clamp(int((hi.y + 6.0f + 0.5f * kWorldSide) / cell), 0, dim - 1);
            for (int mz = z0; mz <= z1; ++mz)
                for (int mx = x0; mx <= x1; ++mx) {
                    const float d = highway::distance((float(mx) + 0.5f) * cell - 0.5f * kWorldSide, (float(mz) + 0.5f) * cell - 0.5f * kWorldSide);
                    if (d > highway::kVerge + 1.0f) continue;
                    uint8_t& r = grass_mask_data_[(size_t(mz) * dim + mx) * 2];
                    r = uint8_t(float(r) * glm::smoothstep(highway::kVerge - 1.2f, highway::kVerge + 0.6f, d));
                }
        }
    }
    // ... nor on the castle lane's car park (terrain.h kCastlePark).
    if (!grass_mask_data_.empty()) {
        const float cell = kWorldSide / float(dim);
        for (int mz = std::max(0, int((kCastlePark[1] - 2.0f + 0.5f * kWorldSide) / cell)); mz <= std::min(dim - 1, int((kCastlePark[3] + 2.0f + 0.5f * kWorldSide) / cell)); ++mz)
            for (int mx = std::max(0, int((kCastlePark[0] - 2.0f + 0.5f * kWorldSide) / cell)); mx <= std::min(dim - 1, int((kCastlePark[2] + 4.5f + 0.5f * kWorldSide) / cell)); ++mx)
                grass_mask_data_[(size_t(mz) * dim + mx) * 2] = 0;
    }
    // VIBE_DUMP_GRASS=<x>,<z>: the grass mask's presence round that place, 3 m apart (0..255), to the log.
    if (const char* dg = std::getenv("VIBE_DUMP_GRASS"); dg && !grass_mask_data_.empty()) {
        float gx = 0.0f, gz = 0.0f;
        std::sscanf(dg, "%f,%f", &gx, &gz);
        const float cell = kWorldSide / float(dim);
        for (int j = -5; j <= 5; ++j) {
            std::string row;
            for (int i = -5; i <= 5; ++i) {
                const int mx = std::clamp(int((gx + float(i) * 3.0f + 0.5f * kWorldSide) / cell), 0, dim - 1), mz = std::clamp(int((gz + float(j) * 3.0f + 0.5f * kWorldSide) / cell), 0, dim - 1);
                char b[16]; std::snprintf(b, sizeof(b), "%4d", int(grass_mask_data_[(size_t(mz) * dim + mx) * 2 + 1]));
                row += b;
            }
            log::infof("[grass] z %+3d m:%s", j * 3, row.c_str());
        }
    }
    // No lawn under the chapel (the blades would stand up through its floor).
    if (church_placed_) {
        const float cell = kWorldSide / float(dim);
        for (int mz = 0; mz < dim; ++mz) {
            const float wz = (float(mz) + 0.5f) * cell - 0.5f * kWorldSide;
            if (std::abs(wz - church_pos_.z) > 4.2f) continue;
            for (int mx = 0; mx < dim; ++mx) {
                const float wx = (float(mx) + 0.5f) * cell - 0.5f * kWorldSide;
                if (wx < church_pos_.x - 6.6f || wx > church_pos_.x + 8.4f) continue;
                grass_mask_data_[(size_t(mz) * size_t(dim) + size_t(mx)) * 2] = 0;
            }
        }
        // The graveyard beside it is kept: short, thin turf so the stones show.
        for (int mz = 0; mz < dim; ++mz) {
            const float wz = (float(mz) + 0.5f) * cell - 0.5f * kWorldSide;
            if (wz < church_pos_.z + 4.6f || wz > church_pos_.z + 15.5f) continue;
            for (int mx = 0; mx < dim; ++mx) {
                const float wx = (float(mx) + 0.5f) * cell - 0.5f * kWorldSide;
                if (wx < church_pos_.x - 7.5f || wx > church_pos_.x + 8.5f) continue;
                uint8_t& g = grass_mask_data_[(size_t(mz) * size_t(dim) + size_t(mx)) * 2];
                g = static_cast<uint8_t>(float(g) * 0.22f);
            }
        }
    }

    // No lawn under or through the stone bridge (its ends lie at ground level: the blades
    // stood up through the deck there).
    if (bridge_placed_) {
        const float cell = kWorldSide / float(dim);
        const glm::vec2 a(bridge_a_.x, bridge_a_.z), b(bridge_b_.x, bridge_b_.z);
        const float L = glm::length(b - a);
        const glm::vec2 d = (b - a) / std::max(L, 1e-3f), n(-d.y, d.x);
        const int x0 = std::max(0, int((std::min(a.x, b.x) - 8.0f + 0.5f * kWorldSide) / cell));
        const int x1 = std::min(dim - 1, int((std::max(a.x, b.x) + 8.0f + 0.5f * kWorldSide) / cell));
        const int z0 = std::max(0, int((std::min(a.y, b.y) - 8.0f + 0.5f * kWorldSide) / cell));
        const int z1 = std::min(dim - 1, int((std::max(a.y, b.y) + 8.0f + 0.5f * kWorldSide) / cell));
        for (int mz = z0; mz <= z1; ++mz)
            for (int mx = x0; mx <= x1; ++mx) {
                const glm::vec2 p((float(mx) + 0.5f) * cell - 0.5f * kWorldSide, (float(mz) + 0.5f) * cell - 0.5f * kWorldSide);
                const float s = glm::dot(p - a, d), u = glm::dot(p - a, n);
                if (s > -4.0f && s < L + 4.0f && std::abs(u) < 5.0f)
                    grass_mask_data_[(size_t(mz) * size_t(dim) + size_t(mx)) * 2] = 0;
            }
    }

    // Upload as a sampled RG8 image.
    VkImageCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = VK_FORMAT_R8G8_UNORM;
    ici.extent = { static_cast<uint32_t>(dim),
                   static_cast<uint32_t>(dim), 1 };
    ici.mipLevels = 1; ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_SAMPLED_BIT |
                VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    vk_check(vmaCreateImage(allocator_, &ici, &aci,
                             &grass_mask_.image, &grass_mask_.alloc, nullptr),
             "grass mask image");
    VkImageViewCreateInfo vci{};
    vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vci.image = grass_mask_.image;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = VK_FORMAT_R8G8_UNORM;
    vci.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    vk_check(vkCreateImageView(device_, &vci, nullptr, &grass_mask_.view),
             "grass mask view");

    // Stage + copy + transition.
    const VkDeviceSize bytes = data.size();
    VkBuffer stage = VK_NULL_HANDLE;
    VmaAllocation stage_alloc = nullptr;
    VkBufferCreateInfo bci{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = nullptr, .flags = 0, .size = bytes,
        .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0, .pQueueFamilyIndices = nullptr,
    };
    VmaAllocationCreateInfo bac{};
    bac.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
    bac.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                VMA_ALLOCATION_CREATE_MAPPED_BIT;
    vmaCreateBuffer(allocator_, &bci, &bac, &stage, &stage_alloc, nullptr);
    VmaAllocationInfo ai{};
    vmaGetAllocationInfo(allocator_, stage_alloc, &ai);
    std::memcpy(ai.pMappedData, data.data(), bytes);
    vkinit::one_time_submit(device_, graphics_queue_, graphics_queue_family_,
                            [&](VkCommandBuffer cb) {
        vkinit::transition_image(cb, grass_mask_.image,
                                  VK_IMAGE_LAYOUT_UNDEFINED,
                                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        VkBufferImageCopy region{};
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.layerCount = 1;
        region.imageExtent = { static_cast<uint32_t>(dim),
                                static_cast<uint32_t>(dim), 1 };
        vkCmdCopyBufferToImage(cb, stage, grass_mask_.image,
                                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                1, &region);
        vkinit::transition_image(cb, grass_mask_.image,
                                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    });
    vmaDestroyBuffer(allocator_, stage, stage_alloc);
    log::infof("grass mask baked: %dx%d RG8 (%.1f KB)", dim, dim,
               static_cast<double>(bytes) / 1024.0);
}

void VulkanEngine::destroy_grass_mask_texture() {
    if (grass_mask_.view)  vkDestroyImageView(device_, grass_mask_.view, nullptr);
    if (grass_mask_.image) vmaDestroyImage(allocator_, grass_mask_.image, grass_mask_.alloc);
    grass_mask_ = {};
}

// Bake the 3-octave fog wisp pattern into a 256² R8 texture so the
// terrain raymarch's volumetric fog density + godray probe can sample
// it once instead of running 3× noise2 per tap. World-space q is
// rescaled to UV by 1/16 so the texture tiles every 16 q-units (в‰€800 m
// of world); the texture sampler uses REPEAT addressing so time
// scrolling and large camera moves stay seamless. Wave-pattern doesn't
// need to bit-match the GLSL noise2 — it just needs to look "wispy".
void VulkanEngine::init_fog_wisp_texture() {
    const int dim = kFogWispDim;
    std::vector<uint8_t> data(static_cast<size_t>(dim) * dim);
    // 16 q-units of pattern fit in the texture; CLAMP/REPEAT works
    // because the FBM is statistically self-similar across periods.
    const float kPeriod = 16.0f;
    for (int y = 0; y < dim; ++y) {
        for (int x = 0; x < dim; ++x) {
            float qx = (static_cast<float>(x) / dim) * kPeriod;
            float qy = (static_cast<float>(y) / dim) * kPeriod;
            // Same weights + scales as the shader's runtime expression:
            //   w = 0.55 noise2(q) + 0.30 noise2(q*2.13) + 0.15 noise2(q*4.27)
            float w = 0.55f * fbm_noise2(qx,         qy)
                    + 0.30f * fbm_noise2(qx * 2.13f, qy * 2.13f)
                    + 0.15f * fbm_noise2(qx * 4.27f, qy * 4.27f);
            w = std::clamp(w, 0.0f, 1.0f);
            data[static_cast<size_t>(y) * dim + x] =
                static_cast<uint8_t>(w * 255.0f);
        }
    }

    VkImageCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = VK_FORMAT_R8_UNORM;
    ici.extent = { static_cast<uint32_t>(dim), static_cast<uint32_t>(dim), 1 };
    ici.mipLevels = 1; ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    vk_check(vmaCreateImage(allocator_, &ici, &aci,
                             &fog_->wisp.image, &fog_->wisp.alloc, nullptr),
             "fog wisp image");
    VkImageViewCreateInfo vci{};
    vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vci.image = fog_->wisp.image;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = VK_FORMAT_R8_UNORM;
    vci.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    vk_check(vkCreateImageView(device_, &vci, nullptr, &fog_->wisp.view),
             "fog wisp view");

    // Stage + copy + transition.
    const VkDeviceSize bytes = data.size();
    VkBuffer stage = VK_NULL_HANDLE;
    VmaAllocation stage_alloc = nullptr;
    VkBufferCreateInfo bci{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = nullptr, .flags = 0, .size = bytes,
        .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0, .pQueueFamilyIndices = nullptr,
    };
    VmaAllocationCreateInfo bac{};
    bac.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
    bac.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                VMA_ALLOCATION_CREATE_MAPPED_BIT;
    vmaCreateBuffer(allocator_, &bci, &bac, &stage, &stage_alloc, nullptr);
    VmaAllocationInfo ai{};
    vmaGetAllocationInfo(allocator_, stage_alloc, &ai);
    std::memcpy(ai.pMappedData, data.data(), bytes);
    vkinit::one_time_submit(device_, graphics_queue_, graphics_queue_family_,
                            [&](VkCommandBuffer cb) {
        vkinit::transition_image(cb, fog_->wisp.image,
                                  VK_IMAGE_LAYOUT_UNDEFINED,
                                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        VkBufferImageCopy region{};
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.layerCount = 1;
        region.imageExtent = { static_cast<uint32_t>(dim),
                                static_cast<uint32_t>(dim), 1 };
        vkCmdCopyBufferToImage(cb, stage, fog_->wisp.image,
                                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                1, &region);
        vkinit::transition_image(cb, fog_->wisp.image,
                                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    });
    vmaDestroyBuffer(allocator_, stage, stage_alloc);
    log::infof("fog wisp baked: %dx%d R8 (%.1f KB)", dim, dim,
               static_cast<double>(bytes) / 1024.0);
}

void VulkanEngine::destroy_fog_wisp_texture() {
    if (fog_->wisp.view)  vkDestroyImageView(device_, fog_->wisp.view, nullptr);
    if (fog_->wisp.image) vmaDestroyImage(allocator_, fog_->wisp.image, fog_->wisp.alloc);
    fog_->wisp = {};
}


} // namespace vibe
