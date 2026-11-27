// Whole-map terrain sun shadow ("sun horizon map").
//
// The per-pixel shadow marches and the ray-traced shadows only reach a few
// hundred metres, so at a low sun a mountain a kilometre away never shadowed
// the valley: the forest stayed sunlit and the fog glowed although the sun
// was behind the ridge.
//
// For the current light direction this bakes, over the whole 2048 m map,
// the height S(x, z) of the terrain's shadow volume: the lowest altitude at
// which the sun is visible above that point. One sweep toward the sun does
// it: S(c) = max(H(c), S(next cell toward the sun) - rise of the sun ray
// over one cell). A point p is lit by the terrain iff p.y >= S(p.xz), which
// works for anything - ground, tree tops, towers, fog in the air.
//
// The texture stores R = S and G = S - ground height (0 on lit ground), see
// sun_horizon.glsl. The bake runs on the worker pool whenever the light has
// moved; the result is copied into the texture at the start of a frame.

#include "engine/vk_engine.h"
#include "engine/vk_engine/internal.h"
#include "engine/vk_engine/state/sun_shadow.h"
#include "engine/cpu_prof.h"
#include "engine/log.h"
#include "engine/vk_initializers.h"

#include <glm/gtc/packing.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>

namespace vibe {

namespace {
constexpr int   kHzDim = 1024;            // 2 m cells over the 2048 m map
constexpr float kHzSide = 2048.0f;
constexpr float kHzLit = -10000.0f;       // "nothing blocks the sun here"
}

void VulkanEngine::init_sun_horizon() {
    const size_t cells = size_t(kHzDim) * kHzDim;
    sun_->horizon_bake.assign(cells * 2, 0);
    sun_->horizon_s_bake.assign(cells, kHzLit);
    sun_->horizon_s.assign(cells, kHzLit);

    VkImageCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = VK_FORMAT_R16G16_SFLOAT;
    ici.extent = { uint32_t(kHzDim), uint32_t(kHzDim), 1 };
    ici.mipLevels = 1; ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO;
    vk_check(vmaCreateImage(allocator_, &ici, &aci, &sun_->horizon_image, &sun_->horizon_alloc, nullptr),
             "sun horizon image");
    VkImageViewCreateInfo vci{};
    vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vci.image = sun_->horizon_image;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = ici.format;
    vci.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    vk_check(vkCreateImageView(device_, &vci, nullptr, &sun_->horizon_view), "sun horizon view");

    VkSamplerCreateInfo sci{};
    sci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sci.magFilter = sci.minFilter = VK_FILTER_LINEAR;
    sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    sci.addressModeU = sci.addressModeV = sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.maxLod = 0.0f;
    vk_check(vkCreateSampler(device_, &sci, nullptr, &sun_->horizon_sampler), "sun horizon sampler");

    for (uint32_t i = 0; i < kFrameOverlap; ++i) {
        VkBufferCreateInfo bci{};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size = cells * 2 * sizeof(uint16_t);
        bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VmaAllocationCreateInfo baci{};
        baci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
        baci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
        VmaAllocationInfo info{};
        vk_check(vmaCreateBuffer(allocator_, &bci, &baci, &sun_->horizon_stage[i], &sun_->horizon_stage_alloc[i], &info),
                 "sun horizon staging");
        sun_->horizon_stage_map[i] = info.pMappedData;
    }

    // Until the first bake lands: everything lit.
    vkinit::one_time_submit(device_, graphics_queue_, graphics_queue_family_, [&](VkCommandBuffer cb) {
        vkinit::transition_image(cb, sun_->horizon_image, VK_IMAGE_LAYOUT_UNDEFINED,
                                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        VkClearColorValue clr{}; clr.float32[0] = kHzLit;
        VkImageSubresourceRange r{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        vkCmdClearColorImage(cb, sun_->horizon_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clr, 1, &r);
        vkinit::transition_image(cb, sun_->horizon_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                 VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    });

    // Scene set binding 31 (raymarched terrain, grass).
    if (scene_desc_set_) {
        VkDescriptorImageInfo ii{ sun_->horizon_sampler, sun_->horizon_view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        VkWriteDescriptorSet w{};
        w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet = scene_desc_set_; w.dstBinding = 31; w.descriptorCount = 1;
        w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        w.pImageInfo = &ii;
        vkUpdateDescriptorSets(device_, 1, &w, 0, nullptr);
    }
}

void VulkanEngine::destroy_sun_horizon() {
    jobs::wait(sun_->horizon_job);
    for (uint32_t i = 0; i < kFrameOverlap; ++i) {
        if (sun_->horizon_stage[i]) vmaDestroyBuffer(allocator_, sun_->horizon_stage[i], sun_->horizon_stage_alloc[i]);
        sun_->horizon_stage[i] = VK_NULL_HANDLE; sun_->horizon_stage_alloc[i] = nullptr;
    }
    if (sun_->horizon_sampler) vkDestroySampler(device_, sun_->horizon_sampler, nullptr);
    if (sun_->horizon_view) vkDestroyImageView(device_, sun_->horizon_view, nullptr);
    if (sun_->horizon_image) vmaDestroyImage(allocator_, sun_->horizon_image, sun_->horizon_alloc);
    sun_->horizon_sampler = VK_NULL_HANDLE; sun_->horizon_view = VK_NULL_HANDLE;
    sun_->horizon_image = VK_NULL_HANDLE; sun_->horizon_alloc = nullptr;
}

// Worker thread. Reads the CPU heightmap, writes sun_->horizon_s_bake and
// sun_->horizon_bake (the main thread does not touch either while a bake is
// in flight).
void VulkanEngine::bake_sun_horizon(glm::vec3 L) {
    CPU_SCOPE("+horizon-job");
    const int N = kHzDim;
    const float cell = kHzSide / float(N);
    std::vector<float>& S = sun_->horizon_s_bake;
    static std::vector<float> H;
    H.resize(size_t(N) * N);
    for (int j = 0; j < N; ++j) {
        const float z = (float(j) + 0.5f) * cell - 0.5f * kHzSide;
        for (int i = 0; i < N; ++i)
            H[size_t(j) * N + i] = sample_terrain_height((float(i) + 0.5f) * cell - 0.5f * kHzSide, z);
    }
    const float ax = std::abs(L.x), az = std::abs(L.z);
    if (L.y <= 0.0f) {
        // Light below the horizon: nothing is lit by it anyway.
        std::copy(H.begin(), H.end(), S.begin());
    } else if (std::max(ax, az) < 1e-4f) {
        std::copy(H.begin(), H.end(), S.begin());     // straight overhead
    } else if (ax >= az) {
        // Columns, starting at the sunward edge. One cell toward the sun in
        // x moves `off` cells in z and the sun ray rises by `rise`.
        const int   sx = L.x > 0.0f ? 1 : -1;
        const float off = L.z / ax;
        const float rise = cell * L.y / ax;
        for (int n = 0; n < N; ++n) {
            const int i = sx > 0 ? N - 1 - n : n;
            const int is = i + sx;
            for (int j = 0; j < N; ++j) {
                float s = H[size_t(j) * N + i];
                if (is >= 0 && is < N) {
                    const float fj = float(j) + off;
                    const int j0 = int(std::floor(fj));
                    if (j0 >= 0 && j0 + 1 < N) {
                        const float t = fj - float(j0);
                        const float src = S[size_t(j0) * N + is] * (1.0f - t) + S[size_t(j0 + 1) * N + is] * t;
                        s = std::max(s, src - rise);
                    }
                }
                S[size_t(j) * N + i] = s;
            }
        }
    } else {
        const int   sz = L.z > 0.0f ? 1 : -1;
        const float off = L.x / az;
        const float rise = cell * L.y / az;
        for (int n = 0; n < N; ++n) {
            const int j = sz > 0 ? N - 1 - n : n;
            const int js = j + sz;
            for (int i = 0; i < N; ++i) {
                float s = H[size_t(j) * N + i];
                if (js >= 0 && js < N) {
                    const float fi = float(i) + off;
                    const int i0 = int(std::floor(fi));
                    if (i0 >= 0 && i0 + 1 < N) {
                        const float t = fi - float(i0);
                        const float src = S[size_t(js) * N + i0] * (1.0f - t) + S[size_t(js) * N + i0 + 1] * t;
                        s = std::max(s, src - rise);
                    }
                }
                S[size_t(j) * N + i] = s;
            }
        }
    }
    uint16_t* out = sun_->horizon_bake.data();
    for (size_t c = 0; c < size_t(N) * N; ++c) {
        out[c * 2 + 0] = glm::packHalf1x16(S[c]);
        out[c * 2 + 1] = glm::packHalf1x16(std::min(S[c] - H[c], 2000.0f));
    }
}

// Main thread, once per frame before draw(): collect a finished bake, start
// the next one when the light has moved.
void VulkanEngine::tick_sun_horizon() {
    if (!sun_->horizon_image || terrain_data_.heights.empty()) return;
    if (sun_->horizon_baking && !sun_->horizon_job.busy()) {
        sun_->horizon_baking = false;
        sun_->horizon_s.swap(sun_->horizon_s_bake);
        sun_->horizon_dir = sun_->horizon_bake_dir;
        sun_->horizon_upload_pending = true;
        // VIBE_LOG_SUN_HZ=1: march from the player's feet toward the sun over the
        // CPU ground for 4 km and compare the highest elevation angle with the map.
        if (std::getenv("VIBE_LOG_SUN_HZ")) {
            const glm::vec3 L = sun_->horizon_dir;
            const glm::vec3 p = player_.eye_position() - glm::vec3(0, 1.6f, 0);
            const glm::vec2 d = glm::normalize(glm::vec2(L.x, L.z));
            float best = -90.0f, best_t = 0.0f, best_h = 0.0f;
            for (float t = 2.0f; t < 4000.0f; t += 2.0f) {
                const float h = sample_terrain_height(p.x + d.x * t, p.z + d.y * t);
                const float a = glm::degrees(std::atan2(h - p.y, t));
                if (a > best) { best = a; best_t = t; best_h = h; }
            }
            const float cell = kHzSide / float(kHzDim);
            const int i = int(std::floor((p.x + 0.5f * kHzSide) / cell));
            const int j = int(std::floor((p.z + 0.5f * kHzSide) / cell));
            const float S = (i >= 0 && j >= 0 && i < kHzDim && j < kHzDim) ? sun_->horizon_s[size_t(j) * kHzDim + i] : kHzLit;
            log::infof("sun-hz: p=(%.1f %.1f %.1f) ground=%.2f sun elev=%.1f deg  max terrain elev=%.1f deg at %.0f m (h %.1f)  map S=%.2f vis=%.2f",
                     p.x, p.y, p.z, sample_terrain_height(p.x, p.z), glm::degrees(std::asin(L.y)), best, best_t, best_h, S,
                     sun_horizon_visibility(p + glm::vec3(0, 0.3f, 0)));
            // (and from the GROUND 0 .. 12 m toward the sun: the bank beside the player)
            for (float o = 0.0f; o <= 12.0f; o += 3.0f) {
                const glm::vec2 q(p.x + d.x * o, p.z + d.y * o);
                const float g = sample_terrain_height(q.x, q.y);
                float bq = -90.0f;
                for (float t = 1.0f; t < 2000.0f; t += 1.0f)
                    bq = std::max(bq, glm::degrees(std::atan2(sample_terrain_height(q.x + d.x * t, q.y + d.y * t) - g, t)));
                log::infof("sun-hz: ground at %.0f m toward the sun (%.1f, %.1f) h %.2f: highest terrain %.1f deg (sun %.1f)",
                           o, q.x, q.y, g, bq, glm::degrees(std::asin(L.y)));
            }
        }
    }
    if (sun_->horizon_baking || sun_->horizon_upload_pending) return;
    const float p = glm::radians(rt_.sun_pitch_deg), y = glm::radians(rt_.sun_yaw_deg);
    const glm::vec3 L(std::cos(p) * std::sin(y), std::sin(p), std::cos(p) * std::cos(y));
    // 0.02 degrees of light motion, or every ~4 s regardless (craters and
    // sculpting change the terrain under a stationary sun).
    ++sun_->horizon_age;
    if (glm::distance(L, sun_->horizon_dir) < 3.5e-4f && sun_->horizon_age < 480) return;
    sun_->horizon_age = 0;
    sun_->horizon_bake_dir = L;
    sun_->horizon_baking = true;
    jobs::run(sun_->horizon_job, [this, L] { bake_sun_horizon(L); });
}

// Frame command buffer, before any pass samples the map.
void VulkanEngine::record_sun_horizon_upload(VkCommandBuffer cmd, uint32_t slot) {
    if (!sun_->horizon_upload_pending || !sun_->horizon_image) return;
    sun_->horizon_upload_pending = false;
    std::memcpy(sun_->horizon_stage_map[slot], sun_->horizon_bake.data(),
                sun_->horizon_bake.size() * sizeof(uint16_t));
    vmaFlushAllocation(allocator_, sun_->horizon_stage_alloc[slot], 0, VK_WHOLE_SIZE);
    vkinit::transition_image(cmd, sun_->horizon_image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                             VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    VkBufferImageCopy reg{};
    reg.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    reg.imageExtent = { uint32_t(kHzDim), uint32_t(kHzDim), 1 };
    vkCmdCopyBufferToImage(cmd, sun_->horizon_stage[slot], sun_->horizon_image,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &reg);
    vkinit::transition_image(cmd, sun_->horizon_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                             VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
}

// 0 = the terrain hides the sun from `p`, 1 = sun visible (CPU side: lens
// flare, sun glare).
float VulkanEngine::sun_horizon_visibility(glm::vec3 p) const {
    if (sun_->horizon_s.empty()) return 1.0f;
    const float cell = kHzSide / float(kHzDim);
    const int i = int(std::floor((p.x + 0.5f * kHzSide) / cell));
    const int j = int(std::floor((p.z + 0.5f * kHzSide) / cell));
    if (i < 0 || j < 0 || i >= kHzDim || j >= kHzDim) return 1.0f;
    return glm::smoothstep(-1.0f, 1.0f, p.y - sun_->horizon_s[size_t(j) * kHzDim + i]);
}

}  // namespace vibe
