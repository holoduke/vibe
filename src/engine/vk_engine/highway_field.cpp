// The highways' field for the shaders (engine/highway.h: bake()).
//
// Two 1024 x 1024 RGBA8 textures in normal-texture slots kTexRoadField and + 1: the distance
// to the nearest road line, the road's height and the distance along it, for a 4096 m window
// round the camera. Baked again when the camera has moved 600 m from the window's centre (a
// few milliseconds on the main thread) and copied into the textures with that frame's other
// uploads, through one staging buffer per frame in flight. The scene block tells the shader
// where the window is (scene.hw_info) and where in it there is road (scene.hw_bbox).
#include "engine/vk_engine/internal.h"
#include "engine/vk_initializers.h"
#include "engine/highway.h"
#include "engine/globe.h"

#include <cstdlib>
#include <cstring>

namespace vibe {

namespace {

struct HwField {
    bool        made = false;                 // the two textures exist and are bound
    VkImage     image[2]{};
    VkBuffer    stage[kFrameOverlap]{};
    VmaAllocation stage_alloc[kFrameOverlap]{};
    void*       stage_map[kFrameOverlap]{};
    std::vector<uint8_t> px[2];
    bool        upload_pending = false;
    bool        valid = false;
    uint32_t    serial = 0;                   // the chart the field was baked in (globe::chart_serial)
    glm::vec2   center{1e9f};
};
HwField g_hw;
constexpr VkDeviceSize kBytes = VkDeviceSize(highway::kFieldN) * highway::kFieldN * 4;

}  // namespace

void VulkanEngine::hw_poll() {
    if (!scene_desc_set_ || !highway::ready()) return;
    HwField& F = g_hw;
    // In home's chart, or in one near enough for home's roads to be in it (engine/globe.h Ring).
    const globe::Ring rg = globe::ring_now();
    const bool home = globe::current().home || rg.on;
    const glm::vec3 eye = player_.eye_position();
    if (F.serial != globe::chart_serial()) { F.serial = globe::chart_serial(); F.valid = false; }     // (another chart: other coordinates)
    if (!home || eye.y - rt_.water_level > 3000.0f) {             // (too far from home's roads, or far above them: no field)
        F.valid = false;
        hw_info_ = glm::vec4(0.0f);
        return;
    }
    const glm::vec2 e2(eye.x, eye.z);
    if (F.valid && glm::distance(e2, F.center) < 600.0f) return;
    const glm::vec2 c = glm::round(e2 / 256.0f) * 256.0f;
    glm::vec4 bbox(0.0f);
    const bool any = highway::bake(c, F.px[0], F.px[1], bbox, rg);
    F.center = c;
    F.valid = true;
    if (!F.made) {
        // The first time: make the textures and put them in their slots.
        for (int k = 0; k < 2; ++k) {
            Texture2D t = upload_texture_from_pixels(device_, allocator_, graphics_queue_, graphics_queue_family_,
                                                     F.px[k].data(), highway::kFieldN, highway::kFieldN, VK_FORMAT_R8G8B8A8_UNORM, "road_field");
            if (!t.ok) return;
            texture_upload_batch_sync(device_, allocator_, graphics_queue_);
            vkDeviceWaitIdle(device_);
            TextureSlot old = normal_textures_[kTexRoadField + k];
            normal_textures_[kTexRoadField + k] = { t.image, t.alloc, t.view };
            F.image[k] = t.image;
            VkDescriptorImageInfo ii{ texture_sampler_, t.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
            VkWriteDescriptorSet w{};
            w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w.dstSet = scene_desc_set_; w.dstBinding = 4; w.dstArrayElement = uint32_t(kTexRoadField + k); w.descriptorCount = 1;
            w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w.pImageInfo = &ii;
            vkUpdateDescriptorSets(device_, 1, &w, 0, nullptr);
            if (old.view) vkDestroyImageView(device_, old.view, nullptr);
            if (old.image) vmaDestroyImage(allocator_, old.image, old.alloc);
        }
        F.made = true;
    } else {
        F.upload_pending = true;                                  // (record_hw_upload, this frame)
    }
    // (the scene block is written this frame, before the upload is recorded: both are this frame's)
    hw_info_ = any ? glm::vec4(c.x, c.y, 1.0f, highway::kFieldSide) : glm::vec4(0.0f);
    hw_bbox_ = bbox;
}

// The engine's coordinates have shifted by whole tiles. What the field holds - distances,
// heights, metres along the road - does not depend on coordinates; only where its window lies
// does. So the window is moved and the field stays as it is.
void VulkanEngine::hw_shift(glm::dvec2 s) {
    HwField& F = g_hw;
    if (!F.valid) return;
    F.center -= glm::vec2(s);
    F.serial = globe::chart_serial();
    if (hw_info_.z > 0.5f) {
        hw_info_.x -= float(s.x); hw_info_.y -= float(s.y);
        hw_bbox_ -= glm::vec4(float(s.x), float(s.y), float(s.x), float(s.y));
    }
}

void VulkanEngine::record_hw_upload(VkCommandBuffer cmd, uint32_t slot) {
    HwField& F = g_hw;
    if (!F.upload_pending || !F.made) return;
    F.upload_pending = false;
    slot %= kFrameOverlap;
    if (!F.stage[slot]) {
        VkBufferCreateInfo bci{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
        bci.size = kBytes * 2;
        bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
        aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
        VmaAllocationInfo ai{};
        if (vmaCreateBuffer(allocator_, &bci, &aci, &F.stage[slot], &F.stage_alloc[slot], &ai) != VK_SUCCESS) return;
        F.stage_map[slot] = ai.pMappedData;
    }
    for (int k = 0; k < 2; ++k) {
        if (F.px[k].size() != size_t(kBytes)) return;
        std::memcpy(static_cast<uint8_t*>(F.stage_map[slot]) + size_t(kBytes) * size_t(k), F.px[k].data(), size_t(kBytes));
    }
    vmaFlushAllocation(allocator_, F.stage_alloc[slot], 0, VK_WHOLE_SIZE);
    for (int k = 0; k < 2; ++k) {
        vkinit::transition_image(cmd, F.image[k], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        VkBufferImageCopy reg{};
        reg.bufferOffset = kBytes * VkDeviceSize(k);
        reg.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
        reg.imageExtent = { uint32_t(highway::kFieldN), uint32_t(highway::kFieldN), 1 };
        vkCmdCopyBufferToImage(cmd, F.stage[slot], F.image[k], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &reg);
        vkinit::transition_image(cmd, F.image[k], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }
}

void VulkanEngine::hw_shutdown() {
    for (uint32_t i = 0; i < kFrameOverlap; ++i)
        if (g_hw.stage[i]) { vmaDestroyBuffer(allocator_, g_hw.stage[i], g_hw.stage_alloc[i]); g_hw.stage[i] = VK_NULL_HANDLE; }
    g_hw = HwField{};
}

}  // namespace vibe
