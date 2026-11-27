// The lighting ports: techniques of the reference game's lighting that Vibe lacked (survey of
// 2026-11-26; docs/ARCHITECTURE.md "Lighting ports"). This file fills their part of the scene
// block (scene_ubo.glsl lp_*): the switches, the projection numbers the screen-space marches
// need, and the sky-change scale of the bounce-light caches. The shading itself is in
// deferred_lighting.frag, contact_shadow.glsl and terrain_rm_main_light.glsl.
//
// Every switch is a setting (RtSettings lp_*, render menu) and has an environment override:
// VIBE_LP_CONTACT, VIBE_LP_FAR_AO, VIBE_LP_LEAF_TINT, VIBE_LP_SKY_RESCALE, VIBE_LP_BOUNCE_ALBEDO,
// VIBE_LP_LAMP_PICK, VIBE_LP_SCREEN_MARCH = 0 / 1. VIBE_LOG_LP=1 logs the state every 2 s.

#include "engine/vk_engine.h"
#include "engine/vk_engine/internal.h"
#include "engine/log.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <vector>

namespace vibe {

namespace {

LightPortFrame g_lp;
float g_sky_ref = -1.0f;      // the sky's light the caches were last in step with
int   g_rescales = 0;

// The lamp list (deferred pass binding 26): two regions of kMaxLamps entries, one a frame in
// flight (a frame writes its region while the GPU may still read the other's); an entry is two
// vec4 - (position, intensity) and (slot in scene.lamp_lights or -1, 0, 0, 0).
constexpr uint32_t kMaxLamps = 512;
constexpr int      kLoopLamps = 6;     // the nearest this many are looped, each with its shadow ray
VkBuffer      g_lamp_buf = VK_NULL_HANDLE;
VmaAllocation g_lamp_alloc = nullptr;
glm::vec4*    g_lamp_map = nullptr;
std::vector<glm::vec4> g_extra;       // this frame's lamps beyond the 24 slots

// A setting, unless the environment says otherwise.
bool lp_switch(const char* env, bool setting) {
    const char* v = std::getenv(env);
    if (!v || !*v) return setting;
    return v[0] != '0';
}

} // namespace

const LightPortFrame& light_port_frame() { return g_lp; }

void light_port_extra_lamps(const glm::vec4* lamps, size_t n) { g_extra.assign(lamps, lamps + n); }

VkBuffer light_port_lamp_buffer(VmaAllocator a) {
    if (!g_lamp_buf && a) {
        VkBufferCreateInfo bci{};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size = sizeof(glm::vec4) * 2 * kMaxLamps * kFrameOverlap;
        bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO;
        aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
        VmaAllocationInfo info{};
        if (vmaCreateBuffer(a, &bci, &aci, &g_lamp_buf, &g_lamp_alloc, &info) == VK_SUCCESS) {
            g_lamp_map = static_cast<glm::vec4*>(info.pMappedData);
            std::memset(g_lamp_map, 0, size_t(bci.size));
        } else {
            g_lamp_buf = VK_NULL_HANDLE;
            log::info("[light-port] lamp list buffer: allocation failed");
        }
    }
    return g_lamp_buf;
}

// Last frame's lit picture (deferred pass binding 28; scene.lp_flags bit 32): a copy of the
// deferred pass's output, made at the end of that pass while the screen march is on.
namespace {
VkImage       g_prev_img = VK_NULL_HANDLE;
VmaAllocation g_prev_alloc = nullptr;
VkImageView   g_prev_view = VK_NULL_HANDLE;
VkExtent2D    g_prev_ext{ 0, 0 };
VkDevice      g_prev_dev = VK_NULL_HANDLE;
bool          g_prev_valid = false;
bool          g_march_want = false;    // the screen march is switched on (light_port_fill)
void prev_free(VmaAllocator a) {
    if (g_prev_view) vkDestroyImageView(g_prev_dev, g_prev_view, nullptr);
    if (g_prev_img) vmaDestroyImage(a, g_prev_img, g_prev_alloc);
    g_prev_img = VK_NULL_HANDLE; g_prev_alloc = nullptr; g_prev_view = VK_NULL_HANDLE;
    g_prev_ext = { 0, 0 }; g_prev_valid = false;
}
} // namespace

VkImageView light_port_prev_lit() { return g_prev_valid ? g_prev_view : VK_NULL_HANDLE; }

void light_port_copy_lit(VkCommandBuffer cmd, VkDevice dev, VmaAllocator a, VkImage src, VkExtent2D ext, VkFormat fmt) {
    if (!g_march_want) return;
    if (g_prev_img && (g_prev_ext.width != ext.width || g_prev_ext.height != ext.height)) {
        vkDeviceWaitIdle(dev);              // (a new render size: rare)
        prev_free(a);
    }
    if (!g_prev_img) {
        VkImageCreateInfo ici{};
        ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ici.imageType = VK_IMAGE_TYPE_2D;
        ici.format = fmt;
        ici.extent = { ext.width, ext.height, 1 };
        ici.mipLevels = 1; ici.arrayLayers = 1;
        ici.samples = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling = VK_IMAGE_TILING_OPTIMAL;
        ici.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        if (vmaCreateImage(a, &ici, &aci, &g_prev_img, &g_prev_alloc, nullptr) != VK_SUCCESS) { g_prev_img = VK_NULL_HANDLE; return; }
        VkImageViewCreateInfo vci{};
        vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vci.image = g_prev_img; vci.viewType = VK_IMAGE_VIEW_TYPE_2D; vci.format = fmt;
        vci.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        vkCreateImageView(dev, &vci, nullptr, &g_prev_view);
        g_prev_ext = ext; g_prev_dev = dev; g_prev_valid = false;
    }
    vkinit::transition_image(cmd, g_prev_img, g_prev_valid ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_UNDEFINED,
                             VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    VkImageCopy r{};
    r.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    r.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    r.extent = { ext.width, ext.height, 1 };
    vkCmdCopyImage(cmd, src, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, g_prev_img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &r);
    vkinit::transition_image(cmd, g_prev_img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    g_prev_valid = true;
}

void light_port_destroy(VmaAllocator a) {
    if (g_lamp_buf) vmaDestroyBuffer(a, g_lamp_buf, g_lamp_alloc);
    g_lamp_buf = VK_NULL_HANDLE; g_lamp_alloc = nullptr; g_lamp_map = nullptr;
    prev_free(a);
}

void VulkanEngine::light_port_fill(SceneUBO& d) {
    static_assert(kTextureCount == 48, "LP_TEX_COUNT in deferred_lighting.frag must match");
    const bool contact = lp_switch("VIBE_LP_CONTACT", rt_.lp_contact_shadows);
    const bool far_ao  = lp_switch("VIBE_LP_FAR_AO", rt_.lp_far_ao);
    const bool leaf    = lp_switch("VIBE_LP_LEAF_TINT", rt_.lp_leaf_tint);
    const bool sky_rs  = lp_switch("VIBE_LP_SKY_RESCALE", rt_.lp_sky_rescale);
    const bool albedo  = lp_switch("VIBE_LP_BOUNCE_ALBEDO", rt_.lp_bounce_albedo);
    const bool lamps   = lp_switch("VIBE_LP_LAMP_PICK", rt_.lp_lamp_pick);
    const bool smarch  = lp_switch("VIBE_LP_SCREEN_MARCH", rt_.lp_screen_march);
    // (the screen march reads last frame's lit picture: on from the frame after one was kept)
    g_march_want = smarch;
    // VIBE_LEAF_SHADE_OLD=1: leaves / grass in ANY shadow lose 55 % of their sky light again (the
    // old rule; now only under a blocker within a few metres - deferred_lighting.frag 'canopy').
    static const bool leaf_shade_old = [] { const char* e = std::getenv("VIBE_LEAF_SHADE_OLD"); return e && e[0] == '1'; }();
    const int bits = (contact ? 1 : 0) | (far_ao ? 2 : 0) | (leaf ? 4 : 0) | (albedo ? 8 : 0) |
                     (lamps ? 16 : 0) | ((smarch && light_port_prev_lit() != VK_NULL_HANDLE) ? 32 : 0) |
                     (leaf_shade_old ? 1024 : 0);
    // (the terrain's slot in the materials table: rt.cpp - after the brushes, when it is traced)
    d.lp_flags = glm::ivec4(bits, 0, tlas_includes_terrain_blas() ? int(world_.brushes.size()) : -1, 0);
    const glm::mat4& P = current_frame_view_.proj;
    d.lp_params = glm::vec4(P[3][2], P[2][2], 1.0f, float(kLoopLamps));
    d.lp_vp = current_frame_view_.vp;

    // The lamp list: the scene block's slots (nearest the eye, rooms, candles, the rocket) and
    // every lamp beyond them that update_scene_ubo found (street lamps out to 800 m), nearest
    // the eye first, up to kMaxLamps.
    if (lamps && light_port_lamp_buffer(allocator_) && g_lamp_map) {
        struct E { float d; glm::vec4 a; float slot; };
        std::vector<E> all;
        all.reserve(32 + g_extra.size());
        const glm::vec3 eye = player_.eye_position();
        const int used = int(d.lamp_info.y + 0.5f);
        for (int k = 0; k < used && k < int(std::size(d.lamp_lights)); ++k)
            if (d.lamp_lights[k].w != 0.0f)
                all.push_back({ glm::distance(eye, glm::vec3(d.lamp_lights[k])), d.lamp_lights[k], float(k) });
        for (const glm::vec4& e : g_extra)
            if (e.w != 0.0f) all.push_back({ glm::distance(eye, glm::vec3(e)), e, -1.0f });
        std::sort(all.begin(), all.end(), [](const E& x, const E& y) { return x.d < y.d; });
        const uint32_t n = uint32_t(std::min<size_t>(all.size(), kMaxLamps));
        const uint32_t base = uint32_t(frame_number_ % kFrameOverlap) * kMaxLamps;
        for (uint32_t i = 0; i < n; ++i) {
            g_lamp_map[2 * (base + i)] = all[i].a;
            g_lamp_map[2 * (base + i) + 1] = glm::vec4(all[i].slot, 0.0f, 0.0f, 0.0f);
        }
        vmaFlushAllocation(allocator_, g_lamp_alloc, 0, VK_WHOLE_SIZE);
        d.lp_flags.y = int(n);
        d.lp_flags.w = int(base);
        static const bool lgl = std::getenv("VIBE_LOG_LP") != nullptr;
        if (lgl && frame_number_ % 120 == 0)
            log::infof("[light-port] lamp list: %u lamps (%d in the scene block's slots, %zu beyond), the nearest %d looped",
                       n, used, g_extra.size(), kLoopLamps);
    }
    g_extra.clear();

    // A sudden drop of the sky's light (the reference game: more than 4x darker at once ->
    // the caches are scaled down rather than left to fade for seconds). The light the caches
    // hold comes from the sun and the sky; a drop that the caches would follow within a
    // fraction of a second (sunset over minutes) never trips it: the reference follows the
    // light slowly, and only a step larger than 4x in about a second does.
    {
        const glm::vec3 kLum(0.2126f, 0.7152f, 0.0722f);
        const float sun_up = std::max(d.sun_direction.y, 0.0f);
        const float cloud = std::clamp(d.crater_info.y, 0.0f, 1.0f);
        const float E = glm::dot(glm::vec3(d.sky_color), kLum) * 0.45f +
                        glm::dot(glm::vec3(d.sun_color), kLum) * d.sun_color.w * sun_up * (1.0f - 0.6f * cloud) +
                        glm::dot(glm::vec3(d.ambient), kLum) + 1e-4f;
        g_lp.cache_scale = 1.0f;
        if (g_sky_ref < 0.0f) g_sky_ref = E;
        if (sky_rs && E < g_sky_ref * 0.25f) {
            g_lp.cache_scale = std::max(E / g_sky_ref, 0.002f);
            g_sky_ref = E;
            ++g_rescales;
        } else {
            // (up at once - a brighter sky is no ghost; down slowly, ~1 s at 60 fps)
            g_sky_ref = E > g_sky_ref ? E : g_sky_ref + (E - g_sky_ref) * 0.05f;
        }
        d.lp_params.z = g_lp.cache_scale;
        static const bool lg = std::getenv("VIBE_LOG_LP") != nullptr;
        if ((lg && frame_number_ % 120 == 0) || (lg && g_lp.cache_scale < 1.0f))
            log::infof("[light-port] bits 0x%x  sky light %.4f (ref %.4f)  cache scale %.3f  rescales %d",
                       bits, double(E), double(g_sky_ref), double(g_lp.cache_scale), g_rescales);
    }
}

} // namespace vibe
