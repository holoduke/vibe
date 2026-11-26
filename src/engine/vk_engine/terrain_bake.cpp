// The baked ground texture: the ground's material kept in pictures round the camera, so the
// terrain pass does not work it out again for every pixel of every frame.
//
// What it is, and what is in the pictures: shaders/terrain_rm_bake.glsl. In short - kLevels
// square pictures of kSize texels, each level's texel twice the last, all centred on the camera;
// a texel's place in a picture is its world index modulo kSize (the sampler repeats), so when
// the camera moves only the strips that came into a level's window are drawn anew. They are
// drawn by a build of the terrain shader itself (terrain_bake.frag.spv, -DVIBE_BAKE): the same
// material code, standing over each texel at the level's own distance.
//
// update_terrain_bake(), once a frame before the world pass:
//   * the material's inputs are hashed (terrain_bake_signature, from the scene constants: snow,
//     season, the palette, the road, the chart ...). When the hash has changed and then stood
//     still for 20 frames, every level is made again - a quarter of a level a frame; until they
//     are whole the terrain pass works the material out per pixel, as without the bake.
//   * a level whose window has moved (by kSnap texels or more) gets its new strips at once.
// The terrain pass (terrain_raymarch.frag, with the patch terrain only) reads the pictures for
// ground nearer than 64 x t0 and works the material out itself beyond that, on steep faces,
// under water and round footprints.
//
// Setting: rt_.terrain_bake (VIBE_TERRAIN_BAKE=0|1). VIBE_LOG_TERRAIN_BAKE=1 logs what is drawn.

#include "engine/vk_engine.h"
#include "engine/vk_engine/internal.h"
#include "engine/vk_engine/state/terrain_bake.h"

#include "engine/globe.h"
#include "engine/log.h"
#include "engine/vk_initializers.h"
#include "engine/vk_pipelines.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>

namespace vibe {

void VulkanEngine::init_terrain_bake() {
    TerrainBakeState& S = *tb_;
    // The normal's two numbers want more than 8 bits (the asphalt's gloss shows steps).
    S.format[0] = VK_FORMAT_R8G8B8A8_UNORM;
    S.format[1] = VK_FORMAT_R16G16_UNORM;
    S.format[2] = VK_FORMAT_R8G8B8A8_UNORM;
    {
        VkFormatProperties fp{};
        vkGetPhysicalDeviceFormatProperties(physical_device_, S.format[1], &fp);
        const VkFormatFeatureFlags need = VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
        if ((fp.optimalTilingFeatures & need) != need) S.format[1] = VK_FORMAT_R16G16_SFLOAT;
    }
    for (int i = 0; i < 3; ++i) {
        VkImageCreateInfo ici{};
        ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ici.imageType = VK_IMAGE_TYPE_2D;
        ici.format = S.format[i];
        ici.extent = { uint32_t(TerrainBakeState::kSize), uint32_t(TerrainBakeState::kSize), 1 };
        ici.mipLevels = 1;
        ici.arrayLayers = uint32_t(TerrainBakeState::kLevels);
        ici.samples = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling = VK_IMAGE_TILING_OPTIMAL;
        ici.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        vk_check(vmaCreateImage(allocator_, &ici, &aci, &S.image[i], &S.alloc[i], nullptr), "terrain bake image");
        VkImageViewCreateInfo vci{};
        vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vci.image = S.image[i];
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
        vci.format = S.format[i];
        vci.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, uint32_t(TerrainBakeState::kLevels) };
        vk_check(vkCreateImageView(device_, &vci, nullptr, &S.array_view[i]), "terrain bake view");
        for (int l = 0; l < TerrainBakeState::kLevels; ++l) {
            vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
            vci.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, uint32_t(l), 1 };
            vk_check(vkCreateImageView(device_, &vci, nullptr, &S.layer_view[i][l]), "terrain bake layer view");
        }
    }
    {
        VkSamplerCreateInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        si.magFilter = VK_FILTER_LINEAR;
        si.minFilter = VK_FILTER_LINEAR;
        si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        si.addressModeU = si.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        vk_check(vkCreateSampler(device_, &si, nullptr, &S.sampler), "terrain bake sampler");
    }
    // (readable from the start: the terrain shader's bindings must be valid before anything is baked)
    vkinit::one_time_submit(device_, graphics_queue_, graphics_queue_family_, [&](VkCommandBuffer cb) {
        for (int i = 0; i < 3; ++i) {
            VkImageMemoryBarrier b{};
            b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            b.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.image = S.image[i];
            b.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, VK_REMAINING_ARRAY_LAYERS };
            b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
        }
    });
    VkDescriptorImageInfo ii[3]{};
    VkWriteDescriptorSet w[3]{};
    for (int i = 0; i < 3; ++i) {
        ii[i] = { S.sampler, S.array_view[i], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[i].dstSet = scene_desc_set_;
        w[i].dstBinding = uint32_t(46 + i);
        w[i].descriptorCount = 1;
        w[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        w[i].pImageInfo = &ii[i];
    }
    vkUpdateDescriptorSets(device_, 3, w, 0, nullptr);

    S.frag = vkpipe::load_shader_module(device_, std::string(VIBE_SHADER_DIR) + "/terrain_bake.frag.spv");
    if (S.frag && terrain_raymarch_vert_module_) {
        vkpipe::GraphicsPipelineConfig cfg{};
        cfg.vert = terrain_raymarch_vert_module_;       // the full-screen triangle
        cfg.frag = S.frag;
        cfg.layout = pipeline_layout_;
        cfg.color_formats = { S.format[0], S.format[1], S.format[2] };
        cfg.depth_format = VK_FORMAT_UNDEFINED;
        cfg.cull = VK_CULL_MODE_NONE;
        cfg.depth_test = false;
        cfg.depth_write = false;
        const VkDevice dev = device_;
        S.build = std::async(std::launch::async, [dev, cfg]() { return vkpipe::build_graphics_pipeline(dev, cfg); });
    }
    log::infof("[terrain] baked ground texture: %d levels of %d x %d texels (3 pictures, %.0f MB), level 0 at %.2f cm a texel",
               TerrainBakeState::kLevels, TerrainBakeState::kSize, TerrainBakeState::kSize,
               double(TerrainBakeState::kLevels) * TerrainBakeState::kSize * TerrainBakeState::kSize * 12.0 / (1024.0 * 1024.0),
               double(TerrainBakeState::kTexel0) * 100.0);
}

void VulkanEngine::destroy_terrain_bake() {
    if (!tb_) return;
    TerrainBakeState& S = *tb_;
    if (S.build.valid()) S.pipeline = S.build.get();
    if (S.pipeline) vkDestroyPipeline(device_, S.pipeline, nullptr);
    if (S.frag) vkDestroyShaderModule(device_, S.frag, nullptr);
    if (S.sampler) vkDestroySampler(device_, S.sampler, nullptr);
    for (int i = 0; i < 3; ++i) {
        for (VkImageView& v : S.layer_view[i]) if (v) vkDestroyImageView(device_, v, nullptr);
        if (S.array_view[i]) vkDestroyImageView(device_, S.array_view[i], nullptr);
        if (S.image[i]) vmaDestroyImage(allocator_, S.image[i], S.alloc[i]);
    }
    S = TerrainBakeState{};
}

// The ground is made anew (sculpting, a crater): so are the pictures.
// (A crater changes it too, but only round itself: update_terrain_bake draws that part again and
//  this call, arriving with the crater or a frame or two later, is then not a reason to make
//  everything again.)
void VulkanEngine::terrain_bake_invalidate() { if (tb_) tb_->height_dirty = true; }
// The ground inside a box changed (a region's height tile: regions.cpp): only that part of the
// pictures is drawn again.
void VulkanEngine::terrain_bake_redraw(float x0, float z0, float x1, float z1) {
    if (tb_ && tb_->redo_boxes.size() < 64) tb_->redo_boxes.push_back(glm::vec4(x0, z0, x1, z1));
    else if (tb_) tb_->height_dirty = true;
}

// What the ground's material is worked out from, hashed: called by update_scene_ubo with the
// frame's scene constants. (Not the sun, not the camera: lighting stays per pixel, and the
// pictures follow the camera by themselves.)
void VulkanEngine::terrain_bake_signature(const SceneUBO& d) {
    uint64_t h = 1469598103934665603ull;
    const auto add = [&](const void* p, size_t n) {
        const unsigned char* b = static_cast<const unsigned char*>(p);
        for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; }
    };
    const auto addv = [&](const glm::vec4& v) { add(&v, sizeof(v)); };
    addv(d.rm_rock); addv(d.rm_ground_dry); addv(d.rm_ground_green); addv(d.rm_red_sand);
    addv(d.terrain_sand_color); addv(d.grass_color_ground); addv(d.grass_color_ground_far); addv(d.grass_color_bottom);
    addv(d.road_info); addv(d.spom_params); addv(d.site_ground); addv(d.site_shape); addv(d.terrain_extra);
    for (const glm::vec4& f : d.flower_field) addv(f);       // (the ground in a flower field takes the field's colour with distance)
    for (const glm::vec4& f : d.hw_jn) addv(f);              // (a junction's paving and markings: highway.h Junction)
    addv(d.stream_info); for (const glm::vec4& f : d.stream_pts) addv(f);      // (the stream's channel is cut into the ground: engine/water_course.h)
    const float q[] = { d.water_params.x, d.water_params.y, std::floor(d.ripple_info.z), d.globe_info.x, d.globe_info.w,
                        std::round(d.season.x * 64.0f), std::round(d.season.y * 64.0f), std::round(d.season.z * 64.0f), std::round(d.season.w * 64.0f),
                        d.viewport.w, float(int(d.rm_perf.x) & (2 | 64)), d.rm_perf.w,
                        float(rt_.terrain_raymarch_octaves), float(rt_.terrain_raymarch_normal_octaves),
                        // (the formula's distances grow with the camera's height over the water - in steps
                        //  here: read to the last digit they changed every frame of a drive over a hill, and
                        //  the pictures were never made)
                        rt_.terrain_raymarch_lod_near_m, rt_.terrain_raymarch_lod_far_m, std::round(std::log2(std::max(rm_air_lod_k_, 1.0f)) * 2.0f),
                        float(rt_.terrain_raymarch_lod_min_octaves), float(globe::chart_serial()), rt_.textures_enabled ? 1.0f : 0.0f };
    add(q, sizeof(q));
    tb_->sig = h;
}

bool VulkanEngine::terrain_bake_ready() const { return tb_ && tb_->ready && terrain_patches_active(); }
// (with the first level in use this frame in the eights: terrain_raymarch.frag takes the two apart)
float VulkanEngine::terrain_bake_t0() const { return tb_ ? tb_->t0 + 8.0f * float(tb_->lmin) : 0.0f; }

void VulkanEngine::update_terrain_bake(VkCommandBuffer cmd) {
    if (!tb_) return;
    TerrainBakeState& S = *tb_;
    S.texels_frame = 0;
    if (S.build.valid() && S.build.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
        S.pipeline = S.build.get();
        log::info("[terrain] baked ground texture: its shader is compiled");
    }
    static const bool kLogState = std::getenv("VIBE_LOG_TERRAIN_BAKE") != nullptr;
    if (kLogState && (frame_number_ % 120) == 60)
        log::infof("[terrain-bake] state: setting %d, shader %d, patches drawn %d, ready %d, inputs unchanged for %d frames%s",
                   rt_.terrain_bake ? 1 : 0, S.pipeline ? 1 : 0, terrain_patches_active() ? 1 : 0, S.ready ? 1 : 0, S.sig_still,
                   S.sig != S.sig_baked ? " (changed since the pictures were made)" : "");
    if (!rt_.terrain_bake || !S.pipeline || !terrain_patches_active()) { S.ready = false; return; }

    // Craters (scene.craters: the bowl in the height, the mud, the paving gone): what is new or
    // changed in the list is drawn again in every level, at once and only within its reach.
    std::vector<glm::vec4> redo;
    for (size_t i = 0; i < craters_.size(); ++i)
        if (i >= S.craters_seen.size() || S.craters_seen[i] != craters_[i]) redo.push_back(craters_[i]);
    if (!redo.empty()) { S.craters_seen = craters_; S.crater_last = redo; S.crater_cool = 4; }
    else if (S.craters_seen.size() != craters_.size()) S.craters_seen = craters_;
    if (S.height_dirty) {
        S.height_dirty = false;
        if (S.crater_cool > 0) redo = S.crater_last;      // (the crater's bowl reached the height texture now)
        else S.sig_baked = ~0ull;                         // (sculpting: anywhere)
    }
    if (S.crater_cool > 0) --S.crater_cool;
    // Made again when the material's inputs have changed and then stood still for a moment
    // (snow that is still falling changes them every few frames: no use chasing it).
    if (S.sig != S.sig_baked) {
        if (S.sig == S.sig_seen) ++S.sig_still; else { S.sig_seen = S.sig; S.sig_still = 0; }
        S.ready = false;
        if (S.sig_still < 20) return;
        S.sig_baked = S.sig;
        for (TerrainBakeState::Level& lv : S.level) { lv.valid = false; lv.tiles = 0; }
        S.redo_boxes.clear();                                 // (everything is drawn again anyway)
    }
    // Level 0 is made for the distance at which one of its texels is one pixel (never further
    // than its window allows: 3.2 m).
    S.t0 = std::min(3.2f, TerrainBakeState::kTexel0 * float(render_extent_.height) / 1.68f);

    const int N = TerrainBakeState::kSize, H = N / 2;
    const glm::vec3 eye = current_frame_view_.eye_pos;
    struct Rect { int level; glm::ivec2 w0; VkRect2D sc; };
    std::vector<Rect> rects;
    // A rectangle of world texel indices [x0, x1) x [z0, z1) of a level, as the one to four
    // rectangles of the picture it falls on (the picture repeats).
    const auto add_world = [&](int level, glm::ivec2 w0, int x0, int x1, int z0, int z1) {
        if (x1 <= x0 || z1 <= z0) return;
        int xs[2][2], zs[2][2], nx = 0, nz = 0;
        const auto spans = [&](int a, int b, int out[2][2], int& n) {
            const int am = a & (N - 1), len = b - a;
            if (am + len <= N) { out[0][0] = am; out[0][1] = len; n = 1; }
            else { out[0][0] = am; out[0][1] = N - am; out[1][0] = 0; out[1][1] = len - (N - am); n = 2; }
        };
        spans(x0, x1, xs, nx); spans(z0, z1, zs, nz);
        for (int j = 0; j < nz; ++j)
            for (int i = 0; i < nx; ++i) {
                rects.push_back({ level, w0, VkRect2D{ { xs[i][0], zs[j][0] }, { uint32_t(xs[i][1]), uint32_t(zs[j][1]) } } });
                S.texels_frame += uint32_t(xs[i][1]) * uint32_t(zs[j][1]);
            }
    };
    for (int l = 0; l < TerrainBakeState::kLevels; ++l) {
        const TerrainBakeState::Level& lv = S.level[l];
        if (!lv.valid) continue;                              // (being filled: it gets the crater anyway)
        const double texel = double(TerrainBakeState::kTexel0) * double(1 << l);
        const glm::ivec2 w0 = lv.c - H;
        for (const glm::vec4& cr : redo) {
            const double reach = 3.0 * double(std::max(cr.w, 0.5f)) + 1.0;
            const int x0 = std::max(int(std::floor((cr.x - reach) / texel)), w0.x), x1 = std::min(int(std::ceil((cr.x + reach) / texel)), w0.x + N);
            const int z0 = std::max(int(std::floor((cr.z - reach) / texel)), w0.y), z1 = std::min(int(std::ceil((cr.z + reach) / texel)), w0.y + N);
            add_world(l, w0, x0, x1, z0, z1);
        }
    }
    // Boxes whose height changed (terrain_bake_redraw): the same. A level that is being filled
    // and has drawn part of such a box already starts again.
    for (int l = 0; l < TerrainBakeState::kLevels; ++l) {
        TerrainBakeState::Level& lv = S.level[l];
        const double texel = double(TerrainBakeState::kTexel0) * double(1 << l);
        const glm::ivec2 w0 = lv.c - H;
        for (const glm::vec4& bx : S.redo_boxes) {
            const int x0 = std::max(int(std::floor(double(bx.x) / texel)), w0.x), x1 = std::min(int(std::ceil(double(bx.z) / texel)), w0.x + N);
            const int z0 = std::max(int(std::floor(double(bx.y) / texel)), w0.y), z1 = std::min(int(std::ceil(double(bx.w) / texel)), w0.y + N);
            if (x1 <= x0 || z1 <= z0) continue;
            if (lv.valid) add_world(l, w0, x0, x1, z0, z1);
            else lv.tiles = 0;
        }
    }
    S.redo_boxes.clear();
    // (a blast the size of a crashed aircraft reaches further than the near levels are wide: that
    //  is not a strip but most of the pictures - they are filled again at the usual rate, and the
    //  terrain pass works the material out per pixel meanwhile)
    if (S.texels_frame > 3u * 1024u * 1024u) {
        rects.clear();
        S.texels_frame = 0;
        for (TerrainBakeState::Level& lv : S.level) { lv.valid = false; lv.tiles = 0; }
    }
    const uint32_t crater_texels = S.texels_frame;
    // At speed the finest levels are left alone: their windows are the narrowest in metres, so
    // they are the ones a moving camera redraws most of (at 40 m/s level 0 alone was a third of
    // the texels drawn per frame, 1 ms in all), and what they hold - detail of millimetres - is
    // not seen on ground that crosses the screen that fast. Above 12 m/s reading starts at level
    // 1, above 28 m/s at level 2; slower again, the finer levels are filled before they are read.
    {
        const float dt = glm::clamp(last_frame_dt_, 1.0f / 500.0f, 0.1f);
        const float v = glm::length(glm::vec2(eye.x - S.prev_eye.x, eye.z - S.prev_eye.z)) / dt;
        S.prev_eye = eye;
        S.speed += (std::min(v, 400.0f) - S.speed) * 0.1f;
        if (S.speed > 28.0f) S.want_min = 2;
        else if (S.speed > 12.0f) S.want_min = std::max(S.want_min == 2 && S.speed > 20.0f ? 2 : 1, 1);
        else if (S.speed < 8.0f) S.want_min = 0;
    }
    int fill_budget = TerrainBakeState::kFillTexels;
    bool all = true;
    for (int l = TerrainBakeState::kLevels - 1; l >= 0; --l) {      // (the coarse levels first: they are needed whatever the speed)
        TerrainBakeState::Level& lv = S.level[l];
        if (l < S.want_min) { lv.valid = false; lv.tiles = 0; continue; }
        const double step = double(TerrainBakeState::kTexel0) * double(1 << l) * double(TerrainBakeState::kSnap);
        const glm::ivec2 c(int(std::llround(double(eye.x) / step)) * TerrainBakeState::kSnap,
                           int(std::llround(double(eye.z) / step)) * TerrainBakeState::kSnap);
        if (lv.valid && c != lv.c) {
            const glm::ivec2 d = c - lv.c;
            if (std::abs(d.x) >= H || std::abs(d.y) >= H) { lv.valid = false; lv.tiles = 0; }      // (a jump: the whole level)
            else {
                const glm::ivec2 w0 = c - H;
                if (d.x > 0) add_world(l, w0, lv.c.x + H, c.x + H, w0.y, w0.y + N);
                if (d.x < 0) add_world(l, w0, w0.x, lv.c.x - H, w0.y, w0.y + N);
                if (d.y > 0) add_world(l, w0, w0.x, w0.x + N, lv.c.y + H, c.y + H);
                if (d.y < 0) add_world(l, w0, w0.x, w0.x + N, w0.y, lv.c.y - H);
                lv.c = c;
            }
        }
        if (!lv.valid) {
            if (lv.tiles == 0) lv.c = c;
            while (lv.tiles < 4 && fill_budget > 0) {
                // (a quarter of the picture as it lies: which part of the world it holds follows from the window)
                rects.push_back({ l, lv.c - H, VkRect2D{ { (lv.tiles & 1) * H, (lv.tiles >> 1) * H }, { uint32_t(H), uint32_t(H) } } });
                S.texels_frame += uint32_t(H) * uint32_t(H);
                fill_budget -= H * H;
                ++lv.tiles;
            }
            if (lv.tiles == 4) lv.valid = true;      // (drawn below, this frame)
        }
        all = all && lv.valid;
    }
    const bool was_ready = S.ready;
    // (read from the finest level below which all are whole - so after a fill, or when the camera
    //  slows down, the coarser levels serve until the finer ones are there)
    int le = TerrainBakeState::kLevels;
    while (le > 0 && S.level[le - 1].valid) --le;
    (void)all;
    S.ready = le <= 2;
    S.lmin = le;
    if (rects.empty()) return;

    const auto barrier = [&](VkImageLayout from, VkImageLayout to) {
        VkImageMemoryBarrier b[3]{};
        for (int i = 0; i < 3; ++i) {
            b[i].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            b[i].oldLayout = from; b[i].newLayout = to;
            b[i].srcQueueFamilyIndex = b[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b[i].image = S.image[i];
            b[i].subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, VK_REMAINING_ARRAY_LAYERS };
            b[i].srcAccessMask = from == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL ? VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT : VK_ACCESS_SHADER_READ_BIT;
            b[i].dstAccessMask = to == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL ? (VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT) : VK_ACCESS_SHADER_READ_BIT;
        }
        const VkPipelineStageFlags fs = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, co = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        vkCmdPipelineBarrier(cmd, from == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL ? co : fs,
                             to == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL ? co : fs, 0, 0, nullptr, 0, nullptr, 3, b);
    };
    barrier(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, S.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout_, 0, 1, &scene_desc_set_, 0, nullptr);
    PushConstants pc{};
    pc.mvp = current_frame_view_.vp;
    pc.prev_mvp = current_frame_view_.vp;
    // (the terrain pass's constants for the height formula - world_render.cpp render_world; the
    //  step factor positive: this is not the full-resolution pass with its per-pixel helpers)
    pc.color = glm::vec4(0.0f, 0.0f, 11.5f, 22.0f);
    pc.emissive = glm::vec4(
        std::clamp(rt_.terrain_raymarch_step_factor, 0.05f, 0.8f),
        std::max(1.0f, rt_.terrain_raymarch_lod_near_m * rm_air_lod_k_),
        std::max(rt_.terrain_raymarch_lod_near_m * rm_air_lod_k_ + 1.0f, rt_.terrain_raymarch_lod_far_m * rm_air_lod_k_),
        static_cast<float>(std::clamp(rt_.terrain_raymarch_lod_min_octaves, 2, 8)));
    pc.tex_params = glm::vec4(
        static_cast<float>(std::clamp(rt_.terrain_raymarch_max_steps, 60, 300)),
        static_cast<float>(std::clamp(rt_.terrain_raymarch_shadow_steps, 16, 96)),
        static_cast<float>(std::clamp(rt_.terrain_raymarch_octaves, 4, 24)),
        static_cast<float>(std::clamp(rt_.terrain_raymarch_normal_octaves, 4, 32)));
    VkViewport vp{};
    vp.width = float(N); vp.height = float(N);
    vp.minDepth = 0.0f; vp.maxDepth = 1.0f;
    for (int l = 0; l < TerrainBakeState::kLevels; ++l) {
        bool any = false;
        for (const Rect& r : rects) any = any || r.level == l;
        if (!any) continue;
        VkRenderingAttachmentInfo att[3]{};
        for (int i = 0; i < 3; ++i) {
            att[i].sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
            att[i].imageView = S.layer_view[i][l];
            att[i].imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            att[i].loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
            att[i].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        }
        VkRenderingInfo ri{};
        ri.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
        ri.renderArea = { { 0, 0 }, { uint32_t(N), uint32_t(N) } };
        ri.layerCount = 1;
        ri.colorAttachmentCount = 3;
        ri.pColorAttachments = att;
        vkCmdBeginRendering(cmd, &ri);
        vkCmdSetViewport(cmd, 0, 1, &vp);
        const float texel = TerrainBakeState::kTexel0 * float(1 << l);
        for (const Rect& r : rects) {
            if (r.level != l) continue;
            vkCmdSetScissor(cmd, 0, 1, &r.sc);
            pc.model = glm::mat4(0.0f);
            pc.model[0] = glm::vec4(float(r.w0.x), float(r.w0.y), texel, S.t0 * float(1 << l));
            vkCmdPushConstants(cmd, pipeline_layout_, kPushConstantStages, 0, sizeof(PushConstants), &pc);
            vkCmdDraw(cmd, 3, 1, 0, 0);
        }
        vkCmdEndRendering(cmd);
    }
    barrier(VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    static const bool kLog = std::getenv("VIBE_LOG_TERRAIN_BAKE") != nullptr;
    if (kLog && crater_texels > 0) log::infof("[terrain-bake] crater: %u texels drawn again round it", crater_texels);
    if (kLog && (S.ready != was_ready || (frame_number_ % 60) == 0))
        log::infof("[terrain-bake] %u texels drawn this frame in %zu rectangles; %s (level 0 for ground nearer than %.2f m)",
                   S.texels_frame, rects.size(), S.ready ? "all levels whole" : "filling", double(S.t0));
}

} // namespace vibe
