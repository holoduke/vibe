// The ground and the objects on it, after the reference game (docs/ARCHITECTURE.md "The ground
// and the objects on it"). One block on the GPU (GroundFxGpuHead + the decals + the decal
// tiles: shaders/ground_fx.glsl) for:
//
//   * the PASTE (shaders/ground_paste.glsl, cube.frag): every mesh that stands on the ground
//     takes the ground's material by height in a zone above its contact line. This file only
//     tells the shader whether the baked ground texture may be read this frame and where.
//   * the far ground's colour of PLACED OBJECTS ("large scale detail", gfx_lsd_capture): the
//     placed meshes drawn from above into a colour + coverage picture over a window round the
//     camera, which the terrain pass lays on the far ground (shaders/ground_lsd.glsl).
//   * the DECALS (gfx_decals_frame; shaders/ground_decals.glsl in deferred_lighting.frag).
//
// Settings: rt_.ground_paste / ground_lsd / ground_decals (VIBE_NO_GROUND_PASTE=1, ...).
// VIBE_LOG_GROUND_FX=1 logs what is drawn and listed.

#include "engine/vk_engine.h"
#include "engine/vk_engine/internal.h"
#include "engine/vk_engine/state/ground_fx.h"
#include "engine/globe.h"
#include "engine/log.h"
#include "engine/terrain.h"
#include "engine/frustum.h"
#include "engine/vk_initializers.h"
#include "engine/vk_pipelines.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>

namespace vibe {

void VulkanEngine::init_ground_fx() {
    GroundFxState& G = *gfx_;
    G.log = std::getenv("VIBE_LOG_GROUND_FX") && std::atoi(std::getenv("VIBE_LOG_GROUND_FX")) != 0;
    G.buf_size = sizeof(GroundFxGpuHead) + sizeof(GfxDecalGpu) * GroundFxState::kMaxDecals +
                 sizeof(uint32_t) * 8ull * GroundFxState::kMaxTiles;
    {
        VkBufferCreateInfo bci{};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size = G.buf_size;
        bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        vk_check(vmaCreateBuffer(allocator_, &bci, &aci, &G.buf, &G.buf_alloc, nullptr), "ground fx buffer");
    }
    // The capture of placed objects: colour + coverage, and its depth (the highest surface wins).
    {
        VkImageCreateInfo ici{};
        ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ici.imageType = VK_IMAGE_TYPE_2D;
        ici.format = VK_FORMAT_R8G8B8A8_UNORM;
        ici.extent = { uint32_t(GroundFxState::kLsdSize), uint32_t(GroundFxState::kLsdSize), 1 };
        ici.mipLevels = 1;
        ici.arrayLayers = 1;
        ici.samples = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling = VK_IMAGE_TILING_OPTIMAL;
        ici.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        vk_check(vmaCreateImage(allocator_, &ici, &aci, &G.lsd_image, &G.lsd_alloc, nullptr), "lsd image");
        VkImageViewCreateInfo vci{};
        vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vci.image = G.lsd_image;
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.format = ici.format;
        vci.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        vk_check(vkCreateImageView(device_, &vci, nullptr, &G.lsd_view), "lsd view");
        ici.format = VK_FORMAT_D32_SFLOAT;
        ici.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
        vk_check(vmaCreateImage(allocator_, &ici, &aci, &G.lsd_depth, &G.lsd_depth_alloc, nullptr), "lsd depth");
        vci.image = G.lsd_depth;
        vci.format = ici.format;
        vci.subresourceRange = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1 };
        vk_check(vkCreateImageView(device_, &vci, nullptr, &G.lsd_depth_view), "lsd depth view");
        VkSamplerCreateInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        si.magFilter = si.minFilter = VK_FILTER_LINEAR;
        si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        vk_check(vkCreateSampler(device_, &si, nullptr, &G.lsd_sampler), "lsd sampler");
    }
    vkinit::one_time_submit(device_, graphics_queue_, graphics_queue_family_, [&](VkCommandBuffer cb) {
        vkCmdFillBuffer(cb, G.buf, 0, VK_WHOLE_SIZE, 0u);
        VkImageMemoryBarrier b{};
        b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = G.lsd_image;
        b.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        b.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
        VkClearColorValue z{};
        VkImageSubresourceRange r{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        vkCmdClearColorImage(cb, G.lsd_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &z, 1, &r);
        b.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        b.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
        VkMemoryBarrier mb{ VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT };
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
    });
    G.lsd_layout_ready = true;
    {
        VkDescriptorBufferInfo bi{ G.buf, 0, VK_WHOLE_SIZE };
        VkDescriptorImageInfo  ii{ G.lsd_sampler, G.lsd_view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        VkWriteDescriptorSet w[2]{};
        for (auto& x : w) { x.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; x.dstSet = scene_desc_set_; x.descriptorCount = 1; }
        w[0].dstBinding = 60; w[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w[0].pBufferInfo = &bi;
        w[1].dstBinding = 61; w[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w[1].pImageInfo = &ii;
        vkUpdateDescriptorSets(device_, 2, w, 0, nullptr);
    }
    init_rock_bake();               // (scene binding 63: the rock kit's bake and the breaker pattern - rock_bake.cpp)
    ground_fx_bind_deferred();
    log::infof("[ground_fx] ready: paste %d, far capture %d, decals %d", int(rt_.ground_paste), int(rt_.ground_lsd), int(rt_.ground_decals));
}

void VulkanEngine::ground_fx_bind_deferred() {
    if (!gfx_ || !gfx_->buf || !deferred_lighting_desc_set_) return;
    VkDescriptorBufferInfo bi{ gfx_->buf, 0, VK_WHOLE_SIZE };
    VkWriteDescriptorSet w{};
    w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w.dstSet = deferred_lighting_desc_set_; w.dstBinding = 30; w.descriptorCount = 1;
    w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w.pBufferInfo = &bi;
    vkUpdateDescriptorSets(device_, 1, &w, 0, nullptr);
}

void VulkanEngine::destroy_ground_fx() {
    destroy_rock_bake();
    if (!gfx_) return;
    GroundFxState& G = *gfx_;
    if (G.lsd_pipeline) vkDestroyPipeline(device_, G.lsd_pipeline, nullptr);
    if (G.lsd_layout) vkDestroyPipelineLayout(device_, G.lsd_layout, nullptr);
    if (G.lsd_sampler) vkDestroySampler(device_, G.lsd_sampler, nullptr);
    if (G.lsd_view) vkDestroyImageView(device_, G.lsd_view, nullptr);
    if (G.lsd_depth_view) vkDestroyImageView(device_, G.lsd_depth_view, nullptr);
    if (G.lsd_image) vmaDestroyImage(allocator_, G.lsd_image, G.lsd_alloc);
    if (G.lsd_depth) vmaDestroyImage(allocator_, G.lsd_depth, G.lsd_depth_alloc);
    if (G.buf) vmaDestroyBuffer(allocator_, G.buf, G.buf_alloc);
    G = GroundFxState{};
}

// Once a frame, after the baked ground texture has been brought up to date (frame.cpp) and
// before the world pass: the capture, the decals' list, then the block.
void VulkanEngine::ground_fx_frame(VkCommandBuffer cmd) {
    if (!gfx_ || !gfx_->buf) return;
    GroundFxState& G = *gfx_;
    // (VIBE_LOG_GROUND_FX=1, once: the most open conifer stands of the home tile - 20 m cells with
    //  3 to 6 conifers and no broadleaf tree - where to look at the needles and cones)
    static bool stands_logged = false;
    if (G.log && !stands_logged && !veg_inst_cpu_.empty()) {
        stands_logged = true;
        std::vector<int> con(100 * 100, 0), brd(100 * 100, 0);
        glm::vec2 sum[100 * 100] = {};
        for (const VegInstance& vi : veg_inst_cpu_) {
            const size_t a = size_t(vi.rot_arch_phase.y + 0.5f);
            if (a >= veg_arch_.size() || vi.pos_scale.w <= 0.0f) continue;
            const int cx = int((vi.pos_scale.x + 1000.0f) / 20.0f), cz = int((vi.pos_scale.z + 1000.0f) / 20.0f);
            if (cx < 0 || cz < 0 || cx >= 100 || cz >= 100) continue;
            const int sp = veg_arch_[a].species;
            if (sp == 0) { ++con[cz * 100 + cx]; sum[cz * 100 + cx] += glm::vec2(vi.pos_scale.x, vi.pos_scale.z); }
            else if (sp == 1) ++brd[cz * 100 + cx];
        }
        int n = 0;
        for (int i = 0; i < 100 * 100 && n < 6; ++i)
            if (con[i] >= 3 && con[i] <= 6 && brd[i] == 0) {
                const glm::vec2 c = sum[i] / float(con[i]);
                const float gy = sample_terrain_height(c.x, c.y);
                if (gy < rt_.water_level + 3.0f) continue;
                log::infof("[ground_fx] open conifer stand: %d conifers round (%.1f, %.1f), ground %.1f m", con[i], c.x, c.y, gy);
                ++n;
            }
    }
    GroundFxGpuHead h{};
    // The paste: the baked ground texture where it is whole this frame (its t0 carries the first
    // level in use in the eights - terrain_bake_t0).
    {
        const float tx = terrain_bake_ready() ? terrain_bake_t0() : 0.0f;
        const float lmin = std::floor(tx / 8.0f);
        h.paste = glm::vec4(rt_.ground_paste ? 1.0f : 0.0f, tx > 0.0f ? tx - 8.0f * lmin : 0.0f, lmin, 0.22f);
    }
    gfx_lsd_capture(cmd);
    h.lsd = glm::vec4(0.0f);
    if (rt_.ground_lsd && G.lsd_valid) {
        const float span = float(GroundFxState::kLsdSize) * GroundFxState::kLsdTexel;
        h.lsd = glm::vec4(1.0f, float(G.lsd_c.x) * GroundFxState::kLsdTexel - 0.5f * span,
                          float(G.lsd_c.y) * GroundFxState::kLsdTexel - 0.5f * span, span);
    }
    // (.w: VIBE_TEST_LSD=1 shows what the capture covers, magenta, at every distance)
    static const bool test_lsd = std::getenv("VIBE_TEST_LSD") && std::atoi(std::getenv("VIBE_TEST_LSD")) != 0;
    // (.z: VIBE_TEST_DEBRIS=1 draws the small debris magenta - scatter.frag)
    static const bool test_debris = std::getenv("VIBE_TEST_DEBRIS") && std::atoi(std::getenv("VIBE_TEST_DEBRIS")) != 0;
    h.lsd_fade = glm::vec4(350.0f, 900.0f, test_debris ? 1.0f : 0.0f, test_lsd ? 1.0f : 0.0f);
    gfx_decals_frame();
    const uint32_t nd = uint32_t(std::min<size_t>(G.frame_list.size(), GroundFxState::kMaxDecals));
    h.dec = glm::vec4(float(nd), float(G.tiles_x), float(G.tiles_y), (rt_.ground_decals && nd > 0) ? 1.0f : 0.0f);
    h.rock = rock_bake_block();

    // (what the frame before read is done before this is written: a barrier, then the copies)
    VkMemoryBarrier mb{ VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT };
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
    vkCmdUpdateBuffer(cmd, G.buf, 0, sizeof(h), &h);
    if (nd > 0) {
        vkCmdUpdateBuffer(cmd, G.buf, sizeof(GroundFxGpuHead), sizeof(GfxDecalGpu) * nd, G.frame_list.data());
        const VkDeviceSize toff = sizeof(GroundFxGpuHead) + sizeof(GfxDecalGpu) * GroundFxState::kMaxDecals;
        const size_t words = std::min<size_t>(G.tiles.size(), 8ull * GroundFxState::kMaxTiles);
        for (size_t o = 0; o < words; o += 16384) {          // (vkCmdUpdateBuffer: 64 KB at most)
            const size_t n = std::min<size_t>(16384, words - o);
            vkCmdUpdateBuffer(cmd, G.buf, toff + o * 4, n * 4, G.tiles.data() + o);
        }
    }
    mb = { VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT };
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 1, &mb, 0, nullptr, 0, nullptr);
}

// ---- The far ground's colour of placed objects: the top-down capture ----------------------
// A window of kLsdSize x kLsdSize texels of kLsdTexel metres round the camera, addressed by
// the world position itself (a texel's place in the picture is its world index modulo the
// size, as the baked ground texture's), moved in steps of kLsdSnap texels. What comes into the
// window as the camera moves is drawn (one or two strips); when the set of placed parts changes
// (a site, a highway chunk or rock formations stream in or out, the engine re-centres) the
// whole window is drawn again - at most once a second. Never the whole world: only the parts
// standing in the window, from what is loaded near the camera anyway (mesh_parts_now).
void VulkanEngine::gfx_lsd_capture(VkCommandBuffer cmd) {
    GroundFxState& G = *gfx_;
    G.lsd_drawn = 0;
    if (!rt_.ground_lsd) { G.lsd_valid = false; return; }
    if (!G.lsd_pipeline) {
        if (!pipeline_layout_) return;
        const std::string sd = VIBE_SHADER_DIR;
        VkShaderModule v = vkpipe::load_shader_module(device_, sd + "/lsd_capture.vert.spv");
        VkShaderModule f = vkpipe::load_shader_module(device_, sd + "/lsd_capture.frag.spv");
        if (!v || !f) { log::warnf("[ground_fx] lsd_capture shaders missing - no far capture"); rt_.ground_lsd = false; return; }
        vkpipe::GraphicsPipelineConfig cfg{};
        cfg.vert = v; cfg.frag = f;
        cfg.layout = pipeline_layout_;
        cfg.color_formats = { VK_FORMAT_R8G8B8A8_UNORM };
        cfg.depth_format = VK_FORMAT_D32_SFLOAT;
        VkVertexInputBindingDescription vb{};
        vb.binding = 0; vb.stride = sizeof(Vertex); vb.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        cfg.vbindings = { vb };
        VkVertexInputAttributeDescription a0{}, a1{}, a2{};
        a0.location = 0; a0.format = VK_FORMAT_R32G32B32_SFLOAT; a0.offset = offsetof(Vertex, position);
        a1.location = 1; a1.format = VK_FORMAT_R32G32B32_SFLOAT; a1.offset = offsetof(Vertex, normal);
        a2.location = 2; a2.format = VK_FORMAT_R32G32_SFLOAT;    a2.offset = offsetof(Vertex, uv);
        cfg.vattrs = { a0, a1, a2 };
        cfg.cull = VK_CULL_MODE_NONE;
        cfg.depth_test = true; cfg.depth_write = true;
        cfg.depth_compare = VK_COMPARE_OP_LESS;
        G.lsd_pipeline = vkpipe::build_graphics_pipeline(device_, cfg);
        vkDestroyShaderModule(device_, v, nullptr);
        vkDestroyShaderModule(device_, f, nullptr);
        if (!G.lsd_pipeline) { rt_.ground_lsd = false; return; }
    }
    const int N = GroundFxState::kLsdSize, S = GroundFxState::kLsdSnap;
    const float T = GroundFxState::kLsdTexel;
    const glm::vec3 eye = current_frame_view_.eye_pos;
    const glm::ivec2 want(int(std::floor(eye.x / T / float(S) + 0.5f)) * S, int(std::floor(eye.z / T / float(S) + 0.5f)) * S);

    // The parts standing on the land now (not the cars, not glass, not what glows), and a hash of them.
    const auto& parts_all = mesh_parts_now();
    std::vector<const CastleMeshGpu*> parts;
    parts.reserve(parts_all.size());
    uint64_t sig = 1469598103934665603ull;
    auto mixh = [&](uint64_t x) { sig = (sig ^ x) * 1099511628211ull; };
    for (const CastleMeshGpu* gp : parts_all) {
        const bool car = !jeep_draw_.empty() && gp >= jeep_draw_.data() && gp < jeep_draw_.data() + jeep_draw_.size();
        if (car || gp->tex_albedo == -3 || gp->emissive.a > 0.5f || !gp->mesh.vertex_buffer) continue;
        parts.push_back(gp);
        mixh(uint64_t(reinterpret_cast<uintptr_t>(gp->mesh.vertex_buffer)));
        mixh(uint64_t(int64_t(std::floor(gp->bmin.x))) * 31ull + uint64_t(int64_t(std::floor(gp->bmin.z))));
    }
    if (G.lsd_cool > 0) --G.lsd_cool;
    std::vector<glm::ivec4> rects;                   // texel index boxes (x0, z0, x1, z1) to draw
    const glm::ivec2 d = want - G.lsd_c;
    if (!G.lsd_valid || std::abs(d.x) >= N / 2 || std::abs(d.y) >= N / 2 || (sig != G.lsd_parts_sig && G.lsd_cool == 0)) {
        rects.push_back({ want.x - N / 2, want.y - N / 2, want.x + N / 2, want.y + N / 2 });
        G.lsd_parts_sig = sig;
        G.lsd_cool = 60;
    } else if (d != glm::ivec2(0)) {
        // (the strips that came in: along x, then along z over the rest)
        if (d.x > 0) rects.push_back({ G.lsd_c.x + N / 2, want.y - N / 2, want.x + N / 2, want.y + N / 2 });
        if (d.x < 0) rects.push_back({ want.x - N / 2, want.y - N / 2, G.lsd_c.x - N / 2, want.y + N / 2 });
        const int xa = std::max(want.x, G.lsd_c.x) - N / 2, xb = std::min(want.x, G.lsd_c.x) + N / 2;
        if (d.y > 0) rects.push_back({ xa, G.lsd_c.y + N / 2, xb, want.y + N / 2 });
        if (d.y < 0) rects.push_back({ xa, want.y - N / 2, xb, G.lsd_c.y - N / 2 });
    }
    G.lsd_c = want;
    G.lsd_valid = true;
    if (rects.empty()) return;

    // Split at the picture's wrap: pieces whose texels are contiguous in the picture.
    struct Piece { glm::ivec4 r; glm::ivec2 o; };
    std::vector<Piece> pieces;
    auto wrapi = [&](int i) { return ((i % N) + N) % N; };
    for (const glm::ivec4& r : rects) {
        if (r.z <= r.x || r.w <= r.y) continue;
        for (int x0 = r.x; x0 < r.z;) {
            const int x1 = std::min(r.z, x0 + (N - wrapi(x0)));
            for (int z0 = r.y; z0 < r.w;) {
                const int z1 = std::min(r.w, z0 + (N - wrapi(z0)));
                pieces.push_back({ { x0, z0, x1, z1 }, { wrapi(x0), wrapi(z0) } });
                z0 = z1;
            }
            x0 = x1;
        }
    }
    vkinit::transition_image(cmd, G.lsd_image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    {
        VkImageMemoryBarrier b{};
        b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        b.oldLayout = G.lsd_depth_ready ? VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL : VK_IMAGE_LAYOUT_UNDEFINED;
        b.newLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = G.lsd_depth;
        b.subresourceRange = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1 };
        b.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        b.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT, VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
        G.lsd_depth_ready = true;
    }
    VkRenderingAttachmentInfo ca = vkinit::color_attachment_info(G.lsd_view, nullptr, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    VkRenderingAttachmentInfo da{};
    da.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    da.imageView = G.lsd_depth_view;
    da.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    da.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    da.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    VkRenderingInfo ri{};
    ri.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    ri.renderArea = { { 0, 0 }, { uint32_t(N), uint32_t(N) } };
    ri.layerCount = 1;
    ri.colorAttachmentCount = 1; ri.pColorAttachments = &ca;
    ri.pDepthAttachment = &da;
    vkCmdBeginRendering(cmd, &ri);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, G.lsd_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout_, 0, 1, &scene_desc_set_, 0, nullptr);
    const float ylo = -600.0f, yhi = 4000.0f;
    VkDeviceSize off = 0;
    for (const Piece& p : pieces) {
        const glm::ivec2 sz(p.r.z - p.r.x, p.r.w - p.r.y);
        VkViewport vp{ float(p.o.x), float(p.o.y), float(sz.x), float(sz.y), 0.0f, 1.0f };
        VkRect2D sc{ { p.o.x, p.o.y }, { uint32_t(sz.x), uint32_t(sz.y) } };
        vkCmdSetViewport(cmd, 0, 1, &vp);
        vkCmdSetScissor(cmd, 0, 1, &sc);
        VkClearAttachment cl[2]{};
        cl[0].aspectMask = VK_IMAGE_ASPECT_COLOR_BIT; cl[0].colorAttachment = 0; cl[0].clearValue.color = { { 0.0f, 0.0f, 0.0f, 0.0f } };
        cl[1].aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT; cl[1].clearValue.depthStencil = { 1.0f, 0 };
        VkClearRect cr{ sc, 0, 1 };
        vkCmdClearAttachments(cmd, 2, cl, 1, &cr);
        const float x0 = float(p.r.x) * T, x1 = float(p.r.z) * T, z0 = float(p.r.y) * T, z1 = float(p.r.w) * T;
        glm::mat4 M(0.0f);                 // (columns) world -> clip, straight down
        M[0][0] = 2.0f / (x1 - x0);
        M[2][1] = 2.0f / (z1 - z0);
        M[1][2] = -1.0f / (yhi - ylo);
        M[3] = glm::vec4(-(x1 + x0) / (x1 - x0), -(z1 + z0) / (z1 - z0), yhi / (yhi - ylo), 1.0f);
        for (const CastleMeshGpu* gp : parts) {
            const CastleMeshGpu& g = *gp;
            if (g.bmax.x < x0 || g.bmin.x > x1 || g.bmax.z < z0 || g.bmin.z > z1) continue;
            PushConstants pc{};
            pc.mvp = M * g.xf;
            pc.model = g.xf;
            pc.prev_mvp = pc.mvp;
            pc.color = glm::vec4(rt_.textures_enabled ? g.tint : g.fallback, 1.0f);
            pc.tex_params = rt_.textures_enabled ? glm::vec4(float(g.tex_albedo), float(g.tex_normal), g.uv_scale, 0.0f) : glm::vec4(-1.0f, -1.0f, 1.0f, 0.0f);
            pc.grass_params = glm::vec4(0.0f, 0.0f, 0.0f, -1.0f);
            vkCmdPushConstants(cmd, pipeline_layout_, kPushConstantStages, 0, sizeof(PushConstants), &pc);
            vkCmdBindVertexBuffers(cmd, 0, 1, &g.mesh.vertex_buffer, &off);
            vkCmdBindIndexBuffer(cmd, g.mesh.index_buffer, 0, VK_INDEX_TYPE_UINT32);
            vkCmdDrawIndexed(cmd, g.mesh.index_count, 1, 0, 0, 0);
            ++G.lsd_drawn;
        }
    }
    vkCmdEndRendering(cmd);
    vkinit::transition_image(cmd, G.lsd_image, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    if (G.log) log::infof("[ground_fx] capture: %zu rects, %zu pieces, %u draws of %zu parts; window centre (%.0f, %.0f)",
                          rects.size(), pieces.size(), G.lsd_drawn, parts.size(), double(want.x) * T, double(want.y) * T);
}
// ---- The decals -----------------------------------------------------------------------------
namespace {
float gfx_hash(float a, float b, float c) {
    const float h = std::sin(a * 12.9898f + b * 78.233f + c * 37.719f) * 43758.5453f;
    return h - std::floor(h);
}
enum GfxKind { kMoss = 1, kDamp = 2, kDirt = 3, kOil = 4, kLitter = 5, kCrack = 6, kMud = 7, kScorch = 8, kHole = 9, kOmni = 16 };
}

// The authored set, by rule, round the camera (made again when the camera enters another
// 64 m cell; within 220 m - streamed with the places they belong to):
//   * the castle's walls (brushes with the wall / keep masonry, home's chart only): moss and
//     damp streaks along the foot of each long face, leaf litter on the ground at the foot,
//     now and then a crack on the keep's walls;
//   * the stone bridge's two ends: moss and damp round the abutments (their faces are not
//     known here: boxes that take any face);
//   * the service station's forecourt: dirt and oil stains; the castle car park: dirt, oil and
//     mud at its entrance (the nearest thing to a field gateway the land has).
// Then every frame: the gun's marks on the still world (scorch / hole, combat.cpp decals_),
// the visible ones nearest first (at most kMaxDecals), and their screen rectangles binned
// into 32-pixel tiles.
void VulkanEngine::gfx_decals_frame() {
    GroundFxState& G = *gfx_;
    G.frame_list.clear();
    G.tiles_x = G.tiles_y = 0;
    if (!rt_.ground_decals) return;
    const glm::vec3 eye = current_frame_view_.eye_pos;
    using D = GroundFxState::Decal;
    const glm::ivec2 cell(int(std::floor(eye.x / 64.0f)), int(std::floor(eye.z / 64.0f)));
    const bool home = globe::current().home;
    if (cell != G.placed_cell) {
        G.placed_cell = cell;
        G.placed.clear();
        const float R = 220.0f;
        auto in_reach = [&](float x, float z) { return std::abs(x - eye.x) < R && std::abs(z - eye.z) < R; };
        auto add = [&](glm::vec3 c, int kind, glm::vec3 ax, glm::vec3 ay, glm::vec3 az, float strength, float ratio) {
            D d{}; d.c = c; d.kind = kind; d.ax = ax; d.ay = ay; d.az = az; d.strength = strength;
            d.seed = gfx_hash(c.x, c.y, c.z); d.ratio = ratio;
            G.placed.push_back(d);
        };
        if (home) {
            for (const auto& b : world_.brushes) {
                if (b.tex_albedo != 1 && b.tex_albedo != 4) continue;
                if (!in_reach(b.center.x, b.center.z) || b.size.y < 2.0f) continue;
                const int la = b.size.x >= b.size.z ? 0 : 2, sa = 2 - la;      // long axis, short axis
                const float len = b.size[la];
                if (len < 3.0f) continue;
                for (int side = -1; side <= 1; side += 2) {
                    glm::vec3 n(0.0f); n[sa] = float(side);
                    glm::vec3 along(0.0f); along[la] = 1.0f;
                    const int steps = int(len / 2.6f);
                    for (int i = 0; i < steps; ++i) {
                        const float s = (float(i) + 0.5f) / float(steps) - 0.5f;
                        glm::vec3 p = b.center + along * (s * len) + n * (0.5f * b.size[sa]);
                        const float h = gfx_hash(p.x, p.z, float(side));
                        const float gy = sample_terrain_height(p.x + n.x * 0.6f, p.z + n.z * 0.6f);
                        const float bot = b.center.y - 0.5f * b.size.y, top = b.center.y + 0.5f * b.size.y;
                        if (gy < bot - 0.3f || gy > top - 1.0f) continue;          // (the face does not meet the ground here)
                        // a band of moss all along the foot (the boxes overlap; the shader closes it at the
                        // ground and lets it climb the joints above), damp darkening over most of it
                        const float half_step = 0.68f * len / float(steps);
                        const float mh = 0.65f + 0.30f * gfx_hash(p.z, p.x, 3.0f);
                        add(glm::vec3(p.x, gy + mh + 0.05f, p.z), kMoss, along * half_step, n * 0.45f, glm::vec3(0.0f, mh + 0.15f, 0.0f), 0.95f, 0.15f);
                        if (h < 0.75f) {
                            const float dh = 0.8f + 0.5f * gfx_hash(p.x, 2.0f, p.z);
                            add(glm::vec3(p.x, gy + dh, p.z), kDamp, along * half_step * 1.1f, n * 0.45f, glm::vec3(0.0f, dh, 0.0f), 0.85f, 0.85f);
                        }
                        if (gfx_hash(p.x, 7.0f, p.z) < 0.55f)
                            add(glm::vec3(p.x, gy, p.z) + n * 0.7f, kLitter, along * 1.4f, glm::vec3(0.0f, 0.5f, 0.0f), n * 0.8f, 0.9f, 0.3f);
                        if (b.tex_albedo == 4 && gfx_hash(p.z, 9.0f, p.x) < 0.25f)
                            add(glm::vec3(p.x, gy + 1.5f + 2.0f * h, p.z), kCrack, along * 0.9f, n * 0.3f, glm::vec3(0.0f, 1.1f, 0.0f), 0.8f, 1.0f);
                    }
                }
            }
            // the stone bridge's ends (vk_engine/bridges.cpp: from (22, 277) to (74, 327))
            const glm::vec2 ends[2] = { { 22.0f, 277.0f }, { 74.0f, 327.0f } };
            const glm::vec2 dir = glm::normalize(ends[1] - ends[0]);
            for (int e = 0; e < 2; ++e) {
                if (!in_reach(ends[e].x, ends[e].y)) continue;
                const glm::vec2 c = ends[e] + dir * (e == 0 ? 5.0f : -5.0f);
                const float gy = sample_terrain_height(c.x, c.y);
                for (int k = 0; k < 3; ++k) {
                    const glm::vec2 q = c + glm::vec2(-dir.y, dir.x) * (float(k - 1) * 3.0f);
                    add(glm::vec3(q.x, gy + 0.8f, q.y), (k == 1 ? kDamp : kMoss) | kOmni,
                        glm::vec3(dir.x, 0.0f, dir.y) * 3.0f, glm::vec3(-dir.y, 0.0f, dir.x) * 2.0f, glm::vec3(0.0f, 1.2f, 0.0f), 0.6f, 0.2f);
                }
            }
            // the castle car park (terrain.h kCastlePark): dirt, oil, mud at its entrance
            if (in_reach(0.5f * (kCastlePark[0] + kCastlePark[2]), 0.5f * (kCastlePark[1] + kCastlePark[3]))) {
                for (int i = 0; i < 14; ++i) {
                    const float x = kCastlePark[0] + 2.0f + (kCastlePark[2] - kCastlePark[0] - 4.0f) * gfx_hash(float(i), 1.0f, 2.0f);
                    const float z = kCastlePark[1] + 2.0f + (kCastlePark[3] - kCastlePark[1] - 4.0f) * gfx_hash(float(i), 3.0f, 4.0f);
                    const float gy = sample_terrain_height(x, z);
                    const float r = 0.8f + 1.4f * gfx_hash(float(i), 5.0f, 6.0f);
                    add(glm::vec3(x, gy, z), i % 3 == 0 ? kOil : kDirt, glm::vec3(r, 0.0f, 0.0f), glm::vec3(0.0f, 0.6f, 0.0f), glm::vec3(0.0f, 0.0f, r * 0.8f), 0.8f, 0.2f);
                }
                for (int i = 0; i < 3; ++i) {
                    const float x = kCastlePark[0] + 1.5f + 3.5f * float(i), z = kCastlePark[3] - 1.0f;
                    add(glm::vec3(x, sample_terrain_height(x, z), z), kMud, glm::vec3(2.2f, 0.0f, 0.0f), glm::vec3(0.0f, 0.6f, 0.0f), glm::vec3(0.0f, 0.0f, 2.5f), 0.9f, 0.0f);
                }
            }
            // the service station's forecourt (terrain.h kStationRect, at kStationH)
            if (in_reach(kStationX, kStationZ)) {
                for (int i = 0; i < 28; ++i) {
                    const float x = kStationRect[0] + 4.0f + (kStationRect[2] - kStationRect[0] - 8.0f) * gfx_hash(float(i), 11.0f, 2.0f);
                    const float z = kStationRect[1] + 4.0f + (kStationRect[3] - kStationRect[1] - 8.0f) * gfx_hash(float(i), 13.0f, 4.0f);
                    const float r = 0.6f + 1.6f * gfx_hash(float(i), 15.0f, 6.0f);
                    add(glm::vec3(x, kStationH, z), i % 2 == 0 ? kOil : kDirt, glm::vec3(r, 0.0f, 0.0f), glm::vec3(0.0f, 0.5f, 0.0f), glm::vec3(0.0f, 0.0f, r * (0.6f + 0.6f * gfx_hash(float(i), 1.0f, 1.0f))), 0.85f, 0.4f);
                }
            }
        }
        if (G.log) log::infof("[ground_fx] decals: %zu placed round cell (%d, %d)", G.placed.size(), cell.x, cell.y);
    }
    // This frame's: the placed ones and the gun's marks, visible, nearest first.
    struct Cand { float d2; GfxDecalGpu g; };
    std::vector<Cand> cand;
    cand.reserve(G.placed.size() + decals_.size());
    const Frustum& fr = current_frame_view_.frustum;
    auto consider = [&](const glm::vec3& c, int kind, const glm::vec3& ax, const glm::vec3& ay, const glm::vec3& az, float strength, float seed, float ratio, float maxd) {
        const glm::vec3 r = glm::abs(ax) + glm::abs(ay) + glm::abs(az);
        const float d2 = glm::dot(c - eye, c - eye);
        if (d2 > maxd * maxd || !aabb_visible(fr, c - r, c + r)) return;
        cand.push_back({ d2, { glm::vec4(c, float(kind)), glm::vec4(ax, strength), glm::vec4(ay, seed), glm::vec4(az, ratio) } });
    };
    for (const D& d : G.placed) consider(d.c, d.kind, d.ax, d.ay, d.az, d.strength, d.seed, d.ratio, 160.0f);
    for (const auto& d : decals_) {
        if (d.parent_body_id != 0 || d.age < 0.5f) continue;
        const float life = 1.0f - d.age / d.ttl;
        const float nl = glm::length(d.normal);
        if (life <= 0.0f || nl < 1e-5f) continue;
        const glm::vec3 n = d.normal / nl;
        const glm::vec3 ref = std::abs(n.y) < 0.95f ? glm::vec3(0, 1, 0) : glm::vec3(1, 0, 0);
        const glm::vec3 x = glm::normalize(glm::cross(ref, n)), z = glm::cross(n, x);
        const float rad = d.size * 1.6f;
        consider(d.pos, kScorch, x * rad, n * 0.14f, z * rad, std::min(1.0f, (d.age - 0.5f) * 2.0f) * std::min(1.0f, life * 4.0f),
                 gfx_hash(d.pos.x, d.pos.y, d.pos.z), 0.6f, 200.0f);
        consider(d.pos, kHole, x * (rad * 0.18f), n * 0.08f, z * (rad * 0.18f), std::min(1.0f, life * 4.0f), 0.5f, 1.0f, 60.0f);
    }
    if (cand.empty()) return;
    std::sort(cand.begin(), cand.end(), [](const Cand& a, const Cand& b) { return a.d2 < b.d2; });
    if (cand.size() > size_t(GroundFxState::kMaxDecals)) cand.resize(GroundFxState::kMaxDecals);
    // Tiles: each decal's box on the screen (its eight corners; the whole screen if one is behind the eye).
    const int W = int(render_extent_.width), H = int(render_extent_.height), K = GroundFxState::kTile;
    G.tiles_x = (W + K - 1) / K; G.tiles_y = (H + K - 1) / K;
    if (G.tiles_x * G.tiles_y > GroundFxState::kMaxTiles) { G.tiles_x = G.tiles_y = 0; return; }
    G.tiles.assign(size_t(G.tiles_x) * G.tiles_y * 8, 0u);
    const glm::mat4& vp = current_frame_view_.vp;
    for (size_t i = 0; i < cand.size(); ++i) {
        const GfxDecalGpu& g = cand[i].g;
        glm::vec2 lo(1e9f), hi(-1e9f);
        bool behind = false;
        for (int k = 0; k < 8; ++k) {
            const glm::vec3 p = glm::vec3(g.c) + glm::vec3(g.ax) * ((k & 1) ? 1.0f : -1.0f) + glm::vec3(g.ay) * ((k & 2) ? 1.0f : -1.0f) + glm::vec3(g.az) * ((k & 4) ? 1.0f : -1.0f);
            const glm::vec4 q = vp * glm::vec4(p, 1.0f);
            if (q.w < 0.05f) { behind = true; break; }
            const glm::vec2 s = (glm::vec2(q) / q.w * 0.5f + 0.5f) * glm::vec2(float(W), float(H));
            lo = glm::min(lo, s); hi = glm::max(hi, s);
        }
        int tx0 = 0, ty0 = 0, tx1 = G.tiles_x - 1, ty1 = G.tiles_y - 1;
        if (!behind) {
            tx0 = std::max(0, int(std::floor(lo.x)) / K - 1); ty0 = std::max(0, int(std::floor(lo.y)) / K - 1);
            tx1 = std::min(G.tiles_x - 1, int(std::floor(hi.x)) / K + 1); ty1 = std::min(G.tiles_y - 1, int(std::floor(hi.y)) / K + 1);
            if (hi.x < 0.0f || hi.y < 0.0f || lo.x > float(W) || lo.y > float(H)) continue;
        }
        const uint32_t word = uint32_t(i >> 5), bit = 1u << (i & 31);
        for (int ty = ty0; ty <= ty1; ++ty)
            for (int tx = tx0; tx <= tx1; ++tx) G.tiles[(size_t(ty) * G.tiles_x + tx) * 8 + word] |= bit;
    }
    G.frame_list.reserve(cand.size());
    for (const Cand& c : cand) G.frame_list.push_back(c.g);
}

} // namespace vibe
