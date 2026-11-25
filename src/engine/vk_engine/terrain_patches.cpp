// The patch terrain: the visible ground found by rasterising a mesh instead of marching a ray.
//
// The ray march (shaders/terrain_raymarch.frag) finds, for every pixel, where its ray meets the
// height formula - dozens of evaluations of the formula a pixel. This renderer finds the same
// point the way the reference game draws its ground (read from its compiled shaders):
//
//   * ONE grid patch of (N + 3)^2 vertices - N quads a side, the outer ring a skirt that hangs
//     down from the edge - drawn once per instance. An instance is a patch's corner and the
//     spacing of its vertices; the spacing IS the level of detail.
//   * A quadtree on the CPU picks the instances every frame: a node near the camera is split
//     into four of half the spacing, a node out of the view is dropped (frustum planes against
//     the node's box, whose lowest and highest ground are kept from the CPU's own ground
//     function). Seams between levels are hidden by the skirts alone.
//   * The vertex shader (shaders/terrain_patch.vert) places each vertex at the height formula's
//     value there - the formula itself, not a copy.
//
// What is done with the mesh is this engine's own: it is rasterised depth-only, before the
// world pass, into a depth picture of its own, and the terrain's full-screen pass reads this
// pixel's distance there in place of marching (terrain_rm_heightfield.glsl rm_patch_hit) - one
// evaluation of the formula a pixel, to put the point exactly on the surface. Everything after
// that - materials, roads, shadows, the water, what stands in front, the far march beyond the
// patches' reach - is the code it always was. Drawing the patches straight into the world pass
// with the full shader would shade the slivers a level ground is at a distance several times
// over (every small triangle shades whole 2 x 2 blocks), and the world pass's depth is too
// coarse beyond a kilometre to sort far hills; a depth picture with reversed depth has neither
// problem.
//
// Per frame (frame.cpp): update_terrain_patches() before the scene constants are written,
// render_terrain_patch_depth() where the low-resolution terrain would be drawn.
// Setting: rt_.terrain_patches (VIBE_TERRAIN_PATCHES=0|1). VIBE_LOG_TERRAIN_PATCHES=1 logs
// what was drawn. See docs/ARCHITECTURE.md "Patch terrain".

#include "engine/vk_engine.h"
#include "engine/vk_engine/internal.h"
#include "engine/vk_engine/state/terrain_patches.h"
#include "engine/vk_engine/state/terrain_raymarch.h"

#include "engine/frustum.h"
#include "engine/globe.h"
#include "engine/log.h"
#include "engine/terrain.h"
#include "engine/vk_initializers.h"
#include "engine/vk_pipelines.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>

namespace vibe {


// The setting is on and the renderer can run. (The low-resolution terrain and the march
// pre-pass are off then: tr_use_lowres, tr_march_prepass_on.)
bool VulkanEngine::terrain_patches_on() const {
    return rt_.terrain_patches && rt_.terrain_raymarch_enabled && tp_ && tp_->pipeline != VK_NULL_HANDLE &&
           tp_->inst_map != nullptr;
}

void VulkanEngine::init_terrain_patches() {
    // The grid: (kN + 3)^2 vertices in rows, two triangles a quad.
    {
        const uint32_t row = uint32_t(TerrainPatchState::kN) + 3u;
        std::vector<uint32_t> idx;
        idx.reserve(size_t(row - 1) * (row - 1) * 6);
        for (uint32_t j = 0; j + 1 < row; ++j)
            for (uint32_t i = 0; i + 1 < row; ++i) {
                const uint32_t a = j * row + i, b = a + 1, c = a + row, d = c + 1;
                idx.insert(idx.end(), { a, c, b, b, c, d });
            }
        tp_->index_count = uint32_t(idx.size());
        VkBufferCreateInfo bci{};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size = idx.size() * sizeof(uint32_t);
        bci.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
        aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
        VmaAllocationInfo ai{};
        vk_check(vmaCreateBuffer(allocator_, &bci, &aci, &tp_->index_buf, &tp_->index_alloc, &ai), "terrain patch index buffer");
        std::memcpy(ai.pMappedData, idx.data(), idx.size() * sizeof(uint32_t));
        vmaFlushAllocation(allocator_, tp_->index_alloc, 0, VK_WHOLE_SIZE);
    }
    {
        VkBufferCreateInfo bci{};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size = VkDeviceSize(kFrameOverlap) * TerrainPatchState::kMaxPatches * sizeof(TerrainPatchState::Instance);
        bci.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
        aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
        VmaAllocationInfo ai{};
        vk_check(vmaCreateBuffer(allocator_, &bci, &aci, &tp_->inst_buf, &tp_->inst_alloc, &ai), "terrain patch instance buffer");
        tp_->inst_map = ai.pMappedData;
    }
    tp_->vert = vkpipe::load_shader_module(device_, std::string(VIBE_SHADER_DIR) + "/terrain_patch.vert.spv");
    if (tp_->vert) {
        VkVertexInputBindingDescription b0{};
        b0.binding = 0; b0.stride = sizeof(TerrainPatchState::Instance);
        b0.inputRate = VK_VERTEX_INPUT_RATE_INSTANCE;
        VkVertexInputAttributeDescription a0{}, a1{};
        a0.location = 0; a0.binding = 0; a0.format = VK_FORMAT_R32G32B32A32_SFLOAT; a0.offset = 0;
        a1.location = 1; a1.binding = 0; a1.format = VK_FORMAT_R32G32B32A32_SFLOAT; a1.offset = 16;
        vkpipe::GraphicsPipelineConfig cfg{};
        cfg.vert = tp_->vert;
        cfg.frag = VK_NULL_HANDLE;                  // depth only
        cfg.layout = pipeline_layout_;
        cfg.color_attachment_count = 0;
        cfg.depth_format = depth_format_;
        // Only the upper side (and a skirt's outer side) is drawn: an eye that is under the surface
        // for a moment - crouched in a hollow of rough ground - then looks through it at the ground
        // beyond, not at a ceiling. (The grid's upper side is the pipeline's default front face.)
        cfg.cull = VK_CULL_MODE_BACK_BIT;
        cfg.depth_test = true;
        cfg.depth_write = true;
        cfg.depth_compare = VK_COMPARE_OP_GREATER;  // reversed depth: see render_terrain_patch_depth
        cfg.vbindings = { b0 };
        cfg.vattrs = { a0, a1 };
        tp_->pipeline = vkpipe::build_graphics_pipeline(device_, cfg);
    }
    log::infof("[terrain] patches: grid %d x %d quads + skirt, %u indices, %s", TerrainPatchState::kN, TerrainPatchState::kN, tp_->index_count,
               tp_->pipeline ? "pipeline built" : "NO pipeline (shader missing)");
}

void VulkanEngine::destroy_terrain_patches() {
    if (!tp_) return;
    if (tp_->pipeline) vkDestroyPipeline(device_, tp_->pipeline, nullptr);
    if (tp_->vert) vkDestroyShaderModule(device_, tp_->vert, nullptr);
    if (tp_->index_buf) vmaDestroyBuffer(allocator_, tp_->index_buf, tp_->index_alloc);
    if (tp_->inst_buf) vmaDestroyBuffer(allocator_, tp_->inst_buf, tp_->inst_alloc);
    tp_->pipeline = VK_NULL_HANDLE; tp_->vert = VK_NULL_HANDLE;
    tp_->index_buf = tp_->inst_buf = VK_NULL_HANDLE;
    tp_->inst_map = nullptr;
}

// Pick this frame's patches. Before update_scene_ubo (the scene constants say whether the
// march pre-pass runs, which it does not while the patches are drawn).
void VulkanEngine::update_terrain_patches() {
    if (!tp_) return;
    TerrainPatchState& S = *tp_;
    S.active = false;
    S.count = 0;
    const bool on = terrain_patches_on();
    // The depth picture is the low-resolution terrain's depth target at full size: a change of
    // the setting makes those targets again (pipelines_setup.cpp init_terrain_raymarch_lowres).
    if (on != S.was_on) {
        S.was_on = on;
        if (rt_.terrain_raymarch_scale < 0.999f) tr_lowres_recreate_pending_ = true;
    }
    if (!on) return;
    if (tr_->lr_extent.width != render_extent_.width || tr_->lr_extent.height != render_extent_.height ||
        tr_->lr_depth_view == VK_NULL_HANDLE) {
        tr_lowres_recreate_pending_ = true;         // (not yet at full size: the march, this frame)
        return;
    }
    const FrameView& fv = current_frame_view_;
    const glm::vec3 eye = fv.eye_pos;
    // From the air the whole landscape is the far march's (terrain_raymarch.frag `air`:
    // the eye above terrainMaxHeight() + 60 = the highest ground, 120 m, + 80 + 60).
    if (eye.y > 260.0f) return;

    const bool home = globe::current().home;
    const float wl = rt_.water_level;
    const auto ground = [&](float x, float z) {
        return home ? sample_terrain_height(x, z) : world_ground_height(x, z, wl, 9);
    };
    S.cam_h = std::max(eye.y - ground(eye.x, eye.z), 0.0f);

    // A node's lowest and highest ground: 5 x 5 samples of the CPU's ground, kept. The CPU's
    // ground is not the drawn one to the centimetre (fewer octaves, no close relief), and the
    // samples miss what lies between them: a margin that grows with the node.
    if (S.bounds_serial != globe::chart_serial() || S.bounds.size() > 60000) {
        S.bounds.clear();
        S.bounds_serial = globe::chart_serial();
    }
    S.bounds_made = 0;
    const auto node_bounds = [&](int level, int ix, int iz, float size) {
        const uint64_t key = (uint64_t(uint32_t(level)) << 56) ^ (uint64_t(uint32_t(ix) & 0xFFFFFFFu) << 28) ^ uint64_t(uint32_t(iz) & 0xFFFFFFFu);
        auto it = S.bounds.find(key);
        if (it != S.bounds.end()) return it->second;
        float lo = 1e30f, hi = -1e30f;
        for (int j = 0; j <= 4; ++j)
            for (int i = 0; i <= 4; ++i) {
                const float h = ground((float(ix) + 0.25f * float(i)) * size, (float(iz) + 0.25f * float(j)) * size);
                lo = std::min(lo, h); hi = std::max(hi, h);
            }
        const float margin = 2.5f + 0.2f * size;
        TerrainPatchState::Bounds b{ lo - margin, hi + margin };
        S.bounds.emplace(key, b);
        ++S.bounds_made;
        return b;
    };

    // Split a node while the camera is nearer than kSplit times its side: vertices are then
    // never further apart than about 1 / (kSplit * kN) of their distance - two to five pixels.
    static const float kSplit = [] { const char* e = std::getenv("VIBE_TEST_PATCH_SPLIT"); return e ? std::clamp(float(std::atof(e)), 0.75f, 6.0f) : 2.0f; }();
    const Frustum& fr = fv.frustum;
    S.list.clear(); S.list_d.clear();
    std::fill(std::begin(S.per_level), std::end(S.per_level), 0);
    const float range = TerrainPatchState::kRange;
    struct Node { int level, ix, iz; };
    Node stack[256];
    int sp = 0;
    {
        const int top = TerrainPatchState::kLevels - 1;
        const float size = float(TerrainPatchState::kN) * TerrainPatchState::kSpacing0 * float(1 << top);
        const int x0 = int(std::floor((eye.x - range) / size)), x1 = int(std::floor((eye.x + range) / size));
        const int z0 = int(std::floor((eye.z - range) / size)), z1 = int(std::floor((eye.z + range) / size));
        for (int iz = z0; iz <= z1; ++iz)
            for (int ix = x0; ix <= x1; ++ix)
                if (sp < 200) stack[sp++] = { top, ix, iz };
    }
    while (sp > 0) {
        const Node n = stack[--sp];
        const float size = float(TerrainPatchState::kN) * TerrainPatchState::kSpacing0 * float(1 << n.level);
        const TerrainPatchState::Bounds b = node_bounds(n.level, n.ix, n.iz, size);
        const glm::vec3 mn(float(n.ix) * size, b.lo, float(n.iz) * size), mx(mn.x + size, b.hi, mn.z + size);
        const glm::vec3 near_p = glm::clamp(eye, mn, mx);
        const float dist = glm::length(near_p - eye);
        if (dist > range) continue;
        // Out of the view: the box (with the skirt that hangs below the edge, up to 30 m) wholly
        // behind one of the view's side planes or its near plane. (Not its far plane: `range`
        // is the limit, and the patches' own projection has no far plane.)
        bool out = false;
        for (int p = 0; p < 5 && !out; ++p) {
            const glm::vec4& pl = fr.planes[p];
            const glm::vec3 v(pl.x > 0.0f ? mx.x : mn.x, pl.y > 0.0f ? mx.y : mn.y - 31.0f, pl.z > 0.0f ? mx.z : mn.z);
            out = glm::dot(glm::vec3(pl), v) + pl.w < 0.0f;
        }
        if (out) continue;
        if (n.level > 0 && dist < kSplit * size && sp + 4 <= 256) {
            for (int k = 0; k < 4; ++k) stack[sp++] = { n.level - 1, n.ix * 2 + (k & 1), n.iz * 2 + (k >> 1) };
            continue;
        }
        if (S.list.size() >= TerrainPatchState::kMaxPatches) continue;
        TerrainPatchState::Instance in{};
        in.patch = glm::vec4(mn.x, mn.z, TerrainPatchState::kSpacing0 * float(1 << n.level), float(n.level));
        in.view = glm::vec4(S.cam_h, 0.0f, 0.0f, 0.0f);
        S.list.push_back(in);
        S.list_d.push_back(dist);
        ++S.per_level[n.level];
    }
    if (S.list.empty()) return;
    // Nearest first: what lies behind is then rejected by the depth test before it is filled.
    S.order.resize(S.list.size());
    std::iota(S.order.begin(), S.order.end(), 0u);
    std::sort(S.order.begin(), S.order.end(), [&](uint32_t a, uint32_t b) { return S.list_d[a] < S.list_d[b]; });
    const uint32_t slot = uint32_t(frame_number_ % kFrameOverlap);
    S.inst_offset = VkDeviceSize(slot) * TerrainPatchState::kMaxPatches * sizeof(TerrainPatchState::Instance);
    auto* dst = reinterpret_cast<TerrainPatchState::Instance*>(static_cast<char*>(S.inst_map) + S.inst_offset);
    for (size_t i = 0; i < S.order.size(); ++i) dst[i] = S.list[S.order[i]];
    S.count = uint32_t(S.list.size());

    // The patches' own projection: the world's (with this frame's sub-pixel offset), its depth
    // replaced by near / distance - reversed, exact at any range, no far plane.
    glm::mat4 proj = fv.proj;
    proj[2][2] = 0.0f;
    proj[3][2] = TerrainPatchState::kNear;
    S.rel_proj = glm::mat4(glm::dmat4(proj) * glm::dmat4(glm::dmat3(fv.view)));
    S.active = true;

    static const bool kLog = std::getenv("VIBE_LOG_TERRAIN_PATCHES") != nullptr;
    if (kLog && (frame_number_ % 120) == 100) {
        const uint32_t vper = uint32_t(TerrainPatchState::kN + 3) * uint32_t(TerrainPatchState::kN + 3);
        std::string lv;
        for (int l = 0; l < TerrainPatchState::kLevels; ++l) lv += (l ? " " : "") + std::to_string(S.per_level[l]);
        log::infof("[terrain-patches] %u patches (per level, finest first: %s), %u vertices, %u triangles; camera %.1f m above the ground; node bounds kept %zu (+%d this frame)",
                   S.count, lv.c_str(), S.count * vper, S.count * (S.index_count / 3), double(S.cam_h), S.bounds.size(), S.bounds_made);
    }
}

// The patches, depth only, into the low-resolution terrain's depth target (full size while the
// patches are on; scene binding 11). In its own pass, before the world pass.
void VulkanEngine::render_terrain_patch_depth(VkCommandBuffer cmd) {
    if (!tp_ || !tp_->active || tp_->count == 0) return;
    TerrainPatchState& S = *tp_;
    vkinit::transition_image_aspect(cmd, tr_->lr_depth_image, VK_IMAGE_LAYOUT_UNDEFINED,
                                    VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT);
    VkClearValue clear{};
    clear.depthStencil.depth = 0.0f;                // reversed depth: 0 = nothing
    VkRenderingAttachmentInfo depth_att{
        .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
        .pNext = nullptr,
        .imageView = tr_->lr_depth_view,
        .imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
        .resolveMode = VK_RESOLVE_MODE_NONE,
        .resolveImageView = VK_NULL_HANDLE,
        .resolveImageLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .clearValue = clear,
    };
    VkRenderingInfo ri{
        .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
        .pNext = nullptr, .flags = 0,
        .renderArea = { {0, 0}, tr_->lr_extent },
        .layerCount = 1, .viewMask = 0,
        .colorAttachmentCount = 0, .pColorAttachments = nullptr,
        .pDepthAttachment = &depth_att, .pStencilAttachment = nullptr,
    };
    vkCmdBeginRendering(cmd, &ri);
    VkViewport vp{};
    vp.width  = static_cast<float>(tr_->lr_extent.width);
    vp.height = static_cast<float>(tr_->lr_extent.height);
    vp.minDepth = 0.0f; vp.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &vp);
    VkRect2D sc{ {0, 0}, tr_->lr_extent };
    vkCmdSetScissor(cmd, 0, 1, &sc);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, S.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout_, 0, 1, &scene_desc_set_, 0, nullptr);
    PushConstants pc{};
    pc.mvp = current_frame_view_.vp;
    pc.model = S.rel_proj;
    pc.prev_mvp = current_frame_view_.vp;
    // (the constants of the terrain's full-screen pass - world_render.cpp render_world - which the
    //  height formula in the vertex shader reads as the fragment shader does)
    pc.color = glm::vec4(0.0f, 0.0f, 11.5f, 22.0f);
    pc.emissive = glm::vec4(
        -std::clamp(rt_.terrain_raymarch_step_factor, 0.05f, 0.8f),
        std::max(1.0f, rt_.terrain_raymarch_lod_near_m * rm_air_lod_k_),
        std::max(rt_.terrain_raymarch_lod_near_m * rm_air_lod_k_ + 1.0f, rt_.terrain_raymarch_lod_far_m * rm_air_lod_k_),
        static_cast<float>(std::clamp(rt_.terrain_raymarch_lod_min_octaves, 2, 8)));
    pc.tex_params = glm::vec4(
        static_cast<float>(std::clamp(rt_.terrain_raymarch_max_steps, 60, 300)),
        static_cast<float>(std::clamp(rt_.terrain_raymarch_shadow_steps, 16, 96)),
        static_cast<float>(std::clamp(rt_.terrain_raymarch_octaves, 4, 24)),
        static_cast<float>(std::clamp(rt_.terrain_raymarch_normal_octaves, 4, 32)));
    vkCmdPushConstants(cmd, pipeline_layout_, kPushConstantStages, 0, sizeof(PushConstants), &pc);
    vkCmdBindVertexBuffers(cmd, 0, 1, &S.inst_buf, &S.inst_offset);
    vkCmdBindIndexBuffer(cmd, S.index_buf, 0, VK_INDEX_TYPE_UINT32);
    vkCmdDrawIndexed(cmd, S.index_count, S.count, 0, 0, 0);
    vkCmdEndRendering(cmd);
    vkinit::transition_image_aspect(cmd, tr_->lr_depth_image, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT);
}

bool VulkanEngine::terrain_patches_active() const { return tp_ && tp_->active && tp_->count > 0; }

} // namespace vibe
