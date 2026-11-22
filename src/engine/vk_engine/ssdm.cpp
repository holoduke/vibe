// Screen-space displacement mapping (SSDM).
//
// The engine side of the port of TechArtAlex's "Screen-Space Raymarched Displacement
// Mapping" (Unreal Engine plugin, shared for free use:
// patreon.com/TechArtAlex/posts/screen-space-171617266), itself made after the look of
// Crimson Desert. The method is described in shaders/ssdm_resolve.frag; this file is the
// plugin's SSDMSceneViewExtension: the render targets and the pass sequence.
//
// WHERE it runs is not the plugin's (a post-process on the lit image) but what is reported of
// Crimson Desert: on the G-buffer, in the middle of the main pass - after the static
// stonework is drawn, before lighting, and before trees, moving things, effects, grass and
// the weapon. So the lighting and shadows fall on the displaced surface, and everything drawn
// afterwards is depth-tested against it instead of being smeared by it.
//
// render_ssdm(), called by render_world() right after the castle meshes:
//   (the main pass is ended)
//   1. displacement mesh pass   every mesh with LevelMesh::ssdm > 0, inflated to a shell
//                               (ssdm_mesh.*) -> displacement, ray data, hit depth, shell depth
//   2. resolve                  ssdm_gbuffer: the march; scene colour, motion vectors and both
//                               G-buffer layers of the displaced surface + the resolve map
//   3. those four back into the main pass's attachments
//   4. depth re-projection      ssdm_depth: the displaced surface's depth into the scene depth
//   (the main pass is taken up again, loading what is there)
#include "engine/vk_engine/internal.h"
#include "engine/vk_pipelines.h"
#include "engine/vk_initializers.h"

#include <array>
#include <cstdlib>
#include <string>

namespace vibe {

namespace {
struct SsdmPC {
    glm::mat4 inv_vp;
    glm::mat4 clip_to_prev;
    glm::vec4 cam, march, guard, misc, mv;
};
constexpr uint32_t kSsdmBindings = 10;
}  // namespace

void VulkanEngine::destroy_ssdm_targets() {
    for (SsdmImage* t : { &ssdm_disp_, &ssdm_ray_, &ssdm_hitz_, &ssdm_shell_z_, &ssdm_color_, &ssdm_resolve_,
                          &ssdm_scene_z_, &ssdm_mv_, &ssdm_g0_, &ssdm_g1_ }) {
        if (t->view) vkDestroyImageView(device_, t->view, nullptr);
        if (t->image) vmaDestroyImage(allocator_, t->image, t->alloc);
        *t = SsdmImage{};
    }
    ssdm_extent_ = { 0, 0 };
}

void VulkanEngine::destroy_ssdm() {
    if (!device_) return;
    destroy_ssdm_targets();
    for (VkPipeline* p : { &ssdm_mesh_pipeline_, &ssdm_color_pipeline_, &ssdm_depth_pipeline_, &ssdm_velocity_pipeline_, &ssdm_gbuffer_pipeline_ }) {
        if (*p) vkDestroyPipeline(device_, *p, nullptr);
        *p = VK_NULL_HANDLE;
    }
    for (VkShaderModule* m : { &ssdm_mesh_vert_, &ssdm_mesh_frag_, &ssdm_color_frag_, &ssdm_depth_frag_, &ssdm_velocity_frag_, &ssdm_gbuffer_frag_ }) {
        if (*m) vkDestroyShaderModule(device_, *m, nullptr);
        *m = VK_NULL_HANDLE;
    }
    if (ssdm_fs_layout_) vkDestroyPipelineLayout(device_, ssdm_fs_layout_, nullptr);
    if (ssdm_set_layout_) vkDestroyDescriptorSetLayout(device_, ssdm_set_layout_, nullptr);
    if (ssdm_pool_) vkDestroyDescriptorPool(device_, ssdm_pool_, nullptr);
    if (ssdm_nearest_) vkDestroySampler(device_, ssdm_nearest_, nullptr);
    ssdm_fs_layout_ = VK_NULL_HANDLE; ssdm_set_layout_ = VK_NULL_HANDLE; ssdm_pool_ = VK_NULL_HANDLE;
    ssdm_nearest_ = VK_NULL_HANDLE; ssdm_set_ = VK_NULL_HANDLE;
    ssdm_ready_ = false;
}

bool VulkanEngine::init_ssdm() {
    if (ssdm_ready_) return true;
    if (ssdm_failed_ || !taa_vert_module_ || !pipeline_layout_) return false;
    const std::string sd = VIBE_SHADER_DIR;
    ssdm_mesh_vert_    = vkpipe::load_shader_module(device_, sd + "/ssdm_mesh.vert.spv");
    ssdm_mesh_frag_    = vkpipe::load_shader_module(device_, sd + "/ssdm_mesh.frag.spv");
    ssdm_gbuffer_frag_ = vkpipe::load_shader_module(device_, sd + "/ssdm_gbuffer.frag.spv");
    ssdm_depth_frag_   = vkpipe::load_shader_module(device_, sd + "/ssdm_depth.frag.spv");
    if (!ssdm_mesh_vert_ || !ssdm_mesh_frag_ || !ssdm_gbuffer_frag_ || !ssdm_depth_frag_) {
        log::warnf("[ssdm] shaders missing - screen-space displacement off");
        ssdm_failed_ = true;
        return false;
    }
    // Sampler: nearest + clamp (the march reads exact texels; G-buffer data must not be blended).
    {
        VkSamplerCreateInfo sci{};
        sci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        sci.magFilter = VK_FILTER_NEAREST; sci.minFilter = VK_FILTER_NEAREST;
        sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        sci.addressModeU = sci.addressModeV = sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.maxLod = 0.0f;
        vk_check(vkCreateSampler(device_, &sci, nullptr, &ssdm_nearest_), "ssdm sampler");
    }
    // The full-screen passes' set: ten sampled images.
    {
        std::array<VkDescriptorSetLayoutBinding, kSsdmBindings> b{};
        for (uint32_t i = 0; i < kSsdmBindings; ++i) {
            b[i].binding = i;
            b[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            b[i].descriptorCount = 1;
            b[i].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        }
        VkDescriptorSetLayoutCreateInfo lci{};
        lci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        lci.bindingCount = kSsdmBindings; lci.pBindings = b.data();
        vk_check(vkCreateDescriptorSetLayout(device_, &lci, nullptr, &ssdm_set_layout_), "ssdm set layout");
        VkDescriptorPoolSize ps{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, kSsdmBindings };
        VkDescriptorPoolCreateInfo pci{};
        pci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pci.maxSets = 1; pci.poolSizeCount = 1; pci.pPoolSizes = &ps;
        vk_check(vkCreateDescriptorPool(device_, &pci, nullptr, &ssdm_pool_), "ssdm pool");
        VkDescriptorSetAllocateInfo dai{};
        dai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dai.descriptorPool = ssdm_pool_; dai.descriptorSetCount = 1; dai.pSetLayouts = &ssdm_set_layout_;
        vk_check(vkAllocateDescriptorSets(device_, &dai, &ssdm_set_), "ssdm set");
        VkPushConstantRange pcr{ VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(SsdmPC) };
        VkPipelineLayoutCreateInfo plci{};
        plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount = 1; plci.pSetLayouts = &ssdm_set_layout_;
        plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pcr;
        vk_check(vkCreatePipelineLayout(device_, &plci, nullptr, &ssdm_fs_layout_), "ssdm layout");
    }
    // Displacement mesh pass: the world's vertex layout, push constants and scene set.
    {
        vkpipe::GraphicsPipelineConfig cfg{};
        cfg.vert = ssdm_mesh_vert_; cfg.frag = ssdm_mesh_frag_;
        cfg.layout = pipeline_layout_;
        cfg.color_formats = { VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R32G32_SFLOAT };
        cfg.depth_format = depth_format_;
        VkVertexInputBindingDescription vb{};
        vb.binding = 0; vb.stride = sizeof(Vertex); vb.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        cfg.vbindings = { vb };
        VkVertexInputAttributeDescription a0{}, a1{}, a2{};
        a0.location = 0; a0.format = VK_FORMAT_R32G32B32_SFLOAT; a0.offset = offsetof(Vertex, position);
        a1.location = 1; a1.format = VK_FORMAT_R32G32B32_SFLOAT; a1.offset = offsetof(Vertex, normal);
        a2.location = 2; a2.format = VK_FORMAT_R32G32_SFLOAT;    a2.offset = offsetof(Vertex, uv);
        cfg.vattrs = { a0, a1, a2 };
        cfg.cull = VK_CULL_MODE_NONE;           // (the depth test keeps the nearest shell face)
        cfg.depth_test = true; cfg.depth_write = true;
        ssdm_mesh_pipeline_ = vkpipe::build_graphics_pipeline(device_, cfg);
    }
    // The full-screen passes.
    {
        vkpipe::GraphicsPipelineConfig cfg{};
        cfg.vert = taa_vert_module_;
        cfg.layout = ssdm_fs_layout_;
        cfg.cull = VK_CULL_MODE_NONE;
        cfg.depth_test = false; cfg.depth_write = false;
        cfg.frag = ssdm_gbuffer_frag_;
        cfg.color_formats = { scene_color_format_, motion_vec_format_, gbuffer0_format_, gbuffer1_format_, VK_FORMAT_R16G16B16A16_SFLOAT };
        ssdm_gbuffer_pipeline_ = vkpipe::build_graphics_pipeline(device_, cfg);
        cfg.frag = ssdm_depth_frag_;
        cfg.color_formats.clear();
        cfg.color_attachment_count = 0;
        cfg.depth_format = depth_format_;
        cfg.depth_test = true; cfg.depth_write = true;
        cfg.depth_compare = VK_COMPARE_OP_ALWAYS;       // (the shader decides: it discards where the scene is nearer)
        ssdm_depth_pipeline_ = vkpipe::build_graphics_pipeline(device_, cfg);
    }
    if (!ssdm_mesh_pipeline_ || !ssdm_gbuffer_pipeline_ || !ssdm_depth_pipeline_) {
        log::warnf("[ssdm] pipeline creation failed - screen-space displacement off");
        ssdm_failed_ = true;
        return false;
    }
    ssdm_ready_ = true;
    log::infof("[ssdm] screen-space displacement mapping ready (G-buffer stage)");
    return true;
}

void VulkanEngine::create_ssdm_targets() {
    destroy_ssdm_targets();
    auto make = [&](VkFormat fmt, VkImageUsageFlags usage, VkImageAspectFlags aspect, SsdmImage& t) {
        VkImageCreateInfo ici{};
        ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ici.imageType = VK_IMAGE_TYPE_2D; ici.format = fmt;
        ici.extent = { render_extent_.width, render_extent_.height, 1 };
        ici.mipLevels = 1; ici.arrayLayers = 1;
        ici.samples = VK_SAMPLE_COUNT_1_BIT; ici.tiling = VK_IMAGE_TILING_OPTIMAL;
        ici.usage = usage; ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        vk_check(vmaCreateImage(allocator_, &ici, &aci, &t.image, &t.alloc, nullptr), "ssdm image");
        VkImageViewCreateInfo vci{};
        vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vci.image = t.image; vci.viewType = VK_IMAGE_VIEW_TYPE_2D; vci.format = fmt;
        vci.subresourceRange = { aspect, 0, 1, 0, 1 };
        vk_check(vkCreateImageView(device_, &vci, nullptr, &t.view), "ssdm view");
    };
    const VkImageUsageFlags rt = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    const VkImageUsageFlags rts = rt | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    const VkImageAspectFlags C = VK_IMAGE_ASPECT_COLOR_BIT, D = VK_IMAGE_ASPECT_DEPTH_BIT;
    make(VK_FORMAT_R16G16B16A16_SFLOAT, rt, C, ssdm_disp_);
    make(VK_FORMAT_R16G16B16A16_SFLOAT, rt, C, ssdm_ray_);
    make(VK_FORMAT_R32G32_SFLOAT, rt, C, ssdm_hitz_);
    make(depth_format_, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, D, ssdm_shell_z_);
    make(scene_color_format_, rts, C, ssdm_color_);
    make(motion_vec_format_, rts, C, ssdm_mv_);
    make(gbuffer0_format_, rts, C, ssdm_g0_);
    make(gbuffer1_format_, rts, C, ssdm_g1_);
    make(VK_FORMAT_R16G16B16A16_SFLOAT, rt, C, ssdm_resolve_);
    make(depth_format_, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, D, ssdm_scene_z_);
    ssdm_extent_ = render_extent_;
    ssdm_first_use_ = true;

    const VkImageLayout RO = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    const VkDescriptorImageInfo ii[kSsdmBindings] = {
        { ssdm_nearest_, scene_color_view_, RO },   { ssdm_nearest_, ssdm_disp_.view, RO },
        { ssdm_nearest_, ssdm_ray_.view, RO },      { ssdm_nearest_, ssdm_hitz_.view, RO },
        { ssdm_nearest_, ssdm_shell_z_.view, RO },  { ssdm_nearest_, ssdm_scene_z_.view, RO },
        { ssdm_nearest_, ssdm_resolve_.view, RO },  { ssdm_nearest_, motion_vec_view_, RO },
        { ssdm_nearest_, gbuffer0_view_, RO },      { ssdm_nearest_, gbuffer1_view_, RO } };
    VkWriteDescriptorSet w[kSsdmBindings]{};
    for (uint32_t i = 0; i < kSsdmBindings; ++i) {
        w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[i].dstSet = ssdm_set_; w[i].dstBinding = i; w[i].descriptorCount = 1;
        w[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        w[i].pImageInfo = &ii[i];
    }
    vkUpdateDescriptorSets(device_, kSsdmBindings, w, 0, nullptr);
}

// Called INSIDE the main pass (render_world, after the static stonework). Returns false and
// touches nothing when there is nothing to displace in view; otherwise it ends the pass, does
// its work on the pass's attachments, begins the pass again (loading them) and returns true -
// the caller then binds its own pipeline / set / buffers again.
bool VulkanEngine::render_ssdm(VkCommandBuffer cmd) {
    static const bool off = std::getenv("VIBE_NO_SSDM") != nullptr;
    if (off || castle_meshes_.empty() || !deferred_lighting_active_) return false;
    const FrameView& fv = current_frame_view_;
    bool any = false;
    for (const auto& g : castle_meshes_)
        if (g.ssdm > 0.0f && aabb_visible(fv.frustum, g.bmin - glm::vec3(g.ssdm), g.bmax + glm::vec3(g.ssdm))) { any = true; break; }
    if (!any) return false;
    if (!init_ssdm()) return false;
    // (destroy_taa() drops these targets with the ones they are sized like, so a missing image
    // here means: first use, or the render size changed and nothing in flight refers to them)
    if (!ssdm_disp_.image || ssdm_extent_.width != render_extent_.width || ssdm_extent_.height != render_extent_.height)
        create_ssdm_targets();
    const VkImageLayout RO = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, CA = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    const VkImageLayout UD = VK_IMAGE_LAYOUT_UNDEFINED, DA = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    const VkImageLayout TS = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, TD = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    const VkImageAspectFlags DEPTH = VK_IMAGE_ASPECT_DEPTH_BIT;
    const VkImageLayout from_c = ssdm_first_use_ ? UD : RO;
    const float rw = float(render_extent_.width), rh = float(render_extent_.height);
    VkViewport vp{ 0.0f, 0.0f, rw, rh, 0.0f, 1.0f };
    VkRect2D sc{ {0, 0}, render_extent_ };
    auto begin = [&](const VkRenderingAttachmentInfo* cols, uint32_t n, const VkRenderingAttachmentInfo* depth) {
        VkRenderingInfo ri{};
        ri.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
        ri.renderArea = sc; ri.layerCount = 1;
        ri.colorAttachmentCount = n; ri.pColorAttachments = cols; ri.pDepthAttachment = depth;
        vkCmdBeginRendering(cmd, &ri);
        vkCmdSetViewport(cmd, 0, 1, &vp);
        vkCmdSetScissor(cmd, 0, 1, &sc);
    };
    auto att = [](VkImageView v, VkImageLayout l, bool clear, VkClearValue cv) {
        VkRenderingAttachmentInfo a{};
        a.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        a.imageView = v; a.imageLayout = l;
        a.loadOp = clear ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD;
        a.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        a.clearValue = cv;
        return a;
    };
    VkClearValue none{};

    // ---- The main pass stops here; its attachments become readable ----
    vkCmdEndRendering(cmd);
    VkImage main_c[4] = { scene_color_image_, motion_vec_image_, gbuffer0_image_, gbuffer1_image_ };
    for (VkImage im : main_c) vkinit::transition_image(cmd, im, CA, RO);
    vkinit::transition_image_aspect(cmd, depth_image_, DA, TS, DEPTH);

    // ---- The scene's depth as it is now (the passes compare against it; step 4 writes the live one) ----
    {
        vkinit::transition_image_aspect(cmd, ssdm_scene_z_.image, from_c, TD, DEPTH);
        VkImageCopy r{};
        r.srcSubresource = { DEPTH, 0, 0, 1 };
        r.dstSubresource = { DEPTH, 0, 0, 1 };
        r.extent = { render_extent_.width, render_extent_.height, 1 };
        vkCmdCopyImage(cmd, depth_image_, TS, ssdm_scene_z_.image, TD, 1, &r);
        vkinit::transition_image_aspect(cmd, ssdm_scene_z_.image, TD, RO, DEPTH);
    }

    // ---- 1. The displacement mesh pass ----
    vkinit::transition_image(cmd, ssdm_disp_.image, from_c, CA);
    vkinit::transition_image(cmd, ssdm_ray_.image, from_c, CA);
    vkinit::transition_image(cmd, ssdm_hitz_.image, from_c, CA);
    vkinit::transition_image_aspect(cmd, ssdm_shell_z_.image, from_c, DA, DEPTH);
    {
        VkClearValue zero{}; VkClearValue far2{}; far2.color = { { 1.0f, 1.0f, 0.0f, 0.0f } };
        VkClearValue dz{}; dz.depthStencil = { 1.0f, 0 };
        const VkRenderingAttachmentInfo cols[3] = { att(ssdm_disp_.view, CA, true, zero), att(ssdm_ray_.view, CA, true, zero),
                                                    att(ssdm_hitz_.view, CA, true, far2) };
        const VkRenderingAttachmentInfo dep = att(ssdm_shell_z_.view, DA, true, dz);
        begin(cols, 3, &dep);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, ssdm_mesh_pipeline_);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout_, 0, 1, &scene_desc_set_, 0, nullptr);
        const glm::vec3 fwd = -glm::vec3(fv.inv_view[2]);
        VkDeviceSize off0 = 0;
        for (const auto& g : castle_meshes_) {
            if (g.ssdm <= 0.0f) continue;
            if (!aabb_visible(fv.frustum, g.bmin - glm::vec3(g.ssdm), g.bmax + glm::vec3(g.ssdm))) continue;
            PushConstants pc{};
            pc.mvp = fv.vp;
            pc.model = glm::mat4(1.0f);
            pc.prev_mvp = fv.vp;                                   // (this pass: the view-projection)
            pc.color = glm::vec4(1.0f / std::max(std::abs(fv.proj[1][1]), 1e-6f), rw / std::max(rh, 1.0f), 0.0f, 0.0f);
            pc.emissive = glm::vec4(fwd, 0.0f);
            pc.tex_params = glm::vec4(float(g.tex_albedo), float(g.tex_normal), g.uv_scale, 0.0f);
            pc.grass_params = glm::vec4(g.ssdm, 0.03f, 1.0f, 0.0f);    // shell, edge fade distance, edge handling mode 1
            vkCmdPushConstants(cmd, pipeline_layout_, kPushConstantStages, 0, sizeof(PushConstants), &pc);
            const Mesh& sm = g.ssdm_mesh.vertex_buffer ? g.ssdm_mesh : g.mesh;
            vkCmdBindVertexBuffers(cmd, 0, 1, &sm.vertex_buffer, &off0);
            vkCmdBindIndexBuffer(cmd, sm.index_buffer, 0, VK_INDEX_TYPE_UINT32);
            vkCmdDrawIndexed(cmd, sm.index_count, 1, 0, 0, 0);
        }
        vkCmdEndRendering(cmd);
    }
    vkinit::transition_image(cmd, ssdm_disp_.image, CA, RO);
    vkinit::transition_image(cmd, ssdm_ray_.image, CA, RO);
    vkinit::transition_image(cmd, ssdm_hitz_.image, CA, RO);
    vkinit::transition_image_aspect(cmd, ssdm_shell_z_.image, DA, RO, DEPTH);

    // VIBE_SSDM_MODE=1: the plain ray march instead of the default bent (NCS) one.
    static const int ssdm_mode = (std::getenv("VIBE_SSDM_MODE") && std::atoi(std::getenv("VIBE_SSDM_MODE")) == 1) ? 1 : 2;
    SsdmPC pc{};
    {
        const glm::dmat4 inv = glm::inverse(glm::dmat4(fv.vp));
        pc.inv_vp = glm::mat4(inv);
        glm::dmat4 c2p(1.0);
        if (prev_view_proj_valid_) {
            glm::dmat4 T(1.0);
            if (rt_.fsr2_enabled) {
                const glm::vec2 jd = -(fv.jitter - prev_jitter_) * 0.5f;
                T[3][0] = 2.0 * jd.x; T[3][1] = 2.0 * jd.y;
            }
            c2p = T * glm::dmat4(prev_view_proj_) * inv;
        }
        pc.clip_to_prev = glm::mat4(c2p);
        // The plugin's defaults: NCS resolver; 64 steps at grazing angles down to 32 head-on,
        // 4 refinement steps, no start jitter; surface tolerance 50 cm; occluder check on,
        // bias 1 cm; edge handling mode 1.
        pc.cam   = glm::vec4(fv.eye_pos, float(ssdm_mode));
        pc.march = glm::vec4(64.0f, 32.0f, 4.0f, 0.0f);
        pc.guard = glm::vec4(0.5f, 0.0f, 0.01f, 0.0f);
        pc.misc  = glm::vec4(1.0f, 1.0f, 0.0f, std::getenv("VIBE_SSDM_DEBUG") ? -1.0f : float(frame_number_ % 8));
        pc.mv    = glm::vec4(0.5f, 0.5f, rw, rh);
    }
    auto fullscreen = [&](VkPipeline p) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, p);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, ssdm_fs_layout_, 0, 1, &ssdm_set_, 0, nullptr);
        vkCmdPushConstants(cmd, ssdm_fs_layout_, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), &pc);
        vkCmdDraw(cmd, 3, 1, 0, 0);
        vkCmdEndRendering(cmd);
    };

    // ---- 2. Resolve: the displaced G-buffer + the resolve map ----
    SsdmImage* outs[5] = { &ssdm_color_, &ssdm_mv_, &ssdm_g0_, &ssdm_g1_, &ssdm_resolve_ };
    for (SsdmImage* t : outs) vkinit::transition_image(cmd, t->image, from_c, CA);
    {
        const VkRenderingAttachmentInfo cols[5] = { att(ssdm_color_.view, CA, false, none), att(ssdm_mv_.view, CA, false, none),
                                                    att(ssdm_g0_.view, CA, false, none), att(ssdm_g1_.view, CA, false, none),
                                                    att(ssdm_resolve_.view, CA, false, none) };
        begin(cols, 5, nullptr);
        fullscreen(ssdm_gbuffer_pipeline_);
    }
    // ---- 3. Back into the main pass's attachments ----
    {
        vkinit::transition_image(cmd, ssdm_resolve_.image, CA, RO);
        VkImage src[4] = { ssdm_color_.image, ssdm_mv_.image, ssdm_g0_.image, ssdm_g1_.image };
        VkImageCopy r{};
        r.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
        r.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
        r.extent = { render_extent_.width, render_extent_.height, 1 };
        for (int i = 0; i < 4; ++i) {
            vkinit::transition_image(cmd, src[i], CA, TS);
            vkinit::transition_image(cmd, main_c[i], RO, TD);
            vkCmdCopyImage(cmd, src[i], TS, main_c[i], TD, 1, &r);
            vkinit::transition_image(cmd, src[i], TS, RO);
            vkinit::transition_image(cmd, main_c[i], TD, CA);
        }
    }
    // ---- 4. Depth re-projection ----
    vkinit::transition_image_aspect(cmd, depth_image_, TS, DA, DEPTH);
    {
        const VkRenderingAttachmentInfo dep = att(depth_view_, DA, false, none);
        begin(nullptr, 0, &dep);
        fullscreen(ssdm_depth_pipeline_);
    }
    ssdm_first_use_ = false;

    // ---- The main pass again, loading what is there ----
    {
        const VkRenderingAttachmentInfo cols[4] = { att(scene_color_view_, CA, false, none), att(motion_vec_view_, CA, false, none),
                                                    att(gbuffer0_view_, CA, false, none), att(gbuffer1_view_, CA, false, none) };
        const VkRenderingAttachmentInfo dep = att(depth_view_, DA, false, none);
        begin(cols, 4, &dep);
    }
    return true;
}

}  // namespace vibe
