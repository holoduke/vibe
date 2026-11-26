// The trail: where people and cars have just been, as two textures round the player.
//
// Grass that is walked or driven through leans away and comes back up by itself. For that the
// plants have to know, cheaply and for any number of walkers, where something pushed and which
// way - also a moment AFTER it has passed. This is how the reference game does it (its
// interaction shader GenerateTrailInteractionCS and GenerateTrailHeightAndDeformedTrailCS, read
// from the compiled shaders; this is our own code for it): top-down textures that follow the
// player, drawn again every frame from the last ones.
//
//   256 x 256 texels of 25 cm (a 64 m square).
//   The trail, RGBA16F (scene binding 41; last frame's at 42):
//     .xy  the direction things were pushed in (world x, z; unit length where .w > 0)
//     .z   how fast what pushed went (0 slow .. 1 a run or a car)
//     .w   how much the plants here are down (0 = standing, 1 = fully pushed over)
//   The heights, RG32F (scene binding 43; last frame's at 44):
//     .x   the underside of the body that pushed here (world y), .y its top
//          (nothing: +1e5, -1e5). A blade reacts only when its root lies between them - less
//          a metre, plus 20 cm: nothing bends under a bridge that is being crossed.
//   Every frame (grass_trail_update, before the world is drawn; shaders/grass_trail.frag and
//   grass_trail_h.frag):
//     - the old pictures are shifted by however many whole texels the window moved,
//     - .z and .w fall off with a half-life of 0.56 s (exp2(-1.7847 dt), the game's rate),
//     - every "pusher" (the player on foot, cars, the world's people and their cars: the
//       nearest twelve, a person 0.6 m, a car 2.1 m) stamps a disc: .w climbs under it at the
//       game's rate (faster the faster it moves), direction outward from its middle, turned
//       toward the way it moves; and its underside and top go into the heights, which are
//       kept for as long as the trail is.
//   (The game finds the bodies in a top-down picture of everything that moves; here they are
//   a list.)
//
// Who reads them: scatter.vert (the grass: tuft_cd.glsl - every blade turns about its root
// away from the push). The window's place goes to it in the plant draws' push constants
// (grass_trail_window).
//
// VIBE_NO_GRASS_TRAIL=1: nothing is stamped (the textures stay empty: plants stand).
// VIBE_TEST_TRAIL_R=<m>: the player's disc that wide (test shots with a standing camera).
#include "engine/vk_engine/internal.h"
#include "engine/vk_engine/state/taa.h"
#include "engine/vk_initializers.h"
#include "engine/vk_pipelines.h"
#include "npc/npc.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <string>

namespace vibe {

namespace {
constexpr uint32_t kTrailN     = 256;       // texels a side
constexpr float    kTrailTexel = 0.25f;     // metres a texel
constexpr VkFormat kTrailFormat  = VK_FORMAT_R16G16B16A16_SFLOAT;
constexpr VkFormat kHeightFormat = VK_FORMAT_R32G32_SFLOAT;
constexpr int      kTrailPushers = 12;      // (three matrices of the push-constant block)

// The heights' own objects (the engine has one trail; the trail's are members of the engine).
struct TrailHeights {
    VkImage        image[2] = {};           // [0] this frame's (binding 43), [1] last frame's (44)
    VmaAllocation  alloc[2] = {};
    VkImageView    view[2] = {};
    VkSampler      sampler = VK_NULL_HANDLE;
    VkPipeline     pipeline = VK_NULL_HANDLE;
    VkShaderModule frag = VK_NULL_HANDLE;
} g_th;

void make_image(VkDevice dev, VmaAllocator al, VkFormat fmt, VkImage& img, VmaAllocation& alloc, VkImageView& view) {
    VkImageCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = fmt;
    ici.extent = { kTrailN, kTrailN, 1 };
    ici.mipLevels = 1; ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    vk_check(vmaCreateImage(al, &ici, &aci, &img, &alloc, nullptr), "grass trail image");
    VkImageViewCreateInfo vci{};
    vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vci.image = img;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = fmt;
    vci.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    vk_check(vkCreateImageView(dev, &vci, nullptr, &view), "grass trail view");
}
}  // namespace

void VulkanEngine::init_grass_trail() {
    VkSamplerCreateInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    si.magFilter = VK_FILTER_LINEAR; si.minFilter = VK_FILTER_LINEAR;
    si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    vk_check(vkCreateSampler(device_, &si, nullptr, &grass_trail_sampler_), "grass trail sampler");
    si.magFilter = VK_FILTER_NEAREST; si.minFilter = VK_FILTER_NEAREST;        // (heights are not blended across a body's edge)
    vk_check(vkCreateSampler(device_, &si, nullptr, &g_th.sampler), "grass trail height sampler");
    for (int l = 0; l < 4; ++l) {
        const bool h = l >= 2;
        VkImage& img = h ? g_th.image[l - 2] : grass_trail_image_[l];
        VkImageView& view = h ? g_th.view[l - 2] : grass_trail_view_[l];
        make_image(device_, allocator_, h ? kHeightFormat : kTrailFormat, img, h ? g_th.alloc[l - 2] : grass_trail_alloc_[l], view);
        // (empty and readable from the start: the bindings must be valid before the first frame)
        vkinit::one_time_submit(device_, graphics_queue_, graphics_queue_family_, [&](VkCommandBuffer cb) {
            vkinit::transition_image(cb, img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            VkClearColorValue zero{};
            if (h) { zero.float32[0] = 1e5f; zero.float32[1] = -1e5f; }
            VkImageSubresourceRange all{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
            vkCmdClearColorImage(cb, img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &zero, 1, &all);
            vkinit::transition_image(cb, img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        });
        VkDescriptorImageInfo ii{};
        ii.sampler = h ? g_th.sampler : grass_trail_sampler_;
        ii.imageView = view;
        ii.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        VkWriteDescriptorSet w{};
        w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet = scene_desc_set_;
        w.dstBinding = 41 + uint32_t(l);
        w.descriptorCount = 1;
        w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        w.pImageInfo = &ii;
        vkUpdateDescriptorSets(device_, 1, &w, 0, nullptr);
    }
    std::string sd = VIBE_SHADER_DIR;
    grass_trail_frag_ = vkpipe::load_shader_module(device_, sd + "/grass_trail.frag.spv");
    g_th.frag = vkpipe::load_shader_module(device_, sd + "/grass_trail_h.frag.spv");
    vkpipe::GraphicsPipelineConfig cfg{};
    cfg.vert = taa_->vert_module;            // (the full-screen triangle)
    cfg.layout = pipeline_layout_;
    cfg.cull = VK_CULL_MODE_NONE;
    cfg.depth_test = false;
    cfg.depth_write = false;
    if (grass_trail_frag_) {
        cfg.frag = grass_trail_frag_;
        cfg.color_format = kTrailFormat;
        grass_trail_pipeline_ = vkpipe::build_graphics_pipeline(device_, cfg);
    }
    if (g_th.frag) {
        cfg.frag = g_th.frag;
        cfg.color_format = kHeightFormat;
        g_th.pipeline = vkpipe::build_graphics_pipeline(device_, cfg);
    }
    if (!grass_trail_pipeline_ || !g_th.pipeline) log::warnf("[grass-trail] shader or pipeline missing - plants will not bend");
    else log::info("grass trail built (64 m round the player, 25 cm a texel)");
}

void VulkanEngine::destroy_grass_trail() {
    if (grass_trail_pipeline_) vkDestroyPipeline(device_, grass_trail_pipeline_, nullptr);
    if (grass_trail_frag_)     vkDestroyShaderModule(device_, grass_trail_frag_, nullptr);
    if (grass_trail_sampler_)  vkDestroySampler(device_, grass_trail_sampler_, nullptr);
    if (g_th.pipeline) vkDestroyPipeline(device_, g_th.pipeline, nullptr);
    if (g_th.frag)     vkDestroyShaderModule(device_, g_th.frag, nullptr);
    if (g_th.sampler)  vkDestroySampler(device_, g_th.sampler, nullptr);
    for (int l = 0; l < 2; ++l) {
        if (grass_trail_view_[l])  vkDestroyImageView(device_, grass_trail_view_[l], nullptr);
        if (grass_trail_image_[l]) vmaDestroyImage(allocator_, grass_trail_image_[l], grass_trail_alloc_[l]);
        grass_trail_view_[l] = VK_NULL_HANDLE; grass_trail_image_[l] = VK_NULL_HANDLE; grass_trail_alloc_[l] = nullptr;
        if (g_th.view[l])  vkDestroyImageView(device_, g_th.view[l], nullptr);
        if (g_th.image[l]) vmaDestroyImage(allocator_, g_th.image[l], g_th.alloc[l]);
    }
    g_th = TrailHeights{};
    grass_trail_pipeline_ = VK_NULL_HANDLE; grass_trail_frag_ = VK_NULL_HANDLE; grass_trail_sampler_ = VK_NULL_HANDLE;
}

// The window as the plant shaders need it: xy = the world x, z of its corner, z = 1 / its side
// in metres, w = 1 when there is a trail to read.
glm::vec4 VulkanEngine::grass_trail_window() const {
    return glm::vec4(grass_trail_org_.x, grass_trail_org_.y, 1.0f / (float(kTrailN) * kTrailTexel),
                     grass_trail_pipeline_ && g_th.pipeline ? 1.0f : 0.0f);
}

// Once a frame, before anything that reads the trail is drawn.
void VulkanEngine::grass_trail_update(VkCommandBuffer cmd) {
    if (grass_trail_pipeline_ == VK_NULL_HANDLE || g_th.pipeline == VK_NULL_HANDLE) return;
    static const bool off = std::getenv("VIBE_NO_GRASS_TRAIL") != nullptr;
    const float dt = std::clamp(anim_time_ - grass_trail_time_, 0.0f, 0.25f);
    grass_trail_time_ = anim_time_;

    // ---- The window: centred on the player, moved in whole texels (so the old pictures are
    // shifted without being blurred) ----
    const glm::vec3 eye = player_.eye_position();
    const glm::vec2 prev_org = grass_trail_org_;
    const float half = 0.5f * float(kTrailN) * kTrailTexel;
    grass_trail_org_ = glm::floor((glm::vec2(eye.x, eye.z) - glm::vec2(half)) / kTrailTexel) * kTrailTexel;

    // ---- The pushers: xz, velocity xz; underside, top; radius. The nearest twelve. ----
    struct Pusher { glm::vec4 p; glm::vec2 h; float r; float d2; };
    std::vector<Pusher> all;
    auto add = [&](glm::vec3 p, glm::vec2 vel, float r, float bottom, float top) {
        const float dx = p.x - eye.x, dz = p.z - eye.z;
        if (std::abs(dx) > half || std::abs(dz) > half) return;
        all.push_back({ glm::vec4(p.x, p.z, vel.x, vel.y), glm::vec2(bottom, top), r, dx * dx + dz * dz });
    };
    constexpr float kPerson = 0.6f, kCar = 2.1f;        // (0.6 m: the game's radius for its player)
    if (!off && globe::current().home) {
        // The player on foot (not in a car's seat).
        if (!body_seated_) {
            // (VIBE_TEST_TRAIL_R=<m>: a wider disc, to see the plants round a standing camera go down)
            static const float test_r = std::getenv("VIBE_TEST_TRAIL_R") ? float(std::atof(std::getenv("VIBE_TEST_TRAIL_R"))) : 0.0f;
            add(eye, glm::vec2(player_.velocity.x, player_.velocity.z), test_r > 0.0f ? test_r : kPerson, eye.y - 1.7f, eye.y + 0.1f);
        }
        // Cars: a disc under the body; their speed from where they were a frame ago.
        static std::vector<glm::vec3> cars_prev;
        std::vector<glm::vec3> cars;
        jeep_positions(cars);
        for (size_t i = 0; i < cars.size(); ++i) {
            const glm::vec3& p = cars[i];
            glm::vec2 vel(0.0f);
            if (cars_prev.size() == cars.size() && dt > 1e-4f) {
                vel = glm::vec2(p.x - cars_prev[i].x, p.z - cars_prev[i].z) / dt;
                if (glm::dot(vel, vel) > 80.0f * 80.0f) vel = glm::vec2(0.0f);      // (put somewhere else, not driven there)
            }
            add(p, vel, kCar, p.y - 1.2f, p.y + 1.0f);
        }
        cars_prev = cars;
        // The people and cars of the world (npc_world.cpp).
        for (const npc::View& v : npc_views_) {
            if (v.hidden) continue;
            const glm::vec2 fw(std::sin(v.yaw), -std::cos(v.yaw));
            const bool car = v.kind == npc::Kind::Car;
            add(v.pos, fw * v.speed, car ? kCar : kPerson, v.pos.y - 0.2f, v.pos.y + (car ? 1.6f : 1.8f));
        }
    }
    std::sort(all.begin(), all.end(), [](const Pusher& a, const Pusher& b) { return a.d2 < b.d2; });
    const int n = int(std::min<size_t>(all.size(), kTrailPushers));
    // (pushers 8..11 are people in the shaders: the cars go first)
    std::stable_sort(all.begin(), all.begin() + n, [](const Pusher& a, const Pusher& b) { return a.r > b.r; });
    glm::vec4 push[kTrailPushers] = {}, push_h[kTrailPushers] = {};
    float     radius[8] = {};
    for (int i = 0; i < n; ++i) {
        const Pusher& P = all[size_t(i)];
        push[i] = P.p;
        push_h[i] = glm::vec4(P.p.x, P.p.y, P.h.x, P.h.y);
        if (i < 8) radius[i] = P.r;
    }

    // ---- Draw them: into [0], reading last frame's [1]; then keep a copy as the next "last".
    // The trail first: the heights read this frame's trail. ----
    const VkImageLayout RO = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, CA = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    const VkImageLayout TS = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, TD = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    PushConstants pc{};          // (as the two shaders read the block)
    pc.color = glm::vec4(grass_trail_org_, prev_org);
    pc.emissive = glm::vec4(dt, float(n), kTrailTexel, float(kTrailN));
    pc.tex_params = glm::vec4(radius[0], radius[1], radius[2], radius[3]);
    pc.grass_params = glm::vec4(radius[4], radius[5], radius[6], radius[7]);
    for (int pass = 0; pass < 2; ++pass) {
        VkImage*     img  = pass == 0 ? grass_trail_image_ : g_th.image;
        VkImageView* view = pass == 0 ? grass_trail_view_ : g_th.view;
        const glm::vec4* src = pass == 0 ? push : push_h;
        vkinit::transition_image(cmd, img[0], RO, CA);
        VkClearValue cv{};
        VkRenderingAttachmentInfo att = vkinit::color_attachment_info(view[0], &cv, CA);
        VkRenderingInfo ri{};
        ri.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
        ri.renderArea = { {0, 0}, { kTrailN, kTrailN } };
        ri.layerCount = 1;
        ri.colorAttachmentCount = 1;
        ri.pColorAttachments = &att;
        vkCmdBeginRendering(cmd, &ri);
        VkViewport vp{};
        vp.width = float(kTrailN); vp.height = float(kTrailN); vp.minDepth = 0.0f; vp.maxDepth = 1.0f;
        vkCmdSetViewport(cmd, 0, 1, &vp);
        VkRect2D sc{ {0, 0}, { kTrailN, kTrailN } };
        vkCmdSetScissor(cmd, 0, 1, &sc);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pass == 0 ? grass_trail_pipeline_ : g_th.pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout_, 0, 1, &scene_desc_set_, 0, nullptr);
        for (int i = 0; i < 4; ++i) { pc.mvp[i] = src[i]; pc.model[i] = src[4 + i]; pc.prev_mvp[i] = src[8 + i]; }
        vkCmdPushConstants(cmd, pipeline_layout_, kPushConstantStages, 0, sizeof(PushConstants), &pc);
        vkCmdDraw(cmd, 3, 1, 0, 0);
        vkCmdEndRendering(cmd);

        vkinit::transition_image(cmd, img[0], CA, TS);
        vkinit::transition_image(cmd, img[1], RO, TD);
        VkImageCopy r{};
        r.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
        r.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
        r.extent = { kTrailN, kTrailN, 1 };
        vkCmdCopyImage(cmd, img[0], TS, img[1], TD, 1, &r);
        vkinit::transition_image(cmd, img[1], TD, RO);
        vkinit::transition_image(cmd, img[0], TS, RO);
    }

    // ---- Tracks in snow and sand (post_fx.cpp post_fx_tracks): from the same bodies, a car's
    // four wheels (from its heading, while it moves) and a person's trodden path; each a short
    // capsule back along this frame's move, so a fast car leaves an unbroken rut. ----
    {
        std::vector<glm::vec4> trk;
        for (int i = 0; i < n; ++i) {
            const Pusher& B = all[size_t(i)];
            const glm::vec2 pos(B.p.x, B.p.y), vel(B.p.z, B.p.w);
            const float sp = glm::length(vel);
            const glm::vec2 back = vel * dt;
            if (B.r > 1.0f) {
                if (sp < 0.3f) continue;
                const glm::vec2 fw = vel / sp, rt(-fw.y, fw.x);
                for (int k = 0; k < 4; ++k) {
                    const glm::vec2 wpos = pos + fw * ((k < 2) ? 1.3f : -1.3f) + rt * ((k & 1) ? 0.78f : -0.78f);
                    trk.push_back(glm::vec4(wpos, back));
                    trk.push_back(glm::vec4(0.15f, 0.10f, 0.0f, 0.0f));     // (a tyre's width, 10 cm deep)
                }
            } else {
                trk.push_back(glm::vec4(pos, back));
                trk.push_back(glm::vec4(0.20f, 0.05f, 0.0f, 0.0f));         // (a path trodden 5 cm deep)
            }
        }
        post_fx_tracks(cmd, trk, eye);
    }
}

}  // namespace vibe
