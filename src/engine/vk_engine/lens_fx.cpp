// Camera weather and heat (shaders/lens_fx.comp; state: vk_engine/state/lens_fx.h).
//
//   Rain on the lens - in the rain, under open sky (64 rays up from the eye against the scene,
//     on the GPU, eased: under a roof, a bridge or a tree's crown fewer drops land; in a car's
//     cabin, under water or inside a known shelter none) drops land on the glass at random,
//     merge, run down when they grow too heavy and leave a wet trail, and dry - slowly in the
//     rain, within a few seconds out of it. compose.frag refracts the picture through the
//     water's surface and adds a small highlight. Setting `lens_rain` (Video tab, next to the
//     lens flare), on by default.
//   Heat shimmer - hot air in front of the scene wavers the picture behind it: idling and
//     revving cars' exhausts, the rocket's plume at launch, explosions for a few seconds, and
//     faintly a hot road far off on a clear summer afternoon. A half-resolution offset target
//     the compose pass resamples through; not drawn (and not read) when nothing is hot.
//   The block compose.frag reads all this from (binding 12) also carries the moon: where it
//     stands, the real sun (its phase follows from the two), how bright it is.
//
// Test hooks: VIBE_TEST_RAIN_LENS=<rain 0..1> forces rain on the lens (open sky still tested);
// VIBE_TEST_SHIMMER=<k> puts a hot source of strength k (1 = a revving car) 6 m in front of the camera; VIBE_LOG_LENS=1 logs the
// state once a second.

#include "engine/vk_engine/internal.h"
#include "engine/vk_engine/state/lens_fx.h"
#include "engine/vk_engine/state/compose.h"
#include "engine/vk_engine/state/post_fx.h"
#include "engine/vk_engine/state/environment.h"
#include "engine/vk_initializers.h"
#include "engine/vk_pipelines.h"
#include "engine/log.h"
#include "engine/globe.h"
#include "engine/terrain.h"

#include <GLFW/glfw3.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>

namespace vibe {

namespace {
// std140 mirror of `LensFx` in shaders/lens_fx.comp and `ComposeExtras` in compose.frag.
constexpr int kHotMax = 16;
struct LensFxUBO {
    glm::vec4 moon_dir;    // xyz toward the moon, w radiance scale (0 = not drawn)
    glm::vec4 sun_true;    // xyz toward the real sun, w 1 = the compose push constant's "sun" is not the sun (the moon, the glow): no sun disc there
    glm::vec4 moon_misc;   // x angular radius (rad), y earthshine share, z, w
    glm::vec4 lens;        // x lens rain shown, y shimmer drawn, z time (s), w frame
    glm::vec4 rain;        // x rain, y lens can be rained on (CPU tests), z dt, w drops this frame
    glm::vec4 cam;         // xyz eye, w aspect
    glm::mat4 inv_vp;      // render view, ndc -> world
    glm::vec4 sh_info;     // x hot sources, y hot-road mirage
    glm::vec4 src[2 * kHotMax];
};
static_assert(sizeof(LensFxUBO) % 16 == 0, "std140");
struct LensPC { glm::vec4 mode; };
constexpr int kLensBindings = 9;

template <class Img> void make_image(VkDevice dev, VmaAllocator al, Img& im, uint32_t w, uint32_t h, VkFormat f) {
    VkImageCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = f;
    ici.extent = { w, h, 1 };
    ici.mipLevels = 1; ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    vk_check(vmaCreateImage(al, &ici, &aci, &im.image, &im.alloc, nullptr), "lens fx image");
    VkImageViewCreateInfo vci{};
    vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vci.image = im.image;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = f;
    vci.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    vk_check(vkCreateImageView(dev, &vci, nullptr, &im.view), "lens fx view");
}
template <class Img> void free_image(VkDevice dev, VmaAllocator al, Img& im) {
    if (im.view) vkDestroyImageView(dev, im.view, nullptr);
    if (im.image) vmaDestroyImage(al, im.image, im.alloc);
    im = {};
}
}  // namespace

void VulkanEngine::init_lens_fx() {
    LensFxState& L = *lens_fx_;
    if (L.pipeline) return;
    L.sh_w = std::max(16u, render_extent_.width / 2);
    L.sh_h = std::max(16u, render_extent_.height / 2);
    make_image(device_, allocator_, L.lens_a, L.lens_w, L.lens_h, VK_FORMAT_R16G16B16A16_SFLOAT);
    make_image(device_, allocator_, L.lens_b, L.lens_w, L.lens_h, VK_FORMAT_R16G16B16A16_SFLOAT);
    make_image(device_, allocator_, L.shimmer, L.sh_w, L.sh_h, VK_FORMAT_R16G16_SFLOAT);
    {
        VkBufferCreateInfo bci{};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size = sizeof(LensFxUBO);
        bci.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;   // (compose reads it as a storage buffer: the post effects' copies of its set come from a pool without uniform buffers)
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        vk_check(vmaCreateBuffer(allocator_, &bci, &aci, &L.ubo, &L.ubo_alloc, nullptr), "lens fx ubo");
        bci.size = 32;
        bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        VmaAllocationCreateInfo hci{};      // (host-visible: VIBE_LOG_LENS reads the largest shimmer offset back)
        hci.usage = VMA_MEMORY_USAGE_AUTO;
        hci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
        vk_check(vmaCreateBuffer(allocator_, &bci, &hci, &L.state, &L.state_alloc, nullptr), "lens fx state");
    }
    {
        VkSamplerCreateInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        si.magFilter = si.minFilter = VK_FILTER_LINEAR;
        si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.maxLod = 0.0f;
        vk_check(vkCreateSampler(device_, &si, nullptr, &L.sampler), "lens fx sampler");
    }
    vkinit::one_time_submit(device_, graphics_queue_, graphics_queue_family_, [&](VkCommandBuffer cb) {
        VkClearColorValue zero{ { 0.0f, 0.0f, 0.0f, 0.0f } };
        VkImageSubresourceRange r{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        for (AtmoImage* im : { &L.lens_a, &L.lens_b, &L.shimmer }) {
            vkinit::transition_image(cb, im->image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
            vkCmdClearColorImage(cb, im->image, VK_IMAGE_LAYOUT_GENERAL, &zero, 1, &r);
        }
        vkCmdFillBuffer(cb, L.state, 0, 32, 0u);
        LensFxUBO u{};
        vkCmdUpdateBuffer(cb, L.ubo, 0, sizeof(u), &u);
    });
    const VkDescriptorType types[kLensBindings] = {
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER };
    {
        VkDescriptorSetLayoutBinding b[kLensBindings]{};
        for (int i = 0; i < kLensBindings; ++i) {
            b[i].binding = uint32_t(i); b[i].descriptorType = types[i];
            b[i].descriptorCount = 1; b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo lci{};
        lci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        lci.bindingCount = kLensBindings; lci.pBindings = b;
        vk_check(vkCreateDescriptorSetLayout(device_, &lci, nullptr, &L.set_layout), "lens fx layout");
        VkDescriptorPoolSize sizes[] = {
            { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 }, { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 3 },
            { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 3 }, { VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1 },
            { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1 } };
        VkDescriptorPoolCreateInfo pci{};
        pci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pci.maxSets = 1; pci.poolSizeCount = 5; pci.pPoolSizes = sizes;
        vk_check(vkCreateDescriptorPool(device_, &pci, nullptr, &L.pool), "lens fx pool");
        VkDescriptorSetAllocateInfo dai{};
        dai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dai.descriptorPool = L.pool; dai.descriptorSetCount = 1; dai.pSetLayouts = &L.set_layout;
        vk_check(vkAllocateDescriptorSets(device_, &dai, &L.set), "lens fx set");
    }
    {
        VkPushConstantRange pr{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(LensPC) };
        VkPipelineLayoutCreateInfo plci{};
        plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount = 1; plci.pSetLayouts = &L.set_layout;
        plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pr;
        vk_check(vkCreatePipelineLayout(device_, &plci, nullptr, &L.pipeline_layout), "lens fx pipeline layout");
        VkShaderModule m = vkpipe::load_shader_module(device_, std::string(VIBE_SHADER_DIR) + "/lens_fx.comp.spv");
        VkComputePipelineCreateInfo cci{};
        cci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        cci.stage.module = m;
        cci.stage.pName = "main";
        cci.layout = L.pipeline_layout;
        vk_check(vkCreateComputePipelines(device_, pipeline_cache_, 1, &cci, nullptr, &L.pipeline), "lens fx pipeline");
        vkDestroyShaderModule(device_, m, nullptr);
    }
    // The fixed bindings (all but the depth and the TLAS, which are written when they change).
    {
        VkDescriptorBufferInfo ub{ L.ubo, 0, sizeof(LensFxUBO) }, sb{ L.state, 0, 32 };
        VkDescriptorImageInfo ii[6] = {
            {}, { L.sampler, L.lens_a.view, VK_IMAGE_LAYOUT_GENERAL }, { L.sampler, L.lens_b.view, VK_IMAGE_LAYOUT_GENERAL },
            { VK_NULL_HANDLE, L.lens_a.view, VK_IMAGE_LAYOUT_GENERAL }, { VK_NULL_HANDLE, L.lens_b.view, VK_IMAGE_LAYOUT_GENERAL },
            { VK_NULL_HANDLE, L.shimmer.view, VK_IMAGE_LAYOUT_GENERAL } };
        VkWriteDescriptorSet w[7]{};
        for (int i = 0; i < 6; ++i) {
            w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[i].dstSet = L.set; w[i].dstBinding = uint32_t(i); w[i].descriptorCount = 1;
            w[i].descriptorType = types[i];
            if (i == 0) w[i].pBufferInfo = &ub; else w[i].pImageInfo = &ii[i];
        }
        w[6] = w[0]; w[6].dstBinding = 8; w[6].descriptorType = types[8]; w[6].pBufferInfo = &sb;
        vkUpdateDescriptorSets(device_, 7, w, 0, nullptr);
    }
    log::infof("[lens-fx] rain on the lens (%ux%u) and heat shimmer (%ux%u) ready", L.lens_w, L.lens_h, L.sh_w, L.sh_h);
}

void VulkanEngine::destroy_lens_fx() {
    if (!lens_fx_) return;
    LensFxState& L = *lens_fx_;
    if (L.pipeline) vkDestroyPipeline(device_, L.pipeline, nullptr);
    if (L.pipeline_layout) vkDestroyPipelineLayout(device_, L.pipeline_layout, nullptr);
    if (L.pool) vkDestroyDescriptorPool(device_, L.pool, nullptr);
    if (L.set_layout) vkDestroyDescriptorSetLayout(device_, L.set_layout, nullptr);
    if (L.sampler) vkDestroySampler(device_, L.sampler, nullptr);
    if (L.ubo) vmaDestroyBuffer(allocator_, L.ubo, L.ubo_alloc);
    if (L.state) vmaDestroyBuffer(allocator_, L.state, L.state_alloc);
    free_image(device_, allocator_, L.lens_a);
    free_image(device_, allocator_, L.lens_b);
    free_image(device_, allocator_, L.shimmer);
    *lens_fx_ = LensFxState{};
}

// Compose bindings 12 (the lens water), 13 (the shimmer offsets), 14 (the block). Called by
// rewrite_compose_image_bindings (post.cpp) and when the shimmer target is resized.
void VulkanEngine::write_lens_fx_compose_bindings() {
    if (!compose_->desc_set_layout) return;
    init_lens_fx();
    LensFxState& L = *lens_fx_;
    if (!L.pipeline) return;
    VkDescriptorImageInfo il{ L.sampler, L.lens_a.view, VK_IMAGE_LAYOUT_GENERAL };
    VkDescriptorImageInfo is{ L.sampler, L.shimmer.view, VK_IMAGE_LAYOUT_GENERAL };
    VkDescriptorBufferInfo ub{ L.ubo, 0, sizeof(LensFxUBO) };
    // (both compose sets, and the post effects' copies of them - post_fx.cpp - when they exist)
    for (int s = 0; s < 2 * kHistorySlots; ++s) {
        const VkDescriptorSet ds = s < kHistorySlots ? compose_->desc_sets[s] : (pfx_ ? pfx_->compose_sets[s - kHistorySlots] : VK_NULL_HANDLE);
        if (!ds) continue;
        VkWriteDescriptorSet w[3]{};
        for (int i = 0; i < 3; ++i) {
            w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[i].dstSet = ds; w[i].dstBinding = uint32_t(12 + i); w[i].descriptorCount = 1;
            w[i].descriptorType = i < 2 ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        }
        w[0].pImageInfo = &il; w[1].pImageInfo = &is; w[2].pBufferInfo = &ub;
        vkUpdateDescriptorSets(device_, 3, w, 0, nullptr);
    }
}

void VulkanEngine::dispatch_lens_fx(VkCommandBuffer cmd) {
    LensFxState& L = *lens_fx_;
    if (!L.pipeline || !depth_view_) return;
    // The shimmer target follows the render resolution.
    {
        const uint32_t w = std::max(16u, render_extent_.width / 2), h = std::max(16u, render_extent_.height / 2);
        if (w != L.sh_w || h != L.sh_h) {
            wait_gpu_frames_idle();
            free_image(device_, allocator_, L.shimmer);
            L.sh_w = w; L.sh_h = h;
            make_image(device_, allocator_, L.shimmer, w, h, VK_FORMAT_R16G16_SFLOAT);
            vkinit::one_time_submit(device_, graphics_queue_, graphics_queue_family_, [&](VkCommandBuffer cb) {
                vkinit::transition_image(cb, L.shimmer.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
                VkClearColorValue zero{ { 0.0f, 0.0f, 0.0f, 0.0f } };
                VkImageSubresourceRange r{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
                vkCmdClearColorImage(cb, L.shimmer.image, VK_IMAGE_LAYOUT_GENERAL, &zero, 1, &r);
            });
            VkDescriptorImageInfo ii{ VK_NULL_HANDLE, L.shimmer.view, VK_IMAGE_LAYOUT_GENERAL };
            VkWriteDescriptorSet wr{};
            wr.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            wr.dstSet = L.set; wr.dstBinding = 5; wr.descriptorCount = 1;
            wr.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; wr.pImageInfo = &ii;
            vkUpdateDescriptorSets(device_, 1, &wr, 0, nullptr);
            write_lens_fx_compose_bindings();
        }
    }
    if (L.depth_bound != depth_view_ || (tlas_ && L.tlas_bound != tlas_)) {
        wait_gpu_frames_idle();
        VkDescriptorImageInfo di{ L.sampler, depth_view_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        VkWriteDescriptorSet w[2]{};
        w[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[0].dstSet = L.set; w[0].dstBinding = 6; w[0].descriptorCount = 1;
        w[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w[0].pImageInfo = &di;
        uint32_t n = 1;
        VkWriteDescriptorSetAccelerationStructureKHR as{};
        if (tlas_) {
            as.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
            as.accelerationStructureCount = 1; as.pAccelerationStructures = &tlas_;
            w[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w[1].pNext = &as;
            w[1].dstSet = L.set; w[1].dstBinding = 7; w[1].descriptorCount = 1;
            w[1].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
            n = 2;
        }
        vkUpdateDescriptorSets(device_, n, w, 0, nullptr);
        L.depth_bound = depth_view_;
        if (tlas_) L.tlas_bound = tlas_;
    }
    const FrameView& fv = current_frame_view_;
    const double now = glfwGetTime();
    const float dt = L.last_t < 0.0 ? 0.016f : float(std::clamp(now - L.last_t, 0.0, 0.1));
    L.last_t = now;
    ++L.frame;
    LensFxUBO u{};

    // ---- the moon ----
    {
        glm::vec3 sun_true, moon = env_->moon_dir;
        bool light_sun = true;
        float moon_k = 0.0f;
        if (rt_.day_night_enabled) {
            sun_true = env_->sun_dir; light_sun = env_->light_is_sun;
            // Its radiance against a full sun's light on the land (sun intensity x pi x
            // atmosphere): the moon's own albedo is about 0.12, made a little brighter than true
            // so it reads at night after the eye's adaptation; by day a pale disc.
            const float E = std::max(0.0f, env_->applied ? env_->base.sun_intensity : rt_.sun_intensity) * 3.14159265f *
                            std::max(0.0f, rt_.atmosphere_intensity);
            const float day = glm::smoothstep(-4.0f, 8.0f, env_->sun_elev);
            moon_k = E * glm::mix(0.10f, 0.035f, day) * glm::clamp(rt_.moon_brightness * 1.25f, 0.0f, 1.25f);
            static const float kMoonGain = std::getenv("VIBE_MOON_GAIN") ? float(std::atof(std::getenv("VIBE_MOON_GAIN"))) : 1.0f;
            moon_k *= kMoonGain;
        } else {
            const float p = glm::radians(rt_.sun_pitch_deg), y = glm::radians(rt_.sun_yaw_deg);
            sun_true = glm::vec3(std::sin(y) * std::cos(p), std::sin(p), std::cos(y) * std::cos(p));
        }
        u.moon_dir = glm::vec4(glm::normalize(moon), moon_k);
        u.sun_true = glm::vec4(glm::normalize(sun_true), light_sun ? 0.0f : 1.0f);
        static const float kMoonSize = std::getenv("VIBE_TEST_MOON_SIZE") ? float(std::atof(std::getenv("VIBE_TEST_MOON_SIZE"))) : 1.0f;   // (tests: a bigger disc to see its phase)
        u.moon_misc = glm::vec4(0.0047f * kMoonSize, 0.012f, 0.0f, 0.0f);     // (0.27 degrees: the real one)
    }

    // ---- rain on the lens ----
    static const float kTestRain = std::getenv("VIBE_TEST_RAIN_LENS") ? float(std::atof(std::getenv("VIBE_TEST_RAIN_LENS"))) : -1.0f;
    const glm::vec3 eye = fv.eye_pos;
    std::vector<glm::vec4> hot;
    const bool cabin = jeep_heat_sources(hot, eye);
    float rain = kTestRain >= 0.0f ? kTestRain : glm::clamp(env_->rain, 0.0f, 1.0f);
    if (env_->snowfall > 0.2f && kTestRain < 0.0f) rain *= 0.3f;            // (snow does not wet the glass as rain does)
    bool open = !cabin;
    if (rt_.water_enabled && eye.y < rt_.water_level + 0.05f && water_at(eye.x, eye.z)) open = false;
    if (open) {
        std::vector<Shelter> roofs;
        shelters_near(eye, 4.0f, roofs, 4);
        for (const Shelter& s : roofs)
            if (eye.x > s.lo.x && eye.x < s.hi.x && eye.z > s.lo.y && eye.z < s.hi.y && eye.y < s.top) { open = false; break; }
    }
    const bool lens_on = rt_.lens_rain && tlas_ != VK_NULL_HANDLE;
    if (rain > 0.01f) L.wet_left = 30.0f; else L.wet_left = std::max(0.0f, L.wet_left - dt);
    const bool lens_run = lens_on && L.wet_left > 0.0f;
    // Drops landing: up to ~25 a second in a downpour (the open-sky test on the GPU thins them
    // where the lens is sheltered - here they are only owed).
    int drops = 0;
    if (lens_run && open) {
        L.drop_acc += dt * 26.0f * rain * std::sqrt(rain);
        drops = std::min(12, int(L.drop_acc));
        L.drop_acc -= float(drops);
    }
    u.lens = glm::vec4(lens_on ? 1.0f : 0.0f, 0.0f, float(std::fmod(now, 3600.0)), float(L.frame % 100000u));
    u.rain = glm::vec4(rain, open ? 1.0f : 0.0f, dt, float(drops));
    u.cam = glm::vec4(eye, float(swapchain_extent_.width) / float(std::max(1u, swapchain_extent_.height)));
    u.inv_vp = fv.inv_vp;

    // ---- hot air ----
    {
        static const float kTestHot = std::getenv("VIBE_TEST_SHIMMER") ? float(std::atof(std::getenv("VIBE_TEST_SHIMMER"))) : 0.0f;
        if (kTestHot > 0.0f) {
            glm::vec3 f = player_.forward(); f.y = 0.0f;
            f = glm::normalize(f + glm::vec3(1e-4f));
            hot.push_back(glm::vec4(eye + f * 6.0f + glm::vec3(0.0f, -0.6f, 0.0f), 1.2f));
            hot.push_back(glm::vec4(1.6f * kTestHot, 1.5f, 1.8f, 9.0f));
        }
        // The rocket: its plume while it is low over the pad.
        if (launch_site_placed_ && rocket_.thr_vis > 0.01f && rocket_.alt < 700.0f && rocket_.stage != 3) {
            const glm::vec3 base = glm::vec3(rocket_.pos);
            const float k = rocket_.thr_vis * (1.0f - glm::smoothstep(150.0f, 700.0f, rocket_.alt));
            if (glm::distance(base, eye) < 2500.0f && k > 0.01f) {
                hot.push_back(glm::vec4(base + glm::vec3(0.0f, -6.0f, 0.0f), 22.0f));
                hot.push_back(glm::vec4(2.6f * k, 9.0f, 1.6f, 0.9f));
            }
        }
        // Explosions: the hot air over a fireball for a few seconds.
        for (const Explosion& e : explosions_) {
            if (e.t > 5.0f) continue;
            const float k = 1.0f - e.t / 5.0f;
            if (glm::distance(e.pos, eye) > 600.0f) continue;
            hot.push_back(glm::vec4(e.pos + glm::vec3(0.0f, 4.0f + 3.0f * e.size * std::min(e.t, 3.0f), 0.0f) + e.drift, 7.0f * e.size));
            hot.push_back(glm::vec4(2.0f * k, 5.0f, 1.5f, 1.6f / std::max(e.size, 0.3f)));
        }
        // Nearest first, at most kHotMax.
        const int n = int(hot.size() / 2);
        std::vector<int> ord(n);
        for (int i = 0; i < n; ++i) ord[i] = i;
        std::sort(ord.begin(), ord.end(), [&](int a, int b) {
            return glm::distance(glm::vec3(hot[2 * a]), eye) < glm::distance(glm::vec3(hot[2 * b]), eye); });
        const int m = std::min(n, kHotMax);
        for (int i = 0; i < m; ++i) { u.src[2 * i] = hot[2 * ord[i]]; u.src[2 * i + 1] = hot[2 * ord[i] + 1]; }
        // A hot road far off: a clear summer afternoon, the sun high, standing on or by a highway.
        float mirage = 0.0f;
        if (rt_.day_night_enabled && globe::current().home) {
            const float tod = std::fmod(rt_.time_of_day + 24.0f, 24.0f);
            const float aft = glm::smoothstep(11.5f, 13.5f, tod) * (1.0f - glm::smoothstep(17.0f, 19.0f, tod));
            const float clear = (1.0f - glm::smoothstep(0.15f, 0.5f, wx_cur_.overcast)) * (1.0f - glm::smoothstep(0.0f, 0.05f, env_->wet));
            const float summer = season_w_[1];
            const float high = glm::smoothstep(25.0f, 40.0f, env_->sun_elev);
            mirage = aft * clear * summer * high;
            if (mirage > 0.01f && world_highway_distance(eye.x, eye.z) > 14.0f) mirage = 0.0f;
        }
        static const float kMirage = std::getenv("VIBE_TEST_MIRAGE") ? float(std::atof(std::getenv("VIBE_TEST_MIRAGE"))) : -1.0f;
        if (kMirage >= 0.0f) mirage = kMirage;
        u.sh_info = glm::vec4(float(m), mirage, 0.0f, 0.0f);
        L.shimmer_on = m > 0 || mirage > 0.01f;
        u.lens.y = L.shimmer_on ? 1.0f : 0.0f;
    }

    if (std::getenv("VIBE_LOG_LENS")) {
        static double t_log = 0.0;
        if (now - t_log > 1.0) {
            t_log = now;
            // (the largest shimmer offset written since the last line, in output pixels; then cleared)
            float sh_px = 0.0f, sky_open = 0.0f;
            VmaAllocationInfo ai{}; vmaGetAllocationInfo(allocator_, L.state_alloc, &ai);
            if (ai.pMappedData) {
                vmaInvalidateAllocation(allocator_, L.state_alloc, 0, VK_WHOLE_SIZE);
                const float* fp = static_cast<const float*>(ai.pMappedData);
                uint32_t* up = reinterpret_cast<uint32_t*>(static_cast<char*>(ai.pMappedData) + 16);
                sky_open = fp[0];
                sh_px = float(*up) * 1e-6f * float(swapchain_extent_.width);
                *up = 0u;
                vmaFlushAllocation(allocator_, L.state_alloc, 0, VK_WHOLE_SIZE);
            }
            log::infof("[lens-fx] shimmer max offset %.1f px, sky over the lens %.2f", double(sh_px), double(sky_open));
            log::infof("[lens-fx] rain %.2f open %d cabin %d drops %d run %d | hot sources %d mirage %.2f | moon (%.2f %.2f %.2f) k %.3f, sun (%.2f %.2f %.2f) is light %d",
                       double(rain), int(open), int(cabin), drops, int(lens_run), int(u.sh_info.x), double(u.sh_info.y),
                       double(u.moon_dir.x), double(u.moon_dir.y), double(u.moon_dir.z), double(u.moon_dir.w),
                       double(u.sun_true.x), double(u.sun_true.y), double(u.sun_true.z), int(u.sun_true.w));
        }
    }

    // Upload (after everything that read the block last frame), then the passes.
    auto barrier = [&](VkPipelineStageFlags2 src, VkAccessFlags2 sa, VkPipelineStageFlags2 dst, VkAccessFlags2 da) {
        VkMemoryBarrier2 mb{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, nullptr, src, sa, dst, da };
        VkDependencyInfo di{}; di.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        di.memoryBarrierCount = 1; di.pMemoryBarriers = &mb;
        vkCmdPipelineBarrier2(cmd, &di);
    };
    barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
    vkCmdUpdateBuffer(cmd, L.ubo, 0, sizeof(u), &u);
    barrier(VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_UNIFORM_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
    if (!lens_run && !L.shimmer_on) return;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, L.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, L.pipeline_layout, 0, 1, &L.set, 0, nullptr);
    auto run = [&](float mode, uint32_t gx, uint32_t gy) {
        LensPC pc{ glm::vec4(mode, 0.0f, 0.0f, 0.0f) };
        vkCmdPushConstants(cmd, L.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, gx, gy, 1);
        barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT);
    };
    if (lens_run) {
        run(0.0f, 1, 1);
        run(1.0f, (L.lens_w + 7) / 8, (L.lens_h + 7) / 8);
        run(2.0f, (L.lens_w + 7) / 8, (L.lens_h + 7) / 8);
    }
    if (L.shimmer_on) run(3.0f, (L.sh_w + 7) / 8, (L.sh_h + 7) / 8);
    prof_mark(cmd, "lens_fx");
}

} // namespace vibe
