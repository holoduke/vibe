// TAA (temporal anti-aliasing) pass — history pingpong + motion-vector
// reproject. Lives in its own TU so post.cpp doesn't carry the TAA
// init/teardown plus the compose pass plus screenshot read-back.
//
// Still implemented as VulkanEngine member methods (init_taa,
// recreate_taa_targets, destroy_taa). State fields stay in vk_engine.h
// for now; a future pass can wrap them in a TaaPass RAII class once
// linear_sampler_/history_view_ ownership is disentangled from the
// compose pass that also samples them.

#include "engine/vk_engine/internal.h"
#include "engine/vk_engine/state/fsr2.h"
#include "engine/vk_engine/state/taau.h"
#include "engine/vk_engine/state/compose.h"
#include "engine/vk_engine/state/taa.h"
#include "engine/vk_engine/state/fsr3.h"
#include "engine/vk_pipelines.h"

#include <string>

namespace vibe {

void VulkanEngine::init_taa() {
    // Linear sampler for history+depth lookups. Owned by TAA but the
    // compose pass also samples through it (created first; compose's
    // init_compose() relies on it being live).
    {
        VkSamplerCreateInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        si.magFilter = VK_FILTER_LINEAR;
        si.minFilter = VK_FILTER_LINEAR;
        si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        si.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.maxLod = 0.0f;
        vk_check(vkCreateSampler(device_, &si, nullptr, &linear_sampler_),
                 "linear sampler");
    }

    // UBO buffer. The motion-vec-driven reproject path (svgf-tao-rewrite)
    // dropped the inverse/prev VP matrices that the depth-reconstruct path
    // used to need; only viewport + params remain.
    struct TaaUBO { glm::vec4 vp; glm::vec4 params; glm::mat4 inv_vp; glm::mat4 prev_vp; glm::vec4 eye; };   // == frame.cpp
    {
        VkBufferCreateInfo bci{
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .pNext = nullptr, .flags = 0,
            .size = sizeof(TaaUBO),
            .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,   // dst of the frame upload ring
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0, .pQueueFamilyIndices = nullptr,
        };
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
        aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                    VMA_ALLOCATION_CREATE_MAPPED_BIT;
        vk_check(vmaCreateBuffer(allocator_, &bci, &aci,
                                 &taa_->ubo_buffer, &taa_->ubo_alloc, nullptr),
                 "taa ubo");
    }

    // Descriptor pool: 2 sets (one per history slot we sample as the "previous"),
    // each with 4 sampled images (current_color + history_color + current_depth
    // + motion_vec) + 1 UBO.
    {
        VkDescriptorPoolSize sizes[] = {
            { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 12 },
            { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 2 },
        };
        VkDescriptorPoolCreateInfo pci{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .pNext = nullptr, .flags = 0,
            .maxSets = kHistorySlots, .poolSizeCount = 2, .pPoolSizes = sizes,
        };
        vk_check(vkCreateDescriptorPool(device_, &pci, nullptr, &taa_->desc_pool),
                 "taa desc pool");
    }

    // Set layout: 0 = current_color, 1 = history_color, 2 = current_depth,
    //             3 = UBO, 4 = motion vector image.
    // 5, 6 = the G-buffer (material and its marks: water pixels take less history, taa.frag).
    VkDescriptorSetLayoutBinding bindings[7]{};
    for (int i = 0; i < 3; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    }
    bindings[3].binding = 3;
    bindings[3].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings[3].descriptorCount = 1;
    bindings[3].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    bindings[4].binding = 4;
    bindings[4].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[4].descriptorCount = 1;
    bindings[4].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    for (int i = 5; i < 7; ++i) { bindings[i] = bindings[4]; bindings[i].binding = uint32_t(i); }

    VkDescriptorSetLayoutCreateInfo lci{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .pNext = nullptr, .flags = 0,
        .bindingCount = 7, .pBindings = bindings,
    };
    vk_check(vkCreateDescriptorSetLayout(device_, &lci, nullptr, &taa_->desc_set_layout),
             "taa set layout");

    VkDescriptorSetLayout layouts[kHistorySlots] = {
        taa_->desc_set_layout, taa_->desc_set_layout
    };
    VkDescriptorSetAllocateInfo dai{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .pNext = nullptr,
        .descriptorPool = taa_->desc_pool,
        .descriptorSetCount = kHistorySlots,
        .pSetLayouts = layouts,
    };
    vk_check(vkAllocateDescriptorSets(device_, &dai, taa_->desc_sets),
             "taa set alloc");

    // Pipeline layout (just the set, no push constants).
    VkPipelineLayoutCreateInfo plci{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext = nullptr, .flags = 0,
        .setLayoutCount = 1, .pSetLayouts = &taa_->desc_set_layout,
        .pushConstantRangeCount = 0, .pPushConstantRanges = nullptr,
    };
    vk_check(vkCreatePipelineLayout(device_, &plci, nullptr, &taa_->pipeline_layout),
             "taa pipeline layout");

    std::string sd = VIBE_SHADER_DIR;
    taa_->vert_module = vkpipe::load_shader_module(device_, sd + "/taa.vert.spv");
    taa_->frag_module = vkpipe::load_shader_module(device_, sd + "/taa.frag.spv");

    vkpipe::GraphicsPipelineConfig cfg{};
    cfg.vert = taa_->vert_module;
    cfg.frag = taa_->frag_module;
    cfg.layout = taa_->pipeline_layout;
    cfg.color_format = scene_color_format_;
    cfg.depth_format = VK_FORMAT_UNDEFINED;
    cfg.cull = VK_CULL_MODE_NONE;
    cfg.depth_test = false;
    cfg.depth_write = false;
    taa_->pipeline = vkpipe::build_graphics_pipeline(device_, cfg);

    recreate_taa_targets();
    log::info("TAA initialized");
}

void VulkanEngine::recreate_taa_targets() {
    // Drop existing targets.
    if (scene_color_view_)  vkDestroyImageView(device_, scene_color_view_, nullptr);
    if (scene_color_image_) vmaDestroyImage(allocator_, scene_color_image_, scene_color_alloc_);
    if (scene_copy_view_)   vkDestroyImageView(device_, scene_copy_view_, nullptr);
    if (scene_copy_image_)  vmaDestroyImage(allocator_, scene_copy_image_, scene_copy_alloc_);
    scene_copy_image_ = VK_NULL_HANDLE; scene_copy_view_ = VK_NULL_HANDLE; scene_copy_alloc_ = nullptr;
    scene_color_image_ = VK_NULL_HANDLE; scene_color_view_ = VK_NULL_HANDLE;
    scene_color_alloc_ = nullptr;
    if (motion_vec_view_)   vkDestroyImageView(device_, motion_vec_view_, nullptr);
    if (motion_vec_image_)  vmaDestroyImage(allocator_, motion_vec_image_, motion_vec_alloc_);
    motion_vec_image_ = VK_NULL_HANDLE; motion_vec_view_ = VK_NULL_HANDLE;
    motion_vec_alloc_ = nullptr;
    if (gbuffer0_view_)  vkDestroyImageView(device_, gbuffer0_view_, nullptr);
    if (gbuffer0_image_) vmaDestroyImage(allocator_, gbuffer0_image_, gbuffer0_alloc_);
    gbuffer0_image_ = VK_NULL_HANDLE; gbuffer0_view_ = VK_NULL_HANDLE;
    gbuffer0_alloc_ = nullptr;
    if (gbuffer1_view_)  vkDestroyImageView(device_, gbuffer1_view_, nullptr);
    if (gbuffer1_image_) vmaDestroyImage(allocator_, gbuffer1_image_, gbuffer1_alloc_);
    gbuffer1_image_ = VK_NULL_HANDLE; gbuffer1_view_ = VK_NULL_HANDLE;
    gbuffer1_alloc_ = nullptr;
    if (staging_color_view_)  vkDestroyImageView(device_, staging_color_view_, nullptr);
    if (staging_color_image_) vmaDestroyImage(allocator_, staging_color_image_, staging_color_alloc_);
    staging_color_image_ = VK_NULL_HANDLE; staging_color_view_ = VK_NULL_HANDLE;
    staging_color_alloc_ = nullptr;
    for (int s = 0; s < 2; ++s) {
        if (deferred_gi_hist_view_[s])
            vkDestroyImageView(device_, deferred_gi_hist_view_[s], nullptr);
        if (deferred_gi_hist_image_[s])
            vmaDestroyImage(allocator_, deferred_gi_hist_image_[s],
                            deferred_gi_hist_alloc_[s]);
        deferred_gi_hist_image_[s] = VK_NULL_HANDLE;
        deferred_gi_hist_view_[s]  = VK_NULL_HANDLE;
        deferred_gi_hist_alloc_[s] = nullptr;
        if (deferred_gi_fast_view_[s])
            vkDestroyImageView(device_, deferred_gi_fast_view_[s], nullptr);
        if (deferred_gi_fast_image_[s])
            vmaDestroyImage(allocator_, deferred_gi_fast_image_[s],
                            deferred_gi_fast_alloc_[s]);
        deferred_gi_fast_image_[s] = VK_NULL_HANDLE;
        deferred_gi_fast_view_[s]  = VK_NULL_HANDLE;
        deferred_gi_fast_alloc_[s] = nullptr;
    }
    for (int s = 0; s < kHistorySlots; ++s) {
        if (history_view_[s])  vkDestroyImageView(device_, history_view_[s], nullptr);
        if (history_image_[s]) vmaDestroyImage(allocator_, history_image_[s], history_alloc_[s]);
        history_image_[s] = VK_NULL_HANDLE; history_view_[s] = VK_NULL_HANDLE;
        history_alloc_[s] = nullptr;
    }

    auto make_image = [&](VkFormat fmt, VkImageUsageFlags usage,
                          VkImage& img, VmaAllocation& alloc, VkImageView& view) {
        VkImageCreateInfo ici{
            .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .pNext = nullptr, .flags = 0,
            .imageType = VK_IMAGE_TYPE_2D, .format = fmt,
            // Render-resolution targets: scene_color / motion_vec / history
            // all sized to render_extent_, not swapchain_extent_, so the
            // user's render_scale slider actually changes per-pixel cost.
            .extent = { render_extent_.width, render_extent_.height, 1 },
            .mipLevels = 1, .arrayLayers = 1,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .tiling = VK_IMAGE_TILING_OPTIMAL,
            .usage = usage,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0, .pQueueFamilyIndices = nullptr,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        };
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        vk_check(vmaCreateImage(allocator_, &ici, &aci, &img, &alloc, nullptr),
                 "taa image");
        VkImageViewCreateInfo vci{
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .pNext = nullptr, .flags = 0, .image = img,
            .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = fmt,
            .components = {},
            .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
        };
        vk_check(vkCreateImageView(device_, &vci, nullptr, &view),
                 "taa view");
    };

    make_image(scene_color_format_,
               VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
               VK_IMAGE_USAGE_SAMPLED_BIT |
               // transfer dst: kept for blit/clear-based debug paths.
               VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
               scene_color_image_, scene_color_alloc_, scene_color_view_);
    make_image(scene_color_format_, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
               scene_copy_image_, scene_copy_alloc_, scene_copy_view_);
    write_scene_copy_binding();
    make_image(motion_vec_format_,
               VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
               VK_IMAGE_USAGE_SAMPLED_BIT |
               VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,     // (ssdm.cpp copies it out)
               motion_vec_image_, motion_vec_alloc_, motion_vec_view_);
    // G-buffer (Phase 1 of the deferred migration). Allocated alongside
    // motion_vec since they're per-frame, render-resolution, and freed
    // together. Phase 1 keeps them ignored — Phase 2 wires the geometry
    // pass to fill them.
    make_image(gbuffer0_format_,
               VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
               VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,     // (ssdm.cpp copies the G-buffer out before it displaces it, and the displaced one back)
               gbuffer0_image_, gbuffer0_alloc_, gbuffer0_view_);
    make_image(gbuffer1_format_,
               VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
               VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
               gbuffer1_image_, gbuffer1_alloc_, gbuffer1_view_);
    // Staging colour for the deferred lighting pass (Phase 3).
    // TRANSFER_SRC_BIT is required because vk_engine.cpp blits this image
    // back into scene_color when the deferred toggle is on. Without it
    // vkCmdBlitImage's read silently returns zeros — every "pass-through"
    // pixel comes out as (0, low, low) instead of the forward HDR value.
    make_image(scene_color_format_,
               VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
               VK_IMAGE_USAGE_SAMPLED_BIT |
               VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
               staging_color_image_, staging_color_alloc_, staging_color_view_);
    for (int s = 0; s < kHistorySlots; ++s) {
        make_image(scene_color_format_,
                   VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                   VK_IMAGE_USAGE_SAMPLED_BIT |
                   VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                   history_image_[s], history_alloc_[s], history_view_[s]);
    }
    // Deferred GI temporal-denoise history (ping-pong storage images).
    // TRANSFER_DST so the init transition can CLEAR them — undefined
    // initial content (NaNs) would poison the EMA forever (NaN survives
    // every comparison-based rejection).
    for (int s = 0; s < 2; ++s) {
        make_image(VK_FORMAT_R16G16B16A16_SFLOAT,
                   VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                   VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                   deferred_gi_hist_image_[s], deferred_gi_hist_alloc_[s],
                   deferred_gi_hist_view_[s]);
        // Fast (3-frame) GI luminance history — the anti-ghost
        // fast/slow divergence detector. Same lifecycle as above.
        make_image(VK_FORMAT_R16_SFLOAT,
                   VK_IMAGE_USAGE_STORAGE_BIT |
                   VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                   deferred_gi_fast_image_[s], deferred_gi_fast_alloc_[s],
                   deferred_gi_fast_view_[s]);
    }

    for (int set = 0; set < kHistorySlots; ++set) {
        VkDescriptorImageInfo i_cur{ linear_sampler_, scene_color_view_,
                                      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        VkDescriptorImageInfo i_hist{ linear_sampler_,
                                       history_view_[1 - set],
                                       VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        VkDescriptorImageInfo i_depth{ linear_sampler_, depth_view_,
                                        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        VkDescriptorImageInfo i_motion{ linear_sampler_, motion_vec_view_,
                                         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        VkDescriptorImageInfo i_g0{ linear_sampler_, gbuffer0_view_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        VkDescriptorImageInfo i_g1{ linear_sampler_, gbuffer1_view_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        VkDescriptorBufferInfo i_ubo{ taa_->ubo_buffer, 0, VK_WHOLE_SIZE };

        VkWriteDescriptorSet w[7]{};
        for (int i = 0; i < 3; ++i) {
            w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[i].dstSet = taa_->desc_sets[set];
            w[i].dstBinding = i;
            w[i].descriptorCount = 1;
            w[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        }
        w[0].pImageInfo = &i_cur;
        w[1].pImageInfo = &i_hist;
        w[2].pImageInfo = &i_depth;
        w[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[3].dstSet = taa_->desc_sets[set];
        w[3].dstBinding = 3;
        w[3].descriptorCount = 1;
        w[3].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        w[3].pBufferInfo = &i_ubo;
        w[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[4].dstSet = taa_->desc_sets[set];
        w[4].dstBinding = 4;
        w[4].descriptorCount = 1;
        w[4].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        w[4].pImageInfo = &i_motion;
        w[5] = w[4]; w[5].dstBinding = 5; w[5].pImageInfo = &i_g0;
        w[6] = w[4]; w[6].dstBinding = 6; w[6].pImageInfo = &i_g1;
        vkUpdateDescriptorSets(device_, 7, w, 0, nullptr);
    }

    prev_view_proj_valid_ = false;

    vkinit::one_time_submit(device_, graphics_queue_, graphics_queue_family_,
                            [&](VkCommandBuffer cb) {
        for (int s = 0; s < kHistorySlots; ++s) {
            vkinit::transition_image(cb, history_image_[s],
                                     VK_IMAGE_LAYOUT_UNDEFINED,
                                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        }
        // Deferred GI history: clear to zero (M=0 → first frame starts
        // a fresh accumulation), then park in GENERAL permanently
        // (imageLoad/Store from the lighting pass; parity ping-pong +
        // the 2-frame fence overlap make per-frame barriers
        // unnecessary, same trick as cube.frag's SVGF history pair).
        for (int s = 0; s < 2; ++s) {
            vkinit::transition_image(cb, deferred_gi_hist_image_[s],
                                     VK_IMAGE_LAYOUT_UNDEFINED,
                                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            VkClearColorValue zero{};
            VkImageSubresourceRange rr{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
            vkCmdClearColorImage(cb, deferred_gi_hist_image_[s],
                                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                 &zero, 1, &rr);
            vkinit::transition_image(cb, deferred_gi_hist_image_[s],
                                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                     VK_IMAGE_LAYOUT_GENERAL);
            vkinit::transition_image(cb, deferred_gi_fast_image_[s],
                                     VK_IMAGE_LAYOUT_UNDEFINED,
                                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            // Clear to -1: the shader treats negative as "no fast
            // history yet" (0 is a legitimate luminance).
            VkClearColorValue neg{ { -1.0f, 0.0f, 0.0f, 0.0f } };
            vkCmdClearColorImage(cb, deferred_gi_fast_image_[s],
                                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                 &neg, 1, &rr);
            vkinit::transition_image(cb, deferred_gi_fast_image_[s],
                                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                     VK_IMAGE_LAYOUT_GENERAL);
        }
    });
}

void VulkanEngine::destroy_taa() {
    destroy_ssdm();          // (its targets are sized like these, its set points at them)
    if (taa_->pipeline)        vkDestroyPipeline(device_, taa_->pipeline, nullptr);
    if (taa_->pipeline_layout) vkDestroyPipelineLayout(device_, taa_->pipeline_layout, nullptr);
    if (taa_->vert_module)     vkDestroyShaderModule(device_, taa_->vert_module, nullptr);
    if (taa_->frag_module)     vkDestroyShaderModule(device_, taa_->frag_module, nullptr);
    if (taa_->desc_set_layout) vkDestroyDescriptorSetLayout(device_, taa_->desc_set_layout, nullptr);
    if (taa_->desc_pool)       vkDestroyDescriptorPool(device_, taa_->desc_pool, nullptr);
    if (taa_->ubo_buffer)      vmaDestroyBuffer(allocator_, taa_->ubo_buffer, taa_->ubo_alloc);
    if (linear_sampler_)      vkDestroySampler(device_, linear_sampler_, nullptr);

    for (int s = 0; s < kHistorySlots; ++s) {
        if (history_view_[s])  vkDestroyImageView(device_, history_view_[s], nullptr);
        if (history_image_[s]) vmaDestroyImage(allocator_, history_image_[s], history_alloc_[s]);
        history_image_[s] = VK_NULL_HANDLE;
        history_view_[s] = VK_NULL_HANDLE;
        history_alloc_[s] = nullptr;
    }
    if (scene_color_view_)  vkDestroyImageView(device_, scene_color_view_, nullptr);
    if (scene_color_image_) vmaDestroyImage(allocator_, scene_color_image_, scene_color_alloc_);
    if (scene_copy_view_)   vkDestroyImageView(device_, scene_copy_view_, nullptr);
    if (scene_copy_image_)  vmaDestroyImage(allocator_, scene_copy_image_, scene_copy_alloc_);
    scene_copy_image_ = VK_NULL_HANDLE; scene_copy_view_ = VK_NULL_HANDLE; scene_copy_alloc_ = nullptr;
    if (motion_vec_view_)   vkDestroyImageView(device_, motion_vec_view_, nullptr);
    if (motion_vec_image_)  vmaDestroyImage(allocator_, motion_vec_image_, motion_vec_alloc_);
    if (gbuffer0_view_)     vkDestroyImageView(device_, gbuffer0_view_, nullptr);
    if (gbuffer0_image_)    vmaDestroyImage(allocator_, gbuffer0_image_, gbuffer0_alloc_);
    if (gbuffer1_view_)     vkDestroyImageView(device_, gbuffer1_view_, nullptr);
    if (gbuffer1_image_)    vmaDestroyImage(allocator_, gbuffer1_image_, gbuffer1_alloc_);
    if (staging_color_view_)  vkDestroyImageView(device_, staging_color_view_, nullptr);
    if (staging_color_image_) vmaDestroyImage(allocator_, staging_color_image_, staging_color_alloc_);
    for (int s = 0; s < 2; ++s) {
        if (deferred_gi_hist_view_[s])
            vkDestroyImageView(device_, deferred_gi_hist_view_[s], nullptr);
        if (deferred_gi_hist_image_[s])
            vmaDestroyImage(allocator_, deferred_gi_hist_image_[s],
                            deferred_gi_hist_alloc_[s]);
        deferred_gi_hist_image_[s] = VK_NULL_HANDLE;
        deferred_gi_hist_view_[s]  = VK_NULL_HANDLE;
        deferred_gi_hist_alloc_[s] = nullptr;
        if (deferred_gi_fast_view_[s])
            vkDestroyImageView(device_, deferred_gi_fast_view_[s], nullptr);
        if (deferred_gi_fast_image_[s])
            vmaDestroyImage(allocator_, deferred_gi_fast_image_[s],
                            deferred_gi_fast_alloc_[s]);
        deferred_gi_fast_image_[s] = VK_NULL_HANDLE;
        deferred_gi_fast_view_[s]  = VK_NULL_HANDLE;
        deferred_gi_fast_alloc_[s] = nullptr;
    }

    taa_->pipeline = VK_NULL_HANDLE;
    taa_->pipeline_layout = VK_NULL_HANDLE;
    taa_->vert_module = taa_->frag_module = VK_NULL_HANDLE;
    taa_->desc_set_layout = VK_NULL_HANDLE;
    taa_->desc_pool = VK_NULL_HANDLE;
    taa_->ubo_buffer = VK_NULL_HANDLE;
    taa_->ubo_alloc = nullptr;
    linear_sampler_ = VK_NULL_HANDLE;
    scene_color_image_ = VK_NULL_HANDLE;
    scene_color_view_ = VK_NULL_HANDLE;
    scene_color_alloc_ = nullptr;
    motion_vec_image_ = VK_NULL_HANDLE;
    motion_vec_view_ = VK_NULL_HANDLE;
    motion_vec_alloc_ = nullptr;
}

// ============================================================================
// TAAU — temporal upsample pass. Lives alongside TAA because it shares the
// linear_sampler + reuses the LR history that TAA produces, but it runs in
// a separate pipeline at native swapchain_extent_ writing to taau_->image.
// ============================================================================

void VulkanEngine::init_taau() {
    // UBO: LR + native viewport sizes + params (history_blend, history_valid).
    struct TaauUBO { glm::vec4 lr_vp; glm::vec4 full_vp; glm::vec4 params; };
    VkBufferCreateInfo bci{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = nullptr, .flags = 0,
        .size = sizeof(TaauUBO),
        .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,   // dst of the frame upload ring
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    VmaAllocationCreateInfo bac{};
    bac.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
    bac.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                VMA_ALLOCATION_CREATE_MAPPED_BIT;
    vk_check(vmaCreateBuffer(allocator_, &bci, &bac,
                             &taau_->ubo_buffer, &taau_->ubo_alloc, nullptr),
             "taau ubo");

    // Descriptor pool: 2 sets, 4 sampled images + 1 UBO each.
    VkDescriptorPoolSize sizes[] = {
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 12 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 2 },
    };
    VkDescriptorPoolCreateInfo pci{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .pNext = nullptr, .flags = 0,
        .maxSets = kHistorySlots, .poolSizeCount = 2, .pPoolSizes = sizes,
    };
    vk_check(vkCreateDescriptorPool(device_, &pci, nullptr, &taau_->desc_pool),
             "taau desc pool");

    // Set layout: 0 cur_lr, 1 prev_full, 2 depth_lr, 3 ubo, 4 motion_lr, 5 / 6 the G-buffer
    // (water pixels keep less history: taau.frag).
    VkDescriptorSetLayoutBinding bindings[7]{};
    for (int i = 0; i < 7; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = (i == 3)
            ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
            : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    }
    VkDescriptorSetLayoutCreateInfo lci{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .pNext = nullptr, .flags = 0,
        .bindingCount = 7, .pBindings = bindings,
    };
    vk_check(vkCreateDescriptorSetLayout(device_, &lci, nullptr,
                                          &taau_->desc_set_layout),
             "taau set layout");

    VkDescriptorSetLayout layouts[kHistorySlots] = {
        taau_->desc_set_layout, taau_->desc_set_layout
    };
    VkDescriptorSetAllocateInfo dai{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .pNext = nullptr,
        .descriptorPool = taau_->desc_pool,
        .descriptorSetCount = kHistorySlots,
        .pSetLayouts = layouts,
    };
    vk_check(vkAllocateDescriptorSets(device_, &dai, taau_->desc_sets),
             "taau set alloc");

    VkPipelineLayoutCreateInfo plci{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext = nullptr, .flags = 0,
        .setLayoutCount = 1, .pSetLayouts = &taau_->desc_set_layout,
        .pushConstantRangeCount = 0, .pPushConstantRanges = nullptr,
    };
    vk_check(vkCreatePipelineLayout(device_, &plci, nullptr,
                                     &taau_->pipeline_layout),
             "taau pipeline layout");

    std::string sd = VIBE_SHADER_DIR;
    taau_->frag_module = vkpipe::load_shader_module(device_, sd + "/taau.frag.spv");

    vkpipe::GraphicsPipelineConfig cfg{};
    cfg.vert = taa_->vert_module;       // reuse fullscreen-tri vert
    cfg.frag = taau_->frag_module;
    cfg.layout = taau_->pipeline_layout;
    cfg.color_format = scene_color_format_;
    cfg.depth_format = VK_FORMAT_UNDEFINED;
    cfg.cull = VK_CULL_MODE_NONE;
    cfg.depth_test = false;
    cfg.depth_write = false;
    taau_->pipeline = vkpipe::build_graphics_pipeline(device_, cfg);

    recreate_taau_targets();
    log::info("TAAU initialized");
}

void VulkanEngine::recreate_taau_targets() {
    for (int s = 0; s < kHistorySlots; ++s) {
        if (taau_->view[s])  vkDestroyImageView(device_, taau_->view[s], nullptr);
        if (taau_->image[s]) vmaDestroyImage(allocator_, taau_->image[s], taau_->alloc[s]);
        taau_->image[s] = VK_NULL_HANDLE; taau_->view[s] = VK_NULL_HANDLE;
        taau_->alloc[s] = nullptr;
    }
    for (int s = 0; s < kHistorySlots; ++s) {
        VkImageCreateInfo ici{
            .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .pNext = nullptr, .flags = 0,
            .imageType = VK_IMAGE_TYPE_2D, .format = scene_color_format_,
            // Native swapchain extent — this is the upscaled output.
            .extent = { swapchain_extent_.width, swapchain_extent_.height, 1 },
            .mipLevels = 1, .arrayLayers = 1,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .tiling = VK_IMAGE_TILING_OPTIMAL,
            .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                     VK_IMAGE_USAGE_SAMPLED_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0, .pQueueFamilyIndices = nullptr,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        };
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        vk_check(vmaCreateImage(allocator_, &ici, &aci,
                                 &taau_->image[s], &taau_->alloc[s], nullptr),
                 "taau image");
        VkImageViewCreateInfo vci{
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .pNext = nullptr, .flags = 0, .image = taau_->image[s],
            .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = scene_color_format_,
            .components = {},
            .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
        };
        vk_check(vkCreateImageView(device_, &vci, nullptr, &taau_->view[s]),
                 "taau view");
    }

    // Wire descriptor sets. Set [s] is the one whose OUTPUT slot is s; it
    // samples prev_full from slot 1-s (the slot written last frame). LR
    // sources come from history_view_ at the same s.
    for (int s = 0; s < kHistorySlots; ++s) {
        VkDescriptorImageInfo i_cur   { linear_sampler_, history_view_[s],
                                         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        VkDescriptorImageInfo i_prev  { linear_sampler_, taau_->view[1 - s],
                                         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        VkDescriptorImageInfo i_depth { linear_sampler_, depth_view_,
                                         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        VkDescriptorImageInfo i_motion{ linear_sampler_, motion_vec_view_,
                                         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        VkDescriptorImageInfo i_g0{ linear_sampler_, gbuffer0_view_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        VkDescriptorImageInfo i_g1{ linear_sampler_, gbuffer1_view_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        VkDescriptorBufferInfo i_ubo  { taau_->ubo_buffer, 0, VK_WHOLE_SIZE };

        VkWriteDescriptorSet w[7]{};
        for (int i = 0; i < 7; ++i) {
            w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[i].dstSet = taau_->desc_sets[s];
            w[i].dstBinding = i;
            w[i].descriptorCount = 1;
            w[i].descriptorType = (i == 3)
                ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
                : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        }
        w[0].pImageInfo = &i_cur;
        w[1].pImageInfo = &i_prev;
        w[2].pImageInfo = &i_depth;
        w[3].pBufferInfo = &i_ubo;
        w[4].pImageInfo = &i_motion;
        w[5].pImageInfo = &i_g0;
        w[6].pImageInfo = &i_g1;
        vkUpdateDescriptorSets(device_, 7, w, 0, nullptr);
    }

    // Transition both ping-pong slots to SHADER_READ_ONLY so the prev-slot
    // sample on the first frame doesn't hit UNDEFINED.
    vkinit::one_time_submit(device_, graphics_queue_, graphics_queue_family_,
                            [&](VkCommandBuffer cb) {
        for (int s = 0; s < kHistorySlots; ++s) {
            vkinit::transition_image(cb, taau_->image[s],
                                      VK_IMAGE_LAYOUT_UNDEFINED,
                                      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        }
    });
    taau_->history_valid = false;
    compose_->uses_taau = false;
}

void VulkanEngine::destroy_taau() {
    if (taau_->pipeline)        vkDestroyPipeline(device_, taau_->pipeline, nullptr);
    if (taau_->pipeline_layout) vkDestroyPipelineLayout(device_, taau_->pipeline_layout, nullptr);
    if (taau_->frag_module)     vkDestroyShaderModule(device_, taau_->frag_module, nullptr);
    if (taau_->desc_set_layout) vkDestroyDescriptorSetLayout(device_, taau_->desc_set_layout, nullptr);
    if (taau_->desc_pool)       vkDestroyDescriptorPool(device_, taau_->desc_pool, nullptr);
    if (taau_->ubo_buffer)      vmaDestroyBuffer(allocator_, taau_->ubo_buffer, taau_->ubo_alloc);
    for (int s = 0; s < kHistorySlots; ++s) {
        if (taau_->view[s])  vkDestroyImageView(device_, taau_->view[s], nullptr);
        if (taau_->image[s]) vmaDestroyImage(allocator_, taau_->image[s], taau_->alloc[s]);
        taau_->image[s] = VK_NULL_HANDLE; taau_->view[s] = VK_NULL_HANDLE;
        taau_->alloc[s] = nullptr;
    }
    taau_->pipeline = VK_NULL_HANDLE;
    taau_->pipeline_layout = VK_NULL_HANDLE;
    taau_->frag_module = VK_NULL_HANDLE;
    taau_->desc_set_layout = VK_NULL_HANDLE;
    taau_->desc_pool = VK_NULL_HANDLE;
    taau_->ubo_buffer = VK_NULL_HANDLE;
    taau_->ubo_alloc = nullptr;
}

// Called from the per-frame settings tick when rt_.taau_enabled flips, or
// when TAAU targets get recreated (swapchain resize). Rebinds compose's
// history sampler to either history_view_ (TAAU off) or taau_->view (on).
// Idempotent — does nothing when bound state already matches.
void VulkanEngine::rewrite_compose_for_taau() {
    // Same predicate as rewrite_compose_image_bindings(): with FSR on, TAAU
    // never becomes the compose source, so comparing against the raw
    // toggle used to trigger vkDeviceWaitIdle + a rebind EVERY frame.
    const bool fsr3 = rt_.fsr2_enabled && rt_.fsr_backend == 1 && fsr3_->output_view != VK_NULL_HANDLE;
    const bool fsr2 = !fsr3 && rt_.fsr2_enabled && (fsr2_->output_view != VK_NULL_HANDLE || dlss_active());
    const bool want_taau = !fsr3 && !fsr2 && rt_.taau_enabled && taau_->image[0] != VK_NULL_HANDLE;
    if (want_taau == compose_->uses_taau) return;
    vkDeviceWaitIdle(device_);   // safest — compose desc may be in flight
    rewrite_compose_image_bindings();
    log::infof("compose source switched to %s", want_taau ? "TAAU" : "TAA");
}

} // namespace vibe
