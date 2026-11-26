// DLSS Ray Reconstruction guide buffers.
//
// DLSS-RR replaces the upscaler AND the denoisers: it takes the noisy
// ray-traced frame plus per-pixel material guides and resolves a clean,
// upscaled image. The guides it needs that the engine did not have as
// separate textures - diffuse albedo, specular albedo, world normal +
// roughness - are decoded from the G-buffer by rr_guides.comp right before
// the DLSS dispatch (dlss.cpp tags them).

#include "engine/vk_engine/internal.h"
#include "engine/vk_engine/state/rr_guides.h"
#include "engine/vk_initializers.h"
#include "engine/vk_pipelines.h"
#include "engine/log.h"

#include <cstdlib>
#include <string>

namespace vibe {

namespace {
constexpr VkFormat kRrFormats[3] = {
    VK_FORMAT_R8G8B8A8_UNORM,        // diffuse albedo
    VK_FORMAT_R8G8B8A8_UNORM,        // specular albedo
    VK_FORMAT_R16G16B16A16_SFLOAT,   // normal xyz + roughness
};
struct RrGuidesPC {
    glm::mat4 inv_vp;
    glm::vec4 cam_pos;
};
}

VkFormat VulkanEngine::rr_guide_format(int i) const { return kRrFormats[i]; }

void VulkanEngine::destroy_rr_guide_images() {
    for (int i = 0; i < 3; ++i) {
        if (rr_->view[i]) vkDestroyImageView(device_, rr_->view[i], nullptr);
        if (rr_->image[i]) vmaDestroyImage(allocator_, rr_->image[i], rr_->alloc[i]);
        rr_->view[i] = VK_NULL_HANDLE; rr_->image[i] = VK_NULL_HANDLE; rr_->alloc[i] = nullptr;
    }
    rr_->extent = { 0, 0 };
}

void VulkanEngine::destroy_rr_guides() {
    destroy_rr_guide_images();
    if (rr_->pipeline) vkDestroyPipeline(device_, rr_->pipeline, nullptr);
    if (rr_->pipeline_layout) vkDestroyPipelineLayout(device_, rr_->pipeline_layout, nullptr);
    if (rr_->pool) vkDestroyDescriptorPool(device_, rr_->pool, nullptr);
    if (rr_->set_layout) vkDestroyDescriptorSetLayout(device_, rr_->set_layout, nullptr);
    if (rr_->sampler) vkDestroySampler(device_, rr_->sampler, nullptr);
    rr_->pipeline = VK_NULL_HANDLE; rr_->pipeline_layout = VK_NULL_HANDLE;
    rr_->pool = VK_NULL_HANDLE; rr_->set_layout = VK_NULL_HANDLE; rr_->set = VK_NULL_HANDLE;
    rr_->sampler = VK_NULL_HANDLE;
}

// Called before command recording (ensure_dlss_bindings), with the GPU idle:
// (re)creates the pipeline once and the images whenever the render extent
// or the G-buffer views changed.
void VulkanEngine::ensure_rr_guides() {
    if (!rr_->pipeline) {
        VkSamplerCreateInfo sci{};
        sci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        sci.magFilter = VK_FILTER_NEAREST; sci.minFilter = VK_FILTER_NEAREST;
        sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        sci.addressModeU = sci.addressModeV = sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        vk_check(vkCreateSampler(device_, &sci, nullptr, &rr_->sampler), "rr sampler");

        VkDescriptorSetLayoutBinding b[7]{};
        for (int i = 0; i < 7; ++i) {
            b[i].binding = uint32_t(i);
            b[i].descriptorType = (i < 3 || i == 6) ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
                                        : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            b[i].descriptorCount = 1;
            b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo lci{};
        lci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        lci.bindingCount = 7; lci.pBindings = b;
        vk_check(vkCreateDescriptorSetLayout(device_, &lci, nullptr, &rr_->set_layout), "rr set layout");
        VkDescriptorPoolSize ps[2] = { { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 4 },
                                       { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 3 } };
        VkDescriptorPoolCreateInfo pci{};
        pci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pci.maxSets = 1; pci.poolSizeCount = 2; pci.pPoolSizes = ps;
        vk_check(vkCreateDescriptorPool(device_, &pci, nullptr, &rr_->pool), "rr pool");
        VkDescriptorSetAllocateInfo dai{};
        dai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dai.descriptorPool = rr_->pool; dai.descriptorSetCount = 1; dai.pSetLayouts = &rr_->set_layout;
        vk_check(vkAllocateDescriptorSets(device_, &dai, &rr_->set), "rr set");

        VkPushConstantRange pcr{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(RrGuidesPC) };
        VkPipelineLayoutCreateInfo plci{};
        plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount = 1; plci.pSetLayouts = &rr_->set_layout;
        plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pcr;
        vk_check(vkCreatePipelineLayout(device_, &plci, nullptr, &rr_->pipeline_layout), "rr layout");
        VkShaderModule m = vkpipe::load_shader_module(device_, std::string(VIBE_SHADER_DIR) + "/rr_guides.comp.spv");
        VkComputePipelineCreateInfo cci{};
        cci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        cci.stage.module = m;
        cci.stage.pName = "main";
        cci.layout = rr_->pipeline_layout;
        vk_check(vkCreateComputePipelines(device_, pipeline_cache_, 1, &cci, nullptr, &rr_->pipeline),
                 "rr guides pipeline");
        vkDestroyShaderModule(device_, m, nullptr);
    }

    const bool stale = rr_->extent.width != render_extent_.width ||
                       rr_->extent.height != render_extent_.height ||
                       rr_->src_views[0] != gbuffer0_view_ || rr_->src_views[1] != gbuffer1_view_ ||
                       rr_->src_views[2] != depth_view_ || rr_->src_views[3] != scene_color_view_;
    if (!stale) return;

    destroy_rr_guide_images();
    for (int i = 0; i < 3; ++i) {
        VkImageCreateInfo ici{};
        ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ici.imageType = VK_IMAGE_TYPE_2D;
        ici.format = kRrFormats[i];
        ici.extent = { render_extent_.width, render_extent_.height, 1 };
        ici.mipLevels = 1; ici.arrayLayers = 1;
        ici.samples = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling = VK_IMAGE_TILING_OPTIMAL;
        ici.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        vk_check(vmaCreateImage(allocator_, &ici, &aci, &rr_->image[i], &rr_->alloc[i], nullptr), "rr guide");
        VkImageViewCreateInfo vci{};
        vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vci.image = rr_->image[i];
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.format = kRrFormats[i];
        vci.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        vk_check(vkCreateImageView(device_, &vci, nullptr, &rr_->view[i]), "rr guide view");
    }
    vkinit::one_time_submit(device_, graphics_queue_, graphics_queue_family_, [&](VkCommandBuffer cb) {
        for (int i = 0; i < 3; ++i)
            vkinit::transition_image(cb, rr_->image[i], VK_IMAGE_LAYOUT_UNDEFINED,
                                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    });
    rr_->extent = render_extent_;
    rr_->src_views[0] = gbuffer0_view_; rr_->src_views[1] = gbuffer1_view_; rr_->src_views[2] = depth_view_;
    rr_->src_views[3] = scene_color_view_;

    VkDescriptorImageInfo ii[7] = {
        { rr_->sampler, gbuffer0_view_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL },
        { rr_->sampler, gbuffer1_view_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL },
        { rr_->sampler, depth_view_,    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL },
        { VK_NULL_HANDLE, rr_->view[0], VK_IMAGE_LAYOUT_GENERAL },
        { VK_NULL_HANDLE, rr_->view[1], VK_IMAGE_LAYOUT_GENERAL },
        { VK_NULL_HANDLE, rr_->view[2], VK_IMAGE_LAYOUT_GENERAL },
        { rr_->sampler, scene_color_view_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL },
    };
    VkWriteDescriptorSet w[7]{};
    for (int i = 0; i < 7; ++i) {
        w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[i].dstSet = rr_->set; w[i].dstBinding = uint32_t(i);
        w[i].descriptorCount = 1;
        w[i].descriptorType = (i < 3 || i == 6) ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
                                    : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        w[i].pImageInfo = &ii[i];
    }
    vkUpdateDescriptorSets(device_, 7, w, 0, nullptr);
    log::infof("[dlss-rr] guide buffers %ux%u", rr_->extent.width, rr_->extent.height);
}

// G-buffer + depth are SHADER_READ_ONLY here (post-world transition). The
// guides are left SHADER_READ_ONLY for the DLSS-RR evaluate.
void VulkanEngine::dispatch_rr_guides(VkCommandBuffer cmd) {
    if (!rr_->pipeline || !rr_->image[0]) return;
    for (int i = 0; i < 3; ++i)
        vkinit::transition_image(cmd, rr_->image[i], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                 VK_IMAGE_LAYOUT_GENERAL);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, rr_->pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, rr_->pipeline_layout, 0, 1, &rr_->set, 0, nullptr);
    RrGuidesPC pc{};
    pc.inv_vp = last_inv_view_proj_;
    { static const bool no_wtaa = std::getenv("VIBE_NO_WATER_TAA") != nullptr; pc.cam_pos = glm::vec4(current_frame_view_.eye_pos, no_wtaa ? 1.0f : 0.0f); }      // (w: 1 = water gets the plain guides)
    vkCmdPushConstants(cmd, rr_->pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, (rr_->extent.width + 7) / 8, (rr_->extent.height + 7) / 8, 1);
    for (int i = 0; i < 3; ++i)
        vkinit::transition_image(cmd, rr_->image[i], VK_IMAGE_LAYOUT_GENERAL,
                                 VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
}

} // namespace vibe
