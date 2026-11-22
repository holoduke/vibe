// Compose pass — samples TAA history (HDR), tonemaps, blends bloom +
// lens flare, encodes to swapchain (LDR). The TAA pass that produces
// that history lives in vk_engine/taa.cpp; the bloom mip chain that
// feeds binding 3 lives in vk_engine/bloom.cpp.

#include "engine/vk_engine/internal.h"
#include "engine/vk_engine/state/atmosphere.h"
#include "engine/vk_engine/state/bloom.h"
#include "engine/vk_engine/state/fsr2.h"
#include "engine/vk_engine/state/fog.h"
#include "engine/vk_engine/state/taau.h"
#include "engine/vk_engine/state/clouds.h"
#include "engine/vk_engine/state/compose.h"
#include "engine/vk_engine/state/taa.h"
#include "engine/vk_engine/state/fsr3.h"
#include "engine/vk_engine/state/sun_shadow.h"
#include "engine/vk_engine/state/dlss.h"
#include "engine/vk_pipelines.h"

#include <GLFW/glfw3.h>
#include <algorithm>

namespace vibe {

void VulkanEngine::init_compose() {
    // Compose set's binding 3 samples bloom_->mip_views[0], so init_bloom()
    // must have run first. linear_sampler_ comes from init_taa(). Both are
    // sequenced from VulkanEngine::init() — this method just assumes them.
    init_exposure();   // SSBO + compute; compose binding 4 reads the buffer
    init_atmosphere();  // LUTs + aerial volume; compose bindings 5-7 sample them
    {
        VkDescriptorPoolSize sizes[] = {
            { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, kHistorySlots * 9 },
            { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,         kHistorySlots },
        };
        VkDescriptorPoolCreateInfo pci{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .pNext = nullptr, .flags = 0,
            .maxSets = kHistorySlots, .poolSizeCount = 2, .pPoolSizes = sizes,
        };
        vk_check(vkCreateDescriptorPool(device_, &pci, nullptr, &compose_->desc_pool),
                 "compose desc pool");
    }
    {
        VkDescriptorSetLayoutBinding b[10]{};
        for (int i = 0; i < 10; ++i) {
            b[i].binding = i;
            b[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            b[i].descriptorCount = 1;
            b[i].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        }
        b[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;   // exposure SSBO
        VkDescriptorSetLayoutCreateInfo lci{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .pNext = nullptr, .flags = 0,
            .bindingCount = 10, .pBindings = b,
        };
        vk_check(vkCreateDescriptorSetLayout(device_, &lci, nullptr,
                                             &compose_->desc_set_layout),
                 "compose set layout");
    }
    {
        VkDescriptorSetLayout layouts[kHistorySlots] = {
            compose_->desc_set_layout, compose_->desc_set_layout
        };
        VkDescriptorSetAllocateInfo dai{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .pNext = nullptr,
            .descriptorPool = compose_->desc_pool,
            .descriptorSetCount = kHistorySlots, .pSetLayouts = layouts,
        };
        vk_check(vkAllocateDescriptorSets(device_, &dai, compose_->desc_sets),
                 "compose set alloc");
        rewrite_compose_image_bindings();
    }

    VkPushConstantRange compose_pc{};
    compose_pc.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    compose_pc.offset = 0;
    // 10 vec4 (viewport, bloom, sun_dir, sun_color, sky, flare, flare2,
    //          sun_screen, sharpen_params, brush_indicator) +
    //          mat4 inv_view_proj.
    compose_pc.size = sizeof(glm::vec4) * 12 + sizeof(glm::mat4);   // + atmo + tone (= 256 B)

    VkPipelineLayoutCreateInfo cplci{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext = nullptr, .flags = 0,
        .setLayoutCount = 1, .pSetLayouts = &compose_->desc_set_layout,
        .pushConstantRangeCount = 1, .pPushConstantRanges = &compose_pc,
    };
    vk_check(vkCreatePipelineLayout(device_, &cplci, nullptr, &compose_->pipeline_layout),
             "compose pipeline layout");

    {
        std::string sd2 = VIBE_SHADER_DIR;
        compose_->frag_module = vkpipe::load_shader_module(device_, sd2 + "/compose.frag.spv");
    }
    vkpipe::GraphicsPipelineConfig ccfg{};
    ccfg.vert = taa_->vert_module;       // reuse the fullscreen-triangle VS
    ccfg.frag = compose_->frag_module;
    ccfg.layout = compose_->pipeline_layout;
    ccfg.color_format = swapchain_format_;
    ccfg.depth_format = VK_FORMAT_UNDEFINED;
    ccfg.cull = VK_CULL_MODE_NONE;
    ccfg.depth_test = false;
    ccfg.depth_write = false;
    compose_->pipeline = vkpipe::build_graphics_pipeline(device_, ccfg);
    // Sky fill (compose.frag -DSKY_FILL): same layout / set / push
    // constants, renders into the render-res scene colour before DLSS.
    {
        std::string sd3 = VIBE_SHADER_DIR;
        sky_fill_frag_module_ = vkpipe::load_shader_module(device_, sd3 + "/sky_fill.frag.spv");
        vkpipe::GraphicsPipelineConfig scfg = ccfg;
        scfg.frag = sky_fill_frag_module_;
        scfg.color_format = scene_color_format_;
        // + the motion vectors of the sky pixels (the camera's turn): without them
        // the upscaler and frame generation treat the sky as standing still and
        // the clouds trail behind when looking around.
        scfg.color_formats = { scene_color_format_, motion_vec_format_ };
        scfg.dst_alpha_add_blend = true;
        sky_fill_pipeline_ = vkpipe::build_graphics_pipeline(device_, scfg);
        // Sun shafts before the upscaler: plain additive into the scene colour.
        sun_->shafts_frag_module = vkpipe::load_shader_module(device_, sd3 + "/sun_shafts.frag.spv");
        if (sun_->shafts_frag_module) {
            vkpipe::GraphicsPipelineConfig hcfg = ccfg;
            hcfg.frag = sun_->shafts_frag_module;
            hcfg.color_format = scene_color_format_;
            hcfg.additive_blend = true;
            sun_->shafts_pipeline = vkpipe::build_graphics_pipeline(device_, hcfg);
        }
        // The cloud layer in front of the land (camera inside / above it), before the upscaler.
        clouds_over_frag_module_ = vkpipe::load_shader_module(device_, sd3 + "/clouds_over.frag.spv");
        if (clouds_over_frag_module_) {
            vkpipe::GraphicsPipelineConfig ocfg = ccfg;
            ocfg.frag = clouds_over_frag_module_;
            ocfg.color_format = scene_color_format_;
            ocfg.premultiplied_blend = true;
            clouds_over_pipeline_ = vkpipe::build_graphics_pipeline(device_, ocfg);
        }
    }
    log::info("Compose pass initialized (ACES Fitted tonemap + sRGB encode)");
}

void VulkanEngine::rewrite_compose_image_bindings() {
    wait_gpu_frames_idle();   // compose set may be in use by the in-flight frame
    if (!compose_->desc_set_layout) return;
    // Four sources for compose's history binding, in priority order:
    //   FSR3  — HR upscaled output via ffx-api (FidelityFX SDK 1.1.4)
    //   FSR2  — HR upscaled output via legacy FSR2 static lib
    //   TAAU  — HR temporal upsample
    //   TAA   — LR temporal AA history (default)
    // FSR3 only wins when fsr2_enabled is on AND fsr_backend == 1; the
    // FSR2/FSR3 selection is mutually exclusive (one upscale per frame).
    const bool use_dlss = dlss_active();   // backend 2 (Streamline)
    const bool use_fsr3 = rt_.fsr2_enabled && rt_.fsr_backend == 1 &&
                          fsr3_->output_view != VK_NULL_HANDLE;
    const bool use_fsr2 = !use_fsr3 && rt_.fsr2_enabled &&
                          !use_dlss &&
                          fsr2_->output_view != VK_NULL_HANDLE;
    const bool use_taau = !use_fsr3 && !use_fsr2 && !use_dlss && rt_.taau_enabled &&
                          taau_->image[0] != VK_NULL_HANDLE;
    compose_->uses_fsr3 = use_fsr3;
    compose_->uses_fsr2 = use_fsr2;
    compose_->uses_taau = use_taau;
    compose_->uses_dlss = use_dlss;
    for (int s = 0; s < kHistorySlots; ++s) {
        VkImageView hist_src = use_dlss ? dlss_->output_view
                              : use_fsr3 ? fsr3_->output_view
                              : use_fsr2 ? fsr2_->output_view
                              : use_taau ? taau_->view[s]
                              : history_view_[s];
        VkDescriptorImageInfo i_hist{ linear_sampler_, hist_src,
                                       VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        VkDescriptorImageInfo i_depth{ linear_sampler_, depth_view_,
                                        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        VkDescriptorImageInfo i_sky { skybox_.sampler, skybox_.view,
                                        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        VkDescriptorImageInfo i_bloom{ linear_sampler_,
                                        bloom_->full_view ? bloom_->full_view
                                                          : bloom_->mip_views[0],
                                        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        VkWriteDescriptorSet w[4]{};
        w[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[0].dstSet = compose_->desc_sets[s];
        w[0].dstBinding = 0;
        w[0].descriptorCount = 1;
        w[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        w[0].pImageInfo = &i_hist;
        w[1] = w[0]; w[1].dstBinding = 1; w[1].pImageInfo = &i_depth;
        w[2] = w[0]; w[2].dstBinding = 2; w[2].pImageInfo = &i_sky;
        w[3] = w[0]; w[3].dstBinding = 3; w[3].pImageInfo = &i_bloom;
        VkDescriptorBufferInfo bi{ exposure_buf_, 0, VK_WHOLE_SIZE };
        VkWriteDescriptorSet wb = w[0];
        wb.dstBinding = 4;
        wb.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        wb.pImageInfo = nullptr;
        wb.pBufferInfo = &bi;
        vkUpdateDescriptorSets(device_, 4, w, 0, nullptr);
        if (exposure_buf_) vkUpdateDescriptorSets(device_, 1, &wb, 0, nullptr);
        // Atmosphere LUTs (GENERAL layout, see atmosphere.cpp).
        if (atmo_->sampler) {
            VkDescriptorImageInfo ai[5] = {
                { atmo_->sampler, atmo_->trans.view, VK_IMAGE_LAYOUT_GENERAL },
                { atmo_->sampler, atmo_->sky.view,   VK_IMAGE_LAYOUT_GENERAL },
                { atmo_->sampler, atmo_->ap.view,    VK_IMAGE_LAYOUT_GENERAL },
                { atmo_->sampler, cloud_->img.view ? cloud_->img.view : atmo_->sky.view, VK_IMAGE_LAYOUT_GENERAL },
                { atmo_->sampler, fog_->integ.view ? fog_->integ.view : atmo_->ap.view, VK_IMAGE_LAYOUT_GENERAL },
            };
            VkWriteDescriptorSet wa[5]{};
            for (int k = 0; k < 5; ++k) {
                wa[k] = w[0];
                wa[k].dstBinding = uint32_t(5 + k);
                wa[k].pImageInfo = &ai[k];
            }
            vkUpdateDescriptorSets(device_, 5, wa, 0, nullptr);
        }
        // Exposure compute reads the same image compose shows.
        if (exposure_sets_[s]) {
            VkWriteDescriptorSet we[3]{};
            we[0] = w[0]; we[0].dstSet = exposure_sets_[s]; we[0].dstBinding = 0;
            we[1] = w[1]; we[1].dstSet = exposure_sets_[s]; we[1].dstBinding = 1;
            we[2] = wb;   we[2].dstSet = exposure_sets_[s]; we[2].dstBinding = 2;
            vkUpdateDescriptorSets(device_, 3, we, 0, nullptr);
        }
    }
}

void VulkanEngine::destroy_compose() {
    destroy_exposure();
    destroy_atmosphere();
    if (sky_fill_pipeline_)        vkDestroyPipeline(device_, sky_fill_pipeline_, nullptr);
    if (sun_->shafts_pipeline)      vkDestroyPipeline(device_, sun_->shafts_pipeline, nullptr);
    if (clouds_over_pipeline_)     vkDestroyPipeline(device_, clouds_over_pipeline_, nullptr);
    if (clouds_over_frag_module_)  vkDestroyShaderModule(device_, clouds_over_frag_module_, nullptr);
    clouds_over_pipeline_ = VK_NULL_HANDLE; clouds_over_frag_module_ = VK_NULL_HANDLE;
    if (sun_->shafts_frag_module)   vkDestroyShaderModule(device_, sun_->shafts_frag_module, nullptr);
    sun_->shafts_pipeline = VK_NULL_HANDLE; sun_->shafts_frag_module = VK_NULL_HANDLE;
    if (sky_fill_frag_module_)     vkDestroyShaderModule(device_, sky_fill_frag_module_, nullptr);
    sky_fill_pipeline_ = VK_NULL_HANDLE; sky_fill_frag_module_ = VK_NULL_HANDLE;
    if (compose_->pipeline)         vkDestroyPipeline(device_, compose_->pipeline, nullptr);
    if (compose_->pipeline_layout)  vkDestroyPipelineLayout(device_, compose_->pipeline_layout, nullptr);
    if (compose_->frag_module)      vkDestroyShaderModule(device_, compose_->frag_module, nullptr);
    if (compose_->desc_set_layout)  vkDestroyDescriptorSetLayout(device_, compose_->desc_set_layout, nullptr);
    if (compose_->desc_pool)        vkDestroyDescriptorPool(device_, compose_->desc_pool, nullptr);
    compose_->pipeline         = VK_NULL_HANDLE;
    compose_->pipeline_layout  = VK_NULL_HANDLE;
    compose_->frag_module      = VK_NULL_HANDLE;
    compose_->desc_set_layout  = VK_NULL_HANDLE;
    compose_->desc_pool        = VK_NULL_HANDLE;
}


// ---------------------------------------------------------------------
// Auto-exposure (shaders/exposure.comp)
// ---------------------------------------------------------------------
void VulkanEngine::init_exposure() {
    {
        VkBufferCreateInfo bci{};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size = 544;   // + vec4 planet ... + the globe's frame and lattice (compose.frag), written every frame
        bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        vk_check(vmaCreateBuffer(allocator_, &bci, &aci, &exposure_buf_, &exposure_alloc_, nullptr),
                 "exposure buffer");
        vkinit::one_time_submit(device_, graphics_queue_, graphics_queue_family_, [&](VkCommandBuffer cb) {
            vkCmdFillBuffer(cb, exposure_buf_, 0, 112, 0u);   // valid = 0 -> first frame snaps
        });
    }
    {
        VkDescriptorSetLayoutBinding b[3]{};
        for (int i = 0; i < 3; ++i) {
            b[i].binding = uint32_t(i);
            b[i].descriptorType = (i == 2) ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
                                           : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            b[i].descriptorCount = 1;
            b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo lci{};
        lci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        lci.bindingCount = 3; lci.pBindings = b;
        vk_check(vkCreateDescriptorSetLayout(device_, &lci, nullptr, &exposure_set_layout_),
                 "exposure set layout");
        VkDescriptorPoolSize sizes[] = {
            { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, kHistorySlots * 2 },
            { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,         kHistorySlots },
        };
        VkDescriptorPoolCreateInfo pci{};
        pci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pci.maxSets = kHistorySlots; pci.poolSizeCount = 2; pci.pPoolSizes = sizes;
        vk_check(vkCreateDescriptorPool(device_, &pci, nullptr, &exposure_pool_), "exposure pool");
        VkDescriptorSetLayout layouts[kHistorySlots] = { exposure_set_layout_, exposure_set_layout_ };
        VkDescriptorSetAllocateInfo dai{};
        dai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dai.descriptorPool = exposure_pool_;
        dai.descriptorSetCount = kHistorySlots; dai.pSetLayouts = layouts;
        vk_check(vkAllocateDescriptorSets(device_, &dai, exposure_sets_), "exposure sets");
    }
    {
        VkPushConstantRange pc{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(float) * 4 };
        VkPipelineLayoutCreateInfo plci{};
        plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount = 1; plci.pSetLayouts = &exposure_set_layout_;
        plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pc;
        vk_check(vkCreatePipelineLayout(device_, &plci, nullptr, &exposure_pipeline_layout_),
                 "exposure pipeline layout");
        VkShaderModule m = vkpipe::load_shader_module(device_, std::string(VIBE_SHADER_DIR) + "/exposure.comp.spv");
        VkComputePipelineCreateInfo cci{};
        cci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        cci.stage.module = m;
        cci.stage.pName = "main";
        cci.layout = exposure_pipeline_layout_;
        vk_check(vkCreateComputePipelines(device_, pipeline_cache_, 1, &cci, nullptr, &exposure_pipeline_),
                 "exposure pipeline");
        vkDestroyShaderModule(device_, m, nullptr);
    }
    log::info("[exposure] luminance-reduction auto-exposure ready");
}

void VulkanEngine::destroy_exposure() {
    if (exposure_pipeline_) vkDestroyPipeline(device_, exposure_pipeline_, nullptr);
    if (exposure_pipeline_layout_) vkDestroyPipelineLayout(device_, exposure_pipeline_layout_, nullptr);
    if (exposure_pool_) vkDestroyDescriptorPool(device_, exposure_pool_, nullptr);
    if (exposure_set_layout_) vkDestroyDescriptorSetLayout(device_, exposure_set_layout_, nullptr);
    if (exposure_buf_) vmaDestroyBuffer(allocator_, exposure_buf_, exposure_alloc_);
    exposure_pipeline_ = VK_NULL_HANDLE; exposure_pipeline_layout_ = VK_NULL_HANDLE;
    exposure_pool_ = VK_NULL_HANDLE; exposure_set_layout_ = VK_NULL_HANDLE;
    exposure_buf_ = VK_NULL_HANDLE; exposure_alloc_ = nullptr;
    exposure_sets_[0] = exposure_sets_[1] = VK_NULL_HANDLE;
}

void VulkanEngine::dispatch_exposure(VkCommandBuffer cmd) {
    if (!exposure_pipeline_ || rt_.auto_exposure_strength <= 0.0f) return;
    const double now = glfwGetTime();
    float dt = exposure_last_t_ > 0.0 ? float(now - exposure_last_t_) : 0.0f;
    exposure_last_t_ = now;
    dt = std::min(dt, 0.25f);
    // Sky luminance estimate: the sky colour the compose pass paints.
    const glm::vec3 sky = rt_.sky_color;
    float sky_lum = 0.2126f * sky.r + 0.7152f * sky.g + 0.0722f * sky.b;
    if (skybox_.hdr && skybox_.avg_lum > 0.0f) sky_lum = skybox_.avg_lum;   // what compose paints
    sky_lum = std::max(0.05f, sky_lum);
    struct { float dt, up, down, sky; } pc{ dt, 2.5f, 0.9f, sky_lum };
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, exposure_pipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, exposure_pipeline_layout_, 0, 1,
                            &exposure_sets_[history_write_slot_], 0, nullptr);
    vkCmdPushConstants(cmd, exposure_pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, 1, 1, 1);
    VkMemoryBarrier2 mb{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, nullptr,
                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                         VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT };
    VkDependencyInfo di{}; di.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    di.memoryBarrierCount = 1; di.pMemoryBarriers = &mb;
    vkCmdPipelineBarrier2(cmd, &di);
}

} // namespace vibe
