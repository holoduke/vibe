// Physically based sky + aerial perspective (Hillaire 2020, "A Scalable
// and Production Ready Sky and Atmosphere Rendering Technique").
//
//   transmittance LUT  256x64   - once (atmosphere parameters are static)
//   multi-scatter LUT  32x32    - once
//   sky-view LUT       192x108  - every frame (sun / camera height)
//   aerial perspective 32^3     - every frame (camera frustum froxels)
//
// One compute pipeline (shaders/atmosphere.comp, pass in a push constant);
// every image stays in VK_IMAGE_LAYOUT_GENERAL so the same images are
// storage targets here and sampled textures in compose.frag.

#include "engine/vk_engine/internal.h"
#include "engine/vk_engine/state/atmosphere.h"
#include "engine/vk_engine/state/fog.h"
#include "engine/vk_engine/state/clouds.h"
#include "engine/vk_engine/state/environment.h"
#include "engine/vk_engine/state/sun_shadow.h"
#include "engine/vk_initializers.h"
#include "engine/vk_pipelines.h"
#include "engine/log.h"

#include <GLFW/glfw3.h>

#include <algorithm>
#include <cmath>

namespace vibe {

namespace {
constexpr uint32_t kTransW = 256, kTransH = 64;
constexpr uint32_t kMsW = 32, kMsH = 32;
constexpr uint32_t kSkyW = 192, kSkyH = 108;
constexpr uint32_t kApN = 32;
constexpr float kBottomRadiusKm = 6360.0f;

struct AtmoPC {
    glm::vec4 sun;     // xyz dir toward sun, w = pass
    glm::vec4 cam;     // xyz planet-space camera (km), w = aerial scale
    glm::vec4 eye;     // xyz world camera (m), w = AP max distance (km)
    glm::mat4 inv_vp;
};
}  // namespace

void VulkanEngine::init_atmosphere() {
    auto make = [&](AtmoImage& im, VkImageType type, uint32_t w, uint32_t h, uint32_t d, const char* name) {
        VkImageCreateInfo ici{};
        ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ici.imageType = type;
        ici.format = VK_FORMAT_R16G16B16A16_SFLOAT;
        ici.extent = { w, h, d };
        ici.mipLevels = 1; ici.arrayLayers = 1;
        ici.samples = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling = VK_IMAGE_TILING_OPTIMAL;
        ici.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        vk_check(vmaCreateImage(allocator_, &ici, &aci, &im.image, &im.alloc, nullptr), name);
        VkImageViewCreateInfo vci{};
        vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vci.image = im.image;
        vci.viewType = (type == VK_IMAGE_TYPE_3D) ? VK_IMAGE_VIEW_TYPE_3D : VK_IMAGE_VIEW_TYPE_2D;
        vci.format = ici.format;
        vci.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        vk_check(vkCreateImageView(device_, &vci, nullptr, &im.view), name);
    };
    make(atmo_->trans, VK_IMAGE_TYPE_2D, kTransW, kTransH, 1, "atmo transmittance");
    make(atmo_->ms,    VK_IMAGE_TYPE_2D, kMsW, kMsH, 1, "atmo multiscatter");
    make(atmo_->sky,   VK_IMAGE_TYPE_2D, kSkyW, kSkyH, 1, "atmo skyview");
    make(atmo_->ap,    VK_IMAGE_TYPE_3D, kApN, kApN, kApN, "atmo aerial");
    vkinit::one_time_submit(device_, graphics_queue_, graphics_queue_family_, [&](VkCommandBuffer cb) {
        for (AtmoImage* im : { &atmo_->trans, &atmo_->ms, &atmo_->sky, &atmo_->ap })
            vkinit::transition_image(cb, im->image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
        // Sky-view LUT is read by the raymarched terrain BEFORE this frame's
        // atmosphere dispatch: start it at zero, not undefined memory.
        VkMemoryBarrier2 mb{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, nullptr,
                             VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT,
                             VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                             VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT };
        VkDependencyInfo di{}; di.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        di.memoryBarrierCount = 1; di.pMemoryBarriers = &mb;
        vkCmdPipelineBarrier2(cb, &di);
        VkClearColorValue z{};
        VkImageSubresourceRange rr{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        vkCmdClearColorImage(cb, atmo_->sky.image, VK_IMAGE_LAYOUT_GENERAL, &z, 1, &rr);
        vkCmdPipelineBarrier2(cb, &di);
    });
    {
        VkSamplerCreateInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        si.magFilter = si.minFilter = VK_FILTER_LINEAR;
        si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.maxLod = 0.0f;
        vk_check(vkCreateSampler(device_, &si, nullptr, &atmo_->sampler), "atmo sampler");
    }
    {
        VkDescriptorSetLayoutBinding b[6]{};
        for (int i = 0; i < 6; ++i) {
            b[i].binding = uint32_t(i);
            b[i].descriptorType = (i < 2) ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
                                          : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            b[i].descriptorCount = 1;
            b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo lci{};
        lci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        lci.bindingCount = 6; lci.pBindings = b;
        vk_check(vkCreateDescriptorSetLayout(device_, &lci, nullptr, &atmo_->set_layout), "atmo set layout");
        VkDescriptorPoolSize sizes[] = {
            { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2 },
            { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          4 },
        };
        VkDescriptorPoolCreateInfo pci{};
        pci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pci.maxSets = 1; pci.poolSizeCount = 2; pci.pPoolSizes = sizes;
        vk_check(vkCreateDescriptorPool(device_, &pci, nullptr, &atmo_->pool), "atmo pool");
        VkDescriptorSetAllocateInfo dai{};
        dai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dai.descriptorPool = atmo_->pool; dai.descriptorSetCount = 1; dai.pSetLayouts = &atmo_->set_layout;
        vk_check(vkAllocateDescriptorSets(device_, &dai, &atmo_->set), "atmo set");

        VkDescriptorImageInfo ii[6] = {
            { atmo_->sampler, atmo_->trans.view, VK_IMAGE_LAYOUT_GENERAL },
            { atmo_->sampler, atmo_->ms.view,    VK_IMAGE_LAYOUT_GENERAL },
            { VK_NULL_HANDLE, atmo_->trans.view, VK_IMAGE_LAYOUT_GENERAL },
            { VK_NULL_HANDLE, atmo_->ms.view,    VK_IMAGE_LAYOUT_GENERAL },
            { VK_NULL_HANDLE, atmo_->sky.view,   VK_IMAGE_LAYOUT_GENERAL },
            { VK_NULL_HANDLE, atmo_->ap.view,    VK_IMAGE_LAYOUT_GENERAL },
        };
        VkWriteDescriptorSet w[6]{};
        for (int i = 0; i < 6; ++i) {
            w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[i].dstSet = atmo_->set;
            w[i].dstBinding = uint32_t(i);
            w[i].descriptorCount = 1;
            w[i].descriptorType = b[i].descriptorType;
            w[i].pImageInfo = &ii[i];
        }
        vkUpdateDescriptorSets(device_, 6, w, 0, nullptr);
    }
    {
        VkPushConstantRange pc{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(AtmoPC) };
        VkPipelineLayoutCreateInfo plci{};
        plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount = 1; plci.pSetLayouts = &atmo_->set_layout;
        plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pc;
        vk_check(vkCreatePipelineLayout(device_, &plci, nullptr, &atmo_->pipeline_layout), "atmo layout");
        VkShaderModule m = vkpipe::load_shader_module(device_, std::string(VIBE_SHADER_DIR) + "/atmosphere.comp.spv");
        VkComputePipelineCreateInfo cci{};
        cci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        cci.stage.module = m;
        cci.stage.pName = "main";
        cci.layout = atmo_->pipeline_layout;
        vk_check(vkCreateComputePipelines(device_, pipeline_cache_, 1, &cci, nullptr, &atmo_->pipeline),
                 "atmo pipeline");
        vkDestroyShaderModule(device_, m, nullptr);
    }
    // Scene set binding 30: the sky-view LUT for raymarched terrain / water.
    if (scene_desc_set_) {
        VkDescriptorImageInfo si{ atmo_->sampler, atmo_->sky.view, VK_IMAGE_LAYOUT_GENERAL };
        VkWriteDescriptorSet w{};
        w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet = scene_desc_set_; w.dstBinding = 30; w.descriptorCount = 1;
        w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        w.pImageInfo = &si;
        vkUpdateDescriptorSets(device_, 1, &w, 0, nullptr);
    }
    atmo_->luts_ready = false;
    log::info("[atmosphere] Hillaire 2020 sky: LUTs + aerial perspective ready");
    init_clouds();
}

// ---------------------------------------------------------------------
// Volumetric clouds (shaders/clouds.comp)
// ---------------------------------------------------------------------
namespace {
constexpr uint32_t kCloudW = 1920, kCloudH = 1080;
constexpr uint32_t kShapeN = 128, kDetailN = 64;
struct CloudPC {
    glm::mat4 inv_vp;
    glm::vec4 eye;      // xyz camera (m), w time (s)
    glm::vec4 sun;      // xyz toward sun, w mode
    glm::vec4 params;   // coverage, density, wind m/s, sun intensity (engine * pi)
    glm::vec4 params2;  // frame, camera planet radius (km), direct-light factor, cloud size
    glm::vec4 params3;  // cloud base altitude (m), tallest cloud (m), edge softness, erosion
    glm::vec4 light_col; // direct light colour
};
}  // namespace

// Cloud shadow map. Created by whoever needs it first (the cloud pass that
// writes it, or the scene / deferred sets that sample it); the scene set is
// written once it exists.
void VulkanEngine::init_cloud_shadow_map() {
    if (!cloud_->shadow_map.image) {
        VkImageCreateInfo ici{};
        ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ici.imageType = VK_IMAGE_TYPE_2D;
        ici.format = VK_FORMAT_R16_SFLOAT;
        ici.extent = { 512, 512, 1 };
        ici.mipLevels = 1; ici.arrayLayers = 1;
        ici.samples = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling = VK_IMAGE_TILING_OPTIMAL;
        ici.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        vk_check(vmaCreateImage(allocator_, &ici, &aci, &cloud_->shadow_map.image, &cloud_->shadow_map.alloc, nullptr), "cloud shadow map");
        VkImageViewCreateInfo vci{};
        vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vci.image = cloud_->shadow_map.image;
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.format = VK_FORMAT_R16_SFLOAT;
        vci.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        vk_check(vkCreateImageView(device_, &vci, nullptr, &cloud_->shadow_map.view), "cloud shadow map view");
        VkSamplerCreateInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        si.magFilter = si.minFilter = VK_FILTER_LINEAR;
        si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.maxLod = 0.0f;
        vk_check(vkCreateSampler(device_, &si, nullptr, &cloud_->shadow_sampler), "cloud shadow sampler");
        vkinit::one_time_submit(device_, graphics_queue_, graphics_queue_family_, [&](VkCommandBuffer cb) {
            vkinit::transition_image(cb, cloud_->shadow_map.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
            VkClearColorValue one{};
            one.float32[0] = one.float32[1] = one.float32[2] = one.float32[3] = 1.0f;     // open sky
            VkImageSubresourceRange r{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
            vkCmdClearColorImage(cb, cloud_->shadow_map.image, VK_IMAGE_LAYOUT_GENERAL, &one, 1, &r);
        });
    }
    if (scene_desc_set_ && !cloud_->shadow_bound) {
        VkDescriptorImageInfo ii{ cloud_->shadow_sampler, cloud_->shadow_map.view, VK_IMAGE_LAYOUT_GENERAL };
        VkWriteDescriptorSet w{};
        w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet = scene_desc_set_; w.dstBinding = 32; w.descriptorCount = 1;
        w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        w.pImageInfo = &ii;
        vkUpdateDescriptorSets(device_, 1, &w, 0, nullptr);
        cloud_->shadow_bound = true;
    }
}

void VulkanEngine::destroy_cloud_shadow_map() {
    if (cloud_->shadow_sampler) vkDestroySampler(device_, cloud_->shadow_sampler, nullptr);
    if (cloud_->shadow_map.view) vkDestroyImageView(device_, cloud_->shadow_map.view, nullptr);
    if (cloud_->shadow_map.image) vmaDestroyImage(allocator_, cloud_->shadow_map.image, cloud_->shadow_map.alloc);
    cloud_->shadow_map = {};
    cloud_->shadow_sampler = VK_NULL_HANDLE;
    cloud_->shadow_bound = false;
}

void VulkanEngine::init_clouds() {
    init_cloud_shadow_map();
    auto make = [&](AtmoImage& im, VkImageType type, VkFormat fmt, uint32_t w, uint32_t h, uint32_t d,
                    const char* name) {
        VkImageCreateInfo ici{};
        ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ici.imageType = type;
        ici.format = fmt;
        ici.extent = { w, h, d };
        ici.mipLevels = 1; ici.arrayLayers = 1;
        ici.samples = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling = VK_IMAGE_TILING_OPTIMAL;
        ici.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        vk_check(vmaCreateImage(allocator_, &ici, &aci, &im.image, &im.alloc, nullptr), name);
        VkImageViewCreateInfo vci{};
        vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vci.image = im.image;
        vci.viewType = (type == VK_IMAGE_TYPE_3D) ? VK_IMAGE_VIEW_TYPE_3D : VK_IMAGE_VIEW_TYPE_2D;
        vci.format = fmt;
        vci.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        vk_check(vkCreateImageView(device_, &vci, nullptr, &im.view), name);
    };
    {
        VkSamplerCreateInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        si.magFilter = si.minFilter = VK_FILTER_LINEAR;
        si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        si.maxLod = 0.0f;
        vk_check(vkCreateSampler(device_, &si, nullptr, &cloud_->sampler), "cloud sampler");
    }
    make(cloud_->img,    VK_IMAGE_TYPE_2D, VK_FORMAT_R16G16B16A16_SFLOAT, kCloudW, kCloudH, 1, "cloud layer");
    make(cloud_->shape,  VK_IMAGE_TYPE_3D, VK_FORMAT_R16_SFLOAT, kShapeN, kShapeN, kShapeN, "cloud shape noise");
    make(cloud_->detail, VK_IMAGE_TYPE_3D, VK_FORMAT_R16_SFLOAT, kDetailN, kDetailN, kDetailN, "cloud detail noise");
    vkinit::one_time_submit(device_, graphics_queue_, graphics_queue_family_, [&](VkCommandBuffer cb) {
        for (AtmoImage* im : { &cloud_->img, &cloud_->shape, &cloud_->detail })
            vkinit::transition_image(cb, im->image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
        VkClearColorValue z{};
        VkImageSubresourceRange r{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        VkMemoryBarrier2 mb{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, nullptr,
                             VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT,
                             VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                             VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT };
        VkDependencyInfo di{}; di.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        di.memoryBarrierCount = 1; di.pMemoryBarriers = &mb;
        vkCmdPipelineBarrier2(cb, &di);
        vkCmdClearColorImage(cb, cloud_->img.image, VK_IMAGE_LAYOUT_GENERAL, &z, 1, &r);
        vkCmdPipelineBarrier2(cb, &di);
    });
    {
        VkDescriptorSetLayoutBinding b[9]{};
        for (int i = 0; i < 9; ++i) {
            b[i].binding = uint32_t(i);
            b[i].descriptorType = (i < 5) ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
                                          : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            b[i].descriptorCount = 1;
            b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo lci{};
        lci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        lci.bindingCount = 9; lci.pBindings = b;
        vk_check(vkCreateDescriptorSetLayout(device_, &lci, nullptr, &cloud_->set_layout), "cloud set layout");
        VkDescriptorPoolSize sizes[] = {
            { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 5 },
            { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          4 },
        };
        VkDescriptorPoolCreateInfo pci{};
        pci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pci.maxSets = 1; pci.poolSizeCount = 2; pci.pPoolSizes = sizes;
        vk_check(vkCreateDescriptorPool(device_, &pci, nullptr, &cloud_->pool), "cloud pool");
        VkDescriptorSetAllocateInfo dai{};
        dai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dai.descriptorPool = cloud_->pool; dai.descriptorSetCount = 1; dai.pSetLayouts = &cloud_->set_layout;
        vk_check(vkAllocateDescriptorSets(device_, &dai, &cloud_->set), "cloud set");
        // Binding 2 (scene depth) is written every dispatch (view changes on resize).
        VkDescriptorImageInfo ii[9] = {
            { atmo_->sampler, atmo_->trans.view,   VK_IMAGE_LAYOUT_GENERAL },
            { atmo_->sampler, atmo_->sky.view,     VK_IMAGE_LAYOUT_GENERAL },
            { atmo_->sampler, VK_NULL_HANDLE,     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL },
            { cloud_->sampler, cloud_->shape.view, VK_IMAGE_LAYOUT_GENERAL },
            { cloud_->sampler, cloud_->detail.view, VK_IMAGE_LAYOUT_GENERAL },
            { VK_NULL_HANDLE, cloud_->img.view,    VK_IMAGE_LAYOUT_GENERAL },
            { VK_NULL_HANDLE, cloud_->shape.view,  VK_IMAGE_LAYOUT_GENERAL },
            { VK_NULL_HANDLE, cloud_->detail.view, VK_IMAGE_LAYOUT_GENERAL },
            { VK_NULL_HANDLE, cloud_->shadow_map.view, VK_IMAGE_LAYOUT_GENERAL },
        };
        VkWriteDescriptorSet w[8]{};
        int n = 0;
        for (int i = 0; i < 9; ++i) {
            if (i == 2) continue;
            w[n].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[n].dstSet = cloud_->set;
            w[n].dstBinding = uint32_t(i);
            w[n].descriptorCount = 1;
            w[n].descriptorType = b[i].descriptorType;
            w[n].pImageInfo = &ii[i];
            ++n;
        }
        vkUpdateDescriptorSets(device_, uint32_t(n), w, 0, nullptr);
    }
    {
        VkPushConstantRange pc{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(CloudPC) };
        VkPipelineLayoutCreateInfo plci{};
        plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount = 1; plci.pSetLayouts = &cloud_->set_layout;
        plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pc;
        vk_check(vkCreatePipelineLayout(device_, &plci, nullptr, &cloud_->pipeline_layout), "cloud layout");
        VkShaderModule m = vkpipe::load_shader_module(device_, std::string(VIBE_SHADER_DIR) + "/clouds.comp.spv");
        VkComputePipelineCreateInfo cci{};
        cci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        cci.stage.module = m;
        cci.stage.pName = "main";
        cci.layout = cloud_->pipeline_layout;
        vk_check(vkCreateComputePipelines(device_, pipeline_cache_, 1, &cci, nullptr, &cloud_->pipeline),
                 "cloud pipeline");
        vkDestroyShaderModule(device_, m, nullptr);
    }
    cloud_->noise_ready = false;
    log::info("[clouds] volumetric cloud layer ready (1920x1080, 128^3 + 64^3 noise)");
    init_fog();
}

void VulkanEngine::destroy_clouds() {
    destroy_fog();
    if (cloud_->sampler) vkDestroySampler(device_, cloud_->sampler, nullptr);
    cloud_->sampler = VK_NULL_HANDLE;
    if (cloud_->pipeline) vkDestroyPipeline(device_, cloud_->pipeline, nullptr);
    if (cloud_->pipeline_layout) vkDestroyPipelineLayout(device_, cloud_->pipeline_layout, nullptr);
    if (cloud_->pool) vkDestroyDescriptorPool(device_, cloud_->pool, nullptr);
    if (cloud_->set_layout) vkDestroyDescriptorSetLayout(device_, cloud_->set_layout, nullptr);
    for (AtmoImage* im : { &cloud_->img, &cloud_->shape, &cloud_->detail }) {
        if (im->view) vkDestroyImageView(device_, im->view, nullptr);
        if (im->image) vmaDestroyImage(allocator_, im->image, im->alloc);
        *im = {};
    }
    cloud_->pipeline = VK_NULL_HANDLE; cloud_->pipeline_layout = VK_NULL_HANDLE;
    cloud_->pool = VK_NULL_HANDLE; cloud_->set_layout = VK_NULL_HANDLE; cloud_->set = VK_NULL_HANDLE;
}

void VulkanEngine::dispatch_clouds(VkCommandBuffer cmd) {
    if (!cloud_->pipeline || !depth_view_) return;
    if (cloud_->depth_bound != depth_view_) {   // depth view changes only on resize
        wait_gpu_frames_idle();
        cloud_->depth_bound = depth_view_;
        VkDescriptorImageInfo di{ atmo_->sampler, depth_view_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        VkWriteDescriptorSet w{};
        w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet = cloud_->set; w.dstBinding = 2; w.descriptorCount = 1;
        w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        w.pImageInfo = &di;
        vkUpdateDescriptorSets(device_, 1, &w, 0, nullptr);
    }
    const FrameView& fv = current_frame_view_;
    float p = glm::radians(rt_.sun_pitch_deg), y = glm::radians(rt_.sun_yaw_deg);
    glm::vec3 sun(std::sin(y) * std::cos(p), std::sin(p), std::cos(y) * std::cos(p));
    CloudPC pc{};
    pc.inv_vp = fv.inv_vp;
    // eye.w x params.z = metres the wind has carried the clouds (added up in descriptors.cpp).
    pc.eye = glm::vec4(fv.eye_pos, float(cloud_->travel));
    const float cl_top = glm::clamp(rt_.cloud_base_m, 400.0f, 8000.0f) + glm::clamp(rt_.cloud_tower_m, 300.0f, 12000.0f);
    // (from above the layer the clouds lie over the land below. The march is made for clouds within some
    // tens of km: from much higher they turn to blocks, so they fade out between 4 and 14 km above the
    // layer's top; the planet has its own painted weather, compose.frag planet_surface)
    const float cl_space = 1.0f - glm::smoothstep(cl_top + 4000.0f, cl_top + 14000.0f, fv.eye_pos.y);
    pc.params = glm::vec4(glm::clamp(rt_.cloud_coverage, 0.0f, 1.0f) * cl_space,
                          std::max(0.0f, rt_.cloud_density) * cl_space,
                          1.0f,
                          (env_->applied ? env_->base.atmosphere_intensity * env_->cloud_k : rt_.atmosphere_intensity) * 3.14159265f *
                          // Night: clouds are lit by the moon and a dark sky only - less than half
                          // of what the sky model alone gave them (they glowed pale).
                          glm::mix(1.0f, 0.40f, glm::clamp(env_->night, 0.0f, 1.0f)));
    pc.params2 = glm::vec4(float(frame_number_ % 64), kBottomRadiusKm + std::max(fv.eye_pos.y, 1.0f) * 0.001f,
                           env_->cloud_direct, glm::clamp(rt_.cloud_scale, 0.3f, 4.0f));
    pc.params3 = glm::vec4(glm::clamp(rt_.cloud_base_m, 400.0f, 8000.0f), glm::clamp(rt_.cloud_tower_m, 300.0f, 12000.0f),
                           glm::clamp(rt_.cloud_softness, 0.01f, 1.0f), glm::clamp(rt_.cloud_erosion, 0.0f, 1.0f));
    pc.light_col = glm::vec4(rt_.sun_color, wx_cur_.overcast);   // w: how grey the deck is (rain clouds are grey, not sky-blue)
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cloud_->pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cloud_->pipeline_layout, 0, 1, &cloud_->set, 0, nullptr);
    auto barrier = [&]() {
        VkMemoryBarrier2 mb{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, nullptr,
                             VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                             VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                             VK_ACCESS_2_SHADER_SAMPLED_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                             VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT };
        VkDependencyInfo dinfo{}; dinfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dinfo.memoryBarrierCount = 1; dinfo.pMemoryBarriers = &mb;
        vkCmdPipelineBarrier2(cmd, &dinfo);
    };
    auto run = [&](float mode, uint32_t gx, uint32_t gy, uint32_t gz) {
        pc.sun = glm::vec4(sun, mode);
        vkCmdPushConstants(cmd, cloud_->pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, gx, gy, gz);
        barrier();
    };
    if (!cloud_->noise_ready) {
        run(0.0f, kShapeN / 8, kShapeN / 8, kShapeN);
        run(1.0f, kDetailN / 8, kDetailN / 8, kDetailN);
        cloud_->noise_ready = true;
    }
    run(2.0f, (cloud_->w + 7) / 8, (cloud_->h + 7) / 8, 1);
    run(3.0f, 512 / 8, 512 / 8, 1);          // sunlight reaching the ground (cloud shadow map)
}

// The cloud layer was a fixed 1920x1080 - 1.76x the pixels of a 1720x685
// internal frame it is sampled into 1:1 (sky fill runs at render
// resolution, before DLSS). Size it to the render extent instead; larger
// render extents are scaled down to the old pixel budget.
void VulkanEngine::ensure_cloud_target() {
    if (!cloud_->pipeline || !cloud_->img.image) return;
    uint32_t w = std::max(16u, render_extent_.width), h = std::max(16u, render_extent_.height);
    const double budget = double(kCloudW) * double(kCloudH);
    if (double(w) * double(h) > budget) {
        const double k = std::sqrt(budget / (double(w) * double(h)));
        w = std::max(16u, uint32_t(double(w) * k));
        h = std::max(16u, uint32_t(double(h) * k));
    }
    if (w == cloud_->w && h == cloud_->h) return;
    wait_gpu_frames_idle();
    if (cloud_->img.view) vkDestroyImageView(device_, cloud_->img.view, nullptr);
    if (cloud_->img.image) vmaDestroyImage(allocator_, cloud_->img.image, cloud_->img.alloc);
    cloud_->img = {};
    VkImageCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    ici.extent = { w, h, 1 };
    ici.mipLevels = 1; ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    vk_check(vmaCreateImage(allocator_, &ici, &aci, &cloud_->img.image, &cloud_->img.alloc, nullptr), "cloud layer");
    VkImageViewCreateInfo vci{};
    vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vci.image = cloud_->img.image;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = ici.format;
    vci.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    vk_check(vkCreateImageView(device_, &vci, nullptr, &cloud_->img.view), "cloud layer view");
    vkinit::one_time_submit(device_, graphics_queue_, graphics_queue_family_, [&](VkCommandBuffer cb) {
        vkinit::transition_image(cb, cloud_->img.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
        VkClearColorValue z{};
        VkImageSubresourceRange r{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        vkCmdClearColorImage(cb, cloud_->img.image, VK_IMAGE_LAYOUT_GENERAL, &z, 1, &r);
    });
    VkDescriptorImageInfo ii{ VK_NULL_HANDLE, cloud_->img.view, VK_IMAGE_LAYOUT_GENERAL };
    VkWriteDescriptorSet wr{};
    wr.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    wr.dstSet = cloud_->set; wr.dstBinding = 5; wr.descriptorCount = 1;
    wr.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    wr.pImageInfo = &ii;
    vkUpdateDescriptorSets(device_, 1, &wr, 0, nullptr);
    cloud_->w = w; cloud_->h = h;
    rewrite_compose_image_bindings();     // compose / sky fill sample it (binding 8)
    log::infof("[clouds] layer %ux%u (render extent %ux%u)", w, h, render_extent_.width, render_extent_.height);
}

void VulkanEngine::destroy_atmosphere() {
    destroy_clouds();
    if (atmo_->pipeline) vkDestroyPipeline(device_, atmo_->pipeline, nullptr);
    if (atmo_->pipeline_layout) vkDestroyPipelineLayout(device_, atmo_->pipeline_layout, nullptr);
    if (atmo_->pool) vkDestroyDescriptorPool(device_, atmo_->pool, nullptr);
    if (atmo_->set_layout) vkDestroyDescriptorSetLayout(device_, atmo_->set_layout, nullptr);
    if (atmo_->sampler) vkDestroySampler(device_, atmo_->sampler, nullptr);
    for (AtmoImage* im : { &atmo_->trans, &atmo_->ms, &atmo_->sky, &atmo_->ap }) {
        if (im->view) vkDestroyImageView(device_, im->view, nullptr);
        if (im->image) vmaDestroyImage(allocator_, im->image, im->alloc);
        *im = {};
    }
    atmo_->pipeline = VK_NULL_HANDLE; atmo_->pipeline_layout = VK_NULL_HANDLE;
    atmo_->pool = VK_NULL_HANDLE; atmo_->set_layout = VK_NULL_HANDLE;
    atmo_->sampler = VK_NULL_HANDLE; atmo_->set = VK_NULL_HANDLE;
}

void VulkanEngine::dispatch_atmosphere(VkCommandBuffer cmd) {
    if (!atmo_->pipeline) return;
    const FrameView& fv = current_frame_view_;
    float p = glm::radians(rt_.sun_pitch_deg), y = glm::radians(rt_.sun_yaw_deg);
    // Dusk / dawn: the sky is lit by the real sun under the horizon.
    if (env_->applied && env_->sky_real) { p = glm::radians(env_->sky_pitch); y = glm::radians(env_->sky_yaw); }
    glm::vec3 sun(std::sin(y) * std::cos(p), std::sin(p), std::cos(y) * std::cos(p));
    AtmoPC pc{};
    // World Y is metres above sea level; planet frame == world axes.
    const float h_km = std::max(fv.eye_pos.y, 1.0f) * 0.001f;
    pc.cam = glm::vec4(0.0f, kBottomRadiusKm + h_km, 0.0f, std::max(0.1f, rt_.aerial_scale));
    pc.eye = glm::vec4(fv.eye_pos, std::max(0.5f, rt_.aerial_max_km));
    pc.inv_vp = fv.inv_vp;

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, atmo_->pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, atmo_->pipeline_layout, 0, 1, &atmo_->set, 0, nullptr);
    auto barrier = [&]() {
        VkMemoryBarrier2 mb{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, nullptr,
                             VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                             VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                             VK_ACCESS_2_SHADER_SAMPLED_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                             VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT };
        VkDependencyInfo di{}; di.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        di.memoryBarrierCount = 1; di.pMemoryBarriers = &mb;
        vkCmdPipelineBarrier2(cmd, &di);
    };
    auto run = [&](float pass, uint32_t gx, uint32_t gy) {
        pc.sun = glm::vec4(sun, pass);
        vkCmdPushConstants(cmd, atmo_->pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, gx, gy, 1);
        barrier();
    };
    if (!atmo_->luts_ready) {
        run(0.0f, (kTransW + 7) / 8, (kTransH + 7) / 8);
        run(1.0f, (kMsW + 7) / 8, (kMsH + 7) / 8);
        atmo_->luts_ready = true;
    }
    run(2.0f, (kSkyW + 7) / 8, (kSkyH + 7) / 8);
    run(3.0f, (kApN + 7) / 8, (kApN + 7) / 8);
    if (rt_.clouds_enabled) dispatch_clouds(cmd);
}


// ---------------------------------------------------------------------
// Froxel volumetric fog (shaders/froxel_fog.comp)
// ---------------------------------------------------------------------
namespace {
constexpr uint32_t kFogX = 160, kFogY = 90, kFogZ = 64;
struct FogPC {
    glm::mat4 inv_vp;
    glm::vec4 eye;      // xyz camera (m), w time
    glm::vec4 sun;      // xyz toward sun, w pass
    glm::vec4 sun_col;  // rgb (engine E * pi), w ground reference height (m)
    glm::vec4 amb;      // rgb ambient, w frame
    glm::vec4 params;   // density, height falloff, anisotropy, max distance
    glm::vec4 flash_pos, flash_dir, flash_col;   // hand flashlight (as in the scene UBO)
};
}  // namespace

void VulkanEngine::init_fog() {
    for (AtmoImage* im : { &fog_->inject, &fog_->integ }) {
        VkImageCreateInfo ici{};
        ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ici.imageType = VK_IMAGE_TYPE_3D;
        ici.format = VK_FORMAT_R16G16B16A16_SFLOAT;
        ici.extent = { kFogX, kFogY, kFogZ };
        ici.mipLevels = 1; ici.arrayLayers = 1;
        ici.samples = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling = VK_IMAGE_TILING_OPTIMAL;
        ici.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        vk_check(vmaCreateImage(allocator_, &ici, &aci, &im->image, &im->alloc, nullptr), "fog volume");
        VkImageViewCreateInfo vci{};
        vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vci.image = im->image;
        vci.viewType = VK_IMAGE_VIEW_TYPE_3D;
        vci.format = ici.format;
        vci.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        vk_check(vkCreateImageView(device_, &vci, nullptr, &im->view), "fog volume view");
    }
    vkinit::one_time_submit(device_, graphics_queue_, graphics_queue_family_, [&](VkCommandBuffer cb) {
        VkMemoryBarrier2 mb{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, nullptr,
                             VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT,
                             VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                             VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT };
        VkDependencyInfo di{}; di.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        di.memoryBarrierCount = 1; di.pMemoryBarriers = &mb;
        for (AtmoImage* im : { &fog_->inject, &fog_->integ })
            vkinit::transition_image(cb, im->image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
        vkCmdPipelineBarrier2(cb, &di);
        // Integrated volume = (L 0, T 1) until the first dispatch: no fog.
        VkClearColorValue clr{ { 0.0f, 0.0f, 0.0f, 1.0f } };
        VkImageSubresourceRange r{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        vkCmdClearColorImage(cb, fog_->integ.image, VK_IMAGE_LAYOUT_GENERAL, &clr, 1, &r);
        vkCmdPipelineBarrier2(cb, &di);
    });
    const VkDescriptorType types[7] = {
        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,     // 5: leaf atlas (card cut-outs)
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER };   // 6: whole-map terrain sun shadow
    {
        VkDescriptorSetLayoutBinding b[7]{};
        for (int i = 0; i < 7; ++i) {
            b[i].binding = uint32_t(i);
            b[i].descriptorType = types[i];
            b[i].descriptorCount = 1;
            b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo lci{};
        lci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        lci.bindingCount = 7; lci.pBindings = b;
        vk_check(vkCreateDescriptorSetLayout(device_, &lci, nullptr, &fog_->set_layout), "fog set layout");
        VkDescriptorPoolSize sizes[] = {
            { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2 },
            { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 4 },
            { VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1 },
        };
        VkDescriptorPoolCreateInfo pci{};
        pci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pci.maxSets = 1; pci.poolSizeCount = 3; pci.pPoolSizes = sizes;
        vk_check(vkCreateDescriptorPool(device_, &pci, nullptr, &fog_->pool), "fog pool");
        VkDescriptorSetAllocateInfo dai{};
        dai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dai.descriptorPool = fog_->pool; dai.descriptorSetCount = 1; dai.pSetLayouts = &fog_->set_layout;
        vk_check(vkAllocateDescriptorSets(device_, &dai, &fog_->set), "fog set");
        VkDescriptorImageInfo ii[4] = {
            { VK_NULL_HANDLE, fog_->inject.view, VK_IMAGE_LAYOUT_GENERAL },
            { atmo_->sampler, fog_->inject.view,  VK_IMAGE_LAYOUT_GENERAL },
            { VK_NULL_HANDLE, fog_->integ.view,  VK_IMAGE_LAYOUT_GENERAL },
            { cloud_->sampler, cloud_->detail.view, VK_IMAGE_LAYOUT_GENERAL },
        };
        VkWriteDescriptorSet w[4]{};
        for (int i = 0; i < 4; ++i) {
            w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[i].dstSet = fog_->set;
            w[i].dstBinding = uint32_t(i);
            w[i].descriptorCount = 1;
            w[i].descriptorType = types[i];
            w[i].pImageInfo = &ii[i];
        }
        vkUpdateDescriptorSets(device_, 4, w, 0, nullptr);
        if (sun_->horizon_view) {
            VkDescriptorImageInfo hz{ sun_->horizon_sampler, sun_->horizon_view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
            VkWriteDescriptorSet wh{};
            wh.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            wh.dstSet = fog_->set; wh.dstBinding = 6; wh.descriptorCount = 1;
            wh.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            wh.pImageInfo = &hz;
            vkUpdateDescriptorSets(device_, 1, &wh, 0, nullptr);
        }
    }
    {
        VkPushConstantRange pc{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(FogPC) };
        VkPipelineLayoutCreateInfo plci{};
        plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount = 1; plci.pSetLayouts = &fog_->set_layout;
        plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pc;
        vk_check(vkCreatePipelineLayout(device_, &plci, nullptr, &fog_->pipeline_layout), "fog layout");
        VkShaderModule m = vkpipe::load_shader_module(device_, std::string(VIBE_SHADER_DIR) + "/froxel_fog.comp.spv");
        VkComputePipelineCreateInfo cci{};
        cci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        cci.stage.module = m;
        cci.stage.pName = "main";
        cci.layout = fog_->pipeline_layout;
        vk_check(vkCreateComputePipelines(device_, pipeline_cache_, 1, &cci, nullptr, &fog_->pipeline),
                 "fog pipeline");
        vkDestroyShaderModule(device_, m, nullptr);
    }
    fog_->tlas_bound = VK_NULL_HANDLE;
    log::info("[fog] froxel volumetric fog ready (160x90x64, RT sun visibility)");
}

void VulkanEngine::destroy_fog() {
    if (fog_->pipeline) vkDestroyPipeline(device_, fog_->pipeline, nullptr);
    if (fog_->pipeline_layout) vkDestroyPipelineLayout(device_, fog_->pipeline_layout, nullptr);
    if (fog_->pool) vkDestroyDescriptorPool(device_, fog_->pool, nullptr);
    if (fog_->set_layout) vkDestroyDescriptorSetLayout(device_, fog_->set_layout, nullptr);
    for (AtmoImage* im : { &fog_->inject, &fog_->integ }) {
        if (im->view) vkDestroyImageView(device_, im->view, nullptr);
        if (im->image) vmaDestroyImage(allocator_, im->image, im->alloc);
        *im = {};
    }
    fog_->pipeline = VK_NULL_HANDLE; fog_->pipeline_layout = VK_NULL_HANDLE;
    fog_->pool = VK_NULL_HANDLE; fog_->set_layout = VK_NULL_HANDLE; fog_->set = VK_NULL_HANDLE;
}

void VulkanEngine::dispatch_fog(VkCommandBuffer cmd) {
    if (!fog_->pipeline || tlas_ == VK_NULL_HANDLE) return;
    if (fog_->tlas_bound != tlas_) {   // the TLAS handle changes when it is reallocated
        wait_gpu_frames_idle();
        VkWriteDescriptorSetAccelerationStructureKHR as{};
        as.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
        as.accelerationStructureCount = 1;
        as.pAccelerationStructures = &tlas_;
        VkWriteDescriptorSet w{};
        w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.pNext = &as;
        w.dstSet = fog_->set; w.dstBinding = 4; w.descriptorCount = 1;
        w.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
        // Leaf atlas (binding 5) rides along: it is loaded long before the
        // first TLAS, and the shader alpha-tests leaf cards with it.
        VkDescriptorImageInfo leaf_ii{ texture_sampler_, albedo_textures_[kTexLeaf].view,
                                       VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        VkWriteDescriptorSet wl{};
        wl.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wl.dstSet = fog_->set; wl.dstBinding = 5; wl.descriptorCount = 1;
        wl.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        wl.pImageInfo = &leaf_ii;
        // Far-mountain sun shadow (binding 6): created after the fog set, so it
        // is bound here too.
        VkDescriptorImageInfo hz_ii{ sun_->horizon_sampler, sun_->horizon_view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        VkWriteDescriptorSet wh = wl;
        wh.dstBinding = 6;
        wh.pImageInfo = &hz_ii;
        const VkWriteDescriptorSet ws[3] = { w, wl, wh };
        vkUpdateDescriptorSets(device_, 3, ws, 0, nullptr);
        fog_->tlas_bound = tlas_;
    }
    const FrameView& fv = current_frame_view_;
    float p = glm::radians(rt_.sun_pitch_deg), y = glm::radians(rt_.sun_yaw_deg);
    glm::vec3 sun(std::sin(y) * std::cos(p), std::sin(p), std::cos(y) * std::cos(p));
    FogPC pc{};
    pc.inv_vp = fv.inv_vp;
    pc.eye = glm::vec4(fv.eye_pos, static_cast<float>(glfwGetTime()));
    pc.sun_col = glm::vec4(last_sun_rgb_ * rt_.sun_intensity * 3.14159265f, rt_.water_level + 1.0f);
    pc.amb = glm::vec4(last_sky_rgb_ * rt_.ambient_strength * 0.8f, float(frame_number_ % 64));
    pc.params = glm::vec4(std::max(0.0f, rt_.fog_density), std::max(0.0f, rt_.fog_height_falloff),
                          glm::clamp(rt_.fog_anisotropy, -0.9f, 0.95f), std::max(20.0f, rt_.fog_max_dist));
    // Mist on the lake: thickest in the cool hours (dawn, dusk, night),
    // nearly gone at midday. Carried in the unused flash_dir.x slot.
    {
        const float h = std::fmod(rt_.time_of_day + 24.0f, 24.0f);
        auto bump = [](float x, float c, float w) { float t = std::clamp(1.0f - std::abs(x - c) / w, 0.0f, 1.0f); return t * t * (3.0f - 2.0f * t); };
        const float cool = std::max({ bump(h, 6.5f, 3.5f), bump(h, 20.5f, 3.0f), (h < 5.0f || h > 22.0f) ? 0.7f : 0.0f });
        pc.flash_dir.x = std::clamp(rt_.lake_mist, 0.0f, 3.0f) * (0.15f + 0.85f * cool);
    }
    // (The flashlight beam is NOT injected here: at froxel resolution it was
    // a foggy ball in front of the lens. It is integrated per pixel instead,
    // see flashlight_beam in scene_ubo.glsl. flash_* stay zero = off.)
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, fog_->pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, fog_->pipeline_layout, 0, 1, &fog_->set, 0, nullptr);
    auto barrier = [&]() {
        VkMemoryBarrier2 mb{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, nullptr,
                             VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                             VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                             VK_ACCESS_2_SHADER_SAMPLED_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                             VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT };
        VkDependencyInfo di{}; di.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        di.memoryBarrierCount = 1; di.pMemoryBarriers = &mb;
        vkCmdPipelineBarrier2(cmd, &di);
    };
    pc.sun = glm::vec4(sun, 0.0f);
    vkCmdPushConstants(cmd, fog_->pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, (kFogX + 7) / 8, (kFogY + 7) / 8, kFogZ);
    barrier();
    pc.sun.w = 1.0f;
    vkCmdPushConstants(cmd, fog_->pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, (kFogX + 7) / 8, (kFogY + 7) / 8, 1);
    barrier();
}

} // namespace vibe
