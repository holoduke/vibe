#pragma once
// Screen-space displacement mapping (ssdm.cpp): its state.
//
// Private to the few files that implement it - vk_engine.h only knows the name of this
// struct and holds a pointer to it (ssdm_), so a change here rebuilds those files and not
// the whole engine. Created and freed in vk_engine/private_state.cpp.

#include "engine/vk_engine.h"

namespace vibe {

struct VulkanEngine::SsdmState {
    // hn: the height feed (vertex normal + height, ssdm_height / ssdm_ground);
    // color / g0 / g1 / motion / scene_z: the main pass's attachments as they were before
    // displacement (what the passes read while the live ones are written);
    // o_color / o_g0 / o_g1: the displaced G-buffer copy (ssdm_resolve, ssdm_behind), copied
    // back into the live attachments at the end;
    // b_mask / b_tiles / b_weight: the behind-material mask, its tiles and the blend weight.
    SsdmImage hn, color, g0, g1, motion, scene_z;
    SsdmImage o_color, o_g0, o_g1;
    SsdmImage b_mask, b_tiles, b_weight;
    VkExtent2D extent{ 0, 0 }, half{ 0, 0 }, tiles{ 0, 0 };
    // raster: one 64-bit word per pixel (ssdm_raster / ssdm_hwraster, atomic max);
    // hw_list: the cells left to the hardware path; hw_ctl: their count + the indirect draw.
    VkBuffer raster = VK_NULL_HANDLE, hw_list = VK_NULL_HANDLE, hw_ctl = VK_NULL_HANDLE;
    VmaAllocation raster_alloc = nullptr, hw_list_alloc = nullptr, hw_ctl_alloc = nullptr;
    VkPipeline height_pipeline = VK_NULL_HANDLE, ground_pipeline = VK_NULL_HANDLE;
    VkPipeline raster_pipeline = VK_NULL_HANDLE, hwargs_pipeline = VK_NULL_HANDLE, hwraster_pipeline = VK_NULL_HANDLE;
    VkPipeline resolve_pipeline = VK_NULL_HANDLE, dilate_pipeline = VK_NULL_HANDLE, behind_pipeline = VK_NULL_HANDLE;
    VkShaderModule height_vert = VK_NULL_HANDLE, height_frag = VK_NULL_HANDLE, ground_frag = VK_NULL_HANDLE;
    VkShaderModule raster_comp = VK_NULL_HANDLE, hwargs_comp = VK_NULL_HANDLE, hwraster_vert = VK_NULL_HANDLE,
                   hwraster_frag = VK_NULL_HANDLE, resolve_frag = VK_NULL_HANDLE, dilate_comp = VK_NULL_HANDLE,
                   behind_comp = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VkDescriptorSet  set = VK_NULL_HANDLE;
    VkSampler nearest = VK_NULL_HANDLE, linear = VK_NULL_HANDLE;
};

} // namespace vibe
