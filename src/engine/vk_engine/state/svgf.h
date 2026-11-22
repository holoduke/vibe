#pragma once
// SVGF denoiser (svgf.cpp): its state.
//
// Private to the files that implement it - vk_engine.h only knows the name of this struct
// and holds a pointer to it (svgf_), so a change here rebuilds those files and not the
// whole engine. Created and freed in vk_engine/private_state.cpp.

#include "engine/vk_engine.h"

namespace vibe {

struct VulkanEngine::SvgfState {
    // SVGF GI denoiser foundation (Session 1 of docs/svgf_plan.md).
    // R16G16B16A16F storage image at render_extent; cube.frag writes
    // the raw (un-albedo-modulated) GI irradiance per pixel via
    // imageStore. Sessions 2-5 add the temporal accumulation + 3-pass
    // à-trous filter that read this; for now it's just allocated and
    // written, never sampled — verifies the plumbing in isolation.
    VkImage         gi_image = VK_NULL_HANDLE;
    VmaAllocation   gi_alloc = nullptr;
    VkImageView     gi_view  = VK_NULL_HANDLE;
    // SVGF temporal-accumulation history (Session 2). 2 R16G16B16A16F
    // storage images at render_extent ping-ponged via frame parity
    // (scene.rt_flags.w & 1) — same race-free model as the ReSTIR ring
    // buffer. .rgb = accumulated irradiance, .a = M-count [1..kMmax].
    VkImage         history_image[2] = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    VmaAllocation   history_alloc[2] = { nullptr, nullptr };
    VkImageView     history_view[2]  = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    // SVGF variance moments (Session 4a-deep). 2 R32G32_SFLOAT images at
    // render_extent ping-ponged by frame parity. .r = EMA mean of scene
    // luminance, .g = EMA mean of luminance^2. Estimated variance per
    // pixel = max(0, g - r*r); fed into the a-trous edge-stop weights so
    // stable surfaces filter tightly while noisy GI pixels can accept
    // bigger spatial neighbours. Bindings 22 (slot 0) / 23 (slot 1) on
    // scene_desc; svgf_moments.frag reprojects the previous-frame slot
    // through the existing motion_vec image and writes the new slot.
    VkImage         moments_image[2] = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    VmaAllocation   moments_alloc[2] = { nullptr, nullptr };
    VkImageView     moments_view[2]  = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    // SVGF a-trous ping-pong pair. Two R16G16B16A16F sampled+attachment
    // images at render_extent. The 3-pass a-trous filter bounces:
    //   pass 0 (stride 1): scene_color -> atrous[0]
    //   pass 1 (stride 2): atrous[0]   -> atrous[1]
    //   pass 2 (stride 4): atrous[1]   -> scene_color  (writes back, so
    //                                                   TAA reads the
    //                                                   denoised result)
    // Only allocated when rt_.svgf_enabled; reused across frames.
    VkImage         atrous_image[2] = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    VmaAllocation   atrous_alloc[2] = { nullptr, nullptr };
    VkImageView     atrous_view[2]  = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    // SVGF moments + a-trous passes (4a-deep). The moments pass updates
    // moments_image at the current frame slot; the a-trous pass
    // runs 3 times with growing stride and edge-stop weights derived
    // from depth, normal-from-depth and the per-pixel luminance
    // variance computed from the moments. Both pipelines stand up at
    // engine init alongside TAA. recreate_svgf_pass_targets() handles
    // descriptor rewrites when scene_color / depth / moments / atrous
    // views change after a swapchain resize.
    VkDescriptorSetLayout moments_set_layout = VK_NULL_HANDLE;
    VkDescriptorPool      moments_desc_pool  = VK_NULL_HANDLE;
    VkDescriptorSet       moments_desc_sets[2] = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    VkPipelineLayout      moments_pipeline_layout = VK_NULL_HANDLE;
    VkPipeline            moments_pipeline = VK_NULL_HANDLE;
    VkShaderModule        moments_frag_module = VK_NULL_HANDLE;
    VkBuffer              moments_ubo_buffer = VK_NULL_HANDLE;
    VmaAllocation         moments_ubo_alloc  = nullptr;
    VkDescriptorSetLayout atrous_set_layout = VK_NULL_HANDLE;
    VkDescriptorPool      atrous_desc_pool  = VK_NULL_HANDLE;
    // 3 sets per output target (scene_color, atrous0, atrous1) x 2
    // frame parities = 6 sets total. svgf_atrous_pass_idx selects which
    // 3-set bank to use per frame so the read views always point at
    // the moments slot just written by svgf_moments.
    VkDescriptorSet       atrous_desc_sets[6] = {};
    VkPipelineLayout      atrous_pipeline_layout = VK_NULL_HANDLE;
    VkPipeline            atrous_pipeline = VK_NULL_HANDLE;
    VkShaderModule        atrous_frag_module = VK_NULL_HANDLE;
    VkBuffer              atrous_ubo_buffer  = VK_NULL_HANDLE;
    VmaAllocation         atrous_ubo_alloc   = nullptr;
};

} // namespace vibe
