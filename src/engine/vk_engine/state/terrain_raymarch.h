#pragma once
// Raymarched terrain: pipelines and the low-resolution march (pipelines_setup.cpp): its state.
//
// Private to the files that implement it - vk_engine.h only knows the name of this struct
// and holds a pointer to it (tr_), so a change here rebuilds those files and not the
// whole engine. Created and freed in vk_engine/private_state.cpp.

#include "engine/vk_engine.h"

namespace vibe {

struct VulkanEngine::TerrainRaymarchState {
    // Low-res raymarch render target. When `terrain_raymarch_scale` < 1
    // the FBM ray-march renders into these (cheaper, since each pixel
    // does ~hundreds of FBM evaluations) and a separate compose pass
    // bilinear-upscales them into the full-res scene_color/depth with
    // depth-aware blending so rasterised cubes/castle still occlude
    // correctly. Each LR* image is sized at render_extent_ × scale.
    VkImage         lr_color_image  = VK_NULL_HANDLE;
    VmaAllocation   lr_color_alloc  = nullptr;
    VkImageView     lr_color_view   = VK_NULL_HANDLE;
    VkImage         lr_motion_image = VK_NULL_HANDLE;
    VmaAllocation   lr_motion_alloc = nullptr;
    VkImageView     lr_motion_view  = VK_NULL_HANDLE;
    VkImage         lr_depth_image  = VK_NULL_HANDLE;
    VmaAllocation   lr_depth_alloc  = nullptr;
    VkImageView     lr_depth_view   = VK_NULL_HANDLE;
    VkSampler       lr_sampler      = VK_NULL_HANDLE;
    // NEAREST sampler dedicated to u_lr_depth (binding 11). Bilinear on
    // depth values bleeds terrain↔sky across silhouettes — at low
    // raymarch scales the >=0.9999 sky-discard then either eats valid
    // terrain (gaps) or writes fake mid-depth halos.
    VkSampler       lr_depth_sampler = VK_NULL_HANDLE;
    // The terrain's low-resolution march ("march pre-pass", rt_.terrain_rm_prepass): one
    // distance per block of pixels (R32F, render_extent_ / march_div) - how far a ray
    // can go before the ground comes near. Drawn before the world pass by a cut-down build
    // of the terrain shader (terrain_march_lo.frag.spv); the full-resolution terrain pass
    // reads it (binding 34) and starts its march there. Made and resized with the tr_lr_*
    // targets above.
    VkImage         march_image = VK_NULL_HANDLE;
    VmaAllocation   march_alloc = nullptr;
    VkImageView     march_view  = VK_NULL_HANDLE;
    // The sun's shadow terms at that march's hits (RGBA16F, same size; binding 35).
    VkImage         shade_image = VK_NULL_HANDLE;
    VmaAllocation   shade_alloc = nullptr;
    VkImageView     shade_view  = VK_NULL_HANDLE;
    VkExtent2D      march_extent{};
    int             march_div   = 2;
    VkShaderModule  march_module   = VK_NULL_HANDLE;
    VkPipeline      march_pipeline = VK_NULL_HANDLE;
    VkExtent2D      lr_extent{};
    // Compose pipeline that samples the low-res raymarch and writes
    // upscaled color/motion + gl_FragDepth into scene_color/depth with
    // depth-test enabled (so closer rasterised geometry survives).
    VkPipeline      compose_pipeline     = VK_NULL_HANDLE;
    VkShaderModule  compose_frag_module  = VK_NULL_HANDLE;
};

} // namespace vibe
