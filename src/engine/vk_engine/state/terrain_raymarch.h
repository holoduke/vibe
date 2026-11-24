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
    // The temporal upscaler (rt_.terrain_rm_temporal, with the scale below 1; terrain_rm_resolve.frag):
    // its two kept pictures of the terrain alone (RGBA16F: colour, distance; bindings 36, 37),
    // written in turn. keep_extent: TWICE the render size - the main upscaler needs a different
    // part of the picture for each of the camera's places inside a pixel (see the shader).
    // (The names are the checkerboard's second colour / depth picture, which they replaced.)
    VkImage         lrb_color_image = VK_NULL_HANDLE;
    VmaAllocation   lrb_color_alloc = nullptr;
    VkImageView     lrb_color_view  = VK_NULL_HANDLE;
    VkImage         lrb_depth_image = VK_NULL_HANDLE;
    VmaAllocation   lrb_depth_alloc = nullptr;
    VkImageView     lrb_depth_view  = VK_NULL_HANDLE;
    bool            lr_cur_b        = false;        // this frame's picture is the second one
    bool            lr_hist_ok      = false;        // last frame's picture is there to be used
    uint64_t        lr_frame        = ~uint64_t(0); // the frame the low-resolution terrain was last drawn in
    glm::vec2       lr_jit{0.0f}, lr_jit_prev{0.0f};   // the offsets (in uv) of this frame's picture and last frame's
    glm::vec2       lr_phase{0.0f};                 // the temporal upscaler: this frame's sample offset in the STILL picture (uv), without the camera's sub-pixel offset
    VkExtent2D      keep_extent{};                   // the kept pictures' size
    int             keep_n = 2;                     // the low-resolution picture goes to keep_n x keep_n places inside its pixel in turn
    glm::mat4       keep_prev_view{0.0f};            // last frame's view, to tell:
    bool            cam_still = false;              // the camera has not moved since last frame (the kept picture then holds the outline against the sky too)
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
    // The ground's shading rate from that march's distances (terrain_vrs.frag): drawn into the
    // rate image every frame, after the march (frame.cpp).
    VkShaderModule  vrs_module   = VK_NULL_HANDLE;
    VkPipeline      vrs_pipeline = VK_NULL_HANDLE;
    VkExtent2D      lr_extent{};
    // Compose pipeline that samples the low-res raymarch and writes
    // upscaled color/motion + gl_FragDepth into scene_color/depth with
    // depth-test enabled (so closer rasterised geometry survives).
    VkPipeline      compose_pipeline     = VK_NULL_HANDLE;
    // The terrain's temporal upscaler (terrain_rm_resolve.frag): with rt_.terrain_rm_temporal the
    // lrb_color_* / lrb_depth_* images above are its two kept pictures (lr_cur_b: this frame's is
    // the second).
    VkPipeline      resolve_pipeline     = VK_NULL_HANDLE;
    VkShaderModule  resolve_frag_module  = VK_NULL_HANDLE;
    VkShaderModule  compose_frag_module  = VK_NULL_HANDLE;
};

} // namespace vibe
