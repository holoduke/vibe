#pragma once
// The sun's shadow map and horizon shadows (sun_shadow.cpp, sun_horizon.cpp): its state.
//
// Private to the files that implement it - vk_engine.h only knows the name of this struct
// and holds a pointer to it (sun_), so a change here rebuilds those files and not the
// whole engine. Created and freed in vk_engine/private_state.cpp.

#include "engine/vk_engine.h"

namespace vibe {

struct VulkanEngine::SunShadowState {
    // Whole-map terrain sun shadow (sun_horizon.cpp): far mountains block
    // the low sun. R16G16F, re-baked on the worker pool as the light moves.
    VkImage       horizon_image = VK_NULL_HANDLE;
    VmaAllocation horizon_alloc = nullptr;
    VkImageView   horizon_view = VK_NULL_HANDLE;
    VkSampler     horizon_sampler = VK_NULL_HANDLE;
    VkBuffer      horizon_stage[kFrameOverlap]{};
    VmaAllocation horizon_stage_alloc[kFrameOverlap]{};
    void*         horizon_stage_map[kFrameOverlap]{};
    std::vector<uint16_t> horizon_bake;     // worker output (RG half floats)
    std::vector<float>    horizon_s_bake;   // worker: shadow-volume height
    std::vector<float>    horizon_s;        // live copy for CPU queries
    jobs::Group   horizon_job;
    bool          horizon_baking = false, horizon_upload_pending = false;
    int           horizon_age = 100000;
    glm::vec3     horizon_dir{0.0f}, horizon_bake_dir{0.0f};
    int             shadow_dim     = 1024;     // current image side
    VkImage         shadow_image   = VK_NULL_HANDLE;
    VmaAllocation   shadow_alloc   = nullptr;
    VkImageView     shadow_view    = VK_NULL_HANDLE;
    VkSampler       shadow_sampler = VK_NULL_HANDLE;
    VkPipeline       shadow_pipeline = VK_NULL_HANDLE;
    // Dedicated pipeline layout — shadow.vert reads only push constants,
    // not the scene descriptor set. Using the cube pipeline_layout_ here
    // forced render_sun_shadow_pass to bind scene_desc_set (TLAS, every
    // texture array) for nothing.
    VkPipelineLayout shadow_pipeline_layout = VK_NULL_HANDLE;
    VkShaderModule   shadow_vert_module = VK_NULL_HANDLE;
    glm::mat4       shadow_light_vp = glm::mat4(1.0f);
    VkShaderModule         shafts_frag_module = VK_NULL_HANDLE;  // compose.frag -DSUN_SHAFTS
    VkPipeline             shafts_pipeline = VK_NULL_HANDLE;
    bool                   shafts_prepass_ran = false;           // this frame: compose skips its own shafts
};

} // namespace vibe
