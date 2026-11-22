#pragma once
// Effects: ribbons, volumes, sparks (fx.cpp): its state.
//
// Private to the files that implement it - vk_engine.h only knows the name of this struct
// and holds a pointer to it (fx_), so a change here rebuilds those files and not the
// whole engine. Created and freed in vk_engine/private_state.cpp.

#include "engine/vk_engine.h"

namespace vibe {

struct VulkanEngine::FxState {
    // FX pass (after deferred lighting, before TAA): premultiplied-alpha
    // blended ribbons (contrails, laser beams, flash) + volumetric
    // explosion volumes into scene_color, depth-tested read-only.
    VkShaderModule vert_module       = VK_NULL_HANDLE;
    VkShaderModule ribbon_frag_module = VK_NULL_HANDLE;
    VkShaderModule volume_frag_module = VK_NULL_HANDLE;
    VkPipeline     ribbon_pipeline   = VK_NULL_HANDLE;
    VkPipeline     ribbon_refl_pipeline = VK_NULL_HANDLE;   // no depth test: mirrored glow in water
    VkPipeline     volume_pipeline   = VK_NULL_HANDLE;
    VkPipeline     volume_inside_pipeline = VK_NULL_HANDLE;
    VkPipeline     decal_pipeline = VK_NULL_HANDLE;
    VkPipeline     glass_pipeline = VK_NULL_HANDLE;       // window panes (glass.frag)
    VkShaderModule glass_vert_module = VK_NULL_HANDLE, glass_frag_module = VK_NULL_HANDLE;
    VkShaderModule decal_frag_module = VK_NULL_HANDLE;
    VkBuffer       vbuf[2]{};        // per frame-slot host-visible ribbon verts
    VmaAllocation  valloc[2]{};
    void*          vmapped[2]{};
};

} // namespace vibe
