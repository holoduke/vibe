#pragma once
// FSR 2 upscaler (fsr2.cpp): its state.
//
// Private to the files that implement it - vk_engine.h only knows the name of this struct
// and holds a pointer to it (fsr2_), so a change here rebuilds those files and not the
// whole engine. Created and freed in vk_engine/private_state.cpp.

#include "engine/vk_engine.h"

namespace vibe {

struct VulkanEngine::Fsr2State {
    // --- FSR2 (vendored AMD FidelityFX SDK, MIT) ---
    // Phase 3 — context lifecycle. Opaque pointer + scratch buffer so
    // the rest of the engine doesn't drag the FSR2 headers in.
    // Phase 4 — per-frame dispatch. Output image is HR (swapchain-sized)
    // R16G16B16A16_SFLOAT, written by FSR2's compute shaders, sampled
    // by compose when fsr2_enabled. compose_uses_fsr2_ tracks the
    // currently-bound source so we only re-write desc sets on toggle.
    void* scratch = nullptr;
    size_t scratch_size = 0;
    void* context = nullptr;   // FfxFsr2Context* (heap-allocated)
    bool  context_valid = false;
    bool  reset_history = true; // first frame after context create
    VkImage       output_image = VK_NULL_HANDLE;
    VmaAllocation output_alloc = nullptr;
    VkImageView   output_view  = VK_NULL_HANDLE;
    VkFormat      output_format = VK_FORMAT_R16G16B16A16_SFLOAT;
};

} // namespace vibe
