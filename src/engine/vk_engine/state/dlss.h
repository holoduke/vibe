#pragma once
// DLSS upscaler (dlss.cpp): its state.
//
// Private to the files that implement it - vk_engine.h only knows the name of this struct
// and holds a pointer to it (dlss_), so a change here rebuilds those files and not the
// whole engine. Created and freed in vk_engine/private_state.cpp.

#include "engine/vk_engine.h"

namespace vibe {

struct VulkanEngine::DlssState {
    bool                   supported = false;
    VkImage                output_image = VK_NULL_HANDLE;
    VmaAllocation          output_alloc = nullptr;
    VkImageView            output_view = VK_NULL_HANDLE;
    VkExtent2D             output_extent{0, 0};
    bool                   reset = true;
    // DLSS Ray Reconstruction (dlss.cpp + rr_guides.cpp).
    bool                   rr_supported = false;
    bool                   rr_was_active = false;
};

} // namespace vibe
