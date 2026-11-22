#pragma once
// Bloom: the mip chain and its passes (bloom.cpp): its state.
//
// Private to the files that implement it - vk_engine.h only knows the name of this struct
// and holds a pointer to it (bloom_), so a change here rebuilds those files and not the
// whole engine. Created and freed in vk_engine/private_state.cpp.

#include "engine/vk_engine.h"

namespace vibe {

struct VulkanEngine::BloomState {
    VkImage      image = VK_NULL_HANDLE;
    VmaAllocation alloc = nullptr;
    VkImageView  mip_views[kBloomMips]{};
    VkExtent2D   mip_extents[kBloomMips]{};
    // All-mips view of image used by compose so cube.frag can
    // textureLod() to the smallest mip for an auto-exposure scene-avg
    // proxy. Per-mip views above are still needed by the down/up passes
    // which must rasterise into each mip individually.
    VkImageView  full_view = VK_NULL_HANDLE;
    VkDescriptorPool       desc_pool = VK_NULL_HANDLE;
    VkDescriptorSetLayout  desc_set_layout = VK_NULL_HANDLE;
    // 2 sets per slot per mip: one to read history (for first downsample),
    // (kBloomMips-1) to read mip k-1 for downsample, (kBloomMips-1) for
    // upsample. Pre-built and indexed.
    VkDescriptorSet down_sets[kHistorySlots][kBloomMips]{};
    VkDescriptorSet up_sets[kBloomMips]{};
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkPipeline       down_pipeline = VK_NULL_HANDLE;
    VkPipeline       up_pipeline = VK_NULL_HANDLE;
    VkShaderModule   down_frag = VK_NULL_HANDLE;
    VkShaderModule   up_frag = VK_NULL_HANDLE;
};

} // namespace vibe
