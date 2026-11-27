#pragma once
// The rock kit's bake and the breaker pattern (vk_engine/rock_bake.cpp): one array texture at
// scene binding 63, read by shaders/rock_breaker.glsl. Private to the files that implement it.

#include "engine/vk_engine.h"

#include <vector>

namespace vibe {

struct VulkanEngine::RockBakeState {
    VkImage        image = VK_NULL_HANDLE;
    VmaAllocation  alloc = nullptr;
    VkImageView    view = VK_NULL_HANDLE;
    VkSampler      sampler = VK_NULL_HANDLE;
    int            size = 0, layers = 0, mips = 0, pieces = 0;      // pieces: 3 layers each; then the breaker's 2
    std::vector<float> range;                                       // per piece: the height's range (m)
    int            switches = 0;                                    // what is on: rock_breaker.glsl RK_*
    // The formations' tops from above (RG32F: world height, flag 1 = ground lies on it / 0.5 = bare rock / 0 = none)
    VkImage        tops_image = VK_NULL_HANDLE;
    VmaAllocation  tops_alloc = nullptr;
    VkImageView    tops_view = VK_NULL_HANDLE;
    VkSampler      tops_sampler = VK_NULL_HANDLE;
};

} // namespace vibe
