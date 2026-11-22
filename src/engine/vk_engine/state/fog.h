#pragma once
// Fog (the low-resolution fog pass): its state.
//
// Private to the files that implement it - vk_engine.h only knows the name of this struct
// and holds a pointer to it (fog_), so a change here rebuilds those files and not the
// whole engine. Created and freed in vk_engine/private_state.cpp.

#include "engine/vk_engine.h"

namespace vibe {

struct VulkanEngine::FogState {
    // Fog wisp pattern, 256² R8. Pre-baked 3-octave noise so the
    // terrain raymarch fog density + godray probe loops sample one
    // textureLod() per tap instead of running 3× noise2() each — the
    // dominant per-pixel cost on godray-on configs.
    TextureSlot wisp{};
    // Froxel volumetric fog (froxel_fog.comp): inject + integrated volumes.
    AtmoImage              inject, integ;
    VkDescriptorSetLayout  set_layout = VK_NULL_HANDLE;
    VkDescriptorPool       pool = VK_NULL_HANDLE;
    VkDescriptorSet        set = VK_NULL_HANDLE;
    VkPipelineLayout       pipeline_layout = VK_NULL_HANDLE;
    VkPipeline             pipeline = VK_NULL_HANDLE;
    VkAccelerationStructureKHR tlas_bound = VK_NULL_HANDLE;
};

} // namespace vibe
