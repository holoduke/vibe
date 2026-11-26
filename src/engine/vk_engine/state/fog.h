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
    // Two inject volumes, used in turn: this frame's is written while last frame's is read
    // as the history (temporal accumulation). sets[k] writes inject_k[k] and reads the other.
    AtmoImage              inject, integ;
    AtmoImage              inject_b;           // the second of the pair
    VkDescriptorSetLayout  set_layout = VK_NULL_HANDLE;
    VkDescriptorPool       pool = VK_NULL_HANDLE;
    VkDescriptorSet        set = VK_NULL_HANDLE;     // = sets[0]
    VkDescriptorSet        sets[2]{ VK_NULL_HANDLE, VK_NULL_HANDLE };
    int                    cur = 0;                  // the set used last
    // What last frame's volume was made with (its reprojection), and whether it may be used:
    // false after start-up, a teleport or a chart change (the eye jumped).
    glm::mat4              last_vp{1.0f};
    glm::vec3              last_eye{0.0f};
    bool                   have_hist = false;
    VkPipelineLayout       pipeline_layout = VK_NULL_HANDLE;
    VkPipeline             pipeline = VK_NULL_HANDLE;
    VkAccelerationStructureKHR tlas_bound = VK_NULL_HANDLE;
};

} // namespace vibe
