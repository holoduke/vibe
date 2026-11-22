#pragma once
// FSR 3.1 upscaler and frame generation (fsr3.cpp): its state.
//
// Private to the files that implement it - vk_engine.h only knows the name of this struct
// and holds a pointer to it (fsr3_), so a change here rebuilds those files and not the
// whole engine. Created and freed in vk_engine/private_state.cpp.

#include "engine/vk_engine.h"

namespace vibe {

struct VulkanEngine::Fsr3State {
    // --- FSR3 upscaler (FidelityFX SDK 1.1.4 ffx-api, prebuilt DLL) ---
    // Parallel path to FSR2; selected via rt_.fsr_backend (0=FSR2 legacy,
    // 1=FSR3 upscaler). FSR3 uses the new ffxContext + ffxDispatch ABI
    // and the prebuilt amd_fidelityfx_vk.dll. Output image is the same
    // HR R16G16B16A16 storage as fsr2_output_image_; compose's history
    // binding flips to output_view when the backend is FSR3.
    void*         upscaler_context = nullptr;  // ffxContext (opaque)
    bool          context_valid    = false;
    bool          reset_history    = true;
    // Set when init_fsr3() AVs inside the SDK — sticks for the process
    // lifetime so we never retry. UI auto-reverts to FSR2 backend.
    bool          fatal            = false;
    VkImage       output_image = VK_NULL_HANDLE;
    VmaAllocation output_alloc = nullptr;
    VkImageView   output_view  = VK_NULL_HANDLE;
    // --- FSR3 frame generation (session 3 of docs/fsr3_plan.md) ---
    // Separate ffxContext for the FG effect — uses the same DLL +
    // backend as the upscaler. Lazy-init when rt_.fg_enabled goes
    // true (and fsr_backend == 1). Output is an HR image we OWN; this
    // session does NOT integrate with the swapchain (session 4).
    void*         fg_context       = nullptr;
    bool          fg_context_valid = false;
    bool          fg_fatal         = false;
    VkImage       fg_output_image  = VK_NULL_HANDLE;
    VmaAllocation fg_output_alloc  = nullptr;
    VkImageView   fg_output_view   = VK_NULL_HANDLE;
    uint64_t      fg_frame_id      = 0;  // must increment by exactly 1
    // --- FSR3 SwapChain proxy (session 4 of docs/fsr3_plan.md) ---
    // The SDK takes ownership of our VkSwapchainKHR and gives back its
    // own wrapped handle plus a function table that REPLACES
    // vkAcquireNextImageKHR / vkQueuePresentKHR. The engine then routes
    // every acquire/present through SDK-provided pointers — when FG is
    // active, those functions internally pace + insert generated frames
    // into the present chain.
    //
    // Lazy: created when fg_enabled goes true, destroyed when it goes
    // false. SEH-guarded init falls back to plain swapchain on AV.
    void* swapchain_context = nullptr;  // ffxContext (opaque)
    bool  swapchain_active  = false;
    bool  swapchain_init_pending = false;   // draw() asks; run() creates between frames
    bool  swapchain_fatal   = false;
    // Extra graphics-family queues for the SDK's distinct-queue check.
    // Pulled at device-create when we requested 4 queues; null on
    // hardware that exposes only 1 graphics queue (FG then unavailable).
    VkQueue extra_queue_present = VK_NULL_HANDLE;
    VkQueue extra_queue_acquire = VK_NULL_HANDLE;
    VkQueue extra_queue_compute = VK_NULL_HANDLE;
    // SDK-provided replacement function pointers — populated after a
    // successful init via ffxQuery. Plain Vulkan when active==false.
    PFN_vkAcquireNextImageKHR  acquire_fn = nullptr;
    PFN_vkQueuePresentKHR      present_fn = nullptr;
    PFN_vkGetSwapchainImagesKHR get_images_fn = nullptr;
    // Session 5 — UI mask via HUDLessColor. Snapshot of swapchain
    // post-compose pre-ImGui each frame; SDK uses it to extract UI
    // pixels by diffing against the final backbuffer (with HUD).
    VkImage       hudless_image = VK_NULL_HANDLE;
    VmaAllocation hudless_alloc = nullptr;
    VkImageView   hudless_view  = VK_NULL_HANDLE;
};

} // namespace vibe
