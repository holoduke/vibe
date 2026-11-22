#pragma once
// NVIDIA Streamline: the proxies and the frame token (dlss.cpp): its state.
//
// Private to the files that implement it - vk_engine.h only knows the name of this struct
// and holds a pointer to it (sl_), so a change here rebuilds those files and not the
// whole engine. Created and freed in vk_engine/private_state.cpp.

#include "engine/vk_engine.h"

namespace vibe {

struct VulkanEngine::StreamlineState {
    // Streamline / DLSS state.
    bool                   ok = false;
    void*                  frame_token = nullptr;   // sl::FrameToken* of the frame being built
    std::vector<std::string> instance_exts, device_exts;
    // Streamline proxies (sl.interposer.dll): vk-bootstrap creates the
    // instance / device / swapchain through gipa, acquire + present go
    // through the proxies so SL's per-frame bookkeeping (presentCommon) runs.
    PFN_vkGetInstanceProcAddr gipa = nullptr;
    PFN_vkGetDeviceProcAddr   gdpa = nullptr;
    PFN_vkAcquireNextImageKHR acquire_fn = nullptr;
    PFN_vkQueuePresentKHR     present_fn = nullptr;
};

} // namespace vibe
