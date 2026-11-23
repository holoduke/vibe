// NVIDIA DLSS Super Resolution / DLAA through Streamline 2.x, using the
// "manual hooking" integration (docs/ProgrammingGuideManualHooking.md):
// the engine keeps calling vulkan-1.dll directly; Streamline only needs
//   slInit (before the VkInstance) -> extra instance/device extensions
//   slSetVulkanInfo (after the VkDevice)
//   per frame: frame token, constants, resource tags, DLSS options,
//              slEvaluateFeature on the graphics command buffer.
// Selected as upscaler backend 2 under the existing fsr2_enabled master
// toggle, so jitter, mip bias, unjittered motion vectors and the compose
// source routing are shared with the FSR paths.

#include "engine/vk_engine/internal.h"
#include "engine/vk_engine/state/streamline.h"
#include "engine/vk_engine/state/compose.h"
#include "engine/vk_engine/state/fsr3.h"
#include "engine/vk_engine/state/dlss.h"
#include "engine/vk_engine/state/rr_guides.h"
#include "engine/vk_initializers.h"
#include "engine/log.h"

#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cstring>
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#ifdef VIBE_HAVE_STREAMLINE
#include <sl.h>
#include <sl_consts.h>
#include <sl_dlss.h>
#include <sl_dlss_d.h>
#include <sl_helpers_vk.h>
#include <sl_dlss_g.h>
#include <sl_reflex.h>
#include <sl_pcl.h>
#endif
#include <fstream>
#include <string>

namespace vibe {

#ifdef VIBE_HAVE_STREAMLINE
namespace {
void sl_log(sl::LogType type, const char* msg) {
    if (type == sl::LogType::eError) log::errorf("[streamline] %s", msg);
    else if (type == sl::LogType::eWarn) log::warnf("[streamline] %s", msg);
}
sl::float4x4 to_sl(const glm::mat4& m) {
    // SL: row-major, row vectors (v * M). GLM: column-major, M * v.
    // Row i of the SL matrix = column i of the GLM matrix.
    sl::float4x4 r;
    for (int i = 0; i < 4; ++i) r.row[i] = sl::float4(m[i][0], m[i][1], m[i][2], m[i][3]);
    return r;
}
sl::Resource vk_res(VkImage img, VkImageView view, VkImageLayout layout, uint32_t w, uint32_t h,
                    VkFormat fmt, VkImageUsageFlags usage) {
    sl::Resource r(sl::ResourceType::eTex2d, reinterpret_cast<void*>(img), nullptr,
                   reinterpret_cast<void*>(view), uint32_t(layout));
    r.width = w; r.height = h;
    r.nativeFormat = uint32_t(fmt);
    r.mipLevels = 1; r.arrayLayers = 1;
    r.flags = 0;
    r.usage = usage;
    return r;
}
// Quality mode for a render scale (super resolution; tolerant of inputs a
// little under a mode's nominal size).
sl::DLSSMode dlss_mode_for_scale(float s) {
    if (s >= 0.99f) return sl::DLSSMode::eDLAA;
    if (s >= 0.64f) return sl::DLSSMode::eMaxQuality;
    if (s >= 0.56f) return sl::DLSSMode::eBalanced;
    if (s >= 0.45f) return sl::DLSSMode::eMaxPerformance;
    return sl::DLSSMode::eUltraPerformance;
}
// Ray Reconstruction is strict: the input may be LARGER than a mode's
// nominal size (dynamic resolution) but never smaller, or every evaluate
// fails with NGX InvalidParameter. So each mode starts exactly at its
// nominal ratio. Ultra performance has no dynamic range at all, which
// leaves no valid mode below 0.5 (see dlss_rr_active).
sl::DLSSMode dlss_rr_mode_for_scale(float s) {
    if (s >= 0.99f)  return sl::DLSSMode::eDLAA;
    if (s >= 0.667f) return sl::DLSSMode::eMaxQuality;
    if (s >= 0.58f)  return sl::DLSSMode::eBalanced;
    return sl::DLSSMode::eMaxPerformance;
}
}  // namespace
#endif

bool VulkanEngine::dlss_active() const {
    return dlss_->supported && rt_.fsr2_enabled && rt_.fsr_backend == 2 && dlss_->output_view != VK_NULL_HANDLE;
}

bool VulkanEngine::dlss_rr_active() const {
    // Under a 0.5 render scale there is no valid RR mode (see
    // dlss_rr_mode_for_scale): fall back to plain super resolution.
    return dlss_active() && dlss_->rr_supported && rt_.dlss_rr && deferred_lighting_active_ &&
           rt_.render_scale >= 0.5f;
}

void VulkanEngine::sl_init_early() {
#ifdef VIBE_HAVE_STREAMLINE
    sl::Preferences pref{};
    pref.showConsole = false;
    pref.logLevel = sl::LogLevel::eDefault;
    pref.logMessageCallback = &sl_log;
    pref.flags |= sl::PreferenceFlags::eUseManualHooking | sl::PreferenceFlags::eUseFrameBasedResourceTagging;
    pref.renderAPI = sl::RenderAPI::eVulkan;
    pref.engine = sl::EngineType::eCustom;
    pref.engineVersion = "1.0";
    pref.projectId = "5b0d6a1e-6c39-4c50-9a1e-7f3f0a6e2b41";
    // DLSS Frame Generation is loaded only when the saved settings ask for
    // it: once loaded it owns the swapchain (off-screen back buffers + its
    // own present queue), which costs a copy even with FG switched off.
    // load_settings() runs after the device exists, so peek the one key.
    int want_fg = 0, backend = 0, upscale = 0;
    {
        std::ifstream cf(kSettingsPath);
        std::string line;
        while (std::getline(cf, line)) {
            const auto eq = line.find('=');
            if (eq == std::string::npos) continue;
            std::string k = line.substr(0, eq);
            while (!k.empty() && (k.back() == ' ' || k.back() == '\t')) k.pop_back();
            const int v = std::atoi(line.c_str() + eq + 1);
            if (k == "dlss_fg") want_fg = v;
            else if (k == "fsr_backend") backend = v;
            else if (k == "fsr2_enabled") upscale = v;
        }
    }
    if (const char* e = std::getenv("VIBE_DLSS_FG")) want_fg = std::atoi(e);   // test override
    const bool load_fg = want_fg > 0 && backend == 2 && upscale != 0;
    static const sl::Feature kFeatures[] = { sl::kFeatureDLSS, sl::kFeatureDLSS_RR, sl::kFeatureReflex,
                                             sl::kFeaturePCL, sl::kFeatureDLSS_G };
    pref.featuresToLoad = kFeatures;
    pref.numFeaturesToLoad = load_fg ? 5 : 4;
    dlssg_loaded_ = load_fg;   // confirmed (or dropped) in sl_set_vulkan_info
    sl::Result r = slInit(pref);
    if (r != sl::Result::eOk) {
        log::warnf("[dlss] slInit failed (%d) - DLSS unavailable", int(r));
        return;
    }
    sl_->ok = true;
    log::infof("[dlss-g] frame generation %s at start-up", load_fg ? "requested" : "not requested");
    // Proxies: SL's vkGetInstanceProcAddr returns vulkan-1 functions except
    // the ones Streamline intercepts (instance / device / swapchain / present).
    if (HMODULE m = GetModuleHandleA("sl.interposer.dll")) {
        sl_->gipa = reinterpret_cast<PFN_vkGetInstanceProcAddr>(GetProcAddress(m, "vkGetInstanceProcAddr"));
        sl_->gdpa = reinterpret_cast<PFN_vkGetDeviceProcAddr>(GetProcAddress(m, "vkGetDeviceProcAddr"));
    }
    sl::FeatureRequirements reqs{};
    if (slGetFeatureRequirements(sl::kFeatureDLSS, reqs) == sl::Result::eOk) {
        for (uint32_t i = 0; i < reqs.vkNumInstanceExtensions; ++i)
            sl_->instance_exts.emplace_back(reqs.vkInstanceExtensions[i]);
        for (uint32_t i = 0; i < reqs.vkNumDeviceExtensions; ++i)
            sl_->device_exts.emplace_back(reqs.vkDeviceExtensions[i]);
        log::infof("[dlss] Streamline up: %u instance + %u device extensions requested",
                   reqs.vkNumInstanceExtensions, reqs.vkNumDeviceExtensions);
    } else {
        log::warn("[dlss] slGetFeatureRequirements(DLSS) failed");
    }
    // Ray Reconstruction: merge whatever it needs on top (usually the same).
    sl::FeatureRequirements rr_reqs{};
    if (slGetFeatureRequirements(sl::kFeatureDLSS_RR, rr_reqs) == sl::Result::eOk) {
        auto add_unique = [](std::vector<std::string>& v, const char* s) {
            if (std::find(v.begin(), v.end(), std::string(s)) == v.end()) v.emplace_back(s);
        };
        for (uint32_t i = 0; i < rr_reqs.vkNumInstanceExtensions; ++i)
            add_unique(sl_->instance_exts, rr_reqs.vkInstanceExtensions[i]);
        for (uint32_t i = 0; i < rr_reqs.vkNumDeviceExtensions; ++i)
            add_unique(sl_->device_exts, rr_reqs.vkDeviceExtensions[i]);
    }
    // Frame Generation / Reflex: same merge (the SL device proxy adds the
    // extra queues itself).
    for (sl::Feature feat : { sl::kFeatureReflex, sl::kFeaturePCL, sl::kFeatureDLSS_G }) {
        if (feat == sl::kFeatureDLSS_G && !load_fg) continue;
        sl::FeatureRequirements fr{};
        if (slGetFeatureRequirements(feat, fr) != sl::Result::eOk) continue;
        auto add_unique = [](std::vector<std::string>& v, const char* s) {
            if (std::find(v.begin(), v.end(), std::string(s)) == v.end()) v.emplace_back(s);
        };
        for (uint32_t i = 0; i < fr.vkNumInstanceExtensions; ++i) add_unique(sl_->instance_exts, fr.vkInstanceExtensions[i]);
        for (uint32_t i = 0; i < fr.vkNumDeviceExtensions; ++i) add_unique(sl_->device_exts, fr.vkDeviceExtensions[i]);
    }
#endif
}

void VulkanEngine::sl_set_vulkan_info() {
#ifdef VIBE_HAVE_STREAMLINE
    if (!sl_->ok) return;
    if (sl_->gipa && sl_->gdpa) {
        // Device + instance came from the SL proxies (vk-bootstrap was handed
        // sl_->gipa), so SL already knows them - slSetVulkanInfo is skipped.
        sl_->acquire_fn = reinterpret_cast<PFN_vkAcquireNextImageKHR>(sl_->gdpa(device_, "vkAcquireNextImageKHR"));
        sl_->present_fn = reinterpret_cast<PFN_vkQueuePresentKHR>(sl_->gdpa(device_, "vkQueuePresentKHR"));
        sl::AdapterInfo ai0{};
        ai0.vkPhysicalDevice = physical_device_;
        dlss_->supported = slIsFeatureSupported(sl::kFeatureDLSS, ai0) == sl::Result::eOk;
        dlss_->rr_supported = dlss_->supported && slIsFeatureSupported(sl::kFeatureDLSS_RR, ai0) == sl::Result::eOk;
        log::infof("[dlss] Ray Reconstruction %s", dlss_->rr_supported ? "supported" : "NOT supported");
        log::infof("[dlss] DLSS %s on this GPU (SL proxies)", dlss_->supported ? "supported" : "NOT supported");
        sl_setup_latency_and_fg(&ai0);
        return;
    }
    sl::VulkanInfo info{};
    info.device = device_;
    info.instance = instance_;
    info.physicalDevice = physical_device_;
    // DLSS SR needs no extra queues; SL queues would start after ours.
    info.graphicsQueueFamily = graphics_queue_family_;
    info.graphicsQueueIndex = 4;   // graphics queues 1..3 belong to the FSR3 proxy
    info.computeQueueFamily = compute_queue_family_;
    info.computeQueueIndex = compute_queue_distinct_ ? 1 : 2;
    if (slSetVulkanInfo(info) != sl::Result::eOk) {
        log::warn("[dlss] slSetVulkanInfo failed - DLSS unavailable");
        sl_->ok = false;
        return;
    }
    sl::AdapterInfo ai{};
    ai.vkPhysicalDevice = physical_device_;
    dlss_->supported = slIsFeatureSupported(sl::kFeatureDLSS, ai) == sl::Result::eOk;
    dlss_->rr_supported = dlss_->supported && slIsFeatureSupported(sl::kFeatureDLSS_RR, ai) == sl::Result::eOk;
    log::infof("[dlss] Ray Reconstruction %s", dlss_->rr_supported ? "supported" : "NOT supported");
    log::infof("[dlss] DLSS %s on this GPU", dlss_->supported ? "supported" : "NOT supported");
    sl_setup_latency_and_fg(&ai);
#endif
}

void VulkanEngine::init_dlss() {
#ifdef VIBE_HAVE_STREAMLINE
    if (!dlss_->supported) return;
    destroy_dlss();
    VkImageCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    ici.extent = { swapchain_extent_.width, swapchain_extent_.height, 1 };
    ici.mipLevels = 1; ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    vk_check(vmaCreateImage(allocator_, &ici, &aci, &dlss_->output_image, &dlss_->output_alloc, nullptr),
             "dlss output");
    VkImageViewCreateInfo vci{};
    vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vci.image = dlss_->output_image;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = ici.format;
    vci.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    vk_check(vkCreateImageView(device_, &vci, nullptr, &dlss_->output_view), "dlss output view");
    vkinit::one_time_submit(device_, graphics_queue_, graphics_queue_family_, [&](VkCommandBuffer cb) {
        vkinit::transition_image(cb, dlss_->output_image, VK_IMAGE_LAYOUT_UNDEFINED,
                                 VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    });
    dlss_->output_extent = swapchain_extent_;
    dlss_->reset = true;
    log::infof("[dlss] output %ux%u", dlss_->output_extent.width, dlss_->output_extent.height);
#endif
}

void VulkanEngine::destroy_dlss() {
    if (dlss_->output_view) vkDestroyImageView(device_, dlss_->output_view, nullptr);
    if (dlss_->output_image) vmaDestroyImage(allocator_, dlss_->output_image, dlss_->output_alloc);
    dlss_->output_view = VK_NULL_HANDLE;
    dlss_->output_image = VK_NULL_HANDLE;
    dlss_->output_alloc = nullptr;
}

void VulkanEngine::sl_shutdown() {
#ifdef VIBE_HAVE_STREAMLINE
    if (sl_->ok) slShutdown();
    sl_->ok = false;
#endif
}

// Called BEFORE command recording (vk_engine.cpp, next to the other compose
// rebinds): (re)creates the DLSS output on first use / resize and points
// compose at it. Doing this inside dispatch_dlss / draw() rewrote descriptor
// sets the frame had already bound (invalid command buffer).
void VulkanEngine::ensure_dlss_bindings() {
#ifdef VIBE_HAVE_STREAMLINE
    if (dlss_->supported && rt_.fsr2_enabled && rt_.fsr_backend == 2 &&
        (!dlss_->output_view || dlss_->output_extent.width != swapchain_extent_.width ||
         dlss_->output_extent.height != swapchain_extent_.height)) {
        vkDeviceWaitIdle(device_);
        init_dlss();
        rewrite_compose_image_bindings();
    }
#endif
    if (dlss_active() != compose_->uses_dlss) {
        vkDeviceWaitIdle(device_);
        rewrite_compose_image_bindings();
        dlss_->reset = true;
        log::infof("compose source switched to %s", compose_->uses_dlss ? "DLSS" : "TAA/FSR");
    }
    if (dlss_rr_active()) {
        const bool stale = !rr_->pipeline || rr_->extent.width != render_extent_.width ||
                           rr_->extent.height != render_extent_.height ||
                           rr_->src_views[0] != gbuffer0_view_ || rr_->src_views[1] != gbuffer1_view_ ||
                           rr_->src_views[2] != depth_view_ || rr_->src_views[3] != scene_color_view_;
        if (stale) {
            vkDeviceWaitIdle(device_);
            ensure_rr_guides();
        }
    }
    if (dlss_rr_active() != dlss_->rr_was_active) {
        dlss_->rr_was_active = dlss_rr_active();
        dlss_->reset = true;   // SR and RR keep separate histories
        log::infof("[dlss] %s", dlss_->rr_was_active ? "Ray Reconstruction ON" : "Super Resolution");
    }
}

void VulkanEngine::dispatch_dlss(VkCommandBuffer cmd) {
#ifdef VIBE_HAVE_STREAMLINE
    if (!dlss_->supported || !dlss_->output_view) return;   // set up by ensure_dlss_bindings()
    // One token per frame, taken in sl_frame_begin (Reflex / PCL / DLSS-G
    // need the same frame identity from simulation to present).
    sl::FrameToken* token = static_cast<sl::FrameToken*>(sl_->frame_token);
    if (!token) {
        const uint32_t frame_idx = static_cast<uint32_t>(frame_number_);
        if (slGetNewFrameToken(token, &frame_idx) != sl::Result::eOk || !token) return;
    }
    const sl::ViewportHandle viewport{ 0 };
    const bool use_rr = dlss_rr_active() && rr_->image[0] != VK_NULL_HANDLE;
    if (use_rr) { dispatch_rr_guides(cmd); prof_mark(cmd, "rr_guides"); }

    // Constants: unjittered matrices, jitter in pixels, motion vectors.
    const FrameView& fv = current_frame_view_;
    glm::mat4 proj = fv.proj;
    proj[2][0] -= fv.jitter.x;                 // remove the sub-pixel jitter
    proj[2][1] -= fv.jitter.y;
    // In double precision: with the eye far from the origin and the far plane tens of km out, a float
    // inverse of the view-projection is noisy from frame to frame. DLSS and frame generation reproject
    // with these - the generated frames put the land a little off, back and forth, the more the higher
    // (and faster) the flight.
    const glm::dmat4 vp_cur_d = glm::dmat4(proj) * glm::dmat4(fv.view);
    const glm::dmat4 vp_prev_d = prev_view_proj_valid_ ? glm::dmat4(prev_view_proj_) : vp_cur_d;
    const glm::dmat4 clip_to_prev_d = vp_prev_d * glm::inverse(vp_cur_d);
    const glm::mat4 clip_to_prev = glm::mat4(clip_to_prev_d);
    sl::Constants c{};
    c.cameraViewToClip = to_sl(proj);
    c.clipToCameraView = to_sl(glm::inverse(proj));
    c.clipToLensClip = to_sl(glm::mat4(1.0f));
    c.clipToPrevClip = to_sl(clip_to_prev);
    c.prevClipToClip = to_sl(glm::mat4(glm::inverse(clip_to_prev_d)));
    // The projection jitter moves the image by -j in NDC (GLM w = -z_view),
    // so the pixel-space sample offset DLSS needs is -j * 0.5 * size on both
    // axes (measured: + made the output shimmer while standing still).
    c.jitterOffset = sl::float2(-fv.jitter.x * 0.5f * float(render_extent_.width),
                                -fv.jitter.y * 0.5f * float(render_extent_.height));
    // Engine motion = current_uv - prev_uv (uv units); DLSS wants
    // prev - current normalised to the render size.
    c.mvecScale = sl::float2(-1.0f, -1.0f);
    c.cameraPinholeOffset = sl::float2(0.0f, 0.0f);
    const glm::vec3 right = glm::vec3(fv.inv_view[0]), up = glm::vec3(fv.inv_view[1]);
    const glm::vec3 fwd = -glm::vec3(fv.inv_view[2]);
    c.cameraPos = sl::float3(fv.eye_pos.x, fv.eye_pos.y, fv.eye_pos.z);
    c.cameraUp = sl::float3(up.x, up.y, up.z);
    c.cameraRight = sl::float3(right.x, right.y, right.z);
    c.cameraFwd = sl::float3(fwd.x, fwd.y, fwd.z);
    c.cameraNear = 0.05f;
    c.cameraFar = view_far_;
    c.cameraFOV = glm::radians(view_fov_deg_);
    c.cameraAspectRatio = float(render_extent_.width) / float(std::max(1u, render_extent_.height));
    c.depthInverted = sl::Boolean::eFalse;
    c.cameraMotionIncluded = sl::Boolean::eTrue;
    c.motionVectors3D = sl::Boolean::eFalse;
    c.motionVectorsJittered = sl::Boolean::eFalse;
    c.reset = dlss_->reset ? sl::Boolean::eTrue : sl::Boolean::eFalse;
    dlss_->reset = false;
    slSetConstants(c, *token, viewport);

    sl::DLSSOptions o{};
    o.mode = dlss_mode_for_scale(rt_.render_scale);
    o.outputWidth = swapchain_extent_.width;
    o.outputHeight = swapchain_extent_.height;
    o.colorBuffersHDR = sl::Boolean::eTrue;
    o.useAutoExposure = sl::Boolean::eTrue;
    // Alpha carries the sky transmittance compose needs to rebuild the sky.
    o.alphaUpscalingEnabled = sl::Boolean::eTrue;
    slDLSSSetOptions(viewport, o);
    if (dlss_->rr_supported) {
        // RR tracks the DLSS options and adds its own; mode eOff parks it.
        sl::DLSSDOptions d{};
        d.mode = use_rr ? dlss_rr_mode_for_scale(rt_.render_scale) : sl::DLSSMode::eOff;
        d.outputWidth = o.outputWidth;
        d.outputHeight = o.outputHeight;
        d.colorBuffersHDR = sl::Boolean::eTrue;
        d.normalRoughnessMode = sl::DLSSDNormalRoughnessMode::ePacked;
        d.worldToCameraView = to_sl(fv.view);
        d.cameraViewToWorld = to_sl(fv.inv_view);
        d.alphaUpscalingEnabled = sl::Boolean::eTrue;   // sky transmittance, as for SR
        // The model (settings: dlss_rr_model; VIBE_DLSS_RR_PRESET=D|E|F overrides, for measuring).
        {
            static const char* pe = std::getenv("VIBE_DLSS_RR_PRESET");
            sl::DLSSDPreset p = rt_.dlss_rr_model == 1 ? sl::DLSSDPreset::ePresetD : sl::DLSSDPreset::eDefault;
            if (pe && pe[0] >= 'D' && pe[0] <= 'F') p = sl::DLSSDPreset(4 + (pe[0] - 'D'));
            d.dlaaPreset = d.qualityPreset = d.balancedPreset = d.performancePreset = d.ultraPerformancePreset = d.ultraQualityPreset = p;
            static int last_model = -1;
            if (last_model != int(p)) { if (last_model >= 0) dlss_->reset = true; last_model = int(p); }      // (another model: its history starts again)
        }
        slDLSSDSetOptions(viewport, d);
    }

    // Resource tags (manual hooking: layouts as they are when DLSS runs).
    vkinit::transition_image(cmd, dlss_->output_image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                             VK_IMAGE_LAYOUT_GENERAL);
    const uint32_t rw = render_extent_.width, rh = render_extent_.height;
    sl::Resource col = vk_res(scene_color_image_, scene_color_view_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                              rw, rh, scene_color_format_,
                              VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
    sl::Resource dep = vk_res(depth_image_, depth_view_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                              rw, rh, depth_format_,
                              VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);
    sl::Resource mv  = vk_res(motion_vec_image_, motion_vec_view_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                              rw, rh, motion_vec_format_,
                              VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
    sl::Resource out = vk_res(dlss_->output_image, dlss_->output_view, VK_IMAGE_LAYOUT_GENERAL,
                              swapchain_extent_.width, swapchain_extent_.height, VK_FORMAT_R16G16B16A16_SFLOAT,
                              VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
    sl::Extent in_ext{ 0, 0, rw, rh };
    sl::Extent out_ext{ 0, 0, swapchain_extent_.width, swapchain_extent_.height };
    sl::ResourceTag tags[4] = {
        sl::ResourceTag{ &col, sl::kBufferTypeScalingInputColor, sl::ResourceLifecycle::eValidUntilPresent, &in_ext },
        sl::ResourceTag{ &dep, sl::kBufferTypeDepth, sl::ResourceLifecycle::eValidUntilPresent, &in_ext },
        sl::ResourceTag{ &mv,  sl::kBufferTypeMotionVectors, sl::ResourceLifecycle::eValidUntilPresent, &in_ext },
        sl::ResourceTag{ &out, sl::kBufferTypeScalingOutputColor, sl::ResourceLifecycle::eValidUntilPresent, &out_ext },
    };
    slSetTagForFrame(*token, viewport, tags, 4, reinterpret_cast<sl::CommandBuffer*>(cmd));
    // Depth + motion vectors above are also DLSS-G's inputs (same frame,
    // valid until present): this frame can be interpolated.
    dlssg_tagged_this_frame_ = true;
    if (use_rr) {
        const VkImageUsageFlags gu = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        sl::Resource alb = vk_res(rr_->image[0], rr_->view[0], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                  rw, rh, rr_guide_format(0), gu);
        sl::Resource spc = vk_res(rr_->image[1], rr_->view[1], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                  rw, rh, rr_guide_format(1), gu);
        sl::Resource nrm = vk_res(rr_->image[2], rr_->view[2], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                  rw, rh, rr_guide_format(2), gu);
        sl::ResourceTag gtags[3] = {
            sl::ResourceTag{ &alb, sl::kBufferTypeAlbedo, sl::ResourceLifecycle::eValidUntilPresent, &in_ext },
            sl::ResourceTag{ &spc, sl::kBufferTypeSpecularAlbedo, sl::ResourceLifecycle::eValidUntilPresent, &in_ext },
            sl::ResourceTag{ &nrm, sl::kBufferTypeNormalRoughness, sl::ResourceLifecycle::eValidUntilPresent, &in_ext },
        };
        slSetTagForFrame(*token, viewport, gtags, 3, reinterpret_cast<sl::CommandBuffer*>(cmd));
    }

    const sl::BaseStructure* inputs[] = { &viewport };
    sl::Result r = slEvaluateFeature(use_rr ? sl::kFeatureDLSS_RR : sl::kFeatureDLSS, *token, inputs, 1,
                                     reinterpret_cast<sl::CommandBuffer*>(cmd));
    static bool logged = false;
    if (r != sl::Result::eOk && !logged) {
        log::warnf("[dlss] slEvaluateFeature failed (%d)", int(r));
        logged = true;
    }
    vkinit::transition_image(cmd, dlss_->output_image, VK_IMAGE_LAYOUT_GENERAL,
                             VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
#else
    (void)cmd;
#endif
}

// ---------------------------------------------------------------------
// Reflex + PCL + DLSS Frame Generation
// ---------------------------------------------------------------------
void VulkanEngine::sl_setup_latency_and_fg(const void* adapter) {
#ifdef VIBE_HAVE_STREAMLINE
    const sl::AdapterInfo& ai = *static_cast<const sl::AdapterInfo*>(adapter);
    reflex_ok_ = slIsFeatureSupported(sl::kFeatureReflex, ai) == sl::Result::eOk;
    pcl_ok_ = slIsFeatureSupported(sl::kFeaturePCL, ai) == sl::Result::eOk;
    if (reflex_ok_) {
        // Low latency on (DLSS-G requires Reflex; it also trims the render
        // queue without FG). No frame limit: the engine's pacer caps.
        sl::ReflexOptions ro{};
        ro.mode = sl::ReflexMode::eLowLatency;
        ro.frameLimitUs = 0;
        if (slReflexSetOptions(ro) != sl::Result::eOk) reflex_ok_ = false;
    }
    if (dlssg_loaded_) {
        dlssg_loaded_ = reflex_ok_ && pcl_ok_ && slIsFeatureSupported(sl::kFeatureDLSS_G, ai) == sl::Result::eOk;
        if (dlssg_loaded_) {
            sl::DLSSGState st{};
            if (slDLSSGGetState(sl::ViewportHandle{ 0 }, st, nullptr) == sl::Result::eOk)
                dlssg_max_frames_ = std::max(1u, st.numFramesToGenerateMax);
        }
    }
    log::infof("[dlss-g] Reflex %s, PCL %s, frame generation %s (max %u generated / rendered frame)",
               reflex_ok_ ? "on" : "unavailable", pcl_ok_ ? "on" : "unavailable",
               dlssg_loaded_ ? "available" : "not loaded", dlssg_max_frames_);
#else
    (void)adapter;
#endif
}

void VulkanEngine::sl_frame_begin() {
#ifdef VIBE_HAVE_STREAMLINE
    sl_->frame_token = nullptr;
    dlssg_tagged_this_frame_ = false;
    if (!sl_->ok) return;
    // FSR3's frame-generation swapchain proxy replaces swapchain_ with its
    // own handle: Reflex sleep / latency markers act on the swapchain and
    // crashed in the driver (vkSetLatencyMarkerNV on a foreign handle).
    const bool reflex_ok = reflex_ok_ && !fsr3_->swapchain_active;
    sl::FrameToken* token = nullptr;
    const uint32_t frame_idx = static_cast<uint32_t>(frame_number_);
    if (slGetNewFrameToken(token, &frame_idx) != sl::Result::eOk || !token) return;
    sl_->frame_token = token;
    if (reflex_ok) slReflexSleep(*token);
    sl_marker(0);   // simulation start
#endif
}

void VulkanEngine::sl_marker(int m) {
#ifdef VIBE_HAVE_STREAMLINE
    if (!pcl_ok_ || !sl_->frame_token || fsr3_->swapchain_active) return;
    // Frame generation: decide right before present whether this frame
    // is a real, fully tagged game frame (menus / loading / FSR paths
    // switch it off instead of interpolating garbage).
    if (m == 4) sl_update_frame_gen(dlssg_tagged_this_frame_ && state_ == State::Playing);
    slPCLSetMarker(static_cast<sl::PCLMarker>(m), *static_cast<sl::FrameToken*>(sl_->frame_token));
    // Every ~2 s on the present thread: frames presented per rendered frame.
    static uint32_t rendered = 0, presented = 0;
    if (m == 5 && dlssg_mode_on_) {
        ++rendered;
        sl::DLSSGState st{};
        if (slDLSSGGetState(sl::ViewportHandle{ 0 }, st, nullptr) == sl::Result::eOk) {
            presented += st.numFramesActuallyPresented;
            if (st.status != sl::DLSSGStatus::eOk) {
                static int warned = 0;
                if (warned++ < 3) log::warnf("[dlss-g] status 0x%x (1 res too low, 2 no Reflex, 4 HDR fmt, 8 constants)", unsigned(st.status));
            }
        }
        if (rendered >= 240) {
            dlssg_present_ratio_ = float(presented) / float(rendered);
            log::infof("[dlss-g] %.2f frames presented per rendered frame", dlssg_present_ratio_);
            rendered = presented = 0;
        }
    }
#else
    (void)m;
#endif
}

void VulkanEngine::sl_update_frame_gen(bool game_frame) {
#ifdef VIBE_HAVE_STREAMLINE
    if (!dlssg_loaded_) return;
    static const int env_fg = std::getenv("VIBE_DLSS_FG") ? std::atoi(std::getenv("VIBE_DLSS_FG")) : -1;   // test override
    const int fg = env_fg >= 0 ? env_fg : rt_.dlss_fg;
    const bool want = game_frame && fg > 0 && dlss_active();
    const int frames = std::clamp(fg, 1, int(dlssg_max_frames_));
    if (want == dlssg_mode_on_ && (!want || frames == dlssg_frames_sent_)) return;
    sl::DLSSGOptions o{};
    o.mode = want ? sl::DLSSGMode::eOn : sl::DLSSGMode::eOff;
    o.numFramesToGenerate = uint32_t(frames);
    o.numBackBuffers = uint32_t(swapchain_images_.size());
    o.mvecDepthWidth = render_extent_.width;
    o.mvecDepthHeight = render_extent_.height;
    o.colorWidth = swapchain_extent_.width;
    o.colorHeight = swapchain_extent_.height;
    o.colorBufferFormat = uint32_t(swapchain_format_);
    o.mvecBufferFormat = uint32_t(motion_vec_format_);
    o.depthBufferFormat = uint32_t(depth_format_);
    const sl::Result r = slDLSSGSetOptions(sl::ViewportHandle{ 0 }, o);
    dlssg_mode_on_ = want && r == sl::Result::eOk;
    dlssg_frames_sent_ = frames;
    log::infof("[dlss-g] %s (%dx)%s", dlssg_mode_on_ ? "ON" : "off", frames + 1,
               r == sl::Result::eOk ? "" : " - slDLSSGSetOptions failed");
#else
    (void)game_frame;
#endif
}

void VulkanEngine::dlssg_destroy_hudless() {
    if (!device_) return;
    if (dlssg_hudless_image_) vkDeviceWaitIdle(device_);
    if (dlssg_hudless_view_) vkDestroyImageView(device_, dlssg_hudless_view_, nullptr);
    if (dlssg_hudless_image_) vmaDestroyImage(allocator_, dlssg_hudless_image_, dlssg_hudless_alloc_);
    dlssg_hudless_view_ = VK_NULL_HANDLE;
    dlssg_hudless_image_ = VK_NULL_HANDLE;
    dlssg_hudless_alloc_ = nullptr;
    dlssg_hudless_extent_ = { 0, 0 };
}

void VulkanEngine::dlssg_capture_hudless(VkCommandBuffer cmd, uint32_t img_index) {
#ifdef VIBE_HAVE_STREAMLINE
    // Only while frame generation runs (mode set at the previous present).
    if (!dlssg_mode_on_ || !sl_->frame_token || !dlssg_tagged_this_frame_ ||
        img_index >= swapchain_images_.size()) return;
    if (!dlssg_hudless_image_ || dlssg_hudless_extent_.width != swapchain_extent_.width ||
        dlssg_hudless_extent_.height != swapchain_extent_.height) {
        dlssg_destroy_hudless();
        VkImageCreateInfo ici{};
        ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ici.imageType = VK_IMAGE_TYPE_2D;
        ici.format = swapchain_format_;
        ici.extent = { swapchain_extent_.width, swapchain_extent_.height, 1 };
        ici.mipLevels = 1; ici.arrayLayers = 1;
        ici.samples = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling = VK_IMAGE_TILING_OPTIMAL;
        ici.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        if (vmaCreateImage(allocator_, &ici, &aci, &dlssg_hudless_image_, &dlssg_hudless_alloc_, nullptr) != VK_SUCCESS) {
            dlssg_hudless_image_ = VK_NULL_HANDLE;
            return;
        }
        VkImageViewCreateInfo vci{};
        vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vci.image = dlssg_hudless_image_;
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.format = ici.format;
        vci.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        vkCreateImageView(device_, &vci, nullptr, &dlssg_hudless_view_);
        dlssg_hudless_extent_ = swapchain_extent_;
    }
    VkImage sc = swapchain_images_[img_index];
    vkinit::transition_image(cmd, sc, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    vkinit::transition_image(cmd, dlssg_hudless_image_, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    VkImageCopy region{};
    region.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    region.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    region.extent = { swapchain_extent_.width, swapchain_extent_.height, 1 };
    vkCmdCopyImage(cmd, sc, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                   dlssg_hudless_image_, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    vkinit::transition_image(cmd, sc, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    vkinit::transition_image(cmd, dlssg_hudless_image_, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                             VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    sl::Resource hud = vk_res(dlssg_hudless_image_, dlssg_hudless_view_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                              swapchain_extent_.width, swapchain_extent_.height, swapchain_format_,
                              VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    sl::Extent ext{ 0, 0, swapchain_extent_.width, swapchain_extent_.height };
    sl::ResourceTag tag{ &hud, sl::kBufferTypeHUDLessColor, sl::ResourceLifecycle::eValidUntilPresent, &ext };
    slSetTagForFrame(*static_cast<sl::FrameToken*>(sl_->frame_token), sl::ViewportHandle{ 0 }, &tag, 1,
                     reinterpret_cast<sl::CommandBuffer*>(cmd));
#else
    (void)cmd; (void)img_index;
#endif
}

} // namespace vibe
