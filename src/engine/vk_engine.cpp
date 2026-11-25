// VulkanEngine implementation.
//
// File map (search "// === Section: " to jump):
//
//   === Section: setup       — instance, device, swapchain, depth, sync
//   === Section: descriptors — scene UBO + materials + texture sampler arrays
//   === Section: rt          — BLAS/TLAS init + per-frame TLAS rebuild
//   === Section: pipeline    — main raster pipeline (cube/cylinder mesh draws)
//   === Section: taa+compose — TAA pingpong, compose pass, bloom mip chain
//   === Section: viewmodel   — procedural / glTF-loaded weapon viewmodel
//   === Section: combat      — projectiles + spark particles + trails
//   === Section: world+ui    — level setup, render_world, ui hooks
//   === Section: settings    — load/save/autosave vibe_settings.cfg
//   === Section: textures    — albedo + normal map upload (calls into
//                              engine/texture.cpp for the per-image work)
//   === Section: skybox      — engine/skybox.cpp wrapper + sun-sync
//   === Section: shutdown    — destroy_* in dependency order
//
// Sub-systems with their own translation units:
//   engine/gltf_loader.cpp — glTF 2.0 → MeshData (used by viewmodel)
//   engine/skybox.cpp      — equirect HDRI/JPG → VkImage + sun extraction
//   engine/texture.cpp     — generic 2D texture upload (used by init_textures)
//   engine/physics_world.cpp — Jolt wrapper (rigid bodies + raycast)
//   engine/mesh.cpp        — cube/cylinder/glTF mesh data → vertex/index VkBuffer
//
#include "engine/vk_engine.h"
#include <ctime>
#include <filesystem>
#include <system_error>
#include "engine/vk_engine/internal.h"
#include "engine/vk_engine/state/bloom.h"
#include "engine/vk_engine/state/fog.h"
#include "engine/vk_engine/state/terrain_raymarch.h"
#include "engine/vk_engine/state/taa.h"
#include "engine/vk_engine/state/fsr3.h"
#include "engine/frame_pacer.h"
// Pulled in here (not in vk_engine.h) so the unique_ptr<voxel::VoxelWorld>
// member's destructor can be generated in this TU where the type is
// complete. Forward-declared in the header to keep its include surface tight.
// Must be at file scope (NOT inside namespace vibe) so voxel_world.h's
// `namespace vibe::voxel` resolves to the same fully-qualified name as
// the forward decl in vk_engine.h.
#include "engine/voxel/voxel_world.h"

#include "engine/audio.h"

#include "engine/frustum.h"
#include "engine/gltf_loader.h"
#include "engine/cpu_prof.h"
#include "engine/log.h"
#include "engine/texture.h"
#include "engine/window.h"
#include "engine/vk_initializers.h"
#include "engine/vk_pipelines.h"

#include <cstdlib>
#include <VkBootstrap.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_vulkan.h>

// stb_image declarations only — implementation owned by skybox.cpp's TU.
#include <stb_image.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <typeinfo>

#ifndef VIBE_SHADER_DIR
#define VIBE_SHADER_DIR "shaders"
#endif

// (declared here instead of including windows.h, whose macros clash with our names)
extern "C" __declspec(dllimport) void* __stdcall GetCurrentProcess();
extern "C" __declspec(dllimport) int __stdcall TerminateProcess(void*, unsigned int);

namespace vibe {
void process_memory_mb(double& working, double& committed);      // (mem_log.cpp)

// PushConstants, SceneUBO, debug_callback, spark_blackbody, align_local_y_to,
// recoil_kick, xorshift32 / frand / frand_range, buffer_device_address,
// as_device_address, write_scene_descriptors_once, the RT extension function
// pointers (g_rt) and load_rt_functions all live in
// src/engine/vk_engine/helpers.cpp; declared in
// src/engine/vk_engine/internal.h which we include above.

// Frustum + extract_frustum + aabb_visible live in engine/frustum.h.

VulkanEngine::VulkanEngine() { alloc_private_state(); }
VulkanEngine::~VulkanEngine() { shutdown(); free_private_state(); }

void VulkanEngine::init() {
    log::info("VulkanEngine::init()");
    // VIBE_TEST_SIZE=WxH: window size for profiling at the played resolution.
    uint32_t win_w = 1280, win_h = 720;
    if (const char* ts = std::getenv("VIBE_TEST_SIZE")) {
        unsigned tw = 0, th = 0;
        if (std::sscanf(ts, "%ux%u", &tw, &th) == 2 && tw >= 320 && th >= 200) { win_w = tw; win_h = th; }
    }
    window_ = std::make_unique<Window>(WindowConfig{ .width = win_w, .height = win_h,
                                                     .title = "Vibe" });
    // Audio engine starts before Vulkan so a missing audio device just
    // means silence; the renderer keeps working. Clip files are loaded
    // by name from assets/sounds/; missing files are ignored.
    audio_ = std::make_unique<AudioEngine>();
    audio_->load_clip(ClipID::Shot,   "assets/sounds/shot.ogg");
    audio_->load_clip(ClipID::Impact, "assets/sounds/impact.ogg");
    audio_->load_clip(ClipID::Jump,   "assets/sounds/jump.ogg");
    audio_->load_clip(ClipID::Land,   "assets/sounds/land.ogg");
    // Single shot cut from laser_gun_#4 (0.045-0.225 s, 100 ms fade, -3.5 dB
    // to sit at the old shot level).
    audio_->load_clip(ClipID::Laser,  "assets/sounds/laser_shot.wav");
    // laser_gun_#3 as recorded: a 12-round burst (80 ms apart) plus its echo.
    audio_->load_clip(ClipID::AaBurst, "assets/sounds/aa_burst.wav");
    audio_->load_clip(ClipID::SamLaunch, "assets/sounds/sam_launch.wav");
    audio_->load_clip(ClipID::Splash, "assets/sounds/water_splash.wav");
    audio_->load_clip(ClipID::Rain, "assets/sounds/rain_bed.wav");
    audio_->load_clip(ClipID::RocketRoar, "assets/sounds/rocket_roar.wav");
    audio_->load_clip(ClipID::Beep, "assets/sounds/beep.wav");
    // Full jet flyby recordings; tick_jet_audio() times their peak to each
    // jet's predicted closest approach.
    audio_->load_clip(ClipID::JetFlyby1, "assets/sounds/jet_flyby_1.mp3");
    audio_->load_clip(ClipID::JetFlyby2, "assets/sounds/jet_flyby_2.mp3");
    audio_->load_clip(ClipID::JetFlyby3, "assets/sounds/jet_flyby_3.mp3");
    // Explosions (play_explosion_audio picks close / far by distance).
    for (int i = 0; i < 5; ++i) {
        audio_->load_clip(ClipID(int(ClipID::ExplClose1) + i), "assets/sounds/expl_close_" + std::to_string(i + 1) + ".mp3");
        audio_->load_clip(ClipID(int(ClipID::ExplFar1) + i),   "assets/sounds/expl_far_" + std::to_string(i + 1) + ".mp3");
    }
    for (int i = 0; i < 3; ++i)
        audio_->load_clip(ClipID(int(ClipID::Thunder1) + i), "assets/sounds/thunder_" + std::to_string(i + 1) + ".mp3");
    init_vulkan();
    init_pipeline_cache();
    init_swapchain();
    init_depth_image();
    init_commands();
    init_upload_ring();     // frames-in-flight staging + gfx timeline semaphore
    init_sync();
    init_gpu_query_pool();
    // FSR2 needs the swapchain (displaySize) before its context can be
    // created; the per-frame dispatch path (Phase 4) is still gated on
    // rt_.fsr2_enabled, so the toggle remains the sole runtime switch.
    init_fsr2();
    init_dlss();   // Streamline DLSS output (no-op without SDK / unsupported GPU)
    // FSR3 init is lazy (only fires when the user picks fsr_backend=1
    // in the UI). The ffx-api SDK's session-2 status was crash-on-
    // create at startup; deferring keeps the engine launchable even
    // if the SDK's create path AVs in the user's environment.
    // Loader is presentable from here on. Each subsequent step is the
    // long-running ones (init_world / init_rt / shadow bake) that the
    // user used to stare at a blank window for.
    present_loader_frame("Building world",        0.05f);
    init_readback_buffer();
    // load_settings() runs AFTER init_world (it restores player pose
    // that init_world otherwise overwrites). But init_world needs to
    // know whether to source the heightmap from the shader-matching
    // FBM (when raymarch is enabled) or the default FastNoiseLite path.
    // Peek just that one flag here.
    preload_terrain_raymarch_flag();
    // Kick off JPG decodes for the world texture set BEFORE init_world
    // starts the multi-second terrain mesh + chunk upload. The decodes
    // run on worker threads while the main thread is busy with terrain;
    // by the time init_textures() needs them they're (mostly) already
    // available and only the serial graphics-queue uploads remain.
    start_world_texture_decodes();
    jobs::init();        // worker pool (world build, frame jobs)
    init_world();
    present_loader_frame("Loading skybox",        0.30f);
    texture_upload_batch_begin();
    init_skybox();
    init_textures();
    texture_upload_batch_flush(device_, allocator_, graphics_queue_);
    if (physics_) physics_->finish_async();     // the terrain's collision shape, built during the texture loading
    present_loader_frame("Loading settings",      0.40f);
    load_settings();
    apply_vendor_upscaler_defaults();
    // load_settings reads rt_.render_scale from disk, but init_swapchain
    // already ran with the default scale of 1.0 — render_extent_ is wrong
    // until something forces a recreate. If the saved scale differs from
    // 1.0, apply it now so the user's preferred scale survives a restart
    // without a manual Apply click.
    // Same for hdr_output: init_swapchain picked the colour space from the
    // DEFAULT before the cfg was read.
    const bool hdr_mismatch = (rt_.hdr_output != 0) != hdr_active_;
    if (std::abs(rt_.render_scale - 1.0f) > 1e-3f || hdr_mismatch) {
        apply_render_scale();
    }
    // FSR2 mip-bias depends on the loaded fsr2_enabled flag + the actual
    // render_extent_, both of which are only known after the two calls
    // above. The recreated sampler is what write_scene_descriptors_once
    // below captures — without this the FSR2 path runs with bias 0.
    update_texture_sampler_for_fsr2();
    init_descriptors();
    present_loader_frame("Building ray tracing",  0.50f);
    init_rt();
    {
        VkImageView alb[kTextureCount]{}, nrm[kTextureCount]{};
        for (int i = 0; i < kTextureCount; ++i) {
            alb[i] = albedo_textures_[i].view;
            nrm[i] = normal_textures_[i].view;
        }
        VkImageView spom[kSpomMaterialCount]{};
        for (int i = 0; i < kSpomMaterialCount; ++i) {
            spom[i] = spom_height_textures_[i].view;
        }
        // Bake grass mask + fog wisp BEFORE descriptor write so the
        // binding-13/14 image views are real (not the slot-0 albedo
        // fallback). Both are CPU-baked, ~1 MB total, runs once.
        init_grass_mask_texture();
        init_terrain_aux_texture();
        init_fog_wisp_texture();
        // ReSTIR reservoir SSBOs at scene_desc bindings 15/16. Allocated
        // here so the descriptor write below captures live handles.
        init_restir();
        write_scene_descriptors_once(device_, scene_desc_set_,
                                     scene_ubo_buffer_, tlas_, materials_buffer_,
                                     prev_transforms_buffer_,
                                     alb, nrm, kTextureCount, texture_sampler_,
                                     spom, kSpomMaterialCount,
                                     grass_mask_.view,
                                     fog_->wisp.view,
                                     // Session 3 aliases both bindings to buf[0]
                                     // (see restir.cpp::init_restir). When session
                                     // 5 ships proper FIF descriptor sets, restore
                                     // buf[1 - write_slot] / buf[write_slot] here.
                                     reservoir_buf_[0],
                                     reservoir_buf_[0]);
        write_depth_copy_binding();   // binding 27 (prepass depth copy)
    }
    present_loader_frame("Compiling pipelines",   0.70f);
    init_pipeline();
    init_voxel();
    init_doors();
    init_sun_horizon();          // before the deferred / fog sets that bind it
    init_cloud_shadow_map();     // likewise (scene set binding 32, deferred 23)
    init_deferred_lighting();
    init_terrain_pipelines();
    init_terrain_raymarch_pipeline();
    init_terrain_patches();      // before the low-resolution targets: their size depends on it
    init_terrain_bake();
    init_terrain_water_pipeline();
    init_terrain_raymarch_compose_pipeline();
    init_terrain_raymarch_lowres();
    init_shadow_lr();
    // SVGF GI denoiser storage image (Session 1 — see docs/svgf_plan.md).
    // cube.frag writes raw irradiance to it via imageStore. No
    // consumer in session 1 — just allocates + binds.
    init_svgf_targets();
    // VRS attachment is sized in LR-tile units, so it must come after
    // init_terrain_raymarch_lowres() has set tr_->lr_extent.
    init_vrs_attachment();
    init_sun_shadow_pipeline();
    init_grass_pipeline();
    init_scatter_pipeline();
    present_loader_frame("Baking shadows",        0.85f);
    // Heightmap raw-heights texture (binding 8) for the procedural
    // raymarched terrain shader. Uploaded once, after descriptors and
    // init_world have populated terrain_data_. Cheap (~16 MB at 2048²
    // R32F).
    init_terrain_height_texture();
    init_terrain_height_full_texture();
    // Heightmap shadow texture must be baked AFTER descriptors and the
    // texture itself can be created; depends on terrain_data_ which
    // init_world fills. Done here so the descriptor write at the end
    // of rebuild_terrain_shadow_texture lands on the live scene set.
    rebuild_terrain_shadow_texture(/*defer_bake=*/true);
    // Async worker thread bakes the heightmap shadow off the main
    // thread. Started after the initial full bake so the texture is
    // valid before grass renders.
    start_terrain_shadow_worker();
    enqueue_terrain_shadow_rebake(terrain_shadow_sun_dir_);   // stream the real bake
    // Sun shadow map (binding 7). Created after descriptors so the
    // one-shot binding-7 write at the end of init_sun_shadow_resources
    // lands on the live scene set.
    init_sun_shadow_resources();
    present_loader_frame("Initialising UI",       0.95f);
    // Order: TAA owns linear_sampler_ + history images that compose samples;
    // bloom must run before compose so binding 3's bloom_->mip_views[0] is
    // live when compose's descriptor set is written.
    init_taa();
    init_taau();
    init_bloom();
    init_compose();
    // SVGF moments + 3-pass a-trous pipelines (4a-deep). Must run
    // AFTER init_taa() (reuses linear_sampler_ + taa_->vert_module) and
    // AFTER init_svgf_targets() (binds the moments + atrous storage
    // image views). Toggled per-frame via rt_.svgf_enabled; pipelines
    // stand up regardless so the toggle is a per-frame branch.
    init_svgf_passes();
    // Must run AFTER init_taa() — needs linear_sampler_ for the binding-18
    // descriptor write that lets cube.frag sample the half-res shadow.
    init_shadow_lr_pipeline();
    // Must run AFTER init_taa() — grass raymarch reuses taa_->vert_module
    // as its fullscreen-triangle vertex shader.
    init_grass_raymarch_pipeline();
    init_wm_cache();              // (after init_taa: the same full-screen triangle)
    init_grass_trail();           // (the same triangle; scene bindings 41, 42)
    init_wind_fluid();            // (scene binding 45)
    init_viewmodel();
    init_spacejet();
    init_fx();
    init_flags();
    init_vegetation_pipelines();
    init_clutter_shadow();
    init_imgui();
    // (The terrain shaders go on compiling in the background; the game starts on the lite build:
    // finish_terrain_raymarch_pipeline(0) in the frame takes them over when they are ready.)
    if (!terrain_raymarch_lite_pipeline_) finish_terrain_raymarch_pipeline(1);
    update_scene_ubo();
    present_loader_frame("Ready",                 1.00f);
    // Loader frames advanced frame_number_ — reset so --frames N and
    // screenshot-after counters in run() count from the first real
    // gameplay frame, not from "real frames + N loader frames".
    frame_number_ = 0;
    if (window_) glfwSetWindowTitle(window_->handle(), "Vibe");
    initialized_ = true;
    log::info("VulkanEngine::init() done");
}


// init_imgui / destroy_imgui live in vk_ui.cpp.



// build_hud_ui / build_menu_ui live in vk_ui.cpp.





// Called on a device-lost (device handle still valid per spec). Asks
// the driver — via VK_EXT_device_fault — for the faulting GPU
// address(es) and vendor fault code, and logs them. This is the only
// way to get hard data on an intermittent device-lost that won't
// reproduce under (GPU-slowing) validation.
void VulkanEngine::dump_device_fault() {
    if (!device_fault_supported_ || !device_) return;
    auto pfn = reinterpret_cast<PFN_vkGetDeviceFaultInfoEXT>(
        vkGetDeviceProcAddr(device_, "vkGetDeviceFaultInfoEXT"));
    if (!pfn) { log::warn("[devfault] vkGetDeviceFaultInfoEXT not loadable"); return; }

    VkDeviceFaultCountsEXT counts{};
    counts.sType = VK_STRUCTURE_TYPE_DEVICE_FAULT_COUNTS_EXT;
    VkResult r = pfn(device_, &counts, nullptr);
    if (r != VK_SUCCESS && r != VK_INCOMPLETE) {
        log::errorf("[devfault] count query failed (VkResult=%d)", static_cast<int>(r));
        return;
    }
    std::vector<VkDeviceFaultAddressInfoEXT> addrs(counts.addressInfoCount);
    std::vector<VkDeviceFaultVendorInfoEXT>  vend(counts.vendorInfoCount);
    counts.vendorBinarySize = 0;   // don't request the opaque vendor blob
    VkDeviceFaultInfoEXT info{};
    info.sType = VK_STRUCTURE_TYPE_DEVICE_FAULT_INFO_EXT;
    info.pAddressInfos = addrs.empty() ? nullptr : addrs.data();
    info.pVendorInfos  = vend.empty()  ? nullptr : vend.data();
    r = pfn(device_, &counts, &info);
    if (r != VK_SUCCESS && r != VK_INCOMPLETE) {
        log::errorf("[devfault] info query failed (VkResult=%d)", static_cast<int>(r));
        return;
    }
    log::errorf("[devfault] --- GPU DEVICE FAULT ---");
    log::errorf("[devfault] description: %s", info.description);
    for (uint32_t i = 0; i < counts.addressInfoCount; ++i) {
        const VkDeviceFaultAddressInfoEXT& a = addrs[i];
        log::errorf("[devfault] addr[%u] type=%d reported=0x%016llX precision=0x%llX",
                    i, static_cast<int>(a.addressType),
                    static_cast<unsigned long long>(a.reportedAddress),
                    static_cast<unsigned long long>(a.addressPrecision));
    }
    for (uint32_t i = 0; i < counts.vendorInfoCount; ++i) {
        const VkDeviceFaultVendorInfoEXT& v = vend[i];
        log::errorf("[devfault] vendor[%u] '%s' code=0x%016llX data=0x%016llX",
                    i, v.description,
                    static_cast<unsigned long long>(v.vendorFaultCode),
                    static_cast<unsigned long long>(v.vendorFaultData));
    }
    log::errorf("[devfault] --- end GPU DEVICE FAULT ---");
}

void VulkanEngine::run(const RunOptions& opts) {
    {   // a lost device anywhere (not only in the frame's try block below): dump the GPU's fault
        static VulkanEngine* self = nullptr;
        self = this;
        g_device_lost_hook = [] { if (self) { self->device_lost_ = true; self->dump_device_fault(); } };
    }
    opts_ = opts;
    if (opts_.deferred >= 0) deferred_lighting_active_ = (opts_.deferred != 0);
    log::infof("run(): max_frames=%d screenshot='%s' after=%d",
               opts_.max_frames, opts_.screenshot_path.c_str(),
               opts_.screenshot_after_frames);

    bool headless = opts_.max_frames > 0 || !opts_.screenshot_path.empty();
    mp_init();
    bool autodemo = opts_.autodemo_seconds > 0.0f;
    if (!headless) window_->set_cursor_captured(true);
    if (autodemo) {
        window_->set_cursor_captured(true);
        state_ = State::Playing;
        log::infof("[autodemo] running for %.2fs (forward+turn+fire)",
                   opts_.autodemo_seconds);
    }

    // FPS cap = primary monitor's refresh rate. Vulkan's FIFO present mode
    // *should* already pace the swap to the display, but we've measured
    // 500+ fps in this loop on some setups (driver fast-path / multi-
    // monitor mismatch). Letting the CPU outrun the GPU saturates the
    // RT pipeline and triggers TDR (DEVICE_LOST). A simple sleep-to-target
    // at the start of each frame caps the loop at the monitor's rate
    // without changing visuals. Falls back to 144 if GLFW can't read the
    // monitor (rare on Windows).
    int monitor_hz = 0;
    if (auto* monitor = glfwGetPrimaryMonitor()) {
        if (auto* mode = glfwGetVideoMode(monitor)) {
            monitor_hz = mode->refreshRate;
        }
    }
    if (monitor_hz <= 0) monitor_hz = 144;
    const auto frame_target = std::chrono::nanoseconds(1'000'000'000 / monitor_hz);
    log::infof("fps cap: %d Hz (monitor refresh)", monitor_hz);
    FramePacer pacer(frame_target);

    auto last = std::chrono::steady_clock::now();
    auto demo_start = last;
    prev_player_position_ = player_.position;

    // Test hook: VIBE_TEST_SITE=<site>[,<east>,<south>,<yaw>,<pitch>,<height>] starts at that
    // site, that far from its arrival point (captures; with VIBE_TEST_HOLD the camera stays).
    if (const char* ts = std::getenv("VIBE_TEST_SITE")) {
        int k = 0; float dx = 0.0f, dz = 0.0f, yaw = 0.0f, pitch = 0.0f, up = 0.0f;
        std::sscanf(ts, "%d,%f,%f,%f,%f,%f", &k, &dx, &dz, &yaw, &pitch, &up);
        travel_to_site(std::abs(k));
        player_.position += glm::vec3(dx, up, dz);
        if (k < 0) {
            // (a negative number: arrive as a traveller does - in a wandering chart anchored
            // at that spot; globe_update then has to take up the site's chart, or not)
            const globe::Chart sc = globe::current();
            GlobeChartRef w; w.c = globe::near_chart(sc, double(player_.position.x), double(player_.position.z));
            globe_enter(w);
        }
        player_.yaw = yaw; player_.pitch = pitch;
        prev_player_position_ = player_.position;
        if (up > 3.0f) fly_mode_ = true;
    }

    while (!window_->should_close()) {
        { CPU_SCOPE("poll-events"); window_->poll_events(); }
        in_frame_ = false;
        if (resize_requested_) { recreate_swapchain(); tr_lowres_recreate_pending_ = false; }      // (rebuilds the low-resolution terrain targets too)
        if (tr_lowres_recreate_pending_) { tr_lowres_recreate_pending_ = false; recreate_terrain_raymarch_lowres(); }
        in_frame_ = true;

        // Frame-rate cap (see frame_pacer.h). Skipped in headless mode
        // (max_frames / screenshot) so test runs are not throttled.
        // With DLSS Frame Generation the displayed rate is (generated + 1) x
        // the rendered one: cap rendering at refresh / multiplier so no
        // generated frame is thrown away by the mailbox swapchain.
        {
            int fg_mult = dlss_fg_active() ? dlssg_frames_sent_ + 1 : 1;
            if (fsr3_->swapchain_active && fsr3_->fg_context_valid && rt_.fg_enabled && !fg_runtime_disabled_)
                fg_mult = 2;   // FSR 3 frame generation: one generated frame per rendered one
            pacer.set_period(frame_target * fg_mult);
        }
        if (!headless) { CPU_SCOPE("pacer"); pacer.wait(); }
        { CPU_SCOPE("reflex"); sl_frame_begin(); }   // Streamline frame token, Reflex sleep, simulation-start marker
        anti_lag_mark(false);   // AMD Anti-Lag 2: input stage

        auto now = std::chrono::steady_clock::now();
        frame_clock_s_ = glfwGetTime();        // (the moment this frame's time step is measured to: what the network stamps and samples with)
        float frame_dt = std::chrono::duration<float>(now - last).count();
        // Hitch diagnostics: section timings, logged for any slow frame.
        auto hz_mark = [](std::chrono::steady_clock::time_point& t) {
            auto n = std::chrono::steady_clock::now();
            float ms = std::chrono::duration<float, std::milli>(n - t).count();
            t = n;
            return ms;
        };
        auto hz_t = now;
        float hz_ms[6] = {};
        const float hz_frame_ms = frame_dt * 1000.0f;
        // (a hitch is a frame much longer than the ones around it - not every frame of a scene
        // that runs under 36 fps, which filled the log with thousands of these - and at most
        // two lines a second)
        static float hz_avg_ms = 16.0f;
        static uint64_t hz_last_frame = 0;
        const bool hz_is_hitch = hz_frame_ms > std::max(28.0f, 1.8f * hz_avg_ms) && frame_number_ > 30 &&
                                 frame_number_ - hz_last_frame > 30;
        hz_avg_ms = glm::mix(hz_avg_ms, std::min(hz_frame_ms, 200.0f), 0.03f);
        if (hz_is_hitch) hz_last_frame = frame_number_;
        if (hz_is_hitch)
            log::warnf("[hitch] frame %.1f ms | prev: update %.1f, jets+fx %.1f, ui %.1f, fence %.1f, acquire %.1f, draw %.1f "
                       "(draw: tlas-wait %.1f, cpu-build %.1f, record+submit %.1f)",
                       hz_frame_ms, hitch_prev_[0], hitch_prev_[1], hitch_prev_[2], hitch_prev_[3],
                       hitch_prev_[4], hitch_prev_[5], hitch_draw_[0], hitch_draw_[1], hitch_draw_[2]);
        {
            // Rolling average of the draw() split (every ~5 s): how long the
            // CPU waits on the GPU vs builds / records while the GPU idles.
            static double acc[8] = {}; static int acc_n = 0;
            acc[0] += hitch_draw_[0]; acc[1] += hitch_draw_[1]; acc[2] += hitch_draw_[2]; acc[3] += hz_frame_ms;
            acc[4] += hitch_prev_[0]; acc[7] += hitch_prev_[1] + hitch_prev_[2];   // update / jets+fx+ui
            acc[5] += hitch_prev_[3];                                      // slot fence (frame N-2)
            acc[6] += hitch_prev_[4];                                      // acquire
            if (++acc_n >= 300) {
                log::infof("[cpu] avg frame %.2f ms | update %.2f, jets+fx+ui %.2f, slot-fence wait %.2f, acquire %.2f, "
                           "draw: pre-wait %.2f, cpu-build %.2f, record+submit %.2f",
                           acc[3] / acc_n, acc[4] / acc_n, acc[7] / acc_n, acc[5] / acc_n, acc[6] / acc_n,
                           acc[0] / acc_n, acc[1] / acc_n, acc[2] / acc_n);
                for (double& a : acc) a = 0.0;
                acc_n = 0;
            }
        }
        last = now;
        const float real_dt = std::min(frame_dt, 2.0f);
        if (frame_dt > 1.0f / 15.0f) frame_dt = 1.0f / 15.0f;
        if (frame_dt < 1e-6f)        frame_dt = 1.0f / 240.0f;
        // Headless captures (--frames N --screenshot) need bit-stable physics
        // and spawn timing across runs; wallclock-derived dt varies by ~30%
        // run-to-run depending on shader compile, GPU contention, etc. Pin
        // it to a fixed 60 Hz step so crate spawns + Jolt integration land
        // at the same place every time and SAD comparisons are meaningful.
        // (VIBE_TEST_REALTIME=1: the clock's time all the same - tests of what must be even in real
        //  time: the cars' motion, the network)
        static const bool kRealTime = std::getenv("VIBE_TEST_REALTIME") != nullptr;
        if ((opts_.max_frames > 0 || !opts_.screenshot_path.empty()) && !kRealTime) {
            frame_dt = 1.0f / 60.0f;
        } else if (state_ == State::Playing) {
            sim_debt_s_ += std::max(0.0f, real_dt - frame_dt);      // (see sim_debt_s_)
        }
        last_frame_dt_ = frame_dt;
        ema_fps_ = ema_fps_ == 0.0f
                   ? 1.0f / frame_dt
                   : (ema_fps_ * 0.9f + (1.0f / frame_dt) * 0.1f);

        InputFrame in = window_->consume_input(input_);
        {   // zoomed in (aiming), the same hand movement turns the view less: the picture moves as much as ever
            const double zoom_k = double(std::tan(glm::radians(view_fov_deg_) * 0.5f) / std::tan(glm::radians(glm::clamp(rt_.fov_deg, 55.0f, 100.0f)) * 0.5f));
            in.mouse_dx *= zoom_k;
            in.mouse_dy *= zoom_k;
            in.look_x *= float(zoom_k);
            in.look_y *= float(zoom_k);
        }
        // Typing into a text field (a preset name in the edit panel): the
        // keys belong to the text, not to movement / E / Q / R.
        if (imgui_initialized_ && ImGui::GetIO().WantTextInput) {
            const double mdx = in.mouse_dx, mdy = in.mouse_dy;
            in = InputFrame{};
            in.mouse_dx = mdx; in.mouse_dy = mdy;
        }
        // The menu is waiting for a key to bind (ui_menu.cpp): nothing reaches the game, and
        // the Esc that cancels it does not also close the menu.
        if (rebind_action_ >= 0) {
            in = InputFrame{};
            prev_menu_key_ = true;
        }
        in_flashlight_ = in.flashlight;
        in_using_pad_ = window_->using_gamepad();
        in_pad_connected_ = in.pad_connected;
        // F12 in-game capture: queue a numbered PPM in the working dir.
        // The render frame below picks it up and writes to disk.
        if (in.screenshot && pending_screenshot_path_.empty()) {
            // PNG in screenshots/, named by date and time (never overwrites).
            char buf[96];
            std::error_code ec;
            std::filesystem::create_directories("screenshots", ec);
            const std::time_t tt = std::time(nullptr);
            std::tm tmv{};
            localtime_s(&tmv, &tt);
            std::snprintf(buf, sizeof(buf), "screenshots/vibe_%04d-%02d-%02d_%02d-%02d-%02d_%02d.png",
                          tmv.tm_year + 1900, tmv.tm_mon + 1, tmv.tm_mday, tmv.tm_hour, tmv.tm_min, tmv.tm_sec,
                          ingame_screenshot_counter_++ % 100);
            pending_screenshot_path_ = buf;
            log::infof("[screenshot] F12 queued %s", buf);
        }
        loop_test_input(in, frame_dt, autodemo, headless, now, demo_start);
        jeep_tick(in, frame_dt);           // (jeep.cpp: the driver's controls, before the physics)
        loop_handle_keys(in, frame_dt);

        loop_weapons_and_edit(in, frame_dt);

        loop_simulate(in, frame_dt);
        jeep_after_sim(frame_dt);
        jeep_net_send(frame_dt);           // (multiplayer: the cars simulated here, as they are now)

        loop_upkeep(in, frame_dt);
        hz_ms[0] = hz_mark(hz_t);
        // Spacejet flyovers: advance active flights, spawn next wave.
        // The pause menu freezes the whole simulation (jets, bombs, AA,
        // particles, their sounds) like it already froze physics; the
        // ticks still run with dt = 0 so everything stays drawn.
        // (In a multiplayer session with others the world is shared and does not stop for one player's menu.)
        const bool world_runs = state_ == State::Playing || (net_ && !net_->players().empty());
        const float sim_dt = world_runs ? frame_dt : 0.0f;
        if (audio_) audio_->set_paused(!world_runs);
        // Aircraft and bombs keep real time at any frame rate (several small
        // steps on a slow frame instead of slowing down): in multiplayer the
        // same airstrike must land at the same moment for everybody.
        {
            float rem = sim_dt;
            for (int n = 0; n < 8; ++n) {
                const float step = std::min(rem, 0.05f);
                { CPU_SCOPE("jets"); tick_spacejet(step); }
                { CPU_SCOPE("fx"); tick_fx(step); }
                rem -= step;
                if (rem <= 1e-5f) break;
            }
        }
        { CPU_SCOPE("water-fx"); tick_water_fx(sim_dt); }
        { CPU_SCOPE("aa"); tick_aa(sim_dt); }
        globe_update();
        { CPU_SCOPE("rocket"); tick_rocket(sim_dt); }
        maybe_autosave_settings(frame_dt);
        hz_ms[1] = hz_mark(hz_t);

        // ImGui frame begins regardless; we just don't fill it when playing.
        if (imgui_initialized_) {
            CPU_SCOPE("imgui");
            // The menu by controller (D-pad / stick to move, A to press, B back): only while
            // the menu is open - in play the HUD's windows must never take the buttons.
            if (state_ != State::Playing && in_pad_connected_ && rebind_action_ < 0)
                ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;
            else
                ImGui::GetIO().ConfigFlags &= ~ImGuiConfigFlags_NavEnableGamepad;
            ImGui_ImplVulkan_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();
            build_hud_ui();
            mp_hud();
            jeep_hud();
            site_labels_hud();                 // the sites' names, from high up (sites.cpp)
            anim_debug_overlay();
            if (terrain_shaders_compiling()) {
                // First start: say why the ground looks plain for a minute.
                ImDrawList* dl = ImGui::GetForegroundDrawList();
                const ImVec2 disp = ImGui::GetIO().DisplaySize;
                const char* msg = "Preparing the detailed terrain for your graphics card - first start only";
                const ImVec2 ts = ImGui::CalcTextSize(msg);
                const ImVec2 p0((disp.x - ts.x) * 0.5f, disp.y * 0.06f);
                dl->AddRectFilled(ImVec2(p0.x - 14.0f, p0.y - 8.0f), ImVec2(p0.x + ts.x + 14.0f, p0.y + ts.y + 8.0f), IM_COL32(0, 0, 0, 150), 6.0f);
                dl->AddText(p0, IM_COL32(255, 255, 255, 230), msg);
            }
            build_menu_ui();
            // VIBE_TEST_SCALE_UI="300:0.72,600:0.5": the render scale changed at those frames from
            // here, where the menu changes it (in the middle of a frame). With
            // VIBE_TEST_SCALE_UI_OLD=1 the targets are rebuilt on the spot, as they were before
            // 2026-11-23 - that loses the device (the test that the cause was found).
            {
                static const char* seq = std::getenv("VIBE_TEST_SCALE_UI");
                static const bool old_way = std::getenv("VIBE_TEST_SCALE_UI_OLD") != nullptr;
                for (const char* p = seq; p && *p; ) {
                    int fr = 0; float sc = 0.0f;
                    if (std::sscanf(p, "%d:%f", &fr, &sc) == 2 && uint64_t(fr) == uint64_t(frame_number_)) {
                        rt_.render_scale = sc;
                        if (old_way) in_frame_ = false;
                        apply_render_scale();
                        in_frame_ = true;
                    }
                    while (*p && *p != ',') ++p;
                    if (*p == ',') ++p;
                }
            }
            ImGui::Render();
        }

        // Frame body wrapped in a try/catch so any vk_check throw from a
        // sync/submit call is caught HERE — at the throw site's stack frame,
        // before the runtime_error's heap message can be invalidated by
        // unwinding through the engine's deeper destructors. We log the type
        // and break the loop; shutdown() then sees device_lost_ and skips the
        // vkDeviceWaitIdle that would otherwise stall on the dead GPU.
        try {
            // SwapChain-proxy teardown gate — runs BEFORE acquire so we
            // don't leak a swapchain semaphore. When the user toggles
            // FSR off (or FG turns off), the SDK's proxied present
            // doesn't gracefully pass-through — symptom is world
            // geometry stops reaching the screen, only sky-clear
            // survives. Tear the proxy down + rebuild plain swapchain
            // here, then `continue` the loop without touching the
            // half-prepared frame state.
            {
                const bool fg_should_run = rt_.fsr2_enabled && rt_.fg_enabled &&
                                           !fsr3_->fg_fatal && !fg_runtime_disabled_ &&
                                           rt_.fsr_backend == 1;
                if (!fg_should_run && fsr3_->swapchain_active) {
                    log::info("[fsr3 sc] FG/FSR turned off — tearing proxy down + restoring plain swapchain");
                    vkDeviceWaitIdle(device_);
                    destroy_fsr3_swapchain();
                    for (auto v : swapchain_views_) vkDestroyImageView(device_, v, nullptr);
                    swapchain_views_.clear();
                    swapchain_images_.clear();
                    for (auto s : render_semaphores_) {
                        if (s) vkDestroySemaphore(device_, s, nullptr);
                    }
                    render_semaphores_.clear();
                    init_swapchain();
                    continue;   // re-enter loop with fresh swapchain
                }
                // Deferred from draw(): create the FG SwapChain proxy between
                // frames (it replaces the swapchain the frame would use).
                if (fsr3_->swapchain_init_pending) {
                    fsr3_->swapchain_init_pending = false;
                    if (fg_should_run && !fsr3_->swapchain_active && !fsr3_->swapchain_fatal) {
                        vkDeviceWaitIdle(device_);
                        init_fsr3_swapchain();
                        continue;
                    }
                }
            }
            hz_ms[2] = hz_mark(hz_t);
            sl_marker(1);   // PCL simulation end
            auto& frame = current_frame();
            VIBE_VK_CHECK(vkWaitForFences(device_, 1, &frame.render_fence,
                                           VK_TRUE, UINT64_MAX),
                           "vkWaitForFences");
            hz_ms[3] = hz_mark(hz_t);
            // GPU watchdog: this slot's previous-iteration GPU work is
            // now finished (we just waited on its fence) so its
            // timestamps are safe to read non-blocking. Internally
            // logs a WARN at >1500 ms (Windows TDR is ~2 s).
            read_gpu_timestamps_for_slot(frame_number_ % kFrameOverlap);
            prof_read_slot(frame_number_ % kFrameOverlap);

            uint32_t img_index = 0;
            VkResult acq = acquire_next_image(UINT64_MAX,
                                              frame.swapchain_semaphore,
                                              VK_NULL_HANDLE, &img_index);
            if (acq == VK_ERROR_OUT_OF_DATE_KHR) { resize_requested_ = true; continue; }
            if (acq != VK_SUCCESS && acq != VK_SUBOPTIMAL_KHR) {
                VIBE_VK_CHECK(acq, "vkAcquireNextImageKHR");
            }

            VIBE_VK_CHECK(vkResetFences(device_, 1, &frame.render_fence),
                           "vkResetFences");
            VIBE_VK_CHECK(vkResetCommandBuffer(frame.command_buffer, 0),
                           "vkResetCommandBuffer");

            hz_ms[4] = hz_mark(hz_t);
            draw(img_index);
            hz_ms[5] = hz_mark(hz_t);
            for (int k = 0; k < 6; ++k) hitch_prev_[k] = hz_ms[k];

            ++frame_number_;
            cpuprof::end_frame();
            anim_time_ = static_cast<float>(std::fmod(glfwGetTime(), 3600.0));
            // VIBE_TEST_ANIM_TIME=<seconds>: the animation clock stands still at that time (water,
            // wind, clouds the same in every capture: pictures of two builds can be compared).
            { static const char* at = std::getenv("VIBE_TEST_ANIM_TIME"); if (at) anim_time_ = static_cast<float>(std::atof(at)); }
            {
                static bool first_logged = false;
                if (!first_logged) {
                    first_logged = true;
                    log::infof("[init] first game frame submitted %.0f ms after launch",
                               glfwGetTime() * 1000.0);
                    {   // video memory in use (all allocations of this process, per the driver)
                        VmaBudget bud[VK_MAX_MEMORY_HEAPS]{};
                        vmaGetHeapBudgets(allocator_, bud);
                        VkPhysicalDeviceMemoryProperties mp{};
                        vkGetPhysicalDeviceMemoryProperties(physical_device_, &mp);
                        double used = 0.0, budget = 0.0, ours = 0.0;
                        for (uint32_t h = 0; h < mp.memoryHeapCount; ++h) {
                            if (!(mp.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)) continue;
                            used += double(bud[h].usage); budget += double(bud[h].budget); ours += double(bud[h].statistics.allocationBytes);
                        }
                        log::infof("[vram] %.0f MB in use by the game (%.0f MB in engine allocations) of %.0f MB available",
                                   used / 1048576.0, ours / 1048576.0, budget / 1048576.0);
                    }
                }
            }
            prof_sweep_tick();
            auto_scale_tick();
            // VIBE_LOG_MEM=1: every 600 frames, what the game holds - to see whether moving
            // about the world leaves anything behind (it must come back to where it was).
            if (static const bool lm = std::getenv("VIBE_LOG_MEM") != nullptr; lm && frame_number_ % 600 == 0) {
                VmaBudget bud[VK_MAX_MEMORY_HEAPS]{};
                vmaGetHeapBudgets(allocator_, bud);
                VkPhysicalDeviceMemoryProperties mp{};
                vkGetPhysicalDeviceMemoryProperties(physical_device_, &mp);
                double used = 0.0, ours = 0.0; uint64_t allocs = 0, blocks = 0;
                for (uint32_t h = 0; h < mp.memoryHeapCount; ++h) {
                    if (!(mp.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)) continue;
                    used += double(bud[h].usage); ours += double(bud[h].statistics.allocationBytes);
                    allocs += bud[h].statistics.allocationCount; blocks += bud[h].statistics.blockCount;
                }
                double ws = 0.0, pv = 0.0;
                process_memory_mb(ws, pv);
                log::infof("[mem] frame %llu at (%.0f, %.0f): vram %.0f MB (engine %.0f MB, %llu allocations in %llu blocks), ram %.0f MB in use, %.0f MB committed | road stretches %zu, mesh parts %zu, boxes %zu, frame %.2f ms",
                           static_cast<unsigned long long>(frame_number_), double(player_.position.x), double(player_.position.z),
                           used / 1048576.0, ours / 1048576.0, static_cast<unsigned long long>(allocs), static_cast<unsigned long long>(blocks),
                           ws, pv, hw_chunks_.size(), mesh_parts_now().size(), world_.aabbs.size() + wv_cols_.size(), double(frame_dt) * 1000.0);
            }
            if (opts_.max_frames >= 0 &&
                static_cast<int>(frame_number_) >= opts_.max_frames) {

                log::infof("max_frames reached (%d), exiting loop",
                           opts_.max_frames);
                break;
            }
            if (!opts_.screenshot_path.empty() && screenshot_taken_) {
                log::info("screenshot captured, exiting loop");
                break;
            }
        } catch (const std::exception& e) {
            // typeid().name() is rdata-stored — safe even if e.what()'s heap
            // string has been freed by unwind.
            log::errorf("[run] frame aborted via std::exception (type=%s) — "
                        "marking device_lost and exiting loop",
                        typeid(e).name());
            device_lost_ = true;
            // The device handle is still valid here (spec allows
            // vkGetDeviceFaultInfoEXT after device-lost) — grab the GPU
            // fault info NOW, before shutdown tears the device down.
            dump_device_fault();
            break;
        }
    }
    // Skip the wait if we already know the device is dead — otherwise we
    // hang in vkDeviceWaitIdle for several seconds before shutdown can run.
    if (device_ && !device_lost_) vkDeviceWaitIdle(device_);
    mp_shutdown();
}

void VulkanEngine::shutdown() {
    if (!initialized_) return;
    // If run() flagged the device as lost, skip vkDeviceWaitIdle — waiting on
    // a hung GPU just stalls shutdown for seconds before returning DEVICE_LOST
    // anyway. Subsequent vkDestroy* calls accept a "device_lost" device.
    if (device_ && !device_lost_) vkDeviceWaitIdle(device_);
    log::infof("VulkanEngine::shutdown() (device_lost=%d)",
               device_lost_ ? 1 : 0);

    // Each cleanup step is wrapped so a single failure can't propagate out
    // and abort the whole shutdown — leaving Vulkan / GLFW in a half-torn-
    // down state on the way to process exit. Any exception is logged
    // defensively (exception_ptr inspected via type_info — what() can be
    // unsafe if the exception's internal storage has already been freed by
    // an earlier unwinding step). Cleanup continues regardless.
    auto guarded = [](const char* label, auto&& fn) {
        try {
            fn();
        } catch (const std::exception& e) {
            // typeid().name() lives in rdata — safe to log even if the
            // exception's heap-allocated what() string is freed/poisoned
            // (we've seen 0xDD). e.what() itself is intentionally untouched.
            log::errorf("[shutdown] %s threw std::exception (type=%s)",
                        label, typeid(e).name());
        } catch (...) {
            log::errorf("[shutdown] %s threw non-std exception", label);
        }
    };

    guarded("save_settings",  [&]{ save_settings(); });
    // FSR2 owns Vulkan resources via its scratch + context — destroy
    // first so the device + descriptor pool tear-down below isn't
    // racing with the SDK's internal allocator.
    guarded("destroy_fsr2",   [&]{ destroy_fsr2(); });
    guarded("destroy_dlss",   [&]{ destroy_dlss(); });   // while the allocator still exists
    guarded("dlssg_hudless",  [&]{ dlssg_destroy_hudless(); });
    guarded("retired_buffers", [&]{
        for (auto& c : pending_craters_) { retire_buffer(c.stage, c.stage_alloc); retire_buffer(c.mask_stage, c.mask_alloc); }
        pending_craters_.clear();
        collect_retired_buffers(true);
    });
    guarded("destroy_rr_guides", [&]{ destroy_rr_guides(); });
    // Destroy SwapChain proxy first — it holds Vulkan resources tied
    // to the FG context's pacing state.
    guarded("destroy_fsr3_sc",[&]{ destroy_fsr3_swapchain(); });
    guarded("destroy_fsr3_fg",[&]{ destroy_fsr3_fg(); });
    guarded("destroy_fsr3",   [&]{ destroy_fsr3(); });
    guarded("destroy_restir", [&]{ destroy_restir(); });
    guarded("destroy_imgui",  [&]{ destroy_imgui(); });
    guarded("destroy_loader_bg", [&]{ destroy_loader_bg_(); });
    guarded("destroy_viewmodel", [&]{ destroy_viewmodel(); });
    guarded("destroy_spacejet",  [&]{ destroy_spacejet(); });
    guarded("destroy_fx",        [&]{ destroy_fx(); });
    // Reverse of init order: compose first (samples bloom + TAA history),
    // then bloom (samples scene), then TAA (owns linear_sampler_ + history).
    guarded("destroy_compose", [&]{ destroy_compose(); });
    guarded("destroy_bloom",   [&]{ destroy_bloom(); });
    guarded("destroy_taau",    [&]{ destroy_taau(); });
    // SVGF passes own pipelines + descriptors that reference
    // linear_sampler_ (owned by TAA) and the SVGF storage image views
    // (owned by init_svgf_targets). Tear them down BEFORE destroy_taa
    // (sampler) and destroy_svgf_targets (image views).
    guarded("destroy_svgf_passes", [&]{ destroy_svgf_passes(); });
    guarded("destroy_taa",     [&]{ destroy_taa(); });
    // Order: tear down the renderer's GPU dependencies on dynamic-prop world
    // matrices (TLAS instance buffer) BEFORE destroying the physics world that
    // owns those bodies. Otherwise the renderer's deferred destruction could
    // deref freed Jolt body data.
    guarded("destroy_veg",    [&]{ destroy_vegetation(); });
    guarded("destroy_aa",     [&]{ destroy_aa(); });
    guarded("destroy_rocket", [&]{ destroy_rocket(); });
    guarded("destroy_clutter_shadow", [&]{ destroy_clutter_shadow(); });
    guarded("destroy_castle", [&]{ destroy_castle_meshes(); });
    guarded("destroy_rt",     [&]{ destroy_rt(); });
    guarded("physics_.reset", [&]{ physics_.reset(); });
    guarded("jobs::shutdown", [&]{ jobs::shutdown(); });
    guarded("destroy_descriptors", [&]{ destroy_descriptors(); });
    guarded("destroy_upload_ring", [&]{ destroy_upload_ring(); });
    guarded("destroy_meshes", [&]{
        destroy_mesh(allocator_, cube_mesh_);
        destroy_mesh(allocator_, cylinder_mesh_);
    });
    guarded("destroy_gpu_query_pool", [&]{ destroy_gpu_query_pool(); });
    guarded("destroy_skybox", [&]{ destroy_skybox_resources(); });
    guarded("destroy_grass_mask_texture", [&]{ destroy_grass_mask_texture(); });
    guarded("destroy_terrain_aux_texture", [&]{ destroy_terrain_aux_texture(); });
    guarded("destroy_sun_horizon", [&]{ destroy_sun_horizon(); });
    guarded("destroy_cloud_shadow_map", [&]{ destroy_cloud_shadow_map(); });
    guarded("destroy_fog_wisp_texture",   [&]{ destroy_fog_wisp_texture(); });
    guarded("destroy_textures", [&]{ destroy_textures(); });
    guarded("destroy_grass_raymarch_pipeline", [&]{ destroy_grass_raymarch_pipeline(); });
    guarded("destroy_wm_cache", [&]{ destroy_wm_cache(); });
    guarded("destroy_grass_trail", [&]{ destroy_grass_trail(); });
    guarded("destroy_wind_fluid", [&]{ destroy_wind_fluid(); });
    guarded("destroy_grass_pipeline", [&]{ destroy_grass_pipeline(); });
    guarded("destroy_scatter_pipeline", [&]{ destroy_scatter_pipeline(); });
    guarded("destroy_terrain_height_texture", [&]{ destroy_terrain_height_texture(); });
    guarded("destroy_vrs_attachment", [&]{ destroy_vrs_attachment(); });
    guarded("destroy_terrain_raymarch_lowres", [&]{ destroy_terrain_raymarch_lowres(); });
    guarded("destroy_shadow_lr", [&]{ destroy_shadow_lr(); });
    guarded("destroy_svgf_targets", [&]{ destroy_svgf_targets(); });
    guarded("destroy_terrain_raymarch_compose_pipeline", [&]{ destroy_terrain_raymarch_compose_pipeline(); });
    guarded("destroy_terrain_raymarch_pipeline", [&]{ destroy_terrain_raymarch_pipeline(); });
    guarded("destroy_terrain_patches", [&]{ destroy_terrain_patches(); });
    guarded("destroy_terrain_bake", [&]{ destroy_terrain_bake(); });
    guarded("destroy_terrain_water_pipeline", [&]{ destroy_terrain_water_pipeline(); });
    guarded("destroy_terrain_pipelines", [&]{ destroy_terrain_pipelines(); });
    guarded("destroy_sun_shadow_pipeline", [&]{ destroy_sun_shadow_pipeline(); });
    guarded("destroy_voxel", [&]{ destroy_voxel(); });
    guarded("destroy_deferred_lighting", [&]{ destroy_deferred_lighting(); });
    guarded("destroy_pipeline", [&]{ destroy_pipeline(); });
    guarded("destroy_grass", [&]{ destroy_grass(allocator_, grass_); });
    guarded("destroy_plant_gpu", [&]{ destroy_plant_gpu(); });
    guarded("destroy_scatter", [&]{ destroy_scatter(allocator_, scatter_); });
    guarded("stop_terrain_shadow_worker", [&]{ stop_terrain_shadow_worker(); });
    guarded("destroy_terrain_shadow_texture", [&]{ destroy_terrain_shadow_texture(); });
    guarded("destroy_sun_shadow_resources", [&]{ destroy_sun_shadow_resources(); });
    guarded("destroy_pipeline_cache", [&]{ destroy_pipeline_cache(); });
    // Everything of ours is saved and released. What is left - tearing down the
    // device, the instance and the upscaler SDK's plug-ins - took 7 seconds and
    // gives nothing back that the operating system does not reclaim at once:
    // leave now. (VIBE_FULL_SHUTDOWN=1 runs the complete teardown, for leak checks.)
    if (!std::getenv("VIBE_FULL_SHUTDOWN")) {
        log::info("Vibe shutdown (fast exit)");
        log::close();
        std::fflush(nullptr);
        TerminateProcess(GetCurrentProcess(), 0);
    }
    guarded("destroy_readback_buffer", [&]{ destroy_readback_buffer(); });

    guarded("destroy_frames", [&]{
        for (auto& f : frames_) {
            if (f.render_fence) vkDestroyFence(device_, f.render_fence, nullptr);
            if (f.swapchain_semaphore) vkDestroySemaphore(device_, f.swapchain_semaphore, nullptr);
            if (f.tlas_build_done) vkDestroySemaphore(device_, f.tlas_build_done, nullptr);
            if (f.command_pool) vkDestroyCommandPool(device_, f.command_pool, nullptr);
            if (f.compute_pool) vkDestroyCommandPool(device_, f.compute_pool, nullptr);
            f = {};
        }
    });

    guarded("destroy_depth_image", [&]{ destroy_depth_image(); });
    guarded("depth_copy_sampler", [&]{
        if (depth_copy_sampler_) vkDestroySampler(device_, depth_copy_sampler_, nullptr);
        depth_copy_sampler_ = VK_NULL_HANDLE;
    });
    guarded("destroy_swapchain",   [&]{ destroy_swapchain(); });

    guarded("vmaDestroyAllocator", [&]{
        if (allocator_) vmaDestroyAllocator(allocator_);
        allocator_ = nullptr;
    });

    guarded("sl_shutdown", [&]{ sl_shutdown(); });   // Streamline: before the device
    guarded("destroy_surface_device_instance", [&]{
        if (surface_) vkDestroySurfaceKHR(instance_, surface_, nullptr);
        surface_ = VK_NULL_HANDLE;
        if (device_) vkDestroyDevice(device_, nullptr);
        device_ = VK_NULL_HANDLE;
        if (debug_messenger_) vkb::destroy_debug_utils_messenger(instance_, debug_messenger_);
        debug_messenger_ = VK_NULL_HANDLE;
        if (instance_) vkDestroyInstance(instance_, nullptr);
        instance_ = VK_NULL_HANDLE;
    });

    guarded("window.reset",   [&]{ window_.reset(); });
    initialized_ = false;
}

} // namespace vibe
