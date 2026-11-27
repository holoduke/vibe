// Loading-screen presentation: decodes the loader background and pushes
// progress frames to the swapchain during the (multi-second) asset load,
// before the main render loop starts. Split out of vk_engine.cpp.

#include "engine/vk_engine.h"
#include "engine/vk_engine/internal.h"
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
#include "engine/log.h"
#include "engine/texture.h"
#include "engine/window.h"
#include "engine/vk_initializers.h"
#include "engine/vk_pipelines.h"

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
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <typeinfo>

#ifndef VIBE_SHADER_DIR
#define VIBE_SHADER_DIR "shaders"
#endif

namespace vibe {

void VulkanEngine::ensure_loader_bg_loaded_() {
    if (loader_bg_tried_) return;
    loader_bg_tried_ = true;
    // Probe a few roots so the engine finds the asset whether the cwd
    // is the project root or `build/`.
    const char* roots[] = {
        "assets/loader/loading_bg.jpg",
        "../assets/loader/loading_bg.jpg",
        "../../assets/loader/loading_bg.jpg",
    };
    std::string path;
    for (const char* r : roots) {
        std::ifstream f(r);
        if (f.good()) { path = r; break; }
    }
    if (path.empty()) {
        log::warnf("[loader] background image missing on all probe paths");
        return;
    }
    Texture2D r = upload_texture_from_file(
        device_, allocator_, graphics_queue_, graphics_queue_family_,
        path, VK_FORMAT_R8G8B8A8_SRGB);
    if (!r.ok) {
        log::warnf("[loader] failed to load %s", path.c_str());
        return;
    }
    // Free the sampler view -- we only need the raw image for
    // vkCmdBlitImage. Saves a tiny bit of memory and keeps the
    // descriptor pool budget intact.
    if (r.view) vkDestroyImageView(device_, r.view, nullptr);
    // Transition once from SHADER_READ_ONLY_OPTIMAL (where upload
    // leaves it) -> TRANSFER_SRC_OPTIMAL so the per-frame blit's
    // src layout matches. The image lives in this layout forever
    // since the loader is the only consumer.
    //
    // Don't assign to loader_bg_image_ until the transition succeeds.
    // present_loader_frame() picks up the asset based on that handle
    // and assumes the layout is TRANSFER_SRC_OPTIMAL; if pool create
    // / cmd alloc / queue submit fail, we want present_loader_frame to
    // fall through to the cleared-navy + vignette fallback instead of
    // issuing a vkCmdBlitImage with the wrong source layout.
    VkCommandPool cp = VK_NULL_HANDLE;
    VkCommandPoolCreateInfo cpi{
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
        .queueFamilyIndex = graphics_queue_family_,
    };
    if (vkCreateCommandPool(device_, &cpi, nullptr, &cp) != VK_SUCCESS) {
        log::warnf("[loader] cmd pool create failed; dropping bg image");
        vmaDestroyImage(allocator_, r.image, r.alloc);
        return;
    }
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkCommandBufferAllocateInfo cai{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .pNext = nullptr,
        .commandPool = cp,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    if (vkAllocateCommandBuffers(device_, &cai, &cmd) != VK_SUCCESS) {
        log::warnf("[loader] cmd alloc failed; dropping bg image");
        vkDestroyCommandPool(device_, cp, nullptr);
        vmaDestroyImage(allocator_, r.image, r.alloc);
        return;
    }
    VkCommandBufferBeginInfo bi{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        .pInheritanceInfo = nullptr,
    };
    vkBeginCommandBuffer(cmd, &bi);
    vkinit::transition_image(cmd, r.image,
                              VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                              VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    vkEndCommandBuffer(cmd);
    VkSubmitInfo si{
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .pNext = nullptr,
        .waitSemaphoreCount = 0, .pWaitSemaphores = nullptr,
        .pWaitDstStageMask = nullptr,
        .commandBufferCount = 1, .pCommandBuffers = &cmd,
        .signalSemaphoreCount = 0, .pSignalSemaphores = nullptr,
    };
    VkResult sub = vkQueueSubmit(graphics_queue_, 1, &si, VK_NULL_HANDLE);
    if (sub == VK_SUCCESS) sub = vkQueueWaitIdle(graphics_queue_);
    vkDestroyCommandPool(device_, cp, nullptr);
    if (sub != VK_SUCCESS) {
        log::warnf("[loader] layout transition submit failed (%d); dropping bg image", int(sub));
        vmaDestroyImage(allocator_, r.image, r.alloc);
        return;
    }
    // Transition succeeded -- publish to engine state. destroy_loader_bg_()
    // at shutdown owns the image from this point.
    loader_bg_image_  = r.image;
    loader_bg_alloc_  = r.alloc;
    loader_bg_extent_ = { static_cast<uint32_t>(r.width),
                          static_cast<uint32_t>(r.height) };
    log::infof("[loader] background loaded (%ux%u) from %s",
                loader_bg_extent_.width, loader_bg_extent_.height, path.c_str());
}

void VulkanEngine::destroy_loader_bg_() {
    if (loader_bg_image_) {
        vmaDestroyImage(allocator_, loader_bg_image_, loader_bg_alloc_);
        loader_bg_image_ = VK_NULL_HANDLE;
        loader_bg_alloc_ = nullptr;
    }
}

void VulkanEngine::present_loader_frame(const char* label, float progress) {
    if (!device_ || !swapchain_ || swapchain_images_.empty()) return;

    if (window_) {
        std::string title = "Vibe — ";
        title += label ? label : "Loading…";
        glfwSetWindowTitle(window_->handle(), title.c_str());
        // Pump GLFW so the OS doesn't mark the window as "not responding"
        // during the long init steps that sit between loader frames.
        window_->poll_events();
    }

    auto& frame = current_frame();

    VIBE_VK_CHECK(vkWaitForFences(device_, 1, &frame.render_fence,
                                    VK_TRUE, UINT64_MAX),
                   "loader: vkWaitForFences");

    uint32_t img_idx = 0;
    // Routes through FSR3 SwapChain proxy fns when active (session 4),
    // plain Vulkan otherwise.
    VkResult acq = acquire_next_image(UINT64_MAX, frame.swapchain_semaphore,
                                       VK_NULL_HANDLE, &img_idx);
    if (acq == VK_ERROR_OUT_OF_DATE_KHR) { resize_requested_ = true; return; }
    if (acq != VK_SUCCESS && acq != VK_SUBOPTIMAL_KHR) {
        VIBE_VK_CHECK(acq, "loader: vkAcquireNextImageKHR");
    }

    VIBE_VK_CHECK(vkResetFences(device_, 1, &frame.render_fence),
                   "loader: vkResetFences");
    VIBE_VK_CHECK(vkResetCommandBuffer(frame.command_buffer, 0),
                   "loader: vkResetCommandBuffer");

    auto bi = vkinit::command_buffer_begin_info(
        VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
    VIBE_VK_CHECK(vkBeginCommandBuffer(frame.command_buffer, &bi),
                   "loader: vkBeginCommandBuffer");

    // Try to load the background image on the first call. Cached
    // across all subsequent loader frames.
    ensure_loader_bg_loaded_();

    // Blit path: if we have a background image, copy it to the
    // swapchain (stretched to fit) BEFORE the render pass begins.
    // vkCmdBlitImage runs without any graphics pipeline -- perfect
    // for the loader screen which fires before pipelines exist.
    // Falls back to the cleared navy + radial vignette if the asset
    // is missing.
    bool used_bg = (loader_bg_image_ != VK_NULL_HANDLE);
    if (used_bg) {
        vkinit::transition_image_src_stage(frame.command_buffer,
                                  swapchain_images_[img_idx],
                                  VK_IMAGE_LAYOUT_UNDEFINED,
                                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                  VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);   // chains with the acquire wait
        VkImageBlit blit{};
        blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        blit.srcSubresource.layerCount = 1;
        blit.srcOffsets[1] = {
            static_cast<int32_t>(loader_bg_extent_.width),
            static_cast<int32_t>(loader_bg_extent_.height), 1 };
        blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        blit.dstSubresource.layerCount = 1;
        blit.dstOffsets[1] = {
            static_cast<int32_t>(swapchain_extent_.width),
            static_cast<int32_t>(swapchain_extent_.height), 1 };
        vkCmdBlitImage(frame.command_buffer,
                        loader_bg_image_,
                        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                        swapchain_images_[img_idx],
                        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                        1, &blit, VK_FILTER_LINEAR);
        vkinit::transition_image(frame.command_buffer,
                                  swapchain_images_[img_idx],
                                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                  VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    } else {
        vkinit::transition_image_src_stage(frame.command_buffer,
                                  swapchain_images_[img_idx],
                                  VK_IMAGE_LAYOUT_UNDEFINED,
                                  VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                  VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);   // chains with the acquire wait
    }

    VkClearValue clear_bg{};
    clear_bg.color = { { 0.020f, 0.025f, 0.040f, 1.0f } };  // deep navy
    // When a BG image was just blit'd into the swapchain, we MUST use
    // LOAD_OP_LOAD on the colour attachment, otherwise the rendering
    // pass starts with LOAD_OP_CLEAR which wipes the blit'd image back
    // to navy before the overlays draw -- "loader image is black".
    auto sw_color = vkinit::color_attachment_info(
        swapchain_views_[img_idx],
        used_bg ? nullptr : &clear_bg,
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    VkRenderingInfo ri{
        .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
        .pNext = nullptr, .flags = 0,
        .renderArea = { {0, 0}, swapchain_extent_ },
        .layerCount = 1, .viewMask = 0,
        .colorAttachmentCount = 1, .pColorAttachments = &sw_color,
        .pDepthAttachment = nullptr, .pStencilAttachment = nullptr,
    };
    vkCmdBeginRendering(frame.command_buffer, &ri);

    // Everything is drawn with vkCmdClearAttachments rectangles — we
    // can't use ImGui/pipelines here (this runs DURING init, before
    // any pipeline exists). The pixel-block font below lets us show
    // a real percentage and title without a font atlas.
    auto clear_rect = [&](int x, int y, int w, int h,
                          float r, float g, float b) {
        if (w <= 0 || h <= 0) return;
        VkClearAttachment ca{};
        ca.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        ca.colorAttachment = 0;
        ca.clearValue.color = { { r, g, b, 1.0f } };
        VkClearRect cr{};
        cr.rect.offset = { x, y };
        cr.rect.extent = { static_cast<uint32_t>(w),
                            static_cast<uint32_t>(h) };
        cr.baseArrayLayer = 0;
        cr.layerCount = 1;
        vkCmdClearAttachments(frame.command_buffer, 1, &ca, 1, &cr);
    };

    int W = static_cast<int>(swapchain_extent_.width);
    int H = static_cast<int>(swapchain_extent_.height);

    // Background gradient: dim radial vignette from centre. Skipped
    // entirely when a loader BG image was blit'd into the swapchain
    // above -- the rects would just cover the image otherwise. The
    // text + progress overlay below still draws on top so the user
    // can see the loading state on either background.
    if (!used_bg) {
        for (int b = 0; b < 8; ++b) {
            float t = float(b) / 8.0f;
            float d = 0.020f - 0.010f * t;
            int pad = static_cast<int>(t * std::max(W, H) * 0.5f);
            clear_rect(pad, pad, W - 2 * pad, H - 2 * pad, d, d * 1.1f, d * 1.4f);
        }
    }

    // ---- Pixel font (5×7), drawn as rectangle blocks at any scale ----
    // Bit 6 of each column byte is the TOP row, bit 0 is the BOTTOM row.
    // Only the chars actually rendered are defined: digits, %, space,
    // and "QUAKELIKEKLAOIDNG" letters.
    struct Glyph { uint8_t col[5]; };
    auto G = [](uint8_t a, uint8_t b, uint8_t c, uint8_t d, uint8_t e) {
        Glyph g{}; g.col[0]=a; g.col[1]=b; g.col[2]=c; g.col[3]=d; g.col[4]=e;
        return g;
    };
    auto glyph_for = [&](char ch) -> Glyph {
        switch (ch) {
            case '0': return G(0x3E,0x51,0x49,0x45,0x3E);
            case '1': return G(0x00,0x42,0x7F,0x40,0x00);
            case '2': return G(0x42,0x61,0x51,0x49,0x46);
            case '3': return G(0x21,0x41,0x45,0x4B,0x31);
            case '4': return G(0x18,0x14,0x12,0x7F,0x10);
            case '5': return G(0x27,0x45,0x45,0x45,0x39);
            case '6': return G(0x3C,0x4A,0x49,0x49,0x30);
            case '7': return G(0x01,0x71,0x09,0x05,0x03);
            case '8': return G(0x36,0x49,0x49,0x49,0x36);
            case '9': return G(0x06,0x49,0x49,0x29,0x1E);
            case '%': return G(0x23,0x13,0x08,0x64,0x62);
            case 'Q': return G(0x3E,0x41,0x51,0x21,0x5E);
            case 'U': return G(0x3F,0x40,0x40,0x40,0x3F);
            case 'A': return G(0x7E,0x09,0x09,0x09,0x7E);
            case 'K': return G(0x7F,0x08,0x14,0x22,0x41);
            case 'E': return G(0x7F,0x49,0x49,0x49,0x41);
            case 'L': return G(0x7F,0x40,0x40,0x40,0x40);
            case 'I': return G(0x00,0x41,0x7F,0x41,0x00);
            case 'O': return G(0x3E,0x41,0x41,0x41,0x3E);
            case 'D': return G(0x7F,0x41,0x41,0x41,0x3E);
            case 'N': return G(0x7F,0x02,0x0C,0x30,0x7F);
            case 'G': return G(0x3E,0x41,0x49,0x49,0x7A);
            case 'B': return G(0x7F,0x49,0x49,0x49,0x36);
            case 'C': return G(0x3E,0x41,0x41,0x41,0x22);
            case 'F': return G(0x7F,0x09,0x09,0x09,0x01);
            case 'H': return G(0x7F,0x08,0x08,0x08,0x7F);
            case 'J': return G(0x20,0x40,0x41,0x3F,0x01);
            case 'M': return G(0x7F,0x02,0x0C,0x02,0x7F);
            case 'P': return G(0x7F,0x09,0x09,0x09,0x06);
            case 'R': return G(0x7F,0x09,0x19,0x29,0x46);
            case 'S': return G(0x46,0x49,0x49,0x49,0x31);
            case 'T': return G(0x01,0x01,0x7F,0x01,0x01);
            case 'V': return G(0x1F,0x20,0x40,0x20,0x1F);
            case 'W': return G(0x3F,0x40,0x38,0x40,0x3F);
            case 'X': return G(0x63,0x14,0x08,0x14,0x63);
            case 'Y': return G(0x07,0x08,0x70,0x08,0x07);
            case 'Z': return G(0x61,0x51,0x49,0x45,0x43);
            case '.': return G(0x00,0x60,0x60,0x00,0x00);
            case '-': return G(0x08,0x08,0x08,0x08,0x08);
            case '/': return G(0x20,0x10,0x08,0x04,0x02);
            case ':': return G(0x00,0x36,0x36,0x00,0x00);
            case '+': return G(0x08,0x08,0x3E,0x08,0x08);
            case ' ': default: return G(0,0,0,0,0);
        }
    };
    auto draw_text = [&](int x, int y, int px, const char* s,
                         float r, float g, float b) {
        for (const char* p_s = s; *p_s; ++p_s) {
            Glyph g_ = glyph_for(*p_s);
            for (int c = 0; c < 5; ++c) {
                uint8_t col = g_.col[c];
                for (int r2 = 0; r2 < 7; ++r2) {
                    if (col & (1 << r2)) {
                        clear_rect(x + c * px, y + r2 * px, px, px, r, g, b);
                    }
                }
            }
            x += 6 * px;        // 5 px + 1 px spacing
        }
    };

    // ---- Title: the key art carries the logo; without it, big letters ----
    if (!used_bg) {
        const char* title = "VIBE";
        int px = std::max(6, W / 90);
        int title_w = static_cast<int>(strlen(title)) * 6 * px - px;
        int tx = (W - title_w) / 2, ty = H / 3;
        draw_text(tx + px / 2, ty + px / 2, px, title, 0.0f, 0.0f, 0.0f);
        draw_text(tx, ty, px, title, 0.96f, 0.96f, 0.98f);
        clear_rect(tx, ty + 9 * px, title_w, std::max(2, px / 3), 1.0f, 0.55f, 0.15f);
    }

    // ---- Bottom strip: what is loading (left), percentage (right), a
    //      slim bar across the screen ----
    float p = progress;
    if (p < 0.0f) p = 0.0f;
    if (p > 1.0f) p = 1.0f;
    const int margin = std::max(40, W / 18);
    const int bar_w = W - 2 * margin;
    const int bar_h = std::max(4, H / 180);
    const int bar_y = H - std::max(48, H / 11);
    const int px = std::max(2, W / 640);
    {
        // Stage name, in capitals.
        char lbl[48];
        size_t n = 0;
        for (const char* c = (label ? label : "Loading"); *c && n + 1 < sizeof(lbl); ++c) {
            unsigned char ch = static_cast<unsigned char>(*c);
            if (ch >= 'a' && ch <= 'z') ch = static_cast<unsigned char>(ch - 32);
            if (ch < 128) lbl[n++] = static_cast<char>(ch);
        }
        lbl[n] = 0;
        const int ty = bar_y - 7 * px - std::max(10, 3 * px);
        draw_text(margin + 1, ty + 1, px, lbl, 0.0f, 0.0f, 0.0f);
        draw_text(margin,     ty,     px, lbl, 0.90f, 0.90f, 0.92f);
        char buf[8];
        std::snprintf(buf, sizeof(buf), "%d%%", static_cast<int>(p * 100.0f + 0.5f));
        const int tw = static_cast<int>(strlen(buf)) * 6 * px - px;
        draw_text(margin + bar_w - tw + 1, ty + 1, px, buf, 0.0f, 0.0f, 0.0f);
        draw_text(margin + bar_w - tw,     ty,     px, buf, 1.0f, 0.62f, 0.22f);
    }
    // Track, fill (warm accent, as the logo's glow), bright leading edge.
    clear_rect(margin - 1, bar_y - 1, bar_w + 2, bar_h + 2, 0.0f, 0.0f, 0.0f);
    clear_rect(margin, bar_y, bar_w, bar_h, 0.16f, 0.17f, 0.19f);
    const int fill_w = static_cast<int>(static_cast<float>(bar_w) * p + 0.5f);
    clear_rect(margin, bar_y, fill_w, bar_h, 1.0f, 0.55f, 0.15f);
    {
        const int edge_w = std::max(3, bar_h);
        if (fill_w >= edge_w)
            clear_rect(margin + fill_w - edge_w, bar_y - 1, edge_w, bar_h + 2, 1.0f, 0.90f, 0.70f);
    }

    vkCmdEndRendering(frame.command_buffer);

    vkinit::transition_image(frame.command_buffer,
                              swapchain_images_[img_idx],
                              VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                              VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);

    VIBE_VK_CHECK(vkEndCommandBuffer(frame.command_buffer),
                   "loader: vkEndCommandBuffer");

    auto wait = vkinit::semaphore_submit_info(
        VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,   // blit (transfer) also writes the image
        frame.swapchain_semaphore);
    auto signal = vkinit::semaphore_submit_info(
        VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT,
        render_semaphores_[img_idx]);
    auto cmd_info = vkinit::command_buffer_submit_info(frame.command_buffer);
    VkSubmitInfo2 submit{
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
        .pNext = nullptr, .flags = 0,
        .waitSemaphoreInfoCount = 1, .pWaitSemaphoreInfos = &wait,
        .commandBufferInfoCount = 1, .pCommandBufferInfos = &cmd_info,
        .signalSemaphoreInfoCount = 1, .pSignalSemaphoreInfos = &signal,
    };
    VIBE_VK_CHECK(vkQueueSubmit2(graphics_queue_, 1, &submit,
                                   frame.render_fence),
                   "loader: vkQueueSubmit2");

    // Loader-frame present — same proxy-aware path as the gameplay one.
    VkResult pres = queue_present_image(img_idx, render_semaphores_[img_idx]);
    if (pres == VK_ERROR_OUT_OF_DATE_KHR || pres == VK_SUBOPTIMAL_KHR) {
        resize_requested_ = true;
    } else if (pres != VK_SUCCESS) {
        VIBE_VK_CHECK(pres, "loader: vkQueuePresentKHR");
    }

    ++frame_number_;
}

// ---------------------------------------------------------------------
// The notice while the large pipelines compile in the background (vk_pipelines.cpp queue).
// Shown once a batch has run for over a second (builds that come from the pipeline cache finish
// long before), kept until the whole batch is done, then "ready" and a fade. Everything it says
// comes from vkpipe::compile_status(): the queue decides what is listed.
// VIBE_TEST_COMPILE_NOTICE=1 shows it with made-up numbers (a picture without a cold compile).
// ---------------------------------------------------------------------
namespace {
std::string compile_time_words(double s) {
    char b[64];
    if (s < 4.0) return "almost done";
    if (s < 60.0) { std::snprintf(b, sizeof(b), "about %d s left", int(std::ceil(s / 5.0) * 5.0)); return b; }
    const int t = int(std::ceil(s / 10.0) * 10.0);
    if (t % 60 == 0) std::snprintf(b, sizeof(b), "about %d min left", t / 60);
    else std::snprintf(b, sizeof(b), "about %d min %02d s left", t / 60, t % 60);
    return b;
}
}  // namespace

void draw_compile_notice() {
    static const bool fake = std::getenv("VIBE_TEST_COMPILE_NOTICE") != nullptr;
    vkpipe::CompileStatus st = vkpipe::compile_status();
    if (fake) {
        st.total = 5; st.done = 1; st.running = 1; st.active = true;
        st.current = "the landscape"; st.current_elapsed = 41.0; st.current_estimate = 124.0;
        st.remaining = 97.0; st.fraction = 0.38; st.batch_elapsed = 55.0;
    }
    static bool shown = false;
    static double done_at = -1.0;
    const double now = ImGui::GetTime();
    if (st.active) {
        done_at = -1.0;
        if (!shown && (st.batch_elapsed > 1.0 || st.current_elapsed > 1.0)) {
            shown = true;
            log::infof("[pipeline] notice: preparing shaders, %d of %d, %s", st.done + 1, st.total, compile_time_words(st.remaining).c_str());
        }
    } else if (shown && done_at < 0.0) {
        done_at = now;
        log::info("[pipeline] notice: shaders ready");
    }
    if (!shown) return;
    float alpha = 1.0f;
    if (!st.active) {
        const double t = now - done_at;
        if (t > 2.5) { shown = false; done_at = -1.0; return; }
        alpha = t < 1.5 ? 1.0f : float(1.0 - (t - 1.5));       // "ready" for 1.5 s, then a 1 s fade
    }
    ImDrawList* dl = ImGui::GetForegroundDrawList();
    const ImVec2 disp = ImGui::GetIO().DisplaySize;
    const float fs = ImGui::GetFontSize();
    auto col = [alpha](int r, int g, int b, float a) { return IM_COL32(r, g, b, int(255.0f * a * alpha)); };
    const char* title = st.active ? "Preparing shaders for your graphics card" : "Shaders ready";
    const char* sub = "This happens once after an update or a driver change. You can play meanwhile;\nthe landscape looks simpler until it is done.";
    char info[256];
    if (st.active) {
        const int n = std::min(st.total, st.done + std::max(st.running, 1));
        std::snprintf(info, sizeof(info), "%d of %d:  %s  -  %s", n, st.total,
                      st.current.empty() ? "waiting for its turn" : st.current.c_str(), compile_time_words(st.remaining).c_str());
    } else {
        std::snprintf(info, sizeof(info), "All %d done in %d s - full detail from now on", st.total, int(st.batch_elapsed + 0.5));
    }
    const ImVec2 ts_title = ImGui::CalcTextSize(title), ts_sub = ImGui::CalcTextSize(sub), ts_info = ImGui::CalcTextSize(info);
    const float pad = fs * 0.9f;
    const float w = std::max({ ts_title.x, ts_sub.x, ts_info.x, fs * 26.0f }) + pad * 2.0f;
    const float bar_h = std::max(4.0f, fs * 0.35f);
    const float h = pad + ts_title.y + fs * 0.35f + ts_sub.y + fs * 0.6f + bar_h + fs * 0.5f + ts_info.y + pad;
    const ImVec2 p0((disp.x - w) * 0.5f, disp.y * 0.05f);
    dl->AddRectFilled(p0, ImVec2(p0.x + w, p0.y + h), col(12, 14, 18, 0.78f), fs * 0.5f);
    dl->AddRect(p0, ImVec2(p0.x + w, p0.y + h), col(255, 255, 255, 0.10f), fs * 0.5f);
    float y = p0.y + pad;
    dl->AddText(ImVec2(p0.x + pad, y), col(255, 255, 255, 0.95f), title);
    y += ts_title.y + fs * 0.35f;
    dl->AddText(ImVec2(p0.x + pad, y), col(200, 205, 215, 0.80f), sub);
    y += ts_sub.y + fs * 0.6f;
    // The bar: by estimated time (done builds + the running one's share).
    const float bx0 = p0.x + pad, bx1 = p0.x + w - pad;
    const float f = float(std::clamp(st.fraction, 0.0, 1.0));
    dl->AddRectFilled(ImVec2(bx0, y), ImVec2(bx1, y + bar_h), col(255, 255, 255, 0.12f), bar_h * 0.5f);
    if (f > 0.0f) dl->AddRectFilled(ImVec2(bx0, y), ImVec2(bx0 + (bx1 - bx0) * f, y + bar_h),
                                    st.active ? col(120, 190, 255, 0.90f) : col(130, 220, 150, 0.90f), bar_h * 0.5f);
    if (st.active) {
        // (a soft light running along the filled part: it is working, even while a fraction barely moves)
        const float ph = float(std::fmod(now * 0.6, 1.0));
        const float cx = bx0 + (bx1 - bx0) * f * ph;
        dl->AddRectFilled(ImVec2(std::max(bx0, cx - fs), y), ImVec2(std::min(bx0 + (bx1 - bx0) * f, cx + fs), y + bar_h),
                          col(255, 255, 255, 0.25f), bar_h * 0.5f);
    }
    y += bar_h + fs * 0.5f;
    dl->AddText(ImVec2(p0.x + pad, y), col(225, 230, 240, 0.90f), info);
}

} // namespace vibe
