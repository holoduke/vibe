// ui_setup.cpp - Dear ImGui setup / teardown, fonts, style
// (split out of vk_ui.cpp; see docs/ARCHITECTURE.md for the map.)
// VulkanEngine UI — ImGui setup, pause menu, HUD overlay, crosshair.
// Split out of vk_engine.cpp because UI is independent of render-state plumbing
// and grew large enough to crowd the main engine file.

#include "engine/vk_engine.h"

#include "engine/log.h"
#include "engine/vk_initializers.h"   // qlike::vk_check
#include "engine/window.h"

#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_vulkan.h>

#include <algorithm>
#include <cstddef>
#include <cstdio>

namespace qlike {


void VulkanEngine::init_imgui() {
    VkDescriptorPoolSize sizes[] = {
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000 },
    };
    VkDescriptorPoolCreateInfo pci{};
    pci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pci.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    pci.maxSets = 1000;
    pci.poolSizeCount = 1;
    pci.pPoolSizes = sizes;
    vk_check(vkCreateDescriptorPool(device_, &pci, nullptr, &imgui_pool_),
             "imgui pool");

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    // Keyboard nav DISABLED. With nav on, ImGui can auto-focus a
    // pause-menu button when the menu opens, then propagate further
    // key events to it — pressing ESC over the pause menu was
    // landing on the Quit button and exiting the game. Mouse-only
    // for the menu is fine here; no controller use.
    io.ConfigFlags &= ~ImGuiConfigFlags_NavEnableKeyboard;
    // Fonts, from Windows itself (nothing shipped): Segoe UI for the menus,
    // Bahnschrift (DIN) for the HUD - small for labels, large for the numbers.
    // Missing files fall back to ImGui's built-in font.
    {
        auto have = [](const char* p) { FILE* fp = std::fopen(p, "rb"); if (fp) std::fclose(fp); return fp != nullptr; };
        const char* kUi = "C:\\Windows\\Fonts\\segoeui.ttf";
        const char* kHud = "C:\\Windows\\Fonts\\bahnschrift.ttf";
        ImFontConfig fc;
        fc.OversampleH = 3; fc.OversampleV = 2;
        if (have(kUi)) io.Fonts->AddFontFromFileTTF(kUi, 18.0f, &fc); else io.Fonts->AddFontDefault();
        if (have(kHud)) {
            ImFontConfig hc;
            hc.OversampleH = 3; hc.OversampleV = 2;
            hc.RasterizerMultiply = 1.15f;
            font_hud_ = io.Fonts->AddFontFromFileTTF(kHud, 26.0f, &hc);
            ImFontConfig bc;
            bc.OversampleH = 2; bc.OversampleV = 1;
            bc.RasterizerMultiply = 1.2f;
            font_hud_big_ = io.Fonts->AddFontFromFileTTF(kHud, 84.0f, &bc);
        }
    }
    // Pin imgui.ini next to vibe_settings.cfg so menu collapse /
    // expand / position state persists across restarts regardless
    // of the launch cwd. Default behaviour writes "imgui.ini" in
    // cwd, which means launching from build/ vs the repo root used
    // two different files and the saved state didn't follow you.
    io.IniFilename = "vibe_imgui.ini";
    ImGui::StyleColorsDark();

    // ===== Synthwave style ============================================
    // Neon magenta + cyan on a deep indigo background — pause menu /
    // HUD reads as a 1980s CRT title screen. Just a colour + spacing
    // override; no custom shaders, no extra geometry.
    {
        ImGuiStyle& s = ImGui::GetStyle();
        s.WindowRounding   = 6.0f;
        s.FrameRounding    = 4.0f;
        s.GrabRounding     = 4.0f;
        s.TabRounding      = 4.0f;
        s.PopupRounding    = 4.0f;
        s.ScrollbarRounding= 6.0f;
        s.WindowPadding    = ImVec2(14.0f, 12.0f);
        s.FramePadding     = ImVec2(8.0f, 5.0f);
        s.ItemSpacing      = ImVec2(10.0f, 8.0f);
        s.WindowBorderSize = 1.5f;
        s.FrameBorderSize  = 0.0f;
        s.TabBorderSize    = 0.0f;
        s.IndentSpacing    = 16.0f;

        const ImVec4 magenta      (0.95f, 0.20f, 0.85f, 1.00f);
        const ImVec4 magentaSoft  (0.65f, 0.15f, 0.60f, 1.00f);
        const ImVec4 magentaDeep  (0.20f, 0.05f, 0.20f, 1.00f);
        const ImVec4 cyan         (0.20f, 0.95f, 0.95f, 1.00f);
        const ImVec4 cyanSoft     (0.15f, 0.55f, 0.65f, 1.00f);
        const ImVec4 indigoBg     (0.05f, 0.03f, 0.10f, 0.96f);
        const ImVec4 indigoFrame  (0.10f, 0.07f, 0.18f, 1.00f);
        const ImVec4 indigoFrameHi(0.18f, 0.12f, 0.32f, 1.00f);
        const ImVec4 textBright   (0.95f, 0.92f, 1.00f, 1.00f);
        const ImVec4 textDim      (0.70f, 0.78f, 0.95f, 1.00f);

        ImVec4* c = s.Colors;
        c[ImGuiCol_Text]                 = textBright;
        c[ImGuiCol_TextDisabled]         = textDim;
        c[ImGuiCol_WindowBg]             = indigoBg;
        c[ImGuiCol_PopupBg]              = indigoBg;
        c[ImGuiCol_Border]               = magenta;
        c[ImGuiCol_BorderShadow]         = ImVec4(0,0,0,0);
        c[ImGuiCol_FrameBg]              = indigoFrame;
        c[ImGuiCol_FrameBgHovered]       = indigoFrameHi;
        c[ImGuiCol_FrameBgActive]        = ImVec4(0.30f, 0.18f, 0.50f, 1.0f);
        c[ImGuiCol_TitleBg]              = magentaDeep;
        c[ImGuiCol_TitleBgActive]        = magentaSoft;
        c[ImGuiCol_TitleBgCollapsed]     = magentaDeep;
        c[ImGuiCol_MenuBarBg]            = magentaDeep;
        c[ImGuiCol_ScrollbarBg]          = ImVec4(0.05f, 0.03f, 0.12f, 0.6f);
        c[ImGuiCol_ScrollbarGrab]        = magentaSoft;
        c[ImGuiCol_ScrollbarGrabHovered] = magenta;
        c[ImGuiCol_ScrollbarGrabActive]  = cyan;
        c[ImGuiCol_CheckMark]            = cyan;
        c[ImGuiCol_SliderGrab]           = cyan;
        c[ImGuiCol_SliderGrabActive]     = magenta;
        c[ImGuiCol_Button]               = ImVec4(0.30f, 0.10f, 0.40f, 1.0f);
        c[ImGuiCol_ButtonHovered]        = magenta;
        c[ImGuiCol_ButtonActive]         = cyan;
        c[ImGuiCol_Header]               = magentaSoft;
        c[ImGuiCol_HeaderHovered]        = magenta;
        c[ImGuiCol_HeaderActive]         = cyan;
        c[ImGuiCol_Separator]            = magentaSoft;
        c[ImGuiCol_SeparatorHovered]     = cyan;
        c[ImGuiCol_SeparatorActive]      = cyan;
        c[ImGuiCol_ResizeGrip]           = magentaSoft;
        c[ImGuiCol_ResizeGripHovered]    = magenta;
        c[ImGuiCol_ResizeGripActive]     = cyan;
        c[ImGuiCol_Tab]                  = ImVec4(0.18f, 0.08f, 0.30f, 1.0f);
        c[ImGuiCol_TabHovered]           = magenta;
        c[ImGuiCol_TabActive]            = ImVec4(0.55f, 0.12f, 0.55f, 1.0f);
        c[ImGuiCol_TabUnfocused]         = ImVec4(0.10f, 0.05f, 0.18f, 1.0f);
        c[ImGuiCol_TabUnfocusedActive]   = magentaSoft;
        c[ImGuiCol_PlotLines]            = cyan;
        c[ImGuiCol_PlotLinesHovered]     = magenta;
        c[ImGuiCol_PlotHistogram]        = magenta;
        c[ImGuiCol_PlotHistogramHovered] = cyan;
        c[ImGuiCol_DragDropTarget]       = cyan;
        c[ImGuiCol_NavHighlight]         = cyan;
    }
    // ===================================================================

    ImGui_ImplGlfw_InitForVulkan(window_->handle(), true);

    VkFormat color_fmt = swapchain_format_;
    VkPipelineRenderingCreateInfo prci{};
    prci.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    prci.colorAttachmentCount = 1;
    prci.pColorAttachmentFormats = &color_fmt;
    prci.depthAttachmentFormat = depth_format_;

    ImGui_ImplVulkan_InitInfo ii{};
    ii.Instance = instance_;
    ii.PhysicalDevice = physical_device_;
    ii.Device = device_;
    ii.QueueFamily = graphics_queue_family_;
    ii.Queue = graphics_queue_;
    ii.DescriptorPool = imgui_pool_;
    ii.MinImageCount = 2;
    ii.ImageCount = static_cast<uint32_t>(swapchain_images_.size());
    ii.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    ii.UseDynamicRendering = true;
    ii.PipelineRenderingCreateInfo = prci;
    ImGui_ImplVulkan_Init(&ii);

    // HUD weapon icons (generated artwork; see ui_gamehud.cpp).
    {
        hud_icons_tex_ = upload_texture_from_file(device_, allocator_, graphics_queue_, graphics_queue_family_,
                                                  "assets/hud/weapon_icons.png", VK_FORMAT_R8G8B8A8_UNORM);
        if (hud_icons_tex_.ok) {
            VkSamplerCreateInfo sci{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
            sci.magFilter = VK_FILTER_LINEAR; sci.minFilter = VK_FILTER_LINEAR;
            sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
            sci.addressModeU = sci.addressModeV = sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            sci.maxLod = VK_LOD_CLAMP_NONE;
            if (vkCreateSampler(device_, &sci, nullptr, &hud_icons_sampler_) == VK_SUCCESS)
                hud_icons_id_ = uint64_t(ImGui_ImplVulkan_AddTexture(hud_icons_sampler_, hud_icons_tex_.view,
                                                                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL));
        } else {
            log::warn("[hud] assets/hud/weapon_icons.png not found: the weapon bay shows names only");
        }
    }

    imgui_initialized_ = true;
    log::info("ImGui initialized (Vulkan, dynamic rendering)");
}

void VulkanEngine::destroy_imgui() {
    if (!imgui_initialized_) return;
    // Force-flush settings to disk before tearing down the context.
    // ImGui's auto-save runs on a timer (default 5 s) and would
    // otherwise lose state if the user quits within a few seconds of
    // collapsing/moving a window.
    ImGuiIO& io = ImGui::GetIO();
    if (io.IniFilename) {
        ImGui::SaveIniSettingsToDisk(io.IniFilename);
    }
    if (hud_icons_sampler_) { vkDestroySampler(device_, hud_icons_sampler_, nullptr); hud_icons_sampler_ = VK_NULL_HANDLE; }
    if (hud_icons_tex_.ok) destroy_texture_2d(device_, allocator_, hud_icons_tex_);
    hud_icons_id_ = 0;
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    if (imgui_pool_) vkDestroyDescriptorPool(device_, imgui_pool_, nullptr);
    imgui_pool_ = VK_NULL_HANDLE;
    imgui_initialized_ = false;
}


} // namespace qlike
