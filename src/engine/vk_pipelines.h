#pragma once

#include <vulkan/vulkan.h>

#include <future>
#include <string>
#include <vector>

namespace vibe::vkpipe {

VkShaderModule load_shader_module(VkDevice device, const std::string& spv_path);

struct GraphicsPipelineConfig {
    VkShaderModule vert = VK_NULL_HANDLE;
    VkShaderModule frag = VK_NULL_HANDLE;
    // Optional tessellation stages. When both are set, the builder adds
    // the TESC/TESE stages, switches input-assembly topology to
    // PATCH_LIST, and emits a VkPipelineTessellationStateCreateInfo with
    // `patch_control_points` control points per patch (3 = triangle
    // patches, which lets us reuse a normal triangle index buffer).
    VkShaderModule tesc = VK_NULL_HANDLE;
    VkShaderModule tese = VK_NULL_HANDLE;
    uint32_t patch_control_points = 0;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    // Single-attachment shorthand — used by every pipeline that has exactly
    // one color attachment. For multi-attachment output (e.g. cube.frag
    // writing scene_color at location 0 and motion vector at location 1),
    // populate `color_formats` instead; if non-empty it overrides this.
    VkFormat color_format = VK_FORMAT_UNDEFINED;
    std::vector<VkFormat> color_formats;
    VkFormat depth_format = VK_FORMAT_UNDEFINED;
    std::vector<VkVertexInputBindingDescription> vbindings;
    std::vector<VkVertexInputAttributeDescription> vattrs;
    // Wireframe debug — requires the fillModeNonSolid device feature.
    VkPolygonMode polygon_mode = VK_POLYGON_MODE_FILL;
    VkCullModeFlags cull = VK_CULL_MODE_BACK_BIT;
    VkFrontFace front_face = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    bool depth_test = true;
    bool depth_write = true;
    // LESS_OR_EQUAL is the safe default — same behaviour as LESS for geometry
    // drawn once, but lets a depth pre-pass + color pass coexist (the color
    // pass's fragment must pass against the *equal* depth value written
    // earlier).
    VkCompareOp depth_compare = VK_COMPARE_OP_LESS_OR_EQUAL;
    int color_attachment_count = 1;

    // Per-attachment additive blend — used by the bloom upsample pass to
    // accumulate into the larger mip. Applied to ALL color attachments when
    // set; if you need mixed blend modes (one additive, others not) this
    // field needs to grow into a per-attachment vector — no caller needs
    // that today.
    bool additive_blend = false;

    // Standard alpha blending (SRC_ALPHA, ONE_MINUS_SRC_ALPHA) on color
    // attachment 0 ONLY. Used by the raymarched grass pipeline so blade
    // edges get real opacity against the underlying terrain while the
    // motion-vector attachment (location 1) stays opaque (TAA needs an
    // unblended motion read). Mutually exclusive with additive_blend.
    bool alpha_blend_color0_only = false;

    // Premultiplied alpha (ONE, ONE_MINUS_SRC_ALPHA) on all colour
    // attachments. Output alpha 0 = purely additive. Used by the FX pass.
    bool premultiplied_blend = false;

    // Sky fill: colour += src * dst.alpha (dst alpha = sky coverage),
    // alpha left untouched.
    bool dst_alpha_add_blend = false;

    // Enables VK_DYNAMIC_STATE_DEPTH_BIAS so the caller can drive depth
    // bias per-frame via vkCmdSetDepthBias. Required for the shadow-map
    // depth-only pipeline (slope-scale + constant bias to combat acne).
    bool depth_bias_enable = false;

    // Enables attachment-based Variable Rate Shading on this pipeline.
    // Chains a VkPipelineFragmentShadingRateStateCreateInfoKHR with combiner
    // ops set to KEEP/REPLACE so the per-pixel shading rate is taken from
    // whatever VRS attachment is bound at draw time. Pipeline-rate and
    // primitive-rate paths are intentionally inert — engine drives VRS
    // entirely via the attachment.
    bool enable_vrs = false;
};

VkPipeline build_graphics_pipeline(VkDevice device, const GraphicsPipelineConfig& cfg);

// The same on a thread of its own, for the very large shaders (the terrain march, its
// variants, the ground bake, the mesh-terrain water) that the game starts without and takes
// over when they are ready. The builds run lowest `priority` first, then in the order they were
// asked for, at most compile_jobs() at once: a first-start compile of one of them takes the
// driver's compiler 3-5 GB of memory, and all five side by side took the game to 15 GB in use,
// 21 GB committed. VIBE_PIPELINE_COMPILE_JOBS=<n> sets the number (default 1).
std::future<VkPipeline> build_graphics_pipeline_async(VkDevice device, const GraphicsPipelineConfig& cfg, int priority = 0);
int compile_jobs();
// At shutdown: the builds still waiting for their turn are not started (their futures give
// VK_NULL_HANDLE); a build already running is finished.
void cancel_queued_builds();
// While held, no queued build starts (so builds asked for together start by priority, not by
// who asked first).
void hold_queued_builds(bool hold);

// What the background builds are doing, for the on-screen notice (loader_screen.cpp
// draw_compile_notice) - read straight from the queue above, so it follows whatever pipelines
// are queued (no list of them anywhere). A "batch" is everything queued since the queue was
// last empty. Estimates come from each build's last real compile time on this machine, kept in
// vibe_pipeline_times.txt next to vibe_pipeline.cache ("<key> <seconds>" a line; only builds
// that took 2 s or more are recorded - a quick one came from the cache and says nothing); a
// build never timed is guessed from its SPIR-V size (12.5 s per MB, at least 2 s: the full
// terrain shader, 10 MB, took 124 s on the RTX 4080).
struct CompileStatus {
    int total = 0;             // builds in this batch
    int done = 0;              // of them finished (or dropped at shutdown)
    int running = 0;
    std::string current;       // the running build's name (the first, if several), in plain words
    double current_elapsed = 0.0, current_estimate = 0.0;   // (s)
    double remaining = 0.0;    // estimated seconds until the whole batch is done
    double fraction = 0.0;     // 0 .. 1, by estimated time
    double batch_elapsed = 0.0;
    bool active = false;       // something is waiting or running
};
CompileStatus compile_status();

// Optional persistent pipeline cache. When set via set_pipeline_cache(), every
// build_graphics_pipeline() call after that hands the cache to Vulkan, so
// equivalent pipelines compile in milliseconds instead of seconds. The engine
// is expected to load the cache from disk at startup, set it here, and write
// the data back at shutdown.
void set_pipeline_cache(VkPipelineCache cache);

} // namespace vibe::vkpipe
