#include "engine/vk_pipelines.h"

#include "engine/log.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <map>
#include <utility>
#include <vector>

namespace vibe::vkpipe {

// Set by the engine via set_pipeline_cache() once it's loaded the disk blob.
// Stays VK_NULL_HANDLE until then; vkCreateGraphicsPipelines accepts null and
// just doesn't cache, which is the correct fallback if the cache file is
// missing or corrupt.
static VkPipelineCache g_pipeline_cache = VK_NULL_HANDLE;

// The shader modules made by load_shader_module: their file name (without the folder and
// ".spv") and SPIR-V size, so a background build can say what it is compiling and guess how long
// (compile_status). A handle reused for another module is simply written over.
namespace {
struct ModuleInfo { std::string name; size_t bytes = 0; };
std::mutex g_mod_mutex;
std::unordered_map<VkShaderModule, ModuleInfo> g_modules;
void note_module(VkShaderModule m, const std::string& path, size_t bytes) {
    std::string n = path;
    if (const size_t s = n.find_last_of("/\\"); s != std::string::npos) n = n.substr(s + 1);
    if (n.size() > 4 && n.compare(n.size() - 4, 4, ".spv") == 0) n.resize(n.size() - 4);
    std::lock_guard<std::mutex> lk(g_mod_mutex);
    g_modules[m] = { n, bytes };
}
ModuleInfo module_info(VkShaderModule m) {
    std::lock_guard<std::mutex> lk(g_mod_mutex);
    auto it = g_modules.find(m);
    return it != g_modules.end() ? it->second : ModuleInfo{};
}
}  // namespace

VkShaderModule load_shader_module(VkDevice device, const std::string& path_in) {
    // The directory compiled in is the build tree of the machine that built the game. A copy
    // somewhere else (a release) has its shaders next to it: build/shaders under the folder the
    // game is started from - looked for first, so a copy never reads another tree's shaders.
    std::string path = path_in;
    {
        const size_t slash = path_in.find_last_of("/\\");
        const std::string local = "build/shaders/" + (slash == std::string::npos ? path_in : path_in.substr(slash + 1));
        if (std::ifstream t(local, std::ios::binary); t.good()) path = local;
        // VIBE_SHADER_OVERRIDE_DIR=<dir>: a shader of the same name in that folder is taken
        // instead (a test copy tries other builds of a shader without touching the shared tree).
        if (const char* od = std::getenv("VIBE_SHADER_OVERRIDE_DIR")) {
            const std::string alt = std::string(od) + "/" + (slash == std::string::npos ? path_in : path_in.substr(slash + 1));
            if (std::ifstream t(alt, std::ios::binary); t.good()) path = alt;
        }
    }
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("can't open shader: " + path);
    std::streamsize sz = f.tellg();
    f.seekg(0);
    std::vector<char> bytes(static_cast<size_t>(sz));
    f.read(bytes.data(), sz);
    // VIBE_TEST_COLD_COMPILE=1: every start a first start - the SPIR-V header's generator word
    // (informational only) is set to a number of this run, so no cache (ours or the driver's
    // own on disk) knows the shaders and the driver compiles them all again.
    // (=<text>: only the shaders whose file name holds that text, e.g. terrain_raymarch)
    if (static const char* cc = std::getenv("VIBE_TEST_COLD_COMPILE"); cc && sz >= 20 &&
        (std::strcmp(cc, "1") == 0 || path.find(cc) != std::string::npos)) {
        static const uint32_t stamp = 0x7E570000u ^ static_cast<uint32_t>(std::chrono::steady_clock::now().time_since_epoch().count());
        std::memcpy(bytes.data() + 8, &stamp, sizeof(stamp));
    }

    VkShaderModuleCreateInfo ci{
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .pNext = nullptr, .flags = 0,
        .codeSize = static_cast<size_t>(sz),
        .pCode = reinterpret_cast<const uint32_t*>(bytes.data()),
    };
    VkShaderModule mod = VK_NULL_HANDLE;
    if (vkCreateShaderModule(device, &ci, nullptr, &mod) != VK_SUCCESS) {
        throw std::runtime_error("vkCreateShaderModule failed for " + path);
    }
    note_module(mod, path, static_cast<size_t>(sz));
    log::infof("loaded shader: %s (%lld bytes)", path.c_str(), static_cast<long long>(sz));
    return mod;
}

VkPipeline build_graphics_pipeline(VkDevice device, const GraphicsPipelineConfig& cfg) {
    VkPipelineShaderStageCreateInfo stages[4]{};
    uint32_t stage_count = 0;
    stages[stage_count].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[stage_count].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[stage_count].module = cfg.vert;
    stages[stage_count].pName = "main";
    ++stage_count;
    const bool use_tess = cfg.tesc && cfg.tese && cfg.patch_control_points > 0;
    if (use_tess) {
        stages[stage_count].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[stage_count].stage = VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT;
        stages[stage_count].module = cfg.tesc;
        stages[stage_count].pName = "main";
        ++stage_count;
        stages[stage_count].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[stage_count].stage = VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;
        stages[stage_count].module = cfg.tese;
        stages[stage_count].pName = "main";
        ++stage_count;
    }
    // Fragment stage is optional — depth-only shadow passes drop it so
    // the rasteriser writes only depth and no color.
    if (cfg.frag) {
        stages[stage_count].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[stage_count].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        stages[stage_count].module = cfg.frag;
        stages[stage_count].pName = "main";
        ++stage_count;
    }

    VkPipelineVertexInputStateCreateInfo vi{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .pNext = nullptr, .flags = 0,
        .vertexBindingDescriptionCount = static_cast<uint32_t>(cfg.vbindings.size()),
        .pVertexBindingDescriptions = cfg.vbindings.data(),
        .vertexAttributeDescriptionCount = static_cast<uint32_t>(cfg.vattrs.size()),
        .pVertexAttributeDescriptions = cfg.vattrs.data(),
    };

    VkPipelineInputAssemblyStateCreateInfo ia{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .pNext = nullptr, .flags = 0,
        .topology = use_tess ? VK_PRIMITIVE_TOPOLOGY_PATCH_LIST
                             : VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
        .primitiveRestartEnable = VK_FALSE,
    };

    VkPipelineTessellationStateCreateInfo tess{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO,
        .pNext = nullptr, .flags = 0,
        .patchControlPoints = cfg.patch_control_points,
    };

    VkPipelineViewportStateCreateInfo vp{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .pNext = nullptr, .flags = 0,
        .viewportCount = 1, .pViewports = nullptr,
        .scissorCount = 1, .pScissors = nullptr,
    };

    VkPipelineRasterizationStateCreateInfo rs{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .pNext = nullptr, .flags = 0,
        .depthClampEnable = VK_FALSE,
        .rasterizerDiscardEnable = VK_FALSE,
        .polygonMode = cfg.polygon_mode,
        .cullMode = cfg.cull,
        .frontFace = cfg.front_face,
        .depthBiasEnable = cfg.depth_bias_enable ? VK_TRUE : VK_FALSE,
        .depthBiasConstantFactor = 0, .depthBiasClamp = 0, .depthBiasSlopeFactor = 0,
        .lineWidth = 1.0f,
    };

    VkPipelineMultisampleStateCreateInfo ms{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .pNext = nullptr, .flags = 0,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
        .sampleShadingEnable = VK_FALSE,
        .minSampleShading = 1.0f, .pSampleMask = nullptr,
        .alphaToCoverageEnable = VK_FALSE, .alphaToOneEnable = VK_FALSE,
    };

    VkPipelineDepthStencilStateCreateInfo ds{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .pNext = nullptr, .flags = 0,
        .depthTestEnable = cfg.depth_test ? VK_TRUE : VK_FALSE,
        .depthWriteEnable = cfg.depth_write ? VK_TRUE : VK_FALSE,
        .depthCompareOp = cfg.depth_compare,
        .depthBoundsTestEnable = VK_FALSE,
        .stencilTestEnable = VK_FALSE,
        .front = {}, .back = {},
        .minDepthBounds = 0.0f, .maxDepthBounds = 1.0f,
    };

    // Resolve color attachment count: prefer the multi-format `color_formats`
    // vector when populated (used by cube.frag's color + motion-vec output);
    // fall back to the single-format `color_attachment_count` shorthand.
    const uint32_t resolved_color_count =
        cfg.color_formats.empty()
            ? static_cast<uint32_t>(cfg.color_attachment_count)
            : static_cast<uint32_t>(cfg.color_formats.size());

    // One blend state per color attachment. additive_blend applies uniformly
    // to all attachments. alpha_blend_color0_only enables SRC_ALPHA blending
    // on attachment 0 only (used by the grass raymarch pipeline so blade
    // edges blend with terrain while motion-vec attachment 1 stays opaque).
    std::vector<VkPipelineColorBlendAttachmentState> blends(resolved_color_count);
    for (uint32_t ai = 0; ai < resolved_color_count; ++ai) {
        VkPipelineColorBlendAttachmentState& blend = blends[ai];
        bool alpha_blend = cfg.alpha_blend_color0_only && ai == 0;
        // (the sky fill's blend is for the colour only; its motion-vector attachment is a plain write)
        const bool dst_add = cfg.dst_alpha_add_blend && ai == 0;
        blend.blendEnable = ((!cfg.dst_alpha_add_blend && (cfg.additive_blend || alpha_blend || cfg.premultiplied_blend)) ||
                             dst_add) ? VK_TRUE : VK_FALSE;
        if (cfg.dst_alpha_add_blend && !dst_add) {
            // no blending
        } else if (dst_add) {
            blend.srcColorBlendFactor = VK_BLEND_FACTOR_DST_ALPHA;
            blend.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
            blend.colorBlendOp        = VK_BLEND_OP_ADD;
            blend.srcAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
            blend.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
            blend.alphaBlendOp        = VK_BLEND_OP_ADD;
        } else if (cfg.premultiplied_blend) {
            blend.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
            blend.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
            blend.colorBlendOp        = VK_BLEND_OP_ADD;
            // Alpha tracks TRANSMITTANCE (dst.a *= 1 - src.a): compose uses
            // it to put the real sky behind FX drawn over sky pixels.
            blend.srcAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
            blend.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
            blend.alphaBlendOp        = VK_BLEND_OP_ADD;
        } else if (cfg.additive_blend) {
            blend.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
            blend.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
            blend.colorBlendOp        = VK_BLEND_OP_ADD;
            blend.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
            blend.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
            blend.alphaBlendOp        = VK_BLEND_OP_ADD;
        } else if (alpha_blend) {
            blend.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
            blend.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
            blend.colorBlendOp        = VK_BLEND_OP_ADD;
            blend.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
            blend.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
            blend.alphaBlendOp        = VK_BLEND_OP_ADD;
        }
        blend.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                               VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    }

    VkPipelineColorBlendStateCreateInfo cb{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .pNext = nullptr, .flags = 0,
        .logicOpEnable = VK_FALSE, .logicOp = VK_LOGIC_OP_COPY,
        .attachmentCount = resolved_color_count,
        .pAttachments = resolved_color_count > 0 ? blends.data() : nullptr,
        .blendConstants = {0,0,0,0},
    };

    VkDynamicState dyn_states[3] = { VK_DYNAMIC_STATE_VIEWPORT,
                                      VK_DYNAMIC_STATE_SCISSOR,
                                      VK_DYNAMIC_STATE_DEPTH_BIAS };
    VkPipelineDynamicStateCreateInfo dyn{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .pNext = nullptr, .flags = 0,
        .dynamicStateCount = cfg.depth_bias_enable ? 3u : 2u,
        .pDynamicStates = dyn_states,
    };

    const VkFormat* color_format_ptr = nullptr;
    if (!cfg.color_formats.empty()) {
        color_format_ptr = cfg.color_formats.data();
    } else if (cfg.color_attachment_count > 0) {
        color_format_ptr = &cfg.color_format;
    }
    VkPipelineRenderingCreateInfo rendering{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
        .pNext = nullptr,
        .viewMask = 0,
        .colorAttachmentCount = resolved_color_count,
        .pColorAttachmentFormats = color_format_ptr,
        .depthAttachmentFormat = cfg.depth_format,
        .stencilAttachmentFormat = VK_FORMAT_UNDEFINED,
    };

    // Optional attachment-based VRS state. Chained AFTER rendering so the
    // pNext walk hits it; combiner ops force the per-fragment rate to come
    // from the bound attachment image (REPLACE on op[1] discards the
    // pipeline/primitive rates from op[0]'s KEEP). fragmentSize is the
    // pipeline-rate slot we set to 1×1; combinerOps[0] = KEEP just passes
    // that through. combinerOps[1] = REPLACE overrides with attachment.
    VkPipelineFragmentShadingRateStateCreateInfoKHR vrs_state{};
    if (cfg.enable_vrs) {
        vrs_state.sType = VK_STRUCTURE_TYPE_PIPELINE_FRAGMENT_SHADING_RATE_STATE_CREATE_INFO_KHR;
        vrs_state.fragmentSize = { 1, 1 };
        vrs_state.combinerOps[0] = VK_FRAGMENT_SHADING_RATE_COMBINER_OP_KEEP_KHR;
        vrs_state.combinerOps[1] = VK_FRAGMENT_SHADING_RATE_COMBINER_OP_REPLACE_KHR;
        vrs_state.pNext = rendering.pNext;
        rendering.pNext = &vrs_state;
    }

    VkGraphicsPipelineCreateInfo pci{
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .pNext = &rendering, .flags = 0,
        .stageCount = stage_count, .pStages = stages,
        .pVertexInputState = &vi,
        .pInputAssemblyState = &ia,
        .pTessellationState = use_tess ? &tess : nullptr,
        .pViewportState = &vp,
        .pRasterizationState = &rs,
        .pMultisampleState = &ms,
        .pDepthStencilState = &ds,
        .pColorBlendState = &cb,
        .pDynamicState = &dyn,
        .layout = cfg.layout,
        .renderPass = VK_NULL_HANDLE,
        .subpass = 0,
        .basePipelineHandle = VK_NULL_HANDLE,
        .basePipelineIndex = -1,
    };

    VkPipeline pipeline = VK_NULL_HANDLE;
    if (vkCreateGraphicsPipelines(device, g_pipeline_cache, 1, &pci, nullptr, &pipeline) != VK_SUCCESS) {
        throw std::runtime_error("vkCreateGraphicsPipelines failed");
    }
    return pipeline;
}

void set_pipeline_cache(VkPipelineCache cache) {
    g_pipeline_cache = cache;
}

int compile_jobs() {
    static const int n = [] {
        const char* e = std::getenv("VIBE_PIPELINE_COMPILE_JOBS");
        const int v = e ? std::atoi(e) : 1;
        return v < 1 ? 1 : (v > 8 ? 8 : v);
    }();
    return n;
}

namespace {
// The queue of the background builds: (priority, ticket) per build, taken on the caller's thread;
// the first of the waiting ones starts when a slot is free.
std::mutex g_q_mutex;
std::condition_variable g_q_cv;
std::set<std::pair<int, uint64_t>> g_q_waiting;
uint64_t g_q_next_ticket = 0;
int g_q_running = 0;
bool g_q_cancel = false;
bool g_q_hold = false;

// What each build of the current batch is (compile_status), and the measured compile times
// (vibe_pipeline_times.txt). All under g_q_mutex.
using Clock = std::chrono::steady_clock;
struct BuildEntry {
    std::string key;      // for the times file: "<frag or vert module>:<colour outputs>[v]"
    std::string label;    // in plain words, for the notice
    double estimate = 0.0;
    int state = 0;        // 0 waiting, 1 running, 2 done
    Clock::time_point start{};
};
std::map<uint64_t, BuildEntry> g_batch;          // by ticket
Clock::time_point g_batch_start{};
std::unordered_map<std::string, double> g_times;
bool g_times_loaded = false;
constexpr const char* kTimesPath = "vibe_pipeline_times.txt";

void times_load_locked() {
    if (g_times_loaded) return;
    g_times_loaded = true;
    std::ifstream f(kTimesPath);
    std::string k; double s = 0.0;
    while (f >> k >> s) if (s > 0.0 && s < 3600.0) g_times[k] = s;
}
void times_save_locked() {
    std::ofstream f(kTimesPath, std::ios::trunc);
    if (!f) return;
    f << "# Vibe: the last real compile time (s) of each background pipeline build on this machine (vk_pipelines.cpp)\n";
    std::map<std::string, double> sorted(g_times.begin(), g_times.end());
    for (const auto& [k, s] : sorted) f << k << " " << s << "\n";
}

// "terrain_raymarch_summer.frag" -> "terrain raymarch summer"; a few names said in plain words.
std::string plain_label(const std::string& mod) {
    std::string n = mod;
    for (const char* ext : { ".frag", ".vert", ".comp" })
        if (n.size() > 5 && n.compare(n.size() - 5, 5, ext) == 0) n.resize(n.size() - 5);
    struct Known { const char* prefix; const char* words; };
    static const Known kKnown[] = {
        { "terrain_raymarch_summer", "the landscape (summer version)" },
        { "terrain_raymarch_lite", "the landscape (simple version)" },
        { "terrain_raymarch", "the landscape" },
        { "terrain_bake", "the ground's texture bake" },
        { "terrain_water", "the water on the landscape" },
        { "water", "the water" },
        { "terrain", "the landscape" },
    };
    for (const Known& k : kKnown)
        if (n.compare(0, std::strlen(k.prefix), k.prefix) == 0) {
            std::string rest = n.substr(std::strlen(k.prefix));
            for (char& c : rest) if (c == '_') c = ' ';
            while (!rest.empty() && rest.front() == ' ') rest.erase(rest.begin());
            return rest.empty() ? std::string(k.words) : std::string(k.words) + " - " + rest;
        }
    for (char& c : n) if (c == '_') c = ' ';
    return n.empty() ? std::string("a shader") : n;
}
}  // namespace

CompileStatus compile_status() {
    CompileStatus st;
    std::lock_guard<std::mutex> lk(g_q_mutex);
    if (g_batch.empty()) return st;
    const auto now = Clock::now();
    double est_total = 0.0, est_done = 0.0, run_left_max = 0.0, wait_sum = 0.0;
    for (const auto& [t, e] : g_batch) {
        ++st.total;
        est_total += e.estimate;
        if (e.state == 2) { ++st.done; est_done += e.estimate; continue; }
        if (e.state == 1) {
            const double el = std::chrono::duration<double>(now - e.start).count();
            if (st.running++ == 0) { st.current = e.label; st.current_elapsed = el; st.current_estimate = e.estimate; }
            // (past its estimate: not done yet, so say a little more is left rather than none)
            const double left = (std::max)(e.estimate - el, (std::max)(2.0, 0.15 * e.estimate));
            run_left_max = (std::max)(run_left_max, left);
            est_done += (std::max)(0.0, e.estimate - left);
        } else {
            wait_sum += e.estimate;
        }
    }
    st.active = st.done < st.total;
    st.remaining = st.active ? run_left_max + wait_sum / double(compile_jobs()) : 0.0;
    st.fraction = est_total > 0.0 ? std::clamp(est_done / est_total, 0.0, 1.0) : 1.0;
    if (!st.active) st.fraction = 1.0;
    st.batch_elapsed = std::chrono::duration<double>(now - g_batch_start).count();
    return st;
}

void cancel_queued_builds() {
    { std::lock_guard<std::mutex> lk(g_q_mutex); g_q_cancel = true; }
    g_q_cv.notify_all();
}

void hold_queued_builds(bool hold) {
    { std::lock_guard<std::mutex> lk(g_q_mutex); g_q_hold = hold; }
    g_q_cv.notify_all();
}

std::future<VkPipeline> build_graphics_pipeline_async(VkDevice device, const GraphicsPipelineConfig& cfg, int priority) {
    std::pair<int, uint64_t> key;
    {
        // What it is (the fragment shader's file, else the vertex shader's) and how long it took last time.
        const ModuleInfo fm = module_info(cfg.frag), vm = module_info(cfg.vert);
        const ModuleInfo& m = fm.name.empty() ? vm : fm;
        const size_t outs = cfg.color_formats.empty() ? size_t(cfg.color_attachment_count) : cfg.color_formats.size();
        BuildEntry e;
        e.key = (m.name.empty() ? std::string("unknown") : m.name) + ":" + std::to_string(outs) + (cfg.enable_vrs ? "v" : "");
        e.label = plain_label(m.name);
        std::lock_guard<std::mutex> lk(g_q_mutex);
        key = { priority, g_q_next_ticket++ };
        g_q_waiting.insert(key);
        times_load_locked();
        bool batch_done = true;
        for (const auto& [t, b] : g_batch) if (b.state != 2) batch_done = false;
        if (batch_done) { g_batch.clear(); g_batch_start = Clock::now(); }    // (a new batch)
        for (const auto& [t, b] : g_batch)
            if (b.label == e.label || b.label.rfind(e.label + " (", 0) == 0) { e.label += " (another pass)"; break; }
        auto it = g_times.find(e.key);
        e.estimate = it != g_times.end() ? it->second
                                         : (std::max)(2.0, 12.5 * double(fm.bytes + vm.bytes) / (1024.0 * 1024.0));
        if (m.name.empty() && it == g_times.end()) e.estimate = 30.0;
        g_batch[key.second] = e;
    }
    return std::async(std::launch::async, [device, cfg, key]() {
        {
            std::unique_lock<std::mutex> lk(g_q_mutex);
            g_q_cv.wait(lk, [key] { return g_q_cancel || (!g_q_hold && *g_q_waiting.begin() == key && g_q_running < compile_jobs()); });
            g_q_waiting.erase(key);
            if (g_q_cancel) { g_batch[key.second].state = 2; return VkPipeline(VK_NULL_HANDLE); }   // (shutting down: not started, not built)
            ++g_q_running;
            BuildEntry& e = g_batch[key.second];
            e.state = 1;
            e.start = Clock::now();
            int n = 0, m = 0;
            for (const auto& [t, b] : g_batch) { ++m; if (b.state != 0) ++n; }
            log::infof("[pipeline] compiling %s (%s) - %d of %d, estimate %.0f s", e.label.c_str(), e.key.c_str(), n, m, e.estimate);
        }
        g_q_cv.notify_all();
        struct Done { uint64_t t; ~Done() { { std::lock_guard<std::mutex> lk(g_q_mutex); --g_q_running; g_batch[t].state = 2; } g_q_cv.notify_all(); } } done{ key.second };   // (also when the build throws)
        const auto t0 = Clock::now();
        VkPipeline p = build_graphics_pipeline(device, cfg);
        const double took = std::chrono::duration<double>(Clock::now() - t0).count();
        {
            std::lock_guard<std::mutex> lk(g_q_mutex);
            BuildEntry& e = g_batch[key.second];
            e.state = 2;
            log::infof("[pipeline] background build %llu (priority %d) %s took %.1f s (estimate was %.0f s)",
                       static_cast<unsigned long long>(key.second), key.first, e.label.c_str(), took, e.estimate);
            if (took >= 2.0) { g_times[e.key] = took; times_save_locked(); }   // (a quick one came from the cache)
            bool all = true;
            for (const auto& [t, b] : g_batch) if (b.state != 2) all = false;
            if (all) log::infof("[pipeline] all %d background builds done after %.1f s", int(g_batch.size()),
                                std::chrono::duration<double>(Clock::now() - g_batch_start).count());
        }
        return p;
    });
}

} // namespace vibe::vkpipe
