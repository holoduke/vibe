// The water's simulation on the GPU: the wind sea's FFT cascades and the ripple field.
//
// After the reference game's water (its formulas and constants where it has them; the code is
// ours; shaders/water_sim.comp has each pass):
//   * FOUR FFT CASCADES of 64 x 64 (41, 13.7, 4.4 and 1.37 m): a Phillips spectrum with the
//     standard wind length U^2 / g (the reference has ln(wind + 10)^2 / 9.8), waves against the wind damped, deep-water
//     dispersion, choppy (horizontal) displacement. Each cascade holds its own band of wave
//     numbers, so the four are one spectrum from 41 m waves down to 4 cm. Computed every frame:
//     spectrum -> FFT along x -> FFT along z -> displacement (scene binding 52) -> slope, fold and
//     height with mips (51; the mips are the distance filter, as the reference's). The
//     reference's per-cascade amplitudes are runtime values; here the long and the short cascades
//     each get a gain so that the wind sea carries what the swell leaves of a fully developed sea's
//     height (Hs = 0.21 U^2 / g) and its slope variance is 80 % of Cox-Munk (0.003 + 0.00512 W) - the rest
//     is below the finest cascade and goes into the glint's width.
//   * THE CPU TWIN: the displacement is copied back every frame (three slots; the one two frames
//     old is complete) and water_fft_height() samples it as the shader does - swimmers, splashes
//     and bullets meet the surface the open sea draws.
//   * THE RIPPLE FIELD: the wave equation on 512 x 512 cells of 12.5 cm (64 m) round the camera,
//     two steps a frame, scrolling with it in whole cells; the banks are walls (the home tile's
//     baked ground above the water - the sea's level or the stream's surface), the river's and the
//     stream's current carries it (a 16 x 16 grid of the CPU's current, water_river_flow), and it
//     is fed by people wading, cars fording, the world's people, and every new splash ring
//     (bullets, droplets, bodies). Scene binding 53.
//
// Switches: VIBE_NO_FFT=1 (the old analytic wind waves), VIBE_NO_RIPPLE_FIELD=1 (the analytic
// rings alone). Hooks: VIBE_TEST_RIPPLE=<k> (a made-up wader circling 6 m in front of the camera,
// k = its strength), VIBE_LOG_WATER_SIM=1 (the spectrum's figures, the field's window, once a second).
#include "engine/vk_engine/internal.h"
#include "engine/vk_engine/state/environment.h"
#include "engine/vk_initializers.h"
#include "engine/vk_pipelines.h"
#include "engine/globe.h"
#include "engine/water_course.h"
#include "npc/npc.h"
#include <glm/gtc/packing.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace vibe {

namespace {
constexpr uint32_t kN = 64;                     // FFT size (the reference's)
constexpr uint32_t kCasc = 4;
constexpr uint32_t kMips = 7;                   // 64 .. 1
constexpr float    kL[kCasc] = { 41.0f, 13.7f, 4.4f, 1.37f };      // keep in step: nothing else (the shader reads scene.water_fft_L)
constexpr uint32_t kRipN = 512;
constexpr float    kRipCell = 0.125f;
constexpr int      kImp = 64;
constexpr int      kFlowN = 16;
constexpr uint32_t kSlots = 3;
constexpr float    kWsPi = 3.14159265358979f;
constexpr float    kChoppy = 2.0f;               // the choppy displacement's scale (the reference's _choppyScale is a runtime value): sets how often the surface folds - the whitecaps

struct Params {
    glm::vec4 wind, L, band, gain, rip, rip_misc, flow_org, pad;
    glm::vec4 imp[kImp];
    glm::vec4 flow[kFlowN * kFlowN / 2];
};
struct Push { glm::ivec4 m; glm::vec4 s; };

struct Img {
    VkImage image = VK_NULL_HANDLE;
    VmaAllocation alloc = nullptr;
    VkImageView view = VK_NULL_HANDLE;      // all mips / layers
    VkImageView mip0 = VK_NULL_HANDLE;      // level 0 (storage)
};

struct WaterSim {
    Img spec, tmp, disp, grad, ripA, ripB;
    VkSampler s_grad = VK_NULL_HANDLE, s_disp = VK_NULL_HANDLE, s_rip = VK_NULL_HANDLE;
    VkBuffer params = VK_NULL_HANDLE;
    VmaAllocation params_alloc = nullptr;
    VkBuffer rb[kSlots] = {};
    VmaAllocation rb_alloc[kSlots] = {};
    void* rb_ptr[kSlots] = {};
    uint64_t rb_frame[kSlots] = {};
    bool rb_valid[kSlots] = {};
    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VkDescriptorSet set[2] = {};
    VkImageView set_hview[2] = {};
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipe = VK_NULL_HANDLE;
    bool ok = false;
    // spectrum
    float norm_wind = -1.0f, gain = 1.0f, sig_h = 0.0f, var_slope = 0.0f;
    glm::vec4 cg{ 1.0f };          // each cascade's gain
    glm::vec4 band{ 0.0f };
    // the CPU twin: heights of the four cascades (last complete read-back)
    std::vector<float> h;
    bool h_valid = false;
    // ripple field
    glm::vec2 org{ 0.0f }, prev_org{ 0.0f };
    bool have_org = false;
    float last_t = 0.0f;
    std::vector<glm::vec4> flow;
    int flow_age = 1000;
    glm::vec2 flow_org{ 1e9f };
    std::vector<glm::vec3> cars_prev;
    float log_t = 0.0f;
} g_ws;

bool env_on(const char* k) { const char* v = std::getenv(k); return v && *v && *v != '0'; }

// The reference's Phillips spectrum (water_sim.comp phillips; keep in step).
float phillips(glm::vec2 k, int c, glm::vec2 wd, float w, const glm::vec4& band) {
    const float kk = glm::dot(k, k);
    if (kk < 1e-10f) return 0.0f;
    const float kl = std::sqrt(kk);
    const float lo = band[c], hi = c < 3 ? band[c + 1] : 1e9f;
    if (kl < lo || kl >= hi) return 0.0f;
    const float Lw = std::max(w * w / 9.81f, 0.05f);      // (the standard Phillips length U^2 / g; the reference's ln(U + 10)^2 / 9.8 put all the wind sea in waves under 10 m: no sea state of a gale)
    const float kw = glm::dot(k / kl, wd);
    float p = std::max(0.25f, glm::clamp(w / 30.0f, 0.0f, 1.0f)) * kw * kw * std::exp(-1.0f / (kk * Lw * Lw)) / (kk * kk) *
              std::exp(-kk * (0.001f * Lw) * (0.001f * Lw));
    if (kw < 0.0f) p *= 0.25f;
    return p;
}

void make_img(VkDevice dev, VmaAllocator al, Img& im, VkImageType, VkFormat fmt, uint32_t w, uint32_t h, uint32_t layers, uint32_t mips, const char* name) {
    VkImageCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = fmt;
    ici.extent = { w, h, 1 };
    ici.mipLevels = mips; ici.arrayLayers = layers;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    vk_check(vmaCreateImage(al, &ici, &aci, &im.image, &im.alloc, nullptr), name);
    VkImageViewCreateInfo vci{};
    vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vci.image = im.image;
    vci.viewType = layers > 1 ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D;
    vci.format = fmt;
    vci.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, mips, 0, layers };
    vk_check(vkCreateImageView(dev, &vci, nullptr, &im.view), name);
    vci.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, layers };
    vk_check(vkCreateImageView(dev, &vci, nullptr, &im.mip0), name);
}
void free_img(VkDevice dev, VmaAllocator al, Img& im) {
    if (im.mip0) vkDestroyImageView(dev, im.mip0, nullptr);
    if (im.view) vkDestroyImageView(dev, im.view, nullptr);
    if (im.image) vmaDestroyImage(al, im.image, im.alloc);
    im = Img{};
}
void barrier_all(VkCommandBuffer cmd) {
    VkMemoryBarrier2 mb{};
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
    mb.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
    mb.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_UNIFORM_READ_BIT;
    mb.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
    mb.dstAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_UNIFORM_READ_BIT;
    VkDependencyInfo dep{};
    dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.memoryBarrierCount = 1; dep.pMemoryBarriers = &mb;
    vkCmdPipelineBarrier2(cmd, &dep);
}
}  // namespace

// The wind sea's height at (x, z) on the CPU: the four cascades' heights, sampled bilinearly as
// the shader's linear repeat sampler does (water_waves.glsl wt_fft_height; m, about the level).
float water_fft_height(float x, float z) {
    if (!g_ws.h_valid) return 0.0f;
    float h = 0.0f;
    for (uint32_t c = 0; c < kCasc; ++c) {
        const float u = x / kL[c] * float(kN) - 0.5f, v = z / kL[c] * float(kN) - 0.5f;
        const float fu = std::floor(u), fv = std::floor(v);
        const float tu = u - fu, tv = v - fv;
        const int iu = int(fu) & int(kN - 1), iv = int(fv) & int(kN - 1);
        const int iu1 = (iu + 1) & int(kN - 1), iv1 = (iv + 1) & int(kN - 1);
        const float* L = &g_ws.h[size_t(c) * kN * kN];
        const float a = L[iv * kN + iu], b = L[iv * kN + iu1], cc = L[iv1 * kN + iu], d = L[iv1 * kN + iu1];
        h += (a * (1.0f - tu) + b * tu) * (1.0f - tv) + (cc * (1.0f - tu) + d * tu) * tv;
    }
    return h;
}
// The bound the shader's hit search uses (m): 4 standard deviations of the height.
float water_fft_bound() { return 4.0f * g_ws.sig_h; }
void water_sim_grad_view(VkImageView& view, VkSampler& sampler) { view = g_ws.grad.view; sampler = g_ws.s_grad; }

void VulkanEngine::init_water_sim() {
    WaterSim& W = g_ws;
    const VkFormat f32 = VK_FORMAT_R32G32B32A32_SFLOAT, f16 = VK_FORMAT_R16G16B16A16_SFLOAT;
    make_img(device_, allocator_, W.spec, VK_IMAGE_TYPE_2D, f32, kN, kN, kCasc, 1, "water spectrum");
    make_img(device_, allocator_, W.tmp,  VK_IMAGE_TYPE_2D, f32, kN, kN, kCasc, 1, "water fft rows");
    make_img(device_, allocator_, W.disp, VK_IMAGE_TYPE_2D, f16, kN, kN, kCasc, 1, "water displacement");
    make_img(device_, allocator_, W.grad, VK_IMAGE_TYPE_2D, f16, kN, kN, kCasc, kMips, "water gradient");
    make_img(device_, allocator_, W.ripA, VK_IMAGE_TYPE_2D, f16, kRipN, kRipN, 1, 1, "ripple field A");
    make_img(device_, allocator_, W.ripB, VK_IMAGE_TYPE_2D, f16, kRipN, kRipN, 1, 1, "ripple field B");
    vkinit::one_time_submit(device_, graphics_queue_, graphics_queue_family_, [&](VkCommandBuffer cb) {
        for (Img* im : { &W.spec, &W.tmp, &W.disp, &W.grad, &W.ripA, &W.ripB }) {
            VkImageMemoryBarrier2 b{};
            b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
            b.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
            b.dstStageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT;
            b.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
            b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED; b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.image = im->image;
            b.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS };
            VkDependencyInfo d{}; d.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            d.imageMemoryBarrierCount = 1; d.pImageMemoryBarriers = &b;
            vkCmdPipelineBarrier2(cb, &d);
            VkClearColorValue z{};
            VkImageSubresourceRange rr{ VK_IMAGE_ASPECT_COLOR_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS };
            vkCmdClearColorImage(cb, im->image, VK_IMAGE_LAYOUT_GENERAL, &z, 1, &rr);
        }
    });
    {
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(physical_device_, &props);
        VkSamplerCreateInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        si.magFilter = si.minFilter = VK_FILTER_LINEAR;
        si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        si.maxLod = float(kMips);
        const float aniso = std::min(8.0f, props.limits.maxSamplerAnisotropy);
        si.anisotropyEnable = aniso > 1.0f ? VK_TRUE : VK_FALSE;
        si.maxAnisotropy = aniso;
        vk_check(vkCreateSampler(device_, &si, nullptr, &W.s_grad), "water gradient sampler");
        si.anisotropyEnable = VK_FALSE; si.maxAnisotropy = 1.0f;
        si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST; si.maxLod = 0.0f;
        vk_check(vkCreateSampler(device_, &si, nullptr, &W.s_disp), "water displacement sampler");
        si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        si.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
        vk_check(vkCreateSampler(device_, &si, nullptr, &W.s_rip), "ripple sampler");
    }
    {
        VkBufferCreateInfo bci{};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size = sizeof(Params);
        bci.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        vk_check(vmaCreateBuffer(allocator_, &bci, &aci, &W.params, &W.params_alloc, nullptr), "water sim params");
        for (uint32_t s = 0; s < kSlots; ++s) {
            VkBufferCreateInfo r = bci;
            r.size = size_t(kN) * kN * kCasc * 8;
            r.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            VmaAllocationCreateInfo ra{};
            ra.usage = VMA_MEMORY_USAGE_AUTO;
            ra.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
            VmaAllocationInfo info{};
            vk_check(vmaCreateBuffer(allocator_, &r, &ra, &W.rb[s], &W.rb_alloc[s], &info), "water read-back");
            W.rb_ptr[s] = info.pMappedData;
        }
    }
    // The passes' set: 0 the scene block, 1 the parameters, 2..7 the pictures, 8 the home tile's
    // ground, 9 / 10 the ripple pictures to read from.
    {
        VkDescriptorSetLayoutBinding b[11]{};
        for (uint32_t i = 0; i < 11; ++i) {
            b[i].binding = i; b[i].descriptorCount = 1; b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            b[i].descriptorType = i <= 1 ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
                                : (i <= 7 ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
        }
        VkDescriptorSetLayoutCreateInfo lci{};
        lci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        lci.bindingCount = 11; lci.pBindings = b;
        vk_check(vkCreateDescriptorSetLayout(device_, &lci, nullptr, &W.dsl), "water sim dsl");
        VkDescriptorPoolSize ps[3] = { { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 4 }, { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 12 }, { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 6 } };
        VkDescriptorPoolCreateInfo pci{};
        pci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pci.maxSets = 2; pci.poolSizeCount = 3; pci.pPoolSizes = ps;
        vk_check(vkCreateDescriptorPool(device_, &pci, nullptr, &W.pool), "water sim pool");
        VkDescriptorSetLayout ls[2] = { W.dsl, W.dsl };
        VkDescriptorSetAllocateInfo dai{};
        dai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dai.descriptorPool = W.pool; dai.descriptorSetCount = 2; dai.pSetLayouts = ls;
        vk_check(vkAllocateDescriptorSets(device_, &dai, W.set), "water sim sets");
        for (int s = 0; s < 2; ++s) {
            VkDescriptorBufferInfo bu{ scene_ubo_buffer_, 0, VK_WHOLE_SIZE }, bp{ W.params, 0, VK_WHOLE_SIZE };
            VkDescriptorImageInfo ii[6] = {
                { VK_NULL_HANDLE, W.spec.view, VK_IMAGE_LAYOUT_GENERAL }, { VK_NULL_HANDLE, W.tmp.view, VK_IMAGE_LAYOUT_GENERAL },
                { VK_NULL_HANDLE, W.disp.view, VK_IMAGE_LAYOUT_GENERAL }, { VK_NULL_HANDLE, W.grad.mip0, VK_IMAGE_LAYOUT_GENERAL },
                { VK_NULL_HANDLE, W.ripA.view, VK_IMAGE_LAYOUT_GENERAL }, { VK_NULL_HANDLE, W.ripB.view, VK_IMAGE_LAYOUT_GENERAL } };
            VkDescriptorImageInfo rs[2] = { { W.s_rip, W.ripA.view, VK_IMAGE_LAYOUT_GENERAL }, { W.s_rip, W.ripB.view, VK_IMAGE_LAYOUT_GENERAL } };
            VkWriteDescriptorSet w[10]{};
            for (int i = 0; i < 10; ++i) { w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w[i].dstSet = W.set[s]; w[i].descriptorCount = 1; }
            w[0].dstBinding = 0; w[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; w[0].pBufferInfo = &bu;
            w[1].dstBinding = 1; w[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; w[1].pBufferInfo = &bp;
            for (int i = 0; i < 6; ++i) { w[2 + i].dstBinding = uint32_t(2 + i); w[2 + i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; w[2 + i].pImageInfo = &ii[i]; }
            w[8].dstBinding = 9;  w[8].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w[8].pImageInfo = &rs[0];
            w[9].dstBinding = 10; w[9].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w[9].pImageInfo = &rs[1];
            vkUpdateDescriptorSets(device_, 10, w, 0, nullptr);
        }
        VkPushConstantRange pcr{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Push) };
        VkPipelineLayoutCreateInfo pli{};
        pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pli.setLayoutCount = 1; pli.pSetLayouts = &W.dsl;
        pli.pushConstantRangeCount = 1; pli.pPushConstantRanges = &pcr;
        vk_check(vkCreatePipelineLayout(device_, &pli, nullptr, &W.layout), "water sim layout");
        VkShaderModule m = vkpipe::load_shader_module(device_, std::string(VIBE_SHADER_DIR) + "/water_sim.comp.spv");
        if (m) {
            VkComputePipelineCreateInfo cci{};
            cci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
            cci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            cci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            cci.stage.module = m; cci.stage.pName = "main";
            cci.layout = W.layout;
            vk_check(vkCreateComputePipelines(device_, pipeline_cache_, 1, &cci, nullptr, &W.pipe), "water sim pipeline");
            vkDestroyShaderModule(device_, m, nullptr);
        }
    }
    // The scene set: 51 slope / fold / height (mips, anisotropic, repeat), 52 the displacement, 53 the ripple field.
    {
        VkDescriptorImageInfo ii[3] = { { W.s_grad, W.grad.view, VK_IMAGE_LAYOUT_GENERAL },
                                        { W.s_disp, W.disp.view, VK_IMAGE_LAYOUT_GENERAL },
                                        { W.s_rip,  W.ripA.view, VK_IMAGE_LAYOUT_GENERAL } };
        VkWriteDescriptorSet w[3]{};
        for (int i = 0; i < 3; ++i) {
            w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w[i].dstSet = scene_desc_set_;
            w[i].dstBinding = uint32_t(51 + i); w[i].descriptorCount = 1;
            w[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w[i].pImageInfo = &ii[i];
        }
        vkUpdateDescriptorSets(device_, 3, w, 0, nullptr);
    }
    W.h.assign(size_t(kN) * kN * kCasc, 0.0f);
    W.ok = W.pipe != VK_NULL_HANDLE;
    if (!W.ok) log::warnf("[water] water_sim shader missing - no FFT cascades, no ripple field");
    else log::infof("[water] simulation ready: 4 FFT cascades of %u^2 (%.1f / %.1f / %.2f / %.2f m), ripple field %u^2 x %.3f m",
                    kN, double(kL[0]), double(kL[1]), double(kL[2]), double(kL[3]), kRipN, double(kRipCell));
}

void VulkanEngine::destroy_water_sim() {
    WaterSim& W = g_ws;
    if (W.pipe) vkDestroyPipeline(device_, W.pipe, nullptr);
    if (W.layout) vkDestroyPipelineLayout(device_, W.layout, nullptr);
    if (W.pool) vkDestroyDescriptorPool(device_, W.pool, nullptr);
    if (W.dsl) vkDestroyDescriptorSetLayout(device_, W.dsl, nullptr);
    for (VkSampler s : { W.s_grad, W.s_disp, W.s_rip }) if (s) vkDestroySampler(device_, s, nullptr);
    if (W.params) vmaDestroyBuffer(allocator_, W.params, W.params_alloc);
    for (uint32_t s = 0; s < kSlots; ++s) if (W.rb[s]) vmaDestroyBuffer(allocator_, W.rb[s], W.rb_alloc[s]);
    for (Img* im : { &W.spec, &W.tmp, &W.disp, &W.grad, &W.ripA, &W.ripB }) free_img(device_, allocator_, *im);
    g_ws = WaterSim{};
}

// The scene block's part (update_scene_ubo).
void VulkanEngine::water_sim_fill(SceneUBO& d) {
    static const bool no_fft = env_on("VIBE_NO_FFT"), no_rip = env_on("VIBE_NO_RIPPLE_FIELD");
    static const bool no_refr = env_on("VIBE_NO_WATER_REFRACT"), no_caus = env_on("VIBE_NO_REFL_CAUSTICS");
    static const bool no_sdf = env_on("VIBE_NO_SHORE_SDF"), no_sky = env_on("VIBE_NO_WATER_SKY_AMB");
    const bool ok = g_ws.ok && rt_.water_enabled;
    d.water_fft = glm::vec4(ok && !no_fft ? 1.0f : 0.0f, ok && !no_fft ? 1.0f : 0.0f, water_fft_bound(), g_ws.var_slope);
    d.water_fft_L = glm::vec4(kL[0], kL[1], kL[2], kL[3]);
    d.water_rip = glm::vec4(g_ws.org, float(kRipN) * kRipCell, ok && !no_rip && g_ws.have_org ? 1.0f : 0.0f);
    d.water_flags = glm::vec4(no_refr ? 0.0f : 1.0f, no_caus ?  0.0f : (env_on("VIBE_DEBUG_REFL_CAUSTICS") ? 2.0f : 1.0f), no_sdf ? 0.0f : 1.0f, no_sky ? 0.0f : 1.0f);
    // The sun's disc through the cloud deck: a closed overcast hides it (the glint and the caustics are
    // images of the disc - the diffuse light under the deck is the scene's dimmed sun, kept for the rest).
    { const float oc = rt_.weather_enabled ? glm::clamp(wx_cur_.overcast, 0.0f, 1.0f) : 0.0f;
      d.water_sun = glm::vec4(std::pow(1.0f - oc, 2.0f), oc, env_ ? env_->sea_wind : 7.0f, 0.0f); }      // (z: the open sea's wind, environment.cpp)
}

void VulkanEngine::water_sim_update(VkCommandBuffer cmd) {
    WaterSim& W = g_ws;
    if (!W.ok || !rt_.water_enabled || !env_) return;
    const uint32_t slot = uint32_t(frame_number_ % kSlots);
    const uint32_t fset = uint32_t(frame_number_ % 2);

    // ---- the CPU twin: the read-back two frames old is complete (its frame's fence was waited on)
    {
        const uint32_t rs = uint32_t((frame_number_ + kSlots - 2) % kSlots);
        if (W.rb_valid[rs] && frame_number_ >= W.rb_frame[rs] + 2) {
            vmaInvalidateAllocation(allocator_, W.rb_alloc[rs], 0, VK_WHOLE_SIZE);
            const uint16_t* src = static_cast<const uint16_t*>(W.rb_ptr[rs]);
            for (size_t i = 0; i < size_t(kN) * kN * kCasc; ++i) W.h[i] = glm::unpackHalf1x16(src[i * 4 + 1]);
            W.h_valid = true;
        }
    }

    // ---- the spectrum's scale (Cox-Munk) when the wind speed changed
    const glm::vec4 wnd = env_->water_wind;
    // (the open sea's wind, not the local one: environment.cpp water_wind_now - a sea breeze at least)
    const float Usea = env_->sea_wind;
    const glm::vec2 wd = glm::normalize(glm::vec2(wnd.x, wnd.y) + glm::vec2(1e-6f, 0.0f));
    for (uint32_t c = 0; c < kCasc; ++c)
        W.band[int(c)] = c == 0 ? 0.0f : std::sqrt((2.0f * kWsPi / kL[c]) * (2.0f * kWsPi * 32.0f / kL[c - 1]));
    if (std::abs(Usea - W.norm_wind) > 0.1f) {
        W.norm_wind = Usea;
        double vs[kCasc] = {}, vh[kCasc] = {};
        for (uint32_t c = 0; c < kCasc; ++c) {
            const float dk = 2.0f * kWsPi / kL[c];
            for (uint32_t m = 0; m < kN; ++m)
                for (uint32_t n = 0; n < kN; ++n) {
                    const glm::vec2 k = dk * glm::vec2(float(int(n) - 32), float(int(m) - 32));
                    const double p = double(phillips(k, int(c), wd, Usea, W.band)) * double(dk) * double(dk);
                    vh[c] += p; vs[c] += p * double(glm::dot(k, k));
                }
        }
        // Two conditions, two gains (the long cascades', the short ones'): the wind sea's height and
        // its slope. A fully developed sea has Hs = 0.21 U^2 / g (Pierson-Moskowitz); the swell
        // (environment.cpp water_wind_now) carries 90 % of that height in its long waves, so the
        // cascades carry the rest: sd_h = sqrt(1 - 0.9^2) Hs / 4. Their slope variance is 80 % of
        // Cox-Munk's for the wind (0.003 + 0.00512 U) - the rest is below the finest cascade.
        const double U = std::max(double(Usea), 0.0);
        const double Hs = 0.21 * U * U / 9.81;
        const double th = std::pow(std::sqrt(1.0 - 0.36) * Hs / 4.0, 2.0);      // (the swell carries 60 % since 2026-11-27: environment.cpp)
        const double ts = 0.8 * (0.003 + 0.00512 * U);
        const double a = vh[0] + vh[1], b = vh[2] + vh[3], c2 = vs[0] + vs[1], d = vs[2] + vs[3];
        const double det = a * d - b * c2;
        double gl = 0.0, gs = 0.0;     // squared gains, long and short
        if (std::abs(det) > 1e-30) { gl = (th * d - b * ts) / det; gs = (a * ts - c2 * th) / det; }
        if (gl <= 0.0 || gs <= 0.0) {  // (not both reachable: the slope rules, one gain)
            const double vsum = c2 + d;
            gl = gs = vsum > 0.0 ? ts / vsum : 0.0;
        }
        W.cg = glm::vec4(float(std::sqrt(gl)), float(std::sqrt(gl)), float(std::sqrt(gs)), float(std::sqrt(gs)));
        W.gain = W.cg.x;
        W.sig_h = float(std::sqrt(gl * a + gs * b));
        W.var_slope = float(gl * c2 + gs * d);
    }

    // ---- the ripple field's window (whole cells round the camera) and what feeds it
    const glm::vec3 eye = current_frame_view_.eye_pos;
    const float half = 0.5f * float(kRipN) * kRipCell;
    W.prev_org = W.have_org ? W.org : glm::vec2(1e9f);
    W.org = glm::floor((glm::vec2(eye.x, eye.z) - glm::vec2(half)) / kRipCell) * kRipCell;
    if (!W.have_org) W.prev_org = W.org + glm::vec2(1e6f);      // (first frame: everything new)
    W.have_org = true;
    const float dt = std::clamp(anim_time_ - W.last_t, 0.0f, 0.05f);
    W.last_t = anim_time_;

    Params P{};
    P.wind = glm::vec4(wd, Usea, anim_time_);      // (the clock the water shader uses: scene.water_params.w)
    P.L = glm::vec4(kL[0], kL[1], kL[2], kL[3]);
    P.band = W.band;
    static const bool no_fft = env_on("VIBE_NO_FFT"), no_rip = env_on("VIBE_NO_RIPPLE_FIELD");
    P.gain = glm::vec4(1.0f, kChoppy, 0.25f, no_fft ? 0.0f : 1.0f);
    P.pad = W.cg;                  // (the cascades' own gains: water_sim.comp P.cgain)
    P.rip = glm::vec4(W.org, kRipCell, float(kRipN));
    // impulses
    int ni = 0;
    auto add_imp = [&](glm::vec2 p, float r, float h) {
        if (ni >= kImp || std::abs(h) < 1e-6f) return;
        if (p.x < W.org.x || p.y < W.org.y || p.x > W.org.x + 2.0f * half || p.y > W.org.y + 2.0f * half) return;
        P.imp[ni++] = glm::vec4(p, r, h);
    };
    // a body moving through the water heaps it up in front and draws it down behind
    auto mover = [&](glm::vec3 p, glm::vec3 vel, float radius, float bottom) {
        if (!water_at(p.x, p.z)) return;
        const float surf = water_surface_y(p.x, p.z);
        const float sub = glm::clamp((surf - bottom) / std::max(radius * 1.2f, 0.3f), 0.0f, 1.0f);
        const glm::vec2 v(vel.x, vel.z);
        const float sp = std::min(glm::length(v), 8.0f);
        if (sub <= 0.0f || sp < 0.15f) return;
        const glm::vec2 dir = v / std::max(glm::length(v), 1e-4f);
        const float a = 0.12f * sub * sp * dt;
        add_imp(glm::vec2(p.x, p.z) + dir * radius, radius, a);
        add_imp(glm::vec2(p.x, p.z) - dir * (0.6f * radius), radius * 0.9f, -0.7f * a);
    };
    if (!no_rip && dt > 0.0f && globe::current().home) {
        if (!body_seated_) mover(player_.position, player_.velocity, 0.35f, player_.position.y - 0.9f);
        std::vector<glm::vec3> cars;
        jeep_positions(cars);
        for (size_t i = 0; i < cars.size(); ++i)
            if (W.cars_prev.size() == cars.size()) {
                const glm::vec3 vel = (cars[i] - W.cars_prev[i]) / dt;
                if (glm::dot(vel, vel) < 80.0f * 80.0f) mover(cars[i], vel, 1.6f, cars[i].y - 0.4f);
            }
        W.cars_prev = cars;
        for (const npc::View& v : npc_views_) {
            if (v.hidden) continue;
            const bool car = v.kind == npc::Kind::Car;
            mover(v.pos, glm::vec3(std::sin(v.yaw), 0.0f, -std::cos(v.yaw)) * v.speed, car ? 1.6f : 0.35f, v.pos.y - (car ? 0.4f : 0.9f));
        }
        // new splash rings (bullets, bodies, droplets): one push down at the impact
        for (const auto& r : water_ripples_)
            if (r.age <= dt + 1e-4f) {
                const float s = glm::clamp(r.amp * 28.5f, 1.0f, 6.0f);
                add_imp(r.xz, 0.12f * std::sqrt(s) + 0.05f, -1.2f * r.amp);
            }
        // VIBE_TEST_RIPPLE=<k>: a made-up wader circling 6 m in front of the camera
        static const float test = std::getenv("VIBE_TEST_RIPPLE") ? float(std::atof(std::getenv("VIBE_TEST_RIPPLE"))) : 0.0f;
        if (test > 0.0f) {
            const glm::mat4& vm = current_frame_view_.view;
            glm::vec2 fw(-vm[0][2], -vm[2][2]);
            fw = glm::dot(fw, fw) > 1e-6f ? glm::normalize(fw) : glm::vec2(0.0f, -1.0f);
            const glm::vec2 c = glm::vec2(eye.x, eye.z) + fw * 6.0f;
            const float a = anim_time_ * 0.6f;
            const glm::vec2 p = c + 2.0f * glm::vec2(std::cos(a), std::sin(a));
            const glm::vec2 v = 1.2f * glm::vec2(-std::sin(a), std::cos(a));
            add_imp(p + glm::normalize(v) * 0.35f, 0.35f, 0.12f * test * 1.2f * dt);
            add_imp(p - glm::normalize(v) * 0.2f, 0.3f, -0.08f * test * 1.2f * dt);
            if (std::fmod(anim_time_, 1.5f) < dt) add_imp(c + glm::vec2(-3.0f, 1.0f), 0.2f, -0.04f * test);     // (and a stone dropped in every 1.5 s)
        }
    }
    // the current on a 16 x 16 grid of 4 m (the CPU's twin of the shader's; every 10th frame or when the window moved)
    const float fcell = 2.0f * half / float(kFlowN);
    if (W.flow.size() != size_t(kFlowN * kFlowN)) W.flow.assign(size_t(kFlowN * kFlowN), glm::vec4(0.0f));
    if (++W.flow_age > 10 || glm::distance(W.flow_org, W.org) > fcell) {
        W.flow_age = 0; W.flow_org = W.org;
        for (int j = 0; j < kFlowN; ++j)
            for (int i = 0; i < kFlowN; ++i) {
                const glm::vec2 p = W.org + (glm::vec2(float(i), float(j)) + 0.5f) * fcell;
                const glm::vec2 f = water_river_flow(p.x, p.y);
                W.flow[size_t(j * kFlowN + i)] = glm::vec4(f, 0.0f, 0.0f);
            }
    }
    for (int i = 0; i < kFlowN * kFlowN; i += 2)
        P.flow[i / 2] = glm::vec4(glm::vec2(W.flow[size_t(i)]), glm::vec2(W.flow[size_t(i + 1)]));
    P.rip_misc = glm::vec4(float(ni), 0.6f, 1.6f, fcell);
    P.flow_org = glm::vec4(W.flow_org, float(kFlowN), 1.0f);

    static const bool log_on = env_on("VIBE_LOG_WATER_SIM");
    if (log_on && (W.log_t -= std::max(dt, 1.0f / 60.0f)) <= 0.0f) {
        W.log_t = 1.0f;
        log::infof("[water-sim] wind %.1f m/s, gain %.4g, slope var %.4f, height sd %.3f m (bound %.2f), CPU fft height at eye %.3f, ripple window (%.1f, %.1f), impulses %d",
                   double(wnd.z), double(W.gain), double(W.var_slope), double(W.sig_h), double(water_fft_bound()),
                   double(water_fft_height(eye.x, eye.z)), double(W.org.x), double(W.org.y), ni);
    }

    // ---- the passes
    barrier_all(cmd);       // (last frame's readers of the pictures and the parameters are done)
    vkCmdUpdateBuffer(cmd, W.params, 0, sizeof(P), &P);
    // the home tile's ground for the banks (its view changes when the ground is baked again)
    if (W.set_hview[fset] != terrain_height_full_view_ && terrain_height_full_view_) {
        VkDescriptorImageInfo hi{ terrain_height_full_sampler_, terrain_height_full_view_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        VkWriteDescriptorSet w{};
        w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w.dstSet = W.set[fset]; w.dstBinding = 8; w.descriptorCount = 1;
        w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w.pImageInfo = &hi;
        vkUpdateDescriptorSets(device_, 1, &w, 0, nullptr);
        W.set_hview[fset] = terrain_height_full_view_;
    }
    if (!W.set_hview[fset]) return;
    barrier_all(cmd);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, W.pipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, W.layout, 0, 1, &W.set[fset], 0, nullptr);
    auto run = [&](int mode, int src, int first, glm::vec4 s, uint32_t gx, uint32_t gy, uint32_t gz) {
        Push p{ glm::ivec4(mode, src, first, 0), s };
        vkCmdPushConstants(cmd, W.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(p), &p);
        vkCmdDispatch(cmd, gx, gy, gz);
        barrier_all(cmd);
    };
    run(0, 0, 0, glm::vec4(0.0f), 1, kN, kCasc);         // spectrum
    run(1, 0, 0, glm::vec4(0.0f), kN, kCasc, 1);         // FFT along x
    run(2, 0, 0, glm::vec4(0.0f), kN, kCasc, 1);         // FFT along z -> displacement
    run(3, 0, 0, glm::vec4(0.0f), 1, kN, kCasc);         // slope, fold, height
    // the gradient's mips (the distance filter)
    for (uint32_t l = 1; l < kMips; ++l) {
        VkImageBlit b{};
        b.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, l - 1, 0, kCasc };
        b.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, l, 0, kCasc };
        b.srcOffsets[1] = { int32_t(kN >> (l - 1)), int32_t(kN >> (l - 1)), 1 };
        b.dstOffsets[1] = { int32_t(kN >> l), int32_t(kN >> l), 1 };
        vkCmdBlitImage(cmd, W.grad.image, VK_IMAGE_LAYOUT_GENERAL, W.grad.image, VK_IMAGE_LAYOUT_GENERAL, 1, &b, VK_FILTER_LINEAR);
        barrier_all(cmd);
    }
    // the read-back for the CPU twin
    {
        VkBufferImageCopy r{};
        r.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, kCasc };
        r.imageExtent = { kN, kN, 1 };
        vkCmdCopyImageToBuffer(cmd, W.disp.image, VK_IMAGE_LAYOUT_GENERAL, W.rb[slot], 1, &r);
        W.rb_frame[slot] = frame_number_; W.rb_valid[slot] = true;
    }
    // the ripple field: two steps, A -> B (scroll, impulses) and B -> A (the scene reads A)
    if (!no_rip) {
        const float sdt = std::max(dt * 0.5f, 1e-4f);
        run(4, 0, 1, glm::vec4(W.prev_org, sdt, 0.0f), kRipN / 64, kRipN, 1);
        run(4, 1, 0, glm::vec4(W.org, sdt, 0.0f), kRipN / 64, kRipN, 1);
    }
}

}  // namespace vibe
