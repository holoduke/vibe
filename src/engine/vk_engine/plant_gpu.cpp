// Small plants on the GPU path (see shaders/plant_cull.comp).
//
// scatter.cpp places ~260 000 flowers, ferns, reeds, tufts and lilies. They
// used to be drawn in 32 m cells chosen on the CPU: every plant of a visible
// cell went through the vertex shader, also the thinned-out, the off-screen
// and the hidden ones. Now a compute pass looks at every plant each frame
// and packs the ones that can be seen into one list per (species, level of
// detail); each list is one instanced indirect draw. scatter.vert is
// unchanged - the packed list is bound as its per-instance vertex buffer.
//
// The plants' shadow map (clutter_shadow.cpp) still uses the CPU cells: it
// is drawn from the light, which this camera-side selection knows nothing of.
#include "engine/vk_engine.h"
#include "engine/vk_engine/internal.h"
#include "engine/vk_engine/state/plant_gpu.h"
#include "engine/vk_engine/state/environment.h"
#include "engine/vk_pipelines.h"
#include "engine/vk_initializers.h"
#include "engine/log.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>

namespace vibe {

namespace {
constexpr int kLevels = 3;
// The kinds of grass tuft_place.comp places (the game: the objects of a placement group), and
// how many vec4 each has after the species table in the buffer at binding 3.
constexpr int kGrassKinds = 3;           // 0 = long tufts (species 15), 1 = turf (species 17), 2 = flowering meadow (species 18, in fields only)
constexpr int kGrassParams = 14;
constexpr uint32_t kTuftRoom = 60000u, kTurfRoom = 150000u, kMeadowRoom = 120000u;      // room in each list of the species
VkBuffer pg_buffer(VmaAllocator a, VkDeviceSize size, VkBufferUsageFlags usage, bool host, VmaAllocation& alloc, void** mapped) {
    VkBufferCreateInfo bci{};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = std::max<VkDeviceSize>(size, 64);
    bci.usage = usage;
    VmaAllocationCreateInfo aci{};
    aci.usage = host ? VMA_MEMORY_USAGE_AUTO_PREFER_HOST : VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    if (host) aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    VkBuffer b = VK_NULL_HANDLE;
    VmaAllocationInfo ai{};
    vk_check(vmaCreateBuffer(a, &bci, &aci, &b, &alloc, &ai), "plant gpu buffer");
    if (mapped) *mapped = ai.pMappedData;
    return b;
}
}  // namespace

// After build_scatter (also after a re-plant in the editor).
void VulkanEngine::init_plant_gpu() {
    destroy_plant_gpu();
    if (std::getenv("VIBE_NO_PLANT_GPU") || !scatter_.plant_buffer) return;
    uint32_t total = 0;
    for (int sp = 0; sp < ScatterSet::kPlantSpecies; ++sp) total += scatter_.plant_count[sp];
    // The automatic grass - long tufts and turf - is in no list: tuft_place.comp makes it every
    // frame, into the lists of species 15 and 17, which get room for it.
    const bool tufts_gpu = !scatter_.tuft_rules.empty() && !std::getenv("VIBE_NO_TUFTS_GPU");
    pg_->tuft_room = tufts_gpu ? kTuftRoom : 0u;
    pg_->turf_room = tufts_gpu && !std::getenv("VIBE_LAWN_RM") ? kTurfRoom : 0u;
    pg_->meadow_room = tufts_gpu && !std::getenv("VIBE_NO_MEADOW") ? kMeadowRoom : 0u;
    if (total == 0 && !tufts_gpu) return;
    pg_->total = total;
    log::infof("[plants] GPU culling: %u plants", total);

    // Buckets: (species, level) -> a range of the packed list big enough for
    // every plant of the species (near levels are capped: only so many plants
    // fit inside their distance).
    std::vector<VkDrawIndexedIndirectCommand> tmpl(size_t(ScatterSet::kPlantSpecies) * kLevels);
    uint32_t base = 0;
    for (int sp = 0; sp < ScatterSet::kPlantSpecies; ++sp) {
        const Mesh* lods[kLevels] = { &scatter_.plant_mesh[sp], &scatter_.plant_mesh_far[sp], &scatter_.plant_mesh_far2[sp] };
        if (!lods[1]->vertex_buffer) lods[1] = lods[0];
        if (!lods[2]->vertex_buffer) lods[2] = lods[1];
        for (int l = 0; l < kLevels; ++l) {
            VkDrawIndexedIndirectCommand& c = tmpl[size_t(sp) * kLevels + size_t(l)];
            c.indexCount = lods[l]->index_count;
            c.instanceCount = 0;
            c.firstIndex = 0;
            c.vertexOffset = 0;
            c.firstInstance = base;
            base += scatter_.plant_count[sp] + (sp == 15 ? pg_->tuft_room : sp == 17 ? pg_->turf_room : sp == 18 ? pg_->meadow_room : 0u);
        }
    }
    pg_->packed_cap = std::max(base, 1u);

    const VkBufferUsageFlags ssbo = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    pg_->packed_buf = pg_buffer(allocator_, VkDeviceSize(pg_->packed_cap) * sizeof(ScatterInstance),
                               ssbo | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, false, pg_->packed_alloc, nullptr);
    const VkDeviceSize cmd_bytes = tmpl.size() * sizeof(VkDrawIndexedIndirectCommand);
    pg_->cmd_buf = pg_buffer(allocator_, cmd_bytes, ssbo | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                            false, pg_->cmd_alloc, nullptr);
    void* mp = nullptr;
    pg_->tmpl_buf = pg_buffer(allocator_, cmd_bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, true, pg_->tmpl_alloc, &mp);
    if (mp) std::memcpy(mp, tmpl.data(), size_t(cmd_bytes));
    vmaFlushAllocation(allocator_, pg_->tmpl_alloc, 0, VK_WHOLE_SIZE);
    // (in GPU memory, refreshed each frame with vkCmdUpdateBuffer: the shader reads it per plant)
    pg_->spec_buf = pg_buffer(allocator_, sizeof(glm::vec4) * (ScatterSet::kPlantSpecies + kGrassKinds * kGrassParams), ssbo | VK_BUFFER_USAGE_TRANSFER_DST_BIT, false, pg_->spec_alloc, nullptr);
    pg_->spec_mapped = this;       // (only says: the path is up)
    pg_->inst_buf = pg_buffer(allocator_, VkDeviceSize(total) * sizeof(ScatterInstance), ssbo | VK_BUFFER_USAGE_TRANSFER_DST_BIT, false, pg_->inst_alloc, nullptr);
    pg_->inst_gen = 0;

    VkDescriptorSetLayoutBinding b[9]{};
    for (uint32_t i = 0; i < 9; ++i) {
        b[i].binding = i;
        b[i].descriptorType = i >= 4 ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        b[i].descriptorCount = 1;
        b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo lci{};
    lci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    lci.bindingCount = 9; lci.pBindings = b;
    vk_check(vkCreateDescriptorSetLayout(device_, &lci, nullptr, &pg_->set_layout), "plant gpu set layout");
    VkDescriptorPoolSize ps[2] = { { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4 }, { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 5 } };
    VkDescriptorPoolCreateInfo pci{};
    pci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pci.maxSets = 1; pci.poolSizeCount = 2; pci.pPoolSizes = ps;
    vk_check(vkCreateDescriptorPool(device_, &pci, nullptr, &pg_->pool), "plant gpu pool");
    VkDescriptorSetAllocateInfo dai{};
    dai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dai.descriptorPool = pg_->pool; dai.descriptorSetCount = 1; dai.pSetLayouts = &pg_->set_layout;
    vk_check(vkAllocateDescriptorSets(device_, &dai, &pg_->set), "plant gpu set");
    VkDescriptorBufferInfo bi[4] = {
        { pg_->inst_buf, 0, VK_WHOLE_SIZE }, { pg_->packed_buf, 0, VK_WHOLE_SIZE },
        { pg_->cmd_buf, 0, VK_WHOLE_SIZE }, { pg_->spec_buf, 0, VK_WHOLE_SIZE } };
    VkWriteDescriptorSet w[4]{};
    for (uint32_t i = 0; i < 4; ++i) {
        w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[i].dstSet = pg_->set; w[i].dstBinding = i; w[i].descriptorCount = 1;
        w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w[i].pBufferInfo = &bi[i];
    }
    vkUpdateDescriptorSets(device_, 4, w, 0, nullptr);
    pg_->hiz_bound = false;
    write_plant_hiz_binding();

    VkPushConstantRange pcr{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(glm::vec4) * 16 };
    VkPipelineLayoutCreateInfo pli{};
    pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pli.setLayoutCount = 1; pli.pSetLayouts = &pg_->set_layout;
    pli.pushConstantRangeCount = 1; pli.pPushConstantRanges = &pcr;
    vk_check(vkCreatePipelineLayout(device_, &pli, nullptr, &pg_->layout), "plant gpu layout");
    const std::string sd = VIBE_SHADER_DIR;
    pg_->module = vkpipe::load_shader_module(device_, sd + "/plant_cull.comp.spv");
    VkComputePipelineCreateInfo cci{};
    cci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cci.stage.module = pg_->module;
    cci.stage.pName = "main";
    cci.layout = pg_->layout;
    vk_check(vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &cci, nullptr, &pg_->pipeline), "plant cull pipeline");
    log::infof("[plants] GPU path: %u plants, %zu buckets, packed list %.1f MB", pg_->total, tmpl.size(),
               double(pg_->packed_cap) * sizeof(ScatterInstance) / 1048576.0);

    // ---- The automatic grass: the rules picture, the ground it stands on, the placement pipeline ----
    if (tufts_gpu) {
        const uint32_t N = uint32_t(ScatterSet::kTuftCapN);
        VkSamplerCreateInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        si.magFilter = VK_FILTER_NEAREST; si.minFilter = VK_FILTER_NEAREST;
        si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        vk_check(vkCreateSampler(device_, &si, nullptr, &pg_->tuft_sampler), "tuft rules sampler");
        VkImageCreateInfo ici{};
        ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ici.imageType = VK_IMAGE_TYPE_2D;
        ici.format = VK_FORMAT_R8G8B8A8_UNORM;
        ici.extent = { N, N, 1 };
        ici.mipLevels = 1; ici.arrayLayers = 1;
        ici.samples = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling = VK_IMAGE_TILING_OPTIMAL;
        ici.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        vk_check(vmaCreateImage(allocator_, &ici, &aci, &pg_->tuft_img, &pg_->tuft_img_alloc, nullptr), "tuft rules image");
        VkImageViewCreateInfo vci{};
        vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vci.image = pg_->tuft_img;
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.format = VK_FORMAT_R8G8B8A8_UNORM;
        vci.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        vk_check(vkCreateImageView(device_, &vci, nullptr, &pg_->tuft_view), "tuft rules view");
        pg_->ground_view = VK_NULL_HANDLE; pg_->lawn_view = VK_NULL_HANDLE; pg_->near_view = VK_NULL_HANDLE;
        write_tuft_ground_bindings();
        pg_->tuft_gen = 0;
        upload_tuft_ground(true);
        pg_->tuft_module = vkpipe::load_shader_module(device_, sd + "/tuft_place.comp.spv");
        if (pg_->tuft_module) {
            VkComputePipelineCreateInfo tci = cci;
            tci.stage.module = pg_->tuft_module;
            vk_check(vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &tci, nullptr, &pg_->tuft_pipeline), "tuft place pipeline");
        }
        if (!pg_->tuft_pipeline) log::warnf("[plants] tuft_place shader missing - no automatic grass");
        else log::infof("[plants] grass placed on the GPU: room for %u long tufts and %u turf clumps a list%s",
                        pg_->tuft_room, pg_->turf_room, pg_->turf_room ? " (the turf replaces the raymarched lawn)" : "");
    }
}

// The pictures tuft_place.comp reads the ground from: 5 = the rules (ours), 6 = the terrain's
// height map, 7 = the grass mask of the lawn, 8 = the ground round the camera beyond the home tile
// (veg_stream.cpp: until its first bake a placeholder - the shader looks for the mark of the bake). Written again when the engine has made one of them anew.
void VulkanEngine::write_tuft_ground_bindings() {
    if (!pg_->set || !pg_->tuft_view || !terrain_height_full_view_ || !grass_mask_.view) return;
    const VkImageView near_v = normal_textures_[kTexVegShade].view ? normal_textures_[kTexVegShade].view : grass_mask_.view;
    if (pg_->ground_view == terrain_height_full_view_ && pg_->lawn_view == grass_mask_.view && pg_->near_view == near_v) return;
    VkDescriptorImageInfo ii[4] = {
        { pg_->tuft_sampler, pg_->tuft_view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL },
        { terrain_height_full_sampler_, terrain_height_full_view_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL },
        { terrain_height_full_sampler_, grass_mask_.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL },
        { texture_sampler_, near_v, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL } };
    VkWriteDescriptorSet w[4]{};
    for (uint32_t i = 0; i < 4; ++i) {
        w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[i].dstSet = pg_->set; w[i].dstBinding = 5 + i; w[i].descriptorCount = 1;
        w[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        w[i].pImageInfo = &ii[i];
    }
    vkUpdateDescriptorSets(device_, 4, w, 0, nullptr);
    pg_->ground_view = terrain_height_full_view_; pg_->lawn_view = grass_mask_.view; pg_->near_view = near_v;
}

// Is the ground's short grass drawn as turf clumps this frame (and the raymarched lawn left out)?
bool VulkanEngine::turf_on() const {
    static const bool off = std::getenv("VIBE_NO_TUFTS") != nullptr;
    return pg_->tuft_pipeline && pg_->ground_view && pg_->turf_room > 0 && pg_->drawn && !off && globe::current().home;
}

// The rules picture the placement reads, sent to the GPU: when it is made and when a crater
// has changed it. (A blocking upload of 4 MB: neither happens in passing.)
void VulkanEngine::upload_tuft_ground(bool first) {
    if (!pg_->tuft_img || scatter_.tuft_rules.empty() || pg_->tuft_gen == scatter_.tuft_cap_gen) return;
    const uint32_t N = uint32_t(ScatterSet::kTuftCapN);
    const VkDeviceSize bytes = VkDeviceSize(scatter_.tuft_rules.size());
    VmaAllocation sa = nullptr;
    void* mp = nullptr;
    VkBuffer stage = pg_buffer(allocator_, bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, true, sa, &mp);
    if (!mp) { vmaDestroyBuffer(allocator_, stage, sa); return; }
    std::memcpy(mp, scatter_.tuft_rules.data(), size_t(bytes));
    vmaFlushAllocation(allocator_, sa, 0, VK_WHOLE_SIZE);
    vkinit::one_time_submit(device_, graphics_queue_, graphics_queue_family_, [&](VkCommandBuffer cb) {
        vkinit::transition_image(cb, pg_->tuft_img, first ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        VkBufferImageCopy r{};
        r.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
        r.imageExtent = { N, N, 1 };
        vkCmdCopyBufferToImage(cb, stage, pg_->tuft_img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &r);
        vkinit::transition_image(cb, pg_->tuft_img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    });
    vmaDestroyBuffer(allocator_, stage, sa);
    pg_->tuft_gen = scatter_.tuft_cap_gen;
}

void VulkanEngine::write_plant_hiz_binding() {
    if (!pg_->set || !veg_hiz_view_ || !veg_hiz_sampler_) return;
    VkDescriptorImageInfo ii{ veg_hiz_sampler_, veg_hiz_view_, VK_IMAGE_LAYOUT_GENERAL };
    VkWriteDescriptorSet w{};
    w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w.dstSet = pg_->set; w.dstBinding = 4; w.descriptorCount = 1;
    w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    w.pImageInfo = &ii;
    vkUpdateDescriptorSets(device_, 1, &w, 0, nullptr);
    pg_->hiz_bound = true;
}

void VulkanEngine::destroy_plant_gpu() {
    pg_->drawn = false;
    if (pg_->pipeline) vkDestroyPipeline(device_, pg_->pipeline, nullptr);
    if (pg_->layout) vkDestroyPipelineLayout(device_, pg_->layout, nullptr);
    if (pg_->module) vkDestroyShaderModule(device_, pg_->module, nullptr);
    if (pg_->pool) vkDestroyDescriptorPool(device_, pg_->pool, nullptr);
    if (pg_->set_layout) vkDestroyDescriptorSetLayout(device_, pg_->set_layout, nullptr);
    if (pg_->packed_buf) vmaDestroyBuffer(allocator_, pg_->packed_buf, pg_->packed_alloc);
    if (pg_->cmd_buf) vmaDestroyBuffer(allocator_, pg_->cmd_buf, pg_->cmd_alloc);
    if (pg_->tmpl_buf) vmaDestroyBuffer(allocator_, pg_->tmpl_buf, pg_->tmpl_alloc);
    if (pg_->spec_buf) vmaDestroyBuffer(allocator_, pg_->spec_buf, pg_->spec_alloc);
    if (pg_->inst_buf) vmaDestroyBuffer(allocator_, pg_->inst_buf, pg_->inst_alloc);
    if (pg_->tuft_pipeline) vkDestroyPipeline(device_, pg_->tuft_pipeline, nullptr);
    if (pg_->tuft_module) vkDestroyShaderModule(device_, pg_->tuft_module, nullptr);
    if (pg_->tuft_sampler) vkDestroySampler(device_, pg_->tuft_sampler, nullptr);
    if (pg_->tuft_view) vkDestroyImageView(device_, pg_->tuft_view, nullptr);
    if (pg_->tuft_img) vmaDestroyImage(allocator_, pg_->tuft_img, pg_->tuft_img_alloc);
    pg_->tuft_view = VK_NULL_HANDLE; pg_->tuft_img = VK_NULL_HANDLE; pg_->tuft_img_alloc = nullptr;
    pg_->tuft_pipeline = VK_NULL_HANDLE; pg_->tuft_module = VK_NULL_HANDLE; pg_->tuft_sampler = VK_NULL_HANDLE;
    pg_->tuft_room = 0; pg_->turf_room = 0; pg_->tuft_gen = 0;
    pg_->ground_view = VK_NULL_HANDLE; pg_->lawn_view = VK_NULL_HANDLE; pg_->near_view = VK_NULL_HANDLE; pg_->meadow_room = 0;
    pg_->inst_buf = VK_NULL_HANDLE;
    pg_->pipeline = VK_NULL_HANDLE; pg_->layout = VK_NULL_HANDLE; pg_->module = VK_NULL_HANDLE; pg_->pool = VK_NULL_HANDLE;
    pg_->set_layout = VK_NULL_HANDLE; pg_->set = VK_NULL_HANDLE;
    pg_->packed_buf = pg_->cmd_buf = pg_->tmpl_buf = pg_->spec_buf = VK_NULL_HANDLE;
    pg_->spec_mapped = nullptr; pg_->total = 0; pg_->hiz_bound = false;
}

// Each frame, after cull_vegetation (it decides whether the depth pyramid may be used).
void VulkanEngine::cull_plants(VkCommandBuffer cmd) {
    pg_->drawn = false;
    if (!pg_->pipeline || !pg_->hiz_bound || (pg_->total == 0 && !pg_->tuft_pipeline) || !pg_->spec_mapped) return;
    upload_tuft_ground(false);       // (a crater has changed where long grass may grow)
    write_tuft_ground_bindings();    // (the height map or the lawn's mask has been made anew)
    const FrameView& fv = current_frame_view_;
    static const bool tufts_off = std::getenv("VIBE_NO_TUFTS") != nullptr;
    const bool place = pg_->tuft_pipeline && pg_->ground_view && !tufts_off && globe::current().home;      // (bound: the height map and the lawn mask exist)
    // Per species: where its plants lie and how far it is drawn (a setting). After them: the
    // numbers of each kind of grass tuft_place.comp places.
    uint32_t grass_pts[kGrassKinds] = {};
    {
        glm::vec4 spec[ScatterSet::kPlantSpecies + kGrassKinds * kGrassParams] = {};
        for (int sp = 0; sp < ScatterSet::kPlantSpecies; ++sp)
            spec[sp] = glm::vec4(float(scatter_.plant_first[sp]), float(scatter_.plant_count[sp]), plant_max_d(sp), 0.0f);
        // The numbers the game keeps per kind of plant and per quality setting (they are not in
        // its shaders): chosen here.
        struct Kind {
            int species; float room, range, candidates;      // candidates a square metre, before any rejection
            float density, noise, cluster, slope;            // keeps 1 - 1/(10 d + 1); clustering noise (1/m), cluster size; steepest (deg)
            float size0, size1, grow, share, band, fade0, fade1;           // sizes; growth with distance; share that keeps its size to the end, and its dissolve band (the game: neither for placed grass - all of it shrinks away)
            const float* shape; int lawn; glm::vec3 c0, c1, c2;     // lawn: 0 = by the rules picture, 1 = where the lawn grows, 2 = ... and only in a field (flower_fields_)
        };
        const glm::vec3 top = rt_.grass_color_top, bot = rt_.grass_color_bottom, straw(0.34f, 0.31f, 0.13f);
        const Kind kinds[kGrassKinds] = {
            // long tufts: patches some 15 m across with bare ground between, a third of the candidates
            { 15, float(pg_->tuft_room), std::min(plant_max_d(15), 300.0f), 3.2f, 0.05f, 0.12f, 0.15f, 19.8f,
              0.45f, 1.5f, 0.12f, 0.0f, 0.0f, 0.5f, 1.0f, ScatterSet::kTuftShape, 0,
              glm::vec3(0.085f, 0.20f, 0.045f), glm::vec3(0.20f, 0.30f, 0.07f), straw },
            // turf: everywhere the lawn's mask has grass, every candidate, clumps growing with
            // distance as their number thins
            { 17, float(pg_->turf_room), std::clamp(rt_.grass_raymarch_distance, 24.0f, 75.0f), 9.0f * std::clamp(rt_.grass_density, 0.4f, 1.6f),
              12.0f, 0.0f, 0.0f, 38.0f,
              0.65f, 1.05f, 0.45f, 0.0f, 0.0f, 0.5f, 1.0f, ScatterSet::kTurfShape, 1,
              glm::mix(bot, top, 0.8f), top * 1.15f, glm::mix(top, straw, 0.6f) },
            // flowering meadow: only in the fields (flower_fields_), dense, seen from far - close by
            // every clump, from an eighth of the range on a growing share shrinks away and the rest
            // (a fifth) grows to keep the field closed, dissolving at the very end
            { 18, float(pg_->meadow_room), 230.0f, 7.0f * std::clamp(rt_.grass_density, 0.4f, 1.6f),
              12.0f, 0.0f, 0.0f, 30.0f,
              0.75f, 1.25f, 0.55f, 0.20f, 0.10f, 0.12f, 0.50f, ScatterSet::kMeadowShape, 2,
              glm::vec3(0.16f, 0.27f, 0.05f), glm::vec3(0.26f, 0.34f, 0.07f), straw },
        };
        for (int k = 0; k < kGrassKinds; ++k) {
            const Kind& K = kinds[k];
            if (!place || K.room <= 0.0f) continue;
            if (K.lawn == 2) {          // (a kind of the fields: not dispatched at all away from them)
                bool near_field = false;
                for (const glm::vec4& F : flower_fields_) near_field |= glm::distance(glm::vec2(fv.eye_pos.x, fv.eye_pos.z), glm::vec2(F.x, F.y)) < F.z + F.w + K.range;
                if (!near_field) continue;
            }
            const float cs = std::max(K.range / 3.0f, 4.0f);                     // half the side of a cell: the range is three of them
            grass_pts[k] = uint32_t(std::clamp(K.candidates * 4.0f * cs * cs / 5.0f, 64.0f, 40000.0f));
            glm::vec4* P = &spec[ScatterSet::kPlantSpecies + k * kGrassParams];
            P[0] = glm::vec4(cs, std::floor(fv.eye_pos.x / (2.0f * cs) + 0.5f), std::floor(fv.eye_pos.z / (2.0f * cs) + 0.5f), float(grass_pts[k]));
            P[1] = glm::vec4(K.fade0 * K.range, K.fade1 * K.range, K.share, float(k) * 977.0f);
            P[2] = glm::vec4(K.density, K.noise, K.cluster, K.slope);
            P[3] = glm::vec4(K.size0, K.size1, K.grow, 1.0f);
            P[4] = glm::vec4(scatter_.tuft_cap_ext, float(K.species), K.room, float(K.lawn));
            P[5] = glm::vec4(K.shape[0], 2.0f * K.shape[2] + 0.5f * K.shape[0], K.range, K.band);
            P[6] = glm::vec4(K.c0, 0.0f); P[7] = glm::vec4(K.c1, 0.0f); P[8] = glm::vec4(K.c2, 0.0f);
            for (size_t i = 0; i < 4; ++i) P[9 + i] = K.lawn == 2 && i < flower_fields_.size() ? flower_fields_[i] : glm::vec4(0.0f);
            P[13] = glm::vec4(K.fade0, K.fade1, glm::clamp(env_->snow, 0.0f, 1.0f), 0.0f);     // (.z: winter's snow cover - short grass is under it)
        }
        vkCmdUpdateBuffer(cmd, pg_->spec_buf, 0, sizeof(spec), spec);
    }
    // The plants themselves, when the CPU has rewritten them (a crater, the surface they stand on).
    if (pg_->total > 0 && pg_->inst_gen != scatter_.plant_gen) {
        VkBufferCopy ic{ 0, 0, VkDeviceSize(pg_->total) * sizeof(ScatterInstance) };
        vkCmdCopyBuffer(cmd, scatter_.plant_buffer, pg_->inst_buf, 1, &ic);
        pg_->inst_gen = scatter_.plant_gen;
    }
    // Counts back to zero: the command template over the command buffer.
    VkBufferCopy cp{ 0, 0, VkDeviceSize(ScatterSet::kPlantSpecies) * kLevels * sizeof(VkDrawIndexedIndirectCommand) };
    vkCmdCopyBuffer(cmd, pg_->tmpl_buf, pg_->cmd_buf, 1, &cp);
    VkMemoryBarrier2 mb{};
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
    mb.srcStageMask  = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT | VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT | VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT;
    mb.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;
    mb.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    mb.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    VkDependencyInfo dep{};
    dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.memoryBarrierCount = 1; dep.pMemoryBarriers = &mb;
    vkCmdPipelineBarrier2(cmd, &dep);

    struct { glm::vec4 planes[6]; glm::vec4 cam; glm::vec4 params; glm::mat4 prev_vp; glm::vec4 hiz; } pc{};
    for (int i = 0; i < 6; ++i) pc.planes[i] = fv.frustum.planes[i];
    const float ppr = float(swapchain_extent_.height) / (2.0f * std::tan(glm::radians(80.0f) * 0.5f));
    pc.cam = glm::vec4(fv.eye_pos, ppr);
    const bool occl = veg_cull_hiz_on_ && !std::getenv("VIBE_NO_PLANT_OCCL");
    pc.params = glm::vec4(float(pg_->total), float(ScatterSet::kPlantSpecies), occl ? 1.0f : 0.0f, ScatterSet::kPlantFarLod);
    pc.prev_vp = veg_cull_prev_vp_;
    pc.hiz = glm::vec4(float(veg_hiz_w_), float(veg_hiz_h_), float(veg_hiz_mips_), ScatterSet::kPlantFarLod2);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pg_->pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pg_->layout, 0, 1, &pg_->set, 0, nullptr);
    vkCmdPushConstants(cmd, pg_->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    if (pg_->total > 0) vkCmdDispatch(cmd, (pg_->total + 127) / 128, 1, 1);

    // ---- The automatic grass: placed now, into the same lists (tuft_place.comp) ----
    if (place) {
        VkMemoryBarrier2 cb{};
        cb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        cb.srcStageMask = cb.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        cb.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        cb.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        VkDependencyInfo cdep{};
        cdep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        cdep.memoryBarrierCount = 1; cdep.pMemoryBarriers = &cb;
        vkCmdPipelineBarrier2(cmd, &cdep);
        struct { glm::vec4 planes[6]; glm::vec4 cam; glm::mat4 prev_vp; glm::vec4 misc; } tp{};
        for (int i = 0; i < 6; ++i) tp.planes[i] = fv.frustum.planes[i];
        tp.cam = glm::vec4(fv.eye_pos, occl ? 1.0f : 0.0f);
        tp.prev_vp = veg_cull_prev_vp_;
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pg_->tuft_pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pg_->layout, 0, 1, &pg_->set, 0, nullptr);
        for (int k = 0; k < kGrassKinds; ++k) {
            if (grass_pts[k] == 0) continue;
            tp.misc = glm::vec4(float(ScatterSet::kPlantSpecies + k * kGrassParams), rt_.terrain_raymarch_enabled ? 1.0f : 0.0f, rt_.water_level, 1.0f);
            vkCmdPushConstants(cmd, pg_->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(tp), &tp);
            vkCmdDispatch(cmd, (grass_pts[k] * 80u + 127u) / 128u, 1, 1);
        }
    }

    mb.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    mb.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    mb.dstStageMask  = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT;
    mb.dstAccessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT;
    vkCmdPipelineBarrier2(cmd, &dep);
    pg_->drawn = true;
}

// One species, with the plant pipeline and its push constants already set.
// Returns false when the GPU list is not available this frame (CPU cells then).
bool VulkanEngine::draw_plants_gpu(VkCommandBuffer cmd, int sp) {
    if (!pg_->drawn) return false;
    // VIBE_SKIP_SP=<bit mask>: leave species out (to see which plant is which).
    { static const char* sk = std::getenv("VIBE_SKIP_SP"); if (sk && ((std::atoi(sk) >> sp) & 1)) return true; }
    const Mesh* lods[kLevels] = { &scatter_.plant_mesh[sp], &scatter_.plant_mesh_far[sp], &scatter_.plant_mesh_far2[sp] };
    if (!lods[1]->vertex_buffer) lods[1] = lods[0];
    if (!lods[2]->vertex_buffer) lods[2] = lods[1];
    for (int l = 0; l < kLevels; ++l) {
        VkBuffer vb[2] = { lods[l]->vertex_buffer, pg_->packed_buf };
        VkDeviceSize off[2] = { 0, 0 };
        vkCmdBindVertexBuffers(cmd, 0, 2, vb, off);
        vkCmdBindIndexBuffer(cmd, lods[l]->index_buffer, 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexedIndirect(cmd, pg_->cmd_buf, VkDeviceSize(sp * kLevels + l) * sizeof(VkDrawIndexedIndirectCommand), 1,
                                 sizeof(VkDrawIndexedIndirectCommand));
    }
    return true;
}

}  // namespace vibe
