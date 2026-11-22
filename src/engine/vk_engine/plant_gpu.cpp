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
#include "engine/vk_pipelines.h"
#include "engine/log.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>

namespace vibe {

namespace {
constexpr int kLevels = 3;
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
    if (total == 0) return;
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
            base += scatter_.plant_count[sp];
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
    pg_->spec_buf = pg_buffer(allocator_, sizeof(glm::vec4) * ScatterSet::kPlantSpecies, ssbo | VK_BUFFER_USAGE_TRANSFER_DST_BIT, false, pg_->spec_alloc, nullptr);
    pg_->spec_mapped = this;       // (only says: the path is up)
    pg_->inst_buf = pg_buffer(allocator_, VkDeviceSize(total) * sizeof(ScatterInstance), ssbo | VK_BUFFER_USAGE_TRANSFER_DST_BIT, false, pg_->inst_alloc, nullptr);
    pg_->inst_gen = 0;

    VkDescriptorSetLayoutBinding b[5]{};
    for (uint32_t i = 0; i < 5; ++i) {
        b[i].binding = i;
        b[i].descriptorType = i == 4 ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        b[i].descriptorCount = 1;
        b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo lci{};
    lci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    lci.bindingCount = 5; lci.pBindings = b;
    vk_check(vkCreateDescriptorSetLayout(device_, &lci, nullptr, &pg_->set_layout), "plant gpu set layout");
    VkDescriptorPoolSize ps[2] = { { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4 }, { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1 } };
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

    VkPushConstantRange pcr{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(glm::vec4) * 13 };
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
    pg_->inst_buf = VK_NULL_HANDLE;
    pg_->pipeline = VK_NULL_HANDLE; pg_->layout = VK_NULL_HANDLE; pg_->module = VK_NULL_HANDLE; pg_->pool = VK_NULL_HANDLE;
    pg_->set_layout = VK_NULL_HANDLE; pg_->set = VK_NULL_HANDLE;
    pg_->packed_buf = pg_->cmd_buf = pg_->tmpl_buf = pg_->spec_buf = VK_NULL_HANDLE;
    pg_->spec_mapped = nullptr; pg_->total = 0; pg_->hiz_bound = false;
}

// Each frame, after cull_vegetation (it decides whether the depth pyramid may be used).
void VulkanEngine::cull_plants(VkCommandBuffer cmd) {
    pg_->drawn = false;
    if (!pg_->pipeline || !pg_->hiz_bound || pg_->total == 0 || !pg_->spec_mapped) return;
    // Per species: where its plants lie and how far it is drawn (a setting).
    {
        glm::vec4 spec[ScatterSet::kPlantSpecies];
        for (int sp = 0; sp < ScatterSet::kPlantSpecies; ++sp)
            spec[sp] = glm::vec4(float(scatter_.plant_first[sp]), float(scatter_.plant_count[sp]), plant_max_d(sp), 0.0f);
        vkCmdUpdateBuffer(cmd, pg_->spec_buf, 0, sizeof(spec), spec);
    }
    // The plants themselves, when the CPU has rewritten them (a crater, the surface they stand on).
    if (pg_->inst_gen != scatter_.plant_gen) {
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

    const FrameView& fv = current_frame_view_;
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
    vkCmdDispatch(cmd, (pg_->total + 127) / 128, 1, 1);

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
