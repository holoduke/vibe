// Cluster LOD, renderer side (see engine/cluster_lod.h for the idea).
//
// The clustered meshes are the rock archetypes of the vegetation system
// (vegetation.cpp builds their hierarchies and sorts their instances to
// the end of the instance buffer). Every frame cluster_cull.comp picks the
// clusters to draw for each visible rock and writes one draw command per
// cluster; draw_clusters() issues them with one indirect-count draw, with
// the vegetation pipelines and veg.vert (the instance is found through a
// slot in the vegetation "visible" list reserved for each clustered rock).
#include "engine/vk_engine.h"
#include "engine/vk_engine/internal.h"
#include "engine/vk_pipelines.h"
#include "engine/log.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>

namespace qlike {

namespace {
VkBuffer cl_buffer(VmaAllocator a, VkDeviceSize size, VkBufferUsageFlags usage, VmaAllocation& alloc,
                   const void* data, VkDeviceSize data_size) {
    VkBufferCreateInfo bci{};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = std::max<VkDeviceSize>(size, 16);
    bci.usage = usage;
    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO;
    if (data) aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    VkBuffer b = VK_NULL_HANDLE;
    VmaAllocationInfo ai{};
    vk_check(vmaCreateBuffer(a, &bci, &aci, &b, &alloc, &ai), "cluster buffer");
    if (data && ai.pMappedData) {
        std::memcpy(ai.pMappedData, data, size_t(data_size));
        vmaFlushAllocation(a, alloc, 0, VK_WHOLE_SIZE);
    }
    return b;
}
}  // namespace

// After init_vegetation_data: veg_cluster_meshes_ built, clustered instances
// are [veg_cluster_first_, +veg_cluster_count_) and own the visible slots
// from cl_vis_base_.
void VulkanEngine::init_cluster_draw() {
    if (veg_cluster_count_ == 0 || !veg_inst_buf_) return;
    // One vertex buffer (the finest mesh of every clustered archetype), one
    // index buffer (all clusters), one cluster table.
    std::vector<Vertex> verts;
    std::vector<uint32_t> idx;
    std::vector<ClusterGpu> clusters;
    std::vector<glm::uvec4> table(veg_arch_.size(), glm::uvec4(0u));
    for (size_t a = 0; a < veg_arch_.size(); ++a) {
        const ClusterMesh& cm = veg_cluster_meshes_[a];
        if (cm.clusters.empty()) continue;
        const auto& src = veg_arch_[a].lods[0];
        table[a] = glm::uvec4(uint32_t(clusters.size()), uint32_t(cm.clusters.size()), uint32_t(idx.size()), uint32_t(verts.size()));
        for (const auto& v : src.verts) verts.push_back({ v.p, v.n, glm::vec2(v.u, v.v) });
        idx.insert(idx.end(), cm.indices.begin(), cm.indices.end());
        clusters.insert(clusters.end(), cm.clusters.begin(), cm.clusters.end());
    }
    if (clusters.empty()) { veg_cluster_count_ = 0; return; }
    cl_mesh_ = create_mesh_from_data(device_, allocator_, graphics_queue_, graphics_queue_family_,
                                     verts.data(), uint32_t(verts.size()), idx.data(), uint32_t(idx.size()));
    const VkBufferUsageFlags ssbo = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    cl_cluster_buf_ = cl_buffer(allocator_, sizeof(ClusterGpu) * clusters.size(), ssbo, cl_cluster_alloc_,
                                clusters.data(), sizeof(ClusterGpu) * clusters.size());
    cl_table_buf_ = cl_buffer(allocator_, sizeof(glm::uvec4) * table.size(), ssbo, cl_table_alloc_,
                              table.data(), sizeof(glm::uvec4) * table.size());
    cl_cmd_cap_ = 32768;
    cl_cmd_buf_ = cl_buffer(allocator_, 16 + VkDeviceSize(cl_cmd_cap_) * sizeof(VkDrawIndexedIndirectCommand),
                            ssbo | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                            cl_cmd_alloc_, nullptr, 0);

    // Descriptor set: instances, visible, fade, arch info, clusters, table, commands.
    VkDescriptorSetLayoutBinding b[8]{};
    for (uint32_t i = 0; i < 8; ++i) {
        b[i].binding = i;
        b[i].descriptorType = i == 7 ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        b[i].descriptorCount = 1;
        b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo lci{};
    lci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    lci.bindingCount = 8; lci.pBindings = b;
    vk_check(vkCreateDescriptorSetLayout(device_, &lci, nullptr, &cl_set_layout_), "cluster set layout");
    VkDescriptorPoolSize ps[2] = { { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 7 }, { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1 } };
    VkDescriptorPoolCreateInfo pci{};
    pci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pci.maxSets = 1; pci.poolSizeCount = 2; pci.pPoolSizes = ps;
    vk_check(vkCreateDescriptorPool(device_, &pci, nullptr, &cl_pool_), "cluster pool");
    VkDescriptorSetAllocateInfo dai{};
    dai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dai.descriptorPool = cl_pool_; dai.descriptorSetCount = 1; dai.pSetLayouts = &cl_set_layout_;
    vk_check(vkAllocateDescriptorSets(device_, &dai, &cl_set_), "cluster set");
    VkDescriptorBufferInfo bi[7] = {
        { veg_inst_buf_, 0, VK_WHOLE_SIZE }, { veg_vis_buf_, 0, VK_WHOLE_SIZE }, { veg_fade_buf_, 0, VK_WHOLE_SIZE },
        { veg_arch_buf_, 0, VK_WHOLE_SIZE }, { cl_cluster_buf_, 0, VK_WHOLE_SIZE }, { cl_table_buf_, 0, VK_WHOLE_SIZE },
        { cl_cmd_buf_, 0, VK_WHOLE_SIZE } };
    VkWriteDescriptorSet w[7]{};
    for (uint32_t i = 0; i < 7; ++i) {
        w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[i].dstSet = cl_set_; w[i].dstBinding = i; w[i].descriptorCount = 1;
        w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w[i].pBufferInfo = &bi[i];
    }
    vkUpdateDescriptorSets(device_, 7, w, 0, nullptr);
    write_cluster_hiz_binding();

    VkPushConstantRange pcr{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(glm::vec4) * 14 };
    VkPipelineLayoutCreateInfo pli{};
    pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pli.setLayoutCount = 1; pli.pSetLayouts = &cl_set_layout_;
    pli.pushConstantRangeCount = 1; pli.pPushConstantRanges = &pcr;
    vk_check(vkCreatePipelineLayout(device_, &pli, nullptr, &cl_layout_), "cluster layout");
    const std::string sd = QLIKE_SHADER_DIR;
    cl_module_ = vkpipe::load_shader_module(device_, sd + "/cluster_cull.comp.spv");
    VkComputePipelineCreateInfo cci{};
    cci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cci.stage.module = cl_module_;
    cci.stage.pName = "main";
    cci.layout = cl_layout_;
    vk_check(vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &cci, nullptr, &cl_pipeline_), "cluster cull pipeline");
    log::infof("[cluster] drawing %u rocks as clusters: %zu clusters, %zu triangles in all levels",
               veg_cluster_count_, clusters.size(), idx.size() / 3);
}

// The depth pyramid (re-made when the window size changes: vegetation.cpp calls this again).
void VulkanEngine::write_cluster_hiz_binding() {
    if (!cl_set_ || !veg_hiz_view_ || !veg_hiz_sampler_) return;
    VkDescriptorImageInfo ii{ veg_hiz_sampler_, veg_hiz_view_, VK_IMAGE_LAYOUT_GENERAL };
    VkWriteDescriptorSet w{};
    w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w.dstSet = cl_set_; w.dstBinding = 7; w.descriptorCount = 1;
    w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    w.pImageInfo = &ii;
    vkUpdateDescriptorSets(device_, 1, &w, 0, nullptr);
}

void VulkanEngine::destroy_cluster_draw() {
    if (cl_pipeline_) vkDestroyPipeline(device_, cl_pipeline_, nullptr);
    if (cl_layout_) vkDestroyPipelineLayout(device_, cl_layout_, nullptr);
    if (cl_module_) vkDestroyShaderModule(device_, cl_module_, nullptr);
    if (cl_pool_) vkDestroyDescriptorPool(device_, cl_pool_, nullptr);
    if (cl_set_layout_) vkDestroyDescriptorSetLayout(device_, cl_set_layout_, nullptr);
    if (cl_cluster_buf_) vmaDestroyBuffer(allocator_, cl_cluster_buf_, cl_cluster_alloc_);
    if (cl_table_buf_) vmaDestroyBuffer(allocator_, cl_table_buf_, cl_table_alloc_);
    if (cl_cmd_buf_) vmaDestroyBuffer(allocator_, cl_cmd_buf_, cl_cmd_alloc_);
    if (cl_mesh_.vertex_buffer) destroy_mesh(allocator_, cl_mesh_);
    cl_pipeline_ = VK_NULL_HANDLE; cl_layout_ = VK_NULL_HANDLE; cl_module_ = VK_NULL_HANDLE; cl_pool_ = VK_NULL_HANDLE;
    cl_set_layout_ = VK_NULL_HANDLE; cl_cluster_buf_ = cl_table_buf_ = cl_cmd_buf_ = VK_NULL_HANDLE;
    cl_set_ = VK_NULL_HANDLE;
}

// Each frame, after cull_vegetation (same place in the frame).
void VulkanEngine::cull_clusters(VkCommandBuffer cmd) {
    if (!cl_pipeline_ || veg_cluster_count_ == 0) return;
    const FrameView& fv = current_frame_view_;
    // Count back to zero, then the selection.
    vkCmdFillBuffer(cmd, cl_cmd_buf_, 0, 16, 0u);
    VkMemoryBarrier2 mb{};
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
    mb.srcStageMask  = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    mb.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    mb.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    mb.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    VkDependencyInfo dep{};
    dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.memoryBarrierCount = 1; dep.pMemoryBarriers = &mb;
    vkCmdPipelineBarrier2(cmd, &dep);

    struct { glm::vec4 planes[6]; glm::vec4 cam; glm::uvec4 range; glm::vec4 params; glm::mat4 prev_vp; glm::vec4 hiz; } pc{};
    for (int i = 0; i < 6; ++i) pc.planes[i] = fv.frustum.planes[i];
    // Detail for the OUTPUT resolution, like the vegetation.
    const float ppr = float(swapchain_extent_.height) / (2.0f * std::tan(glm::radians(80.0f) * 0.5f));
    pc.cam = glm::vec4(fv.eye_pos, ppr);
    pc.range = glm::uvec4(veg_cluster_first_, veg_cluster_count_, cl_vis_base_, cl_cmd_cap_);
    // One pixel of error at detail 1; the vegetation detail slider scales it.
    const bool occl = veg_cull_hiz_on_ && veg_hiz_view_ && !std::getenv("QLIKE_NO_CLUSTER_OCCL");
    pc.params = glm::vec4(1450.0f, 1.0f / std::max(0.25f, rt_.veg_lod_bias), occl ? 1.0f : 0.0f, 0.0f);
    pc.prev_vp = veg_cull_prev_vp_;
    pc.hiz = glm::vec4(float(veg_hiz_w_), float(veg_hiz_h_), float(veg_hiz_mips_), 0.0f);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cl_pipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cl_layout_, 0, 1, &cl_set_, 0, nullptr);
    vkCmdPushConstants(cmd, cl_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, (veg_cluster_count_ + 63) / 64, 1, 1);

    mb.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT;
    mb.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT;
    mb.dstStageMask  = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT;
    mb.dstAccessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
    vkCmdPipelineBarrier2(cmd, &dep);
}

// Inside draw_vegetation: the vegetation pipeline, sets and push constants are bound.
void VulkanEngine::draw_clusters(VkCommandBuffer cmd) {
    if (!cl_pipeline_ || veg_cluster_count_ == 0) return;
    VkDeviceSize off = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &cl_mesh_.vertex_buffer, &off);
    vkCmdBindIndexBuffer(cmd, cl_mesh_.index_buffer, 0, VK_INDEX_TYPE_UINT32);
    vkCmdDrawIndexedIndirectCount(cmd, cl_cmd_buf_, 16, cl_cmd_buf_, 0, cl_cmd_cap_, sizeof(VkDrawIndexedIndirectCommand));
}

}  // namespace qlike
