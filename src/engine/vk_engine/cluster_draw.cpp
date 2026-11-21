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

#include <GLFW/glfw3.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <thread>
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
    if (scl_set_ && veg_hiz_view_ && veg_hiz_sampler_) {
        VkDescriptorImageInfo si{ veg_hiz_sampler_, veg_hiz_view_, VK_IMAGE_LAYOUT_GENERAL };
        VkWriteDescriptorSet sw{};
        sw.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        sw.dstSet = scl_set_; sw.dstBinding = 3; sw.descriptorCount = 1;
        sw.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        sw.pImageInfo = &si;
        vkUpdateDescriptorSets(device_, 1, &sw, 0, nullptr);
        if (scl_set_shadow_) { sw.dstSet = scl_set_shadow_; vkUpdateDescriptorSets(device_, 1, &sw, 0, nullptr); }
        scl_hiz_bound_ = true;
    }
    if (!cl_set_ || !veg_hiz_view_ || !veg_hiz_sampler_) return;
    VkDescriptorImageInfo ii{ veg_hiz_sampler_, veg_hiz_view_, VK_IMAGE_LAYOUT_GENERAL };
    VkWriteDescriptorSet w{};
    w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w.dstSet = cl_set_; w.dstBinding = 7; w.descriptorCount = 1;
    w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    w.pImageInfo = &ii;
    vkUpdateDescriptorSets(device_, 1, &w, 0, nullptr);
}

// The vegetation instance / visible / fade buffers were re-made (re-plant in the editor).
void VulkanEngine::rebind_cluster_instance_buffers() {
    if (!cl_set_) return;
    VkDescriptorBufferInfo bi[3] = { { veg_inst_buf_, 0, VK_WHOLE_SIZE }, { veg_vis_buf_, 0, VK_WHOLE_SIZE }, { veg_fade_buf_, 0, VK_WHOLE_SIZE } };
    VkWriteDescriptorSet w[3]{};
    for (uint32_t i = 0; i < 3; ++i) {
        w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[i].dstSet = cl_set_; w[i].dstBinding = i; w[i].descriptorCount = 1;
        w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w[i].pBufferInfo = &bi[i];
    }
    vkUpdateDescriptorSets(device_, 3, w, 0, nullptr);
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
    const float ppr = float(swapchain_extent_.height) / (2.0f * std::tan(glm::radians(view_fov_deg_) * 0.5f));
    pc.cam = glm::vec4(fv.eye_pos, ppr);
    pc.range = glm::uvec4(veg_cluster_first_, veg_cluster_count_, cl_vis_base_, cl_cmd_cap_);
    // One pixel of error at detail 1; the vegetation detail slider scales it.
    const bool occl = veg_cull_hiz_on_ && veg_hiz_view_ && !std::getenv("QLIKE_NO_CLUSTER_OCCL");
    pc.params = glm::vec4(object_far_, 1.0f / std::max(0.25f, rt_.veg_lod_bias), occl ? 1.0f : 0.0f, 0.0f);
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

// ---------------------------------------------------------------------
// Static world meshes (castle, chapel, gravestones) as clusters.
//
// Each dense part has its cluster hierarchy; two selections are made per
// frame from the same data - one for the camera (cut + view + occlusion)
// and one for the sun's shadow map (cut by the camera's distance, culled
// to the light's box) - each an indirect-count draw per part.
//
// Streaming: the finest level (the original triangles, about half of all
// index data) is NOT kept on the GPU. A part's index buffer holds the
// simplified levels only; when the camera comes close enough that the
// finest level could be selected, a background thread reads it from
// cache/stream_*.idx and a full buffer replaces the coarse one; it is
// dropped again when the camera has moved well away. While it is not
// loaded the selection stops at level 1 (the shader's loaded mask).
// ---------------------------------------------------------------------
namespace {
struct StreamJobData { std::vector<uint32_t> data; std::atomic<int> done{0}; double t0 = 0.0; };
}

void VulkanEngine::init_static_clusters() {
    if (std::getenv("QLIKE_NO_CLUSTERS") || std::getenv("QLIKE_NO_STATIC_CLUSTERS") || scl_pipeline_) return;
    std::vector<ClusterGpu> clusters;
    std::vector<glm::uvec4> meta;
    size_t src_tris = 0, coarse_idx = 0, fine_idx = 0;
    size_t ci = 0;                     // castle_meshes_ holds the non-empty meshes of world_.meshes, in order
    std::error_code ec;
    std::filesystem::create_directories("cache", ec);
    for (size_t mi = 0; mi < world_.meshes.size() && scl_parts_.size() < 64; ++mi) {
        const auto& lm = world_.meshes[mi];
        if (lm.verts.empty() || lm.idx.empty()) continue;
        if (ci >= castle_meshes_.size()) break;
        CastleMeshGpu& g = castle_meshes_[ci++];
        // Dense, solid, still parts only (not the swaying trees of the level, not lamp glass / flames).
        if (lm.idx.size() < 3 * 3000 || lm.sway || lm.full_emissive || g.mesh.index_count != lm.idx.size()) continue;
        ClusterMesh cm = build_cluster_mesh_cached(reinterpret_cast<const Vertex*>(lm.verts.data()), lm.verts.size(),
                                                   lm.idx.data(), lm.idx.size(), lm.triplanar ? 0.0f : 0.25f);
        if (cm.clusters.empty() || cm.depth.size() != cm.clusters.size()) continue;
        SclPart P{};
        P.bmin = lm.bmin; P.bmax = lm.bmax;
        g.cl_part = int(scl_parts_.size());
        g.cl_count = uint32_t(cm.clusters.size());
        g.cl_cmd_base = uint32_t(clusters.size());
        // Index order inside the part: the simplified levels first, the finest
        // level last - the coarse buffer is a prefix of the full one.
        std::vector<uint32_t> fine;
        for (int pass = 0; pass < 2; ++pass) {
            for (size_t k = 0; k < cm.clusters.size(); ++k) {
                // (level-0 clusters of a group that could not be simplified have no coarser stand-in: they stay resident)
                const bool is_fine = cm.depth[k] == 0 && cm.clusters[k].parent_error < 1e29f;
                if (is_fine != (pass == 1)) continue;
                ClusterGpu& c = cm.clusters[k];
                const uint32_t* src = cm.indices.data() + c.first_index;
                std::vector<uint32_t>& dst = is_fine ? fine : P.coarse;
                const uint32_t at = uint32_t(dst.size());
                dst.insert(dst.end(), src, src + c.index_count);
                c.first_index = at | (is_fine ? 0x80000000u : 0u);       // fine: offset after the coarse block (fixed up below)
                if (cm.depth[k] == 1) P.e1max = std::max(P.e1max, c.self_error);
            }
        }
        for (size_t k = 0; k < cm.clusters.size(); ++k) {
            ClusterGpu c = cm.clusters[k];
            const bool fine_k = (c.first_index & 0x80000000u) != 0u;
            if (fine_k) c.first_index = (c.first_index & 0x7FFFFFFFu) + uint32_t(P.coarse.size());
            meta.push_back(glm::uvec4(uint32_t(g.cl_part), g.cl_cmd_base, fine_k ? 0u : std::max<uint32_t>(cm.depth[k], 1u), 0u));
            clusters.push_back(c);
        }
        P.fine_count = uint32_t(fine.size());
        // The finest level goes to its stream file (written once).
        {
            uint64_t h = 1469598103934665603ull;
            for (uint32_t v : fine) { h ^= v; h *= 1099511628211ull; }
            h ^= uint64_t(fine.size()) * 0x9E3779B97F4A7C15ull ^ uint64_t(lm.verts.size());
            char name[64];
            std::snprintf(name, sizeof(name), "cache/stream_%016llx.idx", static_cast<unsigned long long>(h));
            P.stream_file = name;
            if (!std::filesystem::exists(P.stream_file, ec)) {
                if (FILE* fp = std::fopen(name, "wb")) { std::fwrite(fine.data(), sizeof(uint32_t), fine.size(), fp); std::fclose(fp); }
            }
        }
        // How wide the finest triangles are when they are first drawn (in pixels):
        // edge length over the error at which the next level stops being enough.
        {
            double sum = 0.0; size_t n = 0;
            for (size_t t = 0; t + 2 < fine.size(); t += 3 * 16) {
                const glm::vec3 a = lm.verts[fine[t]].p, b = lm.verts[fine[t + 1]].p;
                sum += double(glm::length(b - a)); ++n;
            }
            P.tri_px = (n && P.e1max > 0.0f) ? float(sum / double(n)) / P.e1max : 0.0f;
        }
        P.ib_coarse = cl_buffer(allocator_, sizeof(uint32_t) * P.coarse.size(), VK_BUFFER_USAGE_INDEX_BUFFER_BIT, P.coarse_alloc,
                                P.coarse.data(), sizeof(uint32_t) * P.coarse.size());
        coarse_idx += P.coarse.size(); fine_idx += fine.size();
        src_tris += lm.idx.size() / 3;
        scl_parts_.push_back(std::move(P));
    }
    if (clusters.empty()) return;
    scl_total_ = uint32_t(clusters.size());
    const VkBufferUsageFlags ssbo = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    scl_cluster_buf_ = cl_buffer(allocator_, sizeof(ClusterGpu) * clusters.size(), ssbo, scl_cluster_alloc_,
                                 clusters.data(), sizeof(ClusterGpu) * clusters.size());
    scl_meta_buf_ = cl_buffer(allocator_, sizeof(glm::uvec4) * meta.size(), ssbo, scl_meta_alloc_, meta.data(), sizeof(glm::uvec4) * meta.size());
    const VkDeviceSize cmd_bytes = 256 + VkDeviceSize(scl_total_) * sizeof(VkDrawIndexedIndirectCommand);
    const VkBufferUsageFlags cu = ssbo | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    scl_cmd_buf_ = cl_buffer(allocator_, cmd_bytes, cu, scl_cmd_alloc_, nullptr, 0);
    scl_cmd_shadow_buf_ = cl_buffer(allocator_, cmd_bytes, cu, scl_cmd_shadow_alloc_, nullptr, 0);

    VkDescriptorSetLayoutBinding b[4]{};
    for (uint32_t i = 0; i < 4; ++i) {
        b[i].binding = i;
        b[i].descriptorType = i == 3 ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        b[i].descriptorCount = 1;
        b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo lci{};
    lci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    lci.bindingCount = 4; lci.pBindings = b;
    vk_check(vkCreateDescriptorSetLayout(device_, &lci, nullptr, &scl_set_layout_), "static cluster set layout");
    VkDescriptorPoolSize ps[2] = { { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 6 }, { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2 } };
    VkDescriptorPoolCreateInfo pci{};
    pci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pci.maxSets = 2; pci.poolSizeCount = 2; pci.pPoolSizes = ps;
    vk_check(vkCreateDescriptorPool(device_, &pci, nullptr, &scl_pool_), "static cluster pool");
    for (int si = 0; si < 2; ++si) {
        VkDescriptorSet& set = si == 0 ? scl_set_ : scl_set_shadow_;
        VkDescriptorSetAllocateInfo dai{};
        dai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dai.descriptorPool = scl_pool_; dai.descriptorSetCount = 1; dai.pSetLayouts = &scl_set_layout_;
        vk_check(vkAllocateDescriptorSets(device_, &dai, &set), "static cluster set");
        VkDescriptorBufferInfo bi[3] = { { scl_cluster_buf_, 0, VK_WHOLE_SIZE }, { scl_meta_buf_, 0, VK_WHOLE_SIZE },
                                         { si == 0 ? scl_cmd_buf_ : scl_cmd_shadow_buf_, 0, VK_WHOLE_SIZE } };
        VkWriteDescriptorSet w[3]{};
        for (uint32_t i = 0; i < 3; ++i) {
            w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[i].dstSet = set; w[i].dstBinding = i; w[i].descriptorCount = 1;
            w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            w[i].pBufferInfo = &bi[i];
        }
        vkUpdateDescriptorSets(device_, 3, w, 0, nullptr);
    }
    write_cluster_hiz_binding();

    VkPushConstantRange pcr{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(glm::vec4) * 14 };
    VkPipelineLayoutCreateInfo pli{};
    pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pli.setLayoutCount = 1; pli.pSetLayouts = &scl_set_layout_;
    pli.pushConstantRangeCount = 1; pli.pPushConstantRanges = &pcr;
    vk_check(vkCreatePipelineLayout(device_, &pli, nullptr, &scl_layout_), "static cluster layout");
    const std::string sd = QLIKE_SHADER_DIR;
    scl_module_ = vkpipe::load_shader_module(device_, sd + "/static_cluster_cull.comp.spv");
    VkComputePipelineCreateInfo cci{};
    cci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cci.stage.module = scl_module_;
    cci.stage.pName = "main";
    cci.layout = scl_layout_;
    vk_check(vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &cci, nullptr, &scl_pipeline_), "static cluster pipeline");
    float px_min = 1e9f;
    for (const auto& P : scl_parts_) if (P.tri_px > 0.0f) px_min = std::min(px_min, P.tri_px);
    log::infof("[cluster] static meshes: %zu parts (%zu tris) -> %u clusters; resident %.1f MB of indices, %.1f MB streamed on demand",
               scl_parts_.size(), src_tris, scl_total_, double(coarse_idx) * 4.0 / 1048576.0, double(fine_idx) * 4.0 / 1048576.0);
    log::infof("[cluster] finest triangles are at least %.1f px wide when they are first drawn (sub-pixel triangles do not occur)", px_min);
}

void VulkanEngine::destroy_static_clusters() {
    scl_ready_ = false;
    for (auto& P : scl_parts_) {
        if (P.job) { auto j = std::static_pointer_cast<StreamJobData>(P.job); while (!j->done.load()) std::this_thread::yield(); }
        if (P.ib_coarse) vmaDestroyBuffer(allocator_, P.ib_coarse, P.coarse_alloc);
        if (P.ib_full) vmaDestroyBuffer(allocator_, P.ib_full, P.full_alloc);
    }
    scl_parts_.clear();
    for (auto& d : scl_retired_) vmaDestroyBuffer(allocator_, d.buf, d.alloc);
    scl_retired_.clear();
    if (scl_pipeline_) vkDestroyPipeline(device_, scl_pipeline_, nullptr);
    if (scl_layout_) vkDestroyPipelineLayout(device_, scl_layout_, nullptr);
    if (scl_module_) vkDestroyShaderModule(device_, scl_module_, nullptr);
    if (scl_pool_) vkDestroyDescriptorPool(device_, scl_pool_, nullptr);
    if (scl_set_layout_) vkDestroyDescriptorSetLayout(device_, scl_set_layout_, nullptr);
    if (scl_cluster_buf_) vmaDestroyBuffer(allocator_, scl_cluster_buf_, scl_cluster_alloc_);
    if (scl_meta_buf_) vmaDestroyBuffer(allocator_, scl_meta_buf_, scl_meta_alloc_);
    if (scl_cmd_buf_) vmaDestroyBuffer(allocator_, scl_cmd_buf_, scl_cmd_alloc_);
    if (scl_cmd_shadow_buf_) vmaDestroyBuffer(allocator_, scl_cmd_shadow_buf_, scl_cmd_shadow_alloc_);
    scl_pipeline_ = VK_NULL_HANDLE; scl_layout_ = VK_NULL_HANDLE; scl_module_ = VK_NULL_HANDLE; scl_pool_ = VK_NULL_HANDLE;
    scl_set_layout_ = VK_NULL_HANDLE; scl_set_ = VK_NULL_HANDLE; scl_set_shadow_ = VK_NULL_HANDLE; scl_total_ = 0; scl_hiz_bound_ = false;
    scl_cluster_buf_ = scl_meta_buf_ = scl_cmd_buf_ = scl_cmd_shadow_buf_ = VK_NULL_HANDLE;
}

// Load / drop the finest level of each part by the camera's distance.
void VulkanEngine::stream_static_clusters(glm::vec3 eye, float ppr) {
    for (size_t r = 0; r < scl_retired_.size(); ) {
        if (--scl_retired_[r].frames <= 0) {
            vmaDestroyBuffer(allocator_, scl_retired_[r].buf, scl_retired_[r].alloc);
            scl_retired_.erase(scl_retired_.begin() + ptrdiff_t(r));
        } else ++r;
    }
    static const bool kAll = std::getenv("QLIKE_STREAM_ALL") != nullptr;     // keep everything loaded
    for (size_t pi = 0; pi < scl_parts_.size(); ++pi) {
        SclPart& P = scl_parts_[pi];
        const glm::vec3 q = glm::clamp(eye, P.bmin, P.bmax);
        const float d = glm::distance(q, eye);
        // The finest level can be selected inside this distance (error of level 1 = one pixel).
        const float need_d = P.e1max * ppr;
        const bool want = kAll || d < need_d * 1.6f + 12.0f;
        const bool far_away = !kAll && d > need_d * 2.6f + 30.0f;
        if (P.state == 0 && want && P.fine_count > 0) {
            auto job = std::make_shared<StreamJobData>();
            job->t0 = glfwGetTime();
            P.job = job;
            P.state = 1;
            const std::string file = P.stream_file;
            const uint32_t count = P.fine_count;
            std::thread([job, file, count]() {
                std::vector<uint32_t> buf(count);
                bool ok = false;
                if (FILE* fp = std::fopen(file.c_str(), "rb")) {
                    ok = std::fread(buf.data(), sizeof(uint32_t), count, fp) == count;
                    std::fclose(fp);
                }
                if (ok) job->data = std::move(buf);
                job->done.store(ok ? 1 : -1);
            }).detach();
        } else if (P.state == 1) {
            auto job = std::static_pointer_cast<StreamJobData>(P.job);
            const int done = job->done.load();
            if (done == 1) {
                std::vector<uint32_t> all;
                all.reserve(P.coarse.size() + job->data.size());
                all.insert(all.end(), P.coarse.begin(), P.coarse.end());
                all.insert(all.end(), job->data.begin(), job->data.end());
                P.ib_full = cl_buffer(allocator_, sizeof(uint32_t) * all.size(), VK_BUFFER_USAGE_INDEX_BUFFER_BIT, P.full_alloc,
                                      all.data(), sizeof(uint32_t) * all.size());
                P.state = 2;
                log::infof("[stream] part %zu: finest level in (%u triangles, %.0f ms, camera at %.0f m)", pi, P.fine_count / 3,
                           (glfwGetTime() - job->t0) * 1000.0, d);
                P.job.reset();
            } else if (done == -1) {
                log::warnf("[stream] part %zu: could not read %s", pi, P.stream_file.c_str());
                P.fine_count = 0; P.state = 0; P.job.reset();
            }
        } else if (P.state == 2 && far_away) {
            scl_retired_.push_back({ P.ib_full, P.full_alloc, 4 });       // frames in flight may still draw from it
            P.ib_full = VK_NULL_HANDLE; P.full_alloc = nullptr;
            P.state = 0;
            log::infof("[stream] part %zu: finest level out (camera at %.0f m)", pi, d);
        }
    }
}

void VulkanEngine::cull_static_clusters(VkCommandBuffer cmd) {
    scl_ready_ = false;
    if (!scl_pipeline_ || !scl_hiz_bound_ || scl_total_ == 0) return;
    const FrameView& fv = current_frame_view_;
    const float ppr = float(swapchain_extent_.height) / (2.0f * std::tan(glm::radians(view_fov_deg_) * 0.5f));
    stream_static_clusters(fv.eye_pos, ppr);
    glm::uvec4 loaded(0u);
    for (size_t pi = 0; pi < scl_parts_.size(); ++pi)
        if (scl_parts_[pi].state == 2) loaded[int(pi / 32)] |= 1u << (pi % 32);

    vkCmdFillBuffer(cmd, scl_cmd_buf_, 0, 256, 0u);
    vkCmdFillBuffer(cmd, scl_cmd_shadow_buf_, 0, 256, 0u);
    VkMemoryBarrier2 mb{};
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
    mb.srcStageMask  = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT | VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT;
    mb.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;
    mb.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    mb.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    VkDependencyInfo dep{};
    dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.memoryBarrierCount = 1; dep.pMemoryBarriers = &mb;
    vkCmdPipelineBarrier2(cmd, &dep);
    struct { glm::vec4 planes[6]; glm::vec4 cam; glm::vec4 params; glm::mat4 prev_vp; glm::vec4 hiz; glm::uvec4 loaded; } pc{};
    pc.cam = glm::vec4(fv.eye_pos, ppr);
    pc.prev_vp = veg_cull_prev_vp_;
    pc.hiz = glm::vec4(float(veg_hiz_w_), float(veg_hiz_h_), float(veg_hiz_mips_), 0.0f);
    pc.loaded = loaded;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, scl_pipeline_);
    // 1. The camera's view.
    for (int i = 0; i < 6; ++i) pc.planes[i] = fv.frustum.planes[i];
    const bool occl = veg_cull_hiz_on_ && !std::getenv("QLIKE_NO_CLUSTER_OCCL");
    pc.params = glm::vec4(float(scl_total_), 1.0f, occl ? 1.0f : 0.0f, 0.0f);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, scl_layout_, 0, 1, &scl_set_, 0, nullptr);
    vkCmdPushConstants(cmd, scl_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, (scl_total_ + 63) / 64, 1, 1);
    // 2. The sun's shadow map: the same hierarchy, cut a little coarser (a shadow
    // edge is soft), culled to the light's box, nothing hidden by the camera's depth.
    {
        const Frustum lf = extract_frustum(sun_shadow_light_vp_);
        for (int i = 0; i < 6; ++i) pc.planes[i] = lf.planes[i];
        pc.params = glm::vec4(float(scl_total_), 2.0f, 0.0f, 0.0f);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, scl_layout_, 0, 1, &scl_set_shadow_, 0, nullptr);
        vkCmdPushConstants(cmd, scl_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, (scl_total_ + 63) / 64, 1, 1);
    }
    mb.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT;
    mb.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT;
    mb.dstStageMask  = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT;
    mb.dstAccessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;
    vkCmdPipelineBarrier2(cmd, &dep);
    scl_ready_ = true;
}

}  // namespace qlike
