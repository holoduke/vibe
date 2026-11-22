// The soldiers in the ray-traced scene. Skinned characters are deformed in the vertex shader,
// so the ray tracer never saw them: no shadow on walls, props or each other, no reflection, no
// bounce light. Here the nearest few get a ray-tracing copy every frame:
//
//   bone palette --(skin_rt.comp)--> positions --(BLAS build)--> an instance in the TLAS
//
// on the compute queue, in the same command buffer as the TLAS build (rebuild_tlas, rt.cpp).
// The copy is the lowest level of detail (body + head, ~9k triangles), in the character's model
// space; its TLAS transform is the character's model matrix. Everything that changes per frame
// exists once per frame in flight (positions, BLAS, scratch): the graphics queue may still be
// tracing the previous frame's.
//
// Only close by (kRange, the nearest kBodies): a shadow of a soldier 60 m away is a few pixels
// and the small-object shadow map (clutter_shadow.cpp) already puts it on the ground.

#include "engine/vk_engine.h"
#include "engine/vk_engine/internal.h"
#include "engine/vk_engine/state/multiplayer.h"

#include "engine/log.h"
#include "engine/vk_pipelines.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <string>

#ifndef VIBE_SHADER_DIR
#define VIBE_SHADER_DIR "shaders"
#endif

namespace vibe {

namespace {
const float kRange = 40.0f;         // metres from the camera
}

void VulkanEngine::init_body_rt() {
    static const bool kOff = std::getenv("VIBE_NO_BODY_RT") != nullptr;
    if (kOff || !rt_initialized_ || body_rt_idx_.empty() || !body_palette_buf_) {
        if (!kOff && rt_initialized_) log::info("[body] no level 2 meshes: the soldiers are not in the ray-traced scene");
        return;
    }
    const uint32_t nv = uint32_t(body_rt_rest_.size()), ni = uint32_t(body_rt_idx_.size());
    const uint32_t shared_families[2] = { graphics_queue_family_, compute_queue_family_ };
    auto make = [&](VkDeviceSize size, VkBufferUsageFlags usage, bool mapped, VmaAllocation& alloc, void** map) {
        VkBufferCreateInfo bci{};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size = size;
        bci.usage = usage;
        bci.sharingMode = compute_queue_distinct_ ? VK_SHARING_MODE_CONCURRENT : VK_SHARING_MODE_EXCLUSIVE;
        bci.queueFamilyIndexCount = compute_queue_distinct_ ? 2u : 0u;
        bci.pQueueFamilyIndices = compute_queue_distinct_ ? shared_families : nullptr;
        VmaAllocationCreateInfo aci{};
        aci.usage = mapped ? VMA_MEMORY_USAGE_AUTO : VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        if (mapped) aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
        VmaAllocationInfo info{};
        VkBuffer b = VK_NULL_HANDLE;
        vk_check(vmaCreateBuffer(allocator_, &bci, &aci, &b, &alloc, &info), "body rt buffer");
        if (map) *map = info.pMappedData;
        return b;
    };
    // The mesh: written once (host-visible is fine for 7k vertices read by a compute shader).
    const VkBufferUsageFlags ssbo = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    const VkBufferUsageFlags as_in = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                                     VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    void* map = nullptr;
    body_rt_rest_buf_ = make(sizeof(Vertex) * nv, ssbo, true, body_rt_rest_alloc_, &map);
    std::memcpy(map, body_rt_rest_.data(), sizeof(Vertex) * nv);
    body_rt_skin_buf_ = make(sizeof(Vertex) * nv, ssbo, true, body_rt_skin_alloc_, &map);
    std::memcpy(map, body_rt_skin_.data(), sizeof(Vertex) * nv);
    body_rt_index_buf_ = make(sizeof(uint32_t) * ni, as_in, true, body_rt_index_alloc_, &map);
    std::memcpy(map, body_rt_idx_.data(), sizeof(uint32_t) * ni);
    vmaFlushAllocation(allocator_, body_rt_rest_alloc_, 0, VK_WHOLE_SIZE);
    vmaFlushAllocation(allocator_, body_rt_skin_alloc_, 0, VK_WHOLE_SIZE);
    vmaFlushAllocation(allocator_, body_rt_index_alloc_, 0, VK_WHOLE_SIZE);
    body_rt_verts_ = nv;
    body_rt_tris_ = ni / 3;

    // Size of one soldier's BLAS (built from scratch every frame: fast build, no update).
    VkAccelerationStructureGeometryKHR geom{};
    geom.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geom.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    geom.flags = VK_GEOMETRY_OPAQUE_BIT_KHR | VK_GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION_BIT_KHR;
    geom.geometry.triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
    geom.geometry.triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
    geom.geometry.triangles.vertexStride = sizeof(glm::vec4);
    geom.geometry.triangles.maxVertex = nv - 1;
    geom.geometry.triangles.indexType = VK_INDEX_TYPE_UINT32;
    VkAccelerationStructureBuildGeometryInfoKHR bgi{};
    bgi.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    bgi.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    bgi.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR;
    bgi.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    bgi.geometryCount = 1;
    bgi.pGeometries = &geom;
    VkAccelerationStructureBuildSizesInfoKHR sizes{};
    sizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    g_rt.get_as_build_sizes(device_, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &bgi, &body_rt_tris_, &sizes);

    for (uint32_t s = 0; s < kFrameOverlap; ++s) {
        body_rt_out_buf_[s] = make(sizeof(glm::vec4) * nv * kRtBodies, ssbo | as_in, false, body_rt_out_alloc_[s], nullptr);
        body_rt_out_addr_[s] = buffer_device_address(device_, body_rt_out_buf_[s]);
        for (uint32_t k = 0; k < kRtBodies; ++k) {
            RtBody& b = body_rt_[s][k];
            b.buf = make(sizes.accelerationStructureSize,
                         VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, false, b.alloc, nullptr);
            VkAccelerationStructureCreateInfoKHR ci{};
            ci.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
            ci.buffer = b.buf;
            ci.size = sizes.accelerationStructureSize;
            ci.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
            vk_check(g_rt.create_as(device_, &ci, nullptr, &b.as), "body blas");
            b.addr = as_device_address(device_, b.as);
            b.scratch = make(sizes.buildScratchSize + kAsScratchPad, ssbo | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, false, b.scratch_alloc, nullptr);
            b.scratch_addr = scratch_device_address(device_, b.scratch);
        }
    }
    body_rt_index_addr_ = buffer_device_address(device_, body_rt_index_buf_);

    // The skinning pass.
    VkDescriptorSetLayoutBinding lb[4]{};
    for (uint32_t i = 0; i < 4; ++i) {
        lb[i].binding = i;
        lb[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        lb[i].descriptorCount = 1;
        lb[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo lci{};
    lci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    lci.bindingCount = 4;
    lci.pBindings = lb;
    vk_check(vkCreateDescriptorSetLayout(device_, &lci, nullptr, &body_rt_set_layout_), "body rt set layout");
    VkDescriptorPoolSize ps{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4 * kFrameOverlap };
    VkDescriptorPoolCreateInfo pci{};
    pci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pci.maxSets = kFrameOverlap;
    pci.poolSizeCount = 1;
    pci.pPoolSizes = &ps;
    vk_check(vkCreateDescriptorPool(device_, &pci, nullptr, &body_rt_pool_), "body rt pool");
    for (uint32_t s = 0; s < kFrameOverlap; ++s) {
        VkDescriptorSetAllocateInfo dai{};
        dai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dai.descriptorPool = body_rt_pool_;
        dai.descriptorSetCount = 1;
        dai.pSetLayouts = &body_rt_set_layout_;
        vk_check(vkAllocateDescriptorSets(device_, &dai, &body_rt_set_[s]), "body rt set");
        VkDescriptorBufferInfo bi[4] = { { body_rt_rest_buf_, 0, VK_WHOLE_SIZE }, { body_rt_skin_buf_, 0, VK_WHOLE_SIZE },
                                         { body_palette_buf_, 0, VK_WHOLE_SIZE }, { body_rt_out_buf_[s], 0, VK_WHOLE_SIZE } };
        VkWriteDescriptorSet w[4]{};
        for (uint32_t i = 0; i < 4; ++i) {
            w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[i].dstSet = body_rt_set_[s];
            w[i].dstBinding = i;
            w[i].descriptorCount = 1;
            w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            w[i].pBufferInfo = &bi[i];
        }
        vkUpdateDescriptorSets(device_, 4, w, 0, nullptr);
    }
    VkPushConstantRange pcr{ VK_SHADER_STAGE_COMPUTE_BIT, 0, 16 };
    VkPipelineLayoutCreateInfo pli{};
    pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &body_rt_set_layout_;
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges = &pcr;
    vk_check(vkCreatePipelineLayout(device_, &pli, nullptr, &body_rt_layout_), "body rt layout");
    body_rt_module_ = vkpipe::load_shader_module(device_, std::string(VIBE_SHADER_DIR) + "/skin_rt.comp.spv");
    if (!body_rt_module_) { log::warn("[body] skin_rt.comp.spv missing: the soldiers are not in the ray-traced scene"); return; }
    VkComputePipelineCreateInfo cci{};
    cci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cci.stage.module = body_rt_module_;
    cci.stage.pName = "main";
    cci.layout = body_rt_layout_;
    vk_check(vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &cci, nullptr, &body_rt_pipeline_), "body rt pipeline");
    log::infof("[body] ray-traced soldiers: up to %u within %.0f m, %u triangles each (%.0f kB of acceleration structure a body)",
               kRtBodies, double(kRange), body_rt_tris_, double(sizes.accelerationStructureSize) / 1024.0);
}

void VulkanEngine::destroy_body_rt() {
    if (body_rt_pipeline_) vkDestroyPipeline(device_, body_rt_pipeline_, nullptr);
    if (body_rt_module_) vkDestroyShaderModule(device_, body_rt_module_, nullptr);
    if (body_rt_layout_) vkDestroyPipelineLayout(device_, body_rt_layout_, nullptr);
    if (body_rt_pool_) vkDestroyDescriptorPool(device_, body_rt_pool_, nullptr);
    if (body_rt_set_layout_) vkDestroyDescriptorSetLayout(device_, body_rt_set_layout_, nullptr);
    body_rt_pipeline_ = VK_NULL_HANDLE; body_rt_module_ = VK_NULL_HANDLE; body_rt_layout_ = VK_NULL_HANDLE;
    body_rt_pool_ = VK_NULL_HANDLE; body_rt_set_layout_ = VK_NULL_HANDLE;
    for (uint32_t s = 0; s < kFrameOverlap; ++s) {
        for (uint32_t k = 0; k < kRtBodies; ++k) {
            RtBody& b = body_rt_[s][k];
            if (b.as) g_rt.destroy_as(device_, b.as, nullptr);
            if (b.buf) vmaDestroyBuffer(allocator_, b.buf, b.alloc);
            if (b.scratch) vmaDestroyBuffer(allocator_, b.scratch, b.scratch_alloc);
            b = RtBody{};
        }
        if (body_rt_out_buf_[s]) vmaDestroyBuffer(allocator_, body_rt_out_buf_[s], body_rt_out_alloc_[s]);
        body_rt_out_buf_[s] = VK_NULL_HANDLE;
        body_rt_set_[s] = VK_NULL_HANDLE;
    }
    if (body_rt_rest_buf_) vmaDestroyBuffer(allocator_, body_rt_rest_buf_, body_rt_rest_alloc_);
    if (body_rt_skin_buf_) vmaDestroyBuffer(allocator_, body_rt_skin_buf_, body_rt_skin_alloc_);
    if (body_rt_index_buf_) vmaDestroyBuffer(allocator_, body_rt_index_buf_, body_rt_index_alloc_);
    body_rt_rest_buf_ = body_rt_skin_buf_ = body_rt_index_buf_ = VK_NULL_HANDLE;
    body_rt_list_.clear();
}

// Picks this frame's ray-traced soldiers (the nearest with a pose), skins their copies and
// rebuilds their acceleration structures. Called at the top of rebuild_tlas, which then adds
// body_rt_list_ to the TLAS. The poses of this frame must exist (pose_characters_for_frame).
void VulkanEngine::record_body_rt(VkCommandBuffer cmd) {
    body_rt_list_.clear();
    if (!body_rt_pipeline_ || !game_.player_body) return;
    struct Cand { float d2; glm::mat4 model, prev; int palette; };
    Cand cand[1 + net::kMaxPlayers + 8];
    int nc = 0;
    const glm::vec3 eye = current_frame_view_.eye_pos;
    auto add = [&](const glm::mat4& model, const glm::mat4& prev, int palette) {
        if (palette < 0 || nc >= int(std::size(cand))) return;
        const glm::vec3 d = glm::vec3(model[3]) + glm::vec3(0.0f, 0.9f, 0.0f) - eye;
        const float d2 = glm::dot(d, d);
        if (d2 > kRange * kRange) return;
        cand[nc++] = { d2, model, prev, palette };
    };
    if (body_drawn_) add(body_model_, body_model_, body_palette_);
    if (net_)
        for (const MpAvatar& a : mp_->avatars)
            if (a.id && a.drawn) add(a.model, a.prev_model, a.palette);
    for (const Corpse& c : corpses_)
        if (c.rag && c.av.drawn) add(c.av.model, c.av.model, c.av.palette);
    // Test hook: VIBE_BODY_RT_TEST=<n> adds n copies of the nearest soldier, a metre apart (the cost of n more).
    static const int kTest = std::getenv("VIBE_BODY_RT_TEST") ? std::atoi(std::getenv("VIBE_BODY_RT_TEST")) : 0;
    for (int i = 0; i < kTest && nc > 0 && nc < int(std::size(cand)); ++i) {
        cand[nc] = cand[0];
        cand[nc].model[3].x += float(i + 1);
        cand[nc].prev = cand[nc].model;
        cand[nc].d2 += float(i + 1);
        ++nc;
    }
    if (nc == 0) return;
    std::sort(cand, cand + nc, [](const Cand& a, const Cand& b) { return a.d2 < b.d2; });
    const uint32_t n = uint32_t(std::min(nc, int(kRtBodies)));
    const uint32_t slot = frame_number_ % kFrameOverlap;
    static const float kInset = std::getenv("VIBE_BODY_RT_INSET") ? float(std::atof(std::getenv("VIBE_BODY_RT_INSET"))) : 0.012f;

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, body_rt_pipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, body_rt_layout_, 0, 1, &body_rt_set_[slot], 0, nullptr);
    for (uint32_t k = 0; k < n; ++k) {
        struct { uint32_t palette, count, dst; float inset; } pc{ uint32_t(cand[k].palette), body_rt_verts_, k * body_rt_verts_, kInset };
        vkCmdPushConstants(cmd, body_rt_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, (body_rt_verts_ + 63u) / 64u, 1, 1);
    }
    {   // the positions are written before the builds read them
        VkMemoryBarrier2 mb{};
        mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        mb.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        mb.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
        mb.dstStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
        mb.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
        VkDependencyInfo dep{};
        dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.memoryBarrierCount = 1;
        dep.pMemoryBarriers = &mb;
        vkCmdPipelineBarrier2(cmd, &dep);
    }
    VkAccelerationStructureGeometryKHR geom[kRtBodies]{};
    VkAccelerationStructureBuildGeometryInfoKHR bgi[kRtBodies]{};
    VkAccelerationStructureBuildRangeInfoKHR range[kRtBodies]{};
    const VkAccelerationStructureBuildRangeInfoKHR* ranges[kRtBodies]{};
    for (uint32_t k = 0; k < n; ++k) {
        geom[k].sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
        geom[k].geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
        geom[k].flags = VK_GEOMETRY_OPAQUE_BIT_KHR | VK_GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION_BIT_KHR;
        auto& tri = geom[k].geometry.triangles;
        tri.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
        tri.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
        tri.vertexData.deviceAddress = body_rt_out_addr_[slot] + VkDeviceAddress(k) * body_rt_verts_ * sizeof(glm::vec4);
        tri.vertexStride = sizeof(glm::vec4);
        tri.maxVertex = body_rt_verts_ - 1;
        tri.indexType = VK_INDEX_TYPE_UINT32;
        tri.indexData.deviceAddress = body_rt_index_addr_;
        bgi[k].sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
        bgi[k].type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        bgi[k].flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR;
        bgi[k].mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
        bgi[k].dstAccelerationStructure = body_rt_[slot][k].as;
        bgi[k].geometryCount = 1;
        bgi[k].pGeometries = &geom[k];
        bgi[k].scratchData.deviceAddress = body_rt_[slot][k].scratch_addr;
        range[k].primitiveCount = body_rt_tris_;
        ranges[k] = &range[k];
        body_rt_list_.push_back({ cand[k].model, cand[k].prev, body_rt_[slot][k].addr });
    }
    g_rt.cmd_build_as(cmd, n, bgi, ranges);
    // (rebuild_tlas follows with the barrier between these builds and the TLAS build)
}

} // namespace vibe
