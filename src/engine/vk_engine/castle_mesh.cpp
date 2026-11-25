// Castle mesh parts (round towers, conical roofs, gate arch).
//
// Level::meshes is built in game/level.cpp in castle-local space; init_world
// lifts it onto the plateau and calls init_castle_meshes() to upload it.
// init_rt() calls build_castle_mesh_blas() (one BLAS per part); the parts
// are written into the TLAS every rebuild like the gate doors. Raster goes
// through the regular world pipeline with grass_params.w = -1, which cube.frag
// / cube.vert treat as "static mesh": vertex-UV texturing with a derivative
// tangent frame, and none of the axis-aligned-box-only paths (box relief,
// SPOM, corner softening, shell extrusion).

#include "engine/vk_engine.h"
#include "engine/vk_engine/internal.h"
#include "engine/vk_engine/state/environment.h"
#include "engine/vk_engine/state/static_clusters.h"
#include "engine/vk_engine/river_stream.h"
#include <unordered_map>
#include <cmath>
#include "engine/log.h"
#include "engine/vk_initializers.h"

#include <cstring>

namespace vibe {

static_assert(sizeof(game::LevelVertex) == sizeof(Vertex),
              "LevelVertex must match the engine Vertex layout");

// One mesh part of a level onto the GPU (home's at start-up; a site's when it is loaded).
VulkanEngine::CastleMeshGpu VulkanEngine::make_mesh_part(const game::LevelMesh& lm) {
    {
        CastleMeshGpu g{};
        g.mesh = create_mesh_from_data(
            device_, allocator_, graphics_queue_, graphics_queue_family_,
            reinterpret_cast<const Vertex*>(lm.verts.data()),
            static_cast<uint32_t>(lm.verts.size()),
            lm.idx.data(), static_cast<uint32_t>(lm.idx.size()));
        g.tex_albedo = lm.tex_albedo;
        g.tex_normal = lm.tex_normal;
        g.tint = lm.tint;
        g.fallback = lm.fallback;
        g.bmin = lm.bmin;
        g.bmax = lm.bmax;
        g.triplanar = lm.triplanar;
        g.sway = lm.sway;
        g.uv_scale = lm.uv_scale;
        g.relief = lm.relief;
        // Screen-space displacement (ssdm.cpp): how far the stones stand out, in metres. Given
        // outright (LevelMesh::ssdm), or from the relief depth for masonry that has a height
        // texture (cube.frag height_idx_for_albedo: 1 castle walls, 4 keep walls) - the same
        // ~20 cm at relief 1 the parallax relief shows.
        g.ssdm = lm.ssdm > 0.0f ? lm.ssdm
               : ((lm.tex_albedo == 1 || lm.tex_albedo == 4) && !lm.triplanar) ? 0.28f * glm::clamp(lm.relief, 0.0f, 1.0f) : 0.0f;
        g.emissive = glm::vec4(lm.emissive, lm.full_emissive ? (lm.lit_by_day ? 2.0f : 1.0f) : 0.0f);      // (.a 2: as bright by day)
        return g;
    }
}

void VulkanEngine::init_castle_meshes() {
    size_t total_tris = 0;
    castle_of_level_mesh_.assign(world_.meshes.size(), -1);
    for (size_t lmi = 0; lmi < world_.meshes.size(); ++lmi) {
        const auto& lm = world_.meshes[lmi];
        if (lm.verts.empty() || lm.idx.empty()) continue;
        castle_of_level_mesh_[lmi] = int(castle_meshes_.size());
        castle_meshes_.push_back(make_mesh_part(lm));
        total_tris += lm.idx.size() / 3;
    }
    log::infof("[castle] %zu static mesh parts, %zu tris", castle_meshes_.size(), total_tris);
    init_static_clusters();
}

void VulkanEngine::build_castle_mesh_blas() { build_mesh_part_blas(castle_meshes_); }

void VulkanEngine::build_mesh_part_blas(std::vector<CastleMeshGpu>& parts) {
    const uint32_t shared_families[2] = { graphics_queue_family_, compute_queue_family_ };
    const VkSharingMode rt_sharing = compute_queue_distinct_
        ? VK_SHARING_MODE_CONCURRENT : VK_SHARING_MODE_EXCLUSIVE;
    for (auto& g : parts) {
        if (g.blas) continue;
        const uint32_t tri_count = g.mesh.index_count / 3;
        VkAccelerationStructureGeometryTrianglesDataKHR tri{};
        tri.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
        tri.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
        tri.vertexData.deviceAddress = buffer_device_address(device_, g.mesh.vertex_buffer);
        tri.vertexStride = sizeof(Vertex);
        tri.maxVertex = g.mesh.vertex_count - 1;
        tri.indexType = VK_INDEX_TYPE_UINT32;
        tri.indexData.deviceAddress = buffer_device_address(device_, g.mesh.index_buffer);

        VkAccelerationStructureGeometryKHR geom{};
        geom.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
        geom.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
        geom.geometry.triangles = tri;
        geom.flags = VK_GEOMETRY_OPAQUE_BIT_KHR |
                     VK_GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION_BIT_KHR;   // see rt.cpp

        VkAccelerationStructureBuildGeometryInfoKHR bgi{};
        bgi.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
        bgi.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        bgi.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
        bgi.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
        bgi.geometryCount = 1;
        bgi.pGeometries = &geom;

        VkAccelerationStructureBuildSizesInfoKHR sizes{};
        sizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
        g_rt.get_as_build_sizes(device_, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                                &bgi, &tri_count, &sizes);

        {
            VkBufferCreateInfo bci{};
            bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            bci.size = sizes.accelerationStructureSize;
            bci.usage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
                        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
            bci.sharingMode = rt_sharing;
            bci.queueFamilyIndexCount = compute_queue_distinct_ ? 2u : 0u;
            bci.pQueueFamilyIndices = compute_queue_distinct_ ? shared_families : nullptr;
            VmaAllocationCreateInfo aci{};
            aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
            vk_check(vmaCreateBuffer(allocator_, &bci, &aci, &g.blas_buf,
                                     &g.blas_alloc, nullptr),
                     "castle mesh blas buffer");
        }
        VkAccelerationStructureCreateInfoKHR ci{};
        ci.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
        ci.buffer = g.blas_buf;
        ci.size = sizes.accelerationStructureSize;
        ci.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        vk_check(g_rt.create_as(device_, &ci, nullptr, &g.blas), "castle mesh blas");
        bgi.dstAccelerationStructure = g.blas;

        VkBuffer scratch = VK_NULL_HANDLE;
        VmaAllocation scratch_alloc = nullptr;
        {
            VkBufferCreateInfo bci{};
            bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            bci.size = sizes.buildScratchSize + kAsScratchPad;
            bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
            bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            VmaAllocationCreateInfo aci{};
            aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
            vk_check(vmaCreateBuffer(allocator_, &bci, &aci, &scratch,
                                     &scratch_alloc, nullptr),
                     "castle mesh blas scratch");
        }
        bgi.scratchData.deviceAddress = scratch_device_address(device_, scratch);
        vkinit::one_time_submit(device_, graphics_queue_, graphics_queue_family_,
                                [&](VkCommandBuffer cb) {
            VkAccelerationStructureBuildRangeInfoKHR range{};
            range.primitiveCount = tri_count;
            const VkAccelerationStructureBuildRangeInfoKHR* p_range = &range;
            g_rt.cmd_build_as(cb, 1, &bgi, &p_range);
        });
        vmaDestroyBuffer(allocator_, scratch, scratch_alloc);
        g.blas_addr = as_device_address(device_, g.blas);
    }
}

// Every static mesh part there is to draw or trace now: home's, and the site's while the
// engine is in its chart.
const std::vector<const VulkanEngine::CastleMeshGpu*>& VulkanEngine::mesh_parts_now() {
    mesh_parts_now_.clear();
    // (home's own parts only in home's chart: in any other the same coordinates are another place)
    if (globe::current().home) for (const auto& g : castle_meshes_) mesh_parts_now_.push_back(&g);
    if (site_here()) for (const auto& g : site_meshes_) mesh_parts_now_.push_back(&g);
    for (const auto& ch : hw_chunks_) for (const auto& g : ch.second) mesh_parts_now_.push_back(&g);
    if (globe::current().home) for (const auto& g : jeep_draw_) mesh_parts_now_.push_back(&g);              // (jeep.cpp)       // (a highway's roadside objects: highway_furniture.cpp)
    return mesh_parts_now_;
}

void VulkanEngine::destroy_castle_meshes() {
    destroy_sites();
    hw_objects_destroy();
    river_stream::destroy(allocator_);
    destroy_jeep();
    destroy_mesh_parts(castle_meshes_);
}

void VulkanEngine::destroy_mesh_parts(std::vector<CastleMeshGpu>& parts) {
    for (auto& g : parts) {
        if (g.blas) g_rt.destroy_as(device_, g.blas, nullptr);
        if (g.blas_buf) vmaDestroyBuffer(allocator_, g.blas_buf, g.blas_alloc);
        if (g.mesh.vertex_buffer) destroy_mesh(allocator_, g.mesh);
    }
    parts.clear();
}

void VulkanEngine::draw_castle_meshes(VkCommandBuffer cmd, const glm::mat4& vp,
                                      const glm::mat4& prev_vp, const Frustum& fr,
                                      bool depth_only, VkPipelineLayout shadow_layout) {
    if (castle_meshes_.empty()) return;
    const bool tex_on = rt_.textures_enabled;
    VkDeviceSize off = 0;
    for (const CastleMeshGpu* gp : mesh_parts_now()) {
        const CastleMeshGpu& g = *gp;
        if (!aabb_visible(fr, g.bmin, g.bmax)) continue;
        // Glass (material -3): not in the depth pre-pass nor in any shadow pass - it is drawn
        // see-through (cube.frag), and what is behind it must be there to see.
        // Nor here at all: the panes are drawn after the lighting (fx.cpp, glass.frag).
        if (g.tex_albedo == -3) continue;
        // Depth PRE-pass: skip the relief stone parts. cube.frag discards
        // their silhouette pixels (parallax relief clipping on curved
        // towers); the flat pre-pass depth would hide those cut-outs.
        if (depth_only && !shadow_layout && rt_.spom_strength > 0.01f &&
            (g.tex_albedo == 1 || g.tex_albedo == 4)) continue;
        // The clusters picked for this frame (cull_static_clusters): one selection for the camera, one for the sun.
        const bool clustered = scl_->ready && g.cl_part >= 0 && size_t(g.cl_part) < scl_->parts.size();
        vkCmdBindVertexBuffers(cmd, 0, 1, &g.mesh.vertex_buffer, &off);
        VkBuffer ib = g.mesh.index_buffer;
        if (clustered) { const SclPart& P = scl_->parts[size_t(g.cl_part)]; ib = (P.state == 2 && P.ib_full) ? P.ib_full : P.ib_coarse; }
        vkCmdBindIndexBuffer(cmd, ib, 0, VK_INDEX_TYPE_UINT32);
        if (shadow_layout) {
            glm::mat4 mvp = vp * g.xf;        // (the part's own shift: CastleMeshGpu::xf)
            vkCmdPushConstants(cmd, shadow_layout, VK_SHADER_STAGE_VERTEX_BIT,
                               0, sizeof(glm::mat4), &mvp);
        } else {
            PushConstants pc{};
            pc.mvp = vp * g.xf;
            pc.model = g.xf;
            pc.prev_mvp = prev_vp * g.xf_prev;
            // Static-mesh mode flag (see file header).
            // .y = wind sway (uv = weight, phase), .z = world triplanar.
            // .x <= 0: relief reduction (0 = full depth, -1 = flat).
            // (a part displaced in screen space is drawn flat: its stones are moved there)
            const float relief = (g.ssdm > 0.0f && ssdm_active()) ? 0.0f : glm::clamp(g.relief, 0.0f, 1.0f);
            pc.grass_params = glm::vec4(relief - 1.0f, g.sway ? 1.0f : 0.0f,
                                        g.triplanar ? 1.0f : 0.0f, -1.0f);
            if (!depth_only) {
                pc.color = glm::vec4(tex_on ? g.tint : g.fallback, 1.0f);
                // Self-lit parts (lantern glass) glow faintly by day, fully at night.
                pc.emissive = g.emissive;
                if (g.emissive.a > 0.5f && g.emissive.a < 1.5f && g.emissive.r < 3.5f) {       // (flames burn at full strength by day too)
                    const float night = env_->night >= 0.0f ? env_->night : 1.0f - glm::smoothstep(4.0f, 16.0f, rt_.sun_pitch_deg);
                    pc.emissive = glm::vec4(glm::vec3(g.emissive) * glm::mix(0.12f, 1.0f, night), g.emissive.a);
                    pc.color = glm::vec4(glm::vec3(pc.color) * glm::mix(0.45f, 1.0f, night), 1.0f);
                }
                pc.tex_params = tex_on
                    ? glm::vec4(static_cast<float>(g.tex_albedo),
                                static_cast<float>(g.tex_normal), g.uv_scale, 0.0f)
                    : glm::vec4(-1.0f, -1.0f, 1.0f, 0.0f);
            }
            vkCmdPushConstants(cmd, pipeline_layout_, kPushConstantStages,
                               0, sizeof(PushConstants), &pc);
        }
        if (clustered) {
            VkBuffer cb = shadow_layout ? scl_->cmd_shadow_buf : scl_->cmd_buf;       // the sun's selection / the camera's
            vkCmdDrawIndexedIndirectCount(cmd, cb, 256 + VkDeviceSize(g.cl_cmd_base) * sizeof(VkDrawIndexedIndirectCommand),
                                          cb, VkDeviceSize(g.cl_part) * 4, g.cl_count, sizeof(VkDrawIndexedIndirectCommand));
        }
        else
            vkCmdDrawIndexed(cmd, g.mesh.index_count, 1, 0, 0, 0);
    }
}

} // namespace vibe
