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
#include "engine/vk_engine/state/static_clusters.h"
#include <unordered_map>
#include <cmath>
#include "engine/log.h"
#include "engine/vk_initializers.h"

#include <cstring>

namespace vibe {

static_assert(sizeof(game::LevelVertex) == sizeof(Vertex),
              "LevelVertex must match the engine Vertex layout");

namespace {

// The shell mesh of a displaced mesh (screen-space displacement mapping, ssdm.cpp).
//
// The displacement pass inflates the mesh along its vertex normals. A flat-shaded mesh has
// one normal per face, so at every hard edge the inflated faces would part and leave the
// corner uncovered. The shell mesh is the same surface with normals that round over the
// edges: every quad is cut into 3 x 3, the inner four vertices keep the face's normal, and
// the vertices on the quad's border take the average of the distinct normals of all faces
// meeting at that position. Away from the edges (beyond `inset`) nothing changes; within it
// the normal turns from one face to the next, and the shell is closed.
// Meshes that are not made of quads (4 vertices, 6 indices each) get the position-averaged
// normals without the cut.
void build_ssdm_shell(const game::LevelMesh& lm, float inset, std::vector<Vertex>& out_v, std::vector<uint32_t>& out_i) {
    out_v.clear(); out_i.clear();
    const size_t nv = lm.verts.size(), ni = lm.idx.size();
    bool quads = nv % 4 == 0 && ni == nv / 4 * 6;
    for (size_t q = 0; quads && q < nv / 4; ++q)
        for (int k = 0; k < 6; ++k) {
            const uint32_t ix = lm.idx[q * 6 + k];
            if (ix < q * 4 || ix >= q * 4 + 4) { quads = false; break; }
        }
    std::vector<uint8_t> border;                     // 1: takes the averaged normal
    if (!quads) {
        out_v.resize(nv);
        for (size_t i = 0; i < nv; ++i) out_v[i] = { lm.verts[i].p, lm.verts[i].n, glm::vec2(lm.verts[i].u, lm.verts[i].v) };
        out_i = lm.idx;
        border.assign(nv, 1);
    } else {
        out_v.reserve(nv * 4); out_i.reserve(ni * 9);
        for (size_t q = 0; q < nv / 4; ++q) {
            const auto& a = lm.verts[q * 4], &b = lm.verts[q * 4 + 1], &c = lm.verts[q * 4 + 2], &d = lm.verts[q * 4 + 3];
            const float lu = std::max(0.5f * (glm::distance(a.p, b.p) + glm::distance(d.p, c.p)), 1e-4f);
            const float lv = std::max(0.5f * (glm::distance(a.p, d.p) + glm::distance(b.p, c.p)), 1e-4f);
            const float iu = std::min(inset / lu, 0.33f), iv = std::min(inset / lv, 0.33f);
            const float us[4] = { 0.0f, iu, 1.0f - iu, 1.0f }, vs[4] = { 0.0f, iv, 1.0f - iv, 1.0f };
            const uint32_t base = uint32_t(out_v.size());
            for (int j = 0; j < 4; ++j)
                for (int i = 0; i < 4; ++i) {
                    const float u = us[i], v = vs[j];
                    auto bl = [&](auto A, auto B, auto C, auto Dd) { return glm::mix(glm::mix(A, B, u), glm::mix(Dd, C, u), v); };
                    Vertex x;
                    x.position = bl(a.p, b.p, c.p, d.p);
                    x.normal   = glm::normalize(bl(a.n, b.n, c.n, d.n));
                    x.uv       = bl(glm::vec2(a.u, a.v), glm::vec2(b.u, b.v), glm::vec2(c.u, c.v), glm::vec2(d.u, d.v));
                    out_v.push_back(x);
                    border.push_back((i == 0 || i == 3 || j == 0 || j == 3) ? 1 : 0);
                }
            // (the winding of the original quad: its first triangle tells which way it faces)
            const glm::vec3 gn = glm::cross(lm.verts[lm.idx[q * 6 + 1]].p - lm.verts[lm.idx[q * 6]].p,
                                            lm.verts[lm.idx[q * 6 + 2]].p - lm.verts[lm.idx[q * 6]].p);
            for (int j = 0; j < 3; ++j)
                for (int i = 0; i < 3; ++i) {
                    const uint32_t p00 = base + uint32_t(j * 4 + i), p10 = p00 + 1, p01 = p00 + 4, p11 = p01 + 1;
                    const glm::vec3 tn = glm::cross(out_v[p10].position - out_v[p00].position, out_v[p11].position - out_v[p00].position);
                    if (glm::dot(tn, gn) >= 0.0f) out_i.insert(out_i.end(), { p00, p10, p11, p00, p11, p01 });
                    else                          out_i.insert(out_i.end(), { p00, p11, p10, p00, p01, p11 });
                }
        }
    }
    // Average the distinct normals of everything meeting at a position (to the millimetre).
    struct Acc { glm::vec3 n[6]; int c = 0; };
    std::unordered_map<uint64_t, Acc> acc;
    auto key = [](glm::vec3 p) {
        const int64_t x = int64_t(std::llround(double(p.x) * 1000.0)), y = int64_t(std::llround(double(p.y) * 1000.0)), z = int64_t(std::llround(double(p.z) * 1000.0));
        return uint64_t(x * 73856093LL) ^ (uint64_t(y * 19349663LL) << 1) ^ (uint64_t(z * 83492791LL) << 2);
    };
    for (size_t i = 0; i < out_v.size(); ++i) {
        if (!border[i]) continue;
        Acc& A = acc[key(out_v[i].position)];
        bool seen = false;
        for (int k = 0; k < A.c; ++k) if (glm::dot(A.n[k], out_v[i].normal) > 0.995f) { seen = true; break; }
        if (!seen && A.c < 6) A.n[A.c++] = out_v[i].normal;
    }
    for (size_t i = 0; i < out_v.size(); ++i) {
        if (!border[i]) continue;
        const Acc& A = acc[key(out_v[i].position)];
        glm::vec3 s(0.0f);
        for (int k = 0; k < A.c; ++k) s += A.n[k];
        if (glm::dot(s, s) > 1e-8f) out_v[i].normal = glm::normalize(s);
    }
}
}  // namespace

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
        g.ssdm = lm.ssdm;
        if (g.ssdm > 0.0f) {
            // (its shell for the displacement pass: the same surface, normals rounded over the edges)
            std::vector<Vertex> sv; std::vector<uint32_t> si;
            build_ssdm_shell(lm, std::max(1.5f * g.ssdm, 0.05f), sv, si);
            g.ssdm_mesh = create_mesh_from_data(device_, allocator_, graphics_queue_, graphics_queue_family_,
                                                sv.data(), uint32_t(sv.size()), si.data(), uint32_t(si.size()));
        }
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
    for (const auto& ch : hw_chunks_) for (const auto& g : ch.second) mesh_parts_now_.push_back(&g);       // (a highway's roadside objects: highway_furniture.cpp)
    return mesh_parts_now_;
}

void VulkanEngine::destroy_castle_meshes() {
    destroy_sites();
    hw_objects_destroy();
    destroy_mesh_parts(castle_meshes_);
}

void VulkanEngine::destroy_mesh_parts(std::vector<CastleMeshGpu>& parts) {
    for (auto& g : parts) {
        if (g.blas) g_rt.destroy_as(device_, g.blas, nullptr);
        if (g.blas_buf) vmaDestroyBuffer(allocator_, g.blas_buf, g.blas_alloc);
        if (g.mesh.vertex_buffer) destroy_mesh(allocator_, g.mesh);
        if (g.ssdm_mesh.vertex_buffer) destroy_mesh(allocator_, g.ssdm_mesh);
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
        if (depth_only && g.tex_albedo == -3) continue;
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
            glm::mat4 mvp = vp;   // model = identity (world-space verts)
            vkCmdPushConstants(cmd, shadow_layout, VK_SHADER_STAGE_VERTEX_BIT,
                               0, sizeof(glm::mat4), &mvp);
        } else {
            PushConstants pc{};
            pc.mvp = vp;
            pc.model = glm::mat4(1.0f);
            pc.prev_mvp = prev_vp;
            // Static-mesh mode flag (see file header).
            // .y = wind sway (uv = weight, phase), .z = world triplanar.
            // .x <= 0: relief reduction (0 = full depth, -1 = flat).
            pc.grass_params = glm::vec4(glm::clamp(g.relief, 0.0f, 1.0f) - 1.0f, g.sway ? 1.0f : 0.0f,
                                        g.triplanar ? 1.0f : 0.0f, -1.0f);
            if (!depth_only) {
                pc.color = glm::vec4(tex_on ? g.tint : g.fallback, 1.0f);
                // Self-lit parts (lantern glass) glow faintly by day, fully at night.
                pc.emissive = g.emissive;
                if (g.emissive.a > 0.5f && g.emissive.a < 1.5f && g.emissive.r < 3.5f) {       // (flames burn at full strength by day too)
                    const float night = 1.0f - glm::smoothstep(4.0f, 16.0f, rt_.sun_pitch_deg);
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
