// Dynamic craters: big ground blasts (bombs, crashes) press a bowl with a
// raised, slightly irregular rim into the terrain. Everything that holds a
// copy of the ground height is updated BY REGION only (a sculpt-style full
// refresh would rebuild 2049^2 vertices, the whole heightfield and the
// horizon bake per explosion):
//   * terrain_data_.heights (CPU: gameplay height queries, sample_terrain_height)
//   * CLOD min/max pyramid (culling) for the touched nodes
//   * both height textures (absolute + sculpt delta), sub-rect copies
//   * the merged terrain mesh vertices (rows of the rect) + terrain BLAS
//     refit, so ray-traced sun shadows / GI see the hole (a raster surface
//     below the BLAS would self-shadow black)
//   * the Jolt heightfield (HeightFieldShape::SetHeights) - player, crates
//     and debris collide with the crater
//   * flowers / rocks: removed inside the bowl, re-seated on the slopes.
// Not on the castle plateau and not in the water.

#include "engine/vk_engine.h"
#include "engine/vk_engine/internal.h"
#include "engine/log.h"
#include "engine/physics_world.h"
#include "engine/terrain_clod.h"
#include "engine/vk_initializers.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>

namespace qlike {

void VulkanEngine::make_crater(glm::vec3 pos, float size) {
    TerrainData& td = terrain_data_;
    if (td.heights.empty() || size < 0.6f) return;
    // (Raymarched terrain too: its FBM adds the sculpt-delta height texture
    // this updates, and the baked height the grass / air-skip read.)
    // Only at the surface (callers pass ground blasts, but stay defensive).
    if (pos.y > sample_terrain_height(pos.x, pos.z) + 4.0f) return;
    if (std::max(std::abs(pos.x), std::abs(pos.z)) < 26.0f) return;      // castle plateau
    const float h0 = sample_terrain_height(pos.x, pos.z);
    if (h0 < rt_.water_level + 0.4f) return;                              // water / shoreline
    const auto t0 = std::chrono::steady_clock::now();
    // No GPU drain: the texture / vertex copies and the BLAS refit are
    // recorded into the next frame's command buffer (flush_pending_craters).
    // The CPU-side state (heights, physics, plants) changes right away.
    auto lap = [&, last = std::chrono::steady_clock::now()]() mutable {
        const auto n = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(n - last).count();
        last = n;
        return ms;
    };
    double tp[6] = {};
    tp[0] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();   // GPU drain
    lap();

    auto& R = spacejet_rng_state_;
    const float Rc   = 2.4f + 2.4f * size;          // bowl radius (m): bomb ~4.8, crash ~7.9
    // A real crater: a hole with a floor, a steep wall and a raised rim (it was a shallow dish,
    // a metre deep over five metres - a dent, not a crater).
    const float D    = 1.0f + 1.5f * size;          // depth of the floor: bomb ~2.5 m, crash ~4.4 m
    const float rimh = 0.30f * D;                   // raised ejecta rim
    const float Rmax = Rc * 1.7f;
    const float phase = frand_range(R, 0.0f, 6.28f);
    // Height change at a point: paraboloid bowl + Gaussian rim, both with
    // a little angular irregularity so it doesn't look stamped.
    auto delta = [&](float x, float z) {
        const float dx = x - pos.x, dz = z - pos.z;
        const float r = std::sqrt(dx * dx + dz * dz);
        if (r >= Rmax) return 0.0f;
        const float ang = std::atan2(dz, dx);
        const float wob = 1.0f + 0.12f * std::sin(ang * 3.0f + phase) + 0.07f * std::sin(ang * 7.0f + phase * 2.1f);
        const float rr = r / (Rc * wob);
        // Floor out to ~45 % of the radius (slightly dished), then the wall up to the rim.
        float d = 0.0f;
        if (rr < 1.0f) {
            const float t = std::clamp((rr - 0.45f) / 0.55f, 0.0f, 1.0f);
            const float wall = t * t * (3.0f - 2.0f * t);
            d = -D * (1.0f - wall) * (1.0f - 0.12f * rr * rr / 0.2f * 0.2f);
        }
        // The rim: sharp on the inside (the wall's lip), falling away gently outside.
        const float k = (rr - 1.02f) / (rr < 1.02f ? 0.10f : 0.34f);
        d += rimh * std::exp(-k * k);
        return d;
    };

    const int W = td.dim + 1;
    auto to_ix = [&](float w, float o) { return int(std::floor((w - o) / td.cell)); };
    const int x0 = std::clamp(to_ix(pos.x - Rmax, td.origin_x) - 1, 0, W - 1);
    const int x1 = std::clamp(to_ix(pos.x + Rmax, td.origin_x) + 2, 0, W - 1);
    const int z0 = std::clamp(to_ix(pos.z - Rmax, td.origin_z) - 1, 0, W - 1);
    const int z1 = std::clamp(to_ix(pos.z + Rmax, td.origin_z) + 2, 0, W - 1);
    const float floor_h = rt_.water_level + 0.15f;   // never dig below the water plane
    for (int iz = z0; iz <= z1; ++iz)
        for (int ix = x0; ix <= x1; ++ix) {
            const float wx = td.origin_x + float(ix) * td.cell, wz = td.origin_z + float(iz) * td.cell;
            float& h = td.heights[size_t(iz) * size_t(W) + size_t(ix)];
            const float d = delta(wx, wz);
            if (d < 0.0f) h = std::max(h + d, std::min(h, floor_h));
            else          h += d;
        }

    // Culling pyramid for the touched nodes.
    update_clod_pyramid_region(terrain_clod_pyramid_, td.heights.data(), td.dim, td.cell, 16.0f,
                               x0, x1, z0, z1);

    tp[1] = lap();   // heights + culling pyramid
    // ---- GPU: height textures (sub-rect) + terrain vertices + BLAS refit ----
    const int rw = x1 - x0 + 1, rh = z1 - z0 + 1;
    // Vertex rect one cell wider: normals of the neighbours change too.
    const int vx0 = std::max(0, x0 - 1), vx1 = std::min(W - 1, x1 + 1);
    const int vz0 = std::max(0, z0 - 1), vz1 = std::min(W - 1, z1 + 1);
    const int vw = vx1 - vx0 + 1, vh = vz1 - vz0 + 1;
    const VkDeviceSize tex_bytes = VkDeviceSize(rw) * rh * sizeof(float);
    const VkDeviceSize vtx_bytes = VkDeviceSize(vw) * vh * sizeof(Vertex);
    const VkDeviceSize stage_bytes = tex_bytes * 2 + vtx_bytes;
    VkBufferCreateInfo bci{};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = stage_bytes;
    bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
    aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    VkBuffer stage = VK_NULL_HANDLE; VmaAllocation stage_alloc = nullptr; VmaAllocationInfo sai{};
    if (vmaCreateBuffer(allocator_, &bci, &aci, &stage, &stage_alloc, &sai) != VK_SUCCESS) return;
    auto* full = static_cast<float*>(sai.pMappedData);
    float* dlt = full + size_t(rw) * size_t(rh);
    auto* vtx = reinterpret_cast<Vertex*>(dlt + size_t(rw) * size_t(rh));
    const bool has_base = terrain_baseline_heights_.size() == td.heights.size();
    for (int z = 0; z < rh; ++z)
        for (int x = 0; x < rw; ++x) {
            const size_t i = size_t(z0 + z) * size_t(W) + size_t(x0 + x);
            full[size_t(z) * rw + x] = td.heights[i];
            dlt[size_t(z) * rw + x] = has_base ? td.heights[i] - terrain_baseline_heights_[i] : td.heights[i];
        }
    // Same layout as gen_chunk_verts (interior grid, row-major, 1-cell
    // central-difference normal, uv = world xz).
    auto H = [&](int x, int z) {
        x = std::clamp(x, 0, W - 1); z = std::clamp(z, 0, W - 1);
        return td.heights[size_t(z) * size_t(W) + size_t(x)];
    };
    for (int z = 0; z < vh; ++z)
        for (int x = 0; x < vw; ++x) {
            const int gx = vx0 + x, gz = vz0 + z;
            const glm::vec3 p(td.origin_x + float(gx) * td.cell, H(gx, gz), td.origin_z + float(gz) * td.cell);
            const int xm = std::max(0, gx - 1), xp = std::min(W - 1, gx + 1);
            const int zm = std::max(0, gz - 1), zp = std::min(W - 1, gz + 1);
            const float dhdx = (H(xp, gz) - H(xm, gz)) / (float(xp - xm) * td.cell);
            const float dhdz = (H(gx, zp) - H(gx, zm)) / (float(zp - zm) * td.cell);
            vtx[size_t(z) * vw + x] = { p, glm::normalize(glm::vec3(-dhdx, 1.0f, -dhdz)), glm::vec2(p.x, p.z) };
        }
    vmaFlushAllocation(allocator_, stage_alloc, 0, VK_WHOLE_SIZE);

    {
        PendingCrater pcr{};
        pcr.stage = stage; pcr.stage_alloc = stage_alloc;
        pcr.x0 = x0; pcr.z0 = z0; pcr.rw = rw; pcr.rh = rh;
        pcr.vx0 = vx0; pcr.vz0 = vz0; pcr.vw = vw; pcr.vh = vh; pcr.W = W;
        pcr.tex_bytes = tex_bytes;
        pending_craters_.push_back(pcr);
    }
    tp[2] = lap();   // staging + GPU copies + BLAS refit (blocking submit)
    // ---- Physics: heightfield region in place ----
    if (physics_) physics_->update_heightfield_region(td.heights.data(), W, x0, z0, x1, z1);

    tp[3] = lap();   // physics heightfield
    // ---- Flowers / rocks: gone in the bowl, re-seated on slope and rim ----
    auto reseat = [&](std::vector<ScatterInstance>& cpu, void* mapped,
                      const std::vector<float>& dy) {
        if (!mapped) return;
        auto* gpu = static_cast<ScatterInstance*>(mapped);
        for (size_t i = 0; i < cpu.size(); ++i) {
            ScatterInstance& s = cpu[i];
            if (s.pos_scale.w <= 0.0f) continue;
            const float dx = s.pos_scale.x - pos.x, dz = s.pos_scale.z - pos.z;
            const float r2 = dx * dx + dz * dz;
            if (r2 >= Rmax * Rmax) continue;
            if (r2 < (0.85f * Rc) * (0.85f * Rc)) s.pos_scale.w = 0.0f;   // blasted away
            else s.pos_scale.y += delta(s.pos_scale.x, s.pos_scale.z);
            ScatterInstance g = s;
            if (scatter_.rm_surface && i < dy.size()) g.pos_scale.y += dy[i];
            gpu[i] = g;
        }
    };
    reseat(scatter_.inst_cpu, scatter_.inst_mapped, scatter_.inst_dy);
    reseat(scatter_.plant_cpu, scatter_.plant_mapped, scatter_.plant_dy);
    ++scatter_.plant_gen;

    tp[4] = lap();   // plant / stone reseat
    // Mud: remember the crater for the terrain shader (newest kMaxCraters).
    craters_.push_back(glm::vec4(pos.x, h0, pos.z, Rc));
    if (int(craters_.size()) > kMaxCraters) craters_.erase(craters_.begin());
    // No grass in the bowl or on the ejecta ring (R channel of the RG8
    // presence mask, 2 m texels; the raymarched blades and the green ground
    // tint both read it).
    if (!grass_mask_data_.empty()) {
        const int dim = int(std::lround(std::sqrt(double(grass_mask_data_.size() / 2))));
        const float cpm = float(dim) / 2048.0f;
        const float cxm = (pos.x + 1024.0f) * cpm, czm = (pos.z + 1024.0f) * cpm;
        const float rr = Rc * 1.35f * cpm;
        for (int mz = std::max(0, int(czm - rr) - 1); mz <= std::min(dim - 1, int(czm + rr) + 1); ++mz)
            for (int mx = std::max(0, int(cxm - rr) - 1); mx <= std::min(dim - 1, int(cxm + rr) + 1); ++mx) {
                const float d = std::sqrt((float(mx) + 0.5f - cxm) * (float(mx) + 0.5f - cxm) +
                                          (float(mz) + 0.5f - czm) * (float(mz) + 0.5f - czm));
                const float keep = glm::smoothstep(rr * 0.75f, rr + 0.5f, d);
                uint8_t& m = grass_mask_data_[(size_t(mz) * size_t(dim) + size_t(mx)) * 2];
                m = uint8_t(float(m) * keep);
            }
        // Upload just this rect with the crater's other GPU work.
        const int mx0 = std::max(0, int(cxm - rr) - 1), mx1 = std::min(dim - 1, int(cxm + rr) + 1);
        const int mz0 = std::max(0, int(czm - rr) - 1), mz1 = std::min(dim - 1, int(czm + rr) + 1);
        const int mw = mx1 - mx0 + 1, mh = mz1 - mz0 + 1;
        if (mw > 0 && mh > 0 && grass_mask_.image && !pending_craters_.empty()) {
            VkBufferCreateInfo mbci{};
            mbci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            mbci.size = VkDeviceSize(mw) * mh * 2;
            mbci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
            VmaAllocationCreateInfo maci{};
            maci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
            maci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
            VmaAllocationInfo mai{};
            PendingCrater& pcr = pending_craters_.back();
            if (vmaCreateBuffer(allocator_, &mbci, &maci, &pcr.mask_stage, &pcr.mask_alloc, &mai) == VK_SUCCESS) {
                auto* dst = static_cast<uint8_t*>(mai.pMappedData);
                for (int z = 0; z < mh; ++z)
                    std::memcpy(dst + size_t(z) * size_t(mw) * 2,
                                &grass_mask_data_[(size_t(mz0 + z) * size_t(dim) + size_t(mx0)) * 2], size_t(mw) * 2);
                vmaFlushAllocation(allocator_, pcr.mask_alloc, 0, VK_WHOLE_SIZE);
                pcr.mx0 = mx0; pcr.mz0 = mz0; pcr.mw = mw; pcr.mh = mh;
            }
        }
    }

    tp[5] = lap();   // grass mask
    drop_loose_objects(pos, Rc * 1.05f);
    log::infof("[crater] r %.1f m depth %.1f m at (%.0f, %.0f, %.0f): %dx%d cells, %.2f ms "
               "(drain %.2f, heights %.2f, gpu copies %.2f, physics %.2f, reseat %.2f, grass mask %.2f)",
               Rc, D, pos.x, pos.y, pos.z, rw, rh,
               std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(),
               tp[0], tp[1], tp[2], tp[3], tp[4], tp[5]);
}

} // namespace qlike

namespace qlike {
// Records the GPU half of every crater made since the last frame: height
// texture sub-rects, terrain vertex rows + BLAS refit, grass-mask sub-rect.
// Runs right after the frame-start barrier, i.e. ordered after everything
// the previous frame did with these resources.

// Lamp posts and gravestones inside a fresh crater have nothing to stand on. Each is taken out
// of the static world - its vertices in the shared mesh are collapsed, its collider and (for a
// post) its light are removed, the mesh's ray-tracing structure is rebuilt without it - and a
// physics body of its size is dropped in its place, to topple into the hole.
// Rare (a bomb must land within a few metres of one), so it simply waits for the GPU once.
void VulkanEngine::drop_loose_objects(glm::vec3 pos, float radius) {
    std::vector<size_t> hit;
    for (size_t i = 0; i < world_.loose.size(); ++i) {
        const auto& L = world_.loose[i];
        if (L.gone) continue;
        const glm::vec3 c = (L.lo + L.hi) * 0.5f;
        const float dx = c.x - pos.x, dz = c.z - pos.z;
        if (dx * dx + dz * dz < (radius + 0.3f) * (radius + 0.3f) && std::abs(L.lo.y - pos.y) < 8.0f) hit.push_back(i);
    }
    if (hit.empty()) return;
    vkDeviceWaitIdle(device_);
    std::vector<int> touched;
    for (size_t hi_ : hit) {
        auto& L = world_.loose[hi_];
        L.gone = true;
        const glm::vec3 c = (L.lo + L.hi) * 0.5f;
        // Collapse its vertices (zero-area triangles far below the ground).
        for (int k = 0; k < 3; ++k) {
            if (L.mesh[k] < 0 || size_t(L.mesh[k]) >= castle_of_level_mesh_.size() || L.v1[k] <= L.v0[k]) continue;
            const int gi = castle_of_level_mesh_[size_t(L.mesh[k])];
            if (gi < 0 || size_t(gi) >= castle_meshes_.size()) continue;
            CastleMeshGpu& g = castle_meshes_[size_t(gi)];
            if (L.v1[k] > g.mesh.vertex_count) continue;
            const VkDeviceSize bytes = VkDeviceSize(L.v1[k] - L.v0[k]) * sizeof(Vertex);
            VkBufferCreateInfo bci{};
            bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            bci.size = bytes; bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
            VmaAllocationCreateInfo aci{};
            aci.usage = VMA_MEMORY_USAGE_AUTO;
            aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
            VkBuffer stage = VK_NULL_HANDLE; VmaAllocation stage_alloc = nullptr; VmaAllocationInfo ai{};
            if (vmaCreateBuffer(allocator_, &bci, &aci, &stage, &stage_alloc, &ai) != VK_SUCCESS) continue;
            Vertex dead{ glm::vec3(c.x, L.lo.y - 400.0f, c.z), glm::vec3(0.0f, 1.0f, 0.0f), glm::vec2(0.0f) };
            Vertex* dst = static_cast<Vertex*>(ai.pMappedData);
            for (uint32_t v = 0; v < L.v1[k] - L.v0[k]; ++v) dst[v] = dead;
            vmaFlushAllocation(allocator_, stage_alloc, 0, VK_WHOLE_SIZE);
            vkinit::one_time_submit(device_, graphics_queue_, graphics_queue_family_, [&](VkCommandBuffer cb) {
                VkBufferCopy r{ 0, VkDeviceSize(L.v0[k]) * sizeof(Vertex), bytes };
                vkCmdCopyBuffer(cb, stage, g.mesh.vertex_buffer, 1, &r);
            });
            vmaDestroyBuffer(allocator_, stage, stage_alloc);
            if (std::find(touched.begin(), touched.end(), gi) == touched.end()) touched.push_back(gi);
        }
        // Collider and light.
        if (L.aabb >= 0 && size_t(L.aabb) < world_.aabbs.size())
            world_.aabbs[size_t(L.aabb)] = { glm::vec3(0.0f, -9000.0f, 0.0f), glm::vec3(0.0f, -9000.0f, 0.0f) };
        if (L.kind == 0)
            for (auto& l : lamp_lights_)
                if (std::abs(l.x - c.x) < 1.6f && std::abs(l.z - c.z) < 1.6f && l.y > L.lo.y && l.y < L.hi.y + 1.0f) l.y = -9000.0f;
        // The body that falls: a timber beam for a post, a slab for a stone.
        if (physics_) {
            const glm::vec3 he = L.kind == 0 ? glm::vec3(0.09f, (L.hi.y - L.lo.y) * 0.5f - 0.2f, 0.09f)
                                             : glm::max((L.hi - L.lo) * 0.5f * glm::vec3(0.8f, 0.9f, 0.8f), glm::vec3(0.06f));
            auto& R = spacejet_rng_state_;
            const glm::vec3 tilt(frand_range(R, -0.22f, 0.22f), frand_range(R, 0.0f, 6.28f), frand_range(R, -0.22f, 0.22f));
            const uint32_t id = physics_->add_dynamic_box(c + glm::vec3(0.0f, 0.25f, 0.0f), he, tilt, L.kind == 0 ? 600.0f : 2400.0f);
            if (id != 0) {
                DynamicProp p{};
                p.body_id = id;
                p.jolt_handle = physics_->handle_of(id);
                p.full_size = he * 2.0f;
                p.tex_albedo = p.tex_normal = (L.kind == 0) ? kTexTimber : kTexGrave;
                p.uv_scale = 1.0f;
                p.color = (L.kind == 0) ? glm::vec4(0.85f, 0.80f, 0.74f, 1.0f) : glm::vec4(0.86f, 0.93f, 1.0f, 1.0f);
                p.fallback_color = (L.kind == 0) ? glm::vec4(0.35f, 0.24f, 0.15f, 1.0f) : glm::vec4(0.42f, 0.40f, 0.36f, 1.0f);
                dyn_props_.push_back(p);
            }
        }
    }
    near_static_src_[0] = SIZE_MAX;             // the player's colliders: taken again
    // The touched meshes' ray-tracing structures: built again from the new vertices.
    for (int gi : touched) {
        CastleMeshGpu& g = castle_meshes_[size_t(gi)];
        if (g.blas) g_rt.destroy_as(device_, g.blas, nullptr);
        if (g.blas_buf) vmaDestroyBuffer(allocator_, g.blas_buf, g.blas_alloc);
        g.blas = VK_NULL_HANDLE; g.blas_buf = VK_NULL_HANDLE; g.blas_alloc = nullptr; g.blas_addr = 0;
    }
    if (!touched.empty()) build_mesh_part_blas(castle_meshes_);
    log::infof("[crater] %zu loose object(s) lost their ground and fell", hit.size());
}

void VulkanEngine::flush_pending_craters(VkCommandBuffer cb) {
    if (pending_craters_.empty()) return;
    for (PendingCrater& c : pending_craters_) {
        auto copy_tex = [&](VkImage img, VkBuffer src, VkDeviceSize off, int ox, int oz, int w, int h) {
            if (!img || !src) return;
            vkinit::transition_image(cb, img, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            VkBufferImageCopy r{};
            r.bufferOffset = off;
            r.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            r.imageSubresource.layerCount = 1;
            r.imageOffset = { ox, oz, 0 };
            r.imageExtent = { uint32_t(w), uint32_t(h), 1 };
            vkCmdCopyBufferToImage(cb, src, img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &r);
            vkinit::transition_image(cb, img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        };
        copy_tex(terrain_height_full_image_, c.stage, 0, c.x0, c.z0, c.rw, c.rh);
        copy_tex(terrain_height_image_, c.stage, c.tex_bytes, c.x0, c.z0, c.rw, c.rh);
        if (c.mask_stage) copy_tex(grass_mask_.image, c.mask_stage, 0, c.mx0, c.mz0, c.mw, c.mh);
        if (terrain_mesh_.vertex_buffer) {
            std::vector<VkBufferCopy> rows(static_cast<size_t>(c.vh));
            for (int z = 0; z < c.vh; ++z)
                rows[size_t(z)] = { c.tex_bytes * 2 + VkDeviceSize(z) * c.vw * sizeof(Vertex),
                                    (VkDeviceSize(c.vz0 + z) * c.W + c.vx0) * sizeof(Vertex),
                                    VkDeviceSize(c.vw) * sizeof(Vertex) };
            vkCmdCopyBuffer(cb, c.stage, terrain_mesh_.vertex_buffer, uint32_t(rows.size()), rows.data());
            VkMemoryBarrier2 mb{};
            mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
            mb.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
            mb.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
            mb.dstStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR | VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT;
            mb.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT;
            VkDependencyInfo di{};
            di.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            di.memoryBarrierCount = 1; di.pMemoryBarriers = &mb;
            vkCmdPipelineBarrier2(cb, &di);
            VkBuffer scratch = VK_NULL_HANDLE; VmaAllocation scratch_alloc = nullptr;
            record_terrain_blas_refit(cb, &scratch, &scratch_alloc);
            retire_buffer(scratch, scratch_alloc);
            mb.srcStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
            mb.srcAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
            mb.dstStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR |
                              VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            mb.dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR;
            vkCmdPipelineBarrier2(cb, &di);
        }
        retire_buffer(c.stage, c.stage_alloc);
        retire_buffer(c.mask_stage, c.mask_alloc);
    }
    pending_craters_.clear();
    // The TLAS caches the terrain BLAS bounds at build time (a crater can
    // dig below them): force a full rebuild next frame.
    prev_dyn_signature_ = 0;
    prev_tlas_n_ = UINT32_MAX;
}

void VulkanEngine::collect_retired_buffers(bool all) {
    for (size_t i = 0; i < retired_buffers_.size();) {
        if (all || retired_buffers_[i].frame + uint64_t(kFrameOverlap) + 2 <= frame_number_) {
            vmaDestroyBuffer(allocator_, retired_buffers_[i].buf, retired_buffers_[i].alloc);
            retired_buffers_[i] = retired_buffers_.back();
            retired_buffers_.pop_back();
        } else {
            ++i;
        }
    }
}
}  // namespace qlike
