// Cluster LOD hierarchy builder - see cluster_lod.h.
#include "engine/cluster_lod.h"

#include <meshoptimizer.h>
#ifdef min
#undef min      // (windows.h macros; clusterlod.h uses std::min / std::max)
#endif
#ifdef max
#undef max
#endif
#define CLUSTERLOD_IMPLEMENTATION
#include <clusterlod.h>

#include <algorithm>
#include <cfloat>

namespace vibe {

ClusterMesh build_cluster_mesh(const Vertex* verts, size_t vertex_count, const uint32_t* indices, size_t index_count,
                               float uv_weight) {
    ClusterMesh out;
    if (!verts || !indices || index_count < 3 * 256) return out;

    clodConfig cfg = clodDefaultConfig(128);
    cfg.optimize_bounds = true;        // cluster bounds from its own triangles (they are culled one by one)
    clodMesh mesh{};
    mesh.indices = indices;
    mesh.index_count = index_count;
    mesh.vertex_count = vertex_count;
    mesh.vertex_positions = &verts[0].position.x;
    mesh.vertex_positions_stride = sizeof(Vertex);
    // Normals guide the simplifier (keep creases); uv is not weighted (triplanar rocks).
    mesh.vertex_attributes = &verts[0].normal.x;
    mesh.vertex_attributes_stride = sizeof(Vertex);
    // (normal and uv follow each other in Vertex)
    const float weights[5] = { 0.5f, 0.5f, 0.5f, uv_weight, uv_weight };
    mesh.attribute_weights = weights;
    mesh.attribute_count = uv_weight > 0.0f ? 5 : 3;

    std::vector<clodBounds> groups;         // simplified bounds per group id
    int max_depth = 0;
    std::vector<uint32_t> tris_at_depth;
    clodBuild(cfg, mesh, [&](clodGroup group, const clodCluster* clusters, size_t cluster_count) -> int {
        const int gid = int(groups.size());
        groups.push_back(group.simplified);
        max_depth = std::max(max_depth, group.depth);
        if (size_t(group.depth) >= tris_at_depth.size()) tris_at_depth.resize(size_t(group.depth) + 1, 0u);
        for (size_t i = 0; i < cluster_count; ++i) {
            const clodCluster& c = clusters[i];
            ClusterGpu g{};
            g.bounds = glm::vec4(c.bounds.center[0], c.bounds.center[1], c.bounds.center[2], c.bounds.radius);
            // Its own error: that of the group it was simplified from (0 for original triangles).
            if (c.refined >= 0 && size_t(c.refined) < groups.size()) {
                const clodBounds& b = groups[size_t(c.refined)];
                g.self = glm::vec4(b.center[0], b.center[1], b.center[2], b.radius);
                g.self_error = b.error;
            } else {
                g.self = glm::vec4(c.bounds.center[0], c.bounds.center[1], c.bounds.center[2], c.bounds.radius);
                g.self_error = 0.0f;
            }
            g.parent = glm::vec4(group.simplified.center[0], group.simplified.center[1], group.simplified.center[2],
                                 group.simplified.radius);
            g.parent_error = group.simplified.error >= FLT_MAX * 0.5f ? 1e30f : group.simplified.error;
            g.first_index = uint32_t(out.indices.size());
            g.index_count = uint32_t(c.index_count);
            out.indices.insert(out.indices.end(), c.indices, c.indices + c.index_count);
            out.clusters.push_back(g);
            out.depth.push_back(uint8_t(std::min(group.depth, 255)));
            tris_at_depth[size_t(group.depth)] += uint32_t(c.index_count / 3);
        }
        return gid;
    });
    out.levels = max_depth + 1;
    if (!tris_at_depth.empty()) { out.base_tris = tris_at_depth.front(); out.coarsest_tris = tris_at_depth.back(); }
    return out;
}

}  // namespace vibe

// ---------------------------------------------------------------------
// Disk cache: the hierarchy of a mesh only depends on the mesh, and building
// it for the castle takes over a second. Keyed by a hash of the input.
// ---------------------------------------------------------------------
#include <cstdio>
#include <filesystem>
#include <string>

namespace vibe {

namespace {
uint64_t fnv1a(const void* data, size_t n, uint64_t h) {
    const unsigned char* p = static_cast<const unsigned char*>(data);
    for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ull; }
    return h;
}
constexpr uint32_t kCacheVersion = 4;      // bump when the builder or ClusterGpu changes
}  // namespace

ClusterMesh build_cluster_mesh_cached(const Vertex* verts, size_t vertex_count, const uint32_t* indices, size_t index_count,
                                      float uv_weight) {
    if (!verts || !indices || index_count < 3 * 256) return {};
    uint64_t h = 1469598103934665603ull;
    h = fnv1a(verts, vertex_count * sizeof(Vertex), h);
    h = fnv1a(indices, index_count * sizeof(uint32_t), h);
    h = fnv1a(&uv_weight, sizeof(uv_weight), h);
    char name[64];
    std::snprintf(name, sizeof(name), "cache/cluster_%016llx.bin", static_cast<unsigned long long>(h));
    struct Header { uint32_t version, clusters, indices; int32_t levels; uint32_t base_tris, coarsest_tris; };
    if (FILE* fp = std::fopen(name, "rb")) {
        Header hd{};
        ClusterMesh m;
        bool ok = std::fread(&hd, sizeof(hd), 1, fp) == 1 && hd.version == kCacheVersion && hd.clusters < (1u << 24) && hd.indices < (1u << 28);
        if (ok) {
            m.clusters.resize(hd.clusters); m.indices.resize(hd.indices); m.depth.resize(hd.clusters);
            ok = std::fread(m.clusters.data(), sizeof(ClusterGpu), hd.clusters, fp) == hd.clusters &&
                 std::fread(m.indices.data(), sizeof(uint32_t), hd.indices, fp) == hd.indices &&
                 std::fread(m.depth.data(), 1, hd.clusters, fp) == hd.clusters;
            m.levels = hd.levels; m.base_tris = hd.base_tris; m.coarsest_tris = hd.coarsest_tris;
        }
        std::fclose(fp);
        if (ok) return m;
    }
    ClusterMesh m = build_cluster_mesh(verts, vertex_count, indices, index_count, uv_weight);
    if (m.clusters.empty()) return m;
    std::error_code ec;
    std::filesystem::create_directories("cache", ec);
    if (FILE* fp = std::fopen(name, "wb")) {
        Header hd{ kCacheVersion, uint32_t(m.clusters.size()), uint32_t(m.indices.size()), m.levels, m.base_tris, m.coarsest_tris };
        std::fwrite(&hd, sizeof(hd), 1, fp);
        std::fwrite(m.clusters.data(), sizeof(ClusterGpu), m.clusters.size(), fp);
        std::fwrite(m.indices.data(), sizeof(uint32_t), m.indices.size(), fp);
        std::fwrite(m.depth.data(), 1, m.depth.size(), fp);
        std::fclose(fp);
    }
    return m;
}

}  // namespace vibe
