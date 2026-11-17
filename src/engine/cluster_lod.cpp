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

namespace qlike {

ClusterMesh build_cluster_mesh(const Vertex* verts, size_t vertex_count, const uint32_t* indices, size_t index_count) {
    ClusterMesh out;
    if (!verts || !indices || index_count < 3 * 256) return out;

    clodConfig cfg = clodDefaultConfig(128);
    clodMesh mesh{};
    mesh.indices = indices;
    mesh.index_count = index_count;
    mesh.vertex_count = vertex_count;
    mesh.vertex_positions = &verts[0].position.x;
    mesh.vertex_positions_stride = sizeof(Vertex);
    // Normals guide the simplifier (keep creases); uv is not weighted (triplanar rocks).
    mesh.vertex_attributes = &verts[0].normal.x;
    mesh.vertex_attributes_stride = sizeof(Vertex);
    static const float kWeights[3] = { 0.5f, 0.5f, 0.5f };
    mesh.attribute_weights = kWeights;
    mesh.attribute_count = 3;

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
            tris_at_depth[size_t(group.depth)] += uint32_t(c.index_count / 3);
        }
        return gid;
    });
    out.levels = max_depth + 1;
    if (!tris_at_depth.empty()) { out.base_tris = tris_at_depth.front(); out.coarsest_tris = tris_at_depth.back(); }
    return out;
}

}  // namespace qlike
