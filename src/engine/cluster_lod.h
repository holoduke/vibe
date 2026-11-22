// Cluster level of detail ("Nanite-style"): a mesh is cut into clusters of
// up to ~128 triangles; groups of neighbouring clusters are simplified
// together (their outer border locked, so neighbours always fit) and cut
// into clusters again, level after level, down to a handful of triangles.
// At run time every cluster is tested on its own: it is drawn when it is
// detailed enough for its size on screen while the coarser version above it
// is not. The cut through the hierarchy is seamless and changes gradually
// across one object - detail follows the distance to each PART of a big
// mesh, not to the object as a whole.
//
// Built with meshoptimizer's clusterlod.h. No Vulkan in here; the renderer
// side is vk_engine/cluster_draw.cpp.
#pragma once

#include "engine/mesh.h"

#include <glm/vec4.hpp>

#include <cstdint>
#include <vector>

namespace vibe {

// GPU layout (std430, 64 bytes) - keep in sync with cluster_cull.comp.
struct ClusterGpu {
    glm::vec4 bounds;      // xyz centre, w radius of this cluster's own triangles (culling)
    glm::vec4 self;        // xyz centre, w radius of the group this cluster was simplified FROM (its own error sphere)
    glm::vec4 parent;      // xyz centre, w radius of the group this cluster is IN (the coarser version above it)
    float     self_error;  // object-space error of this cluster's geometry (0 = original triangles)
    float     parent_error;// error of the coarser version (huge for the coarsest level: always "too coarse")
    uint32_t  first_index; // into the mesh's cluster index buffer
    uint32_t  index_count;
};

struct ClusterMesh {
    std::vector<ClusterGpu> clusters;
    std::vector<uint32_t>   indices;       // all clusters' triangles, back to back (refer to the source vertices)
    std::vector<uint8_t>    depth;         // per cluster: its level (0 = the original triangles)
    int      levels = 0;
    uint32_t base_tris = 0;                // triangles of the finest level
    uint32_t coarsest_tris = 0;            // triangles of the coarsest level
};

// Build the hierarchy for one mesh. Empty result if the mesh is too small to be worth it.
// `uv_weight` > 0: the simplifier also keeps texture coordinates in place (meshes with a mapped texture).
ClusterMesh build_cluster_mesh(const Vertex* verts, size_t vertex_count, const uint32_t* indices, size_t index_count,
                               float uv_weight = 0.0f);
// The same, read from / written to `cache/cluster_<hash>.bin` (the hierarchy depends only on the mesh).
ClusterMesh build_cluster_mesh_cached(const Vertex* verts, size_t vertex_count, const uint32_t* indices, size_t index_count,
                                      float uv_weight = 0.0f);

}  // namespace vibe
