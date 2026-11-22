// The engine's private state structs (vk_engine/state/*.h): subsystems whose variables only
// their own files need. vk_engine.h holds a pointer to each; this file is the one place
// that knows them all, to create and free them.

#include "engine/vk_engine.h"
#include "engine/vk_engine/state/ssdm.h"
#include "engine/vk_engine/state/cluster_draw.h"
#include "engine/vk_engine/state/plant_gpu.h"
#include "engine/vk_engine/state/static_clusters.h"
#include "engine/vk_engine/state/rr_guides.h"

namespace vibe {

void VulkanEngine::alloc_private_state() {
    ssdm_ = new SsdmState();
    cl_ = new ClusterDrawState();
    pg_ = new PlantGpuState();
    scl_ = new StaticClusterState();
    rr_ = new RrGuideState();
}

void VulkanEngine::free_private_state() {
    delete ssdm_; ssdm_ = nullptr;
    delete cl_; cl_ = nullptr;
    delete pg_; pg_ = nullptr;
    delete scl_; scl_ = nullptr;
    delete rr_; rr_ = nullptr;
}

} // namespace vibe
