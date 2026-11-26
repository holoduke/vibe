// The engine's private state structs (vk_engine/state/*.h): subsystems whose variables only
// their own files need. vk_engine.h holds a pointer to each; this file is the one place
// that knows them all, to create and free them.

#include "engine/vk_engine.h"
#include "engine/vk_engine/state/aim_spread.h"
#include "engine/vk_engine/state/atmosphere.h"
#include "engine/vk_engine/state/bloom.h"
#include "engine/vk_engine/state/clouds.h"
#include "engine/vk_engine/state/cluster_draw.h"
#include "engine/vk_engine/state/car_traffic.h"
#include "engine/vk_engine/state/clutter_shadow.h"
#include "engine/vk_engine/state/compose.h"
#include "engine/vk_engine/state/post_fx.h"
#include "engine/vk_engine/state/dlss.h"
#include "engine/vk_engine/state/environment.h"
#include "engine/vk_engine/state/fog.h"
#include "engine/vk_engine/state/fsr2.h"
#include "engine/vk_engine/state/fsr3.h"
#include "engine/vk_engine/state/fx.h"
#include "engine/vk_engine/state/glass.h"
#include "engine/vk_engine/state/lens_fx.h"
#include "engine/vk_engine/state/multiplayer.h"
#include "engine/vk_engine/state/plant_gpu.h"
#include "engine/vk_engine/state/regions.h"
#include "engine/vk_engine/state/rr_guides.h"
#include "engine/vk_engine/state/sound_env.h"
#include "engine/vk_engine/state/spacejets.h"
#include "engine/vk_engine/state/ssdm.h"
#include "engine/vk_engine/state/static_clusters.h"
#include "engine/vk_engine/state/streamline.h"
#include "engine/vk_engine/state/sun_shadow.h"
#include "engine/vk_engine/state/svgf.h"
#include "engine/vk_engine/state/taa.h"
#include "engine/vk_engine/state/taau.h"
#include "engine/vk_engine/state/terrain_bake.h"
#include "engine/vk_engine/state/ground_fx.h"
#include "engine/vk_engine/state/terrain_patches.h"
#include "engine/vk_engine/state/terrain_raymarch.h"
#include "engine/vk_engine/state/voxel.h"

namespace vibe {

void VulkanEngine::alloc_private_state() {
    atmo_ = new AtmosphereState();
    spread_ = new AimSpreadState();
    sound_env_ = new SoundEnvState();
    bloom_ = new BloomState();
    cloud_ = new CloudState();
    cl_ = new ClusterDrawState();
    clutter_ = new ClutterShadowState();
    cars_x_ = new CarTrafficState();
    compose_ = new ComposeState();
    pfx_ = new PostFxState();
    dlss_ = new DlssState();
    env_ = new EnvState();
    fog_ = new FogState();
    fsr2_ = new Fsr2State();
    fsr3_ = new Fsr3State();
    fx_ = new FxState();
    glass_ = new GlassState();
    lens_fx_ = new LensFxState();
    mp_ = new MultiplayerState();
    pg_ = new PlantGpuState();
    rr_ = new RrGuideState();
    spacejet_ = new SpacejetState();
    ssdm_ = new SsdmState();
    scl_ = new StaticClusterState();
    sl_ = new StreamlineState();
    sun_ = new SunShadowState();
    svgf_ = new SvgfState();
    taa_ = new TaaState();
    taau_ = new TaauState();
    tr_ = new TerrainRaymarchState();
    tp_ = new TerrainPatchState();
    tb_ = new TerrainBakeState();
    gfx_ = new GroundFxState();
    rg_ = new RegionState();
    voxel_ = new VoxelState();
}

void VulkanEngine::free_private_state() {
    delete atmo_; atmo_ = nullptr;
    delete spread_; spread_ = nullptr;
    delete sound_env_; sound_env_ = nullptr;
    delete bloom_; bloom_ = nullptr;
    delete cloud_; cloud_ = nullptr;
    delete cl_; cl_ = nullptr;
    delete clutter_; clutter_ = nullptr;
    delete cars_x_; cars_x_ = nullptr;
    delete compose_; compose_ = nullptr;
    delete pfx_; pfx_ = nullptr;
    delete dlss_; dlss_ = nullptr;
    delete env_; env_ = nullptr;
    delete fog_; fog_ = nullptr;
    delete fsr2_; fsr2_ = nullptr;
    delete fsr3_; fsr3_ = nullptr;
    delete fx_; fx_ = nullptr;
    delete glass_; glass_ = nullptr;
    delete lens_fx_; lens_fx_ = nullptr;
    delete mp_; mp_ = nullptr;
    delete pg_; pg_ = nullptr;
    delete rr_; rr_ = nullptr;
    delete spacejet_; spacejet_ = nullptr;
    delete ssdm_; ssdm_ = nullptr;
    delete scl_; scl_ = nullptr;
    delete sl_; sl_ = nullptr;
    delete sun_; sun_ = nullptr;
    delete svgf_; svgf_ = nullptr;
    delete taa_; taa_ = nullptr;
    delete taau_; taau_ = nullptr;
    delete tr_; tr_ = nullptr;
    delete tp_; tp_ = nullptr;
    delete tb_; tb_ = nullptr;
    delete gfx_; gfx_ = nullptr;
    delete rg_; rg_ = nullptr;
    delete voxel_; voxel_ = nullptr;
}

} // namespace vibe
