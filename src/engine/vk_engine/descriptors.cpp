// Scene descriptor set + UBO update. The TLAS / materials / prev_transforms
// SSBOs and per-texture sampler arrays all live behind one descriptor set
// (scene_desc_set_) bound by every world raster pipeline. Once init_rt()
// produces the TLAS handle, helpers.cpp's write_scene_descriptors_once()
// fills the bindings; from then on update_scene_ubo() is the only per-frame
// descriptor work (memcpy into a host-mapped UBO buffer).

#include "engine/vk_engine/internal.h"
#include "engine/vk_engine/state/atmosphere.h"
#include "engine/vk_engine/state/voxel.h"
#include "engine/vk_engine/state/clutter_shadow.h"
#include "engine/vk_engine/state/clouds.h"
#include "engine/vk_engine/state/terrain_raymarch.h"
#include "engine/vk_engine/state/environment.h"
#include "engine/vk_engine/state/sun_shadow.h"
#include "engine/globe.h"
#include "engine/voxel/voxel_world.h"

#include <GLFW/glfw3.h>
#include <algorithm>
#include <cmath>
#include <cstring>

namespace vibe {

void VulkanEngine::init_descriptors() {
    // Scene UBO buffer.
    {
        VkBufferCreateInfo bci{
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .pNext = nullptr, .flags = 0,
            .size = sizeof(SceneUBO),
            .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,   // dst of the frame upload ring
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0, .pQueueFamilyIndices = nullptr,
        };
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;      // (GPU memory the CPU can write: every shader reads this, per pixel)
        aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                    VMA_ALLOCATION_CREATE_MAPPED_BIT;
        vk_check(vmaCreateBuffer(allocator_, &bci, &aci,
                                 &scene_ubo_buffer_, &scene_ubo_alloc_, nullptr),
                 "scene ubo buffer");
    }

    // Descriptor pool: 1 UBO + 1 AS + 2 SSBO (materials, prev_transforms) +
    // 2 sampler arrays (albedo[N], normal[N]).
    VkDescriptorPoolSize sizes[] = {
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 },
        { VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1 },
        // 2 originals (materials, prev_transforms) + 2 ReSTIR reservoirs
        // + 2 voxel SSBOs (brick atlas + shape directory, bindings 24/25).
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 7 },      // + the bone palette (33)
        // SVGF GI denoiser storage images: 1 raw gi (binding 19) +
        // 2 history ping-pong (bindings 20, 21) +
        // 2 variance-moments ping-pong (bindings 22, 23 — 4a-deep).
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 5 },
        // +1 for the heightmap shadow texture at binding 6.
        // +1 for the sun shadow map at binding 7 (sampler2DShadow).
        // +1 for the raw heightmap texture at binding 8 (R32_SFLOAT) —
        // sampled by terrain_raymarch.frag so the procedural raymarched
        // terrain follows the gameplay heightmap shape.
        // +3 for the low-res raymarch upscale targets (bindings 9..11):
        // color, motion vector, depth — sampled by the compose pass
        // when terrain_raymarch_scale < 1.0.
        // +kSpomMaterialCount for the SPOM height-map array (binding 12).
        // +1 for the grass-density mask R8 texture (binding 13).
        // +1 for the fog wisp R8 texture (binding 14).
        // +1 (now 9) for the FULL terrain-height R32F texture at
        // binding 17 — the water path samples the real mesh surface.
        // +1 (now 10) for the half-rate shadow R8 image at binding 18
        // — cube.frag's bilateral upsample consumer (roadmap item #4
        // Phase 3) samples it in place of the inline PCSS trace.
        // +1 (now 11) for the terrain aux bake (curvature + sky-vis)
        // at binding 26 — AAA terrain Phases 4/5.
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            kTextureCount * 2 + 30 + kSpomMaterialCount },      // + the terrain's low-resolution march (34), its second low-resolution picture (36, 37)
    };
    VkDescriptorPoolCreateInfo pci{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .pNext = nullptr, .flags = 0,
        .maxSets = 1,
        .poolSizeCount = 5,
        .pPoolSizes = sizes,
    };
    vk_check(vkCreateDescriptorPool(device_, &pci, nullptr, &scene_desc_pool_),
             "scene desc pool");

    // Bindings: 0=UBO, 1=TLAS, 2=materials, 3=albedo[N], 4=normal[N],
    //           5=prev_transforms, 6=heightmap shadow, 7=sun shadow map,
    //           8=raw heightmap (R32_SFLOAT) for terrain_raymarch.frag,
    //           9=LR raymarch color, 10=LR raymarch motion-vec, 11=LR
    //              raymarch depth — read by the compose pass when
    //              terrain_raymarch_scale < 1.
    //          15=ReSTIR reservoir SSBO (read prev), 16=write cur.
    //              Owned by restir.cpp; cube.frag's GI loop reads/writes.
    VkDescriptorSetLayoutBinding bindings[46]{};
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings[0].descriptorCount = 1;
    // TESS_EVAL flag added so terrain_tess.tese can read scene.spom_params.z
    // (UI-driven ground tile metres). Without it the .tese hardcodes
    // kGroundTile=2.5 and the displacement detaches from the texture
    // tiling whenever the user moves the "ground material repeat" slider.
    bindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT |
                             VK_SHADER_STAGE_FRAGMENT_BIT |
                             VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;

    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    bindings[2].binding = 2;
    bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[2].descriptorCount = 1;
    bindings[2].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    bindings[3].binding = 3;
    bindings[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[3].descriptorCount = kTextureCount;
    bindings[3].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    bindings[4].binding = 4;
    bindings[4].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[4].descriptorCount = kTextureCount;
    bindings[4].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    bindings[5].binding = 5;
    bindings[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[5].descriptorCount = 1;
    bindings[5].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    // Binding 6: pre-baked heightmap sun-shadow R8 texture. Sampled by
    // grass.vert AND cube.frag — the latter uses it as a distant-terrain
    // shadow fallback so the BLAS-vs-LOD-raster mismatch can't produce
    // false-hit shadow rays on far ridges.
    bindings[6].binding = 6;
    bindings[6].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[6].descriptorCount = 1;
    bindings[6].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

    // Binding 7: sun shadow map (single-cascade D32). Sampled by grass.vert
    // as a sampler2DShadow for hardware PCF + comparison. Replaces the
    // heightmap-bake path going forward (binding 6 stays for fallback).
    bindings[7].binding = 7;
    bindings[7].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[7].descriptorCount = 1;
    bindings[7].stageFlags = VK_SHADER_STAGE_VERTEX_BIT |
                              VK_SHADER_STAGE_FRAGMENT_BIT;

    // Binding 8: raw heightmap as R32_SFLOAT, sampled by
    // terrain_raymarch.frag so the procedural ray-marched terrain
    // follows the gameplay heightmap shape exactly (castle / cubes
    // sit correctly on the visible surface).
    bindings[8].binding = 8;
    bindings[8].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[8].descriptorCount = 1;
    bindings[8].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    // Bindings 9..11: low-res raymarch upscale targets, sampled by the
    // compose pass when terrain_raymarch_scale < 1. Color and motion
    // are HDR / SFLOAT; depth is R32 sampled as a regular texture so
    // the compose can write gl_FragDepth = upscaled depth.
    for (int b = 9; b <= 11; ++b) {
        bindings[b].binding = static_cast<uint32_t>(b);
        bindings[b].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[b].descriptorCount = 1;
        bindings[b].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    }

    // Binding 12: SPOM displacement map array. cube.frag samples one entry
    // per parallax-using material (currently 5: castle walls, keep walls,
    // courtyard floor, keep interior floor, terrain rocky-grass). Array
    // index = SPOM material ID returned by cube.frag::height_idx_for_albedo().
    // Also visible to terrain_tess.tese: the rocky-grass slot drives
    // per-vertex displacement at the grass band, matching cube.frag's
    // splat UV so the bumps line up with the textured pixels.
    bindings[12].binding = 12;
    bindings[12].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[12].descriptorCount = kSpomMaterialCount;
    bindings[12].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT |
                              VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT |
                              // terrain.vert (non-tess path) now also
                              // displaces vertices from slot 5 so the
                              // rocky relief survives even when tess
                              // is disabled.
                              VK_SHADER_STAGE_VERTEX_BIT;

    // Binding 13: grass-density mask (R8). Both the raymarched grass
    // pass and the terrain raymarch's getMaterial sample this for
    // eligibility — replaces the 9-cell noised() storm and ensures
    // grass blades and green ground tint always agree.
    bindings[13].binding = 13;
    bindings[13].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[13].descriptorCount = 1;
    bindings[13].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_VERTEX_BIT;      // (vertex: the grass blades are cut where the mask has none - tuft_cd.glsl)

    // Binding 14: fog wisp pattern (R8 256², REPEAT). One textureLod()
    // replaces the 3-octave noise2 calls in terrain_raymarch.frag's
    // fog density + godray probe loops.
    bindings[14].binding = 14;
    bindings[14].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[14].descriptorCount = 1;
    bindings[14].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    // Bindings 15/16: ReSTIR GI reservoir SSBOs (ping-pong). Session 2
    // ships the bindings + writes; session 3 (temporal) starts reading
    // binding 15. Layout matches the GLSL Reservoir struct in cube.frag.
    bindings[15].binding = 15;
    bindings[15].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[15].descriptorCount = 1;
    bindings[15].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    bindings[16].binding = 16;
    bindings[16].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[16].descriptorCount = 1;
    bindings[16].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    // Binding 17: FULL baked terrain height (R32_SFLOAT, dim+1²) — the
    // exact array the CDLOD mesh + physics use. terrain_raymarch.frag's
    // water path samples this in mesh-terrain mode so shore /
    // showthrough / reflection / self-shadow follow the real mesh
    // surface with zero procedural FBM evaluation.
    bindings[17].binding = 17;
    bindings[17].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[17].descriptorCount = 1;
    // VERTEX added for the quadtree CLOD terrain: terrain.vert fetches
    // absolute height + gradient normals from this texture instead of
    // carrying per-chunk vertex buffers.
    bindings[17].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT |
                              VK_SHADER_STAGE_VERTEX_BIT;

    // Binding 18: half-rate sun-shadow R8 image (roadmap item #4
    // Phase 3). Produced by render_world_shadow_lr_pass at
    // render_extent_/2; consumed by cube.frag's bilateral upsample
    // for brush/dyn surfaces when rt_.half_rate_shadows is on. Written
    // by init_shadow_lr_pipeline and recreate_shadow_lr.
    bindings[18].binding = 18;
    bindings[18].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[18].descriptorCount = 1;
    bindings[18].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    // Binding 19: SVGF GI denoiser storage image. cube.frag writes raw
    // per-pixel GI irradiance via imageStore; sessions 2+ (in
    // docs/svgf_plan.md) add the temporal-accumulation and à-trous
    // passes that read it. R16G16B16A16F at render_extent_.
    bindings[19].binding = 19;
    bindings[19].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[19].descriptorCount = 1;
    bindings[19].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    // Bindings 20 / 21: SVGF history ping-pong pair (Session 2). cube.frag
    // selects read vs write by frame parity (scene.rt_flags.w & 1) —
    // same race-free model as the ReSTIR ring buffer.
    bindings[20].binding = 20;
    bindings[20].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[20].descriptorCount = 1;
    bindings[20].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    bindings[21].binding = 21;
    bindings[21].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[21].descriptorCount = 1;
    bindings[21].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    // Bindings 22 / 23: SVGF variance-moments ping-pong pair (4a-deep).
    // R32G32 SFLOAT per pixel: .r = EMA mean of scene luminance,
    // .g = EMA mean of luminance^2. svgf_moments.frag reads the prev-
    // frame slot via motion-vector reprojection and writes the new slot
    // each frame; svgf_atrous.frag reads the freshly-written slot to
    // compute per-pixel variance for its luminance edge-stop weight.
    bindings[22].binding = 22;
    bindings[22].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[22].descriptorCount = 1;
    bindings[22].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    bindings[23].binding = 23;
    bindings[23].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[23].descriptorCount = 1;
    bindings[23].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    // Bindings 24 / 25: voxel building brick atlas + shape directory
    // (Session B). cube.frag's inline-RT shadow + GI rays DDA-march the
    // brickmap to occlude / colour-bleed. Written from init_voxel() after
    // the buffers exist (see voxel.cpp), not in write_scene_descriptors_once.
    bindings[24].binding = 24;
    bindings[24].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[24].descriptorCount = 1;
    bindings[24].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    bindings[25].binding = 25;
    bindings[25].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[25].descriptorCount = 1;
    bindings[25].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    // Binding 26: terrain aux bake (RG8 1024² over the 2048 m world).
    // R = height curvature (0.5 flat, <0.5 concave gullies, >0.5 convex
    // ridges) — splat rules + cavity darkening. G = sky visibility
    // (horizon-scan AO) — multiplied into terrain ambient. Baked once
    // at load + after sculpt strokes (AAA terrain Phases 4/5).
    bindings[26].binding = 26;
    bindings[26].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[26].descriptorCount = 1;
    bindings[26].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    // Binding 27: copy of the depth PRE-PASS (D32, sampled). Read by the
    // grass raymarcher (inside the colour pass) to stop its march at the
    // nearest opaque surface. Rewritten on swapchain recreate.
    bindings[27].binding = 27;
    bindings[27].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[27].descriptorCount = 1;
    bindings[27].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    // Binding 28: clutter (small-object) sun shadow map, D32 - the
    // raymarched terrain samples it (it is pre-shaded, so the deferred
    // pass cannot apply it there).
    bindings[28].binding = 28;
    bindings[28].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[28].descriptorCount = 1;
    bindings[28].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    // Binding 29: the FULL scene depth (depth_view_), read by the FX volume
    // pass while the same image is its DEPTH_READ_ONLY attachment - the
    // prepass copy (27) lacks the gun, props and dynamic objects.
    bindings[29].binding = 29;
    bindings[29].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[29].descriptorCount = 1;
    bindings[29].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    // Binding 30: atmosphere sky-view LUT (GENERAL) - raymarched terrain /
    // water reflections + GI misses use the physical sky (atmosphere.cpp).
    bindings[30].binding = 30;
    bindings[30].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[30].descriptorCount = 1;
    bindings[30].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    // Binding 31: whole-map terrain sun shadow (sun_horizon.cpp).
    bindings[31].binding = 31;
    bindings[31].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[31].descriptorCount = 1;
    bindings[31].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    // Binding 32: sunlight through the cloud layer (atmosphere.cpp, clouds.comp).
    bindings[32] = bindings[31];
    bindings[32].binding = 32;

    // Binding 33: bone palette of the skinned characters (cube_skin.vert; player_body.cpp).
    bindings[33].binding = 33;
    bindings[33].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[33].descriptorCount = 1;
    bindings[33].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    // Binding 34: the terrain's low-resolution march (R32F; pipelines_setup.cpp
    // init_terrain_raymarch_lowres) - the full-resolution terrain pass starts from it.
    bindings[34] = bindings[31];
    bindings[34].binding = 34;
    // Binding 35: the sun-shadow terms at that march's hits (RGBA16F, the same size).
    bindings[35] = bindings[31];
    bindings[35].binding = 35;
    // Bindings 36, 37: the low-resolution terrain's second picture, colour and depth (the
    // "temporal" strategy draws into the two in turn; terrain_raymarch_compose.frag reads
    // both). Without it: the first picture again (init_terrain_raymarch_lowres).
    bindings[36] = bindings[31];
    bindings[36].binding = 36;
    bindings[37] = bindings[31];
    bindings[37].binding = 37;
    // Binding 38: a copy of the lit frame, for the glass (glass.frag).
    bindings[38] = bindings[29];
    bindings[38].binding = 38;
    // Bindings 39, 40: the world's slow fields round the camera, near and far (RG32F; wm_cache.cpp).
    bindings[39] = bindings[31];
    bindings[39].binding = 39;
    bindings[40] = bindings[31];
    bindings[40].binding = 40;
    // Bindings 41, 42: the trail round the player - where people and cars have just been
    // (grass_trail.cpp). 41 is read by the plants' vertex shader, 42 (last frame's) by the
    // pass that draws the next one.
    bindings[41] = bindings[31];
    bindings[41].binding = 41;
    bindings[41].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    bindings[42] = bindings[31];
    bindings[42].binding = 42;
    // 43 / 44: the trail's heights, this frame's and last frame's (grass_trail.cpp).
    // 45: the wind field round the camera, a 3D texture (wind_fluid.cpp).
    bindings[43] = bindings[41];
    bindings[43].binding = 43;
    bindings[44] = bindings[31];
    bindings[44].binding = 44;
    bindings[45] = bindings[41];
    bindings[45].binding = 45;

    VkDescriptorSetLayoutCreateInfo lci{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .pNext = nullptr, .flags = 0,
        .bindingCount = 46, .pBindings = bindings,
    };
    vk_check(vkCreateDescriptorSetLayout(device_, &lci, nullptr,
                                         &scene_desc_set_layout_),
             "scene desc set layout");

    VkDescriptorSetAllocateInfo ai{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .pNext = nullptr,
        .descriptorPool = scene_desc_pool_,
        .descriptorSetCount = 1,
        .pSetLayouts = &scene_desc_set_layout_,
    };
    vk_check(vkAllocateDescriptorSets(device_, &ai, &scene_desc_set_),
             "scene desc set alloc");
    {
        // The bone palette exists from the start (the set must be complete whether or not a
        // character is ever loaded): written by the CPU every frame, read by the vertex shader.
        VkBufferCreateInfo bci{};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size = sizeof(glm::vec4) * 3ull * kSkinBones * kSkinCharacters * kPaletteSlots;
        bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        // (also read on the compute queue: the ray-tracing copies of the soldiers, body_rt.cpp)
        const uint32_t palette_families[2] = { graphics_queue_family_, compute_queue_family_ };
        if (compute_queue_distinct_) {
            bci.sharingMode = VK_SHARING_MODE_CONCURRENT;
            bci.queueFamilyIndexCount = 2;
            bci.pQueueFamilyIndices = palette_families;
        }
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO;
        aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
        VmaAllocationInfo info{};
        vk_check(vmaCreateBuffer(allocator_, &bci, &aci, &body_palette_buf_, &body_palette_alloc_, &info), "bone palette buffer");
        body_palette_map_ = static_cast<glm::vec4*>(info.pMappedData);
        VkDescriptorBufferInfo bi{ body_palette_buf_, 0, VK_WHOLE_SIZE };
        VkWriteDescriptorSet w{};
        w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet = scene_desc_set_; w.dstBinding = 33;
        w.descriptorCount = 1;
        w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w.pBufferInfo = &bi;
        vkUpdateDescriptorSets(device_, 1, &w, 0, nullptr);
    }
    log::info("descriptors created");
}

void VulkanEngine::destroy_descriptors() {
    if (scene_desc_set_layout_) vkDestroyDescriptorSetLayout(device_, scene_desc_set_layout_, nullptr);
    if (scene_desc_pool_)       vkDestroyDescriptorPool(device_, scene_desc_pool_, nullptr);
    if (scene_ubo_buffer_)      vmaDestroyBuffer(allocator_, scene_ubo_buffer_, scene_ubo_alloc_);
    if (body_palette_buf_)      vmaDestroyBuffer(allocator_, body_palette_buf_, body_palette_alloc_);
    body_palette_buf_ = VK_NULL_HANDLE;
    body_palette_alloc_ = nullptr;
    body_palette_map_ = nullptr;
    scene_desc_set_layout_ = VK_NULL_HANDLE;
    scene_desc_pool_ = VK_NULL_HANDLE;
    scene_desc_set_ = VK_NULL_HANDLE;
    scene_ubo_buffer_ = VK_NULL_HANDLE;
    scene_ubo_alloc_ = nullptr;
}

void VulkanEngine::update_scene_ubo() {
    // Per-frame: only update the UBO contents (memcpy to mapped buffer).
    // Descriptor set bindings are written once after init_rt().
    SceneUBO data{};
    update_clutter_shadow_vp();
    data.clutter_vp = clutter_->shadow_vp;
    data.atmo_params = glm::vec4((rt_.sky_mode == 1 && atmo_->sky.view) ? 1.0f : 0.0f,
                                 rt_.atmosphere_intensity * 3.14159265f, 0.0f, 0.0f);
    {
        // Road lanterns (dusk / night only): the two nearest to the eye.
        const float night = env_->night >= 0.0f ? env_->night
                                               : 1.0f - glm::smoothstep(4.0f, 16.0f, rt_.sun_pitch_deg);
        // Up to kLampSlots (24; it was 8), nearest the eye first - those the camera looks toward before
        // those behind it. Chosen from kLampPick metres (it was 220: looking down a road at night only
        // the nearest few lamps lit it). VIBE_LAMPS_OLD=1: as it was (8 from 220 m), to compare.
        static const bool kLampsOld = std::getenv("VIBE_LAMPS_OLD") != nullptr;
        const size_t kLampSlots = kLampsOld ? 8 : std::size(data.lamp_lights);
        const float  kLampPick = kLampsOld ? 220.0f : 800.0f;
        for (auto& l : data.lamp_lights) l = glm::vec4(0.0f);
        // A light with a slot of its own (a room's, the rocket's flame) does not put out the
        // lantern that had the slot: that one moves to a free slot further up.
        auto put = [&](size_t slot, glm::vec4 v) {
            if (data.lamp_lights[slot].w != 0.0f && !kLampsOld)
                for (size_t k = 8; k < std::size(data.lamp_lights); ++k)
                    if (data.lamp_lights[k].w == 0.0f) { data.lamp_lights[k] = data.lamp_lights[slot]; break; }
            data.lamp_lights[slot] = v;
        };
        // (with the road's street lamps near the eye: highway_furniture.cpp - they come and go with
        //  the place, so they are not in lamp_lights_)
        std::vector<glm::vec3> all_lamps = lamp_lights_;
        if (night > 0.01f) hw_lamps_near(player_.eye_position(), kLampPick, all_lamps);
        if (night > 0.01f && !all_lamps.empty()) {
            const glm::vec3 eye = player_.eye_position();
            std::vector<std::pair<float, size_t>> order;
            const glm::vec3 look = -glm::vec3(current_frame_view_.inv_view[2]);
            for (size_t i = 0; i < all_lamps.size(); ++i) {
                const float d = glm::distance(eye, all_lamps[i]);
                if (d >= kLampPick) continue;
                // (a lamp more than 60 m off and behind the camera counts as twice as far: its pool
                //  of light is not in the picture, one ahead is)
                const bool behind = !kLampsOld && d > 60.0f && glm::dot(all_lamps[i] - eye, look) < 0.0f;
                order.push_back({ behind ? d * 2.0f : d, i });
            }
            std::sort(order.begin(), order.end());
            lamp_map_slot_ = -1;
            if (all_lamps.size() <= 8) {
                // Few enough lanterns: fixed slots (slot = lantern), so each keeps
                // its own shadow-map quarter. One map is refreshed per frame, the
                // lanterns near the camera in turn.
                for (const auto& o : order)
                    data.lamp_lights[o.second] = glm::vec4(all_lamps[o.second], 5.0f * night);
                for (size_t k = 0; k < all_lamps.size(); ++k) {
                    const int i = (lamp_map_next_ + int(k)) % int(all_lamps.size());
                    if (glm::distance(eye, all_lamps[size_t(i)]) < 70.0f) {
                        lamp_map_slot_ = i;
                        lamp_map_next_ = i + 1;
                        lamp_map_valid_ |= 1u << i;
                        break;
                    }
                }
                static const bool kNoLampMap = std::getenv("VIBE_NO_LAMP_SHADOWMAP") != nullptr;   // A/B test hook
                data.lamp_info = glm::vec4(kNoLampMap ? 0.0f : float(lamp_map_valid_), 0.0f, 0.0f, 0.0f);
            } else {
                for (size_t k = 0; k < order.size() && k < kLampSlots; ++k)
                    data.lamp_lights[k] = glm::vec4(all_lamps[order[k].second], (order[k].second >= lamp_lights_.size() ? 16.0f : 5.0f) * night);      // (a street lamp hangs at 8 m: three times a lantern's light)
            }
        } else {
            lamp_map_slot_ = -1;
        }
        // The pavilion's room lights by day (at night they are lanterns like the rest, above): a
        // deep room is not lit by its windows alone.
        if (night <= 0.01f && !office_lights_.empty() && glm::distance(player_.eye_position(), office_pos_) < 110.0f)
            for (size_t k = 0; k < office_lights_.size() && k < 6; ++k)
                put(k, glm::vec4(office_lights_[k], 3.4f));
        // ... and the service station's, the six nearest the eye.
        if (night <= 0.01f && !station_lights_.empty() && glm::distance(player_.eye_position(), station_pos_) < 110.0f) {
            std::vector<std::pair<float, size_t>> st_near;
            for (size_t k = 0; k < station_lights_.size(); ++k) st_near.push_back({ glm::distance(player_.eye_position(), station_lights_[k]), k });
            std::sort(st_near.begin(), st_near.end());
            for (size_t k = 0; k < st_near.size() && k < 6; ++k)
                put(k, glm::vec4(station_lights_[st_near[k].second], 3.4f));
        }
        // The launch site's own lights, day and night: the lamp in the control house, and the
        // rocket's exhaust while it burns nearby.
        if (launch_site_placed_) {
            const glm::vec3 leye = player_.eye_position();
            if (glm::distance(leye, launch_house_) < 45.0f)
                put(7, glm::vec4(launch_house_ + glm::vec3(0.0f, 2.95f, 0.0f), 2.4f));
            if (rocket_.thr_vis > 0.02f && rocket_.stage != 3 && glm::distance(leye, glm::vec3(rocket_.pos)) < 1200.0f)
                put(6, glm::vec4(glm::vec3(rocket_.pos) - rocket_.axis * 10.0f, 70.0f * rocket_.thr_vis));
        }
    }
    // Candles in the chapel burn day and night. Each is a small light AT its
    // flame with a short reach (negative intensity = candle, see
    // deferred_lighting.frag). Inside, all eight candles take the lantern
    // slots; from outside only the four by the windows / door axis do.
    if (church_placed_) {
        const float dc = glm::distance(player_.eye_position(), church_pos_ + glm::vec3(0.0f, 1.5f, 0.0f));
        if (dc < 70.0f) {
            // Five lights, the same from inside and outside (so nothing goes
            // out when the player walks through the door): one between the
            // altar candles (it also stands in for the two tall stands beside
            // them) and one at each wall candle. They take lantern slots 3-7;
            // the three nearest road lanterns keep theirs.
            const glm::vec4 c[5] = {
                { 3.25f, 2.15f, 0.0f, -2.2f },
                { -1.3f, 2.53f, 2.3f, -1.5f }, { -1.3f, 2.53f, -2.3f, -1.5f },
                { 1.3f, 2.53f, 2.3f, -1.5f }, { 1.3f, 2.53f, -2.3f, -1.5f },
            };
            for (int i = 0; i < 5; ++i) {
                // (the lantern that had the slot moves up to a free one, as for the rooms' lights above)
                static const bool lamps_old = std::getenv("VIBE_LAMPS_OLD") != nullptr;
                if (data.lamp_lights[3 + i].w != 0.0f && !lamps_old)
                    for (size_t k = 8; k < std::size(data.lamp_lights); ++k)
                        if (data.lamp_lights[k].w == 0.0f) { data.lamp_lights[k] = data.lamp_lights[3 + i]; break; }
                data.lamp_lights[3 + i] = glm::vec4(church_pos_ + glm::vec3(c[i]), c[i].w);
            }
        }
    }
    // How many slots the shaders walk (lamp_info.y): up to the last one that is lit.
    {
        int used = 0;
        for (int k = 0; k < int(std::size(data.lamp_lights)); ++k) if (data.lamp_lights[k].w != 0.0f) used = k + 1;
        data.lamp_info.y = float(used);
        if (static const bool lg = std::getenv("VIBE_LOG_LAMPS") != nullptr; lg && frame_number_ % 120 == 0) {
            float far_d = 0.0f; int lit = 0;
            for (int k = 0; k < used; ++k)
                if (data.lamp_lights[k].w != 0.0f) { ++lit; far_d = (std::max)(far_d, glm::distance(glm::vec3(data.lamp_lights[k]), player_.eye_position())); }
            log::infof("[lamps] %d light(s) lit (slots walked: %d), the furthest %.0f m from the eye", lit, used, double(far_d));
        }
    }
    float pitch = glm::radians(rt_.sun_pitch_deg);
    float yaw   = glm::radians(rt_.sun_yaw_deg);
    glm::vec3 sun(std::cos(pitch) * std::sin(yaw),
                  std::sin(pitch),
                  std::cos(pitch) * std::cos(yaw));
    sun = glm::normalize(sun);
    data.sun_direction = glm::vec4(sun, 0.0f);
    // Auto golden-hour tint. When the sun is low (pitch ≤ 20° elevation),
    // smoothly bias the sun colour toward warm orange and the sky toward
    // pinkish-amber — the standard sunset/sunrise look. Drives every
    // shader that reads scene.sun_color / scene.sky_color (cube,
    // terrain_raymarch, grass, water) from a single uniform source.
    // Toggle via rt_.auto_golden_hour; user's manual sun_color / sky_color
    // are still the basis, just modulated by altitude when enabled.
    glm::vec3 sun_rgb = rt_.sun_color;
    glm::vec3 sky_rgb = rt_.sky_color;
    if (rt_.auto_golden_hour) {
        // Elevation of the real sun (with the clock running; the light's
        // own direction otherwise). The warmth comes in slowly from mid
        // afternoon and is strongest in the last few degrees; after sunset
        // it fades with the twilight (env_->golden).
        const float elev = env_->sun_elev > -500.0f ? env_->sun_elev : glm::degrees(std::asin(glm::clamp(sun.y, -1.0f, 1.0f)));
        float t = 1.0f - glm::smoothstep(1.0f, 34.0f, elev);
        t = t * t * (env_->applied ? env_->golden : 1.0f);
        // Warm sunset tint: orange-amber at horizon, peach at ~15°.
        // No brightness boost — just shift the spectrum. The previous
        // 1.35x lift over-cranked the sun and washed everything out.
        glm::vec3 warm     = glm::vec3(1.00f, 0.55f, 0.25f);
        glm::vec3 sky_warm = glm::vec3(0.95f, 0.50f, 0.40f);
        sun_rgb = glm::mix(sun_rgb, sun_rgb * warm, t);
        sky_rgb = glm::mix(sky_rgb, sky_warm,       t * 0.85f);
    }
    data.sun_color     = glm::vec4(sun_rgb, rt_.sun_intensity);
    last_sun_rgb_ = sun_rgb;   // volumetric fog uses the same (golden-hour) light
    last_sky_rgb_ = sky_rgb;
    data.ambient       = glm::vec4(rt_.ground_ambient * env_->amb_k, rt_.terrain_ao_punch);   // darkens with the night fill light
    data.sky_color     = glm::vec4(sky_rgb, 1.0f);
    // DLSS Ray Reconstruction resolves the noise itself, so with it active
    // the ray-traced terms are traced at a low sample count: brute-force
    // multi-sampling there only pre-blurs what RR would reconstruct sharper.
    const bool rr_low = dlss_rr_active() && rt_.dlss_rr_low_samples;
    // FSR accumulates over frames as well (not as strongly as Ray Reconstruction):
    // 6 shadow + 3 bounce rays look the same as 18 + 12 and on a Radeon
    // take the lighting pass from 7.9 ms to 3 ms.
    const bool fsr_low = !rr_low && rt_.fsr2_enabled && rt_.fsr_backend == 1 && rt_.dlss_rr_low_samples;
    const int shadow_samples_eff = rr_low ? std::min(rt_.shadow_samples, 2)
                                 : fsr_low ? std::min(rt_.shadow_samples, 6) : rt_.shadow_samples;
    const int ao_samples_eff     = rr_low ? std::min(rt_.ao_samples, 1) : rt_.ao_samples;
    // .x: 0 = RT shadows off, 1 = on, 3 = on + Ray Reconstruction quality
    // mode (deferred_lighting.frag: longer / rougher reflections, a second
    // GI bounce on every ray). Shaders test != 0 for "shadows on".
    data.rt_flags = glm::ivec4(rt_.shadow_enabled ? ((rr_low ? 3 : 1) | (std::clamp(rt_.foliage_ray_mode, 0, 2) << 2) | (rt_.gi_leaf_lite ? 16 : 0) | (rt_.irradiance_cache && rt_.radiance_cache ? 32 : 0)) : 0,   // bits 2-3: foliage ray mode; bit 4: one bounce-light ray on leaves
                               shadow_samples_eff,
                               ao_samples_eff,
                               static_cast<int>(frame_number_));
    data.rt_params = glm::vec4(rt_.shadow_softness,
                               rt_.ao_radius,
                               rt_.ambient_strength,
                               rt_.shadow_strength);
    // ReSTIR-lite: when on, hand the shader gi_samples / 4 (rounded down,
    // floored at 1 if gi_samples > 0). The 4x reduction is the visible
    // perf win for session 1; sessions 2+ replace this with the actual
    // reservoir-resampled algorithm — see docs/restir_plan.md.
    int gi_samples_eff = rt_.gi_samples;
    if (rt_.gi_restir_enabled && gi_samples_eff > 0) {
        gi_samples_eff = std::max(1, gi_samples_eff / 4);
    }
    if (rr_low) gi_samples_eff = std::min(gi_samples_eff, 2);
    if (fsr_low) gi_samples_eff = std::min(gi_samples_eff, 2);   // (was 3: the third ray, with its second bounce, cost as much as the first two together - 4.7 ms of 20 in a forest on an RX 9070 XT; the accumulation over 28 frames hides the difference)
    data.rt_flags2 = glm::ivec4(gi_samples_eff,
                                (rt_.reflections_enabled ? 1 : 0) | shade_flat_bits_,      // (bits 1-3: the shading test - scene_ubo.glsl FLAT_*)
                                rt_.gi_bounces,
                                rt_.ao_mode);  // 0=off, 1=fast, 2=RTAO
    // ReSTIR session 3+4 (temporal + spatial reservoir blend) was
    // quarantined after shipping (visible wobble). Session 5 fixed both
    // root causes and re-enables it (see the restir_params line below).
    //
    // The session-1 gi_samples / 4 reduction also stays in effect when
    // rt_.gi_restir_enabled (see gi_samples_eff above): ReSTIR converges
    // the quartered sample count back to full quality via temporal +
    // spatial reservoir reuse.
    //
    // restir_params packing:
    // .x: ReSTIR enabled (rt_.gi_restir_enabled). Consumed ONLY by the
    //     two reservoir gates in cube.frag — verified no other shader
    //     reads restir_params.x, so this is independent of FSR.
    // .y: M_max (sample-count cap; bounds temporal variance/lag).
    // .zw: jitter UV delta (cur−prev) for FSR3 motion-vec jitter
    //      cancellation in cube.frag and terrain_raymarch.frag.
    //      Only non-zero when FSR is on (TAA expects jittered motion
    //      vec, would freeze if subtracted unconditionally).
    glm::vec2 jdelta_uv(0.0f, 0.0f);
    if (rt_.fsr2_enabled) {
        // proj[2][xy] += j shifts clip.xy by -j*w (GLM w = -z_view), i.e. the
        // image moves by -j in NDC: the jitter part of raw motion is
        // -(cur - prev) * 0.5 in uv. (Was +: static scenes got 2x the jitter
        // delta as motion and DLSS shimmered while standing still.)
        jdelta_uv = -(current_frame_view_.jitter - prev_jitter_) * 0.5f;
    }
    // .x re-enabled in ReSTIR session 5. It is consumed ONLY by the two
    // temporal/spatial reservoir gates in cube.frag (verified — no other
    // shader reads restir_params.x), so driving it from the user toggle
    // is independent of FSR. The session 3+4 quarantine root causes are
    // now fixed: (a) the single-buffer aliasing race is gone — the
    // reservoir SSBO is a 3-region ring indexed by frame%3 (race-free
    // under kFrameOverlap=2; see restir.cpp), and (b) disocclusion is
    // depth-aware — cube.frag carries prev cam-distance in the reservoir
    // pad slot and rejects reprojected/neighbour samples whose surface
    // is too far in normal OR depth. .z is no longer the (jitter-
    // polluted) normal threshold; cube.frag uses fixed constants.
    float restir_on = rt_.gi_restir_enabled ? 1.0f : 0.0f;
    data.restir_params = glm::vec4(restir_on, 32.0f, jdelta_uv.x, jdelta_uv.y);
    // .x = SPOM bump-depth. .yzw = procedural ground splat knobs
    // (free slots — no UBO layout change): y = material strength,
    // z = metres per material repeat, w = detail-normal strength.
    data.spom_params   = glm::vec4(std::max(0.0f, rt_.spom_strength),
                                    glm::clamp(rt_.ground_mat_strength, 0.0f, 1.0f),
                                    std::max(0.25f, rt_.ground_mat_tile_m),
                                    glm::clamp(rt_.ground_mat_normal, 0.0f, 1.0f));
    // .x = water-only flag for the terrain_raymarch pass: 1.0 when
    // the mesh terrain is the active ground (raymarch terrain OFF) but
    // water is enabled — the pass then skips the terrain march and
    // only renders the water surface on top of the rasterised mesh.
    // .z = half-rate-shadow toggle for cube.frag (Phase 3 consumer).
    //      When 1, brush/dyn surfaces skip the inline blocker+PCSS
    //      block and bilinear-sample u_shadow_lr (binding 18) instead.
    // .w = GI temporal accumulator enable. When >0.5 cube.frag
    //      EMA-blends shade_radiance with the reprojected history
    //      sample from binding 20/21 (ping-pong by frame parity).
    //      ALWAYS ON, decoupled from rt_.svgf_enabled: the accumulator
    //      denoises ONLY the raw GI irradiance (pre-albedo, motion-
    //      reprojected — sharp textures untouched), which is exactly
    //      the scoped denoise the GI signal needs. svgf_enabled now
    //      gates only the full-screen a-trous passes, whose scene-wide
    //      blur is opt-in.
    data.terrain_local_info = glm::vec4(
        (!rt_.terrain_raymarch_enabled && rt_.water_enabled) ? 1.0f : 0.0f,
        terrain_height_max_ + 5.0f,   // .y = mesh max height (air early-out)
        rt_.half_rate_shadows ? 1.0f : 0.0f,
        1.0f);
    // Copy the baked 32×32 hi-Z max-cell grid into the UBO (1024
    // floats → 256 vec4). The grid is static after level load (baked
    // in init_world from terrain_data_.heights); when it isn't ready
    // yet, write a high constant so the shader's cell-skip never
    // wrongly skips real terrain on the first frames.
    if (terrain_max_grid_ready_) {
        std::memcpy(data.terrain_max_grid, terrain_max_grid_.data(),
                    sizeof(float) * 1024);
    } else {
        for (int i = 0; i < 256; ++i)
            data.terrain_max_grid[i] = glm::vec4(1.0e9f);
    }
    data.rt_params2 = glm::vec4(rt_.gi_strength,
                                rt_.gi_radius,
                                rt_.reflection_strength,
                                rt_.shadow_curve);
    // Use the lerped eye captured by compute_frame_view, NOT the raw
    // physics-tick player_.eye_position(). Mismatch caused raymarched
    // grass/terrain to render one frame behind rasterised geometry —
    // their ray origin came from the raw eye while their projection
    // matrix used the smoothed render_pos.
    glm::vec3 eye = current_frame_view_.eye_pos;
    data.camera_pos = glm::vec4(eye, 0.0f);
    // (scene_ray: the projection and the view's rotation only - no position)
    data.ray_mat = glm::mat4(glm::inverse(glm::dmat4(current_frame_view_.proj) * glm::dmat4(glm::dmat3(current_frame_view_.view))));
    {   // The grass's blade lattice at the camera: which cell the camera is in and where in it,
        // in doubles - the shader then works from the camera outward, in small numbers, and a
        // blade (millimetres wide) is whole however large the engine's coordinates are.
        // (kPeriod as grass_raymarch.frag has it: TILE_CELL(0.17).)
        const float P = 6400.0f / std::max(std::round(6400.0f / 0.17f), 1.0f);
        const glm::dvec2 U = glm::dvec2(double(eye.x), double(eye.z)) + 0.5 * double(P);
        const glm::dvec2 n0 = glm::floor(U / double(P)), f0 = U - n0 * double(P);
        data.grass_cam = glm::vec4(float(n0.x), float(n0.y), float(f0.x), float(f0.y));
    }
    // Distance LOD: 15 m = full samples (gameplay-relevant range — shadows
    // and AO are most visible up close); past 50 m fades to 1 sample so the
    // skybox-distance cubes don't waste a full ray budget per pixel.
    // .z carries gi_shadow_max_bounce — see RtSettings; cast to int in
    // cube.frag where it gates the per-bounce sun shadow ray.
    // .z = gi_shadow_max_bounce (cube.frag casts to int).
    // .w = ao_floor — see RtSettings; cube.frag remaps raw AO to
    //      [ao_floor, 1.0] so corner darkness doesn't compound.
    data.rt_lod = glm::vec4(15.0f, 50.0f,
                             static_cast<float>(rt_.gi_shadow_max_bounce),
                             rt_.ao_floor);
    {
        // cube.frag renders at render_extent_ (so does its motion-vec output);
        // gl_FragCoord.xy / viewport.zw is in render-resolution UV space.
        float w = static_cast<float>(render_extent_.width);
        float h = static_cast<float>(render_extent_.height);
        data.viewport = glm::vec4(w, h, 1.0f / w, 1.0f / h);
    }
    {
        // Muzzle flash — origin a short distance ahead of the eye so it
        // doesn't get trapped inside the player capsule. Intensity falls
        // linearly from 1 at fire to 0 over kMuzzleFlashDuration so the
        // pop is sharp and brief. Color is warm and quite hot — at intensity
        // 1.0 a single shot lights nearby walls visibly through RT shadows.
        if (muzzle_flash_timer_ > 0.0f) {
            float t = std::min(1.0f, muzzle_flash_timer_ / kMuzzleFlashDuration);
            glm::vec3 fwd = player_.forward();
            glm::vec3 pos = player_.eye_position() + fwd * 0.35f;
            float intensity = 2.2f * t;        // HDR pop (deferred applies it to the gun + surroundings)
            data.muzzle_pos   = glm::vec4(pos, intensity);
            data.muzzle_color = glm::vec4(1.00f, 0.85f, 0.55f, 6.0f);  // 6 m radius
        } else {
            data.muzzle_pos   = glm::vec4(0.0f);
            data.muzzle_color = glm::vec4(0.0f);
        }
        // Explosions share the single dynamic point light: take the one
        // that is brightest at the camera (flash, then decaying fireball
        // glow) if it beats the muzzle flash.
        float best = data.muzzle_pos.w * 0.05f;
        const glm::vec3 eye = player_.eye_position();
        for (const auto& e : explosions_) {
            if (e.t > 3.5f) continue;
            float inten = e.size * (e.t < 0.12f ? 70.0f : 22.0f * std::exp(-e.t * 1.2f));
            float radius = 170.0f * e.size;
            float d = glm::distance(eye, e.pos);
            float score = inten * std::max(0.05f, 1.0f - d / (radius * 2.5f));
            if (score > best) {
                best = score;
                data.muzzle_pos   = glm::vec4(e.pos + glm::vec3(0.0f, 9.0f * e.size, 0.0f), inten);
                data.muzzle_color = glm::vec4(1.0f, 0.52f, 0.20f, radius);
            }
        }
    }
    data.terrain_params = glm::vec4(rt_.terrain_fog_strength,
                                    rt_.terrain_wrap_strength,
                                    rt_.terrain_detail_strength,
                                    rt_.terrain_shadow_softness_scale);
    data.terrain_h_low  = glm::vec4(rt_.terrain_h_sand_grass_start,
                                    rt_.terrain_h_sand_grass_end,
                                    rt_.terrain_h_grass_dirt_start,
                                    rt_.terrain_h_grass_dirt_end);
    data.terrain_h_high = glm::vec4(rt_.terrain_h_dirt_rock_start,
                                    rt_.terrain_h_dirt_rock_end,
                                    rt_.terrain_h_rock_snow_start,
                                    rt_.terrain_h_rock_snow_end);
    data.grass_extra = glm::vec4(rt_.grass_height_scale,
                                 rt_.grass_alpha_cutoff,
                                 rt_.grass_slope_n_min,
                                 rt_.grass_distance_density);
    data.grass_extra2 = glm::vec4(rt_.grass_alt_min,
                                  rt_.grass_alt_max,
                                  rt_.shadow_map_world_half,
                                  static_cast<float>(rt_.terrain_debug_mode));
    data.light_vp = sun_->shadow_light_vp;
    data.terrain_extra = glm::vec4(rt_.terrain_shading_contrast,
                                    rt_.shadow_near_mult,
                                    rt_.gi_softener,
                                    rt_.gi_debug_viz ? 1.0f : 0.0f);
    // Rocky-grass displacement + POM far-fade knobs (terrain_disp_params
    // tail field on the UBO; see SceneUBO comment in internal.h).
    data.terrain_disp_params = glm::vec4(
        std::max(0.0f, rt_.terrain_disp_amp),
        glm::clamp(rt_.terrain_disp_smooth_mip, 0.0f, 6.0f),
        std::max(20.0f, rt_.terrain_pom_far_m),
        // .w = 1 when the deferred pass owns lighting for material 1/3
        // pixels: cube.frag then writes the G-buffer and skips its own
        // shadow/GI/AO/fog (deferred overwrites that colour anyway).
        // Kept 0 when SVGF or ReSTIR need cube.frag's GI side outputs.
        deferred_owns_lighting() ? 1.0f : 0.0f);
    // Anti-tile blend params -- gates the rotated extra sample inside
    // cube.frag's terrain splat. .x is the master toggle; .y the blend
    // weight of the rotated sample (0 = single sample, 1 = full blend).
    data.terrain_antitile_params = glm::vec4(
        rt_.terrain_antitile ? 1.0f : 0.0f,
        glm::clamp(rt_.terrain_antitile_strength, 0.0f, 1.0f),
        0.0f, 0.0f);
    // Grass-cast shadow on rasterised terrain. Strength=0 OR toggle=0
    // -> cube.frag fast-exits the mask march (no extra texture taps).
    data.grass_shadow_on_terrain_params = glm::vec4(
        glm::clamp(rt_.grass_shadow_on_terrain_strength, 0.0f, 1.0f),
        static_cast<float>(glm::clamp(rt_.grass_shadow_on_terrain_samples,
                                       1, 8)),
        std::max(0.1f, rt_.grass_shadow_on_terrain_dist),
        rt_.grass_shadow_on_terrain ? 1.0f : 0.0f);
    // Blade-shadow march (cube.frag terrain): same blade field as
    // grass_raymarch.frag (height scale, wind, time, density, slope gate).
    {
        float sn = std::max(rt_.grass_slope_n_min, 1e-2f);
        float slope_max = std::sqrt(std::max(1e-4f, 1.0f / (sn * sn) - 1.0f));
        data.grass_rm_params = glm::vec4(rt_.grass_height_scale, rt_.grass_wind,
                                         anim_time_,
                                         glm::clamp(rt_.grass_density, 0.0f, 1.0f));
        data.grass_rm_params2 = glm::vec4(
            slope_max,
            (rt_.grass_shadow_on_terrain && rt_.grass_raymarch_enabled) ? 1.0f : 0.0f,
            glm::clamp(rt_.grass_shadow_on_terrain_strength, 0.0f, 1.0f),
            rt_.terrain_rm_contrast);   // .w: raymarched-terrain contrast
    }
    // Side-lit grass shading. .z is reused (was reserved) as the
    // cube.frag wall-displacement mode: 0 = legacy SPOM, 1 = SSDM
    // depth-override relief. grass.frag / grass_raymarch.frag only read
    // .x/.y, so .z is free to carry this cross-cutting toggle without
    // growing the UBO (cube.frag declares the struct only through
    // grass_side_lit_params, so a true append would land on water_params).
    data.grass_side_lit_params = glm::vec4(
        glm::clamp(rt_.grass_side_lit_strength, 0.0f, 1.0f),
        rt_.grass_side_lit_enabled ? 1.0f : 0.0f,
        // (while the screen-space displacement is on - ssdm.cpp - the walls are drawn flat: it moves the stones)
        static_cast<float>(ssdm_active() ? 0 : rt_.spom_mode),
        glm::clamp(rt_.grass_edge_softness, 0.0f, 1.0f));   // .w: grass patch edge softness
    // Ocean / water plane params. Time is derived from frame number
    // at a nominal 60 Hz — exact wall-clock isn't needed for waves
    // and avoids threading a real timer here.
    float water_t = anim_time_;
    data.water_params = glm::vec4(rt_.water_enabled ? 1.0f : 0.0f,
                                   rt_.water_level,
                                   rt_.water_wave_strength,
                                   water_t);
    // water_color.w doubles as the "shadows enabled" flag (0 / 1).
    data.water_color  = glm::vec4(rt_.water_color,
                                   rt_.water_shadows_enabled ? 1.0f : 0.0f);
    data.water_color_shallow = glm::vec4(rt_.water_color_shallow, 0.0f);
    // water_color_shallow.w doubles as the wave-frequency multiplier
    // (was unused). Read on the shader side as wave_scale.
    // (Must come AFTER the vec4 assignment above, which used to zero it.)
    data.water_color_shallow.w = std::max(0.05f, rt_.water_wave_scale);
    // water_shore.w doubles as the transparency / underwater
    // showthrough strength (0 = opaque, 1 = shallow shows terrain).
    data.water_shore = glm::vec4(rt_.water_shore_blend,
                                  rt_.water_shore_noise,
                                  rt_.water_tlas_reflections ? 1.0f : 0.0f,
                                  rt_.water_transparency);
    data.water_foam_color  = glm::vec4(rt_.water_foam_color,
                                        rt_.water_foam_strength);
    // Water style + river-style knobs packed into water_foam_params.yzw:
    //   y = water_style (int, 0 = default, 1 = river)
    //   z = water_river_speed
    //   w = water_river_normal_str
    // (water_river_extinct_mix lives in water_color_shallow.w — that slot
    //  was wave_scale; we keep both by repurposing only when style >= 1.)
    // foam_params.w is repurposed across styles:
    //   river: river normal_str
    //   lake : lake bump_strength
    float foam_w_slot = (rt_.water_style == 2)
        ? std::max(0.0f, rt_.water_lake_bump_strength)
        : std::max(0.0f, rt_.water_river_normal_str);
    data.water_foam_params = glm::vec4(std::max(0.05f, rt_.water_foam_width),
                                        static_cast<float>(rt_.water_style),
                                        std::max(0.0f, rt_.water_river_speed),
                                        foam_w_slot);
    data.fog_band = glm::vec4(rt_.terrain_raymarch_fog_y_start,
                               rt_.terrain_raymarch_fog_y_top,
                               rt_.terrain_raymarch_fog_noise,
                               rt_.terrain_rt_lod_distance);
    data.terrain_rt_extra = glm::vec4(
        static_cast<float>(rt_.terrain_pcss_samples_cap),
        static_cast<float>(rt_.terrain_gi_samples_cap),
        rt_.terrain_ao_final_strength,
        static_cast<float>(rt_.terrain_gi_bounces_cap));
    data.grass_color_top    = glm::vec4(rt_.grass_color_top,
                                         rt_.grass_density);
    data.grass_color_bottom = glm::vec4(rt_.grass_color_bottom,
                                         rt_.grass_base_ao_floor);
    data.grass_color_ground = glm::vec4(rt_.grass_color_ground,
                                         rt_.grass_ground_tint_strength);
    data.grass_color_ground_far = glm::vec4(rt_.grass_color_ground_far,
                                              std::max(50.0f, rt_.grass_ground_tint_far_distance));
    data.grass_shadow_params = glm::vec4(
        rt_.grass_shadow_strength,
        static_cast<float>(rt_.grass_shadow_samples),
        rt_.grass_shadow_max_dist,
        0.0f);
    // Shoreline grass tint — see SceneUBO comment in internal.h.
    data.grass_shore_color = glm::vec4(rt_.grass_shore_color,
                                        rt_.grass_shore_strength);
    data.grass_shore_params = glm::vec4(std::max(0.05f, rt_.grass_shore_distance),
                                         glm::clamp(rt_.grass_near_boost, 0.0f, 2.0f), 0.0f, 0.0f);   // .y near density boost
    // Shoreline TERRAIN tint — same packing.
    data.terrain_shore_color = glm::vec4(rt_.terrain_shore_color,
                                          rt_.terrain_shore_strength);
    data.terrain_shore_params = glm::vec4(std::max(0.05f, rt_.terrain_shore_distance),
                                           0.0f, 0.0f, 0.0f);
    // Distance fog — strength=0 disables in-shader.
    // (The haze is lit by the sky: at night it darkens with the fill light.)
    data.distance_fog_color = glm::vec4(rt_.distance_fog_color * env_->amb_k,
                                         std::max(0.0f, std::min(1.0f, rt_.distance_fog_strength)));
    data.distance_fog_params = glm::vec4(std::max(0.0f, rt_.distance_fog_density),
                                          std::max(0.0f, rt_.distance_fog_start),
                                          std::max(0.0f, rt_.distance_fog_height),
                                          std::max(0.0f, std::min(1.0f, rt_.distance_fog_max)));
    data.terrain_shore_general_color = glm::vec4(rt_.terrain_shore_general_color,
                                                  rt_.terrain_shore_general_strength);
    data.terrain_shore_general_params = glm::vec4(std::max(0.05f, rt_.terrain_shore_general_distance),
                                                   0.0f, 0.0f, 0.0f);
    data.terrain_sand_color = glm::vec4(rt_.terrain_sand_color, 0.0f);
    data.rm_ground_dry   = glm::vec4(rt_.rm_ground_dry, glm::clamp(rt_.rm_green_amount, 0.0f, 1.0f));
    data.rm_ground_green = glm::vec4(rt_.rm_ground_green, glm::clamp(rt_.rm_red_sand_amount, 0.0f, 1.0f));
    data.rm_red_sand     = glm::vec4(rt_.rm_red_sand, glm::clamp(rt_.rm_red_sand_scale, 0.2f, 5.0f));
    for (int i = 0; i < 24; ++i) data.craters[i] = i < int(craters_.size()) ? craters_[size_t(i)] : glm::vec4(0.0f);
    // y / z: cloud cover and cloud-shadow contrast (scene_ubo.glsl cloud_shadow).
    data.crater_info = glm::vec4(float(craters_.size()),
                                 (rt_.clouds_enabled && rt_.sky_mode == 1) ? glm::clamp(rt_.cloud_coverage * 0.80f, 0.0f, 1.0f) : 0.0f,
                                 // (a cumulus takes about 70% of the direct sun; the map itself goes to black under thick cloud)
                                 (rt_.clouds_enabled && rt_.sky_mode == 1) ? env_->cloud_shadow_k * 0.96f : 0.0f,
                                 // .w: light level of the haze / fog colours (1 by day, the
                                 // night fill-light factor after dark)
                                 env_->amb_k);
    // The cloud layer for the cloud-shadow function (scene_ubo.glsl): the
    // same numbers clouds.comp renders with, incl. its wind offset.
    {
        // How far the wind has carried the clouds: wind speed ADDED UP over
        // time. (Was speed x time: when the weather changed the wind speed, the
        // whole sky jumped by hours of travel - the clouds raced during a change.)
        const double ct = glfwGetTime();
        if (cloud_->travel_t >= 0.0) cloud_->travel += double(rt_.cloud_speed) * std::min(ct - cloud_->travel_t, 0.25);
        else cloud_->travel = double(rt_.cloud_speed) * ct;
        cloud_->travel_t = ct;
        const glm::vec2 wind = glm::vec2(1.0f, 0.35f) * (float(cloud_->travel) * 0.6f);
        cloud_->a = glm::vec4(glm::clamp(rt_.cloud_coverage, 0.0f, 1.0f), glm::clamp(rt_.cloud_scale, 0.3f, 4.0f),
                             glm::clamp(rt_.cloud_base_m, 400.0f, 8000.0f), glm::clamp(rt_.cloud_tower_m, 300.0f, 12000.0f));
        cloud_->b = glm::vec4(wind, std::max(0.0f, rt_.cloud_density), 0.0f);
        data.cloud_a = cloud_->a;
        data.cloud_b = cloud_->b;
        cloud_->shadow_k_now = data.crater_info.z;
    }
    for (size_t i = 0; i < 16; ++i)
        data.ripples[i] = i < water_ripples_.size()
            ? glm::vec4(water_ripples_[i].xz, water_ripples_[i].age, water_ripples_[i].amp) : glm::vec4(0.0f);
    for (int i = 0; i < 32; ++i) data.footprints[i] = footprints_[i];
    data.season = glm::vec4(season_w_[0], season_w_[1], season_w_[2], season_w_[3]);
    {
        const globe::Chart& gc = globe::current();
        data.globe_frame = glm::mat4(glm::mat3(gc.F));
        data.globe_info = glm::vec4(gc.home ? 0.0f : 1.0f, float(gc.ox()), wv_nf_ok() ? 1.0f : 0.0f,
                                    // VIBE_RM_SKIP=<bits>: leave sections of the terrain shader out, to measure them (terrain_raymarch.frag RMDBG)
                                    float((prof_rm_skip_ | shade_rm_skip_ | (rt_.terrain_rm_skip & (128 | 1024 | 32768))) | []{ static const char* e = std::getenv("VIBE_RM_SKIP"); return e ? std::atoi(e) : 0; }()));
        int32_t cells[globe::kOctaves][4]; float fracs[globe::kOctaves][4];
        globe::lattice(gc.F[1] * globe::kR, cells, fracs);
        for (int i = 0; i < 12; ++i) {
            data.globe_cell[i] = glm::ivec4(cells[i][0], cells[i][1], cells[i][2], 0);
            data.globe_frac[i] = glm::vec4(fracs[i][0], fracs[i][1], fracs[i][2], 0.0f);
        }
    }
    data.site_ground = site_uniform(&data.site_shape);       // (sites.cpp)
    data.hw_info = hw_info_;
    data.hw_bbox = hw_bbox_;
    {   // home's flat formula from this chart, where it applies (engine/globe.h Ring)
        const globe::Ring& rg = globe::ring_now();
        data.ring_info = glm::vec4(rg.on ? 1.0f : 0.0f, float(rg.wp0.x), float(rg.wp0.y), float(globe::current().org.y));     // (.w: the chart origin's local z; its x is globe_info.y)
        data.ring_h0 = glm::vec4(float(rg.h0.x), float(rg.h0.y), 0.0f, 0.0f);
        data.ring_j = glm::vec4(float(rg.J[0][0]), float(rg.J[1][0]), float(rg.J[0][1]), float(rg.J[1][1]));
        float lat[globe::kFlatOctaves][4];
        globe::flat_lattice(rg.h0, lat);
        for (int i = 0; i < globe::kFlatOctaves; ++i)
            data.ring_cell[i] = rg.on ? glm::vec4(lat[i][0], lat[i][1], lat[i][2], lat[i][3]) : glm::vec4(0.0f);
    }
    {   // what is roofed near the camera (no rain, wet or snow inside: scene_ubo.glsl scene_sheltered)
        std::vector<Shelter> sh;
        shelters_near(player_.eye_position(), 400.0f, sh, 4);
        for (int i = 0; i < 4; ++i) {
            const bool on = size_t(i) < sh.size();
            data.shelter_a[i] = on ? glm::vec4(sh[size_t(i)].lo, sh[size_t(i)].hi) : glm::vec4(0.0f);
            data.shelter_b[i] = on ? glm::vec4(sh[size_t(i)].top, 1.0f, 0.0f, 0.0f) : glm::vec4(0.0f);
        }
    }
    // The sea's wave field (terrain_raymarch.frag sea_map). It is a sum of octaves, each a
    // pattern of its own coordinate  c = f_i * M_i * (0.75 x, z)  - turned and 1.9 x finer per
    // octave, up to 34 x the world coordinate. Worked out from world coordinates in floats
    // that ran out of precision a couple of kilometres from the origin: the surface cracked
    // along the noise lattice. So the shader works from a point near the camera (sea_org),
    // and what that point contributes to each octave's coordinate is worked out here in
    // doubles and handed over in pieces that stay small: modulo 2 pi for the sines, fraction
    // and whole part (modulo the lattice's period) for the noise. The result is the same
    // pattern whatever the point, so moving it does not show.
    {
        const double ox = std::round(double(eye.x) / 64.0) * 64.0, oz = std::round(double(eye.z) / 64.0) * 64.0;
        data.sea_org = glm::vec4(float(ox), float(oz), 0.0f, 0.0f);
        double ux = ox * 0.75, uy = oz, freq = 0.16;
        const double kTwoPi = 6.283185307179586;
        for (int i = 0; i < 5; ++i) {
            const double cx = ux * freq, cy = uy * freq;
            const double fx = std::floor(cx), fy = std::floor(cy);
            auto wrap = [](double v, double m) { double r = std::fmod(v, m); return r < 0.0 ? r + m : r; };
            data.sea_a[i] = glm::vec4(float(wrap(cx, kTwoPi)), float(wrap(cy, kTwoPi)), float(cx - fx), float(cy - fy));
            data.sea_b[i] = glm::vec4(float(wrap(fx, 4096.0)), float(wrap(fy, 4096.0)), 0.0f, 0.0f);
            const double nx = 1.6 * ux - 1.2 * uy, ny = 1.2 * ux + 1.6 * uy;     // (the shader's mat2(1.6, 1.2, -1.2, 1.6))
            ux = nx; uy = ny; freq *= 1.9;
        }
    }
    // Raymarched terrain, speed settings (terrain_raymarch.frag; scene_ubo.glsl rm_perf).
    data.rm_perf = glm::vec4((tr_march_prepass_on() ? 1.0f : 0.0f) + (rt_.terrain_rm_relief_lite ? 2.0f : 0.0f) +
                                 ((tr_march_prepass_on() && !tr_use_lowres() && rt_.terrain_rm_shade_lo) ? 4.0f : 0.0f) +
                                 ((tr_march_prepass_on() && tr_use_lowres()) ? 32.0f : 0.0f) +       // (the low-resolution terrain pass starts at the march)
                                 ((tr_march_prepass_on() && !tr_use_lowres() && rt_.terrain_rm_start_at_hit) ? 8.0f : 0.0f),
                             float(tr_->march_div),
                             rt_.terrain_rm_converge,
                             rt_.terrain_rm_normal_px);
    data.ripple_info = glm::vec4(float(std::min<size_t>(16, water_ripples_.size())), env_->rain,
                                   std::min(env_->wet, 0.99f) + std::floor(glm::clamp(env_->snow, 0.0f, 1.0f) * 255.0f),   // fract = wet, whole = snow (scene_ubo.glsl)
                                   0.0f);
    // Hand flashlight.
    flash_vp_ = glm::mat4(0.0f);
    static const bool kNoFlashMap = std::getenv("VIBE_NO_FLASH_SHADOWMAP") != nullptr;   // A/B test hook
    if (flashlight_on_ && clutter_->shadow_flash_view && scatter_.plant_buffer && !kNoFlashMap) {
        const glm::vec3 lens = flash_pos_ + flash_fwd_ * 0.118f;
        glm::mat4 proj = glm::perspective(glm::radians(46.0f), 1.0f, 0.15f, 60.0f);
        proj[1][1] *= -1.0f;
        flash_vp_ = proj * glm::lookAt(lens, lens + flash_fwd_, flash_up_);
    }
    data.flash_vp = flash_vp_;
    if (flashlight_on_) {
        // Four brightness levels: low, medium, high, turbo (more light and more reach).
        const float kPower[4] = { 7.0f, 16.0f, 32.0f, 60.0f };
        const float kReach[4] = { 60.0f, 85.0f, 115.0f, 150.0f };
        const int lv = std::clamp(flash_level_, 0, 3);
        data.flash_pos = glm::vec4(flash_pos_ + flash_fwd_ * 0.118f, kReach[lv]);
        data.flash_dir = glm::vec4(flash_fwd_, std::cos(glm::radians(21.0f)));
        data.flash_col = glm::vec4(glm::vec3(1.0f, 0.95f, 0.84f) * kPower[lv], std::cos(glm::radians(9.0f)));
    } else {
        data.flash_pos = data.flash_dir = data.flash_col = glm::vec4(0.0f);
        // A car's head lights (jeep.cpp) are the same kind of light: a cone that lights what
        // it falls on and shows in the air. They take it while the hand torch is off.
        if (jeep_headlights(data.flash_pos, data.flash_dir, data.flash_col)) data.flash_vp = glm::mat4(0.0f);
    }
    // Glow lights: anti-aircraft muzzle flashes and the tracer rounds
    // themselves light what they pass. Up to 8, the ones that matter most
    // to the camera; rounds already high above the ground light nothing
    // and are skipped.
    {
        struct Cand { glm::vec3 pos; float radius; glm::vec3 col; float score; float shadow; };
        static std::vector<Cand> cands;
        cands.clear();
        const glm::vec3 eye = player_.eye_position();
        auto add = [&](glm::vec3 p, float radius, glm::vec3 col, float inten, bool shadow) {
            const float d = glm::distance(p, eye);
            if (d > 900.0f) return;
            cands.push_back({ p, radius, col * inten, inten * radius / (d + 25.0f), shadow ? 1.0f : 0.0f });
        };
        for (const auto& v : aa_) {
            if (v.flash_t <= 0.0f || v.kind == 1) continue;
            glm::mat4 m[4];
            aa_part_models(v, false, m);
            const glm::vec3 mz = glm::vec3(m[2] * glm::vec4(aa_muzzle_[v.flash_barrel], 1.0f));
            add(mz, 26.0f, glm::vec3(1.0f, 0.62f, 0.30f), 7.0f * glm::clamp(v.flash_t / 0.05f, 0.0f, 1.0f), true);
        }
        for (const auto& r : aa_rounds_) {
            const float radius = 18.0f;
            const float gh = std::max(sample_terrain_height(r.pos.x, r.pos.z), rt_.water_level);
            if (r.pos.y - gh > radius + 12.0f) continue;      // +12: tree crowns, towers
            add(r.pos, radius, glm::vec3(1.0f, 0.22f, 0.10f), 2.6f, false);   // 720 m/s: gone in a frame
        }
        // The player's laser bolts light what they fly past.
        for (const auto& p : projectiles_)
            add(p.trail_prev, 12.0f, glm::vec3(1.0f, 0.28f, 0.80f), 3.2f, true);
        const size_t n = std::min<size_t>(8, cands.size());
        std::partial_sort(cands.begin(), cands.begin() + static_cast<std::ptrdiff_t>(n), cands.end(),
                          [](const Cand& a, const Cand& b) { return a.score > b.score; });
        for (size_t i = 0; i < 8; ++i) {
            data.fx_lights[i]    = i < n ? glm::vec4(cands[i].pos, cands[i].radius) : glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
            data.fx_light_col[i] = i < n ? glm::vec4(cands[i].col, cands[i].shadow) : glm::vec4(0.0f);
        }
        data.ripple_info.w = float(n);
    }
    // Seasons: the grass and the green of the ground. Spring is a fresher,
    // lighter green; autumn dries toward straw and ochre.
    {
        const float sp = season_w_[0], au = season_w_[2];
        if (sp + au > 0.004f) {
            auto tint = [&](glm::vec4& c, glm::vec3 spring_k, glm::vec3 autumn_k) {
                glm::vec3 v(c);
                const float lum = glm::dot(v, glm::vec3(0.3f, 0.5f, 0.2f));
                v = glm::mix(v, lum * spring_k, sp * 0.5f);
                v = glm::mix(v, lum * autumn_k, au * 0.6f);
                c = glm::vec4(v, c.w);
            };
            const glm::vec3 kSpring(0.95f, 1.40f, 0.60f), kAutumn(1.55f, 1.25f, 0.55f);
            tint(data.grass_color_top, kSpring, kAutumn);
            tint(data.grass_color_bottom, kSpring, kAutumn);
            tint(data.grass_color_ground, kSpring, kAutumn);
            tint(data.grass_color_ground_far, kSpring, kAutumn);
            tint(data.rm_ground_green, kSpring, kAutumn);
        }
    }
    // Wet ground is darker and a little more saturated.
    if (env_->wet > 0.001f) {
        const float k = 1.0f - 0.34f * env_->wet;
        for (glm::vec4* c : { &data.rm_ground_dry, &data.rm_ground_green, &data.rm_red_sand })
            *c = glm::vec4(glm::vec3(*c) * k, c->w);
    }
    data.rm_rock         = glm::vec4(rt_.rm_rock_color * (1.0f - 0.30f * env_->wet),
                                     glm::clamp(rt_.rm_ground_saturation * (1.0f + 0.25f * env_->wet), 0.0f, 2.0f));
    {
        const float deg2rad = 3.14159265358979f / 180.0f;
        if (rt_.water_style == 2) {
            // Lake style: pack lake knobs into the same UBO slots.
            //   .x = unused, .y = lake time speed,
            //   .z = lake uv scale, .w = unused
            data.water_river_extra = glm::vec4(
                0.0f,
                std::max(0.0f, rt_.water_lake_time_speed),
                std::max(0.05f, rt_.water_lake_uv_scale),
                0.0f);
            // extinct.w holds the lake's bump-fade distance / 100
            // (shader multiplies by 100 to get metres).
            data.water_river_extinct = glm::vec4(
                rt_.water_river_extinct_color,  // unused by lake shader
                std::max(0.05f, rt_.water_lake_bump_dist) * 0.01f);
        } else {
            data.water_river_extra = glm::vec4(
                rt_.water_river_flow_angle * deg2rad,
                std::max(0.0f, rt_.water_river_time_speed),
                std::max(0.05f, rt_.water_river_detail),
                std::max(0.0f, rt_.water_river_foam_amount));
            data.water_river_extinct = glm::vec4(
                rt_.water_river_extinct_color,
                std::max(0.0f, rt_.water_river_extinct_density));
        }
    }

    // Voxel building params for cube.frag's inline-RT shadow + GI rays,
    // water_vox_march reflection lookups, and the sun-block AABB in
    // terrain_raymarch.frag. MUST be shape 0's bounds — vox_entries[]
    // is indexed in shape 0's brick grid, so any other origin makes
    // the brick lookup misaligned (every brick coord points off by the
    // delta between shape 0 and the chosen origin → holes shot in the
    // main castle stop showing in water reflections after the first
    // chunk falls and shifts the union origin).
    //
    // Previous "union of all shapes" attempt broke water_vox_march on
    // post-collapse frames; the sun-block AABB in terrain_raymarch is
    // approximate anyway (chunks rarely shadow much terrain) so the
    // minor regression there is the right trade.
    if (voxel_->world && !voxel_->world->shapes().empty()) {
        const auto& s0 = voxel_->world->shapes()[0];
        glm::vec3 ext0(s0.dim_bricks[0] * voxel::kBrickSize,
                       s0.dim_bricks[1] * voxel::kBrickSize,
                       s0.dim_bricks[2] * voxel::kBrickSize);
        data.voxel_origin = glm::vec4(s0.origin_world, 1.0f);
        data.voxel_dims   = glm::vec4(ext0, voxel::kVoxelSize);
        data.voxel_grid   = glm::vec4(
            static_cast<float>(s0.dim_bricks[0]),
            static_cast<float>(s0.dim_bricks[1]),
            static_cast<float>(s0.dim_bricks[2]),
            voxel::kBrickSize);
    } else {
        data.voxel_origin = glm::vec4(0.0f);  // .w = 0 → disabled
    }

    // Castle road polyline (static after init_world; cached once).
    {
        // Cached resample, rebuilt whenever the road itself changes (was
        // built once and never refreshed).
        static std::vector<glm::vec4> road_gpu;
        static size_t road_sig = 0;
        static glm::ivec2 road_spur(0);
        // (the road of where the engine is: home's in home's chart, a site's in its own, none
        // anywhere else - the shaders know one road at a time)
        static const Road kNoRoad{};
        const globe::Chart& road_chart = globe::current();
        const Road& the_road = road_chart.home ? castle_road_ : site_here() ? site_road_ : kNoRoad;
        size_t sig = the_road.pts.size();
        if (!the_road.pts.empty()) {
            const glm::vec3 a = the_road.pts.front(), b = the_road.pts.back();
            sig = sig * 1000003u ^ std::hash<float>{}(a.x + 7.0f * a.z) ^ (std::hash<float>{}(b.x - 3.0f * b.z) << 1);
        }
        if (sig != road_sig) {
            road_spur = glm::ivec2(0);
            road_gpu = the_road.empty() ? std::vector<glm::vec4>{} : road_gpu_points(the_road, 64, &road_spur);
            road_sig = sig;
        }
        const int n = static_cast<int>(std::min<size_t>(road_gpu.size(), 64));
        // .w: the side road's segments (first * 128 + one-past-last; 0 = none).
        data.road_info = glm::vec4(static_cast<float>(n), the_road.half_width,
                                   the_road.shoulder, float(road_spur.x * 128 + road_spur.y));
        // (the old road's own bounds: its paving is drawn inside them - not the wider ones that take in the lane)
        data.road_bbox = the_road.with_highways ? glm::vec4(the_road.paved_bb_min, the_road.paved_bb_max) : glm::vec4(the_road.bb_min, the_road.bb_max);
        for (int i = 0; i < n; ++i) data.road_pts[i] = road_gpu[size_t(i)];
    }

    VmaAllocationInfo ai{};
    vmaGetAllocationInfo(allocator_, scene_ubo_alloc_, &ai);
    wm_cache_plan(data);
    upload(scene_ubo_buffer_, 0, &data, sizeof(data), ai.pMappedData);   // staged while a frame is in flight
    // Defensive flush — no-op on coherent host memory but required by spec
    // for non-coherent allocations (e.g. some non-NVIDIA drivers).
    vmaFlushAllocation(allocator_, scene_ubo_alloc_, 0, sizeof(data));
}

} // namespace vibe
