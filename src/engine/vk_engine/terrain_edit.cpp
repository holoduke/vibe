// terrain_edit.cpp - terrain sculpt brush, grass-mask painting, collision / BLAS refresh after edits, saving heights
// (split out of terrain_resources.cpp; see docs/ARCHITECTURE.md for the map.)
// Terrain CPU/GPU resources: heightmap sculpting (FBM noise/erosion brush),
// grass-eligibility mask, height textures, quadtree CLOD instancing, the
// async sun-shadow bake worker, and the curvature/sky-vis aux + grass-mask
// + fog-wisp procedural textures. Split out of world.cpp (maintainability).

#include "engine/vk_engine/internal.h"
#include "engine/gltf_loader.h"
#include "engine/grass.h"
#include "engine/skybox.h"
#include "engine/terrain.h"
#include "engine/texture.h"

#include <GLFW/glfw3.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <future>
#include <limits>
#include <utility>
#include <vector>

namespace vibe {


// Classic GLSL-style hash + value-noise + 3-octave FBM, used by every
// CPU sculpt / plateau-noise / erosion path. Was previously inlined as
// nested lambdas in three sites with three different signatures
// (cleanness review #8). Centralising it removes the duplicate
// `12.9898/78.233/43758.5453` magic triple and lets the compiler pick
// one inlining strategy.

namespace {
inline float fract_f(float v) { return v - std::floor(v); }
inline float terrain_hash21(float px, float pz) {
    // Mirrors GLSL hash21(vec2 p) — dot/fract over the (x,y,x) triple.
    float p3x = fract_f(px * 0.1031f);
    float p3y = fract_f(pz * 0.1031f);
    float p3z = fract_f(px * 0.1031f);
    float dot_p = p3x * (p3y + 19.19f) +
                  p3y * (p3z + 19.19f) +
                  p3z * (p3x + 19.19f);
    p3x += dot_p; p3y += dot_p; p3z += dot_p;
    return fract_f((p3x + p3y) * p3z);
}
struct NoiseDeriv { float v, dx, dz; };
inline NoiseDeriv terrain_noised(float px, float pz) {
    float ix = std::floor(px), iz = std::floor(pz);
    float fx = px - ix,        fz = pz - iz;
    float ux  = fx * fx * (3.0f - 2.0f * fx);
    float uz  = fz * fz * (3.0f - 2.0f * fz);
    float dux = 6.0f * fx * (1.0f - fx);
    float duz = 6.0f * fz * (1.0f - fz);
    float a = terrain_hash21(ix,        iz);
    float b = terrain_hash21(ix + 1.0f, iz);
    float c = terrain_hash21(ix,        iz + 1.0f);
    float d = terrain_hash21(ix + 1.0f, iz + 1.0f);
    float value = a + (b - a) * ux + (c - a) * uz + (a - b - c + d) * ux * uz;
    float dvx = dux * ((b - a) + (a - b - c + d) * uz);
    float dvz = duz * ((c - a) + (a - b - c + d) * ux);
    return { value, dvx, dvz };
}
// Same FBM as terrain_raymarch.frag::terrainM — `noised` accumulates
// the analytical derivative `d`, and `n.x / (1 + dot(d,d))` damps
// high-frequency octaves on steep slopes, producing the ridge/valley
// structure the user sees in the raymarched terrain.
// Per-octave rotation matrix matches the shader's `m2`.
inline float terrain_fbm_eroded(float wx, float wz, float scale,
                                  int octaves) {
    constexpr float kM00 =  0.8f, kM01 = -0.6f;
    constexpr float kM10 =  0.6f, kM11 =  0.8f;
    float px = wx * scale, pz = wz * scale;
    float a = 0.0f, b = 1.0f;
    float dx_acc = 0.0f, dz_acc = 0.0f;
    octaves = std::clamp(octaves, 1, 12);
    for (int i = 0; i < octaves; ++i) {
        NoiseDeriv n = terrain_noised(px, pz);
        dx_acc += n.dx;
        dz_acc += n.dz;
        a += b * n.v / (1.0f + dx_acc * dx_acc + dz_acc * dz_acc);
        b *= 0.5f;
        float npx = kM00 * px + kM01 * pz;
        float npz = kM10 * px + kM11 * pz;
        px = npx * 2.0f;
        pz = npz * 2.0f;
    }
    // Re-centre roughly around 0 — fbm with `n.v` in [0,1] sums to в‰€
    // half the harmonic series. The bias is approximate and harmless;
    // the brush noise scale only needs symmetric variation around 0.
    return a - 0.5f;
}
} // namespace

// Distance-based LOD selector for terrain chunks. Used by all three
// terrain draw paths (depth pre-pass, color pass, shadow pass) so LOD
// stays consistent — if the pre-pass picked half-LOD and the color
// pass picked full, the depth values would mismatch and the color
// pass's LESS_OR_EQUAL test would discard fragments. Thresholds are
// pulled from rt_ so the user can push higher-resolution terrain
// further out via the sliders in the Graphics Settings menu.
// pick_terrain_lod moved to terrain.h so sun_shadow.cpp can share it.
// (The old pick_terrain_morph helper was dead code — removed.)

// Paint the R channel of the grass eligibility mask. Add: push toward
// 1, Remove: push toward 0. Uses the same world hit + radius the height
// brush uses, but indexes the 1024² mask over a 2048 m world (matches
// the bake in init_grass_mask_texture). Marks `grass_mask_dirty_` so
// the next safe frame re-uploads the mask to the GPU. Does NOT touch
// heights / chunks / BLAS / Jolt — pure mask edit.
static void paint_grass_mask(std::vector<uint8_t>& mask, int dim,
                              const glm::vec3& hit, float radius_m,
                              float strength_per_sec, float dt,
                              bool add) {
    if (mask.empty()) return;
    const float kWorldSide = 2048.0f;   // matches init_grass_mask_texture
    const float cells_per_metre = static_cast<float>(dim) / kWorldSide;
    // World → mask cell coords. Mask covers [-1024..+1024] m centred at
    // origin; cell (0,0) = world (-1024, -1024).
    float cx = (hit.x + kWorldSide * 0.5f) * cells_per_metre;
    float cz = (hit.z + kWorldSide * 0.5f) * cells_per_metre;
    float rcells_f = radius_m * cells_per_metre;
    int rcells = static_cast<int>(std::ceil(rcells_f)) + 1;
    int ix0 = std::max(0, static_cast<int>(std::floor(cx)) - rcells);
    int ix1 = std::min(dim - 1, static_cast<int>(std::ceil(cx))  + rcells);
    int iz0 = std::max(0, static_cast<int>(std::floor(cz)) - rcells);
    int iz1 = std::min(dim - 1, static_cast<int>(std::ceil(cz))  + rcells);
    if (ix0 > ix1 || iz0 > iz1) return;
    const float dstrength = strength_per_sec * dt;
    for (int iz = iz0; iz <= iz1; ++iz) {
        for (int ix = ix0; ix <= ix1; ++ix) {
            float dx = static_cast<float>(ix) - cx;
            float dz = static_cast<float>(iz) - cz;
            float d  = std::sqrt(dx * dx + dz * dz);
            if (d > rcells_f) continue;
            float t = 1.0f - (d / rcells_f);
            float falloff = t * t * (3.0f - 2.0f * t);
            size_t idx = (static_cast<size_t>(iz) * static_cast<size_t>(dim) +
                          static_cast<size_t>(ix)) * 2;
            float v = static_cast<float>(mask[idx]) / 255.0f;
            float delta = dstrength * falloff;
            v += add ? delta : -delta;
            v = std::clamp(v, 0.0f, 1.0f);
            mask[idx] = static_cast<uint8_t>(v * 255.0f);
        }
    }
}

void VulkanEngine::reupload_grass_mask() {
    if (grass_mask_data_.empty() || grass_mask_.image == VK_NULL_HANDLE) return;
    const VkDeviceSize bytes = grass_mask_data_.size();
    VkBuffer stage = VK_NULL_HANDLE;
    VmaAllocation stage_alloc = nullptr;
    VkBufferCreateInfo bci{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = nullptr, .flags = 0, .size = bytes,
        .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0, .pQueueFamilyIndices = nullptr,
    };
    VmaAllocationCreateInfo bac{};
    bac.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
    bac.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                VMA_ALLOCATION_CREATE_MAPPED_BIT;
    vmaCreateBuffer(allocator_, &bci, &bac, &stage, &stage_alloc, nullptr);
    VmaAllocationInfo ai{};
    vmaGetAllocationInfo(allocator_, stage_alloc, &ai);
    std::memcpy(ai.pMappedData, grass_mask_data_.data(), bytes);
    vkinit::one_time_submit(device_, graphics_queue_, graphics_queue_family_,
                            [&](VkCommandBuffer cb) {
        vkinit::transition_image(cb, grass_mask_.image,
                                  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        VkBufferImageCopy region{};
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.layerCount = 1;
        region.imageExtent = { static_cast<uint32_t>(kGrassMaskDim),
                                static_cast<uint32_t>(kGrassMaskDim), 1 };
        vkCmdCopyBufferToImage(cb, stage, grass_mask_.image,
                                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                1, &region);
        vkinit::transition_image(cb, grass_mask_.image,
                                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    });
    vmaDestroyBuffer(allocator_, stage, stage_alloc);
    grass_mask_dirty_ = false;
}

bool VulkanEngine::save_grass_mask() {
    if (grass_mask_data_.empty()) return false;
    std::ofstream f("assets/level1_grass_mask.bin", std::ios::binary);
    if (!f.is_open()) {
        log::error("save_grass_mask: failed to open assets/level1_grass_mask.bin");
        return false;
    }
    int32_t dim = kGrassMaskDim;
    f.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
    f.write(reinterpret_cast<const char*>(grass_mask_data_.data()),
            grass_mask_data_.size());
    log::infof("[grass] saved mask to assets/level1_grass_mask.bin (%zu bytes)",
               grass_mask_data_.size());
    return true;
}

bool VulkanEngine::load_grass_mask() {
    std::ifstream f("assets/level1_grass_mask.bin", std::ios::binary);
    if (!f.is_open()) return false;
    int32_t fdim = 0;
    f.read(reinterpret_cast<char*>(&fdim), sizeof(fdim));
    if (fdim != kGrassMaskDim) {
        log::warnf("[grass] saved mask dim mismatch (saved %d, want %d) — ignoring",
                   fdim, kGrassMaskDim);
        return false;
    }
    const size_t want = static_cast<size_t>(kGrassMaskDim) * kGrassMaskDim * 2;
    grass_mask_data_.resize(want);
    f.read(reinterpret_cast<char*>(grass_mask_data_.data()), want);
    log::infof("[grass] loaded mask from assets/level1_grass_mask.bin (%zu bytes)",
               want);
    return true;
}

void VulkanEngine::apply_terrain_brush(float dt) {
    if (!terrain_brush_has_hit_) return;
    // Grass-paint modes only touch the eligibility mask — bail before
    // the height-array path so they work even when chunks / heights
    // happen to be empty (defensive — Phase 4 always populates both).
    if (brush_mode_is_grass(terrain_brush_mode_)) {
        paint_grass_mask(grass_mask_data_, kGrassMaskDim,
                          terrain_brush_world_pos_,
                          terrain_brush_radius_,
                          terrain_brush_strength_, dt,
                          terrain_brush_mode_ == TerrainBrushMode::GrassAdd);
        grass_mask_dirty_ = true;
        return;
    }
    if (terrain_data_.heights.empty()) return;
    const float r = terrain_brush_radius_;
    const float w = terrain_data_.cell;
    const int dim = terrain_data_.dim;
    const int W = dim + 1;

    // Convert world hit point to grid coords + brush radius in cells.
    float cx = (terrain_brush_world_pos_.x - terrain_data_.origin_x) / w;
    float cz = (terrain_brush_world_pos_.z - terrain_data_.origin_z) / w;
    int rcells = static_cast<int>(std::ceil(r / w)) + 1;
    int ix0 = std::max(0, static_cast<int>(std::floor(cx)) - rcells);
    int ix1 = std::min(W - 1, static_cast<int>(std::ceil(cx))  + rcells);
    int iz0 = std::max(0, static_cast<int>(std::floor(cz)) - rcells);
    int iz1 = std::min(W - 1, static_cast<int>(std::ceil(cz))  + rcells);
    if (ix0 > ix1 || iz0 > iz1) return;

    const float strength = terrain_brush_strength_ * dt;

    // 3-octave value-noise via cheap hash. Centred around 0 so it
    // multiplies the sculpt delta as `1 + noise·strength` — values
    // > 1 raise more, < 1 raise less. Result: sculpted patches
    // pick up natural-looking surface variation instead of flat
    // squares. Cheap (3 hashes per cell during sculpt only).
    const float n_str  = std::clamp(terrain_brush_noise_strength_, 0.0f, 1.0f);
    const float n_freq = std::max(0.02f, terrain_brush_noise_freq_);
    const bool  use_fbm_eroded = terrain_brush_use_fbm_erosion_;
    const int   fbm_octaves    = terrain_brush_fbm_octaves_;
    auto fbm_var = [n_freq, use_fbm_eroded, fbm_octaves](float wx, float wz) {
        if (use_fbm_eroded) {
            // Mesh has 1 m cells. The raymarcher's macro-mountain
            // scale (0.003 в‰€ 333 m base wavelength) means adjacent
            // cells sample nearly identical noise — the brush ends up
            // applying a smooth blob with no per-cell variation, and
            // the surface reads as faceted because the geometry has
            // no high-freq detail to break up the triangles. Higher
            // base scale (0.05) puts ~20 m at octave 0 and ≤ 0.5 m
            // at 6 octaves — every 1 m cell gets a distinct sample,
            // so brushed areas inherit cell-level ridge variation
            // similar to the procedural FBM heights.
            const float kBrushScale = 0.05f;
            return terrain_fbm_eroded(wx, wz, kBrushScale * n_freq * 4.0f,
                                       fbm_octaves);
        }
        float v = 0.5f  * fbm_noise2(wx * n_freq,         wz * n_freq) +
                  0.25f * fbm_noise2(wx * n_freq * 2.07f, wz * n_freq * 2.07f) +
                  0.125f* fbm_noise2(wx * n_freq * 4.13f, wz * n_freq * 4.13f);
        return v - 0.4375f;   // re-centre on 0
    };

    for (int iz = iz0; iz <= iz1; ++iz) {
        for (int ix = ix0; ix <= ix1; ++ix) {
            float dx = (static_cast<float>(ix) - cx) * w;
            float dz = (static_cast<float>(iz) - cz) * w;
            float d = std::sqrt(dx * dx + dz * dz);
            if (d > r) continue;
            // Smoothstep falloff so the brush has a soft edge.
            float t = 1.0f - (d / r);
            float falloff = t * t * (3.0f - 2.0f * t);
            size_t idx = static_cast<size_t>(iz) * static_cast<size_t>(W) +
                         static_cast<size_t>(ix);
            float& h = terrain_data_.heights[idx];
            // Per-cell noise multiplier — only kicks in for additive
            // modes (raise / lower), not smooth / flatten.
            float wx = terrain_data_.origin_x + ix * terrain_data_.cell;
            float wz = terrain_data_.origin_z + iz * terrain_data_.cell;
            float n_mult = 1.0f + 2.0f * fbm_var(wx, wz) * n_str;
            switch (terrain_brush_mode_) {
                case TerrainBrushMode::Raise:
                    h += strength * falloff * n_mult;
                    break;
                case TerrainBrushMode::Lower:
                    h -= strength * falloff * n_mult;
                    break;
                case TerrainBrushMode::Smooth: {
                    // 4-tap blur toward neighbour mean.
                    int xm = std::max(0, ix - 1);
                    int xp = std::min(W - 1, ix + 1);
                    int zm = std::max(0, iz - 1);
                    int zp = std::min(W - 1, iz + 1);
                    auto at = [&](int x, int z) {
                        return terrain_data_.heights[
                            static_cast<size_t>(z) * static_cast<size_t>(W) +
                            static_cast<size_t>(x)];
                    };
                    float mean = 0.25f * (at(xm, iz) + at(xp, iz) +
                                          at(ix, zm) + at(ix, zp));
                    h += (mean - h) * std::min(1.0f, strength * 4.0f * falloff);
                    break;
                }
                case TerrainBrushMode::Flatten: {
                    float target = terrain_brush_flatten_target_;
                    h += (target - h) * std::min(1.0f, strength * 0.5f * falloff);
                    break;
                }
                case TerrainBrushMode::Erode: {
                    // Add FBM-eroded detail (same n.v/(1+|в€‡d|²) recipe
                    // as the raymarched terrain), scaled by brush
                    // falloff. The base FBM scale here is much higher
                    // than the raymarcher's 0.003 — at typical brush
                    // radii (≤ 10 m) the raymarcher's macro-mountain
                    // wavelength means the WHOLE brush samples nearly
                    // the same noise value, which feels like a smooth
                    // uniform bump (the "smoothing" symptom). Pumping
                    // the base scale to 0.05 gives ~20 m at octave 0
                    // and ~30 cm at 6 octaves — actual ridges and
                    // valleys appear inside the brush footprint.
                    const float kErodeScale = 0.05f;
                    float n = terrain_fbm_eroded(
                        wx, wz, kErodeScale * n_freq * 4.0f,
                        terrain_brush_fbm_octaves_);
                    // Erosion amplitude scales with brush radius so a
                    // larger brush makes proportionally taller ridges
                    // (a 1 m brush shouldn't punch a 4 m peak). Per-
                    // frame accumulation = noise·radius·dt · falloff;
                    // a 1-sec hold on a 6 m brush gives ~±1.5 m peak
                    // displacement — visible without obliterating the
                    // surrounding terrain.
                    float amp = terrain_brush_radius_ * 0.6f;
                    h += n * amp * dt * falloff;
                    break;
                }
                case TerrainBrushMode::ErodeSmooth: {
                    // Inverse: pull detail back toward the 4-neighbour
                    // mean. Same kernel as Smooth but applied with the
                    // erosion-strength scaling so the brush feels like
                    // the inverse of Erode rather than a duplicate of
                    // the existing Smooth mode.
                    int xm = std::max(0, ix - 1);
                    int xp = std::min(W - 1, ix + 1);
                    int zm = std::max(0, iz - 1);
                    int zp = std::min(W - 1, iz + 1);
                    auto at = [&](int x, int z) {
                        return terrain_data_.heights[
                            static_cast<size_t>(z) * static_cast<size_t>(W) +
                            static_cast<size_t>(x)];
                    };
                    float mean = 0.25f * (at(xm, iz) + at(xp, iz) +
                                          at(ix, zm) + at(ix, zp));
                    h += (mean - h) * std::min(1.0f, strength * 2.0f * falloff);
                    break;
                }
                case TerrainBrushMode::GrassAdd:
                case TerrainBrushMode::GrassRemove:
                    // Grass-paint modes branched out at the top of
                    // apply_terrain_brush — the height loop never sees them.
                    break;
            }
        }
    }
    // Castle-courtyard sink (same clamp init_world applies to the source
    // heights): a brush stroke over the castle must not raise the ground
    // through the courtyard floor slab.
    {
        const float kSinkR = 10.5f, kSinkH = 22.0f - 0.6f;
        for (int iz = iz0; iz <= iz1; ++iz) {
            const float wz = terrain_data_.origin_z + float(iz) * w;
            if (std::abs(wz) >= kSinkR) continue;
            for (int ix = ix0; ix <= ix1; ++ix) {
                const float wx = terrain_data_.origin_x + float(ix) * w;
                if (std::abs(wx) >= kSinkR) continue;
                float& hv = terrain_data_.heights[size_t(iz) * W + ix];
                hv = (std::min)(hv, kSinkH);
            }
        }
    }

    // CLOD raster picks the edit up from the heightmap TEXTURE — no mesh
    // rebake. Refresh the min/max pyramid for the touched region (culling
    // AABBs) and flag the textures + BLAS + collision for re-upload.
    update_clod_pyramid_region(terrain_clod_pyramid_,
                               terrain_data_.heights.data(),
                               terrain_data_.dim, terrain_data_.cell,
                               16.0f, ix0, ix1, iz0, iz1);
    terrain_blas_dirty_ = true;
    terrain_jolt_dirty_ = true;
    terrain_height_dirty_ = true;   // re-upload delta + full height textures
}

void VulkanEngine::refresh_terrain_collision() {
    if (!terrain_jolt_dirty_) return;
    if (terrain_data_.heights.empty()) return;
    // Jolt has no incremental update for HeightFieldShape — we rebuild
    // the whole shape. Mirrors the sub-grid trim done at level load:
    // when sample_count would be odd Jolt's block-size requirement
    // forces us onto a 2048×2048 sub-grid taken from the 2049² heightmap.
    //
    // For each Jolt sample we store max(self, neighbours). Reason:
    // Jolt's HeightFieldShape is a per-cell triangulation — the
    // collision surface is a piecewise-linear plane between corners,
    // which sits BELOW the visual surface (whether the rasterized
    // mesh's smooth-shaded reading or the raymarched FBM) at sub-cell
    // peaks. Dynamic boxes settle on the linear plane and visually
    // appear "a bit inside" the terrain. Lifting the corner samples
    // to the local max raises the triangle plane so boxes sit AT or
    // slightly ABOVE the visual surface instead of below it. Player
    // ground-clamp still uses the un-lifted terrain_data_.heights
    // (separate sample_terrain_height path), so the player's feet
    // stay anchored to the visual mesh exactly as before.
    if (physics_) {
        const int W = terrain_data_.dim + 1;
        const int physics_samples = (W % 2 == 0) ? W : (W - 1);
        if (physics_samples == W) {
            physics_->add_static_heightfield(terrain_data_.heights.data(),
                                             terrain_data_.dim,
                                             glm::vec2(terrain_data_.origin_x,
                                                        terrain_data_.origin_z),
                                             terrain_data_.cell);
        } else {
            std::vector<float> jolt_heights(static_cast<size_t>(physics_samples) *
                                             static_cast<size_t>(physics_samples));
            for (int z = 0; z < physics_samples; ++z) {
                std::memcpy(jolt_heights.data() +
                                static_cast<size_t>(z) *
                                static_cast<size_t>(physics_samples),
                            terrain_data_.heights.data() +
                                static_cast<size_t>(z) * static_cast<size_t>(W),
                            static_cast<size_t>(physics_samples) * sizeof(float));
            }
            physics_->add_static_heightfield(jolt_heights.data(),
                                             physics_samples - 1,
                                             glm::vec2(terrain_data_.origin_x,
                                                        terrain_data_.origin_z),
                                             terrain_data_.cell);
        }
    }
    terrain_jolt_dirty_ = false;
}

void VulkanEngine::ensure_terrain_raster_built() {
    if (terrain_clod_patch_.vertex_buffer) return;       // already built
    if (terrain_data_.heights.empty()) return;          // no heightmap to build from
    log::info("[terrain] lazy-building rasterised mesh + CLOD patch");
    vkDeviceWaitIdle(device_);
    Heightmap hm{};
    hm.dim       = terrain_data_.dim;
    hm.cell      = terrain_data_.cell;
    hm.origin_x  = terrain_data_.origin_x;
    hm.origin_z  = terrain_data_.origin_z;
    hm.heights   = terrain_data_.heights;
    terrain_mesh_ = build_terrain_mesh(device_, allocator_,
                                       graphics_queue_, graphics_queue_family_, hm);
    init_terrain_clod();
    // Note: the terrain BLAS is built once in init_rt and not rebuilt
    // here. Toggling raymarch off mid-session draws raster terrain
    // correctly but RT shadow/AO/GI rays from cube.frag will not see
    // it until restart. tlas_includes_terrain_blas() handles the
    // address==0 case already, so this is a quality fallback, not a
    // crash risk.
}

void VulkanEngine::ensure_grass_raster_built() {
    // Attempted once: a build that places 0 blades (empty altitude band)
    // used to repeat every frame - device wait + 800k-candidate placement
    // + a leaked blade mesh per frame.
    if (grass_raster_attempted_ || grass_.instance_count > 0) return;
    grass_raster_attempted_ = true;
    if (terrain_data_.heights.empty()) return;
    log::info("[grass] lazy-building rasterised blade placement");
    vkDeviceWaitIdle(device_);
    Heightmap hm{};
    hm.dim       = terrain_data_.dim;
    hm.cell      = terrain_data_.cell;
    hm.origin_x  = terrain_data_.origin_x;
    hm.origin_z  = terrain_data_.origin_z;
    hm.heights   = terrain_data_.heights;
    GrassParams gp{};
    gp.height_min  = -20.0f;
    gp.height_max  = 200.0f;
    gp.half_extent = 200.0f;
    gp.keep_out_xz = glm::vec2(12.0f, 12.0f);  // full castle footprint
    gp.road = &castle_road_;
    gp.max_blades  = 800000;
    destroy_grass(allocator_, grass_);   // a previous empty build still owns its blade mesh
    grass_ = build_grass(device_, allocator_,
                         graphics_queue_, graphics_queue_family_,
                         hm, gp);
}

void VulkanEngine::refresh_terrain_blas() {
    if (!terrain_blas_dirty_) return;
    // BLAS rebuild is the heavy part. Defer to mouse-up so per-frame
    // cost stays low during a stroke. The merged terrain mesh's vertex
    // buffer is rebuilt from the current heightmap, then the BLAS is
    // re-built in place. RT shadows/GI will be slightly stale until
    // this fires, which reads as "lighting catches up after you stop
    // sculpting" — acceptable.
    if (terrain_mesh_.vertex_buffer && !terrain_data_.heights.empty()) {
        Heightmap hm{};
        hm.dim = terrain_data_.dim;
        hm.cell = terrain_data_.cell;
        hm.origin_x = terrain_data_.origin_x;
        hm.origin_z = terrain_data_.origin_z;
        hm.heights = terrain_data_.heights;
        // Rebuild the full-mesh vertex buffer in place by abusing the
        // chunk rebuild helper on a virtual "chunk" covering the whole
        // mesh.
        TerrainChunk fake{};
        fake.mesh = terrain_mesh_;
        fake.origin_ix = 0;
        fake.origin_iz = 0;
        fake.sample_dim = terrain_data_.dim + 1;
        // Frames in flight: the previous frame may still read the vertex
        // buffer / BLAS. Mouse-up only, so a full drain is fine.
        vkDeviceWaitIdle(device_);
        rebuild_chunk_vertices(device_, allocator_, graphics_queue_,
                               graphics_queue_family_, hm, fake);
        // Refit the terrain BLAS to the new vertices (it was never updated:
        // RT shadows / AO / GI kept tracing the pre-sculpt surface).
        VkBuffer scratch = VK_NULL_HANDLE; VmaAllocation scratch_alloc = nullptr;
        vkinit::one_time_submit(device_, graphics_queue_, graphics_queue_family_, [&](VkCommandBuffer cb) {
            VkMemoryBarrier2 mb{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
            mb.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
            mb.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
            mb.dstStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
            mb.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR;
            VkDependencyInfo di{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
            di.memoryBarrierCount = 1; di.pMemoryBarriers = &mb;
            vkCmdPipelineBarrier2(cb, &di);
            record_terrain_blas_refit(cb, &scratch, &scratch_alloc);
            mb.srcStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
            mb.srcAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
            mb.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
            mb.dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR;
            vkCmdPipelineBarrier2(cb, &di);
        });
        if (scratch) vmaDestroyBuffer(allocator_, scratch, scratch_alloc);
        // The TLAS caches BLAS bounds at build time: force a full rebuild.
        prev_dyn_signature_ = 0;
        prev_tlas_n_ = UINT32_MAX;
    }
    terrain_blas_dirty_ = false;
}


bool VulkanEngine::save_terrain_heights() {
    if (terrain_data_.heights.empty()) return false;
    std::ofstream f("assets/level1_heights.bin", std::ios::binary);
    if (!f.is_open()) {
        log::error("save_terrain_heights: failed to open assets/level1_heights.bin");
        return false;
    }
    int32_t dim = terrain_data_.dim;
    f.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
    f.write(reinterpret_cast<const char*>(&terrain_data_.cell), sizeof(float));
    f.write(reinterpret_cast<const char*>(&terrain_data_.origin_x), sizeof(float));
    f.write(reinterpret_cast<const char*>(&terrain_data_.origin_z), sizeof(float));
    f.write(reinterpret_cast<const char*>(terrain_data_.heights.data()),
            terrain_data_.heights.size() * sizeof(float));
    log::infof("[terrain] saved %zu heights to assets/level1_heights.bin",
               terrain_data_.heights.size());
    return true;
}

void VulkanEngine::add_plateau_noise(float amplitude_m, float frequency) {
    if (terrain_data_.heights.empty()) return;
    // Gameplay plateau is centred at origin with a 28 m half-extent
    // (matches HeightmapParams in init_world). Apply noise inside,
    // taper it off across the 24 m blend ring so the rim stays smooth.
    const glm::vec2 plat_centre(0.0f, 0.0f);
    const glm::vec2 plat_ext   (28.0f, 28.0f);
    const float     plat_blend = 24.0f;

    auto fbm2 = [](glm::vec2 p) {
        float v = 0.5f  * fbm_noise2(p.x,         p.y) +
                  0.25f * fbm_noise2(p.x * 2.07f, p.y * 2.07f) +
                  0.125f* fbm_noise2(p.x * 4.13f, p.y * 4.13f);
        return v - 0.4375f;   // re-centre around zero
    };

    const int W = terrain_data_.dim + 1;
    const int H = terrain_data_.dim + 1;
    for (int iz = 0; iz < H; ++iz) {
        float wz = terrain_data_.origin_z + iz * terrain_data_.cell;
        for (int ix = 0; ix < W; ++ix) {
            float wx = terrain_data_.origin_x + ix * terrain_data_.cell;
            glm::vec2 d = glm::abs(glm::vec2(wx, wz) - plat_centre) - plat_ext;
            float dout = std::max(std::max(d.x, 0.0f), std::max(d.y, 0.0f));
            float t = std::clamp(dout / plat_blend, 0.0f, 1.0f);
            float w = 1.0f - t * t * (3.0f - 2.0f * t);    // 1 inside, 0 outside
            if (w <= 0.001f) continue;
            float n = fbm2(glm::vec2(wx, wz) * frequency);
            terrain_data_.heights[static_cast<size_t>(iz) *
                                   static_cast<size_t>(W) +
                                   static_cast<size_t>(ix)] +=
                n * amplitude_m * w;
        }
    }
    log::infof("[terrain] plateau noise applied (amp=%.2f m, freq=%.3f)",
               amplitude_m, frequency);
    terrain_height_dirty_ = true;
    // Physics, RT and culling must follow the new surface too (only the
    // height textures did: walk-through bumps, stale RT shadows).
    terrain_clod_pyramid_ = build_clod_pyramid(terrain_data_.heights.data(),
                                               terrain_data_.dim,
                                               terrain_data_.cell, 16.0f);
    terrain_blas_dirty_ = true;
    terrain_jolt_dirty_ = true;
    refresh_terrain_blas();
    refresh_terrain_collision();
}

void VulkanEngine::add_fbm_erosion_to_sculpted(float amplitude_factor,
                                                 float frequency) {
    if (terrain_data_.heights.empty()) return;
    if (terrain_baseline_heights_.size() != terrain_data_.heights.size()) return;

    auto fbm5 = [](glm::vec2 p) {
        // 5-octave value-noise FBM via the shared fbm_noise2 helper.
        // Decreasing amplitude per octave produces the classic
        // ridged-mountain look.
        float v = 0.50000f * fbm_noise2(p.x,          p.y) +
                  0.25000f * fbm_noise2(p.x * 2.07f,  p.y * 2.07f) +
                  0.12500f * fbm_noise2(p.x * 4.13f,  p.y * 4.13f) +
                  0.06250f * fbm_noise2(p.x * 8.21f,  p.y * 8.21f) +
                  0.03125f * fbm_noise2(p.x * 16.47f, p.y * 16.47f);
        return v - 0.484375f;   // re-centre around zero
    };

    const int W = terrain_data_.dim + 1;
    const int H = terrain_data_.dim + 1;
    int touched = 0;
    for (int iz = 0; iz < H; ++iz) {
        float wz = terrain_data_.origin_z + iz * terrain_data_.cell;
        for (int ix = 0; ix < W; ++ix) {
            const size_t idx = static_cast<size_t>(iz) *
                               static_cast<size_t>(W) +
                               static_cast<size_t>(ix);
            float delta = terrain_data_.heights[idx] -
                          terrain_baseline_heights_[idx];
            // Only erode cells the user actually raised — preserve
            // procedural terrain + lowered areas as-is.
            if (delta <= 0.25f) continue;
            float wx = terrain_data_.origin_x + ix * terrain_data_.cell;
            float n = fbm5(glm::vec2(wx, wz) * frequency);
            terrain_data_.heights[idx] += n * delta * amplitude_factor;
            ++touched;
        }
    }
    log::infof("[terrain] FBM erosion baked into %d sculpted cells "
               "(amp=%.2f, freq=%.3f)", touched, amplitude_factor, frequency);
    terrain_height_dirty_ = true;
    // CLOD raster follows the heightmap texture; refresh the whole
    // min/max pyramid since the erosion touched sculpted cells anywhere.
    terrain_clod_pyramid_ = build_clod_pyramid(terrain_data_.heights.data(),
                                               terrain_data_.dim,
                                               terrain_data_.cell, 16.0f);
    terrain_blas_dirty_ = true;     // physics + RT follow the eroded surface
    terrain_jolt_dirty_ = true;
    refresh_terrain_blas();
    refresh_terrain_collision();
}


} // namespace vibe
