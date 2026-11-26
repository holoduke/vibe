// regions.cpp - stored ground anywhere on the planet: the regions' height tiles, streamed round
// the camera (docs/ARCHITECTURE.md "Regions"; engine/region_tile.h is the reference for the
// tiles themselves, engine/sector.h for their addresses).
//
//   init_regions()              the array texture (a layer a slot), the table the shaders read
//   regions_update()            once a frame: which tiles the index files list within reach,
//                               one file read at a time on a worker, slots freed out of reach,
//                               the loaded set published (CPU: region::set_active; GPU: the table)
//   record_region_uploads()     a read (or edited) tile -> its layer, one a frame
//   regions_ground_changed()    tells what stands on the ground that a tile came, went or changed
//   regions_brush* / _menu      the terrain brush on region tiles; new tile; the test shapes
//
// Hooks: VIBE_LOG_REGIONS=1, VIBE_NO_REGIONS=1, VIBE_REGION_DIR=<folder> (instead of
// cache/regions), VIBE_REGION_TEST=<mesa|crater>,<east m>,<south m>[,<radius>,<height>],
// VIBE_REGION_NEW=1.

#include "engine/vk_engine/internal.h"
#include "engine/vk_engine/state/regions.h"
#include "engine/vk_engine/state/terrain_patches.h"
#include "engine/globe.h"
#include "engine/region_tile.h"
#include "engine/sector.h"
#include "engine/terrain.h"

#include <GLFW/glfw3.h>
#include <imgui.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <string>

namespace vibe {

namespace {
constexpr int kN = region::kTileN;
constexpr VkDeviceSize kTileBytes = VkDeviceSize(kN) * kN * 2;

void layer_barrier(VkCommandBuffer cmd, VkImage image, uint32_t layer, uint32_t count, VkImageLayout from, VkImageLayout to) {
    VkImageMemoryBarrier b{};
    b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    b.oldLayout = from; b.newLayout = to;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = image;
    b.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, layer, count };
    const bool to_dst = to == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    b.srcAccessMask = from == VK_IMAGE_LAYOUT_UNDEFINED ? 0 : to_dst ? VK_ACCESS_SHADER_READ_BIT : VK_ACCESS_TRANSFER_WRITE_BIT;
    b.dstAccessMask = to_dst ? VK_ACCESS_TRANSFER_WRITE_BIT : VK_ACCESS_SHADER_READ_BIT;
    const VkPipelineStageFlags sh = VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    vkCmdPipelineBarrier(cmd, from == VK_IMAGE_LAYOUT_UNDEFINED ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT : to_dst ? sh : VK_PIPELINE_STAGE_TRANSFER_BIT,
                         to_dst ? VK_PIPELINE_STAGE_TRANSFER_BIT : sh, 0, 0, nullptr, 0, nullptr, 1, &b);
}

double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}
}  // namespace

void VulkanEngine::init_regions() {
    RegionState& S = *rg_;
    S.off = std::getenv("VIBE_NO_REGIONS") != nullptr;
    S.log = std::getenv("VIBE_LOG_REGIONS") != nullptr;
    if (const char* d = std::getenv("VIBE_REGION_DIR")) S.store.cache_dir = d;

    // The tiles: 16-bit signed, a layer a slot. (Made even when the system is off: the terrain
    // shaders' bindings must be valid.)
    VkImageCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = VK_FORMAT_R16_SNORM;
    ici.extent = { uint32_t(kN), uint32_t(kN), 1 };
    ici.mipLevels = 1;
    ici.arrayLayers = uint32_t(region::kSlots);
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    vk_check(vmaCreateImage(allocator_, &ici, &aci, &S.image, &S.alloc, nullptr), "region tiles image");
    VkImageViewCreateInfo vci{};
    vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vci.image = S.image;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    vci.format = VK_FORMAT_R16_SNORM;
    vci.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, uint32_t(region::kSlots) };
    vk_check(vkCreateImageView(device_, &vci, nullptr, &S.view), "region tiles view");
    VkSamplerCreateInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    si.magFilter = si.minFilter = VK_FILTER_NEAREST;        // (read with texelFetch: region_tiles.glsl)
    si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    vk_check(vkCreateSampler(device_, &si, nullptr, &S.sampler), "region tiles sampler");
    vkinit::one_time_submit(device_, graphics_queue_, graphics_queue_family_, [&](VkCommandBuffer cb) {
        layer_barrier(cb, S.image, 0, uint32_t(region::kSlots), VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    });

    // The table (region::GpuTable): written by the CPU when the loaded set changes.
    VkBufferCreateInfo bci{};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = sizeof(region::GpuTable);
    bci.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    VmaAllocationCreateInfo bac{};
    bac.usage = VMA_MEMORY_USAGE_AUTO;
    bac.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    VmaAllocationInfo info{};
    vk_check(vmaCreateBuffer(allocator_, &bci, &bac, &S.ubo, &S.ubo_alloc, &info), "region table buffer");
    S.ubo_map = info.pMappedData;
    region::GpuTable empty;
    region::fill_gpu(region::ActiveSet{}, empty);
    std::memcpy(S.ubo_map, &empty, sizeof(empty));
    vmaFlushAllocation(allocator_, S.ubo_alloc, 0, VK_WHOLE_SIZE);

    VkDescriptorBufferInfo bi{ S.ubo, 0, sizeof(region::GpuTable) };
    VkDescriptorImageInfo ii{ S.sampler, S.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
    VkWriteDescriptorSet w[2]{};
    w[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w[0].dstSet = scene_desc_set_; w[0].dstBinding = 49; w[0].descriptorCount = 1;
    w[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    w[0].pBufferInfo = &bi;
    w[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w[1].dstSet = scene_desc_set_; w[1].dstBinding = 50; w[1].descriptorCount = 1;
    w[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    w[1].pImageInfo = &ii;
    vkUpdateDescriptorSets(device_, 2, w, 0, nullptr);
    log::infof("[regions] height tiles: %d slots of %d x %d (%.1f MB), folders %s, %s%s", region::kSlots, kN, kN,
               double(kTileBytes) * region::kSlots / (1024.0 * 1024.0), S.store.local_dir.c_str(), S.store.cache_dir.c_str(), S.off ? " - OFF" : "");
}

void VulkanEngine::destroy_regions() {
    if (!rg_) return;
    RegionState& S = *rg_;
    if (S.job_active && S.job.valid()) S.job.wait();
    S.job_active = false;
    region::set_active(nullptr);
    for (uint32_t i = 0; i < kFrameOverlap; ++i)
        if (S.stage[i]) { vmaDestroyBuffer(allocator_, S.stage[i], S.stage_alloc[i]); S.stage[i] = VK_NULL_HANDLE; }
    if (S.ubo) vmaDestroyBuffer(allocator_, S.ubo, S.ubo_alloc);
    if (S.sampler) vkDestroySampler(device_, S.sampler, nullptr);
    if (S.view) vkDestroyImageView(device_, S.view, nullptr);
    if (S.image) vmaDestroyImage(allocator_, S.image, S.alloc);
    S.ubo = VK_NULL_HANDLE; S.sampler = VK_NULL_HANDLE; S.view = VK_NULL_HANDLE; S.image = VK_NULL_HANDLE;
    S.ubo_map = nullptr;
}

// A read or edited tile goes into its layer: one a frame (526 KB), with the frame's other uploads.
void VulkanEngine::record_region_uploads(VkCommandBuffer cmd, uint32_t fslot) {
    if (!rg_ || !rg_->image) return;
    RegionState& S = *rg_;
    int s = -1;
    for (int k = 0; k < region::kSlots && s < 0; ++k)
        if (S.slot[k].tile && (S.slot[k].state == RegionState::kLoaded || (S.slot[k].state == RegionState::kActive && S.slot[k].dirty))) s = k;
    if (s < 0) return;
    fslot %= kFrameOverlap;
    if (!S.stage[fslot]) {
        VkBufferCreateInfo bci{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
        bci.size = kTileBytes;
        bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
        aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
        VmaAllocationInfo ai{};
        if (vmaCreateBuffer(allocator_, &bci, &aci, &S.stage[fslot], &S.stage_alloc[fslot], &ai) != VK_SUCCESS) return;
        S.stage_map[fslot] = ai.pMappedData;
    }
    RegionState::Slot& sl = S.slot[s];
    std::memcpy(S.stage_map[fslot], sl.tile->h.data(), size_t(kTileBytes));
    vmaFlushAllocation(allocator_, S.stage_alloc[fslot], 0, VK_WHOLE_SIZE);
    layer_barrier(cmd, S.image, uint32_t(s), 1, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    VkBufferImageCopy reg{};
    reg.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, uint32_t(s), 1 };
    reg.imageExtent = { uint32_t(kN), uint32_t(kN), 1 };
    vkCmdCopyBufferToImage(cmd, S.stage[fslot], S.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &reg);
    layer_barrier(cmd, S.image, uint32_t(s), 1, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    sl.dirty = false;
    if (sl.state == RegionState::kLoaded) {
        // (part of the loaded set once no frame in flight can read the layer half-written)
        sl.state = RegionState::kUploaded;
        sl.ready_frame = frame_number_ + kFrameOverlap;
        S.ev_upload = frame_number_;
    }
}

void VulkanEngine::regions_update() {
    if (!rg_ || rg_->off || !rg_->image) return;
    RegionState& S = *rg_;
    const auto t0 = std::chrono::steady_clock::now();
    // (wall-clock time from this call to the next: the test hooks run the game on a fixed step)
    S.dt_ring[frame_number_ % 16] = float(ms_since(S.last_call) * 0.001);
    S.last_call = t0;
    if (S.log && S.ev_active != ~0ull && S.ev_active >= 5 && frame_number_ == S.ev_active + 4) {
        // (the frame time stored at frame f is that of frame f - 1)
        std::string line;
        for (uint64_t f = S.ev_active - 5; f <= S.ev_active + 3; ++f) {
            char b[48];
            std::snprintf(b, sizeof(b), " %.2f%s", double(S.dt_ring[(f + 1) % 16]) * 1000.0, f == S.ev_upload ? "(upload)" : f == S.ev_active ? "(joins the set)" : "");
            line += b;
        }
        log::infof("[regions] frame times round it, ms:%s", line.c_str());
        S.ev_active = ~0ull;
    }
    const double now = glfwGetTime();
    const globe::Chart& chart = globe::current();
    const uint32_t serial = globe::chart_serial();
    const glm::vec3 eye = player_.eye_position();
    const glm::dvec3 G = globe::to_planet(chart, double(eye.x), double(eye.z));
    const auto dist_to = [&](const glm::dvec3& c, double r) { return std::max(0.0, globe::ground_distance(G, c) - r); };

    if (!S.started) {
        S.started = true;
        // VIBE_REGION_TEST=<mesa|crater>,<east m>,<south m>[,<radius>,<height>]: a test shape at
        // that place (home coordinates), written by code - unless stored ground is there already.
        if (const char* e = std::getenv("VIBE_REGION_TEST")) {
            char kind[16] = {};
            double east = 0.0, south = 0.0, radius = 120.0, height = 18.0;
            if (std::sscanf(e, "%15[a-z],%lf,%lf,%lf,%lf", kind, &east, &south, &radius, &height) >= 3) {
                const glm::dvec3 Gs = globe::to_planet(globe::Chart{}, east, south);
                std::vector<sector::Id> reach;
                sector::cover(Gs, radius, region::kTileLevel, reach);
                bool there = false;
                for (sector::Id id : reach) there = there || S.store.has_tile(id);
                if (there) log::infof("[regions] test shape: stored ground is there already (%s) - left as it is", S.store.cache_dir.c_str());
                else {
                    const auto ts = std::chrono::steady_clock::now();
                    const auto ids = region::stamp(S.store, Gs, std::string(kind) == "crater" ? region::Shape::Crater : region::Shape::Mesa, radius, height);
                    std::string names;
                    for (sector::Id id : ids) names += " " + sector::name(id);
                    log::infof("[regions] test shape: a %s of %.0f m, %.0f m high at %.0f m east, %.0f m south of home -> %zu tiles (%s) in %.0f ms", kind, radius, height, east, south,
                               ids.size(), names.c_str(), ms_since(ts));
                }
            }
        }
        if (std::getenv("VIBE_REGION_NEW")) regions_new_tile_here();
    }

    bool changed = false;
    const auto free_slot = [&](int s, const char* why) {
        RegionState::Slot& sl = S.slot[s];
        if (sl.unsaved && sl.tile) S.store.save_tile(*sl.tile);
        if (S.log) log::infof("[regions] slot %d freed: tile %s (%s)", s, sector::name(sl.tile->id).c_str(), why);
        if (sl.state == RegionState::kActive) { changed = true; S.changed.push_back(sl.tile->id); }
        sl = RegionState::Slot{};
    };

    // 1. A file the worker has read.
    if (S.job_active && S.job.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
        RegionState::Read rd = S.job.get();
        S.job_active = false;
        if (!rd.tile) {
            // Listed, and not there (or damaged): asked for once, looked for again every 3 s.
            S.missing[S.job_id.v] = now + 3.0;
            if (S.asked.insert(S.job_id.v).second) {
                if (S.store.request) S.store.request(region::Store::tile_file(S.job_id));
                log::warnf("[regions] tile %s is listed and its file is not there - %s", sector::name(S.job_id).c_str(),
                           S.store.request ? "requested" : "nothing to fetch it with: the land stays the formula's");
            }
        } else {
            const glm::dvec3 c = sector::centre(rd.tile->id);
            const double r = sector::radius_m(rd.tile->id), d = dist_to(c, r);
            int s = -1;
            for (int k = 0; k < region::kSlots && s < 0; ++k) if (S.slot[k].state == RegionState::kEmpty) s = k;
            if (s < 0) {
                // No slot: the farthest loaded tile makes room, if it is farther than this one.
                int far_s = -1; double far_d = d;
                for (int k = 0; k < region::kSlots; ++k) {
                    const double dk = dist_to(S.slot[k].c, S.slot[k].r);
                    if (dk > far_d) { far_d = dk; far_s = k; }
                }
                if (far_s >= 0) {
                    log::infof("[regions] budget: all %d slots in use - tile %s (%.0f m) is put out for %s (%.0f m)", region::kSlots,
                               sector::name(S.slot[far_s].tile->id).c_str(), far_d, sector::name(rd.tile->id).c_str(), d);
                    free_slot(far_s, "the budget");
                    s = far_s;
                }
            }
            if (s >= 0) {
                RegionState::Slot& sl = S.slot[s];
                sl.tile = rd.tile; sl.state = RegionState::kLoaded; sl.c = c; sl.r = r;
                if (S.log) log::infof("[regions] tile %s read in %.2f ms (worker) -> slot %d, %.0f m away", sector::name(rd.tile->id).c_str(), rd.ms, s, d);
            }
        }
    }
    // 2. Uploaded two frames ago: part of the loaded set from now.
    for (int k = 0; k < region::kSlots; ++k) {
        RegionState::Slot& sl = S.slot[k];
        if (sl.state != RegionState::kUploaded || frame_number_ < sl.ready_frame) continue;
        sl.state = RegionState::kActive;
        changed = true;
        S.changed.push_back(sl.tile->id);
        S.ev_active = frame_number_;
        if (S.log) log::infof("[regions] slot %d: tile %s is part of the ground (frame %llu)", k, sector::name(sl.tile->id).c_str(), (unsigned long long)frame_number_);
    }
    // 3. What is within reach: the index files of the sectors round the camera, then the tiles.
    if (now >= S.next_scan) {
        S.next_scan = now + 0.25;
        std::vector<sector::Id> secs;
        sector::cover(G, RegionState::kLoadM, region::kIndexLevel, secs);
        for (sector::Id sec : secs) {
            RegionState::Index& I = S.index[sec.v];
            if (now < I.next_look) continue;
            I.next_look = now + 3.0;
            std::vector<sector::Id> ids;
            I.have = S.store.load_index(sec, ids);
            std::vector<RegionState::Known> known;
            for (sector::Id id : ids) {
                const auto old = std::find_if(I.tiles.begin(), I.tiles.end(), [&](const RegionState::Known& k) { return k.id == id; });
                if (old != I.tiles.end()) known.push_back(*old);
                else known.push_back({ id, sector::centre(id), sector::radius_m(id) });
            }
            I.tiles = std::move(known);
            if (!I.have && S.store.request && !I.asked) { I.asked = true; S.store.request(region::Store::index_file(sec)); }
        }
        for (int k = 0; k < region::kSlots; ++k)
            if (S.slot[k].state != RegionState::kEmpty && dist_to(S.slot[k].c, S.slot[k].r) > RegionState::kFreeM) free_slot(k, "out of reach");
        if (!S.job_active) {
            const RegionState::Known* best = nullptr;
            double best_d = 1e30;
            for (sector::Id sec : secs) {
                const auto it = S.index.find(sec.v);
                if (it == S.index.end()) continue;
                for (const RegionState::Known& K : it->second.tiles) {
                    const double d = dist_to(K.c, K.r);
                    if (d > RegionState::kLoadM || d >= best_d) continue;
                    bool held = false;
                    for (int k = 0; k < region::kSlots; ++k) held = held || (S.slot[k].tile && S.slot[k].tile->id == K.id);
                    if (held) continue;
                    const auto m = S.missing.find(K.id.v);
                    if (m != S.missing.end() && now < m->second) continue;
                    best = &K; best_d = d;
                }
            }
            if (best) {
                bool room = false;
                for (int k = 0; k < region::kSlots; ++k)
                    room = room || S.slot[k].state == RegionState::kEmpty || dist_to(S.slot[k].c, S.slot[k].r) > best_d + 50.0;
                if (room) {
                    S.budget_logged = false;
                    S.job_id = best->id;
                    S.job_active = true;
                    const region::Store store = S.store;
                    const sector::Id id = best->id;
                    S.job = std::async(std::launch::async, [store, id]() {
                        const auto ts = std::chrono::steady_clock::now();
                        RegionState::Read out;
                        auto t = std::make_shared<region::Tile>();
                        if (store.load_tile(id, *t)) out.tile = std::move(t);
                        out.ms = ms_since(ts);
                        return out;
                    });
                } else if (!S.budget_logged) {
                    S.budget_logged = true;
                    log::infof("[regions] budget: all %d slots hold nearer tiles - tile %s (%.0f m) waits; the land there is the formula's", region::kSlots,
                               sector::name(best->id).c_str(), best_d);
                }
            }
        }
    }
    // 4. The loaded set: made again when a tile joined or left it, an edge changed, the camera
    //    left the middle of the window of sectors, or the chart changed.
    const glm::dvec2 cam(double(eye.x), double(eye.z));
    if (changed || S.rebuild || (S.cur.count > 0 && !S.cur.windows_fit(serial, cam))) {
        for (int k = 0; k < region::kSlots; ++k)
            S.cur.slot[k].tile = S.slot[k].state == RegionState::kActive ? std::shared_ptr<const region::Tile>(S.slot[k].tile) : nullptr;
        S.cur.update_fades();
        for (int k = 0; k < region::kSlots; ++k) {
            const uint64_t idv = S.cur.slot[k].tile ? S.cur.slot[k].tile->id.v : 0;
            // (a tile that stayed but fades differently now - a neighbour came or went)
            if (idv && idv == S.prev_id[k] && S.cur.slot[k].fade != S.prev_fade[k] &&
                std::find(S.changed.begin(), S.changed.end(), sector::Id{ idv }) == S.changed.end()) S.changed.push_back(sector::Id{ idv });
            S.prev_id[k] = idv; S.prev_fade[k] = S.cur.slot[k].fade;
        }
        S.cur.place_windows(chart, serial, cam);
        region::set_active(std::make_shared<region::ActiveSet>(S.cur));
        region::GpuTable T;
        region::fill_gpu(S.cur, T);
        std::memcpy(S.ubo_map, &T, sizeof(T));
        vmaFlushAllocation(allocator_, S.ubo_alloc, 0, VK_WHOLE_SIZE);
        S.rebuild = false;
    }
    // 5. Tell what stands on the ground.
    if (!S.changed.empty()) {
        for (sector::Id id : S.changed) regions_ground_changed(id.v);
        S.changed.clear();
        if (S.home_dirty) regions_refresh_home();
    }
    S.update_ms = ms_since(t0);
    if (S.log && S.update_ms > 0.5) log::infof("[regions] update took %.2f ms this frame (frame %llu)", S.update_ms, (unsigned long long)frame_number_);
}

// The ground under a tile has changed - it joined the loaded set, left it, fades differently
// or was sculpted. Everything that keeps something worked out from the ground's height is told
// here; a new such thing adds its line.
void VulkanEngine::regions_ground_changed(uint64_t tile) {
    const sector::Id id{ tile };
    const globe::Chart& chart = globe::current();
    const glm::dvec2 c = globe::to_chart(chart, sector::centre(id));
    const float r = float(sector::radius_m(id)) + 2.0f;
    regions_box_changed(float(c.x) - r, float(c.y) - r, float(c.x) + r, float(c.y) + r);
    wv_nf_invalidate();                                     // the ground the grass stands on (veg_stream.cpp)
    wv_ground_changed(glm::vec2(c), r);                     // the streamed woods' cells there
    // Home's own systems are built from its heightmap; a tile over the home tile adds to that.
    if (chart.home && c.x - r < 1024.0 && c.x + r > -1024.0 && c.y - r < 1024.0 && c.y + r > -1024.0) rg_->home_dirty = true;
}

// ... the part that is cheap enough for every frame of a brush stroke: the patch terrain's kept
// node bounds and the baked ground texture, inside a box (chart x, z).
void VulkanEngine::regions_box_changed(float x0, float z0, float x1, float z1) {
    if (tp_) {
        for (auto it = tp_->bounds.begin(); it != tp_->bounds.end();) {
            const uint64_t key = it->first;
            const int level = int(key >> 56);
            // (28 bits each, sign carried: terrain_patches.cpp node_bounds)
            const int ix = int(int32_t(uint32_t((key >> 28) & 0xFFFFFFFu) << 4) >> 4), iz = int(int32_t(uint32_t(key & 0xFFFFFFFu) << 4) >> 4);
            const float size = float(TerrainPatchState::kN) * TerrainPatchState::kSpacing0 * float(1 << level);
            const bool hit = float(ix) * size < x1 && float(ix + 1) * size > x0 && float(iz) * size < z1 && float(iz + 1) * size > z0;
            if (hit) it = tp_->bounds.erase(it); else ++it;
        }
    }
    terrain_bake_redraw(x0, z0, x1, z1);
}

// ---- Home ----
// Home's heightmap with the loaded tiles that lie over it added: what home's full-height
// texture, physics heightfield and ray-tracing mesh are built from. `buf` is used only when a
// tile does lie over home.
const std::vector<float>& VulkanEngine::regions_home_heights(std::vector<float>& buf) const {
    const std::shared_ptr<const region::ActiveSet> set = region::active();
    if (!set || set->count == 0 || terrain_data_.heights.empty()) return terrain_data_.heights;
    const globe::Chart home;
    const int W = terrain_data_.dim + 1;
    bool any = false;
    for (int s = 0; s < region::kSlots; ++s) {
        if (!set->slot[s].tile) continue;
        const region::Tile& t = *set->slot[s].tile;
        glm::dvec2 lo(1e30), hi(-1e30);
        for (int k = 0; k < 4; ++k) {
            const glm::dvec2 p = sector::to_chart(home, t.id, double(k & 1), double(k >> 1));
            lo = glm::min(lo, p); hi = glm::max(hi, p);
        }
        const int x0 = std::max(0, int(std::floor((lo.x - terrain_data_.origin_x) / terrain_data_.cell))), x1 = std::min(W - 1, int(std::ceil((hi.x - terrain_data_.origin_x) / terrain_data_.cell)));
        const int z0 = std::max(0, int(std::floor((lo.y - terrain_data_.origin_z) / terrain_data_.cell))), z1 = std::min(W - 1, int(std::ceil((hi.y - terrain_data_.origin_z) / terrain_data_.cell)));
        if (x0 > x1 || z0 > z1) continue;
        if (!any) { buf = terrain_data_.heights; any = true; }
        // (cell coordinates relative to the tile itself: 0..1 across it)
        const sector::ChartMap m = sector::chart_map(home, 0.5 * (lo + hi), t.id.face(), t.id.level(), int64_t(t.id.i()), int64_t(t.id.j()));
        for (int iz = z0; iz <= z1; ++iz)
            for (int ix = x0; ix <= x1; ++ix) {
                const glm::dvec2 ab = m.eval(glm::dvec2(double(terrain_data_.origin_x) + double(ix) * terrain_data_.cell, double(terrain_data_.origin_z) + double(iz) * terrain_data_.cell));
                if (ab.x < 0.0 || ab.y < 0.0 || ab.x >= 1.0 || ab.y >= 1.0) continue;
                buf[size_t(iz) * W + ix] += region::sample(t, set->slot[s].fade, float(ab.x), float(ab.y));
            }
    }
    return any ? buf : terrain_data_.heights;
}

// A tile over the home tile came, went or changed: home's full-height texture (the grass and
// the road read it), its physics heightfield, its ray-tracing mesh and its fixed trees follow.
// This is the brush's end-of-stroke work on home's own sculpt: tens of milliseconds, with a
// wait for the GPU - paid only for a tile that overlaps home.
void VulkanEngine::regions_refresh_home() {
    RegionState& S = *rg_;
    S.home_dirty = false;
    if (terrain_data_.heights.empty()) return;
    const auto t0 = std::chrono::steady_clock::now();
    wait_gpu_frames_idle();
    refresh_terrain_height_full_texture();
    terrain_blas_dirty_ = true;
    terrain_jolt_dirty_ = true;
    refresh_terrain_blas();
    refresh_terrain_collision();
    // Home's fixed trees: lifted by what the tiles add where each stands.
    const uint32_t n = uint32_t(std::min<size_t>(veg_world_first_ == UINT32_MAX ? veg_inst_cpu_.size() : size_t(veg_world_first_), veg_inst_cpu_.size()));
    S.veg_lift.resize(n, { -1e30f, 0.0f });
    VegInstance* gpu = static_cast<VegInstance*>(veg_inst_mapped_);
    uint32_t moved = 0;
    for (uint32_t i = 0; i < n; ++i) {
        VegInstance& vi = veg_inst_cpu_[i];
        if (vi.pos_scale.w <= 0.0f) continue;
        const float before = S.veg_lift[i].first == vi.pos_scale.y ? S.veg_lift[i].second : 0.0f;
        const float lift = region::ground_delta(vi.pos_scale.x, vi.pos_scale.z);
        if (lift == before) continue;
        vi.pos_scale.y += lift - before;
        S.veg_lift[i] = { vi.pos_scale.y, lift };
        if (gpu) gpu[i].pos_scale = vi.pos_scale;
        ++moved;
    }
    if (moved) {
        if (gpu) vmaFlushAllocation(allocator_, veg_inst_alloc_, 0, sizeof(VegInstance) * n);
        veg_rt_valid_ = false;
    }
    log::infof("[regions] a tile over the home tile: home's height texture, physics ground and ray-tracing mesh rebuilt, %u trees moved, in %.0f ms", moved, ms_since(t0));
}

// ---- Editing ----
bool VulkanEngine::regions_edit_on() const { return rg_ && !rg_->off && rg_->edit; }

// Where the brush meets the ground: a march over the CPU's ground along the view (the physics
// world has no ground away from home to cast a ray at).
void VulkanEngine::regions_brush_aim() {
    const glm::vec3 o = player_.eye_position(), d = player_.forward();
    terrain_brush_has_hit_ = false;
    float t = 0.5f, prev = 0.0f;
    while (t < 800.0f) {
        const glm::vec3 p = o + d * t;
        if (p.y < sample_terrain_height(p.x, p.z)) {
            float a = prev, b = t;
            for (int k = 0; k < 10; ++k) {
                const float m = 0.5f * (a + b);
                const glm::vec3 q = o + d * m;
                if (q.y < sample_terrain_height(q.x, q.z)) b = m; else a = m;
            }
            terrain_brush_world_pos_ = o + d * b;
            terrain_brush_has_hit_ = true;
            return;
        }
        prev = t;
        t += std::max(0.5f, 0.03f * t);
    }
}

// One frame of the brush on the region tiles under it. False: region editing is off (the brush
// is home's). Raise, lower, smooth and flatten; the other modes do nothing here.
bool VulkanEngine::regions_brush(float dt) {
    if (!regions_edit_on()) return false;
    RegionState& S = *rg_;
    if (!terrain_brush_has_hit_) return true;
    const TerrainBrushMode mode = terrain_brush_mode_;
    if (mode != TerrainBrushMode::Raise && mode != TerrainBrushMode::Lower && mode != TerrainBrushMode::Smooth && mode != TerrainBrushMode::Flatten) return true;
    const globe::Chart& chart = globe::current();
    const glm::dvec2 P(double(terrain_brush_world_pos_.x), double(terrain_brush_world_pos_.z));
    const float r = terrain_brush_radius_, strength = terrain_brush_strength_ * dt;
    const int steps = kN - 1;
    std::vector<int> touched;
    for (int s = 0; s < region::kSlots; ++s) {
        RegionState::Slot& sl = S.slot[s];
        if (sl.state != RegionState::kActive) continue;
        region::Tile& t = *sl.tile;
        if (glm::distance(P, globe::to_chart(chart, sl.c)) > double(r) + sl.r) continue;
        // The tile's corners in the chart; a sample's place is taken between them (to a few
        // centimetres - a brush's falloff does not need more, and the samples two tiles share
        // on an edge come out at the same place from either).
        glm::dvec2 q[4];
        for (int k = 0; k < 4; ++k) q[k] = sector::to_chart(chart, t.id, double(k & 1), double(k >> 1));
        const glm::dvec2 ab = sector::chart_cell_exact(chart, P, t.id.face(), t.id.level(), int64_t(t.id.i()), int64_t(t.id.j()));
        const double step_m = std::min(glm::distance(q[0], q[1]), glm::distance(q[0], q[2])) / double(steps);
        const int rr = int(std::ceil(double(r) / step_m)) + 2;
        const int a0 = std::max(0, int(std::floor(ab.x * steps)) - rr), a1 = std::min(steps, int(std::ceil(ab.x * steps)) + rr);
        const int b0 = std::max(0, int(std::floor(ab.y * steps)) - rr), b1 = std::min(steps, int(std::ceil(ab.y * steps)) + rr);
        if (a0 > a1 || b0 > b1) continue;
        if (sl.edit.size() != t.h.size()) {
            sl.edit.resize(t.h.size());
            for (size_t k = 0; k < t.h.size(); ++k) sl.edit[k] = float(t.h[k]) * t.unit_m;
        }
        bool any = false;
        const std::vector<float> before = mode == TerrainBrushMode::Smooth ? sl.edit : std::vector<float>();
        for (int ib = b0; ib <= b1; ++ib)
            for (int ia = a0; ia <= a1; ++ia) {
                const double fa = double(ia) / steps, fb = double(ib) / steps;
                const glm::dvec2 p = glm::mix(glm::mix(q[0], q[1], fa), glm::mix(q[2], q[3], fa), fb);
                const float d = float(glm::distance(p, P));
                if (d > r) continue;
                const float tt = 1.0f - d / r, fall = tt * tt * (3.0f - 2.0f * tt);
                float& e = sl.edit[size_t(ib) * kN + ia];
                switch (mode) {
                    case TerrainBrushMode::Raise: e += strength * fall; break;
                    case TerrainBrushMode::Lower: e -= strength * fall; break;
                    case TerrainBrushMode::Smooth: {
                        const auto at = [&](int x, int y) { return before[size_t(std::clamp(y, 0, steps)) * kN + size_t(std::clamp(x, 0, steps))]; };
                        const float mean = 0.25f * (at(ia - 1, ib) + at(ia + 1, ib) + at(ia, ib - 1) + at(ia, ib + 1));
                        e += (mean - e) * std::min(1.0f, strength * 4.0f * fall);
                        break;
                    }
                    case TerrainBrushMode::Flatten:
                        // (toward the wanted height of the ground itself, whatever the formula does under it)
                        e += (terrain_brush_flatten_target_ - sample_terrain_height(float(p.x), float(p.y))) * std::min(1.0f, strength * 0.5f * fall);
                        break;
                    default: break;
                }
                e = std::clamp(e, -32767.0f * t.unit_m, 32767.0f * t.unit_m);
                any = true;
            }
        if (any) touched.push_back(s);
    }
    if (touched.empty()) return true;
    // Two tiles that continue into each other hold the same samples on their shared edge.
    for (int s : touched)
        for (int e = 1; e <= 3; e += 2) {                   // its east and south neighbours (same face)
            RegionState::Slot& A = S.slot[s];
            const sector::Id nb = sector::neighbour(A.tile->id, e == 1 ? 1 : 0, e == 3 ? 1 : 0);
            if (!nb.valid() || nb.face() != A.tile->id.face() || !(A.tile->joined & (1u << e))) continue;
            for (int k = 0; k < region::kSlots; ++k) {
                RegionState::Slot& B = S.slot[k];
                if (B.state != RegionState::kActive || B.tile->id != nb || B.edit.size() != A.edit.size()) continue;
                for (int i = 0; i <= steps; ++i) {
                    float& va = e == 1 ? A.edit[size_t(i) * kN + steps] : A.edit[size_t(steps) * kN + i];
                    float& vb = e == 1 ? B.edit[size_t(i) * kN] : B.edit[size_t(i)];
                    va = vb = 0.5f * (va + vb);
                }
                if (std::find(touched.begin(), touched.end(), k) == touched.end()) { B.dirty = B.unsaved = true; for (size_t i = 0; i < B.edit.size(); ++i) B.tile->h[i] = region::to_sample(B.edit[i], B.tile->unit_m); }
            }
        }
    for (int s : touched) {
        RegionState::Slot& sl = S.slot[s];
        // (the tile object is the one the loaded set holds: the CPU's ground follows at once)
        for (size_t i = 0; i < sl.edit.size(); ++i) sl.tile->h[i] = region::to_sample(sl.edit[i], sl.tile->unit_m);
        sl.dirty = sl.unsaved = true;
    }
    S.stroke = true;
    regions_box_changed(float(P.x) - r - 2.0f, float(P.y) - r - 2.0f, float(P.x) + r + 2.0f, float(P.y) + r + 2.0f);
    return true;
}

// The stroke is over: the tiles are saved, and what stands on the ground is told.
void VulkanEngine::regions_stroke_end() {
    if (!rg_ || !rg_->stroke) return;
    RegionState& S = *rg_;
    S.stroke = false;
    int saved = 0;
    for (int s = 0; s < region::kSlots; ++s) {
        RegionState::Slot& sl = S.slot[s];
        if (!sl.unsaved || !sl.tile) continue;
        if (S.store.save_tile(*sl.tile)) ++saved;
        sl.unsaved = false;
        regions_ground_changed(sl.tile->id.v);
    }
    if (S.home_dirty) regions_refresh_home();
    log::infof("[regions] stroke ended: %d tiles saved to %s", saved, S.store.cache_dir.c_str());
}

// An empty tile for the sector under the camera, joined to the tiles round it whose edges are
// level with it; the streamer picks it up like any other.
bool VulkanEngine::regions_new_tile_here() {
    if (!rg_ || rg_->off) return false;
    RegionState& S = *rg_;
    const glm::vec3 eye = player_.eye_position();
    const sector::Id id = sector::from_chart(globe::current(), double(eye.x), double(eye.z), region::kTileLevel);
    if (S.store.has_tile(id)) { log::infof("[regions] new tile: sector %s has one already", sector::name(id).c_str()); return false; }
    region::Tile t(id);
    sector::Id nb[8];
    const int n = sector::neighbours(id, nb);
    std::vector<region::Tile> round;
    for (int k = 0; k < n; ++k) {
        region::Tile o;
        if (region::edge_toward(id, nb[k]) < 0 || !S.store.load_tile(nb[k], o)) continue;
        if (region::join_tiles(t, o)) round.push_back(std::move(o));
    }
    region::join_corners(t, [&](sector::Id q) { return S.store.has_tile(q); });
    for (region::Tile& o : round) {
        region::join_corners(o, [&](sector::Id q) { return q == id || S.store.has_tile(q); });
        S.store.save_tile(o);
        // (a neighbour that is loaded: it learns that it continues now)
        for (int k = 0; k < region::kSlots; ++k)
            if (S.slot[k].tile && S.slot[k].tile->id == o.id) { S.slot[k].tile->joined = o.joined; S.rebuild = true; }
    }
    if (!S.store.save_tile(t)) { log::warnf("[regions] new tile: could not write to %s", S.store.cache_dir.c_str()); return false; }
    for (auto& kv : S.index) kv.second.next_look = 0.0;
    S.next_scan = 0.0;
    log::infof("[regions] new tile %s (%s/%s), joined to %zu neighbours", sector::name(id).c_str(), S.store.cache_dir.c_str(), region::Store::tile_file(id).c_str(), round.size());
    return true;
}

// A test shape (0 mesa, 1 crater) where the brush points, or under the camera.
void VulkanEngine::regions_stamp_here(int shape, float radius, float height) {
    if (!rg_ || rg_->off) return;
    RegionState& S = *rg_;
    const glm::vec3 at = terrain_brush_has_hit_ && terrain_edit_mode_ ? terrain_brush_world_pos_ : player_.eye_position();
    const glm::dvec3 G = globe::to_planet(globe::current(), double(at.x), double(at.z));
    for (int k = 0; k < region::kSlots; ++k)
        if (S.slot[k].unsaved && S.slot[k].tile) { S.store.save_tile(*S.slot[k].tile); S.slot[k].unsaved = false; }
    const auto t0 = std::chrono::steady_clock::now();
    const auto ids = region::stamp(S.store, G, shape == 1 ? region::Shape::Crater : region::Shape::Mesa, double(radius), double(height));
    // The tiles it wrote are read again.
    for (sector::Id id : ids)
        for (int k = 0; k < region::kSlots; ++k)
            if (S.slot[k].tile && S.slot[k].tile->id == id) {
                if (S.slot[k].state == RegionState::kActive) { S.changed.push_back(id); S.rebuild = true; }
                S.slot[k] = RegionState::Slot{};
            }
    for (auto& kv : S.index) kv.second.next_look = 0.0;
    S.next_scan = 0.0;
    log::infof("[regions] stamped a %s (%.0f m, %.0f m high) into %zu tiles in %.0f ms", shape == 1 ? "crater" : "mesa", double(radius), double(height), ids.size(), ms_since(t0));
}

void VulkanEngine::regions_menu() {
    if (!rg_ || rg_->off) return;
    RegionState& S = *rg_;
    ImGui::SeparatorText("Region tiles: stored ground anywhere on the planet");
    const glm::vec3 eye = player_.eye_position();
    const sector::Id id = sector::from_chart(globe::current(), double(eye.x), double(eye.z), region::kTileLevel);
    int used = 0; bool here = false;
    for (int k = 0; k < region::kSlots; ++k) {
        used += S.slot[k].state != RegionState::kEmpty ? 1 : 0;
        here = here || (S.slot[k].state == RegionState::kActive && S.slot[k].tile->id == id);
    }
    ImGui::Text("sector under the camera: %s - %s; %d of %d slots in use", sector::name(id).c_str(), here ? "its tile is loaded" : "no tile loaded", used, region::kSlots);
    ImGui::Checkbox("Edit region tiles (the brush: raise, lower, smooth, flatten)", &S.edit);
    if (ImGui::Button("New tile here")) regions_new_tile_here();
    ImGui::SameLine();
    ImGui::TextDisabled("-> %s", S.store.cache_dir.c_str());
    ImGui::SliderFloat("shape radius (m)", &S.stamp_radius, 20.0f, 400.0f);
    ImGui::SliderFloat("shape height (m)", &S.stamp_height, 2.0f, 60.0f);
    if (ImGui::Button("Stamp mesa here")) regions_stamp_here(0, S.stamp_radius, S.stamp_height);
    ImGui::SameLine();
    if (ImGui::Button("Stamp crater here")) regions_stamp_here(1, S.stamp_radius, S.stamp_height);
}

} // namespace vibe
