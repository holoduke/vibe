// The river's plants beyond the home tile, streamed (see river_stream.h; the rule itself:
// engine/river_plants.h).
//
// A stretch is kPer stations of the river (70 m). Its plants are worked out on a background
// thread (one stretch at a time; some thousands of ground heights), sorted by species, and
// written into one of kSlots equal parts of a single instance buffer. A slot that was given back
// is not written again for kRest frames (a frame in flight may still be drawing from it).
// No shadows of their own (the small plants' shadow map is drawn from the start-up set).
//
// VIBE_NO_RIVER_PLANTS=1: none (also not the home tile's: scatter.cpp). VIBE_LOG_RIVER=1: logs
// every stretch made and freed.
#include "engine/vk_engine/river_stream.h"
#include "engine/river_plants.h"
#include "engine/terrain.h"
#include "engine/globe.h"
#include "engine/log.h"

#include <glm/glm.hpp>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <future>
#include <vector>

namespace vibe { float water_level_off(float x, float z); }     // (vk_engine/water_fx.cpp: the river falls to the sea)

namespace vibe::river_stream {

namespace {

constexpr int      kPer = 10;                       // stations a stretch
constexpr int      kSlots = 12;
constexpr uint32_t kCap = 2048;                     // instances a slot
constexpr float    kLoad = 260.0f, kFree = 340.0f;  // from the camera to a stretch's middle
constexpr int      kRest = 10;                      // frames a freed slot rests
// What the rule plants out here (no long grass: beyond the home tile that is the ground's own).
constexpr int      kSpecies[] = { 0, 5, 7, 11, 12, 13, 16 };
constexpr int      kNSpecies = int(sizeof(kSpecies) / sizeof(kSpecies[0]));

struct Made {
    int stretch = -1;
    uint32_t serial = 0;
    std::vector<ScatterInstance> inst;              // sorted by species
    uint32_t first[kNSpecies] = {}, count[kNSpecies] = {};
    glm::vec3 mn{ 1e9f }, mx{ -1e9f };
    river_rule::Counts n;
    double ms = 0.0;
};
struct Slot {
    int stretch = -1;
    int rest = 0;
    ScatterCell cell[kNSpecies];
};

VkBuffer      g_buf = VK_NULL_HANDLE;
VmaAllocation g_alloc = nullptr;
ScatterInstance* g_mapped = nullptr;
Slot          g_slot[kSlots];
std::future<Made> g_job;
bool          g_busy = false;
int           g_stretches = -1;                     // (-1: not counted yet)
std::vector<uint8_t>   g_wanted;                    // per stretch: has stations beyond the home tile
std::vector<glm::vec2> g_mid;                       // per stretch: its middle
std::vector<uint8_t>   g_empty;                     // per stretch: made once and nothing grew there

const bool kOff = std::getenv("VIBE_NO_RIVER_PLANTS") != nullptr;
const bool kLog = std::getenv("VIBE_LOG_RIVER") != nullptr;

Made make(int stretch, float wl, int octaves, uint32_t serial) {
    const auto t0 = std::chrono::steady_clock::now();
    Made m; m.stretch = stretch; m.serial = serial;
    std::vector<ScatterInstance> per[ScatterSet::kPlantSpecies];
    auto h_at = [&](float x, float z) { return world_ground_height(x, z, wl, octaves); };
    auto blocked = [&](float x, float z, bool water) {
        // (a highway or the lane and their verges; for what stands in the water also under their
        //  bridges - the distance is in the road's own measure, 8 = its verge's edge: highway.h)
        if (world_highway_distance(x, z) < (water ? 14.0f : 9.5f)) return true;
        // (the service station's plot and its wall down to the river: terrain.h)
        return x > kStationRect[0] - 3.0f && x < kStationRect[2] + 3.0f && z > kStationRect[1] - 3.0f && z < kStationRect[3] + 3.0f;
    };
    const int total = river_rule::station_count();
    for (int k = stretch * kPer; k < std::min((stretch + 1) * kPer, total); ++k) {
        if (river_rule::in_home(river_rule::station(k).mid)) continue;          // (the home tile's: scatter.cpp)
        river_rule::plants_at(k, wl + water_level_off(river_rule::station(k).mid.x, river_rule::station(k).mid.y), h_at, blocked, false, per, m.n);      // (the level there: the river falls)
    }
    for (int s = 0; s < kNSpecies; ++s) {
        std::vector<ScatterInstance>& v = per[kSpecies[s]];
        m.first[s] = uint32_t(m.inst.size());
        if (m.inst.size() + v.size() > kCap) v.resize(kCap - m.inst.size());
        m.count[s] = uint32_t(v.size());
        m.inst.insert(m.inst.end(), v.begin(), v.end());
    }
    for (const ScatterInstance& si : m.inst) { m.mn = glm::min(m.mn, glm::vec3(si.pos_scale)); m.mx = glm::max(m.mx, glm::vec3(si.pos_scale)); }
    m.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return m;
}

void count_stretches() {
    const int total = river_rule::station_count();
    g_stretches = (total + kPer - 1) / kPer;
    g_wanted.assign(size_t(g_stretches), 0);
    g_empty.assign(size_t(g_stretches), 0);
    g_mid.assign(size_t(g_stretches), glm::vec2(0.0f));
    for (int s = 0; s < g_stretches; ++s) {
        for (int k = s * kPer; k < std::min((s + 1) * kPer, total); ++k)
            if (!river_rule::in_home(river_rule::station(k).mid)) g_wanted[size_t(s)] = 1;
        g_mid[size_t(s)] = river_rule::station(std::min(s * kPer + kPer / 2, total - 1)).mid;
    }
}

}  // namespace

void poll(VmaAllocator alloc, glm::vec3 eye, bool here, float water_level, int octaves) {
    if (kOff) return;
    if (g_stretches < 0) count_stretches();
    for (Slot& s : g_slot) if (s.rest > 0) --s.rest;
    // Free: beyond kFree, or the engine has left home's chart.
    for (Slot& s : g_slot) {
        if (s.stretch < 0) continue;
        const glm::vec2 m = g_mid[size_t(s.stretch)];
        if (here && glm::distance(m, glm::vec2(eye.x, eye.z)) < kFree && eye.y < 900.0f) continue;
        if (kLog) log::infof("[river] stretch %d freed", s.stretch);
        s.stretch = -1; s.rest = kRest;
    }
    // A finished stretch goes into a slot.
    if (g_busy && g_job.valid() && g_job.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
        Made m = g_job.get();
        g_busy = false;
        const bool ok = here && m.serial == globe::chart_serial();
        if (ok && m.inst.empty()) g_empty[size_t(m.stretch)] = 1;
        if (ok && !m.inst.empty()) {
            if (!g_buf) {
                VkBufferCreateInfo bci{};
                bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
                bci.size = VkDeviceSize(kSlots) * kCap * sizeof(ScatterInstance);
                bci.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
                VmaAllocationCreateInfo aci{};
                aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
                aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
                VmaAllocationInfo ai{};
                if (vmaCreateBuffer(alloc, &bci, &aci, &g_buf, &g_alloc, &ai) == VK_SUCCESS) g_mapped = static_cast<ScatterInstance*>(ai.pMappedData);
                else { g_buf = VK_NULL_HANDLE; log::warn("[river] instance buffer alloc failed"); }
            }
            int free_slot = -1;
            for (int i = 0; i < kSlots; ++i) if (g_slot[i].stretch < 0 && g_slot[i].rest == 0) { free_slot = i; break; }
            if (g_mapped && free_slot >= 0) {
                Slot& s = g_slot[free_slot];
                const uint32_t base = uint32_t(free_slot) * kCap;
                std::memcpy(g_mapped + base, m.inst.data(), m.inst.size() * sizeof(ScatterInstance));
                vmaFlushAllocation(alloc, g_alloc, VkDeviceSize(base) * sizeof(ScatterInstance), m.inst.size() * sizeof(ScatterInstance));
                for (int k = 0; k < kNSpecies; ++k) {
                    s.cell[k].first = base + m.first[k];
                    s.cell[k].count = m.count[k];
                    // (one box for the stretch; plants stand up to a metre and a half over their roots)
                    s.cell[k].mn = m.mn - glm::vec3(1.5f, 0.5f, 1.5f);
                    s.cell[k].mx = m.mx + glm::vec3(1.5f, 2.2f, 1.5f);
                }
                s.stretch = m.stretch;
                if (kLog) log::infof("[river] stretch %d made in %.1f ms: %d lily rafts, %d reeds, %d bank plants (slot %d, middle %.0f, %.0f)", m.stretch, m.ms,
                                     m.n.lilies, m.n.reeds, m.n.bank, free_slot, double(g_mid[size_t(m.stretch)].x), double(g_mid[size_t(m.stretch)].y));
                // (where to look: the middle one of its lily rafts)
                if (kLog && m.count[kNSpecies - 1] > 0) { const glm::vec4 p = m.inst[m.first[kNSpecies - 1] + m.count[kNSpecies - 1] / 2].pos_scale; log::infof("[river]   a lily raft at (%.1f, %.1f, %.1f)", double(p.x), double(p.y), double(p.z)); }
            }
        }
    }
    if (!here || g_busy || eye.y > 900.0f) return;
    // Make: the nearest wanted stretch within kLoad that is in no slot.
    int want = -1; float best = kLoad;
    for (int s = 0; s < g_stretches; ++s) {
        if (!g_wanted[size_t(s)] || g_empty[size_t(s)]) continue;
        const float d = glm::distance(g_mid[size_t(s)], glm::vec2(eye.x, eye.z));
        if (d >= best) continue;
        bool have = false;
        for (const Slot& sl : g_slot) if (sl.stretch == s) { have = true; break; }
        if (!have) { want = s; best = d; }
    }
    if (want < 0) return;
    bool room = false;
    for (const Slot& sl : g_slot) if (sl.stretch < 0) { room = true; break; }
    if (!room) return;
    const uint32_t serial = globe::chart_serial();
    g_busy = true;
    g_job = std::async(std::launch::async, [want, water_level, octaves, serial]() { return make(want, water_level, octaves, serial); });
}

void draw(VkCommandBuffer cmd, const ScatterSet& set, glm::vec3 eye,
          const std::function<float(int)>& begin_species,
          const std::function<bool(const ScatterCell&, float)>& cell_in) {
    if (!g_buf) return;
    bool any = false;
    for (const Slot& s : g_slot) any |= s.stretch >= 0;
    if (!any) return;
    for (int k = 0; k < kNSpecies; ++k) {
        const int sp = kSpecies[k];
        const Mesh* lods[3] = { &set.plant_mesh[sp], &set.plant_mesh_far[sp], &set.plant_mesh_far2[sp] };
        if (!lods[0]->vertex_buffer) continue;
        if (!lods[1]->vertex_buffer) lods[1] = lods[0];
        if (!lods[2]->vertex_buffer) lods[2] = lods[1];
        float max_d = -1.0f;
        int bound = -1;
        for (const Slot& s : g_slot) {
            const ScatterCell& c = s.cell[k];
            if (s.stretch < 0 || c.count == 0) continue;
            if (max_d < 0.0f) max_d = begin_species(sp);
            if (!cell_in(c, max_d)) continue;
            // (the level of detail by the stretch's nearest point, as the pass does for its own cells)
            const glm::vec3 q = glm::clamp(eye, c.mn, c.mx);
            const float d2 = glm::dot(q - eye, q - eye);
            const int l = d2 < ScatterSet::kPlantFarLod * ScatterSet::kPlantFarLod ? 0 : (d2 < ScatterSet::kPlantFarLod2 * ScatterSet::kPlantFarLod2 ? 1 : 2);
            if (l != bound) {
                VkBuffer vb[2] = { lods[l]->vertex_buffer, g_buf };
                VkDeviceSize off[2] = { 0, 0 };
                vkCmdBindVertexBuffers(cmd, 0, 2, vb, off);
                vkCmdBindIndexBuffer(cmd, lods[l]->index_buffer, 0, VK_INDEX_TYPE_UINT32);
                bound = l;
            }
            vkCmdDrawIndexed(cmd, lods[l]->index_count, c.count, 0, 0, c.first);
        }
    }
}

void destroy(VmaAllocator alloc) {
    if (g_busy && g_job.valid()) g_job.wait();
    g_busy = false;
    if (g_buf) vmaDestroyBuffer(alloc, g_buf, g_alloc);
    g_buf = VK_NULL_HANDLE; g_alloc = nullptr; g_mapped = nullptr;
    for (Slot& s : g_slot) s = Slot{};
}

}  // namespace vibe::river_stream
