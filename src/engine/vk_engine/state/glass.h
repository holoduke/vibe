#pragma once
// Breakable glass and the weapons' lenses (glass.cpp): its state.
//
// Private to the files that implement it - vk_engine.h only knows the name of this struct
// and holds a pointer to it (glass_). Created and freed in vk_engine/private_state.cpp.

#include "engine/vk_engine.h"

#include <memory>
#include <vector>

namespace vibe {

struct VulkanEngine::GlassState {
    // A damaged pane's own pipeline (glass_pane.vert / .frag); a whole pane, a shard and a lens
    // are drawn with the FX pass's glass pipeline (state/fx.h).
    VkPipeline     pane_pipeline = VK_NULL_HANDLE;
    VkShaderModule pane_vert = VK_NULL_HANDLE, pane_frag = VK_NULL_HANDLE;

    static constexpr int kMaxHits = 7;          // impacts a pane shows (what fits a draw's push constants)
    static constexpr size_t kMaxShards = 1400;  // pieces in the air or on the floor, all panes together
    static constexpr float kTeethSeconds = 62.0f;   // how long the teeth of a burst pane stand

    struct Hit {
        glm::vec2 uv{0.0f};     // on the pane: metres from its lower left corner
        float fall = 0.0f;      // 0 .. 3.2: how much of the web round it has dropped out
        float seed = 0.0f;      // 0 .. 1: the web's shape (glass_pane.frag)
    };
    struct Pane {
        glm::vec2 lo{0.0f}, hi{0.0f};   // in its sheet (metres from the sheet's origin)
        int     sheet = 0;
        uint8_t state = 0;              // 0 whole, 1 cracked, 2 burst (teeth stand), 3 gone
        bool    toughened = false;      // toughened glass: breaks all at once, into small dice
        float   health = 1.0f;          // hits it still takes
        float   seed = 0.0f;
        float   burst_t = 0.0f;         // seconds since it burst
        int     n_hits = 0;
        Hit     hits[kMaxHits];
        glm::vec2 last_uv{-99.0f};      // (the same bolt is tested over several steps: one impact)
        double  last_time = -1.0;
    };
    struct Sheet {
        glm::vec3 origin{0.0f}, u{1.0f, 0.0f, 0.0f}, v{0.0f, 1.0f, 0.0f}, n{0.0f, 0.0f, 1.0f};
        glm::vec2 size{0.0f};
        int   first_pane = 0, n_panes = 0;
        // What a walker bumps into: the sheet's box in Level::aabbs (-1: it has none). While
        // every pane stands that box is the collider. With the first pane out it is switched
        // off (col_off) and each pane that still stands gets a box of its own every tick
        // (glass_player_colliders).
        int   col = -1;
        collision::AABB col_box{};
        bool  col_off = false;
    };
    // One glass part of the level that has been looked at: its sheets and panes, and the mesh
    // that replaces the part's own in the glass pass (a rectangle a pane, so that a pane can be
    // drawn - or left out - by itself; then the triangles of the sheets that cannot break).
    struct Part {
        int   level_mesh = -1;                  // index into Level::meshes
        const CastleMeshGpu* gpu = nullptr;     // the part it replaces
        Mesh  mesh{};
        std::vector<Sheet> sheets;
        std::vector<Pane>  panes;
        uint32_t loose_first = 0, loose_count = 0;      // indices of what cannot break
        glm::vec3 bmin{0.0f}, bmax{0.0f};
        glm::vec3 tint{0.05f};
        int   damaged = 0;                      // panes not whole
    };
    std::vector<std::unique_ptr<Part>> parts;
    std::vector<int> part_of_mesh;              // per Level::meshes entry: -1 not looked at, else index into `parts`
    std::vector<int> glass_meshes;              // the level's glass parts (indices into Level::meshes)
    size_t meshes_seen = 0;

    // A piece of a broken pane: a flat triangle, thrown, tumbling, falling, then lying still.
    struct Shard {
        glm::vec3 pos{0.0f}, vel{0.0f};
        glm::vec3 ex{1.0f, 0.0f, 0.0f}, ey{0.0f, 1.0f, 0.0f};     // its plane as it left the pane
        glm::vec3 axis{1.0f, 0.0f, 0.0f};
        float ang = 0.0f, spin = 0.0f;
        glm::vec2 c[3];                         // its corners in its own plane
        float wait = 0.0f;                      // seconds it still hangs in the frame
        float age = 0.0f, life = 6.0f;
        float floor_y[2]{ -1e9f, -1e9f };       // the floor behind and in front of the pane (when it burst; the first guess)
        float fy = -1e9f;                       // the floor under where it is now (glass_tick: measured again every 25 cm it travels)
        glm::vec2 fy_at{ 1e9f, 1e9f };          // where that was measured (x, z)
        glm::vec3 pane_o{0.0f}, pane_n{0.0f, 0.0f, 1.0f};
        float lie = 0.0f;                       // which way it lies once it is down
        bool  rest = false, bounced = false;
    };
    std::vector<Shard> shards;
    // The eye under a roof (a shelter: rocket.cpp shelters_near) this frame, 0 or 1: the panes
    // then mirror the room, not the sky (glass_common.glsl glass_reflection). Set by fx.cpp.
    float indoor = 0.0f;

    // The lenses of the sights of the weapons drawn this frame (draw_viewmodel leaves them here;
    // the glass pass draws them).
    struct Lens {
        const Mesh* mesh = nullptr;
        glm::mat4 mvp{1.0f}, model{1.0f}, to_clip{1.0f};       // model: relative to the eye
        glm::vec3 centre{0.0f}, axis{0.0f, 0.0f, 1.0f};
        float radius = 0.01f;
        glm::vec3 glow{0.0f};
        float aim = 0.0f, mag = 1.0f, reticle = 0.0f;
    };
    std::vector<Lens> lenses;
    uint64_t lens_frame = ~0ull;

    double   now = 0.0;
    float    scan_in = 0.0f;
    uint32_t rng = 0x2F6E2B1u;
    bool     sounds_tried = false, sounds = false;
    bool     log = false;
    int      test_shots_left = -1;
};

} // namespace vibe
