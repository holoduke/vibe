#pragma once
// Regions: stored ground on the procedural planet (docs/ARCHITECTURE.md, "Regions").
//
// The land is a formula everywhere. A HEIGHT TILE is a piece of stored ground laid over it: for
// one sector of the planet (engine/sector.h, level kTileLevel: about 610 m) a grid of heights
// that are ADDED to the formula's. So a tile of zeros is the formula's own land, the formula's
// detail carries on under whatever was sculpted, and land with no tile is just land.
//
//   Grid       kTileN x kTileN samples (513: 512 steps a side, about 1.2 m each), 16-bit, in
//              units of `unit_m` (1/64 m: 1.6 cm steps, +-512 m). Sample (ia, ib) lies at
//              (ia / 512, ib / 512) of the sector - the outer rows lie ON its edges, and two
//              tiles that continue into each other hold the same values there.
//   Reading    bilinear between the four samples round the point, times the edge mask.
//   Edge mask  a tile fades to nothing over kFade of its side (38 m) toward every edge and
//              corner where the ground does NOT continue into a loaded neighbour - so there is
//              never a step at a tile's border, whatever was painted there. Toward a neighbour
//              that is loaded and that both tiles declare joined (`joined`), it does not fade.
//   File       h_<sector>.vht, little endian:
//                 0  char[4]  "VHT1"
//                 4  u16      format version (1)
//                 6  u16      samples a side (513)
//                 8  u64      sector id (sector::Id::v)
//                16  f32      unit_m
//                20  u8       joined (bits: kW kE kN kS, then the corners kNW kNE kSW kSE)
//                21  u8       compression (0: none - the only one this build writes or reads)
//                22  u16      0
//                24  u32      payload bytes
//                28  u32      FNV-1a of the payload
//                32  i16[n*n] the samples, row by row (b outer, a inner)
//              526 KB a tile. (No compressor is linked into vibe_core; the field is there for
//              one. A newer version or an unknown compression is refused, not guessed at.)
//   Index      idx_<sector of kIndexLevel>.txt, text: one line `tile <sector>` per height tile
//              inside that 39 km sector ('#' starts a comment; other first words are skipped,
//              so later kinds of content can be listed in the same file). It is what tells the
//              engine that a place has stored ground at all - nothing is ever probed per cell.
//   Folders    `local` (assets/regions, shipped) then `cache` (cache/regions, fetched or made
//              in the editor). A file that is wanted and in neither is asked for through
//              `Store::request` - the door for a server client, as in engine/site_store.h.
//
// The loaded tiles (the "active set") are published as one unchanging object (ActiveSet):
// ground_delta() - the CPU's twin of shaders/region_tiles.glsl - reads it from any thread, and
// world_ground_height (terrain.cpp) adds it. The same numbers go to the shader (GpuTable).

#include "engine/globe.h"
#include "engine/sector.h"

#include <glm/glm.hpp>

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace vibe::region {

constexpr int   kTileLevel  = 14;       // a height tile is a sector of this level (610 m at most)
constexpr int   kIndexLevel = 8;        // an index file lists what lies in a sector of this level (39 km)
constexpr int   kTileN      = 513;      // samples a side
constexpr int   kSlots      = 32;       // tiles loaded at once (shaders/region_tiles.glsl kRegionSlots)
constexpr int   kWin        = 16;       // the window of sectors round the camera, cells a side
constexpr float kFade       = 0.0625f;  // the edge mask's width, of the tile's side
constexpr float kUnitM      = 1.0f / 64.0f;

// Edges and corners of a tile (a grows W -> E, b grows N -> S; on face 2 near home that is the
// compass). Used for `joined` and for the fade bits.
enum : uint32_t { kW = 1, kE = 2, kN = 4, kS = 8, kNW = 16, kNE = 32, kSW = 64, kSE = 128 };

struct Tile {
    sector::Id id;
    float      unit_m = kUnitM;
    uint8_t    joined = 0;              // the edges / corners where it continues into a neighbour
    std::vector<int16_t> h;             // kTileN * kTileN, row b, column a; never -32768
    Tile() = default;
    explicit Tile(sector::Id id_) : id(id_), h(size_t(kTileN) * kTileN, 0) {}
    int16_t& at(int ia, int ib) { return h[size_t(ib) * kTileN + size_t(ia)]; }
    int16_t  at(int ia, int ib) const { return h[size_t(ib) * kTileN + size_t(ia)]; }
};

// ---- Reading a tile (the rule the shader follows too: region_tiles.glsl) ----
// The edge mask at (a, b) of a tile, 0..1; `fade`: the edges and corners that fade (k* bits).
float edge_mask(float a, float b, uint32_t fade);
// Metres to add to the formula at (a, b) of the tile.
float sample(const Tile& t, uint32_t fade, float a, float b);
// Quantise metres to a stored sample.
int16_t to_sample(float metres, float unit_m);

// ---- Files ----
std::vector<uint8_t> encode(const Tile& t);
bool decode(const uint8_t* data, size_t size, Tile& out);

class Store {
public:
    std::string local_dir = "assets/regions";
    std::string cache_dir = "cache/regions";
    // A file that is wanted and in neither folder (an index, or a tile an index lists): for a
    // server client to fetch into `cache_dir`. Called on the main thread, at most every few
    // seconds per file. Unset: the place stays the formula's land.
    std::function<void(const std::string& file)> request;

    static std::string tile_file(sector::Id tile);          // "h_2-14-8191-8192.vht"
    static std::string index_file(sector::Id sector);       // "idx_2-8-127-128.txt"
    // The height tiles listed for one index sector (both folders). False: no index there.
    bool load_index(sector::Id sector, std::vector<sector::Id>& tiles) const;
    bool load_tile(sector::Id tile, Tile& out) const;
    bool has_tile(sector::Id tile) const;
    // Written to the cache folder (or the local one), and listed in its sector's index there.
    bool save_tile(const Tile& t, bool local = false) const;

    static bool parse_index(const std::string& text, std::vector<sector::Id>& tiles);
};

// ---- The loaded tiles ----
struct Window {
    bool on = false;
    sector::ChartMap map;               // chart position -> cell coordinates relative to the window's first cell
    glm::dvec2 lo{0.0}, hi{0.0};        // the part of the window that is on the face (cell coordinates)
    uint8_t slot[kWin * kWin] = {};     // 0: no tile; else slot + 1
};
struct ActiveSet {
    uint32_t chart_serial = ~0u;        // the chart the windows' mappings were made for
    int      count = 0;
    // A circle (chart coordinates) outside which no loaded tile reaches.
    glm::dvec2 bound_c{0.0};
    double     bound_r = 0.0;
    Window   win[2];                    // the camera's face; a second face when one is within reach
    struct Slot { std::shared_ptr<const Tile> tile; uint32_t fade = 0xFFu; };
    Slot     slot[kSlots];

    int  find(sector::Id tile) const;   // its slot, -1: not loaded
    // The fade bits of every slot, from the tiles' `joined` and which neighbours are loaded.
    void update_fades();
    // The windows round a camera position in a chart (whose serial is given): the camera's
    // face, and a second face if a loaded tile lies on one. Calls update_windows.
    void place_windows(const globe::Chart& chart, uint32_t serial, const glm::dvec2& cam_xz);
    // Whether the windows still serve: made for this chart, the camera well inside the first,
    // and every loaded tile in one of them. If not: place_windows.
    bool windows_fit(uint32_t serial, const glm::dvec2& cam_xz) const;
    // The windows' slot tables and the bounding circle, from the slots (the windows' mappings
    // stay as they are).
    void update_windows(const globe::Chart& chart);
    // Metres added to the formula at a chart position: by the windows' series (what the shader
    // does), and by the exact mapping (any chart).
    float delta_series(double x, double z) const;
    float delta_exact(const globe::Chart& chart, double x, double z) const;
};
std::shared_ptr<const ActiveSet> active();
void set_active(std::shared_ptr<const ActiveSet> s);
// Metres the loaded tiles add to the formula's ground at a position in the CURRENT chart. Any
// thread. Costs one load of a number when nothing is loaded.
float ground_delta(float x, float z);

// What the shader reads (std140: every member a vec4 or uvec4).
struct GpuTable {
    float head[4];                      // tiles loaded; the bounding circle's centre x, z; its radius
    struct Win {
        float c_anchor[4];              // c.i, c.j, anchor.x, anchor.z
        float A[4];                     // di/dx, di/dz, dj/dx, dj/dz
        float Hi[4], Hj[4];             // xx, xz, zz of cell.i / cell.j
        float lim[4];                   // lo.i, lo.j, hi.i, hi.j (hi < lo: the window is off)
    } win[2];
    float    slot[kSlots][4];           // unit_m, fade bits, -, -
    uint32_t map[2][kWin * kWin / 16][4];       // a byte a cell: 0 none, else slot + 1
};
void fill_gpu(const ActiveSet& s, GpuTable& out);

// ---- Making tiles by code (the test generator; also what an importer would call) ----
enum class Shape { Mesa, Crater };
// Metres the shape adds at a distance from its middle.
float shape_profile(Shape s, double dist, double radius, double height);
// Adds the shape, centred on the planet point G, to every height tile it reaches - made if not
// there - joins the tiles it wrote where their shared edges agree, and saves them (cache
// folder, or the local one). Returns the tiles written.
std::vector<sector::Id> stamp(const Store& store, const glm::dvec3& G, Shape shape, double radius, double height, bool local = false);
// Sets `joined` between two tiles wherever they share an edge whose samples agree (and makes
// them the same to the last unit on the same face). True if anything changed.
bool join_tiles(Tile& a, Tile& b);
// The corner bits: a corner continues when the tile is joined along both edges that meet there
// and `has(diagonal tile)`.
void join_corners(Tile& t, const std::function<bool(sector::Id)>& has);
// Which edge (0 W, 1 E, 2 N, 3 S) of `from` faces `to`; -1: they do not share an edge.
int edge_toward(sector::Id from, sector::Id to);

}  // namespace vibe::region
