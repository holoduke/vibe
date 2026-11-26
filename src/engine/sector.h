#pragma once
// Sectors: one address for a piece of the planet's surface, the same from every chart.
//
// The engine's coordinates are a chart (engine/globe.h): small numbers that stand for another
// piece of the planet every few kilometres. Whatever has a place of its own on the planet - a
// stored height tile, a region's content, and later what a server owns and what a client is
// told about - needs a name for that place that does not change with the chart. That name is a
// sector:
//
//   The planet is a cube blown up to a sphere. Each of the cube's six faces is a quadtree: level
//   0 is the whole face (a quarter of the way round the planet, 9,990 km), every level halves
//   the cells. A sector is (face, level, i, j) - 64 bits (Id).
//
//       level   cell (m, at most)     used for
//         8         39,000            an index file / a server's area (kIndexLevel)
//        10          9,760            (a town and its country)
//        14            610            a height tile (engine/region_tile.h kTileLevel)
//        20              9.5
//        28              0.037        (the finest: 28 bits a side)
//
//   Face coordinates are equal-angle (s = 4/pi atan(u), u the gnomonic coordinate): cells are
//   the same width along a face's axes and at most 1.41 times smaller in area at its corners
//   (the plain cube map: 5.2 times). Cells are NOT squares of a round number of metres - they
//   cannot be on a sphere - so whatever is stored per cell is laid out in the cell's own
//   coordinates (0..1 across it), and the engine maps those into the chart.
//
//   Everything here is integers and doubles: a sector is as exact 5000 km from home as at home
//   (a double holds a face coordinate to 2e-16 of 10,000 km = 2 nm).
//
// The faces, in the planet's frame (home is the planet's top, +y; engine/globe.h):
//       face   outward   a grows toward   b grows toward
//        0       +x          +y               +z
//        1       -x          +z               +y
//        2       +y          +x (east)        +z (south)       <- home is its middle (a = b = 0.5)
//        3       -y          +z               +x
//        4       +z          +x               +y
//        5       -z          +y               +x
//
// How a per-sector server model maps onto this is written in docs/ARCHITECTURE.md, "Regions".

#include "engine/globe.h"

#include <glm/glm.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace vibe::sector {

constexpr int kMaxLevel = 28;
// Metres across a level-0 cell along a face's axis: a quarter of the planet's circumference.
constexpr double kFaceM = globe::kR * 1.5707963267948966;

// A sector's address. 0 is "none". Bits: 63..61 face + 1, 60..56 level, 55..28 i, 27..0 j.
struct Id {
    uint64_t v = 0;
    bool valid() const { return v != 0; }
    int  face()  const { return int(v >> 61) - 1; }
    int  level() const { return int((v >> 56) & 31u); }
    uint32_t i() const { return uint32_t((v >> 28) & 0xFFFFFFFu); }
    uint32_t j() const { return uint32_t(v & 0xFFFFFFFu); }
    bool operator==(const Id& o) const { return v == o.v; }
    bool operator!=(const Id& o) const { return v != o.v; }
    bool operator<(const Id& o) const { return v < o.v; }
};
Id make(int face, int level, uint32_t i, uint32_t j);

// A place on a face: a, b in 0..1 across it.
struct FacePos { int face = 2; double a = 0.5, b = 0.5; };
// The face a point of the planet (unit vector, planet frame) lies on, and where on it.
FacePos face_pos(const glm::dvec3& G);
// The same point's coordinates on a GIVEN face (the formula carried on past the face's edge:
// smooth there, which is what a mapping worked out near an edge needs). Only for points
// within a quarter turn of the face's middle.
glm::dvec2 face_ab(int face, const glm::dvec3& G);
// The point of the planet at (a, b) of a face (a, b may lie a little outside 0..1).
glm::dvec3 point(int face, double a, double b);

// The sector of `level` that holds the point G; `frac` (optional): where in it, 0..1 each way.
Id cell_at(const glm::dvec3& G, int level, glm::dvec2* frac = nullptr);
// A point of a sector: (fa, fb) = (0, 0) its first corner, (0.5, 0.5) its middle.
glm::dvec3 point(Id id, double fa, double fb);
inline glm::dvec3 centre(Id id) { return point(id, 0.5, 0.5); }

// The hierarchy.
Id   parent(Id id);                     // none for level 0
Id   child(Id id, int k);               // k = 0..3: (i, j) + (k & 1, k >> 1) of the next level
Id   ancestor(Id id, int level);        // the sector of a coarser (or the same) level that holds it
bool contains(Id outer, Id inner);        // `inner` is `outer` or lies in it

// The sector next to `id` one step along i (di = +-1) or along j (dj = +-1) - across the edge
// of the face where that is where the step leads. Diagonal steps: only within a face (at a
// face's edge a diagonal neighbour is found through the two straight ones; at a cube corner
// three cells meet and there is none). None if there is no such sector.
Id  neighbour(Id id, int di, int dj);
// All the sectors of the same level that touch `id` (at most 8; 7 at a cube corner). The four
// across its edges come first, in the order -i, +i, -j, +j.
int neighbours(Id id, Id out[8]);

// Sizes on the ground: the length of the cell's two middle lines (m), and the largest distance
// from its middle to a corner.
glm::dvec2 size_m(Id id);
double     radius_m(Id id);
// Every sector of `level` that has a point within `radius` metres (over the ground) of G -
// "the sectors within reach": what is streamed, and what a client is interested in. A few
// more than touch the circle may be returned (a cell is tested by its bounding circle).
void cover(const glm::dvec3& G, double radius, int level, std::vector<Id>& out);

// "2-14-8191-8192": face, level, i, j. What files and messages are named by.
std::string name(Id id);
bool        parse(const std::string& s, Id& out);

// ---- Sectors and the chart ----
// Where a point of a sector lies in a chart, and the sector a chart position is in.
glm::dvec2 to_chart(const globe::Chart& c, Id id, double fa, double fb);
Id         from_chart(const globe::Chart& c, double x, double z, int level, glm::dvec2* frac = nullptr);

// Chart position -> cell coordinates of one face at one level, near an anchor, as a second
// order series - what a shader can evaluate in floats (the exact mapping goes through the
// planet's unit vector, which a float holds to half a metre). Cell coordinates are relative to
// (base_i, base_j), so they stay small: cell = base + c + A d + (d' H d) / 2, d = p - anchor.
// Within 4 km of the anchor it is off by under 2 mm (tests/test_sector.cpp).
struct ChartMap {
    int        face = 2, level = 0;
    int64_t    base_i = 0, base_j = 0;
    glm::dvec2 anchor{0.0};
    glm::dvec2 c{0.0};
    glm::dmat2 A{1.0};                  // A[col][row]: column 0 = d(cell)/dx, column 1 = d(cell)/dz
    glm::dvec3 Hi{0.0}, Hj{0.0};        // second derivatives of cell.i and cell.j: xx, xz, zz
    glm::dvec2 eval(const glm::dvec2& p) const {
        const glm::dvec2 d = p - anchor;
        return c + A * d + 0.5 * glm::dvec2(Hi.x * d.x * d.x + 2.0 * Hi.y * d.x * d.y + Hi.z * d.y * d.y,
                                            Hj.x * d.x * d.x + 2.0 * Hj.y * d.x * d.y + Hj.z * d.y * d.y);
    }
};
ChartMap chart_map(const globe::Chart& c, const glm::dvec2& anchor, int face, int level, int64_t base_i, int64_t base_j);
// The exact mapping the series stands for (cell coordinates relative to the same base).
glm::dvec2 chart_cell_exact(const globe::Chart& c, const glm::dvec2& p, int face, int level, int64_t base_i, int64_t base_j);

}  // namespace vibe::sector
