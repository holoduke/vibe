// region_tiles.glsl - stored ground laid over the formula's: the height tiles of the regions
// (engine/region_tile.h; vk_engine/regions.cpp; docs/ARCHITECTURE.md "Regions").
//
// region_delta(wp): the metres the loaded tiles add to the formula's ground at a chart
// position. Called by sampleHeightDelta (terrain_rm_height_tex.glsl), so by every height the
// terrain shader works out. The CPU's twin is region::ActiveSet::delta_series (region_tile.cpp)
// - the same steps in the same order: keep them in step.
//
// How the tile under a point is found without a loop over tiles:
//   1. everything loaded lies inside one circle (head): most ground fails this test and is done;
//   2. the chart position becomes cell coordinates of the planet's sector grid, relative to the
//      first cell of a 16 x 16 window round the camera - a second-order series the CPU worked
//      out in doubles (engine/sector.h chart_map);
//   3. a byte per window cell says which slot of the array texture holds that sector's tile;
//   4. the fraction within the cell is the place in the tile: four texel fetches, bilinear,
//      times the edge mask (a tile fades toward every edge where it does not continue).
// A second window serves another face of the planet's cube when one is within reach.

const int   kRegionSlots = 32;          // = region::kSlots
const int   kRegionWin   = 16;          // = region::kWin
const float kRegionSteps = 512.0;       // = region::kTileN - 1
const float kRegionFade  = 0.0625;      // = region::kFade

struct RegionWin {
    vec4 c_anchor;      // xy: cell coordinates at the anchor; zw: the anchor (chart x, z)
    vec4 A;             // di/dx, di/dz, dj/dx, dj/dz
    vec4 Hi;            // second derivatives of the cell's i: xx, xz, zz
    vec4 Hj;            // ... of its j
    vec4 lim;           // the part of the window that is on the face: lo.ij, hi.ij (hi < lo: off)
};
layout(std140, set = 0, binding = 49) uniform RegionTiles {
    vec4      head;                     // x: tiles loaded; yz: the bounding circle's centre; w: its radius
    RegionWin win[2];
    vec4      slot[kRegionSlots];       // x: metres per unit, y: the fade bits
    uvec4     map[2 * 16];              // per window: a byte a cell (0 none, else slot + 1)
} u_region;
layout(set = 0, binding = 50) uniform sampler2DArray u_region_h;        // R16_SNORM, 513 x 513 a layer

float region_sm(float t) { t = clamp(t, 0.0, 1.0); return t * t * (3.0 - 2.0 * t); }

// The edge mask (region::edge_mask). Bits: 1 W (a = 0), 2 E, 4 N (b = 0), 8 S, then the corners
// 16 NW, 32 NE, 64 SW, 128 SE.
float region_mask(vec2 ab, uint fade) {
    if (fade == 0u) return 1.0;
    float w = kRegionFade;
    float m = 1.0;
    if ((fade & 1u) != 0u) m *= region_sm(ab.x / w);
    if ((fade & 2u) != 0u) m *= region_sm((1.0 - ab.x) / w);
    if ((fade & 4u) != 0u) m *= region_sm(ab.y / w);
    if ((fade & 8u) != 0u) m *= region_sm((1.0 - ab.y) / w);
    if ((fade & 16u) != 0u)  m *= region_sm(length(ab) / w);
    if ((fade & 32u) != 0u)  m *= region_sm(length(vec2(1.0 - ab.x, ab.y)) / w);
    if ((fade & 64u) != 0u)  m *= region_sm(length(vec2(ab.x, 1.0 - ab.y)) / w);
    if ((fade & 128u) != 0u) m *= region_sm(length(vec2(1.0) - ab) / w);
    return m;
}

float region_delta(vec2 wp) {
    if (u_region.head.x < 0.5) return 0.0;
    vec2 bd = wp - u_region.head.yz;
    if (dot(bd, bd) > u_region.head.w * u_region.head.w) return 0.0;
    for (int k = 0; k < 2; ++k) {
        vec4 lim = u_region.win[k].lim;
        vec4 ca = u_region.win[k].c_anchor, A = u_region.win[k].A, Hi = u_region.win[k].Hi, Hj = u_region.win[k].Hj;
        vec2 d = wp - ca.zw;
        vec2 cc = ca.xy + vec2(A.x * d.x + A.y * d.y, A.z * d.x + A.w * d.y) +
                  0.5 * vec2(Hi.x * d.x * d.x + 2.0 * Hi.y * d.x * d.y + Hi.z * d.y * d.y,
                             Hj.x * d.x * d.x + 2.0 * Hj.y * d.x * d.y + Hj.z * d.y * d.y);
        if (any(lessThan(cc, lim.xy)) || any(greaterThanEqual(cc, lim.zw))) continue;
        ivec2 ci = ivec2(cc);
        int c = ci.y * kRegionWin + ci.x;
        uint s = (u_region.map[k * 16 + (c >> 4)][(c >> 2) & 3] >> uint((c & 3) * 8)) & 255u;
        if (s == 0u) return 0.0;
        vec2 ab = cc - vec2(ci);
        vec4 sl = u_region.slot[s - 1u];
        vec2 x = clamp(ab, 0.0, 1.0) * kRegionSteps;
        ivec2 i0 = min(ivec2(x), ivec2(int(kRegionSteps) - 1));
        vec2 f = x - vec2(i0);
        int layer = int(s) - 1;
        float h00 = texelFetch(u_region_h, ivec3(i0, layer), 0).r;
        float h10 = texelFetch(u_region_h, ivec3(i0 + ivec2(1, 0), layer), 0).r;
        float h01 = texelFetch(u_region_h, ivec3(i0 + ivec2(0, 1), layer), 0).r;
        float h11 = texelFetch(u_region_h, ivec3(i0 + ivec2(1, 1), layer), 0).r;
        float h0 = h00 + (h10 - h00) * f.x, h1 = h01 + (h11 - h01) * f.x;
        return (h0 + (h1 - h0) * f.y) * 32767.0 * sl.x * region_mask(ab, uint(sl.y + 0.5));
    }
    return 0.0;
}
