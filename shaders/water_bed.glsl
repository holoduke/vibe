// water_bed.glsl - the bed under rivers, the stream and lakes from photo-scanned sets (2026-11-26).
// After the reference game: its bed is its ordinary ground layers continuing under the water,
// plus a "growth" set (moss / weed: colour, normal, height, roughness) tiled at 8 m and 3.15 m on
// submerged up-facing ground. Here four CC0 scans (assets/terrain_pbr/LICENSE.md), texture slots
// 43..46 (vk_engine.h kTexBed; the normal's alpha = height, the colour's alpha = roughness):
//   43 river stones (Rocks024S)   - where the water runs fast
//   44 silt / wet sand (Ground083) - slow and deep water
//   45 rock with moss (Rock046S)   - where the bed is steep
//   46 growth (Moss001)            - on up-facing bed below ~0.35 m (shore_wet.glsl sw_growth's mask)
// Laid with the ground's anti-tiling (rm_at), chart-independent (TILE_CELL), blended by height (the
// higher material shows where two meet). Switch: scene.water_wet bit 128 off = the code-made
// pebbles and weed as before (VIBE_NO_BED_SCAN=1).
// Needs: u_albedo / u_normal (terrain_raymarch.frag), rm_at (terrain_rm_shading_helpers.glsl),
// shore_wet.glsl (sw_growth).
#ifndef VIBE_WATER_BED_GLSL
#define VIBE_WATER_BED_GLSL

const int kWbSlot = 43;
// tile sizes (m) and the scans' height range (m)
const vec4 kWbCell = vec4(1.7, 2.6, 3.4, 3.15);
const vec4 kWbAmp  = vec4(0.07, 0.025, 0.09, 0.035);

struct WbW { vec4 w; float lod; int n; };      // the four sets' weights before the height blend, and the mip

bool wb_on() { return SW_ON(128); }

// The sets' weights at a point of the bed: speed of the water (m/s), depth under the surface (m),
// the ground's up component, and growth (0..1, sw_growth's mask: moss over the others).
WbW wb_weights(vec2 xz, float speed, float below, float ny, float px) {
    WbW W;
    float steep = 1.0 - smoothstep(0.55, 0.80, ny);
    float n = tile_noise2(xz, 0.21, vec2(13.0, 7.0));       // (patches: a gravel bar here, a silt pocket there)
    float fast = smoothstep(0.25, 1.0, speed + 0.5 * (n - 0.5));
    float deep = smoothstep(1.2, 3.5, below);
    float stones = fast * (1.0 - 0.7 * deep);
    W.w = vec4(stones * (1.0 - steep), (1.0 - stones) * (1.0 - steep), steep, 0.0);
    W.lod = log2(max(px / kWbCell.x * 1024.0, 1.0));
    W.n = min(3, int(scene.water_wet.x + 0.5) + 3);      // (3 - from the scene, so the loops are not unrolled)
    return W;
}

// One set's height (0..1) at xz.
float wb_h1(int i, vec2 xz, float cell, float lod) {
    RmAt A = rm_at(xz / TILE_CELL(cell), lod);
    return rm_at_normal(kWbSlot + i, A, lod).a;
}
// The bed's height (m) over its base at xz with the weights fixed: the height blend of the sets.
float wb_height(vec2 xz, WbW W) {
    // (the dominant set only: three anti-tiled reads per step of the march made the terrain shader
    //  too large for the optimizer - "ID overflow")
    int d = W.w.x >= W.w.y ? (W.w.x >= W.w.z ? 0 : 2) : (W.w.y >= W.w.z ? 1 : 2);
    return wb_h1(d, xz, kWbCell[d], max(W.lod + log2(kWbCell.x / kWbCell[d]), 0.0)) * kWbAmp[d];
}

// The bed's material at xz: colour (linear), tangent normal (x east, y north), roughness, height
// (m) and height 0..1 (for the growth's hollows and the displacement's height byte).
struct WbMat { vec3 alb; vec2 tn; float rough; float h; float hn; };
WbMat wb_material(vec2 xz, WbW W) {
    WbMat M;
    M.alb = vec3(0.0); M.tn = vec2(0.0); M.rough = 0.0; M.h = 0.0; M.hn = 0.5;
    vec4 hv = vec4(0.0);
    vec3 al[3]; vec4 nr[3]; float rg[3];
    for (int i = 0; i < W.n; ++i) {
        al[i] = vec3(0.0); nr[i] = vec4(0.5, 0.5, 1.0, 0.0); rg[i] = 0.8;
        if (W.w[i] < 0.02) continue;
        float li = max(W.lod + log2(kWbCell.x / kWbCell[i]), 0.0);
        RmAt A = rm_at(xz / TILE_CELL(kWbCell[i]), li);
        nr[i] = rm_at_normal(kWbSlot + i, A, li);
        vec4 a = vec4(0.0);
        for (int j = 0; j < A.n; ++j) a += textureLod(u_albedo[kWbSlot + i], A.uv[j], li) * A.w[j];
        al[i] = a.rgb; rg[i] = a.a;
        hv[i] = nr[i].a;
    }
    vec4 s = hv * kWbAmp + W.w * 0.06;
    float m = max(max(s.x, s.y), s.z);
    vec4 b = max(s - (m - 0.02), vec4(0.0)) * step(vec4(0.02), W.w);
    b.w = 0.0;
    float bs = max(b.x + b.y + b.z, 1e-5);
    b /= bs;
    for (int i = 0; i < 3; ++i) {
        M.alb += al[i] * b[i];
        M.tn += (nr[i].xy * 2.0 - 1.0) * b[i];
        M.rough += rg[i] * b[i];
        M.h += hv[i] * kWbAmp[i] * b[i];
        M.hn += (hv[i] - 0.5) * b[i];
    }
    M.hn = clamp(M.hn, 0.0, 1.0);
    return M;
}
// The dominant set alone (the bank strip, terrain_rm_main_light.glsl: the full height blend of the
// three a second time ran the terrain shader past SPIR-V's id limit). rgb colour, a height 0..1;
// tn the tangent normal.
vec4 wb_dominant(vec2 xz, WbW W, out vec2 tn) {
    int d = W.w.x >= W.w.y ? (W.w.x >= W.w.z ? 0 : 2) : (W.w.y >= W.w.z ? 1 : 2);
    float li = max(W.lod + log2(kWbCell.x / kWbCell[d]), 0.0);
    RmAt A = rm_at(xz / TILE_CELL(kWbCell[d]), li);
    vec4 nr = rm_at_normal(kWbSlot + d, A, li);
    vec4 a = vec4(0.0);
    for (int j = 0; j < A.n; ++j) a += textureLod(u_albedo[kWbSlot + d], A.uv[j], li) * A.w[j];
    tn = nr.xy * 2.0 - 1.0;
    return vec4(a.rgb, nr.a);
}

// The growth: the moss scan at the reference's two scales (8 m and 3.15 m), darker and more olive
// under water, its own height; where the mask (shore_wet.glsl sw_growth) holds it covers the bed.
void wb_growth(inout WbMat M, vec2 xz, float lod, float growth) {
    if (growth > 0.0) {
        float lg = lod + log2(kWbCell.x / 3.15);
        RmAt A1 = rm_at(xz / TILE_CELL(3.15), max(lg, 0.0));
        RmAt A2 = rm_at(xz / TILE_CELL(8.0) + 0.37, max(lg - 1.34, 0.0));
        vec4 n1 = rm_at_normal(kWbSlot + 3, A1, max(lg, 0.0)), n2 = rm_at_normal(kWbSlot + 3, A2, max(lg - 1.34, 0.0));
        vec4 a1 = vec4(0.0), a2 = vec4(0.0);
        for (int j = 0; j < A1.n; ++j) a1 += textureLod(u_albedo[kWbSlot + 3], A1.uv[j], max(lg, 0.0)) * A1.w[j];
        for (int j = 0; j < A2.n; ++j) a2 += textureLod(u_albedo[kWbSlot + 3], A2.uv[j], max(lg - 1.34, 0.0)) * A2.w[j];
        vec4 a = mix(a1, a2, 0.4);
        vec4 nn = mix(n1, n2, 0.4);
        float l = dot(a.rgb, vec3(0.2126, 0.7152, 0.0722));
        vec3 gc = mix(vec3(l), a.rgb, 0.7) * vec3(0.80, 0.85, 0.55);      // (under water: less saturated, toward olive)
        M.alb = mix(M.alb, gc, growth);
        M.tn = mix(M.tn, nn.xy * 2.0 - 1.0, growth);
        M.rough = mix(M.rough, a.a, growth);
        M.h = mix(M.h, nn.a * kWbAmp.w + 0.02, growth);
        M.hn = clamp(mix(M.hn, 0.5 + 0.5 * nn.a, growth), 0.0, 1.0);
    }
}

#endif
