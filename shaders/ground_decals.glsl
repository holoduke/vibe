// ground_decals.glsl - deferred box decals, after the reference game's (its deferred decal
// shader, read from the compiled code): a box in the world, projected onto what the G-buffer
// holds inside it - colour, normal, roughness - with HEIGHT-AWARE blending,
// `alpha *= 1 - saturate(30 h^2 (1 - ratio))`: what stands out of the surface (the stones of
// a wall, the clods and pebbles of the ground) pokes through the mud, the moss, the litter;
// `ratio` 1 covers everything. Included by deferred_lighting.frag (its set: binding 30, the
// ground effects' block - vk_engine/ground_fx.cpp fills the list, culls it and bins it into
// 32-pixel tiles, a bit per decal, so a pixel tests only the decals that reach its tile).
//
// The looks are generated here, no textures: 1 moss, 2 damp streaks, 3 dirt, 4 oil, 5 leaf
// litter, 6 cracks, 7 mud, 8 scorch (the gun's marks on the still world), 9 a bullet's hole.
// +16 on the kind: the decal ignores which way the surface faces (a box laid round a bridge's
// end, whose faces are not known).
//
// The height h of the surface: the raymarched ground writes how far its material stands out
// into G-buffer 1 blue (0.7 + 0.3 h); for a mesh it is taken from the colour's brightness.
// The ground's pixels are lit already (material 8): there the decal tints the lit colour.
#ifndef GROUND_DECALS_GLSL
#define GROUND_DECALS_GLSL
#define GFX_BINDING 30
#include "ground_fx.glsl"

float gd_h21(vec2 p) { vec3 p3 = fract(vec3(p.xyx) * 0.1031); p3 += dot(p3, p3.yzx + 33.33); return fract((p3.x + p3.y) * p3.z); }
float gd_vn(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(gd_h21(i), gd_h21(i + vec2(1.0, 0.0)), f.x), mix(gd_h21(i + vec2(0.0, 1.0)), gd_h21(i + vec2(1.0, 1.0)), f.x), f.y);
}
float gd_fbm(vec2 p) { return gd_vn(p) * 0.55 + gd_vn(p * 2.3 + 7.1) * 0.30 + gd_vn(p * 5.1 + 3.3) * 0.15; }

// Applies every decal that reaches this pixel. wp: world position; N, albedo, rough: the
// G-buffer's, changed in place; h: the surface's height 0..1. Returns how much was covered.
float gfx_decals_apply(vec3 wp, ivec2 pix, inout vec3 albedo, inout vec3 N, inout float rough, float h) {
    if (gfx.dec.w < 0.5) return 0.0;
    int tx = pix.x / kGfxTile, ty = pix.y / kGfxTile;
    int ntx = int(gfx.dec.y), nty = int(gfx.dec.z);
    if (tx >= ntx || ty >= nty) return 0.0;
    int base = (ty * ntx + tx) * 8;
    float covered = 0.0;
    for (int w = 0; w < 8; ++w) {
        uint bits = gfx.tiles[base + w];
        while (bits != 0u) {
            int b = findLSB(bits);
            bits &= bits - 1u;
            GfxDecal d = gfx.decals[w * 32 + b];
            vec3 r = wp - d.c.xyz;
            vec3 l = vec3(dot(r, d.ax.xyz) / dot(d.ax.xyz, d.ax.xyz), dot(r, d.ay.xyz) / dot(d.ay.xyz, d.ay.xyz), dot(r, d.az.xyz) / dot(d.az.xyz, d.az.xyz));
            if (any(greaterThan(abs(l), vec3(1.0)))) continue;
            int kind = int(d.c.w + 0.5);
            bool omni = kind >= 16;
            kind = kind & 15;
            vec3 an = normalize(d.ay.xyz);
            float facing = omni ? 1.0 : smoothstep(0.15, 0.5, dot(N, an));
            if (facing <= 0.0) continue;
            vec2 uv = l.xz;                                   // -1..1 across the box
            vec2 wuv = vec2(dot(r, normalize(d.ax.xyz)), dot(r, normalize(d.az.xyz)));   // metres in the box's plane
            float seed = d.ay.w;
            float shape = gd_fbm(wuv * 0.9 + seed * 17.0);
            // the box's edges: ragged, not square
            // (moss and damp on a wall reach right down to its foot: their box is not faded at the bottom)
            float edge = 1.0 - smoothstep(0.55 + 0.25 * shape, 1.0, max(abs(uv.x), (kind == 1 || kind == 2) ? max(uv.y, 0.0) : abs(uv.y)));
            float proj = 1.0 - smoothstep(0.7, 1.0, abs(l.y));
            float a = d.ax.w * edge * proj * facing;
            vec3  col = albedo; float rgh = rough; vec3 nn = N;
            float foot = 0.0;                                 // (moss: how much is the closed band at the ground)
            if (kind == 1) {                // moss: green clumps, thickest low down (uv.y = up the wall: -1 at the foot)
                // a closed band at the ground, ragged on top; above it patches that keep to the joints
                // (the height-aware blend below: near the foot the stones are covered too)
                float up = uv.y * 0.5 + 0.5;                          // 0 at the foot, 1 at the box's top
                float band = 1.0 - smoothstep(0.48, 0.70, up + 0.25 * (gd_fbm(vec2(wuv.x * 1.7, seed * 3.0)) - 0.5));
                float patches = smoothstep(0.50, 0.66, gd_fbm(wuv * 3.1 + seed * 5.0)) * (1.0 - smoothstep(0.6, 1.0, up));
                a *= max(band, patches);
                foot = band;
                col = mix(vec3(0.045, 0.080, 0.022), vec3(0.10, 0.14, 0.040), gd_vn(wuv * 11.0 + seed)) * mix(0.8, 1.15, gd_vn(wuv * 37.0));
                rgh = 0.9;
            } else if (kind == 2) {         // damp: dark streaks running down, wettest at the foot
                float s = smoothstep(0.35, 0.75, gd_vn(vec2(wuv.x * 2.4 + seed * 9.0, wuv.y * 0.25)));
                a *= mix(0.55, 1.0, s) * (1.0 - smoothstep(-0.4, 1.0, uv.y + 0.3 * (s - 0.5)));      // (a damp zone, streaks running up out of it)
                col = albedo * vec3(0.52, 0.55, 0.52);
                rgh = rough * 0.55;
            } else if (kind == 3) {         // dirt: a grey-brown film in patches
                a *= smoothstep(0.35, 0.65, shape) * 0.8;
                col = mix(albedo, vec3(0.13, 0.11, 0.085), 0.7);
                rgh = min(1.0, rough + 0.15);
            } else if (kind == 4) {         // oil: a dark stain, a darker ring, glossy
                float rr = length(uv * vec2(1.0, 1.0 + 0.3 * seed)) + 0.25 * (shape - 0.5);
                a *= 1.0 - smoothstep(0.55, 0.85, rr);
                col = albedo * mix(0.30, 0.18, smoothstep(0.45, 0.7, rr));
                // a faint rainbow sheen in the stain (thin-film colours by a slow noise), dull - at a
                // gloss of 0.28 it was a mirror of the sun
                float film = gd_fbm(wuv * 1.7 + seed * 7.0) * 6.2832;
                col *= 1.0 + 0.35 * (1.0 - smoothstep(0.2, 0.6, rr)) * (vec3(cos(film), cos(film - 2.094), cos(film + 2.094)) * 0.5);
                rgh = 0.75;                 // (at a grazing view lower read as a pale disc of sky)
            } else if (kind == 5) {         // leaf litter: single leaves, autumn colours, thicker at the foot of the wall
                vec2 cell = floor(wuv * 7.0);
                float lh = gd_h21(cell + seed * 31.0);
                vec2 fp = fract(wuv * 7.0) - 0.5;
                float leaf = 1.0 - smoothstep(0.18, 0.34, length(fp * vec2(1.0, 1.7)));
                a *= leaf * step(0.35, lh) * smoothstep(0.3, 0.6, shape + 0.3 * (-uv.y));
                col = mix(vec3(0.30, 0.13, 0.035), vec3(0.42, 0.30, 0.07), gd_h21(cell + 3.1)) * mix(0.6, 1.0, lh);
                rgh = 0.85;
                nn = normalize(N + vec3(fp.x, 0.0, fp.y) * 0.6 * leaf);
            } else if (kind == 6) {         // cracks: dark thin lines along a noise's zero
                float cn = gd_fbm(wuv * 1.3 + seed * 4.0);
                float line = 1.0 - smoothstep(0.0, 0.018, abs(cn - 0.5));
                a *= line * smoothstep(0.3, 0.6, gd_vn(wuv * 0.8 + 2.0));
                col = albedo * 0.35;
                rgh = rough;
            } else if (kind == 7) {         // mud: dark, wet, flat
                a *= smoothstep(0.30, 0.55, shape);
                col = mix(vec3(0.075, 0.058, 0.040), vec3(0.11, 0.085, 0.055), gd_vn(wuv * 5.0));
                rgh = mix(0.35, 0.75, gd_vn(wuv * 2.0 + 9.0));
                nn = normalize(mix(N, an, 0.6));
            } else if (kind == 8) {         // scorch: soot, densest in the middle
                float rr = length(uv) + 0.3 * (shape - 0.5);
                a *= 1.0 - smoothstep(0.25, 0.95, rr);
                col = albedo * 0.12;
                rgh = 0.95;
            } else {                        // a bullet's hole: a dark pit and a pale chipped ring
                float rr = length(uv);
                float pit = 1.0 - smoothstep(0.18, 0.28, rr), ring = smoothstep(0.2, 0.32, rr) * (1.0 - smoothstep(0.4, 0.7, rr));
                col = mix(albedo * 1.25, vec3(0.03), pit);
                a *= max(pit, ring * 0.5);
                rgh = 0.9;
            }
            // HEIGHT-AWARE: the surface's high points poke through (az.w = the ratio)
            float hx = max(h - 0.40, 0.0);
            a *= 1.0 - clamp(30.0 * hx * hx * (1.0 - max(d.az.w, 0.85 * foot)), 0.0, 1.0);
            a = clamp(a, 0.0, 1.0);
            if (a <= 0.002) continue;
            albedo = mix(albedo, col, a);
            rough  = mix(rough, rgh, a);
            N      = normalize(mix(N, nn, a));
            covered = max(covered, a);
        }
    }
    return covered;
}
#endif
