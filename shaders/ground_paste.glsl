// ground_paste.glsl - every mesh that stands on the ground is pasted into it at its contact line
// (the reference game's paste pass, read from its compiled terrain-blend shader): within a zone
// above the ground - `1 - h / (zone + 2.9 saturate((dist - 500) 0.002))`, a hand's width close
// by and 3 m from a kilometre and a half - the ground's weight falls from 1 at the ground to 0
// at the zone's top, and of the two materials the one whose weight x height stands higher shows
// over a band of 0.15 (score = weight x height): soil in the mesh's low spots (mortar joints,
// the rock's clefts), its high points (stones, ribs) standing clear. Above the zone a stain of
// the same earth and the damp creep a little way up the foot, ragged; at the very foot the
// contact's own shade.
//
// The ground's height: the home tile's height map (the surface the terrain is drawn from, less
// its close relief - which is why the near zone is 22 cm and not the reference's 10), beyond it
// the ground baked round the camera for the world's grass (veg_stream.cpp wv_nf_poll; 512 m at
// 1 m - grass_raymarch.frag nf_*: keep in step), else the caller's own ground line. The
// ground's MATERIAL there: the baked ground texture (terrain_bake.cpp; its colour, its
// material height, its normal - the same pictures the terrain is lit from, so the soil on the
// wall is the soil beside it), and where that is not readable (too far, not baked yet, a steep
// face) the soil scan the terrain lays (slot 9, 0.45 repeats a metre), green-brown where the
// grass mask has grass, the rock's tone on steep ground.
//
// Used by cube.frag for every surface that may touch the ground (see the gate there: which
// surfaces are left clean). Needs before it: scene_ubo.glsl, ground_fx.glsl, u_heightmap_full,
// u_grass_mask, u_albedo[], u_normal[], tnoise (cube_noise_rays.glsl).
#ifndef GROUND_PASTE_GLSL
#define GROUND_PASTE_GLSL

layout(set = 0, binding = 46) uniform sampler2DArray u_gp_bake_a;    // (terrain_rm_bake.glsl u_bake_a / _n / _m)
layout(set = 0, binding = 47) uniform sampler2DArray u_gp_bake_n;
layout(set = 0, binding = 48) uniform sampler2DArray u_gp_bake_m;

// The share of the second of two materials where both are present, by height (the reference
// game's blend; terrain_rm_heightfield.glsl rm_height_blend, rock_form.glsl rf_height_blend).
float gp_height_blend(float ha, float hb, float w) {
    float a = max(ha, 0.01) * (1.0 - w), b = max(hb, 0.01) * w;
    float d = max(max(a, b) - 0.15, 0.0);
    a = max(a - d, 0.0); b = max(b - d, 0.0);
    return b / max(a + b, 1e-5);
}

struct GpGround {
    float above;     // metres of this point above the ground (1e9: unknown)
    vec3  n;         // the ground's normal there
    float grassy;    // 0..1 the lawn's mask (home tile only)
};

// The ground under `wp`. fallback_above: the caller's own idea (a rock piece's ground line), 1e9 if none.
GpGround gp_ground(vec3 wp, float fallback_above) {
    GpGround g;
    g.above = fallback_above; g.n = vec3(0.0, 1.0, 0.0); g.grassy = 0.0;
    if (scene.globe_info.x < 0.5 && max(abs(wp.x), abs(wp.z)) < 1020.0) {
        vec2 guv = wp.xz / 2048.0 + 0.5;
        g.above = wp.y - textureLod(u_heightmap_full, guv, 0.0).r;
        if (g.above < 2.5) {
            const float e = 1.5;
            float hx = textureLod(u_heightmap_full, guv + vec2(e / 2048.0, 0.0), 0.0).r - textureLod(u_heightmap_full, guv - vec2(e / 2048.0, 0.0), 0.0).r;
            float hz = textureLod(u_heightmap_full, guv + vec2(0.0, e / 2048.0), 0.0).r - textureLod(u_heightmap_full, guv - vec2(0.0, e / 2048.0), 0.0).r;
            g.n = normalize(vec3(-hx, 2.0 * e, -hz));
            g.grassy = textureLod(u_grass_mask, guv, 0.0).r;
        }
        return g;
    }
    // (row 0 of the near-field bake = its origin and a mark: grass_raymarch.frag nf_*)
    vec3 nm = texelFetch(u_normal[22], ivec2(2, 0), 0).rgb * 255.0;
    if (scene.globe_info.z > 0.5 && abs(nm.r - 77.0) < 0.5 && abs(nm.g - 123.0) < 0.5 && abs(nm.b - 201.0) < 0.5) {
        vec3 na = texelFetch(u_normal[22], ivec2(0, 0), 0).rgb * 255.0, nb = texelFetch(u_normal[22], ivec2(1, 0), 0).rgb * 255.0;
        vec2 org = (vec2(na.r * 65536.0 + na.g * 256.0 + na.b, nb.r * 65536.0 + nb.g * 256.0 + nb.b) - 8388608.0) * 64.0;
        vec2 l = wp.xz - org;
        if (all(greaterThan(l, vec2(4.0))) && all(lessThan(l, vec2(507.0)))) {
            #define GP_NF(q) (dot(textureLod(u_normal[22], ((q) + 0.5) / 512.0, 0.0).rgb, vec3(255.0, 255.0 / 256.0, 255.0 / 65536.0)) - 64.0)
            g.above = wp.y - GP_NF(l);
            if (g.above < 2.5) {
                float hx = GP_NF(l + vec2(1.5, 0.0)) - GP_NF(l - vec2(1.5, 0.0)), hz = GP_NF(l + vec2(0.0, 1.5)) - GP_NF(l - vec2(0.0, 1.5));
                g.n = normalize(vec3(-hx, 3.0, -hz));
            }
            #undef GP_NF
        }
    }
    return g;
}

// The baked ground's material at the ground point under `wp`, `dist` from the eye (the level
// goes by the distance, as terrain_rm_bake.glsl rm_bake_fetch). False: not readable there.
bool gp_bake(vec3 wp, float dist, out vec3 alb, out float h, out vec3 n) {
    alb = vec3(0.0); h = 0.5; n = vec3(0.0, 1.0, 0.0);
    float t0 = gfx.paste.y;
    if (t0 <= 0.0) return false;
    float lf = max(log2(max(dist / t0, 1.0)), gfx.paste.z);
    if (lf > 6.0) return false;
    float l0 = floor(lf), f = lf - l0;
    float span0 = 2048.0 * 0.0075 * exp2(l0);              // (kBakeSize x kBakeTexel0, terrain_rm_bake.glsl)
    if (max(abs(wp.x - scene.camera_pos.x), abs(wp.z - scene.camera_pos.z)) > 0.47 * span0 * (f > 0.02 ? 2.0 : 1.0)) return false;
    vec3 u0 = vec3(wp.xz / span0, l0);
    vec4 m = textureLod(u_gp_bake_m, u0, 0.0);
    vec4 a = textureLod(u_gp_bake_a, u0, 0.0);
    vec2 nn = textureLod(u_gp_bake_n, u0, 0.0).rg;
    if (f > 0.02 && l0 < 5.5) {
        vec3 u1 = vec3(wp.xz / (span0 * 2.0), l0 + 1.0);
        m = mix(m, textureLod(u_gp_bake_m, u1, 0.0), f);
        a = mix(a, textureLod(u_gp_bake_a, u1, 0.0), f);
        nn = mix(nn, textureLod(u_gp_bake_n, u1, 0.0).rg, f);
    }
    if (m.a < 0.995) return false;
    nn = nn * 2.0 - 1.0;
    n = normalize(vec3(nn.x, sqrt(max(1.0 - dot(nn, nn), 0.04)), nn.y));
    alb = a.rgb * a.rgb;
    h = a.a;
    return true;
}

// The paste. albedo / N: the mesh's, changed in place. Ng: its geometric normal. mh: the mesh
// material's height 0..1 at the pixel (1 = what stands out). self_rock: the mesh is rock - on
// steep ground the ground's material IS its own (only the normal and the tone meet). shade:
// how deep the contact's own shade goes (0.62 for a loose rock, less for a building that has
// its own). Returns the ground's share (for the roughness).
float gp_paste(inout vec3 albedo, inout vec3 N, vec3 wp, vec3 Ng, float mh, float dist, GpGround g, bool self_rock, float shade) {
    if (g.above > 2.5) return 0.0;
    float zone = gfx.paste.w + 2.9 * clamp((dist - 500.0) * 0.002, 0.0, 1.0);
    float gn1 = tnoise(wp.xz * 3.1 + wp.y * 2.3), gn2 = tnoise(wp.xz * 0.7 + 9.0);
    float wg = clamp(1.0 - (g.above + 0.05 * (gn1 - 0.5)) / zone, 0.0, 1.0);
    float g_rock = smoothstep(0.56, 0.78, 1.0 - g.n.y);        // (steep ground is the cliff's rock: terrain_rm_main_ground.glsl)
    vec3  earth = mix(vec3(0.105, 0.086, 0.060), vec3(0.060, 0.082, 0.036), max(smoothstep(0.3, 0.7, g.grassy), 0.5 * smoothstep(0.35, 0.70, gn2)));
    vec3  g_col, g_nrm;
    float g_h;
    vec3  b_alb, b_n;
    if (gp_bake(wp, dist, b_alb, g_h, b_n)) {
        g_col = b_alb;
        g_nrm = normalize(g.n + vec3(b_n.x, 0.0, b_n.z));
        earth = mix(earth, b_alb, 0.6);                        // (the stain takes the ground's own colour)
        if (self_rock) g_col = mix(g_col, albedo, g_rock);
    } else {
        vec2  suv = wp.xz * 0.45;
        vec4  sn = texture(u_normal[9], suv);
        vec3  sa = texture(u_albedo[9], suv).rgb / max(textureLod(u_albedo[9], suv, 12.0).rgb, vec3(0.02));
        g_h = sn.a;
        vec3 rockc = self_rock ? albedo : vec3(0.16, 0.15, 0.135) * clamp(sa, vec3(0.6), vec3(1.5));
        g_col = mix(earth * clamp(sa, vec3(0.5), vec3(1.7)) * mix(0.70, 1.08, smoothstep(0.15, 0.85, sn.a)), rockc, g_rock);
        g_nrm = normalize(g.n + vec3(sn.x - 0.5, 0.0, sn.y - 0.5) * 2.0 * (1.0 - g_rock));
    }
    float share = gp_height_blend(mh, g_h, wg);
    // above the zone: earth thrown up by rain, thinner with height and ragged; the damp below it
    // (not on the cliff's own rock)
    float stain = (1.0 - smoothstep(0.0, 0.75, g.above - 0.20 * gn1 + 0.05)) * (1.0 - g_rock) * 0.45;
    float damp  = (1.0 - smoothstep(0.0, 0.40, g.above - 0.12 * gn2)) * (1.0 - g_rock);
    albedo = mix(albedo, earth * 1.05, stain * (1.0 - share));
    albedo *= mix(1.0, 0.78, damp * (1.0 - share) * mix(0.5, 1.0, 1.0 - mh));
    albedo = mix(albedo, g_col, share);
    N = normalize(mix(N, g_nrm, share));
    albedo *= mix(shade, 1.0, smoothstep(0.0, 0.30, g.above + 0.06));
    return max(share, 0.5 * stain);
}

#endif
