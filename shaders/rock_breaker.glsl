// rock_breaker.glsl - what every steep rock face shares, placed formation or the terrain's own
// cliff, and the rock kit's bake (vk_engine/rock_bake.cpp; made by tools/rocks/make_formations.py).
// After the reference game's rock material (read from its compiled terrain-mesh shaders):
//
//   * THE BREAKER (rk_breaker): one pattern for the region, 24 m a tile, laid on by world
//     position on what stands up - beds of uneven thickness and hardness, offset at faults, cut
//     by joints; streaks and blotches. Colour by `alpha x (1 - N.y)`; a bump
//     `saturate((0.2 k d2 + T k) x 4)^4` (T = the pattern's relief, 0 in its cracks) whose slope
//     bends the normal from 2 m on, full from 7 m (close by the fine scan carries the surface).
//     A formation and the cliff behind it carry the same beds on.
//   * THE COLOUR STRIP BY ALTITUDE (rk_gradient): bands of colour that follow height, with a
//     dip and bent by a noise, a darker foot at the water and a paler top, overlay-blended.
//   * THE BAKE of a kit piece (rk_bake): its sculpt's normal (the piece's own axes), height,
//     ambient occlusion and the ground's weight on its tops, in the piece's own unwrap.
//
// One array texture, two channels a layer: three layers a piece, then the breaker's two.
// Needs before it: ground_fx.glsl (gfx.rock: x pieces with a bake, y switches, z the breaker's
// first layer, w the layers' size in texels).
#ifndef ROCK_BREAKER_GLSL
#define ROCK_BREAKER_GLSL

layout(set = 0, binding = 63) uniform sampler2DArray u_rock_bake;

const int RK_TOPS = 1, RK_BAKE = 2, RK_BREAKER = 4, RK_ACCUM = 8;       // (rock_bake.cpp: VIBE_NO_ROCK_*)
#define RK_ON(b) ((int(gfx.rock.y + 0.5) & (b)) != 0)
const float kRkTile = 1.0 / 24.0;       // the breaker's repeats a metre

float rk_h21(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}
float rk_vn(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(rk_h21(i), rk_h21(i + vec2(1.0, 0.0)), f.x), mix(rk_h21(i + vec2(0.0, 1.0)), rk_h21(i + vec2(1.0, 1.0)), f.x), f.y);
}

// (rc: the pattern's relief and cracks; k = 1 - N.y)
float rk_b(vec2 rc, float k) { return pow(clamp((0.2 * k * 0.3 + rc.x * (1.0 - rc.y) * k) * 4.0, 0.0, 1.0), 4.0); }

// The breaker at wp on a face with normal Ng, `dist` from the eye, fp = metres a pixel there.
// Returns what the colour is multiplied by. grad: subtract from the shading normal (then
// normalise). bump: the pattern's height 0..1.
vec3 rk_breaker(vec3 wp, vec3 Ng, float dist, float fp, out vec3 grad, out float bump) {
    grad = vec3(0.0); bump = 0.0;
    float k = 1.0 - clamp(Ng.y, 0.0, 1.0);
    if (gfx.rock.x < 0.5 || !RK_ON(RK_BREAKER) || k < 0.03) return vec3(1.0);
    float l0 = gfx.rock.z;
    // (the wall's plane: the larger of the two level axes, blended only near the diagonal)
    vec2 w = Ng.xz * Ng.xz; w *= w;
    float ws = w.x + w.y;
    if (ws < 1e-6) return vec3(1.0);
    w /= ws;
    float lod = log2(max(fp * kRkTile * gfx.rock.w, 1.0));
    float e = 1.5 * exp2(lod) / gfx.rock.w;
    float nfade = clamp((dist - 2.0) * 0.2, 0.0, 1.0);
    vec2 tone = vec2(0.0);
    float crack = 0.0;
    for (int p = 0; p < 2; ++p) {
        float wq = p == 0 ? w.x : w.y;
        if (wq < 0.02) continue;
        vec2 uv = (p == 0 ? wp.zy : wp.xy) * kRkTile;
        vec2 t  = textureLod(u_rock_bake, vec3(uv, l0), lod).rg;
        vec2 r  = textureLod(u_rock_bake, vec3(uv, l0 + 1.0), lod).rg;
        float b0 = rk_b(r, k);
        tone += t * wq; crack += r.y * wq; bump += b0 * wq;
        if (nfade > 0.01) {
            float bu = rk_b(textureLod(u_rock_bake, vec3(uv + vec2(e, 0.0), l0 + 1.0), lod).rg, k);
            float bv = rk_b(textureLod(u_rock_bake, vec3(uv + vec2(0.0, e), l0 + 1.0), lod).rg, k);
            vec3  au = p == 0 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
            grad += (au * (bu - b0) + vec3(0.0, 1.0, 0.0) * (bv - b0)) * (wq * kRkTile / e);
        }
    }
    ws = max((w.x >= 0.02 ? w.x : 0.0) + (w.y >= 0.02 ? w.y : 0.0), 1e-4);
    tone /= ws; crack /= ws; bump /= ws;
    // (10 cm of relief; the slope kept within what a normal can show)
    grad = (grad - Ng * dot(grad, Ng)) * (0.10 * nfade / ws);
    float gl = length(grad);
    if (gl > 1.4) grad *= 1.4 / gl;
    vec3 c = vec3(0.42 + 1.16 * tone.x) * mix(vec3(0.66, 0.58, 0.48), vec3(1.26, 1.28, 1.30), tone.y) * (1.0 - 0.55 * crack);      // (2026-11-26: wider than at first - on a sunlit face the beds did not read)
    return mix(vec3(1.0), c, 0.85 * k);
}

// The colour strip by altitude laid over colour c (overlay). k = 1 - N.y; sea: the water's level.
vec3 rk_gradient(vec3 wp, vec3 c, float k, float sea) {
    if (!RK_ON(RK_BREAKER)) return c;
    float t  = wp.y - 0.035 * (wp.x + wp.z) + 6.0 * (rk_vn(wp.xz * 0.02) - 0.5);
    float b1 = rk_vn(vec2(t / 7.0, 3.3)), b2 = rk_vn(vec2(t / 2.3, 11.7));
    vec3  G  = vec3(0.5) + vec3(0.17, 0.13, 0.07) * (b1 - 0.5) * 2.0 + vec3(-0.05, 0.0, 0.06) * (b2 - 0.5) * 2.0;
    G += 0.07 * smoothstep(25.0, 70.0, t) - 0.17 * (1.0 - smoothstep(0.5, 7.0, t - sea));
    G = mix(vec3(0.5), G, smoothstep(0.1, 0.5, k));
    return mix(2.0 * c * G, 1.0 - 2.0 * (1.0 - c) * (1.0 - G), step(vec3(0.5), c));
}

// A kit piece's bake at the mesh's uv. piece: its number (< 0: none). m: the piece's axes in
// the world. Nv: the mesh's own normal (world). k: how much of the baked normal and height is
// in (it leaves as a texel shrinks under a pixel: the unwrap's islands run into each other in
// the small mips).
struct RkBake { vec3 n; float h; float ao; float w; float k; };
RkBake rk_bake(vec2 uv, float piece, mat3 m, vec3 Nv) {
    RkBake b;
    b.n = Nv; b.h = 0.5; b.ao = 1.0; b.w = 0.0; b.k = 0.0;
    if (piece < -0.5 || piece > gfx.rock.x - 0.5) return b;
    float l = floor(piece + 0.5) * 3.0;
    vec2 dx = dFdx(uv), dy = dFdy(uv);
    vec2 c = textureGrad(u_rock_bake, vec3(uv, l + 2.0), dx, dy).rg;
    b.ao = c.r; b.w = c.g;
    float fpx = max(length(dx), length(dy)) * gfx.rock.w;
    b.k = 1.0 - smoothstep(5.0, 14.0, fpx);
    if (b.k > 0.01) {
        vec2 a = textureGrad(u_rock_bake, vec3(uv, l), dx, dy).rg, d = textureGrad(u_rock_bake, vec3(uv, l + 1.0), dx, dy).rg;
        vec3 no = vec3(a.x, d.x, a.y) * 2.0 - 1.0;
        b.n = normalize(mix(Nv, normalize(m * no), b.k));
        b.h = mix(0.5, d.y, b.k);
    }
    return b;
}

#endif
