#version 460
// Screen-space displacement - the resolve pass: our code for the reference game's
// SSDM_ReconstructorPS (its full background variant, with the behind-material masks), read
// from its compiled shader. Full screen, after the rasterizers.
//
// Per pixel, from the rasterizer's 64-bit word (ssdm_common.glsl):
//   * covered by displaced stonework: the G-buffer of the SOURCE pixel the word points to,
//     the depth of the displaced surface, and within 32 m the surface normal turned from the
//     source's vertex normal onto the displaced one (the shading normal rotated by the same
//     turn); where the displaced surface leans away from the flat one (a stone's side) the
//     normal and the albedo are taken from four pixels around the source (W/480, H/270 apart),
//     blending to the rotated normal by 20 m;
//   * not covered: the pixel as it was, but its depth pulled toward the camera by what its
//     height would have raised it (1.5 x, at most 0.3 m, for the terrain), fading out over the
//     last 8 m before the reach - so what is drawn later is tested against the raised surface;
//     where that pull is large close by (< 16 m) the half-resolution behind-material mask is
//     marked for ssdm_dilate.comp / ssdm_behind.comp.
// Writes the displaced G-buffer copy (colour, albedo, normal), the motion and the depth.
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#define SSDM_RASTER_BUF
#define SSDM_BEHIND_IMAGES
#include "ssdm_common.glsl"

layout(location = 0) out vec4 outColor;      // the displaced copy of the scene colour
layout(location = 1) out vec2 outMotion;     // the live motion attachment
layout(location = 2) out vec4 outG0;         // the displaced copy of the albedo (sRGB-encoded) + material id
layout(location = 3) out uint outG1;         // the displaced copy of the normal layer (10:10:10:2)

vec2 camera_motion(ivec2 p, float z) {
    vec2 uv = (vec2(p) + 0.5) * pc.size.zw;
    vec3 P  = ssdm_unproject(uv * 2.0 - 1.0, z);
    vec4 q  = pc.prev_vp * vec4(P, 1.0);
    return q.w > 0.0 ? uv - (q.xy / q.w * 0.5 + 0.5) : vec2(0.0);
}

void main() {
    ivec2 p  = ivec2(gl_FragCoord.xy);
    uint64_t r = ssdm_r[uint(p.y) * uint(pc.size.x) + uint(p.x)];
    uint  lo = uint(r & 0xFFFFFFFFUL), hi = uint(r >> 32);
    float sd = texelFetch(u_depth, p, 0).r;
    float rz = uintBitsToFloat(hi & 0xFFFFFF00u);                  // 1 - z of the displaced surface
    bool  covered = ((hi & 192u) == 64u) && !(rz < 1e-05);

    // ---------------- not covered, a surface: as it was, the depth pulled ----------------
    if (sd != 0.0 && !(covered || sd >= 1.0)) {
        vec4  col = texelFetch(u_color, p, 0);
        vec4  g0  = texelFetch(u_g0, p, 0);
        vec4  g1  = texelFetch(u_g1, p, 0);
        uint  cls = ssdm_class(p);
        vec3  Ng  = ssdm_vn(p);
        vec3  P   = ssdm_pos_corner(p, sd);
        vec3  toCam = -P * inversesqrt(dot(P, P));
        float viewZ = dot(P, pc.view.xyz);
        float s   = pc.depth.w * 0.2;
        float rate;
        if ((cls - 1u) > 1u && pc.misc.z > 0.0) rate = 1.0;
        else {
            float fade = ssdm_screen_exit(P, Ng, s);
            rate = ssdm_puddle5(vec2(p) * pc.size.zw) * fade;
        }
        float h   = ((cls - 52u) < 16u) ? 0.2 : s;
        float amt = rate * h * clamp(ssdm_h(p) * 255.0 * 0.005098039, 0.0, 1.0)
                  / max(0.125, clamp(dot(toCam, Ng), 0.0, 1.0));
        amt *= clamp((pc.depth.z - viewZ) * 0.125, 0.0, 1.0);
        float off;
        if (cls == 97u || cls == 98u || (cls & 126u) == 2u || cls == 20u) off = min(0.3, amt * 1.5);
        else if ((cls - 11u) < 9u || cls == 107u)                         off = 0.0;
        else                                                              off = amt;
        float outDepth = sd;
        if (off > 0.0) outDepth = ssdm_z_of_lin(dot(pc.view.xyz, P + off * toCam) * 0.999999);
        if (viewZ < 16.0 && off > viewZ * 0.02) {
            ivec2 hp = p >> 1;
            ivec2 hs = (ivec2(pc.size.xy) + 1) >> 1;
            if (all(lessThan(hp, hs))) imageStore(o_behind_mask, hp, vec4(clamp((16.0 - viewZ) * 0.25, 0.0, 1.0)));
            ivec2 t  = (hp - 3) >> 3;
            ivec2 ts = imageSize(o_behind_tiles);
            for (int k = 0; k < 4; ++k) {
                ivec2 q = t + ivec2(k & 1, k >> 1);
                if (all(greaterThanEqual(q, ivec2(0))) && all(lessThan(q, ts))) imageStore(o_behind_tiles, q, uvec4(1u));
            }
        }
        outColor  = col;
        outG0     = vec4(ssdm_to_srgb(g0.rgb), g0.a);
        outG1     = ssdm_pack1010102(g1);
        outMotion = off > 0.0 ? camera_motion(p, outDepth) : texelFetch(u_motion, p, 0).xy;
        gl_FragDepth = outDepth;
        return;
    }

    // ---------------- covered (or nothing there): rebuilt from the source pixel ----------------
    int  dxs = (((hi >> 5) & 1u) == 1u) ? 1 : -1;
    int  dys = (((lo >> 28) & 1u) == 1u) ? 1 : -1;
    ivec2 s  = p + ivec2(dxs * int((r >> 29) & 255UL), dys * int((lo >> 20) & 255u));
    s = clamp(s, ivec2(0), ivec2(pc.size.xy) - 1);
    vec2  e  = vec2(float((lo >> 2) & 511u), float((lo >> 11) & 511u)) * 0.003913894 - 1.0;
    vec3  Nd = vec3(e, 1.0 - abs(e.x) - abs(e.y));
    if (!(Nd.z >= 0.0)) Nd.xy = (1.0 - abs(e.yx)) * vec2(e.x >= 0.0 ? 1.0 : -1.0, e.y >= 0.0 ? 1.0 : -1.0);
    Nd = normalize(Nd);
    vec4  col = texelFetch(u_color, s, 0);
    vec4  g0  = texelFetch(u_g0, s, 0);
    vec4  g1  = texelFetch(u_g1, s, 0);
    vec3  Ng  = ssdm_vn(s);
    vec3  D   = ssdm_oct_decode(g1.xy);                 // the shading normal (the game's detail normal, in world space)
    float zOut = 1.0 - rz;
    float linZ = ssdm_lin(zOut);

    vec3  Dout = D;
    vec3  albOut = ssdm_to_srgb(g0.rgb);                // (the game averages its albedo bytes: here the sRGB bytes)
    vec4  colOut = col;
    if (linZ < 32.0) {
        bool  steep = clamp(exp2(log2(1.0 - clamp(dot(Nd, Ng) * 0.5 + 0.5, 0.0, 1.0)) * 0.2), 0.0, 1.0) > 0.3;
        // the turn that takes Ng onto Nd, as a quaternion, applied to the shading normal
        // (the game divides by zero when Nd = -Ng; such a pixel keeps its normal here)
        float d1 = dot(Ng, Nd) + 1.0;
        vec3  Dr = D;
        if (d1 > 1e-6) {
            float sq = sqrt(d1 * 2.0);
            vec3  q  = cross(Ng, Nd) / sq;
            float w  = d1 / sq;
            Dr = (w * w - dot(q, q)) * D + 2.0 * w * cross(q, D) + 2.0 * dot(q, D) * q;
        }
        Dout = Dr;
        if (steep) {
            int ox = int(pc.size.x * 0.002083333), oy = int(pc.size.y * 0.003703704);
            ivec2 lim = ivec2(pc.size.xy) - 1;
            ivec2 t0 = clamp(s + ivec2(ox, 0), ivec2(0), lim), t1 = clamp(s + ivec2(0, oy), ivec2(0), lim);
            ivec2 t2 = clamp(s - ivec2(ox, 0), ivec2(0), lim), t3 = clamp(s - ivec2(0, oy), ivec2(0), lim);
            vec3 Davg = 0.25 * (ssdm_oct_decode(texelFetch(u_g1, t0, 0).xy) + ssdm_oct_decode(texelFetch(u_g1, t1, 0).xy)
                              + ssdm_oct_decode(texelFetch(u_g1, t2, 0).xy) + ssdm_oct_decode(texelFetch(u_g1, t3, 0).xy));
            Dout   = mix(Davg, Dr, clamp(linZ * 0.05, 0.0, 1.0));
            albOut = 0.25 * (ssdm_to_srgb(texelFetch(u_g0, t0, 0).rgb) + ssdm_to_srgb(texelFetch(u_g0, t1, 0).rgb)
                           + ssdm_to_srgb(texelFetch(u_g0, t2, 0).rgb) + ssdm_to_srgb(texelFetch(u_g0, t3, 0).rgb));
            // (the raymarched ground is lit already: its colour is what its albedo is to the rest)
            colOut.rgb = 0.25 * (texelFetch(u_color, t0, 0).rgb + texelFetch(u_color, t1, 0).rgb
                               + texelFetch(u_color, t2, 0).rgb + texelFetch(u_color, t3, 0).rgb);
        }
    }
    if (dot(Dout, Dout) <= 1e-12) Dout = D;
    outColor  = colOut;
    outG0     = vec4(albOut, g0.a);
    outG1     = ssdm_pack1010102(vec4(ssdm_oct_encode(normalize(Dout)), g1.b, g1.a));
    outMotion = covered ? camera_motion(p, zOut) : texelFetch(u_motion, p, 0).xy;
    if (covered && pc.misc.y > 0.5) { outG0.rgb = vec3(0.8, 0.1, 0.1); outColor.rgb = vec3(0.6, 0.05, 0.05); }   // debug view
    gl_FragDepth = zOut;
}
