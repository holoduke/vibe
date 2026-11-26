// ssdm_common.glsl - shared by the screen-space displacement passes (ssdm_raster.comp,
// ssdm_hwraster.*, ssdm_resolve.frag, ssdm_dilate.comp, ssdm_behind.comp, ssdm_ground.frag):
// the push-constant block, the images and buffers, and how the G-buffer is read.
//
// The method is the reference game's, read from its compiled shaders (its "rasterizer /
// reconstruct" route); docs/ARCHITECTURE.md has the pass list, ssdm_displace.glsl the
// per-vertex displacement, ssdm_resolve.frag the rebuild of the G-buffer.
//
// Positions are CAMERA-RELATIVE world space (world - camera), as in the game: the matrices
// are built that way on the CPU (vk_engine/ssdm.cpp).
//
// Our depth runs the other way to the game's (0 near, 1 far; the game: reverse-Z, 0 = sky).
// Where the game compares raw depths the mapping is written out at the place:
//   game "raw < 1e-7" (sky)  = ours z >= 1;   game "raw == 1" (the near plane) = ours z == 0;
//   the rasterizer's packed depth is (1 - z), which orders like the game's raw depth.

layout(push_constant) uniform SsdmPC {
    mat4 inv_vp;     // clip -> camera-relative world
    mat4 vp;         // camera-relative world -> clip (this frame, as the scene was drawn)
    mat4 prev_vp;    // camera-relative world -> the clip space of the last frame (jitter difference taken out)
    vec4 size;       // render size in pixels, and 1 / it               (game: _bufferSizeAndInvSize)
    vec4 depth;      // x, y: linear depth = x / (y - z); z the reach T in m (game: _terrainNormalParams.x); w the amplitude scale (game: _deformBlurParam.y)
    vec4 view;       // xyz the camera's forward axis (game: _viewDir); w tan(horizontal fov / 2) (game: _screenPercentage.x)
    vec4 misc;       // x frame number, y debug view, z game _deformBlurParam.x (0: the edge / puddle fade for every class), w the height 0..1 a full value of the ground's G-buffer height stands for (ssdm_ground.frag)
} pc;

layout(set = 0, binding = 0) uniform sampler2D u_hn;      // the height feed: xyz vertex normal, w height 0..1 in 1/255 steps (w < 0: none)
layout(set = 0, binding = 1) uniform sampler2D u_depth;   // the scene's depth, before displacement
layout(set = 0, binding = 2) uniform sampler2D u_color;   // copies of the main pass's attachments,
layout(set = 0, binding = 3) uniform sampler2D u_g0;      //   before displacement
layout(set = 0, binding = 4) uniform sampler2D u_g1;
layout(set = 0, binding = 5) uniform sampler2D u_motion;

#ifdef SSDM_RASTER_BUF
// The rasterizer's target: one 64-bit word per pixel, kept by atomic max (game: g_rasterizerDepth).
//   63..40 the depth (float bits 31..8 of 1 - z)    38 written by the background pass
//   37 dx >= 0, 36..29 |dx|; 28 dy >= 0, 27..20 |dy|   (dx, dy: from this pixel to the source pixel)
//   19..11 / 10..2 the displaced normal, octahedral 9 + 9 bits      0 written by the hardware path
layout(std430, set = 0, binding = 6) buffer SsdmRaster { uint64_t ssdm_r[]; };
#endif
layout(std430, set = 0, binding = 7) buffer SsdmHwList { uint ssdm_hw_idx[]; };     // cells left to the hardware path: x << 16 | y
layout(std430, set = 0, binding = 8) buffer SsdmHwCtl {                             // its count, and the indirect draw made of it
    uint ssdm_hw_count;
    uint ssdm_hw_vertices, ssdm_hw_instances, ssdm_hw_first_vertex, ssdm_hw_first_instance;
};
#ifdef SSDM_OUT_IMAGES
layout(set = 0, binding = 9,  rgba16f) uniform image2D  o_color;     // the displaced G-buffer (game: g_baseColorUAV, g_normalUAV)
layout(set = 0, binding = 10, rgba8)   uniform image2D  o_g0;        //   (sRGB-encoded bytes; copied raw into the sRGB attachment)
layout(set = 0, binding = 11, r32ui)   uniform uimage2D o_g1;        //   (A2B10G10R10 packed by hand)
#endif
#ifdef SSDM_BEHIND_IMAGES
layout(set = 0, binding = 12, r32f)  uniform image2D  o_behind_mask;   // half resolution (game: g_behindReconstructMask)
layout(set = 0, binding = 13, r32ui) uniform uimage2D o_behind_tiles;  // one per 8 x 8 half-res texels (g_behindReconstructTileMask)
layout(set = 0, binding = 14, r32f)  uniform image2D  o_behind_weight; // half resolution (g_behindReconstructWeight)
#endif
layout(set = 0, binding = 15) uniform sampler2D u_behind_weight;      // the same, sampled bilinear

// ---- depth ----
float ssdm_lin(float z)      { return pc.depth.x / max(pc.depth.y - z, 1e-9); }   // game: _nearFarProj.x / max(1e-7, raw)
float ssdm_z_of_lin(float l) { return pc.depth.y - pc.depth.x / max(l, 1e-9); }
vec3 ssdm_unproject(vec2 ndc, float z) {
    vec4 q = pc.inv_vp * vec4(ndc, z, 1.0);
    return q.xyz / q.w;
}
float ssdm_depth_at(ivec2 p) { return texelFetch(u_depth, clamp(p, ivec2(0), ivec2(pc.size.xy) - 1), 0).r; }

// ---- the G-buffer as the game's displacement reads it ----
vec3 ssdm_oct_decode(vec2 e) {          // (deferred_lighting.frag octa_decode)
    e = e * 2.0 - 1.0;
    vec3 n = vec3(e.x, e.y, 1.0 - abs(e.x) - abs(e.y));
    float t = max(-n.z, 0.0);
    n.x += (n.x > 0.0) ? -t : t;
    n.y += (n.y > 0.0) ? -t : t;
    return normalize(n);
}
vec2 ssdm_oct_encode(vec3 n) {          // (cube.frag octa_encode)
    n /= (abs(n.x) + abs(n.y) + abs(n.z));
    vec2 e = (n.z >= 0.0) ? n.xy : ((1.0 - abs(n.yx)) * vec2(n.x >= 0.0 ? 1.0 : -1.0, n.y >= 0.0 ? 1.0 : -1.0));
    return e * 0.5 + 0.5;
}
bool ssdm_in(ivec2 p) { return all(greaterThanEqual(p, ivec2(0))) && all(lessThan(p, ivec2(pc.size.xy))); }
// The height byte (game: g_baseColor.z & 255, / 255): 0 where nothing feeds one.
float ssdm_h(ivec2 p) {
    if (!ssdm_in(p)) return 0.0;
    float w = texelFetch(u_hn, p, 0).w;
    return w < 0.0 ? 0.0 : w;
}
// The vertex normal (game: g_normal.xyz). Where the height feed drew nothing, the G-buffer's
// own normal stands in (our G-buffer keeps no separate vertex normal).
vec3 ssdm_vn(ivec2 p) {
    p = clamp(p, ivec2(0), ivec2(pc.size.xy) - 1);
    vec4 f = texelFetch(u_hn, p, 0);
    if (f.w >= 0.0) return normalize(f.xyz);
    return ssdm_oct_decode(texelFetch(u_g1, p, 0).xy);
}
// The surface byte compared between neighbours (game: g_baseColor.y & 255): our material id.
float ssdm_mat(ivec2 p) { return floor(texelFetch(u_g0, clamp(p, ivec2(0), ivec2(pc.size.xy) - 1), 0).a * 255.0 + 0.5); }
// The game's stencil class (& 127), mapped from what our G-buffer says:
//   0 nothing (sky, outside the picture), 3 the terrain height field (raymarched ground, material 8,
//   and the mesh terrain, material 3), 12 foliage (never displaced), 1 every other opaque surface.
uint ssdm_class(ivec2 p) {
    if (!ssdm_in(p)) return 0u;
    if (texelFetch(u_depth, p, 0).r >= 1.0) return 0u;
    uint m = uint(texelFetch(u_g0, p, 0).a * 255.0 + 0.5);
    if (m == 8u || m == 3u) return 3u;
    if (texelFetch(u_g1, p, 0).a > 0.83) return 12u;
    return 1u;
}
// The game's wet-ground rate (g_ssdmPuddleRate, 1 = dry): this game has no puddle map, so dry.
float ssdm_puddle(vec2 uv) { return 1.0; }
float ssdm_puddle5(vec2 uv) {
    vec2 o = 4.0 * pc.size.zw;
    return min(min(min(min(ssdm_puddle(uv), ssdm_puddle(uv - vec2(0.0, o.y))), ssdm_puddle(uv - vec2(o.x, 0.0))),
                   ssdm_puddle(uv + vec2(o.x, 0.0))), ssdm_puddle(uv + vec2(0.0, o.y)));
}
// The game's "screen exit" fade: 0 where the point mirrored by -N*s leaves the picture.
float ssdm_screen_exit(vec3 P, vec3 N, float s) {
    vec4 c0 = pc.vp * vec4(P, 1.0);
    vec4 c1 = pc.vp * vec4(P + s * N, 1.0);
    vec2 s0 = c0.xy / c0.w, d = c1.xy / c1.w - s0, e = s0 - d;
    float t = 0.0;
    if (e.x < -1.0) t = (-1.0 - e.x) / max(1e-06, d.x);
    if (e.x >  1.0) t = max(t, (e.x - 1.0) / max(1e-06, -d.x));
    if (e.y < -1.0) t = max(t, (-1.0 - e.y) / max(1e-06, d.y));
    if (e.y >  1.0) t = max(t, (e.y - 1.0) / max(1e-06, -d.y));
    return clamp(1.0 - t, 0.0, 1.0);
}
// Camera-relative position of pixel p (its corner, as the game's post passes take it).
vec3 ssdm_pos_corner(ivec2 p, float z) {
    return ssdm_unproject(vec2(float(p.x) * 2.0 * pc.size.z - 1.0, float(p.y) * 2.0 * pc.size.w - 1.0), z);
}
// 10:10:10:2 as the attachment A2B10G10R10_UNORM_PACK32 stores it (R in the low bits).
uint ssdm_pack1010102(vec4 v) {
    uvec4 q = uvec4(round(clamp(v, 0.0, 1.0) * vec4(1023.0, 1023.0, 1023.0, 3.0)));
    return q.x | (q.y << 10) | (q.z << 20) | (q.w << 30);
}
vec4 ssdm_unpack1010102(uint u) {
    return vec4(float(u & 1023u), float((u >> 10) & 1023u), float((u >> 20) & 1023u), float(u >> 30)) / vec4(1023.0, 1023.0, 1023.0, 3.0);
}
vec3 ssdm_to_srgb(vec3 c) {
    c = clamp(c, 0.0, 1.0);
    return mix(c * 12.92, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, step(vec3(0.0031308), c));
}
vec3 ssdm_from_srgb(vec3 c) {
    return mix(c / 12.92, pow((c + 0.055) / 1.055, vec3(2.4)), step(vec3(0.04045), c));
}
