#version 460
#extension GL_GOOGLE_include_directive : require

// Bilinear upscale + depth-aware composite for the half-res procedural
// raymarched terrain. Runs as a fullscreen-tri AT THE START of the main
// world pass (before any cube/castle/dyn-prop draw) so the rasterised
// foreground geometry can layer on top via the standard depth test.
//
//   1. sample u_lr_color, u_lr_motion, u_lr_depth at the current
//      fragment's UV
//   2. write color and motion as outputs
//   3. write gl_FragDepth = upscaled depth so the existing
//      depth_compare = LESS_OR_EQUAL on this pipeline only commits the
//      raymarch where it's closer than the cube/castle depth that the
//      depth pre-pass already wrote into scene_depth
//   4. discard sky pixels (lr_depth ≈ 1.0 = far) so the cleared sky
//      colour shows through unchanged

layout(location = 0) out vec4 outColor;
layout(location = 1) out vec2 outMotion;
layout(location = 2) out vec4 outGBuffer0;
layout(location = 3) out vec4 outGBuffer1;

#include "scene_ubo.glsl"
#include "terrain_rm_upscale.glsl"

void main() {
    g_lr_offset = pc.emissive.xy;
    if (!tr_upscale()) discard;
    vec3  c = g_c;
    float lr_depth = g_depth;
    vec3  edge_motion = g_edge_motion;
    vec2  uv = g_uv;
    // The terrain's own full-resolution picture (terrain_rm_resolve.frag, drawn just before
    // the world pass; bindings 36 / 37 in turn, pc.tex_params.y = this frame's is the second):
    // the colour comes from there - this frame's samples put together with the frames before.
    // Depth, outlines and motion stay this frame's own.
    // (That picture is kept in the STILL picture's pixels; this pass's pixel lies part of a
    //  pixel beside them - the main upscaler's camera offset: pc.emissive.zw - xy. Not at an
    //  outline, and only where it shows ground at this pixel's distance: beside the sky its
    //  neighbours are empty.)
    if (pc.tex_params.x > 0.5 && edge_motion.z < 0.5) {
        vec2 hq = g_uv_true + pc.emissive.zw - pc.emissive.xy;
        vec4 h = (pc.tex_params.y > 0.5) ? textureLod(u_lrb_depth, hq, 0.0) : textureLod(u_lrb_color, hq, 0.0);
        float zc = 0.05 / max(1.0000333 - lr_depth, 1e-7);
        if (h.a > 0.0 && abs(h.a - zc) < 0.06 * zc + 0.03) c = h.rgb;
    }
    // The G-buffer, as the full-resolution terrain writes it (terrain_raymarch.frag): material 8
    // = already lit, a flat placeholder normal. This pass draws into all four attachments; left
    // unwritten the two were undefined - what deferred lighting and the Ray Reconstruction
    // guides (rr_guides.comp) read for every ground pixel.
    outGBuffer0  = vec4(clamp(c, vec3(0.0), vec3(1.0)), 8.0 / 255.0);
    outGBuffer1  = vec4(0.5, 0.5, 0.6, 0.0);       // (.b 0.6 marks "upscaled from the low-resolution picture": rr_guides.comp)
    outColor     = vec4(c, 1.0);
    outMotion    = edge_motion.z > 0.5 ? edge_motion.xy : texture(u_lr_motion, uv).rg;
    gl_FragDepth = lr_depth;
}
