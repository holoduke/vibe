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
    bool  cur = tr_upscale();          // this frame's low-resolution picture shows ground here
    vec3  c = g_c;
    float lr_depth = g_depth;
    vec3  edge_motion = g_edge_motion;
    vec2  uv = g_uv;
    bool  kept = false;
    // The outline of the ground against the sky, while the camera stands still
    // (pc.tex_params.w; terrain_rm_resolve.frag "The outline"): the kept picture says for each
    // of its pixels whether it is ground (a > 0), sky (a < 0) or not known yet (0). Where the
    // four of them round this pixel are not all ground, they decide - ground where they cover
    // half of it or more, with their colour and distance - and not this frame's
    // low-resolution picture, whose outline lies elsewhere every frame.
    bool  outline_done = false;
    if (pc.tex_params.w > 0.5 && pc.tex_params.x > 0.5 && pc.tex_params.z > 0.5) {
        vec2 hq = g_uv_true + pc.emissive.zw - pc.emissive.xy;
        vec4 a4 = (pc.tex_params.y > 0.5) ? textureGather(u_lrb_depth, hq, 3) : textureGather(u_lrb_color, hq, 3);
        bool all_ground = all(greaterThan(a4, vec4(0.0)));
        if ((!cur || !all_ground) && !any(equal(a4, vec4(0.0)))) {
            vec2  ks = vec2(textureSize(u_lrb_color, 0));
            vec2  st = hq * ks - 0.5;
            ivec2 b  = ivec2(floor(st));
            vec2  fr = st - vec2(b);
            // textureGather order: .x (0,1)  .y (1,1)  .z (1,0)  .w (0,0)
            vec4  bw = vec4((1.0 - fr.x) * fr.y, fr.x * fr.y, fr.x * (1.0 - fr.y), (1.0 - fr.x) * (1.0 - fr.y));
            vec4  gm = vec4(greaterThan(a4, vec4(0.0)));
            float wg = dot(bw, gm);
            if (wg < 0.5) discard;
            const ivec2 kOff[4] = ivec2[4](ivec2(0, 1), ivec2(1, 1), ivec2(1, 0), ivec2(0, 0));
            vec3 cs = vec3(0.0);
            for (int i = 0; i < 4; ++i) {
                if (gm[i] < 0.5) continue;
                ivec2 tx = clamp(b + kOff[i], ivec2(0), ivec2(ks) - 1);
                cs += bw[i] * ((pc.tex_params.y > 0.5) ? texelFetch(u_lrb_depth, tx, 0).rgb : texelFetch(u_lrb_color, tx, 0).rgb);
            }
            c = cs / wg;
            float z = dot(bw * gm, a4) / wg;
            lr_depth = min(1.0000333 - 0.05 / max(z, 0.05), 0.999989);
            edge_motion = vec3(0.0, 0.0, 1.0);          // (a still camera: nothing moves)
            kept = true;
            outline_done = true;
        }
    }
    if (!outline_done && !cur) discard;
    // The terrain's own kept picture (terrain_rm_resolve.frag, drawn just before the world
    // pass; bindings 36 / 37 in turn, pc.tex_params.y = this frame's is the second): the colour
    // comes from there - this frame's samples put together with the frames before. Depth,
    // outlines and motion stay this frame's own.
    // (That picture is the STILL picture, at twice this pass's size; this pass's pixel lies
    //  part of a pixel aside in it - the main upscaler's camera offset: pc.emissive.zw - xy -
    //  and that is where it is read: a different part of it for each of the camera's places,
    //  which is what the main upscaler builds the screen's detail from. Not at an outline, and
    //  only where it shows ground at this pixel's distance: beside the sky its neighbours are
    //  empty.)
    if (!outline_done && pc.tex_params.x > 0.5 && edge_motion.z < 0.5) {
        vec2 hq = g_uv_true + pc.emissive.zw - pc.emissive.xy;
        vec4 h = (pc.tex_params.y > 0.5) ? textureLod(u_lrb_depth, hq, 0.0) : textureLod(u_lrb_color, hq, 0.0);
        float zc = 0.05 / max(1.0000333 - lr_depth, 1e-7);
        if (h.a > 0.0 && abs(h.a - zc) < 0.06 * zc + 0.03) { c = h.rgb; kept = true; }
    }
    // The G-buffer, as the full-resolution terrain writes it (terrain_raymarch.frag): material 8
    // = already lit, a flat placeholder normal. This pass draws into all four attachments; left
    // unwritten the two were undefined - what deferred lighting and the Ray Reconstruction
    // guides (rr_guides.comp) read for every ground pixel.
    outGBuffer0  = vec4(clamp(c, vec3(0.0), vec3(1.0)), 8.0 / 255.0);
    // (.b 0.6 marks "enlarged from the low-resolution picture": rr_guides.comp hands Ray
    //  Reconstruction a flat guide there, so the blocks that change from frame to frame are
    //  taken for noise. Not what comes from the kept picture: that is the ground itself, as
    //  steady as the full-resolution pass draws it, and is marked as that pass marks it - with
    //  the flat guide its detail was smoothed over as if it were noise.)
    outGBuffer1  = vec4(0.5, 0.5, kept ? 0.7 : 0.6, 0.0);
    outColor     = vec4(c, 1.0);
    outMotion    = edge_motion.z > 0.5 ? edge_motion.xy : texture(u_lr_motion, uv).rg;
    gl_FragDepth = lr_depth;
}
