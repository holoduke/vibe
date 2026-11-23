#version 460
#extension GL_GOOGLE_include_directive : require

// The terrain's own temporal upscaler ("temporal reconstruction", with the terrain drawn at less
// than full size). Drawn just before the world pass into a full-resolution picture of the
// terrain alone, which is kept from frame to frame - two of them, written in turn (scene
// bindings 36 / 37): rgb = the colour, a = the distance (0 = no terrain there).
//
// The low-resolution picture is offset by a quarter of its pixel, to four places in turn
// (frame.cpp): at half size every full-resolution pixel is sampled exactly once in four
// frames. A pixel takes this frame's sample where one falls on it, and otherwise keeps what
// it had - last frame's picture, moved by the motion vectors, only where it shows the same
// ground (the distance agrees) and never further from this frame's own colours than their
// range allows (so light that changes, or ground that was hidden, does not leave a trail).
// A still view comes to full sharpness in four frames; a moving one stays between that and
// the plain enlargement.
//
// It was: the one frame before, in a second low-resolution picture (a checkerboard of two) -
// smooth, but the paving's joints and the sand's grain were gone and high contrast showed a
// stipple. A picture of the terrain alone is needed because the world pass draws trees and
// buildings over the terrain: the frame's final picture cannot serve as its history.

layout(location = 0) out vec4 outResolved;

#include "scene_ubo.glsl"
#include "terrain_rm_upscale.glsl"

void main() {
    g_lr_offset = pc.emissive.zw;      // (this picture's pixels are the still picture's: frame.cpp lr_phase)
    if (!tr_upscale()) { outResolved = vec4(0.0); return; }
    vec3  c  = g_c;
    float zc = 0.05 / max(1.0000333 - g_depth, 1e-7);       // view distance (near 0.05)
    if (pc.tex_params.z > 0.5 && g_edge_motion.z < 0.5) {
        vec2 mv  = texture(u_lr_motion, g_uv).rg;
        vec2 uvp = g_uv_true - mv;      // (no camera offset in it: both pictures are the still one's)
        if (all(greaterThan(uvp, vec2(0.0))) && all(lessThan(uvp, vec2(1.0)))) {
            // last frame's picture: the other of the two
            vec4 h = (pc.tex_params.y > 0.5) ? textureLod(u_lrb_color, uvp, 0.0) : textureLod(u_lrb_depth, uvp, 0.0);
            if (h.a > 0.0 && abs(h.a - zc) < 0.08 * zc + 0.05) {
                // (wide: the four low-resolution colours round a pixel do not span what lies between
                //  them - a paving joint that none of the four happened to hit was cut out of
                //  the kept picture again, and the paving came out in blotches)
                vec3 pad = 1.5 * (g_hi - g_lo) + 0.05;
                vec3 hc = clamp(h.rgb, g_lo - pad, g_hi + pad);
                // How near one of this frame's samples is to this pixel: 1 on it, 0 midway
                // between two. On a sample: the sample. Between: what was there, and a
                // little of this frame's enlargement, so an error does not stay for ever.
                vec2 ts = vec2(textureSize(u_lr_color, 0));
                vec2 fc = fract(g_uv * ts - 0.5);
                vec2 dc = min(fc, 1.0 - fc);
                float wc = (1.0 - 2.0 * dc.x) * (1.0 - 2.0 * dc.y);
                // (not all of the sample at once: the terrain's light is traced with a few rays a
                //  pixel and each sample carries that frame's noise - taken whole, it stood in the
                //  picture for four frames as blotches. Half of it: a pixel is the mean of its
                //  last few samples.)
                c = mix(hc, c, clamp(wc * wc, 0.04, 1.0) * 0.5);
            }
        }
    }
    outResolved = vec4(c, zc);
}
