#version 460
#extension GL_GOOGLE_include_directive : require

// The terrain's own temporal upscaler ("temporal reconstruction", with the terrain drawn at less
// than full size). Drawn just before the world pass into a picture of the terrain alone, which
// is kept from frame to frame - two of them, written in turn (scene bindings 36 / 37):
// rgb = the colour, a = the distance (0 = no terrain there).
//
// The kept picture is TWICE the render size each way. The main upscaler (DLSS, FSR) moves the
// camera by part of a pixel every frame and builds the screen's detail from what it finds at
// those places: it has to be handed, every frame, what the ground looks like at that frame's
// place inside each pixel. A kept picture of the render size holds one fixed place per pixel -
// read between its pixels it gave the upscaler the same picture every frame, and the far
// paving came out in blocks the size of a rendered pixel (the joints between the stones are
// one SCREEN pixel wide). At twice the size the compose pass has a different part of it to
// read for each of the camera's places.
//
// The low-resolution picture is offset inside its own pixel to n x n places in turn (frame.cpp
// tr_keep_phase; n = 4 with the terrain at half size: every kept pixel is sampled once in 16
// frames). A kept pixel takes this frame's sample where one falls on it, and otherwise keeps
// what it had - last frame's picture, moved by the motion vectors, only where it shows the
// same ground (the distance agrees) and never further from this frame's own colours than
// their range allows (so light that changes, or ground that was hidden, does not leave a
// trail). A still view comes to full sharpness in n x n frames; a moving one stays between
// that and the plain enlargement.
//
// A picture of the terrain alone is needed because the world pass draws trees and buildings
// over the terrain: the frame's final picture cannot serve as its history.

layout(location = 0) out vec4 outResolved;

#include "scene_ubo.glsl"
#include "terrain_rm_upscale.glsl"

// Last frame's kept picture (the other of the two).
vec4 keptPrev(vec2 uv) {
    return (pc.tex_params.y > 0.5) ? textureLod(u_lrb_color, uv, 0.0) : textureLod(u_lrb_depth, uv, 0.0);
}

// ... read where the ground has moved from. Between its pixels, and so with a Catmull-Rom
// filter (five bilinear taps: the 4 x 4 footprint without its corners): read with the plain
// bilinear filter, a picture that moves is blurred a little more every frame it is carried
// along, and a turning camera had no more detail than the plain enlargement. On a pixel's
// centre (a still camera) this is that pixel. Where the footprint reaches past the ground -
// an empty pixel, or ground at another distance - the bilinear tap alone.
vec4 keptPrevSharp(vec2 uv, vec2 size) {
    vec2 sp  = uv * size;
    vec2 tc1 = floor(sp - 0.5) + 0.5;
    vec2 f   = sp - tc1;
    vec2 w0  = f * (-0.5 + f * (1.0 - 0.5 * f));
    vec2 w1  = 1.0 + f * f * (-2.5 + 1.5 * f);
    vec2 w2  = f * (0.5 + f * (2.0 - 1.5 * f));
    vec2 w3  = f * f * (-0.5 + 0.5 * f);
    vec2 w12 = w1 + w2;
    vec2 t0  = (tc1 - 1.0) / size;
    vec2 t3  = (tc1 + 2.0) / size;
    vec2 t12 = (tc1 + w2 / w12) / size;
    vec4 m = keptPrev(t12);
    vec4 a = keptPrev(vec2(t12.x, t0.y)), b = keptPrev(vec2(t0.x, t12.y));
    vec4 d = keptPrev(vec2(t3.x, t12.y)), e = keptPrev(vec2(t12.x, t3.y));
    float lo = min(min(a.a, b.a), min(min(d.a, e.a), m.a)), hi = max(max(a.a, b.a), max(max(d.a, e.a), m.a));
    if (lo <= 0.0 || hi > lo * 1.06 + 0.03) return keptPrev(uv);
    float wa = w12.x * w0.y, wb = w0.x * w12.y, wm = w12.x * w12.y, wd = w3.x * w12.y, we = w12.x * w3.y;
    vec4 r = (a * wa + b * wb + m * wm + d * wd + e * we) / (wa + wb + wm + wd + we);
    // (no further than the five taps themselves reach: the filter's negative lobes overshoot)
    r.rgb = clamp(r.rgb, min(min(a.rgb, b.rgb), min(min(d.rgb, e.rgb), m.rgb)), max(max(a.rgb, b.rgb), max(max(d.rgb, e.rgb), m.rgb)));
    return r;
}

void main() {
    vec2 keep = vec2(textureSize(u_lrb_color, 0));
    g_inv_target = 1.0 / keep;
    g_lr_offset  = pc.emissive.zw;     // (this picture's pixels are the still picture's: frame.cpp lr_phase)
    vec2 uv_true = gl_FragCoord.xy * g_inv_target;
    vec2 uv_lr   = uv_true - g_lr_offset;
    // What this pixel had: last frame's picture (the other of the two), where the ground has
    // moved from. (No camera offset in it: both pictures are the still one's.)
    vec4  h = vec4(0.0);
    float rush = 0.0;       // 0..1: the ground here is stretching too fast for anything kept to be of use
    if (pc.tex_params.z > 0.5) {
        vec2 ts0 = vec2(textureSize(u_lr_motion, 0));
        vec2 mv  = texture(u_lr_motion, uv_lr).rg;
        // Ground that comes toward the camera grows in the picture from frame to frame, and what
        // was kept of it is stretched along: a sample a round old covers several pixels by now.
        // Driving at 70 km/h the road came out in blotches the size of a low-resolution pixel,
        // worse than the plain enlargement. So: how much the picture stretches here in one
        // frame (the motion's change over one low-resolution pixel, as a share of that pixel) -
        // up to about 1 % the kept picture holds (a sixth over a round of 16 frames), from 4 %
        // on this frame's enlargement stands alone. And nothing kept that moves more than a
        // few low-resolution pixels a frame. At 70 km/h: kept beyond some 30 m, plain inside 8 m;
        // walking: plain inside 2 m. A turning camera stretches nothing.
        vec2 dx = texture(u_lr_motion, uv_lr + vec2(1.0 / ts0.x, 0.0)).rg - texture(u_lr_motion, uv_lr - vec2(1.0 / ts0.x, 0.0)).rg;
        vec2 dy = texture(u_lr_motion, uv_lr + vec2(0.0, 1.0 / ts0.y)).rg - texture(u_lr_motion, uv_lr - vec2(0.0, 1.0 / ts0.y)).rg;
        vec2 gx = 0.5 * abs(dx) * ts0, gy = 0.5 * abs(dy) * ts0;
        float stretch = max(max(gx.x, gx.y), max(gy.x, gy.y));
        float speed   = length(mv * ts0);                       // low-resolution pixels a frame
        rush = max(smoothstep(0.012, 0.04, stretch), smoothstep(5.0, 10.0, speed));
        vec2 uvp = uv_true - mv;
        if (rush < 0.999 && all(greaterThan(uvp, vec2(0.0))) && all(lessThan(uvp, vec2(1.0))))
            h = keptPrevSharp(uvp, keep);
    }
    // (nothing is kept there at all: the compose pass then enlarges this frame's picture itself,
    //  and marks the pixel for the denoiser as it does without this pass)
    if (rush >= 0.999) { outResolved = vec4(0.0); return; }
    // (with something kept, this frame's enlargement only has to stand in a little: the plain one)
    g_plain = h.a > 0.0;
    bool cur = tr_upscale();        // this frame's enlargement shows ground here
    // This frame's sample nearest this pixel, and how near it is: 1 on the pixel, 0 a
    // whole kept pixel away or more.
    vec2  ts = vec2(textureSize(u_lr_color, 0));
    vec2  s  = uv_lr * ts;
    vec2  k  = floor(s);
    vec2  d  = abs(s - k - 0.5) * (keep / ts);
    float w  = max(1.0 - d.x, 0.0) * max(1.0 - d.y, 0.0);
    ivec2 ki = clamp(ivec2(k), ivec2(0), ivec2(ts) - 1);
    vec3  smp = texelFetch(u_lr_color, ki, 0).rgb;
    // The outline of the ground against the sky, while the camera stands still
    // (pc.tex_params.w). This frame's enlargement draws the outline from a picture that lies
    // at another place inside its pixel every frame: taken from there, a mountain's edge
    // against the sky changed its shape sixteen times a round - it flickered. So at an
    // outline the kept picture is put together from the samples alone: a kept pixel is what
    // the last sample that fell ON it said - ground (its colour and distance) or sky
    // (a = -1; 0 stays "not known yet") - and is left as it is by every frame whose samples
    // fall elsewhere. After a round the outline stands, at the kept picture's resolution.
    // (Only with a still camera: the sky has no motion vectors to move its pixels along by.)
    if (pc.tex_params.w > 0.5 && pc.tex_params.z > 0.5) {
        ivec2 px = ivec2(gl_FragCoord.xy);
        vec4  hr = (pc.tex_params.y > 0.5) ? texelFetch(u_lrb_color, px, 0) : texelFetch(u_lrb_depth, px, 0);
        if (!cur || g_edge_motion.z > 0.5 || hr.a <= 0.0) {
            if (w > 0.24) {
                float sd = texelFetch(u_lr_depth, ki, 0).r;
                if (sd >= 0.99999) { outResolved = vec4(0.0, 0.0, 0.0, -1.0); return; }
                outResolved = vec4(hr.a > 0.0 ? mix(hr.rgb, smp, 0.6) : smp, 0.05 / max(1.0000333 - sd, 1e-7));
                return;
            }
            if (hr.a != 0.0) { outResolved = hr; return; }
        }
    }
    if (!cur) { outResolved = vec4(0.0); return; }
    vec3  c  = g_c;
    float zc = 0.05 / max(1.0000333 - g_depth, 1e-7);       // view distance (near 0.05)
    if (h.a > 0.0 && g_edge_motion.z < 0.5 && abs(h.a - zc) < 0.08 * zc + 0.05) {
        // What was kept may not be further from this frame's own colour here than the four
        // low-resolution colours round the pixel differ, and more. (Wide: those four do not span
        // what lies between them - a paving joint that none of them happened to hit was cut
        // out of the kept picture again, and the paving came out in blotches. And round the
        // plain enlargement `c`, which changes evenly from pixel to pixel - round the four
        // colours' own range, which changes from one low-resolution pixel to the next, ground
        // whose light changes (the shadow of grass in the wind) came out in blocks that size.)
        vec3  wid = 2.0 * (g_hi - g_lo) + 0.05;
        vec3  hc  = clamp(h.rgb, c - wid, c + wid);
        vec3  off = abs(h.rgb - hc) / wid;
        float cut = clamp(2.0 * max(off.r, max(off.g, off.b)), 0.0, 1.0);     // how much of it was cut off: the light here has changed
        // Between samples: what was there, and a very little of this frame's enlargement, so
        // an error does not stay for ever - more of it where the light has changed: soft and
        // of this frame rather than sharp and old. On a sample: most of the sample - not all of it:
        // the terrain's light is traced with a few rays a pixel and each sample carries that
        // frame's noise; a pixel is the mean of its last few.
        c = mix(hc, c, max(max(0.006, cut), rush));
        c = mix(c, smp, w * 0.6 * (1.0 - rush));
    }
    outResolved = vec4(c, zc);
}
