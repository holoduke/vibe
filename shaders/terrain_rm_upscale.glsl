// The low-resolution terrain enlarged to the full picture: shared by the pass that keeps the
// terrain's own full-resolution picture from frame to frame (terrain_rm_resolve.frag) and the
// pass that puts the terrain into the world (terrain_raymarch_compose.frag). Needs
// scene_ubo.glsl before it. What tr_upscale() found is left in the g_* variables below.

// Push constants — same layout as cube.frag's PushConstants. We use
//   pc.color.x = sharpen strength (0 = off, 1 = aggressive)
//   pc.color.y = LR width  (texels)
//   pc.color.z = LR height (texels)
// The remaining fields are ignored.
layout(push_constant) uniform PC {
    mat4 mvp;
    mat4 model;
    mat4 prev_mvp;
    vec4 color;
    vec4 emissive;
    vec4 tex_params;
    vec4 grass_params;
} pc;

// Half-res raymarch outputs from the previous pass.
layout(set = 0, binding = 9)  uniform sampler2D u_lr_color;
layout(set = 0, binding = 10) uniform sampler2D u_lr_motion;
layout(set = 0, binding = 11) uniform sampler2D u_lr_depth;
// The temporal upscaler's two kept pictures of the terrain (terrain_rm_resolve.frag): larger than
// this pass's target, written in turn - rgb colour, a distance. pc.tex_params.x = they are
// there, .y = this frame's is the second one (binding 37), .z = last frame's may be used.
layout(set = 0, binding = 36) uniform sampler2D u_lrb_color;
layout(set = 0, binding = 37) uniform sampler2D u_lrb_depth;
vec3  lrC(vec2 uv)   { return textureLod(u_lr_color, uv, 0.0).rgb; }
float lrD(vec2 uv)   { return textureLod(u_lr_depth, uv, 0.0).r; }
vec4  lrCg0(vec2 uv) { return textureGather(u_lr_color, uv, 0); }
vec4  lrCg1(vec2 uv) { return textureGather(u_lr_color, uv, 1); }
vec4  lrCg2(vec2 uv) { return textureGather(u_lr_color, uv, 2); }
vec4  lrDg(vec2 uv)  { return textureGather(u_lr_depth, uv); }

// CAS-style adaptive sharpen — same idea, takes 4 corner neighbours
// (from textureGather) instead of cardinals.
vec3 casSharpen(vec3 c, vec3 a, vec3 b, vec3 d, vec3 e, float strength) {
    vec3 mn = min(min(min(a, b), min(d, e)), c);
    vec3 mx = max(max(max(a, b), max(d, e)), c);
    vec3 ratio = (min(mn, 1.0 - mx)) / max(mx, vec3(0.001));
    float amount = strength * 0.5 * clamp(min(ratio.r, min(ratio.g, ratio.b)), 0.0, 1.0);
    vec3 blur = 0.25 * (a + b + d + e);
    return c + amount * (c - blur);
}

// Catmull-Rom upsample in 9 bilinear taps (the 4x4 footprint folded into
// 3x3 fetches). Continuous across low-res texel borders and it keeps
// edges crisp, where the old "average of the bilinear tap and the four
// gathered corners" changed in texel-sized steps: distant mountains came
// out as a grid of blocks.
vec3 lrCatmullRom(vec2 uv) {
    vec2 ts  = vec2(textureSize(u_lr_color, 0));
    vec2 sp  = uv * ts;
    vec2 tc1 = floor(sp - 0.5) + 0.5;
    vec2 f   = sp - tc1;
    vec2 w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
    vec2 w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
    vec2 w2 = f * (0.5 + f * (2.0 - 1.5 * f));
    vec2 w3 = f * f * (-0.5 + 0.5 * f);
    vec2 w12 = w1 + w2;
    vec2 t0  = (tc1 - 1.0) / ts;
    vec2 t3  = (tc1 + 2.0) / ts;
    vec2 t12 = (tc1 + w2 / w12) / ts;
    vec3 r = vec3(0.0);
    r += lrC(vec2(t0.x,  t0.y)) * (w0.x  * w0.y);
    r += lrC(vec2(t12.x, t0.y)) * (w12.x * w0.y);
    r += lrC(vec2(t3.x,  t0.y)) * (w3.x  * w0.y);
    r += lrC(vec2(t0.x,  t12.y)) * (w0.x  * w12.y);
    r += lrC(vec2(t12.x, t12.y)) * (w12.x * w12.y);
    r += lrC(vec2(t3.x,  t12.y)) * (w3.x  * w12.y);
    r += lrC(vec2(t0.x,  t3.y)) * (w0.x  * w3.y);
    r += lrC(vec2(t12.x, t3.y)) * (w12.x * w3.y);
    r += lrC(vec2(t3.x,  t3.y)) * (w3.x  * w3.y);
    return max(r, vec3(0.0));
}

vec3  g_c = vec3(0.0);            // the colour, from this frame's low-resolution picture alone
float g_depth = 1.0;              // its depth (as the depth buffer has it)
vec3  g_edge_motion = vec3(0.0);  // xy: the motion at an outline, z: 1 = this pixel is on one
vec2  g_uv = vec2(0.0), g_uv_true = vec2(0.0);
vec3  g_lo = vec3(0.0), g_hi = vec3(0.0);     // the range of the four low-resolution colours round the pixel
vec2  g_lr_offset = vec2(0.0);    // set before tr_upscale(): the low-resolution picture's pixel at uv shows what this pass's pixel at uv + g_lr_offset shows
vec2  g_inv_target = vec2(0.0);   // set before tr_upscale() by a pass whose target is not the render size: 1 / its size in pixels
bool  g_plain = false;            // set before tr_upscale(): the plain (bilinear) enlargement will do - no Catmull-Rom, no sharpening

// false: sky (nothing of the terrain at this pixel).
bool tr_upscale() {

    // Map full-res fragment to a UV in [0, 1] and sample the LR
    // textures. The LR sampler is LINEAR, so the centre tap is a
    // free bilinear upscale — the sky-sentinel discard below
    // avoids dragging sky colour into terrain edges.
    // (this frame's picture is offset: its pixel at uv shows the scene at uv + pc.emissive.xy)
    vec2 uv_true = gl_FragCoord.xy * (g_inv_target.x > 0.0 ? g_inv_target : scene.viewport.zw);
    vec2 uv = uv_true - g_lr_offset;
    float lr_depth = lrD(uv);
    g_uv = uv; g_uv_true = uv_true;

    // Sky pixels: raymarch leaves depth = 1.0 on miss. Discard preserves
    // the existing sky clear in scene_color and the cube prepass depth.
    if (lr_depth >= 0.99999) return false;

    // Depth-aware bilateral 4-tap upsample, using textureGather: one
    // instruction returns the 4 LR-depth corners surrounding `uv`,
    // replacing the 4 separate `texture()` calls of the previous
    // implementation. Three more gathers fetch the same corners' RGB.
    // Total texture ops: 1 (centre depth) + 1 (depth gather) + 3 (RGB
    // gathers) + 1 (centre color) + 1 (motion) = 7, down from 11.
    //
    // Bilateral weight kernel: 1/(1 + α·d²) — a rational kernel with
    // visually identical edge-preserving falloff to the previous
    // exp(-k·|d|) at a fraction of the ALU. α=64 gives the same kink
    // depth as the old kSharp=8.
    vec3 edge_motion = vec3(0.0);
    vec4 d4 = lrDg(uv);                   // .x TL, .y TR, .z BR, .w BL
    vec4 r4 = lrCg0(uv);
    vec4 g4 = lrCg1(uv);
    vec4 b4 = lrCg2(uv);
    vec3 c0  = lrC(uv);
    vec3 cTL = vec3(r4.x, g4.x, b4.x);
    vec3 cTR = vec3(r4.y, g4.y, b4.y);
    vec3 cBR = vec3(r4.z, g4.z, b4.z);
    vec3 cBL = vec3(r4.w, g4.w, b4.w);

    // Smooth interior: Catmull-Rom, then an optional contrast-adaptive
    // sharpen whose neighbours are bilinear taps one low-res texel away
    // (continuous, unlike the gathered corners).
    vec3 c = g_plain ? c0 : lrCatmullRom(uv);
    // Anti-ringing: clamp to the 2x2 bilinear neighbourhood. The 4x4
    // Catmull-Rom footprint reaches sky texels two texels out (which the
    // 2x2 depth-edge test below doesn't see); its negative outer lobes
    // then overshot into a bright / dark halo along ridges against the sky.
    c = clamp(c, min(min(cTL, cTR), min(cBR, cBL)), max(max(cTL, cTR), max(cBR, cBL)));
    float strength = pc.color.x;
    if (!g_plain && strength > 0.0 && pc.color.y > 0.5 && pc.color.z > 0.5) {
        vec2 lt = 1.0 / vec2(textureSize(u_lr_color, 0));
        vec3 nA = lrC(uv + vec2(-lt.x, 0.0));
        vec3 nB = lrC(uv + vec2( lt.x, 0.0));
        vec3 nD = lrC(uv + vec2(0.0, -lt.y));
        vec3 nE = lrC(uv + vec2(0.0,  lt.y));
        c = max(casSharpen(c, nA, nB, nD, nE, strength), vec3(0.0));
    }

    // Depth edges. Where the four low-res texels straddle a silhouette
    // (a near bump in front of far ground), the bilinear depth above is a
    // mix of the two: just inside the outline it lands far BEHIND the near
    // surface, so rocks, trunks and bushes standing behind the bump were
    // drawn over it in a fringe one low-res texel wide. Instead pick the
    // side that owns the pixel (bilinear weight >= 1/2, the same outline
    // the blend would give) and take depth AND colour from that side only.
    {
        vec4 z4 = 0.05 / max(vec4(1.0000016667) - d4, vec4(1e-7));   // view distance (near 0.05, far 1500)
        float zmin = min(min(z4.x, z4.y), min(z4.z, z4.w));
        float zmax = max(max(z4.x, z4.y), max(z4.z, z4.w));
        // A depth step, and not just a plane seen at a grazing angle: NDC
        // depth is linear across the screen on a plane, so its second
        // difference is ~0 there (flat ground far away also spans > 20 %
        // in depth per texel and used to take this path needlessly).
        vec2  lt_d = 1.0 / vec2(textureSize(u_lr_depth, 0));
        float dL = lrD(uv - vec2(lt_d.x, 0.0));
        float dR = lrD(uv + vec2(lt_d.x, 0.0));
        float dU = lrD(uv - vec2(0.0, lt_d.y));
        float dD = lrD(uv + vec2(0.0, lt_d.y));
        float curv = max(abs(dL + dR - 2.0 * lr_depth), abs(dU + dD - 2.0 * lr_depth));
        bool  stepped = curv > 0.12 * (1.0000016667 - lr_depth);
        if (stepped && zmax > zmin * 1.2 + 0.05) {
            vec2 fr = fract(uv * vec2(textureSize(u_lr_depth, 0)) - 0.5);
            // textureGather order: .x (0,1)  .y (1,1)  .z (1,0)  .w (0,0)
            vec4 bw = vec4((1.0 - fr.x) * fr.y, fr.x * fr.y, fr.x * (1.0 - fr.y), (1.0 - fr.x) * (1.0 - fr.y));
            vec4 nearm = vec4(lessThan(z4, vec4(zmin * 1.2 + 0.05)));
            vec4 skym  = vec4(greaterThanEqual(d4, vec4(0.99999)));
            vec4 farm  = (1.0 - nearm) * (1.0 - skym);
            float wn = dot(bw, nearm), ws = dot(bw, skym), wf = dot(bw, farm);
            vec4 pick;
            if (wn >= 0.5)      pick = bw * nearm;
            else if (ws > wf)   return false;
            else                pick = bw * farm;
            float ps = pick.x + pick.y + pick.z + pick.w;
            if (ps < 1e-5) return false;
            pick /= ps;
            c = cTL * pick.x + cTR * pick.y + cBR * pick.z + cBL * pick.w;
            lr_depth = dot(pick, d4);
            // Motion from the same side (a bilinear mix dragged in the sky's
            // zero motion / the far side's).
            vec4 mx4 = textureGather(u_lr_motion, uv, 0), my4 = textureGather(u_lr_motion, uv, 1);
            edge_motion = vec3(dot(pick, mx4), dot(pick, my4), 1.0);
        }
    }
    g_c = c; g_depth = lr_depth; g_edge_motion = edge_motion;
    g_lo = min(min(cTL, cTR), min(cBR, cBL)); g_hi = max(max(cTL, cTR), max(cBR, cBL));
    return true;
}
