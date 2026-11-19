#version 460
// Screen-space displacement mapping - the full-screen passes.
//
// A port of "Screen-Space Raymarched Displacement Mapping" by TechArtAlex (an Unreal Engine
// plugin, shared for free use: patreon.com/TechArtAlex/posts/screen-space-171617266), file
// SSDMScreenSpaceDisplacement.usf: both resolvers (the plain ray march and the default,
// "NCS" - after Chen & Chang, "A Prism-Free Method for Silhouette Rendering in Inverse
// Displacement Mapping", Pacific Graphics 2008), the occluder tests, the edge handling, and
// the depth and motion-vector re-projection. Not ported: the plugin's debug visualisations.
//
// The method. ssdm_mesh.* has drawn the displaced meshes as an inflated shell and stored,
// per shell pixel q, the screen-UV offset D(q) to where q's surface point really lies (it is
// sunk into the shell by the height map). Surface at q appears at q + D(q). This pass solves,
// for every output pixel p, the inverse: the source q with q + D(q) = p.
//
//   March:  s(t) = p - t * dir,  dir = the view ray's direction along the surface (screen UV)
//   Height: h(t) = dot(D(s(t)), dir)
//   At the true source h == t; before it h > t (the surface there is displaced past p), after
//   it h < t. The first h < t crossing is the source - self-occlusion included.
//   NCS bends the march: at every step the direction becomes that of the local D, so the ray
//   follows the surface's curvature and leaves the mesh at its real, displaced outline.
//
// Where no source is found the pixel keeps the scene's colour as it was (outside the original
// mesh's outline that is the background: the displaced silhouette). Where one is found the
// pixel takes the scene colour of the ORIGINAL surface under q (q + contraction(q)).
//
// Conventions here: screen UV = ndc * 0.5 + 0.5 (Vulkan, y down); depth 0 near .. 1 far, 1 =
// nothing (the original is reversed-Z with 0 = nothing); metres; up is +y.
//
// Built four ways: SSDM_GBUFFER (what the engine runs: the resolve applied to the whole
// G-buffer - colour, motion vectors, albedo/material, normal - before lighting, as Crimson
// Desert is reported to do it; everything drawn afterwards, and the lighting itself, sees the
// displaced surface), SSDM_DEPTH (the displaced depth, from the resolve map), and the
// plugin's own arrangement on the lit image: SSDM_COLOR (colour + resolve map) and
// SSDM_VELOCITY (motion vectors, from the resolve map).

layout(set = 0, binding = 0) uniform sampler2D u_scene_color;   // the lit scene (linear, bilinear sampler)
layout(set = 0, binding = 1) uniform sampler2D u_disp;          // ssdm_mesh outDisp
layout(set = 0, binding = 2) uniform sampler2D u_ray;           // ssdm_mesh outRay
layout(set = 0, binding = 3) uniform sampler2D u_hit_z;         // ssdm_mesh outHitZ
layout(set = 0, binding = 4) uniform sampler2D u_shell_z;       // the shell's depth buffer
layout(set = 0, binding = 5) uniform sampler2D u_scene_z;       // the scene's depth, before this pass
layout(set = 0, binding = 6) uniform sampler2D u_resolve;       // SSDM_COLOR's second output (depth / velocity variants)
layout(set = 0, binding = 7) uniform sampler2D u_scene_mv;      // the scene's motion vectors, before this pass
layout(set = 0, binding = 8) uniform sampler2D u_gbuffer0;      // SSDM_GBUFFER: albedo + material
layout(set = 0, binding = 9) uniform sampler2D u_gbuffer1;      // SSDM_GBUFFER: normal + roughness + metallic

layout(push_constant) uniform PC {
    mat4 inv_vp;        // clip -> world, of the view-projection the scene and the shell were drawn with
    mat4 clip_to_prev;  // clip -> last frame's clip
    vec4 cam;           // xyz camera position, w resolver (1 ray march, 2 NCS)
    vec4 march;         // x steps (grazing), y min steps (head-on), z binary steps, w start phase jitter 0..1
    vec4 guard;         // x surface height tolerance (m), y max normal deviation (0 off), z occluder bias (m), w max march distance (m, 0 off)
    vec4 misc;          // x edge handling mode, y occluder check on, z secant refinement on, w frame index
    vec4 mv;            // xy motion-vector scale: stored = (ndc_now - ndc_prev) * xy;  zw output size in pixels
} pc;

// (drawn with the vertexless full-screen triangle, taa.vert: the UV comes from the pixel)

#if defined(SSDM_GBUFFER)
// The engine's use (ssdm.cpp): on the G-buffer, mid-frame, before lighting - the outputs are
// the main pass's attachments in their order, then the resolve map.
layout(location = 0) out vec4 outColor;
layout(location = 1) out vec2 outVelocity;
layout(location = 2) out vec4 outG0;
layout(location = 3) out vec4 outG1;
layout(location = 4) out vec4 outResolve;
#elif defined(SSDM_COLOR)
layout(location = 0) out vec4 outColor;
layout(location = 1) out vec4 outResolve;    // xy: where the colour was taken from; zw: the hit's source pixel (-1 = no hit)
#elif defined(SSDM_VELOCITY)
layout(location = 0) out vec2 outVelocity;
#endif

const float kCoherence = 0.01;               // SSDM_HIT_COHERENCE_THRESHOLD
const float kBiasDither = 0.15;              // SSDM_OCCLUDER_BIAS_DITHER_FRACTION
const float kNcsStepScale = 1.0;             // SSDM_NCS_STEP_SCALE

float residual_sq(vec2 pixel, vec2 src, vec2 d) { vec2 r = src + d - pixel; return dot(r, r); }
float alignment01(vec2 v, vec2 ref) {
    float l = length(v);
    return clamp(l > 1e-5 ? 0.5 + 0.5 * dot(v / l, ref) : 0.5, 0.0, 1.0);
}
bool outside01(vec2 v) { return v.x < 0.0 || v.x > 1.0 || v.y < 0.0 || v.y > 1.0; }
vec2 sat(vec2 v) { return clamp(v, vec2(0.0), vec2(1.0)); }

vec3 world_at(vec2 uv, float z) {
    vec4 w = pc.inv_vp * vec4(uv * 2.0 - 1.0, z, 1.0);
    return w.xyz / w.w;
}

// Interleaved gradient noise (Jimenez 2014), stepped by the golden ratio per frame.
float ign(vec2 pix) {
    float n = fract(52.9829189 * fract(dot(pix, vec2(0.06711056, 0.00583715))));
    return fract(n + mod(pc.misc.w, 8.0) * 1.61803398875);
}
float start_phase_jitter(vec2 uv, float step_size) {
    float j = clamp(pc.march.w, 0.0, 1.0);
    if (j <= 0.0) return 0.0;
    vec2 tile = floor(uv * pc.mv.zw / 8.0) * 8.0 + 4.0;       // one value per 8 x 8 pixels
    return j * ign(tile) * step_size;
}
float occluder_bias(float nov, vec2 uv) {
    float base = pc.guard.z / max(nov, 0.0001);
    return max(base * (1.0 + (ign(uv * pc.mv.zw) - 0.5) * 2.0 * kBiasDither), 0.0001);
}

// Is the hit under another mesh's shell? Two pixels of the same displaced surface lie at
// nearly the same height (the original compares world Z, its up axis).
bool cross_surface(vec2 start_uv, float start_z, vec2 hit_uv, float tol) {
    if (tol <= 0.0 || start_z >= 1.0) return false;
    float hz = textureLod(u_shell_z, sat(hit_uv), 0.0).r;
    if (hz >= 1.0) return true;
    return abs(world_at(hit_uv, hz).y - world_at(start_uv, start_z).y) > tol;
}

// The scene's own surface at `uv` is nearer to the camera than the displaced hit seen there.
bool scene_nearer_than_hit(vec2 uv, float hit_z, float nov) {
    float sz = textureLod(u_scene_z, uv, 0.0).r;
    if (sz >= 1.0 || hit_z >= 1.0) return false;
    if (sz >= hit_z) return false;
    float bias = occluder_bias(nov, uv);
    return distance(world_at(uv, sz), pc.cam.xyz) + bias < distance(world_at(uv, hit_z), pc.cam.xyz);
}
// What the scene shows at the sample pixel is not the displaced mesh's original surface.
bool source_depth_mismatch(vec2 sample_uv, float expected_z, float nov) {
    if (expected_z >= 1.0) return false;
    float sz = textureLod(u_scene_z, sample_uv, 0.0).r;
    if (sz >= 1.0) return false;
    vec3 d = world_at(sample_uv, sz) - world_at(sample_uv, expected_z);
    float bias = occluder_bias(nov, sample_uv);
    return dot(d, d) > bias * bias;
}
// Where the scene colour for a hit at `src` is fetched: the original surface under it, when
// that pixel belongs to a displaced mesh too.
bool contracted_sample(vec2 src, out vec2 final_uv, out vec2 unclamped) {
    vec2 c   = textureLod(u_ray, src, 0.0).ba;
    unclamped = src + c;
    vec2 cu  = sat(unclamped);
    float ca = textureLod(u_disp, cu, 0.0).a;
    vec2 cc  = textureLod(u_ray, cu, 0.0).ba;
    bool moved   = dot(c, c) > 1e-10;
    bool is_ssdm = dot(cc, cc) > 1e-10;
    bool use = ca >= 0.5 && (!moved || is_ssdm);
    final_uv = use ? cu : src;
    if (!use) unclamped = src;
    return use;
}
bool occluded_by_scene(vec2 uv, vec2 src, float nov) {
    vec2 fin, unc;
    contracted_sample(src, fin, unc);
    vec2 hz = textureLod(u_hit_z, sat(src), 0.0).rg;
    if (hz.r >= 1.0 && hz.g >= 1.0) return false;
    if (scene_nearer_than_hit(uv, hz.r, nov)) return true;            // something stands in front of the hit
    return source_depth_mismatch(fin, hz.g, nov);                     // the source pixel shows something else
}
bool pixel_is_ssdm(vec2 uv) {
    if (textureLod(u_disp, uv, 0.0).a < 0.5) return false;
    float shz = textureLod(u_shell_z, uv, 0.0).r;
    if (shz >= 1.0) return false;
    float sz = textureLod(u_scene_z, uv, 0.0).r;
    if (sz >= 1.0) return true;
    return !(distance(world_at(uv, sz), pc.cam.xyz) < distance(world_at(uv, shz), pc.cam.xyz));
}
// The pixel shows something in front of the shell that is not the displaced mesh: no march.
bool skip_for_pixel(vec2 uv, float nov) {
    if (pc.misc.y < 0.5) return false;
    if (pixel_is_ssdm(uv)) return false;
    return source_depth_mismatch(uv, textureLod(u_hit_z, uv, 0.0).g, nov);
}

// A found source: the edge and occluder tests, then where to take the colour from.
// Returns false when the hit must be given up (the pixel keeps its own colour).
bool accept_hit(vec2 uv, vec2 src, vec2 src_unclamped, float nov, out vec2 final_uv) {
    final_uv = uv;
    int edge = int(pc.misc.x + 0.5);
    if ((edge == 1 || edge == 2) && outside01(src_unclamped)) return false;
    if (pc.misc.y > 0.5 && occluded_by_scene(uv, src, nov)) return false;
    vec2 fin, unc;
    bool used = contracted_sample(src, fin, unc);
    bool valid = used ? !outside01(unc) : !outside01(src);
    if ((edge == 1 || edge == 2) && !valid) return false;
    final_uv = fin;
    return true;
}

// ---------------------------------------------------------------------------------------
// The plain ray march (the original's SSDMResolveRayMarchResult).
// ---------------------------------------------------------------------------------------
bool resolve_raymarch(vec2 uv, out vec2 final_uv, out vec2 hit_uv) {
    final_uv = uv; hit_uv = uv;
    vec4 ds = textureLod(u_disp, uv, 0.0);
    if (ds.a < 0.5) return false;
    float start_z = textureLod(u_shell_z, uv, 0.0).r;
    float nov = ds.b;
    if (skip_for_pixel(uv, nov)) return false;
    if (pc.guard.w > 0.0 && start_z < 1.0 && distance(world_at(uv, start_z), pc.cam.xyz) > pc.guard.w) return false;
    vec2 d_start = ds.rg;
    if (dot(d_start, d_start) < 1e-10) return false;
    vec2 rd = textureLod(u_ray, uv, 0.0).rg;
    float rdl = length(rd);
    if (rdl < 1e-5) return false;
    vec2  dir = rd / rdl;
    float len = min(rdl, 1.0);
    float grazing = 1.0 - ds.b;
    int   total = max(1, int(round(mix(pc.march.y, pc.march.x, grazing))));
    float step_size = (len + len / float(total)) / float(total);     // one step past the first-order source
    float jit = start_phase_jitter(uv, step_size);
    float accept_r2 = step_size * step_size * 0.25;
    float h_prev = dot(d_start, dir), t_prev = 0.0;
    vec2  best = uv, best_unc = uv;
    float best_r2 = 1e30;
    bool  has_best = false;
    float min_align = alignment01(d_start, dir);
    float max_dev = 0.0;
    float dsl = length(d_start);
    vec2  d_prev_dir = dsl > 1e-5 ? d_start / dsl : dir;
    int   nbin = int(pc.march.z + 0.5);

    for (int i = 1; i <= total; ++i) {
        bool  false_cross = false;
        float t = jit + float(i - 1) * step_size;
        vec2  s_unc = uv - t * dir;
        vec2  s = sat(s_unc);
        vec4  hs = textureLod(u_disp, s, 0.0);
        vec2  d_here = hs.rg;
        float h_here = dot(d_here, dir);
        float r2 = residual_sq(uv, s, d_here);
        float al = alignment01(d_here, dir);
        // Background: no surface. Keep the bracket's start moving, the ray stays "high".
        if (hs.a < 0.5) { min_align = 0.0; t_prev = t; continue; }
        min_align = min(min_align, al);
        {
            float l = length(d_here);
            if (l > 1e-5) { vec2 dd = d_here / l; max_dev = max(max_dev, 1.0 - dot(dd, d_prev_dir)); d_prev_dir = dd; }
        }
        if (al >= kCoherence && r2 < best_r2) { best_r2 = r2; best = s; best_unc = s_unc; has_best = true; }
        // Solving q + D(q) = p directly: near head-on the projected crossing is numerically weak.
        if (has_best && min_align >= kCoherence && r2 <= accept_r2) {
            if (pc.guard.y > 0.0 && max_dev > pc.guard.y) false_cross = true;
            else {
                if (!accept_hit(uv, best, best_unc, nov, final_uv)) { final_uv = uv; return false; }
                hit_uv = best;
                return true;
            }
        }
        // (h > 0: a mesh without displacement has h = 0, below the line at once)
        if (h_here < t && h_here > 0.0) {
            if (cross_surface(uv, start_z, s, pc.guard.x)) { t_prev = t; continue; }
            float t_lo = t_prev, h_lo = h_prev, t_hi = t, h_hi = h_here;
            for (int j = 0; j < nbin; ++j) {
                float g_lo = h_lo - t_lo, g_hi = h_hi - t_hi, den = g_lo - g_hi;
                float lt = (pc.misc.z > 0.5 && abs(den) > 1e-6) ? clamp(g_lo / den, 0.0, 1.0) : 0.5;
                float t_mid = mix(t_lo, t_hi, lt);
                vec2  s_mid = sat(uv - t_mid * dir);
                vec4  ms = textureLod(u_disp, s_mid, 0.0);
                float h_mid = dot(ms.rg, dir);
                if (ms.a < 0.5 || alignment01(ms.rg, dir) < kCoherence || h_mid < t_mid) { t_hi = t_mid; h_hi = h_mid; }
                else { t_lo = t_mid; h_lo = h_mid; }
            }
            float g_lo = h_lo - t_lo, g_hi = h_hi - t_hi, den = g_lo - g_hi;
            float t_x = mix(t_lo, t_hi, abs(den) > 1e-6 ? clamp(g_lo / den, 0.0, 1.0) : 0.5);
            vec2  src_unc = uv - t_x * dir;
            int   edge = int(pc.misc.x + 0.5);
            if ((edge == 1 || edge == 2) && outside01(src_unc)) return false;
            vec2  src = sat(src_unc);
            vec4  ss = textureLod(u_disp, src, 0.0);
            if (ss.a < 0.5) return false;                                    // refined into the background
            if (!(alignment01(ss.rg, dir) >= kCoherence && min_align >= kCoherence)) return false;
            if (residual_sq(uv, src, ss.rg) <= t_x * t_x * 0.04) {
                if (!(pc.guard.y > 0.0 && max_dev > pc.guard.y)) {
                    if (!accept_hit(uv, src, src_unc, nov, final_uv)) { final_uv = uv; return false; }
                    hit_uv = src;
                    return true;
                }
            }
            false_cross = true;     // a valley of the texture at this step, not the source: march on
        }
        if (!false_cross) { h_prev = h_here; t_prev = t; }
    }
    return false;       // no crossing: silhouette, or hidden
}

// ---------------------------------------------------------------------------------------
// NCS: the march bends with the surface (the original's SSDMResolveNCSResult - its default).
// ---------------------------------------------------------------------------------------
bool resolve_ncs(vec2 uv, out vec2 final_uv, out vec2 hit_uv) {
    final_uv = uv; hit_uv = uv;
    vec4 ds = textureLod(u_disp, uv, 0.0);
    if (ds.a < 0.5) return false;
    float start_z = textureLod(u_shell_z, uv, 0.0).r;
    float nov = ds.b;
    if (skip_for_pixel(uv, nov)) return false;
    if (pc.guard.w > 0.0 && start_z < 1.0 && distance(world_at(uv, start_z), pc.cam.xyz) > pc.guard.w) return false;
    if (dot(ds.rg, ds.rg) < 1e-10) return false;
    vec2 rd = textureLod(u_ray, uv, 0.0).rg;
    float rdl = length(rd);
    if (rdl < 1e-5) return false;
    vec2  dir0 = rd / rdl;                                               // the fixed reference direction
    float len = min(rdl, 1.0);
    float grazing = 1.0 - ds.b;
    int   total = max(1, int(round(mix(pc.march.y, pc.march.x, grazing))));
    float step_size = (len + len / float(total)) / float(total);
    float jit = start_phase_jitter(uv, step_size);
    float accept_r2 = step_size * step_size * 0.25;
    vec2  s_start_unc = uv - jit * dir0;
    vec2  s_start = sat(s_start_unc);
    vec2  d_sp = textureLod(u_disp, s_start, 0.0).rg;

    vec2  s_cur = s_start, s_prev = s_start, s_cur_unc = s_start_unc, s_prev_unc = s_start_unc;
    vec2  dir_cur = dir0;
    float t_cur = dot(uv - s_start, dir0);
    float h_prev = dot(d_sp, dir0);
    vec2  best = uv, best_unc = uv;
    float best_r2 = 1e30;
    bool  has_best = false;
    float min_align = alignment01(d_sp, dir0);
    float max_dev = 0.0;
    float dspl = length(d_sp);
    vec2  d_prev_dir = dspl > 1e-5 ? d_sp / dspl : dir0;
    int   nbin = int(pc.march.z + 0.5);
    int   edge = int(pc.misc.x + 0.5);

    for (int i = 1; i <= total; ++i) {
        bool false_cross = false;
        // The march keeps its spacing along dir0. A bent ray that no longer advances along it
        // has turned outward at the silhouette.
        float proj = dot(dir_cur, dir0);
        min_align = min(min_align, clamp(0.5 + 0.5 * proj, 0.0, 1.0));
        if (proj <= 1e-4) return false;
        vec2  s_unc = s_cur_unc - (kNcsStepScale * step_size / proj) * dir_cur;
        vec2  s = sat(s_unc);
        float t_here = dot(uv - s, dir0);
        if (t_here <= t_cur + 1e-5) return false;                       // (stopped at the screen's edge)
        vec4  hs = textureLod(u_disp, s, 0.0);
        vec2  d_here = hs.rg;
        float h_here = dot(d_here, dir0);
        float r2 = residual_sq(uv, s, d_here);
        float al = alignment01(d_here, dir0);
        min_align = min(min_align, al);
        // The NCS silhouette condition: the bent ray walked off the mesh, or the displacement
        // curved past the horizon.
        if (hs.a < 0.5) return false;
        if (h_here < 0.0) return false;
        {
            float l = length(d_here);
            if (l > 1e-5) { vec2 dd = d_here / l; max_dev = max(max_dev, 1.0 - dot(dd, d_prev_dir)); d_prev_dir = dd; }
        }
        if (al >= kCoherence && r2 < best_r2) { best_r2 = r2; best = s; best_unc = s_unc; has_best = true; }
        if (has_best && min_align >= kCoherence && r2 <= accept_r2) {
            if (cross_surface(uv, start_z, best, pc.guard.x)) false_cross = true;
            else if (pc.guard.y > 0.0 && max_dev > pc.guard.y) false_cross = true;
            else {
                if (!accept_hit(uv, best, best_unc, nov, final_uv)) { final_uv = uv; return false; }
                hit_uv = best;
                return true;
            }
        }
        if (h_here < t_here && h_here > 0.0) {
            if (cross_surface(uv, start_z, s, pc.guard.x)) false_cross = true;
            else {
                vec2  s_lo = s_prev, s_hi = s, s_lo_unc = s_prev_unc, s_hi_unc = s_unc;
                float h_lo = h_prev, h_hi = h_here;
                float t_lo = dot(uv - s_prev, dir0), t_hi = t_here;
                for (int j = 0; j < nbin; ++j) {
                    float g_lo = h_lo - t_lo, g_hi = h_hi - t_hi, den = g_lo - g_hi;
                    float lt = (pc.misc.z > 0.5 && abs(den) > 1e-6) ? clamp(g_lo / den, 0.0, 1.0) : 0.5;
                    vec2  s_mid_unc = mix(s_lo_unc, s_hi_unc, lt);
                    vec2  s_mid = sat(s_mid_unc);
                    vec4  ms = textureLod(u_disp, s_mid, 0.0);
                    float h_mid = dot(ms.rg, dir0), t_mid = dot(uv - s_mid, dir0);
                    if (ms.a < 0.5 || alignment01(ms.rg, dir0) < kCoherence || h_mid < t_mid) {
                        s_hi = s_mid; s_hi_unc = s_mid_unc; h_hi = h_mid; t_hi = t_mid;
                    } else {
                        s_lo = s_mid; s_lo_unc = s_mid_unc; h_lo = h_mid; t_lo = t_mid;
                    }
                }
                float den = (h_lo - t_lo) - (h_hi - t_hi);
                float lt  = abs(den) > 1e-6 ? clamp((h_lo - t_lo) / den, 0.0, 1.0) : 0.5;
                vec2  src_unc = mix(s_lo_unc, s_hi_unc, lt);
                if ((edge == 1 || edge == 2) && outside01(src_unc)) return false;
                vec2  src = sat(src_unc);
                vec4  ss = textureLod(u_disp, src, 0.0);
                if (ss.a < 0.5) return false;
                float t_x = dot(uv - src, dir0);
                bool  r_big = residual_sq(uv, src, ss.rg) > t_x * t_x * 0.04;
                bool  dev_big = pc.guard.y > 0.0 && max_dev > pc.guard.y;
                if (r_big || dev_big) false_cross = true;
                else if (cross_surface(uv, start_z, src, pc.guard.x)) false_cross = true;
                else {
                    if (!accept_hit(uv, src, src_unc, nov, final_uv)) { final_uv = uv; return false; }
                    hit_uv = src;
                    return true;
                }
            }
        }
        // Bend the ray: the march direction becomes that of the local displacement.
        float l = length(d_here);
        if (l > 1e-5) {
            vec2 nd = d_here / l;
            // (a direction that turns sharply against the bent ray: a surface facing the other way)
            if (dot(nd, dir_cur) < 0.99) return false;
            dir_cur = nd;
        }
        s_prev = s_cur; s_prev_unc = s_cur_unc;
        s_cur = s; s_cur_unc = s_unc;
        t_cur = t_here;
        if (!false_cross) h_prev = h_here;
    }
    return false;
}

void resolve_from_map(vec2 uv, out vec2 final_uv, out bool hit, out vec2 hit_uv) {
    vec4 r = textureLod(u_resolve, uv, 0.0);
    final_uv = sat(r.xy);
    hit = r.z >= 0.0;
    hit_uv = hit ? sat(r.zw) : uv;
}

vec2 camera_mv(vec2 uv, float z) {                 // the motion vector of a static point seen at (uv, z)
    vec4 now = vec4(uv * 2.0 - 1.0, z, 1.0);
    vec4 prev = pc.clip_to_prev * now;
    return (now.xy - prev.xy / prev.w) * pc.mv.xy;
}

// The motion vector of a displaced pixel: the displaced surface is seen AT this pixel, at the
// hit's displaced depth - its camera motion is that of this screen ray at that depth. A moving
// object's own motion (the source pixel's vector less the camera's share there) is added.
vec2 displaced_mv(vec2 uv, vec2 final_uv, vec2 hit_uv, vec2 mv_here) {
    vec2  hz = textureLod(u_hit_z, hit_uv, 0.0).rg;
    float shz = textureLod(u_shell_z, uv, 0.0).r;
    float z_here = hz.r < 1.0 ? hz.r : (shz < 1.0 ? shz : hz.g);
    if (z_here >= 1.0) return mv_here;
    vec2  cam_here = camera_mv(uv, z_here);
    vec2  mv_src   = textureLod(u_scene_mv, final_uv, 0.0).rg;
    float z_src    = textureLod(u_hit_z, final_uv, 0.0).g;
    if (z_src >= 1.0) z_src = textureLod(u_scene_z, final_uv, 0.0).r;
    vec2  object_mv = z_src < 1.0 ? mv_src - camera_mv(final_uv, z_src) : vec2(0.0);
    if (dot(object_mv, object_mv) < 1e-10) object_mv = vec2(0.0);       // (a static mesh: zero up to rounding)
    return cam_here + object_mv;
}

void main() {
    vec2 uv = gl_FragCoord.xy / pc.mv.zw;
#if defined(SSDM_GBUFFER)
    vec2 final_uv, hit_uv;
    bool hit = (pc.cam.w < 1.5) ? resolve_raymarch(uv, final_uv, hit_uv) : resolve_ncs(uv, final_uv, hit_uv);
    if (!hit) final_uv = uv;
    outResolve = vec4(sat(final_uv), hit ? sat(hit_uv) : vec2(-1.0));
    // (G-buffer data is packed: exact texels, never a blend of neighbours)
    ivec2 src = ivec2(clamp(final_uv * pc.mv.zw, vec2(0.0), pc.mv.zw - 1.0));
    outColor = texelFetch(u_scene_color, src, 0);
    outG0    = texelFetch(u_gbuffer0, src, 0); if (pc.misc.w < -0.5) outG0.rgb = hit ? vec3(1.0, 0.0, 0.0) : outG0.rgb * vec3(0.3, 0.3, 1.0);
    outG1    = texelFetch(u_gbuffer1, src, 0);
    vec2 mv_here = textureLod(u_scene_mv, uv, 0.0).rg;
    outVelocity = hit ? displaced_mv(uv, final_uv, hit_uv, mv_here) : mv_here;
#elif defined(SSDM_COLOR)
    vec2 final_uv, hit_uv;
    bool hit = (pc.cam.w < 1.5) ? resolve_raymarch(uv, final_uv, hit_uv) : resolve_ncs(uv, final_uv, hit_uv);
    if (!hit) final_uv = uv;
    outResolve = vec4(sat(final_uv), hit ? sat(hit_uv) : vec2(-1.0));
    outColor = textureLod(u_scene_color, final_uv, 0.0);
#elif defined(SSDM_DEPTH)
    // The displaced surface's depth, where it is nearer than what the scene has there.
    vec2 final_uv, hit_uv; bool hit;
    resolve_from_map(uv, final_uv, hit, hit_uv);
    if (!hit) discard;
    float hz = textureLod(u_hit_z, hit_uv, 0.0).r;
    if (hz >= 1.0) discard;
    float sz = textureLod(u_scene_z, uv, 0.0).r;
    if (sz < 1.0 && hz >= sz) discard;
    gl_FragDepth = hz;
#elif defined(SSDM_VELOCITY)
    vec2 final_uv, hit_uv; bool hit;
    resolve_from_map(uv, final_uv, hit, hit_uv);
    vec2 mv_here = textureLod(u_scene_mv, uv, 0.0).rg;
    if (!hit) { outVelocity = mv_here; return; }
    // The displaced surface is seen AT this pixel, at the hit's displaced depth: its camera
    // motion is that of this screen ray at that depth. A moving object's own motion (the
    // source pixel's vector less the camera's share there) is carried over on top.
    vec2  hz = textureLod(u_hit_z, hit_uv, 0.0).rg;
    float shz = textureLod(u_shell_z, uv, 0.0).r;
    float z_here = hz.r < 1.0 ? hz.r : (shz < 1.0 ? shz : hz.g);
    if (z_here >= 1.0) { outVelocity = mv_here; return; }
    vec2  cam_here = camera_mv(uv, z_here);
    vec2  mv_src   = textureLod(u_scene_mv, final_uv, 0.0).rg;
    float z_src    = textureLod(u_hit_z, final_uv, 0.0).g;
    if (z_src >= 1.0) z_src = textureLod(u_scene_z, final_uv, 0.0).r;
    vec2  object_mv = z_src < 1.0 ? mv_src - camera_mv(final_uv, z_src) : vec2(0.0);
    // (a static mesh: object_mv is zero up to rounding)
    if (dot(object_mv, object_mv) < 1e-12) object_mv = vec2(0.0);
    outVelocity = cam_here + object_mv;
#endif
}
