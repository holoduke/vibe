// water_relief.glsl - the stream's surface in RELIEF (2026-11-27, the owner: "I don't see a lot of
// water bumping, like the effect when water goes over a boulder underwater ... when water is
// squeezed between a narrow place it forms a bubble - much more elevation in the water with
// rounded bulges"). Real height on top of the course's surface (water_course.glsl), used by the
// hit (terrain_raymarch.frag: the stream's crossing is searched on this surface) and by the
// water's shading (its slope tilts the normal: terrain_rm_main_water.glsl). CPU twin:
// vk_engine/water_fx.cpp stream_relief (water_surface_y: swimmers, splashes, bullets) - keep the
// two in step (the solver's part is the GPU's only).
//
// From the stones in the water (scene.river_piers: xy, z = radius, w = the top's height, 1e5 for a
// pier or a rock standing out of the water - water_fx.cpp river_pier_add) and the current (the
// reference's speed rule U = 0.75 + 3 sin(slope)), with the velocity head hv = U^2 / 2g:
//   * a stone UNDER the surface (its top s below it): a smooth rounded hump over it, a little
//     downstream of its top, crest 0.9 hv a / (a + s) (at most 0.45 m); then a dip just behind
//     it and a train of standing waves of the stationary length 2 pi U^2 / g, decaying over ~1.5
//     wavelengths, widening downstream;
//   * a stone STANDING OUT of it: the water heaped against its upstream face (0.8 hv, a pillow)
//     and drawn down in its lee;
//   * where stones (or a stone and the bank) squeeze the flow into a gap narrower than 3/4 of the
//     channel: a rounded pillow across the gap (0.6 hv);
//   * plus, inside the local solver's window, the solver's own surface over the authored top
//     (water_solver.glsl ws_dh, clamped to -0.15 .. 0.3 m).
// None of it on a fall's face; faded toward the water's edge (the bank stays the edge).
// water_wet bit 4096 (VIBE_NO_STREAM_RELIEF=1: off). Needs water_course.glsl and water_solver.glsl.
#ifndef VIBE_WATER_RELIEF_GLSL
#define VIBE_WATER_RELIEF_GLSL

float wr_relief(vec2 p, WcAt A, bool with_solver) {
    if (!SW_ON(4096) || !A.on) return 0.0;
    float sl = max(A.sslope, A.drop > 0.01 ? A.slope : 0.0);      // (continuous across the segments: sslope is blended between them, a face's slope is 0 at its ends)
    float sn = sl / sqrt(1.0 + sl * sl);
    float U = min(0.75 + 3.0 * sn, 4.5);
    float hv = U * U / 19.62;
    float lam = clamp(6.2832 * U * U / 9.81, 0.6, 4.0);
    vec2 fd = A.dir, ac = vec2(-fd.y, fd.x);
    float S0 = A.surf;
    float h = 0.0;
    float gl = 1e9, gr = 1e9;      // the free water to the left and right of p between stones standing out
    int n = int(scene.water_swell.w + 0.5);
    for (int i = 0; i < n; ++i) {
        vec4 P = scene.river_piers[i];
        vec2 d = p - P.xy;
        float a = max(P.z, 0.12);
        float reach = 2.0 * a + 3.0 * lam + 1.0;
        float r2 = dot(d, d);
        if (r2 > reach * reach) continue;
        float down = dot(d, fd), lat = dot(d, ac);
        float r = sqrt(r2);
        float rf = 1.0 - smoothstep(0.75 * reach, reach, r);      // (to nothing at the reach: the surface must stay continuous - the hit's search ends on a jump and takes it for the edge)
        // (under the surface or standing out of it: blended over 10 cm of the top against the surface
        //  at the stone - a hard switch was a jump in the surface)
        float S0c = S0 + sl * down;
        float bsub = smoothstep(-0.03, 0.07, S0c - P.w);
        {
            // under the surface: the hump, the dip, the standing waves
            float s = max(S0c - P.w, 0.0);
            float hc = min(0.9 * hv * a / (a + s), 0.45);
            vec2  dd = d - fd * (0.35 * a);
            float R = 1.4 * a + 0.3;
            float q = dot(dd, dd) / (R * R);
            h += bsub * (q < 1.0 ? hc * (1.0 - q) * (1.0 - q) : 0.0);      // (well inside the reach)
            float xi = down - (1.5 * a + 0.25);
            float wd = 1.1 * a + 0.2 + 0.15 * max(down, 0.0);
            float wl = exp(-lat * lat / (wd * wd));
            h += 0.55 * hc * (-cos(6.2832 * xi / lam)) * exp(-max(xi, 0.0) / (1.5 * lam)) * smoothstep(-0.6 * a - 0.2, 0.0, xi) * wl * rf * bsub;
        }
        {
            // standing out: the pillow upstream, the drawdown in the lee
            float ring = max(r - a, 0.0);
            float inside = smoothstep(0.8 * a, a, r);
            h += 0.8 * hv * exp(-ring / (0.3 + 0.2 * a)) * smoothstep(-0.3 * a, 0.6 * a, -down) * inside * rf * (1.0 - bsub);
            h -= 0.5 * hv * exp(-lat * lat / (a * a)) * exp(-max(down, 0.0) / (1.2 * a + 0.5)) * smoothstep(0.0, 0.4 * a, down) * inside * rf * (1.0 - bsub);
            // (the gap this stone makes on its side of p - pushed away softly as p leaves its reach
            //  along the flow: a min of continuous terms stays continuous)
            float e = max(abs(lat) - a, 0.0) + 0.05 + 20.0 * max(smoothstep(a + 0.3, a + 0.8, abs(down)), bsub);
            if (lat > 0.0) gr = min(gr, e); else gl = min(gl, e);
        }
    }
    // the squeeze: a stone on at least one side, the bank (or another stone) on the other (with no
    // stone near, the gap is the channel's own width: no squeeze)
    {
        float hwb = A.hw;
        gl = min(gl, max(hwb - A.lat, 0.05));
        gr = min(gr, max(hwb + A.lat, 0.05));
        float g = gl + gr;
        float sq = 1.0 - smoothstep(0.35, 0.75, g / max(2.0 * hwb, 0.1));
        h += 0.6 * hv * sq * pow(sin(3.1416 * clamp(gl / g, 0.0, 1.0)), 0.7);
    }
#ifdef WS_TEX
    if (with_solver) h += clamp(ws_dh(p), -0.15, 0.3);
#endif
    // none on a fall's face; toward the water's edge it fades (the bank stays the edge)
    h *= 1.0 - smoothstep(0.3, 0.8, A.slope) * step(0.01, A.drop);
    h *= 1.0 - smoothstep(0.65, 1.0, A.dist / max(wc_wet_reach(A), 0.1));
    // (a dip never bares the bed: at most a third of the channel's depth, less toward the banks where
    //  the bed shallows - VIBE_DEBUG_STREAM_EDGE showed dry ground in the lee of the stones)
    h = max(h, -0.33 * A.depth * (1.0 - smoothstep(0.3, 0.9, A.dist / max(A.hw, 0.1))));
    return clamp(h, -0.3, 0.5);
}

// The stream's surface with its relief at p, -1e9 where there is no stream (wc_stream_surface).
float wr_stream_surface(vec2 p) {
    if (scene.stream_info.x < 1.5) return -1e9;
    WcAt A = wc_stream_at(p);
    if (!A.on) return -1e9;
    float over = A.dist - wc_wet_reach(A);
    if (over >= 1.5) return -1e9;
    return A.surf - max(over, 0.0) + (over < 0.0 ? wr_relief(p, A, true) : 0.0);
}

#endif
