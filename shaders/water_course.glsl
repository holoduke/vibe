// water_course.glsl - water above the sea: the stream (CPU twin: src/engine/water_course.cpp -
// keep the two in step). Its points come from the scene block (scene_ubo.glsl stream_*):
// x, z, the water surface, the channel's half width (negative: a STEP ends at this point).
// Along a segment the surface runs evenly, or - for a step - lies level (the pool) and drops at
// the lower end, its lip wandering up and down the stream across the width (wc_n1 of the
// distance across). Included by terrain_rm_noise_road.glsl (after scene_ubo.glsl) and cube.frag.

struct WcAt {
    bool  on;       // within the carve's reach (40 m)
    float dist;     // m from the centre line
    float surf;     // the water surface here
    float hw;       // the channel's half width
    float depth;    // its depth under the surface
    float slope;    // the surface's fall per metre downstream
    float f;        // 0 at the spring .. 1 at the mouth (by points)
    vec2  dir;      // the way it flows
    float lat;      // m across from the centre line (left of the flow > 0)
    float seg;      // segment + how far along it
    float foot;     // 0..1: in the boiling pool at a step's foot
    float drop;     // the height of the step this segment ends in (0: a run)
    float sslope;   // the run's slope smoothed across the points (for shading: no line where a pool meets a sheet)
};

// 1D value noise from an integer hash (water_course.cpp n1: the same bits).
float wc_h1(int i) {
    uint v = uint(i) * 0x9E3779B1u;
    v ^= v >> 15; v *= 0x85EBCA6Bu; v ^= v >> 13;
    return float(v >> 8) * (1.0 / 16777216.0);
}
float wc_n1(float x) {
    float fl = floor(x), f = x - fl;
    int i = int(fl);
    float u = f * f * (3.0 - 2.0 * f);
    return mix(wc_h1(i), wc_h1(i + 1), u);
}

// The channel's depth at point k (water_course.cpp depth_pt): deep in pools and runs, a thin
// sheet (0.22 m) where the water runs down steep ground.
float wc_seg_slope(int j, int n) {
    if (j < 0 || j >= n - 1) return 0.0;
    if (scene.stream_pts[j + 1].w < 0.0) return 0.0;
    return (scene.stream_pts[j].z - scene.stream_pts[j + 1].z) / max(distance(scene.stream_pts[j].xy, scene.stream_pts[j + 1].xy), 1e-3);
}
float wc_depth_pt(int k, int n) {
    float sheet = smoothstep(0.06, 0.12, max(wc_seg_slope(k - 1, n), wc_seg_slope(k, n)));
    return mix(mix(scene.stream_info.y, scene.stream_info.z, float(k) / float(max(n - 1, 1))), 0.22, sheet);
}

WcAt wc_stream_at(vec2 wp) {
    WcAt r;
    r.on = false; r.dist = 1e9; r.surf = 0.0; r.hw = 0.0; r.depth = 0.0; r.slope = 0.0; r.f = 0.0; r.dir = vec2(1.0, 0.0);
    r.lat = 0.0; r.seg = 0.0; r.foot = 0.0; r.drop = 0.0; r.sslope = 0.0;
    int n = int(scene.stream_info.x + 0.5);
    if (n < 2) return r;
    float best = 1e18, bu = 0.0;
    int bk = -1;
    for (int b = 0; b < 8; ++b) {
        vec4 B = scene.stream_box[b];
        if (wp.x < B.x || wp.y < B.y || wp.x > B.z || wp.y > B.w) continue;
        int k1 = min((b + 1) * 8, n - 1);
        for (int k = b * 8; k < k1; ++k) {
            vec2 a = scene.stream_pts[k].xy, ab = scene.stream_pts[k + 1].xy - a;
            float u = clamp(dot(wp - a, ab) / max(dot(ab, ab), 1e-6), 0.0, 1.0);
            vec2 dv = wp - (a + ab * u);
            float d2 = dot(dv, dv);
            if (d2 < best) { best = d2; bk = k; bu = u; }
        }
    }
    if (bk < 0 || best > 1600.0) return r;
    vec4 P0 = scene.stream_pts[bk], P1 = scene.stream_pts[bk + 1];
    float L = distance(P0.xy, P1.xy);
    r.dir = (P1.xy - P0.xy) / max(L, 1e-4);
    r.lat = dot(wp - P0.xy, vec2(-r.dir.y, r.dir.x));
    float drop = P0.z - P1.z;
    if (P1.w < 0.0 && drop > 0.05) {
        float fl = min(clamp(drop * 0.5, 0.4, 2.0), 0.6 * L);
        float shift = 0.9 * (wc_n1(r.lat * 0.9 + float(bk) * 3.17) - 0.5) * 2.0;
        float a = max(L - fl - 0.9 + shift, 0.5);
        float t = clamp((bu * L - a) / fl, 0.0, 1.0);
        r.slope = drop * 6.0 * t * (1.0 - t) / fl;
        r.surf = P0.z - drop * t * t * (3.0 - 2.0 * t);
        r.drop = drop;
    } else {
        r.slope = drop / max(L, 1e-3);
        r.surf = P0.z - drop * bu;
    }
    if (bk > 0 && P0.w < 0.0) r.foot = exp(-bu * L / 4.0) * smoothstep(0.0, 1.2, bu * L) * min((scene.stream_pts[bk - 1].z - P0.z) / 1.2, 1.0);
    r.sslope = mix(max(wc_seg_slope(bk - 1, n), wc_seg_slope(bk, n)), max(wc_seg_slope(bk, n), wc_seg_slope(bk + 1, n)), bu);
    r.on = true;
    r.dist = sqrt(best);
    r.hw = mix(abs(P0.w), abs(P1.w), bu);
    r.seg = float(bk) + bu;
    r.f = r.seg / float(max(n - 1, 1));
    r.depth = mix(wc_depth_pt(bk, n), wc_depth_pt(bk + 1, n), bu);
    return r;
}

// The channel's half width at the bed on this side (water_course.cpp bank_hw), and how far the
// water can reach (wet_reach: the bank's foot + the bank up to the surface + a margin).
float wc_bank_hw(WcAt A) { return A.hw * (0.85 + 0.4 * wc_n1(A.seg * 0.7 + (A.lat >= 0.0 ? 37.0 : 0.0))); }
float wc_wet_reach(WcAt A) { return wc_bank_hw(A) + 0.65 * A.depth / 0.55 + 0.35; }

// The channel cut into the ground height h (water_course.cpp carve): its banks wander in and out
// (each side its own way) and rise gently; the bed shallows toward them. The rounding where the
// bank meets the land is never deeper than half the meeting point's height above the water (a
// fixed 4 m one dug the ground under the surface: a slab of water with a cut edge).
float wc_stream_carve(float h, vec2 wp) {
    if (scene.stream_info.x < 1.5) return h;
    WcAt A = wc_stream_at(wp);
    if (!A.on) return h;
    float hw = wc_bank_hw(A);
    float e = max(A.dist - hw, 0.0);
    float bed = A.surf - A.depth * (1.0 - 0.35 * smoothstep(0.5, 1.0, A.dist / max(hw, 0.1)));
    float cap = bed + e * 0.55 + max(e - 2.5, 0.0) * 0.45;
    float kr = clamp(2.0 * (min(cap, h) - A.surf), 0.0, 4.0);
    if (kr < 1e-4) return min(cap, h);
    float t = clamp(0.5 + 0.5 * (cap - h) / kr, 0.0, 1.0);
    return mix(cap, h, t) - kr * t * (1.0 - t);
}

// The stream's water surface at wp when wp lies within its wet reach, else -1e9
// (water_course.cpp stream_surface).
float wc_stream_surface(vec2 wp) {
    if (scene.stream_info.x < 1.5) return -1e9;
    WcAt A = wc_stream_at(wp);
    return (A.on && A.dist < wc_wet_reach(A)) ? A.surf : -1e9;
}
