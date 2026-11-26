#version 460
// A pane that has been shot (glass.cpp draws each damaged pane with this; a whole one: glass.frag).
//
// The pane carries up to 7 impacts (pc.hits: where, how far it has fallen out, a seed). Each
// impact is a WEB, worked out here from its seed alone:
//   * a small hole with a crushed white rim,
//   * radial cracks running out from it - each its own length, kinked every few centimetres,
//   * ring cracks: straight breaks between two neighbouring radial cracks,
//   * a crazed, frosted patch round the hole.
// The radial cracks and the rings cut the glass into pieces. As the impact weakens (`fall`
// grows with every hit near it) the pieces next to the hole drop out, sector by sector, ring by
// ring: the hole grows as an irregular polygon along the cracks. Where there is hole there is
// no glass: the fragment is discarded and the scene shows as it is.
// A later web's cracks stop where they meet an earlier web's cracks (they stay inside the
// piece of the earlier web they started in).
// A pane that has burst (misc.x = 2) keeps only jagged teeth along its frame; they drop away
// over a minute.

layout(location = 0) in vec3 vWorld;
layout(location = 1) in vec2 vUv;          // metres from the pane's lower left corner
layout(location = 0) out vec4 outColor;

layout(push_constant) uniform PC {
    mat4 vp;
    vec4 origin_w;      // xyz the pane's lower left corner (world), w its width (m)
    vec4 uaxis_h;       // xyz the direction of u, w the pane's height (m)
    vec4 vaxis_s;       // xyz the direction of v, w the pane's seed
    vec4 tint_n;        // rgb the glass's tint, a the number of impacts
    vec4 misc;          // x: 1 cracked, 2 burst; y: seconds since it burst; z: 1 = toughened glass (small dice)
    vec4 hits[7];       // xy where (pane uv, m), z fall (0 .. 3.2), w seed (0 .. 1)
} pc;

#include "scene_ubo.glsl"

layout(set = 0, binding = 29) uniform sampler2D u_scene_depth;
layout(set = 0, binding = 38) uniform sampler2D u_scene_copy;

#include "glass_common.glsl"

const float TAU = 6.2831853;

float h11(float x) { return fract(sin(x * 127.1) * 43758.5453); }
// Straight between its values at the whole numbers: a line with a kink at each.
float kinked(float x) { float i = floor(x); return mix(h11(i), h11(i + 1.0), x - i); }

struct Web {
    float K;        // radial cracks
    float reach;    // the longest of them (m)
    float rc;       // the hole's radius as shot (m)
    float r1;       // the spacing of the rings (m)
    float s;        // seed
};

Web web_of(vec4 h) {
    Web w;
    w.s = h.w;
    w.K = 8.0 + floor(h11(h.w * 37.0 + 1.3) * 7.0);
    w.reach = mix(0.30, 0.78, h11(h.w * 53.0 + 4.1)) * (1.0 + 0.45 * min(h.z, 3.2));
    w.rc = 0.010 + 0.009 * h11(h.w * 91.0 + 2.7);
    w.r1 = w.reach * mix(0.13, 0.19, h11(h.w * 17.0 + 8.2));
    return w;
}

// Radial crack j of a web: where it is at the distance rho (in sectors: crack j sits near j),
// and how long it is.
float crack_at(Web w, float j, float rho) {
    float jm = mod(j, w.K);
    float jit = (h11(jm * 7.31 + w.s * 113.0) - 0.5) * 0.56;
    float kink = (kinked(rho * 15.0 + jm * 5.7 + w.s * 29.0) - 0.5) * 0.34 * smoothstep(0.0, 0.05, rho);
    return j + jit + kink;
}
float crack_len(Web w, float j) {
    return w.reach * (0.25 + 0.75 * h11(mod(j, w.K) * 3.17 + w.s * 71.0));
}

// Where a point (d: from the web's centre) lies: between which two radial cracks.
struct Sec {
    float rho, a;           // distance; angle in sectors
    float lo, hi;           // the cracks either side
    float alo, ahi;         // where they are at this distance
    float llo, lhi;         // their lengths
};

Sec sector_of(Web w, vec2 d) {
    Sec S;
    S.rho = length(d);
    S.a = (atan(d.y, d.x) / TAU + 0.5 + w.s) * w.K;
    float j0 = floor(S.a);
    S.lo = j0 - 2.0; S.alo = -1e9;
    S.hi = j0 + 3.0; S.ahi = 1e9;
    for (int k = -1; k <= 2; ++k) {
        float j = j0 + float(k);
        float aj = crack_at(w, j, S.rho);
        if (aj <= S.a) { if (aj > S.alo) { S.alo = aj; S.lo = j; } }
        else           { if (aj < S.ahi) { S.ahi = aj; S.hi = j; } }
    }
    S.llo = crack_len(w, S.lo);
    S.lhi = crack_len(w, S.hi);
    return S;
}

vec2 polar(Web w, float ang, float r) {
    float t = (ang / w.K - 0.5 - w.s) * TAU;
    return r * vec2(cos(t), sin(t));
}

// Ring m (1 .. 3) of a sector: a straight break from one radial crack to the next. `there`:
// whether it is a crack at all (the hole's edge follows it either way).
void ring_of(Web w, Sec S, float m, out vec2 A, out vec2 B, out bool there) {
    float sm = mod(S.lo, w.K);
    float r = w.r1 * (m + 0.55 * (h11(sm * 9.7 + m * 3.3 + w.s * 41.0) - 0.5));
    float lim = min(S.llo, S.lhi) * 0.96;
    there = r < lim && h11(sm * 5.1 + m * 7.7 + w.s * 19.0) < 0.88 - 0.13 * m;
    r = min(r, lim);
    A = polar(w, crack_at(w, S.lo, r), r);
    B = polar(w, crack_at(w, S.hi, r), r);
}

float seg_dist(vec2 p, vec2 A, vec2 B) {
    vec2 ab = B - A;
    float t = clamp(dot(p - A, ab) / max(dot(ab, ab), 1e-9), 0.0, 1.0);
    return length(p - A - ab * t);
}
float side_of(vec2 p, vec2 A, vec2 B) { return (B.x - A.x) * (p.y - A.y) - (B.y - A.y) * (p.x - A.x); }
bool beyond(vec2 d, vec2 A, vec2 B) { return side_of(d, A, B) * side_of(vec2(0.0), A, B) < 0.0; }

// How many rings have dropped out of the sector that begins at crack `lo`.
float fallen_at(Web w, float lo, float fall) {
    float sm = mod(lo, w.K);
    return clamp(floor(min(fall, 3.2) * (0.55 + 0.9 * h11(sm * 13.3 + w.s * 7.0))), 0.0, 3.0);
}
float fallen(Web w, Sec S, float fall) { return fallen_at(w, S.lo, fall); }

// How far out the hole reaches in the sector that begins at crack `lo` (near enough: along
// the sector's cracks, where its outermost dropped ring meets them).
float hole_reach(Web w, float lo, float fall) {
    float sm = mod(lo, w.K);
    float nf = fallen_at(w, lo, fall);
    if (nf < 0.5) return w.rc * (0.7 + 0.6 * h11(sm * 2.1 + w.s * 3.0));
    float r = w.r1 * (nf + 0.55 * (h11(sm * 9.7 + nf * 3.3 + w.s * 41.0) - 0.5));
    return min(r, min(crack_len(w, lo), crack_len(w, lo + 1.0)) * 0.96);
}

// The piece of a web a point lies in (a number), or -1 outside the web.
float piece_of(Web w, vec2 d) {
    Sec S = sector_of(w, d);
    if (S.rho > min(S.llo, S.lhi)) return -1.0;
    float band = 0.0;
    for (int m = 1; m <= 3; ++m) {
        vec2 A, B; bool there;
        ring_of(w, S, float(m), A, B, there);
        if (there && beyond(d, A, B)) band = float(m);
    }
    return mod(S.lo, w.K) * 4.0 + band;
}

void main() {
    vec2 uv = gl_FragCoord.xy * scene.viewport.zw;
    vec2 p = vUv;
    float W = pc.origin_w.w, H = pc.uaxis_h.w;
    vec3 T = normalize(pc.uaxis_h.xyz), Bt = normalize(pc.vaxis_s.xyz);
    vec3 N = normalize(cross(T, Bt));
    vec3 V = normalize(scene.camera_pos.xyz - vWorld);
    if (dot(N, V) < 0.0) N = -N;
    float ndv = clamp(dot(N, V), 0.0, 1.0);
    float dist = distance(scene.camera_pos.xyz, vWorld);
    float px = max(max(length(dFdx(p)), length(dFdy(p))), 1e-5);          // a pixel, in metres on the pane
    float seed = pc.vaxis_s.w;

    float line = 0.0;       // a crack
    float core = 0.0;       // its dark middle
    float frost = 0.0;      // crazed glass
    float rim = 0.0;        // the crushed white edge of a hole or of a tooth
    vec2  bend = vec2(0.0); // which way a crack's facet throws the view
    float facet = 0.0;      // a number per stretch of crack (its glint)

    if (pc.misc.x > 1.5) {
        // ---- burst: teeth along the frame ----
        float t = pc.misc.y;
        float dl = p.x, dr = W - p.x, db = p.y, dt = H - p.y;
        float de = min(min(dl, dr), min(db, dt));
        float s = de == dl ? p.y : de == dr ? p.y + 37.0 : de == db ? p.x + 71.0 : p.x + 113.0;
        bool dice = pc.misc.z > 0.5;
        float x = s / (dice ? 0.018 : 0.065) + seed * 100.0;
        float i0 = floor(x);
        float th = 0.0;
        float hh[2];
        for (int k = 0; k < 2; ++k) {
            float i = i0 + float(k);
            float r = h11(i * 1.37 + seed * 9.0);
            float big = h11(floor(i / 6.0) * 2.1 + seed * 3.0);
            float hmax = dice ? 0.028 : mix(0.05, 0.42, big * big);
            float h = max(r - 0.3, 0.0) / 0.7 * hmax;
            if (h11(i * 5.3 + seed * 2.0) < smoothstep(3.0, 60.0, t) * 1.05) h = 0.0;      // (it has dropped)
            hh[k] = h;
        }
        th = mix(hh[0], hh[1], x - i0);
        if (de == db) th *= 0.55;                              // (little stays standing in the sill)
        if (de >= th) discard;
        rim = 1.0 - smoothstep(0.0, 0.004 + px, th - de);
        // the teeth are cracked through: fine lines across them
        float hair = pow(abs(sin(x * 3.1 + de * 90.0 * (0.5 + h11(i0)))), 14.0);
        line = hair * 0.55;
        frost = 0.22 + 0.25 * h11(floor(x * 2.0) + floor(de * 60.0) * 7.0);
        if (dice) frost = 0.55;
        facet = floor(x * 2.0);
        bend = vec2(h11(facet) - 0.5, h11(facet + 3.0) - 0.5);
    } else {
        // ---- cracked: the webs ----
        int n = int(pc.tint_n.a + 0.5);
        float wl = max(0.0011, px * 0.75);                     // half the width a crack is drawn with
        float thin = clamp(0.0011 / wl, 0.0, 1.0);             // (wider than it is: fainter)
        for (int i = 0; i < n && i < 7; ++i) {
            vec4 h = pc.hits[i];
            Web w = web_of(h);
            vec2 d = p - h.xy;
            if (dot(d, d) > w.reach * w.reach) continue;
            Sec S = sector_of(w, d);
            float sm = mod(S.lo, w.K);

            // the hole
            float nf = fallen(w, S, h.z);
            float edge;                                        // distance to the hole's edge, from outside
            if (nf < 0.5) {
                float rh = w.rc * (0.7 + 0.6 * h11(sm * 2.1 + w.s * 3.0));
                if (S.rho < rh) discard;
                edge = S.rho - rh;
            } else {
                vec2 A, B; bool there;
                ring_of(w, S, nf, A, B, there);
                if (!beyond(d, A, B)) discard;
                edge = seg_dist(d, A, B);
            }
            rim = max(rim, 1.0 - smoothstep(0.0, 0.0035 + px, edge));
            // ... and along a radial crack where the piece on its other side is gone
            if (S.rho < hole_reach(w, S.lo - 1.0, h.z))
                rim = max(rim, 1.0 - smoothstep(0.0, 0.003 + px, S.rho * abs(sin((S.a - S.alo) * TAU / w.K))));
            if (S.rho < hole_reach(w, S.hi, h.z))
                rim = max(rim, 1.0 - smoothstep(0.0, 0.003 + px, S.rho * abs(sin((S.ahi - S.a) * TAU / w.K))));

            // its cracks here: the two radial ones of this sector, and the rings
            float li = 0.0, ci = 0.0;
            vec2 perp = S.rho > 1e-5 ? vec2(-d.y, d.x) / S.rho : vec2(0.0);
            float ang_w = TAU / w.K;
            if (S.rho < S.llo) {
                float dd = S.rho * abs(sin((S.a - S.alo) * ang_w));
                float taper = pow(1.0 - S.rho / S.llo, 0.4);
                float l = (1.0 - smoothstep(wl * 0.45, wl, dd)) * taper;
                if (l > li) { li = l; ci = 1.0 - smoothstep(0.0, wl * 0.35, dd); bend = perp; facet = S.lo * 13.0 + floor(S.rho * 22.0) + w.s * 100.0; }
            }
            if (S.rho < S.lhi) {
                float dd = S.rho * abs(sin((S.ahi - S.a) * ang_w));
                float taper = pow(1.0 - S.rho / S.lhi, 0.4);
                float l = (1.0 - smoothstep(wl * 0.45, wl, dd)) * taper;
                if (l > li) { li = l; ci = 1.0 - smoothstep(0.0, wl * 0.35, dd); bend = -perp; facet = S.hi * 13.0 + floor(S.rho * 22.0) + w.s * 100.0; }
            }
            for (int m = 1; m <= 3; ++m) {
                if (float(m) <= nf) continue;                  // (dropped out with its piece)
                vec2 A, B; bool there;
                ring_of(w, S, float(m), A, B, there);
                if (!there) continue;
                float dd = seg_dist(d, A, B);
                float l = (1.0 - smoothstep(wl * 0.45, wl, dd)) * 0.85;
                if (l > li) { li = l; ci = 1.0 - smoothstep(0.0, wl * 0.35, dd); bend = normalize(d + vec2(1e-5)); facet = sm * 7.0 + float(m) * 31.0 + floor(S.a * 3.0) * 3.7 + w.s * 100.0; }
            }
            // an earlier web's cracks stop it
            if (li > 0.0) {
                for (int k = 0; k < i; ++k) {
                    vec4 hk = pc.hits[k];
                    Web wk = web_of(hk);
                    vec2 dk = p - hk.xy;
                    if (dot(dk, dk) > wk.reach * wk.reach) continue;
                    if (piece_of(wk, dk) != piece_of(wk, h.xy - hk.xy)) { li = 0.0; break; }
                }
            }
            if (li > line) { line = li; core = ci; }

            // crazed round the hole: hairlines and a milky bloom
            float fr = 1.0 - smoothstep(w.rc * 1.2, w.rc * (3.6 + 1.6 * h.z), S.rho);
            if (nf >= 0.5) fr = max(fr, 1.0 - smoothstep(0.0, 0.03, edge));
            float hair = pow(abs(sin(S.a * 3.14159 * 5.0 + h11(floor(S.rho * 90.0) + sm) * 6.0)), 8.0);
            frost = max(frost, fr * (0.40 + 0.60 * hair));
        }
        line *= mix(0.45, 1.0, thin);
        core *= thin * thin;
    }

    // ---- the glass itself ----
    vec3 extra = (T * bend.x + Bt * bend.y) * (0.035 * line + 0.012 * frost);
    vec4 behind = glass_through(vWorld, N, V, pc.vp, uv, gl_FragCoord.z, dist, extra);
    vec3 col = behind.rgb * glass_absorb(ndv);
    float fres = glass_fresnel(ndv);
    vec3 R = reflect(-V, N);
    vec3 sky = scene.sky_color.rgb;
    vec3 env = glass_mirror(R) + glass_glint(R, 900.0, 60.0, 24.0, 0.12);
    col = mix(col, env, fres);
    float dust = fract(sin(dot(floor(vUv * 37.0), vec2(12.9898, 78.233))) * 43758.5453);
    col += sky * 0.012 * dust * (0.4 + 0.6 * (1.0 - ndv));

    // ---- the damage on it ----
    vec3 sun = glass_sunlight();
    vec3 sun_dir = normalize(scene.sun_direction.xyz);
    // A crack is a pair of rough faces inside the glass: it scatters the sky, and each stretch
    // of it throws the sun back when it happens to face the right way - or lets it through
    // when one looks toward the sun.
    float ph = h11(facet * 0.731 + 0.17) * TAU;
    vec3 Nf = normalize(N + (T * cos(ph) + Bt * sin(ph)) * 0.75);
    float glint = pow(max(dot(reflect(-V, Nf), sun_dir), 0.0), 40.0) * 3.0
                + pow(max(dot(-V, sun_dir), 0.0), 6.0) * 1.6 * (0.4 + 0.6 * h11(facet + 5.0));
    vec3 pale = sky * 0.62 + scene.ambient.rgb * 0.25 + sun * 0.10;
    float l1 = clamp(line, 0.0, 1.0);
    vec3 crack_col = pale * vec3(0.92, 1.0, 0.97) + sun * (glint * l1 * l1);       // (the glint from the middle of the crack only)
    col = mix(col, crack_col, l1 * 0.92);
    col *= 1.0 - 0.55 * core * line;                           // the dark middle of a crack seen close
    col = mix(col, pale * 1.05 + sun * 0.06, clamp(frost, 0.0, 1.0) * 0.5);
    col = mix(col, pale * 1.35 + sun * (0.20 + 0.5 * glint), clamp(rim, 0.0, 1.0) * 0.9);
    outColor = vec4(col, behind.a);
}
