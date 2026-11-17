// terrain_rm_pebbles.glsl - part of terrain_raymarch.frag (included there, in place; not a standalone shader).
// Procedural pebbles, lake-bed stones, foam cells

// ---- Procedural pebbles -------------------------------------------------
// Loose stones computed by formula (jittered cells, one lens-shaped stone
// per occupied cell), in three sizes: pebbles, gravel, grit. Each size
// fades in once its stones cover a few pixels, so the ground never runs
// out of detail however close the camera gets - there is always a finer
// layer underneath. They are real height on the raymarched ground (the
// ray lands ON the stones near the camera), not a picture.
vec3 peb_h3(vec2 c) {
    vec3 p3 = fract(vec3(c.xyx) * vec3(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, p3.yxz + 33.33);
    return fract((p3.xxy + p3.yzz) * p3.zyx);
}
float g_peb_id = 0.0;    // random 0..1 of the stone at the last query
float g_peb_k  = 0.0;    // 0 at its rim .. 1 at its centre (0 = bare soil)
float g_peb_sz = 0.0;    // its layer: 1 pebble, 2 gravel, 3 grit, 4 big cobble
float peb_layer(vec2 p, float cell, float dens, float layer_id, float hf, inout float best) {
    vec2 q  = p / cell;
    vec2 ip = floor(q), fq = fract(q);
    float h = 0.0;
    // A stone reaches at most 0.9 cell from its cell centre, so only the
    // 2x2 cells on this point's side of its own cell can cover it.
    vec2 gb = step(vec2(0.5), fq) - 1.0;
    for (int j = 0; j <= 1; ++j)
    for (int i = 0; i <= 1; ++i) {
        vec2 g = gb + vec2(float(i), float(j));
        vec3 r3 = peb_h3(ip + g + layer_id * 71.3);
        if (r3.z > dens) continue;
        vec2  o  = g + 0.5 + (r3.xy - 0.5) * 0.55 - fq;
        float rr = mix(0.27, 0.50, fract(r3.z * 17.3));
        if (dot(o, o) > 1.55 * rr * rr) continue;             // cannot reach this point (cheap reject)
        float a  = r3.x * 6.2832;
        vec2  d  = vec2(cos(a) * o.x + sin(a) * o.y, -sin(a) * o.x + cos(a) * o.y);
        d.y *= mix(1.0, 1.75, r3.y);                         // oval stones
        // Shape: from round river pebbles to blocky broken stone.
        vec2  d2 = d * d;
        float bx = fract(r3.z * 29.1) * fract(r3.x * 3.3 + 0.37);
        float k  = 1.0 - mix(d2.x + d2.y, max(d2.x, d2.y) * 1.3, bx) / (rr * rr);
        if (k <= 0.0) continue;
        float hh = hf * cell * rr * mix(0.55, 1.05, fract(r3.x * 7.7)) * k * (1.75 - 0.75 * k);
        if (hh > h) h = hh;
        if (hh > best) { best = hh; g_peb_id = fract(r3.y * 13.7 + r3.x); g_peb_k = k; g_peb_sz = layer_id; }
    }
    return h;
}
// Foam lace: distance between the nearest and second-nearest cell point
// (small on the walls between cells). Foam sits on those walls and the
// cell centres are open water - the net of bubbles real surf leaves.
float foam_cells(vec2 p) {
    vec2 ip = floor(p), fp = fract(p);
    float f1 = 9.0, f2 = 9.0;
    for (int j = -1; j <= 1; ++j)
    for (int i = -1; i <= 1; ++i) {
        vec2 g = vec2(float(i), float(j));
        vec2 o = g + peb_h3(ip + g + 113.0).xy - fp;
        float d = dot(o, o);
        if (d < f1) { f2 = f1; f1 = d; } else if (d < f2) f2 = d;
    }
    return sqrt(f2) - sqrt(f1);
}
// Lake / river bed: a bed of closely packed pebbles (after "Raymarching
// Pebbles", shadertoy wlsBRX: a pebble height field, shaded with its slope,
// darkened in the crevices between the stones, with a wet highlight).
// Two sizes, every cell occupied; the small ones fill the gaps.
float bed_h(vec2 wp, float wA, float wB, out float id, out float k) {
    float best = 0.0, h = 0.0;
    g_peb_id = 0.0; g_peb_k = 0.0;
    if (wA > 0.0) {
        h = wA * peb_layer(wp, 0.22, 1.0, 5.0, 1.0, best);
        h = max(h, wA * peb_layer(wp + vec2(0.11, 0.11), 0.22, 1.0, 7.0, 0.85, best));   // second set in the gaps: packed
    }
    if (wB > 0.0) h = max(h, wB * peb_layer(wp + 5.3, 0.075, 1.0, 6.0, 1.0, best));
    id = g_peb_id; k = g_peb_k;
    return h;
}
// One smooth random value per ~1/f metres (4 hashes; noise2 also computes
// derivatives nobody needs here).
float peb_vnoise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    float a = peb_h3(i).x, b = peb_h3(i + vec2(1.0, 0.0)).x;
    float c = peb_h3(i + vec2(0.0, 1.0)).x, d = peb_h3(i + vec2(1.0, 1.0)).x;
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}
// 0 = everything (shading), 1 = while marching (only what moves the
// silhouette, near the camera), 2 = shadow walk (no grit), -1 = off.
int g_peb_mode = 0;
// The stone under the pixel being shaded (left by calcNormal's centre tap).
float g_pebN_h = 0.0, g_pebN_id = 0.0, g_pebN_k = 0.0, g_pebN_sz = 0.0;
// Height of the stones at wp (metres). t = distance along the view ray.
float rm_pebbles(vec2 wp, float t) {
    g_peb_k = 0.0; g_peb_sz = 0.0;
    if (g_peb_mode < 0 || t > 26.0) return 0.0;
    float fp = max(t, 0.05) * 2.0 * scene.viewport.w;        // ~metres per pixel
    float wA = smoothstep(2.5, 6.0, 0.110 / fp);
    float wB = smoothstep(2.5, 6.0, 0.034 / fp);
    float wC = smoothstep(2.5, 6.0, 0.010 / fp);
    float wD = 1.0;
    if (g_peb_mode == 1) {
        wA *= 1.0 - smoothstep(1.5, 2.5, t);
        wB = 0.0; wC = 0.0;
        wD = 1.0 - smoothstep(4.0, 6.0, t);
    } else if (g_peb_mode == 2) {
        wC = 0.0;
        wB *= 1.0 - smoothstep(2.0, 3.5, t);
    }
    float best = 0.0, h = 0.0;
    if (wA + wB + wC > 0.0) {
        // Stones lie in drifts: dense patches, thin scatter, bare soil.
        float dens = smoothstep(0.30, 0.75, peb_vnoise(wp * 0.45 + 7.1));
        if (wA > 0.0) h = max(h, wA * peb_layer(wp, 0.110, 0.10 + 0.55 * dens, 1.0, 1.0, best));
        if (wB > 0.0) h = max(h, wB * peb_layer(wp + 3.7, 0.034, 0.15 + 0.60 * dens, 2.0, 1.0, best));
        if (wC > 0.0) h = max(h, wC * peb_layer(wp + 9.1, 0.010, 0.45 + 0.40 * dens, 3.0, 1.0, best));
    }
    // Big cobbles: fist- to head-sized stones, few, and in OTHER places
    // than the pebble drifts (their own patch pattern) - singles and
    // small groups half sunk in the soil.
    if (wD > 0.0) {
        float d2 = 0.03 + 0.30 * smoothstep(0.45, 0.80, peb_vnoise(wp * 0.30 + 31.7));
        float sid = g_peb_id, sk = g_peb_k, ssz = g_peb_sz, bb = 0.0;
        float hb = wD * peb_layer(wp + 17.3, 0.420, d2, 4.0, 0.5, bb);
        if (hb > h) h = hb;
        else { g_peb_id = sid; g_peb_k = sk; g_peb_sz = ssz; }   // a smaller stone on top keeps its own look
    }
    return h;
}
float rm_pebbles_march(vec2 wp, float t) {
    if (t > 6.0 || g_peb_mode < 0) return 0.0;
    int m = g_peb_mode;
    g_peb_mode = 1;
    float h = rm_pebbles(wp, t);
    g_peb_mode = m;
    return h;
}

// Depth (m) of the deepest footprint at world xz `p` (scene.footprints):
// a sole - the ball of the foot and a heel - pressed 7 cm into the snow.
float foot_depth(vec2 p) {
    float d = 0.0;
    for (int i = 0; i < 32; ++i) {
        vec4 f = scene.footprints[i];
        if (f.w <= 0.0) continue;
        vec2 q = p - f.xy;
        if (dot(q, q) > 0.06) continue;
        float c = cos(f.z), s = sin(f.z);
        float lx = c * q.x + s * q.y;              // across the foot
        float lz = s * q.x - c * q.y;              // along it (toes forward)
        float ball = (length(vec2(lx / 0.058, (lz - 0.055) / 0.105)) - 1.0) * 0.058;
        float heel = length(vec2(lx, lz + 0.105)) - 0.047;
        float k = 0.03, h = clamp(0.5 + 0.5 * (heel - ball) / k, 0.0, 1.0);
        float sdf = mix(heel, ball, h) - k * h * (1.0 - h);          // smooth union
        d = max(d, f.w * 0.07 * smoothstep(0.012, -0.014, sdf));
    }
    return d;
}
