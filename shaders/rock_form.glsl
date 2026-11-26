// rock_form.glsl - the material of the rock formations (vk_engine/rock_formations.h; the
// meshes are made by tools/rocks/make_formations.py). Included by cube.frag (colour, normal,
// the blend into the ground) and by ssdm_height.frag (RF_HEIGHT_ONLY: the height alone, for
// the screen-space displacement) - one layout of the texture for both, so the stones that
// are moved are the stones that are drawn.
//
// The mesh carries the shape down to a hand's width; everything finer is the landscape's own
// rock scan (slot 7: colour, normal, height in the normal's alpha), laid on by WORLD position
// exactly as terrain_rm_main_ground.glsl lays it on a cliff - the same three planes, the same
// 0.30 repeats a metre, the same growth of the tile with distance (each band four times the
// last, two neighbouring bands laid over each other) and the same shift per band - so a
// formation standing in a cliff carries the cliff's rock on, and a large face never shows the
// tile as a grid. Within a plane the repeats are broken as the ground's are (rm_at): cells of
// two tiles, each turned by quarter turns and shifted, their borders bent, blended across
// the border.
//
// Needs before it: scene_ubo.glsl, u_normal[] (and u_albedo[] unless RF_HEIGHT_ONLY).

const int   kRfRock = 7;
const float kRfTile = 0.30;

float rf_h21(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}
float rf_vn(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(rf_h21(i), rf_h21(i + vec2(1.0, 0.0)), f.x), mix(rf_h21(i + vec2(0.0, 1.0)), rf_h21(i + vec2(1.0, 1.0)), f.x), f.y);
}
// The share of the second of two materials where both are present, by height (the reference
// game's blend; terrain_rm_heightfield.glsl rm_height_blend - keep the two the same).
float rf_height_blend(float ha, float hb, float w) {
    float a = max(ha, 0.01) * (1.0 - w), b = max(hb, 0.01) * w;
    float d = max(max(a, b) - 0.15, 0.0);
    a = max(a - d, 0.0); b = max(b - d, 0.0);
    return b / max(a + b, 1e-5);
}

// One cell's layout of the scan. alb += colour, nh += (normal xy turned back, height, 1).
void rf_cell(vec2 cell, vec2 uv, vec2 dx, vec2 dy, float w, inout vec3 alb, inout vec4 nh) {
    float h = rf_h21(cell - 32.0 * floor(cell / 32.0) + 0.37);
    float k = floor(h * 4.0);
    vec2 cs = k < 0.5 ? vec2(1.0, 0.0) : (k < 1.5 ? vec2(0.0, 1.0) : (k < 2.5 ? vec2(-1.0, 0.0) : vec2(0.0, -1.0)));
    vec2 u  = vec2(cs.x * uv.x - cs.y * uv.y, cs.y * uv.x + cs.x * uv.y) + vec2(fract(h * 7.13), fract(h * 13.7));
#ifdef RF_HEIGHT_ONLY
    // (the mip of the LONG side of the footprint: heights that jump from pixel to pixel comb
    //  the displacement grid into stripes at a slant - ssdm_height.frag)
    float fp = max(length(dx), length(dy)) * float(textureSize(u_normal[kRfRock], 0).x);
    nh += vec4(0.0, 0.0, textureLod(u_normal[kRfRock], u, log2(max(fp, 1.0)) + 0.5).a, 1.0) * w;
#else
    vec2 gx = vec2(cs.x * dx.x - cs.y * dx.y, cs.y * dx.x + cs.x * dx.y);
    vec2 gy = vec2(cs.x * dy.x - cs.y * dy.y, cs.y * dy.x + cs.x * dy.y);
    alb += textureGrad(u_albedo[kRfRock], u, gx, gy).rgb * w;
    vec4 s = textureGrad(u_normal[kRfRock], u, gx, gy);
    vec2 v = s.xy - 0.5;
    nh += vec4(cs.x * v.x + cs.y * v.y, cs.x * v.y - cs.y * v.x, s.a, 1.0) * w;
#endif
}
void rf_plane(vec2 uv, vec2 dx, vec2 dy, float w, inout vec3 alb, inout vec4 nh) {
    vec2 p = uv * 0.5 + (vec2(rf_vn(uv * 0.61 + 3.7), rf_vn(uv * 0.61 + 21.3)) - 0.5) * 0.30;
    vec2 c = floor(p), f = p - c;
    // (a tile under a few pixels: no seam is resolved, one sample)
    bool seams = max(length(dx), length(dy)) * float(textureSize(u_normal[kRfRock], 0).x) < 128.0;
    vec2 wn = seams ? 0.5 * (1.0 - smoothstep(0.0, 0.09, min(f, 1.0 - f))) : vec2(0.0);
    rf_cell(c, uv, dx, dy, w * ((1.0 - wn.x) * (1.0 - wn.y) + wn.x * wn.y), alb, nh);
    if (wn.x > 0.004) rf_cell(c + vec2(f.x < 0.5 ? -1.0 : 1.0, 0.0), uv, dx, dy, w * wn.x * (1.0 - wn.y), alb, nh);
    if (wn.y > 0.004) rf_cell(c + vec2(0.0, f.y < 0.5 ? -1.0 : 1.0), uv, dx, dy, w * wn.y * (1.0 - wn.x), alb, nh);
}

struct RfMat {
    vec3  alb;      // the scan's colour (as it is in the texture)
    vec3  nrm;      // the surface normal with the scan's relief
    float h;        // the scan's height 0..1
};
// wp: the point; N: the mesh's normal there (unit); dpx / dpy: the point's change across one
// pixel; dist: metres from the eye.
RfMat rf_material(vec3 wp, vec3 N, vec3 dpx, vec3 dpy, float dist) {
    vec3 w = abs(N);
    w *= w; w *= w;
    w /= (w.x + w.y + w.z);
    float bL = max(log2(max(dist, 1e-3) / 6.0) * 0.5, 0.0);
    float bi = floor(bL), bw = smoothstep(0.0, 1.0, bL - bi);
    float sA = pow(0.25, bi);
    vec3 alb = vec3(0.0), bump = vec3(0.0);
    float hs = 0.0, ws = 0.0;
    for (int b = 0; b < 2; ++b) {
        float wb = b == 0 ? 1.0 - bw : bw;
        if (wb < 0.004) continue;
        float sc = kRfTile * (b == 0 ? sA : sA * 0.25);
        vec2  bo = vec2(17.37, 9.13) * (bi + float(b));
        if (w.x > 0.03) {
            vec3 a = vec3(0.0); vec4 nh = vec4(0.0);
            rf_plane(wp.zy * sc + bo, dpx.zy * sc, dpy.zy * sc, 1.0, a, nh);
            float k = w.x * wb;
            alb += a * k; hs += nh.z * k; ws += k;
            bump += (vec3(0.0, 0.0, 1.0) * nh.x - vec3(0.0, 1.0, 0.0) * nh.y) * k;
        }
        if (w.y > 0.03) {
            vec3 a = vec3(0.0); vec4 nh = vec4(0.0);
            rf_plane(wp.xz * sc + bo, dpx.xz * sc, dpy.xz * sc, 1.0, a, nh);
            float k = w.y * wb;
            alb += a * k; hs += nh.z * k; ws += k;
            bump += (vec3(1.0, 0.0, 0.0) * nh.x - vec3(0.0, 0.0, 1.0) * nh.y) * k;
        }
        if (w.z > 0.03) {
            vec3 a = vec3(0.0); vec4 nh = vec4(0.0);
            rf_plane(wp.xy * sc + bo, dpx.xy * sc, dpy.xy * sc, 1.0, a, nh);
            float k = w.z * wb;
            alb += a * k; hs += nh.z * k; ws += k;
            bump += (vec3(1.0, 0.0, 0.0) * nh.x - vec3(0.0, 1.0, 0.0) * nh.y) * k;
        }
    }
    ws = max(ws, 1e-4);
    RfMat m;
    m.alb = alb / ws;
    m.h   = hs / ws;
    // (the scan's normal is stored 0..1: twice (xy - 0.5); 1.35 as on the cliff, a little flatter far off)
    m.nrm = normalize(N + bump / ws * (2.0 * mix(1.35, 0.95, smoothstep(80.0, 600.0, dist))));
    return m;
}

#ifndef RF_HEIGHT_ONLY
// The rock's colour: the cliff's own tone, beds and patches (terrain_rm_main_ground.glsl
// "A steep face is ROCK" - the same formulas, so a formation is the colour of the face it
// stands in), then what the mesh knows and the ground does not: dark and damp where little
// sky is seen (ao: the generator's ambient occlusion, per vertex), pale on worn edges and
// dirty in creases (curv: its curvature), and on what faces up dust, lichen, moss, and snow
// above the snow line.
vec3 rf_colour(vec3 wp, vec3 Ng, vec3 Nd, RfMat m, float ao, float curv) {
    vec3 mR = max(textureLod(u_albedo[kRfRock], vec2(0.5), 12.0).rgb, vec3(0.02));
    vec3 aK = m.alb / mR;
    float along  = dot(wp.xz, vec2(0.071, 0.053));
    float bed    = wp.y * 0.42 + 1.7 * tile_noise2(wp.xz, 0.021, vec2(3.0)) + 0.35 * sin(along);
    float bands  = mix(0.80, 1.14, smoothstep(0.25, 0.75, fract(bed))) * mix(0.90, 1.08, fract(bed * 3.1 + 0.37));
    float patchy = mix(0.82, 1.16, tile_noise2(wp.xz + wp.y * 0.6, 0.045, vec2(9.0)));
    vec3  tone   = mix(vec3(0.150, 0.140, 0.128), vec3(0.205, 0.188, 0.165), tile_noise2(wp.xz - wp.y * 0.4, 0.011, vec2(1.0)));
    float face   = smoothstep(0.15, 0.6, 1.0 - abs(Ng.y));
    vec3 c = tone * clamp(aK, vec3(0.45), vec3(1.9)) * mix(1.0, bands, face) * patchy;
    c *= mix(vec3(1.0), vec3(0.80, 0.95, 0.70), 0.35 * smoothstep(0.55, 0.8, tile_noise2(wp.xz, 0.09, vec2(4.0))));
    c *= mix(0.74, 1.08, smoothstep(0.15, 0.85, m.h));                       // the scan's pits darker
    float occ = 1.0 - clamp(ao, 0.0, 1.0);
    c *= mix(1.0, 0.40, smoothstep(0.12, 0.85, occ));                        // clefts, the roof of an overhang
    c  = mix(c, c * vec3(0.78, 0.88, 0.74), smoothstep(0.35, 0.9, occ));     // ... damp, a little green
    float edge = smoothstep(0.58, 0.85, curv) * (1.0 - occ);
    c  = mix(c, c * 1.30 + vec3(0.012), 0.5 * edge);                         // worn edges
    c *= 1.0 - 0.25 * smoothstep(0.42, 0.20, curv);                          // creases
    float up = smoothstep(0.55, 0.90, Nd.y);
    float n1 = tile_noise2(wp.xz, 0.9, vec2(13.0)), n2 = tile_noise2(wp.xz + wp.y * 0.7, 2.7, vec2(5.0)), n3 = tile_noise2(wp.xz, 0.23, vec2(7.0));
    c = mix(c, mix(vec3(dot(c, vec3(0.333))), vec3(0.23, 0.21, 0.18), 0.5), up * 0.35);                       // dust on the tops
    float lichen = smoothstep(0.60, 0.80, n2 * 0.6 + n1 * 0.4) * smoothstep(0.05, 0.6, Nd.y) * (1.0 - occ) * (1.0 - edge);
    c = mix(c, mix(vec3(0.30, 0.31, 0.24), vec3(0.34, 0.27, 0.13), step(0.5, n3)), lichen * 0.40);
    float moss = up * smoothstep(0.42, 0.70, n1 * 0.55 + n3 * 0.45 + 0.30 * occ) * (1.0 - edge);
    c = mix(c, vec3(0.070, 0.105, 0.038) * mix(0.8, 1.25, n2), clamp(moss, 0.0, 1.0) * 0.72);
    // (the ground's own snow line - terrain_rm_shading_helpers.glsl getMaterial: 70 - 90 m, lower
    //  on a north face, higher on a south one, on what is not steep; keep the two in step)
    float snow = smoothstep(70.0, 90.0, wp.y - clamp(Nd.z * 30.0, -16.0, 16.0)) * smoothstep(0.3, 0.7, Nd.y);
    return mix(c, vec3(0.62, 0.65, 0.70), clamp(snow * (0.85 + 0.15 * n2), 0.0, 1.0));
}
#endif

// ---- CHALK: the second look (a sea cliff; normal index -10) --------------------------------
// The same meshes, the same scan for the grain - but the face's own fine structure is worked
// out here, because it is what makes chalk chalk and no scan in the game has it: level beds
// about half a metre thick with a parting between them, cut into blocks by short upright
// joints that do not line up from bed to bed. rf_chalk_h is that as a height 0..1 (1 = a
// block's face, 0 = in a parting or a joint): the screen-space displacement lifts the blocks
// by it (ssdm_height.frag), cube.frag bends the light by its slope and darkens its lines.
// `flint`: 1 on a flint band - every few beds a parting carries a broken line of dark nodules.
float rf_chalk_h(vec3 wp, vec3 N, out float flint) {
    float u   = abs(N.x) > abs(N.z) ? wp.z : wp.x;                    // along the face
    float bed = wp.y / 0.55 + 0.6 * rf_vn(vec2(u * 0.05, wp.y * 0.3));
    float bi  = floor(bed), bf = bed - bi;
    float uj  = u / 1.1 + rf_h21(vec2(bi, 3.0)) * 7.0 + 0.15 * rf_vn(vec2(u * 0.7, wp.y * 0.9));
    float ji  = floor(uj), jf = uj - ji;
    float part  = smoothstep(0.0, 0.09, min(bf, 1.0 - bf));
    float joint = smoothstep(0.0, 0.045, min(jf, 1.0 - jf));
    float fb = step(0.5, rf_h21(vec2(floor(bi / 4.0), 9.0)) ) * step(abs(mod(bi, 4.0) - 1.0), 0.5);      // (a band on about half of every fourth parting)
    flint = fb * (1.0 - smoothstep(0.03, 0.08, bf)) * step(0.42, rf_vn(vec2(u * 3.1, bi * 7.0)));
    float face = smoothstep(0.25, 0.6, 1.0 - abs(N.y));               // (tops and undersides carry no beds)
    return mix(1.0, part * joint * (0.6 + 0.4 * rf_h21(vec2(ji, bi))), face);
}
#ifndef RF_HEIGHT_ONLY
// Chalk's colour. hgt: metres above the piece's ground line (the sea, for a sea cliff);
// lines: rf_chalk_h's value (faded out by the caller where the blocks are smaller than a pixel).
vec3 rf_chalk_colour(vec3 wp, vec3 Ng, vec3 Nd, RfMat m, float ao, float curv, float hgt, float lines, float flint) {
    vec3  mR  = max(textureLod(u_albedo[kRfRock], vec2(0.5), 12.0).rgb, vec3(0.02));
    float lum = clamp(dot(m.alb / mR, vec3(0.333)), 0.55, 1.5);
    float u   = abs(Ng.x) > abs(Ng.z) ? wp.z : wp.x;
    float face = smoothstep(0.25, 0.6, 1.0 - abs(Ng.y));
    float n1 = rf_vn(wp.xz * 0.09 + wp.y * 0.05), n2 = rf_vn(vec2(u * 0.31, wp.y * 0.27) + 7.0), n3 = rf_vn(wp.xz * 1.3 + wp.y * 0.7);
    // white to cream, the scan's grain at a third of its contrast; the freshest falls palest
    vec3 c = mix(vec3(0.74, 0.72, 0.65), vec3(0.86, 0.85, 0.81), n1) * mix(1.0, lum, 0.35);
    c = mix(c, vec3(0.90, 0.90, 0.87), smoothstep(0.62, 0.80, n2) * 0.7);
    c *= mix(0.80, 1.04, smoothstep(0.15, 0.85, m.h));
    // streaks under seeps: rust and grey, long and upright
    float s1 = rf_vn(vec2(u * 0.9, wp.y * 0.055)), s2 = rf_vn(vec2(u * 0.55 + 40.0, wp.y * 0.04 + 3.0));
    c = mix(c, c * vec3(0.84, 0.66, 0.46), face * 0.55 * smoothstep(0.62, 0.86, s1));
    c = mix(c, c * vec3(0.60, 0.62, 0.61), face * 0.55 * smoothstep(0.58, 0.84, s2));
    // the blocks' joints and partings, and the flint
    c *= mix(0.66, 1.0, lines);
    c = mix(c, vec3(0.045, 0.045, 0.05), flint * 0.9);
    // algae from the splash zone up, the foot dark and wet
    float alg = 1.0 - smoothstep(1.5, 9.0, hgt + 3.0 * (n3 - 0.5) + 2.5 * (n1 - 0.5));
    c = mix(c, vec3(0.17, 0.21, 0.12) * (0.7 + 0.5 * lum), alg * 0.78);
    c *= mix(0.55, 1.0, smoothstep(-0.3, 2.2, hgt));
    float occ = 1.0 - clamp(ao, 0.0, 1.0);
    c *= mix(1.0, 0.48, smoothstep(0.12, 0.85, occ));
    c  = mix(c, c * vec3(0.78, 0.88, 0.74), smoothstep(0.35, 0.9, occ));
    c *= 1.0 - 0.2 * smoothstep(0.42, 0.20, curv);
    // turf: on the top and on every ledge that holds earth (not on the fallen blocks at the foot)
    float turf = smoothstep(0.62, 0.86, Nd.y) * smoothstep(3.5, 6.0, hgt) * (1.0 - smoothstep(0.3, 0.7, occ));
    vec3  green = mix(vec3(0.060, 0.105, 0.030), vec3(0.105, 0.135, 0.045), n3) * mix(0.8, 1.2, n1);
    c = mix(c, green, turf * 0.92);
    float snow = smoothstep(70.0, 90.0, wp.y - clamp(Nd.z * 30.0, -16.0, 16.0)) * smoothstep(0.3, 0.7, Nd.y);
    return mix(c, vec3(0.62, 0.65, 0.70), clamp(snow, 0.0, 1.0));
}
#endif
