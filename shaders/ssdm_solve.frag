#version 460
// Screen-space displacement mapping - the solve pass: per pixel of displaced stonework, the
// vector its surface point moves by, and how much light the stones around it leave it.
// See ssdm_grid.vert for the whole method.
//
// What decides how the stones meet at corners, outlines and other materials is here, and it
// is the rule the SSDM_Background_TileVS shader of the reference game uses (read from the compiled
// shader of the game; this is our own code for it):
//
//   * The height is averaged with the pixels around it. A neighbour only adds to the sum
//     as far as it agrees with this pixel: same depth range, a normal that points the same
//     way, a similar height. Its weight by distance counts in the divisor regardless.
//     So wherever the surroundings disagree - round a corner onto the other face, at the
//     foot of the wall where the ground begins, next to anything that is not displaced -
//     the height sinks to nothing, smoothly, and both faces of a corner come down to the
//     same undisplaced edge. No gap can open between them.
//   * A neighbour's height counts only as far as its displaced point stands out along THIS
//     pixel's normal: on the other face of a corner that is nothing.
//   * A pixel that has something much NEARER beside it (it lies behind another thing's
//     outline) is not displaced at all: the grid is cut there, so nothing is ever stretched
//     from one surface to another across a jump in depth.
//   * Toward the screen border the displacement is turned from "along the normal" to
//     "along the line of sight": the pixel keeps its place on screen (pixels that would have
//     to come from outside the picture do not exist) but still gets its displaced depth.
//
// Ours, not the game's:
//   * What lies far BEHIND a pixel (the sky, the land beyond the wall's outline) neither
//     holds it down nor cuts it: the outline itself is displaced, the stones stand out of it.
//   * The light. The ray-traced sun shadow is cast by the flat wall, which knows nothing of
//     the stones, so each pixel finds here how much the stones toward the sun shade it (a
//     short march over the height field, in screen space) and how deep it lies in a joint.
//     The grid pass darkens the albedo by it, as the parallax relief did (cube.frag).
#extension GL_GOOGLE_include_directive : require
#include "ssdm_common.glsl"

layout(location = 0) out vec4 outOff;       // xyz displacement vector (m, world axes), a light kept (0 = not a displaced pixel)

void main() {
    ivec2 p   = ivec2(gl_FragCoord.xy);
    vec4  hn0 = texelFetch(u_hn, p, 0);
    if (hn0.a <= 0.0) { outOff = vec4(0.0); return; }
    float d0;
    vec3  P0 = ssdm_rel(p, texelFetch(u_depth, p, 0).r, d0);
    vec3  N0 = normalize(hn0.xyz);
    float h0 = max(hn0.a - 1e-4, 0.0);
    const float sigma  = pc.par.x;
    const float ampRef = pc.par.y;

    vec3  V   = normalize(-P0);                                  // toward the camera
    float ndv = clamp(dot(N0, V), 0.0, 1.0);
    float tol_near = max(ampRef * (1.0 + d0 / 32.0) / max(ndv, 0.02), 0.05 * d0);
    float tol_far  = max(0.02, 0.05 * d0);

    // Two pixels apart close by (where a pixel is far smaller than a joint), one further off.
    int   stp  = d0 < 8.0 ? 2 : 1;
    float pixw = d0 * pc.par.w * float(stp);
    const ivec2 kTap[12] = ivec2[12](
        ivec2(-1, 0), ivec2(1, 0), ivec2(0, -1), ivec2(0, 1),
        ivec2(-1, -1), ivec2(1, -1), ivec2(-1, 1), ivec2(1, 1),
        ivec2(-2, 0), ivec2(2, 0), ivec2(0, -2), ivec2(0, 2));
    ivec2 lim = ivec2(pc.size.xy) - 1;
    float sum = h0, wsum = 1.0, dmin = d0;
    float raw = h0, rsum = 1.0;                 // plain mean of the heights around (for the joint shade)
    for (int i = 0; i < 12; ++i) {
        ivec2 q  = clamp(p + kTap[i] * stp, ivec2(0), lim);
        float r  = length(vec2(kTap[i])) * pixw / sigma;
        float g  = exp(-r * r);
        float zi = texelFetch(u_depth, q, 0).r;
        if (zi >= 1.0) continue;                                 // (sky: far behind)
        float di;
        vec3  Pi = ssdm_rel(q, zi, di);
        if (di > d0 + tol_far) continue;                         // (far behind: this pixel is on an outline)
        dmin = min(dmin, di);
        wsum += g;
        vec4  hni = texelFetch(u_hn, q, 0);
        if (hni.a <= 0.0 || abs(di - d0) > 0.2 * d0) continue;
        vec3  Ni = normalize(hni.xyz);
        float hi = max(hni.a - 1e-4, 0.0);
        float hc = min(max(dot(Pi + Ni * hi - P0, N0), 0.0), hi);
        float agree = clamp(dot(N0, Ni), 0.0, 1.0);
        sum += g * hc * agree * clamp(1.0 - abs(hc - h0) / ampRef, 0.0, 1.0);
        float gr = exp(-0.25 * dot(vec2(kTap[i]), vec2(kTap[i]))) * agree;
        raw += gr * hi; rsum += gr;
    }
    if (dmin < d0 - tol_near) { outOff = vec4(0.0); return; }
    float h = sum / wsum;

    // ---- Room to stand out (ours) ----
    // The averaging above reaches two pixels: enough to close a corner, not enough to keep
    // a stone that stands 20 cm out from being pushed across a ledge 5 cm wide, or to slope
    // the stones down toward a corner. So: along the four screen directions, how far is it
    // to where the surface turns away or something that is not displaced begins, measured
    // against the height in pixels? The height is scaled by the nearest of the four - nothing
    // stands out further than it is from an edge. (Sky and things far behind are no edge:
    // the outline stays displaced.)
    {
        float rpx = clamp(h0 / max(d0 * pc.par.w, 1e-6), 2.0, 24.0);     // the height, in pixels
        const ivec2 kDir[4] = ivec2[4](ivec2(1, 0), ivec2(-1, 0), ivec2(0, 1), ivec2(0, -1));
        float room = 1.0;
        for (int i = 0; i < 4; ++i) {
            float free = 1.0;                                            // fraction of rpx that is clear this way
            for (int k = 4; k >= 1; --k) {
                ivec2 q  = clamp(p + ivec2(vec2(kDir[i]) * (rpx * float(k) * 0.25) + sign(vec2(kDir[i])) * 0.5), ivec2(0), lim);
                float zi = texelFetch(u_depth, q, 0).r;
                if (zi >= 1.0) continue;
                float di;
                ssdm_rel(q, zi, di);
                if (di > d0 + tol_far || di < d0 - tol_near) continue;
                vec4  hni = texelFetch(u_hn, q, 0);
                float agree = hni.a <= 0.0 ? 0.0 : smoothstep(0.55, 0.9, dot(N0, normalize(hni.xyz)));
                free = min(free, mix(float(k - 1) * 0.25, 1.0, agree));
            }
            room = min(room, free);
        }
        h *= room; h0 *= room;
    }

    // Screen border: w = how much of the sideways shift is kept.
    float w = 1.0;
    {
        vec4 c0 = pc.vp * vec4(P0, 1.0);
        vec4 c1 = pc.vp * vec4(P0 + N0 * max(h, 1e-4), 1.0);
        vec2 n0 = c0.xy / c0.w, dl = c1.xy / c1.w - n0;
        vec2 m  = n0 - dl;                 // where the pixel that lands HERE would have to come from
        float f = 0.0;
        if (m.x < -1.0) f = max(f, (-1.0 - m.x) / max(dl.x, 1e-6));
        if (m.x >  1.0) f = max(f, (m.x - 1.0) / max(-dl.x, 1e-6));
        if (m.y < -1.0) f = max(f, (-1.0 - m.y) / max(dl.y, 1e-6));
        if (m.y >  1.0) f = max(f, (m.y - 1.0) / max(-dl.y, 1e-6));
        w = clamp(1.0 - f, 0.0, 1.0);
    }
    vec3 off = N0 * (h * w);
    if (w < 1.0 && ndv > 0.02) off += V * (h * (1.0 - w) / ndv);

    // ---- The light the stones leave this point ----
    // In a joint: lower than what is around it.
    float keep = 1.0 - pc.rect.z * clamp((raw / rsum - h0) / ampRef, 0.0, 1.0);
    // Toward the sun: a ray from the displaced point, as far as it takes to rise clear of
    // the tallest stone; at each step the stone standing at that place is compared with it.
    vec3  L  = pc.sun.xyz;
    float nl = dot(N0, L);
    if (pc.sun.w > 0.0 && nl > 0.03) {
        vec3  S0   = P0 + N0 * h0;
        float top  = 2.2 * ampRef;                               // (no stone is taller)
        float tmax = min((top - h0) / max(nl, 0.08), 0.6);
        const int kSteps = 8;
        float occ = 0.0;
        for (int k = 0; k < kSteps; ++k) {
            float t = tmax * (float(k) + 0.6) / float(kSteps);
            vec3  S = S0 + L * t;
            vec4  c = pc.vp * vec4(S, 1.0);
            if (c.w <= 0.0) break;
            ivec2 q = ivec2((c.xy / c.w * 0.5 + 0.5) * pc.size.xy);
            if (any(lessThan(q, ivec2(0))) || any(greaterThan(q, lim))) break;
            if (q == p) continue;
            vec4  hq = texelFetch(u_hn, q, 0);
            if (hq.a <= 0.0) continue;
            float dq;
            vec3  Pq = ssdm_rel(q, texelFetch(u_depth, q, 0).r, dq);
            vec3  Nq = normalize(hq.xyz);
            float above = dot(S - Pq, Nq);                       // the ray's height over the drawn surface there
            if (abs(above) > 3.0 * ampRef) continue;             // (another surface)
            occ = max(occ, clamp((hq.a - above) / (0.3 * ampRef), 0.0, 1.0));
        }
        keep *= 1.0 - pc.sun.w * occ * smoothstep(0.03, 0.2, nl);
    }
    outOff = vec4(off, max(keep, 0.05));
}
