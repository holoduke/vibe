// ssdm_displace.glsl - the per-vertex displacement of the screen-space displacement passes:
// our code for the reference game's SSDM_Background_TileVS body, which its compute rasterizer
// (ssdm_raster.comp) and its hardware fallback (ssdm_hwraster.vert) share. Read from the
// compiled shaders; every constant is the game's. Include after ssdm_common.glsl.
//
// One vertex per pixel CORNER of the grid. Per vertex:
//   * reach: beyond T (pc.depth.z) nothing; the amplitude for height 1 is
//       0.2 * scale * (1 - saturate(max(0, d - 0.75 T) / (0.25 T))) * (1 + 4 * saturate(d / 100))
//     i.e. full to 0.75 T, nothing at T, and 1 x close by growing to 5 x by 100 m;
//   * the height evened out with its neighbours: the terrain (class 3) over a 5-tap cross, the
//     rest over 4 x 4 taps at offsets -1..2, both a step of 2 px close by and 1 px from 16 m,
//     Gaussian weights from the world span between taps; a neighbour counts as far as it agrees
//     in depth (20 %), normal, surface byte and height, and as far as its displaced point stands
//     out over this vertex's tangent plane;
//   * no vertex where the depth around it jumps (nearer than the displacement could explain,
//     or farther than 5 %): outlines stay where they are;
//   * the direction: the vertex normal evened out over 2 x 2 by depth similarity;
//   * the screen edge: what would have to come from outside the picture fades, and the faded
//     part is moved along the line of sight instead.

struct SsdmVtx {
    vec4  clip;     // the displaced point in clip space (NaN = no vertex)
    vec2  uv;       // its grid pixel / size (0 = no vertex)
    vec3  N;        // the displacement direction
    float state;    // 1 displaced, -1 cut at a depth jump, 0 any other reason
};

float ssdm_gauss(float r_uv, float s) { float x = r_uv * s; return exp2(x * x * -1.442695); }

SsdmVtx ssdm_reject(float state) {
    SsdmVtx v;
    v.clip = vec4(uintBitsToFloat(0x7fc00000u));
    v.uv = vec2(0.0); v.N = vec3(0.0); v.state = state;
    return v;
}

// pxy: the grid pixel. tilevs = true: the vertex shader's rules (the hardware path) - nothing
// beyond T, and a vertex cut at a depth jump is lost; false: the compute rasterizer's - no far
// test, and a cut vertex keeps its undisplaced place unless the jump is large.
SsdmVtx ssdm_displace(vec2 pxy, bool tilevs) {
    float W = pc.size.x, H = pc.size.y, iW = pc.size.z, iH = pc.size.w;
    float T = pc.depth.z;
    float px = pxy.x, py = pxy.y;
    vec2  uv  = vec2(px * iW, py * iH);                  // the pixel CORNER
    vec2  ndc = uv * 2.0 - 1.0;

    // ---- 2 x 2 depth gather (its weights are always 0.25: frac(floor(x) - 0.5) = 0.5) ----
    vec4  g  = textureGather(u_depth, vec2(iW * (px + 0.5), iH * (py + 0.5)), 0);
    vec4  dl = vec4(ssdm_lin(g.x), ssdm_lin(g.y), ssdm_lin(g.z), ssdm_lin(g.w));
    float d  = dl.w;
    vec2  b  = floor(uv * vec2(W, H)) - 0.5;
    ivec2 i0 = ivec2(floor(b));
    vec2  f  = fract(b);
    float wX = (1.0 - f.y) * (1.0 - f.x);
    float wY = (1.0 - f.y) * f.x;
    float wZ = (1.0 - f.x) * f.y;
    float wW = f.y * f.x;
    float tol = d * 0.05;

    if (tilevs && d > T) return ssdm_reject(0.0);
    ivec2 pcx = ivec2(uv * vec2(W, H) + 0.5);
    uint  cls = ssdm_class(pcx);
    if ((cls & 126u) == 12u || cls == 16u || (cls - 17u) < 2u || cls == 107u || cls == 19u || cls == 0u || cls == 10u)
        return ssdm_reject(0.0);
    bool noSmooth;
    if (d < 64.0) noSmooth = (cls > 11u) ? ((cls < 20u) ? true : (cls - 52u) < 16u) : false;
    else          noSmooth = (cls == 13u) ? true : (cls - 52u) < 16u;

    // ---- the centre: the mean linear depth of the 2 x 2, at the corner ----
    float rawC = ssdm_z_of_lin(dot(dl, vec4(wZ, wW, wY, wX)));
    vec3  P    = ssdm_unproject(ndc, rawC);
    float cx = min(max(px + 0.5, 0.0), W - 1.0);
    float cy = min(max(py + 0.5, 0.0), H - 1.0);
    ivec2 ci = ivec2(uint(cx), uint(cy));
    float idC = ssdm_mat(ci);
    float hC  = ssdm_h(ci);
    vec3  Nc  = ssdm_vn(ci);

    float amp = (1.0 - clamp(max(0.0, d - T * 0.75) / max(0.01, T * 0.25), 0.0, 1.0)) * 0.2 * pc.depth.w
              * (clamp(d * 0.01, 0.0, 1.0) * 4.0 + 1.0);

    float k   = 2.0 - clamp(d * 0.0625, 0.0, 1.0);
    float sx  = k * iW, sy = k * iH;
    float sig = (d * 100.0) * pc.view.w;
    float gX  = ssdm_gauss(sx, sig);
    float gY  = ssdm_gauss(sy, sig);

    float blurH, dMin, dMax, wSum;
    vec3  blurN;
    if (cls == 3u) {
        // ---- the terrain: a 5-tap cross of plain heights ----
        ivec2 q0 = ivec2(int(cx), int(cy));
        ivec2 qr = ivec2(int(min(max(cx + k, 0.0), W - 1.0)), int(cy));
        ivec2 ql = ivec2(int(min(max(cx - k, 0.0), W - 1.0)), int(cy));
        ivec2 qd = ivec2(int(cx), int(min(max(cy + k, 0.0), H - 1.0)));
        ivec2 qu = ivec2(int(cx), int(min(max(cy - k, 0.0), H - 1.0)));
        blurH = (ssdm_h(qd) + ssdm_h(qu)) * gY + ssdm_h(q0) + (ssdm_h(qr) + ssdm_h(ql)) * gX;
        blurN = ssdm_vn(qr) * gX + ssdm_vn(ql) * gX + ssdm_vn(q0) + ssdm_vn(qu) * gY + ssdm_vn(qd) * gY;
        float l0 = ssdm_lin(ssdm_depth_at(q0)), lr = ssdm_lin(ssdm_depth_at(qr)), ll = ssdm_lin(ssdm_depth_at(ql));
        float ld = ssdm_lin(ssdm_depth_at(qd)), lu = ssdm_lin(ssdm_depth_at(qu));
        dMin = min(1000000.0, min(min(min(l0, lr), min(ll, ld)), lu));
        dMax = max(0.0, max(max(max(l0, lr), max(ll, ld)), lu));
        wSum = gX * 2.0 + 1.0 + gY * 2.0;
    } else {
        // ---- everything else: 4 x 4 taps at -1..2 steps, each as far as it agrees ----
        float sum = 0.0;
        dMin = 1000000.0; dMax = 0.0;
        for (int j = -1; j <= 2; ++j)
            for (int i = -1; i <= 2; ++i) {
                float qx = min(max(cx + float(i) * k, 0.0), W - 1.0);
                float qy = min(max(cy + float(j) * k, 0.0), H - 1.0);
                float raw = ssdm_depth_at(ivec2(int(qx), int(qy)));
                vec2  qndc = vec2(qx * (iW * 2.0) - 1.0, qy * (iH * 2.0) - 1.0);
                vec3  Pq = ssdm_unproject(qndc, raw);
                float dq = ssdm_lin(raw);
                ivec2 qi = ivec2(int(floor(qx)), int(floor(qy)));
                float hq = ssdm_h(qi);
                vec3  Nq = ssdm_vn(qi);
                float Hq = clamp(dot((Pq - P) + Nq * (hq * amp), Nc) / amp, 0.0, hq);
                float depthOK = (abs(dq - d) > d * 0.2) ? 0.0 : 1.0;
                float nAgree  = clamp(dot(Nc, Nq), 0.0, 1.0);
                float idAgree = clamp(1.0 - abs((ssdm_mat(qi) - idC) * 0.003921569) * 5.0, 0.0, 1.0);
                float hAgree  = clamp(1.0 - abs(Hq - hC), 0.0, 1.0);
                float G       = ssdm_gauss(sqrt(float(i * i) * sx * sx + float(j * j) * sy * sy), sig);
                sum  += G * Hq * depthOK * nAgree * idAgree * hAgree;
                dMin  = min(dMin, dq);
                dMax  = max(dMax, dq);
            }
        blurH = sum;
        blurN = vec3(0.0);
        wSum  = 1.0 + ssdm_gauss(2.0 * sx, sig) + ssdm_gauss(2.0 * sy, sig) + ssdm_gauss(length(vec2(2.0 * sx, 2.0 * sy)), sig)
              + 2.0 * ssdm_gauss(length(vec2(sx, 2.0 * sy)), sig) + 2.0 * ssdm_gauss(length(vec2(2.0 * sx, sy)), sig)
              + 4.0 * ssdm_gauss(length(vec2(sx, sy)), sig) + 2.0 * gX + 2.0 * gY;
    }
    float Hb = (1.0 / max(1.0, wSum)) * blurH;

    // ---- depth jumps ----
    float facing = max(0.02, clamp(-dot(Nc, pc.view.xyz), 0.0, 1.0));
    if (dMin < d - max(amp * (d * 0.03125 + 1.0) / facing, tol) || dMax > max(0.02, tol) + d) {
        if (tilevs) return ssdm_reject(-1.0);
        bool far = abs(dMin - d) > max(2.0, d * 0.2) || abs(dMax - d) > max(2.0, d * 0.2);
        SsdmVtx v = ssdm_reject(-1.0);
        if (!far) v.clip = pc.vp * vec4(P, 1.0);
        return v;
    }

    // ---- the direction ----
    vec3 N;
    if (d < T && !noSmooth) {
        float c  = -2885.39 / (tol * d + 0.1);
        float aX = clamp(exp2(pow(abs(dl.x - d), 2.0) * c) * wZ, 0.0, 1.0);
        float aY = clamp(exp2(pow(abs(dl.y - d), 2.0) * c) * wW, 0.0, 1.0);
        float aZ = clamp(exp2(pow(abs(dl.z - d), 2.0) * c) * wY, 0.0, 1.0);
        float aW = clamp(wX, 0.0, 1.0);
        float inv = 1.0 / max(1e-06, aX + aY + aZ + aW);
        vec3 n1 = ssdm_vn(ivec2(i0.x + 1, i0.y));
        vec3 n2 = ssdm_vn(ivec2(i0.x, i0.y + 1));
        vec3 n3 = ssdm_vn(ivec2(i0.x + 1, i0.y + 1));
        N = (inv * aZ) * n1 + (inv * aW) * Nc + (inv * aX) * n2 + (inv * aY) * n3;
    } else {
        N = blurN;
    }
    // (a zero direction is a lost vertex in the game: its position comes out not-a-number)
    if (dot(N, N) <= 0.0) return ssdm_reject(0.0);
    N = normalize(N);

    // ---- the screen edge ----
    vec4  c0 = pc.vp * vec4(P, 1.0);
    vec4  c1 = pc.vp * vec4(P + N * amp, 1.0);
    vec2  s0 = c0.xy / c0.w, dlt = c1.xy / c1.w - s0;
    vec2  src = s0 - dlt;
    float t = 0.0;
    if (src.x < -1.0) t = (-1.0 - src.x) / max(1e-06, dlt.x);
    if (src.x >  1.0) t = max(t, (src.x - 1.0) / max(1e-06, -dlt.x));
    if (src.y < -1.0) t = max(t, (-1.0 - src.y) / max(1e-06, dlt.y));
    if (src.y >  1.0) t = max(t, (src.y - 1.0) / max(1e-06, -dlt.y));
    float pr = ssdm_puddle5(uv);
    float fade = ((cls - 1u) > 1u && pc.misc.z > 0.0) ? 1.0 : pr * clamp(1.0 - t, 0.0, 1.0);

    if (!(fade * Hb <= 0.0)) {
        vec3  V  = -(P * inversesqrt(dot(P, P)));
        float hS = clamp(Hb, 0.0, 1.0);
        vec3  Pd = P + N * (hS * (fade * amp));
        float NV = clamp(dot(N, V), 0.0, 1.0);
        if (fade < 1.0 && NV > 0.02) Pd += V * (hS * (pr * amp)) * clamp(1.0 - fade, 0.0, 1.0) / NV;
        SsdmVtx v;
        v.clip = pc.vp * vec4(Pd, 1.0);
        v.uv = uv; v.N = N; v.state = 1.0;
        return v;
    }
    return ssdm_reject(0.0);
}
