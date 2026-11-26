// The tone curves (compose.frag, and the local tone mapping passes that must see the picture
// through the same curve: ltm_lum.comp). Needs post_fx.glsl (pfx_agx_look) included first.

// AgX (Troy Sobotka, Blender 4) - minimal version with the 6th-order
// sigmoid fit (Benjamin Wrensch). Smooth highlight desaturation toward
// white instead of ACES' hue skews; outputs display-referred sRGB-linear.
// look_on > 0.5: the AgX look (post_fx.glsl pfx_agx_look) with scene luminance L, night factor n.
vec3 tm_agx(vec3 c, float look_on, float L, float n) {
    const mat3 agx_in = mat3(0.842479062253094, 0.0423282422610123, 0.0423756549057051,
                             0.0784335999999992, 0.878468636469772, 0.0784336,
                             0.0792237451477643, 0.0791661274605434, 0.879142973793104);
    const mat3 agx_out = mat3(1.19687900512017, -0.0528968517574562, -0.0529716355144438,
                              -0.0980208811401368, 1.15190312990417, -0.0980434501171241,
                              -0.0990297440797205, -0.0989611768448433, 1.15107367264116);
    const float min_ev = -12.47393, max_ev = 4.026069;
    c = agx_in * max(c, vec3(1e-10));
    c = clamp(log2(c), min_ev, max_ev);
    c = (c - min_ev) / (max_ev - min_ev);
    vec3 x2 = c * c, x4 = x2 * x2;
    c = 15.5 * x4 * x2 - 40.14 * x4 * c + 31.96 * x4 - 6.868 * x2 * c
        + 0.4298 * x2 + 0.1191 * c - 0.00232;
    if (look_on > 0.5) c = pfx_agx_look(c, L, n);
    c = agx_out * c;
    return pow(max(c, vec3(0.0)), vec3(2.2));   // AgX base is ~gamma 2.2 encoded
}
// Khronos PBR Neutral (2024): hue-preserving, keeps base colours exact
// below 0.8 and compresses only the highlights.
vec3 tm_pbr_neutral(vec3 color) {
    const float start = 0.8 - 0.04, desat = 0.15;
    float x = min(color.r, min(color.g, color.b));
    float off = x < 0.08 ? x - 6.25 * x * x : 0.04;
    color -= off;
    float peak = max(color.r, max(color.g, color.b));
    if (peak < start) return color;
    float d = 1.0 - start;
    float np = 1.0 - d * d / (peak + d - start);
    color *= np / peak;
    float g = 1.0 - 1.0 / (desat * (peak - np) + 1.0);
    return mix(color, vec3(np), g);
}

vec3 tm_aces_fitted(vec3 x) {
    const mat3 in_m = mat3(
        0.59719, 0.07600, 0.02840,
        0.35458, 0.90834, 0.13383,
        0.04823, 0.01566, 0.83777);
    const mat3 out_m = mat3(
         1.60475, -0.10208, -0.00327,
        -0.53108,  1.10813, -0.07276,
        -0.07367, -0.00605,  1.07602);
    vec3 v = in_m * x;
    vec3 a = v * (v + 0.0245786) - 0.000090537;
    vec3 b = v * (0.983729 * v + 0.4329510) + 0.238081;
    return clamp(out_m * (a / b), 0.0, 1.0);
}

// mode: 0 ACES fitted, 1 AgX, 2 PBR Neutral.
vec3 tone_curve(vec3 hdr, int mode, float look_on, float L, float n) {
    return (mode == 1) ? tm_agx(hdr, look_on, L, n) : (mode == 2) ? tm_pbr_neutral(hdr) : tm_aces_fitted(hdr);
}
