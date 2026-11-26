// water_solver.glsl - what the water's surface takes from the local shallow-water solver round
// the camera (vk_engine/water_solver.cpp, shaders/water_solver.comp). After the reference game:
// the drawn water is NOT moved by it; it takes the solver's height gradient into its normal, its
// velocity into the carried patterns, and its foam - faded out between 0.4 and 0.5 of the window
// from the eye, and where the cell is not the solver's (the open sea, off the home tile).
//
// The includer declares the render picture and names it before including (after scene_ubo.glsl):
//   #define WS_TEX <sampler2D>   x: the surface over the authored top (m), yz: velocity (m/s),
//                                w: foam (0..1; -1 = not the solver's cell)   (scene binding 54)
#ifndef VIBE_WATER_SOLVER_GLSL
#define VIBE_WATER_SOLVER_GLSL

struct WsAt {
    float k;        // how much of the solver is used here (0 outside its window / on the sea)
    vec2  grad;     // the surface's slope (rise per metre, x and z) over the authored top's
    vec2  vel;      // its current (m/s)
    float foam;     // its foam
    float dh;       // the surface over the authored top (m)
};

WsAt ws_at(vec2 p) {
    WsAt r;
    r.k = 0.0; r.grad = vec2(0.0); r.vel = vec2(0.0); r.foam = 0.0; r.dh = 0.0;
#ifdef WS_TEX
    if (scene.water_sw.w < 0.5) return r;
    float L = scene.water_sw.z;
    vec2 uv = (p - scene.water_sw.xy) / L;
    if (any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0)))) return r;
    vec2 de = abs(p - scene.camera_pos.xz);
    float k = 1.0 - clamp((max(de.x, de.y) - 0.4 * L) / (0.1 * L), 0.0, 1.0);
    if (k <= 0.0) return r;
    vec4 c = textureLod(WS_TEX, uv, 0.0);
    if (c.w < 0.0) k *= clamp(1.0 + c.w, 0.0, 1.0);          // (next to cells that are not the solver's)
    if (k <= 0.0) return r;
    float tx = 1.0 / float(textureSize(WS_TEX, 0).x);
    float cell = L * tx;
    float hr = textureLod(WS_TEX, uv + vec2(tx, 0.0), 0.0).x, hl = textureLod(WS_TEX, uv - vec2(tx, 0.0), 0.0).x;
    float hu = textureLod(WS_TEX, uv + vec2(0.0, tx), 0.0).x, hd = textureLod(WS_TEX, uv - vec2(0.0, tx), 0.0).x;
    r.k = k;
    r.grad = vec2(hr - hl, hu - hd) / (2.0 * cell);
    r.grad *= 1.0 / max(1.0, length(r.grad) / 0.6);          // (a cell-sized spike at a stone's face is not a facet)
    r.vel = c.yz;
    r.foam = max(c.w, 0.0);
    r.dh = c.x;
#endif
    return r;
}

// Only the surface over the authored top (m), faded like ws_at: one tap (the wet band on the banks).
float ws_dh(vec2 p) {
#ifdef WS_TEX
    if (scene.water_sw.w < 0.5) return 0.0;
    float L = scene.water_sw.z;
    vec2 uv = (p - scene.water_sw.xy) / L;
    if (any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0)))) return 0.0;
    vec2 de = abs(p - scene.camera_pos.xz);
    float k = 1.0 - clamp((max(de.x, de.y) - 0.4 * L) / (0.1 * L), 0.0, 1.0);
    vec4 c = textureLod(WS_TEX, uv, 0.0);
    return c.w < 0.0 ? 0.0 : c.x * k;
#else
    return 0.0;
#endif
}

#endif
