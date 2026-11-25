// The world-space cache's buffers, and the bounce light ARRIVING at a cell (2026-11-24).
// docs/ARCHITECTURE.md "Bounce light from the cache"; radiance_cache.cpp; rc_resolve.comp.
//
// Before including, define:
//   IC_BIND_KEYS, IC_BIND_ACC, IC_BIND_RES   the three bindings in set 0
//   IC_SALT                                  a number of the shader's own: two shaders whose
//                                            bounce light is not the same quantity must not
//                                            share entries
// and have `scene` (scene_ubo.glsl) declared.
//
// One hash grid: cell size doubles every octave of camera distance from 0.5 m. An entry has
// two slots in `acc` and `res`. Bit 1 of an entry's key: 0 = outgoing radiance (the objects'
// pass, rc_store / rc_lookup there), 1 = irradiance (here).
//
// Irradiance: six entries per cell, as an ambient cube. A sample goes to ONE of the three
// axes its normal has a part in, picked with probability n_axis^2 - and is read the same
// way - so a round trunk has no seam where its normal changes its largest axis. The read is
// moved by up to half a cell at random: over pixels and frames that is the trilinear blend
// of the cells around, without the eight look-ups.
// Second slot: x = the part of the rays that reached the sky, y = how many samples the value
// rests on.
layout(std430, set = 0, binding = IC_BIND_KEYS) buffer RcKeys { uint rc_keys[]; };
layout(std430, set = 0, binding = IC_BIND_ACC)  buffer RcAcc  { uvec4 rc_acc[]; };
layout(std430, set = 0, binding = IC_BIND_RES)  buffer RcRes  { vec4 rc_res[]; };
const uint kRcEntries = 1u << 20;
const int  kRcProbe = 8;
// A cell's value is used once it rests on this many one-pixel samples.
const float kIcMinSamples = 32.0;

uint rc_pcg(uint v) {
    uint s = v * 747796405u + 2891336453u;
    uint w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}
int ic_axis(vec3 n, float r) {
    vec3 a = n * n;
    if (r < a.x) return n.x > 0.0 ? 0 : 1;
    if (r < a.x + a.y) return n.y > 0.0 ? 2 : 3;
    return n.z > 0.0 ? 4 : 5;
}
uvec2 ic_hash(vec3 p, int ax, vec3 jit) {
    vec3 an = vec3(0.0);
    an[ax >> 1] = (ax & 1) == 0 ? 1.0 : -1.0;
    float d = distance(p, scene.camera_pos.xyz);
    int lvl = clamp(int(floor(log2(max(d, 1.0) / 8.0))) + 1, 0, 9);
    float cell = 0.5 * exp2(float(lvl));
    ivec3 ip = ivec3(floor((p + an * 0.05) / cell + jit));
    uint h = rc_pcg(uint(ip.x) + rc_pcg(uint(ip.y) + rc_pcg(uint(ip.z) +
             rc_pcg(uint(lvl * 8 + ax) + IC_SALT))));
    uint k = rc_pcg(h ^ 0x9E3779B9u) | 3u;
    return uvec2(h, k);
}
void ic_store(vec3 p, vec3 n, float r_axis, vec3 irradiance, float sky) {
    uvec2 hk = ic_hash(p, ic_axis(n, r_axis), vec3(0.0));
    for (int i = 0; i < kRcProbe; ++i) {
        uint s = (hk.x + uint(i)) & (kRcEntries - 1u);
        uint prev = atomicCompSwap(rc_keys[s], 0u, hk.y);
        if (prev == 0u || prev == hk.y) {
            uvec3 q = uvec3(clamp(irradiance, vec3(0.0), vec3(64.0)) * 256.0 + 0.5);
            atomicAdd(rc_acc[2u * s].x, q.x);
            atomicAdd(rc_acc[2u * s].y, q.y);
            atomicAdd(rc_acc[2u * s].z, q.z);
            atomicAdd(rc_acc[2u * s].w, 1u);
            atomicAdd(rc_acc[2u * s + 1u].x, uint(clamp(sky, 0.0, 1.0) * 256.0 + 0.5));
            return;
        }
    }
}
bool ic_read(uvec2 hk, out vec3 irradiance, out float sky) {
    for (int i = 0; i < kRcProbe; ++i) {
        uint s = (hk.x + uint(i)) & (kRcEntries - 1u);
        uint k = rc_keys[s];
        if (k == hk.y) {
            vec4 r = rc_res[2u * s];
            vec4 r2 = rc_res[2u * s + 1u];
            // (resolved, seen within ~4 s, and settled)
            if (r.w > 0.5 && r.w < 240.0 && r2.y >= kIcMinSamples) { irradiance = r.rgb; sky = r2.x; return true; }
            return false;
        }
        if (k == 0u) return false;
    }
    return false;
}
// jit: three numbers in 0..1; r_axis: one.
bool ic_lookup(vec3 p, vec3 n, vec3 jit, float r_axis, out vec3 irradiance, out float sky) {
    int ax = ic_axis(n, r_axis);
    if (ic_read(ic_hash(p, ax, jit - 0.5), irradiance, sky)) return true;
    return ic_read(ic_hash(p, ax, vec3(0.0)), irradiance, sky);      // (the cell beside is empty or new: the pixel's own)
}
// One pixel in sixteen of a settled cell still traces, and feeds the cell.
bool ic_feeds(uvec2 px, uint frame) {
    return (rc_pcg(px.x * 2713u + px.y * 7919u + frame * 15731u) & 15u) == 0u;
}
