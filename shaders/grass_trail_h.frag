#version 460
// The trail's second picture: between which heights the body stood that pushed here.
// the reference game keeps one (interaction shader GenerateTrailHeightAndDeformedTrailCS, read from
// the compiled shader; its values come from a top-down depth picture of everything that
// moves, ours from the list of pushers): a blade only reacts when its root lies between the
// body's underside (less a metre) and its top (plus 20 cm) - so grass under a bridge stays up
// when somebody crosses the bridge, and nothing bends under a jump.
//
//   out .x  the underside (world y, m)     .y  the top
//   (nothing here: x = +1e5, y = -1e5)
// Kept for as long as the trail itself says the plants here are down (its .w > 0.01).
//
// Drawn right after grass_trail.frag, with the same window.

layout(set = 0, binding = 41) uniform sampler2D u_trail;        // this frame's trail (just drawn)
layout(set = 0, binding = 44) uniform sampler2D u_height_prev;  // last frame's heights

layout(push_constant) uniform PC {
    vec4 pusher[12];    // xy = world x, z; z = its underside (world y), w = its top
    vec4 org;           // xy = this frame's window corner (world x, z), zw = last frame's
    vec4 par;           // x = dt (s), y = pushers in use, z = metres a texel, w = texels a side
    vec4 radius0;       // radius (m) of pushers 0..3
    vec4 radius1;       // ... 4..7 (the rest: 0.6, a person)
} pc;

layout(location = 0) out vec4 outHeight;

float radius_of(int i) { return i < 4 ? pc.radius0[i] : (i < 8 ? pc.radius1[i - 4] : 0.6); }

void main() {
    ivec2 t  = ivec2(gl_FragCoord.xy);
    vec2  w  = pc.org.xy + (vec2(t) + 0.5) * pc.par.z;
    ivec2 pt = t + ivec2(round((pc.org.xy - pc.org.zw) / pc.par.z));
    vec2  hh = vec2(1e5, -1e5);
    if (all(greaterThanEqual(pt, ivec2(0))) && all(lessThan(pt, ivec2(int(pc.par.w)))) &&
        texelFetch(u_trail, t, 0).w > 0.01)
        hh = texelFetch(u_height_prev, pt, 0).xy;
    int n = int(pc.par.y + 0.5);
    for (int i = 0; i < n; ++i) {
        if (distance(w, pc.pusher[i].xy) >= radius_of(i)) continue;
        hh = vec2(min(hh.x, pc.pusher[i].z), max(hh.y, pc.pusher[i].w));
    }
    outHeight = vec4(hh, 0.0, 0.0);
}
