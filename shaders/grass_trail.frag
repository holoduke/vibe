#version 460
// The trail: where people and cars have just been, as a texture round the player - one
// frame's step of it. See vk_engine/grass_trail.cpp for what it is and who reads it.
//
//   out .xy  direction things were pushed in (world x, z)
//       .z   how hard (0 a walk .. 1 a run or a car)
//       .w   how much the plants here are down (0 standing .. 1 pushed over)
//
// Drawn with the full-screen triangle into the trail texture, reading last frame's.

layout(set = 0, binding = 42) uniform sampler2D u_trail_prev;

// (the engine's one push-constant block, read as what grass_trail.cpp puts in it)
layout(push_constant) uniform PC {
    vec4 pusher[12];    // xy = world x, z; zw = velocity x, z (m/s)
    vec4 org;           // xy = this frame's window corner (world x, z), zw = last frame's
    vec4 par;           // x = dt (s), y = pushers in use, z = metres a texel, w = texels a side
    vec4 radius0;       // radius (m) of pushers 0..3
    vec4 radius1;       // ... 4..7 (the rest: 0.75)
} pc;

layout(location = 0) out vec4 outTrail;

float radius_of(int i) { return i < 4 ? pc.radius0[i] : (i < 8 ? pc.radius1[i - 4] : 0.75); }

void main() {
    ivec2 t  = ivec2(gl_FragCoord.xy);
    vec2  w  = pc.org.xy + (vec2(t) + 0.5) * pc.par.z;          // this texel's place in the world

    // Last frame's value for this place: the window moves in whole texels.
    ivec2 pt = t + ivec2(round((pc.org.xy - pc.org.zw) / pc.par.z));
    vec4  prev = vec4(0.0);
    if (all(greaterThanEqual(pt, ivec2(0))) && all(lessThan(pt, ivec2(int(pc.par.w)))))
        prev = texelFetch(u_trail_prev, pt, 0);
    vec2  dir = prev.xy;
    // Coming back up: half of it in 0.56 s.
    float decay = exp2(-1.784658 * pc.par.x);
    float hard = prev.z * decay;
    float down = prev.w * decay;
    if (down < 0.004) { down = 0.0; hard = 0.0; dir = vec2(0.0); }

    int n = int(pc.par.y + 0.5);
    for (int i = 0; i < n; ++i) {
        vec2  dl = w - pc.pusher[i].xy;
        float r  = radius_of(i);
        float d  = length(dl);
        if (d >= r) continue;
        float f  = 1.0 - smoothstep(0.35 * r, r, d);            // full under it, fading to its rim
        vec2  out_dir = d > 1e-3 ? dl / d : vec2(0.0, 1.0);
        // Outward from its middle, turned toward the way it goes: what is walked through is
        // laid forward and aside, not only aside.
        vec2  vel = pc.pusher[i].zw;
        float sp  = length(vel);
        vec2  nd  = normalize(out_dir + (sp > 0.05 ? vel / sp : vec2(0.0)) * 0.6 * clamp(sp * 0.5, 0.0, 1.0));
        // The older push keeps its say as far as it is still down.
        vec2  mixd = dir * down * (1.0 - f) + nd * f;
        dir  = dot(mixd, mixd) > 1e-8 ? normalize(mixd) : nd;
        down = max(down, f);
        hard = max(hard, f * smoothstep(0.0, 1.0, clamp(max(sp * 0.3, r > 1.5 ? 1.0 : 0.0), 0.0, 1.0)));
    }
    outTrail = vec4(dir, hard, down);
}
