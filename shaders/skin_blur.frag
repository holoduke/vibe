#version 460
// Light under the skin (skin_blur.cpp): one direction of the separable blur over the skin's
// pixels (G-buffer material 4). Every other pixel is discarded - left as it was.
// The light is blurred, not the colour: lit / albedo, blurred, x the albedo again.
// Per colour its own width: red goes furthest under the skin, blue least.

layout(set = 0, binding = 0) uniform sampler2D u_src;
layout(set = 0, binding = 1) uniform sampler2D u_g0;
layout(set = 0, binding = 2) uniform sampler2D u_depth;
layout(push_constant) uniform PC {
    vec4 dir;      // xy: the direction (1, 0) or (0, 1)
    vec4 proj;     // x, y: depth -> metres (P[2][2], P[3][2]); z: pixels a metre covers at 1 m; w: the width, metres
} pc;
layout(location = 0) out vec4 o_color;

float metres(float d) { return pc.proj.y / (d + pc.proj.x); }

void main() {
    ivec2 px = ivec2(gl_FragCoord.xy);
    vec4 g0 = texelFetch(u_g0, px, 0);
    if (int(g0.a * 255.0 + 0.5) != 4) discard;
    float z0 = abs(metres(texelFetch(u_depth, px, 0).r));
    float wpx = min(pc.proj.w * pc.proj.z / max(z0, 0.05), 40.0);
    if (wpx < 0.75) discard;                          // (far off: less than a pixel - nothing to do)
    vec3 alb0 = max(g0.rgb, vec3(0.02));
    vec3 irr0 = texelFetch(u_src, px, 0).rgb / alb0;
    const vec3 kSigma = vec3(0.50, 0.24, 0.15);       // of wpx: red widest
    vec3 sum = irr0, wsum = vec3(1.0);
    ivec2 lim = textureSize(u_src, 0) - 1;
    for (int i = 1; i <= 12; ++i) {
        float off = float(i) / 12.0 * wpx;
        vec3 g = exp(-0.5 * (off / (kSigma * wpx)) * (off / (kSigma * wpx)));
        if (g.r < 1e-3) break;
        for (int s = -1; s <= 1; s += 2) {
            ivec2 q = clamp(px + ivec2(round(pc.dir.xy * off * float(s))), ivec2(0), lim);
            vec4 gq = texelFetch(u_g0, q, 0);
            vec3 irr = irr0;                          // (off the skin: as if it were the centre - the light is not taken from cloth or air)
            float k = 1.0;
            if (int(gq.a * 255.0 + 0.5) == 4) {
                float dz = abs(metres(texelFetch(u_depth, q, 0).r)) - z0;
                k = exp(-(dz * dz) / (0.012 * 0.012));      // (held to the surface: a step of a centimetre or more counts little)
                irr = mix(irr0, texelFetch(u_src, q, 0).rgb / max(gq.rgb, vec3(0.02)), k);
            }
            sum += irr * g;
            wsum += g;
        }
    }
    o_color = vec4(sum / wsum * alb0, 0.0);
}
