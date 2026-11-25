#version 460
// Screen-space displacement mapping - the grid pass (fragment stage): the G-buffer of the
// pixel this surface point came from, at the place it is displaced to. See ssdm_grid.vert.
#extension GL_GOOGLE_include_directive : require
#include "ssdm_common.glsl"

layout(location = 0) noperspective in vec2 vSrc;
layout(location = 1) in vec4 vPrevClip;
layout(location = 2) in float vKeep;

layout(location = 0) out vec4 outColor;
layout(location = 1) out vec2 outMotion;
layout(location = 2) out vec4 outGBuffer0;
layout(location = 3) out vec4 outGBuffer1;

void main() {
    ivec2 lim = ivec2(pc.size.xy) - 1;
    ivec2 s = clamp(ivec2(vSrc), ivec2(0), lim);
    outColor    = texelFetch(u_color, s, 0);
    outGBuffer0 = texelFetch(u_g0, s, 0);
    outGBuffer1 = texelFetch(u_g1, s, 0);
    // Where the grid is pulled apart - the side of a stone, which the flat wall never showed -
    // one source pixel covers several here. Its colour is taken smoothly between the source
    // pixels instead of in blocks, and darkened: it is a face turned away, down in a joint.
    float span = max(length(dFdx(vSrc)), length(dFdy(vSrc)));        // source pixels per pixel
    // (pulled over more than ten pixels it is a tear, not a stone's side: leave the flat wall)
    if (span < 0.1) discard;
    float pulled = 1.0 - smoothstep(0.35, 0.8, span);
    if (pulled > 0.0) {
        vec2  t  = vSrc - 0.5;
        ivec2 b  = ivec2(floor(t));
        vec2  f  = t - vec2(b);
        vec3  c00 = texelFetch(u_g0, clamp(b, ivec2(0), lim), 0).rgb;
        vec3  c10 = texelFetch(u_g0, clamp(b + ivec2(1, 0), ivec2(0), lim), 0).rgb;
        vec3  c01 = texelFetch(u_g0, clamp(b + ivec2(0, 1), ivec2(0), lim), 0).rgb;
        vec3  c11 = texelFetch(u_g0, clamp(b + ivec2(1, 1), ivec2(0), lim), 0).rgb;
        outGBuffer0.rgb = mix(outGBuffer0.rgb, mix(mix(c00, c10, f.x), mix(c01, c11, f.x), f.y), pulled)
                        * (1.0 - 0.35 * pulled);
    }
    // The raymarched ground (material 8, ssdm_ground.frag) is lit already: the lighting pass
    // hands its colour on untouched, so what is done to the albedo above and below has to be
    // done to the colour itself.
    bool prelit = int(outGBuffer0.a * 255.0 + 0.5) == 8;
    if (prelit) {
        if (pulled > 0.0) {
            vec2  t  = vSrc - 0.5;
            ivec2 b  = ivec2(floor(t));
            vec2  f  = t - vec2(b);
            vec3  c00 = texelFetch(u_color, clamp(b, ivec2(0), lim), 0).rgb;
            vec3  c10 = texelFetch(u_color, clamp(b + ivec2(1, 0), ivec2(0), lim), 0).rgb;
            vec3  c01 = texelFetch(u_color, clamp(b + ivec2(0, 1), ivec2(0), lim), 0).rgb;
            vec3  c11 = texelFetch(u_color, clamp(b + ivec2(1, 1), ivec2(0), lim), 0).rgb;
            outColor.rgb = mix(outColor.rgb, mix(mix(c00, c10, f.x), mix(c01, c11, f.x), f.y), pulled)
                         * (1.0 - 0.35 * pulled);
        }
        outColor.rgb *= vKeep;
    }
    // The stones shade each other (ssdm_solve): into the albedo, as the parallax relief did -
    // the lighting pass only knows the flat wall's shadow.
    outGBuffer0.rgb *= vKeep;
    // The displaced point's own motion (the stonework stands still: it is the camera's).
    vec2 cur_uv = gl_FragCoord.xy * pc.size.zw;
    outMotion = vPrevClip.w > 0.0 ? cur_uv - (vPrevClip.xy / vPrevClip.w * 0.5 + 0.5) : vec2(0.0);
    if (pc.par.z > 0.5) {
        // Debug view: red = moved by more than half a pixel, green = on the grid but in place.
        bool moved = distance(vSrc, gl_FragCoord.xy) > 0.5;
        outGBuffer0.rgb = moved ? vec3(0.8, 0.1, 0.1) : vec3(0.1, 0.6, 0.1);
        outColor.rgb    = outGBuffer0.rgb;
    }
}
