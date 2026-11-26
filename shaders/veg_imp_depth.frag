#version 460
#extension GL_GOOGLE_include_directive : require
// Depth pre-pass for the far trees' cards (vegetation.cpp draws detail level 4 with it): the
// same lookup as cube.frag (veg_leaf.glsl veg_imp_sample), to the bit - the colour pass tests
// depth EQUAL. The card stands in front of the tree; the depth written is the baked
// surface's, so it is never nearer than the card.
layout(location = 2) in vec3 vWorldPos;
layout(location = 6) in vec3 vObjectPos;        // the tree's foot
layout(location = 7) in vec3 vObjectNormal;     // its block in the atlas, the baked sphere's radius, its middle's height
layout(location = 9) in vec3 vShellBase;        // size, y squash, turn
layout(location = 10) flat in vec4 vVegInfo;
layout(depth_greater) out float gl_FragDepth;
#include "scene_ubo.glsl"
#include "veg_leaf.glsl"
layout(set = 0, binding = 4) uniform sampler2D u_normal[19];      // 15 / 18: the baked tree impostors
layout(push_constant) uniform PC {
    mat4 mvp;
    mat4 model;
    mat4 prev_mvp;
    vec4 color;
    vec4 emissive;
    vec4 tex_params;
    vec4 grass_params;
} pc;
void main() {
    gl_FragDepth = gl_FragCoord.z;
    float px = max(length(dFdx(vWorldPos)), length(dFdy(vWorldPos)));
    if (vVegInfo.x > 0.0) {
        float n = fract(52.9829189 * fract(dot(gl_FragCoord.xy + 5.588238 * float(uint(scene.rt_flags.w) & 7u),
                                                  vec2(0.06711056, 0.00583715))));
        if (vVegInfo.y > 0.5 ? (n >= vVegInfo.x) : (n < vVegInfo.x)) discard;
    }
    vec3 col, nrm, surf; float leaf;
    vec3 info = vec3(vObjectNormal.x, abs(vObjectNormal.y), vObjectNormal.z);      // (radius < 0: a far card - veg.vert)
    if (!veg_imp_sample(u_normal[15], u_normal[18], vWorldPos, scene.camera_pos.xyz, vObjectPos, vShellBase, info, px,
                        col, nrm, leaf, surf)) discard;
    if (veg_imp_bare(surf, leaf, vVegInfo.z, scene.season.w)) discard;
    precise vec4 sc = pc.mvp * vec4(surf, 1.0);
    precise float z = sc.z / sc.w;
    if (vObjectNormal.y > 0.0) gl_FragDepth = max(gl_FragCoord.z, z);      // (a far card keeps its own depth)
}
