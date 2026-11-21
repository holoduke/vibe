#version 460
#extension GL_GOOGLE_include_directive : require

// Skinned characters (the soldier: the local body, other players, the fallen). The rest-pose
// mesh and its bone indices / weights are static buffers; the pose is a palette of 3x4 matrices
// (model space, rest -> posed) in one storage buffer, written once a frame for every drawn
// character (player_body.cpp: body_push_pose). pc.grass_params.y = this character's first row;
// .z = 1 + its first row of the frame before (0 = none): where the vertex was, for the motion vector.
// Same outputs as cube.vert: the fragment shaders do not know the difference.

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUv;
// Second stream, packed as floats in a Vertex: (bone0, bone1, bone2), (bone3, weight0, weight1), (weight2, weight3)
layout(location = 3) in vec3 inSkinA;
layout(location = 4) in vec3 inSkinB;
layout(location = 5) in vec2 inSkinC;

layout(push_constant) uniform PC {
    mat4 mvp;
    mat4 model;
    mat4 prev_mvp;
    vec4 color;
    vec4 emissive;
    vec4 tex_params;
    vec4 grass_params;
} pc;

#include "scene_ubo.glsl"

layout(set = 0, binding = 33, std430) readonly buffer BonePalette {
    vec4 bone_rows[];          // three rows per bone
};

layout(location = 0) out vec3 vNormal;
layout(location = 1) flat out vec3 vColor;
layout(location = 2) out vec3 vWorldPos;
layout(location = 3) flat out vec4 vEmissive;
layout(location = 4) out vec2 vUv;
layout(location = 5) flat out vec4 vTexParams;
layout(location = 6) out vec3 vObjectPos;
layout(location = 7) out vec3 vObjectNormal;
layout(location = 8) out vec4 vPrevClip;
layout(location = 9) out vec3 vShellBase;
layout(location = 10) flat out vec4 vVegInfo;

void main() {
    const int base = int(pc.grass_params.y + 0.5);
    const float bone[4] = float[4](inSkinA.x, inSkinA.y, inSkinA.z, inSkinB.x);
    const float weight[4] = float[4](inSkinB.y, inSkinB.z, inSkinC.x, inSkinC.y);
    const vec4 rest = vec4(inPosition, 1.0);
    vec3 p = vec3(0.0), n = vec3(0.0);
    for (int k = 0; k < 4; ++k) {
        const float w = weight[k];
        if (w <= 0.0) continue;
        const int r = base + int(bone[k] + 0.5) * 3;
        const vec4 r0 = bone_rows[r], r1 = bone_rows[r + 1], r2 = bone_rows[r + 2];
        p += vec3(dot(r0, rest), dot(r1, rest), dot(r2, rest)) * w;
        n += vec3(dot(r0.xyz, inNormal), dot(r1.xyz, inNormal), dot(r2.xyz, inNormal)) * w;
    }
    n = normalize(n);
    // The same vertex in last frame's pose: a moving arm has motion of its own, not only the body's.
    vec3 p_prev = p;
    if (pc.grass_params.z > 0.5) {
        const int pbase = int(pc.grass_params.z - 0.5);
        p_prev = vec3(0.0);
        for (int k = 0; k < 4; ++k) {
            const float w = weight[k];
            if (w <= 0.0) continue;
            const int r = pbase + int(bone[k] + 0.5) * 3;
            p_prev += vec3(dot(bone_rows[r], rest), dot(bone_rows[r + 1], rest), dot(bone_rows[r + 2], rest)) * w;
        }
    }

    gl_Position = pc.mvp * vec4(p, 1.0);
    const vec4 wp = pc.model * vec4(p, 1.0);
    vWorldPos = wp.xyz;
    vShellBase = wp.xyz;
    vNormal = mat3(pc.model) * n;
    vColor = pc.color.rgb;
    vEmissive = pc.emissive;
    vUv = inUv;
    vTexParams = pc.tex_params;
    vObjectPos = p;
    vObjectNormal = n;
    vVegInfo = vec4(0.0);
    vPrevClip = pc.prev_mvp * vec4(p_prev, 1.0);
}
