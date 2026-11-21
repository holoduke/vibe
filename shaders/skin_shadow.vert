#version 460
#extension GL_GOOGLE_include_directive : require

// Skinned characters into a shadow map (depth only): the same skinning as cube_skin.vert, with
// pc.mvp = light view-projection x the character's model matrix and pc.grass_params.y = its
// first row in the bone palette. Drawn into the small-object shadow map (clutter_shadow.cpp).

layout(location = 0) in vec3 inPosition;
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

void main() {
    const int base = int(pc.grass_params.y + 0.5);
    const float bone[4] = float[4](inSkinA.x, inSkinA.y, inSkinA.z, inSkinB.x);
    const float weight[4] = float[4](inSkinB.y, inSkinB.z, inSkinC.x, inSkinC.y);
    const vec4 rest = vec4(inPosition, 1.0);
    vec3 p = vec3(0.0);
    for (int k = 0; k < 4; ++k) {
        const float w = weight[k];
        if (w <= 0.0) continue;
        const int r = base + int(bone[k] + 0.5) * 3;
        p += vec3(dot(bone_rows[r], rest), dot(bone_rows[r + 1], rest), dot(bone_rows[r + 2], rest)) * w;
    }
    gl_Position = pc.mvp * vec4(p, 1.0);
}
