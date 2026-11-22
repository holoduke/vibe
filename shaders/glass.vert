#version 460
// Glass (glass.frag): a static-mesh part's vertices, placed by pc.model.

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUv;

layout(push_constant) uniform PC {
    mat4 mvp;
    mat4 model;
    mat4 prev_mvp;
    vec4 color;
    vec4 emissive;
    vec4 tex_params;
    vec4 grass_params;
} pc;

layout(location = 0) out vec3 vWorld;
layout(location = 1) out vec3 vNormal;
layout(location = 2) out vec2 vUv;

void main() {
    vWorld  = (pc.model * vec4(inPosition, 1.0)).xyz;
    vNormal = mat3(pc.model) * inNormal;
    vUv     = inUv;
    gl_Position = pc.mvp * vec4(inPosition, 1.0);
}
