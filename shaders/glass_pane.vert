#version 460
// A cracked or broken pane (glass_pane.frag; glass.cpp): one rectangle, its corners in world
// space, uv = metres from its lower left corner.

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUv;

layout(push_constant) uniform PC {
    mat4 vp;
    vec4 origin_w;
    vec4 uaxis_h;
    vec4 vaxis_s;
    vec4 tint_n;
    vec4 misc;
    vec4 hits[7];
} pc;

layout(location = 0) out vec3 vWorld;
layout(location = 1) out vec2 vUv;

void main() {
    vWorld = inPosition;
    vUv    = inUv;
    gl_Position = pc.vp * vec4(inPosition, 1.0);
}
