#version 460
// lsd_capture.vert - the top-down capture of placed objects (vk_engine/ground_fx.cpp
// gfx_lsd_capture): a mesh part seen straight from above with an orthographic matrix (pc.mvp),
// for the far ground's colour (ground_lsd.glsl). The world's vertex layout and push constants.
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

layout(location = 0) out vec3 vN;

void main() {
    gl_Position = pc.mvp * vec4(inPosition, 1.0);
    vN = mat3(pc.model) * inNormal;
}
