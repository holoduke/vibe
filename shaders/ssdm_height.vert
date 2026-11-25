#version 460
// Screen-space displacement mapping - the height pass (vertex stage). See ssdm_grid.vert for
// the method and vk_engine/ssdm.cpp for the pass sequence.
//
// The displaced stonework is drawn once more, exactly where it stands, against the scene's
// depth: only its visible pixels pass. Same vertex layout and push-constant block as
// cube.vert (the castle-mesh draw loop pushes it).

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUv;

layout(push_constant) uniform PC {
    mat4 mvp;
    mat4 model;
    mat4 prev_mvp;
    vec4 color;
    vec4 emissive;
    vec4 tex_params;    // x albedo index, y normal index, z uv scale
    vec4 grass_params;  // HERE: x displacement amplitude (m), y fade start (m), z fade end (m)
} pc;

layout(location = 0) out vec3 vWorldPos;
layout(location = 1) out vec3 vNormal;
layout(location = 2) out vec2 vUv;

void main() {
    vWorldPos = (pc.model * vec4(inPosition, 1.0)).xyz;
    vNormal   = mat3(pc.model) * inNormal;
    vUv       = inUv;
    gl_Position = pc.mvp * vec4(inPosition, 1.0);
    // Toward the camera by a hair: this is the surface already in the depth buffer, and the
    // test must pass on it whatever the last bit of the two vertex shaders' arithmetic says.
    gl_Position.z -= 4e-7 * gl_Position.w;
}
