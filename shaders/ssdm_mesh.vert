#version 460
// Screen-space displacement mapping - the displacement mesh pass (vertex stage).
//
// A port of "Screen-Space Raymarched Displacement Mapping" by TechArtAlex (an Unreal Engine
// plugin, shared for free use: patreon.com/TechArtAlex/posts/screen-space-171617266), file
// SSDMDisplacementPass.usf. See ssdm_resolve.frag for the method.
//
// The mesh is drawn a second time, pushed outward along its normals by the shell thickness:
// a shell around the real surface, so that the full-screen march (ssdm_resolve.frag) can
// reach pixels just outside the mesh's own outline.
//
// Same vertex layout and push-constant block as cube.vert (the castle-mesh draw loop pushes
// it); in this pass:  prev_mvp = this frame's view-projection (the one the scene was drawn
// with), color = (tan half fov, aspect), emissive.xyz = camera forward,
// grass_params = (shell thickness m, edge fade distance m, edge handling mode, 0).
#extension GL_GOOGLE_include_directive : require

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUv;

layout(push_constant) uniform PC {
    mat4 mvp;
    mat4 model;
    mat4 prev_mvp;      // HERE: view-projection
    vec4 color;
    vec4 emissive;
    vec4 tex_params;    // x albedo index, y normal index, z uv scale
    vec4 grass_params;  // HERE: x shell thickness (m), y edge fade distance (m), z edge handling mode
} pc;

layout(location = 0) out vec3 vWorldPos;     // on the shell
layout(location = 1) out vec3 vNormal;
layout(location = 2) out vec2 vUv;

void main() {
    vec3 n = normalize(mat3(pc.model) * inNormal);
    vec3 w = (pc.model * vec4(inPosition, 1.0)).xyz + n * pc.grass_params.x;
    vWorldPos = w;
    vNormal   = n;
    vUv       = inUv;
    gl_Position = pc.prev_mvp * vec4(w, 1.0);
}
