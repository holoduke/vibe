// ssdm_common.glsl - shared by the screen-space displacement passes (ssdm_solve.frag,
// ssdm_grid.vert, ssdm_grid.frag): the push-constant block and the images.
//
// Positions in these passes are CAMERA-RELATIVE world space (world - camera): the matrices
// are built that way on the CPU (vk_engine/ssdm.cpp), so a wall 10 km from the origin is
// displaced with the same precision as one at it.

layout(push_constant) uniform SsdmPC {
    mat4 inv_vp;     // clip -> camera-relative world
    mat4 vp;         // camera-relative world -> clip (this frame, as the scene was drawn)
    mat4 prev_vp;    // camera-relative world -> the clip space of the last frame (jitter difference taken out)
    vec4 size;       // render size in pixels, and 1 / it
    vec4 rect;       // xy first pixel of the screen rectangle that holds displaced stonework; z strength of the joint shade
    vec4 par;        // x smoothing radius (m), y reference height (m), z debug view, w world size of a pixel at 1 m
    vec4 sun;        // xyz direction to the sun, w strength of the sun shadow the stones cast on each other (0 at night)
} pc;

layout(set = 0, binding = 0) uniform sampler2D u_hn;      // ssdm_height: normal + height
layout(set = 0, binding = 1) uniform sampler2D u_depth;   // the scene's depth, before displacement
layout(set = 0, binding = 2) uniform sampler2D u_off;     // ssdm_solve: displacement vector + light kept (0 = not displaced)
layout(set = 0, binding = 3) uniform sampler2D u_color;   // copies of the main pass's attachments,
layout(set = 0, binding = 4) uniform sampler2D u_g0;      //   before displacement
layout(set = 0, binding = 5) uniform sampler2D u_g1;

// Camera-relative world position of pixel p at depth z; d = its distance along the view axis.
vec3 ssdm_rel(ivec2 p, float z, out float d) {
    vec2 ndc = (vec2(p) + 0.5) * pc.size.zw * 2.0 - 1.0;
    vec4 q = pc.inv_vp * vec4(ndc, z, 1.0);
    d = 1.0 / q.w;
    return q.xyz / q.w;
}
