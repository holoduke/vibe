#version 460
// Screen-space displacement mapping - the height pass for the GROUND (the raymarched terrain).
//
// The stonework's height pass (ssdm_height) draws the displaced meshes once more. The ground is
// not a mesh: terrain_raymarch.frag itself says, per pixel, how far the material it shaded
// there stands out of the drawn surface (cobbles, the stones and clods of the scanned soil,
// rock, roadside gravel) - in the blue channel of the second G-buffer layer, which nothing
// reads for an already-lit pixel except as the two marks 0.6 / 0.7 (rr_guides.comp):
//     b = 0.7 + 0.3 * height / kGndSsdmMax        (0.7 = out of reach, not on the grid)
// This pass turns that into what ssdm_displace.glsl reads - normal + height in metres - for
// every terrain pixel (material 8), over the whole screen, drawn into the same target after
// the meshes. The reference game does the same: its terrain pixel shader writes a height
// byte into the G-buffer and its screen-space tessellation pass moves the pixels by it.
//
// The direction the ground is displaced in is its geometric normal, taken from the depth
// buffer (the terrain writes no normal) and leaned toward straight up on ground that is not
// steep: the soil's height texture is laid on from above, and the small relief already in
// the marched surface (rills, stones) must not turn the direction from pixel to pixel.
//
// Rule kept (docs/ARCHITECTURE.md): the ground only rises from the drawn surface.

layout(push_constant) uniform SsdmPC {      // (ssdm_common.glsl; not included: this pass must
    mat4 inv_vp;                            //  not declare the height target it draws into)
    mat4 vp;
    mat4 prev_vp;
    vec4 size;
    vec4 depth;
    vec4 view;
    vec4 misc;       // w: the height 0..1 a full value in the G-buffer stands for
} pc;

layout(set = 0, binding = 1) uniform sampler2D u_depth;
layout(set = 0, binding = 4) uniform sampler2D u_g0;
layout(set = 0, binding = 5) uniform sampler2D u_g1;

layout(location = 0) out vec4 outHeight;

vec3 rel(ivec2 p, out float z) {
    z = texelFetch(u_depth, p, 0).r;
    vec2 ndc = (vec2(p) + 0.5) * pc.size.zw * 2.0 - 1.0;
    vec4 q = pc.inv_vp * vec4(ndc, z, 1.0);
    return q.xyz / q.w;
}

void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    vec4  g0 = texelFetch(u_g0, p, 0);
    if (int(g0.a * 255.0 + 0.5) != 8) discard;                   // (not the raymarched ground)
    float e = (texelFetch(u_g1, p, 0).b - 0.7) / 0.3;
    ivec2 lim = ivec2(pc.size.xy) - 1;
    float z0, zl, zr, zu, zd;
    vec3 P0 = rel(p, z0);
    vec3 Pl = rel(max(p - ivec2(1, 0), ivec2(0)), zl), Pr = rel(min(p + ivec2(1, 0), lim), zr);
    vec3 Pu = rel(max(p - ivec2(0, 1), ivec2(0)), zu), Pd = rel(min(p + ivec2(0, 1), lim), zd);
    // (toward the neighbour that lies on the same surface: the smaller step in depth)
    vec3 dx = abs(zl - z0) < abs(zr - z0) ? P0 - Pl : Pr - P0;
    vec3 dy = abs(zu - z0) < abs(zd - z0) ? P0 - Pu : Pd - P0;
    vec3 n = cross(dx, dy);
    float nl = length(n);
    n = nl > 1e-12 ? n / nl : vec3(0.0, 1.0, 0.0);
    if (dot(n, P0) > 0.0) n = -n;                                // (the side that is seen)
    n = normalize(n + vec3(0.0, 2.5, 0.0) * smoothstep(0.45, 0.8, n.y));
    // (the terrain shader's smallest value within reach is 0.006: "not raised"; 0.7 and below:
    //  out of its reach - fed with height 0, so the ground still counts as a neighbour)
    float h = max(e - 0.006, 0.0) * pc.misc.w;
    outHeight = vec4(n, round(clamp(h, 0.0, 1.0) * 255.0) / 255.0);
}
