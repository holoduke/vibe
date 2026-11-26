#version 460
// Glare sprites, step 2 (post_fx.cpp): one quad per sprite found by glare_find.comp, drawn
// additively into the post colour (HDR, before the exposure and the tone curve).
layout(location = 0) out vec2 v_q;        // -1 .. 1 over the quad
layout(location = 1) out vec4 v_col;      // rgb: colour x brightness x weight
struct Sprite { vec4 pos; vec4 col; };
layout(std430, set = 0, binding = 0) readonly buffer Sprites {
    uint vertex_count, instance_count, first_vertex, first_instance;
    Sprite s[];
} spr;
layout(push_constant) uniform PC { vec4 p; } pc;   // xy: output size, z: strength
void main() {
    const vec2 k[6] = vec2[](vec2(-1, -1), vec2(1, -1), vec2(1, 1), vec2(-1, -1), vec2(1, 1), vec2(-1, 1));
    uint i = uint(gl_InstanceIndex);
    if (i >= 1000u) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); v_q = vec2(0.0); v_col = vec4(0.0); return; }
    Sprite sp = spr.s[i];
    vec2 q = k[gl_VertexIndex];
    vec2 px = sp.pos.xy + q * sp.pos.z;
    gl_Position = vec4(px / pc.p.xy * 2.0 - 1.0, 0.0, 1.0);
    v_q = q;
    v_col = vec4(sp.col.rgb * sp.col.w * sp.pos.w * pc.p.z, 0.0);
}
