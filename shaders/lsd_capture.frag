#version 460
// lsd_capture.frag - the colour of a placed object seen from above, as the far ground will
// carry it (ground_lsd.glsl): the material's mean colour (its texture's smallest mip x its tint),
// the rock formations' tone (rock_form.glsl rf_colour, its mean), a plain tint where there is
// no texture. Colour only: the far ground lights it with its own normal. Out: sqrt(colour), 1.
layout(location = 0) in vec3 vN;
layout(location = 0) out vec4 outColor;

layout(push_constant) uniform PC {
    mat4 mvp;
    mat4 model;
    mat4 prev_mvp;
    vec4 color;
    vec4 emissive;
    vec4 tex_params;
    vec4 grass_params;
} pc;

const int kTextureCount = 48;     // (cube.frag kTextureCount)
layout(set = 0, binding = 3) uniform sampler2D u_albedo[kTextureCount];

void main() {
    int ai = int(pc.tex_params.x + 0.5 * sign(pc.tex_params.x));
    int ni = int(pc.tex_params.y + 0.5 * sign(pc.tex_params.y));
    vec3 c = pc.color.rgb;
    if (ni == -9)       c = vec3(0.165, 0.152, 0.133) * pc.color.rgb;        // rock formation (its mean tone)
    else if (ni == -10) c = vec3(0.74, 0.73, 0.68) * pc.color.rgb;           // chalk
    else if (ai >= 0 && ai < kTextureCount) c = textureLod(u_albedo[ai], vec2(0.5), 14.0).rgb * pc.color.rgb;
    // (what faces the sky is seen most from above: a slanting face a little darker, as a mean of its shade)
    float up = clamp(normalize(vN).y, 0.0, 1.0);
    c *= mix(0.75, 1.0, up);
    outColor = vec4(sqrt(clamp(c, vec3(0.0), vec3(1.0))), 1.0);
}
