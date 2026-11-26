#version 460
// Glare sprite shape (the reference game's): a 4-point star - arms pow(1 - |x|, 10) across,
// pow(1 - r, 7) along - and a thin horizontal streak in three spectral bands (red a little
// above, blue a little below). Output alpha 0: premultiplied blend = purely additive, the
// colour's alpha (sky coverage under DLSS) is kept.
layout(location = 0) in vec2 v_q;
layout(location = 1) in vec4 v_col;
layout(location = 0) out vec4 o_col;
void main() {
    vec2 a = abs(v_q);
    float r = length(v_q);
    if (r >= 1.0) discard;
    float along = pow(1.0 - r, 7.0);
    float star = along * (pow(max(1.0 - a.x * 6.0, 0.0), 10.0) + pow(max(1.0 - a.y * 6.0, 0.0), 10.0));
    float core = exp(-r * r * 60.0) * 0.5;
    float ex = pow(1.0 - a.x, 3.0);
    vec3 band = vec3(exp(-pow((v_q.y - 0.012) * 90.0, 2.0)),
                     exp(-pow(v_q.y * 90.0, 2.0)),
                     exp(-pow((v_q.y + 0.012) * 90.0, 2.0)));
    vec3 streak = band * ex * 0.35;
    o_col = vec4(v_col.rgb * ((star + core) + streak), 0.0);
}
