#version 460
#extension GL_GOOGLE_include_directive : require
// Depth-prepass fragment for GPU vegetation: same LOD-fade dither and
// impostor silhouette cut as cube.frag so the prepass depth matches the
// colour pass exactly (LESS_OR_EQUAL).
layout(location = 0) in vec3 vNormal;
layout(location = 2) in vec3 vWorldPos;
layout(location = 4) in vec2 vUv;
layout(location = 6) in vec3 vObjectPos;
layout(location = 7) in vec3 vObjectNormal;
layout(location = 10) flat in vec4 vVegInfo;
#include "scene_ubo.glsl"
#include "veg_leaf.glsl"
layout(set = 0, binding = 3) uniform sampler2D u_albedo[19];
layout(set = 0, binding = 4) uniform sampler2D u_normal[19];      // 15 / 18: the baked tree impostors (colour + coverage, normal + leaf mark)
void main() {
    if (vVegInfo.x > 0.0) {
        float n = fract(52.9829189 * fract(dot(gl_FragCoord.xy + 5.588238 * float(uint(scene.rt_flags.w) & 7u),
                                                  vec2(0.06711056, 0.00583715))));
        if (vVegInfo.y > 0.5 ? (n >= vVegInfo.x) : (n < vVegInfo.x)) discard;
    }
    // (Rocks are plain meshes: they take part in the depth pre-pass.)
    if (vVegInfo.z < -0.5) return;
    if (vVegInfo.z > 0.5) {
        // The baked card (veg_impostor.cpp): the same cut as cube.frag, to the bit.
        float rows = float(textureSize(u_normal[15], 0).y) / kImpTile;
        vec2 iuv = veg_imp_uv(vUv, vObjectNormal, veg_imp_noise(gl_FragCoord.xy, uint(scene.rt_flags.w)), rows);
        float ilod = veg_imp_lod(vUv);
        if (textureLod(u_normal[15], iuv, ilod).a < 0.26) discard;
        if (veg_imp_bare(iuv, textureLod(u_normal[18], iuv, ilod).a, vVegInfo.z, scene.season.w)) discard;
    } else {
        int vm = int(vUv.y + 0.5);
        if ((vm == 2 || vm == 3) && veg_leaf_cut(vObjectPos, vWorldPos, vNormal, scene.camera_pos.xyz)) discard;
        if (vm == 7 && texture(u_albedo[15], veg_card_uv(vObjectNormal.xy, vObjectNormal.z, 0.5 * max(fwidth(vObjectNormal.x), fwidth(vObjectNormal.y))), -0.7).a < 0.33) discard;
    }
}
