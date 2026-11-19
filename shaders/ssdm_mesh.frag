#version 460
// Screen-space displacement mapping - the displacement mesh pass (fragment stage).
// Port of SSDMDisplacementPass.usf (TechArtAlex, see ssdm_mesh.vert). Written per pixel of the
// shell, for ssdm_resolve.frag:
//
//   outDisp    .rg  D: the screen-UV offset from this shell pixel to where its surface point
//                   really lies (the shell point moved INTO the surface by the displacement)
//              .b   N.V
//              .a   1 = covered by a displaced mesh
//   outRay     .rg  the march direction (the view ray's direction along the surface, in
//                   screen UV) times the search length (the full depth range projected on it)
//              .ba  contraction: the UV offset from this shell pixel to the pixel of the
//                   ORIGINAL (un-inflated) surface - where the scene colour is taken from
//   outHitZ    .r   depth of the displaced point, .g depth of the original surface point
//
// Differences from the Unreal original, all conventions: Vulkan clip space (y down, so UV =
// ndc * 0.5 + 0.5 with no flip), depth 0 near .. 1 far, metres (the original's 5 cm margin on
// the search depth is 0.05 here). The height comes from the wall's own height texture, with
// exactly the UVs cube.frag shades the mesh with, so the displaced stones are the drawn ones.
#extension GL_GOOGLE_include_directive : require
#include "scene_ubo.glsl"

layout(set = 0, binding = 12) uniform sampler2D u_height[6];

layout(push_constant) uniform PC {
    mat4 mvp;
    mat4 model;
    mat4 prev_mvp;      // view-projection
    vec4 color;         // HERE: x tan(half vertical fov), y aspect (width / height)
    vec4 emissive;      // HERE: xyz the camera's forward direction
    vec4 tex_params;
    vec4 grass_params;  // x shell thickness (m), y edge fade distance (m), z edge handling mode
} pc;

layout(location = 0) in vec3 vWorldPos;
layout(location = 1) in vec3 vNormal;
layout(location = 2) in vec2 vUv;

layout(location = 0) out vec4 outDisp;
layout(location = 1) out vec4 outRay;
layout(location = 2) out vec2 outHitZ;

int height_idx_for_albedo(int a) {         // (cube.frag)
    if (a == 1) return 0;
    if (a == 4) return 1;
    return -1;
}

void main() {
    const float shell = pc.grass_params.x;
    vec3  P  = vWorldPos;
    vec3  Nv = normalize(vNormal);
    vec3  V  = normalize(scene.camera_pos.xyz - P);            // toward the camera
    mat4  VP = pc.prev_mvp;

    vec4 origClip = VP * vec4(P, 1.0);
    vec2 origNdc  = origClip.xy / origClip.w;

    // The displacement: how far below the shell the surface lies here. The height texture's
    // 1 is the top of the stones (on the shell), 0 the bottom of the joints (the real mesh).
    float h = 1.0;
    {
        int ai  = int(pc.tex_params.x + 0.5);
        int hix = height_idx_for_albedo(ai);
        if (hix >= 0) {
            vec2 muv = vUv * pc.tex_params.z;
            if (hix == 0 || hix == 2) muv.x *= 0.5;            // 2:1 masonry (as cube.frag)
            h = texture(u_height[hix], muv).r;
        }
    }
    float disp = (1.0 - h) * shell;

    // Edge handling mode 1: toward the screen's edges the displacement is eased to the full
    // shell depth (the flat original surface), where the march would otherwise look for its
    // source off-screen. The fade width is a world distance, seen at this pixel's depth.
    if (int(pc.grass_params.z + 0.5) == 1) {
        vec2  uv0      = origNdc * 0.5 + 0.5;
        float fadeDist = max(pc.grass_params.y, 1e-4);
        float aspect   = max(pc.color.y, 1e-6);
        float slope    = clamp(abs(Nv.y), 0.0, 1.0);
        float pixDepth = max(abs(origClip.w), 1e-4);
        float tanHalf  = max(pc.color.x, 1e-6);
        float viewH    = max(pixDepth * tanHalf * 2.0, 1e-6);
        float yFadeUv  = fadeDist / viewH;
        float xFadeUv  = yFadeUv / aspect;
        float xFade    = min(uv0.x, 1.0 - uv0.x) / max(xFadeUv, 1e-6);
        float yFade    = min(uv0.y, 1.0 - uv0.y) / max(yFadeUv, 1e-6);
        float axis     = mix(xFade, yFade, slope);             // walls: the x edges; floors: the y edges
        float slopeMid = clamp(1.0 - abs(2.0 * slope - 1.0), 0.0, 1.0);
        float slopeMix = pow(slopeMid, 0.1);
        float facing   = clamp(abs(dot(Nv, pc.emissive.xyz)), 0.0, 1.0);
        float facingMix = smoothstep(-1.0, 0.33, facing);
        float combined = clamp(max(slopeMix, facingMix), 0.0, 1.0);
        float fin      = mix(axis, min(xFade, yFade), combined);
        float blend    = smoothstep(max(shell, 1e-4), 0.0, fin);
        disp = mix(disp, max(disp, shell), blend);
    }

    // D: shell point -> displaced point, in screen UV.
    vec3 Pd       = P - Nv * disp;
    vec4 dispClip = VP * vec4(Pd, 1.0);
    vec2 D        = (dispClip.xy / dispClip.w - origNdc) * 0.5;

    // The march direction: the view ray's direction along the surface (its component in the
    // tangent plane), as the parallax of a point sunk by the search depth.
    vec3  Vt   = V - Nv * dot(V, Nv);
    float vtl  = length(Vt);
    vec3  rayW = (vtl > 1e-5 * max(abs(dot(V, Nv)), 1e-4)) ? -(Vt / vtl) : Nv;
    float maxDepth = shell + 0.05;
    vec4  parClip = VP * vec4(P - rayW * maxDepth, 1.0);
    vec2  parUv   = (parClip.xy / parClip.w - origNdc) * 0.5;
    float parLen  = length(parUv);
    vec2  parDir  = (parLen > 1e-7) ? parUv / parLen : vec2(1.0, 0.0);
    // The search length: the full depth range along the normal, projected on that direction
    // (the t at which the height line h == t is met at full depth).
    vec4  nrmClip = VP * vec4(P + Nv * maxDepth, 1.0);
    vec2  nrmUv   = (nrmClip.xy / nrmClip.w - origNdc) * 0.5;
    float depthLen = abs(dot(nrmUv, parDir));

    // Contraction: shell point -> the original surface point under it.
    vec4 unexClip = VP * vec4(P - Nv * shell, 1.0);
    vec2 contr    = (unexClip.xy / unexClip.w - origNdc) * 0.5;

    outDisp = vec4(D, clamp(dot(V, Nv), 0.0, 1.0), 1.0);
    outRay  = vec4(parDir * depthLen, contr);
    outHitZ = vec2(dispClip.z / max(dispClip.w, 1e-6), unexClip.z / max(unexClip.w, 1e-6));
}
