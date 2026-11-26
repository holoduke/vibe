// ground_fx.glsl - the block of the ground's object effects (vk_engine/ground_fx.cpp; state
// vk_engine/state/ground_fx.h GroundFxGpu - keep the two layouts the same):
//   * the paste of meshes into the ground at their contact line (ground_paste.glsl, cube.frag),
//   * the colour of placed objects on the far ground ("large scale detail": a top-down capture
//     round the camera, ground_lsd.glsl, the terrain pass),
//   * the deferred decals (ground_decals.glsl, deferred_lighting.frag).
// The scene set has it at binding 60; deferred_lighting.frag at its own binding 30
// (define GFX_BINDING before the include).
#ifndef GROUND_FX_GLSL
#define GROUND_FX_GLSL
#ifndef GFX_BINDING
#define GFX_BINDING 60
#endif

// One box decal: centre c.xyz, kind c.w; the box's three half axes (world, metres): ax and az
// lie in the surface, ay is the projection axis (out of the surface). ax.w strength 0..1,
// ay.w a seed, az.w the height-blending ratio (1 = covers everything, 0 = only the low spots).
struct GfxDecal { vec4 c; vec4 ax; vec4 ay; vec4 az; };
const int kGfxMaxDecals = 256;
const int kGfxTile = 32;             // decal tiles: pixels a side (8 words of bits per tile)

layout(std430, set = 0, binding = GFX_BINDING) readonly buffer GroundFxB {
    vec4 paste;      // x 1 = on, y the baked ground's level-0 distance t0 (0 = not readable), z its first level in use, w the zone's near height (m)
    vec4 lsd;        // x 1 = on, y z the capture window's corner (world x, z), w its size (m)
    vec4 lsd_fade;   // x fade in from (m), y full by (m), z 1 = VIBE_TEST_DEBRIS (the debris magenta: scatter.frag), w 1 = VIBE_TEST_LSD
    vec4 dec;        // x decals in the list, y tiles across, z tiles down, w 1 = on
    GfxDecal decals[kGfxMaxDecals];
    uint tiles[];    // per tile 8 words: bit i = decal i reaches the tile
} gfx;

#endif
