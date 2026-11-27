// PEOPLE: how a skinned character's surface is coloured (cube.frag, the skinned branch of
// cube_main_surfaces.glsl). docs/ARCHITECTURE.md "People who look like people".
//
// The character files (tools/character/make_people.py) carry a REGION in the whole part of each
// vertex's u: u = region * 4 + u_atlas. Skin, the garments to be tinted, hair and the iris are
// stored normalised in the atlas (a linear 0.5 is the tile's mean); their colour is the person's,
// from the LOOK BLOCK: six rows of the bone palette after the 62 bones of the character's region
// (player_body.cpp people_look), found through vTexParams.z = the character's first palette row.
//   L0 skin tone (linear albedo), a: 1 = the older skin's texture
//   L1 hair colour,               a: scalp hair (0 shaved .. 1 a full head)
//   L2 the unit's shade (camouflage, vest, helmet; 1 = as painted), a: beard
//   L3 top garment colour,        a: moustache
//   L4 bottom garment colour,     a: stubble
//   L5 iris colour,               a: shoes (0 black .. 1 brown)
// What it writes besides the colour: the G-buffer material (4 skin, 9 hair, 1 the rest - the
// deferred pass lights skin and hair their own way) and the roughness.

layout(set = 0, binding = 33, std430) readonly buffer PeopleLook { vec4 people_rows[]; };

const int   kPeopleLookRow = 62 * 3;           // (player_body.cpp kLookRow)
const float kAtlas = 4096.0;
// atlas tiles (make_people.py LAYOUT), in texture space: origin (u, v of the tile's bottom row
// in the image the game samples - v is flipped there) and size
const vec2  kSkinYoung = vec2(0.0, 2048.0) / kAtlas, kSkinOld = vec2(1024.0, 2048.0) / kAtlas, kMaskT = vec2(2048.0, 3584.0) / kAtlas;
const float kSkinSize = 1024.0 / kAtlas, kMaskSize = 512.0 / kAtlas;

int   g_people_mat = 0;        // 0 = not a person's pixel (or the plain material)
float g_people_rough = -1.0;

float people_hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }

// The image is stored bottom row first: a tile at atlas (x, y) is at v = 1 - (y + s) / 4096 .. 1 - y / 4096.
vec2 people_tile_uv(vec2 uv, vec2 from, float from_size, vec2 to, float to_size) {
    vec2 a = vec2(uv.x, 1.0 - uv.y);                         // back to the atlas' own rows
    vec2 t = (a - from) / from_size;
    vec2 b = to + t * to_size;
    return vec2(b.x, 1.0 - b.y);
}

vec3 people_surface(sampler2D atlas, vec2 vuv, int base, inout vec3 N) {
    int   region = int(floor(vuv.x * 0.25));
    vec2  uv = vec2(vuv.x - 4.0 * float(region), vuv.y);
    vec4  t = texture(atlas, uv);
    g_people_mat = 1;
    g_people_rough = 0.85;
    if (region == 0) return t.rgb;
    vec4 L0 = people_rows[base + kPeopleLookRow + 0], L1 = people_rows[base + kPeopleLookRow + 1],
         L2 = people_rows[base + kPeopleLookRow + 2], L3 = people_rows[base + kPeopleLookRow + 3],
         L4 = people_rows[base + kPeopleLookRow + 4], L5 = people_rows[base + kPeopleLookRow + 5];
    if (region == 1) {
        // SKIN: the detail of the younger or the older texture, the person's own tone; hair and
        // beard painted where they grow (the mask in the skin's texture space)
        vec2 suv = uv;
        if (L0.a > 0.5) suv = people_tile_uv(uv, kSkinYoung, kSkinSize, kSkinOld, kSkinSize);
        vec3 d = (L0.a > 0.5 ? texture(atlas, suv).rgb : t.rgb) * 2.0;
        vec3 alb = d * L0.rgb;
        vec3 m = texture(atlas, people_tile_uv(uv, kSkinYoung, kSkinSize, kMaskT, kMaskSize)).rgb;
        m = pow(max(m, vec3(0.0)), vec3(1.0 / 2.2));           // (stored as data, sampled as colour)
        // a strand of hair, a pore: noise at the texel scale of the skin (about 0.3 mm on a face)
        vec2  q = uv * kAtlas * 2.0;
        float n = people_hash(floor(q)), n2 = people_hash(floor(q * 0.5) + 17.0);
        float scalp = m.r * L1.a, beard = max(m.g * L2.a, m.b * max(L3.a, L2.a));
        float stub = max(m.g, m.b) * L4.a + m.r * (L1.a < 0.05 ? 0.35 : 0.0);      // (shaved: the scalp shows the dark of the roots)
        vec3  hc = L1.rgb;
        // a cropped head of hair or a beard: the colour of the hair over the skin, thinner at the edge
        float cover = clamp(max(scalp, beard), 0.0, 1.0);
        float dense = smoothstep(0.15, 0.85, cover) * mix(0.75, 1.0, n);
        alb = mix(alb, hc * mix(0.7, 1.15, n2), dense * 0.92);
        alb *= 1.0 - clamp(stub, 0.0, 1.0) * 0.45 * mix(0.6, 1.0, n) * (1.0 - dense);
        // cavities (the darker detail of the texture) hold less oil: rougher
        float cav = clamp(dot(d, vec3(0.333)), 0.4, 1.3);
        g_people_rough = clamp(0.50 + (1.0 - cav) * 0.35, 0.35, 0.8);
        g_people_rough = mix(g_people_rough, 0.72, dense);
        g_people_mat = dense > 0.6 ? 9 : 4;
        return alb;
    }
    if (region == 2) { g_people_rough = 0.88; return t.rgb * L2.rgb; }
    if (region == 3) { g_people_rough = 0.86; return t.r * 2.0 * L3.rgb; }
    if (region == 4) { g_people_rough = 0.80; return t.r * 2.0 * L4.rgb; }
    if (region == 5 || region == 6) {
        // HAIR CARDS, BROWS: the coverage dithered (the temporal filter makes it soft; no blending)
        float k = people_hash(gl_FragCoord.xy + vec2(float(scene.rt_flags.w & 63) * 3.17, 0.0));
        if (t.a < mix(0.25, 0.75, k)) discard;
        g_people_mat = region == 5 ? 9 : 1;
        g_people_rough = region == 5 ? 0.45 : 0.7;
        return t.r * 2.0 * L1.rgb * (region == 6 ? 0.75 : 1.0);
    }
    if (region == 7) {
        // THE EYE: wet; the iris the person's colour
        g_people_rough = 0.06;
        return mix(t.rgb, t.r * 2.0 * L5.rgb, t.a);
    }
    if (region == 8) { g_people_rough = 0.72; return t.rgb * L2.rgb; }
    if (region == 9) { g_people_rough = 0.5; return t.r * 2.0 * mix(vec3(0.030, 0.028, 0.026), vec3(0.20, 0.11, 0.055), L5.a); }
    return t.rgb;
}
