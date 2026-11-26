// Leaf cut-out for canopy / bush meshes (vertex material 2 / 3).
// The crowns are built from displaced blobs. A 3D cellular pattern places
// one elongated "leaf" per ~15 cm cell; where the blob surface passes
// between leaves the pixel is discarded, so a crown reads as a mass of
// separate leaves with gaps (deeper foliage or sky showing through)
// instead of a smooth ball. Shared by cube.frag (colour / G-buffer) and
// veg_depth.frag (depth prepass) - both MUST take the same decision or the
// prepass depth would block the pixels behind the holes.
//   rp   rest position (world space BEFORE wind sway, so the pattern is
//        glued to the crown and does not swim while it moves)
//   wp   actual world position, n = interpolated normal, cam = camera pos
const float kVegLeafCells = 6.5;        // cells per metre
float veg_leaf_h(vec3 p) {
    p = fract(p * vec3(0.1031, 0.1030, 0.0973));
    p += dot(p, p.yxz + 33.33);
    return fract((p.x + p.y) * p.z);
}
vec3 veg_leaf_h3(vec3 p) {
    p = fract(p * vec3(0.1031, 0.1030, 0.0973));
    p += dot(p, p.yxz + 33.33);
    return fract((p.xxy + p.yxx) * p.zyx);
}
// Nearest leaf: returns its normalised distance (< 1 = on the leaf) and
// its per-leaf random triple in `id`. 2x2x2 cell search.
float veg_leaf_cell(vec3 rp, out vec3 id) {
    vec3 p = rp * kVegLeafCells;
    vec3 ip = floor(p - 0.5);
    float best = 1e9;
    id = vec3(0.5);
    for (int k = 0; k < 8; ++k) {
        vec3 c = ip + vec3(float(k & 1), float((k >> 1) & 1), float((k >> 2) & 1));
        vec3 h = veg_leaf_h3(c);
        vec3 v = p - (c + 0.5 + (h - 0.5) * 0.7);
        vec3 a = normalize(h.yzx - 0.5 + vec3(0.0, 1e-3, 0.0));   // long axis of the leaf
        float av = dot(v, a);
        float r  = 0.40 + 0.20 * h.z;
        float d  = (dot(v, v) - 0.62 * av * av) / (r * r);
        if (d < best) { best = d; id = h; }
    }
    return best;
}
bool veg_leaf_cut(vec3 rp, vec3 wp, vec3 n, vec3 cam) {
    vec3  to_cam = cam - wp;
    float dist = length(to_cam);
    // Leaves are ~10 cm: beyond ~60 m the gaps are sub-pixel shimmer, so
    // they close up (the leaves grow until they merge).
    float near_k = 1.0 - smoothstep(34.0, 62.0, dist);
    if (near_k <= 0.0) return false;
    // More open toward the silhouette, denser where the surface faces the
    // viewer (there the gaps would only show the inside of the same clump).
    float facing = abs(dot(normalize(n), to_cam / max(dist, 1e-3)));
    float open = mix(1.0, 0.72, smoothstep(0.25, 0.9, facing));
    vec3 id;
    float d = veg_leaf_cell(rp, id);
    return d * open * near_k * near_k > 1.0;
}

// Leaf card (vertex material 7, the EZ-Tree style leaf quads). `uv` runs
// 0..1 across and 0 (attachment) .. 1 (tip) along the card; floor(kind) is
// 0 for a broadleaf spray, 1 for a needle spray. Returns false where the
// card is empty. `shade`: -1 on the twig, else 0 (leaf edge) .. 1 (midrib).
bool veg_card(vec2 uv, float kind, out float shade) {
    vec2 p = vec2(uv.x - 0.5, uv.y);
    shade = 0.5;
    bool twig = abs(p.x) < 0.012 && p.y < 0.84;
    if (kind < 1.0) {
        // Eight leaves alternating along a twig, plus a terminal one.
        float best = 1e9;
        for (int i = 0; i < 8; ++i) {
            float fi = float(i);
            bool  term = (i == 7);
            float side = ((i & 1) == 0) ? 1.0 : -1.0;
            float ang  = term ? 0.0 : side * (0.98 - 0.05 * fi);
            vec2  base = vec2(0.0, term ? 0.78 : 0.08 + 0.10 * fi);
            float len  = term ? 0.22 : 0.42 - 0.022 * fi;
            vec2  d = vec2(sin(ang), cos(ang));
            vec2  q = p - base;
            float l = dot(q, d) / len;
            if (l <= 0.0 || l >= 1.0) continue;
            float w = (q.x * d.y - q.y * d.x) / len;
            float prof = 0.27 * sin(3.14159 * pow(l, 0.72));
            best = min(best, abs(w) / max(prof, 1e-4));
        }
        if (best < 1.0) { shade = 1.0 - best; return true; }
    } else {
        // Needles: two fans of thin blades along the twig. They widen with
        // the pixel footprint so a distant spray closes up instead of
        // dissolving into sub-pixel shimmer.
        const float a = 0.62, P = 0.055;
        float hw = 0.011 + 0.8 * fwidth(uv.y);
        for (int s = 0; s < 2; ++s) {
            float sg = (s == 0) ? 1.0 : -1.0;
            float n = p.x / (sg * sin(a));
            if (n <= 0.0) continue;
            float t = p.y - n * cos(a);
            float len = 0.36 * (1.0 - 0.55 * smoothstep(0.45, 0.86, t));
            if (t < 0.02 || t > 0.86 || n > len) continue;
            float d = abs(fract(t / P) - 0.5) * P * sin(a);
            if (d < hw) { shade = 1.0 - d / hw; return true; }
        }
    }
    if (twig) { shade = -1.0; return true; }
    return false;
}

// Leaf card from the foliage atlas (albedo slot 15: broadleaf twig in the
// left half, conifer spray in the right). `uv` as for veg_card. Returns
// rgb + coverage; `lod_bias` < 0 keeps thin needles from fading with the mips.
// `m` = edge margin in cell uv: half the pixel footprint keeps the
// bilinear taps of the chosen mip inside the cell (a fixed 0.006 let the
// neighbouring card bleed in from mip ~3 on: coloured fringes far away).
vec2 veg_card_uv(vec2 uv, float kind, float m) {
    // 2 x 2 cells: 0 oak, 1 spruce (top row), 2 birch, 3 pine (bottom row).
    float cell = floor(kind);
    m = clamp(m, 0.006, 0.2);
    return vec2(mod(cell, 2.0) * 0.5 + clamp(uv.x, m, 1.0 - m) * 0.5,
                (cell >= 2.0 ? 0.5 : 0.0) + (1.0 - clamp(uv.y, m, 1.0 - m)) * 0.5);
}
vec2 veg_card_uv(vec2 uv, float kind) { return veg_card_uv(uv, kind, 0.006); }

// ---- Baked tree impostors (src/engine/vk_engine/veg_impostor.cpp) ----
// the reference game's impostors (its impostor shader, read from the compiled shaders; our code):
// every tree archetype is baked from kImpN x kImpN directions laid out on a hemi-octahedron,
// one kImpTile px tile each, orthographic, framed on the sphere round the tree's middle -
//   colour atlas : rgb colour, a coverage
//   second atlas : rg the surface normal in the tree's own space (octahedral), b depth along
//                  the view (0.5 = the plane through the middle, 1 = a radius nearer),
//                  a 1 for leaves, 0 for wood
// The atlas holds one block of kImpN x kImpN tiles per archetype, the blocks in rows of
// kImpCols. Looking up a pixel: the view ray is cut with the plane through the tree's middle
// that faces the camera; that point is projected into the three baked views round the
// direction to the camera, and they are blended - only in the middle fifth between two views,
// so most of the time one view is shown sharp. Their depths move the point off the plane: the
// card gets a real depth (it cuts into the ground and other trees as the tree would).
const float kImpN = 8.0;
const float kImpTile = 96.0;
const float kImpCols = 6.0;

vec2 veg_imp_enc(vec3 n) {                    // direction (y up, y >= 0) -> -1..1 square
    n.y = max(n.y, 0.0);
    n /= abs(n.x) + abs(n.y) + abs(n.z);
    return vec2(n.x + n.z, n.x - n.z);
}
vec3 veg_imp_dir(vec2 e) {
    vec2 t = vec2(e.x + e.y, e.x - e.y) * 0.5;
    return normalize(vec3(t.x, 1.0 - abs(t.x) - abs(t.y), t.y));
}
vec3 veg_oct_dec(vec2 e) {                    // full octahedral (the baked normal)
    e = e * 2.0 - 1.0;
    vec3 n = vec3(e.x, 1.0 - abs(e.x) - abs(e.y), e.y);
    float t = max(-n.y, 0.0);
    n.x += n.x >= 0.0 ? -t : t;
    n.z += n.z >= 0.0 ? -t : t;
    return normalize(n);
}
// (a baked view's right and up: veg_impostor.cpp imp_axes - keep in step)
void veg_imp_axes(vec3 f, out vec3 rt, out vec3 up) {
    rt = normalize(cross(vec3(0.0, 1.0, 0.0), f) + vec3(1e-6, 0.0, 0.0));
    up = cross(f, rt);
}
// wpos: the pixel on the card. base / ssy (size, y squash, turn) / info (block, the baked
// sphere's radius, its middle's height): the tree. px: the pixel's footprint on the card (m).
// False: nothing of the tree here. col: its colour; nrm: the baked normal, world space; leaf:
// 1 on leaves; surf: the surface point in the world (for the depth).
bool veg_imp_sample(sampler2D colT, sampler2D auxT, vec3 wpos, vec3 cam, vec3 base, vec3 ssy, vec3 info, float px,
                    out vec3 col, out vec3 nrm, out float leaf, out vec3 surf) {
    col = vec3(0.0); nrm = vec3(0.0, 1.0, 0.0); leaf = 0.0; surf = wpos;
    float c = cos(ssy.z), sn = sin(ssy.z);
    vec3  sc = vec3(ssy.x, ssy.x * ssy.y, ssy.x);
    vec3  a = cam - base, b = wpos - base;
    vec3  camL = vec3(c * a.x + sn * a.z, a.y, -sn * a.x + c * a.z) / sc;
    vec3  posL = vec3(c * b.x + sn * b.z, b.y, -sn * b.x + c * b.z) / sc;
    vec3  c0 = vec3(0.0, info.z, 0.0);
    float R  = max(info.y, 1e-3);
    vec3  vd = normalize(camL - c0);
    vec3  d  = posL - camL;
    float t  = dot(c0 - camL, vd) / min(dot(d, vd), -1e-6);
    vec3  p  = camL + d * t - c0;                                  // on the plane through the middle
    if (dot(p, p) > R * R) return false;

    vec2  g  = (veg_imp_enc(vd) * 0.5 + 0.5) * (kImpN - 1.0);
    vec2  ij = min(floor(g), vec2(kImpN - 2.0));
    vec2  fr = clamp((g - ij - 0.4) * 5.0, 0.0, 1.0);
    vec2  fs = fr * fr * (3.0 - 2.0 * fr);
    vec3  w  = vec3(min(1.0 - fs.x, 1.0 - fs.y), abs(fs.x - fs.y), min(fs.x, fs.y));
    vec2  fc[3] = vec2[3](ij, ij + (fs.x > fs.y ? vec2(1.0, 0.0) : vec2(0.0, 1.0)), ij + 1.0);

    vec2  asz = vec2(textureSize(colT, 0));
    float blk = floor(info.x + 0.5);
    vec2  org = vec2(mod(blk, kImpCols), floor(blk / kImpCols)) * kImpN;       // the block's first tile
    // (the mip by hand: the uv jumps between views. Never past 4 - a tile is 6 texels there, and
    //  the margin below keeps the taps inside it; it was 2, and a tree a few pixels tall was read
    //  from a 24-texel tile at a handful of scattered points: it flickered and lost its thin parts.)
    float lod_raw = log2(px / (2.0 * R * ssy.x) * kImpTile + 1e-6);
    float lod = clamp(lod_raw, 0.0, 4.0);
    // The cut follows the mip, as in the reference game: a small mip's coverage is an average
    // over leaves and the gaps between them, and cut at 0.5 a far crown shrinks and a far wood
    // thins out. 0.5 up to mip 2, 0.3 from mip 4 on.
    float cut = 0.5 - 0.1 * clamp(lod_raw - 2.0, 0.0, 2.0);
    // (keeps the taps inside the tile. Past mip 2 half a texel of the coarser mip in use is enough -
    //  a tile is 96 = 3 x 32 texels, so the mips up to 5 never mix two tiles - and a wider margin
    //  cut the top and the foot off a small far tree and smeared its edge: a pillar.)
    float m   = (lod <= 2.0 ? 0.5 + exp2(lod) : 0.5 * exp2(ceil(lod))) / kImpTile;
    float al = 0.0, lf = 0.0;
    vec3  off = vec3(0.0), nn = vec3(0.0);
    for (int k = 0; k < 3; ++k) {
        if (w[k] <= 0.001) continue;
        vec3 f = veg_imp_dir(fc[k] / (kImpN - 1.0) * 2.0 - 1.0), rt, up;
        veg_imp_axes(f, rt, up);
        vec2 uv = vec2(dot(rt, p), -dot(up, p)) / (2.0 * R) + 0.5;
        if (any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0)))) continue;
        vec2 auv = (org + fc[k] + clamp(uv, m, 1.0 - m)) * kImpTile / asz;
        vec4 tc = textureLod(colT, auv, lod);
        vec4 ta = textureLod(auxT, auv, lod);
        float wa = w[k] * tc.a;
        al  += wa;
        col += wa * tc.rgb;
        off += wa * (ta.b - 0.5) * 2.0 * R * f;
        nn  += wa * veg_oct_dec(ta.rg);
        lf  += wa * ta.a;
    }
    if (al < cut) return false;                                     // (the game's cut, by mip)
    col /= al; off /= al; lf /= al;
    leaf = lf;
    vec3 sl = (p + c0 + off) * sc;
    surf = base + vec3(c * sl.x - sn * sl.z, sl.y, sn * sl.x + c * sl.z);
    vec3 nl = normalize(nn / vec3(1.0, ssy.y, 1.0) + vec3(0.0, 1e-5, 0.0));
    nrm = vec3(c * nl.x - sn * nl.z, nl.y, sn * nl.x + c * nl.z);
    return true;
}
// Winter: the leaf trees' cards lose their leaf texels as the mesh loses its leaves.
bool veg_imp_bare(vec3 surf, float leafy, float kind, float winter) {
    if (leafy < 0.5 || kind < 1.5 || winter <= 0.0) return false;
    float tone = fract(sin(dot(floor(surf * 4.0), vec3(12.9898, 78.233, 37.719))) * 43758.5453);
    return smoothstep(tone * 0.6, tone * 0.6 + 0.4, winter) > 0.5;      // (veg.vert: when this leaf has let go)
}
