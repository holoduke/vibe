// Baked tree impostors.
//
// Far trees are drawn as one camera-facing card each (vegetation.cpp: LOD 4). What the card
// shows is baked here, once at start-up, from the tree's own full-detail mesh: every tree
// archetype is drawn from 8 directions round it (straight on, orthographic) by a small
// software rasteriser into two atlases -
//
//   colour : rgb = the leaf / bark colour as cube.frag would give it without the per-tree
//            tint and the season (those are applied per pixel, so one bake serves all
//            seasons); a = coverage (the silhouette)
//   normal : rgb = the surface normal in the tree's own space (0.5 + 0.5 n), a = 1 for
//            leaves, 0 for wood (bare trees in winter drop the leaf texels)
//
// One row per archetype, one 128 x 128 tile per direction. veg.vert picks the row and the
// direction, veg_leaf.glsl (veg_imp_uv) the texel; cube.frag and veg_depth.frag cut the
// silhouette from the alpha and cube.frag lights it with the baked normal - so the card has
// the shape, the colour and the shading of that very tree, where it used to be a painted
// outline in one flat green. That is what lets it take over from the mesh much sooner
// (settings: tree_impostor_px).
//
// The numbers shared with the shaders (8 views, 128 px tiles, the 3 px margin) are in
// veg_leaf.glsl as kImpViews / kImpTile / kImpMargin.

#include "engine/vk_engine/internal.h"
#include "engine/world_props.h"

#include <algorithm>
#include <cmath>
#include <future>
#include <vector>

namespace qlike {

namespace {
constexpr int   kViews = 8;
constexpr int   kTile = 128;
constexpr int   kSS = 2;                       // samples per texel, each way
constexpr float kMargin = 3.0f / 128.0f;

float srgb_to_linear(float c) { return c <= 0.04045f ? c / 12.92f : std::pow((c + 0.055f) / 1.055f, 2.4f); }
float linear_to_srgb(float c) { c = glm::clamp(c, 0.0f, 1.0f); return c <= 0.0031308f ? c * 12.92f : 1.055f * std::pow(c, 1.0f / 2.4f) - 0.055f; }
float smooth(float a, float b, float x) { const float t = glm::clamp((x - a) / (b - a), 0.0f, 1.0f); return t * t * (3.0f - 2.0f * t); }
float hash3(const glm::vec3& p) { return glm::fract(std::sin(glm::dot(p, glm::vec3(12.9898f, 78.233f, 37.719f))) * 43758.5453f); }

// One vertex as veg.vert would place it at full detail (nothing faded out, no wind).
struct BV {
    glm::vec3 p, n;
    glm::vec2 cuv{0.0f};     // leaf card: its corner (0..1)
    glm::vec3 origin{0.0f};  // leaf card: the point it grows from
    int mat = 0;             // 0 bark, 1 needles, 2 leaves, 3 bush, 6 leaf blade, 7 leaf card
    int cell = 0;            // leaf card: its cell of the foliage atlas
};
BV decode(const game::LevelVertex& v) {
    BV o; o.p = v.p; o.n = v.n;
    const int mr = int(v.v + 0.5f);
    o.mat = mr;
    if (mr == 7 || mr >= 9) {
        const float q = std::floor(v.u * 0.25f);
        const float corner = v.u - 4.0f * q;
        o.cuv = glm::vec2(std::fmod(corner, 2.0f), std::floor(corner * 0.5f + 0.01f));
        o.origin = v.p - v.n;
        o.cell = mr == 7 ? 0 : mr - 8;
        o.mat = 7;
    } else if (mr == 8) {
        o.mat = 0;           // branch tube, at its full thickness
    }
    return o;
}
} // namespace

VegImpostorAtlas bake_veg_impostors(const std::vector<VegArchetype>& archs,
                                    const unsigned char* leaf_px, int leaf_w, int leaf_h) {
    VegImpostorAtlas out;
    out.row.assign(archs.size(), -1);
    for (size_t a = 0; a < archs.size(); ++a)
        if (archs[a].species <= 1 && !archs[a].lods[0].verts.empty()) out.row[a] = out.rows++;
    if (out.rows == 0) return out;
    out.w = kViews * kTile;
    out.h = out.rows * kTile;
    out.color.assign(size_t(out.w) * out.h * 4, 0);
    out.normal.assign(size_t(out.w) * out.h * 4, 0);

    // The leaf atlas in linear light (rgb) + coverage. Missing file: plain green, a soft disc.
    auto leaf = [&](glm::vec2 uv, int cell, glm::vec3& rgb) -> float {
        if (!leaf_px || leaf_w <= 0) {
            rgb = glm::vec3(0.30f, 0.42f, 0.16f);
            return 1.0f - 1.6f * glm::length(uv - 0.5f);
        }
        const float m = 0.006f;
        const float au = float(cell % 2) * 0.5f + glm::clamp(uv.x, m, 1.0f - m) * 0.5f;
        const float av = (cell >= 2 ? 0.5f : 0.0f) + (1.0f - glm::clamp(uv.y, m, 1.0f - m)) * 0.5f;
        const int x = glm::clamp(int(au * float(leaf_w)), 0, leaf_w - 1);
        const int y = glm::clamp(int(av * float(leaf_h)), 0, leaf_h - 1);
        const unsigned char* p = leaf_px + (size_t(y) * leaf_w + x) * 4;
        rgb = glm::vec3(srgb_to_linear(p[0] / 255.0f), srgb_to_linear(p[1] / 255.0f), srgb_to_linear(p[2] / 255.0f));
        return p[3] / 255.0f;
    };

    // (one archetype per task: each writes its own row of the atlas)
    auto bake_one = [&](size_t a) {
        const int R = kTile * kSS;
        std::vector<float> zb(size_t(R) * R);
        std::vector<glm::vec3> cb(size_t(R) * R), nb(size_t(R) * R);
        std::vector<float> lb(size_t(R) * R);
        std::vector<glm::vec4> tc(size_t(kTile) * kTile), tn(size_t(kTile) * kTile);
        const game::LevelMesh& m = archs[a].lods[0];
        std::vector<BV> bv(m.verts.size());
        float rmax = 0.0f, ymax = 0.0f;
        for (size_t i = 0; i < m.verts.size(); ++i) {
            bv[i] = decode(m.verts[i]);
            // (the card's size, as vegetation.cpp takes it: from the stored positions)
            rmax = std::max(rmax, std::max(std::abs(m.verts[i].p.x), std::abs(m.verts[i].p.z)));
            ymax = std::max(ymax, m.verts[i].p.y);
        }
        rmax = std::max(rmax, 1e-3f); ymax = std::max(ymax, 1e-3f);
        const bool conifer = archs[a].species == 0;
        const float H = std::max(archs[a].height, 0.3f);

        // Colour and normal of a surface point, as the shaders give them (veg.vert's tints,
        // the crown normal, cube.frag's foliage shading at its average).
        auto shade = [&](const BV& v0, const BV& v1, const BV& v2, float w0, float w1, float w2,
                         glm::vec3& col, glm::vec3& nrm, float& leafy) -> bool {
            const glm::vec3 p = v0.p * w0 + v1.p * w1 + v2.p * w2;
            glm::vec3 n = glm::normalize(v0.n * w0 + v1.n * w1 + v2.n * w2 + glm::vec3(0.0f, 1e-5f, 0.0f));
            const int mat = v0.mat;
            // Outward from the trunk (conifer) or from the crown's centre (leaf tree).
            glm::vec3 dc;
            if (conifer) {
                const float ol = std::sqrt(p.x * p.x + p.z * p.z);
                dc = glm::normalize(glm::vec3(p.x, 0.45f * ol + 0.05f, p.z));
            } else {
                dc = glm::normalize(p - glm::vec3(0.0f, H * 0.68f, 0.0f) + glm::vec3(0.0f, 1e-4f, 0.0f));
            }
            const float kCrown = 0.47f;             // cube.frag's crown gaps / undersides, on average - and the leaves'
            // shadows on each other, which a flat card cannot receive (set by eye against the meshes)
            if (mat == 7) {
                const glm::vec2 uv = v0.cuv * w0 + v1.cuv * w1 + v2.cuv * w2;
                glm::vec3 tex;
                const float al = leaf(uv, v0.cell, tex);
                if (al < 0.33f) return false;
                const float tone = hash3(v0.origin);
                float rr;
                if (v0.cell % 2 == 1) {
                    const float rad = std::max(0.34f * H * (1.0f - glm::clamp(p.y / H, 0.0f, 0.95f)), 0.25f);
                    rr = std::sqrt(p.x * p.x + p.z * p.z) / rad;
                } else {
                    const float cy = H * 0.66f;
                    rr = glm::length(glm::vec3(p.x, (p.y - cy) * 1.25f, p.z)) / std::max(0.36f * H, 0.3f);
                }
                const float inner = glm::mix(0.42f, 1.0f, smooth(0.15f, 0.85f, rr));
                const glm::vec3 kind = v0.cell % 2 == 1 ? glm::vec3(0.62f, 0.80f, 0.63f)
                                     : v0.cell >= 2 ? glm::vec3(0.46f, 0.52f, 0.40f) : glm::vec3(0.60f, 0.64f, 0.50f);
                const float cl = glm::dot(tex, glm::vec3(0.3f, 0.6f, 0.1f));
                col = tex * kind * (0.80f + 0.40f * tone) * inner *
                      glm::mix(0.55f, 1.15f, smooth(0.10f, 0.45f, cl)) *
                      glm::mix(0.58f, 1.10f, smooth(0.33f, 0.95f, al)) * kCrown;
                nrm = glm::normalize(glm::mix(glm::vec3(0.0f, 1.0f, 0.0f), dc, 0.8f));
                leafy = 1.0f;
                return true;
            }
            if (mat == 0) {
                col = glm::vec3(0.085f, 0.065f, 0.048f);       // bark texture x its tint, in the crown's shade
                nrm = n;
                leafy = 0.0f;
                return true;
            }
            glm::vec3 base = mat == 1 ? glm::vec3(0.075f, 0.16f, 0.08f)
                           : mat == 3 ? glm::vec3(0.14f, 0.24f, 0.08f)
                           : mat == 6 ? glm::vec3(0.17f, 0.31f, 0.07f) * (0.78f + 0.44f * hash3(glm::floor(p * 2.3f)))
                           : glm::vec3(0.16f, 0.28f, 0.07f);
            nrm = glm::normalize(glm::mix(n, dc, mat == 1 ? 0.45f : mat == 6 ? 0.4f : 0.6f));
            col = base * glm::mix(0.55f, 1.0f, glm::clamp(nrm.y * 0.5f + 0.6f, 0.0f, 1.0f)) * kCrown;
            leafy = 1.0f;
            return true;
        };

        for (int view = 0; view < kViews; ++view) {
            const float ang = 6.2831853f * float(view) / float(kViews);
            const glm::vec3 d(std::sin(ang), 0.0f, std::cos(ang));       // from the tree to the camera
            const glm::vec3 rt(d.z, 0.0f, -d.x);                          // the card's right (veg.vert)
            std::fill(zb.begin(), zb.end(), -1e30f);
            std::fill(lb.begin(), lb.end(), 0.0f);
            auto to_px = [&](const glm::vec3& p) {
                const float u = glm::dot(p, rt) / rmax, v = p.y / ymax;
                return glm::vec3((kMargin + (u * 0.5f + 0.5f) * (1.0f - 2.0f * kMargin)) * float(R),
                                 (kMargin + (1.0f - v) * (1.0f - 2.0f * kMargin)) * float(R),
                                 glm::dot(p, d));
            };
            for (size_t t = 0; t + 2 < m.idx.size(); t += 3) {
                const BV& v0 = bv[m.idx[t]]; const BV& v1 = bv[m.idx[t + 1]]; const BV& v2 = bv[m.idx[t + 2]];
                const glm::vec3 a0 = to_px(v0.p), a1 = to_px(v1.p), a2 = to_px(v2.p);
                const float area = (a1.x - a0.x) * (a2.y - a0.y) - (a2.x - a0.x) * (a1.y - a0.y);
                if (std::abs(area) < 1e-6f) continue;
                const int x0 = std::max(0, int(std::floor(std::min({ a0.x, a1.x, a2.x }))));
                const int x1 = std::min(R - 1, int(std::ceil(std::max({ a0.x, a1.x, a2.x }))));
                const int y0 = std::max(0, int(std::floor(std::min({ a0.y, a1.y, a2.y }))));
                const int y1 = std::min(R - 1, int(std::ceil(std::max({ a0.y, a1.y, a2.y }))));
                for (int y = y0; y <= y1; ++y)
                    for (int x = x0; x <= x1; ++x) {
                        const float px = float(x) + 0.5f, py = float(y) + 0.5f;
                        const float w0 = ((a1.x - px) * (a2.y - py) - (a2.x - px) * (a1.y - py)) / area;
                        const float w1 = ((a2.x - px) * (a0.y - py) - (a0.x - px) * (a2.y - py)) / area;
                        const float w2 = 1.0f - w0 - w1;
                        if (w0 < 0.0f || w1 < 0.0f || w2 < 0.0f) continue;
                        const float z = a0.z * w0 + a1.z * w1 + a2.z * w2;
                        const size_t k = size_t(y) * R + size_t(x);
                        if (z <= zb[k]) continue;
                        glm::vec3 col, nrm; float leafy;
                        if (!shade(v0, v1, v2, w0, w1, w2, col, nrm, leafy)) continue;
                        zb[k] = z; cb[k] = col; nb[k] = nrm; lb[k] = leafy;
                    }
            }
            // Down to the tile: coverage in alpha, colour and normal over what is covered.
            for (int y = 0; y < kTile; ++y)
                for (int x = 0; x < kTile; ++x) {
                    glm::vec3 c(0.0f), n(0.0f); float lf = 0.0f; int cnt = 0;
                    for (int sy = 0; sy < kSS; ++sy)
                        for (int sx = 0; sx < kSS; ++sx) {
                            const size_t k = size_t(y * kSS + sy) * R + size_t(x * kSS + sx);
                            if (zb[k] < -1e29f) continue;
                            c += cb[k]; n += nb[k]; lf += lb[k]; ++cnt;
                        }
                    const size_t k = size_t(y) * kTile + size_t(x);
                    if (cnt > 0) {
                        tc[k] = glm::vec4(c / float(cnt), float(cnt) / float(kSS * kSS));
                        tn[k] = glm::vec4(glm::normalize(n + glm::vec3(0.0f, 1e-5f, 0.0f)), lf / float(cnt));
                    } else {
                        tc[k] = glm::vec4(0.0f); tn[k] = glm::vec4(0.0f, 1.0f, 0.0f, 0.0f);
                    }
                }
            // Spread colour and normal into the empty texels round the tree (a few steps), so
            // filtering and the smaller mips mix in neighbouring leaves, not black.
            for (int pass = 0; pass < 6; ++pass) {
                std::vector<glm::vec4> c2 = tc, n2 = tn;
                for (int y = 0; y < kTile; ++y)
                    for (int x = 0; x < kTile; ++x) {
                        const size_t k = size_t(y) * kTile + size_t(x);
                        if (tc[k].a > 0.0f || tn[k].a < -0.5f) continue;
                        glm::vec3 c(0.0f), n(0.0f); float lf = 0.0f; int cnt = 0;
                        for (int j = -1; j <= 1; ++j)
                            for (int i = -1; i <= 1; ++i) {
                                const int xx = x + i, yy = y + j;
                                if (xx < 0 || yy < 0 || xx >= kTile || yy >= kTile) continue;
                                const size_t q = size_t(yy) * kTile + size_t(xx);
                                if (tc[q].a <= 0.0f && tn[q].a > -0.5f) continue;    // (still empty)
                                c += glm::vec3(tc[q]); n += glm::vec3(tn[q]); lf += std::abs(tn[q].a) > 1.5f ? tn[q].a + 3.0f : tn[q].a; ++cnt;
                            }
                        if (cnt == 0) continue;
                        c2[k] = glm::vec4(c / float(cnt), 0.0f);
                        // (marked as filled - a < -0.5 - with its leaf share kept: -3 + share)
                        n2[k] = glm::vec4(glm::normalize(n + glm::vec3(0.0f, 1e-5f, 0.0f)), -3.0f + glm::clamp(lf / float(cnt), 0.0f, 1.0f));
                    }
                tc.swap(c2); tn.swap(n2);
            }
            const int ox = view * kTile, oy = out.row[a] * kTile;
            for (int y = 0; y < kTile; ++y)
                for (int x = 0; x < kTile; ++x) {
                    const size_t k = size_t(y) * kTile + size_t(x);
                    const size_t o = (size_t(oy + y) * out.w + size_t(ox + x)) * 4;
                    const float lf = tn[k].a < -0.5f ? tn[k].a + 3.0f : tn[k].a;
                    auto b = [](float v) { return static_cast<unsigned char>(glm::clamp(v, 0.0f, 1.0f) * 255.0f + 0.5f); };
                    out.color[o]     = b(linear_to_srgb(tc[k].r));
                    out.color[o + 1] = b(linear_to_srgb(tc[k].g));
                    out.color[o + 2] = b(linear_to_srgb(tc[k].b));
                    out.color[o + 3] = b(tc[k].a);
                    out.normal[o]     = b(tn[k].x * 0.5f + 0.5f);
                    out.normal[o + 1] = b(tn[k].y * 0.5f + 0.5f);
                    out.normal[o + 2] = b(tn[k].z * 0.5f + 0.5f);
                    out.normal[o + 3] = b(lf);
                }
        }
    };
    std::vector<std::future<void>> tasks;
    for (size_t a = 0; a < archs.size(); ++a)
        if (out.row[a] >= 0) tasks.push_back(std::async(std::launch::async, bake_one, a));
    for (auto& t : tasks) t.get();
    return out;
}

} // namespace qlike
