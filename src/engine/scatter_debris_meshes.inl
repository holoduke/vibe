// scatter_debris_meshes.inl - the small debris placed on the GPU every frame round the camera
// (species 19..26 = kinds 0..7, tuft_place.comp "debris"; docs/ARCHITECTURE.md "Small debris").
// Included by scatter.cpp. One mesh per kind, its vertices marked by uv.x = 20 + kind (the
// shaders' sign that this is debris: scatter.vert lays it on the ground, scatter.frag colours it). Everything lies flat: local y is the ground's normal
// (scatter.vert turns it onto the ground under the instance). Sizes are at scale 1 (m).
//   0 twig       a thin crooked stick with a side shoot
//   1 leaf       a broad leaf, a little curled (its colour from the instance: green, yellow, red-brown)
//   2 pebbles    three to five flattened pebbles
//   3 needles    a mat of fallen pine needles
//   4 cone       a pine cone lying on its side
//   5 litter     a crushed can and a scrap of paper
//   6 gravel     a spill of grit and small stones
//   7 stones     two or three fist-sized stones (the foot of a cliff)
// uv.y: 0..1 along the piece (the shader's shading); the levels of detail are the
// simplifier's (scatter.cpp).

namespace {
void debris_box(std::vector<Vertex>& v, std::vector<uint32_t>& ix, glm::vec3 a, glm::vec3 b, float r, float part) {
    // a thin square tube from a to b (a stick), 4 sides
    glm::vec3 d = b - a;
    float L = glm::length(d);
    if (L < 1e-5f) return;
    d /= L;
    glm::vec3 s = glm::normalize(glm::cross(d, std::abs(d.y) < 0.9f ? glm::vec3(0, 1, 0) : glm::vec3(1, 0, 0)));
    glm::vec3 u = glm::cross(s, d);
    for (int k = 0; k < 4; ++k) {
        float a0 = float(k) * 1.5707963f, a1 = a0 + 1.5707963f;
        glm::vec3 o0 = (s * std::cos(a0) + u * std::sin(a0)) * r, o1 = (s * std::cos(a1) + u * std::sin(a1)) * r;
        glm::vec3 n = glm::normalize(o0 + o1);
        uint32_t base = uint32_t(v.size());
        v.push_back({ a + o0, n, { part, 0.0f } });
        v.push_back({ b + o0 * 0.7f, n, { part, 1.0f } });
        v.push_back({ b + o1 * 0.7f, n, { part, 1.0f } });
        v.push_back({ a + o1, n, { part, 0.0f } });
        ix.insert(ix.end(), { base, base + 1, base + 2, base, base + 2, base + 3 });
    }
}
void debris_blob(std::vector<Vertex>& v, std::vector<uint32_t>& ix, glm::vec3 c, glm::vec3 r, float part, uint32_t seed) {
    // a low-poly flattened rock: an octahedron subdivided once, its corners jittered
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> j(0.82f, 1.15f);
    const int RINGS = 4, SEGS = 6;
    uint32_t base = uint32_t(v.size());
    for (int i = 0; i <= RINGS; ++i) {
        float th = 3.1415926f * float(i) / float(RINGS);
        for (int k = 0; k < SEGS; ++k) {
            float ph = 6.2831853f * float(k) / float(SEGS);
            glm::vec3 d(std::sin(th) * std::cos(ph), std::cos(th), std::sin(th) * std::sin(ph));
            float jj = (i == 0 || i == RINGS) ? 1.0f : j(rng);
            glm::vec3 p = c + d * r * jj;
            v.push_back({ p, glm::normalize(d / r), { part, float(i) / float(RINGS) } });
        }
    }
    for (int i = 0; i < RINGS; ++i)
        for (int k = 0; k < SEGS; ++k) {
            uint32_t a = base + uint32_t(i * SEGS + k), b = base + uint32_t(i * SEGS + (k + 1) % SEGS);
            uint32_t c2 = a + SEGS, d2 = b + SEGS;
            ix.insert(ix.end(), { a, c2, b, b, c2, d2 });
        }
}
void debris_quad(std::vector<Vertex>& v, std::vector<uint32_t>& ix, glm::vec3 c, glm::vec3 ax, glm::vec3 az, float lift, float part) {
    // a bent card: four corners, the middle raised by `lift` (two triangles each side of the fold)
    uint32_t base = uint32_t(v.size());
    glm::vec3 n = glm::normalize(glm::cross(az, ax));
    if (n.y < 0.0f) n = -n;
    v.push_back({ c - ax - az, n, { part, 0.0f } });
    v.push_back({ c + ax - az, n, { part, 0.0f } });
    v.push_back({ c + glm::vec3(0, lift, 0), n, { part, 0.5f } });
    v.push_back({ c - ax + az, n, { part, 1.0f } });
    v.push_back({ c + ax + az, n, { part, 1.0f } });
    ix.insert(ix.end(), { base, base + 2, base + 1, base + 1, base + 2, base + 4, base + 4, base + 2, base + 3, base + 3, base + 2, base });
    // (the underside, so a curled edge seen from below is not a hole)
    ix.insert(ix.end(), { base, base + 1, base + 2, base + 1, base + 4, base + 2, base + 4, base + 3, base + 2, base + 3, base, base + 2 });
}
}  // namespace

void build_debris(std::vector<Vertex>& v, std::vector<uint32_t>& ix, int kind) {
    std::mt19937 rng(911u);
    std::uniform_real_distribution<float> u01(0.0f, 1.0f);
    // 0 twig: 0.45 m, crooked, a side shoot
    if (kind == 0) {
        const float P = 20.0f;
        glm::vec3 a(-0.22f, 0.008f, 0.0f), m(0.0f, 0.012f, 0.025f), b(0.23f, 0.008f, -0.01f);
        debris_box(v, ix, a, m, 0.009f, P);
        debris_box(v, ix, m, b, 0.007f, P);
        debris_box(v, ix, m * 0.4f + b * 0.6f, glm::vec3(0.12f, 0.01f, 0.12f), 0.004f, P);
    }
    // 1 leaf: 9 x 6 cm, curled along its midrib
    if (kind == 1) debris_quad(v, ix, glm::vec3(0.0f, 0.004f, 0.0f), glm::vec3(0.03f, 0.006f, 0.0f), glm::vec3(0.0f, 0.0f, 0.045f), 0.006f, 21.0f);
    // 2 pebbles
    if (kind == 2) for (int i = 0; i < 4; ++i) {
        float r = 0.018f + 0.02f * u01(rng);
        debris_blob(v, ix, glm::vec3(u01(rng) * 0.16f - 0.08f, r * 0.3f, u01(rng) * 0.16f - 0.08f), glm::vec3(r, r * 0.55f, r * (0.8f + 0.4f * u01(rng))), 22.0f, 100u + uint32_t(i));
    }
    // 3 needles: 40 thin cards over 0.35 m
    // (a low heap, thickest in the middle, the needles crossing at slants: it reads as a mat of
    //  needles from standing height and stands above the ground's own relief)
    if (kind == 3) for (int i = 0; i < 90; ++i) {
        float ang = u01(rng) * 6.2831853f, l = 0.025f + 0.02f * u01(rng);
        float rx = u01(rng) * 2.0f - 1.0f, rz = u01(rng) * 2.0f - 1.0f;
        float heap = std::max(0.0f, 1.0f - (rx * rx + rz * rz));
        glm::vec3 c(rx * 0.2f, 0.004f + 0.028f * heap * u01(rng), rz * 0.2f);
        float tilt = (u01(rng) - 0.5f) * 0.4f * l;
        glm::vec3 ax(std::cos(ang) * l, tilt, std::sin(ang) * l), az(-std::sin(ang) * 0.0022f, 0.0f, std::cos(ang) * 0.0022f);
        debris_quad(v, ix, c, ax, az, 0.0f, 23.0f);
    }
    // 4 cone: a stack of scales round an axis lying along x
    if (kind == 4) {
        const float P = 24.0f;
        for (int i = 0; i < 5; ++i) {
            float t = float(i) / 4.0f;
            float rr = 0.022f * std::sin(0.3f + 2.6f * t) + 0.006f;
            debris_blob(v, ix, glm::vec3(-0.035f + 0.07f * t, 0.024f, 0.0f), glm::vec3(0.014f, rr, rr), P, 200u + uint32_t(i));
        }
    }
    // 5 litter: a crushed can lying down (an 8-sided tube, squashed) and a scrap of paper
    if (kind == 5) {
        const float P = 25.0f;
        uint32_t base = uint32_t(v.size());
        for (int s = 0; s <= 1; ++s)
            for (int k = 0; k < 8; ++k) {
                float a = 6.2831853f * float(k) / 8.0f;
                glm::vec3 n(0.0f, std::sin(a), std::cos(a));
                glm::vec3 p(s ? 0.06f : -0.06f, 0.033f + 0.033f * std::sin(a) * (s ? 0.45f : 0.9f), 0.033f * std::cos(a) * (s ? 1.25f : 1.0f));
                v.push_back({ p, n, { P, float(s) } });
            }
        for (int k = 0; k < 8; ++k) {
            uint32_t a = base + uint32_t(k), b = base + uint32_t((k + 1) % 8);
            ix.insert(ix.end(), { a, b, a + 8, b, b + 8, a + 8 });
        }
        debris_quad(v, ix, glm::vec3(0.06f, 0.003f, 0.10f), glm::vec3(0.07f, 0.004f, 0.02f), glm::vec3(-0.015f, 0.0f, 0.05f), 0.01f, P + 0.25f);
    }
    // 6 gravel: 14 grains over 0.45 m
    if (kind == 6) for (int i = 0; i < 14; ++i) {
        float r = 0.008f + 0.016f * u01(rng) * u01(rng);
        debris_blob(v, ix, glm::vec3(u01(rng) * 0.45f - 0.225f, r * 0.2f, u01(rng) * 0.45f - 0.225f), glm::vec3(r, r * 0.6f, r), 26.0f, 300u + uint32_t(i));
    }
    // 7 stones: two or three fist-sized
    if (kind == 7) for (int i = 0; i < 3; ++i) {
        float r = 0.045f + 0.06f * u01(rng);
        debris_blob(v, ix, glm::vec3(u01(rng) * 0.3f - 0.15f, r * 0.35f, u01(rng) * 0.3f - 0.15f), glm::vec3(r, r * 0.7f, r * (0.8f + 0.3f * u01(rng))), 27.0f, 400u + uint32_t(i));
    }
}
