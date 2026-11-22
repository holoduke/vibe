// scatter_plant_meshes.inl - mesh builders for the small plants: ferns, flowers, reeds, clover, mushrooms, grass tufts, water lilies, the rock
// Included by scatter.cpp inside namespace vibe (not a compile unit of its own).

namespace {

float hash31(glm::vec3 p) {
    float h = p.x * 12.9898f + p.y * 78.233f + p.z * 37.719f;
    float s = std::sin(h) * 43758.5453f;
    return s - std::floor(s);
}

// Lumpy rock: icosahedron subdivided twice, vertices displaced by a
// per-direction hash so it reads as an irregular stone, bottom
// flattened so it sits on the ground without floating edges.
void build_rock_mesh_data(std::vector<Vertex>& verts,
                          std::vector<uint32_t>& idx) {
    // Icosahedron base.
    const float t = (1.0f + std::sqrt(5.0f)) * 0.5f;
    std::vector<glm::vec3> pos = {
        {-1,  t, 0}, {1,  t, 0}, {-1, -t, 0}, {1, -t, 0},
        {0, -1,  t}, {0, 1,  t}, {0, -1, -t}, {0, 1, -t},
        { t, 0, -1}, {t, 0,  1}, {-t, 0, -1}, {-t, 0,  1},
    };
    for (auto& p : pos) p = glm::normalize(p);
    std::vector<uint32_t> faces = {
        0,11,5, 0,5,1, 0,1,7, 0,7,10, 0,10,11,
        1,5,9, 5,11,4, 11,10,2, 10,7,6, 7,1,8,
        3,9,4, 3,4,2, 3,2,6, 3,6,8, 3,8,9,
        4,9,5, 2,4,11, 6,2,10, 8,6,7, 9,8,1,
    };
    // Two subdivision rounds, vertices SHARED: 162 vertices / 320 triangles
    // of a unit sphere. scatter.vert pushes every vertex out to the stone's
    // own shape (rock_sdf.glsl's radial function with a per-stone seed), so
    // each stone is real geometry - no per-pixel ray march, no hull.
    for (int round = 0; round < 2; ++round) {
        std::vector<glm::vec3> np;
        std::vector<uint32_t>  nf;
        np.reserve(faces.size() * 2);
        for (size_t f = 0; f < faces.size(); f += 3) {
            glm::vec3 a = pos[faces[f]], b = pos[faces[f + 1]],
                      c = pos[faces[f + 2]];
            glm::vec3 ab = glm::normalize(a + b);
            glm::vec3 bc = glm::normalize(b + c);
            glm::vec3 ca = glm::normalize(c + a);
            uint32_t base = uint32_t(np.size());
            np.insert(np.end(), { a, ab, ca,  ab, b, bc,
                                  ca, bc, c,  ab, bc, ca });
            for (int i = 0; i < 12; ++i) nf.push_back(base + i);
        }
        pos = std::move(np);
        faces = std::move(nf);
    }
    // Weld the duplicated corners.
    verts.clear();
    idx.clear();
    std::vector<glm::vec3> uniq;
    for (uint32_t fi : faces) {
        const glm::vec3 p = pos[fi];
        uint32_t found = UINT32_MAX;
        for (uint32_t k = 0; k < uint32_t(uniq.size()); ++k)
            if (glm::dot(uniq[k] - p, uniq[k] - p) < 1e-10f) { found = k; break; }
        if (found == UINT32_MAX) {
            found = uint32_t(uniq.size());
            uniq.push_back(p);
            verts.push_back({ p, p, glm::vec2(0.0f) });
        }
        idx.push_back(found);
    }
}

// ---- Small plant meshes (unit-ish size, y up, base at origin) ----
// uv.x = part id (0 leaf/stem, 1 petal, 2 flower centre), uv.y = sway.
void plant_quad(std::vector<Vertex>& v, std::vector<uint32_t>& ix,
                glm::vec3 a, glm::vec3 b, glm::vec3 c, glm::vec3 d,
                float part, float s0, float s1) {
    glm::vec3 n = glm::normalize(glm::cross(b - a, d - a));
    uint32_t k = uint32_t(v.size());
    v.push_back({ a, n, { part, s0 } });
    v.push_back({ b, n, { part, s0 } });
    v.push_back({ c, n, { part, s1 } });
    v.push_back({ d, n, { part, s1 } });
    ix.insert(ix.end(), { k, k + 1, k + 2, k, k + 2, k + 3 });
}

// Tapered tube along `path` (stems), `sides` around, part 0 (leaf colour);
// uv.y = sway weight w0 -> w1.
void plant_tube(std::vector<Vertex>& v, std::vector<uint32_t>& ix, const std::vector<glm::vec3>& path,
                float r0, float r1, int sides, float w0, float w1) {
    const size_t n = path.size();
    if (n < 2) return;
    const uint32_t base = uint32_t(v.size());
    for (size_t i = 0; i < n; ++i) {
        glm::vec3 t = glm::normalize(path[std::min(i + 1, n - 1)] - path[i > 0 ? i - 1 : 0]);
        glm::vec3 ref = std::abs(t.y) < 0.95f ? glm::vec3(0, 1, 0) : glm::vec3(1, 0, 0);
        glm::vec3 u = glm::normalize(glm::cross(t, ref)), w = glm::cross(t, u);
        float f = float(i) / float(n - 1);
        float r = r0 + (r1 - r0) * f;
        for (int s = 0; s < sides; ++s) {
            float a = 6.2831853f * float(s) / float(sides);
            glm::vec3 nrm = u * std::cos(a) + w * std::sin(a);
            v.push_back({ path[i] + nrm * r, nrm, { 0.0f, w0 + (w1 - w0) * f } });
        }
    }
    for (size_t i = 0; i + 1 < n; ++i)
        for (int s = 0; s < sides; ++s) {
            uint32_t a = base + uint32_t(i) * uint32_t(sides) + uint32_t(s);
            uint32_t b = base + uint32_t(i) * uint32_t(sides) + uint32_t((s + 1) % sides);
            uint32_t c = a + uint32_t(sides), d = b + uint32_t(sides);
            ix.insert(ix.end(), { a, b, d, a, d, c });
        }
}

// Curved, folded leaf blade from `b` along `dir` (4 segments).
void plant_leaf(std::vector<Vertex>& v, std::vector<uint32_t>& ix, glm::vec3 b, glm::vec3 dir,
                float len, float wid, float lift, float w_base, int segs = 8, int across = 4, float round = 0.0f, float teeth = 0.0f) {
    // A smooth curved surface S(t, u): t along the midrib, u across the
    // blade. Shared vertices with analytic (finite-difference) normals, so
    // the leaf shades as one continuous, gently cupped surface instead of
    // a strip of flat facets.
    const glm::vec3 side = glm::normalize(glm::vec3(-dir.z, 0.0f, dir.x));
    // (default 9 x 5 vertices; small, numerous leaves pass fewer.)
    // `round` 0 = lance shape with a pointed tip, 1 = obovate (clover,
    // plantain): widest past the middle, blunt tip.
    auto half_w = [&](float t) {                          // rounded base, widest at 40 %, pointed tip
        float tc = glm::clamp(t, 0.0f, 1.0f);
        float a = std::pow(std::sin(3.14159f * std::pow(tc, 0.75f)), 0.8f);
        float ob = std::sqrt(std::max(0.0f, std::sin(3.14159f * std::pow(tc, 1.6f))));
        a = a + (ob - a) * round;
        // Saw-toothed margin (dandelion, nettle): `teeth` notches along the blade.
        if (teeth > 0.0f) a *= 1.0f - 0.42f * glm::fract(tc * teeth) * (1.0f - tc * 0.5f);
        return wid * (0.06f + 0.94f * a) * (t > 0.999f ? (round > 0.5f ? 0.35f : 0.0f) : 1.0f);
    };
    auto S = [&](float t, float u) {                      // u in -1..1
        glm::vec3 rib = b + dir * (len * t) +
                        glm::vec3(0.0f, std::sin(t * 2.0f) * lift - t * t * lift * 0.6f, 0.0f);
        float w = half_w(t);
        // Cupped along the midrib (edges raised), edges slightly wavy.
        float cup = 0.35f * w * u * u + 0.06f * w * std::sin(t * 9.0f) * u;
        return rib + side * (w * u) + glm::vec3(0.0f, cup, 0.0f);
    };
    const uint32_t base = uint32_t(v.size());
    for (int s = 0; s <= segs; ++s) {
        float t = float(s) / float(segs);
        for (int a = 0; a <= across; ++a) {
            float u = -1.0f + 2.0f * float(a) / float(across);
            const float e = 0.02f;
            glm::vec3 dt = S(std::min(t + e, 1.0f), u) - S(std::max(t - e, 0.0f), u);
            glm::vec3 du = S(std::min(t, 0.97f), std::min(u + 0.1f, 1.0f)) - S(std::min(t, 0.97f), std::max(u - 0.1f, -1.0f));
            glm::vec3 n = glm::cross(dt, du);
            float l = glm::length(n);
            n = l > 1e-9f ? n / l : glm::vec3(0.0f, 1.0f, 0.0f);
            if (n.y < 0.0f) n = -n;
            // uv: part 3 (leaf blade) + the across coordinate in the fraction;
            // sway base in 1/32 steps + the along coordinate in the fraction
            // (decoded by scatter.vert / .frag and clutter_shadow.vert).
            v.push_back({ S(t, u), n, { 3.0f + 0.4f * u, std::floor(w_base * 32.0f) / 32.0f + t * (0.98f / 32.0f) } });
        }
    }
    const uint32_t row = uint32_t(across + 1);
    for (int s = 0; s < segs; ++s)
        for (int a = 0; a < across; ++a) {
            uint32_t i0 = base + uint32_t(s) * row + uint32_t(a), i1 = i0 + 1, i2 = i0 + row, i3 = i2 + 1;
            ix.insert(ix.end(), { i0, i1, i3, i0, i3, i2 });
        }
}

// Flower head at hc facing `up`: segmented rounded petals that cup
// progressively along their length, plus a domed centre.
void plant_head(std::vector<Vertex>& v, std::vector<uint32_t>& ix, glm::vec3 hc, glm::vec3 up,
                int petals, float head_r, float petal_cup, float petal_w, float seed,
                float sway = 1.0f, int segs = 3) {
    glm::vec3 t1 = glm::normalize(glm::cross(up, std::abs(up.z) < 0.9f ? glm::vec3(0, 0, 1) : glm::vec3(1, 0, 0)));
    glm::vec3 t2 = glm::cross(up, t1);
    for (int p = 0; p < petals; ++p) {
        float am = 6.2831853f * (float(p) + 0.5f) / float(petals) + seed;
        float half = 3.14159f / float(petals) * petal_w;
        glm::vec3 em = t1 * std::cos(am) + t2 * std::sin(am);
        glm::vec3 es = t1 * std::cos(am + 1.5708f) + t2 * std::sin(am + 1.5708f);
        auto pt = [&](float t, float sgn) {   // t along the petal, sgn -1..1 across it
            float r = head_r * (0.12f + 0.95f * t);
            float wv = std::sin(3.14159f * std::min(1.0f, 0.25f + 0.85f * t)) * std::tan(half) * r;
            float cup = petal_cup * head_r * t * t;
            return hc + em * r + es * (wv * sgn) + up * (cup - 0.15f * head_r * sgn * sgn * t);
        };
        const uint32_t k = uint32_t(v.size());
        for (int s = 0; s <= segs; ++s) {
            float t = float(s) / segs;
            glm::vec3 a = pt(t, -1.0f), c = pt(t, 0.0f), b = pt(t, 1.0f);
            glm::vec3 nrm = glm::normalize(glm::cross(pt(std::min(1.0f, t + 0.1f), 0.0f) - c, b - a) + up * 0.3f);
            // Part 4: petal with petal-space coordinates (across in uv.x's
            // fraction, along packed with the sway in uv.y - see plant_leaf).
            const float pk = std::floor(sway * 32.0f) / 32.0f + t * (0.98f / 32.0f);
            v.push_back({ a, nrm, { 4.0f - 0.4f, pk } });
            v.push_back({ c, nrm, { 4.0f, pk } });
            v.push_back({ b, nrm, { 4.0f + 0.4f, pk } });
        }
        for (int s = 0; s < segs; ++s) {
            uint32_t r0 = k + uint32_t(s) * 3u, r1 = r0 + 3u;
            ix.insert(ix.end(), { r0, r0 + 1, r1 + 1, r0, r1 + 1, r1,
                                  r0 + 1, r0 + 2, r1 + 2, r0 + 1, r1 + 2, r1 + 1 });
        }
    }
    // Domed centre: two rings + top.
    const uint32_t k = uint32_t(v.size());
    const int cs = 10;
    const float r = head_r * 0.22f;
    for (int ring = 0; ring < 2; ++ring) {
        float rr = ring == 0 ? r : r * 0.6f, hh = ring == 0 ? 0.006f : 0.016f;
        for (int i = 0; i < cs; ++i) {
            float a = 6.2831853f * float(i) / float(cs);
            glm::vec3 d = t1 * std::cos(a) + t2 * std::sin(a);
            v.push_back({ hc + d * rr + up * hh, glm::normalize(up + d * 0.5f), { 2.0f, sway } });
        }
    }
    v.push_back({ hc + up * 0.022f, up, { 2.0f, sway } });
    for (int i = 0; i < cs; ++i) {
        uint32_t a = k + uint32_t(i), b = k + uint32_t((i + 1) % cs), c = a + uint32_t(cs), d = b + uint32_t(cs);
        ix.insert(ix.end(), { a, b, d, a, d, c, c, d, k + uint32_t(2 * cs) });
    }
}

// Fern (L-system): a crown of fronds in a 137.5 deg spiral. Each frond is
// a rachis R(i) that puts out a pair of pinnae per node and arches over
// under gravity; pinna length follows the lance-shaped outline of a real
// frond (short at the base, longest a third up, tapering to the tip).
//   axiom : [ /(137.5 k) &(tilt) R(0, len) ] for every frond k
//   R(i,l): i < n -> F(l/n) [ +(a) P(w_i) ] [ -(a) P(w_i) ] R(i+1, l)
//           i = n -> P(tip)
void build_fern(std::vector<Vertex>& v, std::vector<uint32_t>& ix) {
    using namespace lsys;
    const int fronds = 8, n = 11;
    Grammar g;
    {
        Rng r{ 7u };
        for (int k = 0; k < fronds; ++k) {
            g.axiom.push_back({ '[' });
            g.axiom.push_back({ '/', 137.5f * float(k) + r.range(-12.0f, 12.0f) });
            g.axiom.push_back({ '&', r.range(34.0f, 52.0f) });
            g.axiom.push_back({ 'R', 0.0f, r.range(0.78f, 1.12f) });
            g.axiom.push_back({ ']' });
        }
    }
    g.rules['R'] = [n](const Module& m, Rng& r, Str& o) {
        const int i = int(m.a + 0.5f);
        const float len = m.b;
        if (i >= n) { o.push_back({ 'P', 0.07f * len, 1.0f }); return; }
        const float t = (float(i) + 1.0f) / float(n);
        const float w = len * 0.25f * std::pow(std::sin(3.14159f * (0.10f + 0.88f * t)), 0.85f);
        const float a = 74.0f - 26.0f * t;                    // pinnae sweep forward toward the tip
        o.push_back({ 'F', len / float(n) });
        o.push_back({ '[' }); o.push_back({ '+', a + r.range(-5.0f, 5.0f) }); o.push_back({ 'P', w, t }); o.push_back({ ']' });
        o.push_back({ '[' }); o.push_back({ '-', a + r.range(-5.0f, 5.0f) }); o.push_back({ 'P', w, t }); o.push_back({ ']' });
        o.push_back({ 'R', float(i + 1), len });
    };
    TurtleParams tp;
    tp.radius = 0.007f; tp.taper = 0.9f;
    tp.bend = 1.75f;                                           // fronds arch over
    tp.organs = "P";
    const Shape sh = interpret(g.derive(n + 1, 11u), tp);
    float reach = 1e-3f;
    for (const auto& s : sh.segs) reach = std::max(reach, s.d1);
    // Rachis: a thin ribbon facing up.
    for (const auto& s : sh.segs) {
        glm::vec3 d = glm::normalize(s.b - s.a);
        glm::vec3 side = glm::normalize(glm::cross(d, glm::vec3(0.0f, 1.0f, 0.0f)));
        plant_quad(v, ix, s.a - side * s.r0, s.a + side * s.r0, s.b + side * s.r1, s.b - side * s.r1,
                   0.0f, s.d0 / reach, s.d1 / reach);
    }
    // Pinnae: folded, drooping blades (two triangles each).
    for (const auto& o : sh.organs) {
        const float len = o.a, sw = glm::clamp(o.dist / reach, 0.0f, 1.0f);
        const glm::vec3 droop(0.0f, -0.22f * len, 0.0f);
        const glm::vec3 base = o.pos, tip = o.pos + o.H * len + droop;
        const glm::vec3 mid = o.pos + o.H * (len * 0.42f) + droop * 0.3f;
        // Width runs along the rachis (perpendicular to the pinna, in the frond plane).
        const glm::vec3 w = glm::normalize(glm::cross(o.U, o.H)) * (len * 0.20f + 0.004f);
        const glm::vec3 fold = o.U * (len * 0.05f);
        glm::vec3 n1 = glm::normalize(glm::cross(mid + w - base, tip - base));
        glm::vec3 n2 = glm::normalize(glm::cross(tip - base, mid - w - base));
        if (n1.y < 0.0f) n1 = -n1;
        if (n2.y < 0.0f) n2 = -n2;
        uint32_t k = uint32_t(v.size());
        // Part 3 (leaf blade) with leaf-space coordinates, like plant_leaf.
        const float sk = std::floor(sw * 32.0f) / 32.0f;
        const float t0 = sk, tm = sk + 0.42f * (0.98f / 32.0f), t1 = sk + 0.98f / 32.0f;
        v.push_back({ base, n1, { 3.0f, t0 } }); v.push_back({ mid + w + fold, n1, { 3.4f, tm } }); v.push_back({ tip, n1, { 3.0f, t1 } });
        v.push_back({ base, n2, { 3.0f, t0 } }); v.push_back({ tip, n2, { 3.0f, t1 } }); v.push_back({ mid - w + fold, n2, { 2.6f, tm } });
        ix.insert(ix.end(), { k, k + 1, k + 2, k + 3, k + 4, k + 5 });
    }
}

// Flowering herb (L-system). One grammar covers the species; the numbers
// pick the habit:
//   S(i)  stem apex : i < nodes -> F [leaf] [side branch] /(137.5) S(i+1)
//                     i = nodes -> K (single terminal head) or A(0) (raceme)
//   B     side branch (sympodial): F [leaf] F K - or just a bud
//   A(j)  raceme apex: F [ &(a) F(pedicel) K ] /(roll) A(j+1), flowers
//         shrinking toward the tip (they open from the bottom up)
// Organs: L leaf, M palmate leaf, K flower head (parameter = size).
struct Herb {
    float height = 0.32f;
    int   nodes = 5;            // stem internodes
    int   leaves = 3;           // lowest nodes carrying a leaf
    float leaf_len = 0.20f, leaf_w = 0.035f;
    bool  palmate = false;
    int   branches = 0;         // side branches below the top
    float branch_len = 0.10f;
    int   petals = 8;
    float head_r = 0.05f, cup = 0.1f, petal_w = 0.9f;
    int   raceme = 0;           // flowers along the top of the stem
    float raceme_len = 0.3f;    // fraction of the height
    float raceme_roll = 137.5f; // 137.5 = spiral, ~0 = one-sided
    float pedicel_pitch = 62.0f;
    float lean = 0.0f;          // stem arches over (tropism strength)
    uint32_t seed = 1u;
};

void build_herb(std::vector<Vertex>& v, std::vector<uint32_t>& ix, const Herb& h) {
    using namespace lsys;
    Grammar g;
    g.axiom = { { '!', 0.0068f }, { 'S', 0.0f } };
    const float stem_len = h.height * (h.raceme > 0 ? 1.0f - h.raceme_len : 1.0f);
    g.rules['S'] = [&h, stem_len](const Module& m, Rng& r, Str& o) {
        const int i = int(m.a + 0.5f);
        if (i >= h.nodes) {
            if (h.raceme > 0) o.push_back({ 'A', 0.0f }); else o.push_back({ 'K', 1.0f });
            return;
        }
        o.push_back({ 'F', r.jit(stem_len / float(h.nodes), 0.18f) });
        if (i < h.leaves) {
            const float ll = h.leaf_len * (1.0f - 0.13f * float(i)) * r.range(0.85f, 1.15f);
            o.push_back({ '[' }); o.push_back({ '&', r.range(52.0f, 72.0f) });
            o.push_back({ h.palmate ? 'M' : 'L', ll }); o.push_back({ ']' });
        }
        if (h.branches > 0 && i >= h.nodes - 1 - h.branches && i < h.nodes - 1) {
            o.push_back({ '[' }); o.push_back({ '/', 90.0f }); o.push_back({ '&', r.range(28.0f, 44.0f) });
            o.push_back({ '!', 0.0042f }); o.push_back({ 'B', 1.0f }); o.push_back({ ']' });
        }
        o.push_back({ '/', 137.5f + r.range(-15.0f, 15.0f) });
        o.push_back({ '^', r.range(-5.0f, 5.0f) });           // slight zig-zag between nodes
        o.push_back({ 'S', float(i + 1) });
    };
    g.rules['B'] = [&h](const Module&, Rng& r, Str& o) {
        const float bl = h.branch_len * r.range(0.75f, 1.25f);
        o.push_back({ 'F', bl * 0.5f });
        o.push_back({ '[' }); o.push_back({ '/', 80.0f }); o.push_back({ '&', 60.0f });
        o.push_back({ 'L', h.leaf_len * 0.55f }); o.push_back({ ']' });
        o.push_back({ '^', r.range(8.0f, 20.0f) });           // curves back up toward the light
        o.push_back({ 'F', bl * 0.5f });
        o.push_back({ 'K', r.u() < 0.3f ? 0.42f : r.range(0.7f, 0.9f) });   // bud or smaller flower
    };
    g.rules['A'] = [&h](const Module& m, Rng& r, Str& o) {
        const int j = int(m.a + 0.5f);
        if (j >= h.raceme) { o.push_back({ 'K', 0.34f }); return; }
        const float t = float(j) / float(h.raceme);
        o.push_back({ 'F', h.height * h.raceme_len / float(h.raceme) });
        o.push_back({ '[' }); o.push_back({ '&', h.pedicel_pitch + r.range(-8.0f, 8.0f) });
        o.push_back({ '!', 0.0026f }); o.push_back({ 'F', h.head_r * (0.9f - 0.4f * t) });
        o.push_back({ 'K', 1.0f - 0.55f * t }); o.push_back({ ']' });
        o.push_back({ '/', h.raceme_roll + r.range(-10.0f, 10.0f) });
        o.push_back({ 'A', float(j + 1) });
    };
    TurtleParams tp;
    tp.radius = 0.0068f; tp.taper = 0.93f;
    tp.tropism = glm::normalize(glm::vec3(1.0f, -0.55f, 0.0f));
    tp.bend = h.lean;
    tp.organs = "LMK";
    const Shape sh = interpret(g.derive(h.nodes + h.raceme + 3, h.seed), tp);
    float top = 1e-3f;
    for (const auto& s : sh.segs) top = std::max(top, s.b.y);
    for (const auto& o : sh.organs) top = std::max(top, o.pos.y);
    auto sway = [&](glm::vec3 p) { return glm::clamp(p.y / top, 0.0f, 1.0f); };
    for (const auto& s : sh.segs)
        plant_tube(v, ix, { s.a, s.b }, s.r0, s.r1, s.depth == 0 ? 5 : 4, sway(s.a), sway(s.b));
    for (const auto& o : sh.organs) {
        if (o.s == 'K') {
            const bool main_head = o.a > 0.95f;
            if (o.a < 0.5f) {                                  // closed bud: few narrow petals, strongly cupped
                plant_head(v, ix, o.pos, o.H, 5, h.head_r * 0.42f, 2.6f, 0.9f, o.dist * 31.0f, sway(o.pos), 2);
            } else {
                plant_head(v, ix, o.pos, o.H, h.petals, h.head_r * o.a, h.cup, h.petal_w, o.dist * 31.0f,
                           sway(o.pos), main_head ? 3 : 2);
            }
        } else {
            glm::vec3 d = o.H;
            if (std::abs(d.y) > 0.92f) d = glm::normalize(d + o.U * 0.5f);
            if (o.s == 'L') {
                plant_leaf(v, ix, o.pos, d, o.a, h.leaf_w * (o.a / h.leaf_len), o.a * 0.5f, sway(o.pos));
            } else {                                           // palmate: leaflets fanned around a short stalk
                const glm::vec3 hub = o.pos + d * (o.a * 0.45f);
                plant_tube(v, ix, { o.pos, hub }, 0.0028f, 0.002f, 3, sway(o.pos), sway(hub));
                const glm::vec3 ax = glm::normalize(d + glm::vec3(0.0f, 0.8f, 0.0f));
                glm::vec3 t1 = glm::normalize(glm::cross(ax, glm::vec3(0.31f, 0.0f, 0.95f)));
                glm::vec3 t2 = glm::cross(ax, t1);
                for (int k = 0; k < 7; ++k) {
                    float a = 6.2831853f * float(k) / 7.0f;
                    glm::vec3 ld = glm::normalize(t1 * std::cos(a) + t2 * std::sin(a) + ax * 0.25f);
                    plant_leaf(v, ix, hub, ld, o.a * 0.5f, h.leaf_w * 0.55f, o.a * 0.10f, sway(hub));
                }
            }
        }
    }
}

// Reed / bulrush clump (Typha-like) for the shore. A tuft of long, narrow,
// keeled leaf blades that rise almost vertically and arch over near the
// tip, round flowering stems among them, and the familiar brown cattail
// heads (a velvety cylinder with a thin spike above). Leaves carry
// leaf-space coordinates (part 3, like plant_leaf) so the leaf texture
// and venation run along them; heads are part 5.
void reed_blade(std::vector<Vertex>& v, std::vector<uint32_t>& ix, glm::vec3 base, glm::vec3 lean,
                float len, float wid, float arch, uint32_t seed) {
    const glm::vec3 up(0.0f, 1.0f, 0.0f);
    const glm::vec3 L = glm::normalize(glm::vec3(lean.x, 0.0f, lean.z) + glm::vec3(1e-4f));
    const glm::vec3 side = glm::normalize(glm::cross(up, L));
    const float twist = (float((seed * 2654435761u) >> 8 & 0xFFFF) / 65535.0f - 0.5f) * 1.6f;
    const int segs = 10;
    auto spine = [&](float t) {
        // Straight up most of the way, then bending over toward `lean`.
        float bend = arch * t * t * t;
        return base + up * (len * t * (1.0f - 0.35f * arch * t * t)) + L * (len * (0.10f * t + bend * 0.55f));
    };
    const uint32_t b0 = uint32_t(v.size());
    for (int s = 0; s <= segs; ++s) {
        const float t = float(s) / float(segs);
        const glm::vec3 p = spine(t);
        const glm::vec3 tg = glm::normalize(spine(std::min(t + 0.05f, 1.0f)) - spine(std::max(t - 0.05f, 0.0f)));
        const float tw = twist * t;
        glm::vec3 sd = glm::normalize(side * std::cos(tw) + glm::cross(tg, side) * std::sin(tw));
        const float w = wid * (t < 0.85f ? 1.0f - 0.25f * t : (1.0f - t) / 0.15f * 0.79f);
        glm::vec3 nrm = glm::normalize(glm::cross(sd, tg));
        if (glm::dot(nrm, L) < 0.0f) nrm = -nrm;
        // Keeled (V-shaped) cross-section: the edges sit a little behind the midrib.
        const glm::vec3 keel = -nrm * (w * 0.35f);
        const float pk = t * (0.98f / 32.0f);
        v.push_back({ p - sd * w + keel, nrm, { 3.0f - 0.4f, pk } });
        v.push_back({ p,                 nrm, { 3.0f,        pk } });
        v.push_back({ p + sd * w + keel, nrm, { 3.0f + 0.4f, pk } });
    }
    for (int s = 0; s < segs; ++s) {
        const uint32_t r0 = b0 + uint32_t(s) * 3u, r1 = r0 + 3u;
        ix.insert(ix.end(), { r0, r0 + 1, r1 + 1, r0, r1 + 1, r1, r0 + 1, r0 + 2, r1 + 2, r0 + 1, r1 + 2, r1 + 1 });
    }
}

// Grass tuft: a clump of long, thin, arching blades - the meadow grass
// packs of the EZ-Tree demo scene, as real geometry instead of textured
// cards (no alpha test, proper shading and shadows). Part 8: uv.x = 8 +-
// 0.4 across the blade, uv.y = along-blade coordinate (the sway weight).
void grass_blade(std::vector<Vertex>& v, std::vector<uint32_t>& ix, glm::vec3 base, glm::vec3 lean,
                 float len, float wid, float arch, float twist) {
    const glm::vec3 up(0.0f, 1.0f, 0.0f);
    const glm::vec3 L = glm::normalize(glm::vec3(lean.x, 0.0f, lean.z) + glm::vec3(1e-4f));
    const glm::vec3 side = glm::normalize(glm::cross(up, L));
    const int segs = 4;
    auto spine = [&](float t) {
        return base + up * (len * t * (1.0f - 0.30f * arch * t * t)) + L * (len * (0.06f * t + 0.5f * arch * t * t));
    };
    const uint32_t b0 = uint32_t(v.size());
    for (int s = 0; s <= segs; ++s) {
        const float t = float(s) / float(segs);
        const glm::vec3 p = spine(t);
        const glm::vec3 tg = glm::normalize(spine(std::min(t + 0.05f, 1.0f)) - spine(std::max(t - 0.05f, 0.0f)));
        const float tw = twist * t;
        const glm::vec3 sd = glm::normalize(side * std::cos(tw) + glm::cross(tg, side) * std::sin(tw));
        const float w = wid * (1.0f - 0.92f * t * t);              // tapers to a point
        glm::vec3 nrm = glm::normalize(glm::cross(sd, tg));
        if (glm::dot(nrm, L) < 0.0f) nrm = -nrm;
        const glm::vec3 keel = -nrm * (w * 0.30f);                 // folded along the midrib
        const float pk = t * (0.98f / 32.0f);
        v.push_back({ p - sd * w + keel, nrm, { 8.0f - 0.4f, pk } });
        v.push_back({ p,                 nrm, { 8.0f,        pk } });
        v.push_back({ p + sd * w + keel, nrm, { 8.0f + 0.4f, pk } });
    }
    for (int s = 0; s < segs; ++s) {
        const uint32_t r0 = b0 + uint32_t(s) * 3u, r1 = r0 + 3u;
        ix.insert(ix.end(), { r0, r0 + 1, r1 + 1, r0, r1 + 1, r1, r0 + 1, r0 + 2, r1 + 2, r0 + 1, r1 + 2, r1 + 1 });
    }
}

void build_grass_tuft(std::vector<Vertex>& v, std::vector<uint32_t>& ix) {
    uint32_t seed = 0x6A55u;
    auto rnd = [&]() { seed = seed * 1664525u + 1013904223u; return float((seed >> 8) & 0xFFFFu) / 65535.0f; };
    // 34 blades from a ~0.3 m wide base: upright in the middle, the outer
    // ones shorter and arching over.
    for (int i = 0; i < 34; ++i) {
        const float a = 6.2831853f * rnd();
        const float outer = std::sqrt(rnd());
        const float r = 0.14f * outer;
        // Roots well below the ground line, the outer ones deepest: on a
        // slope the downhill side of the clump would otherwise hang in the air.
        const float root = 0.05f + 1.1f * r;
        const glm::vec3 base(std::cos(a) * r, -root, std::sin(a) * r);
        const float len = root + 0.45f + 0.45f * rnd() - 0.12f * outer;
        const glm::vec3 lean(std::cos(a) + (rnd() - 0.5f) * 0.9f, 0.0f, std::sin(a) + (rnd() - 0.5f) * 0.9f);
        grass_blade(v, ix, base, lean, len, 0.0080f + 0.0050f * rnd(), 0.25f + 0.95f * outer * rnd(),
                    (rnd() - 0.5f) * 1.8f);
    }
}

void build_reed(std::vector<Vertex>& v, std::vector<uint32_t>& ix) {
    uint32_t seed = 0xBEEDu;
    auto rnd = [&]() { seed = seed * 1664525u + 1013904223u; return float((seed >> 8) & 0xFFFFu) / 65535.0f; };
    const glm::vec3 up(0.0f, 1.0f, 0.0f);
    // Leaves: ~26 blades from a ~0.3 m wide base, outer ones lower and more arched.
    for (int i = 0; i < 26; ++i) {
        const float a = 6.2831853f * rnd();
        const float r = 0.16f * std::sqrt(rnd());
        const glm::vec3 base(std::cos(a) * r, -0.05f, std::sin(a) * r);
        const float outer = r / 0.16f;
        const float len = 1.25f + 0.85f * rnd() - 0.45f * outer;
        const glm::vec3 lean(std::cos(a) + (rnd() - 0.5f) * 0.8f, 0.0f, std::sin(a) + (rnd() - 0.5f) * 0.8f);
        reed_blade(v, ix, base, lean, len, 0.011f + 0.006f * rnd(), 0.25f + 0.65f * outer * rnd(), seed);
    }
    // Flowering stems with cattail heads.
    for (int i = 0; i < 4; ++i) {
        const float a = 6.2831853f * rnd(), r = 0.07f * rnd();
        const glm::vec3 b(std::cos(a) * r, -0.05f, std::sin(a) * r);
        const float H = 1.75f + 0.55f * rnd();
        const glm::vec3 lean = glm::vec3(std::cos(a), 0.0f, std::sin(a)) * (0.05f + 0.10f * rnd());
        std::vector<glm::vec3> path;
        for (int k = 0; k <= 8; ++k) {
            const float t = float(k) / 8.0f;
            path.push_back(b + up * (H * t) + lean * (H * t * t));
        }
        plant_tube(v, ix, path, 0.0065f, 0.0040f, 5, 0.0f, 1.0f);
        // Head: a 16-sided cylinder ~0.20 m long below the stem tip, rounded ends.
        const glm::vec3 hb = path[8] - up * 0.34f + lean * 0.2f, ht = hb + up * 0.20f;
        const float rr = 0.016f + 0.004f * rnd();
        std::vector<glm::vec3> hp = { hb - up * 0.012f, hb, hb + up * 0.06f, ht - up * 0.06f, ht, ht + up * 0.012f };
        const uint32_t h0 = uint32_t(v.size());
        const float rad[6] = { 0.45f, 0.92f, 1.0f, 1.0f, 0.92f, 0.45f };
        const int S = 16;
        for (int k = 0; k < 6; ++k)
            for (int s = 0; s < S; ++s) {
                const float ang = 6.2831853f * float(s) / float(S);
                const glm::vec3 d(std::cos(ang), 0.0f, std::sin(ang));
                const glm::vec3 nrm = glm::normalize(d + up * (k == 0 ? -0.8f : (k == 5 ? 0.8f : 0.0f)));
                v.push_back({ hp[k] + d * (rr * rad[k]), nrm, { 5.0f, 1.0f } });
            }
        for (int k = 0; k < 5; ++k)
            for (int s = 0; s < S; ++s) {
                const uint32_t a0 = h0 + uint32_t(k * S + s), a1 = h0 + uint32_t(k * S + (s + 1) % S);
                ix.insert(ix.end(), { a0, a1, a1 + uint32_t(S), a0, a1 + uint32_t(S), a0 + uint32_t(S) });
            }
        // Thin male spike above the head.
        std::vector<glm::vec3> sp = { ht + up * 0.02f, ht + up * 0.16f };
        plant_tube(v, ix, sp, 0.0030f, 0.0012f, 4, 1.0f, 1.0f);
    }
}

// ---- Ground cover (the low layer open-world games fill their ground
// with: clover, plantain, dandelion, nettle, foxglove, thistle, fungi).
// Kept low-poly per leaf - they are scattered by the ten thousand. ----

// Surface of revolution around `axis` through `base`: profile (r, y)
// pairs bottom to top, `part` written to every vertex (uv.y = 0: no sway).
void plant_lathe(std::vector<Vertex>& v, std::vector<uint32_t>& ix, glm::vec3 base, glm::vec3 axis,
                 const std::vector<glm::vec2>& prof, int sides, float part) {
    const glm::vec3 up = glm::normalize(axis);
    const glm::vec3 t1 = glm::normalize(glm::cross(up, std::abs(up.z) < 0.9f ? glm::vec3(0, 0, 1) : glm::vec3(1, 0, 0)));
    const glm::vec3 t2 = glm::cross(up, t1);
    const uint32_t b0 = uint32_t(v.size());
    const size_t n = prof.size();
    for (size_t i = 0; i < n; ++i) {
        // Profile tangent -> outward normal (dr, dy) rotated by -90 deg.
        glm::vec2 d = prof[std::min(i + 1, n - 1)] - prof[i > 0 ? i - 1 : 0];
        glm::vec2 pn = glm::normalize(glm::vec2(d.y, -d.x) + glm::vec2(1e-6f));
        for (int s = 0; s < sides; ++s) {
            const float a = 6.2831853f * float(s) / float(sides);
            const glm::vec3 radial = t1 * std::cos(a) + t2 * std::sin(a);
            v.push_back({ base + radial * prof[i].x + up * prof[i].y,
                          glm::normalize(radial * pn.x + up * pn.y), { part, 0.0f } });
        }
    }
    for (size_t i = 0; i + 1 < n; ++i)
        for (int s = 0; s < sides; ++s) {
            const uint32_t a = b0 + uint32_t(i) * uint32_t(sides) + uint32_t(s);
            const uint32_t b = b0 + uint32_t(i) * uint32_t(sides) + uint32_t((s + 1) % sides);
            ix.insert(ix.end(), { a, b, b + uint32_t(sides), a, b + uint32_t(sides), a + uint32_t(sides) });
        }
}

// Brown seed spike (plantain): a thin cylinder in part 5 (the cattail
// head material) from `a` to `b`.
void seed_spike(std::vector<Vertex>& v, std::vector<uint32_t>& ix, glm::vec3 a, glm::vec3 b, float r) {
    const glm::vec3 ax = b - a;
    const float L = glm::length(ax);
    plant_lathe(v, ix, a, ax, { { r * 0.5f, 0.0f }, { r, L * 0.15f }, { r, L * 0.75f }, { r * 0.4f, L } }, 6, 5.0f);
    // plant_lathe writes sway 0; spikes sit high on a flexible stalk.
    for (size_t i = v.size() - 24; i < v.size(); ++i) v[i].uv.y = 1.0f;
}

uint32_t g_seed = 1u;
float grnd() { g_seed = g_seed * 1664525u + 1013904223u; return float((g_seed >> 8) & 0xFFFFu) / 65535.0f; }

// White clover (Trifolium repens): a creeping patch of trifoliate leaves
// on petioles of different lengths, a few round white flower heads.
void build_clover(std::vector<Vertex>& v, std::vector<uint32_t>& ix) {
    g_seed = 0xC10Fu;
    const glm::vec3 up(0.0f, 1.0f, 0.0f);
    for (int i = 0; i < 11; ++i) {
        const float a = 6.2831853f * grnd(), r = 0.20f * std::sqrt(grnd());
        const glm::vec3 b(std::cos(a) * r, -0.01f, std::sin(a) * r);
        const float hgt = 0.035f + 0.075f * grnd();
        const glm::vec3 top = b + up * hgt + glm::vec3(grnd() - 0.5f, 0.0f, grnd() - 0.5f) * 0.03f;
        plant_tube(v, ix, { b, top }, 0.0016f, 0.0013f, 3, 0.0f, 0.6f);
        const float roll = 6.2831853f * grnd(), size = 0.018f + 0.012f * grnd();
        for (int k = 0; k < 3; ++k) {
            const float la = roll + 2.0944f * float(k);
            const glm::vec3 d = glm::normalize(glm::vec3(std::cos(la), 0.12f, std::sin(la)));
            plant_leaf(v, ix, top, d, size, size * 0.55f, size * 0.12f, 0.6f, 3, 2, 1.0f);
        }
    }
    for (int i = 0; i < 3; ++i) {                       // flower heads on taller stalks
        const float a = 6.2831853f * grnd(), r = 0.14f * std::sqrt(grnd());
        const glm::vec3 b(std::cos(a) * r, -0.01f, std::sin(a) * r);
        const glm::vec3 top = b + up * (0.10f + 0.05f * grnd());
        plant_tube(v, ix, { b, top }, 0.0016f, 0.0012f, 3, 0.0f, 1.0f);
        plant_head(v, ix, top, up, 14, 0.011f, 2.4f, 0.75f, grnd() * 6.0f, 1.0f, 2);
    }
}

// Greater plantain (Plantago major): a flat rosette of broad, ribbed,
// spoon-shaped leaves pressed to the ground and slim brown seed spikes.
void build_plantain(std::vector<Vertex>& v, std::vector<uint32_t>& ix) {
    g_seed = 0x91A7u;
    const glm::vec3 up(0.0f, 1.0f, 0.0f);
    const int n = 8;
    for (int i = 0; i < n; ++i) {
        const float a = 6.2831853f * (float(i) + 0.3f * grnd()) / float(n) + float(i % 2) * 0.3f;
        const glm::vec3 d = glm::normalize(glm::vec3(std::cos(a), 0.22f + 0.15f * grnd(), std::sin(a)));
        const float len = 0.13f + 0.07f * grnd();
        plant_leaf(v, ix, glm::vec3(0.0f, 0.005f, 0.0f), d, len, len * 0.36f, len * 0.08f, 0.2f, 5, 2, 0.85f);
    }
    for (int i = 0; i < 3; ++i) {
        const float a = 6.2831853f * grnd();
        const glm::vec3 lean = glm::vec3(std::cos(a), 0.0f, std::sin(a)) * (0.02f + 0.03f * grnd());
        const float H = 0.16f + 0.10f * grnd();
        const glm::vec3 b(0.0f, 0.0f, 0.0f), m = b + up * (H * 0.55f) + lean, t = b + up * H + lean * 1.8f;
        plant_tube(v, ix, { b, m }, 0.0018f, 0.0015f, 3, 0.0f, 0.6f);
        seed_spike(v, ix, m, t, 0.0035f);
    }
}

// Dandelion: a rosette of deeply toothed leaves, hollow stalks with
// yellow heads (many strap florets) and a closed bud.
void build_dandelion(std::vector<Vertex>& v, std::vector<uint32_t>& ix) {
    g_seed = 0xDA4Du;
    const glm::vec3 up(0.0f, 1.0f, 0.0f);
    for (int i = 0; i < 8; ++i) {
        const float a = 6.2831853f * (float(i) + 0.4f * grnd()) / 8.0f;
        const glm::vec3 d = glm::normalize(glm::vec3(std::cos(a), 0.30f + 0.2f * grnd(), std::sin(a)));
        const float len = 0.12f + 0.06f * grnd();
        plant_leaf(v, ix, glm::vec3(0.0f, 0.004f, 0.0f), d, len, len * 0.17f, len * 0.10f, 0.25f, 10, 2, 0.0f, 4.0f);
    }
    for (int i = 0; i < 2; ++i) {
        const float a = 6.2831853f * grnd();
        const glm::vec3 lean = glm::vec3(std::cos(a), 0.0f, std::sin(a)) * 0.03f;
        const float H = 0.14f + 0.10f * grnd();
        const glm::vec3 m = up * (H * 0.5f) + lean * 0.4f, t = up * H + lean;
        plant_tube(v, ix, { glm::vec3(0.0f), m, t }, 0.0024f, 0.0020f, 4, 0.0f, 1.0f);
        const glm::vec3 face = glm::normalize(up + lean * 6.0f);
        if (i == 0) plant_head(v, ix, t, face, 24, 0.020f, 0.35f, 0.55f, grnd() * 6.0f, 1.0f, 2);
        else        plant_head(v, ix, t, face, 6, 0.010f, 2.6f, 0.9f, grnd() * 6.0f, 1.0f, 2);   // bud
    }
}

// Stinging nettle: a few upright square-ish stems with opposite,
// decussate pairs of heart-shaped, serrated leaves angled down, smaller
// toward the tip. Grows in clumps along forest edges.
void build_nettle(std::vector<Vertex>& v, std::vector<uint32_t>& ix) {
    g_seed = 0x2E77u;
    const glm::vec3 up(0.0f, 1.0f, 0.0f);
    for (int s = 0; s < 4; ++s) {
        const float a = 6.2831853f * grnd(), r = 0.10f * std::sqrt(grnd());
        const glm::vec3 b(std::cos(a) * r, -0.02f, std::sin(a) * r);
        const float H = 0.45f + 0.35f * grnd();
        const glm::vec3 lean = glm::vec3(std::cos(a), 0.0f, std::sin(a)) * (0.04f + 0.05f * grnd());
        auto P = [&](float t) { return b + up * (H * t) + lean * (t * t); };
        std::vector<glm::vec3> path;
        for (int k = 0; k <= 6; ++k) path.push_back(P(float(k) / 6.0f));
        plant_tube(v, ix, path, 0.0035f, 0.0016f, 4, 0.0f, 1.0f);
        const int nodes = 6;
        float roll = grnd() * 3.14159f;
        for (int k = 1; k <= nodes; ++k) {
            const float t = float(k) / float(nodes + 1);
            const glm::vec3 p = P(t);
            const float len = (0.10f - 0.055f * t) * (0.85f + 0.3f * grnd());
            for (int side = 0; side < 2; ++side) {
                const float la = roll + 3.14159f * float(side);
                const glm::vec3 d = glm::normalize(glm::vec3(std::cos(la), -0.15f - 0.25f * t, std::sin(la)));
                plant_leaf(v, ix, p, d, len, len * 0.38f, len * 0.05f, t, 8, 2, 0.35f, 5.0f);
            }
            roll += 1.5708f;                              // decussate
        }
    }
}

// Mushrooms: a small cluster of capped fungi (stem + gills part 6, cap
// part 7 = instance colour). Rigid - no sway.
void build_mushrooms(std::vector<Vertex>& v, std::vector<uint32_t>& ix) {
    g_seed = 0xF0461u;
    const int n = 5;
    for (int i = 0; i < n; ++i) {
        const float a = 6.2831853f * grnd(), r = i == 0 ? 0.0f : 0.05f + 0.08f * grnd();
        const glm::vec3 b(std::cos(a) * r, -0.01f, std::sin(a) * r);
        const float s = (i == 0 ? 1.25f : 0.55f + 0.6f * grnd());
        const glm::vec3 ax = glm::normalize(glm::vec3(std::cos(a) * 0.18f * r / 0.13f, 1.0f, std::sin(a) * 0.18f * r / 0.13f));
        const float sh = 0.055f * s * (0.8f + 0.4f * grnd());   // stem height
        const float cr = 0.026f * s * (0.8f + 0.5f * grnd());   // cap radius
        // Stem: slightly bulbous foot, tapering up.
        plant_lathe(v, ix, b, ax, { { 0.0f, -0.004f }, { 0.0075f * s, 0.0f }, { 0.0068f * s, sh * 0.4f },
                                    { 0.0056f * s, sh } }, 8, 6.0f);
        // Gills: the cap underside, from the stem out to the rim.
        const glm::vec3 c = b + ax * sh;
        plant_lathe(v, ix, c, ax, { { 0.0056f * s, 0.0f }, { cr * 0.6f, -0.002f }, { cr, -0.004f } }, 12, 6.0f);
        // Cap: rim curled under, domed top (flatter on older ones).
        const float dome = 0.55f + 0.35f * grnd();
        plant_lathe(v, ix, c, ax, { { cr, -0.0042f }, { cr * 1.03f, 0.0f }, { cr * 0.93f, cr * 0.32f * dome },
                                    { cr * 0.68f, cr * 0.62f * dome }, { cr * 0.32f, cr * 0.80f * dome },
                                    { 0.0f, cr * 0.84f * dome } }, 12, 7.0f);
    }
}

// Water lilies: one instance = a raft of floating pads, each on its own
// stem down to the lake bed, with one or two flowers.
//   pad    part 9: uv.x = 9 + 0.4 * angle / pi (angle round the pad centre,
//          for the veins), uv.y = radius 0..1 (packed below 1/32 so the wind
//          sway stays a few millimetres - the pads bob a little).
//   petal  part 1 (instance colour), flower centre part 2, stems part 0.
// Pads are real surfaces: 4 rings x 28 segments, gently domed, the rim
// wavy and here and there curled up or dipping under; some pads tilt.
// `far_lod`: the same raft with a tenth of the triangles (coarser pads and
// petals, no stems or stamens) - drawn beyond ~22 m, where the rest is
// smaller than a pixel. The random numbers are drawn the same way in both,
// so the two shapes coincide.
void build_water_lily(std::vector<Vertex>& v, std::vector<uint32_t>& ix, bool far_lod = false) {
    g_seed = 0x11117u;
    const float kPi = 3.14159265f;
    auto tri = [&](uint32_t a, uint32_t b, uint32_t c) { ix.insert(ix.end(), { a, b, c }); };
    auto stem = [&](glm::vec3 top, float r, float lean_a) {
        // A soft S-curve down to (and a little into) the bed.
        std::vector<glm::vec3> path;
        const glm::vec3 drift(std::cos(lean_a) * 0.22f, 0.0f, std::sin(lean_a) * 0.22f);
        for (int k = 0; k <= 7; ++k) {
            const float t = float(k) / 7.0f;
            path.push_back(top + glm::vec3(0.0f, -1.30f * t, 0.0f) + drift * (std::sin(t * kPi * 0.9f) * (1.0f - 0.3f * t)));
        }
        if (!far_lod) plant_tube(v, ix, path, r, r * 1.5f, 5, 0.0f, 0.0f);
    };
    const int pads = 7;
    for (int i = 0; i < pads; ++i) {
        const float a = 2.0f * kPi * (float(i) + 0.7f * grnd()) / float(pads - 1);
        const float d = i == 0 ? 0.0f : 0.22f + 0.30f * grnd();
        const glm::vec3 c(std::cos(a) * d, 0.010f + 0.004f * grnd(), std::sin(a) * d);
        const float r = i == 0 ? 0.19f : 0.10f + 0.09f * grnd();
        const float notch = 2.0f * kPi * grnd();
        const float gap = 0.30f + 0.22f * grnd();                 // the V-shaped cut of a lily pad
        // Tilt: most lie flat, a few ride up on a neighbour.
        const float tilt = grnd() < 0.3f ? 0.10f + 0.12f * grnd() : 0.02f * grnd();
        const float tilt_a = 2.0f * kPi * grnd();
        const float curl_a = 2.0f * kPi * grnd();                  // where the rim curls up
        const float curl = grnd() < 0.5f ? 0.035f + 0.05f * grnd() : 0.008f;
        const float dip_a = curl_a + 2.2f + grnd();                // and where it dips under
        const int seg = far_lod ? 8 : 28, rings = far_lod ? 1 : 4;
        auto P = [&](float rr01, float ang) {
            const float wav = 1.0f + 0.05f * std::sin(ang * 5.0f + float(i)) + 0.025f * std::sin(ang * 11.0f);
            const float rr = r * rr01 * (rr01 > 0.7f ? wav : 1.0f);
            glm::vec3 p = c + glm::vec3(std::cos(ang) * rr, 0.0f, std::sin(ang) * rr);
            // Dome (centre slightly raised where the stem pushes), raised rim.
            p.y += 0.006f * (1.0f - rr01 * rr01) + 0.004f * rr01 * rr01 * rr01;
            // Rim curled up on one side, dipping into the water on another.
            const float e = rr01 * rr01 * rr01;
            p.y += curl * e * std::pow(std::max(std::cos(ang - curl_a), 0.0f), 3.0f);
            p.y -= 0.012f * e * std::pow(std::max(std::cos(ang - dip_a), 0.0f), 4.0f);
            p.y += 0.004f * e * std::sin(ang * 7.0f + float(i) * 1.7f);
            p.y += tilt * rr * std::cos(ang - tilt_a);
            return p;
        };
        auto uvp = [&](float rr01, float ang) {
            float an = std::fmod(ang - notch, 2.0f * kPi); if (an < 0.0f) an += 2.0f * kPi;
            return glm::vec2(9.0f + 0.4f * (an / kPi - 1.0f), rr01 * 0.98f / 32.0f);
        };
        const uint32_t b0 = uint32_t(v.size());
        for (int k = 0; k <= seg; ++k) {
            const float ang = notch + gap * 0.5f + (2.0f * kPi - gap) * float(k) / float(seg);
            for (int q = 0; q <= rings; ++q) {
                const float rr01 = float(q) / float(rings);
                const glm::vec3 p = P(rr01, ang);
                // Normal from neighbouring surface points.
                const glm::vec3 du = P(std::min(rr01 + 0.05f, 1.0f), ang) - P(std::max(rr01 - 0.05f, 0.0f), ang);
                const glm::vec3 dv = P(std::max(rr01, 0.05f), ang + 0.05f) - P(std::max(rr01, 0.05f), ang - 0.05f);
                glm::vec3 n = glm::cross(dv, du);
                n = glm::dot(n, n) > 1e-12f ? glm::normalize(n) : glm::vec3(0.0f, 1.0f, 0.0f);
                if (n.y < 0.0f) n = -n;
                v.push_back({ p, n, uvp(rr01, ang) });
            }
        }
        for (int k = 0; k < seg; ++k)
            for (int q = 0; q < rings; ++q) {
                const uint32_t i0 = b0 + uint32_t(k) * uint32_t(rings + 1) + uint32_t(q), i1 = i0 + uint32_t(rings + 1);
                tri(i0, i1, i1 + 1); tri(i0, i1 + 1, i0 + 1);
            }
        stem(c + glm::vec3(0.0f, 0.004f, 0.0f), 0.0045f, 2.0f * kPi * grnd());
    }
    // Flowers: three rings of curved, cupped petals round a crown of
    // stamens; the first opens wide beside the central pad, the second is
    // a smaller, half-closed one further out.
    auto flower = [&](glm::vec3 fc, float sz, float open) {
        // Four rings: the outer petals lie almost flat on the water, each
        // ring inside stands steeper and is a little shorter - the cup of a
        // real water lily. Petals are broad with a pointed tip.
        for (int ring = 0; ring < 4; ++ring) {
            const int n = ring == 0 ? 9 : (ring == 1 ? 9 : (ring == 2 ? 7 : 5));
            const float len = sz * (ring == 0 ? 0.105f : (ring == 1 ? 0.092f : (ring == 2 ? 0.074f : 0.055f)));
            const float wid = sz * (ring == 0 ? 0.034f : (ring == 1 ? 0.030f : (ring == 2 ? 0.024f : 0.018f)));
            const float rise = (ring == 0 ? 0.16f : (ring == 1 ? 0.55f : (ring == 2 ? 1.15f : 2.4f))) / open;
            for (int k = 0; k < n; ++k) {
                const float ang = 2.0f * kPi * (float(k) + 0.5f * float(ring) + 0.15f * grnd()) / float(n);
                const glm::vec3 dir(std::cos(ang), 0.0f, std::sin(ang)), side(-dir.z, 0.0f, dir.x);
                // Petal surface: 5 stations along, 3 across; it curves up
                // along its length and is cupped across.
                const uint32_t b0 = uint32_t(v.size());
                const int st = far_lod ? 2 : 7;
                for (int s2 = 0; s2 < st; ++s2) {
                    const float t = float(s2) / float(st - 1);
                    const float up_a = std::atan(rise) * (0.55f + 0.75f * t);          // steeper toward the tip
                    const glm::vec3 along = dir * std::cos(up_a) + glm::vec3(0.0f, std::sin(up_a), 0.0f);
                    const glm::vec3 mid = fc + dir * (sz * 0.010f) + (dir * std::cos(std::atan(rise) * 0.8f) +
                                          glm::vec3(0.0f, std::sin(std::atan(rise) * 0.8f), 0.0f)) * (len * t) +
                                          glm::vec3(0.0f, len * 0.18f * t * t, 0.0f);
                    // Lance shape: widest a third of the way out, drawn to a point.
                    const float hw = wid * std::pow(std::sin(kPi * std::pow(std::min(t * 0.97f + 0.03f, 1.0f), 0.62f)), 0.85f) * (1.0f - 0.25f * t);
                    const glm::vec3 nrm0 = glm::normalize(glm::cross(side, along));
                    for (int u2 = -1; u2 <= 1; ++u2) {
                        const float uu = float(u2);
                        const glm::vec3 p = mid + side * (hw * uu) + nrm0 * (hw * 0.35f * uu * uu);
                        v.push_back({ p, glm::normalize(nrm0 - side * (0.5f * uu)), { 1.0f, 0.0f } });
                    }
                }
                for (int s2 = 0; s2 + 1 < st; ++s2)
                    for (int u2 = 0; u2 < 2; ++u2) {
                        const uint32_t i0 = b0 + uint32_t(s2) * 3u + uint32_t(u2), i1 = i0 + 3u;
                        tri(i0, i1, i1 + 1); tri(i0, i1 + 1, i0 + 1);
                    }
            }
        }
        // Stamens: a crown of short golden filaments, and the centre.
        for (int k = 0; k < (far_lod ? 0 : 14); ++k) {
            const float ang = 2.0f * kPi * float(k) / 14.0f;
            const glm::vec3 dir(std::cos(ang), 0.0f, std::sin(ang));
            const glm::vec3 b = fc + dir * (sz * 0.006f) + glm::vec3(0.0f, sz * 0.006f, 0.0f);
            const glm::vec3 t = b + dir * (sz * 0.012f) + glm::vec3(0.0f, sz * 0.026f, 0.0f);
            const glm::vec3 sd(-dir.z * sz * 0.0025f, 0.0f, dir.x * sz * 0.0025f);
            const uint32_t b0 = uint32_t(v.size());
            v.push_back({ b - sd, dir, { 2.0f, 0.0f } }); v.push_back({ b + sd, dir, { 2.0f, 0.0f } });
            v.push_back({ t + sd, dir, { 2.0f, 0.0f } }); v.push_back({ t - sd, dir, { 2.0f, 0.0f } });
            tri(b0, b0 + 1, b0 + 2); tri(b0, b0 + 2, b0 + 3);
        }
        plant_lathe(v, ix, fc, glm::vec3(0.0f, 1.0f, 0.0f), { { 0.0f, 0.0f }, { sz * 0.012f, sz * 0.004f },
                    { sz * 0.010f, sz * 0.016f }, { 0.0f, sz * 0.020f } }, 8, 2.0f);
        stem(fc, 0.0055f, 2.0f * kPi * grnd());
    };
    flower(glm::vec3(0.15f, 0.030f, -0.12f), 1.25f, 1.0f);
    flower(glm::vec3(-0.30f, 0.040f, 0.24f), 0.85f, 0.55f);
}

// Sort instances by 32 m ground cell and emit one contiguous range per
// cell with a conservative AABB (instance size + raymarch-surface offset).
void sort_into_cells(std::vector<ScatterInstance>& v, uint32_t base,
                     std::vector<ScatterCell>& out) {
    const float kCell = 32.0f;
    auto key = [&](const ScatterInstance& s) {
        int cx = int(std::floor(s.pos_scale.x / kCell)) + 512;
        int cz = int(std::floor(s.pos_scale.z / kCell)) + 512;
        return cz * 1024 + cx;
    };
    std::stable_sort(v.begin(), v.end(),
                     [&](const ScatterInstance& a, const ScatterInstance& b) { return key(a) < key(b); });
    out.clear();
    for (size_t i = 0; i < v.size();) {
        const int k = key(v[i]);
        ScatterCell c{};
        c.mn = glm::vec3(1e9f);
        c.mx = glm::vec3(-1e9f);
        size_t j = i;
        for (; j < v.size() && key(v[j]) == k; ++j) {
            glm::vec3 p(v[j].pos_scale);
            float r = v[j].pos_scale.w * 1.5f + 0.5f;
            c.mn = glm::min(c.mn, p - glm::vec3(r, 1.5f, r));
            c.mx = glm::max(c.mx, p + glm::vec3(r, r * 2.0f + 1.5f, r));
        }
        c.first = base + uint32_t(i);
        c.count = uint32_t(j - i);
        out.push_back(c);
        i = j;
    }
}

} // namespace
