// cube_wall_relief.glsl - part of cube.frag (included there, in place; not a standalone shader).
// Castle wall box relief (br_*): solidity margin, exit, open-air probe, silhouette clip

// ---------------------------------------------------------------------
// Box relief: brick relief raymarched through the brush's AABB.
//
// Classic POM shifts UVs on a flat face, so the brush silhouette stays a
// perfectly straight line ("edges have no depth"). Following the bounded-
// box formulation (GPU Gems 2 ch.8) + POM silhouette clipping (discard
// rays that leave the surface bounds), the brush is treated as a volume:
// a point is SOLID when it lies below the height-field surface of every
// EXPOSED face (per-face mask from the CPU, so brush-brush seams stay
// closed). The view ray is marched from the rasterised entry point;
// rays that slip through a mortar groove at an exposed edge exit the box
// and are discarded -> bricks get a real stepped silhouette at corners
// and wall tops.
// ---------------------------------------------------------------------
vec2 br_face_uv(int f, vec3 w, float uv_scale, vec2 aspect) {
    int   a = f >> 1;
    float s = ((f & 1) == 1) ? 1.0 : -1.0;
    vec3 wp = w * uv_scale;
    vec2 uv = (a == 0) ? vec2(wp.z * s, wp.y)
            : (a == 1) ? vec2(wp.x, wp.z * s)
                       : vec2(wp.x * s, wp.y);
    return uv * aspect;
}

// Smooth 3-D value noise (0..1) - how worn a wall corner is at a point.
float br_hash3(vec3 p) {
    p = fract(p * vec3(0.1031, 0.1030, 0.0973));
    p += dot(p, p.yxz + 33.33);
    return fract((p.x + p.y) * p.z);
}
float br_noise(vec3 p) {
    vec3 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(mix(br_hash3(i),                 br_hash3(i + vec3(1, 0, 0)), f.x),
                   mix(br_hash3(i + vec3(0, 1, 0)), br_hash3(i + vec3(1, 1, 0)), f.x), f.y),
               mix(mix(br_hash3(i + vec3(0, 0, 1)), br_hash3(i + vec3(1, 0, 1)), f.x),
                   mix(br_hash3(i + vec3(0, 1, 1)), br_hash3(i + vec3(1, 1, 1)), f.x), f.y), f.z);
}
// Outward direction of the worn corner at the last point br_margin looked
// at (zero away from corners) - bends the shading normal round the corner.
vec3 g_br_round_n = vec3(0.0);
// Inside corner: the face of the neighbouring block this ray runs into
// (-1 = none) and where its plane lies. Its stones are part of the solid
// too, so the two walls meet along the uneven line where their stones
// actually touch instead of the ruler-straight line of the two boxes.
int   g_nb_f  = -1;
// True once the wall relief has run for this pixel.
bool  g_br_relief = false;
float g_nb_pl = 0.0;
// The neighbour is of another material (paving under a wall): it joins as
// a plain level surface, drawn with its own texture where the ray lands on it.
bool  g_nb_flat = false;
// Its own relief (height map, tile size, depth) - so what a wall pixel sees of
// the floor inside its recesses is exactly what the floor pixels next to it draw.
int   g_nb_h = -1;
float g_nb_scale = 1.0;
vec2  g_nb_asp = vec2(1.0);
float g_nb_dm = 0.0;

// Signed solidity margin (>= 0 -> inside the masonry). Only EXPOSED faces
// carry a surface; where a face is covered by a neighbouring block the
// stone simply continues (no boundary there at all), so the relief of one
// block runs on into the next without a seam. bind_face = the face whose
// surface constrains this point most (= the face hit).
float br_margin(vec3 w, vec3 C, vec3 H, int mask, float Dm, int h_idx,
                float lod, float uv_scale, vec2 aspect, int entry_face,
                out int bind_face) {
    vec3 rel = w - C;
    float m = 1e9;
    bind_face = entry_face;
    vec3 qv = vec3(0.0), qs = vec3(1.0);
    int  near_n = 0;
    for (int f = 0; f < 6; ++f) {
        if (((mask >> f) & 1) == 0) continue;
        int   a = f >> 1;
        float s = ((f & 1) == 1) ? 1.0 : -1.0;
        float d = H[a] - s * rel[a];               // depth below face f
        float mf = d;
        if (d < Dm) {
            float h = textureLod(u_height[h_idx],
                                 br_face_uv(f, w, uv_scale, aspect), lod).r;
            mf = d - Dm * (1.0 - h);
        }
        if (mf < m) { m = mf; bind_face = f; }
        if (mf < Dm) {
            ++near_n;
            if (Dm - mf > qv[a]) { qv[a] = Dm - mf; qs[a] = s; }
        }
    }
    // Where two (or three) exposed faces meet, the corner is not a razor
    // edge: it is rounded off and chipped, by an amount that wanders along
    // the edge - so a wall corner reads as worn stone from any direction,
    // not only where a joint happens to cut it.
    g_br_round_n = vec3(0.0);
    if (near_n >= 2) {
        float n = br_noise(w * 3.1) * 0.65 + br_noise(w * 9.7) * 0.35;
        float r = Dm * mix(0.20, 0.95, n);
        vec3  q = max(qv - (Dm - r), vec3(0.0));     // = max(r - mf, 0)
        int   k = int(q.x > 0.0) + int(q.y > 0.0) + int(q.z > 0.0);
        if (k >= 2) {
            m = min(m, r - length(q));
            g_br_round_n = q * qs;
        }
    }
    if (g_nb_f >= 0) {
        int   a = g_nb_f >> 1;
        float s = ((g_nb_f & 1) == 1) ? 1.0 : -1.0;
        float d = s * (g_nb_pl - w[a]);              // depth below the neighbour's face
        if (d > -0.06) {
            float mb = d;
            if (!g_nb_flat) {
                if (d < Dm) {
                    float h = textureLod(u_height[h_idx],
                                         br_face_uv(g_nb_f, w, uv_scale, aspect), lod).r;
                    mb = d - Dm * (1.0 - h);
                }
            } else if (g_nb_h >= 0 && d < g_nb_dm) {
                float h = textureLod(u_height[g_nb_h],
                                     br_face_uv(g_nb_f, w, g_nb_scale, g_nb_asp), lod).r;
                mb = d - g_nb_dm * (1.0 - h);
            }

            // Union of the two walls, with a little mortar packed into
            // the angle.
            const float k = 0.035;
            float hh = clamp(0.5 + 0.5 * (mb - m) / k, 0.0, 1.0);
            if (mb > m) { bind_face = g_nb_f; g_br_round_n = vec3(0.0); }
            m = mix(m, mb, hh) + k * hh * (1.0 - hh);
        }
    }
    return m;
}

// Where the ray leaves the block through one of its exposed faces
// (covered faces are no boundary). Returns the distance; `face` = -1 and a
// huge distance when it never does.
float br_exit(vec3 ro, vec3 rd, vec3 C, vec3 H, int mask, out int face) {
    float t_exit = 1e9;
    face = -1;
    for (int a = 0; a < 3; ++a) {
        if (abs(rd[a]) < 1e-6) continue;
        bool pos = rd[a] > 0.0;
        int  f = a * 2 + (pos ? 1 : 0);
        if (((mask >> f) & 1) == 0) continue;
        float t = (C[a] + (pos ? H[a] : -H[a]) - ro[a]) / rd[a];
        if (t < t_exit) { t_exit = max(t, 0.0); face = f; }
    }
    return t_exit;
}

// Is the point where a ray left a block in open air? Asked of the real
// castle geometry (blocks + tower / arch meshes carry TLAS mask bit 7):
// if the first thing the ray meets from there is the INSIDE of a surface,
// the point lies within another piece of masonry and the wall is solid
// there. Only a "yes" lets a pixel be cut away, so a wrong answer can
// never open a hole into a neighbouring block.
bool br_open_air(vec3 p, vec3 rd) {
    // Every surface the ray crosses from here is counted: leaving a solid
    // (+1) or entering one (-1). A point in open air enters and leaves each
    // piece of masonry in pairs; a point inside stone leaves once more than
    // it enters. (Judging by the first surface alone failed where pieces
    // overlap - a corbel poking into a tower shell is entered first, and
    // the wall next to the tower was cut open.)
    rayQueryEXT rq;
    rayQueryInitializeEXT(rq, topLevelAS, gl_RayFlagsNoOpaqueEXT, 0x80u,
                          p, 0.004, rd, 30.0);
    int inside = 0;
    while (rayQueryProceedEXT(rq)) {
        if (rayQueryGetIntersectionTypeEXT(rq, false) ==
            gl_RayQueryCandidateIntersectionTriangleEXT)
            inside += BR_FRONT(rayQueryGetIntersectionFrontFaceEXT(rq, false)) ? -1 : 1;
    }
    return inside <= 0;
}

// Relief silhouette on curved meshes (round towers), from the real
// geometry instead of a curvature estimate (which punched holes wherever
// the profile has a sharp step: string courses, parapet rims). The view
// ray goes on into the stone from this pixel; the relief surface lies
// `depth_m` below the mesh surface. If the ray comes OUT of the masonry
// again before it has got that deep, and out into open air, it passed over
// the stone: this pixel shows what is behind. c = N.V.
bool pom_silhouette_clip(float depth_m, float c, float curv) {
    c = clamp(c, 0.0, 1.0);
    // (curv >= 100: a flat-faced mesh that asked for its edges to be cut - LevelMesh::relief > 1.
    // There is no curvature to go by, and the edge can be met at any angle.)
    bool edges = curv >= 100.0;
    if (depth_m < 0.008 || (!edges && (curv < 0.02 || c > 0.5))) return false;
    vec3  rd = normalize(vWorldPos - scene.camera_pos.xyz);
    float t_hit = min(depth_m / max(c, 0.04), 4.0);
    rayQueryEXT rq;
    rayQueryInitializeEXT(rq, topLevelAS, gl_RayFlagsOpaqueEXT, 0x80u,
                          vWorldPos, 0.004, rd, t_hit);
    while (rayQueryProceedEXT(rq)) {}
    if (rayQueryGetIntersectionTypeEXT(rq, true) !=
        gl_RayQueryCommittedIntersectionTriangleEXT) return false;   // still in the stone
    if (BR_FRONT(rayQueryGetIntersectionFrontFaceEXT(rq, true))) return false;   // runs into more masonry
    float t_out = rayQueryGetIntersectionTEXT(rq, true);
    return br_open_air(vWorldPos + rd * t_out, rd);
}

