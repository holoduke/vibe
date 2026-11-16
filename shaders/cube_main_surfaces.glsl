// cube_main_surfaces.glsl - part of cube.frag (included there, in place; not a standalone shader).
// main(): non-terrain surfaces (wall box relief march, castle meshes / towers, props)

        // SPOM (parallax-occlusion) — for any material that has a height
        // map registered in u_height[]. height_idx_for_albedo() returns
        // -1 for non-SPOM materials so they fall through to the standard
        // triplanar path. obj_space brushes (dyn props) skip SPOM because
        // their normal isn't world-aligned; terrain skips because it has
        // its own raymarched normal path.
        int spom_h_idx = height_idx_for_albedo(albedo_idx_raw);
        bool spom_path = use_albedo && spom_h_idx >= 0 && !obj_space &&
                         !is_terrain && !is_mesh_draw &&
                         ((albedo_idx_raw != 5 && albedo_idx_raw != 6) || N.y > 0.9);
        if (spom_path) {
            // pc.model is translate * scale (no rotation for static brushes,
            // and obj_space — which would rotate — already failed the gate
            // above). Each column's length is the brush's world-space size
            // along that local axis, which for axis-aligned brushes equals
            // the face dimensions we need for the silhouette discard.
            vec3 face_size = vec3(length(pc.model[0].xyz),
                                  length(pc.model[1].xyz),
                                  length(pc.model[2].xyz));
            int axis;
            vec3 spom_T, spom_B, spom_face_n;
            bool spom_disc;
            vec3 spom_world;     // lateral-only (RT origin; no inward push)
            vec3 spom_recessed;  // lateral + inward (SSDM depth-override)
            vec2 spom_uv0;
            vec2 spom_uv_off = spom_uv(sample_pos, proj_n, face_size, scale,
                                        spom_h_idx,
                                        axis, spom_T, spom_B, spom_face_n,
                                        spom_disc, spom_world, spom_recessed,
                                        spom_uv0);
            // Wall displacement mode (UI "Wall displacement"): carried in
            // the reserved grass_side_lit_params.z slot. 0 = legacy SPOM
            // (ray-query seam probe), 1 = SSDM depth-override relief.
            // 0 = Flat (no per-pixel parallax shift; bricks look
            //     painted on with normal-mapped relief only),
            // 1 = SPOM legacy (ray-query silhouette resolution),
            // 2 = SSDM (depth-override silhouette + recessed depth).
            int spom_mode = int(scene.grass_side_lit_params.z + 0.5);
            vec3  br_round_n = vec3(0.0);
            bool  br_nb_hit   = false;       // the ray landed on the neighbour of another material
            int   br_nb_idx   = 0;
            float br_nb_scale = 1.0;
            vec3  br_nb_tint  = vec3(1.0);
            float br_ao       = 1.0;         // soft shadow in the angle between wall and floor
            float br_relief_m = 0.0;   // box relief depth in metres (0 = not that path)
            if (spom_mode == 0) {
                // Flat path: forget the parallax UV shift entirely.
                // Bricks use the un-displaced sample point; silhouette
                // is just the geometric edge (no sky-substitution).
                spom_uv_off = spom_uv0;
                spom_world  = vWorldPos;
            } else
            if (spom_mode == 2 && pc.grass_params.w > 0.5) {
                // ---- Box relief (static brushes with a face mask) ----
                int   br_mask = int(pc.grass_params.z + 0.5);
                int   br_h    = clamp(spom_h_idx, 0, 5);
                vec2  br_asp  = (br_h == 0 || br_h == 2) ? vec2(0.5, 1.0)
                                                         : vec2(1.0);
                vec3  br_C    = pc.model[3].xyz;
                vec3  br_H    = face_size * 0.5;
                float br_dist = distance(vWorldPos, scene.camera_pos.xyz);
                // Full relief up close, fades out by 60 m (sub-pixel there).
                // Depth proportional to the texture tile (castle walls tile
                // every ~8 m -> big stones -> ~20 cm mortar relief; a fixed
                // 4 cm read as flat). Capped so thin brushes stay solid.
                float br_Dm   = clamp(0.05 / max(scale, 0.01), 0.06, 0.3);
                br_Dm = min(br_Dm, 0.3 * min(min(face_size.x, face_size.y),
                                             face_size.z));
                br_Dm *= max(0.0, scene.spom_params.x) *
                         (1.0 - smoothstep(35.0, 70.0, br_dist));
                if (br_Dm > 0.002) {
                    vec3  ro = vWorldPos;
                    vec3  rd = normalize(vWorldPos - scene.camera_pos.xyz);
                    int   br_ef = axis * 2 + (spom_face_n[axis] > 0.0 ? 1 : 0);
                    // Blocks that are in the depth pre-pass (floors) must
                    // cover every pixel they rasterise: nothing behind them
                    // survives the depth test. Only walls may be cut away.
                    bool  br_cut = albedo_idx_raw == 1 || albedo_idx_raw == 4;
                    // Past depth Dm below the entry face every point is
                    // solid, so the march never needs to go further.
                    // (True angle, however flat: with the camera almost in
                    // the plane of a face - hugging a wall, peering round a
                    // corner - the ray runs a long way through the relief
                    // layer and must be followed until it meets a stone or
                    // leaves the block; cutting it short painted smeared
                    // stripes where there is nothing but air.)
                    float cosn  = max(dot(rd, -spom_face_n), 0.05);
                    float t_deep = br_Dm / max(dot(rd, -spom_face_n), 0.002) * 1.02;
                    float lod = textureQueryLod(u_height[br_h], spom_uv0).x;
                    // More steps at grazing angles (long march, thin
                    // features) - fewer head-on.
                    int   kSteps = int(mix(48.0, 16.0, cosn));
                    float dt = 0.0;
                    // Start offset dithered per pixel: the fixed layers of a
                    // linear march show as bands on walls seen at a slant.
                    float t_lo = 0.0, t_hi = -1.0;
                    float t_j = 0.0;
                    int   br_n = 0;
                    int   bf = br_ef;
                    br_relief_m = br_Dm;
                    g_br_relief = true;
                    // The ray is marched until it meets stone. If it first
                    // leaves the block through an exposed face, that is a
                    // silhouette only when the exit point really is in open
                    // air; where another block or a tower stands there, that
                    // face stops being a boundary and the march goes on.
                    // A side face of this block only shapes the stone here
                    // (relief cut in from the side, worn corner) if it is
                    // open at THIS spot. Where another block sits flush
                    // against it - a lintel on a door post, the next piece
                    // of the same wall - there is no corner, the masonry
                    // just carries on. Asked once per pixel, and only within
                    // reach of such a face.
                    for (int f = 0; f < 6; ++f) {
                        if (f == br_ef || ((br_mask >> f) & 1) == 0) continue;
                        int   a = f >> 1;
                        float sg = ((f & 1) == 1) ? 1.0 : -1.0;
                        if (a == (br_ef >> 1)) continue;
                        float pl = br_C[a] + sg * br_H[a];
                        if (sg * (pl - ro[a]) > br_Dm * 1.3) continue;
                        vec3 q = ro - spom_face_n * (br_Dm * 0.5);
                        q[a] = pl + sg * 0.012;
                        vec3 od = vec3(0.0); od[a] = sg;
                        if (!br_open_air(q, normalize(od + vec3(0.013, 0.017, 0.011))))
                            br_mask &= ~(1 << f);
                    }
                    // A side face that is only partly covered by a neighbour
                    // (wall beside a doorway: open at the door, joined to the
                    // lintel above it) counts as open for the whole block.
                    // Close to such a face, ask whether stone really ends
                    // there AT THIS PIXEL; if a neighbour carries on, the face
                    // is no edge here - no worn corner, no cut - and the seam
                    // stays invisible.
                    for (int sf = 0; sf < 6; ++sf) {
                        if (((br_mask >> sf) & 1) == 0 || (sf >> 1) == axis) continue;
                        int   sa = sf >> 1;
                        float ss = ((sf & 1) == 1) ? 1.0 : -1.0;
                        float sd = br_H[sa] - ss * (ro[sa] - br_C[sa]);
                        if (sd > br_Dm * 1.3) continue;
                        vec3 sp = ro - spom_face_n * 0.03;
                        sp[sa] = br_C[sa] + ss * (br_H[sa] + 0.012);
                        if (!br_open_air(sp, rd)) br_mask &= ~(1 << sf);
                    }
                    // Which wall does this ray run into? Asked just above
                    // the face, along it, as far as the march can travel.
                    g_nb_f = -1;
                    g_nb_flat = false;
                    // Soft shadow where wall and floor meet: how close is
                    // the other one? (walls look down, floors look round.)
                    if (br_dist < 30.0) {
                        float reach = 0.7, nearest = reach;
                        vec3  ao_o = ro + spom_face_n * 0.04;
                        int   n_dir = (axis == 1) ? 8 : 1;        // floors: 8 directions (round towers, corners)
                        for (int k = 0; k < n_dir; ++k) {
                            vec3 ad = (axis != 1) ? vec3(0.0, -1.0, 0.0)
                                    : vec3(cos(float(k) * 0.7853982), 0.0, sin(float(k) * 0.7853982));
                            rayQueryEXT aq;
                            rayQueryInitializeEXT(aq, topLevelAS, gl_RayFlagsOpaqueEXT, 0x80u,
                                                  ao_o, 0.0, ad, reach);
                            while (rayQueryProceedEXT(aq)) {}
                            if (rayQueryGetIntersectionTypeEXT(aq, true) ==
                                gl_RayQueryCommittedIntersectionTriangleEXT)
                                nearest = min(nearest, rayQueryGetIntersectionTEXT(aq, true));
                        }
                        if (axis != 1)
                            nearest = min(nearest, max(ao_o.y - textureLod(u_heightmap_full, ao_o.xz / 2048.0 + 0.5, 0.0).r, 0.0));
                        float a01 = clamp(nearest / reach, 0.0, 1.0);
                        a01 = 0.5 * a01 + 0.5 * a01 * a01 * (3.0 - 2.0 * a01);   // stays dark near the contact
                        br_ao = mix(1.0, mix(0.48, 1.0, a01),
                                    1.0 - smoothstep(20.0, 30.0, br_dist));
                    }
                    {
                        vec3  td = rd - spom_face_n * dot(rd, spom_face_n);
                        float tl = length(td);
                        if (tl > 0.05) {
                            td /= tl;
                            vec3  po = ro + spom_face_n * 0.015;
                            rayQueryEXT nq;
                            rayQueryInitializeEXT(nq, topLevelAS, gl_RayFlagsOpaqueEXT, 0x80u,
                                                  po, 0.0, td, min(t_deep * tl, 2.5) + 0.02);
                            while (rayQueryProceedEXT(nq)) {}
                            if (rayQueryGetIntersectionTypeEXT(nq, true) ==
                                    gl_RayQueryCommittedIntersectionTriangleEXT &&
                                rayQueryGetIntersectionInstanceCustomIndexEXT(nq, true) == kStaticBlasSentinel &&
                                BR_FRONT(rayQueryGetIntersectionFrontFaceEXT(nq, true))) {
                                int  prim = rayQueryGetIntersectionPrimitiveIndexEXT(nq, true);
                                int  cf   = (prim % kCubeTrisPerBox) / 2;   // +X -X +Y -Y +Z -Z
                                int  na   = cf >> 1;
                                Material nm = materials[prim / kCubeTrisPerBox];
                                if (na != axis) {
                                    g_nb_f  = na * 2 + ((cf & 1) == 0 ? 1 : 0);
                                    g_nb_pl = (po + td * rayQueryGetIntersectionTEXT(nq, true))[na];
                                    g_nb_flat = !(int(nm.tex.x) == albedo_idx_raw &&
                                                  abs(nm.tex.z - scale) < 0.002);
                                    br_nb_idx   = clamp(int(nm.tex.x), 0, kTextureCount - 1);
                                    br_nb_scale = max(nm.tex.z, 0.001);
                                    br_nb_tint  = nm.color.rgb;
                                    if (int(nm.tex.x) < 0) g_nb_f = -1;
                                    if (g_nb_flat) {
                                        // A floor never shows the wall beside
                                        // it (from a low angle that read as
                                        // seeing under the paving): only the
                                        // surface a wall STANDS ON joins in.
                                        if (g_nb_f != 3) g_nb_f = -1;
                                        g_nb_h     = height_idx_for_albedo(br_nb_idx);
                                        g_nb_scale = br_nb_scale;
                                        g_nb_asp   = (g_nb_h == 0 || g_nb_h == 2) ? vec2(0.5, 1.0) : vec2(1.0);
                                        g_nb_dm    = clamp(0.05 / br_nb_scale, 0.06, 0.3) *
                                                     max(0.0, scene.spom_params.x) *
                                                     (1.0 - smoothstep(35.0, 70.0, br_dist));
                                        g_nb_dm    = min(g_nb_dm, 0.15);
                                    }
                                }
                            }
                        }
                    }
                    for (int pass = 0; pass < 3 && t_hi < 0.0; ++pass) {
                        int   xf;
                        float t_exit = br_exit(ro, rd, br_C, br_H, br_mask, xf);
                        // The march runs all the way to the exit. Deeper than
                        // Dm below the entry face the stone is USUALLY solid
                        // (and the march stops there by itself) - but not
                        // next to another open face, whose own relief and
                        // worn corner cut into it: there the ray is still in
                        // air and has to go on to the exit.
                        float L = t_exit;
                        if (pass == 0) {
                            // Steps sized to the stretch actually travelled
                            // (never coarser than ~half a joint width).
                            dt  = min(min(t_exit, t_deep), 8.0) / float(kSteps);
                            dt  = min(dt, max(br_Dm * 0.5, 0.03));
                            t_j = dt * ign(gl_FragCoord.xy);
                        }
                        for (float t = (pass == 0 ? t_j : t_lo); t < L && br_n < 160; t += dt, ++br_n) {
                            if (br_margin(ro + rd * t, br_C, br_H, br_mask, br_Dm,
                                          br_h, lod, scale, br_asp, br_ef, bf) >= 0.0) {
                                t_hi = t;
                                break;
                            }
                            t_lo = t;
                        }
                        if (t_hi >= 0.0) break;
                        if (t_exit > 1e8 || br_n >= 160) { t_hi = t_lo + dt; break; }   // no way out: solid
                        if (br_cut && br_open_air(ro + rd * t_exit, rd)) discard;
                        br_mask &= ~(1 << xf);
                        t_lo = max(t_lo, t_exit - dt);
                    }
                    if (t_hi < 0.0) t_hi = t_lo + dt;
                    // Binary refinement of the surface crossing.
                    for (int i = 0; i < 5 && t_hi > 0.0; ++i) {
                        float tm = 0.5 * (t_lo + t_hi);
                        int bft;
                        if (br_margin(ro + rd * tm, br_C, br_H, br_mask, br_Dm,
                                      br_h, lod, scale, br_asp, br_ef, bft) >= 0.0)
                            t_hi = tm;
                        else
                            t_lo = tm;
                    }
                    vec3 p_hit = ro + rd * t_hi;
                    br_margin(p_hit, br_C, br_H, br_mask, br_Dm, br_h, lod,
                              scale, br_asp, br_ef, bf);
                    br_round_n = g_br_round_n;
                    // Shade with the frame of the face actually hit (at an
                    // edge that can be the neighbouring face of the box).
                    int   fa = bf >> 1;
                    float fs = ((bf & 1) == 1) ? 1.0 : -1.0;
                    axis = fa;
                    if (fa == 0) {
                        spom_face_n = vec3(fs, 0.0, 0.0);
                        spom_T = vec3(0.0, 0.0, fs); spom_B = vec3(0.0, 1.0, 0.0);
                    } else if (fa == 1) {
                        spom_face_n = vec3(0.0, fs, 0.0);
                        spom_T = vec3(1.0, 0.0, 0.0); spom_B = vec3(0.0, 0.0, fs);
                    } else {
                        spom_face_n = vec3(0.0, 0.0, fs);
                        spom_T = vec3(fs, 0.0, 0.0); spom_B = vec3(0.0, 1.0, 0.0);
                    }
                    spom_uv_off = br_face_uv(bf, p_hit, scale, br_asp);
                    if (g_nb_flat && bf == g_nb_f) {
                        br_nb_hit = true;
                        int nh = height_idx_for_albedo(br_nb_idx);
                        spom_uv_off = br_face_uv(bf, p_hit, br_nb_scale,
                                                 (nh == 0 || nh == 2) ? vec2(0.5, 1.0) : vec2(1.0));
                    }
                    // Depth + RT origin stay on the rasterised face: the box
                    // BLAS has culling disabled, so lighting from a point
                    // recessed INSIDE the box would self-shadow every brick.
                    // The silhouette comes entirely from the discard above.
                    spom_world = vWorldPos;
                }
            } else
            if (spom_mode == 2) {
                // SSDM mode = shell-mapping silhouette + depth-override
                // relief. cube.vert extruded wall verts outward by
                // kShellThickness * spom_strength; cube.frag here marches
                // the height map inward and discards pixels in the shell
                // band where the brick at this lateral UV doesn't reach
                // up far enough -- those discards produce the stepped
                // per-brick silhouette past the brush corner.
                //
                // Floors (axis == 1) are excluded -- cube.vert leaves
                // them un-extruded, so shell_offset would be 0 and the
                // discard would wipe every brick crevice on the floor.
                float ss_strength = max(0.0, scene.spom_params.x);
                if (axis != 1 && ss_strength > 0.01) {
                    // The corner "sky-through-bricks" silhouette discard
                    // that used to live here is REMOVED. It wrote
                    // transparent far-plane pixels wherever the march
                    // went deep at a brush corner — with the larger
                    // stone tiling that punched visible see-through
                    // holes in the walls around the gate (and the same
                    // class of artifact had already been reported twice
                    // before). The depth-override relief below carries
                    // the visual interest; corners stay solid.
                    float parallax_hit_depth = dot(vWorldPos - spom_recessed,
                                                   spom_face_n);
                    // Face-on gate on the recessed depth write. At
                    // grazing view angles a few cm of recession along
                    // the face normal becomes a LARGE depth change,
                    // opening a screen-space band at wall silhouettes
                    // where later passes (grass, terrain behind) win
                    // the depth test and bleed INTO the wall — read as
                    // "walls are transparent from certain angles".
                    {
                        vec3 Vw_g = normalize(scene.camera_pos.xyz - vWorldPos);
                        float vn_g = abs(dot(Vw_g, spom_face_n));
                        parallax_hit_depth *= smoothstep(0.30, 0.60, vn_g);
                    }
                    // Brick reaches up to this fragment -- write
                    // the parallax-hit depth so downstream passes
                    // see the correct brick-top geometry.
                    // pc.mvp maps OBJECT space to clip; static brushes are
                    // translate*scale only, so undo the model transform
                    // analytically instead of a generic per-fragment
                    // mat4 inverse() (~100 ALU on every wall pixel).
                    vec3 hit_world = vWorldPos - proj_n * parallax_hit_depth;
                    vec3 inv_sc = 1.0 / vec3(dot(pc.model[0].xyz, pc.model[0].xyz),
                                             dot(pc.model[1].xyz, pc.model[1].xyz),
                                             dot(pc.model[2].xyz, pc.model[2].xyz));
                    vec3 rel_h  = hit_world - pc.model[3].xyz;
                    vec3 hit_obj = vec3(dot(rel_h, pc.model[0].xyz),
                                        dot(rel_h, pc.model[1].xyz),
                                        dot(rel_h, pc.model[2].xyz)) * inv_sc;
                    vec4 rc = pc.mvp * vec4(hit_obj, 1.0);
                    gl_FragDepth = max(gl_FragCoord.z, rc.z / rc.w);
                }
            } else if (spom_mode == 3) {
                // ---- Godot-style POM + Silhouette Clipping (full port) ----
                //
                // Faithful port of the user-supplied Godot shader, adapted
                // to our triplanar tiled-UV walls. Three parts:
                //
                //   1. POM march with reference_plane = 0.5
                //      Independent of the existing spom_uv() (which goes
                //      inward-only) so bricks can stick BOTH OUT and IN
                //      of the geometric face. Same loop as Godot's:
                //      starts at base_uv + P * reference_plane, marches
                //      backward, contact-refines with binary search.
                //   2. Silhouette UV clipping (per-tile-adapted)
                //      Godot's `if (base_uv outside [0,1]) discard` —
                //      adapted as "if the parallax shift crossed the
                //      local brick-tile boundary, discard". Each brick
                //      texture tile is [0,1] in fract(uv) space; if the
                //      pre-fract shift goes outside the tile of the
                //      starting UV, the brick at this lateral position
                //      doesn't extend that far -> sky shows through.
                //   3. Curved silhouette (NdotV + curvature mask)
                //      Godot's `use_curved_silhouette` block — with the
                //      `fwidth(NORMAL)` curvature mask. On flat brush
                //      faces fwidth(N) is 0 so the mask is 0 and this
                //      part doesn't fire (matches Godot's actual
                //      behaviour on flat surfaces — the effect is for
                //      curved meshes like the sphere in their demo).
                //      Kept for correctness in case a tessellated /
                //      curved brush is added later.
                //
                // Self-shadow (GetParallaxShadow) intentionally skipped:
                // cube.frag has its own RT-based shadow + GI pipeline,
                // re-implementing Godot's height-march would conflict
                // with that. Same for the PBR light() function.
                if (axis != 1) {
                    // ---- Setup: tangent-space view, march params ----
                    vec3  V_w = normalize(scene.camera_pos.xyz - vWorldPos);
                    vec3  V_t = vec3(dot(V_w, spom_T),
                                     dot(V_w, spom_B),
                                     dot(V_w, spom_face_n));
                    int   h_idx = clamp(spom_h_idx, 0, 5);

                    // Distance fade — same as spom_uv() so mode 3 looks
                    // consistent with the other modes' fade behaviour.
                    float d = distance(vWorldPos, scene.camera_pos.xyz);
                    float depth_w = 1.0 - smoothstep(8.0, 30.0, d);

                    // Skip path: grazing or distance-faded -> just shade
                    // the un-displaced sample, no silhouette clipping
                    // (the parallax shift is too small to be meaningful).
                    if (V_t.z < 0.2 || depth_w <= 0.001 ||
                        max(0.0, scene.spom_params.x) < 0.01) {
                        spom_uv_off = spom_uv0;
                        spom_world  = vWorldPos;
                    } else {
                        // ---- Part 1: POM march with reference_plane ----
                        // Use the same scale formula as spom_uv() so the
                        // brick depth matches across modes.
                        const float kHeightScale  = 0.04;
                        const float kReferencePlane = 0.5;
                        float spom_strength = max(0.0, scene.spom_params.x);
                        // Per-axis number of layers (Godot mixes by
                        // |z_view|: head-on -> few layers, grazing ->
                        // many). 8..32 here, same defaults Godot ships.
                        float v_z_abs = clamp(abs(V_t.z), 0.0, 1.0);
                        float num_layers = mix(32.0, 8.0, v_z_abs);
                        float layer_depth = 1.0 / num_layers;
                        // Tangent-space shift per unit depth, scaled by
                        // depth_w for the same distance-fade as the
                        // other SPOM modes.
                        vec2  P = (V_t.xy / max(V_t.z, 0.1)) *
                                  (kHeightScale * spom_strength * depth_w);
                        // Match spom_uv()'s 2:1 texture-aspect handling so
                        // the march direction agrees with the scaled UVs.
                        P *= (h_idx == 0 || h_idx == 2) ? vec2(0.5, 1.0)
                                                        : vec2(1.0);
                        vec2  delta = P / num_layers;

                        // Start the march at the reference plane.
                        vec2  ofs = spom_uv0 + P * kReferencePlane;
                        float depth_at_ofs = 1.0 -
                            texture(u_height[h_idx], ofs).r;
                        float current_depth = 0.0;
                        // Steep parallax: march backward (ofs -= delta)
                        // until current_depth catches the height map's
                        // depth. Capped at 32 — matches num_layers max.
                        for (int i = 0; i < 32; ++i) {
                            if (current_depth >= depth_at_ofs) break;
                            ofs -= delta;
                            depth_at_ofs = 1.0 -
                                texture(u_height[h_idx], ofs).r;
                            current_depth += layer_depth;
                        }
                        // Contact refinement (binary search). 3 iters
                        // matches Godot's default `heightmap_refinement_steps`.
                        ofs += delta;
                        current_depth -= layer_depth;
                        vec2  half_delta = delta * 0.5;
                        float half_depth = layer_depth * 0.5;
                        for (int i = 0; i < 3; ++i) {
                            ofs -= half_delta;
                            current_depth += half_depth;
                            float dh = 1.0 - texture(u_height[h_idx], ofs).r;
                            if (current_depth > dh) {
                                ofs += half_delta;
                                current_depth -= half_depth;
                            }
                            half_delta *= 0.5;
                            half_depth *= 0.5;
                        }
                        spom_uv_off = ofs;

                        // ---- Part 2: Per-tile silhouette clipping ----
                        // Godot's "discard if base_uv outside [0,1]"
                        // adapted to tiled UVs. Compute the shift from
                        // the un-displaced UV; if the fractional position
                        // crossed the tile boundary, the brick at this
                        // lateral position doesn't extend that far ->
                        // discard so sky shows through.
                        vec2 shift = ofs - spom_uv0;
                        vec2 base_frac = fract(spom_uv0);
                        vec2 ofs_in_tile = base_frac + shift;
                        if (ofs_in_tile.x < 0.0 || ofs_in_tile.x > 1.0 ||
                            ofs_in_tile.y < 0.0 || ofs_in_tile.y > 1.0) {
                            outColor     = vec4(0.0);
                            outMotion    = vec2(0.0);
                            gl_FragDepth = 1.0;
                            return;
                        }

                        // ---- Part 3: Curved silhouette (Godot-style) ----
                        // Curvature mask via fwidth(N) / fwidth(VERTEX) —
                        // 0 on flat surfaces (our brush faces), 1 on
                        // curved. Multiplied into horizon_factor so this
                        // part does nothing on flat walls (matches
                        // Godot's actual behaviour on flat meshes).
                        // Kept verbatim for correctness in case a curved
                        // brush is added later.
                        float NdotV = abs(dot(spom_face_n, V_w));
                        float normal_change  = length(fwidth(vNormal));
                        float pixel_size     = length(fwidth(vWorldPos));
                        float true_curvature = normal_change /
                                               max(pixel_size, 0.0001);
                        const float kCurvatureActivation  = 0.05;
                        float curvature_mask =
                            smoothstep(0.0, kCurvatureActivation,
                                       true_curvature);
                        const float kHorizonSafeThreshold = 0.30;
                        const float kHorizonFalloffPower  = 2.0;
                        const float kHorizonClipStrength  = 1.0;
                        const float kHorizonHeightBias    = 0.0;
                        float t = clamp(
                            1.0 - NdotV /
                            max(kHorizonSafeThreshold, 0.001), 0.0, 1.0);
                        float horizon_factor =
                            pow(t, kHorizonFalloffPower) * curvature_mask;
                        float height_threshold = clamp(
                            horizon_factor * kHorizonClipStrength,
                            0.0, 1.0);
                        float surface_h = texture(u_height[h_idx],
                                                  ofs).r -
                                          kHorizonHeightBias;
                        if (surface_h < height_threshold) {
                            outColor     = vec4(0.0);
                            outMotion    = vec2(0.0);
                            gl_FragDepth = 1.0;
                            return;
                        }

                        // ---- Pixel depth offset (write_to_depth) ----
                        // Godot formula:
                        //   local_z = (current_depth - ref) * scale
                        //   view_dist = local_z / max(N·V, 0.0001)
                        //   offset = VERTEX - VIEW * view_dist
                        //   DEPTH = clip_pos.z / clip_pos.w
                        float local_z = (current_depth - kReferencePlane) *
                                        kHeightScale * spom_strength;
                        float view_dist = local_z /
                                          max(dot(spom_face_n, V_w), 0.0001);
                        // VIEW in Godot is camera-relative; in world we
                        // displace along V_w (camera→fragment) by -view_dist
                        // to push toward camera when local_z < 0 (brick
                        // out) or away when > 0 (brick in).
                        vec3 hit_world = vWorldPos - V_w * view_dist;
                        // Same analytic translate*scale un-transform as the
                        // mode-2 path — avoids a per-fragment mat4 inverse().
                        vec3 inv_sc3 = 1.0 / vec3(dot(pc.model[0].xyz, pc.model[0].xyz),
                                                  dot(pc.model[1].xyz, pc.model[1].xyz),
                                                  dot(pc.model[2].xyz, pc.model[2].xyz));
                        vec3 rel_h3  = hit_world - pc.model[3].xyz;
                        vec3 hit_obj3 = vec3(dot(rel_h3, pc.model[0].xyz),
                                             dot(rel_h3, pc.model[1].xyz),
                                             dot(rel_h3, pc.model[2].xyz)) * inv_sc3;
                        vec4 rc = pc.mvp * vec4(hit_obj3, 1.0);
                        gl_FragDepth = max(gl_FragCoord.z, rc.z / rc.w);

                        // Update spom_world for the downstream RT lighting
                        // shaders so shadow / AO rays trace from the
                        // displaced surface.
                        spom_world = hit_world;
                    }
                }
            } else if (spom_disc) {
                // ---- Legacy SPOM: ray-query seam probe (original path) ----
                //
                // Reworked seam ray (trade-off, not a clean fix). Old
                // version fired the ray purely laterally along this
                // wall's tangent plane — at coplanar wall segments
                // meeting edge-to-edge it skimmed parallel to both
                // faces and never hit the neighbour, so silhouette
                // discarded → user saw dark sky panes at the seam.
                //
                // New version fires perpendicularly INTO the wall plane
                // (-face_n) FROM the displaced lateral point. This
                // reliably catches a coplanar neighbour: the neighbour's
                // face is in the same plane, our ray crosses it within
                // ~10 mm. BUT the discard fires on a SINGLE-axis
                // overshoot (vis_T OR vis_B past the padded edge), so
                // the displaced point is often still inside the current
                // brush on the other axis. The -face_n probe then hits
                // the current brush's own front face, has_neighbour
                // reports true, and we fall back to flat un-parallaxed
                // brick. The visible net: no more dark sky panes (the
                // user's complaint), at the cost of silhouette
                // extension at true outside corners — those now also
                // fall back to flat instead of showing sky-through
                // bumped bricks. Accepted trade-off; revisit only if
                // outside-corner silhouette becomes desirable again
                // (would need instance-ID rejection or per-axis ray
                // dispatch — see commit notes).
                bool has_neighbour = false;
                {
                    vec3 origin = spom_world + spom_face_n * 0.005;
                    vec3 dir    = -spom_face_n;
                    rayQueryEXT rq_seam;
                    rayQueryInitializeEXT(rq_seam, topLevelAS,
                                          gl_RayFlagsTerminateOnFirstHitEXT |
                                          gl_RayFlagsOpaqueEXT,
                                          0xFFu,
                                          origin, 0.001, dir, 0.12);
                    while (rayQueryProceedEXT(rq_seam)) {}
                    has_neighbour =
                        rayQueryGetIntersectionTypeEXT(rq_seam, true) ==
                        gl_RayQueryCommittedIntersectionTriangleEXT;
                }
                if (!has_neighbour) {
                    // True outer corner — silhouette extension. Sky
                    // shows behind the bumped bricks. depth=1 lets
                    // compose's sky branch (depth >= 0.99999) paint
                    // the sky color over this pixel.
                    outColor     = vec4(0.0);
                    outMotion    = vec2(0.0);
                    gl_FragDepth = 1.0;
                    return;
                }
                // Inner seam — fall back to the un-parallaxed wall so
                // we don't black-hole the joint. Lose silhouette bricks
                // at this specific edge; the neighbour brush conceals
                // the loss.
                spom_uv_off = spom_uv0;
                spom_world  = vWorldPos;
            }
            // RT origins from the brick crevice — but only on WALLS
            // (axis 0 / 2). On floors (axis 1) the lateral tangent
            // offset can push the AO/shadow ray origin past the floor
            // brush's XZ extent, where rays land in adjacent wall
            // colliders and produce noise that reads as gray patches
            // when TAA history is unavailable (camera moving / falling).
            // Floors keep shading_pos = vWorldPos so AO/shadow rays
            // trace from the actual rasterised hit point.
            // Depth cues for the deferred path (cavity + self-shadow), see
            // pom_shade(). The old forward path got them from RT rays fired
            // out of the displaced crevice; deferred shades the flat face.
            {
                float fadeB = 1.0 - smoothstep(30.0, 65.0, distance(vWorldPos, scene.camera_pos.xyz));
                if (fadeB > 0.01 && scene.spom_params.x > 0.01 && !br_nb_hit) {
                    int  hb = clamp(spom_h_idx, 0, 5);
                    vec3 Lt = vec3(dot(L, spom_T), dot(L, spom_B), dot(L, spom_face_n));
                    if (hb == 0 || hb == 2) Lt.x *= 0.5;   // 2:1 brick texture aspect (spom_uv)
                    // Box relief: the real depth of the stones (metres -> uv).
                    float hsB = br_relief_m > 0.0 ? br_relief_m * scale : 0.04 * scene.spom_params.x;
                    mesh_pom_shade = mix(1.0, pom_shade(hb, spom_uv_off, Lt, hsB,
                                                        dFdx(spom_uv0), dFdy(spom_uv0),
                                                        br_relief_m > 0.0 ? 1.0 : 0.0), fadeB);
                }
            }
            if (axis != 1) shading_pos = spom_world;
            int  a_idx = clamp(albedo_idx_raw, 0, kTextureCount - 1);
            // The shifted uv is discontinuous across a quad (per-pixel march
            // result): sample with the smooth un-shifted uv's gradients.
            vec2 sgdx = dFdx(spom_uv0), sgdy = dFdy(spom_uv0);
            vec3 sh_tint = vColor;
            if (br_nb_hit) {
                a_idx = br_nb_idx;
                sh_tint = br_nb_tint;
                sgdx *= br_nb_scale / scale; sgdy *= br_nb_scale / scale;
            }
            vec3 tex_albedo = textureGrad(u_albedo[a_idx], spom_uv_off, sgdx, sgdy).rgb;
            albedo = tex_albedo * sh_tint * mesh_pom_shade * br_ao;
            if (use_normal) {
                vec3 n_t = textureGrad(u_normal[a_idx], spom_uv_off, sgdx, sgdy).rgb * 2.0 - 1.0;
                N = normalize(spom_T  * n_t.x +
                              spom_B  * n_t.y +
                              spom_face_n * n_t.z);
                if (br_relief_m > 0.0 && !br_nb_hit) {
                    // Box relief: the stones are really that deep, so their
                    // flanks have to turn toward / away from the light. Slope
                    // of the height field itself (metres per metre) on top of
                    // the fine normal map.
                    int   hn  = clamp(spom_h_idx, 0, 5);
                    vec2  asp = (hn == 0 || hn == 2) ? vec2(0.5, 1.0) : vec2(1.0);
                    float hl  = textureQueryLod(u_height[hn], spom_uv0).x + 1.0;
                    vec2  e   = vec2(exp2(hl)) / vec2(textureSize(u_height[hn], 0));
                    float hx  = textureLod(u_height[hn], spom_uv_off + vec2(e.x, 0.0), hl).r -
                                textureLod(u_height[hn], spom_uv_off - vec2(e.x, 0.0), hl).r;
                    float hy  = textureLod(u_height[hn], spom_uv_off + vec2(0.0, e.y), hl).r -
                                textureLod(u_height[hn], spom_uv_off - vec2(0.0, e.y), hl).r;
                    vec2  slope = vec2(hx, hy) / (2.0 * e) * (scale * asp) * br_relief_m;
                    slope = clamp(slope, vec2(-2.5), vec2(2.5));
                    N = normalize(N - (spom_T * slope.x + spom_B * slope.y) * 0.7);
                    // Worn corner: the surface really turns round the edge.
                    if (dot(br_round_n, br_round_n) > 1e-10)
                        N = normalize(N + normalize(br_round_n) - spom_face_n);
                }
            }
        } else if (use_albedo && is_mesh_draw && pc.grass_params.z < 0.5) {
            // Static mesh: real UVs + derivative (cotangent) tangent frame
            // (Schueler 2013). NormalGL green = image-up = -dP/dv.
            int  ai  = clamp(albedo_idx_raw, 0, kTextureCount - 1);
            vec2 muv = vUv * scale;
            // 2:1 masonry textures: same stone proportions as on the wall blocks.
            { int hq = height_idx_for_albedo(albedo_idx_raw); if (hq == 0 || hq == 2) muv.x *= 0.5; }
            // Cotangent frame (Schueler 2013) - needed for the normal map AND
            // the parallax march; derivatives taken before any uv shift.
            vec3 dp1 = dFdx(vWorldPos), dp2 = dFdy(vWorldPos);
            vec2 du1 = dFdx(muv),       du2 = dFdy(muv);
            vec3 dp2p = cross(dp2, N), dp1p = cross(N, dp1);
            vec3 Tm = dp2p * du1.x + dp1p * du2.x;
            vec3 Bm = dp2p * du1.y + dp1p * du2.y;
            // The frame must point the way u / v actually increase across
            // the surface. With this renderer's screen-space handedness the
            // construction above comes out NEGATED (measured: T and B both
            // opposite to dP/du, dP/dv on every mesh) - which lit the normal
            // map as hollows and ran the parallax backwards, so tower stones
            // swam against the camera and never sat still on the ground.
            // Test it instead of assuming: moving along +T must raise u.
            if (dot(Tm, dp1) * du1.x + dot(Tm, dp2) * du2.x < 0.0) { Tm = -Tm; Bm = -Bm; }
            float im = inversesqrt(max(max(dot(Tm, Tm), dot(Bm, Bm)), 1e-20));
            // Parallax occlusion on UV-mapped meshes (castle towers / keep).
            // How far below this point is the ground (paving or terrain)?
            // The stone relief is flattened out toward it: relief only cuts
            // INTO the wall, so at full depth the stones appear to stand a
            // hand's width inside the tower's footprint, and the tower looked
            // as if it hovered over a gap. Pinned at the ground line it sits.
            float m_ground = 1e9, m_gfade = 0.0;
            {
                float md = distance(vWorldPos, scene.camera_pos.xyz);
                if (height_idx_for_albedo(albedo_idx_raw) >= 0 && md < 40.0 && abs(N.y) < 0.6) {
                    m_gfade = 1.0 - smoothstep(28.0, 40.0, md);
                    vec3 gp = vWorldPos + N * 0.12;
                    m_ground = gp.y - textureLod(u_heightmap_full, gp.xz / 2048.0 + 0.5, 0.0).r;
                    rayQueryEXT fq;
                    rayQueryInitializeEXT(fq, topLevelAS, gl_RayFlagsOpaqueEXT, 0x80u,
                                          gp, 0.0, vec3(0.0, -1.0, 0.0), 1.1);
                    while (rayQueryProceedEXT(fq)) {}
                    if (rayQueryGetIntersectionTypeEXT(fq, true) ==
                        gl_RayQueryCommittedIntersectionTriangleEXT)
                        m_ground = min(m_ground, rayQueryGetIntersectionTEXT(fq, true));
                    m_ground = max(m_ground, 0.0);
                }
            }
            float m_dm = 0.0;            // depth of the relief at the hit (metres)
            float m_hs = 0.0;            // relief depth in uv units
            float m_vz = 1.0;
            vec3  m_Ng = N;              // surface normal before the normal map
            {
                int   hix   = height_idx_for_albedo(albedo_idx_raw);
                float fadeP = 1.0 - smoothstep(30.0, 65.0, distance(vWorldPos, scene.camera_pos.xyz));
                if (hix >= 0 && fadeP > 0.01 && scene.spom_params.x > 0.01) {
                    vec3 Tn = normalize(Tm), Bn = normalize(Bm);
                    vec3 Vw = normalize(scene.camera_pos.xyz - vWorldPos);
                    vec3 Vt = vec3(dot(Vw, Tn), dot(Vw, Bn), dot(Vw, N));
                    m_vz = Vt.z;
                    if (Vt.z > 0.04) {
                        // uv per metre from the derivatives -> relief in metres
                        float uvpm = sqrt(max(dot(du1, du1), dot(du2, du2)) /
                                          max(max(dot(dp1, dp1), dot(dp2, dp2)), 1e-10));
                        float hs = clamp(0.05 / max(uvpm, 0.01), 0.06, 0.3) * uvpm *
                                   scene.spom_params.x * fadeP;
                        // On a round tower a ray that meets the surface at a
                        // slant can only get so far below it before it is
                        // out of the stone again (R * (1 - sqrt(1 - c^2))).
                        // Marching deeper than that dragged the texture into
                        // long streaks where the tower curves away.
                        hs *= mix(1.0, smoothstep(0.0, 0.45, m_ground), m_gfade);
                        // Seen at a flat slant (where a tower curves away) the
                        // shift gets long and drags the stones into streaks:
                        // ease the relief out there.
                        hs *= smoothstep(0.04, 0.38, Vt.z);
                        if (mesh_curv > 0.02) {
                            float can = (1.0 - sqrt(max(1.0 - Vt.z * Vt.z, 0.0))) / mesh_curv;
                            hs *= clamp(can * uvpm / max(hs, 1e-6), 0.12, 1.0);
                        }
                        vec2 uvp = pom_march(hix, muv, Vt, hs, du1, du2);
                        float dm_hit = 0.0;
                        {   // relief silhouette on round towers
                            float dm = (1.0 - textureGrad(u_height[hix], uvp, du1, du2).r) * hs / max(uvpm, 1e-4);
                            if (pom_silhouette_clip(dm, Vt.z, mesh_curv)) discard;
                            dm_hit = dm;
                        }
                        vec3 Lt = vec3(dot(L, Tn), dot(L, Bn), dot(L, N));
                        mesh_pom_shade = mix(1.0, pom_shade(hix, uvp, Lt, hs, du1, du2, 1.0), fadeP);
                        muv = uvp;
                        m_dm = dm_hit; m_hs = hs;
                    } else if (mesh_curv > 0.02) {
                        // Grazing silhouette pixels (no march): clip by the
                        // height at the unshifted uv - the stone outline.
                        float uvpm = sqrt(max(dot(du1, du1), dot(du2, du2)) /
                                          max(max(dot(dp1, dp1), dot(dp2, dp2)), 1e-10));
                        float hs = clamp(0.05 / max(uvpm, 0.01), 0.06, 0.3) * uvpm *
                                   scene.spom_params.x * fadeP;
                        float dm = (1.0 - textureGrad(u_height[hix], muv, du1, du2).r) * hs / max(uvpm, 1e-4);
                        if (pom_silhouette_clip(dm, Vt.z, mesh_curv)) discard;
                    }
                }
            }
            albedo = textureGrad(u_albedo[ai], muv, du1, du2).rgb * vColor * mesh_pom_shade;
            if (use_normal) {
                int  ni = clamp(normal_idx_raw, 0, kTextureCount - 1);
                vec3 nt = textureGrad(u_normal[ni], muv, du1, du2).xyz * 2.0 - 1.0;
                N = normalize(Tm * (im * nt.x) - Bm * (im * nt.y) + N * nt.z);
                if (m_hs > 0.0) {
                    // The stones are really that deep: their flanks turn
                    // toward / away from the light (same as the straight
                    // walls) - slope of the height field on top of the fine
                    // normal map.
                    int   hn = height_idx_for_albedo(albedo_idx_raw);
                    vec2  e  = 1.5 * vec2(length(du1), length(du2)) + 1.0 / vec2(textureSize(u_height[hn], 0));
                    float hx = textureGrad(u_height[hn], muv + vec2(e.x, 0.0), du1, du2).r -
                               textureGrad(u_height[hn], muv - vec2(e.x, 0.0), du1, du2).r;
                    float hy = textureGrad(u_height[hn], muv + vec2(0.0, e.y), du1, du2).r -
                               textureGrad(u_height[hn], muv - vec2(0.0, e.y), du1, du2).r;
                    vec2  slope = clamp(vec2(hx, hy) / (2.0 * e) * m_hs, vec2(-2.5), vec2(2.5));
                    N = normalize(N - (normalize(Tm) * slope.x + normalize(Bm) * slope.y) * 0.7);
                }
            }
            // Soft shadow toward the ground the tower stands on.
            { float g01 = clamp(m_ground / 0.7, 0.0, 1.0); g01 = 0.5 * g01 + 0.5 * g01 * g01 * (3.0 - 2.0 * g01);
              albedo *= mix(1.0, mix(0.48, 1.0, g01), m_gfade); }
        } else if (use_albedo && int(vTexParams.y - 0.5) == -8) {
            // Skinned character: its own UVs into its texture atlas.
            albedo = texture(u_albedo[clamp(albedo_idx_raw, 0, kTextureCount - 1)], vUv).rgb * vColor;
        } else if (use_albedo) {
            int  albedo_idx = clamp(albedo_idx_raw, 0, kTextureCount - 1);
            vec3 tex_albedo = triplanar_sample(u_albedo[albedo_idx],
                                               sample_pos, proj_n);
            albedo = tex_albedo * vColor * mesh_pom_shade;
            if (use_normal) {
                int  normal_idx = clamp(normal_idx_raw, 0, kTextureCount - 1);
                N = triplanar_normal(u_normal[normal_idx], sample_pos, N);
            }
        } else if (use_normal) {
            int  normal_idx = clamp(normal_idx_raw, 0, kTextureCount - 1);
            N = triplanar_normal(u_normal[normal_idx], sample_pos, N);
        }
