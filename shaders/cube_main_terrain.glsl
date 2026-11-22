// cube_main_terrain.glsl - part of cube.frag (included there, in place; not a standalone shader).
// main(): mesh-terrain surface (splat layers, road, terrain parallax + self-shadow)

        // ---- Terrain layer blend (height + slope) ----
        // Layer palette tuned for "natural earth" — desaturated, not too
        // toy-coloured. Heights match the heightmap params:
        //   plateau ~22, mountain crests ~120-140 (height_scale=140 in
        //   the generator). Sea level isn't real here, but we still
        //   give the bottom band a sandy hue.
        const vec3 sand   = vec3(0.78, 0.71, 0.50);
        const vec3 grass  = vec3(0.31, 0.45, 0.18);
        const vec3 dirt   = vec3(0.42, 0.30, 0.18);
        const vec3 rock   = vec3(0.40, 0.38, 0.36);
        const vec3 snow   = vec3(0.95, 0.95, 0.97);

        // ---- Raymarch-matched relief (per-pixel normal) ----
        // 1. Base normal from the full 1 m heightmap per PIXEL at every
        //    distance (the CLOD vertex normal at range comes from a
        //    coarse LOD grid and smooths away ridges / rock exposure).
        // 2. Plus the FBM octaves the heightmap can't hold (raymarcher
        //    octaves 9..19), faded per octave by pixel footprint.
        // 3. Plus the raymarcher's close-up micro-bump.
        // Done before slope / material classification so the detail also
        // drives rock-vs-snow / grass exposure like the raymarched terrain.
        {
            float fp = cam_dist * 2.0 / max(scene.viewport.y, 1.0);   // ~m per pixel
            float he = max(1.0, fp);
            #define HM(xz) textureLod(u_heightmap_full, (((xz) - pc.emissive.xy) * pc.emissive.z + 0.5) * pc.emissive.w, 0.0).r
            float hxp = HM(vWorldPos.xz + vec2(he, 0.0)), hxm = HM(vWorldPos.xz - vec2(he, 0.0));
            float hzp = HM(vWorldPos.xz + vec2(0.0, he)), hzm = HM(vWorldPos.xz - vec2(0.0, he));
            vec2 g = vec2(hxp - hxm, hzp - hzm) / (2.0 * he);
            // Far pixels: the footprint-spaced point taps undersample the
            // 1 m heightmap (normal sparkle / crawl on ridges). Average in
            // a 45-degree rotated estimate - 8-tap rotated-grid filter.
            if (fp > 1.5) {
                const float r = 0.70710678;
                float a0 = HM(vWorldPos.xz + vec2( he,  he) * r), a1 = HM(vWorldPos.xz - vec2( he,  he) * r);
                float b0 = HM(vWorldPos.xz + vec2( he, -he) * r), b1 = HM(vWorldPos.xz - vec2( he, -he) * r);
                vec2 gd = vec2(a0 - a1, b0 - b1) / (2.0 * he);        // along (1,1)/sqrt2 and (1,-1)/sqrt2
                vec2 g2 = vec2(gd.x + gd.y, gd.x - gd.y) * r;         // back to x / z
                g = mix(g, 0.5 * (g + g2), smoothstep(1.5, 3.0, fp));
            }
            #undef HM
            if (fp < 0.33) {
                float eps = max(0.02, fp);
                float h0 = terrain_fbm_detail(vWorldPos.xz, fp);
                float hx = terrain_fbm_detail(vWorldPos.xz + vec2(eps, 0.0), fp);
                float hz = terrain_fbm_detail(vWorldPos.xz + vec2(0.0, eps), fp);
                g += vec2(hx - h0, hz - h0) / eps;
            }
            N = normalize(vec3(-g.x, 1.0, -g.y));
            float mw = 1.0 - smoothstep(15.0, 60.0, cam_dist);
            if (mw > 0.05) {
                vec3 d1 = t_noised(vWorldPos.xz * 4.0), d2 = t_noised(vWorldPos.xz * 13.0);
                vec2 gm = (d1.yz * 0.30 + d2.yz * 0.18) * mw;
                N = normalize(N + vec3(-gm.x, 0.0, -gm.y) * 0.45);
            }
        }
        float h = vWorldPos.y;
        // Slope: 0 = flat (normal up), 1 = vertical wall.
        float slope = 1.0 - clamp(N.y, 0.0, 1.0);

        // ---- Baked curvature + sky-vis (AAA Phase 4) ----
        vec2 t_aux = vec2(0.5, 1.0);
        {
            vec2 aux_uv = (vWorldPos.xz / 2048.0) + vec2(0.5);
            t_aux = textureLod(u_terrain_aux,
                               clamp(aux_uv, vec2(0.0), vec2(1.0)), 0.0).rg;
        }
        // -1 = gully floor .. +1 = exposed ridge.
        float convexity = clamp((t_aux.x - 0.5) * 2.0, -1.0, 1.0);
        terrain_sky_vis = t_aux.y;

        // ---- Automatic beach band from the live water level ----
        // scene.water_params aliases the C++ water_params
        // (.x = water enabled, .y = level). Sand owns everything from
        // below the waterline up into a band above it; grass/dirt NEVER
        // appear under or right at the water, with zero manual tuning.
        //
        // The band edge is NOT a flat height contour (that read as a
        // straight "line" up slopes): a low-frequency world-XZ meander
        // wobbles the threshold so the sand/grass boundary follows an
        // irregular natural coastline. It's also slope-aware — a broad
        // beach on gentle ground, a thin sand strip on steep coast — so
        // it visually hugs the shore instead of ringing the terrain at
        // one elevation.
        float wtr_lvl = scene.water_params.y;
        float wtr_on  = step(0.5, scene.water_params.x);
        // (1) HARD guarantee: everything at or below the water line is
        // sand, ALWAYS — independent of any noise. This is what stops
        // grass ever appearing under water.
        float under = 1.0 - smoothstep(wtr_lvl - 0.20, wtr_lvl + 0.05, h);
        // (2) The sand strip ABOVE water. A flat height threshold on a
        // slope is a straight iso-line; we displace the threshold by a
        // strong MID-frequency multi-octave world-XZ wander whose
        // amplitude is much larger than the local height span across
        // the strip, so the sand→grass edge swings widely in 2D and
        // reads as an irregular coast, not a contour. Soft (±1 m) so
        // it's a sand→grass gradient, not a hard line.
        // Big sweeping bays + mid coves + fine crenellation. Amplitude
        // is deliberately huge vs the slope so the sand→grass edge
        // swings tens of metres in world-XZ — it cannot read as a
        // straight contour even on a uniform ramp.
        float wrp = (tnoise(vWorldPos.xz * 0.008) - 0.5) * 14.0   // bays ~125 m
                  + (tnoise(vWorldPos.xz * 0.028) - 0.5) * 6.0    // coves ~36 m
                  + (tnoise(vWorldPos.xz * 0.090) - 0.5) * 2.0;   // crenel ~11 m
        float topH  = wtr_lvl + 2.0 + wrp;
        float above = 1.0 - smoothstep(topH - 1.5, topH + 1.5, h);
        float beach = wtr_on * max(under, above);

        // ---- Layer-transition break-up ----
        // Without a per-pixel offset, layer boundaries form perfectly
        // horizontal contour lines that read as obvious "stripes" on
        // gentle slopes. We jitter each transition's effective height
        // by a low-frequency hash on world XZ — same as how natural
        // soils don't change at exactly the same elevation everywhere.
        // The jitter is in metres and matched to ~half the smoothstep
        // width so it breaks up the line without erasing the gradient.
        // Multi-octave, large-amplitude jitter so the layer boundaries
        // dissolve into natural patches instead of horizontal contour
        // "line banding". (The old ±3 m single-ish octave was too weak
        // once the detail/noise layers were stripped.)
        // Only low-frequency octaves: the boundary should *meander*
        // smoothly (a wandering coastline), not flicker per-metre. The
        // old 0.30-freq (~3 m) octave made the threshold jump up/down
        // every few metres, which with real per-texel materials reads
        // as a spiky salt-and-pepper mix of two grounds instead of a
        // gradient. Drop it; keep a broad 40 m + gentle 11 m wander,
        // smaller amplitude so it never overshoots the blend ramp.
        float n0 = tnoise(vWorldPos.xz * 0.012) - 0.5;
        float n1 = tnoise(vWorldPos.xz * 0.045 + vec2(13.7, 41.3)) - 0.5;
        float jitter = n0 + 0.35 * n1;                // ~±0.7, slow
        float jh = h + jitter * 3.0;                  // ±~2 m meander

        // Widen every band transition into a LONG gradient. On steep
        // mountain flanks a narrow height band projects to a thin spiky
        // line on screen; a very wide ramp (≈5.6x the configured span,
        // around the same centre) turns grass→rock→snow into a smooth
        // fade. The +8 m floor guarantees a usable ramp even if a band's
        // start/end are configured close together.
        #define WIDE_BAND(a, b, x) smoothstep(                              \
            0.5 * ((a) + (b)) - (1.4 * ((b) - (a)) + 8.0),                  \
            0.5 * ((a) + (b)) + (1.4 * ((b) - (a)) + 8.0), (x))
        float t_sand = WIDE_BAND(scene.terrain_h_low.x,  scene.terrain_h_low.y,  jh);
        float t_dirt = WIDE_BAND(scene.terrain_h_low.z,  scene.terrain_h_low.w,  jh);
        float t_rock = WIDE_BAND(scene.terrain_h_high.x, scene.terrain_h_high.y, jh);
        // (the snow line by the way the slope faces, as in terrain_rm_shading_helpers.glsl getMaterial)
        float t_snow = WIDE_BAND(scene.terrain_h_high.z, scene.terrain_h_high.w, jh - clamp(N.z * 30.0, -16.0, 16.0));
        // Snow only settles on flatter faces (raymarcher getMaterial rule);
        // steep micro-faces stay bare rock -> mottled snow / rock slopes.
        t_snow *= smoothstep(0.30, 0.72, N.y);
        #undef WIDE_BAND

        vec3 base = mix(sand, grass, t_sand);
        base = mix(base, dirt, t_dirt);
        base = mix(base, rock, t_rock);
        base = mix(base, snow, t_snow);

        // ---- Procedural surface detail noise ----
        // Two-octave value noise on world XZ. Fine octave (0.45 m
        // freq) modulates albedo within ±8% so flat patches don't
        // read as a single uniform colour; coarser octave (3 m freq)
        // adds large-scale tonal variation that survives at distance.
        // Cheap (4 ign() calls) and unlike the triplanar detail it
        // doesn't depend on N so it's stable across LOD-mismatched
        // triangle edges.
        // Now that the Ground054 texture tiles at a sane size it
        // carries the surface detail; this procedural modulation only
        // needs to subtly break up flat patches. The old coarse 3 m
        // octave at ±8% read as big blurry blotches OVER the texture —
        // drop it almost entirely (mostly the fine 0.45 m octave) and
        // halve the amplitude.
        float dn_fine = tnoise(vWorldPos.xz * 2.2 + vec2(11.0, 23.0));
        float dn_far  = tnoise(vWorldPos.xz * 0.33 + vec2(91.0, 47.0));
        float dn = mix(dn_far, dn_fine, 0.9);          // 0..1, mostly fine
        float noise_amp = 0.07;                        // ±3.5% albedo swing
        base *= 1.0 + (dn - 0.5) * noise_amp;

        // Steep faces become rocky regardless of altitude. Slope jitter
        // breaks up the cliff/grass border the same way the height
        // jitter breaks up horizontal layer transitions.
        // Fade slope-driven rock with camera distance: at LOD 2/3 the
        // Gouraud-interpolated normals between widely-spaced verts
        // produce per-triangle slope swings, and adjacent large
        // triangles end up classified as rock vs grass at random,
        // reading as patchy "different faces". Past ~120 m we let the
        // height-band layering carry the look — visually identical at
        // distance, no per-triangle classification noise.
        // Slope→rock is the main grass↔rock spike: `slope` comes from
        // the per-pixel mesh normal, which flickers across the dense
        // sculpt, so a tight slope window sprinkles rock through grass.
        // Use a LOW-frequency meander (not the old ~11 m hash) and a
        // very wide smoothstep so the cliff/grass border is a long soft
        // gradient, and cap the max so slope never fully erases grass.
        float slope_jitter = (tnoise(vWorldPos.xz * 0.02 + vec2(7.0, 19.0)) - 0.5) * 0.12;
        float steep_raw = smoothstep(0.40 + slope_jitter, 1.02 + slope_jitter, slope);
        // Phase 4: convex ridges expose extra rock (erosion strips the
        // soil off ridgelines); the baked curvature is mesh-independent
        // so this doesn't reintroduce per-triangle classification noise.
        steep_raw = clamp(steep_raw + max(convexity, 0.0) * 0.30 * slope, 0.0, 1.0);
        float steep_dist = cam_dist;    // P5: use hoisted distance
        // No distance fade any more: N is now a per-pixel heightmap+FBM
        // normal (not the coarse LOD vertex normal that caused per-
        // triangle rock/grass flicker), so far cliffs expose rock like
        // the raymarched terrain.
        float steep_w = 1.0 + 0.0 * steep_dist;
        float steep = steep_raw * steep_w * 0.85;
        base = mix(base, rock, steep);

        // ---- Cavity AO from local height curvature ----
        // dFdx(N)·dFdx(vWorldPos) gives concavity per fragment. Adds
        // "natural shadowing in cracks" for free near the camera, but
        // on distant LOD-2/3 chunks the screen-space derivatives of
        // Gouraud-interpolated normals between widely-spaced verts
        // become erratic (large per-pixel jumps), and the cavity term
        // oscillates 0.45↔1.0 across adjacent triangles — visible as
        // patchy dark faces on far ridges. Fade the effect out past
        // ~120 m so distant terrain reads as smooth.
        // DISABLED: this curvature term uses dFdx(N) on the 1 m faceted
        // mesh, so every triangle gets a different darkening — that is
        // the blocky "pixelated dark overlay baked into the heightmap"
        // (a faceted-normal artifact, not real AO). Removed for terrain.
        // float curvature = -dot(dFdx(N), dFdx(vWorldPos)) -
        //                   dot(dFdy(N), dFdy(vWorldPos));
        // float cav_w     = 1.0 - smoothstep(80.0, 200.0, cam_dist);
        // float cavity    = clamp(0.5 - curvature * 0.4, 0.45, 1.0);
        // base *= mix(1.0, cavity, cav_w);

        // ---- Per-pixel FBM-eroded micro-detail ----
        // The rasterised mesh has 1 m vertex spacing, so anything
        // finer than that gets flattened between vertices — sculpted
        // / brushed terrain reads as "faceted" compared with the
        // raymarched terrain (which evaluates noise per pixel). This
        // pass evaluates the same FBM-with-derivative-erosion in the
        // fragment shader and pushes its analytical gradient into N,
        // adding sub-cell ridges/valleys at any zoom level. Geometry
        // stays at mesh resolution; only the SHADING normal moves.
        // Faded with camera distance so distant chunks (where 1-pixel
        // surfaces span many noise cycles) don't shimmer.
        // DISABLED: this FBM micro-detail pushed a per-pixel noise
        // gradient into the shading normal AND ridge-darkened the
        // albedo. On the faceted 1 m mesh it read as a noisy
        // "low-resolution heightmap baked in" overlay. Removed so the
        // terrain is just the clean height-band colour + fine grain.
        // (Re-introduce later as a proper, toggleable detail-normal.)
        // {
        //     float det_w = 1.0 - smoothstep(40.0, 120.0, cam_dist);
        //     if (det_w > 0.001) {
        //         vec3 dn = terrain_detail_fbm(vWorldPos.xz, 0.30, 6);
        //         N = normalize(N + vec3(-dn.y, 0.0, -dn.z) * 0.25 * det_w);
        //         float ridge = clamp(dn.x * 0.6 + 0.5, 0.0, 1.0);
        //         base *= mix(1.0, mix(0.85, 1.10, ridge), det_w);
        //     }
        // }

        // Optional triplanar detail using the engine's Ground054 albedo
        // (texture index 0). When textures are off (use_albedo=false)
        // we just show the layer-blended colour.
        // The Ground054 triplanar "detail" overlay is disabled for
        // terrain: that texture is too low-res to tile per-metre over a
        // 2 km field — at any tiling it read as a big pixelated stamp
        // on top of the ground. The look is now the height-band layer
        // colours + the subtle procedural fine-grain noise above +
        // anisotropic filtering, which stays clean at all distances.
        // ---- Full material splatting ----
        // Real per-texel rock/grass/dirt/sand/snow (procedurally baked,
        // seamless tiling) blended by the SAME height/slope weights that
        // drive the art-directed band colour `base`. `base` is reused as
        // the distance-fade target so far chunks settle to the clean
        // band colour (no tiling shimmer, no LOD-facet normal noise).
        // Runtime knobs (free slots of spom_params — see descriptors.cpp):
        //   .y = material strength (0 = old flat band look, 1 = full),
        //   .z = metres per material repeat,
        //   .w = detail-normal strength.
        float g_str  = clamp(scene.spom_params.y, 0.0, 1.0);
        float g_tile = max(scene.spom_params.z, 0.25);
        vec2  uvm = vWorldPos.xz / g_tile;          // material repeat
        vec2  uvM = vWorldPos.xz / (g_tile * 8.0);  // macro break-up

        // ---- Parallax-occlusion (per-pixel displacement) ----
        // Pixel-displaces the ROCK heightfield (baked into the rock
        // normal map's alpha, with extra eroded gullies) so cliffs /
        // snow read as carved, weathered rock without any geometry.
        // Cost is bounded three ways: (1) only where it matters —
        // rocky/snow pixels (grass/sand have ~0 weight → skipped, the
        // common case); (2) step count ramps with distance to ZERO by
        // ~45 m (a per-pixel LOD); (3) 1 height sample/step + 2 refine.
        // Strength comes from the terrain draw's pc.tex_params.x (the
        // terrain path ignores that slot otherwise); 0 = off.
        vec3 Tt = normalize(vec3(1.0, 0.0, 0.0) - N * N.x);
        vec3 Bt = cross(Tt, N);   // +Z, matches uvm.y = world z
        float pom_str = clamp(vTexParams.x, 0.0, 1.0);
        // Camera-to-fragment view direction is identical for both POM
        // branches below (grass POM + rock POM) — was being normalised
        // twice per pixel. Hoist once; both inner gates still early-out
        // before any other work when their weight is below threshold.
        vec3 V_pom = normalize(scene.camera_pos.xyz - vWorldPos);
        // ---- Terrain relief (parallax occlusion + self-shadow) ----
        // One march over the height field of whichever layer dominates
        // here (grass band, or rock/dirt/snow), refined by bisection,
        // then a short march toward the sun through the same field for
        // per-pixel self-shadow (stones shadow the ground next to them).
        // Depth is in UV units of the material repeat (uvm); with the
        // default 1.7 m repeat 0.07 -> ~12 cm, 0.12 -> ~20 cm.
        //
        // Grass-present mask, needed here so the grass-band march uses
        // the SAME height map as the texture the splat shows below.
        float _grass_present = 0.0;
        {
            vec2 _gm_uv = (vWorldPos.xz / 2048.0) + vec2(0.5);
            if (_gm_uv.x >= 0.0 && _gm_uv.x <= 1.0 &&
                _gm_uv.y >= 0.0 && _gm_uv.y <= 1.0) {
                _grass_present = textureLod(u_grass_mask, _gm_uv, 0.0).r;
            }
            // Soft edge so the green doesn't stop on a hard line.
            _grass_present = smoothstep(0.05, 0.55, _grass_present);
        }
        float grass_w_band = clamp(t_sand * (1.0 - t_dirt) *
                                   (1.0 - max(t_rock, steep)),
                                   0.0, 1.0);
        float rockiness = max(max(t_rock, steep),
                              max(t_dirt * 0.6, t_snow * 0.7));
        float pdw = clamp(t_dirt * (1.0 - max(t_rock, steep)), 0.0, 1.0);
        // ---- Castle road (cobbles + dirt verge) ----
        // Exact XZ distance to the road polyline (SceneUBO road_pts), so
        // the paving edge is crisp at any distance. A little noise on
        // the distance roughens the edge into irregular kerb stones.
        float road_w = 0.0, verge_w = 0.0;
        // Road-space UV (x = across, y = along the road, metres / g_tile) and
        // the road frame, so the cobble rows follow the curve.
        vec2  road_uv  = uvm;
        vec3  road_T   = Tt, road_B = Bt;
        {
            int rn = int(scene.road_info.x + 0.5);
            vec2 q = vWorldPos.xz;
            if (rn >= 2 && all(greaterThan(q, scene.road_bbox.xy)) &&
                all(lessThan(q, scene.road_bbox.zw))) {
                float best = 1e9;
                for (int i = 0; i < 63; ++i) {
                    if (i + 1 >= rn) break;
                    vec2 a = scene.road_pts[i].xz, b = scene.road_pts[i + 1].xz;
                    vec2 ab = b - a;
                    float t = clamp(dot(q - a, ab) / max(dot(ab, ab), 1e-6), 0.0, 1.0);
                    vec2 dv = q - (a + ab * t);
                    best = min(best, dot(dv, dv));
                }
                float rd = sqrt(best);
                // Bisector-partitioned segments: q belongs to segment i
                // when it lies between the corner bisectors at i and i+1;
                // along = s_i + fraction * len is then continuous through
                // every bend (no seams in the cobble rows).
                float best_lat = 1e9, r_along = 0.0, r_side = 0.0;
                vec2  r_dir = vec2(0.0, 1.0);
                for (int i = 0; i < 63; ++i) {
                    if (i + 1 >= rn) break;
                    vec2 a = scene.road_pts[i].xz, b = scene.road_pts[i + 1].xz;
                    vec2 ta = normalize(b - scene.road_pts[max(i - 1, 0)].xz);
                    vec2 tb = normalize(scene.road_pts[min(i + 2, rn - 1)].xz - a);
                    float da = dot(q - a, ta), db = dot(q - b, tb);
                    if (da < 0.0 && i > 0) continue;
                    if (db > 0.0 && i + 2 < rn) continue;
                    vec2 sd = normalize(b - a);
                    float lat = sd.x * (q - a).y - sd.y * (q - a).x;   // signed
                    if (abs(lat) < abs(best_lat)) {
                        best_lat = lat;
                        float f = da / max(da - db, 1e-4);
                        r_along = mix(scene.road_pts[i].w, scene.road_pts[i + 1].w, f);
                        r_dir = normalize(mix(ta, tb, clamp(f, 0.0, 1.0)));
                    }
                }
                if (best_lat < 1e8) {
                    road_uv = vec2(best_lat, r_along) / g_tile;
                    // Frame matching road_uv: U grows to the road-left
                    // (lat sign), V along the travel direction.
                    vec3 dirw  = vec3(r_dir.x, 0.0, r_dir.y);
                    vec3 sidew = vec3(-r_dir.y, 0.0, r_dir.x);
                    road_T = normalize(sidew - N * dot(sidew, N));
                    road_B = normalize(dirw  - N * dot(dirw,  N));
                }
                float hw = scene.road_info.y, vw = scene.road_info.z;
                float jit = (tnoise(q * 1.7) - 0.5) * 0.45 +
                            (tnoise(q * 0.31 + 17.0) - 0.5) * 0.6;
                road_w  = 1.0 - smoothstep(hw - 0.12, hw + 0.12, rd + jit);
                verge_w = (1.0 - smoothstep(hw + 0.4 * vw, hw + vw + 0.5,
                                            rd + jit * 1.6)) * (1.0 - road_w);
            }
        }
        // Pick the dominant layer's field (one march, not two).
        bool  pom_grass = grass_w_band >= rockiness;
        int   pom_kind  = road_w > 0.5 ? 2 : (pom_grass ? 0 : 1);
        float pom_k     = pom_grass ? _grass_present : pdw;
        terrain_grass_occ = mix(1.0, 0.42, _grass_present * (1.0 - smoothstep(40.0, 90.0, cam_dist)));
        float pom_far   = max(70.0, scene.terrain_disp_params.z);
        float pom_fade  = 1.0 - smoothstep(0.5 * pom_far, pom_far, cam_dist);
        float pom_w     = pom_str * pom_fade *
                          max(pom_grass ? grass_w_band : rockiness, 0.35);
        if (pom_kind == 2) pom_w = pom_str * pom_fade;
        float pom_D     = (pom_kind == 2 ? 0.035 : (pom_grass ? 0.05 : 0.10)) * pom_w;
        // Road parallax runs in road space (road_uv + road frame).
        vec2 pom_uv0 = (pom_kind == 2) ? road_uv : uvm;
        vec3 pT = (pom_kind == 2) ? road_T : Tt;
        vec3 pB = (pom_kind == 2) ? road_B : Bt;
        g_tpom_lod = pom_kind == 2 ? textureQueryLod(u_normal[kMatRoad], road_uv).x
                   : pom_grass     ? textureQueryLod(u_height[5], uvm).x
                                   : textureQueryLod(u_normal[kMatRock], uvm).x;
        if (pom_D > 0.002) {
            vec3 Vt = vec3(dot(V_pom, pT), dot(V_pom, pB), dot(V_pom, N));
            Vt.z = max(Vt.z, 0.22);
            // More layers at grazing angles where the shift is long.
            int   steps = int(mix(24.0, 8.0, Vt.z));
            float layer = 1.0 / float(steps);
            vec2  dtex  = (Vt.xy / Vt.z) * pom_D * layer;
            vec2  uvc   = pom_uv0;
            float curD  = 0.0;
            float hh    = tpom_h(uvc, pom_kind, pom_k);
            for (int i = 0; i < 24; ++i) {
                if (i >= steps || curD >= 1.0 - hh) break;
                uvc -= dtex;
                curD += layer;
                hh = tpom_h(uvc, pom_kind, pom_k);
            }
            // Bisection between the last layer above (lo) and the first
            // below (hi) the surface: no stair-step banding.
            vec2  lo = uvc + dtex;   float dLo = max(curD - layer, 0.0);
            vec2  hi = uvc;          float dHi = curD;
            for (int b = 0; b < 4; ++b) {
                vec2  mid  = (lo + hi) * 0.5;
                float dMid = (dLo + dHi) * 0.5;
                if (dMid < 1.0 - tpom_h(mid, pom_kind, pom_k)) { lo = mid; dLo = dMid; }
                else                                           { hi = mid; dHi = dMid; }
            }
            vec2  uv_hit = (lo + hi) * 0.5;
            float h_hit  = 1.0 - 0.5 * (dLo + dHi);
            vec2  duv = uv_hit - pom_uv0;
            if (pom_kind == 2) {
                road_uv += duv;
            } else {
                uvm += duv;
                uvM += duv * 0.125;
            }
            vec3 Lt = vec3(dot(L, pT), dot(L, pB), dot(L, N));
            float vis = tpom_shadow(uv_hit, h_hit, Lt, pom_D, pom_kind, pom_k);
            terrain_pom_vis = mix(1.0, vis, clamp(pom_w * 1.5, 0.0, 1.0));
        }

        // Grass blade shadows (raymarched grass on): exact blade-shadow
        // march near the camera, fading to an averaged mask darkening
        // (blades become sub-pixel) further out. Rides the same sun-
        // visibility channel as the relief self-shadow (gbuffer1.b ->
        // deferred_lighting), so it works in the deferred path too.
        if (scene.grass_rm_params2.y > 0.5 && _grass_present > 0.02 &&
            (scene.water_params.x < 0.5 || vWorldPos.y > scene.water_params.y + 0.05)) {
            float gstr = scene.grass_rm_params2.z;
            float near_w = 1.0 - smoothstep(13.0, 26.0, cam_dist);
            float far_vis = 1.0 - 0.45 * gstr * _grass_present *
                            smoothstep(0.1, 0.6, L.y) * scene.grass_rm_params.w;
            float gvis = far_vis;
            if (near_w > 0.01) {
                float exact = grass_blade_shadow(vWorldPos, L);
                gvis = mix(far_vis, mix(1.0, exact, gstr * 0.85 + 0.15), near_w);
            }
            terrain_pom_vis *= gvis;
        }

        // Anti-tiling: blend a SECOND sample at a rotated/offset UV with
        // the standard one, weighted by a low-frequency noise field. Cuts
        // the visible grid repeat on terrain (the typical Heitz-Neyret
        // problem) without paying for the full stochastic / hex-tile
        // approach. ~2x texture taps per ground pixel when on; gated on
        // the master toggle so the user can A/B for free.
        //
        // Implementation:
        //   - Sample a low-freq noise (~100 m feature size) at vWorldPos.xz
        //     to get a per-pixel rotation angle. Smooth so neighbouring
        //     pixels rotate by the same amount (no per-pixel hash noise).
        //   - Rotate uvm/uvM about the local repeat origin and add a
        //     translation offset so it doesn't share any lattice nodes
        //     with the strict sample.
        //   - Blend with weight scene.terrain_antitile_params.y * cube-
        //     ramped noise so the rotated copy dominates inside the
        //     "alternate" patches and the strict copy dominates outside.
        bool  at_on = scene.terrain_antitile_params.x > 0.5;
        float at_w  = at_on ? clamp(scene.terrain_antitile_params.y, 0.0, 1.0)
                            : 0.0;
        float at_a    = 0.0;
        float at_blend = 0.0;
        vec2  uvm_r = uvm;
        vec2  uvM_r = uvM;
        if (at_w > 0.001) {
            // Low-freq angle field. tnoise is in [0,1]; map to roughly
            // [-pi/3, +pi/3] so the rotated sample is visibly different
            // from the strict one but not 90 deg (which would re-align
            // axis-aligned bricky textures back to the same look).
            float ang_n = tnoise(vWorldPos.xz * 0.011 + vec2(31.7, 17.3));
            at_a = (ang_n - 0.5) * 2.094;  // ~ +/- 60 deg in radians
            // Patch-presence noise: a separate field that decides where
            // the rotated copy "wins" -- 0..1 with soft edges so the
            // transition doesn't look like a procedural mask.
            float patch_n = tnoise(vWorldPos.xz * 0.025 + vec2(7.1, 91.7));
            at_blend = smoothstep(0.35, 0.75, patch_n) * at_w;
            float cs = cos(at_a), sn = sin(at_a);
            mat2 R = mat2(cs, -sn, sn, cs);
            // Translation breaks lattice alignment between the two
            // samples. World-derived so it's spatially consistent.
            vec2 offm = vec2(13.7, 27.1);
            vec2 offM = vec2(5.3, 41.9);
            uvm_r = R * uvm + offm;
            uvM_r = R * uvM + offM;
        }
        // MAT_A samples the strict copy by default; when anti-tile is on,
        // blend in the rotated copy by `at_blend`. Strict path is exactly
        // the previous code (same two-fetch macro/micro mix) so toggling
        // the slider off restores the previous look bit-for-bit.
        // Per-layer relative UV scale (AAA Phase 1): real photogrammetry
        // sets have different natural feature sizes — one global tile
        // rate made snow grain match boulder size. Relative so the
        // user's metres-per-repeat slider still scales everything.
        const float kScRock  = 1.0;
        const float kScGrass = 1.5;
        const float kScDirt  = 1.25;
        const float kScSand  = 1.85;
        const float kScSnow  = 2.3;
        #define MAT_A(i, s) ( \
            mix( \
                texture(u_albedo[i], uvm   * (s)).rgb * 0.75 + \
                texture(u_albedo[i], uvM   * (s)).rgb * 0.25, \
                texture(u_albedo[i], uvm_r * (s)).rgb * 0.75 + \
                texture(u_albedo[i], uvM_r * (s)).rgb * 0.25, \
                at_blend))
        // Per-layer surface height for the height-aware blend — the
        // displacement packed into each layer's normal alpha.
        #define MAT_H(i, s) (texture(u_normal[i], uvm * (s)).a)
        // Pick the grass-band TEXTURE based on the eligibility mask:
        // slot 8 (stylized PNG, green blade ground) where grass grows,
        // slot 12 (procedural grass, the OLD pre-stylized look) where
        // it doesn't. The grass-band WEIGHT (t_sand) is unchanged so
        // the height/slope blend is identical -- only the texture
        // sampled in the grass slot changes.
        // Rocky-rugged slot 12: STRICT single-tap at uvm only, no
        // rotation AND no macro mix. The macro tap at uvM=uvm/8 was
        // contributing 25% of the visible texture but the tess height
        // sampler only reads uvm, so visible stones at the macro tile
        // had bumps at a different world position -> "some stones
        // perfectly fine, many in the wrong place". Macro break-up
        // disabled here trades some visible tiling for stone-on-bump
        // alignment (essential for vertex displacement to read).
        vec3 rocky_strict = texture(u_albedo[12], uvm).rgb;
        vec3 grass_albedo = mix(rocky_strict,
                                 MAT_A(kMatGrass, kScGrass),
                                 _grass_present);
        // Height-aware splat (AAA Phase 1): every transition is contested
        // by the two layers' surface heights, so boundaries interlock
        // (grass between stones, dirt in cracks) instead of cross-fading.
        // `hcur` tracks the accumulated surface height alongside `m`.
        float h_grassband = mix(textureLod(u_height[5], uvm, 0.0).r,
                                MAT_H(kMatGrass, kScGrass),
                                _grass_present);
        float h_dirt = MAT_H(kMatDirt, kScDirt);
        float h_rock = MAT_H(kMatRock, kScRock);
        float h_snow = MAT_H(kMatSnow, kScSnow);
        float h_sand = MAT_H(kMatSand, kScSand);
        vec3  m    = MAT_A(kMatSand, kScSand);
        float hcur = h_sand;
        // Weights are saved so the detail-normal splat below blends the
        // normal maps with EXACTLY the same height-aware weights as the
        // albedo (mismatched weights make lighting drift off the texture).
        float w_grass = height_lerp(t_sand, hcur, h_grassband);
        m = mix(m, grass_albedo, w_grass);
        hcur = mix(hcur, h_grassband, w_grass);
        float w_dirt = height_lerp(t_dirt, hcur, h_dirt);
        m = mix(m, MAT_A(kMatDirt, kScDirt), w_dirt);
        hcur = mix(hcur, h_dirt, w_dirt);
        float w_rock = height_lerp(t_rock, hcur, h_rock);
        m = mix(m, MAT_A(kMatRock, kScRock), w_rock);
        hcur = mix(hcur, h_rock, w_rock);
        float w_snow = height_lerp(t_snow, hcur, h_snow);
        m = mix(m, MAT_A(kMatSnow, kScSnow), w_snow);
        hcur = mix(hcur, h_snow, w_snow);
        float w_steep = height_lerp(steep, hcur, h_rock);
        m = mix(m, MAT_A(kMatRock, kScRock), w_steep);
        hcur = mix(hcur, h_rock, w_steep);
        float w_beach = height_lerp(beach, hcur, h_sand);
        m = mix(m, MAT_A(kMatSand, kScSand), w_beach);  // sand at waterline
        // Road: trodden dirt verge, then the cobbles (POM-shifted uvm).
        m = mix(m, MAT_A(kMatDirt, kScDirt) * vec3(0.82, 0.76, 0.68), verge_w);
        m = mix(m, texture(u_albedo[kMatRoad], road_uv).rgb, road_w);
        hcur = mix(hcur, texture(u_normal[kMatRoad], road_uv).a, road_w);
        #undef MAT_A
        #undef MAT_H
        // Carry the final splat height + a rock/snow specular weight to
        // the lighting section (Phase 5).
        terrain_micro_h = hcur;
        terrain_spec_w  = clamp(w_rock + w_steep + w_snow * 1.6, 0.0, 1.0)
                          * 0.22;

        // Tint the neutral material toward the art-directed band hue so
        // the overall palette stays controlled, keeping the per-texel
        // detail as a multiplicative term; then fade to the flat band
        // colour with distance.
        // Force the band/flat colour to sand at the waterline too, so
        // the distance-fade target and the g_str=0 look also have no
        // grass under water.
        base = mix(base, sand, beach);
        float m_lum  = max(dot(m, vec3(0.299, 0.587, 0.114)), 1e-3);
        vec3  tinted = mix(m, m * (base / m_lum), 0.55);
        // Cobbles keep their own grey stone colour (no band tint).
        tinted = mix(tinted, m, road_w);   // no band (grass / sand) tint on the cobbles
        // g_str blends old flat band colour ↔ full per-texel material
        // (g_str = 0 is the exact previous look — an A/B toggle).
        vec3  splat = mix(base, tinted, g_str);
        // Fade to the flat band colour only at long range so mid-
        // distance terrain still shows real material (the prior
        // 110→300 m fade made everything past the plateau look flat).
        // Push the far-flatten fade WAY out so distant terrain keeps
        // its splat textures (rocky-rugged, etc.) visible. Was 280-760
        // -> too aggressive, distant ground read as untextured colour
        // bands. 450-1500 keeps splat detail across the whole visible
        // valley before the per-texel filter loses energy.
        float far_fade = smoothstep(450.0, 1500.0, cam_dist);
        albedo = mix(splat, base, far_fade);

        // ---- Macro variation (AAA Phase 1) ----
        // World-scale moisture/soil tint over every layer so the ground
        // never reads as one uniform hue at middle distance. Two slow
        // octaves (~250 m and ~60 m); damp patches darken + cool, dry
        // patches lift + warm. Applied to the final albedo so it also
        // survives the far band-colour fade.
        float mv0 = tnoise(vWorldPos.xz * 0.004 + vec2(53.1, 7.7));
        float mv1 = tnoise(vWorldPos.xz * 0.017 + vec2(3.3, 77.7));
        float mv  = clamp(mv0 * 0.7 + mv1 * 0.3, 0.0, 1.0);
        albedo *= mix(vec3(1.0), mix(vec3(0.86, 0.89, 0.84), vec3(1.10, 1.06, 0.99), mv), 1.0 - road_w);   // not on the road

        // ---- Gully sediment + cavity darkening (AAA Phase 4) ----
        // Concave drainage floors collect sediment (warm-brown tint) and
        // sit lower in the light (cheap baked cavity term). Curvature
        // comes from the heightmap bake, so unlike the old dFdx(N)
        // attempt this is stable across triangle/LOD boundaries.
        float gully = clamp(-convexity, 0.0, 1.0) * (1.0 - road_w);   // the carved road reads as a gully: keep it off the cobbles
        albedo = mix(albedo, albedo * vec3(0.96, 0.88, 0.74), gully * 0.40);
        albedo *= mix(1.0, 0.80, gully * 0.65);

        // ---- Detail normal mapping ----
        // Splat the matching baked normal maps, rotate into world space
        // via a tangent frame built on the geometric terrain normal.
        // Strength fades out by ~140 m: distant LOD chunks have widely
        // spaced verts whose interpolated N makes per-texel normals
        // shimmer (the old faceted-overlay artifact) — let the band
        // colour carry the distance.
        // Anti-tile normal: same rotated-second-sample trick. The
        // rotated tangent-space normal needs its XY counter-rotated so
        // the bump direction stays consistent with the world-aligned
        // tangent frame (otherwise the lighting would visibly swirl
        // across the rotation transition).
        mat2 N_rot_back = at_blend > 0.001
            ? mat2(cos(-at_a), -sin(-at_a), sin(-at_a), cos(-at_a))
            : mat2(1.0);
        #define MAT_N_STRICT(i, s) (texture(u_normal[i], uvm * (s)).xyz * 2.0 - 1.0)
        #define MAT_N_ROT(i, s)    vec3(N_rot_back * \
            (texture(u_normal[i], uvm_r * (s)).xy * 2.0 - 1.0), \
            texture(u_normal[i], uvm_r * (s)).z * 2.0 - 1.0)
        #define MAT_N(i, s) mix(MAT_N_STRICT(i, s), MAT_N_ROT(i, s), at_blend)
        vec3 nm = MAT_N(kMatSand, kScSand);
        // Same un-rotated treatment for the rocky-rugged normal: the
        // tess-side displacement uses the unrotated height-map UV, so
        // the normal map MUST match or lit highlights drift off the
        // displaced bumps.
        vec3 rocky_strict_n = texture(u_normal[12], uvm).xyz * 2.0 - 1.0;
        vec3 grass_nm = mix(rocky_strict_n,
                            MAT_N(kMatGrass, kScGrass), _grass_present);
        // Same height-aware weights as the albedo splat above.
        nm = mix(nm, grass_nm, w_grass);
        nm = mix(nm, MAT_N(kMatDirt, kScDirt), w_dirt);
        nm = mix(nm, MAT_N(kMatRock, kScRock), w_rock);
        nm = mix(nm, MAT_N(kMatSnow, kScSnow), w_snow);
        nm = mix(nm, MAT_N(kMatRock, kScRock), w_steep);
        nm = mix(nm, MAT_N(kMatSand, kScSand), w_beach); // sand at waterline
        nm = mix(nm, MAT_N(kMatDirt, kScDirt), verge_w);
        {
            // Road normal is in road space; rotate its XY into the terrain
            // tangent frame (Tt, Bt) the splat normal is expressed in.
            vec3 rn = texture(u_normal[kMatRoad], road_uv).xyz * 2.0 - 1.0;
            vec3 rw = road_T * rn.x + road_B * rn.y;
            nm = mix(nm, vec3(dot(rw, Tt), dot(rw, Bt), rn.z), road_w);
        }
        #undef MAT_N
        #undef MAT_N_ROT
        #undef MAT_N_STRICT

        // ---- Micro-normal (AAA Phase 2) ----
        // High-frequency second normal tap in the last metres so the
        // ground keeps granular relief at boot height where the base
        // tiling has gone magnified-blurry. Granular dirt detail works
        // for every layer at this frequency; fades out by ~12 m.
        float micro_w = (1.0 - smoothstep(3.0, 12.0, cam_dist)) * g_str * (1.0 - road_w);
        if (micro_w > 0.01) {
            vec3 mn2 = texture(u_normal[kMatDirt], uvm * 5.7).xyz * 2.0 - 1.0;
            nm = normalize(vec3(nm.xy + mn2.xy * 0.55 * micro_w, nm.z));
        }

        // ---- Shore-following beach ripples ----
        // Real sand ripples, but the phase is HEIGHT ABOVE WATER
        // (h - waterLevel), so ripple crests are iso-height contours.
        // The waterline itself is a height contour, so the ripples are
        // automatically parallel to the coast and curve with every bay
        // — never a fixed world-axis stripe. Only on near-flat sand by
        // the water (beach), fading out up the slope; a gentle noise
        // wander keeps them from looking mechanical.
        // Phase = height above water → crests are iso-height contours,
        // so the lines run parallel to the coast and curve with it
        // (never a fixed world axis). Independent scales: sand uses
        // pc.grass_params.x, grass uses pc.grass_params.y (0 = off).
        // Both fade out on slopes; the normal is tilted along the world
        // uphill (height-gradient) direction so the lines catch light
        // as ridges.
        float flatv = clamp((N.y - 0.55) * 2.5, 0.0, 1.0);
        vec2  up2   = -N.xz;
        float ulen  = length(up2);
        if (ulen > 1e-3 && flatv > 0.001) {
            vec3  uw  = vec3(up2.x / ulen, 0.0, up2.y / ulen);
            vec2  ut  = vec2(dot(uw, Tt), dot(uw, Bt));
            float wob = (tnoise(vWorldPos.xz * 0.05) - 0.5) * 3.0;
            // Distance LOD: the ripple is sin() of world height, so far
            // away its spatial frequency outruns the pixel grid and
            // moirés into banding. Fade it out by ~110 m so distant
            // sand is smooth (no bands); near sand keeps the ripple.
            float rip_dist = 1.0 - smoothstep(35.0, 110.0, cam_dist);
            // Sand ripples: phase = height above water → crests are
            // iso-height ≈ parallel to the shore, curving with it.
            // Smooth (no texture-UV rotation → no per-triangle seams);
            // the normal tilt direction is the tangent-space uphill.
            float s_sc = pc.grass_params.x > 0.01 ? pc.grass_params.x : 9.0;
            float s_w  = beach * flatv * rip_dist * (1.0 - road_w);
            if (s_w > 0.001) {
                float ph = (h - wtr_lvl) * s_sc + wob;
                albedo  *= 1.0 + sin(ph) * 0.10 * s_w;
                nm.xy   += ut * (cos(ph) * 0.7 * s_w);
            }
            // Grass contour rows — independent scale, 0 disables.
            float g_sc = pc.grass_params.y;
            if (g_sc > 0.01) {
                float g_w = t_sand * (1.0 - t_dirt) * (1.0 - steep)
                            * (1.0 - beach) * flatv * rip_dist * (1.0 - road_w);
                if (g_w > 0.001) {
                    float ph = (h - wtr_lvl) * g_sc + wob;
                    albedo *= 1.0 + sin(ph) * 0.05 * g_w;
                    nm.xy  += ut * (cos(ph) * 0.4 * g_w);
                }
            }
            nm = normalize(nm);
        }

        // Splat-normal fade now 250-900 m (was 200-650). Past 900 m
        // the LOD-N coarse mesh shimmers if per-texel normals push it,
        // so still fades fully there -- but the distance-active FBM
        // detail below kicks in starting at 200 m to ADD relief that
        // grows with distance and compensates for the lost splat
        // texture detail.
        float nf = (1.0 - smoothstep(250.0, 900.0, cam_dist)) *
                   clamp(scene.spom_params.w, 0.0, 1.0) * g_str;
        // Tt/Bt were built above for the parallax march — reuse them.
        vec3  Nt = normalize(Tt * nm.x + Bt * nm.y + N * max(nm.z, 0.2));
        N = normalize(mix(N, Nt, nf));

        // Distance-active procedural FBM detail. Strengthens with
        // camera distance to compensate for the splat-texture detail
        // fading out at the same range. Two scales added: ~5 m for
        // erosion gullies + ~1.5 m for rock striations. The world-
        // gradient is folded into the shading normal so distant
        // surfaces get crinkled relief, and a ridge term darkens the
        // albedo in the cavities so the eye reads "eroded rocky" not
        // "flat coloured".
        // Start gentler at 250 m (was 150) and cap the cavity-darken
        // tighter (was 0.55..1.0 -> 0.85..1.0). The aggressive setting
        // produced visible black blobs on flat far terrain when tess
        // was off (more far terrain visible -> more pixels qualifying
        // for the cavity darken, stacking with shadow + AO to near-
        // black). Normal amplitude also reduced 0.55 -> 0.35 to avoid
        // grazing-angle pitch-black patches.
        // Additionally GATED to above-water pixels: underwater terrain
        // visible through the transparent water plane would otherwise
        // get the cavity-darken stacked on top of the water tint and
        // read as dark blotches on the lake floor.
        float _wtr_lvl_far = scene.water_params.y;
        float _wtr_on_far  = scene.water_params.x;
        float above_water = (_wtr_on_far > 0.5)
            ? smoothstep(_wtr_lvl_far - 0.20, _wtr_lvl_far + 0.40, vWorldPos.y)
            : 1.0;
        float dd_w = smoothstep(250.0, 700.0, cam_dist) * above_water * (1.0 - road_w);
        if (dd_w > 0.001) {
            vec3 dn1 = terrain_detail_fbm(vWorldPos.xz, 0.18, 4);
            vec3 dn2 = terrain_detail_fbm(vWorldPos.xz, 0.65, 3);
            // Floor N.y so the resulting vec3 can never collapse toward
            // (0,0,0) (which would NaN-out the normalize). The earlier
            // normal-mapping blend can drive N.y arbitrarily close to 0
            // on grazing facets; the final N.y = max(N.y, 0.18) clamp
            // below runs AFTER this block so it's not enough on its own.
            float dn_ny = max(N.y, 0.05);
            vec3 dN = normalize(vec3(
                N.x - (dn1.y + 0.45 * dn2.y) * 0.35 * dd_w,
                dn_ny,
                N.z - (dn1.z + 0.45 * dn2.z) * 0.35 * dd_w));
            N = normalize(mix(N, dN, dd_w));
            float cavity = smoothstep(-0.6, 0.4, dn1.x);
            albedo *= mix(1.0, 0.85 + 0.15 * cavity, dd_w);
        }
        // Up-bias floor so terrain never self-shadows to black under the
        // RT sun query (hard-won fix — do not remove).
        N.y = max(N.y, 0.18);
        N = normalize(N);
