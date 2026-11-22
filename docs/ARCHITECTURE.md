# Vibe — code map

Where things live and which file to open for a given change. Written for an
AI agent (or a person) who has to change something quickly without reading
the whole code base. Keep it short and true: when a file is split, renamed
or gets a new job, update the matching line here in the same commit.

## Shape of the project

```
src/main.cpp                 command line, log, creates the engine
src/game/                    gameplay that knows nothing about Vulkan
  level.cpp                    the castle: walls, towers, keep, stairs, colliders
  player.cpp/.h                movement, crouch, eye position
  collision.cpp, physics.cpp   AABB collision for the player
src/anim/                    character animation library (no Vulkan): see docs/animation/guide.md
src/engine/                  engine modules that are NOT the VulkanEngine class
  terrain.cpp, terrain_clod.cpp  height map, CPU twins of the terrain shader
  world_props.cpp              tree / bush / rock archetypes + where they stand
  world_props_meshes.inl       ... their mesh builders (conifer, broadleaf, shrub, rock)
  scatter.cpp                  small plants + loose stones: placement, buffers
  scatter_plant_meshes.inl     ... their meshes (fern, flowers, reed, water lily, ...)
  veg_paint.h                  the editor's paint map + presets
  road.cpp, grass.cpp, audio.cpp, physics_world.cpp (Jolt), texture.cpp, window.cpp
src/engine/vk_engine.h       the one big class: every member + method declaration
src/engine/vk_engine.cpp     init(), run() (the frame loop + input), shutdown()
src/engine/vk_engine/        the class's methods, one topic per file (see below)
shaders/                     GLSL 4.60; *.glsl = includes, compiled into the others
tools/                       scripts (character build, rock bake, split_shader.pl, split_cpp.pl, perf/bench.sh)
docs/                        this file, design notes, screenshots
```

Everything is in `namespace vibe`. Runtime files live in the project root:
`vibe_settings.cfg` (all settings + player pose, restored at start),
`vibe.log`, `screenshots/` (F12).

## `src/engine/vk_engine/` by topic

| Topic | Files |
|---|---|
| Frame: order of passes, submit, present | `frame.cpp`, `frame_upload.cpp` |
| Drawing the world (depth pre-pass + colour pass) | `world_render.cpp` |
| Vulkan setup, pipelines, descriptors, scene UBO | `setup.cpp`, `pipelines_setup.cpp`, `descriptors.cpp`, `helpers.cpp` |
| Ray tracing (BLAS / TLAS, instance masks) | `rt.cpp` |
| Deferred lighting, GI caches, denoise | `deferred_lighting.cpp`, `radiance_cache.cpp`, `restir.cpp`, `svgf.cpp`, `rr_guides.cpp` |
| Upscaling / frame generation | `dlss.cpp`, `fsr2.cpp`, `fsr3.cpp`, `taa.cpp` |
| Post: bloom, exposure, compose | `bloom.cpp`, `post.cpp` |
| Sky, clouds, froxel fog (incl. lake mist) | `atmosphere.cpp` |
| Day / night, weather, rain, lightning | `environment.cpp` |
| Sun shadows | `sun_shadow.cpp`, `sun_horizon.cpp`, `clutter_shadow.cpp`, `terrain_shadow_bake.cpp` |
| Terrain | `terrain_textures.cpp` (GPU textures, CLOD), `terrain_edit.cpp` (sculpt, grass mask), `craters.cpp` |
| Trees and bushes (GPU culled) | `vegetation.cpp` |
| Far trees as baked cards (8 views per tree type, baked at start-up) | `veg_impostor.cpp` (bake), `veg_leaf.glsl` `veg_imp_uv` (lookup), setting `tree_impostor_px` |
| World editor (item bar, painting, replant) | `veg_zones.cpp` |
| Castle meshes, props, doors, lanterns, flags | `castle_mesh.cpp`, `props.cpp`, `world.cpp`, `world_textures.cpp` |
| Launch site, rocket, its flight (orbit and back), launch button | `rocket.cpp`, `engine/rocket_sim.cpp` (the flight, no engine dependencies), models: `tools/rocket/make_rocket.py` |
| Combat: gun, projectiles, airstrike, jets, AA | `combat.cpp`, `aa.cpp`, `fx.cpp` |
| Player body (soldier, hands on the gun) | `player_body.cpp`, `flashlight.cpp` |
| Character animation: skeleton + clips (ozz), motion matching, jumps, inertialisation, warps, foot / hand IK, weapon layers, levels of detail | `src/anim/` (library, tested by `tests/anim/`); asset `assets/character/soldier.qanim` cooked by `vibe_animcook` (`tools/anim/cook_main.cpp`); docs in `docs/animation/` |
| LAN multiplayer: discovery, host election, snapshots, interpolation | `src/engine/net/net.h`, `net.cpp` (ENet; no engine types); what is replicated, its wire form, buffering, events: `net_state.h`; state -> a remote player's animator input: `net_anim.h` |
| Multiplayer game side: other players (avatars), their shots, hits, health, HUD | `multiplayer.cpp` |
| A killed player falls as a physics ragdoll | `ragdoll.cpp`, `physics_world.cpp` (`add_ragdoll`) |
| In-game HUD (health, weapon, slots, frame rate); fonts are loaded in `ui_setup.cpp` | `ui_gamehud.cpp` |
| Water effects (splashes, ripples) | `water_fx.cpp` |
| UI | `ui_setup.cpp`, `ui_hud.cpp` (HUD, crosshair), `ui_menu.cpp` (pause menu) |
| Settings: load / save / presets / types | `settings.cpp`, `settings_types.h` |
| Loading screen, screenshots | `loader_screen.cpp`, `screenshot.cpp` |
| Voxel tower | `voxel.cpp`, `../voxel/` |

## Shaders

| Shader | What it draws | Its includes |
|---|---|---|
| `cube.frag` (+ `cube.vert`) | every mesh: castle walls, towers, props, trees, mesh terrain. Built 3x: forward, `cube_gbuf` (G-buffer only, the normal case), `cube_gbuf_terrain` | `cube_surface_helpers.glsl`, `cube_noise_rays.glsl`, `cube_wall_relief.glsl`, `cube_main_terrain.glsl`, `cube_main_surfaces.glsl`, `cube_main_forward_lighting.glsl` |
| `terrain_raymarch.frag` | the raymarched ground AND the water (built twice, `VIBE_WATER`) | `terrain_rm_noise_road.glsl`, `terrain_rm_pebbles.glsl`, `terrain_rm_heightfield.glsl`, `terrain_rm_shading_helpers.glsl`, `terrain_rm_water.glsl` |
| `deferred_lighting.frag` | sun, shadows, AO, GI for everything in the G-buffer | |
| `grass_raymarch.frag` | the lawn (raymarched blades) | `grass_blades.glsl` |
| `scatter.vert/.frag` | small plants, flowers, loose stones, water lilies | `rock_sdf.glsl` |
| `veg_cull.comp` | tree culling + LOD on the GPU | |
| `froxel_fog.comp` | volumetric fog, lake mist | |
| `compose.frag` | sky fill, fog apply, tonemap, final image | |
| `scene_ubo.glsl` | the scene uniform block shared by all | |

`*_main_*.glsl` files are pieces of one `main()` included in place: they share
its local variables and are not self-contained.

## "I want to change ..." -> open this

| Change | File(s) |
|---|---|
| Castle layout, wall sizes, stairs, tower colliders | `src/game/level.cpp` |
| The chapel (shape) | `tools/church/make_church.py` (run in Blender, writes `assets/church/church.mesh`); its materials + colliders: `append_church` in `level.cpp`; its site + ground pad: `world.cpp` (search `[church]`) |
| Stone relief on straight walls (depth, corners, edges) | `shaders/cube_wall_relief.glsl` (helpers), `shaders/cube_main_surfaces.glsl` (the march); which faces count as open: `compute_brush_face_masks` in `world_render.cpp` |
| Round towers' stonework | `shaders/cube_main_surfaces.glsl` (static-mesh branch) |
| Tree look (crown shading, leaf cards) | `shaders/cube.frag` (foliage block after the includes); tree shapes: `world_props_meshes.inl`; where trees stand: `world_props.cpp` |
| A flower / plant / water lily shape | `scatter_plant_meshes.inl`; its colours: `shaders/scatter.frag`; where it grows: `scatter.cpp` |
| Ground colour, pebbles, puddles, shore | `shaders/terrain_rm_shading_helpers.glsl`, `terrain_rm_pebbles.glsl`, tail of `terrain_raymarch.frag` |
| Water surface, foam, lake bed | `shaders/terrain_rm_water.glsl` + the water block in `terrain_raymarch.frag` |
| Lawn grass | `shaders/grass_raymarch.frag` |
| Fog / mist | `shaders/froxel_fog.comp`, `atmosphere.cpp` |
| Weather, rain, thunder timing | `environment.cpp`; lightning bolt drawing: `ui_hud.cpp` |
| A new setting | `settings_types.h` (field + default), `settings.cpp` (`f("name", ...)` line + clamp), `ui_menu.cpp` (slider) |
| A HUD element | `ui_hud.cpp` |
| Editor items | `veg_zones.cpp` (`kItems`) |
| Gun, bullets, fire rate | `combat.cpp`; bolt glow: `fx.cpp` |
| The soldier's meshes: body without head, head with helmet, three levels of detail each | built by `tools/character/make_soldier.py`, exported by `tools/character/export_fp.py` into `assets/character/soldier_fp*.qchar`, `soldier_head*.qchar`, `soldier_atlas.jpg`; loaded and chosen in `player_body.cpp` (`load_skin_mesh`, `body_mesh_lod`) |
| The soldiers' shadows | Ray traced for the nearest six: `vk_engine/body_rt.cpp` (`record_body_rt`, called from `rebuild_tlas`), `shaders/skin_rt.comp`. On the ground for all: `clutter_shadow.cpp` (`draw_skinned_shadows`), `shaders/skin_shadow.vert` |
| Order of a frame's first steps (camera, poses, TLAS) | `frame.cpp`: `pose_characters_for_frame` runs before `submit_tlas_build` because the TLAS build skins the soldiers' ray-tracing copies |
| What the animator is told about the local player; the bone palette; the animation debug overlay | `player_body.cpp`; the skinning itself: `shaders/cube_skin.vert` (pipelines in `pipelines_setup.cpp`, palette buffer = scene set binding 33 in `descriptors.cpp`) |
| How a pose is made (order of layers, warps, feet, arms, recoil, reload) | `src/anim/character_animator.cpp`; tuning: `AnimTuning` in its header |
| Which captured frame is played next | `src/anim/motion_matcher.cpp` (ground), `src/anim/airborne.cpp` (jumps, falls, landings) |
| Where the hands hold the weapon, recoil and reload timing | `WeaponDef` in `src/anim/character_animator.h` |
| What is replicated about a player | `src/engine/net/net_state.h` (bump `kProtocolVersion`) |
| Player speed, jump, crouch | `src/game/player.cpp` |
| Anti-aircraft vehicles | `aa.cpp` |
| Loading screen | `loader_screen.cpp`, art in `assets/loader/loading_bg.jpg` |
| Key bindings and the controller | `src/engine/window.cpp` (`Action`, `InputBindings`: reads keys, mouse and gamepad into one `InputFrame`), `vk_engine.cpp` `run()` (acts on it), `ui_menu.cpp` Controls tab (rebinding; saved as `bind_key_*` / `bind_pad_*`) |

## Working on it

- Build: `./build.ps1 -Config Release`. Run from the project root: `build/vibe.exe`.
- Capture one frame: `build/vibe.exe --frames 260 --screenshot out.png --screenshot-after 230`
  (PNG if the path ends in `.png`). The pose comes from `vibe_settings.cfg`
  (`player_pos_*`, `player_yaw` and `player_pitch` in radians, `time_of_day`).
- Test hooks (environment variables): `VIBE_TEST_HOLD=1` pin the camera,
  `VIBE_TEST_CRAWL=1` crouch, `VIBE_TEST_SIZE=3440x1400` window size,
  `VIBE_TEST_EDIT=1` open the editor, `VIBE_TEST_ZONE=<preset>` paint a test
  patch, `VIBE_TEST_LIGHTNING=1`, `VIBE_TEST_FIRE=1`, `VIBE_TEST_MENU=1`.
- GPU cost per pass: the `gpu-passes ...` line in `vibe.log` (once a second).
  Measure at the played resolution (`VIBE_TEST_SIZE`), not the 1280x720 default.
- Anything added to the full terrain height function (`rm_ground_detail`) is
  paid on every terrain shadow / reflection step: add detail in the normal or
  the shading instead.
- Splitting a file that has grown too big: `tools/split_shader.pl` (GLSL line
  ranges -> includes) and `tools/split_cpp.pl` (a .cpp -> several).

## Conventions

- One topic per file; a file over ~1200 lines is a candidate for a split.
- New `VulkanEngine` methods go in the topic file under `vk_engine/`; declare
  them in `vk_engine.h` next to the related members.
- Settings names are `snake_case` and identical in the cfg file, the struct
  field and the `f("...")` line.
- Shader includes are named `<owner>_<topic>.glsl`.
- Internal names keep the old `vibe` / `VIBE_` prefix (namespace, test
  hooks, the `vibe_engine` library); the product name is Vibe.

## Multiplayer (LAN)

Model and rules are described at the top of `src/engine/net/net.h`. In short: every instance
broadcasts a beacon (UDP 47777); the lowest id hosts (ENet, UDP 47778+), the rest join; when the
host leaves, the others elect again. Each client simulates its own player and sends its state 30x
a second; the host relays snapshots at 60 Hz; other players are drawn 100 ms in the past,
interpolated. Shots / hits / deaths are reliable events; a hit is decided on the shooter's machine
against the avatar as drawn there (`mp_bolt_hits`).

Other players each have an animator of their own (`MpAvatar::animator`): `mp_pose_avatars` turns
their interpolated state into its input (`net::remote_animation_input`), picks a level of detail
by their size on screen, and writes their bone palette (the GPU skins). No pose data is sent: position,
velocity, view, movement keys, stance, weapon and a shot counter are (35 bytes, 30 times a second).
Shots are stamped events, de-duplicated and played when the delayed picture reaches their moment.
`draw_remote_players` draws body + what they carry through `draw_viewmodel` with their camera
matrix (`mp_draw_`). Whether a player can be hit never depends on whether or how they are drawn.

Off for automated runs (`--frames`, `--screenshot`) unless `VIBE_NET=1`; `VIBE_NO_NET=1` turns it
off entirely. Two instances on one PC work (the second takes the next game port).

## Levels of detail and shader builds

- **Trees / bushes / rocks** (`vegetation.cpp`, `veg_cull.comp`): GPU-driven - every instance is frustum- and Hi-Z-culled and given one of 4 mesh levels (or an impostor) on the GPU; drawn with indirect draws, near levels first. The colour pass tests depth EQUAL against the pre-pass.
- **Trees beyond the home tile** (`vk_engine/veg_stream.cpp`): the world is cut into 128 m cells; a cell's trees and bushes are a pure function of its coordinates and the world's forest rule (`wm_forest` in `shaders/world_map.glsl` / `world_forest` in `terrain.cpp` - also what colours the far landscape as woods). Cells are generated on background threads, nearest first, and their instances are written into a pool of `kWvPool` slots inside the vegetation instance buffer (between the home tile's instances and the clustered rocks), so the same GPU culling / LOD / impostor / shadow / ray-tracing path draws them. With distance only a leading share of a cell's rank-ordered trees is live (all, half, a quarter, a twelfth; bushes within 260 m). Trunk colliders for cells within 200 m (`wv_cols_`). `VIBE_NO_WORLD_VEG=1` turns it off. When the camera is too high to see a tree, or moving faster than 900 m/s, everything is given back and nothing is queued (space flight). **Grass beyond home**: `wv_nf_poll` bakes the ground around the camera (512 m at 1 m, the world formula, packed into an RGBA8 texture in normal-texture slot 22, re-baked every 72 m of travel) and `grass_raymarch.frag` (`nf_*`) stands its blades on it, where a rule of height and slope allows. Not yet: bullets / physics bodies against these trunks, blast damage that survives a cell reload, manual add / remove beyond home.
- **Screen-space displacement mapping (SSDM)** (`vk_engine/ssdm.cpp`, `shaders/ssdm_mesh.*`, `shaders/ssdm_resolve.frag`): a port of TechArtAlex's Unreal plugin "Screen-Space Raymarched Displacement Mapping". A mesh with `LevelMesh::ssdm > 0` (shell thickness, m; the stone bridge) is drawn a second time inflated to a shell, storing per pixel the screen-UV displacement of its height map; a full-screen pass then marches the screen for each pixel's source (plain march or the default bent "NCS" march). It runs on the G-buffer in the middle of the main pass (`render_ssdm`, called by `render_world` right after the castle meshes: the pass is ended, the displaced colour / motion vectors / G-buffer layers and depth are written back, the pass is begun again) - as Crimson Desert is reported to do it - so lighting and shadows fall on the displaced surface and trees, effects, grass and the weapon are drawn after it, against its depth. The shell mesh has normals rounded over hard edges (`build_ssdm_shell` in `castle_mesh.cpp`). `VIBE_NO_SSDM=1` off, `VIBE_SSDM_MODE=1` the plain march, `VIBE_SSDM_DEBUG=1` paints hits red and the rest blue. Not ported: the plugin's other debug views.
- **Small plants** (`scatter.cpp`): 32 m cells, culled on the CPU (frustum, distance, walls, hills). Each species has three meshes: full, ~32 % (beyond 22 m) and ~11 % (beyond 48 m), made at load time with meshoptimizer (`simplified` in `build_scatter`). `VIBE_NO_PLANT_LOD=1` turns them off.
- **meshoptimizer** (FetchContent) is available engine-wide: simplification, index / vertex ordering, meshlets. `clusterlod.h` from the same project is the starting point for Nanite-style cluster LOD if that is built.
- **terrain_raymarch.frag** is compiled three ways: normal (with winter: snow layer, footprints, ice), `VIBE_SUMMER` (no winter code; bound while there is no snow) and `VIBE_WATER`.

## Cluster LOD (Nanite-style) - rocks

- `src/engine/cluster_lod.h/.cpp`: builds a cluster hierarchy for a mesh with meshoptimizer's `clusterlod.h` (clusters of <=128 triangles, simplified level by level; each cluster carries its own and its parent group's error sphere).
- `vk_engine/cluster_draw.cpp` + `shaders/cluster_cull.comp`: every frame, one compute thread per clustered instance culls it and tests each cluster (drawn when its error is under one pixel while its parent's is not), writing one draw command per cluster; `draw_clusters` issues them with `vkCmdDrawIndexedIndirectCount` through the vegetation pipelines.
- Static world meshes (castle, chapel, gravestones; parts of 3000+ triangles): ``init_static_clusters`` / ``cull_static_clusters`` in the same file with ``static_cluster_cull.comp`` - one thread per cluster, one indirect-count draw per mesh part (camera passes only; the sun shadow map draws whole parts). ``VIBE_NO_STATIC_CLUSTERS=1`` turns it off.
- The sun shadow map draws from the same hierarchy: a second selection per frame (light box, cut at 2 px by the camera's distance).
- Streaming: a part's finest level (the original triangles, half the index data) lives in ``cache/stream_*.idx`` and is read by a background thread into a full index buffer when the camera comes within reach (``stream_static_clusters``); the coarse buffer is a prefix of the full one, the shader's ``loaded`` mask makes level 1 stand in meanwhile. ``VIBE_STREAM_ALL=1`` keeps everything loaded. Vertices and textures are not streamed.
- Sub-pixel triangles: the cut keeps the finest triangles at about a pixel or more when first drawn (logged at start), so no software rasteriser is used.
- Hierarchies are cached in ``cache/cluster_<hash>.bin`` (``build_cluster_mesh_cached``; bump ``kCacheVersion`` when the builder changes).
- Occlusion uses LAST frame's depth pyramid on purpose: the main occluder, the terrain, is ray-marched after the object passes, so this frame's depth does not contain it when the culling runs (a two-pass scheme would lose the hills).
- Clustered rocks: the Blender-baked rock archetypes of the vegetation system (their instances are sorted to the end of the instance buffer; `veg_cluster_first_` / `veg_cluster_count_`). `VIBE_NO_CLUSTERS=1` draws them with the old discrete levels.
- To cluster another mesh type: build a `ClusterMesh`, add it to the cluster / table buffers in `init_cluster_draw`, and give its instances slots like the rocks. Not yet done: per-cluster frustum and occlusion culling, ray-tracing structures from clusters.

## Leaving the planet (state: 2026-11-18)

**Rule: nothing below 1 km altitude may change.** Every piece here is gated by the camera's height.

What exists:
- **Flight**: free-fly (V) speeds up with height above the ground (x altitude / 60 m, up to 2000x), ceiling 1000 km (`vk_engine.cpp`, fly block).
- **The planet as a far field** (`compose.frag` `planet_surface`, used by `atmo_sky` in the sky-fill and compose passes): view rays that leave the local landscape and meet the planet's sphere (radius 6360 km, the atmosphere model's) are shaded as land / sea / snow / weather systems with day and night; fades in between 1 and 6 km. Above the atmosphere (40 - 110 km) the sky-view LUT is replaced by black space, a veil over the day side and a limb glow.
- **Directions at height**: above 2 km the sky passes get the inverse of projection x rotation only (`frame.cpp`), because the full inverse view-projection loses the view directions in float rounding.
- **Local systems step aside**: the terrain march returns at once when the landscape lies beyond its 1500 m range (`terrain_rm_heightfield.glsl`); the water plane is not drawn beyond 2.5 km from above 1 km; the local cloud layer fades out over 3 km above its top.

- **The middle distance** (`planet_surface`, the `w_mid` block): out to 45 - 90 km from home the sphere is shaded from the landscape's own height formula (`pl_terrain` = `terrainM` without plateau and sculpting, with only the octaves the view resolves), with the same height layers (sand / grass / dirt / rock / snow from the terrain settings), water level, ground snow and the aerial-perspective volume's haze. So the country seen from an aircraft is the one walked on. The CPU hands the shader the camera's map position, water level, snow and the height layers through the tail of the exposure buffer (`frame.cpp`, before the culls; the compose push constants are full at 256 B). The planet's map is fixed to the world now (the camera's travel rotates the sphere under it); it used to follow the camera. Fades in between 1 and 1.2 km altitude; from 1 km up the water plane ends where the terrain march ends (1.5 km).
- **The raymarched landscape from the air** (`terrain_rm_heightfield.glsl` `rm_far_limit` / `raymarch_far`, called from `terrain_raymarch.frag` only above 1 km for rays the normal march cannot reach): the march range grows with altitude (1.5 km at 1 km up, 80 km from about 1.5 km up), with all octaves so lakes and the snow line stay put. Depth for those hits is clamped just in front of the sky value, as before. The ground haze grows slowly past 1500 m and counts only the part of the ray in the 2.5 km of air near the ground (`rm_haze_dist`; `planet_surface` applies the same to the far field behind it). 
- **Objects from the air** (`world_render.cpp` `view_far_` / `object_far_`): up to 1 km altitude the projection is 0.05 .. 1500 m as always; above it the far plane moves out (to 30 km) and trees, rocks and buildings are drawn to 3.8 km (`veg_cull.comp`, `cluster_cull.comp` take `object_far_`; the upscalers get `view_far_`). The near plane stays (the weapon in the hand), so depth out there is metres coarse - fine for objects a few pixels tall. The 3.8 km follows from where the terrain shader clamps its depth (0.999989); going further needs reversed depth, and camera-relative positions beyond ~1000 km.
- **The hand-overs, castle to space** (state 2026-11-18; all thresholds are camera altitude):
  - The far march runs from the ground (range 1.5 km + 100 m per metre of height, 80 km from 0.8 km up); the far field comes in between 100 and 220 m and the cross-fade into it with it (distance = where the ray first passes below the highest ground, so it starts above 220 m); the water plane ends with the range from 220 m; it draws in again between 25 and 60 km, where the painted far field and its weather take over); the water plane ends with it; the planet's far field fades in (300 - 450 m).
  - The raymarched ground cross-fades into the far field over the outer half of its range, in `compose.frag` (after the aerial perspective; the distance is that of the ground under the pixel's ray). Keep `rm_far_limit` and that block in step.
  - Far hits get their normal from the height formula (`calcNormal`, t > 1500), a fixed share of occlusion and no contrast curve - beyond the range of the AO / shadow terms the ground was a step brighter and sunlit snow went to plain white.
  - 1 km: far plane and object distance grow (above).
  - Above the cloud layer's top the clouds are composited over the land (`clouds.comp` renders over far geometry then; `compose.frag` filters them wide); they fade out 4 - 14 km above the top.
  - 20 - 110 km: sunlight on the far field goes from the engine's own term to the orbit value; from 40 km the sky-view LUT gives way to space and the veil.
  - The view-projection inverse is computed in double precision, the terrain rays and the viewmodel are camera-relative (float positions shook with altitude).
  - The far field's home country (same height layers, snow, average look) reaches 350 - 900 km from home.
  - **Clouds: one weather map for every view.** `clouds.comp` grows its volumetric clouds where a 2D weather field says so; `compose.frag` `weather_cloud()` is the same field (same hash, cells, wind drift, coverage). The clouds painted on the raymarched land (compose, before the cross-fade) and on the planet's far field (`planet_surface`) come from it, weighted by `weather_paint_w()` = how far the volumetric ones have faded. So what is seen from orbit over home is what is found on the way down. From inside or above the layer the volumetric clouds are rendered over the land as well (the land lies below the layer's base) and composited before the upscaler by the `CLOUDS_OVER` variant of compose.frag (`frame.cpp`, next to the sun-shaft pre-pass). Change the field in both places.
  - Test hooks: `VIBE_TEST_FLIGHT=<y0>,<y1>,<frames>[,<yaw per frame>]` (with `VIBE_TEST_HOLD`) flies the camera between two altitudes in one motion; with `--frames` a given frame is the same in every run, so a flight can be captured frame by frame.
  - **One lighting at every altitude.** The far field's land (`planet_surface`) and the painted clouds are lit as `terrain_raymarch.frag` lights the landscape: the same sunlight (`u_exposure.sun_land` = scene.sun_color, golden hour included), the same sky fill (scene.sky_color x 0.75 x (0.5 + 0.5 up)) and the same direction; around the planet the sun's strength changes only relative to home. There is no dimming toward orbit. In the terrain shader `rm_air_look()` blends the ground look (occlusion, contrast curve) into the air look by distance and by camera height (250 - 1200 m), so the land right below is not a brighter patch.
  - **The world as one formula at every zoom level: `shaders/world_map.glsl`.** Included (the same text) by `terrain_rm_noise_road.glsl` for the raymarched landscape and by `compose.frag` for the planet's far field, each with its own `WM_NOISE` (same hash). `wm_continent(km)`: > 0 land, < 0 sea - a domain-warped fractal for continents (2400 .. 150 km), a detail band for coasts (90 .. 5 km) and for the shoreline (3 km .. 700 m); home is placed on a continent's east coast by two broad terms laid out on a bent copy of the map. `wm_ranges`: where the high ranges stand. `wm_height(a, wp, water)`: every height the landscape formula makes passes through it - home (1.5 km) untouched, then low country with ranges, a coastal plain coming down to the beach, shallows, shelf and deep sea bed. The far field takes its land mask, shallows, ranges and snow from the same functions. The CPU has its own copy (`terrain.cpp` namespace `wm`, `world_ground_height`; keep it in step line for line): outside the baked home tile `sample_terrain_height` returns it, so the player can land, walk and swim out there. Physics bodies (Jolt) still exist on the home tile only. `VIBE_TEST_GROUND=1` (with `VIBE_TEST_HOLD`) stands the held camera on that ground - a capture shows whether it is the drawn ground. Cost: it roughly tripled the terrain march from the air (0.6 -> 1.8 ms at 1280x720); the continent value could be evaluated once per pixel instead of per step.

**The globe** (`engine/globe.h/.cpp`; state 2026-11-20). The planet is a real sphere (radius 6360 km); flying on round it comes back home. Three parts:
- *Charts.* The engine still works in flat float coordinates (x east, y up, z south). They are a chart of part of the planet: the sphere unrolled round one point (distance and bearing from it kept), with a frame there (which way its east / up / south point in the planet's frame; home is the planet's top, identity). Home's chart reaches 100 km. Beyond it `globe_update` (rocket.cpp, every frame) gives the player a far chart, re-anchored under them every 16 km, its origin at local (64 km, 0) - so coordinates stay under 100 km (float fine to a centimetre) and nothing of home's content, which sits at its home coordinates near (0, 0), is ever in range. Within 92 km of home the home chart is taken up again. `globe_enter` moves the player between charts (same place and heading on the planet). The shaders get the chart in the scene UBO (`globe_frame`, `globe_info`).
- *The world on the sphere* (`shaders/world_map.glsl` "The globe", mirrored in `terrain.cpp` + `globe.cpp`). Near home the world is the flat formula it always was (home coordinates = the azimuthal map round home). That map cannot go round a sphere, so away from home the world is made from 3D noise at the point: the landscape FBM between 40 and 90 km from home (`wm_shape3` / `rm_shape`), continents and ranges between 1800 and 3200 km (`wm_continent_globe`). A float cannot hold a position on the planet finely enough for the FBM's small octaves: the CPU supplies, per octave, the lattice cell and fraction of a nearby origin in double precision (`globe::lattice`; `globe_cell` / `globe_frac` in the scene UBO for the chart's origin, `gcell` / `gfrac` in the exposure buffer for the camera's ground point) and the shader adds the local offset. In a far chart the terrain shader switches off everything that belongs to home's tile (`kHeightmapSide` is nothing there, no plateau, no road, `rm_home_dist`).
- *The far field* (`compose.frag` `planet_surface`) gets the frame at the camera's ground point (`gf0..2`) instead of the old "how far has the camera travelled over the flat map", and its mid-distance heights (`pl_terrain`) come from the same formula everywhere on the planet.
- The rocket's simulation plane is placed on the planet through the pad; its drawn position is the straight line from the eye in the eye's own frame, in whatever chart. The rocket camera simply sits beside it and the chart follows.
- The sun (`environment.cpp`, `from_here`): the clock is home's; the sun and moon stand where they stand over the planet and are seen from the player's own east / up / north, so it is another time of day elsewhere and night on the far side.
- Not done: weather, the shader's fine ground noise and the streamed trees are laid out in chart coordinates, so they change when a far chart is re-anchored; multiplayer positions assume home's chart. Test: set `player_pos_x` (home coordinates, m) beyond 100000 in the cfg copy - the engine re-anchors on the first frame (`[globe]` in the log).

**The launch site** (`vk_engine/rocket.cpp`): a terrace in the world formula (`wm_pad` in `shaders/world_map.glsl`, `wm::pad` in `terrain.cpp`, `kLaunchSite*` in `terrain.h` - keep in step) at (1600, 830) on the coast, outside the home tile. Apron, mount, tower, tanks, masts and the control house are static level meshes from `assets/rocket/*.mesh` (written by `tools/rocket/make_rocket.py`, run in Blender; it also writes the hull and concrete textures, slots 23 / 24, whose normal maps the engine derives from the `_h` height images). The rocket is drawn with a model matrix (`draw_rocket`, parts in `rocket_parts_`, in the TLAS while near). E at the button in the house starts a 10 s count. The flight is `RocketSim` (`engine/rocket_sim.cpp`): a point mass in the orbit's plane, planet-centred, doubles, fixed 0.05 s step - gravity turn, closed-loop climb to a 180 km circular orbit, one revolution, a retrograde burn timed by running a copy of the flight ahead, drag-only entry, zero-effort-miss landing burn onto the mount (93 minutes of flight). `tick_rocket` runs it faster than the clock while the rocket is out of sight (x5 above the air, x60 in orbit). Its world position is the simulation's on the round planet (it sets below the horizon downrange and returns from the west). Beyond the object range it is drawn as a proxy on the line of sight (`rocket_view`: pulled in to 0.9 x the object far plane and scaled by the same ratio; hidden when the planet or - by a CPU march over the ground - the landscape is in the way); exhaust, entry glow, smoke trail and the pad's clouds are FX ribbons (`rocket_build_fx`, consumed in `fx.cpp`). C toggles the rocket camera (`rocket_toggle_camera`; the cam block in `tick_rocket`): the view rides 190 m beside the rocket, the mouse turns it. The rocket is drawn relative to the eye (`far_eye` in `draw_rocket`) wherever the eye is far from the chart's origin. `compose.frag` leaves an object in the air out of the land's cross-fade into the far field (`in_air`). The control house's lamp and the exhaust are point lights in lamp slots 7 and 6 (`descriptors.cpp`). Profiling the terrain shader: `VIBE_RM_SKIP=<mask>` leaves sections of it out at run time (bits listed at `RMDBG` in `terrain_rm_noise_road.glsl`); the per-pass GPU times are the `gpu-passes` line in the log. Do not time a section by cutting the shader short with an early return: the compiler removes whatever no longer reaches the output and the numbers mean nothing. Test hooks: `VIBE_TEST_NF_REBAKE=1` re-bakes the near-field ground every few seconds (its in-frame upload); `VIBE_TEST_ROCKET_CAM=1` turns the camera on at lift-off; `VIBE_ROCKET_SIMTEST=1` logs the whole mission in numbers at start-up; `VIBE_TEST_ROCKET=<s>` presses the button after s seconds, with `VIBE_TEST_ROCKET_T=<mission s>` jumping the flight ahead at lift-off (5490 = just before the landing burn).

**The river** (`wm_river` in `shaders/world_map.glsl`, `wm::river` in `src/engine/terrain.cpp` - same course, keep in step): a polyline from the castle lake to the sea, cut into every height the formula makes (home tile bake included; a loaded sculpt gets the cut applied in `world.cpp`). To lay out another: `VIBE_DUMP_HEIGHTS=<file>` (with `VIBE_TEST_HOLD`) writes the CPU ground as raw float32 every 8 m over x -1400..3000, z -1400..1400.
  - Verified with a contact sheet per altitude (28 m .. 400 km), frame-to-frame difference of a held camera and GPU frame cost.

- **Number precision (audit 2026-11-18).** Positions are 32-bit floats in world space. Done in double precision or relative to the camera, because they visibly broke with altitude: the view matrix (`player.cpp`), the inverse view-projection (`world_render.cpp`), the sky passes' rotation-only inverse (`frame.cpp`), DLSS / frame generation's clip-to-previous-clip (`dlss.cpp`), the terrain's view rays and the viewmodel. Still plain floats: every other rasterised object's `vp * model` (trees, avatars, projectiles, particles, jets). That is exact to about 2 mm at 20 km and 3 cm at 400 km from the origin - only matters for things drawn close to a camera that is very high up or far out (another player beside you in orbit would shake). The cure is camera-relative rendering for all objects.

What is still missing (in the order to build):
1. **Objects in the middle distance**: trees and buildings end at 3.8 km from the air (1.5 km on the ground). Beyond the raymarched range the far field is shading only. Older note: it is shading on the sphere - no silhouettes, no parallax; the real landscape (drawn to 1.5 km) is a little brighter than the layer around it between 1 and 1.5 km up. Trees and buildings are not in it.
2. **Camera-relative rendering**: positions are plain floats; above ~1000 km (and for anything drawn near the camera at height) precision runs out. Render relative to the camera, keep world positions in doubles on the CPU.
3. **Depth range**: the projection is 0.05 .. 1500 m (also in `fsr3.cpp`, `dlss.cpp`, several shaders). Move to reversed-Z with the far plane at infinity before anything real is drawn far away.
4. Other planets / a moon as bodies, and gravity that points at the planet's centre.

## Character animation (`src/anim`, `docs/animation/`)

A library of its own (`vibe_anim`: glm + ozz-animation, no Vulkan), one `CharacterAnimator` per
character. It is told what the controller says (position, velocity, wish direction, view, ground
contact, stance, what is held, shot and reload counters) and returns joint matrices; it never
writes to the controller. `docs/animation/guide.md` is the working manual (API, cooking, tuning,
hooks, tests, known gaps), `architecture.md` the design and status, `performance.md` the numbers.

- **Ground**: motion matching on 100STYLE capture (`motion_matcher.cpp`), the path predicted with
  the controller's own movement model (`trajectory.cpp`).
- **Air**: captured two-foot jumps scrubbed by vertical speed and predicted time to contact
  (`airborne.cpp`): no clip can run out, a landing never holds the player.
- **Every change of source** is hidden by inertialisation (`inertializer.cpp`).
- **Fitting to the game**: game speed mapped onto the pace the capture sustains in that direction,
  playback rate, stride (a foot beyond the leg's reach leaves the ground), legs turned to the
  travel direction (diagonals included), lean; feet locked in the world below 4 m/s and following
  the ground. `vibe_animbench <asset> slide` measures what is left of foot slide.
- **Weapon**: the weapon frame comes from the view; both arms are fitted to its grips last;
  recoil, reload and equip move the frame (`weapon_offset`, which `draw_viewmodel` uses too).
- **Data**: `assets/character/soldier.qanim`, cooked by `build/vibe_animcook.exe <100STYLE dir>
  assets/character/soldier_fp.qchar assets/character/soldier.qanim` (raw BVH not in the repo).
- Hooks: `VIBE_ANIM_DEBUG=1` (overlay), `VIBE_BODY_FRONT` / `_TURN` / `_WALK` / `_STRAFE` / `_AIR` /
  `_FIRE` / `_RELOAD`, `VIBE_ANIM_TUNE="name=value,..."`. R reloads (cosmetic; firing ends it).

## Ragdolls (`vk_engine/ragdoll.cpp`, `PhysicsWorld::add_ragdoll`)

A killed player's body goes to the physics: eleven capsules (pelvis, torso, head, upper arms,
forearms, thighs, shins) joined by cone joints and knee hinges, created in the pose and with the
velocity of the moment of death. The `Hit` network event carries where the shot struck and its
direction; the nearest part gets a knock there, the whole body a shove, both scaled by the damage.
Each frame the capsules are read back into the skeleton and the mesh is skinned (until the
physics sleeps). Up to 6 bodies, 25 s each. The parts of one body do not collide with each other.
Each ragdoll has a floor slab of its own at the height the game's terrain function gives, so it
also works beyond the home tile. Test: `VIBE_TEST_RAGDOLL=<frame>` (+ `_Y` hit height, `_POWER`)
with `VIBE_BODY_FRONT`; `VIBE_RAGDOLL_DEBUG=1` logs positions.

## First start: shaders compile while you play

The terrain shader is ~10 MB of SPIR-V; a driver needs a minute or more to compile it the first
time (afterwards it comes from `vibe_pipeline.cache` / the driver's cache). The game does not wait:

- `terrain_raymarch_lite.frag.spv` (`-DVIBE_LITE`, built from the same source): the same march and
  normal, so the ground is where it belongs, with the simplest shading (`lite_out`, the two
  `#ifdef VIBE_LITE` blocks in `main`). Compiles in ~5 s. Drawn until the full one is ready.
- The full pipeline and its two variants (summer, low-res) build on their own threads from
  `init_terrain_raymarch_pipeline`; `finish_terrain_raymarch_pipeline(0)` polls them every frame and
  the draw switches the moment the main one arrives. A HUD line says so meanwhile.
- The water-plane shader (mesh terrain only) also builds on a thread and is never waited for.
- `VIBE_LITE_TERRAIN=1` shows the lite build. When adding heavy code to the terrain shader, keep it
  after the lite blocks' `return` so the lite build stays small.

## Frame time: how to measure it, and the speed settings (state: 2026-11-21)

**Measure with `tools/perf/bench.sh`** (Git Bash, from anywhere in the repo): three fixed views (close ground on the sand road, a far view, the forest), 400 frames each at 3440x1400, on a copy of `vibe_settings.cfg` with time, weather and season pinned; it prints the main passes and the sum per view. `key=value` arguments override settings in that copy, so an A/B is two runs (`-n off terrain_rm_prepass=0`, `-n on`). It makes a throw-away run first: after a terrain shader change the driver compiles the full shader for a minute or two and the game shows the lite build meanwhile - numbers from that time are not a result. Nothing else may use the GPU (another `vibe.exe`, a session's `vibe_test.exe`, a game): the script warns, and the numbers swing by 2-4x. The second PC (garage430, Radeon) gives clean A/B numbers when this one is busy: `BenchTest.cmd` there, started with schtasks (memory note `reference_vibe_amd_test_pc`).

**Where the time goes** (RTX 4080, 3440x1400 at render scale 0.5, before this work): the frame is GPU-bound (the CPU's own work is ~2 ms). Open ground: the terrain raymarch (half the frame). Forest: the deferred lighting's rays. Wide views: vegetation. DLSS is a fixed ~2.6 ms.

**Terrain raymarch** (`terrain_raymarch.frag`, `terrain_rm_heightfield.glsl`; settings `terrain_rm_*`, Graphics > terrain > "Speed"; shaders read them from `scene.rm_perf`):
- *Low-resolution march first* (`terrain_rm_prepass`, block size `terrain_rm_prepass_div`): `terrain_march_lo.frag.spv` (the same source with `VIBE_MARCH_ONLY`) draws one ray per block of pixels into an RG32F target (`tr_march_image_`, binding 34) before the world pass (`frame.cpp` "Pass 0.6"): `.r` = the last distance at which the ray was clear of the ground by more than the block's width, `.g` = the same for a blade's height (the grass pass starts its march there). The full pass starts each pixel at the nearest of its four blocks (`raymarch(..., t_start)`); a pixel that turns out to start inside the ground marches from the eye. The block's ray stops at the farthest depth-pre-pass pixel under it (it must not march behind trees - that made the pass cost more than it saved).
- *Fast final approach* (`terrain_rm_converge`): close over the ground the step `kStep * h` shrinks with the clearance; two real-surface steps give the rate at which it is lost and the next step aims at where it runs out (capped), overshoot is caught by a three-step root find. 40-64 steps became 12-20.
- *Normal detail limit* (`terrain_rm_normal_px`) drops normal octaves finer than that many pixels; *simpler relief under dense grass* (`terrain_rm_relief_lite`).
- The sand's and the snow's own normals reuse the pixel's plain surface normal (`g_nrm_plain`, set in `calcNormal`) - they used to take four more taps of the whole landscape.
- *Sun shadows at low resolution* (`terrain_rm_shade_lo`, OFF): the two shadow terms worked out once per block in the low-resolution pass (second attachment `tr_shade_image_`, binding 35) and read back with a same-surface test. Measured: it costs that pass as much as it saves the main one (a loss on the Radeon). Kept as an option.
- *Ground shading rate* (`terrain_vrs`, OFF): VRS on the terrain pipeline in the world pass (rate image from `init_vrs_attachment`). -15 % / -24 % of the pass, but the 2x2 blocks show toward the screen edges.
- Profiling switches: `VIBE_RM_SKIP` bits (see `RMDBG`): 64 = the march alone, drawn as a step count; 128 = march without relief; 256 = green where low-resolution shadows were taken.

**Rays through foliage** (`foliage_ray_mode`, default 1; `deferred_lighting.frag` `g_rq_leaf_solid`, bits 2-3 of `scene.rt_flags.x`): a ray through a tree crown stops at every leaf card to ask the leaf texture. Bounce light, ambient occlusion, the shadow blocker search and sun shadows beyond 70 m now take the first card as solid (ray flag OPAQUE); sun shadows near the camera keep the exact leaf shapes. Forest lighting on the Radeon: 41 ms -> 17 ms. Not in winter (bare trees keep their cards in the BLAS). In the terrain shader only the single far shadow ray uses it: with solid cards the blocker search "finds a blocker" for nearly every pixel under a tree and the full set of shadow rays follows - slower.

**FSR path** (non-NVIDIA): `fsr_min_render_scale` (0.77) is the least it renders at (`setup.cpp`), and the sky-fill pass runs for FSR 3 as it does for DLSS (`frame.cpp` "Pass 2.15", `compose.frag` `pc.tone.x > 15.5`) - without it objects kept a rim of the placeholder sky colour.

**Did not pay / not done**: `half_rate_shadows` has no effect with deferred lighting on; `terrain_raymarch_scale` 75 % helps open ground but costs in forest (the low-resolution terrain path does not stop at objects); trees switch to their impostor at 7 px because the impostor is a generic procedural silhouette - switching earlier needs baked multi-view impostors.

**Corrections to "Leaving the planet" above (checked against the code, 2026-11-21):** the planet's far field comes in between 100 and 220 m of altitude (`compose.frag` `planet_surface`: `smoothstep(0.10, 0.22, alt)`), not at 1 km or 300 - 450 m; its colour target then blends toward the full sky model over 220 - 700 m. The far march's range is `1500 + 100 x altitude` metres, 80 km from about 785 m up, drawing in again between 25 and 60 km. The water plane is cut where that range ends only from 220 m up, and the cut closes in from the horizon between 220 and 1200 m (`terrain_raymarch.frag`, `t_water_plane`). The sky passes use the rotation-only inverse view-projection at every height (the full one lost the view directions in float rounding well below 2 km: bands in the sky); compose keeps the full matrix only for the terrain brush ring and the underwater view. Water: style 2 ("lake") runs the river branch (`water_style >= 1`) - the lake branch below it is never reached; left so because the look in use was tuned that way.
