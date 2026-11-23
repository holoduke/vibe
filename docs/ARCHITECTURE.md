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
src/engine/vk_engine.cpp     init(), run() (the frame loop), shutdown()
src/engine/vk_engine/game_loop.cpp   one turn of run(), the game half: loop_test_input, loop_handle_keys,
                             loop_weapons_and_edit, loop_simulate, loop_upkeep (called in that order)
src/engine/vk_engine/state/*.h       private state of 25 subsystems (see "Private state" below)
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
| Water surface, foam, lake bed | `shaders/terrain_rm_water.glsl` + `terrain_rm_main_water.glsl` (the water part of main()) |
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
| What is replicated about a car, who simulates it | `net_state.h` "Cars", `jeep.cpp` `jeep_net_send` / `jeep_net_receive` |
| The colour far things fade into | `shaders/ground_haze.glsl` |
| Where the ground is shaded coarsely | `shaders/terrain_vrs.frag`, `world_render.cpp` `tr_vrs_near_m` |
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

Cars are replicated too: see "Cars: drawn between physics steps, and on the network" below.

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

**Bounce light on leaves** (`gi_leaf_lite`, default on; `deferred_lighting.frag` `gi_leaf1`, bit 4 of `scene.rt_flags.x`): a leaf pixel takes one bounce-light ray and no second-bounce ray (a fixed average stands in for it). Measured clean on the RX 9070 XT at 2560x1440 (2026-11-22), lighting pass off -> on: forest 13.15 -> 10.59 ms, far view 4.62 -> 4.05, close ground 3.04 -> 2.64. Two frames a frame apart differ no more with it than without. What the forest's 13.2 ms was: bounce light 5.1 (rays and its 13-tap history gather, on vegetation only - the ground takes none), sun shadows 2.2, ambient occlusion 1.4, the rest 4.5 with no rays at all (not yet looked into). *Tried and dropped:* bounce-light rays on alternate pixels in a chequerboard for pixels with a settled history - 0.2 to 0.8 ms, because under trees few pixels keep a history.

**Before trusting a number from the test PC, look for a game left running there** (`tasklist /S 192.168.2.74 | findstr vibe`): on 2026-11-22 an idle `vibe.exe` rendered at 78 fps beside every benchmark from 10:17 on and read as a 2 - 3 ms "regression" (and distorted the ablations). Clean sums that day, default settings: close 24.7, far 21.1, forest 31.2 ms before the leaf change; 24.4 / 20.6 / 28.7 after.

**FSR path** (non-NVIDIA): `fsr_min_render_scale` (0.77) is the least it renders at (`setup.cpp`), and the sky-fill pass runs for FSR 3 as it does for DLSS (`frame.cpp` "Pass 2.15", `compose.frag` `pc.tone.x > 15.5`) - without it objects kept a rim of the placeholder sky colour.

**Where the terrain pass's time goes now** (2026-11-22, same machine and size, `VIBE_RM_SKIP` and setting overrides one at a time; close view 13.6 ms / far view 7.9 ms): the march alone 5.6 / 3.1 (of which the close relief 2.0 in the close view - on the sand road every near step works out the sand drift); everything the close relief costs, march, normal and shadow steps together, 4.7 / 2.5; ray-traced object shadows 2.0 / 1.2; the ground's own sun shadow 1.6 / 0.9; bounce light 1.1 / 0.7 (plus 0.4 - 0.6 in the deferred pass); the finite-difference normal 0.9 / 1.2; stones in the normal 0.7 / 0.8. Close-up textures, grass blade shadows, the small plants' shadow map, ambient occlusion and the stones in the march: under 0.3 each. Shading, not marching, is the larger half: only shading fewer pixels moves it much (`terrain_raymarch_scale`, with `terrain_rm_temporal` to win back detail).

**Object shadows on the ground under Ray Reconstruction** (`terrain_raymarch.frag` `rm_rt_sun_shadow`): 1 or 2 blocker rays instead of 4 and half the shadow rays a frame, the rule `deferred_lighting.frag` already had. -0.6 ms in the forest, -0.2 on open ground (parallel rays are cheap: the first one pays for the traversal).

**Strategies to try** (Graphics > terrain): `terrain_rm_start_at_hit` (the low-resolution march goes on to the ground and each pixel starts just short of that hit: measured a small loss, the march is not the bottleneck) and `terrain_rm_temporal` (with the scale below 100 %: two low-resolution targets drawn in turn a quarter of a texel apart, `terrain_raymarch_compose.frag` puts this frame's together with last frame's, bindings 36/37; costs nothing measurable over the scale alone, softer than full resolution).

**Did not pay / not done**: `half_rate_shadows` has no effect with deferred lighting on; `terrain_raymarch_scale` 75 % helps open ground but costs in forest (the low-resolution terrain path does not stop at objects); trees switch to their impostor at 7 px because the impostor is a generic procedural silhouette - switching earlier needs baked multi-view impostors.

**Corrections to "Leaving the planet" above (checked against the code, 2026-11-21):** the planet's far field comes in between 100 and 220 m of altitude (`compose.frag` `planet_surface`: `smoothstep(0.10, 0.22, alt)`), not at 1 km or 300 - 450 m; its colour target then blends toward the full sky model over 220 - 700 m. The far march's range is `1500 + 100 x altitude` metres, 80 km from about 785 m up, drawing in again between 25 and 60 km. The water plane is cut where that range ends only from 220 m up, and the cut closes in from the horizon between 220 and 1200 m (`terrain_raymarch.frag`, `t_water_plane`). The sky passes use the rotation-only inverse view-projection at every height (the full one lost the view directions in float rounding well below 2 km: bands in the sky); compose keeps the full matrix only for the terrain brush ring and the underwater view. Water: style 2 ("lake") runs the river branch (`water_style >= 1`) - the lake branch below it is never reached; left so because the look in use was tuned that way.

## Private state (2026-11-22)

`vk_engine.h` used to declare every subsystem's variables, so touching one rebuilt all 61 engine files. The variables of 25 subsystems now live in `src/engine/vk_engine/state/<name>.h`, each as a nested struct (`struct VulkanEngine::BloomState { ... }`) that `vk_engine.h` only forward-declares and points to: `bloom_`, `fsr3_`, `env_`, `sun_`, `tr_`, `mp_`, `pg_`, `cl_`, `scl_`, `ssdm_`, `rr_`, `sl_`, `atmo_`, `fog_`, `svgf_`, `taa_`, `taau_`, `voxel_`, `fx_`, `clutter_`, `cloud_`, `compose_`, `spacejet_`, `dlss_`, `fsr2_`.

- A variable that was `bloom_extent_` is now `bloom_->extent`. A file that uses it includes `engine/vk_engine/state/bloom.h`; a change to that header rebuilds only those files (2 to 12 of them, by subsystem).
- The structs are created in the constructor and freed in the destructor (`private_state.cpp`, which lists them all - add a new one there).
- New subsystem state goes into a state header, not into `vk_engine.h`. Still in the big header: the vegetation, the terrain, the player's body, deferred lighting and everything without a clear prefix (about 600 variables).
- Member functions are still declared in `vk_engine.h`.

## The terrain shader's main() (2026-11-22)

`terrain_raymarch.frag` holds the set-up and the march; the rest of its `main()` is in five files that are included, in this order, in the middle of it: `terrain_rm_main_water.glsl`, `terrain_rm_main_ground.glsl`, `terrain_rm_main_light.glsl`, `terrain_rm_main_gi.glsl`, `terrain_rm_main_fog.glsl`. They are not functions - they share main()'s local variables - so they cannot be included anywhere else.

## Names (2026-11-22)

The code's namespace is `vibe`, the libraries `vibe_core` / `vibe_anim` / `vibe_engine`, and every environment variable and shader define starts with `VIBE_` (they were `qlike` / `QLIKE_`). Only the repository's folder is still called `quake-like`.

## Sites: the hand-made places (2026-11-22)

The planet is procedural everywhere; a **site** is a spot where something was built on it. Home (the castle) is site 0 and is special - its content is in code and at the engine's origin. Every other site is **data**, and the design is meant for thousands of them, fetched from a server.

**Where the data comes from** - `src/engine/site_store.h` (the formats are documented there):
- `index.txt`: one line per site - id, place on the planet (km east / south of home), the shape of its ground, its name. Read whole at start-up (about 60 bytes a site).
- `<id>.site`: what stands there (towers, houses, a chapel, lanterns, a road). Read when the player comes near.
- Both are looked for in `assets/sites/` (shipped) and then `cache/sites/` (fetched). **A server client plugs in without touching the engine:** it sets `SiteStore::request`, which is called when a site file is wanted and missing, and puts `<id>.site` (or a newer `index.txt`) into `cache/sites/`. The engine looks again every 3 s; until then the place is plain land. Nothing in the engine does networking yet.

**Why thousands cost nothing** - `src/engine/vk_engine/sites.cpp`:
- **One slot.** The engine holds one site's content at a time. Every site has a chart of its own (`engine/globe.h`) whose origin lies 12,288 m north of it, so the content of whichever site is loaded sits at the same engine coordinates, (0, 12288). Loaded within 9 km (`load_site`, about 0.5 s for Seawatch, on the main thread), dropped beyond 14 km (`unload_site`); the old site's GPU resources are freed a dozen frames later and the next site is not built before that. Drawn, traced and collided with only while the engine is in that site's chart (`site_here()`, `mesh_parts_now()`).
- **The landscape knows shapes, not content.** A site's ground is part of the landscape formula: level at `rise` above the water within `flat` of its centre, meeting the natural ground at `slope`; no woods within `clearing`. CPU: `terrain.cpp` `site_ground`, over `globe::sites_nearby()` - the sites within 300 km, republished when the set changes. Shader: `rm_site_ground` in `terrain_rm_noise_road.glsl`, fed with the nearest site (`scene.site_ground`, `scene.site_shape`).
- **Per frame** the whole index is scanned once (`update_sites`): one dot product a site. Measured with 5,002 sites: start-up unchanged, 60 fps in orbit.

**Small coordinates on purpose.** A wandering far chart puts the player near x = 64,000, where a float resolves 4 to 8 mm: raymarched grass turns blocky and close-up ground textures streak (still so on the planet's land away from home and the sites). At 12 km it is 1 mm. Because (0, 12288) is also a place in home's chart (12 km south of the castle, another part of the planet), site content exists only in its own chart; home's content needs no such test, a site's chart being left 5.4 km from the site.

**Also:** one road at a time (`descriptors.cpp` uploads home's, the site's, or none); grass is cut on the street and inside buildings through the alpha of the near-field bake (`veg_stream.cpp`); from high up `compose.frag` `planet_surface` paints the nearest site and a point of light at night, and `site_labels_hud()` names the ten nearest above 1.5 km (setting `site_labels`); menu > Game > Travel lists home and the seven nearest.

**Test hooks:** `VIBE_TEST_SITE=<site>[,east,south,yaw,pitch,height]` starts there (a negative number arrives through a wandering chart); `VIBE_TEST_SITE_HOP=<frame>,<site>` travels on at that frame (one site out of the slot, the next in). Run a hop with `VIBE_VK_VALIDATION=1` after touching load / unload.

**What a site cannot have yet** (each is a known next step, not a dead end):
- object kinds beyond tower / house / chapel / lamp - and no custom meshes: a `mesh <file>` kind, with the mesh fetched like the site file, is the natural addition (unknown kinds are skipped, so old builds stay compatible);
- a sculpted height tile (the ground is the formula's flat top), a physics heightfield (rigid bodies have no ground away from home), placed vegetation;
- loading in the background (it is a half-second stop on the main thread);
- sites nearer than about 30 km to each other (one slot, one shape in the shader), or within 150 km of home;
- the shader and the far field know the nearest site only.

## The launch operations pavilion, and glass (2026-11-22)

The office north of the launch pad - a steel-framed hall with a sawtooth roof, a glass front to the rocket, a mezzanine, desks, a meeting room, a kitchen - is modelled by a Blender script, like the launch site itself:

```
blender -b --python tools/office/make_office.py      ->  assets/office/office.mesh, office.col, office.blend
```

- `office.mesh`: one group per material (`o_*`, the table in `rocket.cpp` `kMats`). `office.col`: the colliders (`box`) and the room lights (`light`), written by the same script - nothing about the building is typed into C++ by hand. `place_office()` in `rocket.cpp` puts it 86 m north of the pad's centre, on the level launch terrace, as part of home's content (`world_.meshes`, `world_.aabbs`, `lamp_lights_`).
- Change the building: edit the script, run it, restart the game. The measures at the top of `build()` (bay, eave, ridge, mezzanine height) drive everything.
- **Glass** is material `-3` (`o_glass`, the jeep's `j_glass`). It is in no depth, shadow or G-buffer pass and not in the ray-tracing structure (the sun and the bounce light go through it). It is drawn after the lighting - see "Glass" below.
- **Lit by day.** Emissive parts dim to 12 % by day (lantern glass). A `LevelMesh` with `lit_by_day` does not (`emissive.a = 2`): the pavilion's screens and room lights. Its room lights are also real lights by day (`descriptors.cpp`, `office_lights_`), and its ceiling, walls and floor carry a little light of their own - a deep room is darker in this renderer than in life.
- Beyond the home tile grass comes from the near-field bake; `home_grass_cuts_` keeps it out of the building and off its terrace (`veg_stream.cpp`).

**Its materials (2026-11-22).** Texture slots 25..29 (`kTexOffice`; `kTextureCount` is now 30, in `vk_engine.h` and `cube.frag`) hold the pavilion's five surfaces: weathering steel, oak boards, polished concrete, board-marked concrete, woven fabric - `assets/office/tex/*.png`, 2160 px, generated with an image model and upscaled (the 1024 px originals are in `assets/office/src/`). `load_foliage_textures()` in `world_textures.cpp` loads them and derives each normal map and height from the picture's own light and dark. The materials that use them are the `o_*` rows of `kMats` in `rocket.cpp` (slot, tint, metres per repeat).

**Rain stays out (2026-11-22).** A *shelter* is a roofed box: a rectangle on the ground plan and the roof's height (`VulkanEngine::Shelter`; `shelters_` for home, `site_shelters_` for the loaded site). The pavilion's hall and every house register one. `fx.cpp` leaves out the rain and snow streaks that would fall inside one; the four nearest the camera go to the shaders (`scene.shelter_a / shelter_b`), where `scene_sheltered()` keeps surfaces inside dry and free of snow (`deferred_lighting.frag`, `snow_cover`). The castle's rooms, the chapel and the towers do not register one yet.

**Its furniture (2026-11-22).** `tools/office/make_office.py` builds every object from a few helpers - `chair`, `monitor`, `bench_desk`, `sofa`, `planter` (with `leaf`), `books`, `laptop`, `mug`, and `obox` for a box turned about the vertical. Small disorder (chair angles, mugs, books) comes from one seeded `RNG`, so every run gives the same model. Rule kept when placing things: nothing crosses the back wall's frame columns (x = -18, -12, -6, 0, 6, 12) - screens, kitchen units and shelves are split into runs between them, and the stair stands 15 cm clear.

## Highways: roads from a line (2026-11-22)

A highway is a **centre line** - points 20 m apart, each with the height of the road's surface. Everything else is made from the line by rule. `src/engine/highway.h` (read its header first) / `highway.cpp`.

- **Where the lines come from.** `highway::build()` (called from `world.cpp` once the water level is known) makes the one there is today, the **coast road**: it traces the contour of the landscape formula's land / sea value just above the waterline of the sea east of home (`world_continent_home`), 13 km each way from the launch pad, smooths it into a highway's curves (coves and headlands shorter than a few hundred metres are cut across) and gives it the height of the land under it: the natural ground with only car-length bumps smoothed off, never lower than 3 m over the water (coves are crossed on fill) and never steeper than 10 % along the road (`kGrade`; where the land is steeper the road climbs at that limit, cutting the top of the rise and filling its foot by equal measure), with crests and dips rounded. It runs at the pad's own level where it passes the pad's seaward edge. (Until 2026-11-22 the height was averaged over half a kilometre and held 8 - 15 m over the water: the road trenched 30 m deep through every headland.) A line from data (a file, a server) goes into the same table with `highway::add_line`. A spur to the pad is made only when the road passes more than 160 m from it (it does not).
- **The ground** is a shelf along the line: level to 8 m from the centre, met by the natural ground at 30 m - a cutting through a hill, fill over a hollow or a cove. CPU: `highway::ground()`, applied in `world_ground_height` (terrain.cpp), so physics, the near-field bake and everything placed on the land have it; `world_ground_height_natural` is the ground without it. Shader: `rm_highway` in `terrain_rm_noise_road.glsl`, applied in `rm_world_h`. Keep the two in step.
- **The field** is how the shader knows the lines without searching them: `highway::bake()` writes, for every 4 m of a 4096 m window round the camera, the signed distance to the nearest line, the road's height there and the metres along it (two 1024 x 1024 RGBA8 textures, normal-texture slots 30 and 31 = `kTexRoadField`). All three are linear across a straight stretch, so the bilinear read is exact - which is why markings 15 cm wide are sharp from a 4 m grid. `vk_engine/highway_field.cpp`: baked again when the camera has moved 600 m (`hw_poll`, main thread), uploaded with the frame (`record_hw_upload`), window and extent in `scene.hw_info` / `scene.hw_bbox`. Home's chart only, and not above 3 km.
- **The surface** (`terrain_rm_main_ground.glsl` `hw_*`, `terrain_rm_main_light.glsl`): asphalt to 5.5 m with wheel tracks, aggregate and sealed cracks, a broken centre line (3 m of every 12) and solid edge lines 3.75 m out, a gravel verge to 8 m. The close ground relief is off on the asphalt and grows back over the verge (`rm_ground_detail`; CPU twin `raymarch_ground_relief`).
- **What does not grow there:** woods (`world_forest`), grass (the near-field bake's alpha, `veg_stream.cpp`).
- **Cost** (RX 9070 XT, 2560x1440): nothing in the three benchmark views; standing on the road 11.8 ms on the open coast, 23.6 ms in its wooded stretch.
- **Not there yet:** the road is not drawn beyond its 2 km window (nor from orbit); it crosses the river's mouth and the coves as a solid causeway (no bridges); no rails, lamps or signs; the field is not baked in a site's chart; `water_level` changed at run time does not rebuild the line. The log prints where it runs at start-up (`highway: +250 m along: ...`).

**The road surface, made to look used (2026-11-22).** Scanned materials: asphalt `Asphalt031` (slot 32, `kTexAsphalt`) and roadside gravel `Gravel040` (slot 33, `kTexGravel`), both CC0 from ambientCG in `assets/terrain_pbr/`, loaded by `load_terrain_pbr_overrides`; `kTextureCount` is 34 (`vk_engine.h`, `cube.frag`, `kRmTextureCount` in `terrain_raymarch.frag`). `rm_hw_asphalt` (end of `terrain_rm_noise_road.glsl`) works in road coordinates (metres along, metres from the centre line): stretches aged differently, two polished wheel tracks a lane with an oil strip between, rectangular repair patches with tarred joints, sealed cracks across and along, dust toward the edge, thermoplastic markings chipped and worn. The relief is the scans' normal maps (`terrain_rm_main_ground.glsl`); the edge of the asphalt wanders and breaks. Sheen (`terrain_rm_main_gi.glsl`, after `col`): a sun highlight and sky reflection that grow toward grazing angles, stronger on the tracks and the tar. Rain puddles come from the terrain's own wet-ground code.

**Beside the road: streamed** (`vk_engine/highway_furniture.cpp`). A W-beam guard rail on posts every 4 m along the seaward side and reflector posts every 48 m on the landward side. Nothing of it exists at start-up. The road is cut into 300 m stretches; `hw_objects_poll` (once a frame, `game_loop.cpp`) makes a stretch's meshes and ray-tracing structures from the line when the camera comes within 1500 m of its middle - the nearest missing one, one a frame - and frees it beyond 1900 m, or when home's chart is left. Freed parts wait 12 frames before they are destroyed (frames in flight may still draw them) and nothing new is made while any wait - the rule a site's content follows (`sites.cpp`); `hw_gen_` is part of the TLAS signature (`rt.cpp`), and the parts are drawn through `mesh_parts_now()`. What is kept for the whole road is one small record per post position (`g_st`, made on first use). A new kind of roadside object goes into `build_stretch()`. Measured riding 9.6 km at 240 m/s (`VIBE_TEST_HIGHWAY_RIDE=240`, `VIBE_LOG_HIGHWAY=1`): 11 - 12 stretches alive at any time, no errors. No colliders yet.

Cost with all of it (RX 9070 XT, 2560x1440): the three benchmark views 24.7 / 20.9 / 28.0 ms (24.3 / 20.6 / 28.7 before the road; within run-to-run spread except about +0.3 ms in the terrain pass); on the road 12.1 ms on the open coast, 23.4 ms in the wooded stretch.

**Bends (2026-11-22).** The plan of the coast road has a least radius (`kMinRadius` 160 m in `highway::build`: a part that turns harder is relaxed until it does not; the tightest bend of the route is 185 m, logged at start-up) and the line handed on is a Catmull-Rom curve through the 20 m points in 4 m pieces - with 20 m pieces the lane lines and the rail had a corner every 20 m. A post's direction is taken from the posts either side of it. There is no banking in bends.

## Small coordinates everywhere (2026-11-22)

A float resolves 4 - 8 mm at 64 km and 1 mm at 12 km. Grass blades need the millimetre: at 64 km they tear into sheared blocks (seen in every wandering chart, whose origin sat at local x = 64,000, and in home's chart out to its 100 km reach). Now the engine's coordinates are small near the ground everywhere on the planet: the same few kilometres of numbers are used again and again, each time for another piece of it. `engine/globe.h` ("Small coordinates", "Home's landscape from another chart") is the reference; in short:

- **Which chart** (`globe_update` in `rocket.cpp`). Home's chart: within 12 km of home (`kHomeNear`, back at 9 km), and further out only above 900 m over the water, where precision does not show. Near the ground elsewhere: a **near chart** (`globe::near_chart`) - its origin 12,288 m north of the player, so the player starts at local (0, 12288) and is never further than 4 km from there (`kNearReach`): coordinates stay under 16.4 km. (Not round (0, 0): home's content lives at those numbers in every chart; a site's chart keeps clear the same way.) High up (from 1800 m, back at 900 m) and beyond 92 km from home: the old kind (`high_chart`, origin at local x = 64,000, re-anchored every 16 km), because re-anchoring every 4 km at flying speed would cost more than it gives. `Chart::high` tells the two apart. Home's own mesh parts are drawn in home's chart only (`mesh_parts_now`).
- **The same land from any chart.** Within 90 km of home the land is the flat formula, a function of home coordinates. From a near chart those are never formed in a float: `globe::Ring` gives home = h0 + J (p - anchor); the FBM runs on J (p - anchor) and gets h0's share per octave as a lattice cell and a position in it (`globe::flat_lattice` -> `scene.ring_cell[24]`; shader `rm_noised_i`, `rm_flat_p0`, `rm_ring_hp` in `terrain_rm_noise_road.glsl`; CPU `flat_fbm_d` and `home_coords` in `terrain.cpp`, in doubles). Measured at four places: the ground's height differs by 7 mm or less between the two ways. The slow parts (continents, ranges, woods) take home coordinates from the same mapping.
- **The same trees from any chart** (`veg_stream.cpp`). The streamed woods' 128 m cells are cut out of home coordinates (`globe::home_az`), not the chart's: a cell is a piece of the planet. `generate_cell` lays it into the current chart as a parallelogram; far round the planet, where home coordinates stretch, fewer candidates are kept by the same measure. When the engine changes chart, the loaded cells are carried over (`wv_rechart`) instead of being made again. A cell a generator made across a change of chart is dropped (`globe::chart_serial`).
- **Highways from any chart.** Lines stay in home coordinates; the field and the roadside objects are carried into the chart through the ring's mapping (`highway::bake(..., ring)`, `build_stretch`), CPU queries through `home_coords`. On a change of chart the field is baked again and the stretches are made again.
- **Checks made:** the same road spot and the same woodland spot photographed in home's chart and in a near chart (`VIBE_TEST_NEAR_CHART=1` leaves home's chart 3 km out) - road, rail, grass cut and every tree in the same place; a ride along the coast road across the 12 km hand-over (`VIBE_TEST_HIGHWAY_RIDE=120,9000`) without errors; frame time in the three benchmark views unchanged (24.6 / 20.8 / 27.4 ms).
- **Known limits:** grass blades and small ground detail are laid out in chart coordinates and re-arrange when the chart changes (the land, the trees, the road do not); at the change the woods are put back from the carried cells over a few frames; between 12 and 100 km from home at more than 900 m up the coordinates are still home's large ones (by design); the mapping used by the shader is linear (under 2 cm off within a near chart's reach); woods near the antipode of home are thin (the home map is singular there).

**Tiles, and work from the camera (2026-11-22, later).** What the large engines do about precision is one of two things: shift the world's origin now and then (UE4, Kerbal), or keep positions in 64 bits and hand the GPU small offsets - from the camera, or from a tile (UE5's large-world coordinates, Star Citizen, Flight Simulator). This engine now does the second in the form that fits a ray-marched, formula-made world:
- **A near chart's coordinates are the old chart's less a whole number of tiles** (`kTile` = 6400 m; `globe::near_chart`, `Chart::org`, `Chart::tile`). The land round the player keeps its coordinates exactly, up to that shift - from home's chart into a near chart, and from one near chart to the next (checked: the same spot is at z = -5080 in home's chart and z = 14120 in a near chart, x unchanged). The chart is unrolled round the player (`org` is where its origin lies in its own coordinates); it is re-anchored when the player is 3600 m from (0, 12288) along either axis.
- **So whatever repeats once a tile is the same before and after a re-centre.** `scene_ubo.glsl` has the tools: `TILE_CELL` / `TILE_N` (a cell size snapped to fit a tile a whole number of times), `TILE_FREQ`, `TILE_W` (the same for a frequency), `tile_wrap` (a cell's number wrapped at the tile before it is hashed), `tile_noise2(wp, k, off)` (value noise, the same in every tile). Converted: the grass's blade cells, patches, type field and gusts (`grass_raymarch.frag`, `grass_blades.glsl`); 71 fine-noise calls on a world position in the terrain shader's material, light and pebble code; the highway's textures. NOT converted (these still re-arrange at a re-centre): the close ground relief `rm_ground_detail` and the home road's sand / snow patches (they have twins on the CPU), the scanned ground textures' tile sizes, the water's noise, and noise calls written in other forms than `noise2(pos.xz * k ...)`. Use the tile tools for anything new that is drawn from world coordinates at a scale of metres or less.
- **The grass's blade lattice is worked out from the camera** (`scene.grass_cam`, filled in doubles in `descriptors.cpp`; `g_rel_xz` in `grass_raymarch.frag` `map`): the camera's cell and its place in it come from the CPU, the march's offset from the camera is a small number.
- **Cost:** the three benchmark views 24.8 / 20.9 / 27.5 ms (24.6 / 20.8 / 27.4 before the tile work).
- **The blocky grass, found (2026-11-22).** Far from the origin the grass came out as blocks a few pixels across instead of blades - in home's own chart 10 km out as much as anywhere else. The cause was neither the blade lattice nor the size of the coordinates as such: `grass_raymarch.frag` took its view ray as `inverse(view_proj) x ndc - camera`, with the point on the near plane, 5 cm from the eye. That sum puts the camera's coordinates in and takes them out again; at 10 km (a float resolves 1 mm) the direction was left with two digits, and neighbouring pixels got the same ray. (Found with debug views of the ray direction and the hit distance; ruled out on the way: the blade lattice, the low-resolution pass's start distance, the grass's shadows, the upscaler.) Now `scene_ray(ndc)` in `scene_ubo.glsl`, from `scene.ray_mat` = inverse(projection x the view's rotation), worked out in doubles with no position in it. The terrain pass always had its rays that way (`pc_data.inv_view_proj` in `frame.cpp`).

**Rules, so that this stays fixed** (the tests in `tests/test_globe.cpp` hold the chart side of them):
1. **A view ray comes from `scene_ray()`** (or a matrix without the camera's position in it) - never from a world position less the camera's.
2. **Nothing may assume the engine's coordinates are a place.** They are a place less some tiles, in a chart that changes. Content with a place of its own keeps it in home coordinates or as a planet point (the woods' cells, the highways' lines, a site's `G`) and is carried into the chart: `globe::Ring`, `key_to_chart`, `to_chart(to_planet(...))`.
3. **What is laid out in chart coordinates is made again, or carried over, when `globe::chart_serial()` changes** (the highway's field and stretches, the woods through `wv_rechart`).
4. **Fine patterns worked out from a position repeat once a tile**: the `TILE_*` tools and `tile_noise2` in `scene_ubo.glsl`.
5. **Things thinner than a few millimetres are worked out from the camera outward** (the grass's blade lattice: `scene.grass_cam`), with the camera's part done in doubles on the CPU.
6. **The landscape's own formulas take a lattice cell and a position in it per octave** (`globe::lattice`, `globe::flat_lattice`), never a large coordinate times a frequency.
7. **Check a precision fix at full resolution, in daylight, close to the ground, with the same place photographed in home's chart and in a near chart** (`VIBE_TEST_NEAR_CHART=1`; `PrShot10.cmd` on the test PC). A small dim capture hid this fault once.

**Rule 4, state (2026-11-22, evening).** Converted since: the close ground relief (`rm_ground_detail`, with its twins `grass_ground_relief` in the grass shader and `raymarch_ground_relief` on the CPU), the stones (`peb_layer`, `peb_vnoise`), the turf's green and brown patches and the rock's grain (noise on a turned lattice: `tile_noise2_rot`, which turns by the angle of a whole-sided triangle - the only angles at which a tile's shift lands on whole cells), the red sand's patches, the snow's and the sand's grain. The scanned ground textures needed nothing: their repeats (0.45, 0.30, 0.55, 1.9 per metre) already fit 6400 m a whole number of times, as does any frequency with two decimals. The measure: the same ground photographed in home's chart and in a near chart (`PrShot11.cmd`) - the same blades, stones and patches; mean difference 7 grey levels against 2 between two runs in one chart, the rest being the finest grain, which is a little softer where coordinates are 14 km (rule 5 would cure that for the ground as it did for the grass). Not converted: the home road's cobble warp and its sand / snow patches (home only, twins on the CPU), the water's noise, craters, ice.

Stray trees: the woods keep a few lone trees in open country whatever the forest rule says; nothing is planted on a highway or on a site's own ground any more (`world_highway_distance`, `world_site_clearing`).

Cost of all of it: the three benchmark views 25.1 / 20.9 / 27.6 ms (24.6 / 20.8 / 27.4 before the tile work).

**A re-centre is not an event any more (2026-11-22, night).** When the engine shifts its coordinates by whole tiles (`globe_enter` between home's chart and near charts), nothing is torn down and rebuilt; everything is moved in the same step:
- *The renderer's memory of the last frame* - the last view-projection (motion vectors, every temporal history), the vegetation's Hi-Z view, footprints, ripples: `globe_shift_history` (`rocket.cpp`). Before this the frame of the shift came out dark and smeared: every history was reprojected 6 km off.
- *The streamed woods*: `wv_rechart` rewrites every live tree's GPU slot in place (it used to empty the pool and refill it over the next frames), and forgets the camera's last position (the jump read as a speed of hundreds of km/s, and the woods were given up as "outrun").
- *The near-field ground bake* the grass stands on: its origin texels are rewritten and it stays in use (`wv_rechart`).
- *The highway's field*: its window is moved (`hw_shift`) - what it holds does not depend on coordinates.
- *The roadside stretches*: `hw_objects_shift` adds the shift to each part's transform. A mesh part has a transform now (`CastleMeshGpu::xf`: drawn with it, in the TLAS with it; identity for everything else), and a stretch's vertices are kept relative to a point of its own, placed by `xf` - the same path whether or not the engine has re-centred.
Checked on the test PC by capturing the frames either side of both kinds of shift (`RideTest4.cmd`: home's chart to a near chart at frame 951, near chart to near chart at 1778): road, rail, trees and grass carry through; `VIBE_LOG_HIGHWAY=1` logs each shift with its frame number. A change to or from a site's chart or a high one is still a rebuild (those are not shifts).

**Known, not from this work:** the guard rail's steel flips between bright and dark from place to place. It is seen nearly edge-on along the road, and the lighting pass's ray origins - rebuilt from depth and pushed toward the camera (`wp_safe`) - do not clear a surface seen at that angle, so its own beam shadows it or not by rounding. Needs an offset along the surface normal for thin parts.

Cost after all of it: the three benchmark views 25.1 / 21.0 / 27.8 ms.

**Flicker while walking: rays that start inside their own surface (2026-11-22, night).** Three complaints, one family of causes; all in the lighting pass's rays (`deferred_lighting.frag`) or the glass.
- *Walls, the launch tower, the rocket's concrete flickered light and dark in the pattern of their texture.* A wall with relief is drawn with its depth pushed into the stone (`cube.frag`), up to some 20 cm behind the face the ray tracing knows, by an amount that follows the texture and changes with the view. Rays from there - ambient occlusion first, the sun's shadow and bounce light too - met that face from behind and took it for an occluder. Now a ray from a solid receiver that meets a surface from behind within 30 cm is carried on from it (`g_rq_from_solid`, `any_hit_once`, `closest_hit_once`: `rayQueryGetIntersectionFrontFaceEXT`). Measured on the tower's side while walking past (mean change between two frames, 0 - 255): 10.4 -> 2.3, what it is with ambient occlusion off.
- *The geometric normal* (`Ng`, from the depth buffer's neighbours): those rays leave from it and fan out round it, not round the normal-mapped `N`, whose fan dips under the real surface. The guard rail's steel no longer flips between bright and dark from place to place.
- *Ambient occlusion has fixed directions*, the same for every pixel and frame, like the terrain's sky probe: the answer depends on the place alone. (Under Ray Reconstruction it used to take 2 random taps; it takes the 4 fixed ones now.)
- *Everything behind glass flickered, the monitors first.* Glass is drawn as a part of its pixels; the pattern changed every frame for the temporal pass to blend - neither upscaler did. The pattern is fixed on the screen now and sparser head-on (5 %, was 13 %). A stipple is still seen in panes at a grazing angle: real blending needs a pass of its own for glass.
- The model's screens stand 12 mm proud of their panels (were 1.5 - 5 mm: depth fighting from across the hall).
How it was found, for the next time: `VIBE_TEST_DOLLY=<vx>,<vy>,<vz>` moves the held camera at walking speed; capture two consecutive frames (`--screenshot-after 300` and `301`), difference them (a still camera's pair is the noise floor), then switch the lighting's parts off one at a time in the cfg (`FrShot.cmd`, `FqShot.cmd` on the test PC). On that PC a killed run leaves its batch file running: end the `cmd.exe` too, or two runs overlap and the captures are worthless.
Cost: none measured (benchmark 25.3 / 21.3 / 27.7 ms).

## Glass (2026-11-22)

`shaders/glass.vert/.frag`, drawn in the FX pass (`fx.cpp`, before smoke and light ribbons).

- Before the pass the lit frame is copied (`scene_copy_image_`, binding 38) - only when a pane is in view. The pane's shader reads what is behind it from that copy, looked up a little to the side (the refracted line of sight), tints it faintly green, and lays the sky's and the sun's reflection over it by Fresnel (4 % head-on, a mirror at a grazing angle).
- Opaque to the blender, depth-tested, no depth write. Parts are sorted far to near; two panes of ONE part behind each other are not sorted (the nearer one's reflection can be lost where they overlap).
- It replaced a pattern of opaque dots in `cube.frag`, which neither upscaler blended (boiling panes, flickering monitors).
- New glass: give a static-mesh part material `-3`. Nothing else.

## The launch pad's lights (2026-11-22)

`tools/rocket/make_rocket.py` writes `assets/rocket/site.lights` (`kind x y z dx dy dz`): flood lights on the masts, steady red lights round the apron and on the tanks, flashing red ones on the tower. `rocket.cpp` reads them into `pad_lights_`; `fx.cpp` draws them: a flare (two billboards and a streak) and, at night, a soft beam for a flood; a glow for a red one (flashing: 1.5 s). Night is `env_->night` - `rt_.sun_pitch_deg` is not valid at night.

## The course (2026-11-22)

`highway::build` no longer follows the coast only: it searches a closed course of about 12 km round home (`highway.cpp`, a dynamic program over 720 bearings x 40 radii; water, side slope and height cost, the shore is cheap), smooths it, and pushes it clear of the launch pad's apron (`clear_pad`, 96 m). The road is not flat: `along_noise` (0.11 m over 47 m, 0.035 m over 13 m) and a crown (`cross_fall`: 2 % on the asphalt, 5 % on the verge) - the same functions on the CPU (`highway::ground`) and in the shader (`rm_highway`), from the position along the line only, so it is the same every run. The guard rail is material `-5` (galvanised: dull).

## The jeep (2026-11-22)

`vk_engine/jeep.cpp`, `PhysicsWorld::vehicle_*` (`physics_world.cpp`), `tools/jeep/make_jeep.py` -> `assets/jeep/` (`body.mesh`, `wheel.mesh`, `steering.mesh`, `jeep.json`: wheel places, seat, steering hub, collision boxes; `tex/paint.png`, `tex/canvas.png` in texture slots 34 / 35).

- **Physics**: Jolt's `VehicleConstraint` with a wheeled controller: 1.25 t, four sprung wheels (cast cylinders), engine with a torque curve, clutch, four gears and reverse, a differential per axle (40 / 60), anti-roll bars. An ordinary rigid body: it rolls, and stops against what it hits. Front is -z; wheels 0 / 1 front left / right, 2 / 3 rear.
- **The world for it**: Jolt has a heightfield for the home tile only. Round the jeep it gets a ground patch (128 x 128 at 2 m, from `world_ground_height` on a worker thread, renewed 48 m from its centre; a hole where the home tile's own heightfield is) and the obstacles within 70 m as boxes: `wv_cols_` (trunks), `world_.aabbs`, the guard rail (`hw_rail_boxes`). New kinds of solid things must be added in `jeep_feed_world`.
- **Drawing**: through the static-mesh path, `jeep_draw_` (the wheel four times), each part with `xf` and `xf_prev` (`CastleMeshGpu::xf_prev`: motion vectors for moving parts - keep it equal to `xf` for what stands still). `jeep_signature()` is in the ray-tracing structure's signature.
- **Driving**: Use gets in (within 3.6 m) and out. Forward / back: throttle, brake, then reverse. Steering is rate-limited and lighter at speed. Jump: handbrake. On its roof and at rest for 2.5 s it is put back on its wheels. The weapon is hidden. While driving the engine stays in home's chart (`rocket.cpp`): the vehicle's physics is in home's coordinates. Home's chart only, for now.
- **Gravity is 25 m/s2 here, not 9.81** (`physics_world.cpp`, for the feel of the player's jumps). The vehicle falls at 9.81 (`mGravityFactor`): built from real numbers - mass, springs, tyres, engine - it only behaves like a car under the Earth's gravity. At 25 it sat on its bump stops, cornered at 2.5 g and stalled on the slightest hill. The springs are sized from the weight per axle (not Jolt's frequency setting, which uses a much smaller mass). Tyres let go past about 6 degrees of slip and hold about half when sliding; the rear a little less than the front.
- **Camera**: stays behind the jeep by itself and turns with it a little late (`cam_yaw`). Holding the right mouse button the mouse looks round (`orbit_yaw / orbit_pitch`); released, the view returns. The player's own body is not drawn nor ray traced while driving (`body_rt.cpp`). It is parked on the concrete in front of the pavilion.
- **Loop order** (`vk_engine.cpp`): `jeep_tick` (controls; clears the player's inputs) -> `loop_simulate` (physics) -> `jeep_after_sim` (parts, camera).
- **Tests**: `VIBE_TEST_JEEP=<throttle>,<steer>` drives it (steer `9`: follows the course at throttle x 30 m/s), `VIBE_TEST_JEEP_AT=<x>,<z>,<yaw>` places it, `VIBE_LOG_JEEP=1` logs place, speed, gear and wheel contact once a second. Measured: 12 km course at 80-88 km/h, four wheels down throughout.

## Small things (2026-11-22)

- Aiming slows the player to 55 % (`player.cpp`, `aiming`).
- The laser and the muzzle flash start at the barrel's own muzzle (`viewmodel_muzzle_`, found from the weapon's vertices in `props.cpp`); the bolt is drawn from 1 m in front of it.
- Anti-air vehicles keep 45 m from the chapel (`aa.cpp`, `by_chapel`).
- Grass: a ray that looks nearly along the ground took too-small steps above the blade tops and gave up in the middle distance (a band without grass). It now advances by its height above the tops (`grass_raymarch.frag`, `g_above_top`).

## Cars: more than one, marks, smoke, the driver (2026-11-22)

Additions to "The jeep" above (`jeep.cpp`; the type is still called `JeepState`, one per car).

- **Several cars** (`jeep_all_`), one simulated: `jeep_` is the car last got into and the only one in the physics. The others stand where they were left (`jeep_park` / their last state), drawn but not simulated and not solid. Use at another car removes the vehicle from the physics and makes it again for that car where it stands. A new car: a folder under `assets/` with `body.mesh`, `wheel.mesh`, `steering.mesh` and a json like `jeep.json`, a line in `init_jeep` with its `VehicleDesc` (mass, centre of mass, drive, gears, engine, springs, grip) and its materials in `kJMats` / `kSMats`.
- **Centre of mass** is set (`com_height`, `front_weight`; `OffsetCenterOfMassShape`), not left to the collision boxes - theirs is a solid block's, at window height: the car rolled over about the wrong point and far too easily. A box over each axle out to the tyres' outer faces is added, so on its side it lies on its wheels.
- **Tyre marks and smoke** (`jeep_marks`, `jeep_fx`): per wheel, sliding = sideways speed of the contact point, or wheel speed against ground speed (spin, locked). On a hard surface (the road, a slab) it lays a mark - points along the ground, drawn as flat strips (`fx_ribbon.frag` kind 6) - and smoke (mist billboards, kind 5); off-road only dust. 700 points per wheel are kept.
- **The driver**: the player's body is put in the seat (`jeep_driver` -> `player_body.cpp`: `body_model_` from the car's transform, hips on the seat) and posed by `CharacterAnimator::seated` - from the rest pose: thighs forward, shins down, back against the seat, wrists fitted to the rim, the hands going round with the wheel up to a quarter turn. The head is drawn (the camera is outside). `JeepState::hip`, `legs`, `rim_r` per car.
- **Tests**: `VIBE_TEST_CAR=<n>` which car is simulated first, `VIBE_TEST_JEEP_SKID=<s>` full lock and handbrake after s seconds, `VIBE_TEST_JEEP_ORBIT=<yaw>,<pitch>,<distance>` holds the view round the car.

## Round of 2026-11-22, evening: car park, bridge, lights, screens, quality levels

- **Car park** (`rocket.cpp place_parking`): an asphalt slab east of the pavilion, level with the apron it joins, bays (`parking_bays_`), a kerb, lamp posts, a ramp to the highway's edge. The level terrace round the pad was widened for it (`terrain.cpp pad` and `shaders/world_map.glsl wm_pad`: 150 m level, blended out by 270 m - keep the two in step).
- **Paved places and vehicles** (`drive_pads_`): rectangles with a height (a ramp when the two ends differ). The vehicle's ground patch is the higher of terrain and these, and round each the ground comes up to it over 1.5 m (a kerb a wheel rolls over). A floor under the wheels is never an obstacle box (`jeep_feed_world`). The apron is one of them.
- **Cars** (`init_jeep`): kinds in `assets/<kind>/` (jeep, sportscar, saloon, pickup; each from `tools/<kind>/make_*.py`), one or more colours each (`jeep_load(..., paint)`; material `-6` = glossy paint, `cube_surface_helpers.glsl`), each kind its own `VehicleDesc`. They stand in the car park's bays. `car.json` may give `hip_driver` and `legs`.
- **Car lights**: L toggles the head lights of the car driven. The beam is the hand torch's light (the scene's one cone light with its glow in the air: `jeep_headlights` -> `descriptors.cpp`), the lamps are emissive parts (`JeepState::kind`), flares in the FX pass (`jeep_fx`). Brake lights by the pedal or the handbrake.
- **The view from a car**: the middle mouse button steps near / medium / far (`cam_mode`).
- **Bridges** (`bridges.cpp`, `highway.cpp`): a stretch of the course whose natural ground is under water somewhere is a span (`g_spans`). No shelf is made there - `highway::ground` returns the natural ground, the baked field says "no road" - so the river reaches the sea, and `place_bridges` builds a box girder on piers with the road's surface, a walkway, railings and lamps. `highway::deck(x, z, h)` is its surface (the vehicle's ground patch uses it); for a walker it is boxes a metre long. Static meshes with the launch site's (the course lies in home's chart). `VIBE_NO_BRIDGES=1`: none.
- **Concrete without repetition**: the apron's texture carries only grain and joints; tone per slab, stains and cracks come from the place in the world (`cube.frag`, slot 24). Grass is cut on the whole apron square, not a circle (`home_grass_cuts_`).
- **Displays** (`cube.frag`, slot 36 `kTexScreens`, `assets/office/tex/screens.png`: 8 pictures, 4 x 2): `make_office.py screen()` gives every display uv = its number + 0..1; each shows one picture and changes every 16 s, not all at once (time = `scene.water_params.w`).
- **Quality levels** (`settings.cpp apply_foliage_quality` / `apply_shadow_quality`; first in the menu's Game tab; saved as `foliage_quality`, `shadow_quality`): foliage low .. insane sets densities, detail distance, plant and grass draw distances (and replants the forest); shadows low .. high sets the sun's rays per pixel, whether rays through trees test every leaf (`foliage_ray_mode`), screen-space instead of ray-traced ambient occlusion on low and medium, the terrain's shadow steps and the grass's shadow rays. Measured in the forest scene at 1440p on the AMD card: shadows 23.6 / 25.7 / 27.9 ms; foliage low .. insane only 28 .. 30 ms - fewer trees uncover more raymarched ground, which costs as much as the trees did. The terrain march is the floor there. `VIBE_TEST_FOLIAGE`, `VIBE_TEST_SHADOWS` force a level.
- **Memory**: `VIBE_LOG_MEM=1` logs video memory, RAM, allocations and streamed counts every 600 frames (`mem_log.cpp`). Measured over 34,000 frames of riding the course at 110 m/s: flat (3754 MB video, RAM steady, allocations 1601-1614). Never run two test batches on the test PC at once: the second seems to hang.

## Tyres and handling (2026-11-22, late)

What makes the cars drive like cars (`physics_world.cpp vehicle_create`, `jeep.cpp jeep_tick`), each found by measuring (`VIBE_TEST_JEEP_TURN=<s>`: full lock after s seconds, logs speed and sideways g):

- **The tyre's curves are the grip.** Jolt's default combines the tyre's friction with the friction of the body it stands on (made for solids touching); the home tile's heightfield has 0.2, which left under half the grip there. `SetCombineFriction` is a no-op now.
- **Curve shape**: steep rise, a peak (12 % slip along, 8 degrees across), then a gentle fall to about four fifths - a measured tyre's shape. A curve that falls to half past the peak makes the car plough straight on at full lock.
- **One grip per tyre** (`SetTireMaxImpulseCallback`): what a spinning or locked wheel spends along it has not across. Jolt limits the two separately.
- **The air** (`VehicleImpl::Aero`, a step listener): downforce and drag per car (`VehicleDesc::downforce`, `drag`), by speed squared.
- **Steering from a key** is limited to the lock the tyres can use at that speed: atan(wheelbase x grip acceleration / speed squared) + the slip angle of best grip. More scrubs the front tyres or trips a tall car over.
- **Traction control**: the throttle is eased while a driven wheel spins (not from a standstill, not with the handbrake).
- Measured after: sports car about 1.1 g at 60-75 km/h on full lock, jeep about 0.9 g, neither rolls; the course at 108 km/h on autopilot.

## Where the frame goes on the RTX 4080 (2026-11-23)

Measured with the sweep (`VIBE_PROF_SWEEP=150`, `gfx_quality.cpp prof_sweep_tick`: one run, the camera held, ~60 experiments that each switch one thing off or down, the GPU time of every pass logged for each). 3440x1440, DLSS Ray Reconstruction, render scale 0.5, the settings as played. Full tables: `docs/perf/sweep_2026-11-23_*.txt`. Run-to-run noise is about +-0.5 ms.

| view | frame | terrain pass | its first pass | vegetation | grass | lighting | upscale |
|---|---|---|---|---|---|---|---|
| forest by the course (outside the home tile) | 28.1 | 11.8 | 1.3 | 9.0 | 1.0 | 0.3 | 2.6 |
| river valley by the bridge (outside) | 23.6 | 15.4 | 0.8 | 0.4 | 1.8 | 0.4 | 2.9 |
| close ground at home | 32.1 | 16.1 | 0.5 | 4.5 | 4.3 | 0.8 | 2.7 |

What moves it (ms saved, forest / river / close):
- **Shading the terrain, not finding it**: the march alone is 4.3 / 3.7 / 6.1 of the pass; shading is the other 7.5 / 11.6 / 10.0.
- **Close relief** (stones, sand ripples, small bumps: in the march, the normal and the shadows): 4.8 / 4.2 / 5.0.
- **Coarser shading toward the screen's edges** (`terrain_vrs` 1): 3.7 / 3.5 / 5.2. Level 2 adds little.
- **Terrain at lower resolution** (`terrain_raymarch_scale`): 75 % 3.4 / 4.2 / 5.9; 50 % 7.8 / 9.9 / 11.3.
- **Tree detail distance** (`veg_lod_bias` 1.25 -> 0.6): 8.0 in the forest (vegetation 9.0 -> 0.5), nothing elsewhere.
- **Render scale** costs by the pixel, nearly all of the frame: 0.59 +5.9 / +5.7 / +7.1; 0.67 +11.9 / +11.0 / +15.1. Below 0.5 nothing changes (the upscaler's floor).
- The terrain's shadows together 1.8 / 2.3 / 4.8; its bounce light 0.8 / 1.1 / 2.4; grass 1.0 / 1.7 / about 4.
- Under 1 ms each, in every view: sun shadow rays, ambient occlusion, reflections, clouds, fog, bloom, water, step counts, octaves, plant and grass distances and densities. The first pass pays for itself (off: +1.8 / +2.9).

From reading the shader (not yet measured one by one): outside the home tile every height evaluation adds the continent's 21-29 noises to its own 32 hashes and has no baked height to skip along; the eye-in-ground test, the footprint bounds and the mean colours are the same for every pixel of a frame but worked out per pixel; the three normal taps recompute hashes, river and continent that they share; ground under dense grass is fully lit and then covered by the grass.

## Graphics quality rows (2026-11-23)

`gfx_quality.cpp`: 25 rows, each low / medium / high / ultra / insane, and "Everything" that sets them all; first in the settings window's Game tab; saved as `gfx_<key>` and `gfx_all`. `VIBE_TEST_GFX="all=2,forest=4"` sets levels by key at start-up.

Measured on the RTX 4080 at 3440x1440 with Ray Reconstruction (GPU ms a frame, before frame generation), everything at one level:

| view | low | medium | high | ultra | insane |
|---|---|---|---|---|---|
| open ground by the course | 9.2 | 11.6 | 20.4 | 33.5 | 46.9 |
| river valley | 9.9 | 11.1 | 19.4 | 28.5 | 41.7 |
| close ground at home | 11.2 | 15.0 | 22.7 | 40.9 | 61.2 |

Measured again after the trees' triangle rule and with the ground's shape out of the rows (2026-11-23, later): open ground 9.5 / 11.2 / 17.7 / 28.6 / 42.9; river 10.0 / 11.2 / 18.5 / 28.2 / 41.1; close 12.7 / 14.8 / 21.1 / 37.5 / 57.8.

What the levels are made of follows the sweep above: the terrain's resolution (half size at low and medium, three quarters at high), coarser shading toward the screen's edges, the relief kept out of the march at low and medium, the trees' detail distance, and the render scale above the upscaler's floor for ultra and insane. The other rows are worth under a millisecond each and are there for the picture.

Rules learnt making them:
- **Never make `terrain_raymarch_octaves` a quality choice.** It is the ground's shape; with fewer octaves the sand path showed its gravel bed, because the ground no longer lay where everything else (physics, roads, sand) has it. Shading detail is `terrain_raymarch_normal_octaves`.
- **Relief off entirely (`RMDBG` 1024) is not a level**: it changes materials. Low and medium use 128 (relief in the shading, not in the march).
- The render scale cannot go below 0.5 with DLSS: low, medium and high share it.
- Changing the terrain target's size before the targets exist (settings being loaded) crashes: `apply_gfx_row` only resizes once they do.

**Eye-in-ground test** (`terrain_rm_heightfield.glsl`, both marches, 2026-11-23): it evaluated the full height function at the camera for every pixel - the same answer each time, at the price of a march step. It now asks the baked height first (one fetch; the full surface is never more than about a metre above it) and only evaluates for an eye within 1.5 m of the bake. Close view at home: terrain pass 16.1 -> 15.0 ms, frame 32.1 -> 30.7. Nothing outside the home tile (the test is not made there).

**Trees: triangles per pixel** (`veg_cull.comp`, `vegetation.cpp`, 2026-11-23). In the open view by the course the trees filled 3 % of the screen and cost 9 ms (depth pass 3.7 + colour 5.3). The cause was geometry, not pixels: a tree line 200-400 m off was drawn as full meshes - about 20 000 triangles a tree on a crown 40 pixels across - and the occlusion pyramid cannot cull a forest's inside (leaves have holes). Two rules now:
- **The hand-over to the baked card is in pixels of the output** (`tree_impostor_px`, 7-64; the card's tile is 128 px, so it is as sharp as the mesh up to there). It used to be multiplied by the detail setting, so "more detail" also meant "cards later" and the two settings fought.
- **The thinning curve is `(px / 60)^1.4`** (was 0.8): half the size draws 38 % of the tree; the leaves that stay are drawn larger (`veg.vert`), so the crown keeps its cover. `veg_lod_bias` is limited to 0.4-1.5: above about 0.9 the picture did not change, the cost did (same view, hand-over 40 px: 0.6 -> 1.5 ms, 0.9 -> 2.7, the old 2.6 -> 7.2).

Same view after both: frame 28.1 -> 21.9 ms; close view at home 30.7 -> 28.8. The copy of the curve in `vegetation.cpp` (which shape a shadow ray meets) must stay the same as the shader's.

**The normal's three landscape heights in one pass** (`terrainB_slope`, 2026-11-23): the formula's slope is carried through the octave loop (gradient, second derivatives, and the damping term's own derivative), so the formula runs once instead of three times. Same picture. It saved only 0.1-0.45 ms: the formula was NOT where the normal's 5-6 ms goes (the `512` switch also turns off what hangs on the normal - the relief's taps, the stones' three taps at 2 ms, the sand's and snow's slopes). `RMDBG` 2048 brings the three passes back for comparison.

**The stones' slope from their formula** (`rm_pebbles_g`, 2026-11-23): the normal looked every stone layer up three times to difference them; a stone's shape is a formula, so its slope is returned with its height from one look-up. Same picture; terrain pass -0.15 to -0.4 ms at the High level (measured twice per view). `RMDBG` 4096 brings the three look-ups back.

**Where the High level's frame goes** (sweep, 2026-11-23, 17.6 / 18.5 / 21.5 ms): the terrain pass is 9.7 / 11.6 / 10.5 ms, of which the march alone is about 4.5 / 3.3 / 4.3; the upscaler 2.7-3.3; grass 1.1 / 1.6 / 3.2; trees 1.4 / 0.3 / 1.5. What is left in the terrain's shading is spread thin: relief 2.1-2.9, all shadows 0.9-2.7, no single item above 1.5.

**The grass pass, taken apart** (2026-11-23, High, the close view's 3.2 ms; switches in the shader, since removed): the loop over the blade cells 1.4 (four cells, two blades each near the camera); the close ground relief under the blades 0.7; the type field 0.15; gusts, the shading and its shadow ray: nothing measurable. Done: the relief's three noises are asked only where a blade can be - not above the tallest blade on the highest relief, and not over ground the mask says carries no grass (a road through a meadow is half the screen, and its rays paid for the relief all the way down, to be thrown away); the march's step factor 0.95 instead of 0.82 near the camera. Close view 3.2 -> 2.5 ms, woodland 3.6 -> 3.4, the others 0.05 - 0.15; the picture the same in side-by-side crops of blades and patch borders. What is left is the cell loop, which is the density setting.

**The terrain's rays in a wood** (same day, woodland view, terrain pass 8.2 ms): `VIBE_RM_SKIP` bit 32 takes 2.65 ms off - but it switches off three things: the sun's object shadow (one blocker ray and the shadow rays: 0.9 together), the bounce light's rays (two closest-hit rays of 24 m through the leaves: 0.8) and the ambient occlusion rays (2 m, next to nothing). One blocker ray instead of two, a first-hit blocker ray, no rays where the ground's own shadow is full: no difference (the settings already give one blocker ray; nothing in these views lies in a hill's shadow). Done: under DLSS Ray Reconstruction the bounce light traces half its rays a frame, as the shadow rays already did (woodland -0.4 ms, river -0.5, the others nothing). Shading under a dense lawn costs 1.0 ms in the close view and 2.7 in the wood, but the ground shows between the blades there (a flat colour in its place is plainly seen), and the blades blend with the ground from 5 m out, so it cannot simply be left out.

**The largest single item: the ground's normal outside the home tile** (2026-11-23, High; sweep in four views, 17.5 / 18.5 / 20.5 / 20.9 ms). `VIBE_RM_SKIP` bit 512 (the screen-space normal for every pixel; the sweep's label "far section off" is wrong) takes 2.7 ms off the open view and 4.6 off the river view, but 0.3 and 1.3 off the two views inside the home tile. Working the normal out and not using it saves nothing, so it is the normal's own arithmetic. Its parts switched off one at a time (river / open): the landscape's slope pass 3.7 / 2.05, the relief's three taps 1.3 / 0.3, the stones 1.55 / 0.75 (they overlap: all together 4.3 / 2.95). The slope pass's FBM is the same everywhere; what differs beyond 1450 m from home is `rm_world_h`, called at the normal's three points: `wm_continent` (21 noises, 29 near a coast) and `wm_ranges` (2) each time, and the river's 25 segments inside its box. The march pays the same per step out there (march alone: 4.8 / 3.5 ms of the pass).

**Done the same day: the world's slow fields as two textures** (`vk_engine/wm_cache.cpp`, `shaders/wm_cache.frag`, `rm_world_vr` in `terrain_rm_noise_road.glsl`). Land / sea (`wm_continent`) and the ranges (`wm_ranges`) are drawn round the camera into two RG32F pictures by the terrain shader's own functions (the bake shader includes `terrain_rm_noise_road.glsl` with `WM_CACHE_BAKE`), in the chart's coordinates, whichever chart: 2048 x 2048 at 4 m (+-4 km, all the normal march reaches) and 1024 x 1024 at 160 m (+-82 km; the far march ends at 80 km). Scene bindings 39 and 40, `scene.wm_cache` / `wm_cache1` (origin, texels a metre, size). A picture is drawn again (well under a millisecond) when the camera is 1 km / 1.9 km from its middle or the chart's mapping changes; origins are whole texels, so the same places are sampled before and after. Rules:
- *The shader sums the four texels itself* (`texelFetch`). A filtered read moves in 1/256 steps of a texel; the normal takes its slope over 2 cm and would see a staircase.
- *The terrain shader no longer holds the formula.* With a formula fall-back inlined at every height call it grew to 20 MB of SPIR-V and the optimiser ran out of ids ("ID overflow"); without it the shader is 6.4 MB (it was 13). Anything that needs `wm_continent` in that shader asks `rm_world_vr` / `rm_continent`.
- *Over the near picture's last 256 m the far one is blended in*, so the two meet without a step.
- The CPU's twin of the formula (`terrain.cpp`, physics and trees) is untouched; the bilinear read is within millimetres of it at 4 m a texel (the fields' finest feature is 700 m).

Measured (High, 3440 x 1440, frame / terrain pass): river view 17.8 / 11.0 -> 11.3 / 4.6 ms, open view 17.4 / 9.6 -> 11.3 / 3.4; inside the home tile nothing changes (20.3, 20.7). Checked: with the formula still in the shader as a fall-back (the first version) the land in the open and river views was the same picture, mean difference 1.5 grey levels against 1.1 between two runs; the remaining difference was the water and the leaves, whose clock is real time - `VIBE_TEST_ANIM_TIME=<s>` now stops it for such comparisons. A near chart, a site's chart and two views from the air (2.5 and 9 km) were looked at with the final version: trees on the ground, coast continuous to the horizon, no seam at 4 km. Not compared picture for picture against the old build in those four.


**Everything that is not the terrain** (2026-11-23, High, after the two changes above; `pt.sh`-style averages over a run with `VIBE_TEST_ANIM_TIME`, which brings the run-to-run noise of the sweep from 0.5 down to about 0.15 ms). Open / river / close / woodland, frame 11.6 / 11.4 / 20.6 / 19.9 ms:

| pass | open | river | close | wood |
|---|---|---|---|---|
| terrain | 3.7 | 4.5 | 10.2 | 7.0 |
| DLSS Ray Reconstruction (`fsr_taau`) | 2.9 | 2.7 | 2.6 | 2.6 |
| grass | 1.0 | 1.4 | 2.5 | 3.2-3.4 |
| trees, depth + colour (`d_veg`, `w_veg`) | 1.35 | 0.3 | 1.55 | 2.7 |
| lighting pass | 0.3 | 0.4 | 0.7 | 1.8 |
| compose (output resolution) | 0.44 | 0.44 | 0.44 | 0.44 |
| the rest together (sky fill, clouds, small plants' shadow map, depth copy, terrain upsample, scatter) | 1.4 | 1.4 | 1.9 | 1.6 |

- `fsr_taau` is NVIDIA's network and nothing else (cloud layer, sun shafts and RR's guide pass have marks of their own now and stay under the profiler's threshold). Plain DLSS is 1.4 ms instead of 2.9 - but without Ray Reconstruction every ray budget doubles: the open view gains 1.1 ms, the woodland loses 8 (lighting pass 1.8 -> 9.0). RR stays.
- Volumetric fog 0.1-0.4, clouds 0.1-0.4, bloom, water, reflections, radiance cache, ambient occlusion: each within the noise.
- Done: the terrain's bounce rays and the sun ray from where they land take leaf cards as solid cards (as the far shadow ray did; not at 'every leaf', not in the leafless season). Woodland terrain pass 7.75 -> 7.05 ms, close view -0.2; picture 0.09 grey levels from the exact one.
- Tried and not kept: the grass's nine-cell zone from 1.2 m instead of 2.5 m (-0.4 ms in the wood): a few dozen near blades lose pieces, visible in the difference picture.
- What remains in grass and trees is their density and detail settings.

**Later the same day** (2026-11-23, evening):
- *Frame generation* costs about 4 ms of GPU time per rendered frame at 3440 x 1369 (28.2 ms playing against 24.2 with it off, GPU pass sum the same). It is off while the menu is open (`dlss.cpp`: only in `State::Playing`), which is why the rendered frame rate rises there. It is outside every profiler mark: frame time = pass sum + frame generation.
- *Ray Reconstruction models* (`rt_.dlss_rr_model`, quality row "Denoiser model", `VIBE_DLSS_RR_PRESET=D|E|F` to measure): the SDK's default F 2.9 ms, D and E 2.4 ms. E leaves a stipple on the ground; D is a little crisper and less smoothed than F in stills (not judged in motion). Low and Medium use D, High and up the default.
- *The close relief* (`VIBE_RM_SKIP` 1024) is worth 3.3 ms in the close view and 2.2 in the wood, but its own evaluations are a small part: out of the normal's taps 0.8 / 0.25, out of the march 0.4, out of the sun-shadow march and the sand's slope taps nothing. The rest is what the relief brings with it - the sand drift and its shading on the road, a rougher surface to find and to light. A texture of the relief round the camera would not win it back.
- `VIBE_TEST_ANIM_TIME`, `VIBE_PROF_ALWAYS`, `VIBE_TEST_GFX_SEQ`, `VIBE_TEST_SCALE_UI`: test hooks added today (clock stopped; pass times in the menu too; quality levels / render scale changed while running).

**Hold a frame rate** (`rt_.auto_fps`, the panel's "Hold frame rate"; `auto_scale_tick` in `gfx_quality.cpp`; `VIBE_TEST_AUTO_FPS=<n>`): the render scale follows the GPU's load in steps 0.50 / 0.55 / 0.60 / 0.67 / 0.75 / 0.85 / 1.00. The load is the profiler's pass sum plus frame generation (no mark of its own: about 4 ms at 3440 x 1440), not the frame time - a cap or v-sync holds that up whatever the load. Down after 1.5 s over the budget, straight to the highest step that should fit (three quarters of the frame taken to grow with the pixel count); up one step after 4 s with room for it; 6 s of rest after every change, because a change rebuilds the render targets (a hitch of about a tenth of a second, the upscaler's history starts again). Tested: close view, 60 fps asked, from 0.72: one change to 0.50 (34.9 -> 18.4 ms of passes); open view, 45 fps asked, from 0.50: up to 0.55, then 0.60, then it stays. The profiler's running averages now count a pass that did not run in a frame as 0 for that frame (a pass run once used to stay in the sum for ever; one run every other frame counted double).

**Looked at and left** (2026-11-23): the ground's octaves are not a quality setting (they are the ground's shape, which physics and everything standing on it share - the sweep's "octaves 6" row is a measurement, not an option). The small plants' shadow map is already drawn every other frame, in turn with the sun's. Terrain at 50 % instead of High's 75 % saves 4.1 ms in the close view and 3.5 in the wood and flickers no more from frame to frame (0.35 against 0.30 grey levels) - but the paving comes out paler with a stepped outline and the sand's ripples softer, so High stays at 75 %; Medium is the 50 % level.

**Candidates not yet done** (from reading the shader; estimates, to be measured with the sweep): the ground under dense grass is fully lit - two bounce rays, object shadow rays, the blade-shadow march - and then covered by the blades (1-2 ms in a meadow); the 29-fetch cavity occlusion could be a baked texture (0.3-0.6); the two walks along the sun ray over the baked height could be one (0.2-0.4); the three normal taps share lattice cells and could hash once (0.4-0.6); outside the home tile the continent's 21-29 noises are added to every height evaluation (a texture round the camera would halve those steps - but its gradient shapes the coast's slope, so the normal cannot simply reuse one value).

**Shader review** (2026-11-23, night; four readers over all shaders, then measured with `tools/perf/bench.sh` at the settings as played, 3440x1400). Frame, close / far / forest: 24.8 / 19.8 / 23.0 -> 23.6 / 19.1 / 22.6 ms; the terrain pass 11.2 / 6.3 / 3.7 -> 10.0 / 6.0 / 3.4.
- *The ground's own sun shadow at home walks the baked heights* (`calcShadow`, `terrain_rm_shading_helpers.glsl`): the formula was asked for every step within 2.5 m of the bake (the first five or so of each lit pixel), after a 30-fetch walk over the same bake had already tested the same ray. Now one walk, on the bake, as outside the home tile. `RMDBG` 16384 brings the old path back. This is most of the gain (old against new, two runs each: -1.1 / -0.6 / -0.4 ms together with the next item). Pictures at 7.2 h, 14 h and 18.6 h in the close and far views: 0.2 - 0.9 grey levels from the old path, which is what two runs differ by. NOT looked at: a bare hillside in its own shadow at grazing sun, where the bake's 1 m texels could show as facets.
- *The march's root find stops when it is there* (`terrain_rm_heightfield.glsl`; `RMDBG` 8192 = the old three evaluations always).
- Small, same picture, nothing measurable alone: the wide layer of the small plants' shadow map is read only where it shows; the sea shore's sand noises only up to 2.5 m over the water (and a dead turf term is gone); the grass pass asks the height under an object only with the mesh terrain; a leaf card the detail rule leaves out quits the vertex shader at once (trees: no change measured).
- *Measured and dropped*: a separate early-depth variant for the trees' colour pass - the EQUAL test already rejects early (`VIBE_VEG_NO_EQ=1` costs +0.7 ms in the forest).
- *Read, not done* (estimates, not measurements): the relief's slope from one tap with shared lattice corners (the three taps were measured at 0.8 ms close); the froxel fog's shadow rays for slices behind geometry (up to 0.3); EASU at 1:1 in compose under an upscaler (0.1, slightly softer); cavity occlusion baked to a texture instead of 29 fetches (0.1 - 0.3); the road's 63 segments behind a coarse grid (0.1 - 0.2); the footprint bounds and mean colours as UBO values; grass steps bounded by the ray's slope (picture risk). On the FSR path (no Ray Reconstruction): fewer blocker rays far away, no occlusion rays on leaves, a smaller history gather; `descriptors.cpp` clears the foliage-ray bits when shadows are off, which distorts "shadows off" ablations.
- *Settings, not shaders*: with the quality rows "sun shadows" and "bounce light" at their highest (exact leaves for every ray, full bounce light on leaves) the forest's lighting pass is 7.1 ms; at High it is under 2.

## Cars: drawn between physics steps, and on the network (2026-11-23, night)

**A car is drawn between the last two physics steps** (`jeep.cpp jeep_after_sim`, `PhysicsWorld::vehicle_state_prev`). The physics runs in steps of 1/120 s and a frame holds one, two or none of them; drawn where the last step left it, the car moved by a different amount every frame. Measured on the course at 108 km/h (`VIBE_LOG_JEEP=1`, line "uneven by"): a frame's step differed from speed x time by 17 - 20 cm before, under 1 mm after. `VIBE_TEST_JEEP_RAW=1` brings the old drawing back. The camera first moves along with the car and then closes what is left (`cam_pos += velocity * dt`): following alone it hung back by speed x a time that depends on the frame's length. `JeepState::st` is this drawn state (a step behind at most); the physics' own is `vehicle_state`.

**Cars on the network** (`net_state.h` "Cars", `net.cpp` `kMsgCar`, `jeep.cpp jeep_net_send / jeep_net_receive`, protocol version 3). Rules:
- *One machine simulates a car*: its driver's, and still that machine after the driver got out, until somebody else gets in (`JeepState::net_mine`). It sends the car's state 30 times a second while it moves and once a second while it stands (so a player who joins later sees it where it is). Everybody else draws it from those states (`net_owner`) and does not simulate it; getting into it makes it yours. Two machines with the same car: whoever sits in it has it; both sitting: the lower id.
- *What is sent* (47 bytes): place, turn (quaternion), velocity, turn rate, steering angle, each wheel's height and speed, lights / brake / who drives / wheel contact. The wheels' turn is carried on by the receiver from their speed (a wheel goes round several times between two states).
- *Drawn a moment in the past*, between two states: the place along the curve through both velocities (`mix_car`: straight pieces between states 33 ms apart show in a bend), the turn the short way.
- *Times have their fraction of a millisecond* (`CarBuffer`, `MsgCar::t_frac`). At 100 km/h a whole millisecond is 3 cm: stamped and sampled in whole milliseconds the car moved unevenly by that much every frame.
- *A state is stamped with the time it is OF, not the time it is sent.* `jeep_net_send` runs after the simulation (`vk_engine.cpp`, behind `jeep_after_sim`) and gives `queue_car` the frame's clock; the session's update runs earlier in the frame, before the physics, and sent from there a state was last frame's with this frame's time. The network's clock is the frame's (`frame_clock_s_`, read where the frame's dt is measured), not the time a line of code happens to run at.
- *After a long frame the simulation is behind the clock* (the 1/15 s limit, the limit on physics steps): `sim_debt_s_`. The cars' stamps are behind the clock by it, so they stay the times of what was simulated; it drains at 1 % of the time passing, and more than 0.4 s is told as a jump (`kCarTeleport`).
- *The shared clock* (`ClockOffset`): a client's estimate of the host's clock takes the least-delayed of the last 3 s of samples and moves toward it at half a millisecond a second. (It was a running average of every sample: packets are noticed up to a frame late, and the clock followed that noise.) The delay a car is shown with follows the link at 1 % of the time passing (`CarBuffer::advance_delay`): a late packet makes it run 1 % slow, not jump.
- *The driver*: a remote player whose car says "driven" is posed in its seat (`mp_pose_avatars`, `jeep_seat_of`), as the local driver is; the position that player sends while driving is their camera and is not used.
- Not done: a car another machine simulates is not solid for the local player or car (as a parked car is not); a car's owner leaving the game leaves it where it stood on each machine, and a later joiner sees it in its bay until somebody uses it.

Measured with two instances on this PC (one drives the course on autopilot at 108 km/h, the other watches; `VIBE_LOG_JEEP=1` on the watcher, line "remote car"): the change of a frame's step from one frame to the next is 0 - 1 mm. Before the last three rules it was 6 - 13 cm.

Tests: `tests/anim/test_net.cpp` `[car]` (wire form, even motion round a bend at 144 frames a second, loss, stall, jump, parked car, the clock). Hooks: `VIBE_TEST_REALTIME=1` (a `--frames` run on the real clock instead of a fixed 1/60 s: needed for anything about evenness or the network), `VIBE_NET=1 VIBE_NET_PORT=<n>` (two test instances that do not meet a game being played), `VIBE_TEST_CAR_ECHO=1` (sender: what a receiver rebuilds against what is drawn - wheel places, axles), `VIBE_LOG_NET_CAR=1` (receiver: states that are not where the last one and the velocities say; `2`: every state), `VIBE_LOG_JEEP_NET=1` (frames whose step changed by over a centimetre; `2`: every frame's shown state).

## The haze of distance takes the sky's colour (2026-11-23, night)

`shaders/ground_haze.glsl ground_haze_air`: what is far away fades into the sky seen just over the horizon in that direction (the atmosphere's sky-view LUT), not into a fixed grey. The grey gave a far forest a hard outline against a brighter, bluer sky by day, and at night a band LIGHTER than the sky. The LUT is at binding 24 of the deferred lighting set (`deferred_lighting.cpp`) and binding 30 of the scene set (the terrain); the planet's far field in `compose.frag` uses the same colour. Without the sky model (`scene.atmo_params.x` 0) the old grey remains.

## The ground's shading rate (2026-11-23, night)

`shaders/terrain_vrs.frag`, `frame.cpp` "Pass 0.65", `world_render.cpp tr_vrs_near_m`. The rate image is drawn every frame from the terrain's low-resolution march (binding 34): a tile is shaded once per 2 x 2 pixels where ALL the ground in it is nearer than a distance and of one piece; far ground and every outline keep every pixel. It replaces the fixed pattern (coarse toward the screen's edges, whatever was there: far mountains at the sides came out in steps).

- *Coarse shading on ground that stands still in the picture shows.* Close view, everything within reach at 2 x 2: terrain pass 12.6 -> 5.9 ms, and the paving a mosaic of 4 x 4 screen pixels - the upscaler cannot rebuild what was never sampled, and the blocks sit on the pixel grid, so its jitter does not move them. So the distance comes from the camera's speed: `terrain_vrs` 1 = nothing up to 6 m/s, then 1.2 m per m/s (29 m at 108 km/h); 2 = 8 m at any speed, then 1.6 m per m/s. Standing or walking at level 1: every pixel, and the pass is not run.
- Measured on the course on autopilot at 108 km/h, 3440 x 1400: terrain pass 4.3 -> 2.4 ms, frame 19.4 -> 17.1 ms.
- It needs the march pre-pass (`terrain_rm_prepass`) and the full-resolution terrain (`terrain_raymarch_scale` 1). `VIBE_TEST_VRS_NEAR=<m>` sets the distance (tests).
- Known: the validation layer reports that the other pipelines of the world pass were not created for a shading-rate attachment (they have no shading-rate state and draw at 1 x 1; it was so with the fixed pattern too).

**A mesh for the near ground would not make it faster** (measured the same day, close view at home, terrain pass 12.5 ms): the march alone (`VIBE_RM_SKIP=64`) is 4.7 ms, and starting every pixel's march at the hit of the low-resolution march (`terrain_rm_start_at_hit`, what a mesh under the ray would give) makes that 4.6 ms and costs 1.1 ms in the pre-pass. Without the pre-pass at all the pass is 13.4 ms. The march's time is the last few evaluations of the detailed surface (relief, stones) and the root find, not the way to it; the rest of the pass is shading. A near mesh would pay off only as a second ground renderer whose detail is baked (geometry and textures instead of the formula) - and would then have to match the raymarched ground's look and shape at the hand-over.

**The low-resolution terrain's upscaler, reviewed** (`terrain_raymarch_compose.frag`; same view: 50 % with "temporal" 4.7 ms against 12.5 at full size): smooth, but the paving's joints and the sand's grain are gone, and the two-frame checkerboard leaves a fine stipple on high-contrast ground. The pass is a spatial upscale (Catmull-Rom limited to the 2 x 2 neighbourhood, a sharpen) plus the one frame before; it cannot bring back what its samples never held. What would: a full-resolution picture of the terrain alone that is kept from frame to frame (four sample positions in turn, last frame's picture moved by the motion vectors and limited to the range of this frame's neighbourhood) - the terrain's own temporal upscaler in front of DLSS. It needs its own target, because the world pass draws objects over the terrain. Not built.

**The march's reach does not depend on the step factor** (`terrain_rm_heightfield.glsl rm_step_budget`, 2026-11-23). A step is the factor x the clearance, so on a small factor (0.15, set to keep steep rock faces whole) a ray skimming the ground got a fraction as far on the same step count and gave up a few hundred metres out: a strip of sky between the near ground and the far land, from the ground only (from 100 m up every step is large). The step count of the settings now stands for a factor of 0.55, and a smaller factor gets proportionally more (three times at most); the low-resolution march also lengthens its stride with distance, up to what a factor of 0.55 takes and no further (a block's ray that jumps a ridge leaves a block-sized hole). A factor of 0.55 or more marches as before. Cost at 0.15 / 192 steps in the view along the course: the low-resolution march 0.47 -> 0.78 ms, the full pass unchanged.

## Cars: the guard rail, and grass under a car (2026-11-23, late)

- **A car slides along a rail or a wall instead of hanging on it** (`physics_world.cpp`, `highway_furniture.cpp hw_rail_boxes`). Measured: steered gently into the rail at 40 km/h on full throttle it dropped to 5 km/h and crawled along at 10 - 20; now 40 -> 35 -> 41 and on. Four causes, all removed: the wheels' sweep took the rail's side for ground (`VehicleImpl::WheelGround`: the wheels ignore the obstacle body - and the car's own, which the tester's default filter did and a filter of our own replaces); friction 0.5 on the obstacles (now 0.03); corners catching on the joints between a rail's boxes (`mEnhancedInternalEdgeRemoval`); a 12 cm sheet for a rail (now 40 cm, grown away from the road, the boxes overlapping).
- **Grass lies down under a car and comes up behind it** (`grass_raymarch.frag car_press`, scene block `car_press[4]`, `jeep.cpp jeep_grass_press`): the two cars nearest the camera; a blade under one is a tenth of its height, and behind the car it rises again over the stretch driven in the last 1.5 s (25 m at most), swaying as it comes up. The stretch is straight behind where the car faces now. Seen in a test drive: no blades through the car. Not yet looked at: the blades rising behind it.
- Checked: a jeep placed by the test hook at `20,120` (inside the castle's colliders) is thrown about - by the build of release 0.2.0 too - and with this build ended under the ground. Placed on open ground inside the home tile (`300,300`, `-200,400`) it stands and drives. A bad place for the test, not a fault of the home tile.
