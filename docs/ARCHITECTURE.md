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
| Far trees as baked cards (octahedral impostors with depth, 64 views per tree type, baked at start-up) | `veg_impostor.cpp` (bake), `veg_leaf.glsl` `veg_imp_uv` (lookup), setting `tree_impostor_px` |
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
| Gun, bullets, fire rate (laser and rifle) | `combat.cpp`; bolt glow: `fx.cpp`; gun models and where they sit: `props.cpp` `init_viewmodel` |
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
- **Screen-space displacement mapping (SSDM)** (`vk_engine/ssdm.cpp`, `shaders/ssdm_height.*`, `shaders/ssdm_solve.frag`, `shaders/ssdm_grid.*`, `shaders/ssdm_common.glsl`): the method of the reference game, written after reading the compiled screen-space tessellation shaders of the game. The stonework is drawn flat; then, in the middle of the main pass (`render_ssdm`, called by `render_world` right after the castle meshes, before lighting, trees, effects, grass and the weapon): (1) *height pass* - the displaced mesh parts once more against the scene depth, writing normal + height in metres; (2) *solve pass* - per pixel the displacement vector: the height evened out with its neighbours, where a neighbour only counts as far as it agrees (depth, normal, height), so the height sinks to nothing at corners, at the foot of a wall and next to anything not displaced, and a pixel beside a depth jump is not displaced at all; (3) *grid pass* - a grid with one vertex per pixel over the screen rectangle that holds such stonework, each vertex the surface point of its pixel moved by its vector, the fragment stage copying the G-buffer of the source pixel, depth-tested and depth-writing into the main pass's own attachments. Not-displaced vertices get a NaN position, which drops their triangles. Also in the solve pass, ours and not the game's: what lies far behind a pixel neither holds it down nor cuts it (outlines are displaced); the height is limited by the distance to the nearest edge along the four screen directions (nothing stands out further than it is from a corner or ledge - thin ledges stay flat instead of being smeared); and the stones shade each other - a short sun march over the height field plus a joint shade, multiplied into the albedo by the grid pass (the ray-traced sun shadow is cast by the flat wall). The grid pass smooths and darkens pixels where the grid is pulled apart (a stone's side) and drops triangles pulled over more than ten pixels. The height pass samples the height texture at the mip of the LONG footprint axis (per-pixel height jumps comb the grid into stripes at slants). Displaced: mesh parts with `CastleMeshGpu::ssdm > 0` (metres; from `LevelMesh::ssdm`, or `0.28 * relief` for masonry albedo 1 / 4) and home's static box brushes with albedo 1 / 4 (walls) or 5 / 6 (floor tiles, top face; `brush_ssdm_amplitude` in ssdm.cpp). While the pass is on they are drawn flat (`ssdm_active()`: mesh relief off in `draw_castle_meshes`, wall mode 0 in the scene constants). Rule: stones only rise from the drawn surface, never sink below it (ray-traced shadows and collision are those of the drawn surface). On by default with deferred lighting; `VIBE_NO_SSDM=1` off (parallax / box relief back), `VIBE_SSDM_NO_BRUSHES=1` mesh parts only, `VIBE_SSDM_DEBUG=1` paints moved pixels red and grid pixels left in place green. Tunables at the top of ssdm.cpp (fade 40-55 m, smoothing radius, shade strengths). Cost measured before the edge-distance taps were added: 0.09 ms at 640x360 internal, 0.31-0.35 ms at 1280x720 internal; not re-measured since (the GPU was shared during the last runs). Not done: a compute tile list (the grid covers the bounding rectangle of the visible stonework), unit tests (checked by screenshots only).
- **Small plants** (`scatter.cpp`): 32 m cells, culled on the CPU (frustum, distance, walls, hills). Each species has three meshes: full, ~32 % (beyond 22 m) and ~11 % (beyond 48 m), made at load time with meshoptimizer (`simplified` in `build_scatter`). `VIBE_NO_PLANT_LOD=1` turns them off.
- **Trees, the reference game's way** (2026-11-25). Written after reading the game's compiled tree shaders (foliage shader, impostor shader, tree shadow shader, its diffuse-GI pass; notes outside the repo: `dev/cd-shaders/trees/REPORT.md`). The game's trees are authored meshes; ours stay the engine's own (`eztree.cpp`) - what was taken over is how they are drawn.
  - *Far trees: octahedral impostors with depth* (`vk_engine/veg_impostor.cpp` bake, `shaders/veg_leaf.glsl` `veg_imp_sample`, `veg.vert` the card, `cube.frag` / `veg_imp_depth.frag` the pixel). Every tree type is baked from 8 x 8 directions on a hemi-octahedron (96 px tiles, orthographic, framed on the sphere round the tree's middle) into two atlases: colour + coverage, and normal (octahedral, rg) + depth (b) + leaf mark (a). 24 types: 4608 x 3072, baked in about 1.3 s at start-up. The card is a square facing the camera on the NEAR side of that sphere; the pixel cuts its view ray with the plane through the tree's middle, projects that point into the three baked views round the direction to the camera and blends them - only in the middle fifth between two views, so mostly one view is shown sharp - cuts at alpha 0.5, and moves the point by the blended depth: that is the depth written (never nearer than the card, so `depth_greater` holds), in the pre-pass and in the colour pass alike (the colour pass tests EQUAL: both compute it with the same `precise` lines). So a card looks right from above, cuts into the ground and other trees like the tree, and can take over early (`tree_impostor_px`). The normal: the game takes the slope of the depth, turns it to the eye where edge-on and leans it 10 - 20 % up; ours is that, half way to the baked crown normal (so card and mesh are lit alike). Roughness 0.9. Numbers shared between bake and shaders: `kImpN`, `kImpTile`, `kImpCols`. Measured in one view of a tree line at 64 px hand-over: tree passes 2.8 -> 0.45 ms against meshes.
  - *Wind* (`shaders/veg_wind.glsl`, called by `veg.vert` for this and last frame's pose): the game's one body for branch, frond and leaf - the wind here (global, turned and strengthened by the wind field within 64 m), its fade with distance, a few millimetres for the whole plant, the branch swing (the visible motion), a ripple and a tumble for each leaf card about the point it grows from, and low plants leaning away from who walks through them (the trail picture; divided by (height - 1.5)^1.8, so trees do not). What its modelling tool bakes into every vertex of the game's meshes (branch weight, phase, direction, leaf pivot) is worked out from the vertex's place in the tree. The ray-traced shadows are cast by the still tree.
  - *Leaves in the lighting pass*: leaf pixels of trees and bushes are material id 2 like the grass: the game's foliage back-light, roughness 0.45 + 0.005 a metre, and occlusion only coming in with distance.
  - *Gusts* (`shaders/wind_gust.glsl`, ours - not in the game's shaders): the wind's strength is a pattern of fronts travelling down the wind at 6 m/s (about 30 m apart, in groups, with lulls), read at each plant's place by the grass (`tuft_cd.glsl`) and the trees; under a front blades and branches also lean down the wind. This is what makes a field move in waves instead of all at once.
  - *Not taken over*: the game's shadow-map passes for trees (alpha 0.45 on sharper mips, dithered fade, leaf-coloured static shadows, impostor shadows) - shadows here are ray-traced; its compute / mesh-shader paths that draw far impostors in bulk; its coverage trick for leaf edges under 2 x 2 shading; the terrain-colour tint of impostors.
- **Grass, the reference game's way** (2026-11-25; species 15 = long tufts, 17 = turf). Written after reading the game's compiled shaders (foliage shader, placement shader, interaction shader, its diffuse-GI pass); our own code, the game's formulas and constants. The notes on what was read are outside the repo (`dev/cd-shaders/grass/`: `REPORT.md`, `WIND_FLUID.md`).
  - *One kind of grass.* The game has no lawn of its own kind: its ground cover is the same instanced clumps as its long grass, placed densely. So the short grass is **turf** (species 17, `build_grass_turf`: 16 short blades) and replaces the raymarched lawn (`grass_raymarch.frag` is not drawn while `turf_on()`; `VIBE_LAWN_RM=1` brings it back). The turf grows where the lawn's own mask (`grass_mask_`, scene binding 13) has grass, on the whole home tile.
  - *Placement* (`shaders/tuft_place.comp`, run by `cull_plants` in `vk_engine/plant_gpu.cpp` right after `plant_cull.comp`, once per kind): no list of automatic grass exists. Every frame the 4 x 4 block of cells round the camera is gone through: N points of the R2 sequence a cell, 5 candidates a point on a disc of a metre (clumps), range = min(setting, 3 cells), everything shrinking away over the second half of the range (the game's placed grass has no dissolve), then the ground (height and slope from the terrain's height map, binding 6; what may grow from the lawn mask, binding 7, or from `ScatterSet::tuft_rules`, binding 5), density (keep 1 - 1/(10 d + 1)), the six-octave clustering noise (long tufts only), a turn and a size from hashes, frustum and depth-pyramid culling, and the record appended to the species' three (level of detail) lists - the lists `plant_cull.comp` fills with the hand-placed plants, drawn as before. The cells are fixed in the world, so a clump is the same one every frame. Each kind's numbers are in a table in `cull_plants` (`kinds`); they go to the shader after the species table in the buffer at binding 3.
  - *Beyond the home tile* the placement reads the ground from the picture the engine bakes round the camera there (`veg_stream.cpp` `wv_nf_poll`: 512 m at 1 m, heights in RGB, "grass may grow" in A; binding 8) and applies the wider world's grass rule (`grass_raymarch.frag`: keep in step) - so turf and fields carry on past the tile's edge. The vertex shader has no height map there: blades stand upright and are not re-seated.
  - *Fields* (`flower_fields_` in `vk_engine.h`: x, z, radius, how frayed the edge is; at most four, sent after each kind's numbers): a kind with `lawn == 2` grows only inside them, wherever the short grass may. The one kind so far is the **flowering meadow** (species 18, `build_meadow`: 16 to a clump, eleven of them stalks ending in a yellow head - mesh part 9, a grass blade in everything but the colour of its head, `scatter.frag`), placed to 230 m: every clump close by, from 12 % of the range a growing share shrinks away and the fifth that stays grows, so the field stays closed from far off at a fraction of the clumps. The launch site has one (`rocket.cpp`, radius 205 m round the pad). The kind is not dispatched at all when the camera is out of reach of every field. `VIBE_NO_MEADOW=1` leaves it out. A new field: one `push_back`; a new kind of field plant: a mesh, a species number, a row in `kinds`.
  - *Cover, near and far* (2026-11-25, after two rings were seen round the camera): the turf's and the meadow's levels of detail are built by hand (`build_grass_turf` / `build_meadow` with a level: all 32 blades to 22 m, every other one wider to 48 m, every fourth beyond) - 16 blades left the ground showing between them close by when looked at from above, and the general simplifier lost the meadow's flower heads. A kind that thins out with distance (the meadow) grows what stays by 1 / sqrt(share kept), so the ground it covers is the same at every distance. A person standing still presses the grass only 40 % down (`grass_trail.frag`; the game: flat).
  - *The rules picture* for the long tufts (`ScatterSet::tuft_rules`, 1024 x 1024 over the plants' 560 m square, made by `scatter.cpp`): every rule that used to reject a tuft on the CPU (painted grass, buildings, the road, water, rock, thinner under trees), and dryness. A crater clears its patch (`craters.cpp`). Whatever must keep long grass away has to be in `ScatterParams` when the plants are built; whatever must keep turf away has to be cleared from the lawn mask.
  - *The record* is the 32-byte plant instance: a NEGATIVE size marks GPU-placed grass (already faded: `scatter.vert` does not fade it again).
  - *Vertex shader* (`shaders/tuft_cd.glsl`, included by `scatter.vert`): the wind here (global wind, turned and strengthened by the wind field within 64 m), its fade (full to 48 m, none from 64 m), the sway of the whole clump, the per-blade "branch" term, the ripple, the "tumble" (the blade as a whole about its root: a twist and a lean down the wind), the basis on the ground's normal, every blade set on the ground under its own root (and a turf blade over bare ground cut), then the trail, read at the blade's root: the blade turns about its root by up to 0.3 pi away from the push. What the game's meshes carry per vertex (the blade's root, direction, weights) is worked out from the blade's number: the meshes number their blades (uv.y = (blade + along) / 32) and place their roots on a spiral the shader repeats (`tuft_root` / `tuft_cd`; the shape numbers are `ScatterSet::kTuftShape` / `kTurfShape`, sent in `pc.tex_params`). Hand-placed tufts also get a lean of their own, as the game's seeded instances do.
  - *The trail* (`vk_engine/grass_trail.cpp`, `shaders/grass_trail.frag`, `grass_trail_h.frag`; scene bindings 41 - 44): 256 x 256 texels of 25 cm round the player. One picture holds direction, speed and how far down (decay exp2(-1.7847 dt), climbing under a body at the game's rate), the other between which heights the body stood - a blade reacts only when its root lies between them, so nothing bends under a bridge. Pushers: the player on foot, the players' cars, the world's people and their cars - the nearest twelve, a person 0.6 m, a car 2.1 m.
  - *The wind field* (`vk_engine/wind_fluid.cpp`, `shaders/wind_fluid.comp`; scene binding 45): the game's fluid simulation on a 48^3 grid of 2 m cells round the camera - advect, forces, curl, vorticity confinement, 8 pressure steps, gradient, history - stirred by whatever moves near by (the player, cars, people). Grid size, cell and step count are ours (the game's CPU sets them). `VIBE_TEST_WIND=<power>` puts a vortex in front of the camera.
  - *Pixel shader* (`scatter.frag` part 8): the ground's hue at the blade's own luminance, back faces x1.1, roughness 0.45 + 0.005 a metre. Material id 2 = grass: `deferred_lighting.frag` adds the game's back-light for its foliage ids in place of the leaf lobes: diffuse += max(0, 0.3 - N.L) x 0.2319 x the shadow term (the phase-and-thickness formula used at first belongs to other shading ids - corrected after the tree shaders were read, `dev/cd-shaders/trees/REPORT.md`).
  - *Shadows*: GPU-placed grass is drawn into the small plants' shadow maps (`clutter_shadow.cpp`, `shaders/tuft_shadow.vert`: the colour pass's blade maths, so flattened or wind-blown grass casts the shadow of what is drawn) - the 64 m map, the 16 m near map and the flashlight's, not the lanterns' (the meadow: not the 64 m map). It is drawn from the lists the placement filled for the camera, so grass out of view casts no shadow into it. Ours: the game's dynamic cascades leave out everything under 1.5 m bounding radius. `VIBE_NO_GRASS_SHADOW=1` none, `VIBE_GRASS_SHADOW=1` long tufts only, `=2` near map only. Cost read on a busy GPU: about +0.7 ms in `clutter_shadow` on the frames the maps are redrawn (every other one).
  - *Not carried over, and why*: the game's instance record of four rows (the ground normal is looked up again in the vertex shader); its top-down captures of the terrain and of everything that moves (ours: the height map, the lawn mask, a list of pushers); its screen-space contact shadow and AO tweak for foliage; the terrain-paste pass that blends ground material onto the grass's base. All per-asset and per-quality numbers (cell size, points a cell, density, noise size, sizes, fade distances, mesh shapes) are ours: they are not in the shaders.
  - Switches: `VIBE_NO_TUFTS=1` no automatic grass (and the raymarched lawn back); `VIBE_LAWN_RM=1` the raymarched lawn instead of turf; `VIBE_TUFTS_CPU=1` the long tufts placed on the CPU as before (also with `VIBE_NO_PLANT_GPU=1`); `VIBE_NO_TUFTS_GPU=1`; `VIBE_NO_GRASS_TRAIL=1`; `VIBE_TEST_TRAIL_R=<m>`; `VIBE_NO_WIND_FLUID=1`; `VIBE_TEST_WIND=<power>`. GPU mark `trail_wind` = the trail and the wind field. Checked by screenshots only (no unit tests); costs could not be measured on an idle GPU yet.
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
- To cluster another mesh type: build a `ClusterMesh`, add it to the cluster / table buffers in `init_cluster_draw`, and give its instances slots like the rocks. Each drawn cluster is tested on its own against the view and last frame's depth pyramid (`cluster_cull.comp`, `static_cluster_cull.comp` - checked 2026-11-23: this line used to say it was not done). Not yet done: ray-tracing structures from clusters (the structures are built from the archetypes' LOD 1 meshes and the static parts' full meshes).

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
- *The ground's own sun shadow at home walks the baked heights* (`calcShadow`, `terrain_rm_shading_helpers.glsl`): the formula was asked for every step within 2.5 m of the bake (the first five or so of each lit pixel), after a 30-fetch walk over the same bake had already tested the same ray. Now one walk, on the bake, as outside the home tile. `RMDBG` 16384 brings the old path back. This is most of the gain (old against new, two runs each: -1.1 / -0.6 / -0.4 ms together with the next item). Pictures at 7.2 h, 14 h and 18.6 h in the close and far views: 0.2 - 0.9 grey levels from the old path, which is what two runs differ by. Looked at on 2026-11-24 on the sand at home at 18:54 (sun 2 - 3 degrees up), old path against new: 0.8 - 0.9 of 255 apart, the same picture. (What that picture showed in both - a ragged pattern of light and dark over the sand - was another fault, since fixed: "Ground turned away from the sun, in an object's shadow" at the end.) Still not looked at: a bare hillside.
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
- It needs the march pre-pass (`terrain_rm_prepass`) and the full-resolution terrain (`terrain_raymarch_scale` 1; the march itself also runs ahead of the low-resolution terrain - last section). `VIBE_TEST_VRS_NEAR=<m>` sets the distance (tests).
- Known: the validation layer reports that the other pipelines of the world pass were not created for a shading-rate attachment (they have no shading-rate state and draw at 1 x 1; it was so with the fixed pattern too).

**A mesh for the near ground would not make it faster** (measured the same day, close view at home, terrain pass 12.5 ms): the march alone (`VIBE_RM_SKIP=64`) is 4.7 ms, and starting every pixel's march at the hit of the low-resolution march (`terrain_rm_start_at_hit`, what a mesh under the ray would give) makes that 4.6 ms and costs 1.1 ms in the pre-pass. Without the pre-pass at all the pass is 13.4 ms. The march's time is the last few evaluations of the detailed surface (relief, stones) and the root find, not the way to it; the rest of the pass is shading. A near mesh would pay off only as a second ground renderer whose detail is baked (geometry and textures instead of the formula) - and would then have to match the raymarched ground's look and shape at the hand-over.

**The low-resolution terrain's upscaler, reviewed** (`terrain_raymarch_compose.frag`; same view: 50 % with "temporal" 4.7 ms against 12.5 at full size): smooth, but the paving's joints and the sand's grain are gone, and the two-frame checkerboard leaves a fine stipple on high-contrast ground. The pass is a spatial upscale (Catmull-Rom limited to the 2 x 2 neighbourhood, a sharpen) plus the one frame before; it cannot bring back what its samples never held. What would: a full-resolution picture of the terrain alone that is kept from frame to frame (four sample positions in turn, last frame's picture moved by the motion vectors and limited to the range of this frame's neighbourhood) - the terrain's own temporal upscaler in front of DLSS. It needs its own target, because the world pass draws objects over the terrain. Built since: "The terrain's temporal upscaler" below.

**The step factor goes down to 0.05** (2026-11-24, was 0.15: the menu, the three clamps in `world_render.cpp`, and `rm_step_budget` up to eight times the step count instead of three). Far view, terrain at 75 %: 8.8 ms at 0.15, 9.3 at 0.10, 10.6 at 0.05, the low-resolution march 0.47 -> 1.01; no holes seen.

**The march's reach does not depend on the step factor** (`terrain_rm_heightfield.glsl rm_step_budget`, 2026-11-23). A step is the factor x the clearance, so on a small factor (0.15, set to keep steep rock faces whole) a ray skimming the ground got a fraction as far on the same step count and gave up a few hundred metres out: a strip of sky between the near ground and the far land, from the ground only (from 100 m up every step is large). The step count of the settings now stands for a factor of 0.55, and a smaller factor gets proportionally more (three times at most); the low-resolution march also lengthens its stride with distance, up to what a factor of 0.55 takes and no further (a block's ray that jumps a ridge leaves a block-sized hole). A factor of 0.55 or more marches as before. Cost at 0.15 / 192 steps in the view along the course: the low-resolution march 0.47 -> 0.78 ms, the full pass unchanged.

## Cars: the guard rail, and grass under a car (2026-11-23, late)

- **A car slides along a rail or a wall instead of hanging on it** (`physics_world.cpp`, `highway_furniture.cpp hw_rail_boxes`). Measured: steered gently into the rail at 40 km/h on full throttle it dropped to 5 km/h and crawled along at 10 - 20; now 40 -> 35 -> 41 and on. Four causes, all removed: the wheels' sweep took the rail's side for ground (`VehicleImpl::WheelGround`: the wheels ignore the obstacle body - and the car's own, which the tester's default filter did and a filter of our own replaces); friction 0.5 on the obstacles (now 0.03); corners catching on the joints between a rail's boxes (`mEnhancedInternalEdgeRemoval`); a 12 cm sheet for a rail (now 40 cm, grown away from the road, the boxes overlapping).
- **Grass lies down under a car and comes up behind it** (`grass_raymarch.frag car_press`, scene block `car_press[4]`, `jeep.cpp jeep_grass_press`): the two cars nearest the camera; a blade under one is a tenth of its height, and behind the car it rises again over the stretch driven in the last 1.5 s (25 m at most), swaying as it comes up. The stretch is straight behind where the car faces now. Seen in a test drive: no blades through the car. The blades rising behind it, looked at on 2026-11-24 (straight across the meadow by the highway at 28 km/h, from above and behind: `VIBE_TEST_JEEP=0.3,0 VIBE_TEST_JEEP_AT=1697,930,3.14159 VIBE_TEST_JEEP_ORBIT=0.5,-0.55,10`): a flattened lane the car's width behind it, closing to nothing over about ten metres. One still picture - the sway as they come up was not watched.
- Checked: a jeep placed by the test hook at `20,120` (inside the castle's colliders) is thrown about - by the build of release 0.2.0 too - and with this build ended under the ground. Placed on open ground inside the home tile (`300,300`, `-200,400`) it stands and drives. A bad place for the test, not a fault of the home tile.

## Cars: the dial, turbo, a passenger, street lamps (2026-11-23, night 2)

- **Speedometer** (`jeep.cpp jeep_hud`): a dial above the weapon panel - needle and figures for the speed, the gear, the revs as an inner arc, the turbo's tank as the outer arc.
- **Turbo** (Sprint while on the throttle; `JeepState::turbo`): 3.2 s of extra push (7.5 m/s2 along the car, with two wheels down), a blue glow at the tail. It fills again in 9 s once it has been off for 0.8 s; run dry it must reach a quarter before it works again. `VIBE_TEST_JEEP_TURBO=1` holds it (measured: 30 -> 67 km/h after the same six seconds from rest).
- **The view** is nearer: 2.7 / 3.3 / 4.0 m behind (was 4.3 / 6.4 / 10.5).
- **Steering from a key** asks for four fifths of the tyres' grip, the full grip only after the key has been held for over half a second; the wheel turns more slowly the faster the car goes and returns twice as fast as it goes out. **Tyre marks and smoke** go by the tyre's slip angle (from 11 degrees), not by sideways speed alone: at speed every turn of the wheel drew them. The tyre curves themselves are unchanged. Not judged by a driver yet.
- **A passenger** (`jeep_ride_`, `jeep_ride_place`, `jeep_passenger_seat`): Use at a car another player is driving gets in beside them (Use again: out). The passenger is carried at the seat, looks round and may shoot. Others see them seated: the two low bits of the state's weapon byte are what is held, the bits above the car ridden in + 1 (protocol version 4). `VIBE_TEST_RIDE=1` gets in beside the first remote driver (seen working in a two-instance run).
- **Street lamps** (`highway_furniture.cpp`, `g_lamp`): on the landward side of the road every 40 m, for 2 km of road either way from the launch site. Their lights are not in `lamp_lights_` (they would be cut off when a site is left): `hw_lamps_near` adds the near ones when the scene's eight lamp lights are chosen (`descriptors.cpp`), at three times a lantern's strength. Seen at night: posts and lit heads; how much they light the road was not tuned.
- **Lights seen from far (2026-11-25, "distance of lights in the dark is too close").** Two limits, both raised. (a) *A lamp seen as lit*: a lamp's head is an emissive panel a hand wide facing the ground - under a pixel from a few dozen metres, so a road at night showed the nearest lamp or two. `fx.cpp render_fx` now draws a glow for every lantern (`lamp_lights_`) and street lamp (`hw_lamps_near`) between 20 m and 1600 m at night: a soft disc and a core in the `flash` ribbon, depth-tested, never smaller than 2.6 pixels (fades in 20 - 60 m, out 1280 - 1600 m; the posts themselves are streamed to 1500 m). The launch site's pad lights already had flares to 9 km, a car's lamps to 900 m: unchanged. (b) *Lamps that light the ground*: the scene block's `lamp_lights` went from 8 to 24 slots, chosen from 800 m (was 220 m), nearest first, a lamp more than 60 m off and behind the camera counted as twice as far (`descriptors.cpp update_scene_ubo`); `lamp_info.y` is the number of slots in use and every shader's lamp loop runs to it (0 by day). A light with a slot of its own (rooms by day, the control house, the rocket's flame, the chapel's candles) no longer puts out the lantern that had the slot - that one moves to a free slot from 8 up. A lamp's reach on the ground (34 m, `kLampR`) and the plant shadow maps (fixed slots, eight lanterns or fewer, 70 m) are as they were. Measured at night (23.5 h, RTX 4080, three views, 8 from 220 m against 24 from 800 m in the same build - `VIBE_LAMPS_OLD=1`): down the coast road the furthest lit lamp went from 160 m to 471 - 637 m; `w_terrain_raymarch` 2.17 / 2.19 ms and 2.35 / 2.34 ms, `deferred_lighting` 0.10 / 0.10 and 0.08 / 0.085 ms; in the station's car park 2.45 / 2.44 and 0.19 - 0.23 / 0.19 - 0.24 ms; `fx` under 0.1 ms - no cost that can be told from the noise (a pixel still only traces to the lamps within 34 m of it). Hooks: `VIBE_LOG_LAMPS=1` (lights lit and the furthest, glows drawn, every tenth street lamp's place and the road's direction there), `VIBE_LAMPS_OLD=1`.
- Not done from the same list: per-cluster culling and ray-tracing structures for the cluster LOD, the terrain's temporal upscaler, bullets against the streamed trees (`props.cpp` already hands `wv_cols_` to the physics' near set - whether a bolt stops at a trunk was not tested).

**Bolts against what the physics does not hold** (`combat.cpp update_projectiles`, 2026-11-23): a bolt's way since the last step is tested against the trunks of the trees beyond the home tile (`wv_cols_`), the road's guard rail (`hw_rail_boxes`) and the street lamps' poles (`hw_pole_boxes`). `VIBE_LOG_BOLT=1` says what stopped a bolt. Seen: bolts fired at the rail stop on it (they flew through before). The trunks: fired into the wood beside the highway from (1687, 940) in two directions (2026-11-24, `VIBE_TEST_FIRE=1`) - every bolt, 37 and 36 of them, "stopped by a tree beyond the home tile", 55 and 113 m off. (That the collider stands exactly where the drawn trunk does was not checked: the hit lay behind a bush.)

**The sun's glow stays out of the haze on things** (`ground_haze.glsl ground_haze_air`, `compose.frag`, 2026-11-23): the haze colour is the sky's no nearer than 40 degrees round from the sun. Taken straight toward a low sun it washed a mountain orange that had the sun BEHIND it (the air on this side of it lies in its shadow). Looked at on 2026-11-24 at 18:54 (close view at home, toward the sunset glow and to both sides): the far trees and what shows of the mountains keep their own colour, no orange wash. The car's view: 3.6 / 4.6 / 5.8 m behind (2.7 / 3.3 / 4.0 was too near). The street lamps at three times a lantern's strength light the road visibly (seen at night).

## The terrain's temporal upscaler (2026-11-24)

`shaders/terrain_rm_resolve.frag`, `terrain_rm_upscale.glsl` (the enlargement, shared with `terrain_raymarch_compose.frag`), `frame.cpp` "Pass 0.55", `src/engine/terrain_keep.h` (the sample places; `tests/test_terrain_keep.cpp`). With the terrain below full size and "temporal reconstruction" on (`terrain_rm_temporal`; the Medium and High levels of "Terrain resolution"):

- A picture of the terrain alone is kept from frame to frame: two RGBA16F images written in turn (rgb colour, a distance; the `lrb_color_*` / `lrb_depth_*` fields, bindings 36 / 37). It is needed because the world pass draws objects over the terrain - the frame's final picture cannot be its history.
- **The kept picture is twice the render size each way** (`tr_->keep_extent`). The main upscaler (DLSS, FSR) moves the camera by part of a pixel every frame and builds the screen's detail from what it finds at those places, so it must be handed, every frame, what the ground looks like at that frame's place inside each pixel. The first version kept a picture of the render size: one fixed place per pixel. Read between its pixels it gave the upscaler the same picture every frame, and paving ten metres off came out in blocks the size of a rendered pixel (its joints are one screen pixel wide). That was the fault that looked like "the low-resolution pass shades the ground more simply" - it does not; the pass is the same shader with the same push constants.
- The low-resolution terrain is moved inside its own pixel to n x n places in turn (`terrain_keep_phase`, `tr_->lr_phase`): n = 4 at half size (each of its pixels covers 4 x 4 kept pixels, a round is 16 frames), 3 at three quarters, never more than 4. In ordered-dither order: any four frames in a row take one place in each quarter of the pixel.
- *The kept picture lives in the STILL picture's pixels.* A history that followed the camera's sub-pixel offset was sampled between its pixels every frame and only blurred. The low-resolution terrain takes that offset out (`lr_jit = lr_phase - jitter / 2`) and the compose pass reads the kept picture where its own moved pixel lies in it.
- The resolve pass, per kept pixel: last frame's picture where the ground has moved from (Catmull-Rom, five taps - with the plain bilinear filter a turning camera had no more detail than the plain enlargement), used where its distance agrees and limited to a range round this frame's own colour there; this frame's sample where one falls on the pixel (60 % of it: the ground's light is traced with a few rays a pixel, and a pixel is the mean of its last few samples); 0.6 % of this frame's plain enlargement, so an error does not stay.
- The compose pass takes its colour from the kept picture; depth, outlines and motion stay this frame's. Pixels that come from the kept picture are marked as the full-resolution terrain marks its own (`gbuffer1.b` 0.7); only pixels enlarged from this frame alone keep the 0.6 that makes `rr_guides.comp` hand Ray Reconstruction a flat guide.

Measured (close view at home, 3440 x 1400 at render scale 50 %, 14 h, still camera; mean difference from the full-resolution picture, of 255 - two runs of the full-resolution picture differ by 1.1 - 1.4):

| | sand near | sand far | paving near | paving 10 m off | terrain pass |
|---|---|---|---|---|---|
| 50 %, plain enlargement | 8.5 | 4.6 | 12.6 | 18.4 | 7.5 ms |
| 50 %, first version (kept picture of the render size) | 7.5 | 4.4 | 10.4 | 13.8 | 7.5 + 0.4 |
| 50 %, kept picture twice the size | 5.7 | 3.6 | 6.5 | 9.3 | 7.5 + 0.44 |
| 75 %, plain enlargement | 7.8 | 4.1 | 9.6 | 14.2 | 14.6 |
| 75 %, kept picture twice the size | 5.6 | 3.4 | 5.9 | 8.6 | 14.6 + 0.46 |
| full size | | | | | 20.2 - 20.6 |

(The times are with the quality rows as in the player's file that day - sun shadows, bounce light and terrain shadows at their highest; they are higher than the 12.5 ms of the sections above.) Change from one frame to the next, still camera: 1.0 - 1.5 of 255 on sand and paving in all three of full / plain / kept - the kept picture does not flicker. With the kept picture, 75 % is hardly nearer the full picture than 50 % and costs twice as much: the High level could go to 50 % (not changed).

In a steady turn (0.1 rad/s, two rendered pixels a frame): no trails or blocks; the paving keeps its stones and joints, the sand's grain is softer than at full size and a little sharper than the plain enlargement.

Known:
- *Light that changes quickly shows the round.* The shadow of grass swaying in the wind on sand: the sixteen kept pixels under a low-resolution pixel hold sixteen different moments, and the shadow comes out in a pattern four screen pixels wide instead of blades (the plain enlargement shows it as blobs; neither is the full picture). Tried without effect: limiting the history round the plain enlargement instead of round the four neighbours' own range, and going over to this frame's enlargement where the history was cut off (both are in). Also tried, and taken out again (2026-11-24): a measure of change in the compose pass - each of this frame's samples against last frame's kept picture at the very place it falls on (the same point of the ground a round ago, so fine detail does not count), only where the ground stands still, the kept picture itself left alone. It cannot tell a moved shadow from noise: on still, evenly lit sand a fresh sample differs from the kept one by 15 - 25 %, because the ground's light is traced with a few random rays a sample. A threshold above that misses most real change; one below it flags a tenth of all ground pixels at random. It would take the ground's light without its noise (the sun's shadow term kept apart from the traced bounce light) to compare. (Whether that difference is the light's noise was then tested from the other side: with a sample counting for 20 % instead of 60 % - each kept pixel the mean of many more - the kept picture was no nearer the full one, sand 6.1 against 5.7. So it is not simply noise frozen into the kept picture, and what the two samples of one point differ by is not explained.)
- An outline of ground against ground (a near bump in front of far ground) is this frame's low-resolution picture, never the kept one.
- The validation layer reports once that binding 36 is in the colour-attachment layout: the resolve pass writes one of the two kept pictures while both are named in the descriptor set. The shader reads only the other one.
- Rain and snow (same close view, kept / plain, of 255): rain - sand 5.0 / 8.1, paving near 3.7 / 8.2, paving 10 m off 7.2 / 14.2; snow - 0.2 - 4.5 / 0.2 - 8.9, to the eye the same picture as full size. Walking: one capture, mostly tall grass, nothing wrong in it - not a real test.

How it was found, for the next time a low-resolution pass "looks different": the three captures (full, 50 % plain, 50 % kept) of the same held view, cropped at the same place and enlarged eight times, showed blocks exactly one RENDERED pixel in size - the low-resolution picture's pixels are twice that. A debug colour in the compose pass (not in the resolve pass: what that writes is its own history next frame, so the colour feeds itself) showed that nearly every ground pixel did come from the kept picture.

Capture recipe: a copy of `vibe_settings.cfg` with `terrain_raymarch_scale = 0.5`, `terrain_rm_temporal = 1`, `time_of_day = 14`, `day_night_enabled = 0` and the close view of `tools/perf/bench.sh`, run with `VIBE_CFG=<copy> VIBE_TEST_HOLD=1 VIBE_TEST_GROUND=1 VIBE_NO_NET=1 VIBE_TEST_SIZE=3440x1400 build/vibe.exe --frames 330 --screenshot x.png --screenshot-after 300` (`VIBE_TEST_PAN=0.1` and a yaw 0.5 rad less for the turn that ends on the same view). The `gfx_*` rows in a cfg copy do NOT set the settings they stand for at load: set the settings themselves.

**Fast ground goes back to the plain enlargement** (`terrain_rm_resolve.frag`, `rush`). Ground that comes toward the camera grows in the picture from frame to frame, and what was kept of it is stretched along. Driving at 72 km/h the road came out in blotches the size of a low-resolution pixel - worse than the plain enlargement. The resolve pass measures how much the picture stretches at a pixel in one frame (the motion's change over one low-resolution pixel): up to about 1 % the kept picture holds, from 4 % on nothing is kept there (alpha 0: the compose pass enlarges this frame's picture itself and marks the pixel as it does without this pass); likewise from 5 - 10 low-resolution pixels of motion a frame. At 72 km/h that is: kept beyond some 30 m, plain inside 8 m. A turning camera stretches nothing, and the still view and the turn measure as before.

More views, same day (difference from full size, of 255; kept / plain): night at 23 h - 0.4 - 0.9 / 0.5 - 1.9; the far view of `bench.sh` - paving 7.1 / 15.0, sand 5.3 / 8.5, the hillside 3.0 / 4.8, the mountains 5.3 / 6.5. Driving the course on autopilot (`VIBE_TEST_JEEP=1,9 VIBE_TEST_JEEP_AT=1689.5,837.2,3.14159`, no `VIBE_TEST_HOLD`): the same picture as the plain enlargement, as intended. Walking was not captured (the rule above covers it: plain inside about 2 m).

## The low-resolution terrain starts at the march too (2026-11-24)

Seen while driving with the terrain below full size (plain or kept, and at 98 % as at 50 %): a tan strip of far field drawn across the wood beside the highway, IN FRONT of the trunks. A debug colour by distance (compose pass) put the strip about 100 m from the camera; from 15 m up there is no such bank, in either mode. With `terrain_rm_prepass = 0` the full-resolution picture shows the same strip.

So it is the plain march, not the low-resolution path: a ray that skims ground falling away from the camera creeps along it in ever smaller steps and, started at the eye, is taken for a hit far too early (careful stepping - `terrain_rm_converge = 1`, step factor 0.3 - makes the strip nearly vanish; the distance LOD has nothing to do with it). The full-resolution pass has been safe since it starts where the low-resolution march got to. The low-resolution terrain pass never did.

- The march pre-pass now runs ahead of the low-resolution terrain as well (`frame.cpp` "Pass 0.45", before "Pass 0.5"; `tr_march_prepass_on()` no longer excludes it), and that pass starts its rays there (`terrain_raymarch.frag` `lr_start`, `scene.rm_perf.x` bit 32; the place in the march's picture comes from `vNDC`, since this pass's pixels are not the picture's). The shadow terms, the start at the hit and the shading rate stay with the full-resolution pass.
- Ahead of the low-resolution terrain the march is always the coarsest (4 x 4 pixels: `init_terrain_raymarch_lowres`). At 2 x 2 it had as many pixels as the pass it serves: 1.7 ms against 0.7 saved. Highway view, 50 %: march 0.83 + terrain 2.92 ms, against 3.85 ms for the terrain alone before; close view 0.39 + 6.9 against 7.5.
- The low-resolution pass's rays are built from the camera-relative matrix, as the full-resolution pass's (`render_terrain_raymarch_lr`). It did not cause the strip; it is what the march's distances are measured against.
- Still true with the pre-pass switched off (`terrain_rm_prepass = 0`, no quality level does that): the strip, in both paths.

## Ground turned away from the sun, in an object's shadow (2026-11-24)

`terrain_rm_main_light.glsl` (the call of `rm_rt_sun_shadow`). Seen on the sand at home at 18:54, the sun 2 - 3 degrees up behind the wood: a ragged pattern of light and dark over the ripples, pixel by pixel, the same on every frame.

It was not a shadow's edge. The whole of the sand lay in the wood's shadow. The ray-traced shadow of objects does two things to the ground: it takes the sun away, and it takes 70 % of the sky's light with it (`terrain_rm_main_gi.glsl`: under a canopy the sky is covered too). It was only asked for where the ground faces the sun (`dif > 0`). Ground turned away from the sun kept all of its sky light - and came out BRIGHTER than the shadowed ground beside it. At a low sun the sand's fine normal turns to and from the sun from pixel to pixel.

Now asked for whichever way the ground faces. The same view: evenly shaded sand. At midday the picture is the same (1.1 - 1.4 of 255 from before, what two runs differ by) and so is the time (terrain pass 20.7 ms against 20.2 - 20.6; dusk 16.8 against 16.5).

How it was found: `VIBE_RM_SKIP` 4 / 8 / 32 / 1024 in turn - only 32 (object shadows) took the pattern away; then a colour for the blocker's distance (all of it over 15 m: the wood, nothing near); then the same picture with Ray Reconstruction off and with the softness at a sixteenth - so no ray's noise, and nothing the rays' directions decide. What was left was which pixels ask at all.

The cost in a wood at midday, where many more pixels are both turned from the sun and near trees (`VIBE_RM_SKIP=65536` asks only where the ground faces the sun, as it was): the forest view of `bench.sh` 5.58 against 5.58 ms, the same place looking down at the forest floor 10.7 against 10.6 - nothing.

## The far edge of a meadow came out in steps (2026-11-24)

`terrain_rm_heightfield.glsl raymarch_lo` (`t_grass`), read by `grass_raymarch.frag` ("Skip the open air"). Seen from the meadow by the highway looking at the wood, with the terrain at full size only: the line where the grass ends against the trees was notched in rectangles, the size of the march pre-pass's blocks.

The low-resolution march hands the grass pass the place to start from: the last point at which the block's ray was still more than a blade's height above the ground. It went on counting after the ray had first come lower. A ray that skims the crest of a meadow has the ground fall away under it beyond the crest and is "high" again there - so the grass march started behind the blades standing on the crest, for the whole block. Now the point is fixed the first time the ray comes lower.

Cost: nothing measurable (grass pass: meadow view 5.56 -> 5.67 ms, close view 3.40 -> 3.04, forest view 3.49 -> 3.41).

## The outline against the sky under the temporal upscaler (2026-11-24)

`terrain_rm_resolve.frag` and `terrain_raymarch_compose.frag` ("The outline"), `frame.cpp` (`tr_->cam_still`). Reported from play at "Terrain resolution" Medium: the edge of the snow mountains against the sky flickers, standing still.

At an outline the compose pass took this frame's low-resolution picture - and that picture lies at another of sixteen places inside its pixel every frame, so the outline was drawn sixteen different ways a round (frame-to-frame change along the ridge 2.0 of 255, against 0.9 in the open sky and 1.5 for the plain enlargement, which is blocky but steady).

While the camera stands still (the view matrix as last frame's: `cam_still`, `pc.tex_params.w`) the kept picture now holds the outline too. Its alpha says ground (the distance), sky (-1) or not known yet (0). At an outline a kept pixel is what the last sample that fell ON it said, and is left alone by every frame whose samples fall elsewhere; after a round the outline stands, at the kept picture's resolution. The compose pass lets the four kept pixels round its own decide - ground where they cover half of it or more. Measured: 0.9 along the ridge, the same outline on consecutive frames, and finer than before.

Only with a still camera: the sky has no motion vectors to carry its kept pixels along by. Turning or walking, an outline is this frame's low-resolution picture as before (it moves anyway). An outline of ground against ground (a near hill before a far one) is also still this frame's: mixing the distances of two surfaces would put objects behind the near one in front of it.

Not known: whether the player's view is exactly still when standing in a real game (it is in a held test view); if something moves it by a hair every frame, none of this takes effect.

## "Simplest shading": a measuring mode (2026-11-24)

Graphics tab, the checkbox at the top; `VIBE_TEST_SIMPLE=1` for a test run. `gfx_quality.cpp set_simple_shading`, `scene_ubo.glsl FLAT_SHADING` (bit 1 of `scene.rt_flags2.y`) and `flat_light()`.

Everything is still drawn, at the same resolution - the same ground, trees, grass, water - but flat: a surface is its own colour times one light for the whole picture (half sun, half sky; no direction, no shadow, no rays, no haze). What a frame costs then is geometry, marching and the upscaler; the difference to the normal picture is what the shading costs.

- *Settings*: put aside when it is switched on and brought back when it is switched off (`simple_saved_rt_`); while it is on nothing is saved. Off in it: ray-traced shadows, occlusion, bounce light, reflections, the light in the air, clouds, bloom and flares, grass shadows, the surface relief, Ray Reconstruction.
- *Meshes, trees, cars, small plants* (everything in the G-buffer): `deferred_lighting.frag` returns colour x light right after the G-buffer is read. The G-buffer pass itself (`cube.frag`: textures, leaf cards) is as it was.
- *Ground and water*: the first-start build of the terrain shader (`terrain_raymarch_lite`), with the flat light and no haze. Its low-resolution pipeline is now always built, not only when the game starts with the terrain below full size.
- *Grass*: `grass_raymarch.frag` leaves out its lighting (the shadow map, the blade shadows, side light, translucency).

Measured (3440 x 1400, render scale 50 %, 14 h; normal -> flat, ms):

| view | frame | terrain | deferred lighting | upscaler | grass |
|---|---|---|---|---|---|
| close, terrain at 75 % | 28.9 -> 17.5 | 14.1 -> 6.1 | 1.6 -> (none logged) | 2.5 -> 1.3 | 3.5 -> 3.5 |
| close, terrain at full size | 38.9 -> 19.5 | 21.9 -> 7.8 | 1.7 -> - | 2.7 -> 1.3 | 3.9 -> 3.5 |
| forest, terrain at 75 % | 27.0 -> 14.3 | 5.8 -> 1.7 | 6.8 -> - | 2.5 -> 1.3 | 4.1 -> 4.2 |

What it says: two thirds of the ground's time is its shading, one third (6 - 8 ms in the close view) finding the ground at all; the grass pass is all marching - its lighting costs nothing that can be measured; the trees' drawing (3.3 - 4 ms for depth and colour passes together) is geometry and does not change; half of the upscaler's time is Ray Reconstruction.

Not tried: switching it on and off from the menu in a running game (only started with the test variable).

## The shading test: drop-downs per part of the picture (2026-11-24)

Graphics tab, the first section; `VIBE_TEST_SHADE="ground=0,water=1"` for a test run (`VIBE_TEST_SIMPLE=1`: everything plainest). `gfx_quality.cpp` (`shade_rows`, `apply_shade_test`), `scene_ubo.glsl` (`FLAT_OBJECTS` / `FLAT_GRASS` / `FLAT_GROUND`, bits 1 - 3 of `scene.rt_flags2.y`), `shade_rm_skip_` (sections of the terrain shader left out: the `RMDBG` bits). It replaces the single "simplest shading" switch of the section above; what that section says of "flat" still holds.

Measuring switches, not quality levels: one part of the picture is made plain or taken out while the rest stays as it is. The player's settings are put aside when the first drop-down leaves "full" and come back when the last returns to it; nothing is saved meanwhile. Every change starts again from the settings put aside, so the order the drop-downs are used in does not matter.

| drop-down (key) | levels | how |
|---|---|---|
| Ground: light (`ground`) | flat / simple / no shadows or bounce light / full | flat, simple: the first-start shader (`terrain_raymarch_lite`); "no shadows": `RMDBG` 4, 8, 16, 32 and no bounce rays |
| Ground: stones (`stones`) | off / simple / full | `RMDBG` 2 + 32768 / 32768 |
| Ground: close-up (`detail`) | off / full | `RMDBG` 1 + 128 + 1024, no relief on walls |
| Water (`water`) | flat / simple / full | `RMDBG` 131072 (new: `terrain_rm_main_water.glsl`) / no ray-traced reflections, no shadows on it |
| Objects: light (`objects`) | flat / sun and sky only / full | `FLAT_OBJECTS` / shadows, occlusion, bounce light, reflections off |
| Grass (`grass`) | off / flat / no shadows / full | not drawn / `FLAT_GRASS` / its shadow settings |
| Trees (`trees`) | off / full | draw distance 0 |
| Sky and air (`sky`) | off / full | clouds, mist, bloom, flares |
| Denoiser (`denoiser`) | off / on | Ray Reconstruction |

Measured one at a time (close view at home, 3440 x 1400, render scale 50 %, terrain at 75 %, 14 h; full: frame 28.9 ms, terrain 14.0, grass 3.4, deferred lighting 1.6, upscaler 2.5):

| changed | frame | what moved |
|---|---|---|
| ground: no shadows or bounce light | 26.5 | terrain 11.8 |
| ground: simple | 21.9 | terrain 7.5 |
| ground: flat | 22.2 | terrain 7.4 |
| stones: simple / off | 28.6 / 28.4 | terrain 13.8 / 13.4 |
| close-up: off | 23.8 | terrain 9.1 |
| objects: sun and sky only | 26.0 | deferred 0.2, terrain 12.7 (the objects' shadows on the ground go too) |
| objects: flat | 26.4 | deferred gone |
| grass: no shadows / flat | 28.4 / 28.4 | nothing (grass 3.4) |
| grass: off | 25.0 | grass gone |
| trees: off | 28.3 | their passes 3.3 -> 2.3 |
| sky and air: off | 28.4 | clouds 0.3 |
| denoiser: off | 34.4 | SLOWER: upscaler 2.5 -> 1.2, but without it the lighting traces its full ray counts (deferred 1.6 -> 7.1, terrain 14.0 -> 15.5) |

What it says for this view: the largest single thing in the ground's time is the close-up detail (textures and relief: 5 ms of 14); its shadows and bounce light are 2 ms; the stones 0.6. The grass costs what its marching costs. Ray Reconstruction pays for itself several times over.

Not verified: the water drop-down (no water in the views tried: three runs showed no change in picture or time), the stones in a picture (the view's sand has none - only the time moved), and using the drop-downs from the menu in a running game (measured with the test variable only).

## Libraries updated (2026-11-24)

`CMakeLists.txt`, the `FetchContent` pins: glfw 3.4 -> 3.5.1, glm 1.0.1 -> 1.0.3, vk-bootstrap 1.3.295 -> 1.4.341, VulkanMemoryAllocator 3.1.0 -> 3.5.0, Catch2 3.7.1 -> 3.16.1, JoltPhysics 5.2.0 -> 5.7.0, imgui 1.91.5 -> 1.93.0, tinygltf 2.9.3 -> 2.9.7, tinyexr 1.0.8 -> 3.4.0, miniaudio 0.11.21 -> 0.11.26, meshoptimizer 1.0 -> 1.3. FastNoiseLite 1.1.1, enet 1.3.18 and ozz-animation 0.17.0 were the latest already.

- *vk-bootstrap* follows the Vulkan headers' version: 1.4.341 is the one for the installed SDK (1.4.341.1). A newer one wants a newer SDK.
- *tinygltf stays on 2.x.* 3.x is a rewrite with a C API (`tiny_gltf_v3.h`); `tiny_gltf.h` and the `stb_image.h` it carried - which `skybox.cpp`, `frame.cpp` and others include from its folder - moved to an `attic/`. Going to 3.x means rewriting `gltf_loader.cpp` and bringing stb in on its own.
- *imgui 1.92.4+*: the Vulkan backend's pipeline fields moved (`ui_setup.cpp`: `ii.PipelineInfoMain.MSAASamples`, `.PipelineRenderingCreateInfo`). Its `ImGui_ImplVulkan_AddTexture(sampler, view, layout)` now ignores the sampler (the HUD's weapon icons: looks the same).
- After trying a tag and going back (tinygltf 3.1 -> 2.9.7), `build/_deps/<lib>-build` kept the tests the abandoned version had registered and `build.ps1 -Test` stopped at the first of them ("Could not find executable ... tester_v3_c"): delete that folder.
- The vendor SDKs and the Vulkan SDK: the next section.

Checked: all 129 tests, a run of the close view (no errors in the log), the pause menu drawn.

## The sun's shadows: where the rays' time is (2026-11-24)

Asked: review the shadow logic and make it faster. Measured first (3440 x 1400, render scale 50 %, terrain at full size, every quality row at its highest, Ray Reconstruction on; close view / forest view):

| left out | ground pass | objects' lighting pass |
|---|---|---|
| nothing | 18.4 / 5.2 ms | 1.7 / 6.8 ms |
| the ground's own shadow (`VIBE_RM_SKIP=4`) | 18.5 / 4.9 | |
| grass blades' shadow on the ground (8) | 17.8 / 4.8 | |
| the small plants' shadow map on the ground (16) | 18.3 / 5.2 | |
| every ray on the ground (32: object shadows, occlusion AND bounce light) | 15.9 / 3.7 | |
| bounce light on the ground only (`terrain_gi_samples_cap = 0`) | 16.6 / 4.2 | |
| ray-traced occlusion (`ao_mode = 0`) | 17.9 / 4.9 | 1.5 / 6.0 |
| ray-traced sun shadows (`shadow_enabled = 0`) | 17.7 / 5.1 | 1.2 / 4.4 |
| bounce light everywhere (`gi_samples = 0`) | 16.6 / 4.1 | 0.6 / 1.8 |

So the shadows are not where the time is. On the ground all the shadow terms together are under 1 ms (the plants' shadow map is drawn in a pass of its own: another 0.5 - 1.1 ms); on objects the sun's shadow is about 2 ms in a wood. The rays' time is the bounce light: 1.7 ms on the ground close by, and 4.6 of the 6.4 ms of the objects' lighting in the forest (with "Bounce light" at its highest: full bounce light on leaves).

Trap: `VIBE_RM_SKIP=32` is not "object shadows" - `do_rt` in `terrain_rm_main_light.glsl` also gates the occlusion rays and the bounce light.

Changed: under Ray Reconstruction the search for a shadow's blocker takes one ray, not two (`deferred_lighting.frag` `n_block`, `terrain_raymarch.frag` `kBlocker`). The second was a third of the sun's shadow rays, spent on halving the noise of a distance that only sets the penumbra's width. Objects' lighting in the forest 6.8 -> 6.35 ms, close 1.74 -> 1.60; the ground unchanged (its rays were never the cost). Pictures: the same within what two runs differ by (forest floor 5.4 of 255 against 8.3 between two runs of the new build - the leaves move).

Read and left: the ground's own shadow (one walk over the baked heights since 2026-11-23: nothing left to take), the plants' shadow map (two layers drawn every frame, 0.5 - 1.1 ms: it could be drawn less often, but the plants sway), the baked castle shadow (34 texture fetches per mesh-terrain pixel; not on the raymarched ground).

## Vendor SDKs updated (2026-11-24)

- *NVIDIA Streamline 2.14.16 / DLSS 310.9.16, FidelityFX-FSR2 2.2.1*: the newest already (FSR2's own repository ended at 2.2.1). Not touched.
- *AMD FSR SDK 1.1.4 -> 2.4.0* (the `external/FidelityFX-SDK` submodule; the first 2.x with Vulkan on a par with DX12). What changed for us:
  - Layout: headers in `Kits/FidelityFX/{api,upscalers,framegeneration}/include` (included without the `ffx_api/` prefix; the swap chain's structs are in `vk/ffx_api_framegeneration_vk.h`), DLLs in `Kits/FidelityFX/signedbin`.
  - One DLL became three: `fsr3.cpp` loads `amd_fidelityfx_loader_vk.dll`, which picks `amd_fidelityfx_upscaler_vk.dll` / `amd_fidelityfx_framegeneration_vk.dll`. All three are copied next to the exe (`FFX_SDK_DLLS` in `CMakeLists.txt`); `build/amd_fidelityfx_vk.dll` is the old one and can go.
  - `ffxCreateBackendVKDesc`: device, physical device and proc-addr moved into `.deviceContext`.
  - `ffx_api_loader.h` resolves the entry points only with `_WINDOWS` defined (`fsr3.cpp` defines it).
  - `screenshot.cpp` took `stb_image_write.h` out of the old SDK's sample framework; it now takes tinygltf's copy, like `stb_image.h` everywhere else.
  - Checked on the NVIDIA card with `fsr_backend = 1`, `upscaler_auto = 0`: the upscaler context, the frame-generation context and the swap-chain proxy all come up, the picture is right, no errors. Not checked: on an AMD card (where the loader may pick FSR 4), and the frame times.
- *Vulkan SDK 1.4.341.1 -> 1.4.363.1*, vk-bootstrap v1.4.341 -> v1.4.363 to match. Installed beside the old one without admin rights (`vulkansdk-windows-X64-1.4.363.1.exe --root C:\VulkanSDK\1.4.363.1 --accept-licenses --default-answer --confirm-command install copy_only=1`), which sets no environment variable: the *user-level* `VULKAN_SDK` (and `VK_SDK_PATH`) point at it, and `build.ps1` now reads the user-level variable before the machine-level one. Back to the old SDK: delete the user-level variable, delete the `Vulkan_*` lines' values from `build/CMakeCache.txt` (or `-Clean`), vk-bootstrap back to v1.4.341. A changed `glslc` recompiles every shader: the first start after it is the long one.
- Checked after both: all 129 tests, the close view with DLSS + Ray Reconstruction and with FSR + frame generation (no errors, `vk_err=0`, pictures as before). Frame times not compared - another game was using the graphics card during the runs.

## Bounce light from the cache (2026-11-24)

`shaders/irradiance_cache.glsl`, `deferred_lighting.frag` (the bounce-light block), `rc_resolve.comp`, `radiance_cache.cpp`; setting `irradiance_cache` (default 1; Graphics, "Bounce light from the cache"; needs `radiance_cache`); `scene.rt_flags.x` bit 5.

*Before.* The world-space cache held what a surface sends OUT (outgoing radiance), and a bounce ray read it where it landed. Every pixel still traced its own rays each frame, and then ran a 13-tap history filter over the result.

*Now.* The same hash grid also holds what ARRIVES at a cell (irradiance, and the part of the rays that reached the sky). A pixel whose cell has a settled value takes it and traces nothing. One pixel in sixteen of such a cell still traces and feeds the cell. A cell without a value - new in view - has all its pixels trace, exactly as before, so the worst case is the old cost.

- *An entry* has two slots in `acc` and `res` now (bit 1 of the key marks an irradiance entry). Second slot: x = sky visibility, y = the number of samples the value rests on. `rc_resolve.comp`: a running mean while it settles, then a slow follow (alpha at least 0.02, count capped at 2048); used from 32 samples on (`kIcMinSamples`).
- *Six entries per cell as an ambient cube*: a sample goes to one of the three axes its normal has a part in, picked with probability n_axis squared, and is read the same way. With "the largest axis" a round trunk would show a seam where its normal changes axis.
- *The read is moved by up to half a cell at random* (and falls back to the pixel's own cell): over pixels and frames that is the trilinear blend of the cells around, for one look-up instead of eight.
- *Under Ray Reconstruction* the cell's value goes out as it is and the history filter is skipped (the history images are still written, for the frame the pixel has to trace again). *Without it* (FSR) the value goes through the history filter as a traced one did: the rays are saved, the filter is not.
- `IC_SALT`: a second shader whose bounce light is another quantity gets its own entries.

Measured (`tools/perf/bench.sh`, 3440 x 1400, render scale 50 %, every quality row at its highest, Ray Reconstruction; the objects' lighting pass, off -> on): close 1.69 -> 0.77 ms, far 2.49 -> 1.30, forest 6.67 -> 2.87. Frame: 34.2 / 24.0 / 26.0 -> 33.6 / 22.5 / 22.3. On the FSR path in the forest: 16.7 -> 10.2 ms.

Pictures, off against on: the castle's courtyard 1.2 - 1.6 of 255 apart, a dark room in the castle 0.8 - 1.1 (no light through the walls: the six entries a cell keep a wall's two faces apart, and the occlusion rays still give the contact darkening), the forest the same but for the plants' sway, a steady pan clean.

Not checked: light that changes quickly near objects (the torch switched on, a lantern, a door opening) - the cell follows over ten or more frames where a traced pixel followed in a few; night; the default quality level (measured at the highest only).

**The terrain's bounce light is NOT from the cache - tried and taken out.** The ground traces its own bounce rays in `terrain_rm_main_gi.glsl`. Reading and feeding the same cache there worked (every ground pixel in the close view read its cell) and saved 0.6 ms of rays - but the terrain pass got 1.1 ms slower from the code being in the shader at all: with its bounce light switched off it took 17.65 ms with the cache code compiled in against 16.58 without. Read-only buffers (no atomics), two probes instead of eight and a single look-up made no difference to that. What in the driver's compile of this very large shader it is, is not known. So: the ground's rays are 0.5 - 0.6 ms of the 1.7 ms its bounce light costs in the close view, and the cache cannot be had there for less than it saves. (If it is tried again: feed the ground's cells from the objects' lighting pass, which has the cache already, and give the terrain shader nothing but one texture to read.)

## What the close-range relief costs, and two things that did not make it cheaper (2026-11-24)

Asked: the ground's close-up detail. Measured in the close view (3440 x 1400, render scale 50 %, terrain at full size, every quality row at its highest; terrain pass 18.3 ms), one `VIBE_RM_SKIP` bit at a time:

| left out | terrain pass |
|---|---|
| nothing | 18.3 ms |
| close-up textures (1) | 18.1 |
| stones in the normal (2) | 17.4 |
| screen-space normal (512) | 18.7 |
| **the close-range relief, all of it (1024)** | **11.75** |
| the relief in the march only (128) | 15.4 |
| ... and one tap of it in the normal, not three (128 + 1048576) | 14.35 |
| the normal's relief from the main rills alone (524288) | 18.2 |

So the "close-up detail" of the shading test is one thing: the relief (`rm_ground_detail`, `terrain_rm_heightfield.glsl`), 6.5 ms of the pass in this view. Of that:

- *2.8 ms in the march*: over ground with rills and stones on it the ray takes more, smaller steps within the relief's reach (0.34 m over the landscape), and each of them works the whole landscape formula out again.
- *1.1 ms goes with the two outer taps of the relief in the normal* - but not for their arithmetic (below): with them gone the normal is the landscape's alone, and what follows in the shader is cheaper for a smooth normal. Which part of what follows is not found yet.
- *2.6 ms more* between "one tap in the normal" and "no relief": the same question.

`RMDBG` bits added for this: 524288 and 1048576 (above). Bit 128 changed: it now leaves the relief out of the march before any of its work (it used to work the road distance and two noises out first and throw them away, which measured the relief's effect on the step count but not its cost).

**Tried, measured, taken out again:**

- *The normal's three relief taps together* (the road and highway asked once, each noise's lattice corners hashed once for all three; the same values): 18.03 - 18.27 ms against 18.14 - 18.18 with the taps one by one. Nothing. The taps' arithmetic is not what the 1.1 ms is.
- *The march reusing the landscape between steps* (the landscape under the relief taken from the straight line through the last two steps that worked it out, for up to two steps running; a step that would end the march done again with the formula): 22.1 ms against 18.4, and the picture 7.5 of 255 apart on the sand (two runs of one build: 1.2). The landscape's fine octaves are not a straight line over the distance of a step, the line's error sends steps into the ground, and those are done twice.

**Not tried** (what is left for the 2.8 ms): march the landscape without the relief to its hit, then find the relief's surface from there as a relief-mapping search over the local plane - no landscape formula per step at all. It replaces the last stretch of `raymarch()`, which is tuned (fast final approach, the secant search); every view that went wrong there before has to be looked at again.

For the remaining 3.7 ms the next step is finding it: `RMDBG` bits on the sections that follow the normal (the blades' shadow march, the stones, the sand and its ripples, the cavity term) with bit 1048576 set and unset.

## The clouds in a big world (2026-11-24)

Asked: "the cloud system is a little broken; our world is so big, make it right for it, and fast". What was read and seen (`shaders/clouds.comp`, `atmosphere.cpp dispatch_clouds`):

1. *The cloud field was sampled in the engine's coordinates.* Those are re-centred as the player travels (near charts shift by whole 6400 m tiles, far charts are turned): the weather cells (6500 m x size) and the shape texture (11000 m x size) are not periods of that, so every re-centring put other clouds in the sky. The painted clouds seen from orbit (`compose.frag weather_cloud`) use home coordinates - the two only agreed in home's chart.
2. *The layer's curve to the horizon was taken from the engine's origin*, which is 12 km from the player in a near chart: the sky's "top" stood over a point kilometres away.
3. *The cloud shadow map covered 6144 m round the engine's origin.* Beyond 3 km from home, and everywhere in a near chart (the player stands at local (0, 12288)), there were no cloud shadows at all.
4. *Clouds were drawn on sky pixels only* (from below the layer): a cloud in front of a mountain that stands above the cloud base was cut out along the mountain's outline.
5. *An empty sky cost as much as a cloudy one*: every sky pixel walked its 176 steps through nothing (2.1 ms at 3440 x 1400 with the coverage at 0.15).
6. *Contour bands inside a cloud's shaded side*: the four light taps toward the sun sat at fixed distances.

What was changed:

- **The field is a function of the place on the planet.** `atmosphere.cpp` works out the camera's home coordinates (`globe::home_az(globe::to_planet(chart, x, z))`, double precision) plus the wind's travel, and hands the shader pieces that are exact in a float: the weather lattice's cell and the place in it (`wx_a`, `wx_b` for the second octave), the shape texture's coordinate at the camera (`shape0`), and how the engine's axes lie in home's (`jmat`, from differences of the same map). `clouds.comp` adds the small offset from the camera (`home_off`). The weather hash takes whole cells (`wx_hash`, the same numbers as `compose.frag wm_hash`). Checked: the same place from home's chart and from a near chart (`VIBE_TEST_NEAR_CHART=1`, 4.2 km out, wind 0) - the same clouds, cloud for cloud.
- **Curvature round the camera**: `altitude()` and `hit_shell()` take the planet's drop-off from the camera's ground point.
- **The cloud shadow map wraps and follows the camera**: 16384 m (was 6144), texel (i, j) holds the point congruent to it modulo the extent that is nearest the camera (`shadow_map()`); `scene_ubo.glsl cloud_shadow()` reads it with a repeating sampler at `g / kCloudShadowExtent` and fades the shadows out toward the map's edge (keep `kCloudShadowExtent` in step in `scene_ubo.glsl` and `atmosphere.cpp`). No origin has to be passed: a point keeps its texel while the camera moves.
- **Clouds in front of high ground**: from below the layer, a land pixel whose point lies above the cloud base is marched too, up to the land (`t_land`); the clouds-over-land pass (`frame.cpp`, `compose.frag CLOUDS_OVER`) now runs from below the layer as well and discards every pixel the cloud pass left empty.
- **Strides through empty weather**: `sample_cloud` returns -1 where the weather map has no cloud at any height; the march then strides half a kilometre and more, and the step that finds weather after such a stride goes back and walks it at the normal step (`stride_back`, `walk_until`).
- **The light taps are jittered** within their stretch, per pixel and frame.

Checked: clouds at home as before (same shapes in the same places); the two-chart comparison above; shadows on the ground at home with the wrapping map (clouds on against off: 17 - 22 of 255 on the land in view); no errors in any run; the bands inside a cloud fainter (not gone: the view steps' own banding is left to the upscaler, and these captures were 1280 x 536).

**Cloud shadows, settled the next round (same day):**
- They work far from home: at (-300, -5200), with many small clouds (`cloud_scale` 0.3), cloud-shaped patches lie on the ground in the map's view (`VIBE_RM_SKIP=2097152`: the cloud shadow on the ground, white = open sky - a new view, `terrain_rm_main_gi.glsl`).
- Why a first on/off comparison there showed nothing: with the player's own clouds (size 1.27, towers of 7 km) a cloud's shadow is displaced by its height over the sun's slope - kilometres - and none fell in the views taken. A dark count of pixels was no measure either (it counted trees). Lesson: test cloud shadows with small, low clouds.
- *The shadow now uses the cloud as it is drawn*: `shadow_map()` samples with the detail erosion (`sample_cloud(p, true)`) at 32 heights (was 12, without the detail). Without the erosion every wisp of the base shape counted as cloud: one shadow covered its whole weather cell - the ground in view was all in shadow or all out of it, never patterned.
- The lookup takes `fract()` of its coordinate itself (right whatever a sampler does outside 0..1).

**The speed** (3440 x 1400, the `sky_clouds` mark, the launch-site view): a clear sky (coverage 0.15) 2.09 -> 0.11 ms; "Sunny, cloudy" 0.10 - 0.15 ms now (not measured before the change).

**Not confirmed:**
- *Clouds in front of a mountain*: with low small clouds round the summit there is no fault to see, but no capture shows a cloud plainly between the eye and the flank.
- Above the layer, the flight down from orbit, night, rain decks: not looked at after the change.

Seen on the way, not the clouds': from a held camera 900 m up at (-300, -5200) the land is washed out to pale blue-white with the clouds switched off too.

## Cars are solid for a walker (2026-11-24)

`jeep.cpp jeep_player_colliders()`, called at the end of the per-tick collision list (`props.cpp`). The player collides with axis-aligned boxes; a car stands at any angle. So each of a car's collision boxes is cut along its length into pieces about as long as it is wide and each piece's own axis-aligned bound goes into the list: a stair-stepped outline at most a hand wider than the car. Not the car the player sits in (driving, or riding beside another driver). Until now a car was nothing to a walker.

Checked: `VIBE_TEST_WALK=1` (new: the forward key held, the player's place logged once a second) from 12 m in front of the parked jeep at (1638.5, 727.6): the player stops at z = 729.65 and stays there - 2.05 m from the car's centre. Getting in reaches 3.6 m from the centre, so it still works from where the player is stopped. Not tried: a moving car hitting a walker (the walker is an obstacle to nothing: the car's physics does not know of the player), standing on a car, getting out against a wall.

## The land from the air: no longer washed out (2026-11-24)

Seen while testing the clouds: from a camera held 600 m up the land was pale blue-white, from 1500 m a white sheet (snow patches only). Not the clouds, the mist, the ground fog or the distance fog (each switched off: the same picture); with the ground's shading flat the land was there. Two causes:

- *`ground_haze.glsl ground_haze_air()`*: the haze laid over far ground took the colour of the sky just over the horizon for EVERY ray - also one that goes down. That band is the brightest part of the sky (a line of sight through hundreds of kilometres of air). Now a ray that goes down at an angle takes the sky's colour from the same angle up (about as much air), and of that only the share that lies below the camera (`1 - exp(-height / 3000)`; blended in over the first 0.25 of downward slope, so a view along the ground is as it was). Shared with the trees and buildings (`deferred_lighting.frag`).
- *`atmosphere.cpp`, the haze's stretch* (`aerial_scale`, 6 in the player's settings: the game's distances are short, so the air acts as six times as much) is for views along the ground. It eases back to 1 between 300 m and 4 km up: through six times the air the land from 1500 m was the sky's blue.

Looked at (1280 x 536, 14 h, clear): 300 / 600 / 1500 / 4000 m up at (-300, -5200) - the land is there at every height, dark green-blue with its snow, bluer with height; from the launch tower at home the far mountain as before (the land below the eye a little less pale: 12 of 255). Not looked at: dusk and night, the way up to orbit (the far field takes over from 25 km), other weather.

## The service station on the coast highway (2026-11-24)

Asked for at the place the player was looking at: a fuel station and a restaurant, modelled in Blender, in high detail, with its own textures. It stands on the meadow west of the highway, 300 m north of the launch pad: the middle of its plot at (1620, 41.8, 545), the road along its east side.

```
blender -b --python tools/gasstation/make_gasstation.py   ->  assets/gasstation/gasstation.mesh, .col, .blend
```

- **The model** (`tools/gasstation/make_gasstation.py`, 204,000 triangles, one group per material `g_*`): a 54 x 72 m plot - asphalt forecourt under a steel canopy on four columns, two pump islands with two dispensers each (displays, card readers, three nozzles and hoses a face), a sloped apron to the road, a price pylon; one long flat-roofed building with a food mart (three shelving runs with end caps, six drinks fridges, an open chiller, pallet displays, a counter with till and a wall of small goods, a coffee bar) and a diner (seven booths at the windows, a counter with ten stools and a back bar, round tables, a kitchen behind a pass with range, griddle, fryers and hood, wet rooms, a jukebox); outside: parking bays with wheel stops, two car chargers, air and vacuum, ice, propane, newspaper boxes, benches, planters, a picnic terrace with parasols, a bin store, four light masts, roof plant, a rooftop DINER sign. Lettering is geometry (`text()`: Blender's font turned into triangles), not a texture. The same helpers as `tools/office/make_office.py`; small disorder from one seeded `RNG`.
- **`gasstation.col`**, written by the script: `box` (208 colliders), `light` (20), `shelter` (the building, its awning, the canopy), `nograss` (the plot and the apron), `pad` (paved ground a car drives on: the asphalt, the pavers, the apron in six strips that follow the road's slope). `place_gasstation()` in `rocket.cpp` reads all of it; nothing about the site is typed into C++ but its place.
- **Its place and its ground**: `kStationX / Z / H` and `kStationRect` in `terrain.h`. The plot is a level terrace in the landscape formula: `wm_station()` in `shaders/world_map.glsl` and `wm::station()` in `terrain.cpp` (keep in step, as `wm_pad`). The rectangle's blend to the natural ground is 22 m on three sides and 5 m toward the road, so that the ground under the road's centre line - which the road takes its height from - is untouched. The woods' mask leaves the plot out. The river's bank and the road's bank still cut into the terrace's edge: a concrete retaining wall goes down 9 m round the plot.
- **Textures**: slots 37..42 (`kTexStation`; `kTextureCount` is 43 in `vk_engine.h` (as `kTerrainMatCount` 36), `cube.frag`, and `kRmTextureCount` in `terrain_raymarch.frag` and `wm_cache.frag`): brick, asphalt, checker tile, ribbed cladding, pavers, wall tile - `assets/gasstation/tex/*.png`, 1024 px, made with an image model (0.75 credits each); loaded with the pavilion's in `world_textures.cpp`, normal map and height from the picture's light and dark. Materials: the `g_*` rows of `kMats` in `rocket.cpp`. Emissive `g_*` materials are lit by day (signs, displays, room lights), like the pavilion's.
- **Light inside by day**: the rooms' lights (those under 3.2 m, in the building) are `station_lights_`: by day the six nearest the eye are real lights within 110 m (`descriptors.cpp`), as the pavilion's.
- `VIBE_LOG_LOOK=1` (new): logs once the ground point the camera looks at, the ground's heights round it and the highway beside it - what was used to place this.

Looked at (1280 x 536, 14 h): from the player's own pose, from the road north and south, from above, at eye level on the forecourt, inside the diner and the mart. The frame at the station from the road, 3440 x 1400: 12.1 ms (the mesh pass 0.45).

Not done / not checked: night (a capture at "21.5 h" came out as day - the test's clock was held); driving a car in over the apron and the highway's shelf beside it (the pads are registered, not driven on); the doors are open doorways (no leaves that move); rain inside the roofed parts; the plot from the river side below the retaining wall; multiplayer.

## The station joined to the road, flicker, height maps (2026-11-24)

- **The driveway.** The station stands 4 m further west than first built (centre x 1616, `kStationX`). Between the forecourt's edge (`kStationEdge` 1642.6) and the road's verge the ground is one ramp, the same rule in `highway::ground()` (highway.cpp) and `rm_highway` (terrain_rm_noise_road.glsl): the station's level at the edge, the road's own surface at the verge. The model carries a flared mouth over it (kerbs, lines, give-way dashes; `mouth()` in make_gasstation.py). No street lamp stands in the mouth (highway_furniture.cpp).
- **The road's line must not see the station.** The route is planned from the natural ground; with the terrace in it the road moved 20 m east. `highway::build` runs with `world_ground_without_station(true)` (terrain.h).
- **No relief under the paving.** `rm_ground_detail` (terrain_rm_heightfield.glsl) fades the close-range relief to nothing inside the station's rectangle: stones stood up through the asphalt.
- **Flicker = two faces in one place.** Fixed in the script: paint 2 cm over the asphalt, coloured bays 1 cm, pump bands and screens proud of the body, the canopy's sheet under the fascia's top, corner piers that share no face with the end walls, tile linings in front of the plaster, bands as strips round the edge instead of lids. Rule for new models: nothing coplanar - lift or inset by a centimetre at least.
- **Height maps.** `tools/gasstation/make_maps.py` writes `brick_h.png`, `checker_h.png`, `subway_h.png` (joints low). world_textures.cpp: a `<name>_h.png` beside a station texture replaces the brightness as the height the normals come from. There are no roughness or occlusion maps, and meshes have no parallax.
- `VIBE_LOG_LOOK=<metres>`: the grid step of the logged ground heights.

## People and cars that live their own life: the NPC engine (2026-11-24)

**`src/npc/npc.h`, `npc.cpp`** - in `vibe_core`, no Vulkan, physics or network in it. Up to 100 agents (`kMaxAgents`), fixed steps of 0.1 s, a seeded generator: the same seed and number of steps give the same world. It is given
- a **site**: a text file of nodes and links (where one walks), seats, spots (till, stove, shelves, pumps), bays, pump stops and car paths. The station's is `assets/gasstation/gasstation.nav`, written by `nav()` in make_gasstation.py - change the model and the life follows;
- a **road**: a closed course as a function from metres to a point and a direction.

Who lives there: a cashier, two cooks and a server at their places; 14 locals who sit (booths, stools, tables, the benches outside), look at the shelves and wander; 26 cars round the course, half each way, on their right-hand lane, keeping behind the car in front. The cars that pass the station on their side carry one to three people; 38 % of them pull in - to a pump (the driver fills up, may shop and pay) or to a bay (they eat) - wait for their people and drive on. Five cars stand parked. A car that stood 20 s on the site behind another goes anyway: nothing may stand for good.

**Shown by the engine** (`vk_engine/npc_world.cpp`; the cars in jeep.cpp `npc_cars_init` / `npc_cars_place`):
- People: the soldier's mesh and animator, as other players are drawn (`MpAvatar`, `unarmed`), sitting with the animator's seat pose. The nearest 28 within 150 m (a frame can skin 48 bodies in all). There is one body mesh: everybody is a soldier until there are civilians.
- Cars: 15 cars without physics (copies of the sports car, saloon and pickup: the same meshes), lent each frame to the NPC cars within 700 m, nearest first; paint from the agent's `look`. Wheels roll and steer, brake lights, head lights in the dark. A walker collides with them; they do not collide with a driven car and cannot be entered.
- Only in home's chart.

**Multiplayer** (protocol 5, `kMsgNpc`): the host steps the world and sends it four times a second (`serialize()`: 4 + 18 bytes an agent, unreliable); a client `apply()`s it and only blends between the last two states. A client that becomes host starts the world anew from the seed (one jump).

Tests: `tests/test_npc.cpp` (half an hour stays sane, same seed same world, every car leaves again, the wire, taking over). Hooks: `VIBE_LOG_NPC=1` (census and what each car at the site does, every 300 frames), `VIBE_NPC_FAST=<n>` (n times as fast), `VIBE_NO_NPC=1`.

Measured (1280 x 536, in the diner, 18 people posed): CPU frame 3.14 -> 3.28 ms; GPU passes unchanged. Two instances on one PC: the client's counts follow the host's.

Found on the way: a car that reserved a lane's far pump while another stood at the near one never got there, and the near one waited for the far one - both stood for good. Now the far stop is only taken while the near one is free.

Not done / not checked: people are not solid and do not avoid each other or the player; nobody opens a car door or is seen inside a car; cars do not react to a player's car on the road; a still picture cannot show flicker - the fixes above are in the model but were not watched in motion; night.

## NPC cars drive like cars; people in them; the edge that stopped a walker (2026-11-25)

- **How a car moves** (`npc.cpp`, "HOW A CAR MOVES"). It was a point slid along its line with the nose eased round: where the way in left the lane it jumped two metres sideways, it turned on the spot, and south of the mouth it drove out through the air. Now it is a bicycle model: a rear axle, a 2.7 m wheelbase, front wheels that steer (0.6 rad at most, 1.5 rad/s), speed that changes by 1.8 m/s2 up and 3.2 down, and no faster through a bend than 2.6 m/s2 sideways. It steers at a point of its line ahead of it (pure pursuit) - on the road the line is its lane, on the site the nav file's paths - and is never put anywhere: it drives from the lane into the station, out of a bay backwards on an arc, and out on to the road, where it waits for a gap. `s` along the road is the place on the lane beside the car (so the inside of a bend is shorter).
- **What the body does** (jeep.cpp `npc_cars_place`): it lies on the slope, leans out of a bend, squats under power, dips under the brakes; the wheels stay on the ground and point where the turn says. This is a model of how a car moves, not the rigid-body simulation the player's car has: there is one simulated vehicle in the physics world. NPC cars do not collide with a driven car.
- **People in the cars**: a hidden person's `View` says which car and seat (`car`, `seat`; the wire is 19 bytes an agent, format 2). The driver and whoever sits beside him are posed in the shown car's seats (`npc_car_seat`), hands on the wheel. A moving car with nobody of the engine's at its wheel gets a driver all the same. Within 90 m. Seen through the glass they are dark figures: they sit in the roof's shadow behind tinted glass.
- **The ways' heights** (make_gasstation.py `onroad`): the ground east of the plot as the engine makes it, not clamped at the terrace's level.
- **The edge that stopped a walker**: the forecourt's collision slab had its top exactly at the ground, and its east face stopped a player walking in from the road at x 1642.3 (by the drain cover). The slab's top is now 4 cm under the ground; the level ground itself is walked on. Rule: a collider's top that is meant to be flush with the terrain goes under it.
- Tests: "a car only ever drives" (40 minutes step by step: no step longer than its speed allows, no turn over 0.09 rad a step, under 12 cm sideways). Hooks: `VIBE_NPC_WARM=<s>` (so much of their life has passed at start; with `VIBE_LOG_NPC` the moments a car turns in are logged), `VIBE_NPC_NO_RIDERS=1`, `build/trshot/walk.sh` (walks the player forward from a place and prints where it got).
- Looked at: four moments of a pickup turning in, a close view, with and without riders; 20x for 30 minutes of their time on the real site (cars stop within 0.4 m of a pump's mark and 0.1 m of a bay's, all leave again). Not looked at in motion on screen - stills and numbers only; night.

## The castle lane: one lane of asphalt along the river, its bridge, rocks and car park (2026-11-25)

**What it is.** A second road line (`highway.cpp build_castle_lane`, line 1; 1.85 km): off the coast road south of the river's mouth (the far side of the highway bridge from the service station), west along the river's south bank, over the river on its own bridge 1.3 - 1.4 km in, along the north bank, up the little valley east of the knoll and into a car park on the knoll (`terrain.h kCastlePark`, 100..126 x 324..346), 50 m from the stone bridge's far end. Steepest 13 %.

**How its line is found.** From the river's own course (`terrain.h world_river_course`, `world_river_wander`): a station every 8 m of the river; on the chosen bank, the first place outward from the channel where the natural ground stands 5 m over the water, a few metres back from there, smoothed; pushed further out wherever it would stand on wet ground. So it sits on the bank's shoulder with the least earthwork. The crossing and the last climb are given points. Then a road's curves (20 m least radius) and a road's profile (the ground, smoothed, no steeper than 13 %, straight from bank to bank over the water).

**Roads of different widths** (`highway.h add_line(..., width)`; `Line::k`). A road of width w is a highway's cross-section w times as large: asphalt to 5.5 w, verge to 8 w, cut and fill met at 30 w. The lane is 0.364 (4 m of asphalt). **Distances are in the road's own measure** (metres / width) everywhere - `nearest()`, `distance()`, the baked field - so 5.5 is the asphalt's edge on any road and nobody who asks "is this on the road" needs to know its width; the field's spare byte (b.a) carries the width for what is laid out in real metres (`rm_hw_width`, used by `rm_hw_asphalt`: a lane has one pair of wheel tracks, an edge line each side, no centre line). "Nearest" is by that measure too.

**Bridges on any line** (`g_spans`: line, from, to; `span_count / span / point_on`; `bridges.cpp place_bridges` builds each as wide as its road - a lane's has a kerb, railings, piers every 22 m, no walkway or lamps). `deck()` answers for all of them.

**Inside the home tile** the ground is the baked height map. `world.cpp` cuts a road's shelf into it BEFORE the baseline is taken: the shader makes the shelf from the field and adds (bake - baseline), so a shelf cut after the baseline would be there twice. `Road::with_highways` makes the old castle road's `distance()` answer for the lane too (trees, rocks, tufts keep off both); the grass mask is cleared along it (`terrain_textures.cpp`).

**Rocks** (`bridges.cpp place_lane_rocks`): where the natural ground 7 m out lies more than 2.2 m under the road, three rows of boulders from the verge down the slope (2078 of them), made from the line by rule, in 120 m meshes, boxes for a walker. Not on a bridge, not on a road.

**The car park** (`rocket.cpp place_castle_parking`; terrace in `world.cpp`): 26 x 22 m, bays along both long sides, a kerb, two lamps, a P board, the gate in the north side where the lane arrives. The terrace lies 45 cm UNDER the slab's top - level with it, the ground's close-range relief and grass stood up through the asphalt - except under the lane where it comes in. No trees or tufts on it (`veg_zones.cpp`).

**Tool:** `VIBE_DUMP_MAP=<file.ppm>[,x0,z0,x1,z1[,metres per pixel]]` writes a plan (heights shaded, 10 m contours, steep ground brown, water, every road) and logs each line's length, steepest grade, bridges and - for a small plan - the heights every 25 m. `build/trshot/ppm2png.ps1` turns it into a PNG. This is how the route was laid out and checked. `VIBE_NO_LANE=1`, `VIBE_NO_LANE_ROCKS=1`.

**Looked at** (1280 x 536): the junction, a bank stretch with its rocks, the bridge from the side and along its deck, the surface close by, the car park from above and from the lane. **Not done / not checked:** driving a car along it (the vehicle's ground and the deck query are the generalised ones the coast road uses, but nobody drove it); no guard rail, signs or lamps along the lane; NPC cars do not use it; no unit test (highway.cpp is not in the library the tests link - the map dump's figures are the check); night.

## Station and NPC fixes of the same round (2026-11-25)

- **Painted bays flickering against the asphalt**: 1 - 2 cm apart was too little from a distance. Coloured bays are 3 cm over the asphalt, lines and letters 5 cm (make_gasstation.py).
- **The open side toward the river**: the retaining wall goes down 45 m (`WALL_DEEP`), not 9.
- **Tiling of plain surfaces from a distance** (`cube_main_surfaces.glsl`, static mesh branch; slots 38 asphalt, 28 and 24 concrete): once a repeat is under a few hundred pixels the texture goes over into its mean colour, modulated by the same texture read 7 and 34 times larger and turned. Not for patterned textures.
- **How a person walks** (`npc.cpp`, "HOW A PERSON WALKS"): it goes the way it faces, turns at a person's rate, slows through a turn, picks its speed up and lets it go, cuts corners; turns to a seat and sits down over half a second; steps out of it. Before: full speed from the first step straight at the target while the body swung round (sliding), corners on a point.
- **Head up** (`npc_world.cpp`, `VIBE_NPC_UPRIGHT`, default 0.42): the soldier's walk is hunched over a rifle; an NPC's spine, neck and head are raised through the animator's aim pitch, more the faster it walks.

- **Found before hand-over:** the stones were first put in `world_.aabbs`, each a static body in the physics world; with them the body table (2048) was full and the home tile's ground body failed to be made (`[jolt] heightfield body create failed`). They are now `lane_rock_cols_`: added to the walker's colliders near him (props.cpp) and to a car's obstacles (jeep.cpp), not bodies. Watch `Jolt: N static bodies` in the log when adding many small solid things (1343 now).

- **The lane was not level inside the home tile; its stones hung in the air (fixed 2026-11-25, same day).** A saved sculpt (`assets/level1_heights.bin`) is taken in after the baseline is captured, as "saved - baseline" added back. The shelf had been cut into the heights before the baseline, ahead of that load - so the saved land, which has no road in it, came back as a difference from a baseline that had one, and took the shelf out again. The shelf is now cut after the sculpt is loaded, into the heights AND the baseline (`world.cpp`, beside the river's cut into a loaded sculpt). Rule: anything the shader makes by formula and the bake must also hold goes into both, after every load.
- **What stands under the lane's edge** is now set on the ground as it is drawn (`sample_terrain_height` inside the home tile, the formula with the shelf outside it): three stepped courses of concrete blocks (1.5 x 0.9 m, texture slot 24) from the verge down the slope and a row of boulders at the foot - 1896 blocks and 632 stones. `[lane]` in the log counts them.

## HY-Motion 1.0 was asked for and not installed (2026-11-25)

Tencent HY-Motion 1.0 (text to 3D human motion; github.com/Tencent-Hunyuan/HY-Motion-1.0). Two things stand in the way, read from its own files that day:
- **Licence** (`License.txt`, "Tencent HY-Motion 1.0 Community License"): it "does not apply in the European Union, United Kingdom and South Korea"; using the model OR ITS OUTPUT outside the licensed territory is "unlicensed and unauthorized" (section 5c). Animations made with it could not be used in, or shipped to, those places.
- **Hardware**: the README gives 26 GB of video memory as the minimum (24 GB for the Lite model); this PC's RTX 4080 has 16 GB. Community reports say it can run in less with the text encoder quantised or on the CPU - not tried.
Free data that covers part of the wish: the CMU database (already used for the runs) has sitting down on and getting up from stools and step stools (13_01..06, 14_27..36, 15_10); it has nothing for getting in and out of a car.

## Round of 2026-11-25, late: actions from text, the lane's end, the bridge-end strip, cliffs

- **Action clips** (text to motion with HY-Motion 1.0, played by name with inertialisation, hip warping and leg re-solving; the NPC engine's `Act`): see `docs/animation/actions.md`.
- **The strip running out from each bridge's end** (seen from the service station's bridge toward the sea): the baked road field stored "far" as +64 on both sides of a span; on the road's negative side the neighbouring texels read about -20, and the shader's bilinear read between them went through zero - a texel-wide strip of shelf and asphalt straight out from the span's end, on one side, as far as the field reaches. In a span the distance now keeps its sign (`highway.cpp bake`). Found by switching things off (`VIBE_NO_LANE`, `VIBE_NO_BRIDGES`) under a top-down capture.
- **The river's valley** (`wm_river`, both twins): one even side slope (the 1.3 beyond 40 m is gone) and a 12 m rounding where it meets the land. The cut used to END at 120 m from the channel - whatever stood higher there was sliced along that line; with the gentler side that became a wall with the mountain overhanging it. From 70 m out the side now climbs as the square of the distance, so the cut runs out into the land. (`far` and `near` are macros in Windows headers: not variable names.)
- **Cliffs are rock** (`terrain_rm_main_ground.glsl rm_cliff`, used in `terrain_rm_main_light.glsl`): on a steep face the colour is the scanned rock laid on the wall, with level beds and patches, instead of the flat palette (worked out from the ground plan: it smeared on a wall) with the scan as a tint.
- **The lane's end**: it comes up the little valley, turns west and runs its last 14 m straight and level through a gate in the car park's EAST side. (A loop round the knoll's north was drawn first; the least-radius rule flattened it and the lane arrived along the north side at 20 %, side-on to its gate.)
- **Plants and rocks on the lane**: everything that scatters asks "inside the road's bounds?" before "how far is the road" - and the bounds were the old castle road's. `Road::bb_*` now reach round the highways' lines in the home tile (`paved_bb_*` keeps the old road's own, for the shader); `Road::distance` reaches 22 m of a lane (it was 4.4 m); woods keep 7 m from a lane's middle.

## Round of 2026-11-25, night: what the continued checks found

- **The lane, looked at again after the bounds fix**: two stretches inside the home tile - no rock on the asphalt, trees stand back from it.
- **Getting into and out of a car, seen in captures** (a visitor's sports car in a bay, four moments each): in - standing, bending in, seated; out - seated, a foot out, stepping out, standing. Found on the way: with the hips warped a metre into the seat the "standing" foot stayed outside and the leg went through the door. A foot now goes with the body when the warp is large (over 0.18 .. 0.55 m), and stays put only when it is small (`CharacterAnimator::action`).
- **A car drove off across the country.** The rule for which bays a visiting car may use still had its old margin in `plan_visit` (the edit had landed in the other copy of the rule), a car was sent to a bay with no room for the turn, overshot, and with its target BEHIND it pure pursuit steers almost straight - for ever. Now: one constant (`kBayRunUp`) in both places; and a target behind the car means full lock toward it at a crawl (`drive`). Tests: no car is ever further than 6 m from the road outside the site.
- **Cars stood up to 35 degrees askew in their bays.** The way in is now a swing out, a quarter circle of 4.6 m and the rest straight; on a site the car looks less far ahead (follows its line closer); and over the last 7 m before it stands it also steers to lie ALONG the line (`drive(..., straighten)`). Test: within 12 degrees of the bay's line. On the real site: 7 degrees.
- The parked cars that never move stand only in bays visitors cannot turn into.

- **NPCs cast shadows (2026-11-25).** `npc_bodies_` join the same two paths the players use: within 40 m the nearest 6 bodies of everybody are in the ray-traced scene (`body_rt.cpp`: the sun's sharp shadow, reflections); to 90 m they are drawn into the small-object shadow map (`clutter_shadow.cpp`) - to 45 m one mesh level under the drawn one, beyond that the coarsest; further than 90 m none. Seen in a before / after capture: a person by a car now shades the door beside him.

- **Start place, and lights at dusk (2026-11-25).** With no saved place a game starts at the service station (`world.cpp`: 1610, 521, looking south along the mart). `cars_need_lights()` (jeep.cpp): the clock's lantern factor over 0.1, or the sun under 12 degrees - NPC cars that drive have head and tail lights on from then (it was the sun under 7 degrees), with the lamps' glow drawn for them too (`jeep_fx`); the car the player drives puts its lights on once when dusk falls (L still switches them). NPC cars throw no beam on the road: the scene has one cone light, the driven car's. Seen at 18:35 and 20:30: a white glow and a red one up the road.

- **A sunken flat with grass hanging over it, between the station's wall and the highway bridge (2026-11-25).** The station's driveway rule (ground east of the plot = a ramp to the road's verge) ran on to z 587, past where the bridge's span begins (z 577). In a span the field's distance reads "far" (64): the shader built the ramp toward a road 64 m away; the CPU's ground returns the natural land in a span before it reaches the rule. Two grounds: grass, stones and physics on one, the picture on the other. The rule now ends at z 565..573 in both (`highway.cpp ground`, `rm_highway`), and the shader also skips it where the distance is "far". Rule for anything keyed to "distance to the road": say what happens in a span.

- **Still grass in the air there, and the player floating (same evening).** What was left was at the bridge's end itself: the CPU's ground switched from shelf to bare land on the exact line where the span begins; the shader reads "far" from the field's 4 m texels bilinearly and slides between the two over a texel. `highway::ground()` now reads the distance and the height AS THE SHADER DOES within 14 m of any span (the four texel centres, 4 k + 2 in home coordinates, bilinear). Outside home's chart the field is laid out in the chart's coordinates and the two can still differ at a bridge's end.

## Nobody walks through anything; car doors (2026-11-26)

- **NPCs and obstacles** (`npc.cpp`, "NOBODY WALKS THROUGH ANYTHING"). A walker *steers*: whoever or whatever is close and in the way pushes its heading away and round - people pass on the right, a car is gone round by the side nearer the walker's target (`steer_clear`). And it *cannot*: after every step it is put outside every car's outline (2.04 x 4.7 m, plus a body's 0.34 m) and 0.54 m from everybody else (`keep_apart`). Its own car is not in its way. A place it cannot reach (somebody on it, a car over it) is given up after 4 s without getting nearer. A car on the site waits for somebody on foot within 8 m ahead of it (its own people excepted; after 20 s it goes, as before). Test: over 40 minutes no walker is ever inside a standing car's outline or nearer than 0.30 m to another person.
- **Doors** (`jeep.cpp`, "THE FRONT DOORS open"). The car models have no doors - a body is one skin with the gaps drawn on it. At load the two front doors are cut out of `body.mesh`: every triangle whose middle lies in the door's box (beside the front seat, sill to roof, from 1.05 m ahead of the seat to 0.5 m behind) is the door's - skin, glass, mirror, trim. Each turns 64 degrees on a hinge down its front edge; open or shut in under half a second. Shut, nothing shows (the same triangles in the same place); open, the cut follows the triangles' edges, not a drawn gap. Not the jeep (open-sided). `VIBE_NO_CAR_DOORS=1`: none.
  - An NPC's door is open while it gets in or out (`npc_cars_place`: its `Act` and seat), shut again for the last half second.
  - The player's: the driver's door stands open for a second when getting in or out (the getting in itself is still instant).
- Seen: a driver getting out at a pump, five moments - shut, open, open, open, shut. Not seen: the player's own door; a passenger's door; people going round a car (the test holds the rule, no capture was made).

## Car sound (2026-11-25)

**Where the sound comes from.** `engine-sim` (github.com/ange-yaghi/engine-sim, MIT) simulates a piston engine and synthesises its exhaust note. Running it live costs a good part of a core an engine, so it is used OFFLINE: each car's engine is rendered to loops, and the game plays those.

- **The tool** (outside the repo: `C:\Users\gillis\dev\engine-sim\vibe`): `CMakeLists.txt` builds the simulation core, the script interpreter (piranha) and `bake.cpp` - none of the author's graphics engine. Needs flex / bison (`C:\Users\gillis\dev\tools\winflexbison`); Boost is patched out (std::filesystem). Build: `cmd /c C:\Users\gillis\dev\engine-sim\vibe\build.bat`.
- **Bake** (run in `vbuild`, so that `../es/` is the scripts' library): `vibe_enginebake.exe ../assets/vibe_<car>.mr ../vout <car> <rpm lo> <rpm hi> <steps> 2.0`. The engine is held at each speed by the dynamometer and recorded under load (throttle 0.85) and coasting (0); each recording is cross-faded into a 2 s loop. The synthesiser's leveller is set to 13000 (at the application's 30000 every transient clipped). Then copy `vout/*` to `assets/sounds/engines/`.
- **Which engine**: jeep - Subaru EJ25 flat four (800 - 6200 rpm, 8 steps); sports car - Ferrari F136 flat-plane V8 (1000 - 8600, 9); saloon - Toyota 2JZ straight six (800 - 6600, 8); pickup - GM LS V8 (700 - 6400, 8). (`chev_truck_454.mr` does not compile with this build.) 70 files, 11 MB.

**In the game.**
- `AudioEngine::engine_*` (`engine/audio.h`, `audio.cpp`): a voice plays the two loops either side of the engine's speed, each pitched to it, cross-faded (equal power) by speed and by load, through its own low-pass; panned, with Doppler from the car's and the listener's velocity.
- `car_audio_tick` (`vk_engine/jeep.cpp`, called beside `set_listener`): the nearest 6 running cars within 280 m have a voice - the car the player drives, the NPC engine's traffic, cars other players drive. The driven car's engine speed and load are the physics' and the throttle; any other car's come from its road speed through its own gears (it changes up early when cruising, late under load) and from how it gains speed. A car that stands (parked, at a pump) is silent.
- **Far away**: volume / (1 + (d / 20 m)^1.7), and the low-pass falls from 18 kHz at 12 m to 600 Hz at 320 m.
- **In the cabin** (driving, the nearest of the three views): the own engine through 2.4 kHz; everything outside through at most 1.4 kHz and at 45 %.
- Hooks: `VIBE_LOG_CAR_SOUND=1` (each voice every 90 frames: distance, gear, rpm, load, volume, low-pass), `VIBE_CAR_SOUND_GAIN=<x>`, `VIBE_NO_CAR_SOUND=1`.

**Checked by the log, not by ear**: traffic passing a standing player (a pickup from 266 m to 24 m: volume 0.002 -> 0.125, low-pass 800 Hz -> 17.7 kHz) and the self-driving jeep (rpm following the physics through its gear changes). Nobody listened: the levels, the gear rule and the filters are first values.

**Not done**: no idle for a standing car with its engine on; no tyre, wind or brake noise; no engine start or stop; the cabin filter is tied to the nearest view (there is no real interior view); a live simulation for the player's own car (more alive than loops) was not tried.

### Grass mask: the slope is never the saved one (2026-11-25)

`assets/level1_grass_mask.bin` holds the painted grass mask (R presence, G slope). `init_grass_mask_texture` takes only R from it: G is always baked from the land as it is now. The file's slope was that of the day it was saved; after the land changed (river valley, road shelves) it left a round bare patch in a meadow and grass on a 50 degree bank. Hooks: `VIBE_DUMP_GRASS=<x>,<z>` (the mask's slope channel round a place), `VIBE_DUMP_DELTA=<x>,<z>` (heights round a place, decimetres, 1 m apart). Cliff rock (`rm_cliff`) now starts at 1 - normal.y = 0.56 (was 0.40): only on faces that are really walls.

## The station's life, round two (2026-11-25, v0.4.0)

**More to do (animation).** Clips made with HY-Motion (docs/animation/actions.md), played by `npc_world.cpp` "WHAT IT PLAYS":
- `npc::View::style` (new; on the wire in the upper bits of the act's byte, wire format 4, protocol 6) says how somebody sits, stands or works. Sitting: 1 at the bar (`bar_sit`, `bar_talk`), 2 eating at a table (`table_eat`), 3 talking at one (`table_talk`) - from the seat's kind. Standing a while: `look_crossed`, `look_hips`, `look_phone`, `look_far`. Working: 1 a cashier (`cashier`, `cashier_take`), 2 at a pump (`fuel`), 0 `till`.
- A seat's clip is played with the model standing on the floor in front of the seat and the hips warped on to it for the whole loop (`action_warp_full`); the floor is `kSeatHeightCm` under the seat's top (45 cm, a bar stool 75).
- **Ways of walking**: `look % 8` - 1 brisk, 2 tired, 3 hands in the pockets, 4 at a phone, the rest the plain motion-matched walk. The clip is played in place (`action_in_place`: its own travel left out, the NPC engine moves the body) at the rate the body really moves; the NPC engine walks each at that clip's pace (`npc::kPace`).
- **Soldiers on the run**: `npc::World::set_run_path` + role Runner, pose `Run`: two squads of six, two abreast, there and back along the coast road's edge from the launch pad and up the castle lane to the car park (1970 m). They carry their rifles and use the motion-matched run. `VIBE_NO_RUNNERS`.
- Not changed: the driver's pose (hips on the seat from the car's file, hands on the rim by IK). A `drive` clip was made and is in the set, unused.
- **Round three (2026-11-25, stopped half-way; docs/animation/actions.md "At the wheel" and "Round three")**: the driver and whoever sits beside him are fitted to the car itself (`CharacterAnimator::seated` with `CharacterAnimationInput::SeatExtras`, filled by jeep.cpp `jeep_seat` / `jeep_seat_feet` from the car file's `hip_driver`, `pedal_gas`, `floor_y`, `foot_rest`, `seat_back_deg` - measured from the car models); the diner's chairs face their tables (make_gasstation.py; they stood with their backs to them); people who stay take turns between clips (`npc_world.cpp` "WHAT IT PLAYS"). Hooks: `VIBE_NPC_CAM=<who>[,<metres>[,<degrees>[,<height>]]]` (the eye beside somebody: bar, table, eat, talk, counter, look, walk, walk<0..7>, run, driver, id<n>; with `VIBE_TEST_HOLD`), and `VIBE_LOG_NPC` now also says what each shown body plays (`plays: ...`). Test: tests/anim/test_animator.cpp "at the wheel the body is fitted to the car's seat, wheel and pedals".

**Shooting them.** `npc_bolt_hits` (combat.cpp calls it beside `mp_bolt_hits`): a bolt against everybody on foot or on a seat. A hit: blood (`spawn_blood`: mist, drops that fall and leave stains for two minutes - fx.cpp, fx_ribbon.frag kind 7), the body drawn for that person is handed to the physics as a ragdoll (`corpse_spawn`, as for a killed player), and `npc::World::kill` takes the person out of the NPC world: their car leaves without them, and 75 s later somebody new walks on to the site (a soldier is not replaced). For every player: see "NPCs and every player" below.

**Every car can be driven.** `npc_take_car` (jeep.cpp, from `jeep_use`): a car of the NPC engine that stands with nobody in it (parked, or at a pump with its people in the shop) is taken by getting in. `npc::World::take_car` lets it go (its people stay, on foot; 150 s later another car joins the traffic); the model leaves the pool of lent cars and joins `jeep_all_`. For every player: see "NPCs and every player" below. `VIBE_TEST_USE=<frame>` presses the use key.

**Lights.** A lantern's light reached 13 m; now 34 m, with the same brightness near it ((1 - d/34)^5 instead of (1 - d/13)^2 - squared at 34 m it doubled the light in every room). A car's beam reaches 270 m (was 95) and thins out as slowly as it reaches (`flashlight_at`: the falloff constant scales with the range); the torch reaches twice as far (60 - 150 m by level).

**Car sound, second pass.** Half at 9 m, a tenth at 27 m, nothing beyond 170 m; duller from outside (7 kHz at arm's length on foot, down to 350 Hz far off; the own car from behind 12 kHz). Five voices at most; a car beyond 45 m plays one loop instead of four and is brought up to date every fourth frame; below a volume of 0.0015 nothing plays.

**The lane's edge.** `place_lane_rocks` now builds one continuous concrete wall where the land 7 m out lies more than 2.2 m under the road, and only there: a parapet (0.8 m high, 0.3 thick) a car cannot go through, on a battered retaining wall down to the ground. Sections share their corners; the parapet rises out of the verge at each end. (Before: loose blocks and boulders, each on its own bit of ground.)

**The river.** `scatter.cpp` "THE RIVER'S OWN PLANTS": along the whole course inside the home tile (the other passes reach 280 m from the castle) - lily rafts on the shallows, reed beds at the waterline, ferns and tufts up the banks.

**Stopped.** Nothing falls on the castle any more (`VIBE_BOX_RAIN=1` brings the boxes back).

**Tests** (tests/test_npc.cpp, 139 in all): somebody shot is gone and replaced and nobody's car waits for the dead; a standing car can be taken and the traffic gets another; the soldiers keep to their path with no jumps, and pose and style cross the wire.

**Not verified / open**: blood and the falling body were looked at in one capture each; the lights at night, the sound and the walking clips in motion were not looked at or heard. Runners do not avoid cars or the player. The walking clips loop with a blend every 10 s.
**Car sound, third pass: nothing stops at once (2026-11-25).** A car driving away went quiet fast and then cut out: the loudness fell with the square of the distance from 9 m (under the playing threshold by 100 m), a voice was stopped the moment its car left the five nearest or parked, the far voice began at a fixed 45 m and was updated every fourth frame (a staircase). Now (`car_audio_tick`): loudness 1 / (1 + (d / 14)^1.5) - half at 14 m, an eighth at 50 m, a twentieth at 100 m - times a window from 100 m down to nothing at 240 m; each car's loudness FOLLOWS what it should be (`snd_vol`: up in 0.1 s, down in 0.33 s), so a voice that is let go, a car that parks and the player getting out are fades; a voice is released only once it is silent; the one-loop far voice is chosen by loudness (under 0.006, back above 0.010) and updated every frame; nothing plays under 0.0004 (-68 dB). Six voices, a few more while some fade. Measured with `VIBE_LOG_CAR_SOUND=1` on a pickup passing at 33 m: 0.063 there, 0.021 at 81 m, 0.006 at 154 m, 0.001 at 204 m, 0 at 239 m - one slope in and out. Not heard by ear.

**Car sound, fourth pass: muffled and quieter from outside (2026-11-25).** "We need some muffling when you stand outside or far away; more silent the further away; cpu efficient." The curves are two functions in `engine/car_sound_curve.h`, used by `car_audio_tick` for every car the player does not drive (the driven car is as it was: 12 kHz in the chase view, the cabin's 2.4 kHz inside; other cars heard from the cabin still go through 1.4 kHz at most and x 0.45):
- loudness `gain(d) = 1 / (1 + (d / 12)^2) x (1 - smoothstep(90, 180, d))`; low-pass `cutoff_hz(d) = 4500 / (1 + d / 12)`, not under 300 Hz (one biquad a voice, 12 dB an octave - `ma_lpf_node`, as before).

| distance | gain | dB | low-pass | before: gain | before: low-pass |
|---|---|---|---|---|---|
| 0 m | 1.000 | 0 | 4500 Hz | 1.000 | 7000 Hz |
| 5 m | 0.852 | -1.4 | 3176 Hz | 0.824 | 6982 Hz |
| 12 m | 0.500 | -6.0 | 2250 Hz | 0.558 | 6656 Hz |
| 25 m | 0.187 | -14.6 | 1459 Hz | 0.295 | 5301 Hz |
| 50 m | 0.055 | -25.3 | 871 Hz | 0.129 | 2421 Hz |
| 75 m | 0.025 | -32.1 | 621 Hz | 0.075 | 941 Hz |
| 100 m | 0.014 | -37.3 | 482 Hz | 0.050 | 442 Hz |
| 125 m | 0.0061 | -44.3 | 394 Hz | 0.033 | 350 Hz |
| 150 m | 0.0016 | -55.7 | 333 Hz | 0.020 | 350 Hz |
| 180 m | 0 | - | 300 Hz | 0.0084 | 350 Hz |

(the table is the code's own output: `VIBE_LOG_CAR_SOUND_CURVE=1`.) A car has a voice within 180 m (was 240 m); the rest is the third pass's: the loudness follows its target, a voice is released once silent, nothing is mixed under 0.0004, the one-loop far voice under 0.006 - which a cruising car now reaches at about 75 m instead of about 150 m. Checked with `VIBE_LOG_CAR_SOUND=1` at the roadside by the station, a pickup passing: 26 m - wanted 0.0437, 1433 Hz; 38 m - 0.0218, 1073 Hz; 95 m - 0.0040, 503 Hz; 155 m - 0.0003, 324 Hz; 175 m - 0, 300 Hz: the curve times the car's own 0.249.
- **The mixer's cost** is measured now (`AudioEngine::mixer_cost_take`: the device callback is miniaudio's own with a clock round it - two clock reads and three atomic adds a block, no allocation, no virtual call; `VIBE_LOG_AUDIO_COST=1` logs it every 300 frames with the engine voices held and loops mixed). On the 9800X3D, 10 ms blocks: 2.5 - 4 us a block with no car (0.03 % of a core), about 55 us with two near cars (8 loops: 0.55 %) - about 6.5 us a loop a block. One pass of two pickups by the station, the lines with a voice held: 0.41 % of a core before (`VIBE_CAR_SOUND_OLD=1`), 0.27 % after - one sample of sparse traffic, not a benchmark; the saving is the loops that are no longer mixed between 75 and 240 m. Not heard by ear.

**Shot bodies: why they flew, and how they go down now (2026-11-25).**
- THE FAULT: a bolt is a physics body of 10 kg at 220 m/s. The first bolt is taken out by `npc_bolt_hits`, but the next one in flight struck the ragdoll's parts (2 - 28 kg) with its own weight: measured under continuous fire, the pelvis reached 35 m/s (`VIBE_TEST_BOLTS_HIT_BODIES=1` brings that back to compare); in the open the body left the scene. RULE: a ragdoll's parts are in the physics layer `CORPSE` (physics_world.cpp): the world, what moves and other bodies touch them - bolts and sparks do not. A bolt through a fallen body is found by the game (`corpse_bolt_hits`, combat.cpp): a knock of at most 22 N s on the part, and blood. A part's speed is capped at 25 m/s whatever hits it. Now: 8 m/s at most, within 0.2 m of where it fell.
- Every ragdoll says once, 3 s on, how far its pelvis went and its top speed (`[ragdoll] fell within ...`); more than 8 m is a WARN ("a body flew").
- Going down: see docs/animation/actions.md "Death". `VIBE_NO_DEATH_CLIPS=1`: the ragdoll at once, as before. `VIBE_TEST_NPC_KILL=<frame>[,<skip>]`: the six nearest people with a body are shot at that frame.
- The shot's push on a ragdoll is halved (0.7 m/s for the body, 3.5 m/s for the limb it struck; players' bodies too).

**NPCs and every player (2026-11-25, protocol 7).** What a player does to the NPC world is an event (net_state.h `Event`); the machine whose NPC world is the world (the host, or a machine alone) decides, every machine shows it. No bones, clips or poses cross the wire here either: the NPC state is 27 bytes an agent four times a second (place, heading, speed, pose, act, style), and each machine's animator picks and plays the clips from that.
- **A shot** (`Event::NpcHit`: which agent, where, the bolt's direction). The shooter decides the hit against what is on the shooter's screen, shows it at once and tells everybody; `npc_hit` on every machine: blood, and the body drawn for that person falls (each machine its own ragdoll from the same pose - the fall itself is not sent). The authority calls `npc::World::kill`; a copy keeps the person out of sight for 3 s (`npc_gone_`) until the host's state says the same.
- **A car** - three steps, the same alone and on a network (jeep.cpp, the comment above `npc_take_car`): ask (`Event::NpcCarTake` from a client), decide (`npc_host_take`: `take_car`, the next number in `jeep_all_`), make (`Event::NpcCarTaken` -> `npc_car_make` on every machine: a model of that kind and paint leaves the pool of lent cars and joins `jeep_all_` under that number; the one who asked gets in). From then on it is a car like the others (`jeep_net_send` / `jeep_net_receive`). RULE: `jeep_all_` must be the same list on every machine - a car is named by its number - so a car is only ever added through `npc_car_make`, in the order of the numbers (`npc_taken_wait_` holds one that came before the one in front of it).
- **Somebody who joins later**: a second after a new player appears every machine tells of the taken cars it knows, where they stand now (`npc_taken_tick`). Known gap: two machines that each took cars before they met have different cars under the same numbers.
- Tests: `tests/anim/test_net.cpp` `[npc]` (the wire form of a taken car, what is refused, the events pass the gate at once). Two instances on one PC: `VIBE_NET=1 VIBE_NET_PORT=48100 VIBE_TEST_REALTIME=1`, `VIBE_NET_ID=100` on the one that must host (the lower id hosts), `VIBE_NPC_WARM=200` on it; the client with `VIBE_TEST_FIRE=1500` at (1602.6, 42.9, 532.0) yaw 1.945 pitch -0.085, or `VIBE_TEST_USE=1500` at (1617.0, 42.6, 558.5) yaw -1.57. Both logs then show "person 15 shot" / "NPC car 18 ... is a players' car now: number 8", the client "the host gave us NPC car 18 ... in it 1". Checked 2026-11-25 this way; not checked: a third player watching, a late joiner.

**The diner's posters (2026-11-25).** Eighteen posters of the base ("Cape Vibe Space Command": recruiting, 25 years, launch night party, base rules, safety, history, launch schedule, softball, a lost cat, ...) hang in the diner: ten along the party wall, eight in the band over the kitchen's pass. Pictures: `tools/gasstation/posters/p00..p17.png` (image generator, 2:3). `tools/gasstation/make_poster_atlas.ps1` packs them UNDER the pavilion's screens in the same picture, `assets/office/tex/screens.png` (texture slot 36, now 4096 x 4096: rows 0..1151 the 4 x 2 screens from `tools/office/screens_src.png`, below them 6 x 3 poster cells of 682.67 x 981) - no texture slot was free. `poster(i, x, y, z0, face)` in `make_gasstation.py` hangs number i; material `g_poster` (rocket.cpp table) is self-lit and samples the atlas with the cell's own uv, with v + 2 to tell cube.frag's display block (which picks a changing picture by number, and now divides by the screens' share of the atlas) that it is paper. Another poster: put the picture in the folder, run the atlas script, add a `poster()` line, run the Blender script.

**The launch site's lamps behind a mountain (2026-11-25).** The lamps' flares (`fx.cpp`, `pad_lights_`) are drawn over the picture and far land is not in the depth buffer, so from across the hills they shone through. Each lamp further than 350 m now has its line of sight marched over the world's ground formula (a third of the lamps a frame, the answer eased). `VIBE_TEST_PAD_LIGHTS_THROUGH=1`: as before. Not confirmed by a picture (two tries found no spot where the lamps showed either way).

**Cars with people in them, cars on the road, and people in cars (2026-11-25, protocol 8; unit-tested, NOT yet seen working in the game).**
- **Any NPC car that stands or barely rolls can be taken** (`npc::kTakeSpeed`, 1.5 m/s), whoever sits in it: `npc::World::take_car` puts them out beside it (`put_out`: at the site they go about as locals; out on the road they stand 40 s and are gone). Before: only a standing car with nobody in it.
- **Cars do not drive into players on foot** (`npc::World::set_walkers`, given every frame by `npc_tick` on the machine that steps the world: the local player and the other players' avatars). On the road a car brakes for somebody on its lane (within 2.3 m of the lane's middle, looked for 90 m ahead) and waits; on the site as for its own people. That is how a car in the traffic is stopped to be taken. `VIBE_NPC_NO_WALKERS=1`: as before.
- **The pool of lent cars is refilled** (`npc_car_add`, jeep.cpp) when one becomes a players' car; the old limit of eleven taken cars is gone, `net::kMaxCars` (32 players' cars in all) remains.
- **People in cars can be shot** (`npc_bolt_hits`: the bodies drawn in shown cars - driver and front passenger, within 90 m). `npc::World::kill` takes a hidden rider, or a car's number for the driver nobody of the NPC world sits for (the traffic the other way; `Event::NpcHit` then carries the car's number, `npc_nodriver_` hides him on a copy of the host's world). A car whose driver is shot rolls to a stop on its lane (pose Park at once), the others get out, and 120 s later it is cleared away. The body is handed to the physics at the car's speed; the lent car has no physics, so it falls through it on to the road.
- Hooks: `VIBE_TEST_NPC_SHOOT=<frame>[,<who>[,<skip>]]` turns the player to the nearest person the bolt can reach (who 1: in a car) and fires a real bolt (the log says when something stands right in front of the muzzle: `fire_projectile` then fires nothing); `VIBE_TEST_NPC_CAR=<frame>[,<throttle>[,<mode>]]` puts the player at the door of the nearest standing NPC car (mode 0) or on the lane 75 m ahead of the nearest driving one and waits for it to stop (mode 1), presses Use and drives (`[jeep] test:` lines; not with `VIBE_TEST_HOLD`).
- Tests: `tests/test_npc.cpp` - a car stops for a walker on its lane without touching him and is taken with its people in it; a driver shot (his own, and by the car's number).

### Engines and gearboxes: each kind of car its own (2026-11-25)

A kind of car is a handful of numbers in `PhysicsWorld::VehicleDesc` (set in `init_jeep`, jeep.cpp), all shares of `engine_max_rpm`:

| | engine | limit | idle | pulls hardest at | changes up, pedal down / held | change takes | loudness |
|---|---|---|---|---|---|---|---|
| sports car | flat-plane V8 (Ferrari F136) | 9000 | 1000 | 0.68 | 8600 / 3600 | 0.11 s | 1.0 |
| pickup | even-fire V6 | 4800 | 650 | 0.38 | 4320 / 2200 | 0.40 s | 0.85 |
| saloon | four-cylinder (Honda B18) | 6500 | 750 | 0.45 | 5980 / 2200 | 0.28 s | 0.42 |
| jeep | flat four (Subaru EJ25) | 4600 | 800 | 0.35 | 4140 / 2300 | 0.35 s | 0.80 |

- **One rule for the box**, `VehicleDesc::shift_up_rpm(pedal)` / `shift_down_rpm(pedal)`: early when the pedal is only held, at the top of the range with it down, a gear down when it is floored at low revs. The driven car's box (`PhysicsWorld::vehicle_input`, every frame) and the sound of every other car (`car_audio_tick`, which works a car's gear out of its road speed) use the same two functions. The change down is capped under what the engine turns right after a change up (the widest step between two gears x 0.72), or the box would go up and down without end.
- **The pedal the box sees is the driver's**, not what traction control lets through (`vehicle_input`'s `pedal`).
- **Why the cars sat on their limiters**: Jolt's automatic refuses a change up while a driven tyre slips more than 0.1 or the engine's speed is not rising, and a car under full power has its tyres right there (ours grip best at 0.12). A change up that is due and has not come within a tenth of a second is made by `vehicle_input` itself: the transmission is put in manual for that one change (drive cut for `shift_time`, clutch let in over 0.7 of it, throttle following the clutch), then handed back.
- **A change up needs the road to ask for it too**: the engine speed the road speed would give in this gear must be over 0.8 of the change-up speed. Spinning wheels alone take the engine there; changing then leaves the next gear at half the speed (the box went 1-2-1-2 through a wheelspin). So a car with its wheels spinning stays in gear on its limiter until it grips.
- **The torque curve** is made from `torque_peak` (physics_world.cpp, where the controller is set up).
- **Recordings**: `assets/sounds/engines/<kind>.txt` + loops, baked with engine-sim (`dev/engine-sim/assets/vibe_<kind>.mr` names the engine; pickup 650-5200 in 8 steps, saloon 800-6600 in 8, sports car 1000-8900 in 9). Every loop is levelled to the same loudness; `engine_loudness` is what makes one car quieter than another, and it also closes the low-pass of the driven car's near view.
- Test: `build/trshot/drive.sh <car 0..7> [top speed / 30 m/s] [frames]` drives a car along the highway by itself; with `VIBE_LOG_JEEP=1` every change of gear is a log line (from which engine speed, the most reached in that gear, road speed, pedal). `VIBE_TEST_JEEP_VIEW=<0..2>` sets the view.
- Set by numbers, not by ear.

**Tyre smoke stays out of the car** (`jeep_marks`, `jeep_fx`): a puff is a flat round picture turned to the eye that grows to a metre and a half; one at a rear wheel reached across the whole car, and seen from behind the seats stood in smoke. Now a puff is born with a push out to its wheel's side, its middle is kept out of the body's box (`JeepState::body_box`), and it is drawn no wider than its distance from that box plus 0.3 m. `VIBE_TEST_SMOKE_OLD=1` draws it as before. Not confirmed by a picture: the test drives made too little smoke to show a difference.

## The lane's dry-stone wall and the river's plants (2026-11-25, late)

- **The lane's edge is a dry-stone wall, streamed** (`vk_engine/lane_edge.h/.cpp`; it replaces the concrete parapet of `place_lane_rocks`, which is gone from `bridges.cpp`). Rule, from the ground as drawn: a station every 2 - 3 m of every road line but the coast highway; a wall on a side where the ground falls more than 1.5 m within 4 m beyond the verge (`kDrop`, `kWithin`), not on a bridge, not where another road joins, not round the car park (`no_wall` given by the caller), gaps of up to two stations closed, runs under four stations dropped. Stones are rounded blocks (superquadrics, 108 triangles), a function of (station, side, place in the course): a lower course, an upper course half a stone along, now and then a boulder, and a footing course on the side of the drop where the ground lies lower. Three tints of the triplanar rock texture (slot 7). The stations and flags are made once (`lane_edge::make`, called from `hw_objects_poll`); the stones are made in 120 m stretches by `highway_furniture.cpp` (ids `kLaneId + c` in `hw_chunks_`, loaded within 520 m, freed beyond 700 m, home's chart only). Collision: for cars and bolts turned boxes through `hw_rail_boxes` (`lane_edge::car_boxes`); for a walker upright boxes half a metre long in `lane_rock_cols_`. Measured at start-up: 1332 m of wall (the concrete one stood on 752 m, by its own rule). Hooks: `VIBE_LOG_LANE=1` (every run of wall with its end points, stones per stretch), `VIBE_NO_LANE_ROCKS=1`.
- **The river's plants are one rule** (`engine/river_plants.h`): stations 7 m apart along the stream's real middle; per side the bank is found, then lily rafts on the shallows (mostly on the inside of a bend, never in the middle half of the stream, never under a bridge, none below the coast road's bridge), reed beds at the waterline (a second stand in the water inside a bend), and up the bank ferns, long grass and drifts of buttercups, nettles, foxgloves, thistles. `scatter.cpp` asks it for the stations inside the home tile at start-up (1647 lily rafts, 3336 reeds, 4684 bank plants; before: 1297 / 972 / 2540). **Beyond the home tile** `vk_engine/river_stream.cpp` asks for the rest in 70 m stretches on a background thread (2 - 9 ms each) within 260 m of the camera, writes them into one of 12 slots of a 786 KB instance buffer, draws them with the small plants' pipeline and meshes on the CPU-cell path (`world_render.cpp`, after the species loop) and gives them back beyond 340 m. They cast no shadow of their own and get no long grass (species 15). Hooks: `VIBE_LOG_RIVER=1`, `VIBE_NO_RIVER_PLANTS=1` (also the home tile's).
- **River-bank scrub** (`world_props.cpp river_shrub_density`, home tile and streamed cells alike): the gaps between thickets narrower, more of it near the castle.
- **Not measured:** the `w_scatter` pass before / after (other sessions were using the GPU: the same view read 0.7 to 18 ms). Not checked: driving a car into the wall, walking against it, night.
