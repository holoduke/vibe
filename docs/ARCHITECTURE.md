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
tools/                       scripts (character build, rock bake, split_shader.pl, split_cpp.pl)
docs/                        this file, design notes, screenshots
```

Everything is in `namespace qlike`. Runtime files live in the project root:
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
| World editor (item bar, painting, replant) | `veg_zones.cpp` |
| Castle meshes, props, doors, lanterns, flags | `castle_mesh.cpp`, `props.cpp`, `world.cpp`, `world_textures.cpp` |
| Combat: gun, projectiles, airstrike, jets, AA | `combat.cpp`, `aa.cpp`, `fx.cpp` |
| Player body (soldier, hands, walk cycle) | `player_body.cpp`, `flashlight.cpp` |
| LAN multiplayer: discovery, host election, snapshots, interpolation | `src/engine/net/net.h`, `net.cpp` (ENet; no engine types) |
| Multiplayer game side: other players (avatars), their shots, hits, health, HUD | `multiplayer.cpp` |
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
| `terrain_raymarch.frag` | the raymarched ground AND the water (built twice, `QLIKE_WATER`) | `terrain_rm_noise_road.glsl`, `terrain_rm_pebbles.glsl`, `terrain_rm_heightfield.glsl`, `terrain_rm_shading_helpers.glsl`, `terrain_rm_water.glsl` |
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
| Hands, arms, walking animation | `player_body.cpp` |
| Player speed, jump, crouch | `src/game/player.cpp` |
| Anti-aircraft vehicles | `aa.cpp` |
| Loading screen | `loader_screen.cpp`, art in `assets/loader/loading_bg.jpg` |
| Key bindings | `src/engine/window.cpp` (reads keys), `vk_engine.cpp` `run()` (acts on them) |

## Working on it

- Build: `./build.ps1 -Config Release`. Run from the project root: `build/vibe.exe`.
- Capture one frame: `build/vibe.exe --frames 260 --screenshot out.png --screenshot-after 230`
  (PNG if the path ends in `.png`). The pose comes from `vibe_settings.cfg`
  (`player_pos_*`, `player_yaw` and `player_pitch` in radians, `time_of_day`).
- Test hooks (environment variables): `QLIKE_TEST_HOLD=1` pin the camera,
  `QLIKE_TEST_CRAWL=1` crouch, `QLIKE_TEST_SIZE=3440x1400` window size,
  `QLIKE_TEST_EDIT=1` open the editor, `QLIKE_TEST_ZONE=<preset>` paint a test
  patch, `QLIKE_TEST_LIGHTNING=1`, `QLIKE_TEST_FIRE=1`, `QLIKE_TEST_MENU=1`.
- GPU cost per pass: the `gpu-passes ...` line in `vibe.log` (once a second).
  Measure at the played resolution (`QLIKE_TEST_SIZE`), not the 1280x720 default.
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
- Internal names keep the old `qlike` / `QLIKE_` prefix (namespace, test
  hooks, the `qlike_engine` library); the product name is Vibe.

## Multiplayer (LAN)

Model and rules are described at the top of `src/engine/net/net.h`. In short: every instance
broadcasts a beacon (UDP 47777); the lowest id hosts (ENet, UDP 47778+), the rest join; when the
host leaves, the others elect again. Each client simulates its own player and sends its state 30x
a second; the host relays snapshots at 60 Hz; other players are drawn 100 ms in the past,
interpolated. Shots / hits / deaths are reliable events; a hit is decided on the shooter's machine
against the avatar as drawn there (`mp_bolt_hits`).

Other players reuse the local soldier pose solver: `mp_pose_avatars` swaps their state into
`player_` / the body animation members, runs `update_player_body` into their own vertex buffer
(`mp_pose_dst_`), and restores. `draw_remote_players` draws body + gun through `draw_viewmodel`
with their camera matrix (`mp_draw_`).

Off for automated runs (`--frames`, `--screenshot`) unless `QLIKE_NET=1`; `QLIKE_NO_NET=1` turns it
off entirely. Two instances on one PC work (the second takes the next game port).

## Levels of detail and shader builds

- **Trees / bushes / rocks** (`vegetation.cpp`, `veg_cull.comp`): GPU-driven - every instance is frustum- and Hi-Z-culled and given one of 4 mesh levels (or an impostor) on the GPU; drawn with indirect draws, near levels first. The colour pass tests depth EQUAL against the pre-pass.
- **Small plants** (`scatter.cpp`): 32 m cells, culled on the CPU (frustum, distance, walls, hills). Each species has three meshes: full, ~32 % (beyond 22 m) and ~11 % (beyond 48 m), made at load time with meshoptimizer (`simplified` in `build_scatter`). `QLIKE_NO_PLANT_LOD=1` turns them off.
- **meshoptimizer** (FetchContent) is available engine-wide: simplification, index / vertex ordering, meshlets. `clusterlod.h` from the same project is the starting point for Nanite-style cluster LOD if that is built.
- **terrain_raymarch.frag** is compiled three ways: normal (with winter: snow layer, footprints, ice), `QLIKE_SUMMER` (no winter code; bound while there is no snow) and `QLIKE_WATER`.
