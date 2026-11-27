# Vibe - an open-world Vulkan game engine with ray tracing, written by AI

Vibe is a 3D game engine and first-person game in C++20 and Vulkan 1.3, written
entirely by AI (vibe coding with Claude Code): hardware ray tracing, a ray-marched procedural
planet, motion-matched characters, cars to drive, a world with its own people and traffic, and
LAN multiplayer.

The goal is see how far you can get with AI. Feel free to let your agent work on this game. Create pull requests if you want.

Important: This engine was written by AI for humans. No code was (and never will be) seen by humans in development. Same for object, sounds, animations and level design. Everything is done by prompts only

![Vibe game engine: soldiers running beside a jeep on the lane along the river (Vulkan, C++)](docs/screenshots/lane_soldiers_jeep.jpg)

| | |
|---|---|
| ![The castle on its lake, seen over the forest, snow mountains behind](docs/screenshots/castle_lake_forest.jpg) | ![The castle gate and its cobbled road](docs/screenshots/castle_gate.jpg) |
| ![The home valley from above: river, roads and snow ridges](docs/screenshots/valley_from_above.jpg) | ![The coast highway under the mountains, a car on it](docs/screenshots/highway_mountains.jpg) |
| ![A lamp on the cobbled path at night](docs/screenshots/lamp_path_night.jpg) | ![The chapel lit at night](docs/screenshots/chapel_night.jpg) |
| ![A field of yellow flowers by the launch tower, traffic on the highway](docs/screenshots/flower_field_launch.jpg) | ![The launch site on the coast from the air](docs/screenshots/launch_site_coast.jpg) |
| ![Inside the diner: checkered floor, posters, guests](docs/screenshots/diner.jpg) | ![Inside the launch office: desks with screens, glass walls](docs/screenshots/launch_office.jpg) |
| ![The coast from high up, snow fields and the sea](docs/screenshots/coast_high.jpg) | ![The planet under clouds from the edge of space](docs/screenshots/planet_clouds.jpg) |
| ![The procedural planet from orbit, a continent under clouds](docs/screenshots/planet_orbit_2.jpg) | ![A castle on a lake in an open world](docs/screenshots/castle_lake.jpg) |
| ![Another player, a soldier, in daylight](docs/screenshots/soldier_day.png) | ![The same soldier at night under a lamp](docs/screenshots/soldier_night.png) |
| ![A sports car on the coast highway, snow mountains behind](docs/screenshots/car_bend.jpg) | ![The highway running toward the mountains, the car seen from above](docs/screenshots/car_road.jpg) |
| ![Inside the launch pavilion: glass walls, the rocket outside](docs/screenshots/pavilion.jpg) | ![A river valley at dusk, the launch tower on the horizon](docs/screenshots/valley_dusk.jpg) |
| ![The home valley and the launch pad from high up](docs/screenshots/high.jpg) | ![The rocket lifting off, seen from the air](docs/screenshots/launch.jpg) |
| ![The coast from the edge of space](docs/screenshots/coast.jpg) | ![The procedural planet from orbit](docs/screenshots/planet.jpg) |

## The technology

A native C++20 application on Vulkan 1.3, with no game engine underneath: about 85,000 lines of
C++ and 26,000 lines of GLSL in 101 shader files, with 160 unit tests. Windowing, math, physics, audio and networking
come from small libraries; the renderer, the world and the tools are written here.

The lists below name each technique and, where it follows published work, its source. Most are
switches in the settings menu, so they can be compared live.

### Ray tracing

Hardware ray tracing is required. Rays are traced inline from fragment and compute shaders
(ray queries, no separate ray-tracing pipeline).

| Technique | What it does |
|---|---|
| Acceleration structures | One bottom-level structure per mesh, static geometry merged, one top-level structure refit as things move; instance masks select what a ray may hit |
| Soft shadows | Sun shadows with a blocker search and a variable cone (PCSS-style), so they sharpen at the contact and soften with distance |
| Global illumination | Multi-bounce diffuse light from traced paths |
| Radiance cache | World-space spatial hash of outgoing light (SHaRC-style); a ray that hits a cached cell gets several bounces for the price of one |
| ReSTIR GI | Reservoir resampling of GI samples over time (optional) |
| Ambient occlusion | Screen-space GTAO by default, ray-traced AO as the high-quality choice |
| Reflections | Traced reflections of the scene on water and glossy surfaces |
| Light shafts | The volumetric fog traces a ray to the sun per froxel, so shafts fall through trees and windows |
| Denoising | SVGF (temporal variance + à-trous filter), or DLSS Ray Reconstruction at about one sample per pixel |

### Raymarched shaders

Much of the picture has no triangles behind it: it is computed per pixel by marching a ray
through a formula.

| Technique | What it does |
|---|---|
| Terrain | A fractal height formula (layered noise with erosion) marched in the fragment shader; the number of noise octaves falls with distance; pebbles, puddles, snow, footprints and ice are shading detail |
| Terrain upscaling | The ground is marched at reduced resolution and rebuilt by its own temporal upscaler; the slow fields of the world (continents, ranges, forest) are cached in two textures round the camera |
| Water | Sea, lake and river surface with waves, flow, foam, a visible bed and an underwater view |
| Volumetric clouds | Marched in a compute shader, grown from a 2D weather map |
| Atmosphere | Physically based sky from look-up tables (Hillaire 2020): transmittance, multiple scattering, sky view and an aerial-perspective volume |
| Volumetric fog | 160 x 90 x 64 camera-aligned froxels (Wronski 2014, Hillaire 2015), with mist that lies on the lake |
| Stonework | Screen-space displacement mapping: walls are drawn flat, then a per-pixel grid moves every pixel out by the height of its stone, with real outlines and the stones shading each other (after a published game's method); parallax relief as the fallback |
| Explosions | Fireballs and smoke columns as marched noise volumes with a blackbody colour ramp |
| Voxels | A destructible tower traversed per pixel with a brick DDA |

### A planet-sized map

The world is a real sphere with a radius of 6360 km. Fly on round it and you come back home.

| Technique | What it does |
|---|---|
| One world formula | Continents, coasts, mountain ranges, the river and the ground under your feet come from one function, shared by the shaders and the CPU, so you can land, walk and swim anywhere |
| Hand-overs by altitude | Full-detail march near the ground; the march range grows to 80 km with height; then a shaded far field of the sphere takes over; above the atmosphere the sky gives way to space |
| One weather map | The same field grows the volumetric clouds below and paints the cloud systems seen from orbit: what you see from space is what you find on the way down |
| Charts of the sphere | The engine works in local flat coordinates that are re-anchored under the player every 16 km, so positions stay exact to the centimetre |
| Double precision where it matters | Camera matrices, terrain rays and the noise lattice are computed in doubles or relative to the camera; 32-bit floats cannot hold a position on a planet |
| Streamed vegetation | Beyond the home area the world is cut into 128 m cells whose trees are a pure function of their coordinates, generated on background threads, nearest first |

### Geometry at scale

| Technique | What it does |
|---|---|
| GPU-driven culling | Every tree, bush, rock and plant is culled in a compute shader (frustum + Hi-Z depth pyramid) and drawn with indirect draws |
| Levels of detail | Four mesh levels and an impostor per tree, picked on the GPU, with dithered cross-fades |
| Cluster LOD (Nanite-style) | Rocks and static meshes are cut into clusters of up to 128 triangles in a hierarchy (meshoptimizer); per cluster the level with under one pixel of error is drawn |
| Geometry streaming | The finest cluster level is read from disk on a background thread when the camera comes near |
| Grass | No list of grass exists: every frame a compute shader places turf, long tufts and flowering meadow round the camera from rules of the ground, culls them and draws them as instances. Blades bend to a wind field simulated as a fluid round the camera and lie down where somebody walked or drove |
| Procedural trees and plants | Trees are grown at load time, plants and their simplified levels are generated in code; far trees are impostors with depth |
| Mesh terrain | A second terrain renderer with continuous distance-dependent LOD (CDLOD), next to the raymarched one |

### Image

| Technique | What it does |
|---|---|
| Deferred lighting | The scene is drawn into a G-buffer after a depth pre-pass and lit in one pass |
| Upscaling | NVIDIA DLSS (Streamline) or AMD FSR 3.1, with the engine's own TAA and temporal upsampler as fallback |
| Frame generation | DLSS Frame Generation (up to 4x) or FSR 3; AMD Anti-Lag 2 |
| HDR | HDR10 output where the display offers it; ACES, AgX or Khronos PBR Neutral tone mapping |
| Auto-exposure | Eye adaptation from the scene's average luminance |
| Bloom and lens flare | 13-tap downsample and tent upsample chain (Call of Duty: Advanced Warfare), ghosts and halo |
| Decals | Deferred decals that wrap laser burns over any surface |
| Glass | A refractive glass pass for windows and windscreens: what lies behind is bent, tinted and reflected over |
| Shading rate | The ground is shaded more coarsely where the picture moves fast |
| Colour grading | Temperature, vibrance, split toning, film grain and vignette |

### Characters

| Technique | What it does |
|---|---|
| Motion matching | Every frame the next pose of the soldier is searched in a database of motion capture by where the player is heading; legs, hips and stride are fitted to the real speed and ground |
| Hands and aim | The arms are placed on the weapon by inverse kinematics every frame; the head and eyes look where the player looks |
| Action clips | Sitting down, eating, getting into a car, serving at a counter: clips made from text with a motion model, played on the same skeleton |
| Death | Somebody who is shot first reacts - a clip chosen by where the shot struck and what they were doing - and as the body loses its balance the physics takes it over as a ragdoll |
| Many at once | Animation levels of detail by distance, poses computed on worker threads, skinning on the GPU, ray-traced shadows for the nearest |

### A world with life in it

| | What it does |
|---|---|
| Cars | A jeep, sports cars, saloons and pickups to drive: suspension, tyre grip, traction control, turbo, tyre marks and smoke, lights, a passenger seat. Each kind has its own engine and gearbox - a V8 that revs to 9000, a low-revving V6, a quiet four-cylinder |
| Sound in space | The sound of every car is recorded from a physical engine simulation at steps of engine speed and mixed by revs and load. One spatialiser gives every moving source its distance, Doppler shift, muffling behind what stands in the way and the echo of its surroundings |
| Roads | A coast highway and a river lane are built from a centre line: surface, markings, guard rails, signs, lamps and bridges, with the terrain cut to fit |
| People and traffic | An NPC engine runs a hundred agents: cars drive the highway, pull in at the service station, fuel up and park; people get out, shop, eat in the diner and drive on. Drivers differ - slow, fast, overtaking, a speeder with the military police behind him. Any standing car can be taken |
| Places | A castle on a lake, a chapel, a launch site with a rocket and a glass operations pavilion, a service station with fuel pumps, a shop and a diner - modelled by script in Blender, with interiors |
| Sites | Hand-made places are data, laid over the procedural planet; the design is meant for thousands of them |
| Weapons | A laser rifle, an automatic assault rifle and an airstrike called in on a tablet; bolts are physical bodies that knock things over |

### Simulation and tools

| Technique | What it does |
|---|---|
| Physics | Rigid bodies, vehicles and ragdolls with Jolt |
| LAN multiplayer | Up to eight instances find each other, elect a host and share players, shots, cars, the people and traffic of the world, weather, season and clock (ENet); other players and cars are interpolated 100 ms in the past. Characters are sent as position, direction and action - each machine animates them itself |
| Orbital mechanics | The rocket flies a real mission in double precision: gravity turn, 180 km circular orbit, one revolution, a retrograde burn and a powered landing on its mount |
| Day, seasons and weather | A sun, moon with phases and stars; four seasons and eleven kinds of weather that change the land and the light |
| Job system | Background threads for world generation and streaming |
| Self-inspection | The command line renders a chosen frame to a file and the log gives GPU time per pass, which is how the AI looks at and measures its own work |

## Requirements

Windows and a fast, recent graphics card with hardware ray tracing. Tested on a GeForce RTX 4080
(DLSS) and a Radeon RX 9070 XT (FSR). There is no low-end fallback.

## Download

A ready-to-run Windows build is on the [releases page](https://github.com/holoduke/vibe/releases):
unzip it and start `Vibe.cmd`. The first start prepares shaders for your graphics card in the
background; the terrain looks plain for about a minute.

## Build and run

Visual Studio 2022 and the Vulkan SDK 1.3. Other dependencies are fetched by CMake.

```powershell
./build.ps1 -Config Release
./build/vibe.exe        # run from the repository root
```

## Controls

| Key | Action |
|---|---|
| W A S D, mouse | Move, look |
| Space / Shift / Ctrl | Jump / sprint / crawl. In water: rise / swim fast / dive |
| Left mouse | Fire, or mark the airstrike target |
| Right mouse | Aim down the sights |
| R | Reload |
| 1 / 2 / 3 | Laser rifle / airstrike tablet / assault rifle |
| V | Fly |
| H | Back to the start |
| E | Get into or out of a car; at the launch button: start the count; elsewhere: world editor |
| In a car | W A S D drive, Space handbrake, Shift turbo, L lights, middle mouse changes the view |
| C | During a launch: ride along with the rocket (mouse turns the camera) |
| F | Flashlight |
| F12 | Save a screenshot |
| Esc | Menu with all settings and key bindings (a controller works too) |

## Code

- `src/engine/` — the Vulkan engine
- `src/anim/` — character animation (motion matching, hands, action clips)
- `src/npc/` — the people and traffic that live in the world
- `src/game/` — player, collision, level
- `tools/` — the Blender scripts that build the places, the animation cooker, benchmarks
- `tests/` — unit tests
- `shaders/` — GLSL 4.60
- `docs/ARCHITECTURE.md` — the code map, with the details behind every technique above

## Credits

- Volumetric clouds: ported from
  [leoawen/volumetric_cloud_atmosphere_scattering](https://github.com/leoawen/volumetric_cloud_atmosphere_scattering) (MIT).
- Procedural trees: a C++ port of [dgreenheck/ez-tree](https://github.com/dgreenheck/ez-tree) (MIT).
- Sea waves: after "Seascape" by Alexander Alekseev (TDM), [shadertoy.com/view/Ms2SD1](https://www.shadertoy.com/view/Ms2SD1) (CC BY-NC-SA 3.0).
- Screen-space displacement mapping: a published game's method, written after studying how it renders; a first attempt was after
  [TechArtAlex](https://www.patreon.com/TechArtAlex/posts/screen-space-171617266).
- Car engine sound: recorded from [ange-yaghi/engine-sim](https://github.com/ange-yaghi/engine-sim) (MIT).
- Action and death clips: made with Tencent HY-Motion 1.0 (text to motion), under its community licence, which excludes the EU, the UK and South Korea.
- Weapon models (LR-9 laser rifle, MR-7 rifle): original, made for this project in Blender.
- Player character: [MakeHuman / MPFB](https://static.makehumancommunity.org/) community assets;
  tactical vest by MindFront (CC-BY).
- Character movement: motion capture from the 100STYLE dataset by Ian Mason, Sebastian Starke and Taku Komura
  (CC BY 4.0, https://www.ianxmason.com/100style/), retargeted to the soldier (`tools/anim/cook_main.cpp`, see `docs/animation/`).
- Running motion: the CMU Graphics Lab Motion Capture Database (mocap.cs.cmu.edu; created with funding from NSF EIA-0196217).
- Skeletal animation runtime: ozz-animation by Guillaume Blanc (MIT, https://github.com/guillaumeblanc/ozz-animation).
- Libraries: GLFW, glm, Dear ImGui, VMA, vk-bootstrap, Jolt Physics, meshoptimizer, miniaudio,
  NVIDIA Streamline, AMD FidelityFX.

## Licence

The code and shaders written for this project are under the [MIT licence](LICENSE). Third-party
material (motion capture, the sea wave function, character assets, textures, vendor libraries)
keeps the licence of its source; see the list above and [THIRD_PARTY.md](THIRD_PARTY.md).
