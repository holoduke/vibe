# Character animation — architecture, findings and plan

Source of the requirements: `AGENT_ANIMATION_IMPLEMENTATION.md` (animation rewrite, multiplayer
extension, 1,000-player extension). This file is the Phase 0 deliverable and is kept current as
phases land. Status per phase is in the last section.

## 1. What the engine is (findings)

| Topic | Finding | Where |
|---|---|---|
| Language / build | C++20, MSVC (`/W4 /permissive-`, Release `/O2 /GL /arch:AVX2`, no `/fp:fast`), CMake + Ninja, dependencies by `FetchContent` pinned to tags | `CMakeLists.txt`, `build.ps1` |
| Targets | `qlike_core` (gameplay, no Vulkan; what tests link), `qlike_engine` (Vulkan), `vibe` (exe), `qlike_tests` (Catch2 3.7.1, 30 tests) | `CMakeLists.txt`, `tests/` |
| Platform | Windows x64, Vulkan 1.3. Other platforms have compile switches but are not built | |
| Math | glm 1.0.1 (`vec3`, `quat`, `mat4`), radians, depth 0..1 | |
| Coordinates | Right-handed, +Y up, metres. Yaw 0 looks down -Z; forward = `(sin yaw, 0, -cos yaw)`, right = `(cos yaw, 0, sin yaw)` | `game/player.cpp` |
| Character model space | +X right, +Y up, -Z forward (same as view space), feet at the origin | `player_body.cpp` |
| Skeleton | `assets/character/soldier_fp.qchar`: 53 joints (`Root` > `Hips` > spine x3, neck, head, 2 x (shoulder, arm, forearm, hand, 15 finger joints), 2 x (upleg, leg, foot, toe)). Joints have a rest position (`head`) and NO rest rotation: every joint's rest orientation is identity in model space, a pose is "rotation from rest" | `tools/character/export_fp.py`, `init_player_body` |
| Mesh / skinning | 43,155 vertices, 78,832 triangles, 4 weights per vertex, skinned on the CPU into a per-frame vertex buffer, drawn by the ordinary mesh shader. The head is not in the mesh (first person); other players get a box head | `player_body.cpp`, `props.cpp` |
| Camera | First person. The body is seen from outside only on other players, on corpses and through the test hooks | |
| Character controller | Custom Quake-style AABB mover at a fixed 120 Hz tick; the render frame interpolates the camera | `game/player.cpp`, `vk_engine.cpp` |
| Rigid bodies | Jolt 5.2 (props, projectiles, ragdolls). The player is not a Jolt body | `physics_world.cpp` |
| Threads | A job system (`engine/jobs.h`); the frame loop is single threaded apart from named jobs | |
| Serialization | Hand-written little-endian binary files with a 4-char magic | |

### Movement specification (as implemented, `game/player.h`, `kPlayerMove`)

| Quantity | Value |
|---|---|
| Ground speed | 8 m/s; sprint x1.5 = 12 m/s; crawl x0.4 = 3.2 m/s; wading up to -55 % |
| Ground acceleration / friction | Quake `PM_Accelerate` 10, friction 6, stop speed 2.5 m/s |
| Air control | `PM_AirAccelerate` 10 with wish speed capped at 0.75 m/s (strafe jumping gains speed) |
| Gravity | 25 m/s^2, a menu setting (`RtSettings::gravity`) |
| Jump | 8.5 m/s (1.45 m); one air jump: x1.10, up to +20 % more with horizontal speed 8..12 m/s |
| Airtime | ~0.68 s for a ground jump on flat ground; unbounded with terrain, double jumps and the gravity setting |
| Midair turning | Unrestricted (yaw is the mouse, applied at frame rate) |
| Crouch | `crouch_factor` 0..1 eased at 8/s; lowers the eye only, the hitbox does not change |
| Swimming | Separate branch (3D wish direction, drag, buoyancy) |

None of these are changed by the animation work. The animation reads them (`kPlayerMove`) for its
trajectory prediction, so retuning the controller retunes the prediction.

### Animation as found (what is replaced)

- `engine/motion_match.cpp` (290 lines): database load, mirror at load, brute-force search over
  27 features, inertialisation, captured jumps. Model-space rotations per database bone.
- `vk_engine/player_body.cpp` (784 lines): one function that mixes input shaping, motion query,
  jump state, stride / orientation warping, foot locks, arm IK to hard-coded view-space points,
  finger curls, test hooks and CPU skinning, all on `VulkanEngine` members.
- Other players: `mp_pose_avatars` swaps a remote player's state into `player_` and the body
  members, runs the local solver, and restores them.
- `tools/anim/build_motion_db.py`: BVH import, retarget, feature extraction (needs numpy).
- No tests cover any of it. No clip runtime, no blending, no layers, no debug views.

Baseline (Release, 9800X3D, body shown with `QLIKE_BODY_FRONT`, running at 6 m/s):
`player body` 0.80 - 0.90 ms per frame for one character (search + pose + skinning 43k vertices);
database 14.8 MB on disk, about 46 MB in memory (93,838 frames with mirrors x (27 features +
8 header floats + 22 quaternions)). Capture: `docs/animation/captures/baseline_run6.png`.
Visible artefacts: the legs cycle fast at 8 - 12 m/s (playback x1.6 and stride x2.4), feet are
only locked below 4 m/s, a long fall holds one frame, arms are rigid to the weapon, no lean.

### Networking as found (N0 / S0 audit)

| Topic | Finding |
|---|---|
| Transport | ENet 1.3.18 over UDP: channel 0 unreliable (state), channel 1 reliable (events, hello, world) |
| Topology | LAN only. Discovery by UDP broadcast beacon (47777); the instance with the lowest id becomes a **listen host**, others join it; re-election when it leaves. No dedicated server |
| Authority | **Each client is authoritative for its own player** (position, velocity, view). The host relays. There is no server simulation of other players, so no prediction error and no reconciliation |
| Rates | Clients send state at 30 Hz; the host relays one snapshot with everybody at 60 Hz |
| Snapshot | `PlayerState`, 38 bytes, raw floats: position, velocity, yaw, pitch, crouch, flags (ground, swim, flashlight), health. No quantisation, no delta, no interest management |
| Interpolation | Remote players are drawn 100 ms in the past between two states; up to 250 ms of extrapolation; a jump of more than 6 m is a teleport |
| Events | Reliable `Event` (Fire, Hit, Death, Leave, Airstrike) relayed by the host. No sequence numbers, no timestamps |
| Hits | Decided on the **shooter's** machine against the avatar's box as drawn there (`mp_bolt_hits`); the victim applies the damage it is told |
| Limits | `kMaxPlayers = 8`, fixed arrays |
| Not replicated | Active weapon (remote players always show the laser), movement input, jump, any action other than Fire |

Consequences for the requirements document, which assumes a server-authoritative game:

1. Remote animation can be reconstructed from replicated state on this netcode as it is (N1 - N4).
   Hit decisions already do not read a pose: they test an axis-aligned box at the replicated position.
2. "Server-authoritative movement with client prediction and reconciliation" does not exist and is
   a rewrite of the game's simulation model (the host would have to simulate every player against
   the level, terrain, water and props). This is recorded as blocker B1 below; the animation system
   does not depend on it — it takes "the state the netcode hands it", wherever that was decided.
3. 1,000 players in one world is far outside what the session code was written for; see
   `docs/network/1000_player_capacity_plan.md` (Phase S0) for measurements and the staged plan.

## 2. Dependencies

| Library | Decision | Licence | Notes |
|---|---|---|---|
| **ozz-animation 0.17.0** | Adopted: skeleton, compressed clips + sampling, blending, local-to-model; its two-bone and aim IK jobs are available through the interface and tested. The animator fits limbs with its own roll-preserving solver (`CharacterAnimator::fit_limb`): ozz's two-bone job needs a fixed hinge axis per joint, which this rig (rest orientations all identity) does not give for arms | MIT | `FetchContent`, tag pinned. Only `ozz_base`, `ozz_animation`, `ozz_animation_offline` are built (tools, samples, FBX, glTF, tests off). Hidden behind `anim::IAnimationRuntime`; no ozz header is included outside `src/anim/ozz_runtime.cpp` and `src/anim/offline/` |
| Assimp | Not adopted | BSD-3 | The only source format in use is BVH; a 150-line reader is more reliable than Assimp's BVH importer and adds no dependency. glTF is already covered by tinygltf should a rigged glTF source appear |
| ACL | Not adopted | MIT | ozz's own compressed clips are measured first (section 7). A second sampling path is only justified by a measured need |
| Motion matching, inertialisation, trajectory prediction, contacts, foot lock, warping, shooter layers | Written for this engine in `src/anim/` | MIT (project) | No suitable library |

Update policy: bump the tag in `CMakeLists.txt`, run `qlike_anim_tests` (bind-pose parity and
sampling tests catch format or convention changes), re-cook the assets (the runtime archive
version is checked at load).

## 3. Assets and licences

| Asset | Source | Licence | In repo |
|---|---|---|---|
| Soldier mesh + skeleton | MakeHuman / MPFB community assets, built by `tools/character/` | CC0 and CC BY (credited in README) | yes (`soldier_fp.qchar`) |
| Locomotion, crouch, two-foot jumps | 100STYLE (Mason, Starke, Komura), BVH 60 fps | CC BY 4.0 (dataset). The research code released with the paper is a separate work and is not used | cooked data only; raw BVH stays outside the repo |
| Aim, recoil, reload, air and landing layers | Procedural, written for this project | MIT (project) | code |

Not used: LaFAN1 (CC BY-NC-ND), any clip without a stated licence. 100STYLE has no aiming, firing,
reload, fall or hard landing; those are layers and procedural phases here. Every cooked clip
carries its source file, frame range and licence string in the asset (`ClipInfo::provenance`, `ClipInfo::licence`).

## 4. Target architecture

```
tools/anim/vibe_animcook (offline)                    src/anim (runtime library qlike_anim: glm + ozz, no Vulkan)
  BVH -> retarget to the soldier rig                    IAnimationRuntime  <- OzzRuntime (skeleton, clips, blend, FK, IK)
      -> validate (joints, units, contacts, rates)      MotionDatabase     versioned features + metadata
      -> ozz clips (compressed)                         MotionSearch       exhaustive, AVX2, hysteresis
      -> motion database (features, tags)               TrajectoryPredictor  the controller's own movement model
      -> assets/character/soldier.qanim                 Inertializer       quaternion-safe offsets, per joint
                                                        AirborneModel      take-off / rise / apex / fall / land by physics
                                                        Layers             aim, recoil, reload, lean, warps
                                                        FootIK, ArmIK      contacts, locks, ground, weapon hands
                                                        CharacterAnimator  input -> pose (one per character)
                                                        AnimLod, debug data
src/engine/vk_engine/player_body.cpp   builds CharacterAnimationInput from the controller (local) or
                                       from the interpolated network state (remote), skins, draws
```

Runtime order per character (fixed, documented because layers must not fight):

1. Movement mode from the controller state (ground / air / swim) — never from animation.
2. Trajectory prediction with the controller's model.
3. Lower body: motion matching (ground), airborne phase model (air), directional blend fallback
   when the database does not cover the request.
4. Clip sampling -> local pose. Inertialisation of the change.
5. Orientation warp (legs to travel direction; a diagonal request is searched as the nearest captured
   direction), lean, the chest held near its place under the view, stride warp targets.
6. Upper body: aim yaw / pitch spread over pelvis, spine, neck, head with limits.
7. Local -> model.
8. Feet: stride (a foot beyond the leg's reach leaves the ground), world locks, ground probes, pelvis
   compensation, two-bone IK with a knee direction that does not depend on a straight leg.
9. Weapon: the weapon frame comes from gameplay aim; reload / equip layer moves the hands' targets;
   additive recoil; two-bone IK for both arms; finger poses. IK runs after every additive layer.
10. Local -> model for the changed chains; skinning matrices.

Invariants (tested): the animator has no write access to the controller; its input is a `const`
struct; it is deterministic for a given input sequence; no allocation in `update` after warm-up.

## 5. Integration points (exact files)

| Change | File |
|---|---|
| New library `qlike_anim`, cook tool, tests | `CMakeLists.txt`, `src/anim/**`, `tools/anim/cook_main.cpp`, `tests/anim/**` |
| Body: load, per-character animator, skinning, draw | `src/engine/vk_engine/player_body.cpp`, members in `vk_engine.h` |
| Remote players: animator per avatar, no state swapping | `src/engine/vk_engine/multiplayer.cpp` |
| Ragdoll hand-over (reads the model-space pose) | `src/engine/vk_engine/ragdoll.cpp` |
| Weapon draw from the animator's weapon frame | `src/engine/vk_engine/props.cpp` (`draw_viewmodel`) |
| Protocol: aim / weapon / input bits, timestamped events | `src/engine/net/net.h`, `net.cpp` |
| Removed | `src/engine/motion_match.*`, `tools/anim/build_motion_db.py`, `assets/character/motion.qmm` |

## 6. Phases

| Phase | Content | State |
|---|---|---|
| 0 | This document, baseline | done |
| 1 | ozz runtime behind `IAnimationRuntime`, BVH import, retarget, validation, cook tool | done, tested |
| 2 | Motion database, AVX2 exhaustive search, trajectory prediction from the controller's model | done, tested |
| 3 | Inertialisation, orientation + stride warp, lean | done, tested. Distance matching for stops: not built (see gaps) |
| 4 | Jumps, falls, landings from controller state | done, tested |
| 5 | Foot locks, ground adaptation, pelvis compensation, weapon hands, aim, recoil, reload, equip | done, tested |
| 6 | Benchmarks, levels of detail, debug overlay, documentation | done (`performance.md`, `guide.md`) |
| N0 - N3 | Network audit; replicated aim / weapon / keys / shot counter; stamped, de-duplicated events | done, tested |
| N4 | Interpolation that follows the link, loss / jitter / stall tests, net debug figures | done against a modelled link; see gaps |
| N5 | Two-client acceptance from the observer's side | partly: one real two-instance run, standing and firing (`captures/mp_observer.png`) |
| S0 | Capacity audit | written (`docs/network/1000_player_capacity_plan.md`); server-side measurements not possible on this netcode |
| S1 - S4 | Interest management, bandwidth budgets, server scaling, load tests | not started: blocked on B1 |
| S3 (client) | Animation levels of detail, per-level cost at 1 .. 1000 characters | done and measured |

## 7. Risks and open blockers

- **B1 - authority model (open, owner decision).** The requirements assume a dedicated authoritative server; the game is
  a LAN listen-host relay with client-authoritative movement and shooter-decided hits. Moving the
  movement simulation to the host is a gameplay / netcode project of its own and needs an owner
  decision (dedicated server? cheating in scope for a LAN game?). Animation work proceeds on the
  contract "replicated state + timestamped events in, pose out", which holds under either model.
- **B2 — no reload, no weapon switch animation in the game design.** The laser has no magazine.
  Reload exists here as a cosmetic action so the layer, its replication and its interruption can
  be built and tested; whether the game gets ammunition is a design decision.
- **B3 — speed range.** Capture data tops out near 3 m/s; the game sprints at 12 m/s. Coverage
  is extended by bounded playback rate, stride warping and lean, not by more leg cycles; a real
  sprint capture with rights would improve it (asset gap, section 3).
- **B4 — first-person mesh.** The soldier has no head geometry. Third-person views use a
  stand-in head; a full third-person mesh is an asset task (`tools/character/`).
- **B5 — 1,000 players.** An engineering target to be measured, not a promise; see the capacity plan.

## 8. Status log

### Result (2026-11-21)

The game runs on the new system; `engine/motion_match.*`, `tools/anim/build_motion_db.py` and
`assets/character/motion.qmm` are gone. Build: `./build.ps1 -Config Release -Test` - 101 tests
(30 existing + 71 in `qlike_anim_tests`), all passing. The animation library and its tests also
build and pass in Debug (MSVC checked iterators); the whole game was only built in Release.

Second pass, same day: skinning moved to the GPU (`cube_skin.vert`); a real two-instance run with
a player moving by itself; and the fast gaits measured and reworked (`performance.md`, "Foot slide
and joint rates").

| Area | Files |
|---|---|
| Library | `src/anim/` (types, skeleton_def, asset, runtime + ozz_runtime, motion_db, trajectory, motion_matcher, inertializer, airborne, character_animator, anim_lod) |
| Offline | `src/anim/offline/` (bvh, raw_clip, retarget, cook, motion_build), `tools/anim/cook_main.cpp`, `tools/anim/bench_main.cpp` |
| Engine | `vk_engine/player_body.cpp` (rewritten), `multiplayer.cpp`, `ragdoll.cpp`, `props.cpp`, `combat.cpp`, `vk_engine.cpp`, `vk_engine.h` |
| Network | `engine/net/net_state.h` (new), `net_anim.h` (new), `net.h`, `net.cpp` |
| Tests | `tests/anim/` (skeleton, runtime, import, motion, animator, net) |
| Asset | `assets/character/soldier.qanim` (33 clips, 87,224 frames, 490 jumps; 26.8 MB) |

Measured against the baseline (same machine, body shown from outside, running):

| | Before | After |
|---|---|---|
| `player body` per frame (animation + skinning) | 0.80 - 0.90 ms | 0.08 - 0.40 ms on the CPU (animation only); skinning on the GPU, ~0.01 ms a body |
| Data in memory | ~46 MB (frames stored twice for the mirror) | 26.7 MB (14.4 MB clips + 12.3 MB database; mirror searched, not stored) |
| Search | every 0.2 s, scalar | every 0.1 s, AVX2: 0.2 - 0.45 ms over 86,135 frames x 2 |
| Tests | none | 70 |

### Acceptance criteria

| # | Criterion | State |
|---|---|---|
| 1 | Builds cleanly, existing tests green | pass |
| 2 | Demo map showcase | not built as a map; every listed movement runs in the scenario tests and can be shown with the `QLIKE_BODY_*` hooks |
| 3 | No obvious foot sliding in planted phases | pass below 4 m/s (locks, < 1.5 cm in tests) and at the crawl pace (0 - 1 %). At walk and sprint: 9 - 12 % of the body's speed forward, diagonally and sideways (a strafe run), **19 - 25 % backward** (measured, `performance.md`; bounded by the `[asset]` test). Not zero: the capture tops out near 2 m/s (gap G1) |
| 4 | No pops, NaNs, oscillation | pass in tests (joint rates bounded, 0 invalid poses, switch rate bounded at 5 - 10 Hz input), and on the real asset at walk and sprint in every direction (fastest joint 17 - 38 rad/s; it was 100 - 184 until the arm and knee solves were fixed) |
| 5 | Input never delayed or altered | pass: controller trajectories bit-identical with and without animation |
| 6 | Debug tools | pass (`QLIKE_ANIM_DEBUG=1`) |
| 7 | Automated tests | pass |
| 8 | Benchmarks recorded, budgets agreed | recorded (`performance.md`); budgets are the owner's to set |
| 9 | Licences and provenance | pass (per clip in the asset; `THIRD_PARTY.md`) |
| 10 | Integration guide, quality gaps | `guide.md` |

Judged by eye only from stills: nobody has watched this in motion yet. The tests prove it is
continuous, bounded and responsive; they cannot say it looks good.
