# Character animation — working with it

How the system is used, fed, tuned and debugged. Design and findings are in `architecture.md`,
measurements in `performance.md`.

## The shape of it

```
src/anim/                    library qlike_anim: glm + ozz, no Vulkan, no engine headers
  runtime.h, ozz_runtime.cpp   skeleton, clips, sampling, blending, FK, IK (IAnimationRuntime; ozz behind it)
  motion_db.*                  the motion-matching database and its search
  trajectory.*                 where the controller will take the character (its own movement model)
  motion_matcher.*             which captured frame plays on the ground
  airborne.*                   which captured frame shows a jump / fall / landing
  inertializer.*               hides every change of source
  character_animator.*         one character: input -> pose (the order of layers is at the top of the header)
  anim_lod.h                   levels of detail
  offline/                     BVH import, retargeting, validation, cooking (never linked into the game)
src/engine/vk_engine/player_body.cpp   gathers the input, writes the bone palette, draws; debug overlay
shaders/cube_skin.vert                 skins on the GPU (bone palette: binding 33 of the scene set)
src/engine/net/net_state.h, net_anim.h what is replicated and how it becomes a remote animator's input
tools/anim/cook_main.cpp     vibe_animcook
tools/anim/bench_main.cpp    vibe_animbench
tests/anim/                  qlike_anim_tests
```

## Runtime API

```cpp
anim::AnimationSet set;                      // shared: skeleton, clips, database
set.load("assets/character/soldier.qanim", &why);
anim::CharacterAnimator a(set);              // one per character

anim::CharacterAnimationInput in;            // filled every frame from the controller (const to the animator)
in.dt, in.position, in.velocity, in.desired_velocity, in.facing_yaw, in.aim_pitch,
in.grounded, in.swimming, in.crouch, in.eye_local, in.held, in.shot_seq, in.reload_seq, ...
a.update(in);

a.model();          // joint matrices, model space (+X right, +Y up, -Z forward)
a.weapon_offset();  // what is carried is drawn at  view frame * this
a.debug();          // chosen clip, costs, contacts, phases, timings
```

The animator has no pointer to the controller and never writes to it. For another player the
input comes from `net::remote_animation_input(state, in)`; the code path after that is the same.
Actions are counters (`shot_seq`, `reload_seq`): a change triggers the layer, a repeat does not,
so a duplicated or late network event cannot replay an animation.

An animator's `update` is safe on any thread as long as one thread at a time uses that animator
(the set is read-only). `vibe_animbench` updates 1,000 on 16 threads.

## Cooking the asset

```powershell
./build.ps1 -Config Release
./build/vibe_animcook.exe C:/path/to/100STYLE assets/character/soldier_fp.qchar assets/character/soldier.qanim --cmu C:/path/to/cmu/bvh
```

Two seconds. Imports the takes listed in `default_style100_sources()` (`offline/cook.cpp`),
retargets them to the soldier, validates every clip (bone lengths, speeds, contacts, rates) and
refuses the cook on an error, compresses with ozz, builds the motion database. The raw capture
is not in the repository: 100STYLE, CC BY 4.0, https://www.ianxmason.com/100style/.

`--cmu <dir>` adds the runs of the CMU Graphics Lab Motion Capture Database (the takes listed in
`default_cmu_sources()`, as `09_01.bvh` ... in that directory; BVH conversions of the database are
widely mirrored, the originals are at http://mocap.cs.cmu.edu). They are the only fast running the
asset has (3.4 - 4.2 m/s on the soldier); without them the fastest take is 2 m/s. Each is a second
or two long (`kTagShortTake`: the search continues its end at its last pace). The steady ones are
also cooked played backward, as the fast back-pedal nobody captured (`--no-reversed` leaves that out).

Re-cook when: the character mesh / skeleton is re-exported (the asset's rig id must match; the
game says so in the log and draws no body otherwise), the ozz version changes, or the database
or asset version constants change.

### Adding clips

- Another 100STYLE style: add a line to `default_style100_sources()` with its tags
  (`kTagLocomotion`, `kTagCrouch`, `kTagJump`). Jump clips are searched for flights automatically.
- Another capture skeleton: write a `RetargetParams` like `style100_params()` (joint map, axis
  conversion) and call `retarget_bvh`. Other formats: produce a `RawClip` and hand it to the
  same validation and cook steps.
- Every clip must carry `provenance` and `licence`. Do not add LaFAN1 (CC BY-NC-ND) or anything
  without a stated licence.

## Tuning

`AnimTuning` (per animator; `character_animator.h`). Without a rebuild:
`QLIKE_ANIM_TUNE="halflife=0.06,lean=0,foot_lock=0"` — names are listed above
`apply_anim_tuning` in `player_body.cpp`. The switches (`foot_ik`, `foot_lock`, `stride_warp`,
`orient_warp`, `hand_ik`, `lean`, `air`) exist for side-by-side comparison.

The controller's movement model is read from `game::kPlayerMove`; change the controller and the
prediction follows.

## Looking at it

| Hook | Effect |
|---|---|
| `QLIKE_ANIM_DEBUG=1` | Overlay: clip and time, cost by feature group, predicted path, foot targets (green = locked), hand targets, muzzle line, phases, timings, levels of detail, network counters |
| `QLIKE_BODY_DEBUG=1` | One log line a second with the same |
| `QLIKE_BODY_FRONT=<m>`, `QLIKE_BODY_TURN=<deg>` | Stand the local body in front of the camera, turned |
| `QLIKE_BODY_WALK=<m/s>`, `QLIKE_BODY_STRAFE=<m/s>`, `QLIKE_BODY_AIR=<m/s>` | Drive it |
| `QLIKE_BODY_FIRE=<shots/s>`, `QLIKE_BODY_RELOAD=1` | Fire / reload in a loop |
| `QLIKE_TEST_AIM=1` | Hold the aim (right mouse) |
| `QLIKE_BODY_REST=1` | Rest pose |
| `QLIKE_BODY_CROWD=<n>` | Draw the local body n more times (GPU cost of n skinned characters) |
| `QLIKE_TEST_MOVE=1` | The player moves by itself on a 9 s loop of real key input: forward, jump, sprint, strafe, back, a turning arc |
| `QLIKE_ANIM_LOD=<0..3>` | Force other players' level of detail |
| `QLIKE_MESH_LOD=<0..2>` | Force which mesh every soldier is drawn with |
| `QLIKE_NO_BODY_SHADOW=1` | Soldiers are not drawn into the small-object shadow map (comparison) |
| `QLIKE_NO_BODY_RT=1` | Soldiers are not in the ray-traced scene (comparison); `QLIKE_BODY_RT_INSET=<m>` sets how far their copy is pulled in under the drawn skin |
| `QLIKE_ANIM_LOG=1` | Per-frame animation cost, level counts, network bytes, and what each remote player's animator is doing, in the log |
| `QLIKE_NET_PORT=<n>` | Another discovery port: test instances do not meet real games |
| `QLIKE_CFG=<file>` | Another settings file (captures without touching the player's own) |

Two instances on one PC, one observing the other:

```bash
QLIKE_NET=1 QLIKE_NET_PORT=48777 QLIKE_CFG=a.cfg QLIKE_TEST_MOVE=1 QLIKE_TEST_FIRE=1 ./build/vibe.exe --frames 5000 &
QLIKE_NET=1 QLIKE_NET_PORT=48777 QLIKE_CFG=b.cfg QLIKE_TEST_HOLD=1 QLIKE_ANIM_DEBUG=1 QLIKE_ANIM_LOG=1 ./build/vibe.exe --frames 1500 --screenshot obs.png --screenshot-after 130
```

## Measuring the gait

```powershell
./build/vibe_animbench.exe assets/character/soldier.qanim slide
```

Holds the real controller on each key combination and prints, per pace and direction: the clip
chosen, its speed, rate and stride, how fast a planted foot moves over the ground, how much of
the capture's contact the foot really stands, and the fastest joint. `SLIDE_OFF=stride,orient,
lock,footik,hand,lean,rate` switches layers off for comparison; `SLIDE_STRIDE`, `SLIDE_LEG`,
`SLIDE_REACH`, `SLIDE_DROP` set the stride tuning; `SLIDE_TRACE="walk/forward"` (with
`SLIDE_TRACE_T=<s>`) prints half a second of legs and right arm frame by frame.

## Tests

`./build.ps1 -Config Release -Test`, or `build/tests/qlike_anim_tests.exe "[animator]"`.

| Tag | Covers |
|---|---|
| `[skeleton]` `[asset]` | Rig validity, mirror map, scaling, rig id, asset round trip, damaged files |
| `[import]` | BVH parsing and FK, retargeting (axes, scale, bone lengths), contacts, validation |
| `[runtime]` | Bind-pose parity, sampling at boundaries, looping, root motion, blending / masks / additive, IK, mirroring, skinning, threads |
| `[motion]` `[trajectory]` `[matcher]` `[inertializer]` | Features, jump detection, serialization, fast search = reference search, query time, prediction = the controller's own integration, hysteresis, 5 - 10 Hz input, pose and rate continuity |
| `[asset]` | The cooked asset at the controller's real paces (crawl, walk, sprint; forward, diagonal, sideways, backward): foot slide, joint rates, a whole gait, no path-only searching. Skipped when the asset is missing |
| `[animator]` `[lod]` | Against the real controller: reversals, strafes, 1x - 3x airtime, long falls, bunny hops, double jump + 180 in the air, hands on the weapon, recoil / reload / interrupt / switch, foot locks, slopes, platforms, variable frame rates, **controller trajectory identical with and without animation**, determinism, levels of detail |
| `[net]` | Wire form and aim precision, insane packets, interpolation / extrapolation / teleports, adaptive delay, event de-duplication and timing, an observer's reconstruction over a link with 0 - 200 ms latency and 0 - 10 % loss, a stall, respawn, late join |

## Troubleshooting

| Symptom | Cause |
|---|---|
| No body; log says `no animation: ... soldier.qanim` | Asset missing, cooked for another ozz version, or its skeleton is not the mesh's: cook again |
| Log: `the animator produced N pose(s) that were not finite` | A bug; the last good pose was shown. `QLIKE_BODY_DEBUG=1` shows the clip and mode at that time |
| Overlay says `FALLBACK: path only` often | The request is outside what was captured: the clip is chosen by direction alone. Should be rare since requests are mapped onto the pace each direction of the capture sustains |
| Feet slide when running | Some is expected with this data, most of it sideways (G1); `vibe_animbench <asset> slide` gives the numbers |
| An arm or a knee snaps over for a frame | Should not happen (the `[asset]` test bounds every joint's rate): run `vibe_animbench <asset> slide` and look at the last column |
| Remote player's feet ignore a floor far from you | Level floors are probed only near the local player; further out feet stay level (G4) |

## Known gaps

- **G1 - speed.** The capture sustains 3.4 m/s forward (CMU runs), 3.4 backward (the same runs played
  backward: a stand-in, not a captured back-pedal) and 1.4 sideways; the game walks at 8 and sprints
  at 12. Playback rate (to 1.5x) and stride (1.7x at 8 m/s, 2.4x at 12) make up the difference. A
  planted foot slides 8 - 11 % of the body's speed forward, diagonally and sideways (sideways above
  5.5 m/s is a run on legs turned to the side), 14 - 18 % backward, 0 - 1 % at the crawl pace
  (`vibe_animbench <asset> slide`, table in `performance.md`). A real sprint (8 m/s and more) and a
  real fast back-pedal would close the rest.
- **G2 — no captured aiming, firing, reload, fall or hard landing.** These are procedural
  layers; the landing uses the two-foot-jump capture's own landing.
- **G3 — stops.** No distance matching; a stop is found by the search (the StartStop takes help).
- **G4 — ground under remote players** far from the local one is the terrain only when in the air
  (time to contact may be late over a floor) and level when grounded.
- **G5 - at most 48 skinned characters are drawn in a frame** (`kSkinCharacters`). Skinning is on
  the GPU (`cube_skin.vert`); the body and the head each have three meshes (109k / 28k / 9k
  triangles together), chosen by height on screen (`body_mesh_lod`: full above 300 px, the middle
  one above 100). Motion vectors of a body come from its model matrix, not its previous pose.
  Shadows: the six soldiers nearest the camera within 40 m are in the ray-traced scene
  (`body_rt.cpp`: a 9k-triangle copy each, skinned by `skin_rt.comp` and rebuilt into a BLAS
  every frame), so they shadow walls, props, each other and themselves, and show in reflections
  and bounce light. Every soldier also goes into the small-object shadow map
  (`draw_skinned_shadows`, `skin_shadow.vert`): the ground and plants within 32 m, which is
  what is left for those beyond the nearest six.
- **G6 - the head** is a mesh of its own (`soldier_head*.qchar`: head, eyes, M1 helmet), drawn on
  other players and the fallen, never on the player himself. No brows, lashes or teeth; the
  face does not move.
- **G7 — not watched in motion by a person.** Verified by tests, stills, and a two-instance run
  on the real netcode in which one instance walks, sprints, jumps, strafes, back-pedals and swims
  by itself (`QLIKE_TEST_MOVE=1`) and the other logs what its animator reconstructs
  (`captures/mp_moving.png`).
- **G8 — reload is cosmetic** (the laser has no magazine); weapon switching has an equip layer
  but no captured draw / holster.
- **Aim and ready.** Right mouse is the aim (`CharacterAnimationInput::aiming`, replicated as
  `kFlagAim`): the rifle in the shoulder, the head on the sights; in the player's own view where
  it always was. Without it the rifle is carried at the ready: lower and canted, the muzzle down
  (`WeaponDef::ready_*`); a shot from the hip brings it level for `hip_fire_hold` seconds. A stance
  only: the shot goes where the crosshair is either way, and there is no zoom or change of speed.
- **Third person hold.** Seen from outside, the rifle is moved so its butt sits in the right shoulder
  (`WeaponDef::butt`, `shoulder_offset`; `AnimTuning::shoulder_weapon`) and the head leans to the
  stock (`head_to_sights`); the tablet is held in front of the chest (`tablet_chest`). The player
  himself sees both where his view has them. A torch is not shown on other players.
- **Hits.** `hit_seq` / `hit_dir` / `hit_power` in the input: the torso gives the way the shot went
  and comes back within a quarter of a second (`AnimTuning::flinch`); the weapon stays on the
  aim. Death is a ragdoll started from the pose and speed of that moment (`ragdoll.cpp`); there
  is no captured death or stagger.
- **Reload.** Still procedural: the weapon tilts, the support hand goes to the magazine, takes it
  down and comes back with one (`magazine_in_hand`, drawn as a plain box by `draw_viewmodel`).
  Cosmetic: the laser has no magazine to empty.
- **Death** is a ragdoll started from the pose and speed of that moment. Making it hold itself for a
  moment by damping its joints was tried: the body reaches the ground at the same time either way.
  A stagger needs joints driven toward a pose (powered ragdoll), which is not built.
- **G9 — one weapon definition** (`default_rifle()`); `WeaponDef` holds grips, muzzle, recoil
  and reload timing per weapon, but the game has one gun model. No ADS in the game, so none here.
