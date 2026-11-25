# Action clips: motions made from text, played by name

Things a character does once - sits down, gets up, gets into a car - are **action clips**: clips
tagged `kTagAction`, never found by the locomotion search, played by name. As of 2026-11-25 they
are made with Tencent HY-Motion 1.0 (text to motion) on this PC.

## 1. Making a motion (outside the repo: `C:\Users\gillis\dev\hy-motion`)

- The model: a clone of github.com/Tencent-Hunyuan/HY-Motion-1.0 with a Python 3.10 environment
  made by `uv` (`C:\Users\gillis\dev\tools\uv\uv.exe`), weights in `ckpts/` (HY-Motion-1.0 3.9 GB,
  CLIP-L 1.6 GB, Qwen3-8B 16 GB).
- **16 GB of video memory is enough** with one local change: `HY_QWEN_BITS=4` loads the text
  encoder in 4 bits (`hymotion/network/text_encoders/text_encoder.py`, marked LOCAL CHANGE).
  Peak 11 - 12.5 GB a motion, a few seconds each, with the game running beside it.
- `vibe_generate.py <prompt file> --out <dir> [--seeds 11,23] [--only name,name]` - one line a
  motion, `text#frames#name` (30 frames a second). Writes `<name>_s<seed>.bvh`: the SMPL-H body
  (22 joints, root `Hips`), metres, +Y up, +Z forward, +X the body's left, 30 Hz.
  Prompt files: `vibe_prompts/npc.txt`, `vibe_prompts/car.txt`.
- **Check the checkpoint loaded.** The runtime takes the checkpoint's full path; given a bare
  file name it warns "Checkpoint ... not found, skipping model loading" and runs with random
  weights - motions that depend only on the seed and the length. (The first batch was that.)
- Judge a motion by its numbers before looking: hips height at start / lowest / end, travel,
  turn (an `awk` over the BVH's root channels does it). "Gets into a car" as first worded walked
  forward and never sat; "steps sideways and sits down onto a low seat" sits.
- Licence: Tencent HY-Motion 1.0 Community License - it does not apply in the EU, the UK and
  South Korea, and that covers the outputs. The user was told and chose to go on (2026-11-25).

## 2. Into the game

- Copy the chosen take to `C:\Users\gillis\dev\mocap\hy_actions\<clip name>.bvh` (the file's
  name is the clip's).
- Cook:
  `build/vibe_animcook.exe C:/Users/gillis/dev/mocap/100style/100STYLE assets/character/soldier_fp.qchar assets/character/soldier.qanim --cmu C:/Users/gillis/dev/mocap/cmu/bvh --actions C:/Users/gillis/dev/mocap/hy_actions`
  (`--actions`: every `.bvh` there, retargeted with `smplh_params()` in `anim/offline/retarget.cpp`.)
- Clips now: `sit_down`, `stand_up`, `sit_idle`, `car_enter`, `car_exit`, `stand_idle`, `stroll`,
  `fuel`, `till`, `walk_stop`; since 2026-11-25 also `bar_sit`, `bar_talk`, `table_eat`, `table_talk`,
  `cashier`, `cashier_take`, `look_crossed`, `look_hips`, `look_phone`, `look_far`, `walk_brisk`,
  `walk_tired`, `walk_pockets`, `walk_phone`, `drive` (unused). Prompt files `vibe_prompts/life.txt`, `walks.txt`.
  Loops: `action_in_place` (a walk played while the caller moves the body) and `action_warp_full`
  (hips on the seat for the whole loop).

## Death (2026-11-25): shot people react, then fall

- Takes: `vibe_prompts/shot.txt` (8 prompts x seeds 11, 23, 37). Every take stands still for one or two seconds before it reacts, so the chosen ones are CUT with `vibe_trim.sh <in> <out> <first> <last>` (in the hy-motion folder) to begin at the reaction. Chosen by the numbers of the hips (height at start / lowest, when they pass 85 % and 70 % of the start, travel):

  | clip | take | frames kept | the physics takes over at |
  |---|---|---|---|
  | `shot_front` | s37 | 6 - 80 | 1.63 s |
  | `shot_gut` | s11 | 43 - 100 | 1.07 s |
  | `shot_back` | s11 | 43 - 90 | 0.73 s |
  | `shot_head` | s11 | 28 - 70 | 0.63 s |
  | `shot_leg` | s11 | 7 - 55 | 0.77 s |
  | `shot_spin` | s11 | 33 - 90 | 1.00 s |
  | `shot_run` | s23 | 12 - 50 | 0.43 s |
  | `shot_walk` | s11 | 30 - 95 | 1.30 s |

- Played by `npc_world.cpp` (`pick_shot_clip`, `npc_dying_pose`): which clip follows from where the bolt struck (above 1.5 m the head, under 0.75 m a leg), from which side, and how fast the person moved. At the hand-over time (the hips have begun to drop) the pose becomes a ragdoll whose parts go on at the speed the clip moved them (`corpse_spawn(..., part_velocity)`); the fall and the landing are the physics'. On a seat, or in the middle of an act that warps the hips: the ragdoll at once.
- Not yet: players killed in multiplayer still buckle for 0.38 s and fall (multiplayer.cpp); seated people have no clip of their own.

## 3. Playing one (`CharacterAnimationInput`, `CharacterAnimator::action`)

- `action_clip` (from `find_clip`) and `action_time`: the caller owns the time. The clip's own
  travel from its first frame is kept.
- **Inertialisation**: into an action, between two, at a loop's wrap, into and out of the seat
  pose, and back to locomotion (`last_mode_`).
- **Motion warping**: `action_warp` + `action_hips` carry the hips over the clip to where this
  seat really is; `action_warp_from_start` for clips that begin there; `action_end_at_origin`
  takes the clip's travel out again so it ends where the model stands.
- **Feet**: a foot that stands (ankle under 24 cm) stays where the clip put it while the hips
  are warped - the leg is solved again (`fit_limb`).

## 4. Who uses it

The NPC engine (`src/npc`): `Act` (SitDown, StandUp, CarEnter, CarExit) with fixed times
(`kActTime`), in the `View` with its clock and the seat aimed at; 27 bytes an agent on the wire.
`vk_engine/npc_world.cpp` maps act -> clip, scales the clip to the act's time and gives the
warp target (a chair's seat, or the car seat from `npc_car_seat`). People at a till or a pump
loop `till` / `fuel`.

Hooks: `VIBE_NPC_ACT=<clip>` (everybody who stands plays it), `VIBE_NPC_WARM=<s>` with
`VIBE_LOG_NPC=1` (logs the moments acts begin, with the place - to stand there for a capture).

## Not done

- Walking still comes from the soldier's motion-matched set (raised upright); `stroll` is cooked
  but not used - it is not a cycle, and looping it needs a matched loop point.
- Looked at in sequences of captures: sitting down on a bench, getting into and out of a car.
  Getting up from a chair and the till / fuel loops play, but were not judged by eye.
- The right-hand passenger uses the same car clips as the driver (not mirrored).
- No hands on the door or the wheel during the car acts; no door moves.

## At the wheel (2026-11-25): the body is fitted to the car, not played from a clip

- The pose is made by rule (`CharacterAnimator::seated`), not from the `drive` clip (still unused): a clip's seat, wheel and pedals are never this car's.
- Each car's file (`assets/<car>/car.json`, `assets/jeep/jeep.json`) says where things are, measured from its model: `hip_driver` (the hip joint: the cushion's top + 10 cm, 13 - 17 cm in front of the backrest), `pedal_gas` (the middle of the accelerator's pad), `floor_y` (the footwell), `foot_rest` (x, z of the left heel), `seat_back_deg`. jeep.cpp `jeep_seat` / `jeep_seat_feet` hand them over as `CharacterAnimationInput::SeatExtras` (`take_seat`), for the player, other players and NPC drivers and passengers alike.
- The rule: the back against the backrest, unless the arms then do not reach the wheel with the elbows bent - then the trunk comes forward as far as it takes; the palms on the outside of the rim at a quarter to three, knuckles along it, fingers closed round it; the right heel on the floor behind the accelerator with the ball of the foot on its pad, the left foot on the floor beside the pedals; a leg too short for its pedal stops short of it with the knee bent (the sports car). The head turns a little with the wheel and is never quite still.
- Measured with scratch perl scripts over the `.mesh` text (cushion tops: jeep 0.63, pickup 0.78, saloon 0.38, sports car 0.30 m; pedals and floors likewise). A new car needs the same five entries, or it gets the old general pose.

## Round three (2026-11-25) - stopped before it was finished

- Made with HY-Motion (`vibe_prompts/life2.txt`, takes in `C:\Users\gillis\dev\hy-motion\vibe_out2`): `bar_idle`, `table_drink`, `table_idle` (two seeds each). NOT yet chosen, copied to `hy_actions` or cooked. Not made: `walk_relaxed`, `walk_swagger`, `look_watch`, `look_around`, `cashier_idle` (the run died when a game started beside it, with 13.2 GB peak; the second run was stopped).
- `npc_world.cpp` already names all eight: whoever stays takes turns (bar: `bar_sit` / `bar_talk` / `bar_idle`; eating: `table_eat` / `table_drink`; talking: `table_talk` / `table_idle` / `table_drink`; cashier: `cashier` / `cashier_take` / `cashier_idle`; standing: `look_around`, `look_watch` beside the four there were; walking: `look % 8` 5 `walk_relaxed`, 6 `walk_swagger`). A name the cooked set does not have falls back to the clip used before, so nothing breaks until they are cooked. When the two walks are cooked, set their paces in `npc::kPace[5]`, `[6]`.
- The soldiers' run is still the motion-matched run (real capture, rifle in the hands); no clip was made for it.
