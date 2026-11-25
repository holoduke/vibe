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
