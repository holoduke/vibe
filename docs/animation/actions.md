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
- **Finished later that day.** The five that were missing were made (`C:\Users\gillis\dev\hy-motion\vibe_out3`, one run, 13.2 GB peak with no game beside it) and all eight chosen by their numbers (scratch `takes.pl`: hips height, travel, turn, how far the last frame is from the first, how much moves) and cooked - 41 action clips in the set:

  | clip | take | why |
  |---|---|---|
  | `bar_idle` | s11 | the last frame nearer the first (1.8 against 4.2 degrees a channel), more movement |
  | `table_drink` | s23 | nearer a loop (1.8 against 3.9) |
  | `table_idle` | s11 | nearest a loop of all (0.9), moves more |
  | `cashier_idle` | s23 | the hips turn 16 degrees, not 31 |
  | `look_around` | s23 | wanders 0.28 m, not 0.65 |
  | `look_watch` | s23 | 9.6 against 16.3 at the wrap |
  | `walk_relaxed` | s23 | straighter (turns 11 degrees over 10 s); 0.88 m/s cooked - `npc::kPace[5]` = 0.82 |
  | `walk_swagger` | s11 | ends straighter; 1.25 m/s cooked - `npc::kPace[6]` stays 1.0 |

  None of the eight was looked at one by one in the game.

## Life layers (2026-11-25): nobody is the same picture as the next

Two people who play the same clip were the same picture a few frames apart. Now everybody is
somebody, and whatever plays - locomotion, an action clip, the seat pose - has layers on top.
`src/anim/life_layers.cpp`, `src/anim/person_style.h`; the order is fixed:

| | layer | kind | what |
|---|---|---|---|
| - | base | | motion matching, an action clip (inertialised), or the seat pose |
| a | posture | additive, constant | `PersonStyle`: slouch, a lean, shoulders up / dropped / one higher, head tilt / turn / nod, arms held out, on a seat the feet out or tucked, wider or closer |
| b | clip layer | override, masked | `layer_clip` over `LayerMask` (UpperBody, Head, ArmLeft, ArmRight, Arms; a weight a joint, feathered down the spine). The animator owns its clock and its weight (`AnimTuning::layer_halflife`); it lets go for a moment where the clip wraps |
| c | breathing, weight | additive, own clocks | chest and shoulders at the person's own rate; the weight to one leg (standing: the hips move, the feet are solved back) or one side of the seat |
| d | gesture | override, one arm | `Gesture`: ScratchHead, RubNeck, CheckWrist, Shift, Wave. The arm is solved from rest to a place on the body and blended in and out. A fidget is one the animator starts by itself (`fidgets`, `fidget_arms` = the hands that are free); any can be asked for (`gesture` + `gesture_seq`) |
| e | look | additive, limited | head, neck, chest and eyes to `look_at` some of the time (`look_weight`: 1 now and then, 2.5 mostly, 4+ fixed; a new `look_id` catches the eye soon), else a side, a sweep, nowhere. Measured from the head the layers below made and limited from the chest (`look_yaw_max` 60 degrees, `look_pitch_up` / `_down`), so a clip that turns the head is not turned twice |
| - | fitting | | feet (locks, or the action's standing feet), the seat's hips, hands on a weapon or a wheel |

- **Who somebody is**: `person_style(id)` - everything from one number, so the same on every machine and every time that person is shown. Besides the posture it gives the caller `phase` (where in a loop this person starts), `rate` (playback 0.93 - 1.07) and `mirror` (`action_mirror`: the clip left for right, its travel too). `individuality` 0 (the players' soldiers) = nobody in particular: no offsets, the old behaviour.
- **Opting in**: locomotion always had the look, the breath and the weight shift; an action clip or the seat pose gets them with `life_on_clip` (somebody who stays - not sitting down, not dying). A driver at a wheel keeps the pose he had.
- **Every weight is eased** (the clip layer by a spring, a gesture at 4 a second at most, the look by its own spring, clamped to its limits), so something that ends early - shot in the middle of a wave - lets go without a jump.
- **What the station's people get** (`npc_world.cpp` "WHAT IT PLAYS", "COMBINED", "THE LIFE LAYERS"): their own phase, pace and - except the cashier, the pump and the till - side; a second clip part of the time (talking arms on `table_idle`, `bar_talk`'s head on `bar_idle` / `bar_sit`, `look_around`'s head on somebody who walks or stands, a phone or `sit_idle` upper body on a plain seat); fidgets with the hand that holds no fork or glass; and something to look at - the player within 3.5 - 4.5 m, a car that drives past (those outside, and whoever eats by the window), whoever is talked to (the next stool, across the table, the customer at the counter), the plate or the glass, the player out to 9 m. The more curious half of those on foot wave once when the player comes within 6 m in front of them.
- Hooks: `VIBE_NPC_NO_LIFE=1` (everybody as the bare clip, to compare), `VIBE_NPC_GESTURE=<1..5>` (everybody does that gesture every five seconds), and `VIBE_LOG_NPC` says per body the clip, "mirrored", the second clip, the gesture and where the head is.
- Tests (`tests/anim/test_animator.cpp`, tag `[life]`, the real asset): two ids on one clip at the same moment differ and one id twice does not; mirrored is the reflection; the look reaches a target at 40 degrees, stops at the limit for one at 95, ignores one behind, keeps the limit on a clip that turns the head itself; a wave goes up and comes back with the feet where the clip has them; an upper-body layer on the seat pose changes the hands and not the legs.
- Cost: not measured. Per body a handful of joint turns; one more clip sample while a clip layer is in; one two-bone solve while a gesture runs. Posture at every level of detail, the clip layer to level 2, the rest to level 1.
- Not done: no gesture uses both arms; a gesture's hand does not know the head's real size (the wrist is put 16 cm beside the head joint); the eyes need the face joints and level 0; players' soldiers get none of the posture (`individuality` 0); nothing of this crosses the network - each machine draws the same person's own style, but fidgets and glances run on each machine's own clock.

## Desk work, pointing, talking (2026-11-26): made by rule, not from clips

The pavilion's people needed five new motions: typing seated, reading seated, leaning back, pointing while standing, talking while standing. They are NOT text-to-motion clips. Two reasons: the generator needs 11 - 13 GB of video memory and the game was running beside it (8.8 of 16 GB in use; the last run with a game beside it died), and a desk clip's keyboard is never where this desk's is - the hands would have to be solved to the real keyboard anyway. So they are built on what was there:

- **At a desk** (`npc::View::style` 4 of a seat; `npc_world.cpp` "AT A DESK"): the animator's seat pose (`CharacterAnimator::seated`, the one the cars' passengers use) with the wrists given every frame - `seat_desk` (the hands lie flat, fingers forward), `seat_typing` (0 .. 1: each finger goes down and up at its own moment, in bursts with pauses), `seat_back` (how far the back leans). The keyboard's place comes with the seat (`View::aim`, from `assets/office/office.nav`); the mouse lies 30 cm to its left, the screen a hand behind it and 30 cm up. Four things in turn, 13 s or so each, each person at a moment of his own: **typing** (both wrists before the keys, wandering along them), **reading** (the left hand on the mouse, moving it now and then, the right at rest on the desk's edge, the back a little further back), **leaning back** (21 degrees, the forearms on the armrests, the head free to look about), **turned to the neighbour** (the head fixed on whoever sits at the next desk, the right hand talking). The wrists go from one place to the next eased over a fifth of a second; the fingers stop while a hand is on its way. The head is the look layer's: the screen most of the time, the player who comes close. `VIBE_NPC_DESK=<0..3>`: everybody does that one.
- **Pointing** (`Gesture::Point`, `life_layers.cpp`; style 5 of somebody standing): the arm on the side the thing is goes out toward `gesture_at` (the colleague's screen, a screen of the mission wall), not quite straight, the first finger out and the others closed, stirs a little and comes back; every six seconds or so. The head looks where the arm points.
- **Talking** (`Gesture::Talk`; style 6, and at a desk turned to the neighbour): a forearm up in front of the body, the open hand going out and back and up and down on two slow beats. Each person at moments of his own, so in a group now one talks and now another; the head goes to whoever stands nearest.
- **Looking at something** (style 7: the rocket from the apron, the pad from the pavilion's glass, the hall from the mezzanine's rail): the head fixed on `View::aim` by the look layer, the trunk leaning back with a look that goes up.
- **Fingers on every action clip** (`CharacterAnimator::action`): the clips come from a body of 22 joints - no fingers - so every hand was the rest pose's, flat and spread. A hand that holds nothing now hangs with its fingers a little curled.
- Not done: no clip for getting on to a task chair (the same `sit_down` as for a bench, begun beside the chair: the body goes sideways through the armrest); the chairs do not turn or roll; nobody carries anything; the meeting table's people play the diner's `table_talk`. Seen in stills only - how the typing, the pointing and the talking MOVE has to be judged in play.
