# Character animation — measurements

Machine: AMD Ryzen 7 9800X3D (16 threads), Windows 11, MSVC Release (`/O2 /GL /arch:AVX2`).
Date: 2026-11-21. Reproduce: `./build/vibe_animbench.exe assets/character/soldier.qanim 3`.

No budgets are set here: they are the project owner's to decide against a target frame rate
and player count. These are the costs to decide with.

## Animation update (CPU, no skinning)

Each character is a `CharacterAnimator` on the shipped asset, driven by the real controller
with its own scripted input (runs, strafes, turns, jumps, shots). "allocations" counts heap
allocations made inside `update` over the measured frames.

| characters | levels | threads | ms / frame p50 | p95 | p99 | us / character (mean) | allocations in update |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | all 0 (full) | 1 | 0.012 | 0.046 | 0.088 | 18.71 | 0 |
| 8 | all 0 (full) | 1 | 0.135 | 0.355 | 0.522 | 20.81 | 0 |
| 16 | all 0 (full) | 1 | 0.321 | 0.652 | 0.948 | 22.78 | 0 |
| 32 | all 0 (full) | 1 | 0.693 | 1.190 | 1.648 | 22.73 | 0 |
| 64 | all 0 (full) | 1 | 1.350 | 2.282 | 2.573 | 22.50 | 0 |
| 64 | all 1 | 1 | 1.077 | 1.741 | 2.132 | 18.02 | 0 |
| 64 | all 2 | 1 | 0.879 | 1.299 | 1.772 | 14.54 | 0 |
| 64 | all 3 (unseen) | 1 | 0.005 | 0.006 | 0.007 | 0.08 | 0 |
| 64 | all 0 (full) | 16 | 0.614 | 1.173 | 1.742 | 10.70 | - |
| 128 | all 0 (full) | 1 | 2.683 | 4.056 | 4.570 | 21.97 | 0 |
| 128 | mix 8/24/64/rest | 1 | 1.538 | 2.378 | 2.678 | 12.63 | 0 |
| 128 | all 0 (full) | 16 | 0.753 | 1.385 | 1.792 | 6.56 | - |
| 256 | all 0 (full) | 1 | 5.447 | 7.360 | 7.732 | 21.84 | 0 |
| 256 | mix 8/24/64/rest | 1 | 1.477 | 2.180 | 2.548 | 6.03 | 0 |
| 256 | all 0 (full) | 16 | 1.116 | 1.906 | 2.369 | 4.73 | - |
| 512 | all 0 (full) | 1 | 11.012 | 13.849 | 14.990 | 21.87 | 0 |
| 512 | mix 8/24/64/rest | 1 | 1.497 | 2.063 | 2.468 | 3.01 | 0 |
| 512 | all 0 (full) | 16 | 1.709 | 2.358 | 2.640 | 3.51 | - |
| 1000 | all 0 (full) | 1 | 22.579 | 27.053 | 29.325 | 23.01 | 0 |
| 1000 | all 1 | 1 | 19.006 | 22.568 | 23.877 | 19.35 | 0 |
| 1000 | all 2 | 1 | 14.767 | 17.498 | 20.000 | 15.04 | 0 |
| 1000 | all 3 (unseen) | 1 | 0.089 | 0.093 | 0.101 | 0.09 | 0 |
| 1000 | mix 8/24/64/rest | 1 | 1.511 | 1.995 | 2.449 | 1.55 | 0 |
| 1000 | all 0 (full) | 16 | 2.938 | 3.664 | 4.149 | 2.98 | - |

Reading it:

- One fully animated character costs about **48 microseconds** a frame on average; most of that
  is the pose search (0.2 - 0.45 ms, ten times a second). p99 is higher because searches bunch up.
- Level 1 is ~28 us, level 2 ~21 us, level 3 (not in view) 0.09 us.
- A client's realistic mix in a crowd (8 full, 24 at level 1, 64 at level 2, the rest unseen)
  costs **2.3 ms p50 / 4.7 ms p99 on one thread whether 128 or 1,000 players are replicated**.
- Spread over 16 threads, 1,000 fully animated characters take 6.9 ms p50.
- **No allocations** in the update at any count.

The pose search does not look at every frame any more: the database keeps, per block of 32
frames, the least and greatest of every feature, and a block that cannot beat the best match so
far is skipped whole. Same result (the tests compare it with the plain search); in this benchmark
it looks at 5 % of the rows, and a fully animated character costs 19 - 23 us instead of 48 - 56.
In the game the other players' animators are updated together on the job threads
(`pose_update_all`; `QLIKE_ANIM_SERIAL=1` for one after the other).

## In the game

| | |
|---|---|
| Local body, per frame (`player body` in the `[cpuprof]` line) | 0.08 - 0.40 ms: the animation (0.02 idle, up to 0.35 on a frame that searches) plus 0.002 ms to write the bone palette |
| The same before skinning moved to the GPU | 0.75 - 0.85 ms (skinning 43,155 vertices on the CPU ~0.6 ms) |
| Two bodies in view (two-instance run, `QLIKE_ANIM_LOG=1`) | animation 0.03 - 0.3 ms, palettes 0.002 ms |
| GPU, body + weapon (`w_viewmodel` in `gpu-passes`, RTX 4080, 1440p internal) | 0.06 ms |
| GPU, 33 skinned bodies (`QLIKE_BODY_CROWD=32`) | 0.31 - 0.37 ms |
| GPU, 129 skinned bodies (`QLIKE_BODY_CROWD=128`) | 1.25 - 1.47 ms |
| Pose search | 0.20 - 0.45 ms over 86,135 frames, both mirror halves; 0.005 ms while idle |

**Skinning is on the GPU** (`shaders/cube_skin.vert`): the rest-pose mesh and each vertex's bones
and weights are static buffers, a character costs the CPU one 62-bone palette (8.9 kB) a frame.
On the CPU it was ~0.6 ms a body, which put the wall at 20 - 30 drawn players; now about 0.01 ms
of GPU time a body (43k vertices each: the crowd numbers above are vertex work, the block of
copies is mostly off screen, so shading is not in them). The mesh has no levels of detail: at
hundreds in view its 79k triangles per body become the cost, not the skinning.

## Foot slide and joint rates at the controller's paces

`vibe_animbench assets/character/soldier.qanim slide`: the real controller held on one key
combination over flat ground. "Slide" is the speed over the ground of a foot the capture has
on the ground and the animator has not lifted; "on the ground" is the share of the capture's
contact during which the foot really stands (the rest it has left the ground: the leg was not
long enough for the stride).

| pace | direction | speed m/s | clip | clip speed | rate | stride | slide m/s while down | share of speed | foot on the ground while down | locked | fastest joint rad/s |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| crawl key | forward | 3.2 | Crouched_TR1 | 1.36 | 1.50 | 1.57 | 0.00 | 0 % | 100 % | 24 % | 17 LeftToeBase at 3.28 s |
| crawl key | diagonal | 3.2 | Crouched_TR1 | 0.96 | 1.50 | 2.23 | 0.02 | 1 % | 74 % | 43 % | 19 RightLeg at 7.12 s |
| crawl key | sideways | 3.2 | Crouched_TR1 | 1.37 | 1.50 | 1.56 | 0.00 | 0 % | 90 % | 20 % | 14 RightToeBase at 3.12 s |
| crawl key | backward | 3.2 | Crouched_BR | 1.05 | 1.50 | 2.05 | 0.01 | 0 % | 83 % | 33 % | 28 LeftLeg at 2.05 s |
| walk | forward | 8.0 | cmu_35_18 | 3.15 | 1.50 | 1.69 | 0.75 | 9 % | 71 % | 0 % | 24 LeftForeArm at 3.50 s |
| walk | diagonal | 8.0 | cmu_35_18 | 3.15 | 1.50 | 1.69 | 0.76 | 10 % | 71 % | 0 % | 23 LeftForeArm at 4.32 s |
| walk | sideways | 8.0 | cmu_35_17 | 3.15 | 1.50 | 1.69 | 0.91 | 11 % | 70 % | 0 % | 23 LeftForeArm at 3.30 s |
| walk | backward | 8.0 | cmu_35_19_back | 3.18 | 1.50 | 1.68 | 1.44 | 18 % | 49 % | 0 % | 20 LeftLeg at 2.18 s |
| sprint | forward | 12.0 | cmu_35_19 | 3.37 | 1.50 | 2.37 | 1.00 | 8 % | 54 % | 0 % | 26 RightLeg at 2.13 s |
| sprint | diagonal | 12.0 | cmu_35_19 | 3.37 | 1.50 | 2.37 | 1.00 | 8 % | 54 % | 0 % | 26 RightLeg at 2.45 s |
| sprint | sideways | 12.0 | cmu_35_19 | 3.37 | 1.50 | 2.37 | 1.13 | 9 % | 54 % | 0 % | 26 RightLeg at 2.68 s |
| sprint | backward | 12.0 | cmu_35_19_back | 3.35 | 1.50 | 2.39 | 1.66 | 14 % | 53 % | 0 % | 29 LeftLeg at 3.23 s |

Before this pass (same tool, the asset without the Rushed takes, 2026-11-21): walk 32 % forward,
98 % diagonal, 79 % sideways, 87 % backward; sprint 55 / 87 / 86 / 92 %; and at those paces an
upper arm or a knee turning at 100 - 184 rad/s for a frame (a visible snap). What changed:

- a diagonal request is turned onto the nearest captured direction for the search and the legs
  are turned back by the orientation warp (the capture has no diagonal takes);
- game speed is mapped onto the pace the capture sustains in that direction, not onto one number
  (a request for more found one fast moment and replayed a quarter of a second of it for ever);
- the Rushed takes of 100STYLE are in the asset (the fastest of the set);
- stride to 4.5x; a foot that would go beyond the leg leaves the ground instead of being held
  back, the hips come down up to 10 cm;
- the knee points where the hips face while the capture has the leg straight; arms on a weapon
  are solved from the rest pose, not from the captured arm swing; the chest keeps 35 % of its lean;
- the CMU runs (3.4 - 4.2 m/s) are in the asset: at 8 m/s the stride is stretched 1.7x instead of
  2.9x, at 12 m/s 2.4x instead of 4.3x; played backward they replace a back-pedal stretched 3.7x;
- sideways faster than 5.5 m/s is run, not side-stepped: the forward run on legs turned up to 83
  degrees to the side, the torso twisted back onto the aim (the captured sidestep is 1.4 m/s; stretched
  to 8 - 12 it slid 25 - 43 %).

Measured for the ray-traced copies (`QLIKE_BODY_RT_TEST=5`: six soldiers skinned and rebuilt
every frame): 7.7 ms a frame with them and without, 0.13 ms of CPU against 0.09 for the TLAS
build. Nothing to optimise there at six.

## Memory

| | |
|---|---|
| Asset on disk | 24.2 MB |
| Clips in memory (ozz, compressed) | 15.3 MB for 89,508 frames x 62 joints: 179 bytes per frame (raw: 1,736) |
| Motion database | 12.7 MB (32 floats per frame, padded for AVX; the mirrored half is not stored) |
| Per character | ~16 kB of pose buffers plus two sampling caches |
| Before this work | ~46 MB (every frame stored twice) |

ACL was not evaluated: ozz's compression is 9:1 on this data and sampling is not on the profile.

## Network (measured in a two-instance run)

| | |
|---|---|
| A client sends | 44 bytes x 30 /s = 1.3 kB/s (measured 1.31) |
| The host sends each client | (6 + 43 x players) bytes x 60 /s; two players: 5.5 kB/s (measured 5.66) |
| Before (protocol 1) | 47 bytes per state; no weapon, keys or shot counter |

Payload bytes as counted by `net::Session::stats()`; ENet and UDP headers come on top.
