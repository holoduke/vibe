# 1,000 players in one world — capacity audit (Phase S0)

Status: **audit only.** Nothing in this document makes the game support 1,000 players. It records
what the netcode is, what was measured, what follows by arithmetic from the code, what cannot be
known yet, and the order in which it would have to be built. 1,000-player support is not complete
and is not claimed.

## 1. What exists (`src/engine/net/`, `vk_engine/multiplayer.cpp`)

| Item | As built |
|---|---|
| Server simulation loop | **None.** There is no server. Every instance simulates its own player; one instance (elected by lowest id) also relays |
| Transport | ENet 1.3.18 on UDP; channel 0 unreliable (state), channel 1 reliable (events, hello, world state) |
| Snapshot encoder | `PackedState`, 35 bytes: position and velocity as floats, yaw / pitch 16 bit, crouch 8 bit, flags, health, weapon, movement keys, shot counter. No delta, no change mask |
| Rates | Client -> host 30 Hz; host -> each client 60 Hz with every player in one packet |
| Prediction / reconciliation | Not needed today: a player's own machine is authoritative for its movement |
| Interpolation | Per sender, at least 100 ms behind, further on a slow link (`SnapBuffer::delay_ms`); 250 ms of extrapolation at most |
| Interest management | None: everybody receives everybody |
| Collision broadphase | Player vs. level: boxes within 24 m of the local player (`rebuild_tick_aabbs`). Rigid bodies: Jolt, local to each machine, not replicated |
| Hit validation | On the shooter's machine, against the victim's box at the interpolated position (`mp_bolt_hits`); the victim subtracts the damage it is told |
| Ownership / session | Beacon discovery on the LAN, host election, re-election when the host leaves |
| Limits in code | `kMaxPlayers = 8`; avatars in a fixed array of 8; one ENet host with 8 peers |
| Threads | Networking on the main thread, once per frame |

## 2. Measured

Two instances on one machine (9800X3D), `QLIKE_ANIM_LOG=1`, 2026-11-21:

| | |
|---|---|
| Client upstream | 1.31 kB/s payload |
| Host downstream per client, 2 players | 5.66 kB/s payload |
| Client frame cost of one remote body in view | animation 0.03 - 0.3 ms on the CPU; skinning on the GPU, ~0.01 ms (129 bodies: 1.3 - 1.5 ms on an RTX 4080) |

Client-side animation cost by player count (`vibe_animbench`, full table in
`docs/animation/performance.md`):

| Players replicated to one client | Animation CPU, one thread, p50 / p99 |
|---|---|
| 128, mix of levels (8 / 24 / 64 / rest unseen) | 2.2 / 4.3 ms |
| 1,000, same mix | 2.3 / 4.7 ms |
| 1,000, all at full detail | 49 / 66 ms (6.9 / 9.7 ms on 16 threads) |

**Not measured, because there is nothing to measure them on:** server CPU per tick, snapshot
encode time, reliable queue depth, interest-set cost, physics cost at N players, packets per
second at 16 - 1,000 players, dispersed / clustered / hotspot distributions. A headless load
harness needs a server to load; building one is step 1 below.

## 3. What the code implies (arithmetic, not measurement)

With the relay as it is, every client receives every player 60 times a second:

| Players | Host -> one client | Host total upstream |
|---|---|---|
| 8 | (6 + 43 x 8) x 60 = 21 kB/s | 0.15 MB/s |
| 64 | 165 kB/s | 10.4 MB/s |
| 256 | 661 kB/s | 169 MB/s |
| 1,000 | 2.58 MB/s | 2.58 GB/s |

Total traffic grows with the square of the player count. A 1,000-entry snapshot is 43 kB per
packet: it does not fit a datagram and would be fragmented 60 times a second. The relay design
ends somewhere between 32 and 64 players on bandwidth alone, before CPU is considered.

## 4. What stands between this and 1,000

1. **Authority (blocker B1 in `docs/animation/architecture.md`).** A world of 1,000 needs a
   dedicated server that simulates movement and decides hits. Today movement is client
   authoritative and hits are shooter decided: with strangers that is an open door (speed,
   teleport, hit claims). The server has to run the player controller against the level, the
   terrain function, water and props for every player. This is a rewrite of the game's
   simulation model and needs an owner decision first: dedicated or listen server, hosting,
   whether cheating is in scope.
2. **Interest management.** A spatial grid over the home tile and the far charts, enter / leave
   sets with hysteresis, priority by distance, visibility, combat relevance. Per-client byte
   budget per tick; a hotspot of hundreds has to degrade by priority, not by dropping events.
3. **Encoding.** Delta against the last acknowledged snapshot, change masks, quantised velocity
   near the ground (floats stay for flight), lower rates for far entities.
4. **Client.** Skinning is on the GPU (done: it was ~0.6 ms of CPU a body, a wall at 20 - 30
   drawn players); the body has no mesh levels of detail (79k triangles each) and the palette
   holds 48 characters a frame; animation levels of detail exist and are measured; impostors or aggregate
   representations for hundreds in view are a design question.
5. **Structures.** `kMaxPlayers`, the avatar array, one packet for all players, discovery by
   broadcast: all sized for a LAN.

What already fits: animation is reconstructed from 35 bytes of state plus events, no bones are
sent, and the animator's levels of detail keep a client at 2 - 5 ms of animation for 1,000
replicated players. Hit tests do not read poses (tested: they use the replicated box, drawn or
not). The state has sanity checks on receive (`state_is_sane`).

## 5. Proposed state / event schema (for the server model; not implemented)

```
snapshot (unreliable, per client, per tick, budgeted):
  server_tick u32, baseline_tick u32, count u16
  per entity (priority order, as many as the budget allows):
    id u16, mask u8
    [pos]  3 x 24 bit, relative to the client's grid cell     [vel] 3 x i16 (cm/s; escape to floats above 327 m/s)
    [aim]  yaw u16, pitch i16                                  [stance] crouch u8, flags u8
    [arms] weapon u8, shot_seq u16                             [keys] move u8
events (reliable-ordered per sender, stamped):
  kind u8, from u16, to u16, tick u32, seq u16, payload (fire: direction 2 x 16 bit; hit: part, damage)
client -> server (unreliable, redundant last 3):
  input tick u32, keys u8, yaw u16, pitch i16, fire/reload bits
```

The fields are the ones the animator already consumes; `net_anim.h` would not change.

## 6. Order of work

| Step | Deliverable | Gate |
|---|---|---|
| S0 | This audit | done |
| - | Owner decision on B1 | open |
| 1 | Headless dedicated server running the controller for N scripted players; load harness | server tick p50 / p95 / p99 at 2 - 1,000 |
| 2 | Client prediction + reconciliation for the local player | controls feel unchanged; no divergence |
| 3 | Server-side hit validation with proxy volumes and lag compensation | hits independent of any pose |
| 4 | Interest grid, priorities, per-client budgets, delta encoding | bytes per client bounded in a 1,000-player hotspot |
| 5 | GPU skinning (done); mesh levels of detail; crowd representation | frame time with 32 / 64 / 128 drawn |
| 6 | 16 -> 64 -> 128 -> 256 -> 512 -> 1,000, dispersed, clustered, hotspot; 0 - 200 ms, 0 - 10 % loss | owner-set budgets, PASS / FAIL with evidence |

Target hardware, tick rate, expected clustering and acceptable visual compromises in a hotspot
are not known and have to come from the owner before budgets can be written.
