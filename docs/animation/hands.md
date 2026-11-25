# Hands on what is carried

The hands are not played from a capture: they are placed every frame on the weapon (`CharacterAnimator`, the hand block at the end of `character_animator.cpp`), so they stay on it whatever the body does.

## Where a hand goes

- `WeaponDef::grip_r`, `grip_l`: the point of the weapon the palm lies against, in the weapon frame (x right, y up, z towards the viewer). `r_fingers` / `l_fingers` and `r_palm` / `l_palm` give the hand's direction. The wrist is worked back from there and the arm is fitted to it.
- **The right hand** is fixed: each gun model is placed so that its pistol grip lands on `grip_r` (`props.cpp`, `load_gun`'s `place`).
- **The left hand goes under this gun's fore-end.** `load_gun` measures the model: the slice at `grip_l.z`, its lowest point, and the middle of what lies within 3 cm above it (`gun_fore_[0]` the laser, `[1]` the rifle; logged as `viewmodel: <file> fore-end at ...`). `update_player_body` writes it to the animator's `weapon.grip_l` for the gun in the hand. Before this both guns shared one point that lay inside the fore-end, and the hand stood in the gun.
- Not done: the other players' bodies and the NPCs still use the default `grip_l`.

## Fingers

`HandPose` (character_animator.h): for each of the four fingers the bend of its three joints, how far it is fanned from the middle finger's line, and three numbers for the thumb. `WeaponDef::pose_r` (pistol grip: three fingers wrapped, tighter towards the little finger, the index finger laid on the trigger) and `pose_l` (under the fore-end: fingers up round the far side, the thumb along the near side). A hand without a pose (torch, tablet, free arm) keeps the old rule: one bend for all four fingers.

## Where the weapon is held when not aiming

`WeaponDef::ready_shift_fp` (own view only). `VIBE_READY="x,y,z"` tries another place without a build; `VIBE_ADS` does the same for the aimed weapon.

## Looking at it

- `build/trshot/fp.sh <tag> <weapon 0|2>`: the own view with the gun brought up to the middle of the screen, and a crop of the hands (`VIBE_TEST_WEAPON` picks the gun).
- `VIBE_BODY_FRONT` stands the body in front of the camera but does not draw the gun in its hands: no use for this.
- In the own view most of both hands is hidden by the gun; what shows is the back of the left hand and its thumb. The fingers' bends were set by reasoning, and only what shows was checked in pictures.
