# Bimanual YAM arms + ORCA hands — MuJoCo teleop setup

Backup/record of the **bimanual YAM-arm setup** files that are **not** in the
`alerest285/orca-teleop` repo. Built/refined 2026-06-01. The two I2RT YAM arms
(each carrying an ORCA hand) live in the ORCA MuJoCo teleop sim.

Everything here is a **copy** — the live originals are still in
`~/Desktop/orca_sim/...` (so the sim keeps working). To restore, copy the
folders/files below back to the matching path under
`~/Desktop/orca_sim/src/orca_sim/`.

---

## What's in this folder (→ original path under `orca_sim/src/orca_sim/`)

| here | → orca_sim path | what it is |
|---|---|---|
| `scenes/v2/bimanual_yam.xml` | `scenes/v2/` | top scene: composes options+scene+2 hands+2 arms; **arm base poses** |
| `models/v2/mjcf/yam_left.mjcf` | `models/v2/mjcf/` | left arm: mesh assets + 6 position actuators (kp600/kv30, incl. wrist-roll j6) |
| `models/v2/mjcf/yam_left_body.xml` | `models/v2/mjcf/` | left arm body tree + **left hand mount** |
| `models/v2/mjcf/yam_right.mjcf` | `models/v2/mjcf/` | right arm: assets + actuators (de-mirrored → real meshes) |
| `models/v2/mjcf/yam_right_body.xml` | `models/v2/mjcf/` | right arm body tree (de-mirrored) + **right hand mount** |
| `models/v2/assets/stand/frame_part{0,1}.stl` | `models/v2/assets/stand/` | the AgileX Ranger 60×60 stand, split in 2 (MuJoCo 200k-tri limit) |
| `models/v2/assets/yam/*.stl` | `models/v2/assets/yam/` | the YAM arm link meshes (see note on `_mirror` below) |
| `__init__.py`, `envs.py`, `registry.py` | `src/orca_sim/` | **modified** package files: `OrcaYamBimanual` env + `-v2` registration + export |
| `orca-teleop/*` | `~/Desktop/orca-teleop/` | teleop driver + launcher + support (snapshot — see note) |
| `exported_hand_meshes/orca_hand_{left,right}.stl` | (not in orca_sim) | full assembled ORCA hand, single STL, neutral pose, in the hand *tower* frame, **mm** — for placing the hand in CAD |
| `left_yam_arm_descritption/` | (separate) | the source YAM arm ROS/URDF description (you placed this) |

### `orca-teleop/` (snapshot of the teleop side)
- `sim_yam_bimanual.py` — **the driver**. Per-side `HOME_POSE` facing forward −X
  (`left arm_j1=1.5708`, `right arm_j1=4.7124`; j2–j5=0 → palms-down/thumbs-in),
  `ArmController` stub, `home_targets`/`home_qpos`, `build_action`, selftest.
- `run_yam.sh` — launcher: `./run_yam.sh --demo | --selftest | --swap-hands`.
- `sim_common.py` — model-derived `ActuatorMap` + realtime stepper (`parse_sim_name`
  extended to pass through arm joints like `arm_j1`).
- `sim_bimanual.py` — hand config loading + retargeting (reused by the driver).
- `sim_teleop.py`, `solve_pinches.py`, `run_sim.sh` — single-hand teleop + IK solve
  (`solve_pinches` damped-least-squares is reusable for phase-2 arm IK).
> These live in your **`alerest285/orca-teleop`** repo — this is a **snapshot**.
> Commit them in the repo for the canonical version. The driver also needs
> `webcam_teleop.py` (unchanged, in the repo) at runtime.

> `envs.py`/`registry.py`/`__init__.py` are **modified upstream `orcahand/orca_sim`
> files** — most of the content is upstream; the bimanual additions are the
> `OrcaYamBimanual` class (`envs.py`), its `OrcaYamBimanual-v2` registration
> (`registry.py`), and its export (`__init__.py`).

---

## Key parameters (the values that took work to find)

**Stand / frame** — `Assembly Frame 60x60 for Agilex Robotics Ranger.stl`, ~213k
tris, mm. Split into `frame_part{0,1}.stl` (~107k each, full fidelity) because
MuJoCo caps a mesh at 200k tris. Placed in the scene at **z = +0.2475 m** (feet
on floor), visual-only.

**Arm base poses** (in `scenes/v2/bimanual_yam.xml`) — recovered by ICP-registering
the YAM `base_link` mesh onto the CAD assembly (`assembly example both arms.stl`),
RMS ~1.3–1.6 mm. quat = `w x y z`:
- `left_arm_base`  pos `-0.0248 -0.1700 0.6908`  quat `0.49791 0.50194 -0.50011 -0.50004`  (joint-1 axis → world −Y)
- `right_arm_base` pos `0.0101 0.0801 0.6875`    quat `0.49989 0.49988 0.50055 0.49968`    (joint-1 axis → world +Y; = left rotated 180° about Z)

Each arm points world **+Z (up) at q=0**; the per-side home pose (in the driver,
see below) swings them to face forward.

**De-mirror** — both physical arms are identical YAMs, so `yam_right_body.xml` is
now real **left geometry** with `right_*` names (NOT an x-reflection), and
`yam_right.mjcf` points at the non-mirror STLs. ⇒ the `*_mirror.stl` files in
`assets/yam/` are now **UNUSED** (kept only for history; safe to delete).

**Servo gains** — arm position actuators bumped to **kp=600 kv=30** (from 150/15)
so the (un-gravity-compensated) hand doesn't sag a horizontal arm. Holds <1.3°.

**Hand mounts** (`*_hand_mount` in the `yam_*_body.xml`) — the hand's square
back-face center is seated on link5's circular flange center, faces flush
(verified **0.00 mm** center-to-center). From CAD `full base + hand attached.stl`:
link5 circle center = `(-40.5, 49.8, 0) mm` in the link5 frame; tower square
center = `(-9.9, -58, 0) mm` in the tower frame; `pos = circle − R_euler·square`.
- `left_hand_mount`  pos `-0.0504 0.1078 0`  euler `0 3.14159 0` (180° finger-axis roll → palm-down, thumb-in)
- `right_hand_mount` pos `-0.0306 0.1078 0`  euler `0 0 0`
- The L/R 180° euler difference comes from the two arms being 180°-about-Z copies.

---

## Teleop side

The teleop driver + support are in `orca-teleop/` above (snapshot). Their
canonical home is your `alerest285/orca-teleop` repo — **commit them there**.

## NOT here — lives in the `orcahand/orca_sim` repo (upstream, tracked)

Needed for the scene to load (the YAM bodies `<include>` the hand bodies):
`models/v2/mjcf/orcahand_{left,right}.mjcf` + `..._body.xml`,
`models/v2/assets/{left,right}/`, `scenes/v2/{scene,options}.xml`.

---

## Run

```bash
cd ~/Desktop/orca-teleop && ./run_yam.sh --demo      # viewer; arms hold home, hands sweep
./run_yam.sh --selftest                               # headless check
# add --swap-hands when wiring teleop (L/R hands are on crossed sides)
```
Note: `run_yam.sh` hardcodes `$HOME/Desktop/orca_sim/src` — if you move the
project, fix that path.

## Open / next

- **Phase-2 arm control**: swap real teleop (GELLO/VR → IK) into the
  `ArmController` stub in `sim_yam_bimanual.py`. IK can reuse the damped
  least-squares solve in `solve_pinches.py`.
- Hand contact is **flush-at-flange**; re-derive only if a specific non-flush
  adapter offset surfaces.
