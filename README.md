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
| `models/v2/assets/stand/frame_part{0..5}.stl` | `models/v2/assets/stand/` | the **elongated** (1230 mm tall) AgileX Ranger 60×60 stand, split in 6 (MuJoCo 200k-tri limit) |
| `models/v2/assets/yam/*.stl` | `models/v2/assets/yam/` | the YAM arm link meshes (see note on `_mirror` below) |
| `__init__.py`, `envs.py`, `registry.py` | `src/orca_sim/` | **modified** package files: `OrcaYamBimanual` env + `-v2` registration + export |
| `orca-teleop/*` | `~/Desktop/orca-teleop/` | teleop driver + launcher + support (snapshot — see note) |
| `exported_hand_meshes/orca_hand_{left,right}.stl` | (not in orca_sim) | full assembled ORCA hand, single STL, neutral pose, in the hand *tower* frame, **mm** — for placing the hand in CAD |
| `left_yam_arm_descritption/` | (separate) | the source YAM arm ROS/URDF description (you placed this) |

### `orca-teleop/` (snapshot of the teleop side)
- `sim_yam_bimanual.py` — **the driver**. Per-side `HOME_POSE` facing forward −X
  (`left arm_j1=1.5708`, `right arm_j1=4.7124`; j2–j6=0 → palms-down/thumbs-in),
  `ArmController` stub, `home_targets`/`home_qpos`, `build_action`, selftest.
  Now also carries the **phase-2 arm-teleop run loops**: `--vr`/`--vr-selftest`
  (WebXR wrist) and `--orbit`/`--orbit-selftest` (ORBIT Quest app → IK).
- `run_yam.sh` — launcher: `./run_yam.sh --demo | --selftest | --swap-hands`.
- `sim_common.py` — model-derived `ActuatorMap` + realtime stepper (`parse_sim_name`
  extended to pass through arm joints like `arm_j1`).
- `sim_bimanual.py` — hand config loading + retargeting (reused by the driver).
- `sim_teleop.py`, `solve_pinches.py`, `run_sim.sh` — single-hand teleop + IK solve
  (`solve_pinches` damped-least-squares is reusable for phase-2 arm IK).

**Phase-2 VR arm teleop (built + headless-verified 2026-06-04; not yet on hardware):**
- `vr_ik.py` — `ArmIK`: 6-DoF damped-least-squares pose solver (EE = `{side}_hand_mount`,
  `mj_jacBody` + `mju_quat2Vel`, warm-started). Frame-agnostic; selftest PASS.
- `vr_zmq.py` — `OrbitWristReader` (ZeroMQ PULL bind per wrist port + drain-to-newest),
  `parse_wrist` CSV, Unity LH→RH frame fix (`M_rh = S·M·S`), `PoseRetargeter`
  (delta-anchored target `M_robot0·inv(M_vr0)·M_vr_now`), `PoseSmoother` (EMA+nlerp).
- `vr_teleop.py` — WebXR path: cert gen, `VRReceiver` (one port serves the https page
  + wss), quat helpers, `WristMapper` (rotvec → wrist joints), `VRArmController`.
- `vr/vr_client.html` — Quest-browser WebXR client (hand-tracking, sends wrist pose over wss).
- `run_yam_orbit.sh` — launcher for the ORBIT (Quest app + ZMQ) path.
- `run_yam_vr.sh` — launcher for the WebXR path.
> The ORBIT path is preferred (native XR Hands + USB adb-reverse); WebXR is the
> fallback that needs no Quest dev-mode/ownership. See the run section below.

> These live in your **`alerest285/orca-teleop`** repo — this is a **snapshot**.
> Commit them in the repo for the canonical version. The driver also needs
> `webcam_teleop.py` (unchanged, in the repo) at runtime.

> `envs.py`/`registry.py`/`__init__.py` are **modified upstream `orcahand/orca_sim`
> files** — most of the content is upstream; the bimanual additions are the
> `OrcaYamBimanual` class (`envs.py`), its `OrcaYamBimanual-v2` registration
> (`registry.py`), and its export (`__init__.py`).

---

## Key parameters (the values that took work to find)

**Stand / frame** — the **elongated** `Assembly Frame 60x60 for Agilex Robotics
Ranger` (2026-06-05): same 270×400 mm footprint, but **1230 mm tall (+500 mm vs
the old 730 mm)** so the arms ride high enough that the hands clear the floor.
~1.05M tris, mm. Split into `frame_part{0..5}.stl` (~176k each, full fidelity)
because MuJoCo caps a mesh at 200k tris. Placed in the scene at **pos
`0.0017 -0.0210 0.2139`** (feet on floor; the tiny XY offset re-aligns the new
export's mount faces onto where the old frame's were), visual-only.

**Arm base poses** (in `scenes/v2/bimanual_yam.xml`) — recovered by ICP-registering
the YAM `base_link` mesh onto the CAD assembly (`assembly example both arms.stl`),
RMS ~1.3–1.6 mm. quat = `w x y z`. Both z were raised **+0.5 m** (0.69 → 1.19) to
track the elongated frame's mount; XY/quat are unchanged:
- `left_arm_base`  pos `-0.0248 -0.1700 1.1908`  quat `0.49791 0.50194 -0.50011 -0.50004`  (joint-1 axis → world −Y)
- `right_arm_base` pos `0.0101 0.0801 1.1875`    quat `0.49989 0.49988 0.50055 0.49968`    (joint-1 axis → world +Y; = left rotated 180° about Z)

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

# Phase-2 VR arm teleop (wrist pose → IK):
./run_yam_orbit.sh                                    # ORBIT Quest app over ZeroMQ (preferred)
./run_yam_vr.sh                                       # WebXR fallback (Quest browser → wss)
.venv/bin/python sim_yam_bimanual.py --orbit-selftest # headless ORBIT pipeline check
.venv/bin/python sim_yam_bimanual.py --vr-selftest    # headless WebXR check
```
Note: `run_yam.sh` hardcodes `$HOME/Desktop/orca_sim/src` — if you move the
project, fix that path.

## Open / next

- **Phase-2 arm control — BUILT (headless-verified, not yet on hardware).**
  Wrist pose → 6-DoF DLS IK is implemented in `vr_ik.py` + `vr_zmq.py`, driven by
  `sim_yam_bimanual.py --orbit` (ORBIT Quest app over ZeroMQ, preferred) or `--vr`
  (WebXR fallback). Both `--orbit-selftest`/`--vr-selftest` PASS headless.
  Remaining: validate on the real Quest (ORBIT needs the headset owner to clear the
  USB-debugging prompt; chirality/scale/jitter knobs are `--orbit-{flip,scale,smooth}`).
- Next: Quest finger keypoints → 17-DoF ORCA retargeting, and a camera-return stream.
- Hand contact is **flush-at-flange**; re-derive only if a specific non-flush
  adapter offset surfaces.
