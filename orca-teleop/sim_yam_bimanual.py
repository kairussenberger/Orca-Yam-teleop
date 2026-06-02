#!/usr/bin/env python3
"""Bimanual webcam teleop -> two ORCA hands mounted on two I2RT YAM arms.

Phase 1 (hands-only): the 34 ORCA-hand actuators are driven from the webcam
exactly as in sim_bimanual.py; the 10 YAM arm actuators are held at a home pose
by an ``ArmController`` stub. Swap a real arm input (GELLO leader arms, VR 6-DoF
-> IK, full-body pose) into ``ArmController`` later without touching the hand
path -- ``build_action`` routes each actuator by (side, joint) from the model.

macOS: launch the viewer with mjpython:
    uv run mjpython sim_yam_bimanual.py --demo        # hands sweep, arms hold
    uv run mjpython sim_yam_bimanual.py --no-mirror   # webcam -> both sim hands
    uv run python   sim_yam_bimanual.py --selftest    # headless mapping check

If your hands drive the wrong sides, add --swap-hands.
"""
from __future__ import annotations

import argparse
import math
import sys
import time

import numpy as np

import sim_common as sc          # model-derived actuator map + real-time stepper
import sim_bimanual as sb        # reuse hand config loading + retargeting
import webcam_teleop as wt


# Rest pose (radians), PER SIDE. From the CAD-registered base frames each arm
# points straight UP at q=0; j1=+/-90deg swings it to face FORWARD (world -X), so
# the two arms reach parallel side-by-side instead of apart along +/-Y. The signs
# differ because the bases have opposite joint-1 axes (left -> world -Y, right ->
# world +Y). Flip both j1 signs to face +X instead.
# Forward = -X. Chosen because, with the current hand-to-arm assignment + palms
# down, the thumbs only point INWARD when the arms reach -X (facing +X forces
# thumbs out — the two arms are 180deg-about-Z copies, not mirrors, so hand
# chirality is "crossed" vs naming). NB j1 range is [0,2pi] so a -90deg base is
# +270deg=4.7124. (Teleop L/R may then need run_yam's --swap-hands.)
HOME_POSE = {
    "left":  {"arm_j1": 1.5708, "arm_j2": 0.0, "arm_j3": 0.0, "arm_j4": 0.0, "arm_j5": 0.0},
    "right": {"arm_j1": 4.7124, "arm_j2": 0.0, "arm_j3": 0.0, "arm_j4": 0.0, "arm_j5": 0.0},
}


def home_targets(arm_joints: list[tuple[str, str]]) -> dict[tuple[str, str], float]:
    """(side, joint) -> home angle, defaulting to 0 for any unlisted joint."""
    return {(s, j): HOME_POSE.get(s, {}).get(j, 0.0) for (s, j) in arm_joints}


def home_qpos(env, amap, arm_joints: list[tuple[str, str]]) -> np.ndarray:
    """Full reset qpos: hands neutral (0), arm joints at the forward home pose."""
    q = np.zeros_like(env.data.qpos)
    for s, j in arm_joints:
        q[amap.qpos_of(j, s)] = HOME_POSE.get(s, {}).get(j, 0.0)
    return q


class ArmController:
    """Source of YAM arm joint targets (radians), keyed by (side, joint).

    Phase 1 stub: returns a constant home pose. The home defaults to 0 for every
    arm joint (the model's neutral). Replace ``targets()`` with GELLO/VR/IK to
    drive the arms for real; the hand path is unaffected.
    """

    def __init__(self, arm_joints: list[tuple[str, str]], home: dict | None = None) -> None:
        # arm_joints comes from the model (e.g. [('left','arm_j1'), ...]), so the
        # DoF count is whatever the URDF has (5 today, 6 if a wrist joint is added).
        self.home: dict[tuple[str, str], float] = {sj: 0.0 for sj in arm_joints}
        if home:
            self.home.update(home)

    def targets(self, t: float | None = None) -> dict[tuple[str, str], float]:
        return self.home


def split_actuators(amap: sc.ActuatorMap) -> tuple[list[int], list[tuple[str, str]]]:
    """Indices of hand actuators, and (side, joint) of the arm actuators."""
    hand_idx = [k for k, j in enumerate(amap.joints) if not j.startswith("arm")]
    arm_joints = [(amap.sides[k], amap.joints[k]) for k in range(len(amap))
                  if amap.joints[k].startswith("arm")]
    return hand_idx, arm_joints


def build_action(model, amap, cfgs, low, high, smoothed, arm_targets):
    """Full action: hand joints from ``smoothed`` (deg) + arm joints from
    ``arm_targets`` (rad). Both clipped to the model's ctrl ranges."""
    a = np.empty(model.nu, dtype=np.float32)
    for k, (side, joint) in enumerate(amap.side_joints):
        if joint.startswith("arm"):
            a[k] = arm_targets.get((side, joint), 0.0)               # radians
        else:
            neutral = cfgs[side][0]
            a[k] = math.radians(smoothed.get(f"{side}:{joint}", neutral.get(joint, 0.0)))
    return np.clip(a, low, high)


def run_selftest(args) -> int:
    from orca_sim import OrcaYamBimanual
    env = OrcaYamBimanual(render_mode=None)
    cfgs = sb.load_side_configs()
    amap = sc.ActuatorMap(env.model)
    low, high = env.action_low.copy(), env.action_high.copy()
    hand_idx, arm_joints = split_actuators(amap)
    arm = ArmController(arm_joints, home=home_targets(arm_joints))

    nhand, narm = len(hand_idx), len(arm_joints)
    nleft_h = sum(1 for k in hand_idx if amap.sides[k] == "left")
    print(f"actuators: {env.model.nu} | hand={nhand} (left {nleft_h}/right {nhand-nleft_h}) "
          f"| arm={narm} {sorted(arm_joints)}")
    if not (nhand == 34 and narm == 10):
        env.close(); print("SELFTEST: FAIL (expected 34 hand + 10 arm)"); return 1

    # synthetic fist for BOTH hands; arms at home
    fist = {}
    for side, (n, r) in cfgs.items():
        ang = wt.clamp_to_rom(wt.landmarks_to_joint_angles(wt._synthetic_hand(1.0), n, False, True), r)
        for j, v in ang.items():
            fist[f"{side}:{j}"] = v

    env.reset(options={"qpos": home_qpos(env, amap, arm_joints)})
    action = build_action(env.model, amap, cfgs, low, high, fist, arm.targets())
    arm_q0 = np.array([env.data.qpos[amap.qpos_of(j, s)] for s, j in arm_joints])
    for _ in range(200):                          # let servos settle
        env.step(action)
    arm_q = np.array([env.data.qpos[amap.qpos_of(j, s)] for s, j in arm_joints])

    li = amap.qpos_of("index_mcp", "left"); ri = amap.qpos_of("index_mcp", "right")
    in_range = bool(np.all(action >= low) and np.all(action <= high))
    home_vec = np.array([HOME_POSE.get(s, {}).get(j, 0.0) for s, j in arm_joints])
    arm_hold = float(np.max(np.abs(arm_q - home_vec)))   # holds the forward home pose
    arm_drift = float(np.max(np.abs(arm_q - arm_q0)))
    print(f"action in-range: {in_range}")
    print(f"hands curled: left index_mcp={math.degrees(env.data.qpos[li]):+.0f} deg "
          f"right index_mcp={math.degrees(env.data.qpos[ri]):+.0f} deg")
    print(f"arm hold: max |q-home|={math.degrees(arm_hold):.2f} deg, "
          f"max drift from reset={math.degrees(arm_drift):.2f} deg")
    env.close()
    ok = in_range and abs(math.degrees(env.data.qpos[li])) > 30 and math.degrees(arm_hold) < 3.0
    print("SELFTEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def run(args) -> int:
    from orca_sim import OrcaYamBimanual
    env = OrcaYamBimanual(render_mode="human")
    cfgs = sb.load_side_configs()
    amap = sc.ActuatorMap(env.model)
    low, high = env.action_low.copy(), env.action_high.copy()
    _, arm_joints = split_actuators(amap)
    arm = ArmController(arm_joints, home=home_targets(arm_joints))

    env.reset(options={"qpos": home_qpos(env, amap, arm_joints)})
    viewer = env._viewer
    if viewer is None:
        print("ERROR: viewer did not start. On macOS run with `mjpython`."); env.close(); return 1

    smoothed = sb.neutral_flat(cfgs)
    stepper = sc.RealtimeStepper(env)

    if args.demo:
        print("DEMO: both hands sweep, arms hold home. Ctrl+C / close viewer to stop.")
        t0 = time.time()
        try:
            while viewer.is_running():
                c = 0.5 - 0.5 * math.cos((time.time() - t0) * 1.5)
                d = {}
                for side, (n, r) in cfgs.items():
                    for f in ["index", "middle", "ring", "pinky"]:
                        d[f"{side}:{f}_mcp"] = wt.lerp(0, 90, c)
                        d[f"{side}:{f}_pip"] = wt.lerp(0, 95, c)
                action = build_action(env.model, amap, cfgs, low, high, {**smoothed, **d}, arm.targets())
                stepper.step(env, action)
                time.sleep(1 / 60)
        except KeyboardInterrupt:
            pass
        finally:
            env.close()
        return 0

    import cv2
    import mediapipe as mp
    from mediapipe.tasks.python import vision, BaseOptions
    landmarker = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=args.model),
            running_mode=vision.RunningMode.IMAGE, num_hands=2,
            min_hand_detection_confidence=0.4, min_hand_presence_confidence=0.4))
    src = wt.resolve_source(args)
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"ERROR: could not open source {src!r}"); env.close(); return 1
    if isinstance(src, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    filt = wt.OneEuroFilter(args.mincutoff, args.beta)
    print("Running bimanual YAM. Show BOTH hands; arms hold home. Ctrl+C to stop.")
    try:
        while viewer.is_running():
            ok, frame = cap.read()
            if not ok:
                continue
            frame = wt.orient_frame(frame, args.rotate, args.mirror)
            rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            tgt = sb.hand_targets(res, cfgs, args)
            if tgt:
                merged = filt({**smoothed, **tgt}, time.time())
                smoothed.update(merged)
            stepper.step(env, build_action(env.model, amap, cfgs, low, high, smoothed, arm.targets()))
    except KeyboardInterrupt:
        pass
    finally:
        landmarker.close(); cap.release(); env.close()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo", action="store_true", help="No camera: both hands sweep, arms hold home.")
    ap.add_argument("--selftest", action="store_true", help="Headless mapping + arm-hold check.")
    ap.add_argument("--source", default=None, help="Device index or stream URL.")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270])
    ap.add_argument("--mirror", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--world", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--swap-hands", action="store_true", help="Swap which hand drives which side.")
    ap.add_argument("--model", default=wt.DEFAULT_MODEL_PATH)
    ap.add_argument("--mincutoff", type=float, default=wt.ONE_EURO_MINCUTOFF)
    ap.add_argument("--beta", type=float, default=wt.ONE_EURO_BETA)
    args = ap.parse_args()
    return run_selftest(args) if args.selftest else run(args)


if __name__ == "__main__":
    sys.exit(main())
