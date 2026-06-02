#!/usr/bin/env python3
"""Bimanual webcam teleop -> OrcaHandCombined (BOTH hands) in MuJoCo.

num_hands=2: each detected hand is routed by MediaPipe handedness to the left
or right simulated hand. Sim-only, so NO physical calibration is needed. Reuses
the exact retargeting from webcam_teleop (incl. world-landmark angles).

macOS: launch the viewer with mjpython:
    uv run mjpython sim_bimanual.py --demo        # both hands sweep, no camera
    uv run mjpython sim_bimanual.py --no-mirror    # webcam -> both sim hands
    uv run python   sim_bimanual.py --selftest     # headless mapping check

If your hands drive the wrong sides, add --swap-hands.
"""
from __future__ import annotations

import argparse
import math
import sys
import time

import numpy as np

import sim_common as sc  # model-derived actuator map + real-time stepper
import webcam_teleop as wt


def load_side_configs():
    """neutral+ROMs for each side: right = packaged default, left = orcahand_left."""
    from orca_core.hand_config import OrcaHandConfig
    rn, rr, _ = wt.load_neutral_and_roms(None)
    lcfg = OrcaHandConfig.from_config_path(model_name="orcahand_left")
    return {"right": (rn, rr), "left": (dict(lcfg.neutral_position), dict(lcfg.joint_roms_dict))}


def neutral_flat(cfgs):
    return {f"{side}:{j}": v for side, (n, _) in cfgs.items() for j, v in n.items()}


def route_hands(res, swap_hands=False):
    """Route detected hands to sides by horizontal POSITION in the frame
    (leftmost -> 'left'). Deterministic and independent of MediaPipe's flaky
    handedness labels (which flip with mirroring). Returns [(side, hand_index)]."""
    if not res.hand_landmarks:
        return []
    order = sorted((res.hand_landmarks[i][0].x, i) for i in range(len(res.hand_landmarks)))
    sides = ["right", "left"] if swap_hands else ["left", "right"]
    return [(sides[r], i) for r, (_, i) in enumerate(order[:2])]


def hand_targets(res, cfgs, args):
    """{'side:joint': deg} for whichever hands are detected this frame."""
    out = {}
    for side, i in route_hands(res, args.swap_hands):
        neutral, roms = cfgs[side]
        src = (res.hand_world_landmarks[i] if (args.world and res.hand_world_landmarks)
               else res.hand_landmarks[i])
        pts = [(p.x, p.y, p.z) for p in src]
        mir = args.mirror if side == "right" else (not args.mirror)  # left hand is mirrored
        ang = wt.clamp_to_rom(wt.landmarks_to_joint_angles(pts, neutral, False, mir), roms)
        for j, v in ang.items():
            out[f"{side}:{j}"] = v
    return out


def build_action(model, amap, cfgs, low, high, smoothed):
    a = np.empty(model.nu, dtype=np.float32)
    for k, (side, joint) in enumerate(amap.side_joints):
        neutral = cfgs[side][0]
        a[k] = math.radians(smoothed.get(f"{side}:{joint}", neutral.get(joint, 0.0)))
    return np.clip(a, low, high)


def run_selftest(args) -> int:
    from orca_sim import OrcaHandCombined
    env = OrcaHandCombined(render_mode=None)
    cfgs = load_side_configs()
    amap = sc.ActuatorMap(env.model)
    low, high = env.action_low.copy(), env.action_high.copy()
    nleft = amap.sides.count("left")
    nright = amap.sides.count("right")
    print(f"combined actuators: {env.model.nu} | left={nleft} right={nright}")
    if not (nleft == 17 and nright == 17):
        env.close(); print("SELFTEST: FAIL (expected 17+17)"); return 1

    # synthetic fist for BOTH sides
    fist = {}
    for side, (n, r) in cfgs.items():
        ang = wt.clamp_to_rom(wt.landmarks_to_joint_angles(wt._synthetic_hand(1.0), n, False, True), r)
        for j, v in ang.items():
            fist[f"{side}:{j}"] = v
    env.reset()
    action = build_action(env.model, amap, cfgs, low, high, fist)
    for _ in range(40):
        env.step(action)
    in_range = bool(np.all(action >= low) and np.all(action <= high))
    li = amap.qpos_of("index_mcp", "left"); ri = amap.qpos_of("index_mcp", "right")
    print(f"action in-range: {in_range} | left index_mcp qpos={math.degrees(env.data.qpos[li]):+.0f} "
          f"right index_mcp qpos={math.degrees(env.data.qpos[ri]):+.0f}")
    env.close()
    ok = in_range and env.data is not None
    print("SELFTEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def run(args) -> int:
    from orca_sim import OrcaHandCombined
    env = OrcaHandCombined(render_mode="human")
    cfgs = load_side_configs()
    amap = sc.ActuatorMap(env.model)
    low, high = env.action_low.copy(), env.action_high.copy()
    env.reset()
    viewer = env._viewer
    if viewer is None:
        print("ERROR: viewer did not start. On macOS run with `mjpython`."); env.close(); return 1

    smoothed = neutral_flat(cfgs)
    stepper = sc.RealtimeStepper(env)        # keep the sim at real-time speed

    if args.demo:
        print("DEMO: both hands sweep. Ctrl+C / close viewer to stop.")
        t0 = time.time()
        try:
            while viewer.is_running():
                c = 0.5 - 0.5 * math.cos((time.time() - t0) * 1.5)
                d = {}
                for side, (n, r) in cfgs.items():
                    for f in ["index", "middle", "ring", "pinky"]:
                        d[f"{side}:{f}_mcp"] = wt.lerp(0, 90, c)
                        d[f"{side}:{f}_pip"] = wt.lerp(0, 95, c)
                stepper.step(env, build_action(env.model, amap, cfgs, low, high, {**smoothed, **d}))
                time.sleep(1 / 60)
        except KeyboardInterrupt:
            pass
        finally:
            env.close()
        return 0

    import cv2
    import mediapipe as mp
    from mediapipe.tasks.python import vision, BaseOptions
    # IMAGE mode = full detection every frame. VIDEO mode tracks one hand and
    # is lazy about re-detecting the second, so bimanual needs IMAGE mode.
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
    print("Running bimanual. Show BOTH hands. Ctrl+C to stop.")
    try:
        while viewer.is_running():
            ok, frame = cap.read()
            if not ok:
                continue
            frame = wt.orient_frame(frame, args.rotate, args.mirror)
            rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            tgt = hand_targets(res, cfgs, args)
            if tgt:
                merged = filt({**smoothed, **tgt}, time.time())
                smoothed.update(merged)
            stepper.step(env, build_action(env.model, amap, cfgs, low, high, smoothed))
    except KeyboardInterrupt:
        pass
    finally:
        landmarker.close(); cap.release(); env.close()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo", action="store_true", help="No camera: both hands sweep.")
    ap.add_argument("--selftest", action="store_true", help="Headless mapping check.")
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
