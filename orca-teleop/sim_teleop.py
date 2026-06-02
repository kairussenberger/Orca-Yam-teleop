#!/usr/bin/env python3
"""
Drive the ORCA MuJoCo simulation (orca_sim) — a hardware-free twin of
webcam_teleop.py. Reuses the exact same landmark->joint retargeting.

The sim's 17 position actuators map 1:1 to the 17 ORCA joints, in radians,
over the same ranges as the config ROMs. So we just convert our (degree)
joint angles to radians and write them as the action.

macOS: the MuJoCo viewer MUST be launched with `mjpython`, not `python`:

    uv run mjpython sim_teleop.py --demo     # no camera: open/close sweep
    uv run mjpython sim_teleop.py            # webcam -> sim hand mirror
    uv run python   sim_teleop.py --selftest # headless mapping check (no viewer)

In webcam mode there is no camera preview window (the MuJoCo viewer is the
display, and macOS only allows one GUI event loop here). Move your RIGHT hand
in front of the camera and the simulated hand mirrors it. Ctrl+C or close the
viewer to stop.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np

import sim_common as sc  # model-derived actuator map + real-time stepper
import webcam_teleop as wt  # reuse retargeting, config loading, model path


def make_action_builder(model, env, neutral):
    """Return ``(build, amap)`` where build(angles_deg_dict) -> clipped float32
    action in radians, and amap is the model-derived :class:`sim_common.ActuatorMap`."""
    amap = sc.ActuatorMap(model)
    low, high = env.action_low.copy(), env.action_high.copy()

    def build(angles_deg: dict) -> np.ndarray:
        a = np.empty(model.nu, dtype=np.float32)
        for i, jn in enumerate(amap.joints):
            a[i] = math.radians(angles_deg.get(jn, neutral.get(jn, 0.0)))
        return np.clip(a, low, high)

    return build, amap


def demo_angles(neutral: dict, t: float) -> dict:
    """A smooth open<->close sweep of all flexion joints (no camera)."""
    out = dict(neutral)
    c = 0.5 - 0.5 * math.cos(t * 1.5)        # 0..1
    for f in ["index", "middle", "ring", "pinky"]:
        out[f"{f}_mcp"] = wt.lerp(0.0, 90.0, c)
        out[f"{f}_pip"] = wt.lerp(0.0, 95.0, c)
    out["thumb_mcp"] = wt.lerp(0.0, 60.0, c)
    out["thumb_dip"] = wt.lerp(0.0, 70.0, c)
    return out


def run_selftest(args) -> int:
    """Headless: verify the actuator mapping and a couple of poses. No viewer."""
    from orca_sim import OrcaHandRight
    env = OrcaHandRight(render_mode=None)
    neutral, roms, cfg = wt.load_neutral_and_roms(args.config)
    build, amap = make_action_builder(env.model, env, neutral)

    missing = set(cfg.joint_ids) - set(amap.joints)
    print(f"actuators: {env.model.nu} | mapped joints: {amap.joints}")
    if missing:
        print(f"!! actuator mapping missing ORCA joints: {missing}")
        env.close()
        return 1

    imcp_qpos = amap.qpos_of("index_mcp")    # robust: derived from the model
    env.reset()
    for label, ang in [("neutral", neutral),
                       ("fist", wt.clamp_to_rom(demo_angles(neutral, math.pi / 1.5), roms))]:
        action = build(ang)
        for _ in range(40):                  # let position actuators settle
            env.step(action)
        qpos = env.data.qpos.copy()
        print(f"[{label}] action in-range: {bool(np.all(action >= env.action_low) and np.all(action <= env.action_high))} "
              f"| index_mcp cmd={ang.get('index_mcp'):+.1f}deg | i-mcp qpos={math.degrees(qpos[imcp_qpos]):+.1f}deg")
    env.close()
    print("SIM SELFTEST: PASS")
    return 0


def run(args) -> int:
    from orca_sim import OrcaHandRight
    env = OrcaHandRight(render_mode="human")
    neutral, roms, cfg = wt.load_neutral_and_roms(args.config)
    build, _ = make_action_builder(env.model, env, neutral)

    env.reset()                              # creates the viewer (needs mjpython)
    viewer = env._viewer
    if viewer is None:
        print("ERROR: viewer did not start. On macOS run with `mjpython`, not `python`.")
        env.close()
        return 1

    stepper = sc.RealtimeStepper(env)        # keep the sim at real-time speed

    if args.demo:
        print("DEMO: open/close sweep. Close the viewer or Ctrl+C to stop.")
        t0 = time.time()
        try:
            while viewer.is_running():
                stepper.step(env, build(wt.clamp_to_rom(demo_angles(neutral, time.time() - t0), roms)))
                time.sleep(1.0 / 60)
        except KeyboardInterrupt:
            pass
        finally:
            env.close()
        return 0

    # ---- webcam -> sim mirror (camera runs headless; viewer is the display) ----
    import cv2
    import mediapipe as mp
    from mediapipe.tasks.python import vision, BaseOptions

    if not os.path.exists(args.model):
        print(f"ERROR: model not found: {args.model}")
        env.close()
        return 1

    landmarker = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=args.model),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    )
    source = wt.resolve_source(args)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"ERROR: could not open video source {source!r}.")
        env.close()
        return 1
    if isinstance(source, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    smoothed = dict(neutral)
    filt = wt.OneEuroFilter(args.mincutoff, args.beta)
    t_start, last_ts = time.time(), -1
    print("Running. Show your RIGHT hand to the camera; the sim mirrors it. Ctrl+C to stop.")
    try:
        while viewer.is_running():
            ok, frame = cap.read()
            if not ok:
                continue
            frame = wt.orient_frame(frame, args.rotate, args.mirror)
            rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            ts = max(int((time.time() - t_start) * 1000), last_ts + 1)
            last_ts = ts
            res = landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)
            if res.hand_landmarks:
                target = wt.clamp_to_rom(
                    wt.landmarks_to_joint_angles(
                        wt.angle_points(res, args.world), neutral, args.wrist, args.mirror), roms)
                smoothed = filt(target, time.time())
            stepper.step(env, build(smoothed))   # catch up to wall-clock between frames
    except KeyboardInterrupt:
        pass
    finally:
        landmarker.close()
        cap.release()
        env.close()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="config.yaml (default: packaged right v2).")
    ap.add_argument("--demo", action="store_true", help="No camera: open/close sweep in sim.")
    ap.add_argument("--selftest", action="store_true", help="Headless mapping check (no viewer).")
    ap.add_argument("--camera", type=int, default=0, help="Webcam index.")
    ap.add_argument("--source", default=None,
                    help="Device index OR phone stream URL (http://.../video, rtsp://...).")
    ap.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270],
                    help="Rotate frames (sideways-mounted phone).")
    ap.add_argument("--mirror", action=argparse.BooleanOptionalAction, default=True,
                    help="Mirror image; use --no-mirror for an egocentric/head view.")
    ap.add_argument("--world", action=argparse.BooleanOptionalAction, default=False,
                    help="Use metric 3D world landmarks for angles (better under occlusion).")
    ap.add_argument("--model", default=wt.DEFAULT_MODEL_PATH, help="hand_landmarker.task path.")
    ap.add_argument("--mincutoff", type=float, default=wt.ONE_EURO_MINCUTOFF,
                    help="One-Euro min cutoff Hz (higher = snappier).")
    ap.add_argument("--beta", type=float, default=wt.ONE_EURO_BETA,
                    help="One-Euro beta (higher = less lag on fast motion).")
    ap.add_argument("--wrist", action="store_true", help="Enable rough wrist mapping.")
    args = ap.parse_args()

    if args.selftest:
        return run_selftest(args)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
