#!/usr/bin/env python3
"""Bimanual webcam teleop -> two ORCA hands mounted on two I2RT YAM arms.

Phase 1 (hands-only): the 34 ORCA-hand actuators are driven from the webcam
exactly as in sim_bimanual.py; the 12 YAM arm actuators (6/side incl. the
wrist-roll j6) are held at a home pose by an ``ArmController`` stub. Swap a real arm input (GELLO leader arms, VR 6-DoF
-> IK, full-body pose) into ``ArmController`` later without touching the hand
path -- ``build_action`` routes each actuator by (side, joint) from the model.

Phase 2 (VR wrist, orientation-only): ``--vr`` drives the arm wrist joints
(j4/j5/j6) from a Meta Quest's hand-tracked wrist quaternion (streamed over wss
from vr/vr_client.html via vr_teleop.py); j1..j3 hold home, fingers stay neutral.

macOS: launch the viewer with mjpython:
    ./run_yam.sh --demo                               # hands sweep, arms hold
    ./run_yam.sh                                      # webcam -> both sim hands
    ./run_yam_vr.sh                                   # Quest wrist -> arm wrist joints
    .venv/bin/python sim_yam_bimanual.py --selftest     # headless hand/arm check
    .venv/bin/python sim_yam_bimanual.py --vr-selftest  # headless VR-pipeline check

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
# arm_j6 is the wrist-roll added at the link5 flange; home 0 = the welded
# reference (preserves the verified flush hand contact) until phase-2 IK drives it.
HOME_POSE = {
    "left":  {"arm_j1": 1.5708, "arm_j2": 0.0, "arm_j3": 0.0, "arm_j4": 0.0, "arm_j5": 0.0, "arm_j6": 0.0},
    "right": {"arm_j1": 4.7124, "arm_j2": 0.0, "arm_j3": 0.0, "arm_j4": 0.0, "arm_j5": 0.0, "arm_j6": 0.0},
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
        # DoF count is whatever the model has (6 today: j1..j5 + the wrist-roll j6).
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
    if not (nhand == 34 and narm == 12):
        env.close(); print("SELFTEST: FAIL (expected 34 hand + 12 arm)"); return 1

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


def run_vr(args) -> int:
    """Phase-2 milestone: Quest hand-tracking wrist quaternion -> arm wrist joints
    (j4/j5/j6), orientation only. Fingers stay neutral; j1..j3 hold the forward
    home pose. The WebXR page streams pose over wss to an in-process server."""
    import vr_teleop as vr
    from orca_sim import OrcaYamBimanual
    env = OrcaYamBimanual(render_mode="human")
    cfgs = sb.load_side_configs()
    amap = sc.ActuatorMap(env.model)
    low, high = env.action_low.copy(), env.action_high.copy()
    _, arm_joints = split_actuators(amap)
    arm = vr.VRArmController(arm_joints, home=home_targets(arm_joints))

    env.reset(options={"qpos": home_qpos(env, amap, arm_joints)})
    viewer = env._viewer
    if viewer is None:
        print("ERROR: viewer did not start. On macOS run with `mjpython`."); env.close(); return 1

    smoothed = sb.neutral_flat(cfgs)              # fingers neutral in this milestone
    stepper = sc.RealtimeStepper(env)
    rx = vr.VRReceiver(host="0.0.0.0", port=args.vr_port, ip=vr.lan_ip())
    rx.start()
    smoother = vr.QuatSmoother(args.vr_smooth)
    ip = vr.lan_ip() or "<your-mac-ip>"
    print(f"\n  Quest Browser -> https://{ip}:{args.vr_port}/   (accept the cert, 'Enter teleop')")
    print("  Show BOTH hands; hold them in a neutral pose for calibration.\n")

    phase, t_phase, last_count, t_dbg, t_wait = "wait", time.time(), -1, 0.0, 0.0
    announced = set()
    try:
        while viewer.is_running():
            state = smoother.apply(rx.latest())
            arm.update(state)
            both = state.get("left") is not None and state.get("right") is not None
            if phase == "wait":
                if both:
                    phase, t_phase = "count", time.time()
                    print("Both hands detected. Hold them in your NEUTRAL pose...")
                elif time.time() - t_wait > 1.0:        # tell the user WHY it's not starting
                    t_wait = time.time()
                    seen = [s for s in ("left", "right") if state.get(s) is not None] or ["none"]
                    print(f"  waiting for BOTH hands... seen={seen}  (socket frames={rx.msgs})")
            elif phase == "count":
                if not both:                            # a hand dropped: restart the hold
                    t_phase = time.time()
                else:
                    left = args.vr_calib - (time.time() - t_phase)
                    if left <= 0:
                        announced = set(arm.calibrate())
                        print(f"Calibrated {sorted(announced)}. Move your wrists — arms follow. "
                              f"(--vr-debug shows live rotvec + joints.)")
                        phase = "run"
                    elif int(left) != last_count:
                        last_count = int(left); print(f"  calibrating in {int(left)+1}...")
            elif phase == "run":
                for s in arm.calibrate_missing():       # auto-join a hand missing at calibration
                    if s not in announced:
                        announced.add(s); print(f"  {s} hand auto-joined (calibrated).")
            # until a side is calibrated, mapper omits it -> that arm holds home
            tg = arm.targets()
            if args.vr_debug and phase == "run" and time.time() - t_dbg > 0.3:
                t_dbg = time.time()
                rows = arm.status()
                jd = lambda s: tuple(round(math.degrees(tg.get((s, f"arm_j{n}"), 0.0)), 1) for n in (4, 5, 6))
                def fmt(s):
                    r = rows[s]; tag = "CAL" if r["calibrated"] else ("track" if r["tracked"] else "none")
                    return f"{s[0].upper()}[{tag}] rotvec={r['rotvec_deg']} j456={jd(s)}"
                print(f"  {fmt('left')}   {fmt('right')}")
            action = build_action(env.model, amap, cfgs, low, high, smoothed, tg)
            stepper.step(env, action)
    except KeyboardInterrupt:
        pass
    finally:
        env.close()
    return 0


def run_vr_selftest(args) -> int:
    """Headless: module checks (mapper + wss) then a sim integration check that a
    synthetic wrist quaternion actually drives the wrist joints in-range."""
    import vr_teleop as vr
    if vr.selftest() != 0:
        return 1
    from orca_sim import OrcaYamBimanual
    env = OrcaYamBimanual(render_mode=None)
    cfgs = sb.load_side_configs()
    amap = sc.ActuatorMap(env.model)
    low, high = env.action_low.copy(), env.action_high.copy()
    _, arm_joints = split_actuators(amap)
    arm = vr.VRArmController(arm_joints, home=home_targets(arm_joints))

    env.reset(options={"qpos": home_qpos(env, amap, arm_joints)})
    smoothed = sb.neutral_flat(cfgs)
    # NON-identity neutral + BODY-LOCAL motions (a real hand is never at identity;
    # an identity neutral can't catch a wrong-axis MAP). roll = local Z -> j6.
    q0l, q0r = vr._q_about("y", 0.3), vr._q_about("x", -0.2)
    arm.update({"left": {"q": q0l}, "right": {"q": q0r}})
    arm.calibrate()
    ql = vr.q_mul(np.asarray(q0l, float), np.asarray(vr._q_about("z", 0.5), float))  # local ROLL -> left j6
    qr = vr.q_mul(np.asarray(q0r, float), np.asarray(vr._q_about("x", 0.4), float))  # local flex -> right j5
    arm.update({"left": {"q": ql}, "right": {"q": qr}})
    action = build_action(env.model, amap, cfgs, low, high, smoothed, arm.targets())
    in_range = bool(np.all(action >= low) and np.all(action <= high))
    for _ in range(300):
        env.step(action)
    j6l = float(env.data.qpos[amap.qpos_of("arm_j6", "left")])
    j5r = float(env.data.qpos[amap.qpos_of("arm_j5", "right")])
    j4l = float(env.data.qpos[amap.qpos_of("arm_j4", "left")])
    print(f"integration: left j6={j6l:+.3f} (want +0.5)  right j5={j5r:+.3f} (want +0.4)  "
          f"left j4={j4l:+.3f} (want 0)  in-range={in_range}")
    env.close()
    ok = in_range and abs(j6l - 0.5) < 0.05 and abs(j5r - 0.4) < 0.05 and abs(j4l) < 0.05
    print("VR-INTEGRATION SELFTEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def _apply_sol(arm_targets: dict, sol: dict) -> None:
    """Merge ArmIK's {'left_arm_j1': rad, ...} into arm_targets keyed by (side, joint)."""
    for jn, val in sol.items():
        sd, jt = jn.split("_", 1)               # 'left_arm_j6' -> ('left', 'arm_j6')
        arm_targets[(sd, jt)] = val


def run_orbit(args) -> int:
    """Full-pose VR teleop driven by the ORBIT Quest app over ZeroMQ. Each hand's
    wrist pose (delta-anchored to a calibrated neutral, vr_zmq) becomes a 6-DoF
    end-effector target solved by vr_ik.ArmIK; j1..j6 all move, fingers stay neutral."""
    import vr_zmq as vz
    import vr_ik
    from orca_sim import OrcaYamBimanual
    env = OrcaYamBimanual(render_mode="human")
    cfgs = sb.load_side_configs()
    amap = sc.ActuatorMap(env.model)
    low, high = env.action_low.copy(), env.action_high.copy()
    _, arm_joints = split_actuators(amap)

    iks = {s: vr_ik.ArmIK(env.model, s) for s in ("left", "right")}
    reader = vz.OrbitWristReader(args.orbit_right_port, args.orbit_left_port, flip=args.orbit_flip)
    reader.start()
    retarget = vz.PoseRetargeter(pos_scale=args.orbit_scale)
    smoother = vz.PoseSmoother(args.orbit_smooth)

    env.reset(options={"qpos": home_qpos(env, amap, arm_joints)})
    viewer = env._viewer
    if viewer is None:
        print("ERROR: viewer did not start. On macOS run with mjpython."); env.close(); return 1
    smoothed = sb.neutral_flat(cfgs)
    stepper = sc.RealtimeStepper(env)
    arm_targets = home_targets(arm_joints)        # IK overwrites the streaming sides

    def ee_mat(side):
        p, qw = iks[side].ee_pose(env.data)
        return vz.mat_from_pos_quat_xyzw(p, vz.wxyz_to_xyzw(qw))

    print(f"\n  ORBIT app -> wrist ports {args.orbit_right_port}(R)/{args.orbit_left_port}(L). "
          f"Make sure `adb reverse tcp:{args.orbit_right_port} tcp:{args.orbit_right_port}` etc. are set.\n"
          f"  Show BOTH hands; hold a comfortable NEUTRAL pose to calibrate.\n")

    phase, t_phase, last_count, t_dbg, t_wait = "wait", time.time(), -1, 0.0, 0.0
    try:
        while viewer.is_running():
            st = reader.latest()
            both = st["left"] is not None and st["right"] is not None
            if phase == "wait":
                if both:
                    phase, t_phase = "count", time.time()
                    print("Both wrists streaming. Hold your NEUTRAL pose...")
                elif time.time() - t_wait > 1.0:
                    t_wait = time.time()
                    seen = [s for s in ("left", "right") if st[s] is not None] or ["none"]
                    print(f"  waiting for BOTH wrist streams... seen={seen} frames={reader.total()}")
            elif phase == "count":
                if not both:
                    t_phase = time.time()
                else:
                    left = args.vr_calib - (time.time() - t_phase)
                    if left <= 0:
                        for s in ("left", "right"):
                            retarget.calibrate(s, st[s], ee_mat(s)); smoother.reset(s)
                        print("Calibrated both. Move your hands — the arms follow in full 6-DoF.")
                        phase = "run"
                    elif int(left) != last_count:
                        last_count = int(left); print(f"  calibrating in {int(left)+1}...")
            elif phase == "run":
                for s in ("left", "right"):            # auto-join a wrist that appears late
                    if st[s] is not None and not retarget.is_calibrated(s):
                        retarget.calibrate(s, st[s], ee_mat(s)); smoother.reset(s)
                        print(f"  {s} wrist auto-joined.")

            res = {}
            if phase == "run":
                for s in ("left", "right"):
                    if st[s] is not None and retarget.is_calibrated(s):
                        tp, tq = retarget.target(s, st[s])
                        tp, tq = smoother.apply(s, tp, tq)
                        sol, r = iks[s].solve(env.data.qpos.copy(), tp, tq, iters=args.orbit_iters)
                        _apply_sol(arm_targets, sol); res[s] = r
            if args.vr_debug and phase == "run" and time.time() - t_dbg > 0.3:
                t_dbg = time.time()
                print("  IK residual " + "  ".join(
                    f"{s[0].upper()}={res[s]:.3f}" for s in ("left", "right") if s in res))
            action = build_action(env.model, amap, cfgs, low, high, smoothed, arm_targets)
            stepper.step(env, action)
    except KeyboardInterrupt:
        pass
    finally:
        reader.stop(); env.close()
    return 0


def run_orbit_selftest(args) -> int:
    """Headless end-to-end: synthetic ORBIT PUSH sender -> ZMQ -> frame-fix ->
    delta-anchor -> ArmIK -> verify the EE reaches the commanded target. No headset."""
    import time as _t
    import mujoco
    import zmq
    from scipy.spatial.transform import Rotation as Rr
    import vr_zmq as vz
    import vr_ik
    from orca_sim import OrcaYamBimanual
    if vz.selftest() != 0:
        return 1
    env = OrcaYamBimanual(render_mode=None)
    amap = sc.ActuatorMap(env.model)
    _, arm_joints = split_actuators(amap)
    env.reset(options={"qpos": home_qpos(env, amap, arm_joints)})
    mujoco.mj_forward(env.model, env.data)
    iks = {s: vr_ik.ArmIK(env.model, s) for s in ("left", "right")}

    reader = vz.OrbitWristReader(8122, 8123, flip="z"); reader.start(); _t.sleep(0.4)
    ctx = zmq.Context.instance()
    snd = {"right": ctx.socket(zmq.PUSH), "left": ctx.socket(zmq.PUSH)}
    snd["right"].connect("tcp://127.0.0.1:8122"); snd["left"].connect("tcp://127.0.0.1:8123")
    _t.sleep(0.2)

    neutral = {"right": ([0.2, 1.0, 0.3], [0, 0, 0, 1]), "left": ([-0.2, 1.0, 0.3], [0, 0, 0, 1])}
    for s in ("right", "left"):
        snd[s].send_string(vz.encode_wrist(*neutral[s]))
    _t.sleep(0.3)

    def ee_mat(side):
        p, qw = iks[side].ee_pose(env.data); return vz.mat_from_pos_quat_xyzw(p, vz.wxyz_to_xyzw(qw))
    retarget = vz.PoseRetargeter(pos_scale=1.0)
    st = reader.latest()
    for s in ("right", "left"):
        retarget.calibrate(s, st[s], ee_mat(s))

    # small body-local delta in the Unity frame: 3cm along unity +X, 10deg about unity Y
    # (small enough to stay inside BOTH arms' reachable set near home)
    for s in ("right", "left"):
        p, q = neutral[s]
        d = vz.mat_from_pos_quat_xyzw([0.03, 0, 0], Rr.from_euler("y", 10, degrees=True).as_quat())
        Mm = vz.mat_from_pos_quat_xyzw(p, q) @ d
        snd[s].send_string(vz.encode_wrist(Mm[:3, 3], Rr.from_matrix(Mm[:3, :3]).as_quat()))
    _t.sleep(0.3)

    st = reader.latest()
    ok = True
    for s in ("right", "left"):
        tp, tq = retarget.target(s, st[s])
        moved = float(np.linalg.norm(tp - ee_mat(s)[:3, 3]))
        sol, r = iks[s].solve(env.data.qpos.copy(), tp, tq, iters=300, tol=1e-5)
        d2 = mujoco.MjData(env.model); d2.qpos[:] = env.data.qpos
        for jn, val in sol.items():
            jid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, jn)
            d2.qpos[int(env.model.jnt_qposadr[jid])] = val
        mujoco.mj_forward(env.model, d2)
        pe = float(np.linalg.norm(d2.xpos[iks[s].ee] - tp)) * 1000.0
        good = moved > 0.01 and pe < 3.0
        print(f"  {s}: target moved {moved*100:.1f}cm from home, IK reached pos_err={pe:.2f}mm "
              f"residual={r:.1e}  {'ok' if good else 'BAD'}")
        ok &= good
    reader.stop(); env.close()
    print("ORBIT-INTEGRATION SELFTEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo", action="store_true", help="No camera: both hands sweep, arms hold home.")
    ap.add_argument("--selftest", action="store_true", help="Headless mapping + arm-hold check.")
    ap.add_argument("--vr", action="store_true", help="VR wrist teleop: Quest hand-tracking quat -> arm wrist joints (j4/j5/j6).")
    ap.add_argument("--vr-selftest", action="store_true", help="Headless VR pipeline check (mapper + wss + sim), no headset.")
    ap.add_argument("--vr-port", type=int, default=8443, help="HTTPS/wss port the Quest Browser connects to.")
    ap.add_argument("--vr-calib", type=float, default=3.0, help="Seconds to hold a neutral wrist pose before calibration.")
    ap.add_argument("--vr-debug", action="store_true", help="Print live wrist-joint targets (deg) so you can verify/flip signs.")
    ap.add_argument("--vr-smooth", type=float, default=0.4, help="Wrist-quaternion low-pass alpha (1=off, lower=smoother/laggier).")
    # ORBIT (Quest app over ZeroMQ) full-pose IK path — supersedes the WebXR --vr path
    ap.add_argument("--orbit", action="store_true", help="Full-pose VR teleop from the ORBIT Quest app over ZeroMQ (wrist pose -> IK).")
    ap.add_argument("--orbit-selftest", action="store_true", help="Headless ORBIT pipeline check (synthetic sender + IK), no headset.")
    ap.add_argument("--orbit-right-port", type=int, default=8122, help="PULL bind port for the right wrist stream.")
    ap.add_argument("--orbit-left-port", type=int, default=8123, help="PULL bind port for the left wrist stream.")
    ap.add_argument("--orbit-scale", type=float, default=1.0, help="Scale on the delta translation (human reach -> robot reach).")
    ap.add_argument("--orbit-flip", default="z", choices=["x", "y", "z", "none"], help="Handedness axis to negate (chirality knob; flip if motion is mirrored).")
    ap.add_argument("--orbit-smooth", type=float, default=0.5, help="Target-pose low-pass alpha (1=off, lower=smoother).")
    ap.add_argument("--orbit-iters", type=int, default=15, help="DLS IK iterations per frame (warm-started).")
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
    if args.orbit_selftest:
        return run_orbit_selftest(args)
    if args.orbit:
        return run_orbit(args)
    if args.vr_selftest:
        return run_vr_selftest(args)
    if args.vr:
        return run_vr(args)
    return run_selftest(args) if args.selftest else run(args)


if __name__ == "__main__":
    sys.exit(main())
