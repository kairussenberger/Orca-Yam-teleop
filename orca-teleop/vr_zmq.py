"""ORBIT Quest -> MuJoCo bridge: subscribe to ORBIT's wrist streams, convert the
Unity poses to the robot frame, and produce full 6-DoF end-effector targets for
vr_ik.ArmIK via a calibrated delta-anchor.

ORBIT wire format (verified GestureDetector.cs:323), one NetMQ PushSocket per hand
that .Connect()s to 127.0.0.1:<port> (adb-reverse forwards Quest -> host), so WE
bind a PULL socket on each port:
    "<marker>,px,py,pz,qx,qy,qz,qw"
    - marker  = token[0] (ORBIT sends "relative"; we just skip it)
    - px,py,pz = wrist position, METERS, Unity XR-Origin WORLD frame (LEFT-handed, Y-up)
    - qx,qy,qz,qw = wrist rotation quaternion, XYZW (scalar-LAST), same Unity frame
    - right wrist -> port 8122, left wrist -> port 8123

Frame fix is done with MATRICES, not hand-derived sign flips (we got those wrong
repeatedly): LH->RH by congruence  M_rh = S @ M_unity @ S  with S = diag(1,1,-1)
(flips one axis for rotation AND translation together). Then a DELTA-ANCHOR maps VR
motion to robot motion: target = M_robot0 @ (inv(M_vr0) @ M_vr_now). Only the delta
matters, so operator height/standing-pose/headset-yaw all cancel; the ONLY empirical
knob is which axis S negates (chirality) -- flip it if motion is mirrored on hardware.

MuJoCo quats are WXYZ; scipy/ORBIT are XYZW -- we convert at the boundaries.
"""
from __future__ import annotations

import threading

import numpy as np
import zmq
from scipy.spatial.transform import Rotation as R


# ----- quat order + matrix helpers ----------------------------------------- #
def wxyz_to_xyzw(q):
    return np.array([q[1], q[2], q[3], q[0]], dtype=float)


def xyzw_to_wxyz(q):
    return np.array([q[3], q[0], q[1], q[2]], dtype=float)


def mat_from_pos_quat_xyzw(pos, quat_xyzw):
    M = np.eye(4)
    M[:3, :3] = R.from_quat(quat_xyzw).as_matrix()
    M[:3, 3] = pos
    return M


def make_S4(flip: str = "z"):
    """4x4 handedness-flip (congruence) matrix; flip in {x,y,z,none}."""
    s = np.ones(3)
    if flip in ("x", "y", "z"):
        s["xyz".index(flip)] = -1.0
    S4 = np.eye(4)
    S4[:3, :3] = np.diag(s)
    return S4


def parse_wrist(msg: str):
    """'marker,px,py,pz,qx,qy,qz,qw' -> (pos[3], quat_xyzw[4]) or None."""
    t = msg.split(",")
    if len(t) != 8:
        return None
    try:
        pos = np.array([float(t[1]), float(t[2]), float(t[3])])
        q = np.array([float(t[4]), float(t[5]), float(t[6]), float(t[7])])
    except (ValueError, IndexError):
        return None
    n = np.linalg.norm(q)
    if n < 1e-9:
        return None
    return pos, q / n


def encode_wrist(pos, quat_xyzw, marker: str = "relative") -> str:
    """Build an ORBIT wrist message (used by the synthetic sender / tests)."""
    return (f"{marker},{pos[0]},{pos[1]},{pos[2]},"
            f"{quat_xyzw[0]},{quat_xyzw[1]},{quat_xyzw[2]},{quat_xyzw[3]}")


# ----- transport: one PULL bind per wrist, latest-frame-wins --------------- #
class OrbitWristReader:
    """Binds PULL sockets on the right/left wrist ports, converts each incoming
    Unity pose to a robot-frame 4x4, and keeps only the newest frame per side."""

    def __init__(self, right_port: int = 8122, left_port: int = 8123, flip: str = "z"):
        self.ports = {"right": right_port, "left": left_port}
        self.S4 = make_S4(flip)
        self._lock = threading.Lock()
        self._latest: dict[str, np.ndarray | None] = {"right": None, "left": None}
        self.counts = {"right": 0, "left": 0}
        self._ctx = zmq.Context.instance()
        self._stop = False

    def start(self):
        for side, port in self.ports.items():
            threading.Thread(target=self._run, args=(side, port), daemon=True).start()

    def _run(self, side, port):
        sock = self._ctx.socket(zmq.PULL)
        sock.bind(f"tcp://*:{port}")
        poller = zmq.Poller()
        poller.register(sock, zmq.POLLIN)
        while not self._stop:
            if not dict(poller.poll(timeout=200)).get(sock):
                continue
            msg = None                              # drain the queue, keep newest
            while True:
                try:
                    msg = sock.recv_string(zmq.NOBLOCK)
                except zmq.Again:
                    break
            if msg is None:
                continue
            parsed = parse_wrist(msg)
            if parsed is None:
                continue
            pos, quat = parsed
            M = self.S4 @ mat_from_pos_quat_xyzw(pos, quat) @ self.S4
            with self._lock:
                self._latest[side] = M
                self.counts[side] += 1

    def latest(self) -> dict:
        with self._lock:
            return dict(self._latest)

    def has_both(self) -> bool:
        with self._lock:
            return self._latest["right"] is not None and self._latest["left"] is not None

    def total(self) -> dict:
        with self._lock:
            return dict(self.counts)

    def stop(self):
        self._stop = True


# ----- delta-anchor retargeting ------------------------------------------- #
class PoseRetargeter:
    """Per side: target = M_robot0 @ (inv(M_vr0) @ M_vr_now), with an optional scale
    on the delta translation (human reach -> robot reach)."""

    def __init__(self, pos_scale: float = 1.0):
        self.pos_scale = float(pos_scale)
        self.M_vr0: dict[str, np.ndarray] = {}
        self.M_robot0: dict[str, np.ndarray] = {}

    def calibrate(self, side, M_vr_now, M_robot_now):
        self.M_vr0[side] = np.array(M_vr_now, dtype=float)
        self.M_robot0[side] = np.array(M_robot_now, dtype=float)

    def is_calibrated(self, side) -> bool:
        return side in self.M_vr0

    def fully_calibrated(self) -> bool:
        return "left" in self.M_vr0 and "right" in self.M_vr0

    def target(self, side, M_vr_now):
        """-> (target_pos[3], target_quat_wxyz[4]) for ArmIK, or None."""
        if side not in self.M_vr0:
            return None
        dM = np.linalg.inv(self.M_vr0[side]) @ M_vr_now
        dM[:3, 3] *= self.pos_scale
        T = self.M_robot0[side] @ dM
        return T[:3, 3].copy(), xyzw_to_wxyz(R.from_matrix(T[:3, :3]).as_quat())


class PoseSmoother:
    """Per-side low-pass on the TARGET pose (EMA position, nlerp quaternion-wxyz) to
    soften hand-tracking jitter. alpha in (0,1]: 1=off, lower=smoother."""

    def __init__(self, alpha: float = 0.5):
        self.alpha = max(1e-3, min(1.0, float(alpha)))
        self._p: dict[str, np.ndarray] = {}
        self._q: dict[str, np.ndarray] = {}

    def apply(self, side, pos, quat_wxyz):
        a = self.alpha
        sp, sq = self._p.get(side), self._q.get(side)
        pos = np.asarray(pos, float); q = np.asarray(quat_wxyz, float)
        if sp is None:
            np_, nq = pos, q
        else:
            np_ = sp + a * (pos - sp)
            if float(np.dot(sq, q)) < 0.0:
                q = -q
            nq = sq + a * (q - sq)
            nq = nq / (np.linalg.norm(nq) or 1.0)
        self._p[side], self._q[side] = np_, nq
        return np_, nq

    def reset(self, side):
        self._p.pop(side, None); self._q.pop(side, None)


# --------------------------------------------------------------------------- #
# Headless self-test: transport round-trip + frame conversion + delta-anchor
# --------------------------------------------------------------------------- #
def _rand_pose(rng):
    pos = rng.uniform(-0.5, 0.5, 3)
    quat = R.random(random_state=rng).as_quat()       # xyzw
    return mat_from_pos_quat_xyzw(pos, quat)


def selftest() -> int:
    ok = True
    rng = np.random.default_rng(0)

    # 1) transport: PUSH(connect) -> PULL(bind), newest frame, on test ports
    reader = OrbitWristReader(right_port=18122, left_port=18123, flip="z")
    reader.start()
    import time
    time.sleep(0.4)
    ctx = zmq.Context.instance()
    sr = ctx.socket(zmq.PUSH); sr.connect("tcp://127.0.0.1:18122")
    sl = ctx.socket(zmq.PUSH); sl.connect("tcp://127.0.0.1:18123")
    time.sleep(0.2)
    # Unity pose (0.1, 0.2, 0.3), identity rot -> robot pos flips Z -> (0.1,0.2,-0.3)
    sr.send_string(encode_wrist([0.1, 0.2, 0.3], [0, 0, 0, 1]))
    sl.send_string(encode_wrist([0.0, 0.0, 0.0], [0, 0, 0, 1]))
    time.sleep(0.3)
    st = reader.latest()
    t1 = st["right"] is not None and st["left"] is not None and reader.total()["right"] >= 1
    pr = st["right"][:3, 3] if st["right"] is not None else np.zeros(3)
    t2 = np.allclose(pr, [0.1, 0.2, -0.3], atol=1e-6)        # Z flipped by S=diag(1,1,-1)
    print(f"  transport: got right={st['right'] is not None} left={st['left'] is not None}  {'ok' if t1 else 'BAD'}")
    print(f"  frame flip: unity(0.1,0.2,0.3)->robot {np.round(pr,3)} (want 0.1,0.2,-0.3)  {'ok' if t2 else 'BAD'}")
    ok &= t1 and t2

    # 2) delta-anchor: no motion -> target == robot home; known delta -> robot0 @ delta
    rt = PoseRetargeter(pos_scale=1.0)
    M_vr0 = _rand_pose(rng); M_robot0 = _rand_pose(rng)
    rt.calibrate("right", M_vr0, M_robot0)
    p0, q0 = rt.target("right", M_vr0)                       # identity delta
    home_ok = (np.allclose(p0, M_robot0[:3, 3], atol=1e-9)
               and np.allclose(R.from_quat(wxyz_to_xyzw(q0)).as_matrix(), M_robot0[:3, :3], atol=1e-9))
    delta = mat_from_pos_quat_xyzw([0.07, -0.03, 0.05], R.from_euler("xyz", [0.2, -0.1, 0.3]).as_quat())
    M_vr_now = M_vr0 @ delta
    p1, q1 = rt.target("right", M_vr_now)
    T_exp = M_robot0 @ delta
    delta_ok = (np.allclose(p1, T_exp[:3, 3], atol=1e-9)
                and np.allclose(R.from_quat(wxyz_to_xyzw(q1)).as_matrix(), T_exp[:3, :3], atol=1e-9))
    print(f"  anchor no-motion -> home pose: {'ok' if home_ok else 'BAD'}")
    print(f"  anchor delta -> robot0@delta:  {'ok' if delta_ok else 'BAD'}")
    ok &= home_ok and delta_ok

    reader.stop()
    print("VR-ZMQ SELFTEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(selftest())
