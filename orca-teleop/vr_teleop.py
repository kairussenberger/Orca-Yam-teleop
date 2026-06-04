"""Phase-2 VR arm input: Quest hand-tracking wrist quaternion -> YAM wrist joints.

Pipeline (orientation-only milestone):
    Quest Browser (vr/vr_client.html, WebXR hand tracking)
      --wss/JSON-->  VRReceiver (this file, background-thread server)
      -->            WristMapper  (wrist quat, relative to a calibrated neutral,
                                   -> Euler -> arm_j4/j5/j6 angles)
      -->            VRArmController.targets()  (home for j1..j3 + mapped j4..j6)
      -->            sim_yam_bimanual.build_action -> env.step

Only ORIENTATION is used here (position is received and stored for later IK). The
hand/finger actuators are NOT touched by this module -- they keep whatever the
caller feeds (neutral in --vr v1).

WebXR needs HTTPS, so the page + socket are served over TLS with a self-signed
cert (accept the browser warning once). The server runs in a daemon thread with
its own asyncio loop; the sim reads the latest pose synchronously each frame.
"""
from __future__ import annotations

import asyncio
import json
import math
import os
import ssl
import subprocess
import threading
import time

import numpy as np
from websockets.asyncio.server import serve
from websockets.http11 import Response
from websockets.datastructures import Headers

HERE = os.path.dirname(os.path.abspath(__file__))
HTML_PATH = os.path.join(HERE, "vr", "vr_client.html")
CERT_DIR = os.path.join(HERE, "vr", "certs")

# Wrist joint ranges (rad), must match the model (yam_*_body.xml).
JOINT_RANGE = {"arm_j4": (-1.570796, 1.570796),
               "arm_j5": (-1.570796, 1.570796),
               "arm_j6": (-3.141590, 3.141590)}


# --------------------------------------------------------------------------- #
# Quaternion helpers  (all quats are [x, y, z, w], the WebXR/DOMPoint order)
# --------------------------------------------------------------------------- #
def q_conj(q):
    x, y, z, w = q
    return np.array([-x, -y, -z, w])


def q_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ])


def quat_to_rotvec(q):
    """Quaternion -> rotation vector (axis * angle), a 3-vec in the quat's frame.

    Gimbal-free, and for a pure single-axis rotation the other two components are
    exactly 0 -- so each wrist motion drives one joint cleanly. We normalize to the
    shorter rotation (q ~ -q) so the angle stays in [-pi, pi]."""
    x, y, z, w = q
    if w < 0.0:
        x, y, z, w = -x, -y, -z, -w
    v = np.array([x, y, z], dtype=float)
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return 2.0 * v                       # small angle: rotvec ~ 2*(x,y,z)
    return (2.0 * math.atan2(n, w) / n) * v


# --------------------------------------------------------------------------- #
# Wrist mapper: relative wrist orientation -> j4/j5/j6 targets
# --------------------------------------------------------------------------- #
class WristMapper:
    """Map each hand's wrist quaternion (relative to a calibrated neutral) to the
    three wrist joints. Orientation-only: position is ignored here.

    Assignment (rotation-vector component in the calibrated wrist-LOCAL frame ->
    joint), grounded in the W3C WebXR Hand Input spec: the wrist joint's local -Z
    runs along the bone toward the fingers = the forearm / pronation-supination
    (ROLL) axis; local -Y is the palm normal = radial/ulnar deviation; local X is
    flex/extend. q_rel = conj(neutral) * q is the delta in this local frame, so:
        rotvec.x (local X)  wrist flex/extend      -> arm_j5  (+-pi/2)
        rotvec.y (local Y)  radial/ulnar deviation -> arm_j4  (+-pi/2)
        rotvec.z (local Z)  forearm ROLL           -> arm_j6  (+-pi, the big joint)
    Magnitudes are 1:1. SIGNS are empirical (robot j5 axis is -X; j6 follows the
    WebXR -Z-toward-fingers convention) -- watch the rotvec in --vr-debug and flip
    the per-joint sign below if a motion drives its joint the wrong way.
    """

    # (joint, rotvec-component 0=localX/1=localY/2=localZ, sign, gain)
    DEFAULT_MAP = [("arm_j5", 0, +1.0, 1.0),   # local X = flex/extend
                   ("arm_j4", 1, +1.0, 1.0),   # local Y = radial/ulnar deviation
                   ("arm_j6", 2, +1.0, 1.0)]   # local Z = forearm ROLL  (the fix)

    def __init__(self):
        self.neutral: dict[str, np.ndarray] = {}     # side -> neutral quat
        self.MAP = [tuple(m) for m in self.DEFAULT_MAP]

    def calibrate(self, state: dict, side: str | None = None) -> list[str]:
        """Capture the current wrist quat as neutral. Returns sides calibrated."""
        done = []
        for s in (("left", "right") if side is None else (side,)):
            e = state.get(s)
            if e and e.get("q") is not None:
                self.neutral[s] = np.asarray(e["q"], dtype=float)
                done.append(s)
        return done

    def targets(self, state: dict) -> dict[tuple[str, str], float]:
        """{(side, joint): radians} for the wrist joints of any calibrated, tracked
        hand. Joints with no data / no calibration are omitted (caller holds home)."""
        out: dict[tuple[str, str], float] = {}
        for side in ("left", "right"):
            e = state.get(side)
            q0 = self.neutral.get(side)
            if not e or e.get("q") is None or q0 is None:
                continue
            q_rel = q_mul(q_conj(q0), np.asarray(e["q"], dtype=float))
            rotvec = quat_to_rotvec(q_rel)
            for joint, idx, sign, gain in self.MAP:
                lo, hi = JOINT_RANGE[joint]
                out[(side, joint)] = float(np.clip(sign * gain * rotvec[idx], lo, hi))
        return out

    def is_calibrated(self, side: str) -> bool:
        return side in self.neutral

    def fully_calibrated(self) -> bool:
        return "left" in self.neutral and "right" in self.neutral

    def relative_rotvec(self, state: dict, side: str):
        """rotvec of the wrist delta from neutral (None if untracked/uncalibrated)."""
        e = state.get(side)
        q0 = self.neutral.get(side)
        if not e or e.get("q") is None or q0 is None:
            return None
        return quat_to_rotvec(q_mul(q_conj(q0), np.asarray(e["q"], dtype=float)))

    def debug_rows(self, state: dict) -> dict:
        """Per-side {tracked, calibrated, rotvec_deg} for live diagnostics."""
        rows = {}
        for side in ("left", "right"):
            e = state.get(side)
            rv = self.relative_rotvec(state, side)
            rows[side] = {
                "tracked": bool(e and e.get("q") is not None),
                "calibrated": side in self.neutral,
                "rotvec_deg": None if rv is None else [round(math.degrees(v), 1) for v in rv],
            }
        return rows


class QuatSmoother:
    """Per-hand low-pass on the incoming wrist quaternion (nlerp toward each new
    sample) to tame Quest hand-tracking jitter -- worst during pronation, when the
    hand turns edge-on to the cameras. alpha in (0,1]: 1.0 = no smoothing, lower =
    smoother (and laggier). Resets a side when its hand drops tracking. Note: this
    reduces JITTER only, not the deterministic flex/deviation coupling of a roll."""

    def __init__(self, alpha: float = 0.4):
        self.alpha = max(1e-3, min(1.0, float(alpha)))
        self._sm: dict[str, np.ndarray] = {}

    def apply(self, state: dict) -> dict:
        out = {"t": state.get("t", 0.0)}
        for side in ("left", "right"):
            e = state.get(side)
            if not e or e.get("q") is None:
                self._sm.pop(side, None)
                out[side] = e
                continue
            q = np.asarray(e["q"], dtype=float)
            n = float(np.linalg.norm(q))
            q = q / n if n > 1e-9 else q
            prev = self._sm.get(side)
            if prev is None:
                sm = q
            else:
                if float(np.dot(prev, q)) < 0.0:
                    q = -q                         # keep both on the same hemisphere
                sm = prev + self.alpha * (q - prev)
                m = float(np.linalg.norm(sm))
                sm = sm / m if m > 1e-9 else q
            self._sm[side] = sm
            out[side] = {"q": sm.tolist(), "p": e.get("p")}
        return out


# --------------------------------------------------------------------------- #
# Arm controller: home for j1..j3, VR-mapped wrist for j4..j6
# --------------------------------------------------------------------------- #
class VRArmController:
    """Drop-in for sim_yam_bimanual.ArmController. j1..j3 (+ unmapped joints) hold
    ``home``; j4/j5/j6 follow the calibrated wrist orientation."""

    def __init__(self, arm_joints, home: dict | None = None, mapper: WristMapper | None = None):
        self.home = {sj: 0.0 for sj in arm_joints}
        if home:
            self.home.update(home)
        self.mapper = mapper or WristMapper()
        self._state: dict = {"left": None, "right": None}

    def update(self, state: dict) -> None:
        self._state = state or {"left": None, "right": None}

    def calibrate(self) -> list[str]:
        return self.mapper.calibrate(self._state)

    def calibrate_missing(self) -> list[str]:
        """Capture a neutral for any tracked side that isn't calibrated yet (so a
        hand missing at the initial calibration auto-joins when it reappears)."""
        done = []
        for s in ("left", "right"):
            if not self.mapper.is_calibrated(s):
                done += self.mapper.calibrate(self._state, side=s)
        return done

    def fully_calibrated(self) -> bool:
        return self.mapper.fully_calibrated()

    def status(self) -> dict:
        return self.mapper.debug_rows(self._state)

    def targets(self, t: float | None = None) -> dict[tuple[str, str], float]:
        tg = dict(self.home)
        tg.update(self.mapper.targets(self._state))
        return tg


# --------------------------------------------------------------------------- #
# TLS + WebSocket receiver (runs in a background thread)
# --------------------------------------------------------------------------- #
def ensure_cert(ip: str | None = None) -> tuple[str, str]:
    """Self-signed cert+key in vr/certs (generated once via openssl)."""
    os.makedirs(CERT_DIR, exist_ok=True)
    cert, key = os.path.join(CERT_DIR, "cert.pem"), os.path.join(CERT_DIR, "key.pem")
    if os.path.exists(cert) and os.path.exists(key):
        return cert, key
    san = "subjectAltName=DNS:localhost,IP:127.0.0.1" + (f",IP:{ip}" if ip else "")
    subprocess.run(
        ["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
         "-keyout", key, "-out", cert, "-days", "825",
         "-subj", "/CN=yam-teleop", "-addext", san],
        check=True, capture_output=True)
    return cert, key


class VRReceiver:
    """Serves vr_client.html + a wss endpoint; stores the latest wrist pose.

    Thread-safe ``latest()`` returns ``{'left': {'q':[x,y,z,w],'p':[..]} | None,
    'right': {...} | None, 't': <recv time>}``.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 8443, ip: str | None = None):
        self.host, self.port, self.ip = host, port, ip
        self._lock = threading.Lock()
        self._state = {"left": None, "right": None, "t": 0.0}
        self._html = open(HTML_PATH, "rb").read()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self.msgs = 0                       # frames received (for Hz / liveness)

    # ----- public, thread-safe -----
    def latest(self) -> dict:
        with self._lock:
            return dict(self._state)

    def has_hands(self) -> bool:
        s = self.latest()
        return bool(s.get("left") or s.get("right"))

    def start(self) -> None:
        cert, key = ensure_cert(self.ip)
        self._ssl = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        self._ssl.load_cert_chain(cert, key)
        self._thread = threading.Thread(target=lambda: asyncio.run(self._serve()), daemon=True)
        self._thread.start()

    # ----- server internals -----
    def _process_request(self, connection, request):
        if request.path.rstrip("/") == "/ws":
            return None                                   # -> WebSocket handshake
        headers = Headers({"Content-Type": "text/html; charset=utf-8",
                           "Content-Length": str(len(self._html))})
        return Response(200, "OK", headers, self._html)   # serve the page

    async def _handler(self, connection):
        async for raw in connection:
            try:
                m = json.loads(raw)
            except (ValueError, TypeError):
                continue
            with self._lock:
                self._state = {"left": m.get("l"), "right": m.get("r"), "t": time.time()}
                self.msgs += 1

    async def _serve(self):
        self._loop = asyncio.get_running_loop()
        async with serve(self._handler, self.host, self.port,
                         ssl=self._ssl, process_request=self._process_request):
            await asyncio.Future()        # run until the daemon thread dies


def lan_ip() -> str | None:
    for dev in ("en0", "en1"):
        try:
            out = subprocess.run(["ipconfig", "getifaddr", dev],
                                 capture_output=True, text=True).stdout.strip()
            if out:
                return out
        except Exception:
            pass
    return None


# --------------------------------------------------------------------------- #
# Self-test: mapping math + full TLS WebSocket round-trip (no headset needed)
# --------------------------------------------------------------------------- #
def _q_about(axis: str, theta: float):
    s, c = math.sin(theta / 2), math.cos(theta / 2)
    return {"x": [s, 0, 0, c], "y": [0, s, 0, c], "z": [0, 0, s, c]}[axis]


def _test_mapper() -> bool:
    m = WristMapper()
    m.calibrate({"left": {"q": [0, 0, 0, 1]}, "right": {"q": [0, 0, 0, 1]}})
    ok = True
    # CORRECT frame (W3C WebXR wrist-local): X=flex->j5, Y=deviation->j4, Z=ROLL->j6.
    # Each single-axis motion must drive exactly its mapped joint, no leak.
    for axis, joint in (("z", "arm_j6"), ("x", "arm_j5"), ("y", "arm_j4")):
        t = m.targets({"left": {"q": _q_about(axis, 0.3)}, "right": None})
        got = t.get(("left", joint), 0.0)
        others = max(abs(v) for k, v in t.items() if k != ("left", joint)) if len(t) > 1 else 0.0
        good = abs(got - 0.3) < 1e-3 and others < 1e-3
        print(f"  mapper {axis}->{joint}: {got:+.3f} rad (others {others:.1e})  {'ok' if good else 'BAD'}")
        ok &= good
    # a big forearm roll about local Z must reach j6 (+-pi joint), no gimbal blow-up/leak
    t = m.targets({"left": {"q": _q_about("z", 2.6)}, "right": None})
    roll_ok = abs(t[("left", "arm_j6")] - 2.6) < 1e-3 and max(abs(t[("left", "arm_j5")]), abs(t[("left", "arm_j4")])) < 1e-3
    print(f"  mapper big-roll(Z): j6={t[('left','arm_j6')]:+.3f} (want +2.600), no leak  {'ok' if roll_ok else 'BAD'}")
    # CRITICAL non-tautological test: a NON-identity neutral must still route a
    # BODY-LOCAL roll to j6 only (an identity neutral can't catch a wrong-axis MAP).
    m2 = WristMapper()
    q0 = _q_about("x", 0.7)                                   # arbitrary tilted neutral
    m2.calibrate({"left": {"q": q0}, "right": None})
    q_cur = q_mul(np.asarray(q0, float), np.asarray(_q_about("z", 0.6), float))   # local roll
    t = m2.targets({"left": {"q": q_cur}, "right": None})
    leak = max(abs(t[("left", "arm_j5")]), abs(t[("left", "arm_j4")]))
    tilt_ok = abs(t[("left", "arm_j6")] - 0.6) < 1e-3 and leak < 1e-3
    print(f"  mapper tilted-neutral roll: j6={t[('left','arm_j6')]:+.3f} (want +0.600), leak={leak:.1e}  {'ok' if tilt_ok else 'BAD'}")
    # clamping: a 2.5 rad flex about X clamps j5 to its +/-pi/2 range
    t = m.targets({"left": {"q": _q_about("x", 2.5)}, "right": None})
    clamped = abs(t[("left", "arm_j5")] - JOINT_RANGE["arm_j5"][1]) < 1e-6
    print(f"  mapper clamp: j5={t[('left','arm_j5')]:+.3f} == range max  {'ok' if clamped else 'BAD'}")
    return ok and roll_ok and tilt_ok and clamped


def _test_roundtrip() -> bool:
    from websockets.asyncio.client import connect
    port = 8459
    rx = VRReceiver(host="127.0.0.1", port=port, ip=None)
    rx.start()
    time.sleep(0.8)                                  # let the server bind

    async def go():
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        # 1) the page is served over https on the same port
        async with connect(f"wss://127.0.0.1:{port}/ws", ssl=ctx) as ws:
            await ws.send(json.dumps({"l": {"q": [0, 0, 0, 1], "p": [0, 0, 0]}, "r": None}))
            await asyncio.sleep(0.2)
    asyncio.run(go())
    s = rx.latest()
    ok = s["left"] is not None and s["left"]["q"] == [0, 0, 0, 1] and rx.msgs >= 1
    print(f"  roundtrip: received={rx.msgs} left={s['left']}  {'ok' if ok else 'BAD'}")
    return ok


def selftest() -> int:
    print("VR mapping math:")
    a = _test_mapper()
    print("VR wss round-trip:")
    b = _test_roundtrip()
    print("VR SELFTEST:", "PASS" if (a and b) else "FAIL")
    return 0 if (a and b) else 1


if __name__ == "__main__":
    import sys
    sys.exit(selftest())
