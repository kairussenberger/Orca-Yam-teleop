"""Solve each pinch pose in sim: Jacobian IK that drives the thumb + one finger
until their fingertip bodies meet. Saves per-finger joint configs into calib.json
under "pinch" (degrees). Headless — no camera, no hardware."""
import json
import os

import numpy as np
import mujoco
import orca_sim

import sim_common as sc  # shared sim-name -> ORCA-joint parsing
import webcam_teleop as wt

THUMB_J = ["thumb_cmc", "thumb_abd", "thumb_mcp", "thumb_dip"]
FINGER_J = lambda f: [f"{f}_abd", f"{f}_mcp", f"{f}_pip"]


def orca_name(sim_joint):
    return sc.parse_sim_name(sim_joint)[1]


env = orca_sim.OrcaHandRight(render_mode=None)
m, data = env.model, env.data

jname = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)]
idx = {orca_name(j): i for i, j in enumerate(jname)}          # ORCA joint -> qpos/dof index (hinge: 1:1)

bn = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(m.nbody)]
tip = {"thumb": next(i for i, b in enumerate(bn) if "T-DP" in b)}
ftips = [i for i, b in enumerate(bn) if "FingerTipAssembly" in b]   # order: index, middle, ring, pinky
for finger, bid in zip(["index", "middle", "ring", "pinky"], ftips):
    tip[finger] = bid


def solve(finger, n_iter=2000, lr0=4.0):
    allowed = [idx[j] for j in THUMB_J + FINGER_J(finger)]
    mujoco.mj_resetData(m, data)
    # seed a rough pinch: curl the finger, oppose the thumb toward it
    seed = {f"{finger}_mcp": 55, f"{finger}_pip": 70,
            "thumb_cmc": 10, "thumb_abd": 45, "thumb_mcp": 40, "thumb_dip": 20}
    for j, deg in seed.items():
        k = idx[j]; lo, hi = m.jnt_range[k]
        data.qpos[k] = np.clip(np.radians(deg), lo, hi)
    jt = np.zeros((3, m.nv)); jf = np.zeros((3, m.nv)); jr = np.zeros((3, m.nv))
    best = (1e9, data.qpos.copy())
    for it in range(n_iter):
        lr = 0.3 + lr0 * (1 - it / n_iter)        # decay to escape oscillation
        mujoco.mj_forward(m, data)
        d = data.xpos[tip["thumb"]] - data.xpos[tip[finger]]
        dist = float(np.linalg.norm(d))
        if dist < best[0]:
            best = (dist, data.qpos.copy())
        if dist < 0.004:
            break
        mujoco.mj_jacBody(m, data, jt, jr, tip["thumb"])
        mujoco.mj_jacBody(m, data, jf, jr, tip[finger])
        g = (jt - jf).T @ d                       # grad of 0.5|d|^2 wrt q
        for k in allowed:
            lo, hi = m.jnt_range[k]
            data.qpos[k] = np.clip(data.qpos[k] - lr * g[k], lo, hi)
    data.qpos[:] = best[1]
    mujoco.mj_forward(m, data)
    gap = float(np.linalg.norm(data.xpos[tip["thumb"]] - data.xpos[tip[finger]]))
    pose = {j: float(np.degrees(data.qpos[idx[j]])) for j in THUMB_J + FINGER_J(finger)}
    return pose, gap


pinch = {}
for f in ["index", "middle", "ring", "pinky"]:
    pose, gap = solve(f)
    pinch[f] = pose
    print(f"{f:6} pinch: tip gap {gap*1000:5.1f} mm", flush=True)
env.close()

path = wt.CALIB_PATH_DEFAULT
calib = json.load(open(path)) if os.path.exists(path) else {}
calib["pinch"] = pinch
json.dump(calib, open(path, "w"), indent=2)
print(f"saved pinch poses -> {path}", flush=True)
