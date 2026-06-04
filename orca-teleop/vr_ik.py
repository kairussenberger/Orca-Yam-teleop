"""Per-arm 6-DOF damped-least-squares (DLS) inverse kinematics for the YAM arms.

Phase-2 full-pose VR teleop: given a target end-effector POSE (position + orientation)
in the robot world frame, solve for the 6 arm joint angles (arm_j1..arm_j6) that put
the hand-mount frame there. The end-effector body is ``{side}_hand_mount`` -- it sits
after the wrist-roll j6, so its pose is the full hand position+orientation.

Solver: iterate  dq = J^T (J J^T + lambda^2 I)^-1 e  on the 6-vector pose error
e = [pos_err(3); ori_err(3)], where J = [jacp; jacr] restricted to the arm's 6 DOFs.
Runs on a SCRATCH mjData so the live/rendered sim is never disturbed; warm-started
from the previous solution for smooth real-time tracking. Frame-agnostic and fully
testable headless (see selftest): the Quest->robot frame mapping lives elsewhere.

MuJoCo quaternions are [w,x,y,z]; we use mju_* quat ops to avoid order bugs.
"""
from __future__ import annotations

import numpy as np
import mujoco


class ArmIK:
    JOINTS = [f"arm_j{n}" for n in range(1, 7)]

    def __init__(self, model, side: str, damping: float = 0.1, max_dq: float = 0.4):
        self.model = model
        self.side = side
        self.damping = float(damping)
        self.max_dq = float(max_dq)                      # cap per-iter joint step (rad)
        self.ee = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_hand_mount")
        if self.ee < 0:
            raise KeyError(f"no body {side}_hand_mount")
        self.jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_{j}")
                     for j in self.JOINTS]
        self.qadr = [int(model.jnt_qposadr[j]) for j in self.jids]
        self.dadr = [int(model.jnt_dofadr[j]) for j in self.jids]
        self.lo = np.array([model.jnt_range[j][0] for j in self.jids])
        self.hi = np.array([model.jnt_range[j][1] for j in self.jids])
        self._d = mujoco.MjData(model)
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))

    def ee_pose(self, data):
        """Current (pos[3], quat[4] wxyz) of the end-effector from a live mjData."""
        return data.xpos[self.ee].copy(), data.xquat[self.ee].copy()

    def solve(self, q_seed, target_pos, target_quat, iters: int = 20,
              tol: float = 1e-4, step: float = 1.0):
        """Return ({side_arm_jN: rad}, residual). q_seed = full model qpos (warm start);
        only the arm's qpos are optimized. target_quat is [w,x,y,z]."""
        d, m = self._d, self.model
        d.qpos[:] = q_seed
        err = np.zeros(6)
        neg = np.zeros(4); dqt = np.zeros(4); erot = np.zeros(3)
        residual = 1e9
        for _ in range(iters):
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)                       # both cheaper than full mj_forward
            err[:3] = target_pos - d.xpos[self.ee]
            mujoco.mju_negQuat(neg, d.xquat[self.ee])
            mujoco.mju_mulQuat(dqt, target_quat, neg)    # R_err = R_tgt * R_cur^-1 (world)
            mujoco.mju_quat2Vel(erot, dqt, 1.0)
            err[3:] = erot
            residual = float(np.linalg.norm(err))
            if residual < tol:
                break
            mujoco.mj_jacBody(m, d, self._jacp, self._jacr, self.ee)
            J = np.vstack([self._jacp[:, self.dadr], self._jacr[:, self.dadr]])   # 6x6
            dq = J.T @ np.linalg.solve(J @ J.T + (self.damping ** 2) * np.eye(6), err)
            dq = np.clip(step * dq, -self.max_dq, self.max_dq)
            for i in range(6):
                d.qpos[self.qadr[i]] = np.clip(d.qpos[self.qadr[i]] + dq[i], self.lo[i], self.hi[i])
        return {f"{self.side}_{self.JOINTS[i]}": float(d.qpos[self.qadr[i]]) for i in range(6)}, residual


# --------------------------------------------------------------------------- #
# Headless self-test: prove the solver reaches reachable target poses
# --------------------------------------------------------------------------- #
def _apply(model, base_qpos, sol_by_jointname):
    d = mujoco.MjData(model)
    d.qpos[:] = base_qpos
    for jn, val in sol_by_jointname.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        d.qpos[int(model.jnt_qposadr[jid])] = val
    mujoco.mj_forward(model, d)
    return d


def selftest() -> int:
    from orca_sim import OrcaYamBimanual
    env = OrcaYamBimanual(render_mode=None)
    m, d = env.model, env.data
    mujoco.mj_forward(m, d)
    ok = True
    # reachable-by-construction target: perturb the arm joints, FK, and use that EE
    # pose as the target (guaranteed inside the reachable set for BOTH arms -- a
    # fixed world-frame offset is not, since the arms are 180deg-about-Z copies).
    perturb = [0.15, 0.12, -0.18, 0.20, 0.15, 0.25]
    for side in ("left", "right"):
        ik = ArmIK(m, side, damping=0.05)
        d2 = mujoco.MjData(m); d2.qpos[:] = d.qpos
        for i, qa in enumerate(ik.qadr):
            d2.qpos[qa] = np.clip(d.qpos[qa] + perturb[i], ik.lo[i], ik.hi[i])
        mujoco.mj_forward(m, d2)
        tp, tq = d2.xpos[ik.ee].copy(), d2.xquat[ik.ee].copy()
        sol, res = ik.solve(d.qpos.copy(), tp, tq, iters=300, tol=1e-5, step=1.0)
        d2 = _apply(m, d.qpos, sol)
        pe = float(np.linalg.norm(d2.xpos[ik.ee] - tp)) * 1000.0           # mm
        neg = np.zeros(4); de = np.zeros(4); ev = np.zeros(3)
        mujoco.mju_negQuat(neg, d2.xquat[ik.ee]); mujoco.mju_mulQuat(de, tq, neg)
        mujoco.mju_quat2Vel(ev, de, 1.0)
        oe = np.degrees(np.linalg.norm(ev))                                 # deg
        inlim = bool(np.all([ik.lo[i] - 1e-6 <= sol[f"{side}_{ik.JOINTS[i]}"] <= ik.hi[i] + 1e-6 for i in range(6)]))
        good = pe < 2.0 and oe < 1.0 and inlim
        print(f"  {side}: residual={res:.2e}  reached pos_err={pe:.2f}mm ori_err={oe:.2f}deg  in-limits={inlim}  {'ok' if good else 'BAD'}")
        ok &= good
    env.close()
    print("IK SELFTEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(selftest())
