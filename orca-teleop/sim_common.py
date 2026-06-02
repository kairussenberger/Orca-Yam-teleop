"""Shared helpers for the MuJoCo sim teleop drivers (sim_teleop, sim_bimanual,
solve_pinches).

Two things used to be copy-pasted (and drift) across those scripts:

  1. The sim-name -> ORCA-joint mapping. The sim names joints like
     ``right_i-mcp`` / actuators like ``right_i-mcp_actuator``; ORCA calls the
     same joint ``index_mcp``. That translation lived in three places.
  2. The actuator -> qpos index. Two selftests assumed actuator index == qpos
     index, which is NOT guaranteed (qpos is ordered by joint, ctrl by
     actuator). ``ActuatorMap`` derives it from the model via
     ``actuator_trnid`` + ``jnt_qposadr`` instead, the same way orca_sim's
     task envs do.

Plus ``RealtimeStepper``: the live loops are gated by the camera/detector, but
one ``env.step()`` only advances ``frame_skip * timestep`` (= 0.01s) of sim
time, so a single step per ~30-50ms frame ran the hand at ~0.2-0.3x speed.
The stepper steps as many times as wall-clock elapsed, restoring real-time.
"""
from __future__ import annotations

import time

import mujoco

# sim finger letter -> ORCA finger name
_FINGER = {"p": "pinky", "r": "ring", "m": "middle", "i": "index", "t": "thumb"}


def parse_sim_name(name: str) -> tuple[str, str]:
    """Map a sim actuator OR joint name to ``(side, joint)``.

    Hand joints: ``'right_i-mcp_actuator'`` / ``'right_i-mcp'`` ->
    ``('right', 'index_mcp')``. The sim's thumb ``t-pip`` is ORCA's ``thumb_dip``.

    Non-finger joints pass through unchanged: ``'right_wrist'`` ->
    ``('right', 'wrist')`` and YAM arm joints ``'left_arm_j1_actuator'`` ->
    ``('left', 'arm_j1')``. Arm joints carry no ``-`` so they are distinguished
    from finger joints and returned verbatim (lets ``ActuatorMap`` cover the
    bimanual-YAM model without special-casing each arm joint).
    """
    side = "left" if name.startswith("left") else "right"
    core = name.replace("left_", "").replace("right_", "").replace("_actuator", "")
    if "-" not in core:          # wrist + arm joints (e.g. 'arm_j1'): no remap
        return side, core
    f, j = core.split("-")
    finger = _FINGER[f]
    if finger == "thumb" and j == "pip":
        j = "dip"
    return side, f"{finger}_{j}"


class ActuatorMap:
    """Per-model actuator <-> ORCA-joint mapping, in actuator order.

    Attributes (all length ``model.nu``, aligned with ``ctrl``/action order):
        sides       -- 'left'/'right' per actuator
        joints      -- ORCA joint name per actuator (e.g. 'index_mcp')
        qpos_index  -- qpos address of each actuator's driven joint
    """

    def __init__(self, model) -> None:
        self.sides: list[str] = []
        self.joints: list[str] = []
        self.qpos_index: list[int] = []
        for a in range(model.nu):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
            side, joint = parse_sim_name(name)
            joint_id = int(model.actuator_trnid[a, 0])
            self.sides.append(side)
            self.joints.append(joint)
            self.qpos_index.append(int(model.jnt_qposadr[joint_id]))

    def __len__(self) -> int:
        return len(self.joints)

    @property
    def side_joints(self) -> list[tuple[str, str]]:
        """``[(side, joint), ...]`` in actuator order."""
        return list(zip(self.sides, self.joints))

    def qpos_of(self, joint: str, side: str | None = None) -> int:
        """qpos address of ``joint`` (optionally on a given side)."""
        for i, (s, j) in enumerate(zip(self.sides, self.joints)):
            if j == joint and (side is None or s == side):
                return self.qpos_index[i]
        raise KeyError((side, joint))


class RealtimeStepper:
    """Step a sim env to keep pace with wall-clock time.

    One ``env.step()`` advances ``frame_skip * timestep`` of sim time. Calling
    ``step()`` once per (slow) camera frame therefore ran the sim in slow
    motion. This accumulates real elapsed time and steps as many times as is
    needed to stay real-time, capped at ``max_substeps`` so a stalled loop
    can't trigger a step spiral.
    """

    def __init__(self, env, max_substeps: int = 20) -> None:
        self.dt = float(env.model.opt.timestep) * env.frame_skip
        self.max_substeps = max_substeps
        self._t_prev: float | None = None
        self._acc = 0.0

    def step(self, env, action) -> int:
        """Advance the sim to match real time. Returns the number of steps taken."""
        now = time.perf_counter()
        if self._t_prev is None:        # first call: one step, start the clock
            self._t_prev = now
            env.step(action)
            return 1
        self._acc += now - self._t_prev
        self._t_prev = now
        n = int(self._acc / self.dt)
        if n <= 0:                      # loop faster than one sim step; wait
            return 0
        if n > self.max_substeps:       # behind (loop stalled): cap, drop backlog
            n = self.max_substeps
            self._acc = 0.0
        else:
            self._acc -= n * self.dt
        for _ in range(n):
            env.step(action)
        return n
