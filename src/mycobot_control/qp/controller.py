"""One-step QP differential IK: the next joint command from the current one.

Each :meth:`QPController.step` solves, for the joint increment ``dq`` (deg,
over one tick of ``dt``)::

    minimise   |S (J dq - dx_des)|^2                          track the task
             + damping^2 * sum_i w_i dq_i^2                    Levenberg damping
             + smooth_weight * sum_i w_i (dq_i - dq_prev_i)^2   no sudden changes
             + posture_weight * |q + dq - q_rest|^2             (optional)
    subject to q + dq inside the joint table minus a margin, and able to stop there
               |dq|           <= qd_max  * dt                    joint speed
               |dq - dq_prev| <= qdd_max * dt^2                  joint acceleration
               |J_v dq| <= v_max * dt, |J_w dq| <= w_max * dt     TCP speed (per base axis)
               z + J_z dq >= z_min + (1 - floor_gain)(z - z_min)       optional floor

``w_i`` are the per-joint weights: a large ``w_1`` spares J1, whose backlash
makes small oscillations on it unrepeatable (20 on J1 gave repeatable small
circles on this lab's arm). The acceleration bound is what makes a streamed
command free of velocity jumps no matter how the target moves.

Why a QP rather than damped least squares: the limits are constraints, not
penalties, so the command never asks the arm for more than it may do, and
when a limit binds the remaining joints still do as much of the task as they
can.

Units: joints in degrees, positions mm, angles in the config in degrees.
``dq_prev`` is the previous **per-tick increment in degrees** (not deg/s).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from mycobot_control.qp.kinematics import DEG, Kinematics, orientation_error, so3_exp
from mycobot_control.qp.solver import OPTIMAL, solve_qp
from mycobot_control.safety.limits import JOINT_LIMIT_MARGIN_DEG, JOINT_LIMITS_DEG, N_JOINTS

Num = Union[float, Sequence[float]]


@dataclass
class ControllerConfig:
    """Tuning and limits for :class:`QPController`. Degrees, mm, seconds.

    Attributes:
        dt: control period (s). The arm is streamed one command per tick.
        qd_max_deg_s: joint speed limit, scalar or per joint.
            TODO(unverified): what pymycobot ``speed`` this needs; the spec
            sheet's ~120 deg/s at speed 100 has not been measured here.
        qdd_max_deg_s2: joint acceleration limit, scalar or per joint.
        v_max_mm_s: TCP linear speed limit (per base axis as a hard bound; the
            desired motion is clipped by norm). ``None`` disables.
        w_max_deg_s: TCP angular speed limit, likewise. ``None`` disables.
        k_pos, k_rot: task error gains (1/s): a static error shrinks by
            ``k * dt`` of itself per tick, on top of the feed-forward.
        k_joint: the same for joint references.
        damping: Levenberg damping, in mm of task error per degree of joint
            motion. Keeps steps finite near singularities.
        smooth_weight: weight on ``(dq - dq_prev)^2``, (mm/deg)^2 like the
            damping term. Low-passes the command; does not bias steady tracking.
        joint_weights: per-joint multipliers on damping and smoothness. Raise
            J1's to spare it (e.g. ``(20, 1, 1, 1, 1, 1)``).
        orientation_weight: mm of position error that cost the same as 1 rad
            of orientation error. 60 means 1 deg ~ 1 mm.
        position_only: ignore the target orientation (weight 0). The three
            spare DOF are then settled by damping, joint weights and posture.
        posture_weight: weight pulling ``q`` toward ``q_rest_deg`` (per deg^2).
            Off unless ``q_rest_deg`` is set.
        q_rest_deg: rest posture for the posture term.
        z_min_mm: optional floor plane for the TCP (base-frame z), linearised.
            TODO(unverified): only the TCP point is kept above it, not the
            links or the gripper body.
        floor_gain: fraction of the remaining clearance the TCP may close per
            tick (velocity damper); smaller brakes earlier.
        joint_limit_margin_deg: commands stay this far inside the joint table.
            One degree more than the session's own refusal margin, so a solver
            rounding error can never produce a command the session refuses.
    """

    dt: float = 0.05
    qd_max_deg_s: Num = 60.0
    qdd_max_deg_s2: Num = 300.0
    v_max_mm_s: Optional[float] = 150.0
    w_max_deg_s: Optional[float] = 90.0
    k_pos: float = 10.0
    k_rot: float = 10.0
    k_joint: float = 10.0
    damping: float = 0.1
    smooth_weight: float = 0.05
    joint_weights: Num = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    orientation_weight: float = 60.0
    position_only: bool = False
    posture_weight: float = 1e-3
    q_rest_deg: Optional[Sequence[float]] = None
    z_min_mm: Optional[float] = None
    floor_gain: float = 0.1
    joint_limit_margin_deg: float = JOINT_LIMIT_MARGIN_DEG + 1.0
    joint_limits_deg: Sequence[Tuple[float, float]] = field(default=JOINT_LIMITS_DEG)

    def __post_init__(self) -> None:
        if not self.dt > 0.0:
            raise ValueError(f"dt must be positive, got {self.dt}")
        for name in ("qd_max_deg_s", "qdd_max_deg_s2"):
            if np.any(self.vec(getattr(self, name)) <= 0.0):
                raise ValueError(f"{name} must be positive")
        if np.any(self.vec(self.joint_weights) <= 0.0):
            raise ValueError("joint_weights must be positive")
        if self.damping < 0 or self.smooth_weight < 0 or self.posture_weight < 0:
            raise ValueError("weights must be non-negative")
        if not 0.0 < self.floor_gain <= 1.0:
            raise ValueError("floor_gain must be in (0, 1]")

    @staticmethod
    def vec(v: Num) -> np.ndarray:
        a = np.asarray(v, dtype=float).reshape(-1)
        if a.size == 1:
            return np.full(N_JOINTS, float(a[0]))
        if a.size != N_JOINTS:
            raise ValueError(f"need a scalar or {N_JOINTS} values, got {a.size}")
        return a

    def limits_lo_hi(self) -> Tuple[np.ndarray, np.ndarray]:
        lim = np.asarray(self.joint_limits_deg, dtype=float)
        return lim[:, 0] + self.joint_limit_margin_deg, lim[:, 1] - self.joint_limit_margin_deg


@dataclass
class PoseTarget:
    """TCP pose to reach by the end of this tick, with optional feed-forward.

    ``twist``: ``[vx, vy, vz]`` mm/s and ``[wx, wy, wz]`` **deg/s**, base
    frame: the reference's velocity over this tick. The pose one tick earlier
    is taken as ``T`` minus one tick of ``twist``.
    """

    T: np.ndarray
    twist: Optional[Sequence[float]] = None


@dataclass
class JointTarget:
    """Joint angles (deg) to reach by the end of this tick, optional velocity (deg/s)."""

    q_deg: Sequence[float]
    qd_deg_s: Optional[Sequence[float]] = None


Target = Union[PoseTarget, JointTarget]


def as_target(target) -> Target:
    """Accept a PoseTarget/JointTarget, a 4x4 pose, or 6 joint angles (deg)."""
    if isinstance(target, (PoseTarget, JointTarget)):
        return target
    arr = np.asarray(target, dtype=float)
    if arr.shape == (4, 4):
        return PoseTarget(arr)
    if arr.shape == (N_JOINTS,):
        return JointTarget(arr)
    raise ValueError(f"target must be a 4x4 pose or {N_JOINTS} joint angles, got shape {arr.shape}")


@dataclass
class StepInfo:
    """What one :meth:`QPController.step` did.

    ``status``: ``"optimal"``; ``"relaxed"`` (infeasible with the TCP speed
    rows, solved without them); ``"braking"`` (still infeasible - typically the
    floor: the command decelerates as hard as allowed and the caller should
    stop). ``pos_err_mm`` / ``rot_err_deg`` are measured *after* the step
    (pose targets); ``joint_err_deg`` is the worst joint error after the step
    (joint targets). ``active`` names the binding constraints, e.g.
    ``"J1 vel+"``, ``"J3 pos-"``, ``"J2 acc+"``, ``"vz-"``, ``"floor"``.
    ``accel_relaxed`` lists joints (0-based) whose acceleration bound had to
    give way to a position/speed bound (abnormal: a reset with speed, or a
    start outside the margin).
    """

    status: str
    mode: str
    pos_err_mm: float = 0.0
    rot_err_deg: float = 0.0
    joint_err_deg: float = 0.0
    active: List[str] = field(default_factory=list)
    accel_relaxed: List[int] = field(default_factory=list)
    dq_deg: List[float] = field(default_factory=list)
    tcp_z_mm: float = 0.0
    iterations: int = 0
    solve_time_s: float = 0.0


def _stop_bound(room: np.ndarray, amax: np.ndarray) -> np.ndarray:
    """Largest per-tick step toward a limit ``room`` away that can still stop before it.

    With deceleration ``amax`` per tick^2, a step ``v`` followed by braking
    covers at most ``v + v^2 / (2 amax)``. Solving ``= room`` gives the bound.
    The set is invariant: from any state satisfying it, braking by ``amax``
    satisfies it again next tick, so the joint limits and the acceleration
    limit never contradict each other in normal running.
    """
    out = -amax + np.sqrt(amax * amax + 2.0 * amax * np.maximum(room, 0.0))
    # Already past the limit (a start outside the margin): head back, gently.
    return np.where(room >= 0.0, out, np.maximum(room, -amax))


class QPController:
    """Turns a target into the next joint command, one tick at a time. Stateless.

    Args:
        kin: the arm's kinematics (tool included).
        cfg: limits and tuning.
    """

    def __init__(self, kin: Optional[Kinematics] = None, cfg: Optional[ControllerConfig] = None) -> None:
        self.kin = kin if kin is not None else Kinematics()
        self.cfg = cfg if cfg is not None else ControllerConfig()

    def step(self, q_cmd_deg: Sequence[float], dq_prev_deg: Sequence[float],
             target) -> Tuple[np.ndarray, StepInfo]:
        """The next joint command.

        Args:
            q_cmd_deg: the current *commanded* joint angles (deg).
            dq_prev_deg: the previous tick's increment (deg per tick).
            target: :class:`PoseTarget`, :class:`JointTarget`, a 4x4 pose or 6
                joint angles - where to be at the end of this tick.

        Returns:
            ``(q_next_deg, info)``. ``q_next_deg`` is always finite and inside
            the joint table minus ``joint_limit_margin_deg``.
        """
        cfg = self.cfg
        dt = cfg.dt
        tgt = as_target(target)
        q = np.asarray(q_cmd_deg, dtype=float).reshape(N_JOINTS)
        dqp = np.asarray(dq_prev_deg, dtype=float).reshape(N_JOINTS)
        if not (np.all(np.isfinite(q)) and np.all(np.isfinite(dqp))):
            raise ValueError("q_cmd and dq_prev must be finite")

        lo, hi = cfg.limits_lo_hi()
        vmax = cfg.vec(cfg.qd_max_deg_s) * dt
        amax = cfg.vec(cfg.qdd_max_deg_s2) * dt * dt
        w = cfg.vec(cfg.joint_weights)

        # -- per-joint bounds on dq: speed, stop-before-limit, acceleration ----
        hard_ub = np.minimum(vmax, _stop_bound(hi - q, amax))
        hard_lb = np.maximum(-vmax, -_stop_bound(q - lo, amax))
        acc_lb, acc_ub = dqp - amax, dqp + amax
        lb = np.maximum(hard_lb, acc_lb)
        ub = np.minimum(hard_ub, acc_ub)
        lb_src = np.where(acc_lb >= hard_lb, "acc", np.where(-vmax >= hard_lb, "vel", "pos"))
        ub_src = np.where(acc_ub <= hard_ub, "acc", np.where(vmax <= hard_ub, "vel", "pos"))
        relaxed = [i for i in range(N_JOINTS) if lb[i] > ub[i]]
        for i in relaxed:
            # Limits outrank smoothness: take the hard-feasible point nearest
            # to what the acceleration bound wanted.
            v = min(max(dqp[i], hard_lb[i]), hard_ub[i])
            lb[i] = ub[i] = v

        T, J = self.kin.fk_jacobian(q)
        Jd = J * DEG                                  # per degree
        p, R = T[:3, 3], T[:3, :3]

        # -- objective ----------------------------------------------------------
        if isinstance(tgt, PoseTarget):
            mode = "pose"
            Tt = np.asarray(tgt.T, dtype=float).reshape(4, 4)
            tw = np.zeros(6) if tgt.twist is None else np.asarray(tgt.twist, dtype=float).reshape(6)
            v_ff = tw[:3]
            w_ff = tw[3:] * DEG
            # Where the reference is now (one tick before T), and the error to it.
            p_now = Tt[:3, 3] - v_ff * dt
            R_now = so3_exp(-w_ff * dt) @ Tt[:3, :3]
            d_p = v_ff * dt + cfg.k_pos * dt * (p_now - p)
            d_r = w_ff * dt + cfg.k_rot * dt * orientation_error(R_now, R)
            if cfg.v_max_mm_s is not None:
                d_p = _clip_norm(d_p, cfg.v_max_mm_s * dt)
            if cfg.w_max_deg_s is not None:
                d_r = _clip_norm(d_r, cfg.w_max_deg_s * DEG * dt)
            wr = 0.0 if cfg.position_only else cfg.orientation_weight
            if wr > 0.0:
                A = np.vstack([Jd[:3], wr * Jd[3:]])
                bvec = np.concatenate([d_p, wr * d_r])
            else:
                A, bvec = Jd[:3], d_p
            Hm = A.T @ A + np.diag((cfg.damping ** 2) * w)
            gv = -(A.T @ bvec)
        else:
            mode = "joint"
            qt = np.asarray(tgt.q_deg, dtype=float).reshape(N_JOINTS)
            qd_ff = np.zeros(N_JOINTS) if tgt.qd_deg_s is None else np.asarray(tgt.qd_deg_s, float).reshape(N_JOINTS)
            dq_des = qd_ff * dt + cfg.k_joint * dt * (qt - qd_ff * dt - q)
            Hm = np.eye(N_JOINTS)
            gv = -dq_des
        Hm = Hm + np.diag(cfg.smooth_weight * w)
        gv = gv - cfg.smooth_weight * w * dqp
        if cfg.q_rest_deg is not None and cfg.posture_weight > 0.0:
            rest = np.asarray(cfg.q_rest_deg, dtype=float).reshape(N_JOINTS)
            Hm = Hm + cfg.posture_weight * np.eye(N_JOINTS)
            gv = gv - cfg.posture_weight * (rest - q)
        Hm = Hm + 1e-9 * np.eye(N_JOINTS)  # strictly convex even with every weight at 0

        # -- general constraints C dq <= d -------------------------------------
        rows: List[np.ndarray] = []
        rhs: List[float] = []
        names: List[str] = []
        if cfg.v_max_mm_s is not None:
            lim = cfg.v_max_mm_s * dt
            for k, ax in enumerate("xyz"):
                rows += [Jd[k], -Jd[k]]
                rhs += [lim, lim]
                names += [f"v{ax}+", f"v{ax}-"]
        if cfg.w_max_deg_s is not None:
            lim = cfg.w_max_deg_s * DEG * dt
            for k, ax in enumerate("xyz"):
                rows += [Jd[3 + k], -Jd[3 + k]]
                rhs += [lim, lim]
                names += [f"w{ax}+", f"w{ax}-"]
        n_speed = len(rows)
        if cfg.z_min_mm is not None:
            # Velocity damper: the TCP may close floor_gain of the clearance per
            # tick, so it slows smoothly instead of hitting the plane at speed.
            rows.append(-Jd[2])
            rhs.append(cfg.floor_gain * (p[2] - cfg.z_min_mm))
            names.append("floor")

        t0 = time.perf_counter()
        status = OPTIMAL
        C = np.array(rows) if rows else None
        d = np.array(rhs) if rows else None
        res = solve_qp(Hm, gv, C, d, lb, ub, full_output=True)
        used_names = names
        if res.status != OPTIMAL and n_speed:
            # The TCP speed rows are a comfort limit; the joint bounds and the
            # floor are not. Drop the former first.
            C2 = C[n_speed:] if C is not None and C.shape[0] > n_speed else None
            d2 = d[n_speed:] if C2 is not None else None
            res = solve_qp(Hm, gv, C2, d2, lb, ub, full_output=True)
            used_names = names[n_speed:]
            status = "relaxed"
        if res.status == OPTIMAL:
            dq = res.x
        else:
            # Still infeasible: brake as hard as the bounds allow.
            dq = np.clip(np.zeros(N_JOINTS), lb, ub)
            status = "braking"
        solve_time = time.perf_counter() - t0

        q_next = np.clip(q + dq, lo, hi)
        dq = q_next - q

        m = len(used_names) if res.status == OPTIMAL else 0
        active: List[str] = []
        if res.status == OPTIMAL:
            for idx in res.active:
                if idx < m:
                    active.append(used_names[idx])
                elif idx < m + N_JOINTS:
                    j = idx - m
                    active.append(f"J{j + 1} {lb_src[j]}-")
                else:
                    j = idx - m - N_JOINTS
                    active.append(f"J{j + 1} {ub_src[j]}+")

        info = StepInfo(status=status, mode=mode, active=active, accel_relaxed=relaxed,
                        dq_deg=dq.tolist(), iterations=res.iterations, solve_time_s=solve_time)
        T_next = self.kin.fk(q_next)
        info.tcp_z_mm = float(T_next[2, 3])
        if mode == "pose":
            info.pos_err_mm = float(np.linalg.norm(Tt[:3, 3] - T_next[:3, 3]))
            info.rot_err_deg = float(np.linalg.norm(orientation_error(Tt[:3, :3], T_next[:3, :3]))) / DEG
        else:
            info.joint_err_deg = float(np.max(np.abs(qt - q_next)))
        return q_next, info


def _clip_norm(v: np.ndarray, limit: float) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n > limit > 0.0:
        return v * (limit / n)
    return v
