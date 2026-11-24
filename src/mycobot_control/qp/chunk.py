"""Chunk execution: turn motion chunks into a smooth per-tick joint command stream.

A *chunk* is a short trajectory - TCP poses or joint angles sampled in time -
from a VLM, a planner (cuRobo) or a goal. :class:`ChunkExecutor` holds the
**commanded** state ``(q_cmd, dq_prev)`` - never the measured one, which has
backlash in it - and on each :meth:`~ChunkExecutor.tick` asks the
:class:`~mycobot_control.qp.controller.QPController` for the next command.

Replacing or extending a chunk mid-motion never stops the arm and never jumps
its velocity: the new chunk is blended in from the current commanded position
*and velocity* with a quintic that fades the offset out over a short window,
and the controller's acceleration bound caps whatever is left. That is what
turns a series of chunks into one continuous motion.

Time convention: sample 0 of a chunk is "now" (the arm's current reference),
sample k is ``k * dt`` later. A chunk whose sample 0 is not where the arm is
gets blended. With ``lookahead_ticks = L`` the controller runs L ticks ahead
of the arm: the command sent is the reference L ticks in the future, so the
firmware - which plans each ``send_angles`` target to a stop - always has
somewhere further to go and does not brake between ticks.
TODO(unverified): that lookahead removes the stop-go on this arm's firmware,
and which L (with which ``speed``) is best.

Pure: no I/O, no sleeping. The caller owns the clock and calls ``tick()``
every ``cfg.dt`` seconds.
"""

from __future__ import annotations

import dataclasses
import math
from collections import deque
from typing import Deque, List, Optional, Sequence, Tuple, Union

import numpy as np

from mycobot_control.qp.controller import ControllerConfig, JointTarget, PoseTarget, QPController, StepInfo
from mycobot_control.qp.kinematics import DEG, orientation_error, so3_exp, so3_log
from mycobot_control.safety.limits import N_JOINTS

#: Peak velocity of a min-jerk move, in units of distance / duration.
MIN_JERK_PEAK_V = 1.875
#: Peak acceleration of a min-jerk move, in units of distance / duration^2.
MIN_JERK_PEAK_A = 10.0 / math.sqrt(3.0)
#: Most that goal_pose's automatic duration stretches a path for the joint limits.
MAX_TIME_SCALE = 10.0


class TrackingError(RuntimeError):
    """The measured joints are too far from the commanded ones; the caller must stop."""


def min_jerk(tau, derivative: int = 0):
    """Minimum-jerk profile ``s(tau) = 10 tau^3 - 15 tau^4 + 6 tau^5`` on ``tau`` in [0, 1].

    ``tau`` is clipped to [0, 1] (scalar or array). ``derivative`` 1 or 2
    returns ds/dtau or d2s/dtau2 (divide by the duration, or its square, for
    time derivatives). Starts and ends at rest with zero acceleration.
    """
    t = np.clip(np.asarray(tau, dtype=float), 0.0, 1.0)
    if derivative == 0:
        out = t ** 3 * (10.0 - 15.0 * t + 6.0 * t * t)
    elif derivative == 1:
        out = 30.0 * t * t * (1.0 - t) ** 2
    elif derivative == 2:
        out = 60.0 * t * (1.0 - t) * (1.0 - 2.0 * t)
    else:
        raise ValueError("derivative must be 0, 1 or 2")
    return float(out) if np.ndim(out) == 0 else out


def _fade(tau: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Quintic Hermite bases on [0, 1]: H0 carries a start offset, H1 a start velocity.

    H0(0)=1, H1'(0)=1; both are 0 with zero first and second derivatives at 1,
    and have zero second derivative at 0.
    """
    t = np.clip(tau, 0.0, 1.0)
    h0 = 1.0 - t ** 3 * (10.0 - 15.0 * t + 6.0 * t * t)
    h1 = t - 6.0 * t ** 3 + 8.0 * t ** 4 - 3.0 * t ** 5
    return h0, h1


def _blend_time(offset: float, vel_offset: float, vmax: float, amax: float, tmin: float) -> float:
    """Long enough that fading the offsets needs at most half the speed limit and the accel limit."""
    t = tmin
    if vmax > 0:
        t = max(t, MIN_JERK_PEAK_V * offset / (0.5 * vmax))
    if amax > 0:
        t = max(t, math.sqrt(MIN_JERK_PEAK_A * offset / (0.5 * amax)), 4.0 * vel_offset / amax)
    return t


def _hermite_resample(X: np.ndarray, dt_in: float, dt_out: float) -> np.ndarray:
    """Cubic Hermite (Catmull-Rom tangents) resampling of rows of X, sample 0 kept at t=0."""
    N = X.shape[0]
    if N == 1 or abs(dt_in - dt_out) < 1e-12:
        return X.copy()
    T_end = (N - 1) * dt_in
    M = max(1, int(round(T_end / dt_out)))
    ts = np.linspace(0.0, T_end, M + 1)
    tang = np.empty_like(X)
    tang[1:-1] = (X[2:] - X[:-2]) / (2 * dt_in)
    tang[0] = (X[1] - X[0]) / dt_in
    tang[-1] = (X[-1] - X[-2]) / dt_in
    idx = np.minimum((ts / dt_in).astype(int), N - 2)
    u = (ts - idx * dt_in) / dt_in
    u = u[:, None]
    h00 = 2 * u ** 3 - 3 * u ** 2 + 1
    h10 = u ** 3 - 2 * u ** 2 + u
    h01 = -2 * u ** 3 + 3 * u ** 2
    h11 = u ** 3 - u ** 2
    return h00 * X[idx] + h10 * dt_in * tang[idx] + h01 * X[idx + 1] + h11 * dt_in * tang[idx + 1]


def _resample_rotations(Rs: np.ndarray, dt_in: float, dt_out: float) -> np.ndarray:
    """Piecewise-geodesic (slerp) resampling of rotation matrices."""
    N = Rs.shape[0]
    if N == 1 or abs(dt_in - dt_out) < 1e-12:
        return Rs.copy()
    T_end = (N - 1) * dt_in
    M = max(1, int(round(T_end / dt_out)))
    out = np.empty((M + 1, 3, 3))
    for j, t in enumerate(np.linspace(0.0, T_end, M + 1)):
        i = min(int(t / dt_in), N - 2)
        u = (t - i * dt_in) / dt_in
        out[j] = so3_exp(u * so3_log(Rs[i + 1] @ Rs[i].T)) @ Rs[i]
    return out


class _JointRef:
    kind = "joint"

    def __init__(self, Q: np.ndarray, QD: np.ndarray) -> None:
        self.Q, self.QD = Q, QD       # (L, 6) deg; deg/s over each tick

    def __len__(self) -> int:
        return self.Q.shape[0]

    def target(self, j: int, hold: bool = False) -> JointTarget:
        return JointTarget(self.Q[j], None if hold else self.QD[j])

    def append(self, other: "_JointRef") -> None:
        self.Q = np.vstack([self.Q, other.Q])
        self.QD = np.vstack([self.QD, other.QD])

    def drop_before(self, j: int) -> None:
        self.Q, self.QD = self.Q[j:], self.QD[j:]


class _PoseRef:
    kind = "pose"

    def __init__(self, P: np.ndarray, R: np.ndarray, V: np.ndarray, W: np.ndarray) -> None:
        self.P, self.R, self.V, self.W = P, R, V, W   # mm, rot, mm/s, rad/s

    def __len__(self) -> int:
        return self.P.shape[0]

    def target(self, j: int, hold: bool = False) -> PoseTarget:
        T = np.eye(4)
        T[:3, :3] = self.R[j]
        T[:3, 3] = self.P[j]
        if hold:
            return PoseTarget(T)
        return PoseTarget(T, np.concatenate([self.V[j], self.W[j] / DEG]))

    def append(self, other: "_PoseRef") -> None:
        self.P = np.vstack([self.P, other.P])
        self.R = np.concatenate([self.R, other.R])
        self.V = np.vstack([self.V, other.V])
        self.W = np.vstack([self.W, other.W])

    def drop_before(self, j: int) -> None:
        self.P, self.R, self.V, self.W = self.P[j:], self.R[j:], self.V[j:], self.W[j:]


Ref = Union[_JointRef, _PoseRef]


class ChunkExecutor:
    """Executes joint / Cartesian chunks as one continuous, limit-respecting stream.

    Args:
        ctrl: the QP controller (its ``cfg.dt`` is the tick).
        q0_deg: where the arm is (measured) when execution starts.
        lookahead_ticks: run the controller this many ticks ahead of the arm
            (see module doc). 0 sends exactly the next tick's reference.
        max_track_err_deg: if a measured reading passed to :meth:`tick` is
            further than this from where the arm should be, the executor
            faults. The arm server refuses commands 20 deg from the measured
            angles; this trips first.
        raise_on_track_error: raise :class:`TrackingError` on a fault (else
            only set :attr:`fault` and hold the command).
        blend_s: shortest window over which a chunk offset is faded in.
        done_vel_deg_s: "at rest" threshold for :attr:`done`.
    """

    def __init__(self, ctrl: QPController, q0_deg: Sequence[float], *, lookahead_ticks: int = 0,
                 max_track_err_deg: float = 12.0, raise_on_track_error: bool = True,
                 blend_s: float = 0.3, done_vel_deg_s: float = 0.5) -> None:
        if lookahead_ticks < 0:
            raise ValueError("lookahead_ticks must be >= 0")
        self.ctrl = ctrl
        self.kin = ctrl.kin
        self.cfg: ControllerConfig = ctrl.cfg
        self.lookahead_ticks = int(lookahead_ticks)
        self.max_track_err_deg = float(max_track_err_deg)
        self.raise_on_track_error = bool(raise_on_track_error)
        self.blend_s = float(blend_s)
        self.done_vel_deg_s = float(done_vel_deg_s)
        self.reset(q0_deg)

    # -- state -------------------------------------------------------------
    def reset(self, q_deg: Sequence[float]) -> None:
        """Forget all motion: commanded = ``q_deg``, at rest, no chunk, no fault."""
        q = np.asarray(q_deg, dtype=float).reshape(N_JOINTS)
        if not np.all(np.isfinite(q)):
            raise ValueError("q must be finite")
        self._hist: Deque[np.ndarray] = deque([q.copy()])
        self._dq = np.zeros(N_JOINTS)
        self._ref: Optional[Ref] = None
        self._j = 0
        self._hold = JointTarget(q.copy())
        self.fault: Optional[str] = None
        self.ticks = 0
        self.last_info: Optional[StepInfo] = None
        self.last_track_err_deg: Optional[float] = None

    @property
    def q_cmd(self) -> np.ndarray:
        """The last command sent (the controller's state; L ticks ahead of the arm)."""
        return self._hist[-1].copy()

    @property
    def dq_prev(self) -> np.ndarray:
        """The commanded per-tick increment (deg/tick)."""
        return self._dq.copy()

    @property
    def qd_cmd_deg_s(self) -> np.ndarray:
        return self._dq / self.cfg.dt

    @property
    def q_expected(self) -> np.ndarray:
        """Where the arm should be now: the command from ``lookahead_ticks`` ticks ago."""
        return self._hist[0].copy()

    @property
    def remaining_ticks(self) -> int:
        return 0 if self._ref is None else max(0, len(self._ref) - self._j)

    @property
    def done(self) -> bool:
        """The chunk is used up and the commanded motion (incl. the lookahead pipeline) is at rest."""
        if self.remaining_ticks > 0:
            return False
        eps = self.done_vel_deg_s * self.cfg.dt
        if np.max(np.abs(self._dq)) > eps:
            return False
        return bool(np.max(np.abs(self._hist[-1] - self._hist[0])) <= eps * max(1, len(self._hist) - 1))

    # -- anchors -----------------------------------------------------------
    def _lead_time_offset_ticks(self) -> int:
        # The controller is len(hist)-1 ticks ahead of the arm's "now".
        return len(self._hist) - 1

    def _anchor_joint(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._hist[-1].copy(), self._dq / self.cfg.dt

    def _anchor_pose(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        T, J = self.kin.fk_jacobian(self._hist[-1])
        tw = J @ (self._dq * DEG) / self.cfg.dt
        return T[:3, 3].copy(), T[:3, :3].copy(), tw[:3], tw[3:]

    # -- building references ------------------------------------------------
    def _joint_ref(self, Q: np.ndarray, q_a: np.ndarray, qd_a: np.ndarray, skip: int) -> _JointRef:
        dt = self.cfg.dt
        if skip:
            Q = Q[min(skip, Q.shape[0] - 1):]
        M = Q.shape[0] - 1
        qd_new0 = (Q[1] - Q[0]) / dt if M >= 1 else np.zeros(N_JOINTS)
        p0 = q_a - Q[0]
        v0 = qd_a - qd_new0
        vmax = self.cfg.vec(self.cfg.qd_max_deg_s)
        amax = self.cfg.vec(self.cfg.qdd_max_deg_s2)
        Tb = max(_blend_time(abs(p0[i]), abs(v0[i]), vmax[i], amax[i], self.blend_s) for i in range(N_JOINTS))
        L = max(M, int(math.ceil(Tb / dt - 1e-9)), 1)
        j = np.arange(L + 1)
        h0, h1 = _fade(j * dt / Tb)
        Z = Q[np.minimum(j, M)] + h0[:, None] * p0 + (Tb * h1)[:, None] * v0
        return _JointRef(Z[1:], np.diff(Z, axis=0) / dt)

    def _pose_ref(self, P: np.ndarray, Rs: np.ndarray, p_a: np.ndarray, R_a: np.ndarray,
                  v_a: np.ndarray, w_a: np.ndarray, skip: int) -> _PoseRef:
        dt = self.cfg.dt
        if skip:
            k = min(skip, P.shape[0] - 1)
            P, Rs = P[k:], Rs[k:]
        M = P.shape[0] - 1
        if M >= 1:
            v_new0 = (P[1] - P[0]) / dt
            w_new0 = so3_log(Rs[1] @ Rs[0].T) / dt
        else:
            v_new0 = w_new0 = np.zeros(3)
        p0 = p_a - P[0]
        r0 = orientation_error(R_a, Rs[0])
        v0 = v_a - v_new0
        w0 = w_a - w_new0
        cfg = self.cfg
        # Cartesian blend timing only; the joint limits in the QP are what bind.
        vlin = cfg.v_max_mm_s if cfg.v_max_mm_s is not None else 150.0
        vang = (cfg.w_max_deg_s if cfg.w_max_deg_s is not None else 90.0) * DEG
        Tb = max(_blend_time(float(np.linalg.norm(p0)), float(np.linalg.norm(v0)), vlin, 6.0 * vlin, self.blend_s),
                 _blend_time(float(np.linalg.norm(r0)), float(np.linalg.norm(w0)), vang, 6.0 * vang, self.blend_s))
        L = max(M, int(math.ceil(Tb / dt - 1e-9)), 1)
        j = np.arange(L + 1)
        h0, h1 = _fade(j * dt / Tb)
        src = np.minimum(j, M)
        Zp = P[src] + h0[:, None] * p0 + (Tb * h1)[:, None] * v0
        ZR = np.empty((L + 1, 3, 3))
        for i in range(L + 1):
            ZR[i] = so3_exp(h0[i] * r0 + Tb * h1[i] * w0) @ Rs[src[i]]
        V = np.diff(Zp, axis=0) / dt
        W = np.array([so3_log(ZR[i + 1] @ ZR[i].T) / dt for i in range(L)])
        return _PoseRef(Zp[1:], ZR[1:], V, W)

    def _install(self, ref: Ref) -> None:
        self._ref = ref
        self._j = 0

    @staticmethod
    def _poses(poses) -> Tuple[np.ndarray, np.ndarray]:
        A = np.asarray(poses, dtype=float)
        if A.ndim == 2 and A.shape == (4, 4):
            A = A[None]
        if A.ndim != 3 or A.shape[1:] != (4, 4) or A.shape[0] < 1:
            raise ValueError(f"poses must be a list of 4x4 matrices, got shape {A.shape}")
        if not np.all(np.isfinite(A)):
            raise ValueError("poses must be finite")
        return A[:, :3, 3].copy(), A[:, :3, :3].copy()

    @staticmethod
    def _joints(Q) -> np.ndarray:
        A = np.asarray(Q, dtype=float)
        if A.ndim == 1 and A.shape == (N_JOINTS,):
            A = A[None]
        if A.ndim != 2 or A.shape[1] != N_JOINTS or A.shape[0] < 1:
            raise ValueError(f"joint chunk must be N x {N_JOINTS} degrees, got shape {A.shape}")
        if not np.all(np.isfinite(A)):
            raise ValueError("joint chunk must be finite")
        return A

    # -- chunk API -----------------------------------------------------------
    def set_cartesian_chunk(self, poses, dt: Optional[float] = None) -> int:
        """Replace the current motion with TCP ``poses`` (4x4, mm), sample 0 = now.

        ``dt`` is the chunk's sample period (default ``cfg.dt``); other periods
        are resampled. Blends from the current commanded pose and velocity.
        Returns the number of ticks queued.
        """
        P, Rs = self._poses(poses)
        dt_in = self.cfg.dt if dt is None else float(dt)
        P = _hermite_resample(P, dt_in, self.cfg.dt)
        Rs = _resample_rotations(Rs, dt_in, self.cfg.dt)
        p_a, R_a, v_a, w_a = self._anchor_pose()
        self._install(self._pose_ref(P, Rs, p_a, R_a, v_a, w_a, self._lead_time_offset_ticks()))
        return self.remaining_ticks

    def set_joint_chunk(self, Q, dt: Optional[float] = None) -> int:
        """Replace the current motion with joint samples ``Q`` (N x 6 deg), sample 0 = now.

        ``dt`` is the chunk's sample period (default ``cfg.dt``); resampled to
        ``cfg.dt`` with cubic Hermite interpolation. Returns ticks queued.
        """
        Q = self._joints(Q)
        dt_in = self.cfg.dt if dt is None else float(dt)
        Q = _hermite_resample(Q, dt_in, self.cfg.dt)
        q_a, qd_a = self._anchor_joint()
        self._install(self._joint_ref(Q, q_a, qd_a, self._lead_time_offset_ticks()))
        return self.remaining_ticks

    def extend(self, chunk, dt: Optional[float] = None) -> int:
        """Append a chunk (N x 4 x 4 poses or N x 6 joints) after the queued one.

        Sample 0 of ``chunk`` is the time of the last queued sample, so a chunk
        that starts where the queue ends joins seamlessly; one that does not is
        blended from the queue's end state. With nothing queued this is
        ``set_*_chunk``. The kind must match the queued chunk's.
        Returns the total ticks queued.
        """
        arr = np.asarray(chunk, dtype=float)
        is_pose = arr.ndim >= 2 and arr.shape[-2:] == (4, 4)
        if self.remaining_ticks == 0:
            return self.set_cartesian_chunk(arr, dt) if is_pose else self.set_joint_chunk(arr, dt)
        ref = self._ref
        assert ref is not None
        dt_in = self.cfg.dt if dt is None else float(dt)
        if is_pose != (ref.kind == "pose"):
            raise ValueError(f"cannot extend a {ref.kind} chunk with a {'pose' if is_pose else 'joint'} chunk; "
                             "use set_cartesian_chunk / set_joint_chunk to switch")
        # Drop what has been executed so the arrays do not grow without bound.
        ref.drop_before(self._j)
        self._j = 0
        if isinstance(ref, _PoseRef):
            P, Rs = self._poses(arr)
            P = _hermite_resample(P, dt_in, self.cfg.dt)
            Rs = _resample_rotations(Rs, dt_in, self.cfg.dt)
            ref.append(self._pose_ref(P, Rs, ref.P[-1], ref.R[-1], ref.V[-1], ref.W[-1], 0))
        else:
            Q = _hermite_resample(self._joints(arr), dt_in, self.cfg.dt)
            ref.append(self._joint_ref(Q, ref.Q[-1], ref.QD[-1], 0))
        return self.remaining_ticks

    def goal_pose(self, T_goal, duration: Optional[float] = None) -> int:
        """Min-jerk straight line (position) and geodesic (orientation) to ``T_goal``.

        Starts from the current commanded pose; the current velocity is blended
        out, so calling this mid-motion does not stop the arm first.
        ``duration`` defaults to the shortest at which the min-jerk path needs
        at most half of ``v_max`` / ``w_max`` and, per joint, half of
        ``qd_max`` / ``qdd_max`` (at least 0.5 s). The joint part comes from
        tracking the path once with the limits lifted, which costs one QP
        solve per tick of the path.
        """
        Tg = np.asarray(T_goal, dtype=float).reshape(4, 4)
        p_a, R_a, v_a, w_a = self._anchor_pose()
        dist = float(np.linalg.norm(Tg[:3, 3] - p_a))
        rot = orientation_error(Tg[:3, :3], R_a)
        ang = float(np.linalg.norm(rot))

        def path(duration_s: float) -> Tuple[np.ndarray, np.ndarray]:
            n = max(1, int(round(float(duration_s) / self.cfg.dt)))
            s = min_jerk(np.arange(n + 1) / n)
            P = p_a + s[:, None] * (Tg[:3, 3] - p_a)
            Rs = np.array([so3_exp(si * rot) @ R_a for si in s])
            Rs[-1] = Tg[:3, :3]
            return P, Rs

        if duration is None:
            vlin = self.cfg.v_max_mm_s or 150.0
            vang = (self.cfg.w_max_deg_s or 90.0) * DEG
            duration = max(0.5, MIN_JERK_PEAK_V * dist / (0.5 * vlin), MIN_JERK_PEAK_V * ang / (0.5 * vang))
            duration *= self._joint_time_scale(*path(duration))
        P, Rs = path(duration)
        self._install(self._pose_ref(P, Rs, p_a, R_a, v_a, w_a, 0))
        return self.remaining_ticks

    def _joint_time_scale(self, P: np.ndarray, Rs: np.ndarray) -> float:
        """How much to stretch a Cartesian path so its joint motion stays at half the joint limits.

        The probe runs with the limits raised 5x, not removed: an unreachable
        or singular path then saturates instead of whipping the joints, and the
        stretch is capped at ``MAX_TIME_SCALE``.
        """
        cfg = self.cfg
        free = dataclasses.replace(cfg, qd_max_deg_s=5.0 * cfg.vec(cfg.qd_max_deg_s),
                                   qdd_max_deg_s2=5.0 * cfg.vec(cfg.qdd_max_deg_s2),
                                   v_max_mm_s=None, w_max_deg_s=None)
        ctrl = QPController(self.kin, free)
        q = self._hist[-1].copy()
        dq = np.zeros(N_JOINTS)
        dt = cfg.dt
        qs = [q]
        for i in range(1, P.shape[0]):
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = Rs[i], P[i]
            twist = np.concatenate([(P[i] - P[i - 1]) / dt, so3_log(Rs[i] @ Rs[i - 1].T) / dt / DEG])
            q_next, _ = ctrl.step(q, dq, PoseTarget(T, twist))
            dq, q = q_next - q, q_next
            qs.append(q)
        Q = np.array(qs)
        if Q.shape[0] < 3:
            return 1.0
        v_pk = np.max(np.abs(np.diff(Q, axis=0)), axis=0) / dt
        a_pk = np.max(np.abs(np.diff(Q, 2, axis=0)), axis=0) / dt ** 2
        vmax = cfg.vec(cfg.qd_max_deg_s)
        amax = cfg.vec(cfg.qdd_max_deg_s2)
        scale = max(1.0, float(np.max(v_pk / (0.5 * vmax))), float(np.sqrt(np.max(a_pk / (0.5 * amax)))))
        return min(scale, MAX_TIME_SCALE)

    def goal_joints(self, q_goal, duration: Optional[float] = None) -> int:
        """Min-jerk joint-space move to ``q_goal`` (deg) from the commanded state.

        ``duration`` defaults to one where every joint's min-jerk peak speed is
        at most half its ``qd_max`` and peak acceleration half its ``qdd_max``
        (at least 0.5 s).
        """
        qg = np.asarray(q_goal, dtype=float).reshape(N_JOINTS)
        q_a, qd_a = self._anchor_joint()
        if duration is None:
            dq = np.abs(qg - q_a)
            vmax = self.cfg.vec(self.cfg.qd_max_deg_s)
            amax = self.cfg.vec(self.cfg.qdd_max_deg_s2)
            duration = max(0.5, float(np.max(MIN_JERK_PEAK_V * dq / (0.5 * vmax))),
                           float(np.max(np.sqrt(MIN_JERK_PEAK_A * dq / (0.5 * amax)))))
        n = max(1, int(round(float(duration) / self.cfg.dt)))
        s = min_jerk(np.arange(n + 1) / n)
        Q = q_a + s[:, None] * (qg - q_a)
        self._install(self._joint_ref(Q, q_a, qd_a, 0))
        return self.remaining_ticks

    def stop(self) -> None:
        """Drop the chunk and come to rest smoothly at about the accel-limited braking point, then hold.

        The hold target is each joint's braking distance ahead, so the
        controller decelerates within ``qdd_max`` rather than reversing.
        """
        amax = self.cfg.vec(self.cfg.qdd_max_deg_s2) * self.cfg.dt ** 2
        dq = self._dq
        brake = dq * np.abs(dq) / (2.0 * amax) + 0.5 * dq
        self._ref = None
        self._j = 0
        self._hold = JointTarget(self._hist[-1] + brake)

    # -- the tick ------------------------------------------------------------
    def tick(self, q_measured_deg: Optional[Sequence[float]] = None) -> List[float]:
        """Advance one ``cfg.dt`` and return the joint command (deg) to send now.

        If ``q_measured_deg`` is given, it is compared with :attr:`q_expected`
        first; beyond ``max_track_err_deg`` the executor faults (raises
        :class:`TrackingError` unless ``raise_on_track_error=False``, in which
        case it returns the previous command unchanged). A faulted executor
        stays faulted until :meth:`reset`.
        """
        if self.fault is not None:
            if self.raise_on_track_error:
                raise TrackingError(self.fault)
            return self._hist[-1].tolist()
        if q_measured_deg is not None:
            qm = np.asarray(q_measured_deg, dtype=float).reshape(N_JOINTS)
            err = np.abs(qm - self._hist[0])
            worst = int(np.argmax(err))
            self.last_track_err_deg = float(err[worst])
            if err[worst] > self.max_track_err_deg:
                self.fault = (f"J{worst + 1} is {err[worst]:.1f} deg from its commanded angle "
                              f"(limit {self.max_track_err_deg:.1f}); stopping")
                if self.raise_on_track_error:
                    raise TrackingError(self.fault)
                return self._hist[-1].tolist()
        # Fill the lookahead pipeline on the first tick(s), then one step per tick.
        steps = 1 + max(0, self.lookahead_ticks + 1 - len(self._hist))
        for _ in range(steps):
            self._advance()
        while len(self._hist) > self.lookahead_ticks + 1:
            self._hist.popleft()
        self.ticks += 1
        return self._hist[-1].tolist()

    def _advance(self) -> None:
        ref = self._ref
        if ref is not None and self._j < len(ref):
            target = ref.target(self._j)
            self._j += 1
            if self._j == len(ref):
                self._hold = ref.target(len(ref) - 1, hold=True)
        else:
            target = self._hold
        q = self._hist[-1]
        q_next, info = self.ctrl.step(q, self._dq, target)
        self._dq = q_next - q
        self._hist.append(q_next)
        self.last_info = info

    def plan(self, max_ticks: int = 100000) -> np.ndarray:
        """Run :meth:`tick` until :attr:`done` (no measurements) and return the commands (K x 6).

        Mutates the executor; use a fresh one to execute afterwards.
        """
        out = []
        for _ in range(max_ticks):
            out.append(self.tick())
            if self.done:
                break
        return np.array(out).reshape(-1, N_JOINTS)
