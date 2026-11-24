"""myCobot 280 forward kinematics and the analytic geometric Jacobian.

Units, the same as the SDK everywhere this package talks to the arm:

* joint angles in **degrees** (pymycobot units) at every public entry point;
* positions in **mm**;
* orientations as rotation matrices, or as firmware ``rx, ry, rz`` in degrees
  with ``R = Rz(rz) @ Ry(ry) @ Rx(rx)``.

The one exception is the Jacobian, whose columns are **per radian** of joint
motion (the geometric Jacobian is naturally per radian). Multiply by
``pi / 180`` (``DEG``) for per-degree columns, as the QP controller does.

Standard DH, rows ``(theta_offset rad, d mm, a mm, alpha rad)``. Measured on
this lab's arms and validated against firmware ``get_coords()`` (the parent
session's ``vlm/kin.py``); the zero pose gives ``(45.6, -63.4, 420.87)`` mm,
``rx, ry, rz = (-90, 0, -90)``.

Pure numpy: no SDK, no ROS, no session. The web console imports this on a PC
that has never seen pymycobot.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np

from mycobot_control.safety.limits import JOINT_LIMITS_DEG, N_JOINTS

DEG = math.pi / 180.0

#: (theta_offset rad, d mm, a mm, alpha rad) per joint, J1..J6.
DH_PARAMS: Tuple[Tuple[float, float, float, float], ...] = (
    (0.0, 139.42, 0.0, math.pi / 2),
    (-math.pi / 2, 0.0, -110.4, 0.0),
    (0.0, 0.0, -96.0, 0.0),
    (-math.pi / 2, 63.4, 0.0, math.pi / 2),
    (math.pi / 2, 75.05, 0.0, -math.pi / 2),
    (0.0, 45.6, 0.0, 0.0),
)


def _q6(q_deg: Sequence[float]) -> np.ndarray:
    q = np.asarray(q_deg, dtype=float).reshape(-1)
    if q.shape != (N_JOINTS,):
        raise ValueError(f"need {N_JOINTS} joint angles in degrees, got shape {np.shape(q_deg)}")
    if not np.all(np.isfinite(q)):
        raise ValueError(f"joint angles must be finite, got {q.tolist()}")
    return q


def dh_matrix(theta: float, d: float, a: float, alpha: float) -> np.ndarray:
    """Standard DH link transform A_i (theta, alpha in rad; d, a in mm)."""
    ct, st, ca, sa = math.cos(theta), math.sin(theta), math.cos(alpha), math.sin(alpha)
    return np.array([
        [ct, -st * ca, st * sa, a * ct],
        [st, ct * ca, -ct * sa, a * st],
        [0.0, sa, ca, d],
        [0.0, 0.0, 0.0, 1.0],
    ])


# -- rotations ---------------------------------------------------------------

def rpy_to_rot(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    """Firmware convention: ``R = Rz(rz) @ Ry(ry) @ Rx(rx)``, angles in degrees."""
    rx, ry, rz = rx_deg * DEG, ry_deg * DEG, rz_deg * DEG
    cx, sx, cy, sy, cz, sz = math.cos(rx), math.sin(rx), math.cos(ry), math.sin(ry), math.cos(rz), math.sin(rz)
    return np.array([
        [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
        [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
        [-sy, cy * sx, cy * cx],
    ])


def rot_to_rpy(R: np.ndarray) -> np.ndarray:
    """Inverse of :func:`rpy_to_rot`: ``[rx, ry, rz]`` degrees, ``ry`` in [-90, 90].

    At gimbal lock (``|ry| = 90``) everything goes into ``rz``, as in
    ``teleop.keymap.matrix_to_rpy``.
    """
    R = np.asarray(R, dtype=float)
    sy = max(-1.0, min(1.0, -R[2, 0]))
    ry = math.asin(sy)
    if abs(sy) < 1.0 - 1e-9:
        rx = math.atan2(R[2, 1], R[2, 2])
        rz = math.atan2(R[1, 0], R[0, 0])
    else:
        rx = 0.0
        rz = math.atan2(-R[0, 1], R[1, 1])
    return np.array([rx, ry, rz]) / DEG


def skew(w: Sequence[float]) -> np.ndarray:
    x, y, z = (float(v) for v in w)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def so3_exp(w: Sequence[float]) -> np.ndarray:
    """Rotation matrix for the axis-angle vector ``w`` (rad), Rodrigues."""
    w = np.asarray(w, dtype=float).reshape(3)
    th = float(np.linalg.norm(w))
    K = skew(w)
    if th < 1e-9:
        return np.eye(3) + K
    return np.eye(3) + (math.sin(th) / th) * K + ((1.0 - math.cos(th)) / (th * th)) * (K @ K)


def so3_log(R: np.ndarray) -> np.ndarray:
    """Axis-angle vector (rad) of ``R``; robust near 0 and near pi."""
    R = np.asarray(R, dtype=float)
    vee = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    c = max(-1.0, min(1.0, (float(np.trace(R)) - 1.0) / 2.0))
    th = math.acos(c)
    if th < 1e-6:
        return 0.5 * vee
    if th > math.pi - 1e-4:
        # sin(th) ~ 0: the antisymmetric part carries no axis. (R + R^T)/2 - cI
        # = (1 - c) a a^T; take its best-conditioned column.
        B = 0.5 * (R + R.T) - c * np.eye(3)
        i = int(np.argmax(np.diag(B)))
        a = B[:, i] / math.sqrt(max(B[i, i], 1e-300))
        a /= np.linalg.norm(a)
        if float(np.dot(a, vee)) < 0.0:
            a = -a
        return th * a
    return (th / (2.0 * math.sin(th))) * vee


def orientation_error(R_target: np.ndarray, R_current: np.ndarray) -> np.ndarray:
    """Axis-angle vector (rad, base frame) of the rotation taking ``R_current`` to ``R_target``.

    ``so3_exp(e) @ R_current == R_target``. For small errors this is the
    angular displacement the Jacobian's angular rows predict.
    """
    return so3_log(np.asarray(R_target, float) @ np.asarray(R_current, float).T)


# -- poses -------------------------------------------------------------------

def pose_from_xyzrpy(xyzrpy: Sequence[float]) -> np.ndarray:
    """4x4 from firmware-style ``[x, y, z, rx, ry, rz]`` (mm, deg)."""
    v = np.asarray(xyzrpy, dtype=float).reshape(-1)
    if v.shape != (6,):
        raise ValueError(f"need [x, y, z, rx, ry, rz], got shape {np.shape(xyzrpy)}")
    T = np.eye(4)
    T[:3, :3] = rpy_to_rot(v[3], v[4], v[5])
    T[:3, 3] = v[:3]
    return T


def xyzrpy_from_pose(T: np.ndarray) -> np.ndarray:
    """Firmware-style ``[x, y, z, rx, ry, rz]`` (mm, deg) of a 4x4."""
    T = np.asarray(T, dtype=float)
    return np.concatenate([T[:3, 3], rot_to_rpy(T[:3, :3])])


def pose_error(T_target: np.ndarray, T_current: np.ndarray) -> Tuple[float, float]:
    """(position error mm, orientation error deg) between two 4x4 poses."""
    T_target, T_current = np.asarray(T_target, float), np.asarray(T_current, float)
    dp = float(np.linalg.norm(T_target[:3, 3] - T_current[:3, 3]))
    dr = float(np.linalg.norm(orientation_error(T_target[:3, :3], T_current[:3, :3]))) / DEG
    return dp, dr


# -- the arm -----------------------------------------------------------------

class Kinematics:
    """FK and geometric Jacobian of the myCobot 280 (standard DH above).

    Args:
        tool: 4x4 flange-to-TCP transform in mm (identity = the flange, which
            is what firmware ``get_coords()`` reports).
        dh: override the DH table (rows of theta_offset rad, d, a, alpha rad).
    """

    def __init__(self, tool: Optional[np.ndarray] = None,
                 dh: Sequence[Sequence[float]] = DH_PARAMS) -> None:
        self.tool = np.eye(4) if tool is None else np.asarray(tool, dtype=float).reshape(4, 4)
        self.dh = np.asarray(dh, dtype=float).reshape(N_JOINTS, 4)
        #: Source joint table (deg) from ``safety/limits.py``; margins are the controller's job.
        self.joint_limits_deg = np.asarray(JOINT_LIMITS_DEG, dtype=float)

    def frames(self, q_deg: Sequence[float]) -> List[np.ndarray]:
        """``[T_0 = I, T_1, ..., T_6]``: base-to-link-frame transforms, without the tool."""
        q = _q6(q_deg) * DEG
        Ts = [np.eye(4)]
        T = Ts[0]
        for qi, (off, d, a, al) in zip(q, self.dh):
            T = T @ dh_matrix(qi + off, d, a, al)
            Ts.append(T)
        return Ts

    def fk(self, q_deg: Sequence[float]) -> np.ndarray:
        """Base-to-TCP 4x4 (mm) at joint angles ``q_deg`` (deg)."""
        return self.frames(q_deg)[-1] @ self.tool

    def fk_jacobian(self, q_deg: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
        """``(T, J)`` in one pass. See :meth:`jacobian` for J's units."""
        Ts = self.frames(q_deg)
        T = Ts[-1] @ self.tool
        p = T[:3, 3]
        J = np.empty((6, N_JOINTS))
        for i in range(N_JOINTS):
            # Standard DH: joint i+1 turns about z of frame i, through its origin.
            z = Ts[i][:3, 2]
            o = Ts[i][:3, 3]
            J[:3, i] = np.cross(z, p - o)
            J[3:, i] = z
        return T, J

    def jacobian(self, q_deg: Sequence[float]) -> np.ndarray:
        """Geometric Jacobian at the TCP, base frame, 6x6.

        Rows 0-2: TCP linear velocity in **mm per rad** of joint motion.
        Rows 3-5: angular velocity in **rad per rad**.
        Column j: joint j+1, **per radian** (``J @ dq_rad``). For per-degree
        columns use ``J * DEG``.
        """
        return self.fk_jacobian(q_deg)[1]

    def xyzrpy(self, q_deg: Sequence[float]) -> np.ndarray:
        """FK as firmware-style ``[x, y, z, rx, ry, rz]`` (mm, deg)."""
        return xyzrpy_from_pose(self.fk(q_deg))
