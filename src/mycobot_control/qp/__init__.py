"""QP differential IK and chunk streaming for the myCobot 280.

``Kinematics`` (FK, geometric Jacobian) -> ``QPController`` (one tick: the next
joint command under joint position / speed / acceleration, TCP speed and floor
constraints) -> ``ChunkExecutor`` (pose or joint chunks, goals, blending, the
per-tick command stream).

Needs numpy (the ``qp`` extra); nothing else. This package never imports
pymycobot, ROS or the ``sdk`` session modules, so a PC without the arm SDK can
import it to plan and stream. ``qp.cli`` (``mycobot-qp``) is the only module
here that talks to an arm, and it imports the session lazily.
"""

from mycobot_control.qp.chunk import ChunkExecutor, TrackingError, min_jerk
from mycobot_control.qp.controller import ControllerConfig, JointTarget, PoseTarget, QPController, StepInfo
from mycobot_control.qp.kinematics import (
    DEG,
    DH_PARAMS,
    Kinematics,
    orientation_error,
    pose_error,
    pose_from_xyzrpy,
    rot_to_rpy,
    rpy_to_rot,
    so3_exp,
    so3_log,
    xyzrpy_from_pose,
)
from mycobot_control.qp.solver import QPResult, solve_qp

__all__ = [
    "ChunkExecutor",
    "ControllerConfig",
    "DEG",
    "DH_PARAMS",
    "JointTarget",
    "Kinematics",
    "PoseTarget",
    "QPController",
    "QPResult",
    "StepInfo",
    "TrackingError",
    "min_jerk",
    "orientation_error",
    "pose_error",
    "pose_from_xyzrpy",
    "rot_to_rpy",
    "rpy_to_rot",
    "so3_exp",
    "so3_log",
    "solve_qp",
    "xyzrpy_from_pose",
]
