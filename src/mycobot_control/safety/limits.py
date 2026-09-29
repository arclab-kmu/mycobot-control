"""Limits and conservative defaults for the myCobot 280 JN.

Units are the SDK's own: joint angles in **degrees**, positions in **mm**,
orientations in **degrees**. Converting at the edge (recorder, sim bridge) is
cheaper than converting every call and comparing the result with vendor
numbers that are all in degrees.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

N_JOINTS = 6

# ---------------------------------------------------------------------------
# Joint position limits, degrees - SOURCE CONSTANTS
# ---------------------------------------------------------------------------
# Two sources agree joint for joint:
#   * pymycobot 4.0.7 robot_info.py, "MyCobot280" angles_min / angles_max - the
#     table the SDK's own calibration_parameters() checks send_angles() against
#   * mycobot_ros2 mycobot_description/urdf/mycobot_280_jn/
#     mycobot_280_jn_adaptive_gripper.urdf (J1 +/-2.9321 rad ... J6 +/-3.1416)
# Two vendor files agreeing probably means one origin, not two measurements.
#
# TODO(unverified): the controller's own values. get_joint_min_angle(id) /
# get_joint_max_angle(id) read them; `mycobot-info --live --read-only` prints
# both side by side. Record them before relying on these near the edge.
JOINT_LIMITS_DEG: Tuple[Tuple[float, float], ...] = (
    (-168.0, 168.0),  # J1  joint2_to_joint1
    (-140.0, 140.0),  # J2  joint3_to_joint2
    (-150.0, 150.0),  # J3  joint4_to_joint3
    (-150.0, 150.0),  # J4  joint5_to_joint4
    (-155.0, 160.0),  # J5  joint6_to_joint5
    (-180.0, 180.0),  # J6  joint6output_to_joint6
)

#: Commands are refused this close to a limit. The servos are hobby-class and
#: the table above is a source constant, so do not command the last few degrees.
JOINT_LIMIT_MARGIN_DEG = 3.0

# ---------------------------------------------------------------------------
# Cartesian range, mm - SOURCE CONSTANTS, advisory only
# ---------------------------------------------------------------------------
# pymycobot robot_info.py "MyCobot280" coords_min / coords_max. These are what
# the SDK accepts, not what the arm can reach: a point inside this box can
# still come back as error 32 (no IK solution).
COORDS_MIN: Tuple[float, ...] = (-281.45, -281.45, -70.0, -180.0, -180.0, -180.0)
COORDS_MAX: Tuple[float, ...] = (281.45, 281.45, 412.67, 180.0, 180.0, 180.0)

# ---------------------------------------------------------------------------
# Speed
# ---------------------------------------------------------------------------
# pymycobot takes an integer speed 1-100 for send_angles / send_coords /
# set_gripper_value. What that means in deg/s is not documented.
# TODO(unverified): the only figure anywhere is the spec sheet's 120 deg/s,
# presumably at 100. Measure with `mycobot-record` before streaming.
SPEED_MIN = 1
SPEED_MAX = 100
#: Conservative start. The vendor's streaming slider uses 25.
DEFAULT_SPEED = 20
#: Above this, --live asks for --allow-fast. Not a vendor number - a choice.
FAST_SPEED = 50

# ---------------------------------------------------------------------------
# Per-keypress steps
# ---------------------------------------------------------------------------
DEFAULT_JOINT_STEP_DEG = 2.0
MAX_JOINT_STEP_DEG = 15.0
DEFAULT_LINEAR_STEP_MM = 5.0
MAX_LINEAR_STEP_MM = 30.0
DEFAULT_ANGULAR_STEP_DEG = 2.0
MAX_ANGULAR_STEP_DEG = 15.0

#: The largest joint-space jump a single command may make from the MEASURED
#: angles. Larger moves go through mycobot-move-joints, which interpolates.
#: Exceeding it is refused, not clamped: a jump this large usually means a
#: units mistake (rad passed as deg) or a stale reading.
MAX_COMMAND_DELTA_DEG = 20.0

# ---------------------------------------------------------------------------
# Gripper (adaptive gripper, gripper_type=1)
# ---------------------------------------------------------------------------
GRIPPER_TYPE_ADAPTIVE = 1
GRIPPER_MIN = 0
GRIPPER_MAX = 100
DEFAULT_GRIPPER_STEP = 10
DEFAULT_GRIPPER_SPEED = 50
# TODO(unverified): which end of 0..100 is "open". The vendor slider maps the
# URDF driver joint's [-0.74, 0.15] rad linearly onto 0..100, which does not
# say which way the jaws move. Check with one small step before trusting the
# key labels in teleop/keymap.py.


class LimitViolation(ValueError):
    """A requested motion is outside a limit this package enforces."""


def check_speed(speed: int) -> int:
    value = int(speed)
    if not SPEED_MIN <= value <= SPEED_MAX:
        raise LimitViolation(f"speed {speed} outside [{SPEED_MIN}, {SPEED_MAX}]")
    return value


def check_joint_step(step_deg: float) -> None:
    if not 0.0 < step_deg <= MAX_JOINT_STEP_DEG:
        raise LimitViolation(f"joint step {step_deg} deg outside (0, {MAX_JOINT_STEP_DEG}]")


def check_cartesian_step(linear_mm: float, angular_deg: float) -> None:
    if not 0.0 < linear_mm <= MAX_LINEAR_STEP_MM:
        raise LimitViolation(f"linear step {linear_mm} mm outside (0, {MAX_LINEAR_STEP_MM}]")
    if not 0.0 < angular_deg <= MAX_ANGULAR_STEP_DEG:
        raise LimitViolation(f"angular step {angular_deg} deg outside (0, {MAX_ANGULAR_STEP_DEG}]")


def joint_vector(values: Sequence[float]) -> List[float]:
    """Six finite floats, or a ValueError before anything reaches the arm."""
    out = [float(v) for v in values]
    if len(out) != N_JOINTS:
        raise ValueError(f"need {N_JOINTS} joint angles, got {len(out)}")
    if not all(math.isfinite(v) for v in out):
        raise ValueError(f"joint angles must be finite, got {out}")
    return out


def joint_limit_margin(angles_deg: Sequence[float]) -> List[float]:
    """Per joint, degrees to the nearer limit. Negative means outside."""
    q = joint_vector(angles_deg)
    return [min(a - lo, hi - a) for a, (lo, hi) in zip(q, JOINT_LIMITS_DEG)]


def check_joint_target(angles_deg: Sequence[float], margin_deg: float = JOINT_LIMIT_MARGIN_DEG) -> List[float]:
    """Return the target unchanged, or raise if any joint is within ``margin_deg`` of a limit."""
    q = joint_vector(angles_deg)
    bad = [
        f"J{i + 1}={a:+.1f} (limit {lo:+.0f}..{hi:+.0f}, margin {margin_deg:.0f})"
        for i, (a, (lo, hi)) in enumerate(zip(q, JOINT_LIMITS_DEG))
        if not lo + margin_deg <= a <= hi - margin_deg
    ]
    if bad:
        raise LimitViolation("joint target outside limits: " + ", ".join(bad))
    return q


def clamp_joint_target(angles_deg: Sequence[float], margin_deg: float = JOINT_LIMIT_MARGIN_DEG) -> List[float]:
    """Clamp into the limits minus ``margin_deg``. For explicit moves only; teleop refuses instead."""
    q = joint_vector(angles_deg)
    return [min(max(a, lo + margin_deg), hi - margin_deg) for a, (lo, hi) in zip(q, JOINT_LIMITS_DEG)]


def check_command_delta(target_deg: Sequence[float], measured_deg: Sequence[float],
                        max_delta_deg: float = MAX_COMMAND_DELTA_DEG) -> float:
    """Raise if any joint would jump more than ``max_delta_deg``. Returns the largest delta."""
    t, m = joint_vector(target_deg), joint_vector(measured_deg)
    worst = max(abs(a - b) for a, b in zip(t, m))
    if worst > max_delta_deg:
        raise LimitViolation(
            f"joint target is {worst:.1f} deg from the measured angles (max {max_delta_deg:.1f}); "
            "use mycobot-move-joints for large moves"
        )
    return worst


def coords_in_range(coords: Sequence[float]) -> bool:
    """Advisory: inside the SDK's accepted coords box. Says nothing about reachability."""
    return all(lo <= float(c) <= hi for c, lo, hi in zip(coords, COORDS_MIN, COORDS_MAX))
