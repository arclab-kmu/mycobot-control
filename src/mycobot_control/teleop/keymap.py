"""Key -> motion mapping. Pure: no I/O, no SDK, testable exactly.

Two modes:

``joint``
    One key pair per joint, stacked vertically on the keyboard::

        w e r t y u   +J1 .. +J6
        s d f g h j   -J1 .. -J6

``cartesian``
    The nero-control layout, applied to the firmware's ``[x, y, z, rx, ry, rz]``
    (mm, deg); the firmware solves the IK in ``send_coords``::

        w / s  +X / -X     u / j  +rx / -rx
        a / d  +Y / -Y     i / k  +ry / -ry
        r / f  +Z / -Z     o / l  +rz / -rz

Gripper keys are shared: ``,`` / ``.`` step, ``<`` / ``>`` go to the ends.

TODO(unverified): the joint signs and the firmware base-frame axes have not
been watched on this arm. The first live check should be one key at a time, at
the smallest step.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from mycobot_control.safety.limits import GRIPPER_MAX, GRIPPER_MIN

AXIS_NAMES = ("x", "y", "z", "rx", "ry", "rz")


@dataclass(frozen=True)
class Binding:
    """One motion key. ``index`` is a joint (joint mode) or a coords axis (cartesian mode)."""

    index: int
    sign: float
    label: str

    @property
    def is_linear(self) -> bool:
        return self.index < 3


JOINT_BINDINGS: Dict[str, Binding] = {}
for _i, (_plus, _minus) in enumerate(zip("wertyu", "sdfghj")):
    JOINT_BINDINGS[_plus] = Binding(_i, +1.0, f"+J{_i + 1}")
    JOINT_BINDINGS[_minus] = Binding(_i, -1.0, f"-J{_i + 1}")

CARTESIAN_BINDINGS: Dict[str, Binding] = {
    "w": Binding(0, +1.0, "+X"),
    "s": Binding(0, -1.0, "-X"),
    "a": Binding(1, +1.0, "+Y"),
    "d": Binding(1, -1.0, "-Y"),
    "r": Binding(2, +1.0, "+Z"),
    "f": Binding(2, -1.0, "-Z"),
    "u": Binding(3, +1.0, "+rx"),
    "j": Binding(3, -1.0, "-rx"),
    "i": Binding(4, +1.0, "+ry"),
    "k": Binding(4, -1.0, "-ry"),
    "o": Binding(5, +1.0, "+rz"),
    "l": Binding(5, -1.0, "-rz"),
}

#: Gripper keys: (label, kind, amount). kind "step" adds amount x the gripper
#: step; kind "set" goes to the value.
#: TODO(unverified): that larger values open the adaptive gripper - see
#: safety/limits.py. The labels say "open"/"close" on that assumption.
GRIPPER_KEYS: Dict[str, Tuple[str, str, int]] = {
    ",": ("gripper close step", "step", -1),
    ".": ("gripper open step", "step", +1),
    "<": ("gripper fully closed", "set", GRIPPER_MIN),
    ">": ("gripper fully open", "set", GRIPPER_MAX),
}

#: Keys the teleop loop handles itself. Listed so no binding can shadow one.
CONTROL_KEYS: Dict[str, str] = {
    " ": "stop: close the gate and stop() the current motion",
    "[": "halve the step size",
    "]": "double the step size",
    "?": "show this help",
    "q": "quit",
    "\x1b": "quit (Esc)",
    "\x03": "quit (Ctrl-C)",
}

for _table in (JOINT_BINDINGS, CARTESIAN_BINDINGS):
    assert not (set(_table) & set(CONTROL_KEYS)), "motion key shadows a control key"
    assert not (set(_table) & set(GRIPPER_KEYS)), "motion key shadows a gripper key"
assert not (set(GRIPPER_KEYS) & set(CONTROL_KEYS)), "gripper key shadows a control key"


def bindings_for(mode: str) -> Dict[str, Binding]:
    if mode == "joint":
        return JOINT_BINDINGS
    if mode == "cartesian":
        return CARTESIAN_BINDINGS
    raise ValueError(f"mode must be 'joint' or 'cartesian', got {mode!r}")


def apply_joint_key(angles_deg: Sequence[float], binding: Binding, step_deg: float) -> List[float]:
    out = [float(a) for a in angles_deg]
    out[binding.index] += binding.sign * float(step_deg)
    return out


# -- rotations -------------------------------------------------------------
# Hand-written 3x3 math so the core needs no numpy (see pyproject.toml).
#
# TODO(unverified): the firmware's rx/ry/rz convention. This assumes fixed-axis
# XYZ, i.e. R = Rz(rz) Ry(ry) Rx(rx) - the usual robotics RPY, and what
# nero-control uses. For a single small step the choice mostly changes which
# axis a key turns about when the others are far from zero; composing as
# matrices (rather than adding to the angles) is what keeps a step a step
# near ry = +/-90, whichever convention the firmware uses.

Mat3 = List[List[float]]


def _rx(a: float) -> Mat3:
    c, s = math.cos(a), math.sin(a)
    return [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]]


def _ry(a: float) -> Mat3:
    c, s = math.cos(a), math.sin(a)
    return [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]


def _rz(a: float) -> Mat3:
    c, s = math.cos(a), math.sin(a)
    return [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]


def _mul(a: Mat3, b: Mat3) -> Mat3:
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def rpy_to_matrix(rx_deg: float, ry_deg: float, rz_deg: float) -> Mat3:
    rx, ry, rz = (math.radians(v) for v in (rx_deg, ry_deg, rz_deg))
    return _mul(_rz(rz), _mul(_ry(ry), _rx(rx)))


def matrix_to_rpy(r: Mat3) -> Tuple[float, float, float]:
    """Inverse of :func:`rpy_to_matrix`, degrees, ry in [-90, 90]."""
    sy = -r[2][0]
    sy = max(-1.0, min(1.0, sy))
    ry = math.asin(sy)
    if abs(sy) < 1.0 - 1e-9:
        rx = math.atan2(r[2][1], r[2][2])
        rz = math.atan2(r[1][0], r[0][0])
    else:  # gimbal lock: put everything in rz
        rx = 0.0
        rz = math.atan2(-r[0][1], r[1][1])
    return math.degrees(rx), math.degrees(ry), math.degrees(rz)


def apply_cartesian_key(
    coords: Sequence[float],
    binding: Binding,
    linear_step_mm: float,
    angular_step_deg: float,
    rotation_frame: str = "base",
) -> List[float]:
    """Return ``coords`` with ``binding``'s delta. Translation is always along base axes.

    Rotation composes as matrices: base frame ``R_delta @ R``, tool frame
    ``R @ R_delta``. The position is untouched by a rotation key.
    """
    out = [float(v) for v in coords]
    if binding.is_linear:
        out[binding.index] += binding.sign * float(linear_step_mm)
        return out
    delta = [0.0, 0.0, 0.0]
    delta[binding.index - 3] = binding.sign * float(angular_step_deg)
    r_delta = rpy_to_matrix(*delta)
    r = rpy_to_matrix(out[3], out[4], out[5])
    if rotation_frame == "base":
        r_new = _mul(r_delta, r)
    elif rotation_frame == "tool":
        r_new = _mul(r, r_delta)
    else:
        raise ValueError(f"rotation_frame must be 'base' or 'tool', got {rotation_frame!r}")
    out[3], out[4], out[5] = matrix_to_rpy(r_new)
    return out


def apply_gripper_key(value: int, key: str, step: int) -> Optional[int]:
    """New gripper value for a gripper key, clamped to 0-100; ``None`` for other keys."""
    entry = GRIPPER_KEYS.get(key)
    if entry is None:
        return None
    _, kind, amount = entry
    new = int(value) + amount * int(step) if kind == "step" else amount
    return max(GRIPPER_MIN, min(GRIPPER_MAX, new))


def describe_bindings(mode: str, joint_step_deg: float, linear_step_mm: float, angular_step_deg: float) -> str:
    table = bindings_for(mode)
    lines: List[str] = []
    if mode == "joint":
        lines.append(f"joints     ({joint_step_deg:.1f} deg/key)")
        lines.append("    " + "   ".join(f"{k} {b.label}" for k, b in table.items() if b.sign > 0))
        lines.append("    " + "   ".join(f"{k} {b.label}" for k, b in table.items() if b.sign < 0))
    else:
        lines.append(f"translate  ({linear_step_mm:.1f} mm/key, base frame)")
        lines.append("    " + "   ".join(f"{k} {b.label}" for k, b in table.items() if b.is_linear))
        lines.append(f"rotate     ({angular_step_deg:.1f} deg/key)")
        lines.append("    " + "   ".join(f"{k} {b.label}" for k, b in table.items() if not b.is_linear))
    lines.append("gripper")
    lines.append("    " + "   ".join(f"{k} {label}" for k, (label, _, _) in GRIPPER_KEYS.items()))
    lines.append("control")
    shown = {" ": "space", "\x1b": "esc", "\x03": "ctrl-c"}
    for key, meaning in CONTROL_KEYS.items():
        lines.append(f"    {shown.get(key, key):<7} {meaning}")
    return "\n".join(lines)
