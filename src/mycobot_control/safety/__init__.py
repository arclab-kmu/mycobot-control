"""Software interlocks: the command gate, the deadman and the limit tables.

None of this is a safety system. A physical way to cut the arm's power, within
reach, is the only thing between a bug here and the arm moving.
TODO(unverified): whether this lab's myCobot 280 JN has an E-stop button fitted.
"""

from mycobot_control.safety.deadman import DEFAULT_DEADMAN_TIMEOUT_S, Deadman
from mycobot_control.safety.gate import ControlGate, GateClosed
from mycobot_control.safety.limits import LimitViolation

__all__ = [
    "ControlGate",
    "DEFAULT_DEADMAN_TIMEOUT_S",
    "Deadman",
    "GateClosed",
    "LimitViolation",
]
