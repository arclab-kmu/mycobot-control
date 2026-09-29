"""Decoding the arm's health into a policy.

Two reads make up the status: ``get_error_information()`` and
``is_power_on()``. Codes quoted from the ``MyCobot280.get_error_information``
docstring, pymycobot 4.0.7 (read 2026-09-23)::

    0: No error message.
    1 ~ 6: The corresponding joint exceeds the limit position.
    16 ~ 19: Collision protection.
    32: Kinematics inverse solution has no solution.
    33 ~ 34: Linear motion has no adjacent solution.

The split mirrors nero-control's: a *halting* status latches the teleop loop,
a *rejected* status means "the last Cartesian target was refused" and is
reported but does not latch.

TODO(unverified): whether an error code clears by itself once the condition
passes, or stays until ``clear_error_information()``. If it is sticky, a latch
that waits for 0 waits for an explicit clear (``mycobot-info --clear-errors``).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

ERROR_NONE = 0


def error_name(code: Optional[int]) -> str:
    if code is None:
        return "UNKNOWN"
    if code == ERROR_NONE:
        return "NORMAL"
    if 1 <= code <= 6:
        return f"J{code}_LIMIT"
    if 16 <= code <= 19:
        return f"COLLISION_{code}"
    if code == 32:
        return "NO_IK_SOLUTION"
    if code in (33, 34):
        return f"NO_LINEAR_SOLUTION_{code}"
    return f"ERROR_{code}"


@dataclass(frozen=True)
class StatusReading:
    """One health snapshot. ``None`` fields mean the arm did not answer that query."""

    error_code: Optional[int]
    power_on: Optional[bool]
    timestamp: float = field(default_factory=time.time)

    @property
    def error_name(self) -> str:
        return error_name(self.error_code)

    @property
    def was_rejected(self) -> bool:
        """The last Cartesian target had no IK / linear solution. Nothing moved because of it."""
        return self.error_code in (32, 33, 34)

    @property
    def should_halt(self) -> bool:
        """Stop commanding until the arm reports NORMAL.

        Unknown non-zero codes halt: an error we cannot name is not one we can
        call harmless. An unanswered query does not halt - see
        :meth:`ArmSession.read_status` for why a missing status is not a fault.
        """
        if self.power_on is False:
            return True
        if self.error_code is None or self.error_code == ERROR_NONE:
            return False
        return not self.was_rejected

    @property
    def is_normal(self) -> bool:
        return self.error_code == ERROR_NONE and self.power_on is not False

    def describe(self) -> str:
        power = {True: "on", False: "OFF", None: "?"}[self.power_on]
        code = "?" if self.error_code is None else str(self.error_code)
        return f"error {code} ({self.error_name}), power {power}"


def from_raw(error_raw, power_raw) -> StatusReading:
    """Build a reading from the two raw SDK replies. ``-1`` / ``None`` become unknown."""

    def as_int(value) -> Optional[int]:
        if isinstance(value, bool) or not isinstance(value, int) or value == -1:
            return None
        return value

    error = as_int(error_raw)
    power = as_int(power_raw)
    return StatusReading(error_code=error, power_on=None if power is None else bool(power))
