"""Deadman: stop commanding when operator input goes stale.

Guards the same two failure modes as nero-control: the operator stops pressing
keys (walked away, ssh froze) while the process keeps commanding, and the
process dying mid-motion. Both are handled by **closing the gate** - no new
targets leave.

What we deliberately do NOT do on expiry or shutdown is ``release_all_servos()``
or ``power_off()``. Both drop servo torque, and a raised myCobot then falls
under gravity - the vendor's own ``follow_display`` node calls
``release_all_servos()`` precisely so a person can backdrive the arm by hand.
Ceasing to send targets leaves the servos holding the last one, which is the
correct response to a lost operator.
"""

from __future__ import annotations

import logging
import time
from typing import Callable, Optional

from mycobot_control.safety.gate import ControlGate

log = logging.getLogger(__name__)

#: How long after the last operator input the gate closes itself.
DEFAULT_DEADMAN_TIMEOUT_S = 0.5


class Deadman:
    """Closes a :class:`ControlGate` when operator input goes stale.

    Args:
        gate: the gate to close on expiry.
        timeout_s: seconds of silence tolerated before the gate closes.
        clock: monotonic time source, injectable for tests.
    """

    def __init__(
        self,
        gate: ControlGate,
        timeout_s: float = DEFAULT_DEADMAN_TIMEOUT_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if timeout_s <= 0.0:
            raise ValueError(f"deadman timeout must be positive, got {timeout_s}")
        self._gate = gate
        self._timeout_s = float(timeout_s)
        self._clock = clock
        self._last_input: Optional[float] = None

    @property
    def timeout_s(self) -> float:
        return self._timeout_s

    def poke(self, reason: str = "operator input") -> None:
        """Record operator input and open the gate."""
        self._last_input = self._clock()
        self._gate.open(reason)

    def age(self) -> float:
        """Seconds since the last :meth:`poke`; ``inf`` if never poked."""
        if self._last_input is None:
            return float("inf")
        return self._clock() - self._last_input

    def expired(self) -> bool:
        return self.age() > self._timeout_s

    def check(self) -> bool:
        """Close the gate if input has gone stale. Returns True if it expired."""
        if self.expired():
            if self._gate.is_open:
                log.warning("deadman expired after %.3f s of silence", self.age())
            self._gate.close(f"deadman expired ({self._timeout_s:.2f} s)")
            return True
        return False

    def release(self, reason: str = "key released") -> None:
        """Immediately close the gate, e.g. on an explicit stop key."""
        self._last_input = None
        self._gate.close(reason)
