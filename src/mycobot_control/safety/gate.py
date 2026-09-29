"""The command gate.

``pymycobot`` has no gate of its own: ``send_angles()`` goes straight onto the
serial line (or the socket to the Jetson). The vendor ROS 2 nodes do not add one
either - ``slider_control`` forwards every ``joint_states`` message it hears.

:class:`ControlGate` is that gate, in the host process. It is closed on
construction, and every command path in this package calls :meth:`require_open`
before touching the backend. Same design as ``nero_control.safety.gate``.

This is a software interlock in a Python process. It is not a safety system.
"""

from __future__ import annotations

import logging
import time
from typing import Callable

log = logging.getLogger(__name__)


class GateClosed(RuntimeError):
    """A command was attempted while the control gate was closed."""


class ControlGate:
    """A latched open/closed gate that commands must pass.

    Args:
        clock: monotonic time source, injectable for tests.
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._open = False
        self._reason = "never opened"
        self._changed_at = clock()

    @property
    def is_open(self) -> bool:
        return self._open

    @property
    def reason(self) -> str:
        """Why the gate is in its current state."""
        return self._reason

    @property
    def changed_at(self) -> float:
        return self._changed_at

    def open(self, reason: str = "explicitly opened") -> None:
        if not self._open:
            log.info("control gate OPEN (%s)", reason)
        self._open = True
        self._reason = reason
        self._changed_at = self._clock()

    def close(self, reason: str = "explicitly closed") -> None:
        if self._open:
            log.warning("control gate CLOSED (%s)", reason)
        self._open = False
        self._reason = reason
        self._changed_at = self._clock()

    def require_open(self) -> None:
        """Raise :class:`GateClosed` unless the gate is open."""
        if not self._open:
            raise GateClosed(f"control gate is closed: {self._reason}")

    def __enter__(self) -> "ControlGate":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # Any exit, including an exception, closes the gate.
        self.close("scope exit")

    def __repr__(self) -> str:
        state = "open" if self._open else "closed"
        return f"<ControlGate {state}: {self._reason}>"
