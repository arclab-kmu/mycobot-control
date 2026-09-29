"""Session management.

A session owns what must be created and torn down together: the backend
(dry-run or pymycobot), the connection (and the serial port lock), and the
:class:`~mycobot_control.safety.gate.ControlGate`. Use it as a context manager;
on exit the gate closes and the connection is released, in that order.
"""

from __future__ import annotations

import contextlib
import logging
import time
from dataclasses import dataclass
from typing import Callable, Iterator, List, Optional, Sequence, TypeVar

from mycobot_control.safety.gate import ControlGate
from mycobot_control.safety.limits import (
    DEFAULT_GRIPPER_SPEED,
    DEFAULT_SPEED,
    GRIPPER_MAX,
    GRIPPER_MIN,
    GRIPPER_TYPE_ADAPTIVE,
    LimitViolation,
    MAX_COMMAND_DELTA_DEG,
    check_command_delta,
    check_joint_target,
    check_speed,
    coords_in_range,
)
from mycobot_control.sdk.backend import (
    DEFAULT_NETPORT,
    JN_BAUD,
    JN_SERIAL_PORT,
    ArmBackend,
    CoordsReading,
    DryRunBackend,
    JointReading,
    ReadOnlyBackend,
)
from mycobot_control.sdk.status import StatusReading

log = logging.getLogger(__name__)

T = TypeVar("T")


class SessionError(RuntimeError):
    """The session could not be brought into, or kept in, a usable state."""


@dataclass
class ArmConfig:
    """How to open a session.

    Attributes:
        live: ``False`` (the default) builds a :class:`DryRunBackend`; nothing
            is imported from pymycobot and nothing moves. ``True`` builds a
            :class:`~mycobot_control.sdk.backend.PyMyCobotBackend`. Nothing in
            this package flips this for you.
        read_only: connect to the real arm, read, never transmit.
        power_on: call ``power_on()`` at start. The JN's servos are normally
            already holding after boot, so this is only needed after a
            ``release_all_servos()`` (e.g. from the vendor's follow_display).
        transport: ``"serial"`` on the Jetson, ``"socket"`` from a PC.
        speed: pymycobot speed 1-100 for every joint / Cartesian command.
        max_feedback_age_s: a reading whose round trip took longer than this
            is refused as a basis for a command.
        max_command_delta_deg: the largest per-joint jump a single command may
            make from the measured angles.
    """

    live: bool = False
    read_only: bool = False
    power_on: bool = False
    transport: str = "serial"
    port: str = JN_SERIAL_PORT
    baud: int = JN_BAUD
    host: Optional[str] = None
    netport: int = DEFAULT_NETPORT
    gripper_type: int = GRIPPER_TYPE_ADAPTIVE
    speed: int = DEFAULT_SPEED
    gripper_speed: int = DEFAULT_GRIPPER_SPEED
    fresh_mode: int = 1
    max_feedback_age_s: float = 0.5
    max_command_delta_deg: float = MAX_COMMAND_DELTA_DEG
    lock_port: bool = True

    def __post_init__(self) -> None:
        if self.transport not in ("serial", "socket"):
            raise ValueError(f"transport must be 'serial' or 'socket', got {self.transport!r}")
        if self.transport == "socket" and not self.host:
            raise ValueError("transport='socket' needs host (the Jetson's IP)")
        if self.power_on and not self.live:
            raise ValueError("power_on=True requires live=True")
        if self.read_only and not self.live:
            raise ValueError("read_only=True requires live=True (a dry run commands nothing anyway)")
        if self.read_only and self.power_on:
            raise ValueError("read_only=True and power_on=True are contradictory")
        self.speed = check_speed(self.speed)
        if self.fresh_mode not in (0, 1):
            raise ValueError(f"fresh_mode must be 0 or 1, got {self.fresh_mode}")

    def describe_target(self) -> str:
        if not self.live:
            return "dry-run"
        if self.transport == "serial":
            return f"serial {self.port} @ {self.baud}"
        return f"socket {self.host}:{self.netport}"


class ArmSession:
    """A connected arm plus its command gate. The gate starts closed."""

    def __init__(
        self,
        config: ArmConfig,
        backend: Optional[ArmBackend] = None,
        gate: Optional[ControlGate] = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config
        self.gate = gate if gate is not None else ControlGate()
        self.backend: ArmBackend = backend if backend is not None else make_backend(config)
        self._clock = clock
        self._last_joints: Optional[JointReading] = None

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> "ArmSession":
        self.backend.connect()
        if not self.config.read_only:
            # fresh mode 1 = the newest command replaces the pending one. With
            # the queue (0), keypresses faster than the arm pile up and keep
            # executing after the operator stops - the opposite of a deadman.
            self.backend.set_fresh_mode(self.config.fresh_mode)
        if self.config.power_on:
            self.backend.power_on()
        log.info("session started on %s", self.backend.describe())
        return self

    def stop(self) -> None:
        # Gate first: nothing may slip through during teardown. No
        # release_all_servos()/power_off(): both drop torque and a raised arm falls.
        self.gate.close("session stopped")
        with contextlib.suppress(Exception):
            self.backend.disconnect()

    def __enter__(self) -> "ArmSession":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    # -- feedback ----------------------------------------------------------
    def _retry(self, read: Callable[[], Optional[T]], retry_s: float) -> Optional[T]:
        deadline = self._clock() + max(0.0, retry_s)
        value = read()
        while value is None and self._clock() < deadline:
            time.sleep(0.02)
            value = read()
        return value

    def read_joints(self, retry_s: float = 0.5) -> JointReading:
        """Measured joint angles (deg).

        ``None`` / ``-1`` replies are retried for ``retry_s``: they are routine
        right after the port opens. A reading whose round trip took longer than
        ``max_feedback_age_s`` is refused - on a request/response link, the
        latency is how stale the angles were when they arrived.

        Raises:
            SessionError: no usable reading, or a too-slow one.
        """
        reading = self._retry(self.backend.get_joint_state, retry_s)
        if reading is None:
            raise SessionError(
                f"no joint angles from the arm within {retry_s:.2f} s. Check the arm is "
                f"powered, that {self.config.describe_target()} is the right link, and "
                "that no other process (vendor ROS node, myStudio) owns the port."
            )
        if reading.latency_s is not None and reading.latency_s > self.config.max_feedback_age_s:
            raise SessionError(
                f"get_angles() took {reading.latency_s:.3f} s (limit "
                f"{self.config.max_feedback_age_s:.3f} s); refusing to build a command on it"
            )
        self._last_joints = reading
        return reading

    def read_coords(self, retry_s: float = 0.5) -> CoordsReading:
        reading = self._retry(self.backend.get_coords, retry_s)
        if reading is None:
            raise SessionError(f"no coords from the arm within {retry_s:.2f} s")
        if reading.latency_s is not None and reading.latency_s > self.config.max_feedback_age_s:
            raise SessionError(f"get_coords() took {reading.latency_s:.3f} s; refusing to use it")
        return reading

    @property
    def last_joints(self) -> Optional[JointReading]:
        return self._last_joints

    def read_status(self) -> Optional[StatusReading]:
        """Decoded health; ``None`` if unavailable. Never raises.

        A missing joint reading makes the next command wrong, so that stops the
        caller. A missing status only costs the diagnosis.
        """
        try:
            return self.backend.read_status()
        except Exception as exc:  # serial hiccup; the vendor wraps every call too
            log.warning("read_status failed: %s", exc)
            return None

    # -- commands ----------------------------------------------------------
    def command_joints(
        self,
        angles_deg: Sequence[float],
        measured: Optional[JointReading] = None,
        speed: Optional[int] = None,
    ) -> List[float]:
        """Send a joint target (deg) via ``send_angles``, through the gate and the limits.

        ``measured`` defaults to the last :meth:`read_joints` result. There must
        be one: the jump check is against where the arm *is*, and a command with
        nothing to compare against is refused rather than trusted.

        Raises:
            GateClosed, LimitViolation, SessionError.
        """
        self.gate.require_open()
        target = check_joint_target(angles_deg)
        basis = measured if measured is not None else self._last_joints
        if basis is None:
            raise SessionError("no measured joint angles to check the command against; read_joints() first")
        check_command_delta(target, basis.angles, self.config.max_command_delta_deg)
        self.backend.send_angles(target, check_speed(speed if speed is not None else self.config.speed))
        return target

    def command_coords(self, coords: Sequence[float], speed: Optional[int] = None, mode: int = 0) -> List[float]:
        """Send a Cartesian target (mm, deg) via ``send_coords``; the firmware solves the IK.

        Raises:
            GateClosed, LimitViolation (outside the SDK's accepted box).
        """
        self.gate.require_open()
        target = [float(v) for v in coords]
        if len(target) != 6:
            raise ValueError(f"need 6 coords, got {len(target)}")
        if not coords_in_range(target):
            raise LimitViolation(f"coords {target} outside the range pymycobot accepts for the 280")
        if mode not in (0, 1):
            raise ValueError("mode must be 0 (angular) or 1 (linear)")
        self.backend.send_coords(target, check_speed(speed if speed is not None else self.config.speed), mode)
        return target

    def command_gripper(self, value: int, speed: Optional[int] = None) -> int:
        """Gripper 0-100 via ``set_gripper_value``. Through the gate like every command."""
        self.gate.require_open()
        v = int(value)
        if not GRIPPER_MIN <= v <= GRIPPER_MAX:
            raise LimitViolation(f"gripper value {value} outside [{GRIPPER_MIN}, {GRIPPER_MAX}]")
        self.backend.set_gripper_value(v, check_speed(speed if speed is not None else self.config.gripper_speed))
        return v

    def halt(self, reason: str = "halt") -> None:
        """Close the gate and ask the arm to stop its current motion. Torque stays on."""
        self.gate.close(reason)
        try:
            self.backend.stop()
        except Exception as exc:
            log.warning("stop() failed: %s", exc)


def make_backend(config: ArmConfig) -> ArmBackend:
    if not config.live:
        return DryRunBackend()
    # Imported here: importing this module must not require pymycobot.
    from mycobot_control.sdk.backend import PyMyCobotBackend

    backend: ArmBackend = PyMyCobotBackend(
        transport=config.transport,
        port=config.port,
        baud=config.baud,
        host=config.host,
        netport=config.netport,
        gripper_type=config.gripper_type,
        lock=config.lock_port,
    )
    return ReadOnlyBackend(backend) if config.read_only else backend


@contextlib.contextmanager
def open_session(config: ArmConfig, backend: Optional[ArmBackend] = None) -> Iterator[ArmSession]:
    """Open an :class:`ArmSession`, guaranteeing teardown."""
    session = ArmSession(config, backend=backend)
    try:
        yield session.start()
    finally:
        session.stop()
