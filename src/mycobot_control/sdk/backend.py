"""Backends: talking to the arm through ``pymycobot``.

Three implementations sit behind one Protocol, as in nero-control:

:class:`DryRunBackend`
    The default. Keeps joint angles and coords in variables, logs every
    command, touches no hardware and imports no vendor code.

:class:`PyMyCobotBackend`
    The real thing, over the Jetson's UART (``MyCobot280``) or over TCP to a
    server running on the Jetson (``MyCobot280Socket``). ``pymycobot`` is
    imported lazily inside :meth:`connect`, so importing this module - or the
    whole package - works on a machine that has never seen the vendor SDK.

:class:`ReadOnlyBackend`
    A decorator that forwards reads and drops commands.

Signatures are quoted from pymycobot 4.0.7 (``inspect.signature``, read
2026-09-23). Units are the SDK's: degrees for joints, mm + degrees for coords.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, List, Optional, Protocol, Sequence, runtime_checkable

from mycobot_control.safety.limits import GRIPPER_TYPE_ADAPTIVE, N_JOINTS, joint_vector
from mycobot_control.sdk.status import ERROR_NONE, StatusReading, from_raw

log = logging.getLogger(__name__)

#: Defaults for the myCobot 280 JetsonNano: the Jetson's onboard UART, at the
#: baud rate every vendor JN script uses (mycobot_280jn/*.py). Code using this
#: port has to run on the Jetson itself.
JN_SERIAL_PORT = "/dev/ttyTHS1"
JN_BAUD = 1000000
#: MyCobot280Socket's default ``netport``.
DEFAULT_NETPORT = 9000


@dataclass(frozen=True)
class JointReading:
    """Six joint angles in degrees, plus when and how fast they came back.

    ``pymycobot`` reads are request/response on the serial line, not a pushed
    feed, so there is no SDK timestamp. ``timestamp`` is host ``time.time()``
    when the reply was parsed, and ``latency_s`` is how long the round trip
    took - which is the only measure of how old the angles already are.
    """

    angles: List[float]
    timestamp: float
    latency_s: Optional[float] = None

    @property
    def radians(self) -> List[float]:
        return [math.radians(a) for a in self.angles]


@dataclass(frozen=True)
class CoordsReading:
    """``[x, y, z, rx, ry, rz]`` in mm and degrees, in the firmware's base frame.

    TODO(unverified): the firmware base frame is rotated 90 deg about z from
    the URDF's ``joint1`` frame - vendor ``follow_display.py`` maps
    ``urdf_x = -coords[1]``, ``urdf_y = coords[0]``. Asserted by the vendor, not
    checked here. The rx/ry/rz Euler convention is not documented at all.
    """

    coords: List[float]
    timestamp: float
    latency_s: Optional[float] = None


def valid_vector(raw: Any, n: int = N_JOINTS) -> Optional[List[float]]:
    """``raw`` as ``n`` finite floats, or ``None`` if it is not a usable reading.

    pymycobot returns ``-1`` (the int) on a failed read, and can return ``None``
    or an empty list shortly after opening the port. The vendor's own check in
    ``listen_real_service.py`` also rejects a vector containing an element
    equal to -1, and so does this one.
    TODO(unverified): whether the firmware really marks a single failed joint
    as -1. If it does not, the cost of copying the vendor is that a joint
    resting at exactly -1.00 deg reads as "no reading" and is retried.
    """
    if not isinstance(raw, (list, tuple)) or len(raw) != n:
        return None
    try:
        values = [float(v) for v in raw]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in values) or any(v == -1.0 for v in values):
        return None
    return values


@runtime_checkable
class ArmBackend(Protocol):
    """The slice of pymycobot this package uses."""

    def connect(self) -> None: ...

    def disconnect(self) -> None: ...

    def is_connected(self) -> bool: ...

    def set_fresh_mode(self, mode: int) -> None: ...

    def power_on(self) -> None: ...

    def get_joint_state(self) -> Optional[JointReading]: ...

    def get_coords(self) -> Optional[CoordsReading]: ...

    def get_gripper_value(self) -> Optional[int]: ...

    def read_status(self) -> Optional[StatusReading]: ...

    def send_angles(self, angles_deg: Sequence[float], speed: int) -> None: ...

    def send_coords(self, coords: Sequence[float], speed: int, mode: int = 0) -> None: ...

    def set_gripper_value(self, value: int, speed: int) -> None: ...

    def stop(self) -> None: ...

    def clear_errors(self) -> None: ...

    def describe(self) -> str: ...


@dataclass
class DryRunBackend:
    """A backend that moves nothing.

    Bookkeeping only: :meth:`send_angles` teleports the stored angles and
    :meth:`send_coords` teleports the stored coords. There is no kinematics, so
    the two are unrelated - sending angles does not change the reported coords.
    A dry run proves the key mapping, the gate and the command sequence. It
    proves nothing about where the arm can go.
    """

    #: The imaginary arm starts at the zero pose - the one ``go_home()`` sends.
    joint_angles: List[float] = field(default_factory=lambda: [0.0] * N_JOINTS)
    #: A plausible tool-down pose. Not read from an arm.
    coords: List[float] = field(default_factory=lambda: [150.0, -60.0, 200.0, -180.0, 0.0, -90.0])
    gripper_value: int = 50
    #: Overridable so tests can drive the fault-handling paths.
    error_code: int = ERROR_NONE
    powered: bool = True
    #: Commands in call order: ``("send_angles", angles, speed)`` etc.
    commands: List[tuple] = field(default_factory=list)

    _connected: bool = field(init=False, default=False)
    _fresh_mode: Optional[int] = field(init=False, default=None)

    def connect(self) -> None:
        self._connected = True
        log.info("[dry-run] connect()")

    def disconnect(self) -> None:
        self._connected = False
        log.info("[dry-run] disconnect()")

    def is_connected(self) -> bool:
        return self._connected

    def set_fresh_mode(self, mode: int) -> None:
        self._fresh_mode = int(mode)
        self.commands.append(("set_fresh_mode", int(mode)))

    def power_on(self) -> None:
        self.powered = True
        self.commands.append(("power_on",))

    def get_joint_state(self) -> Optional[JointReading]:
        return JointReading(list(self.joint_angles), timestamp=time.time(), latency_s=0.0)

    def get_coords(self) -> Optional[CoordsReading]:
        return CoordsReading(list(self.coords), timestamp=time.time(), latency_s=0.0)

    def get_gripper_value(self) -> Optional[int]:
        return int(self.gripper_value)

    def read_status(self) -> Optional[StatusReading]:
        return StatusReading(error_code=self.error_code, power_on=self.powered)

    def send_angles(self, angles_deg: Sequence[float], speed: int) -> None:
        target = joint_vector(angles_deg)
        self.commands.append(("send_angles", target, int(speed)))
        self.joint_angles = list(target)
        log.info("[dry-run] send_angles(%s, %d)", _fmt(target), speed)

    def send_coords(self, coords: Sequence[float], speed: int, mode: int = 0) -> None:
        target = [float(v) for v in coords]
        self.commands.append(("send_coords", target, int(speed), int(mode)))
        self.coords = list(target)
        log.info("[dry-run] send_coords(%s, %d, mode=%d)", _fmt(target), speed, mode)

    def set_gripper_value(self, value: int, speed: int) -> None:
        self.commands.append(("set_gripper_value", int(value), int(speed)))
        self.gripper_value = int(value)

    def stop(self) -> None:
        self.commands.append(("stop",))
        log.info("[dry-run] stop()")

    def clear_errors(self) -> None:
        self.commands.append(("clear_errors",))
        self.error_code = ERROR_NONE

    def describe(self) -> str:
        return "dry-run (no hardware, no vendor SDK)"


class PyMyCobotBackend:
    """Real hardware via ``pymycobot``.

    Args:
        transport: ``"serial"`` - run on the Jetson, talk to ``/dev/ttyTHS1``
            with ``MyCobot280``. ``"socket"`` - run anywhere, talk TCP to a
            socket server on the Jetson with ``MyCobot280Socket``.
        port, baud: serial only.
        host, netport: socket only.
        gripper_type: 1 is the adaptive gripper this lab's arm carries.
        lock: take the vendor's ``/tmp/mycobot_lock`` (serial only).

    TODO(unverified): the socket path. It needs the vendor's socket server
    running on the Jetson. It is not in the pymycobot wheel; the vendor ships it
    in the pymycobot GitHub repo's ``demo/`` directory. Nobody has started one on
    this arm yet. Note ``MyCobot280Socket.__init__`` sleeps 1.5 s by itself.
    """

    TRANSPORTS = ("serial", "socket")

    def __init__(
        self,
        transport: str = "serial",
        port: str = JN_SERIAL_PORT,
        baud: int = JN_BAUD,
        host: Optional[str] = None,
        netport: int = DEFAULT_NETPORT,
        gripper_type: int = GRIPPER_TYPE_ADAPTIVE,
        lock: bool = True,
    ) -> None:
        if transport not in self.TRANSPORTS:
            raise ValueError(f"transport must be one of {self.TRANSPORTS}, got {transport!r}")
        if transport == "socket" and not host:
            raise ValueError("socket transport needs the Jetson's host/IP")
        self._transport = transport
        self._port = port
        self._baud = int(baud)
        self._host = host
        self._netport = int(netport)
        self._gripper_type = int(gripper_type)
        self._lock = None
        if lock and transport == "serial":
            from mycobot_control.sdk.portlock import PortLock

            self._lock = PortLock()
        self._mc: Any = None

    @property
    def raw(self) -> Any:
        """The underlying ``MyCobot280`` / ``MyCobot280Socket``. Reach through for reads only."""
        return self._mc

    # -- session -----------------------------------------------------------
    def connect(self) -> None:
        """Open the port. pymycobot opens the serial device in the constructor."""
        try:
            from pymycobot import MyCobot280, MyCobot280Socket  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - depends on host install
            raise ImportError(
                "pymycobot is not installed. Install the hardware extra:\n"
                "    uv sync --extra dev --extra arm      (or: pip install '.[arm]')\n"
                "Nothing else in mycobot_control needs it - dry-run works without it."
            ) from exc
        if self._lock is not None:
            self._lock.acquire()
        try:
            if self._transport == "serial":
                # MyCobot280(self, port, baudrate='115200', timeout=0.1, debug=False, thread_lock=True)
                self._mc = MyCobot280(self._port, self._baud)
            else:
                # MyCobot280Socket(self, ip, netport=9000, debug=False)
                self._mc = MyCobot280Socket(self._host, self._netport)
        except Exception:
            if self._lock is not None:
                self._lock.release()
            raise
        # Every vendor JN script sleeps 50 ms after opening before its first call.
        time.sleep(0.05)

    def disconnect(self) -> None:
        mc, self._mc = self._mc, None
        try:
            if mc is not None:
                # close(self) on both classes: the serial port, or the socket.
                mc.close()
        finally:
            if self._lock is not None:
                self._lock.release()

    def is_connected(self) -> bool:
        return self._mc is not None

    def _require(self) -> Any:
        if self._mc is None:
            raise RuntimeError("backend is not connected")
        return self._mc

    def set_fresh_mode(self, mode: int) -> None:
        """``set_fresh_mode(self, mode)``: 1 = always execute the latest command, 0 = queue.

        Teleop wants 1: with a queue, keys pressed faster than the arm moves
        pile up and keep executing after the operator lets go.
        """
        self._require().set_fresh_mode(int(mode))

    def power_on(self) -> None:
        """``power_on(self)`` - torque on for all servos."""
        self._require().power_on()

    # -- feedback ----------------------------------------------------------
    def get_joint_state(self) -> Optional[JointReading]:
        """``get_angles(self)`` -> list of 6 floats in degrees, or -1."""
        mc = self._require()
        t0 = time.monotonic()
        raw = mc.get_angles()
        latency = time.monotonic() - t0
        angles = valid_vector(raw)
        if angles is None:
            log.debug("get_angles() returned unusable %r", raw)
            return None
        return JointReading(angles, timestamp=time.time(), latency_s=latency)

    def get_coords(self) -> Optional[CoordsReading]:
        """``get_coords(self)`` -> ``[x, y, z, rx, ry, rz]`` mm / deg, or -1."""
        mc = self._require()
        t0 = time.monotonic()
        raw = mc.get_coords()
        latency = time.monotonic() - t0
        coords = valid_vector(raw)
        if coords is None:
            return None
        return CoordsReading(coords, timestamp=time.time(), latency_s=latency)

    def get_gripper_value(self) -> Optional[int]:
        """``get_gripper_value(self, gripper_type=None)``; 1 = adaptive gripper."""
        raw = self._require().get_gripper_value(self._gripper_type)
        if isinstance(raw, bool) or not isinstance(raw, int) or raw == -1:
            return None
        return raw

    def read_status(self) -> Optional[StatusReading]:
        """``get_error_information(self)`` and ``is_power_on(self)``, decoded."""
        mc = self._require()
        return from_raw(mc.get_error_information(), mc.is_power_on())

    # -- commands ----------------------------------------------------------
    def send_angles(self, angles_deg: Sequence[float], speed: int) -> None:
        """``send_angles(self, angles, speed, _async=False)`` - degrees, speed 1-100.

        ``_async=True`` as in the vendor's streaming slider: the call returns
        once the frame is written instead of waiting for a reply.
        TODO(unverified): what ``_async=False`` waits for on this firmware.
        """
        self._require().send_angles(joint_vector(angles_deg), int(speed), _async=True)

    def send_coords(self, coords: Sequence[float], speed: int, mode: int = 0) -> None:
        """``send_coords(self, coords, speed, mode=None, _async=False)``.

        ``mode`` 0 = joint-interpolated ("angular"), 1 = straight line. The
        firmware solves the IK; a target it cannot solve is dropped and shows
        up only as error 32-34 in ``get_error_information()``.
        """
        self._require().send_coords([float(v) for v in coords], int(speed), int(mode), _async=True)

    def set_gripper_value(self, value: int, speed: int) -> None:
        """``set_gripper_value(self, gripper_value, speed, gripper_type=None, is_torque=None)``."""
        self._require().set_gripper_value(int(value), int(speed), gripper_type=self._gripper_type)

    def stop(self) -> None:
        """``stop(self)`` - stop the current motion. Torque stays on; the arm holds."""
        self._require().stop()

    def clear_errors(self) -> None:
        """``clear_error_information(self)``."""
        self._require().clear_error_information()

    def describe(self) -> str:
        if self._transport == "serial":
            return f"pymycobot MyCobot280 on {self._port} @ {self._baud}"
        return f"pymycobot MyCobot280Socket to {self._host}:{self._netport}"


class ReadOnlyBackend:
    """Forwards reads, refuses to transmit.

    ``stop()`` is forwarded: blocking a stop is never the safe direction.
    Everything that moves the arm, changes its mode or powers servos is
    dropped and recorded in :attr:`suppressed`.
    """

    def __init__(self, inner: ArmBackend) -> None:
        self._inner = inner
        self.suppressed: List[tuple] = []

    @property
    def inner(self) -> ArmBackend:
        return self._inner

    def connect(self) -> None:
        self._inner.connect()

    def disconnect(self) -> None:
        self._inner.disconnect()

    def is_connected(self) -> bool:
        return self._inner.is_connected()

    def get_joint_state(self) -> Optional[JointReading]:
        return self._inner.get_joint_state()

    def get_coords(self) -> Optional[CoordsReading]:
        return self._inner.get_coords()

    def get_gripper_value(self) -> Optional[int]:
        return self._inner.get_gripper_value()

    def read_status(self) -> Optional[StatusReading]:
        return self._inner.read_status()

    def stop(self) -> None:
        self._inner.stop()

    def set_fresh_mode(self, mode: int) -> None:
        self._suppress("set_fresh_mode", int(mode))

    def power_on(self) -> None:
        raise ReadOnlyViolation("read-only session: refusing to power on the servos")

    def send_angles(self, angles_deg: Sequence[float], speed: int) -> None:
        self._suppress("send_angles", joint_vector(angles_deg), int(speed))

    def send_coords(self, coords: Sequence[float], speed: int, mode: int = 0) -> None:
        self._suppress("send_coords", [float(v) for v in coords], int(speed), int(mode))

    def set_gripper_value(self, value: int, speed: int) -> None:
        self._suppress("set_gripper_value", int(value), int(speed))

    def clear_errors(self) -> None:
        self._suppress("clear_errors")

    def _suppress(self, *command: Any) -> None:
        self.suppressed.append(tuple(command))
        log.warning("[read-only] suppressed %s%r", command[0], command[1:])

    def describe(self) -> str:
        return f"read-only over {self._inner.describe()}"


class ReadOnlyViolation(RuntimeError):
    """Something tried to drive the arm during a read-only session."""


def _fmt(values: Sequence[float]) -> str:
    return "[" + ", ".join(f"{v:+.2f}" for v in values) + "]"
