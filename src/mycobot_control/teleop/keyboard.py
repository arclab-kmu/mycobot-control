"""Keyboard teleop for the myCobot 280: one step per keypress.

**Dry-run by default.** Without ``--live`` no vendor code is imported and
nothing moves; every command is logged. ``--live`` asks for confirmation.

Every motion key: check the fault latch, poke the deadman (the only thing that
opens the gate), read the MEASURED state, add one step, command it. Targets are
never accumulated open-loop - a delta stacked on a pose the arm never reached
compounds error.

Run it in the foreground of a real TTY; it reads single keys in cbreak mode.
"""

from __future__ import annotations

import argparse
import logging
import os
import select
import sys
import time
from typing import List, Optional, Sequence

try:  # POSIX only.
    import termios
    import tty

    HAVE_TERMIOS = True
except ImportError:  # pragma: no cover - Windows
    HAVE_TERMIOS = False

from mycobot_control.safety.deadman import DEFAULT_DEADMAN_TIMEOUT_S, Deadman
from mycobot_control.safety.gate import GateClosed
from mycobot_control.safety.limits import (
    DEFAULT_ANGULAR_STEP_DEG,
    DEFAULT_GRIPPER_STEP,
    DEFAULT_JOINT_STEP_DEG,
    DEFAULT_LINEAR_STEP_MM,
    LimitViolation,
    check_cartesian_step,
    check_joint_step,
)
from mycobot_control.sdk.cliargs import add_hardware_args, config_from_args, confirm_live, preflight
from mycobot_control.sdk.session import ArmSession, SessionError
from mycobot_control.teleop.eventlog import EventLog, default_log_path
from mycobot_control.teleop.keymap import (
    GRIPPER_KEYS,
    apply_cartesian_key,
    apply_gripper_key,
    apply_joint_key,
    bindings_for,
    describe_bindings,
)

log = logging.getLogger(__name__)

EOF_KEY = ""
QUIT_KEYS = frozenset({"q", "\x1b", "\x03", EOF_KEY})


class KeyReader:
    """Single-keystroke reader with a timeout. Works on a TTY (cbreak) and on a pipe (tests)."""

    def __init__(self, stream=None) -> None:
        self._stream = stream if stream is not None else sys.stdin
        try:
            self._fd = self._stream.fileno()
        except (AttributeError, OSError):  # pragma: no cover
            self._fd = -1
        self._saved: Optional[list] = None

    @property
    def is_tty(self) -> bool:
        return self._fd >= 0 and os.isatty(self._fd)

    def __enter__(self) -> "KeyReader":
        if not HAVE_TERMIOS:
            raise RuntimeError("keyboard teleop needs POSIX termios (Linux / macOS)")
        if self.is_tty:
            self._saved = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._saved is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved)
            self._saved = None

    def read(self, timeout: float) -> Optional[str]:
        """One key, ``""`` at EOF, or ``None`` on timeout. Arrow-key sequences read as ``None``."""
        if self._fd < 0:  # pragma: no cover
            return None
        ready, _, _ = select.select([self._fd], [], [], timeout)
        if not ready:
            return None
        data = os.read(self._fd, 1)
        if not data:
            return EOF_KEY
        key = data.decode("utf-8", errors="replace")
        if key == "\x1b" and self._readable_now():
            # Esc followed immediately by more bytes is an arrow / function key,
            # not the Esc quit key. Swallow exactly that sequence.
            nxt = os.read(self._fd, 1)
            if not nxt:
                return key
            if nxt == b"[":
                while self._readable_now():
                    byte = os.read(self._fd, 1)
                    if not byte or 0x40 <= byte[0] <= 0x7E:
                        break
            elif nxt == b"O" and self._readable_now():
                os.read(self._fd, 1)
            return None
        return key

    def _readable_now(self) -> bool:
        ready, _, _ = select.select([self._fd], [], [], 0.0)
        return bool(ready)


def _fmt(values: Sequence[float], unit: str = "") -> str:
    return "[" + " ".join(f"{v:+7.2f}" for v in values) + "]" + unit


def run(
    session: ArmSession,
    reader: KeyReader,
    mode: str = "joint",
    joint_step_deg: float = DEFAULT_JOINT_STEP_DEG,
    linear_step_mm: float = DEFAULT_LINEAR_STEP_MM,
    angular_step_deg: float = DEFAULT_ANGULAR_STEP_DEG,
    gripper_step: int = DEFAULT_GRIPPER_STEP,
    rotation_frame: str = "base",
    coords_mode: int = 0,
    deadman_timeout_s: float = DEFAULT_DEADMAN_TIMEOUT_S,
    poll_interval_s: float = 0.05,
    status_settle_s: float = 0.3,
    out=None,
    eventlog: Optional[EventLog] = None,
) -> int:
    """The teleop loop. Returns a process exit code."""
    out = out if out is not None else sys.stdout
    bindings = bindings_for(mode)
    deadman = Deadman(session.gate, timeout_s=deadman_timeout_s)
    halted_by: Optional[str] = None

    def say(msg: str) -> None:
        print(msg, file=out, flush=True)

    def note(event: str, **fields) -> None:
        if eventlog is not None:
            eventlog.record(event, gate=session.gate.is_open, latched=halted_by is not None, **fields)

    say(f"backend: {session.backend.describe()}")
    if session.config.read_only:
        say("         commands are computed and logged, never transmitted")
    say(f"mode   : {mode}   speed {session.config.speed}")
    say(f"deadman: {deadman_timeout_s:.2f} s of silence closes the control gate")
    if eventlog is not None:
        say(f"log    : {eventlog.csv_path}")
    say("")
    say(describe_bindings(mode, joint_step_deg, linear_step_mm, angular_step_deg))
    say("")

    try:
        joints = session.read_joints()
        coords = session.read_coords() if mode == "cartesian" else None
    except SessionError as exc:
        say(f"cannot read the starting state: {exc}")
        note("end", note=str(exc))
        return 2
    say(f"start  : joints {_fmt(joints.angles, ' deg')}")
    if coords is not None:
        say(f"         coords {_fmt(coords.coords)}")
    note("start", measured=joints.angles, note=session.backend.describe())

    gripper_value: Optional[int] = None
    try:
        gripper_value = session.backend.get_gripper_value()
    except Exception as exc:  # a missing gripper must not stop arm teleop
        log.warning("get_gripper_value failed: %s", exc)

    status_due_at: Optional[float] = None

    def latch_blocks(key: str, action: str) -> bool:
        """True while a halting status is latched. Only the arm reporting NORMAL clears it."""
        nonlocal halted_by
        if halted_by is None:
            return False
        status = session.read_status()
        # An unanswered error query is not a NORMAL report, so it does not clear.
        if status is None or status.error_code is None or status.should_halt:
            detail = status.describe() if status else "status unavailable"
            say(f"  ! still halted ({halted_by}): {detail}")
            note("key", key=key, action=action, outcome="halted", note=detail)
            return True
        say(f"  arm reports {status.error_name}; resuming")
        halted_by = None
        note("latch_clear", note=status.describe())
        return False

    while True:
        key = reader.read(poll_interval_s)

        if status_due_at is not None and time.monotonic() >= status_due_at:
            status_due_at = None
            status = session.read_status()
            if status is not None and not status.is_normal:
                say(f"  ! {status.describe()}")
                if status.should_halt:
                    halted_by = status.error_name
                    deadman.release(f"status {status.error_name}")
                    say("  ! gate closed and latched; it reopens only once the arm reports NORMAL")
                    outcome = "halt"
                else:
                    outcome = "rejected" if status.was_rejected else "abnormal"
                note("status", outcome=outcome, note=status.describe())

        if key is None:
            was_open = session.gate.is_open
            if deadman.check() and was_open:
                note("deadman", note=f"{deadman_timeout_s:.2f} s of silence")
            continue

        if key in QUIT_KEYS:
            deadman.release("quit")
            say("\nstopped. control gate closed.")
            note("key", key=key, action="quit", outcome="quit")
            note("end", note="eof" if key == EOF_KEY else "quit")
            return 0

        if key == " ":
            deadman.release("stop key")
            session.halt("stop key")
            say("stop: gate closed, stop() sent")
            note("key", key=key, action="stop", outcome="stop")
            continue

        if key in ("[", "]"):
            factor = 0.5 if key == "[" else 2.0
            try:
                check_joint_step(joint_step_deg * factor)
                check_cartesian_step(linear_step_mm * factor, angular_step_deg * factor)
            except LimitViolation as exc:
                say(f"step unchanged: {exc}")
                note("key", key=key, action="step", outcome="step_refused", note=str(exc))
                continue
            joint_step_deg *= factor
            linear_step_mm *= factor
            angular_step_deg *= factor
            say(f"step: {joint_step_deg:.2f} deg joint / {linear_step_mm:.2f} mm / {angular_step_deg:.2f} deg")
            note("key", key=key, action="step", outcome="step")
            continue

        if key == "?":
            say(describe_bindings(mode, joint_step_deg, linear_step_mm, angular_step_deg))
            note("key", key=key, action="help", outcome="help")
            continue

        if key in GRIPPER_KEYS:
            label = GRIPPER_KEYS[key][0]
            if latch_blocks(key, label):
                continue
            if gripper_value is None and GRIPPER_KEYS[key][1] == "step":
                say("  gripper value unknown; use < or > first")
                note("key", key=key, action=label, outcome="refused_feedback")
                continue
            target_g = apply_gripper_key(gripper_value if gripper_value is not None else 0, key, gripper_step)
            if target_g is None:  # unreachable: key is in GRIPPER_KEYS
                continue
            deadman.poke(f"key {key!r} -> {label}")
            try:
                gripper_value = session.command_gripper(target_g)
            except (GateClosed, LimitViolation) as exc:
                say(f"refused: {exc}")
                note("key", key=key, action=label, outcome="refused", note=str(exc))
                continue
            say(f"{label:<22} -> {gripper_value}")
            note("key", key=key, action=label, outcome="commanded", target=[gripper_value])
            continue

        binding = bindings.get(key)
        if binding is None:
            say(f"unbound key {key!r} - press ? for help")
            note("key", key=key, outcome="unbound")
            continue

        if latch_blocks(key, binding.label):
            continue

        deadman.poke(f"key {key!r} -> {binding.label}")
        measured: List[float]
        try:
            if mode == "joint":
                reading = session.read_joints()
                measured = reading.angles
                target = apply_joint_key(measured, binding, joint_step_deg)
                session.command_joints(target, measured=reading)
            else:
                creading = session.read_coords()
                measured = creading.coords
                target = apply_cartesian_key(measured, binding, linear_step_mm, angular_step_deg, rotation_frame)
                session.command_coords(target, mode=coords_mode)
        except GateClosed as exc:
            say(f"refused: {exc}")
            note("key", key=key, action=binding.label, outcome="refused_gate", note=str(exc))
            continue
        except LimitViolation as exc:
            say(f"refused: {exc}")
            note("key", key=key, action=binding.label, outcome="refused_limit", note=str(exc))
            continue
        except SessionError as exc:
            deadman.release("feedback lost")
            say(f"refused: {exc}")
            note("key", key=key, action=binding.label, outcome="refused_feedback", note=str(exc))
            continue

        say(f"{binding.label:<5} -> {_fmt(target)}")
        note("key", key=key, action=binding.label, outcome="commanded", measured=measured, target=target)
        # send_* is fire-and-forget (_async=True), so an IK rejection or a limit
        # error only shows up in a later get_error_information(). Read it on a
        # later poll rather than blocking here, which would cap the key rate.
        status_due_at = time.monotonic() + status_settle_s


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mycobot-teleop-keyboard",
        description="Keyboard joint / Cartesian teleop for the myCobot 280. Dry-run unless --live.",
    )
    parser.add_argument("--mode", choices=("joint", "cartesian"), default="joint",
                        help="joint: one key pair per joint (default). cartesian: firmware IK via send_coords.")
    parser.add_argument("--joint-step", type=float, default=DEFAULT_JOINT_STEP_DEG, help="deg per key")
    parser.add_argument("--linear-step", type=float, default=DEFAULT_LINEAR_STEP_MM, help="mm per key")
    parser.add_argument("--angular-step", type=float, default=DEFAULT_ANGULAR_STEP_DEG, help="deg per key")
    parser.add_argument("--gripper-step", type=int, default=DEFAULT_GRIPPER_STEP, help="gripper units per key")
    parser.add_argument("--rotation-frame", choices=("base", "tool"), default="base")
    parser.add_argument("--coords-mode", type=int, choices=(0, 1), default=0,
                        help="send_coords mode: 0 joint-interpolated (default), 1 straight line")
    parser.add_argument("--deadman", type=float, default=DEFAULT_DEADMAN_TIMEOUT_S, help="seconds")
    parser.add_argument("--log", nargs="?", const="auto", default=None,
                        help="write a per-keypress CSV (default path: recordings/teleop-*.csv)")
    parser.add_argument("-v", "--verbose", action="store_true")
    add_hardware_args(parser)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")
    try:
        check_joint_step(args.joint_step)
        check_cartesian_step(args.linear_step, args.angular_step)
    except LimitViolation as exc:
        print(exc, file=sys.stderr)
        return 2
    config = config_from_args(args)
    problem = preflight(config)
    if problem:
        print(problem, file=sys.stderr)
        return 2
    if not confirm_live(config, f"keyboard teleop ({args.mode} mode)", args.assume_yes):
        return 1
    eventlog = None
    if args.log:
        eventlog = EventLog(default_log_path() if args.log == "auto" else args.log)
    try:
        with ArmSession(config) as session, KeyReader() as reader:
            return run(
                session, reader, mode=args.mode,
                joint_step_deg=args.joint_step, linear_step_mm=args.linear_step,
                angular_step_deg=args.angular_step, gripper_step=args.gripper_step,
                rotation_frame=args.rotation_frame, coords_mode=args.coords_mode,
                deadman_timeout_s=args.deadman, eventlog=eventlog,
            )
    except (SessionError, RuntimeError, ImportError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    finally:
        if eventlog is not None:
            eventlog.close()


if __name__ == "__main__":
    sys.exit(main())
