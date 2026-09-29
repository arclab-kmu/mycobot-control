"""``mycobot-info``: read what the arm reports, and nothing else by default.

``--live`` always opens a read-only session, so this command cannot move the
arm. ``--clear-errors`` is the one exception and says so: it sends
``clear_error_information()`` (not a motion) after its own confirmation, by
reaching past the read-only wrapper for that single call.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Callable, Optional, Sequence

from mycobot_control.safety.limits import JOINT_LIMITS_DEG, N_JOINTS
from mycobot_control.sdk.cliargs import add_hardware_args, config_from_args, confirm_live, preflight
from mycobot_control.sdk.session import ArmSession, SessionError


def _call(fn: Callable[[], Any]) -> Any:
    try:
        return fn()
    except Exception as exc:
        return f"<error: {exc}>"


def report(session: ArmSession, out=None) -> None:
    out = out if out is not None else sys.stdout

    def say(msg: str) -> None:
        print(msg, file=out)

    say(f"backend      : {session.backend.describe()}")
    raw = getattr(session.backend, "raw", None) or getattr(getattr(session.backend, "inner", None), "raw", None)
    if raw is not None:
        say(f"system ver   : {_call(raw.get_system_version)}")
        say(f"modify ver   : {_call(raw.get_modify_version)}")
        say(f"fresh mode   : {_call(raw.get_fresh_mode)}")
    status = session.read_status()
    say(f"status       : {status.describe() if status else 'unavailable'}")
    try:
        joints = session.read_joints()
        latency = f"   ({joints.latency_s * 1000:.1f} ms round trip)" if joints.latency_s is not None else ""
        say("joints (deg) : " + " ".join(f"{a:+8.2f}" for a in joints.angles) + latency)
    except SessionError as exc:
        say(f"joints       : {exc}")
    try:
        coords = session.read_coords()
        say("coords       : " + " ".join(f"{c:+8.2f}" for c in coords.coords) + "   (mm, deg)")
    except SessionError as exc:
        say(f"coords       : {exc}")
    say(f"gripper      : {_call(session.backend.get_gripper_value)}")
    say("")
    say("joint limits : table (deg)          controller (deg)")
    for i in range(N_JOINTS):
        lo, hi = JOINT_LIMITS_DEG[i]
        ctrl = ""
        if raw is not None:
            ctrl = f"{_call(lambda: raw.get_joint_min_angle(i + 1))} .. {_call(lambda: raw.get_joint_max_angle(i + 1))}"
        say(f"  J{i + 1}         : {lo:+7.1f} .. {hi:+7.1f}      {ctrl}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mycobot-info", description="Read-only report of the arm's state.")
    parser.add_argument("--clear-errors", action="store_true",
                        help="send clear_error_information() after the report (needs --live, asks first)")
    add_hardware_args(parser, can_command=False)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    config = config_from_args(args, force_read_only=True)
    problem = preflight(config)
    if problem:
        print(problem, file=sys.stderr)
        return 2
    if args.clear_errors:
        if not args.live:
            print("--clear-errors needs --live", file=sys.stderr)
            return 2
        if not confirm_live(config, "read the arm, then clear_error_information()", args.assume_yes):
            return 1
    try:
        with ArmSession(config) as session:
            report(session)
            if args.clear_errors:
                # The session stays read-only; this one call reaches past the
                # wrapper on purpose, so nothing else (fresh mode, motion) can.
                inner = getattr(session.backend, "inner", session.backend)
                inner.clear_errors()
                status = session.read_status()
                print(f"cleared; status now: {status.describe() if status else 'unavailable'}")
    except (RuntimeError, ImportError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
