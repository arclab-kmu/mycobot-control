"""``mycobot-record``: sample the arm at a fixed rate. Cannot command.

``--live`` always builds a read-only session - there is no flag that lets the
recorder transmit. Commanding while recording is the teleop tool's job, and its
``--log`` CSV shares this file's ``t_host`` clock.
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence

from mycobot_control.record.recorder import default_path, record
from mycobot_control.sdk.cliargs import add_hardware_args, config_from_args, preflight
from mycobot_control.sdk.session import ArmSession


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mycobot-record", description="Read-only fixed-rate recorder.")
    parser.add_argument("-o", "--output", default=None, help="CSV path (default recordings/record-*.csv)")
    parser.add_argument("--rate", type=float, default=10.0, help="requested Hz (default 10)")
    parser.add_argument("--duration", type=float, default=None, help="seconds; default until Ctrl-C")
    parser.add_argument("--no-coords", action="store_true", help="skip get_coords() to double the joint rate")
    parser.add_argument("--gripper", action="store_true", help="also read get_gripper_value() each row")
    parser.add_argument("--status-every", type=int, default=10, help="read status every N rows (0 = never)")
    add_hardware_args(parser, can_command=False)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    config = config_from_args(args, force_read_only=True)
    problem = preflight(config)
    if problem:
        print(problem, file=sys.stderr)
        return 2
    path = args.output or default_path()
    try:
        with ArmSession(config) as session:
            print(f"recording {session.backend.describe()} -> {path}  (Ctrl-C to stop)")
            stats = record(
                session.backend, path, rate_hz=args.rate, duration_s=args.duration,
                coords=not args.no_coords, gripper=args.gripper, status_every=args.status_every,
                meta={"backend": session.backend.describe(), "live": config.live},
            )
    except (RuntimeError, ImportError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    hz = f"{stats.achieved_hz:.1f}" if stats.achieved_hz else "n/a"
    print(f"{stats.rows} rows, achieved {hz} Hz, {stats.overruns} overruns, {stats.joint_misses} joint misses")
    return 0


if __name__ == "__main__":
    sys.exit(main())
