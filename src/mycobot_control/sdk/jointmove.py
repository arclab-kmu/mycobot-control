"""``mycobot-move-joints``: go to a joint target (deg), in small verified hops.

The target is split into waypoints no more than ``--max-step`` apart. Each
waypoint is commanded only after the previous one was *measured* as reached,
so a stalled or blocked arm stops the sequence instead of being handed the rest
of it. Targets outside the joint table (minus its margin) are refused, not
clamped - an explicit move should go where it says or nowhere.

There is deliberately no ``--home`` preset: the all-zero pose the vendor's
``go_home()`` sends stands the arm straight up, and whether the path there is
clear depends on the tool and the table, which this program cannot see.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from typing import List, Optional, Sequence

from mycobot_control.safety.limits import LimitViolation, check_joint_target, joint_vector
from mycobot_control.sdk.cliargs import add_hardware_args, config_from_args, confirm_live, preflight
from mycobot_control.sdk.session import ArmSession, SessionError

DEFAULT_MAX_STEP_DEG = 10.0
DEFAULT_TOLERANCE_DEG = 2.0
DEFAULT_WAYPOINT_TIMEOUT_S = 5.0


def plan_waypoints(start: Sequence[float], target: Sequence[float], max_step_deg: float) -> List[List[float]]:
    """Evenly spaced waypoints from ``start`` (exclusive) to ``target`` (inclusive)."""
    if max_step_deg <= 0.0:
        raise ValueError("max_step_deg must be positive")
    s, t = joint_vector(start), joint_vector(target)
    worst = max(abs(b - a) for a, b in zip(s, t))
    n = max(1, int(math.ceil(worst / max_step_deg)))
    return [[a + (b - a) * (k / n) for a, b in zip(s, t)] for k in range(1, n + 1)]


def move(
    session: ArmSession,
    target: Sequence[float],
    max_step_deg: float = DEFAULT_MAX_STEP_DEG,
    tolerance_deg: float = DEFAULT_TOLERANCE_DEG,
    waypoint_timeout_s: float = DEFAULT_WAYPOINT_TIMEOUT_S,
    out=None,
    sleep=time.sleep,
) -> int:
    out = out if out is not None else sys.stdout
    goal = check_joint_target(target)
    start = session.read_joints()
    waypoints = plan_waypoints(start.angles, goal, max_step_deg)
    print(f"from {_fmt(start.angles)}\n  to {_fmt(goal)}  in {len(waypoints)} waypoint(s)", file=out)
    if session.config.read_only:
        # Commands would be dropped, so "waiting to arrive" would only time out.
        print("read-only: plan shown, nothing commanded", file=out)
        return 0
    session.gate.open("mycobot-move-joints")
    try:
        for i, wp in enumerate(waypoints, 1):
            measured = session.read_joints()
            session.command_joints(wp, measured=measured)
            deadline = time.monotonic() + waypoint_timeout_s
            while True:
                sleep(0.05)
                now = session.read_joints()
                err = max(abs(a - b) for a, b in zip(now.angles, wp))
                if err <= tolerance_deg:
                    break
                if time.monotonic() > deadline:
                    session.halt("waypoint timeout")
                    print(f"waypoint {i} not reached within {waypoint_timeout_s:.1f} s "
                          f"(worst error {err:.1f} deg); stopped", file=out)
                    return 3
            print(f"  {i}/{len(waypoints)} {_fmt(now.angles)}", file=out)
    except KeyboardInterrupt:
        session.halt("interrupted")
        print("interrupted; stop() sent", file=out)
        return 130
    finally:
        session.gate.close("move finished")
    return 0


def _fmt(values: Sequence[float]) -> str:
    return "[" + " ".join(f"{v:+7.2f}" for v in values) + "]"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mycobot-move-joints",
                                     description="Move to a joint target in degrees. Dry-run unless --live.")
    parser.add_argument("angles", nargs=6, type=float, metavar="J", help="six joint angles, degrees")
    parser.add_argument("--max-step", type=float, default=DEFAULT_MAX_STEP_DEG, help="deg between waypoints")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE_DEG, help="deg, 'reached' threshold")
    parser.add_argument("--timeout", type=float, default=DEFAULT_WAYPOINT_TIMEOUT_S, help="s per waypoint")
    add_hardware_args(parser)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    config = config_from_args(args)
    try:
        check_joint_target(args.angles)
    except LimitViolation as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    problem = preflight(config)
    if problem:
        print(problem, file=sys.stderr)
        return 2
    if not confirm_live(config, f"move joints to {_fmt(args.angles)}", args.assume_yes):
        return 1
    try:
        with ArmSession(config) as session:
            return move(session, args.angles, args.max_step, args.tolerance, args.timeout)
    except (SessionError, LimitViolation, RuntimeError, ImportError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
