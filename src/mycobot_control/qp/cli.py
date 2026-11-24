"""``mycobot-qp`` / ``mycobot qp``: go to a TCP pose or joint goal through the QP controller.

Dry-run by default: plans the whole command stream, prints a summary (ticks,
peak joint speed / acceleration, TCP speed, final error) and replays it
through a dry-run session so every command passes the same gate and limit
checks the live run will. ``--live`` streams the same plan to the arm with
``ArmSession.command_joints`` at ``--rate``, reading the measured joints for
the tracking check, and stops on Ctrl-C or any error.

The session (``sdk``) is imported inside functions only: ``mycobot_control.qp``
itself must stay importable on a PC without the arm SDK.
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import TYPE_CHECKING, Callable, List, Optional, Sequence, TextIO, Tuple

import numpy as np

from mycobot_control.qp.chunk import ChunkExecutor, TrackingError
from mycobot_control.qp.controller import ControllerConfig, QPController
from mycobot_control.qp.kinematics import Kinematics, pose_error, pose_from_xyzrpy, xyzrpy_from_pose
from mycobot_control.safety.gate import GateClosed
from mycobot_control.safety.limits import LimitViolation, check_joint_target

if TYPE_CHECKING:  # pragma: no cover
    from mycobot_control.sdk.session import ArmSession

#: TODO(unverified): pymycobot speed -> deg/s. The spec sheet's ~120 deg/s at
#: speed 100, assumed linear. Used only to keep the planned joint speed below
#: what the firmware will follow at the chosen ``--speed``.
DEG_S_PER_SPEED_UNIT = 1.2
#: Fraction of that estimate the plan may use.
SPEED_HEADROOM = 0.8
#: "Reached" in the plan.
POS_TOL_MM = 1.0
ROT_TOL_DEG = 1.0
JOINT_TOL_DEG = 0.5
#: Ticks allowed after the nominal duration for the controller to settle.
SETTLE_S = 5.0


def _fmt(values: Sequence[float]) -> str:
    return "[" + " ".join(f"{v:+7.2f}" for v in values) + "]"


def build_parser() -> argparse.ArgumentParser:
    from mycobot_control.sdk.cliargs import add_hardware_args

    p = argparse.ArgumentParser(
        prog="mycobot-qp",
        description="Move to a TCP pose (--to) or joint goal (--joints) through the QP controller. "
                    "Dry-run unless --live.",
    )
    goal = p.add_mutually_exclusive_group(required=True)
    goal.add_argument("--to", nargs=3, type=float, metavar=("X", "Y", "Z"),
                      help="TCP goal, mm, in the firmware coords base frame")
    goal.add_argument("--joints", nargs=6, type=float, metavar="J", help="joint goal, deg")
    p.add_argument("--rpy", nargs=3, type=float, metavar=("RX", "RY", "RZ"),
                   help="TCP orientation goal, deg, firmware convention (default: keep the current one)")
    p.add_argument("--position-only", action="store_true", help="track the position only; orientation is free")
    p.add_argument("--duration", type=float, default=None,
                   help="seconds (default: from the distance, at half the speed limits)")
    p.add_argument("--rate", type=float, default=20.0, help="command rate in Hz (default 20)")
    p.add_argument("--lookahead", type=int, default=2,
                   help="ticks the controller runs ahead of the arm (default 2; TODO(unverified) on this firmware)")
    p.add_argument("--qd-max", type=float, default=None,
                   help="joint speed limit deg/s (default: what --speed is assumed to follow, at most 60)")
    p.add_argument("--j1-weight", type=float, default=1.0, help="joint weight on J1; 20 spares J1 (default 1)")
    p.add_argument("--z-min", type=float, default=None, help="keep the TCP above this z, mm")
    p.add_argument("--max-track-err", type=float, default=12.0,
                   help="stop if a measured joint is this far from its command, deg (default 12)")
    p.add_argument("--read-every", type=int, default=1, help="read the measured joints every N ticks (default 1)")
    p.add_argument("--start", nargs=6, type=float, metavar="J",
                   help="dry-run only: plan from these joint angles instead of the dry-run arm's zero pose")
    add_hardware_args(p)
    return p


def controller_config(args: argparse.Namespace, speed: int) -> ControllerConfig:
    qd_max = args.qd_max
    if qd_max is None:
        qd_max = min(60.0, SPEED_HEADROOM * DEG_S_PER_SPEED_UNIT * speed)
    return ControllerConfig(
        dt=1.0 / args.rate,
        qd_max_deg_s=qd_max,
        joint_weights=(args.j1_weight, 1.0, 1.0, 1.0, 1.0, 1.0),
        position_only=bool(args.position_only),
        z_min_mm=args.z_min,
    )


def make_executor(args: argparse.Namespace, cfg: ControllerConfig, kin: Kinematics,
                  q0: Sequence[float]) -> ChunkExecutor:
    ex = ChunkExecutor(QPController(kin, cfg), q0, lookahead_ticks=args.lookahead,
                       max_track_err_deg=args.max_track_err)
    if args.joints is not None:
        ex.goal_joints(args.joints, args.duration)
    else:
        ex.goal_pose(goal_pose(args, kin, q0), args.duration)
    return ex


def goal_pose(args: argparse.Namespace, kin: Kinematics, q0: Sequence[float]) -> np.ndarray:
    start = xyzrpy_from_pose(kin.fk(q0))
    rpy = args.rpy if args.rpy is not None else start[3:]
    return pose_from_xyzrpy(list(args.to) + list(rpy))


def summarize(Q: np.ndarray, q0: Sequence[float], ex: ChunkExecutor, kin: Kinematics, args: argparse.Namespace,
              statuses: List[str]) -> Tuple[List[str], bool]:
    """Human summary of a planned stream and whether it reaches the goal."""
    dt = ex.cfg.dt
    allq = np.vstack([np.asarray(q0, float)[None], Q])
    dq = np.diff(allq, axis=0) / dt
    ddq = np.diff(np.vstack([np.zeros((1, 6)), dq]), axis=0) / dt
    P = np.array([kin.fk(q)[:3, 3] for q in allq])
    v_tcp = np.linalg.norm(np.diff(P, axis=0), axis=1) / dt if len(P) > 1 else np.zeros(1)
    jv, ja = np.unravel_index(np.argmax(np.abs(dq)), dq.shape)[1], np.unravel_index(np.argmax(np.abs(ddq)), ddq.shape)[1]
    lines = [
        f"plan   : {len(Q)} ticks @ {1 / dt:.0f} Hz = {len(Q) * dt:.2f} s (lookahead {ex.lookahead_ticks} ticks)",
        f"from   : joints {_fmt(q0)}",
        f"         tcp    {_fmt(xyzrpy_from_pose(kin.fk(q0)))}",
        f"peak   : joint {np.abs(dq).max():.1f} deg/s (J{jv + 1}), {np.abs(ddq).max():.0f} deg/s^2 (J{ja + 1}); "
        f"tcp {v_tcp.max():.0f} mm/s",
        f"min z  : {P[:, 2].min():.1f} mm",
    ]
    final = allq[-1]
    if args.joints is not None:
        err = float(np.max(np.abs(np.asarray(args.joints) - final)))
        lines.append(f"goal   : joints {_fmt(args.joints)}")
        lines.append(f"final  : joints {_fmt(final)}  error {err:.2f} deg")
        ok = err <= JOINT_TOL_DEG
    else:
        Tg = goal_pose(args, kin, q0)
        e_p, e_r = pose_error(Tg, kin.fk(final))
        lines.append(f"goal   : tcp    {_fmt(xyzrpy_from_pose(Tg))}")
        lines.append(f"final  : joints {_fmt(final)}  error {e_p:.2f} mm / {e_r:.2f} deg"
                     + ("  (orientation not tracked)" if args.position_only else ""))
        ok = e_p <= POS_TOL_MM and (args.position_only or e_r <= ROT_TOL_DEG)
    bad = [s for s in statuses if s != "optimal"]
    lines.append(f"solver : {len(statuses) - len(bad)}/{len(statuses)} steps optimal"
                 + (f"; {len(bad)} relaxed/braking" if bad else ""))
    return lines, ok


def plan(args: argparse.Namespace, cfg: ControllerConfig, kin: Kinematics,
         q0: Sequence[float]) -> Tuple[np.ndarray, ChunkExecutor, List[str]]:
    ex = make_executor(args, cfg, kin, q0)
    statuses: List[str] = []
    out = []
    max_ticks = ex.remaining_ticks + int(SETTLE_S / cfg.dt) + ex.lookahead_ticks + 1
    for _ in range(max_ticks):
        out.append(ex.tick())
        statuses.append(ex.last_info.status if ex.last_info else "optimal")
        if ex.done:
            break
    return np.array(out).reshape(-1, 6), ex, statuses


def stream(session: "ArmSession", ex: ChunkExecutor, *, read_every: int = 1, status_every_s: float = 1.0,
           max_ticks: Optional[int] = None, out: Optional[TextIO] = None,
           sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic) -> int:
    """Send ``ex``'s command stream to ``session`` at ``1 / ex.cfg.dt`` Hz until done.

    Every command goes through ``session.command_joints`` (gate, joint table,
    20 deg jump check against the measured angles). The gate is opened here
    and always closed on the way out; any error, a tracking fault or Ctrl-C
    calls ``session.halt()`` (gate closed + ``stop()``, torque stays on).

    Returns 0 done, 2 refused/feedback lost, 3 did not finish in ``max_ticks``,
    4 tracking fault or halting arm status, 130 interrupted.
    """
    from mycobot_control.sdk.session import SessionError

    out = out if out is not None else sys.stdout
    dt = ex.cfg.dt
    read_every = max(1, int(read_every))
    status_every = max(1, int(round(status_every_s / dt)))
    n = 0
    session.gate.open("mycobot-qp")
    try:
        reading = session.read_joints()
        t_next = clock()
        while True:
            fresh = None
            if n == 0 or n % read_every == 0:
                if n:
                    reading = session.read_joints()
                fresh = reading.angles
            cmd = ex.tick(fresh)
            session.command_joints(cmd, measured=reading)
            n += 1
            if n % status_every == 0:
                st = session.read_status()
                if st is not None and st.should_halt:
                    session.halt(f"arm status {st.error_name}")
                    print(f"arm reports {st.describe()}; stopped", file=out)
                    return 4
            if ex.done:
                break
            if max_ticks is not None and n >= max_ticks:
                session.halt("qp stream did not finish")
                print(f"not finished after {n} ticks; stopped", file=out)
                return 3
            t_next += dt
            delay = t_next - clock()
            if delay > 0:
                sleep(delay)
            elif delay < -dt:
                t_next = clock()  # fell behind: carry on from now rather than burst
    except KeyboardInterrupt:
        session.halt("interrupted")
        print("interrupted; stop() sent", file=out)
        return 130
    except TrackingError as exc:
        session.halt("tracking error")
        print(f"tracking error: {exc}", file=out)
        return 4
    except (GateClosed, LimitViolation, SessionError, ValueError) as exc:
        session.halt(str(exc))
        print(f"refused: {exc}; stop() sent", file=out)
        return 2
    finally:
        session.gate.close("mycobot-qp finished")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    from mycobot_control.sdk.backend import DryRunBackend
    from mycobot_control.sdk.cliargs import config_from_args, confirm_live, preflight
    from mycobot_control.sdk.session import ArmSession, SessionError

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.joints is not None and (args.rpy is not None or args.position_only):
        parser.error("--rpy / --position-only go with --to, not --joints")
    if args.rate <= 0 or args.read_every < 1 or args.lookahead < 0:
        parser.error("--rate must be positive, --read-every >= 1, --lookahead >= 0")
    config = config_from_args(args)
    if args.start is not None and config.live:
        parser.error("--start is for dry runs; live runs start from the measured joints")
    if args.joints is not None:
        try:
            check_joint_target(args.joints)
        except LimitViolation as exc:
            print(f"refused: {exc}", file=sys.stderr)
            return 2
    problem = preflight(config)
    if problem:
        print(problem, file=sys.stderr)
        return 2
    cfg = controller_config(args, config.speed)
    kin = Kinematics()
    what = f"joints {_fmt(args.joints)}" if args.joints is not None else f"tcp to {_fmt(args.to)}"
    if not confirm_live(config, f"QP move, {what}", args.assume_yes):
        return 1
    backend = None if config.live else DryRunBackend(joint_angles=list(args.start or [0.0] * 6))
    try:
        with ArmSession(config, backend=backend) as session:
            q0 = session.read_joints().angles
            Q, ex, statuses = plan(args, cfg, kin, q0)
            lines, ok = summarize(Q, q0, ex, kin, args, statuses)
            print("\n".join(lines))
            print(f"limits : qd_max {np.max(cfg.vec(cfg.qd_max_deg_s)):.0f} deg/s, "
                  f"qdd_max {np.max(cfg.vec(cfg.qdd_max_deg_s2)):.0f} deg/s^2, speed {config.speed}")
            if not ok:
                print("goal not reached in the plan (unreachable, another IK branch, or a limit); "
                      "nothing commanded")
                return 3
            if config.read_only:
                print("read-only: plan shown, nothing commanded")
                return 0
            ex = make_executor(args, cfg, kin, q0)
            live = config.live
            code = stream(session, ex, read_every=args.read_every,
                          max_ticks=len(Q) + int(SETTLE_S / cfg.dt),
                          sleep=time.sleep if live else (lambda s: None))
            if code == 0:
                if live:
                    final = session.read_joints().angles
                    print(f"done   : measured {_fmt(final)}")
                else:
                    print(f"dry-run: {ex.ticks} commands passed the gate and limit checks; nothing moved")
            return code
    except (SessionError, RuntimeError, ImportError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
