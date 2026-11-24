"""``mycobot <command> ...``: one entry point that dispatches to the individual CLIs.

It adds nothing: every flag, including ``--live`` and its confirmation, is the
sub-command's own.
"""

from __future__ import annotations

import sys
from typing import Callable, Dict, Optional, Sequence


def _commands() -> Dict[str, Callable[[Optional[Sequence[str]]], int]]:
    from mycobot_control.record.cli import main as record_main
    from mycobot_control.sdk.info import main as info_main
    from mycobot_control.sdk.jointmove import main as joints_main
    from mycobot_control.teleop.keyboard import main as teleop_main

    return {"teleop": teleop_main, "info": info_main, "joints": joints_main, "record": record_main}


def _qp_main(argv: Optional[Sequence[str]] = None) -> int:
    # Imported on use: the QP stack needs numpy (the `qp` extra) and the other
    # commands must keep working without it.
    try:
        from mycobot_control.qp.cli import main as qp_main
    except ImportError as exc:
        print(f"mycobot qp needs numpy: install the qp extra ({exc})", file=sys.stderr)
        return 2
    return qp_main(argv)


USAGE = """usage: mycobot <command> [args]

commands:
  teleop   keyboard joint / Cartesian teleop      (mycobot-teleop-keyboard)
  info     read-only report of the arm's state    (mycobot-info)
  joints   move to a joint target in degrees      (mycobot-move-joints)
  record   read-only fixed-rate CSV recorder      (mycobot-record)
  qp       TCP / joint goal via the QP controller (mycobot-qp; needs the qp extra)

Every command is a dry run unless given --live. `mycobot <command> -h` for its flags."""


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help"):
        print(USAGE)
        return 0
    commands = _commands()
    commands["qp"] = _qp_main
    name, rest = args[0], args[1:]
    if name not in commands:
        print(f"unknown command {name!r}\n\n{USAGE}", file=sys.stderr)
        return 2
    return commands[name](rest)


if __name__ == "__main__":
    sys.exit(main())
