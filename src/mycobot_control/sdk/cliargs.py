"""Hardware arguments and the ``--live`` confirmation, shared by every CLI.

One place decides what ``--live`` means, so no CLI can grow a quieter route to
the arm than the others.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional, TextIO

from mycobot_control.safety.limits import DEFAULT_SPEED, FAST_SPEED
from mycobot_control.sdk.backend import DEFAULT_NETPORT, JN_BAUD, JN_SERIAL_PORT
from mycobot_control.sdk.session import ArmConfig


def add_hardware_args(parser: argparse.ArgumentParser, *, can_command: bool = True) -> None:
    hw = parser.add_argument_group("hardware (dry-run unless --live)")
    hw.add_argument("--live", action="store_true", help="talk to the real arm. Without this nothing is sent.")
    if can_command:
        hw.add_argument(
            "--read-only",
            action="store_true",
            help="connect and read, but drop every command (implies --live)",
        )
        hw.add_argument(
            "--power-on",
            action="store_true",
            help="call power_on() at start - only needed after the servos were released",
        )
        hw.add_argument(
            "--speed",
            type=int,
            default=DEFAULT_SPEED,
            help=f"pymycobot speed 1-100 (default {DEFAULT_SPEED}; above {FAST_SPEED} needs --allow-fast)",
        )
        hw.add_argument("--allow-fast", action="store_true", help=f"permit --speed above {FAST_SPEED}")
    hw.add_argument(
        "--transport",
        choices=("serial", "socket"),
        default="serial",
        help="serial: run on the Jetson (default). socket: run on a PC, talk to a server on the Jetson.",
    )
    hw.add_argument("--port", default=JN_SERIAL_PORT, help=f"serial device (default {JN_SERIAL_PORT})")
    hw.add_argument("--baud", type=int, default=JN_BAUD, help=f"serial baud rate (default {JN_BAUD})")
    hw.add_argument("--host", default=None, help="Jetson IP, for --transport socket")
    hw.add_argument("--netport", type=int, default=DEFAULT_NETPORT, help=f"socket port (default {DEFAULT_NETPORT})")
    hw.add_argument("--assume-yes", action="store_true", help="skip the confirmation --live asks for")


def config_from_args(args: argparse.Namespace, *, force_read_only: bool = False) -> ArmConfig:
    """``--read-only`` implies ``--live``. ``force_read_only`` (the recorder) does not:
    it only guarantees that *if* the session is live, it cannot command."""
    ro_flag = bool(getattr(args, "read_only", False))
    live = bool(args.live) or ro_flag
    read_only = live and (force_read_only or ro_flag)
    speed = getattr(args, "speed", DEFAULT_SPEED)
    if speed > FAST_SPEED and not getattr(args, "allow_fast", False):
        raise SystemExit(f"--speed {speed} is above {FAST_SPEED}; pass --allow-fast if you mean it")
    return ArmConfig(
        live=live,
        read_only=read_only,
        power_on=bool(getattr(args, "power_on", False)),
        transport=args.transport,
        port=args.port,
        baud=args.baud,
        host=args.host,
        netport=args.netport,
        speed=speed,
    )


def preflight(config: ArmConfig) -> Optional[str]:
    """A reason the live link cannot work, found before importing anything; else ``None``."""
    if not config.live or config.transport != "serial":
        return None
    if not os.path.exists(config.port):
        return (
            f"{config.port} does not exist. The myCobot 280 JN's UART is only present on the "
            "Jetson itself; from a PC use --transport socket --host <jetson-ip> with the vendor "
            "socket server running on the Jetson."
        )
    if not os.access(config.port, os.R_OK | os.W_OK):
        return f"no read/write permission on {config.port}. Add your user to the dialout group and log in again."
    return None


def confirm_live(config: ArmConfig, action: str, assume_yes: bool,
                 stdin: Optional[TextIO] = None, out: Optional[TextIO] = None) -> bool:
    """Ask the operator to type ``yes`` before a live session. Refuses on a non-TTY."""
    stdin = stdin if stdin is not None else sys.stdin
    out = out if out is not None else sys.stdout
    if not config.live:
        return True
    mode = "READ-ONLY (commands dropped)" if config.read_only else "COMMANDS WILL MOVE THE ARM"
    print(
        f"\n  LIVE: {action}\n"
        f"  link : {config.describe_target()}\n"
        f"  mode : {mode}\n"
        f"  speed: {config.speed}\n"
        "  Keep the arm's power within reach. Clear the workspace.\n",
        file=out,
        flush=True,
    )
    if assume_yes:
        return True
    if not stdin.isatty():
        print("refusing: --live needs an interactive 'yes' (or --assume-yes)", file=out)
        return False
    print("type 'yes' to continue: ", end="", file=out, flush=True)
    return stdin.readline().strip().lower() == "yes"
