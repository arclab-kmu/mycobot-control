"""``mycobot-camera``: show the cameras live in a browser. Read-only; never touches the arm.

Run it on the machine the cameras are plugged into. It binds to loopback by
default; from another machine, tunnel (``ssh -L 8082:127.0.0.1:8082 <host>``)
rather than ``--bind 0.0.0.0``, which shows the cameras to anyone who can reach
the host, with no login.
"""

from __future__ import annotations

import argparse
import getpass
import ipaddress
import socket
import sys
import threading
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from mycobot_control.camera.capture import (
    CaptureWorker,
    FrameSource,
    OpenCVSource,
    SyntheticSource,
    jpeg_encoder,
    require_opencv,
)
from mycobot_control.camera.devices import (
    CameraSpec,
    VideoDevice,
    default_cameras,
    device_name,
    list_video_devices,
    parse_camera,
)
from mycobot_control.camera.server import CameraWebServer
from mycobot_control.camera.slot import FrameSlot

DEFAULT_PORT = 8082  # 8080 is the cuRobo/viser viewer, 8081 nero-camera
STATUS_VERSION = 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mycobot-camera",
        description="Read-only browser viewer for the cameras (MJPEG). Never touches the arm.",
    )
    parser.add_argument(
        "--camera", action="append", metavar="SPEC",
        help="0 or /dev/videoN, a GStreamer pipeline (contains '!'), or 'test' for a synthetic pattern. "
        "Repeat once per camera. Default: the first two capture nodes found (see --list)",
    )
    parser.add_argument("--list", action="store_true", help="list /dev/video* nodes and exit")
    parser.add_argument("--width", type=int, default=640, help="requested width (default 640)")
    parser.add_argument("--height", type=int, default=480, help="requested height (default 480)")
    parser.add_argument("--fps", type=float, default=30.0, help="requested frame rate (default 30)")
    parser.add_argument(
        "--fourcc", default="MJPG",
        help="V4L2 pixel format to request (default MJPG, so two cameras fit one USB 2 bus; '' = driver default)",
    )
    parser.add_argument("--quality", type=int, default=80, choices=range(1, 101), metavar="1-100",
                        help="JPEG quality (default 80)")
    parser.add_argument("--bind", default="127.0.0.1", help="address to listen on (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"port (default {DEFAULT_PORT})")
    return parser


def _print_devices(devices: List[VideoDevice]) -> None:
    if not devices:
        print("no /dev/video* nodes")
    for d in devices:
        role = "capture" if d.index == 0 else f"node {d.index} (metadata, no frames)"
        print(f"{d.path:<14} {role:<34} {d.name}")


def _opener(spec: CameraSpec, args: argparse.Namespace) -> Callable[[], FrameSource]:
    if spec.kind == "test":
        return lambda: SyntheticSource(args.width, args.height, args.fps)
    return lambda: OpenCVSource(spec, args.width, args.height, args.fps, args.fourcc)


def _is_loopback(host: str) -> bool:
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return host == "localhost"


def _report(workers: Sequence[CaptureWorker], seen: Dict[str, Tuple[bool, str]]) -> None:
    """Print each camera's state when it changes, so a dead camera shows in the terminal too."""
    now = time.monotonic()
    for w in workers:
        st = w.status(now)
        ok, err = bool(st["ok"]), str(st["last_error"])
        if not ok and not err:
            continue  # not tried yet
        key = (ok, "" if ok else err)
        if seen.get(w.name) == key:
            continue
        seen[w.name] = key
        if ok:
            # flush: this runs for hours, often with stdout piped to a log
            print(f"{w.name}: streaming {st['negotiated'] or ''}".rstrip(), flush=True)
        else:
            print(f"{w.name}: {err} (retrying)", file=sys.stderr)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    devices = list_video_devices()
    if args.list:
        _print_devices(devices)
        return 0
    if args.camera:
        specs = [parse_camera(c) for c in args.camera]
    else:
        specs = [parse_camera(p) for p in default_cameras(devices)]
        if not specs:
            print("error: no camera found under /dev/video*. Name one with --camera (or --camera test).",
                  file=sys.stderr)
            return 2
    problem = require_opencv()
    if problem:
        print(f"error: {problem}", file=sys.stderr)
        return 2

    slots = [FrameSlot() for _ in specs]
    workers = []
    for i, (spec, slot) in enumerate(zip(specs, slots)):
        name = f"cam{i}"
        device = device_name(spec.source, devices) if spec.kind == "v4l2" else None
        workers.append(CaptureWorker(name, spec, _opener(spec, args), jpeg_encoder(f"{name} {spec.label}", args.quality),
                                     slot, device=device))
    host = socket.gethostname()
    server: Optional[CameraWebServer] = None

    def status() -> Dict[str, object]:
        now = time.monotonic()
        return {
            "v": STATUS_VERSION,
            "host": host,
            "t_host": time.time(),
            "viewers": server.viewers if server else 0,
            "cameras": [w.status(now) for w in workers],
        }

    try:
        server = CameraWebServer((args.bind, args.port), slots, status)
    except OSError as exc:
        print(f"error: cannot listen on {args.bind}:{args.port}: {exc}", file=sys.stderr)
        return 2

    stop = threading.Event()
    threads = [threading.Thread(target=w.run, args=(stop,), name=w.name, daemon=True) for w in workers]
    threads.append(threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.5}, daemon=True))
    for t in threads:
        t.start()
    for w in workers:
        print(f"{w.name}: {w.spec.source}" + (f"  ({w.device})" if w.device else ""), flush=True)
    port = server.server_address[1]
    print(f"serving http://{args.bind}:{port}/  (read-only; Ctrl-C to stop)", flush=True)
    if _is_loopback(args.bind):
        print(f"from another machine: ssh -L {port}:127.0.0.1:{port} {getpass.getuser()}@{host}"
              f"  then open http://localhost:{port}/", flush=True)
    else:
        print("warning: not loopback - anyone who can reach this host can watch the cameras (no login)",
              file=sys.stderr)

    seen: Dict[str, Tuple[bool, str]] = {}
    try:
        while not stop.wait(1.0):
            _report(workers, seen)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        server.stop()
        for t in threads:
            t.join(timeout=2.0)  # a worker blocked in cv2.read() is a daemon; don't hang on it
    return 0


if __name__ == "__main__":
    sys.exit(main())
