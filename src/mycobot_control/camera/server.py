"""A small read-only HTTP server: the page, one MJPEG stream and snapshot per camera, status.

Every route is a fixed GET. There is no route that writes, so nothing reachable
from a browser can reach the arm (this package's camera side never holds a
backend in the first place). Standard library only; Python 3.8 has
``ThreadingHTTPServer``.
"""

from __future__ import annotations

import json
import re
import select
import socket
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Dict, Iterator, Sequence, Tuple
from urllib.parse import urlsplit

from mycobot_control.camera.page import PAGE_HTML
from mycobot_control.camera.slot import FrameSlot

MAX_CONNECTIONS = 32
SEND_TIMEOUT_S = 3.0  # a viewer that stops reading frees its thread after this
POLL_S = 0.5  # how often an idle stream re-checks for shutdown
SNAPSHOT_FRESH_S = 0.5
SNAPSHOT_WAIT_S = 1.5
BOUNDARY = "mycobotframe"

_CAMERA_ROUTE = re.compile(r"/cam(\d+)\.(mjpg|jpg)")


class CameraWebServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        address: Tuple[str, int],
        slots: Sequence[FrameSlot],
        status: Callable[[], Dict[str, object]],
        max_connections: int = MAX_CONNECTIONS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.slots = list(slots)
        self.status = status
        self.clock = clock
        self.stopping = threading.Event()
        self._connections = threading.BoundedSemaphore(max_connections)
        self._viewers_lock = threading.Lock()
        self.viewers = 0
        super().__init__(address, _Handler)

    # Over the cap, close the socket instead of queueing: a thread per
    # connection is what this server costs, and the Jetson has four cores.
    def process_request(self, request, client_address):  # type: ignore[override]
        if not self._connections.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._connections.release()
            raise

    def process_request_thread(self, request, client_address):  # type: ignore[override]
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._connections.release()

    @contextmanager
    def watching(self) -> Iterator[None]:
        with self._viewers_lock:
            self.viewers += 1
        try:
            yield
        finally:
            with self._viewers_lock:
                self.viewers -= 1

    def stop(self) -> None:
        self.stopping.set()
        self.shutdown()
        self.server_close()


class _Handler(BaseHTTPRequestHandler):
    server: CameraWebServer
    timeout = SEND_TIMEOUT_S  # StreamRequestHandler applies it to the socket
    server_version = "mycobot-camera"

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002 - base class's name
        pass  # one line per MJPEG connection is noise in the operator's terminal

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if path in ("/", "/index.html"):
            self._send(200, "text/html; charset=utf-8", PAGE_HTML.encode("utf-8"))
        elif path == "/status.json":
            self._send(200, "application/json", json.dumps(self.server.status()).encode("utf-8"))
        else:
            match = _CAMERA_ROUTE.fullmatch(path)
            if not match or int(match.group(1)) >= len(self.server.slots):
                self.send_error(404)
                return
            slot = self.server.slots[int(match.group(1))]
            if match.group(2) == "mjpg":
                self._stream(slot)
            else:
                self._snapshot(slot)

    def _send(self, code: int, content_type: str, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _snapshot(self, slot: FrameSlot) -> None:
        now = self.server.clock()
        frame = slot.latest()
        if frame is None or now - frame.t_mono > SNAPSHOT_FRESH_S:
            slot.request(now)  # nobody is streaming, so encoding is off: ask for one
            frame = slot.wait_newer(frame.seq if frame else 0, SNAPSHOT_WAIT_S)
        if frame is None:
            self.send_error(503, "no fresh frame from this camera")
            return
        self._send(200, "image/jpeg", frame.jpeg)

    def _stream(self, slot: FrameSlot) -> None:
        self.send_response(200)
        self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={BOUNDARY}")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        seq = 0
        with self.server.watching():
            try:
                while not self.server.stopping.is_set():
                    slot.request(self.server.clock())
                    frame = slot.wait_newer(seq, POLL_S)
                    if frame is None:
                        if self._viewer_gone():
                            return
                        continue
                    seq = frame.seq
                    head = f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\nContent-Length: {len(frame.jpeg)}\r\n\r\n"
                    self.wfile.write(head.encode("ascii") + frame.jpeg + b"\r\n")
            except OSError:
                return  # the tab closed, or stalled past SEND_TIMEOUT_S

    def _viewer_gone(self) -> bool:
        """True once the browser has closed the connection.

        A failed write would say so too, but a camera that sends no frames
        gives nothing to write: without this, every reload of a page with a
        dead camera would leak a thread and one of MAX_CONNECTIONS.
        """
        try:
            readable, _, _ = select.select([self.connection], [], [], 0)
            return bool(readable) and self.connection.recv(1, socket.MSG_PEEK) == b""
        except OSError:
            return True
