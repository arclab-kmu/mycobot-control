"""Exclusive ownership of the arm's serial port.

The vendor ROS 2 nodes (``mycobot_280jn/listen_real_service.py``) serialise
access with an ``fcntl`` lock on ``/tmp/mycobot_lock``. Two processes writing
the same UART interleave their frames, and each then reads the other's
replies. The vendor takes the lock around each call (non-blocking, retried for
up to 50 s); this package takes the same lock and holds it for the whole
session. So while a mycobot-* command runs, a vendor node's calls wait and
then fail, and a mycobot-* command started next to a busy vendor node refuses
to start - refusing each other instead of corrupting each other.

Only the serial transport takes it: over a socket the server on the Jetson owns
the port, and the lock file on this host would protect nothing.
"""

from __future__ import annotations

import os
from typing import Optional

try:  # POSIX only.
    import fcntl

    HAVE_FCNTL = True
except ImportError:  # pragma: no cover - Windows
    HAVE_FCNTL = False

VENDOR_LOCK_PATH = "/tmp/mycobot_lock"


class PortBusy(RuntimeError):
    """Another process holds the arm's lock."""


class PortLock:
    """A non-blocking exclusive lock. Fails fast rather than queueing behind another owner."""

    def __init__(self, path: str = VENDOR_LOCK_PATH) -> None:
        self.path = path
        self._fd: Optional[int] = None

    @property
    def held(self) -> bool:
        return self._fd is not None

    def acquire(self) -> None:
        if not HAVE_FCNTL or self._fd is not None:
            return
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o666)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            os.close(fd)
            raise PortBusy(
                f"{self.path} is held by another process - probably a vendor ROS 2 node "
                "(listen_real_service, slider_control) or another mycobot-* command. "
                "Only one process may own the arm's serial port."
            ) from exc
        self._fd = fd

    def release(self) -> None:
        if self._fd is None:
            return
        try:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
        finally:
            os.close(self._fd)
            self._fd = None

    def __enter__(self) -> "PortLock":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()
