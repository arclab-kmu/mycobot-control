"""The hand-off between one camera's capture thread and the browsers watching it.

A slot holds only the newest JPEG. A viewer that falls behind gets the newest
frame when it next asks and skips the ones in between; nothing queues, so a
slow tab can neither delay the other viewers nor grow memory on the Jetson.

Encoding is the expensive step, so it happens once per captured frame however
many viewers there are, and only while someone has asked recently
(``request`` / ``in_demand``). Capture itself never pauses: a camera that is
not read keeps filling its driver buffers, and the first frame after a pause
would be seconds old.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional

RATE_WINDOW_S = 1.0
DEMAND_HOLD_S = 2.0


@dataclass(frozen=True)
class Frame:
    seq: int
    jpeg: bytes
    t_mono: float  # capture time, on the clock shared with the server
    t_host: float  # capture time as time.time(), the clock the recorders use


class FrameSlot:
    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._frame: Optional[Frame] = None
        self._seq = 0
        self._demand_until = float("-inf")

    def publish(self, jpeg: bytes, t_mono: float, t_host: float) -> Frame:
        with self._cond:
            self._seq += 1
            self._frame = Frame(self._seq, jpeg, t_mono, t_host)
            self._cond.notify_all()
            return self._frame

    def latest(self) -> Optional[Frame]:
        with self._cond:
            return self._frame

    def wait_newer(self, seq: int, timeout: float) -> Optional[Frame]:
        """The newest frame after ``seq``, or None if none arrives within ``timeout``."""
        with self._cond:
            self._cond.wait_for(lambda: self._frame is not None and self._frame.seq > seq, timeout)
            frame = self._frame
            return frame if frame is not None and frame.seq > seq else None

    def request(self, now: float, hold_s: float = DEMAND_HOLD_S) -> None:
        with self._cond:
            self._demand_until = max(self._demand_until, now + hold_s)

    def in_demand(self, now: float) -> bool:
        with self._cond:
            return now <= self._demand_until


class StreamMeter:
    """Capture rate over the last second and age of the newest frame. Not locked; the owner locks."""

    def __init__(self, window_s: float = RATE_WINDOW_S) -> None:
        self._window_s = window_s
        self._recent: Deque[float] = deque()
        self.frames = 0

    def on_frame(self, t: float) -> None:
        self.frames += 1
        self._recent.append(t)
        while t - self._recent[0] > self._window_s:
            self._recent.popleft()

    def rate_hz(self, now: float) -> float:
        recent = [t for t in self._recent if now - t <= self._window_s]
        if len(recent) < 2:
            return 0.0
        span = recent[-1] - recent[0]
        return (len(recent) - 1) / span if span > 0 else 0.0

    def age_s(self, now: float) -> Optional[float]:
        return max(0.0, now - self._recent[-1]) if self._recent else None
