"""One thread per camera: grab every frame; decode, stamp, encode and publish only while watched.

Grabbing (dequeueing without decoding) never stops, so the driver's buffers
never fill with old frames. Decoding the camera's MJPG and re-encoding it is
the CPU cost, and on the Jetson that cost is only paid while a browser is
looking.

``CaptureWorker`` takes its source, encoder and clocks as arguments, so the
loop's behaviour (reconnecting, encoding only on demand, health for the page)
is tested without OpenCV or a camera. The OpenCV pieces below it import
``cv2`` and ``numpy`` inside the functions, never at module level.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any, Callable, Dict, Optional, Protocol, Tuple

from mycobot_control.camera.devices import CameraSpec
from mycobot_control.camera.slot import FrameSlot, StreamMeter

RETRY_S = 1.0


class FrameSource(Protocol):
    """cv2.VideoCapture's split read: grab() takes the next frame off the
    driver, retrieve() decodes the grabbed one."""

    def grab(self) -> bool: ...

    def retrieve(self) -> Tuple[bool, Any]: ...

    def release(self) -> None: ...


Encoder = Callable[[Any, float], bytes]  # (frame, t_host) -> JPEG bytes


class CaptureWorker:
    def __init__(
        self,
        name: str,
        spec: CameraSpec,
        open_source: Callable[[], FrameSource],
        encode: Encoder,
        slot: FrameSlot,
        device: Optional[str] = None,
        clock: Callable[[], float] = time.monotonic,
        wall: Callable[[], float] = time.time,
        retry_s: float = RETRY_S,
    ) -> None:
        self.name = name
        self.spec = spec
        self.slot = slot
        self.device = device
        self._open = open_source
        self._encode = encode
        self._clock = clock
        self._wall = wall
        self._retry_s = retry_s
        self._source: Optional[FrameSource] = None
        self._lock = threading.Lock()
        self._meter = StreamMeter()
        self._ok = False
        self._errors = 0
        self._last_error = ""
        self._size: Optional[Tuple[int, int]] = None
        self._negotiated: Optional[str] = None

    def step(self) -> float:
        """One frame (opening the source first if needed). Returns how long to wait before the next."""
        if self._source is None:
            try:
                self._source = self._open()
            except Exception as exc:  # cv2.error, OSError, ImportError: all mean "not now, retry"
                return self._fail(f"open failed: {exc}")
            with self._lock:
                self._negotiated = getattr(self._source, "negotiated", None)
        try:
            if not self._source.grab():
                return self._lost("no frame (unplugged, or another program holds the camera?)")
        except Exception as exc:
            return self._lost(f"grab failed: {exc}")
        now = self._clock()
        with self._lock:
            self._meter.on_frame(now)
            self._ok = True
        if not self.slot.in_demand(now):
            return 0.0

        try:
            ok, frame = self._source.retrieve()
        except Exception as exc:
            return self._lost(f"decode failed: {exc}")
        if not ok or frame is None:
            return self._lost("grabbed frame did not decode")
        shape = getattr(frame, "shape", None)
        if shape is not None and len(shape) >= 2:
            with self._lock:
                self._size = (int(shape[1]), int(shape[0]))
        t_host = self._wall()
        try:
            jpeg = self._encode(frame, t_host)
        except Exception as exc:
            return self._fail(f"encode failed: {exc}")
        self.slot.publish(jpeg, now, t_host)
        return 0.0

    def _lost(self, reason: str) -> float:
        self.close()  # a USB camera that stops returning frames usually has to be reopened
        return self._fail(reason)

    def run(self, stop: threading.Event) -> None:
        try:
            while not stop.is_set():
                wait = self.step()
                if wait > 0:
                    stop.wait(wait)
        finally:
            self.close()

    def close(self) -> None:
        source, self._source = self._source, None
        if source is not None:
            try:
                source.release()
            except Exception:
                pass

    def _fail(self, reason: str) -> float:
        with self._lock:
            self._ok = False
            self._errors += 1
            self._last_error = reason
        return self._retry_s

    def status(self, now: float) -> Dict[str, object]:
        with self._lock:
            age = self._meter.age_s(now)
            return {
                "name": self.name,
                "source": self.spec.source,
                "kind": self.spec.kind,
                "device": self.device,
                "ok": self._ok,
                "negotiated": self._negotiated,
                "width": self._size[0] if self._size else None,
                "height": self._size[1] if self._size else None,
                "hz": round(self._meter.rate_hz(now), 1),
                "age_s": None if age is None else round(age, 3),
                "frames": self._meter.frames,
                "encoding": self.slot.in_demand(now),
                "errors": self._errors,
                "last_error": self._last_error,
            }


# --- OpenCV: imported here, inside functions, and only by the CLI ---------------------------------


def require_opencv() -> Optional[str]:
    """None if cv2 and numpy import, else what to tell the operator."""
    # Before the first import: a camera that is down is retried every second,
    # OpenCV would print a WARN line each time, and last_error already says why.
    os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
    try:
        import cv2  # noqa: F401
        import numpy  # noqa: F401
    except ImportError as exc:
        return f"{exc}. Install the camera extra: uv sync --extra camera (or pip install -e '.[camera]')"
    return None


class OpenCVSource:
    def __init__(self, spec: CameraSpec, width: int, height: int, fps: float, fourcc: str) -> None:
        import cv2

        if spec.kind == "gstreamer":
            # TODO(unverified): pip's opencv-python-headless is built without
            # GStreamer; a CSI camera on the Jetson needs JetPack's system cv2.
            cap = cv2.VideoCapture(spec.source, cv2.CAP_GSTREAMER)
        else:
            cap = cv2.VideoCapture(spec.source, cv2.CAP_V4L2)
            if fourcc:
                # MJPG, not the usual YUYV default: two uncompressed 640x480@30
                # streams (~18 MB/s each) do not fit one USB 2 bus, and the
                # second camera then fails to start. Costs a decode per frame.
                # TODO(unverified): the cameras' supported modes; `v4l2-ctl --list-formats-ext`.
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # newest frame, not a backlog (V4L2 honours it)
        if not cap.isOpened():
            cap.release()
            # EBUSY (another program is streaming it) looks exactly like this.
            raise OSError(f"cannot open {spec.source} (missing, or in use by another program?)")
        self._cap = cap
        code = int(cap.get(cv2.CAP_PROP_FOURCC))
        fmt = "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4)).strip("\x00 ") if code else "?"
        self.negotiated = (
            f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} "
            f"{fmt or '?'} {cap.get(cv2.CAP_PROP_FPS):.0f} fps"
        )

    def grab(self) -> bool:
        return bool(self._cap.grab())

    def retrieve(self) -> Tuple[bool, Any]:
        return self._cap.retrieve()

    def release(self) -> None:
        self._cap.release()


class SyntheticSource:
    """``--camera test``: a moving bar at the requested rate, to try the viewer with no camera."""

    def __init__(
        self,
        width: int,
        height: int,
        fps: float,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        import numpy as np

        self._np = np
        self._w, self._h = width, height
        self._period = 1.0 / fps if fps > 0 else 0.0
        self._clock, self._sleep = clock, sleep
        self._next = clock()
        self._t0 = self._next
        self._ramp = np.linspace(40, 160, width, dtype=np.uint8)
        self.negotiated = f"{width}x{height} synthetic {fps:.0f} fps"

    def grab(self) -> bool:
        now = self._clock()
        if self._next > now:
            self._sleep(self._next - now)
        self._next = max(self._next + self._period, now)
        return True

    def retrieve(self) -> Tuple[bool, Any]:
        np = self._np
        frame = np.empty((self._h, self._w, 3), np.uint8)
        frame[:] = self._ramp[None, :, None]
        x = int(((self._clock() - self._t0) * self._w / 2.0) % self._w)
        frame[:, max(0, x - 8) : x + 8] = (0, 200, 255)
        return True, frame

    def release(self) -> None:
        pass


def jpeg_encoder(label: str, quality: int) -> Encoder:
    """Stamp ``label`` and the capture time onto the frame, then encode it once.

    The stamp is burned into the pixels so that a frozen picture shows a frozen
    clock even in a screenshot, where the page's stale warning is not visible.
    """
    import cv2

    params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]

    def encode(frame: Any, t_host: float) -> bytes:
        stamp = f"{label}  {time.strftime('%H:%M:%S', time.localtime(t_host))}.{int(t_host % 1 * 1000):03d}"
        # A box, not a thick dark outline: Hershey glyphs advance by their
        # thickness, so an outline drawn thicker drifts away from the text.
        (w, h), base = cv2.getTextSize(stamp, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (4, 4), (12 + w, 12 + h + base), (0, 0, 0), -1)
        cv2.putText(frame, stamp, (8, 8 + h), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        ok, buf = cv2.imencode(".jpg", frame, params)
        if not ok:
            raise RuntimeError("cv2.imencode returned False")
        return buf.tobytes()

    return encode
