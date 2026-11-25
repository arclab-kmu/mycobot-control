import http.client
import json
import sys
import threading
import time

import pytest

from mycobot_control import cli
from mycobot_control.camera import devices, server as camera_server
from mycobot_control.camera.capture import CaptureWorker
from mycobot_control.camera.cli import main as camera_main
from mycobot_control.camera.devices import default_cameras, list_video_devices, parse_camera
from mycobot_control.camera.server import BOUNDARY, CameraWebServer
from mycobot_control.camera.slot import FrameSlot, StreamMeter

JPEG = b"\xff\xd8first\xff\xd9"
JPEG2 = b"\xff\xd8second\xff\xd9"


class Clock:
    def __init__(self, t=0.0):
        self.t = t

    def __call__(self):
        return self.t


class FakeFrame:
    shape = (480, 640, 3)


class FakeSource:
    """Grabs the scripted frames, then reports no frame (an unplugged camera)."""

    def __init__(self, frames):
        self.frames = list(frames)
        self.grabbed = None
        self.decoded = 0
        self.released = False

    def grab(self):
        item = self.frames.pop(0) if self.frames else None
        if isinstance(item, Exception):
            raise item
        self.grabbed = item
        return item is not None

    def retrieve(self):
        self.decoded += 1
        return self.grabbed is not None, self.grabbed

    def release(self):
        self.released = True


def worker(open_source, slot=None, clock=None, encoded=None):
    encoded = [] if encoded is None else encoded
    return CaptureWorker(
        "cam0", parse_camera("test"), open_source, lambda frame, t: encoded.append(t) or JPEG,
        slot or FrameSlot(), clock=clock or Clock(), wall=lambda: 123.0, retry_s=1.0,
    )


# --- slot ------------------------------------------------------------------------------------------


def test_slow_viewer_gets_the_newest_frame_not_a_backlog():
    slot = FrameSlot()
    for i in range(3):
        slot.publish(b"frame%d" % i, float(i), float(i))
    frame = slot.wait_newer(0, timeout=0)
    assert frame.jpeg == b"frame2"
    assert slot.wait_newer(frame.seq, timeout=0) is None


def test_demand_lapses_when_nobody_asks_again():
    slot = FrameSlot()
    assert not slot.in_demand(0.0)
    slot.request(10.0, hold_s=2.0)
    assert slot.in_demand(11.9)
    assert not slot.in_demand(12.1)


def test_meter_rate_and_age():
    meter = StreamMeter()
    for i in range(11):
        meter.on_frame(i * 0.1)
    assert meter.rate_hz(1.0) == pytest.approx(10.0)
    assert meter.age_s(1.25) == pytest.approx(0.25)
    assert meter.rate_hz(5.0) == 0.0  # nothing in the last second
    assert StreamMeter().age_s(0.0) is None


# --- capture worker --------------------------------------------------------------------------------


def test_worker_grabs_always_but_decodes_and_encodes_only_while_watched():
    clock, slot, encoded = Clock(), FrameSlot(), []
    source = FakeSource([FakeFrame()] * 2)
    w = worker(lambda: source, slot, clock, encoded)

    assert w.step() == 0.0
    assert source.decoded == 0 and encoded == [] and slot.latest() is None
    assert w.status(clock())["frames"] == 1  # the camera is still drained

    slot.request(clock())
    assert w.step() == 0.0
    assert len(encoded) == 1 and slot.latest().jpeg == JPEG and slot.latest().t_host == 123.0
    st = w.status(clock())
    assert st["ok"] and st["encoding"] and (st["width"], st["height"]) == (640, 480)


def test_worker_retries_open_and_reopens_a_camera_that_stops():
    sources = []

    def opener():
        if not sources:
            sources.append(None)
            raise OSError("cannot open /dev/video9")
        sources.append(FakeSource([FakeFrame()]))
        return sources[-1]

    clock = Clock()
    w = worker(opener, clock=clock)
    assert w.step() == 1.0
    assert not w.status(clock())["ok"] and w.status(clock())["errors"] == 1

    assert w.step() == 0.0
    assert w.status(clock())["ok"]

    assert w.step() == 1.0  # no frame: released, to be reopened
    assert sources[1].released and not w.status(clock())["ok"]
    assert w.step() == 0.0 and len(sources) == 3


def test_worker_survives_a_grab_that_raises():
    clock = Clock()
    w = worker(lambda: FakeSource([OSError("VIDIOC_DQBUF")]), clock=clock)
    assert w.step() == 1.0
    assert w.status(clock())["errors"] == 1


# --- devices ---------------------------------------------------------------------------------------


def fake_sysfs(root, nodes):
    for number, (name, index) in nodes.items():
        d = root / f"video{number}"
        d.mkdir()
        (d / "name").write_text(name + "\n")
        (d / "index").write_text(f"{index}\n")
    return root


def test_default_cameras_skip_metadata_nodes(tmp_path):
    fake_sysfs(tmp_path, {10: ("C", 0), 0: ("A", 0), 1: ("A", 1), 2: ("B", 0), 3: ("B", 1)})
    found = list_video_devices(str(tmp_path))
    assert [d.path for d in found] == ["/dev/video0", "/dev/video1", "/dev/video2", "/dev/video3", "/dev/video10"]
    assert default_cameras(found) == ["/dev/video0", "/dev/video2"]
    assert list_video_devices(str(tmp_path / "absent")) == []


def test_parse_camera():
    assert parse_camera("1") == parse_camera("/dev/video1")
    assert parse_camera("1").kind == "v4l2"
    assert parse_camera("/dev/v4l/by-id/usb-cam-video-index0").source == "/dev/v4l/by-id/usb-cam-video-index0"
    assert parse_camera("videotestsrc ! appsink").kind == "gstreamer"
    assert parse_camera("test").kind == "test"


# --- server ----------------------------------------------------------------------------------------


@pytest.fixture
def serve():
    started = []

    def start(slots, status=lambda: {"v": 1, "cameras": []}, **kwargs):
        server = CameraWebServer(("127.0.0.1", 0), slots, status, **kwargs)
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
        thread.start()
        started.append((server, thread))
        return server, server.server_address[1]

    yield start
    for server, thread in started:
        server.stop()
        thread.join(timeout=2.0)


def get(port, path, method="GET"):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=3.0)
    conn.request(method, path)
    return conn, conn.getresponse()


def read_part(response):
    assert response.readline().strip() == b"--" + BOUNDARY.encode()
    headers = {}
    while True:
        line = response.readline().strip()
        if not line:
            break
        key, value = line.decode().split(":", 1)
        headers[key.lower()] = value.strip()
    assert headers["content-type"] == "image/jpeg"
    body = response.read(int(headers["content-length"]))
    assert response.readline() == b"\r\n"
    return body


def test_server_routes(serve):
    slot = FrameSlot()
    slot.publish(JPEG, time.monotonic(), 1.0)
    _, port = serve([slot], status=lambda: {"v": 1, "cameras": [{"name": "cam0"}]})

    _, r = get(port, "/")
    assert r.status == 200 and r.getheader("Content-Type").startswith("text/html")
    _, r = get(port, "/status.json")
    assert json.loads(r.read())["cameras"][0]["name"] == "cam0"
    _, r = get(port, "/cam0.jpg")
    assert r.status == 200 and r.read() == JPEG
    for missing in ("/cam1.mjpg", "/cam1.jpg", "/cam0.png", "/nope"):
        _, r = get(port, missing)
        assert r.status == 404, missing


def test_nothing_but_get_is_served(serve):
    _, port = serve([FrameSlot()])
    for method in ("POST", "PUT", "DELETE"):
        _, r = get(port, "/status.json", method)
        assert r.status == 501, method


def test_mjpeg_stream_follows_the_slot_and_keeps_encoding_on(serve):
    slot = FrameSlot()
    slot.publish(JPEG, time.monotonic(), 1.0)
    server, port = serve([slot])

    conn, r = get(port, "/cam0.mjpg")
    assert r.getheader("Content-Type") == f"multipart/x-mixed-replace; boundary={BOUNDARY}"
    assert read_part(r) == JPEG
    assert server.viewers == 1
    assert slot.in_demand(time.monotonic())
    slot.publish(JPEG2, time.monotonic(), 2.0)
    assert read_part(r) == JPEG2
    conn.close()


def wait_until(condition, timeout=3.0):
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.02)
    return True


def test_stream_of_a_silent_camera_is_released_when_the_tab_closes(serve):
    server, port = serve([FrameSlot()])  # never publishes: an unplugged camera
    conn, r = get(port, "/cam0.mjpg")
    assert r.status == 200 and wait_until(lambda: server.viewers == 1)
    conn.close()
    r.close()
    assert wait_until(lambda: server.viewers == 0)


def test_snapshot_switches_encoding_on_when_nobody_streams(serve):
    slot = FrameSlot()
    _, port = serve([slot])
    done = threading.Event()

    def camera():  # stands in for a CaptureWorker: encodes only when asked
        while not done.wait(0.01):
            if slot.in_demand(time.monotonic()):
                slot.publish(JPEG, time.monotonic(), 1.0)
                return

    threading.Thread(target=camera, daemon=True).start()
    try:
        _, r = get(port, "/cam0.jpg")
        assert r.status == 200 and r.read() == JPEG
    finally:
        done.set()


def test_snapshot_without_a_camera_is_503(serve, monkeypatch):
    monkeypatch.setattr(camera_server, "SNAPSHOT_WAIT_S", 0.05)
    _, port = serve([FrameSlot()])
    _, r = get(port, "/cam0.jpg")
    assert r.status == 503


def test_connections_over_the_cap_are_dropped(serve):
    slot = FrameSlot()
    slot.publish(JPEG, time.monotonic(), 1.0)
    _, port = serve([slot], max_connections=1)
    conn, r = get(port, "/cam0.mjpg")
    assert read_part(r) == JPEG  # holds the only connection
    with pytest.raises(ConnectionError):
        get(port, "/status.json")
    conn.close()


# --- cli -------------------------------------------------------------------------------------------


def test_umbrella_dispatches_camera_list(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(devices, "SYSFS_V4L", str(fake_sysfs(tmp_path, {0: ("USB Cam", 0), 1: ("USB Cam", 1)})))
    assert cli.main(["camera", "--list"]) == 0
    out = capsys.readouterr().out
    assert "/dev/video0" in out and "USB Cam" in out


def test_no_camera_found_is_an_error(tmp_path, monkeypatch):
    monkeypatch.setattr(devices, "SYSFS_V4L", str(tmp_path))
    assert camera_main([]) == 2


def test_missing_opencv_stops_before_listening(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "cv2", None)  # makes `import cv2` raise ImportError
    assert camera_main(["--camera", "test"]) == 2
    assert "camera" in capsys.readouterr().err
