"""Which ``/dev/video*`` nodes are cameras, and what a ``--camera`` argument names.

Read from sysfs, so ``--list`` and the default choice work without OpenCV.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

SYSFS_V4L = "/sys/class/video4linux"


@dataclass(frozen=True)
class VideoDevice:
    path: str
    name: str
    # Node number within its USB device. UVC puts the capture node at 0 and,
    # on kernels >= 4.16, a metadata node (no frames) at 1, so one webcam is
    # two /dev/video* nodes and "video0 and video1" is usually one camera.
    # TODO(unverified): the Jetson's L4T 4.9 kernel predates metadata nodes,
    # and a CSI camera there is read through GStreamer (nvarguscamerasrc), not
    # as a V4L2 capture node.
    index: int


@dataclass(frozen=True)
class CameraSpec:
    kind: str  # "v4l2", "gstreamer" or "test"
    source: str
    label: str  # ASCII: it is drawn onto the frame and cv2.putText has no other glyphs


def _read(path: Path) -> str:
    try:
        return path.read_text().strip()
    except OSError:
        return ""


def _node_number(path: str) -> int:
    match = re.search(r"(\d+)$", path)
    return int(match.group(1)) if match else -1


def list_video_devices(sysfs: Optional[str] = None) -> List[VideoDevice]:
    root = Path(sysfs or SYSFS_V4L)
    if not root.is_dir():
        return []
    found = []
    for entry in root.iterdir():
        if not re.fullmatch(r"video\d+", entry.name):
            continue
        index = _read(entry / "index")
        found.append(VideoDevice(f"/dev/{entry.name}", _read(entry / "name") or "?", int(index) if index.isdigit() else 0))
    return sorted(found, key=lambda d: _node_number(d.path))


def default_cameras(devices: List[VideoDevice], count: int = 2) -> List[str]:
    return [d.path for d in devices if d.index == 0][:count]


def device_name(path: str, devices: List[VideoDevice]) -> Optional[str]:
    """The driver's name for ``path``, following /dev/v4l/by-id symlinks."""
    real = os.path.realpath(path)
    return next((d.name for d in devices if d.path == real), None)


def parse_camera(text: Union[str, int]) -> CameraSpec:
    spec = str(text).strip()
    if spec == "test":
        return CameraSpec("test", spec, "test")
    if "!" in spec:
        return CameraSpec("gstreamer", spec, "gstreamer")
    if spec.isdigit():
        spec = f"/dev/video{spec}"
    return CameraSpec("v4l2", spec, spec)
