"""The sampler behind ``mycobot-record``.

Rows carry joints in degrees (the SDK's unit) *and* radians (the unit of the
URDF and the arc-mycobot Isaac Lab tasks), so a recording can be compared with
a sim rollout without a conversion step that could go wrong silently.

Reads on this arm are request/response over one serial line, so each field
costs a round trip and the achievable rate is set by the link, not by
``--rate``. Every row records its own read latency and the loop records
overruns in the sidecar, so a recording says what rate it actually achieved.
TODO(unverified): the achievable rate on the JN's UART at 1 Mbaud.
"""

from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

from mycobot_control.safety.limits import N_JOINTS
from mycobot_control.sdk.backend import ArmBackend

JOINT_DEG = [f"j{i}_deg" for i in range(1, N_JOINTS + 1)]
JOINT_RAD = [f"j{i}_rad" for i in range(1, N_JOINTS + 1)]
COORDS = ["x_mm", "y_mm", "z_mm", "rx_deg", "ry_deg", "rz_deg"]
FIELDS = (
    ["t_host", "t_rel", "joints_ok"] + JOINT_DEG + JOINT_RAD + ["joints_latency_s", "coords_ok"]
    + COORDS + ["coords_latency_s", "gripper", "error_code", "power_on"]
)

SCHEMA = "mycobot-record/1"


def default_path(directory: Union[str, Path] = "recordings") -> Path:
    return Path(directory) / time.strftime("record-%Y%m%d-%H%M%S.csv")


@dataclass
class RecordStats:
    rows: int = 0
    joint_misses: int = 0
    coord_misses: int = 0
    overruns: int = 0
    started: float = 0.0
    ended: float = 0.0
    extra: Dict[str, object] = field(default_factory=dict)

    @property
    def achieved_hz(self) -> Optional[float]:
        span = self.ended - self.started
        return (self.rows - 1) / span if self.rows > 1 and span > 0 else None


def sample_row(backend: ArmBackend, t0: float, coords: bool, gripper: bool, status: bool) -> Dict[str, object]:
    now = time.time()
    row: Dict[str, object] = {name: "" for name in FIELDS}
    row["t_host"] = f"{now:.6f}"
    row["t_rel"] = f"{now - t0:.6f}"
    joints = backend.get_joint_state()
    row["joints_ok"] = int(joints is not None)
    if joints is not None:
        for i, a in enumerate(joints.angles):
            row[JOINT_DEG[i]] = f"{a:.3f}"
            row[JOINT_RAD[i]] = f"{math.radians(a):.6f}"
        row["joints_latency_s"] = f"{joints.latency_s:.4f}" if joints.latency_s is not None else ""
    if coords:
        c = backend.get_coords()
        row["coords_ok"] = int(c is not None)
        if c is not None:
            for name, v in zip(COORDS, c.coords):
                row[name] = f"{v:.3f}"
            row["coords_latency_s"] = f"{c.latency_s:.4f}" if c.latency_s is not None else ""
    if gripper:
        g = backend.get_gripper_value()
        row["gripper"] = "" if g is None else g
    if status:
        s = backend.read_status()
        if s is not None:
            row["error_code"] = "" if s.error_code is None else s.error_code
            row["power_on"] = "" if s.power_on is None else int(s.power_on)
    return row


def record(
    backend: ArmBackend,
    path: Union[str, Path],
    rate_hz: float = 10.0,
    duration_s: Optional[float] = None,
    coords: bool = True,
    gripper: bool = False,
    status_every: int = 10,
    meta: Optional[Dict[str, object]] = None,
    should_stop: Callable[[], bool] = lambda: False,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> RecordStats:
    """Sample ``backend`` into ``path`` until ``duration_s`` or ``should_stop()``.

    Only read methods are called. The caller is responsible for handing in a
    read-only backend when live; ``mycobot-record`` always does.
    """
    if rate_hz <= 0:
        raise ValueError("rate_hz must be positive")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    period = 1.0 / rate_hz
    stats = RecordStats(started=time.time())
    t0 = stats.started
    start = clock()
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        try:
            _loop(writer, backend, stats, t0, start, period, duration_s, coords, gripper,
                  status_every, should_stop, clock, sleep)
        except KeyboardInterrupt:
            pass  # Ctrl-C is the normal way to end an open-ended recording
        fh.flush()
    stats.ended = time.time()
    sidecar = {
        "schema": SCHEMA,
        "csv": path.name,
        "requested_hz": rate_hz,
        "achieved_hz": stats.achieved_hz,
        "rows": stats.rows,
        "joint_misses": stats.joint_misses,
        "coord_misses": stats.coord_misses,
        "overruns": stats.overruns,
        "units": {"joints": "deg and rad", "coords": "mm, deg (firmware base frame)", "gripper": "0-100"},
        "joint_order": ["joint2_to_joint1", "joint3_to_joint2", "joint4_to_joint3",
                        "joint5_to_joint4", "joint6_to_joint5", "joint6output_to_joint6"],
        "started": stats.started,
        "ended": stats.ended,
    }
    if stats.extra:
        sidecar["errors"] = stats.extra
    sidecar.update(meta or {})
    path.with_suffix(".meta.json").write_text(json.dumps(sidecar, indent=2))
    return stats


def _loop(writer, backend, stats, t0, start, period, duration_s, coords, gripper,
          status_every, should_stop, clock, sleep) -> None:
    next_due = start
    while not should_stop():
        # The tolerance stops 10 x 0.1 s summing to 0.999... and taking an 11th row.
        if duration_s is not None and clock() - start >= duration_s - 1e-6 * period:
            break
        try:
            row = sample_row(backend, t0, coords, gripper, status_every > 0 and stats.rows % status_every == 0)
        except Exception as exc:  # a serial hiccup costs a row, not the recording
            row = {name: "" for name in FIELDS}
            row.update(t_host=f"{time.time():.6f}", joints_ok=0, coords_ok=0 if coords else "")
            stats.extra["last_error"] = str(exc)
        writer.writerow(row)
        stats.rows += 1
        stats.joint_misses += 0 if row["joints_ok"] else 1
        if coords and not row["coords_ok"]:
            stats.coord_misses += 1
        next_due += period
        delay = next_due - clock()
        if delay > 0:
            sleep(delay)
        else:
            stats.overruns += 1
            next_due = clock()  # do not try to catch up with a burst


def read_rows(path: Union[str, Path]) -> List[Dict[str, str]]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))
