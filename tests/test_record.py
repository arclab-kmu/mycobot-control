import json

from mycobot_control.record.recorder import FIELDS, read_rows, record
from mycobot_control.sdk.backend import DryRunBackend, ReadOnlyBackend


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, dt):
        self.t += dt


def test_record_writes_rows_and_sidecar(tmp_path):
    clock = Clock()
    backend = ReadOnlyBackend(DryRunBackend(joint_angles=[90.0, 0, 0, 0, 0, 0]))
    path = tmp_path / "r.csv"
    stats = record(backend, path, rate_hz=10, duration_s=1.0, gripper=True, clock=clock, sleep=clock.sleep)
    rows = read_rows(path)
    assert len(rows) == stats.rows == 10
    assert list(rows[0].keys()) == FIELDS
    assert float(rows[0]["j1_rad"]) == 1.570796
    meta = json.loads(path.with_suffix(".meta.json").read_text())
    assert meta["schema"] == "mycobot-record/1" and meta["rows"] == 10
    assert backend.suppressed == []
    assert backend.inner.commands == []


def test_record_survives_read_errors(tmp_path):
    class Broken(DryRunBackend):
        def get_joint_state(self):
            raise OSError("serial hiccup")

    clock = Clock()
    stats = record(Broken(), tmp_path / "r.csv", rate_hz=10, duration_s=0.5, clock=clock, sleep=clock.sleep)
    assert stats.rows == 5 and stats.joint_misses == 5
