import io

from conftest import FakeReader

from mycobot_control.sdk.backend import DryRunBackend
from mycobot_control.sdk.session import ArmConfig, ArmSession
from mycobot_control.teleop.eventlog import EventLog
from mycobot_control.teleop.keyboard import run


def _run(keys, backend=None, **kw):
    backend = backend or DryRunBackend()
    session = ArmSession(ArmConfig(), backend=backend).start()
    out = io.StringIO()
    code = run(session, FakeReader(keys), out=out, status_settle_s=0.0, **kw)
    return code, backend, session, out.getvalue()


def _sent(backend, name):
    return [c for c in backend.commands if c[0] == name]


def test_joint_key_commands_one_step_from_measured():
    code, backend, _, _ = _run(["w", "w", "s"], joint_step_deg=2.0)
    assert code == 0
    assert [c[1][0] for c in _sent(backend, "send_angles")] == [2.0, 4.0, 2.0]


def test_quit_closes_gate():
    _, _, session, _ = _run(["w", "q"])
    assert not session.gate.is_open


def test_stop_key_sends_stop_and_closes_gate():
    _, backend, session, _ = _run(["w", " "])
    assert backend.commands[-1] == ("stop",)
    assert not session.gate.is_open


def test_cartesian_mode_uses_send_coords():
    start = DryRunBackend().coords
    _, backend, _, _ = _run(["r"], mode="cartesian", linear_step_mm=5.0)
    (cmd,) = _sent(backend, "send_coords")
    assert cmd[1][2] == start[2] + 5.0
    assert not _sent(backend, "send_angles")


def test_step_keys_respect_caps():
    _, backend, _, out = _run(["]"] * 10 + ["w"], joint_step_deg=2.0)
    assert "step unchanged" in out
    assert _sent(backend, "send_angles")[0][1][0] <= 15.0


def test_limit_refusal_does_not_command():
    backend = DryRunBackend(joint_angles=[164.0, 0, 0, 0, 0, 0])
    _, backend, _, out = _run(["w"], backend=backend, joint_step_deg=2.0)
    assert not _sent(backend, "send_angles")
    assert "refused" in out


def test_gripper_keys():
    _, backend, _, _ = _run([">", ",", ","], gripper_step=10)
    assert [c[1] for c in _sent(backend, "set_gripper_value")] == [100, 90, 80]


class FaultAfterFirst(DryRunBackend):
    """Reports a joint-limit error after the first command, until cleared."""

    def send_angles(self, angles_deg, speed):
        super().send_angles(angles_deg, speed)
        self.error_code = 2


def test_fault_latches_and_deadman_cannot_reopen():
    # Poll timeouts (None) let the deferred status read run before the next keys.
    backend = FaultAfterFirst()
    _, backend, session, out = _run(["w", None, None, "w", "w", "e"], backend=backend)
    assert len(_sent(backend, "send_angles")) == 1
    assert "still halted" in out
    assert not session.gate.is_open


class FaultThenClear(FaultAfterFirst):
    def __init__(self):
        super().__init__()
        self.reads_until_clear = 3

    def read_status(self):
        s = super().read_status()
        if self.error_code:
            self.reads_until_clear -= 1
            if self.reads_until_clear <= 0:
                self.error_code = 0
        return s


def test_latch_clears_only_when_arm_reports_normal():
    backend = FaultThenClear()
    _, backend, _, out = _run(["w", None, None, "w", "w", "w"], backend=backend)
    assert "resuming" in out
    assert len(_sent(backend, "send_angles")) >= 2


def test_rejected_ik_does_not_latch():
    class Rejects(DryRunBackend):
        def send_coords(self, coords, speed, mode=0):
            super().send_coords(coords, speed, mode)
            self.error_code = 32

    _, backend, _, out = _run(["r", None, None, "r"], backend=Rejects(), mode="cartesian")
    assert len(_sent(backend, "send_coords")) == 2
    assert "NO_IK_SOLUTION" in out


def test_eventlog_has_a_row_per_key(tmp_path):
    path = tmp_path / "log.csv"
    with EventLog(path) as log:
        _run(["w", "x", "q"], eventlog=log)
    text = path.read_text()
    assert text.count("\nkey") == 0  # rows start with t_host, not the event
    rows = text.strip().splitlines()[1:]
    events = [r.split(",")[1] for r in rows]
    assert events.count("key") == 3
    assert events[0] == "start" and events[-1] == "end"
