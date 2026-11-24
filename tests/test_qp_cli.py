import io
import itertools

import pytest

np = pytest.importorskip("numpy")

from conftest import FakeReader  # noqa: E402

from mycobot_control import cli  # noqa: E402
from mycobot_control.qp import ChunkExecutor, ControllerConfig, Kinematics, QPController  # noqa: E402
from mycobot_control.qp import cli as qpcli  # noqa: E402
from mycobot_control.sdk.backend import DryRunBackend  # noqa: E402
from mycobot_control.sdk.session import ArmConfig, ArmSession  # noqa: E402
from mycobot_control.teleop.keyboard import build_parser as teleop_parser  # noqa: E402
from mycobot_control.teleop.keyboard import run  # noqa: E402

KIN = Kinematics()
READY = ["0", "-30", "-60", "0", "0", "0"]


def _sent(backend, name):
    return [c for c in backend.commands if c[0] == name]


def test_dry_run_to_pose_plans_and_replays_through_the_gate(capsys):
    assert qpcli.main(["--to", "150", "-60", "200", "--start", *READY]) == 0
    out = capsys.readouterr().out
    assert "error 0.00 mm / 0.00 deg" in out
    assert "dry-run:" in out and "passed the gate" in out


def test_dry_run_joint_goal(capsys):
    assert qpcli.main(["--joints", "10", "-20", "-70", "5", "10", "20", "--start", *READY,
                       "--duration", "2", "--rate", "10"]) == 0
    assert "error 0.00 deg" in capsys.readouterr().out


def test_unreachable_goal_is_refused_before_anything_moves(capsys):
    assert qpcli.main(["--to", "600", "0", "200", "--start", *READY]) == 3
    assert "nothing commanded" in capsys.readouterr().out


def test_joint_goal_outside_the_table_is_refused():
    assert qpcli.main(["--joints", "0", "150", "0", "0", "0", "0"]) == 2


def test_start_is_dry_run_only_and_rpy_needs_to():
    with pytest.raises(SystemExit):
        qpcli.main(["--to", "150", "-60", "200", "--start", *READY, "--live", "--assume-yes"])
    with pytest.raises(SystemExit):
        qpcli.main(["--joints", *READY, "--rpy", "0", "0", "0"])


def test_qd_max_follows_speed_unless_given():
    args = qpcli.build_parser().parse_args(["--to", "1", "2", "3"])
    assert qpcli.controller_config(args, 20).qd_max_deg_s == pytest.approx(20 * 1.2 * 0.8)
    assert qpcli.controller_config(args, 100).qd_max_deg_s == 60.0
    args = qpcli.build_parser().parse_args(["--to", "1", "2", "3", "--qd-max", "33"])
    assert qpcli.controller_config(args, 20).qd_max_deg_s == 33.0


def test_dispatcher_runs_qp(capsys):
    assert cli.main(["qp", "--to", "150", "-60", "200", "--start", *READY]) == 0
    assert cli.main([]) == 0
    assert "qp" in capsys.readouterr().out


def _session(backend):
    return ArmSession(ArmConfig(), backend=backend).start()


def _executor(q0, goal, lookahead=0, **kw):
    ex = ChunkExecutor(QPController(KIN, ControllerConfig()), q0, lookahead_ticks=lookahead, **kw)
    ex.goal_joints(goal, 1.0)
    return ex


def test_stream_sends_every_tick_through_command_joints_and_closes_the_gate():
    q0 = [float(v) for v in READY]
    backend = DryRunBackend(joint_angles=list(q0))
    session = _session(backend)
    goal = np.array(q0) + [10, 0, 0, 0, 0, 10]
    sleeps = []
    code = qpcli.stream(session, _executor(q0, goal, lookahead=2), out=io.StringIO(), sleep=sleeps.append,
                        clock=itertools.count(0.0, 0.01).__next__)
    assert code == 0
    sent = _sent(backend, "send_angles")
    assert len(sent) >= 18  # 1 s at 20 Hz, finishing 2 ticks early with lookahead 2
    assert np.allclose(sent[-1][1], goal, atol=0.01)
    assert not session.gate.is_open
    assert sleeps and all(s > 0 for s in sleeps)


def test_stream_stops_on_tracking_error():
    class Stuck(DryRunBackend):
        def send_angles(self, angles_deg, speed):
            self.commands.append(("send_angles", list(angles_deg), speed))  # never moves

    q0 = [float(v) for v in READY]
    backend = Stuck(joint_angles=list(q0))
    session = _session(backend)
    ex = _executor(q0, np.array(q0) + [40, 0, 0, 0, 0, 0], max_track_err_deg=3.0)
    out = io.StringIO()
    assert qpcli.stream(session, ex, out=out, sleep=lambda s: None) == 4
    assert "tracking error" in out.getvalue()
    assert backend.commands[-1] == ("stop",)
    assert not session.gate.is_open


def test_stream_stops_on_a_halting_status():
    class Faults(DryRunBackend):
        def send_angles(self, angles_deg, speed):
            super().send_angles(angles_deg, speed)
            self.error_code = 2  # joint limit

    q0 = [float(v) for v in READY]
    backend = Faults(joint_angles=list(q0))
    session = _session(backend)
    ex = _executor(q0, np.array(q0) + [20, 0, 0, 0, 0, 0])
    assert qpcli.stream(session, ex, out=io.StringIO(), sleep=lambda s: None, status_every_s=0.05) == 4
    assert backend.commands[-1] == ("stop",)


# -- teleop --ik qp ----------------------------------------------------------

def _fake_clock(step=0.05):
    t = [0.0]

    def clock():
        t[0] += step
        return t[0]
    return clock


def test_teleop_cartesian_qp_streams_joint_commands_not_coords():
    q0 = [float(v) for v in READY]
    backend = DryRunBackend(joint_angles=list(q0))
    session = _session(backend)
    keys = ["r"] + [None] * 30  # +Z, then let the chunk run out
    code = run(session, FakeReader(keys), mode="cartesian", ik="qp", linear_step_mm=5.0,
               out=io.StringIO(), status_settle_s=0.0, clock=_fake_clock())
    assert code == 0
    assert not _sent(backend, "send_coords")
    sent = _sent(backend, "send_angles")
    assert len(sent) > 5
    z0 = KIN.fk(q0)[2, 3]
    z1 = KIN.fk(backend.joint_angles)[2, 3]
    assert z1 - z0 == pytest.approx(5.0, abs=0.2)


def test_teleop_qp_stop_key_cancels_the_stream():
    q0 = [float(v) for v in READY]
    backend = DryRunBackend(joint_angles=list(q0))
    session = _session(backend)
    run(session, FakeReader(["w", " "] + [None] * 10), mode="cartesian", ik="qp",
        out=io.StringIO(), status_settle_s=0.0, clock=_fake_clock())
    i_stop = backend.commands.index(("stop",))
    assert not [c for c in backend.commands[i_stop:] if c[0] == "send_angles"]


def test_teleop_firmware_ik_still_available_and_qp_is_the_cli_default():
    args = teleop_parser().parse_args(["--mode", "cartesian"])
    assert args.ik == "qp"
    args = teleop_parser().parse_args(["--mode", "cartesian", "--ik", "firmware"])
    assert args.ik == "firmware"
    backend = DryRunBackend()
    session = _session(backend)
    run(session, FakeReader(["r"]), mode="cartesian", ik="firmware", out=io.StringIO(), status_settle_s=0.0)
    assert len(_sent(backend, "send_coords")) == 1
