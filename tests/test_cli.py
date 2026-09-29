import io

import pytest

from mycobot_control import cli
from mycobot_control.sdk import cliargs, jointmove
from mycobot_control.sdk.backend import DryRunBackend
from mycobot_control.sdk.portlock import PortBusy, PortLock
from mycobot_control.sdk.session import ArmConfig, ArmSession


def test_umbrella_lists_commands(capsys):
    assert cli.main([]) == 0
    assert "teleop" in capsys.readouterr().out
    assert cli.main(["nope"]) == 2


def test_info_dry_run(capsys):
    assert cli.main(["info"]) == 0
    assert "dry-run" in capsys.readouterr().out


def test_record_live_is_always_read_only():
    parser = cli._commands  # noqa: F841 - import check only
    from mycobot_control.record.cli import build_parser

    args = build_parser().parse_args(["--live"])
    cfg = cliargs.config_from_args(args, force_read_only=True)
    assert cfg.live and cfg.read_only


def test_read_only_flag_implies_live():
    from mycobot_control.teleop.keyboard import build_parser

    cfg = cliargs.config_from_args(build_parser().parse_args(["--read-only"]))
    assert cfg.live and cfg.read_only


def test_fast_speed_needs_opt_in():
    from mycobot_control.teleop.keyboard import build_parser

    with pytest.raises(SystemExit):
        cliargs.config_from_args(build_parser().parse_args(["--speed", "80"]))
    cfg = cliargs.config_from_args(build_parser().parse_args(["--speed", "80", "--allow-fast"]))
    assert cfg.speed == 80


def test_live_confirmation_refuses_without_tty():
    cfg = ArmConfig(live=True)
    out = io.StringIO()
    assert not cliargs.confirm_live(cfg, "x", assume_yes=False, stdin=io.StringIO("yes\n"), out=out)
    assert cliargs.confirm_live(cfg, "x", assume_yes=True, out=out)
    assert cliargs.confirm_live(ArmConfig(), "x", assume_yes=False, out=out)


def test_preflight_names_missing_port(tmp_path):
    cfg = ArmConfig(live=True, port=str(tmp_path / "ttyTHS1"))
    assert "--transport socket" in cliargs.preflight(cfg)
    assert cliargs.preflight(ArmConfig()) is None


def test_port_lock_is_exclusive(tmp_path):
    path = str(tmp_path / "lock")
    with PortLock(path):
        with pytest.raises(PortBusy):
            PortLock(path).acquire()
    PortLock(path).acquire()


def test_plan_waypoints_bounded():
    wps = jointmove.plan_waypoints([0] * 6, [25, 0, -5, 0, 0, 0], 10.0)
    assert len(wps) == 3
    assert wps[-1] == [25, 0, -5, 0, 0, 0]


def test_move_dry_run_reaches_target():
    backend = DryRunBackend()
    session = ArmSession(ArmConfig(), backend=backend).start()
    out = io.StringIO()
    assert jointmove.move(session, [30, 0, 0, 0, 0, 0], max_step_deg=10, out=out, sleep=lambda s: None) == 0
    assert backend.joint_angles == [30, 0, 0, 0, 0, 0]
    assert not session.gate.is_open


def test_move_stops_when_arm_does_not_follow():
    class Stuck(DryRunBackend):
        def send_angles(self, angles_deg, speed):
            self.commands.append(("send_angles", list(angles_deg), speed))

    backend = Stuck()
    session = ArmSession(ArmConfig(), backend=backend).start()
    code = jointmove.move(session, [30, 0, 0, 0, 0, 0], max_step_deg=10, waypoint_timeout_s=0.0,
                          out=io.StringIO(), sleep=lambda s: None)
    assert code == 3
    assert len([c for c in backend.commands if c[0] == "send_angles"]) == 1
    assert backend.commands[-1] == ("stop",)


def test_move_joints_refuses_out_of_limits(capsys):
    assert jointmove.main(["0", "150", "0", "0", "0", "0"]) == 2
