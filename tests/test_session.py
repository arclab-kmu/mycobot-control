import pytest

from mycobot_control.safety.gate import GateClosed
from mycobot_control.safety.limits import LimitViolation
from mycobot_control.sdk.backend import DryRunBackend, JointReading, ReadOnlyBackend, ReadOnlyViolation, valid_vector
from mycobot_control.sdk.session import ArmConfig, ArmSession, SessionError, make_backend
from mycobot_control.sdk.status import StatusReading, from_raw


def test_default_config_is_dry_run():
    assert isinstance(make_backend(ArmConfig()), DryRunBackend)


def test_config_rejects_contradictions():
    with pytest.raises(ValueError):
        ArmConfig(power_on=True)  # needs live
    with pytest.raises(ValueError):
        ArmConfig(live=True, read_only=True, power_on=True)
    with pytest.raises(ValueError):
        ArmConfig(live=True, transport="socket")  # needs host


def test_live_read_only_wraps_backend():
    backend = make_backend(ArmConfig(live=True, read_only=True))
    assert isinstance(backend, ReadOnlyBackend)


def test_start_sets_fresh_mode(session, dry):
    assert ("set_fresh_mode", 1) in dry.commands


def test_commands_need_the_gate(session):
    session.read_joints()
    with pytest.raises(GateClosed):
        session.command_joints([1, 0, 0, 0, 0, 0])
    with pytest.raises(GateClosed):
        session.command_coords([150, 0, 200, 180, 0, 0])
    with pytest.raises(GateClosed):
        session.command_gripper(50)


def test_command_joints_needs_a_measurement(dry):
    s = ArmSession(ArmConfig(), backend=dry).start()
    s.gate.open("test")
    with pytest.raises(SessionError):
        s.command_joints([1, 0, 0, 0, 0, 0])


def test_command_joints_refuses_jumps_and_limits(session, dry):
    session.read_joints()
    session.gate.open("test")
    with pytest.raises(LimitViolation):
        session.command_joints([45, 0, 0, 0, 0, 0])
    with pytest.raises(LimitViolation):
        session.command_joints([0, 139, 0, 0, 0, 0])
    session.command_joints([5, 0, 0, 0, 0, 0])
    assert dry.commands[-1] == ("send_angles", [5.0, 0, 0, 0, 0, 0], session.config.speed)


def test_coords_outside_sdk_box_refused(session):
    session.gate.open("test")
    with pytest.raises(LimitViolation):
        session.command_coords([500, 0, 200, 180, 0, 0])


def test_halt_closes_gate_and_stops(session, dry):
    session.gate.open("test")
    session.halt()
    assert not session.gate.is_open
    assert dry.commands[-1] == ("stop",)


def test_stop_closes_gate(session):
    session.gate.open("test")
    session.stop()
    assert not session.gate.is_open


class FlakyBackend(DryRunBackend):
    def __init__(self, misses, latency=0.0):
        super().__init__()
        self.misses = misses
        self.latency = latency

    def get_joint_state(self):
        if self.misses > 0:
            self.misses -= 1
            return None
        r = super().get_joint_state()
        return JointReading(r.angles, r.timestamp, latency_s=self.latency)


def test_read_joints_retries_missing_replies():
    s = ArmSession(ArmConfig(), backend=FlakyBackend(misses=2)).start()
    assert s.read_joints(retry_s=1.0).angles == [0.0] * 6


def test_read_joints_gives_up():
    s = ArmSession(ArmConfig(), backend=FlakyBackend(misses=10**6)).start()
    with pytest.raises(SessionError):
        s.read_joints(retry_s=0.05)


def test_slow_reading_refused():
    s = ArmSession(ArmConfig(max_feedback_age_s=0.1), backend=FlakyBackend(0, latency=0.3)).start()
    with pytest.raises(SessionError):
        s.read_joints()


def test_read_only_drops_commands_but_forwards_stop():
    inner = DryRunBackend()
    ro = ReadOnlyBackend(inner)
    ro.send_angles([1, 0, 0, 0, 0, 0], 20)
    ro.send_coords([150, 0, 200, 180, 0, 0], 20)
    ro.set_gripper_value(10, 50)
    ro.set_fresh_mode(1)
    ro.stop()
    assert inner.commands == [("stop",)]
    assert [c[0] for c in ro.suppressed] == ["send_angles", "send_coords", "set_gripper_value", "set_fresh_mode"]
    with pytest.raises(ReadOnlyViolation):
        ro.power_on()


@pytest.mark.parametrize("raw", [None, -1, [], [0.0] * 5, [0, 0, -1, 0, 0, 0], "x", [float("nan")] * 6])
def test_valid_vector_rejects(raw):
    assert valid_vector(raw) is None


def test_valid_vector_accepts():
    assert valid_vector([1, 2, 3, 4, 5, 6.5]) == [1.0, 2.0, 3.0, 4.0, 5.0, 6.5]


@pytest.mark.parametrize(
    "code,power,halt,rejected",
    [(0, True, False, False), (3, True, True, False), (17, True, True, False),
     (32, True, False, True), (99, True, True, False), (0, False, True, False), (None, None, False, False)],
)
def test_status_policy(code, power, halt, rejected):
    s = StatusReading(error_code=code, power_on=power)
    assert s.should_halt is halt
    assert s.was_rejected is rejected


def test_status_from_raw_treats_minus_one_as_unknown():
    s = from_raw(-1, -1)
    assert s.error_code is None and s.power_on is None
    assert from_raw(0, 1).is_normal
