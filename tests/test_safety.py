import pytest

from mycobot_control.safety.deadman import Deadman
from mycobot_control.safety.gate import ControlGate, GateClosed
from mycobot_control.safety import limits
from mycobot_control.safety.limits import LimitViolation


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_gate_starts_closed_and_refuses():
    gate = ControlGate()
    with pytest.raises(GateClosed):
        gate.require_open()
    gate.open("test")
    gate.require_open()


def test_gate_closes_on_scope_exit_even_on_error():
    gate = ControlGate()
    with pytest.raises(ValueError):
        with gate:
            gate.open("x")
            raise ValueError
    assert not gate.is_open


def test_deadman_expires_after_silence():
    clock = Clock()
    gate = ControlGate(clock=clock)
    dm = Deadman(gate, timeout_s=0.5, clock=clock)
    dm.poke()
    clock.t = 0.4
    assert not dm.check() and gate.is_open
    clock.t = 0.51
    assert dm.check() and not gate.is_open


def test_joint_target_refused_near_limits():
    lo, hi = limits.JOINT_LIMITS_DEG[1]
    ok = [0.0] * 6
    limits.check_joint_target(ok)
    near = list(ok)
    near[1] = hi - limits.JOINT_LIMIT_MARGIN_DEG / 2
    with pytest.raises(LimitViolation):
        limits.check_joint_target(near)


def test_clamp_stays_inside_margin():
    q = limits.clamp_joint_target([999, -999, 0, 0, 0, 0])
    assert q[0] == limits.JOINT_LIMITS_DEG[0][1] - limits.JOINT_LIMIT_MARGIN_DEG
    assert q[1] == limits.JOINT_LIMITS_DEG[1][0] + limits.JOINT_LIMIT_MARGIN_DEG


def test_command_delta_catches_radians_passed_as_degrees():
    # 1.0 "deg" where 57 deg was meant, or the other way round: a units mistake is a jump.
    with pytest.raises(LimitViolation):
        limits.check_command_delta([57.3, 0, 0, 0, 0, 0], [1.0, 0, 0, 0, 0, 0])


def test_joint_vector_rejects_nan_and_wrong_length():
    with pytest.raises(ValueError):
        limits.joint_vector([0.0] * 5)
    with pytest.raises(ValueError):
        limits.joint_vector([float("nan")] + [0.0] * 5)


@pytest.mark.parametrize("speed", [0, 101])
def test_speed_range(speed):
    with pytest.raises(LimitViolation):
        limits.check_speed(speed)
