import math

import pytest

from mycobot_control.teleop import keymap
from mycobot_control.teleop.keymap import (
    CARTESIAN_BINDINGS,
    JOINT_BINDINGS,
    apply_cartesian_key,
    apply_gripper_key,
    apply_joint_key,
    matrix_to_rpy,
    rpy_to_matrix,
)

START_COORDS = [150.0, -60.0, 200.0, -170.0, 10.0, -90.0]


def _pairs(table):
    by = {}
    for key, b in table.items():
        by.setdefault(b.index, {})[b.sign] = key
    return [(v[+1.0], v[-1.0]) for v in by.values()]


def test_every_joint_has_a_pair():
    assert sorted(b.index for b in JOINT_BINDINGS.values()) == sorted(list(range(6)) * 2)


def test_joint_opposite_keys_cancel():
    q = [10.0, -20.0, 30.0, 0.0, 5.0, -5.0]
    for plus, minus in _pairs(JOINT_BINDINGS):
        out = apply_joint_key(apply_joint_key(q, JOINT_BINDINGS[plus], 2.0), JOINT_BINDINGS[minus], 2.0)
        assert out == pytest.approx(q)


def test_joint_key_moves_only_its_joint():
    out = apply_joint_key([0.0] * 6, JOINT_BINDINGS["r"], 3.0)
    assert out == [0.0, 0.0, 3.0, 0.0, 0.0, 0.0]


@pytest.mark.parametrize("frame", ["base", "tool"])
def test_cartesian_opposite_keys_cancel(frame):
    for plus, minus in _pairs(CARTESIAN_BINDINGS):
        once = apply_cartesian_key(START_COORDS, CARTESIAN_BINDINGS[plus], 5.0, 2.0, frame)
        back = apply_cartesian_key(once, CARTESIAN_BINDINGS[minus], 5.0, 2.0, frame)
        assert _same_pose(back, START_COORDS), (plus, minus, back)


def test_rotation_key_leaves_position():
    out = apply_cartesian_key(START_COORDS, CARTESIAN_BINDINGS["o"], 5.0, 2.0)
    assert out[:3] == START_COORDS[:3]


def test_base_yaw_is_addition_but_roll_is_not():
    # Base-frame rz is the outermost factor, so it happens to add. rx does not
    # (at ry != 0) - which is why rotation composes as matrices.
    yaw = apply_cartesian_key(START_COORDS, CARTESIAN_BINDINGS["o"], 5.0, 2.0)
    assert yaw[5] == pytest.approx(START_COORDS[5] + 2.0)
    roll = apply_cartesian_key(START_COORDS, CARTESIAN_BINDINGS["u"], 5.0, 2.0)
    assert roll[3:] != pytest.approx([START_COORDS[3] + 2.0, START_COORDS[4], START_COORDS[5]])


def test_rpy_round_trip():
    for rpy in [(0, 0, 0), (-170, 10, -90), (30, -45, 120), (179, 89, -179)]:
        assert _same_rot(matrix_to_rpy(rpy_to_matrix(*rpy)), rpy)


def test_gripper_keys_clamp():
    assert apply_gripper_key(95, ".", 10) == 100
    assert apply_gripper_key(5, ",", 10) == 0
    assert apply_gripper_key(40, "<", 10) == 0
    assert apply_gripper_key(40, ">", 10) == 100
    assert apply_gripper_key(40, "w", 10) is None


def test_no_key_bound_twice():
    for table in (JOINT_BINDINGS, CARTESIAN_BINDINGS):
        assert not set(table) & set(keymap.CONTROL_KEYS)
        assert not set(table) & set(keymap.GRIPPER_KEYS)


def _same_rot(a, b):
    ra, rb = rpy_to_matrix(*a), rpy_to_matrix(*b)
    return all(math.isclose(ra[i][j], rb[i][j], abs_tol=1e-9) for i in range(3) for j in range(3))


def _same_pose(a, b):
    return a[:3] == pytest.approx(b[:3]) and _same_rot(a[3:], b[3:])
