import math

import pytest

np = pytest.importorskip("numpy")

from mycobot_control.qp.kinematics import (  # noqa: E402
    DEG,
    Kinematics,
    orientation_error,
    pose_from_xyzrpy,
    rot_to_rpy,
    rpy_to_rot,
    so3_exp,
    so3_log,
    xyzrpy_from_pose,
)
from mycobot_control.teleop.keymap import rpy_to_matrix  # noqa: E402

KIN = Kinematics()


def test_fk_zero_pose_matches_hand_computation():
    # Chain of the six DH matrices at q = 0 (theta = the offsets), multiplied out
    # by hand: the arm stands up, the flange points along base +x.
    T = KIN.fk([0, 0, 0, 0, 0, 0])
    assert np.allclose(T[:3, 3], [45.6, -63.4, 139.42 + 110.4 + 96.0 + 75.05], atol=1e-9)
    assert np.allclose(T[:3, :3], [[0, 0, 1], [-1, 0, 0], [0, -1, 0]], atol=1e-12)
    assert np.allclose(xyzrpy_from_pose(T), [45.6, -63.4, 420.87, -90.0, 0.0, -90.0], atol=1e-9)


@pytest.mark.parametrize("q, xyz, rpy", [
    # From the parent session's vlm/kin.py, which was validated against firmware get_coords().
    ([10, -20, 30, -40, 50, -60], [87.6653, -13.4498, 388.0433], [-112.7959, -33.9514, -12.7959]),
    ([-45, 1.75, 7.82, -31.64, 3.16, -119.44], [-6.9469, -79.1594, 396.8756], [130.6166, -54.8093, 13.3838]),
    ([0, -30, -60, 0, 0, 0], [226.25, -63.4, 189.4292], [-180.0, 0.0, -90.0]),
])
def test_fk_matches_validated_reference(q, xyz, rpy):
    v = KIN.xyzrpy(q)
    assert np.allclose(v[:3], xyz, atol=1e-3)
    d = (v[3:] - np.asarray(rpy) + 180.0) % 360.0 - 180.0  # -180 and +180 are the same angle
    assert np.allclose(d, 0.0, atol=1e-3)


def _fd_jacobian(kin, q_deg, h=1e-6):
    J = np.zeros((6, 6))
    for j in range(6):
        e = np.zeros(6)
        e[j] = h / DEG  # h radians, in degrees
        Tp, Tm = kin.fk(np.asarray(q_deg) + e), kin.fk(np.asarray(q_deg) - e)
        J[:3, j] = (Tp[:3, 3] - Tm[:3, 3]) / (2 * h)
        J[3:, j] = so3_log(Tp[:3, :3] @ Tm[:3, :3].T) / (2 * h)
    return J


@pytest.mark.parametrize("tool", [None, "offset"])
def test_jacobian_matches_finite_differences(tool):
    T_tool = None
    if tool:
        T_tool = pose_from_xyzrpy([12.0, -5.0, 85.0, 10.0, -20.0, 30.0])
    kin = Kinematics(tool=T_tool)
    rng = np.random.default_rng(3)
    lim = kin.joint_limits_deg
    for _ in range(40):
        q = rng.uniform(lim[:, 0] + 5, lim[:, 1] - 5)
        J = kin.jacobian(q)
        Jfd = _fd_jacobian(kin, q)
        assert np.allclose(J, Jfd, atol=1e-5 * max(1.0, np.abs(J).max())), q


def test_jacobian_units_are_per_radian():
    q = np.array([20.0, -30.0, -60.0, 10.0, 40.0, 5.0])
    J = KIN.jacobian(q)
    dq_deg = np.array([0.01, -0.02, 0.015, 0.01, -0.01, 0.02])
    dp = KIN.fk(q + dq_deg)[:3, 3] - KIN.fk(q)[:3, 3]
    assert np.allclose(J[:3] @ (dq_deg * DEG), dp, atol=1e-5)


def test_fk_jacobian_agrees_with_separate_calls():
    q = [5, -40, -70, 20, 60, -30]
    T, J = KIN.fk_jacobian(q)
    assert np.allclose(T, KIN.fk(q)) and np.allclose(J, KIN.jacobian(q))


def test_rpy_is_the_firmware_convention_and_matches_the_teleop_keymap():
    rng = np.random.default_rng(0)
    for _ in range(50):
        rx, ry, rz = rng.uniform(-179, 179), rng.uniform(-89, 89), rng.uniform(-179, 179)
        R = rpy_to_rot(rx, ry, rz)
        c, s = math.cos, math.sin
        a, b, g = rx * DEG, ry * DEG, rz * DEG
        Rx = np.array([[1, 0, 0], [0, c(a), -s(a)], [0, s(a), c(a)]])
        Ry = np.array([[c(b), 0, s(b)], [0, 1, 0], [-s(b), 0, c(b)]])
        Rz = np.array([[c(g), -s(g), 0], [s(g), c(g), 0], [0, 0, 1]])
        assert np.allclose(R, Rz @ Ry @ Rx)
        assert np.allclose(R, np.array(rpy_to_matrix(rx, ry, rz)))
        assert np.allclose(rot_to_rpy(R), [rx, ry, rz], atol=1e-9)


def test_pose_roundtrip():
    v = np.array([120.0, -80.0, 210.0, -170.0, 15.0, -95.0])
    assert np.allclose(xyzrpy_from_pose(pose_from_xyzrpy(v)), v)


@pytest.mark.parametrize("angle", [0.0, 1e-8, 0.3, 2.0, math.pi - 1e-6, math.pi])
def test_so3_log_exp_roundtrip_including_near_pi(angle):
    axis = np.array([0.3, -0.5, 0.81])
    axis /= np.linalg.norm(axis)
    w = angle * axis
    R = so3_exp(w)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
    w2 = so3_log(R)
    assert np.all(np.isfinite(w2))
    assert np.allclose(so3_exp(w2), R, atol=1e-9)


def test_orientation_error_is_the_base_frame_rotation_from_current_to_target():
    Rc = rpy_to_rot(-170, 10, -80)
    w = np.array([0.02, -0.01, 0.03])
    Rt = so3_exp(w) @ Rc
    assert np.allclose(orientation_error(Rt, Rc), w, atol=1e-12)


def test_bad_joint_vector_rejected():
    with pytest.raises(ValueError):
        KIN.fk([0, 0, 0])
    with pytest.raises(ValueError):
        KIN.fk([0, 0, float("nan"), 0, 0, 0])
