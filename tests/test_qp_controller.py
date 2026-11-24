import pytest

np = pytest.importorskip("numpy")

from mycobot_control.qp.controller import ControllerConfig, JointTarget, PoseTarget, QPController  # noqa: E402
from mycobot_control.qp.kinematics import DEG, Kinematics, pose_error, so3_exp  # noqa: E402

KIN = Kinematics()
#: Well-conditioned, tool pointing down, z ~ 189 mm.
Q_READY = np.array([0.0, -30.0, -60.0, 0.0, 0.0, 0.0])


def _run(ctrl, q0, target, ticks):
    q, dq = np.asarray(q0, float), np.zeros(6)
    info = None
    for _ in range(ticks):
        qn, info = ctrl.step(q, dq, target)
        dq, q = qn - q, qn
    return q, dq, info


def _branch_pair(rng):
    """A goal and a start on the same elbow / wrist branch (local IK cannot flip branches).

    J3 = 0 (arm stretched) and J5 = +-90 (J4 and J6 axes aligned) are
    singular; goals stay clear of them.
    """
    qg = np.array([rng.uniform(-90, 90), rng.uniform(-50, 20), rng.uniform(-120, -30),
                   rng.uniform(-60, 60), rng.uniform(20, 70), rng.uniform(-90, 90)])
    q0 = qg + rng.uniform(-20, 20, 6)
    q0[2] = min(q0[2], -25.0)
    q0[4] = min(max(q0[4], 15.0), 75.0)
    return qg, q0


def test_converges_to_reachable_poses():
    ctrl = QPController(KIN, ControllerConfig())
    rng = np.random.default_rng(10)
    for _ in range(25):
        qg, q0 = _branch_pair(rng)
        Tg = KIN.fk(qg)
        q, dq, info = _run(ctrl, q0, Tg, 200)  # 10 s at 20 Hz
        e_p, e_r = pose_error(Tg, KIN.fk(q))
        assert e_p < 0.1 and e_r < 0.1, (qg, q0, e_p, e_r)
        assert np.abs(dq).max() < 1e-2  # settled (deg/tick)
        assert info.status == "optimal"


def test_feed_forward_tracks_a_moving_target_without_lag():
    ctrl = QPController(KIN, ControllerConfig())
    T0 = KIN.fk(Q_READY)
    q, dq = Q_READY.copy(), np.zeros(6)
    v = np.array([0.0, 40.0, 0.0])  # mm/s
    errs = []
    for k in range(1, 61):
        T = T0.copy()
        T[:3, 3] += v * k * ctrl.cfg.dt
        qn, info = ctrl.step(q, dq, PoseTarget(T, np.concatenate([v, [0, 0, 0]])))
        dq, q = qn - q, qn
        errs.append(info.pos_err_mm)
    assert max(errs[20:]) < 0.05  # steady state: no lag


def _limits(cfg):
    lo, hi = cfg.limits_lo_hi()
    return lo, hi, cfg.vec(cfg.qd_max_deg_s) * cfg.dt, cfg.vec(cfg.qdd_max_deg_s2) * cfg.dt ** 2


def test_long_random_run_never_violates_joint_speed_or_accel_limits():
    cfg = ControllerConfig(z_min_mm=40.0)
    ctrl = QPController(KIN, cfg)
    lo, hi, vmax, amax = _limits(cfg)
    rng = np.random.default_rng(11)
    q, dq = Q_READY.copy(), np.zeros(6)
    target = None
    statuses = set()
    for k in range(3000):
        if k % 40 == 0:
            r = rng.random()
            if r < 0.4:  # a pose, often unreachable
                T = np.eye(4)
                T[:3, :3] = so3_exp(rng.normal(size=3))
                T[:3, 3] = rng.uniform([-350, -350, -100], [350, 350, 500])
                target = PoseTarget(T)
            elif r < 0.7:  # joints, some beyond the joint table
                target = JointTarget(rng.uniform(-200, 200, 6))
            else:  # a moving pose with feed-forward
                target = PoseTarget(KIN.fk(q), rng.normal(size=6) * [80, 80, 80, 40, 40, 40])
        qn, info = ctrl.step(q, dq, target)
        statuses.add(info.status)
        assert np.all(np.isfinite(qn))
        assert np.all(qn >= lo - 1e-9) and np.all(qn <= hi + 1e-9), (k, qn)
        step = qn - q
        assert np.all(np.abs(step) <= vmax + 1e-9), (k, step)
        assert np.all(np.abs(step - dq) <= amax + 1e-9), (k, step - dq)
        assert info.accel_relaxed == []
        dq, q = step, qn
    assert "optimal" in statuses


def test_joint_targets_beyond_the_table_stop_at_the_margin_without_overshoot():
    cfg = ControllerConfig()
    ctrl = QPController(KIN, cfg)
    lo, hi, vmax, amax = _limits(cfg)
    q, dq = Q_READY.copy(), np.zeros(6)
    for _ in range(400):
        qn, info = ctrl.step(q, dq, JointTarget([400, 400, 400, 400, 400, 400]))
        assert np.all(qn <= hi + 1e-9)
        dq, q = qn - q, qn
    assert np.allclose(q, hi, atol=1e-6)
    assert any("pos+" in a for a in info.active) or np.abs(dq).max() < 1e-9


def test_joint_reference_converges():
    ctrl = QPController(KIN, ControllerConfig())
    goal = Q_READY + [20, -10, 15, 30, -25, 40]
    q, dq, info = _run(ctrl, Q_READY, JointTarget(goal), 150)
    assert info.mode == "joint" and info.joint_err_deg < 1e-3


def _circle(weights, wls=False):
    """Two laps of an r = 10 mm, 4 s circle, position only. Returns per-joint ranges, worst error."""
    cfg = ControllerConfig(position_only=True, joint_weights=weights)
    ctrl = QPController(KIN, cfg)
    q0 = np.array([-10.0, -40.0, -50.0, -20.0, 30.0, 0.0])
    c = KIN.fk(q0)[:3, 3]
    q, dq = q0.copy(), np.zeros(6)
    r, period = 10.0, 4.0
    qs, errs = [], []
    for k in range(1, int(2 * period / cfg.dt) + 1):
        th = 2 * np.pi * k * cfg.dt / period
        p = c + r * np.array([np.cos(th) - 1.0, np.sin(th), 0.0])
        if wls:
            # Exact weighted least-norm IK (what the stir experiment used), for reference.
            Wi = np.diag(1.0 / np.asarray(weights, float))
            for _ in range(10):
                J = KIN.jacobian(q)[:3] * DEG
                q = q + Wi @ J.T @ np.linalg.solve(J @ Wi @ J.T, p - KIN.fk(q)[:3, 3])
            errs.append(float(np.linalg.norm(p - KIN.fk(q)[:3, 3])))
        else:
            T = np.eye(4)
            T[:3, 3] = p
            v = r * 2 * np.pi / period * np.array([-np.sin(th), np.cos(th), 0.0])
            qn, info = ctrl.step(q, dq, PoseTarget(T, np.concatenate([v, [0, 0, 0]])))
            dq, q = qn - q, qn
            errs.append(info.pos_err_mm)
        qs.append(q.copy())
    return np.ptp(np.array(qs), axis=0), max(errs[10:])


def test_j1_weight_spares_j1_on_a_small_circle():
    plain, err_plain = _circle((1, 1, 1, 1, 1, 1))
    spared, err_spared = _circle((20, 1, 1, 1, 1, 1))
    assert spared[0] < 0.7 * plain[0], (plain, spared)
    assert err_plain < 0.1 and err_spared < 0.1
    # The QP's null-space choice is the weighted least-norm one. How much J1 that
    # saves depends on the geometry (a long tool gives the wrist more leverage).
    ref, _ = _circle((20, 1, 1, 1, 1, 1), wls=True)
    assert abs(spared[0] - ref[0]) < 0.3, (spared, ref)


def test_floor_keeps_the_tcp_above_z_min():
    z0 = KIN.fk(Q_READY)[2, 3]
    cfg = ControllerConfig(z_min_mm=z0 - 20.0)
    ctrl = QPController(KIN, cfg)
    T = KIN.fk(Q_READY)
    T[2, 3] -= 80.0
    q, dq = Q_READY.copy(), np.zeros(6)
    zs = []
    for _ in range(200):
        qn, info = ctrl.step(q, dq, T)
        dq, q = qn - q, qn
        zs.append(info.tcp_z_mm)
    assert min(zs) >= cfg.z_min_mm - 0.05
    assert zs[-1] < cfg.z_min_mm + 1.0  # it does go down to the floor
    assert "floor" in info.active


def test_infeasible_floor_brakes_and_stays_finite():
    z0 = KIN.fk(Q_READY)[2, 3]
    ctrl = QPController(KIN, ControllerConfig(z_min_mm=z0 + 500.0))  # TCP far below the "floor"
    qn, info = ctrl.step(Q_READY, np.zeros(6), KIN.fk(Q_READY))
    assert info.status == "braking"
    assert np.all(np.isfinite(qn)) and np.allclose(qn, Q_READY)


def test_position_only_ignores_orientation():
    ctrl = QPController(KIN, ControllerConfig(position_only=True))
    T = KIN.fk(Q_READY)
    T[:3, 3] += [10, -15, 20]
    T[:3, :3] = so3_exp([0.0, 0.0, 1.0]) @ T[:3, :3]  # 57 deg off, must be ignored
    q, _, info = _run(ctrl, Q_READY, T, 100)
    assert info.pos_err_mm < 0.05
    assert info.rot_err_deg > 30.0


def test_tcp_speed_limit_holds():
    cfg = ControllerConfig(v_max_mm_s=50.0)
    ctrl = QPController(KIN, cfg)
    T = KIN.fk(Q_READY)
    T[:3, 3] += [60, 60, -40]
    q, dq = Q_READY.copy(), np.zeros(6)
    for _ in range(80):
        qn, _ = ctrl.step(q, dq, T)
        v = (KIN.fk(qn)[:3, 3] - KIN.fk(q)[:3, 3]) / cfg.dt
        assert np.all(np.abs(v) <= 50.0 * 1.02)  # per axis, linearised
        dq, q = qn - q, qn


def test_accepts_raw_pose_and_joint_arrays():
    ctrl = QPController(KIN, ControllerConfig())
    _, a = ctrl.step(Q_READY, np.zeros(6), KIN.fk(Q_READY))
    _, b = ctrl.step(Q_READY, np.zeros(6), Q_READY)
    assert a.mode == "pose" and b.mode == "joint"
    with pytest.raises(ValueError):
        ctrl.step(Q_READY, np.zeros(6), [1, 2, 3])


def test_dq_prev_units_are_degrees_per_tick():
    # A previous increment of 1 deg/tick is 20 deg/s at dt 0.05: the next one may
    # differ by at most qdd_max * dt^2 = 0.75 deg.
    cfg = ControllerConfig()
    ctrl = QPController(KIN, cfg)
    dq_prev = np.array([1.0, 0, 0, 0, 0, 0])
    qn, _ = ctrl.step(Q_READY, dq_prev, Q_READY)  # target: stay
    assert (qn - Q_READY)[0] >= 1.0 - 300.0 * cfg.dt ** 2 - 1e-9
    assert DEG > 0
