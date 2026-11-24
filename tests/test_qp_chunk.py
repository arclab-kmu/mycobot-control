import pytest

np = pytest.importorskip("numpy")

from mycobot_control.qp import (  # noqa: E402
    ChunkExecutor,
    ControllerConfig,
    Kinematics,
    QPController,
    TrackingError,
    min_jerk,
    pose_error,
)

KIN = Kinematics()
Q_READY = np.array([0.0, -30.0, -60.0, 0.0, 0.0, 0.0])


def _ctrl(**kw):
    return QPController(KIN, ControllerConfig(**kw))


def _run(ex, ticks=None, until_done=True, limit=2000):
    out = [ex.q_cmd]
    for _ in range(limit if ticks is None else ticks):
        out.append(np.array(ex.tick()))
        if until_done and ticks is None and ex.done:
            break
    return np.array(out)


def _assert_smooth(Q, cfg):
    vmax = cfg.vec(cfg.qd_max_deg_s) * cfg.dt
    amax = cfg.vec(cfg.qdd_max_deg_s2) * cfg.dt ** 2
    dq = np.diff(Q, axis=0)
    assert np.all(np.abs(dq) <= vmax + 1e-9)
    assert np.all(np.abs(np.diff(dq, axis=0)) <= amax + 1e-9)


def test_min_jerk_profile():
    assert min_jerk(0.0) == 0.0 and min_jerk(1.0) == 1.0 and min_jerk(0.5) == pytest.approx(0.5)
    assert min_jerk(0.0, 1) == 0.0 and min_jerk(1.0, 1) == 0.0 and min_jerk(0.5, 1) == pytest.approx(1.875)
    assert min_jerk(0.0, 2) == 0.0 and min_jerk(1.0, 2) == 0.0
    assert min_jerk(-1.0) == 0.0 and min_jerk(2.0) == 1.0
    s = min_jerk(np.linspace(0, 1, 11))
    assert np.all(np.diff(s) > 0)


def test_goal_pose_reaches_and_finishes_at_rest():
    ctrl = _ctrl()
    ex = ChunkExecutor(ctrl, Q_READY)
    Tg = KIN.fk(Q_READY)
    Tg[:3, 3] += [-40, 30, 25]
    ex.goal_pose(Tg)
    Q = _run(ex)
    assert ex.done
    e_p, e_r = pose_error(Tg, KIN.fk(Q[-1]))
    assert e_p < 0.05 and e_r < 0.05
    _assert_smooth(Q, ctrl.cfg)
    assert {ex.last_info.status} == {"optimal"}


def test_goal_joints_with_given_duration():
    ctrl = _ctrl()
    ex = ChunkExecutor(ctrl, Q_READY)
    goal = Q_READY + [20, 10, -15, 30, 20, -40]
    n = ex.goal_joints(goal, duration=2.0)
    assert n == 40
    Q = _run(ex)
    assert np.abs(Q[-1] - goal).max() < 0.01
    _assert_smooth(Q, ctrl.cfg)


def test_replacing_a_chunk_mid_motion_keeps_q_and_dq_continuous_and_does_not_stop():
    ctrl = _ctrl()
    cfg = ctrl.cfg
    ex = ChunkExecutor(ctrl, Q_READY)
    T0 = KIN.fk(Q_READY)
    # Chunk 1: straight line in +x at 50 mm/s for 2 s.
    line = []
    for k in range(41):
        T = T0.copy()
        T[0, 3] += 50.0 * k * cfg.dt
        line.append(T)
    ex.set_cartesian_chunk(line)
    Q = [ex.q_cmd] + [np.array(ex.tick()) for _ in range(20)]
    # Mid-motion: a new chunk that heads +y from a point 15 mm away from where we are.
    T1 = KIN.fk(ex.q_cmd)
    T1[:3, 3] += [5.0, 10.0, -10.0]
    turn = []
    for k in range(41):
        T = T1.copy()
        T[1, 3] += 40.0 * k * cfg.dt
        turn.append(T)
    ex.set_cartesian_chunk(turn)
    while not ex.done:
        Q.append(np.array(ex.tick()))
    Q = np.array(Q)
    _assert_smooth(Q, cfg)  # max |d(dq)| <= qdd_max dt^2 everywhere, incl. the switch
    speed = np.linalg.norm(np.diff(Q, axis=0), axis=1)
    assert speed[18:26].min() > 0.3 * speed[18]  # no stop at the switch (tick 20)
    e_p, _ = pose_error(turn[-1], KIN.fk(Q[-1]))
    assert e_p < 0.05


def test_extend_with_a_continuing_chunk_is_seamless():
    ctrl = _ctrl()
    cfg = ctrl.cfg
    T0 = KIN.fk(Q_READY)

    def pose(k):
        T = T0.copy()
        T[:3, 3] += np.array([30.0, -20.0, 10.0]) * k * cfg.dt
        return T

    whole = ChunkExecutor(ctrl, Q_READY)
    whole.set_cartesian_chunk([pose(k) for k in range(61)])
    Qw = _run(whole, ticks=50)

    split = ChunkExecutor(ctrl, Q_READY)
    split.set_cartesian_chunk([pose(k) for k in range(31)])
    Qs = [split.q_cmd] + [np.array(split.tick()) for _ in range(20)]
    split.extend([pose(k) for k in range(30, 61)])  # starts where the queue ends
    Qs += [np.array(split.tick()) for _ in range(30)]
    assert np.allclose(np.array(Qs), Qw, atol=1e-6)


def test_extend_with_an_offset_chunk_blends():
    ctrl = _ctrl()
    ex = ChunkExecutor(ctrl, Q_READY)
    goal1 = Q_READY + [10, 0, 0, 0, 0, 0]
    ex.goal_joints(goal1, 1.0)
    Q = [ex.q_cmd] + [np.array(ex.tick()) for _ in range(10)]
    ex.extend([goal1 + [0, 5, 0, 0, 0, 0], goal1 + [0, 5, 0, 0, 0, 20]], dt=1.0)
    while not ex.done:
        Q.append(np.array(ex.tick()))
    Q = np.array(Q)
    _assert_smooth(Q, ctrl.cfg)
    # The extension is two samples, i.e. a constant-speed segment that ends at
    # speed; the hold brakes under the accel limit, so "done" is near, not exact.
    assert np.abs(Q[-1] - (goal1 + [0, 5, 0, 0, 0, 20])).max() < 0.05


def test_joint_chunk_is_resampled_to_the_control_period():
    ctrl = _ctrl()
    ex = ChunkExecutor(ctrl, Q_READY)
    t = np.arange(0.0, 2.0001, 0.1)
    Q = Q_READY + np.outer(min_jerk(t / 2.0), [30, 10, -20, 5, 5, 40])
    n = ex.set_joint_chunk(Q, dt=0.1)
    assert n == 40
    out = _run(ex)
    assert np.abs(out[-1] - Q[-1]).max() < 0.01
    _assert_smooth(out, ctrl.cfg)


def test_extend_kind_mismatch_is_refused():
    ex = ChunkExecutor(_ctrl(), Q_READY)
    ex.goal_joints(Q_READY + 10, 1.0)
    ex.tick()
    with pytest.raises(ValueError):
        ex.extend([KIN.fk(Q_READY)])


def test_tracking_error_raises_and_latches():
    ex = ChunkExecutor(_ctrl(), Q_READY, max_track_err_deg=12.0)
    ex.goal_joints(Q_READY + [20, 0, 0, 0, 0, 0], 2.0)
    ex.tick(Q_READY)  # in step
    far = Q_READY + [0, 0, 13.0, 0, 0, 0]
    with pytest.raises(TrackingError):
        ex.tick(far)
    assert ex.fault
    with pytest.raises(TrackingError):
        ex.tick()  # latched until reset
    ex.reset(Q_READY)
    ex.tick(Q_READY)


def test_tracking_error_can_flag_instead_of_raising():
    ex = ChunkExecutor(_ctrl(), Q_READY, raise_on_track_error=False)
    ex.goal_joints(Q_READY + [20, 0, 0, 0, 0, 0], 2.0)
    first = ex.tick(Q_READY)
    held = ex.tick(Q_READY + 30.0)
    assert ex.fault and held == first


def test_lookahead_runs_the_same_plan_l_ticks_earlier():
    goal = Q_READY + [25, -10, 15, 0, 20, -30]
    a = ChunkExecutor(_ctrl(), Q_READY, lookahead_ticks=0)
    b = ChunkExecutor(_ctrl(), Q_READY, lookahead_ticks=2)
    a.goal_joints(goal, 2.0)
    b.goal_joints(goal, 2.0)
    Qa, Qb = _run(a), _run(b)
    # b's command at tick k is a's at tick k + 2: the arm is sent the reference two ticks ahead.
    m = min(len(Qa) - 2, len(Qb) - 1)
    assert np.allclose(Qb[1:m], Qa[3:m + 2], atol=1e-9)
    assert np.abs(Qb[-1] - goal).max() < 0.01


def test_lookahead_tracking_check_compares_with_where_the_arm_should_be():
    ex = ChunkExecutor(_ctrl(), Q_READY, lookahead_ticks=3, max_track_err_deg=2.0)
    ex.goal_joints(Q_READY + [40, 0, 0, 0, 0, 0], 2.0)
    for _ in range(15):
        expected = ex.q_expected
        ex.tick(expected)  # an arm exactly L ticks behind the commands is on track
    assert ex.q_cmd[0] > ex.q_expected[0]


def test_stop_decelerates_within_limits_and_comes_to_rest():
    ctrl = _ctrl()
    ex = ChunkExecutor(ctrl, Q_READY)
    ex.goal_joints(Q_READY + [80, 0, 0, 0, 0, 0], 3.0)
    Q = [ex.q_cmd] + [np.array(ex.tick()) for _ in range(30)]
    ex.stop()
    while not ex.done:
        Q.append(np.array(ex.tick()))
        assert len(Q) < 300
    _assert_smooth(np.array(Q), ctrl.cfg)


def test_idle_executor_is_done_and_holds():
    ex = ChunkExecutor(_ctrl(), Q_READY)
    assert ex.done
    assert np.allclose(ex.tick(), Q_READY)


def test_cartesian_chunk_with_floor_never_goes_below():
    ctrl = _ctrl(z_min_mm=150.0)
    ex = ChunkExecutor(ctrl, Q_READY)
    T0 = KIN.fk(Q_READY)
    dive = []
    for k in range(41):
        T = T0.copy()
        T[2, 3] -= 2.5 * k  # down 100 mm, through the floor
        dive.append(T)
    ex.set_cartesian_chunk(dive)
    Q = _run(ex)
    z = np.array([KIN.fk(q)[2, 3] for q in Q])
    assert z.min() >= 150.0 - 0.05
