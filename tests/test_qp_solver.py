import itertools
import time

import pytest

np = pytest.importorskip("numpy")

from mycobot_control.qp.solver import solve_qp  # noqa: E402


def _random_pd(rng, n, cond=1.0):
    M = rng.normal(size=(n, n))
    return M @ M.T + cond * np.eye(n)


def _kkt(H, g, C, d, lb, ub, res):
    """Worst stationarity, primal infeasibility, complementarity; and min multiplier."""
    x, lam = res.x, res.multipliers
    m, n = C.shape[0], g.size
    stat = H @ x + g + C.T @ lam[:m] - lam[m:m + n] + lam[m + n:]
    slack = np.concatenate([d - C @ x, np.where(np.isfinite(lb), x - lb, 0.0), np.where(np.isfinite(ub), ub - x, 0.0)])
    return np.abs(stat).max(), max(0.0, -slack.min()), np.abs(lam * slack).max(), lam.min()


def test_unconstrained_is_newton_step():
    rng = np.random.default_rng(0)
    H, g = _random_pd(rng, 6), rng.normal(size=6)
    assert np.allclose(solve_qp(H, g), -np.linalg.solve(H, g))


def test_box_with_diagonal_h_is_clipping():
    h = np.array([1.0, 2.0, 0.5, 4.0])
    g = np.array([-3.0, 5.0, 0.1, -0.2])
    lb, ub = np.array([-1.0, -1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0, 1.0])
    x = solve_qp(np.diag(h), g, lb=lb, ub=ub)
    assert np.allclose(x, np.clip(-g / h, lb, ub))


def test_random_feasible_problems_satisfy_kkt():
    rng = np.random.default_rng(1)
    for _ in range(300):
        n, m = 6, 40
        H, g = _random_pd(rng, n, 0.1), rng.normal(size=n) * 10
        C = rng.normal(size=(m, n))
        x0 = rng.normal(size=n)
        d = C @ x0 + rng.uniform(0.0, 1.0, size=m)
        lb, ub = x0 - rng.uniform(0, 1, size=n), x0 + rng.uniform(0, 1, size=n)
        lb[rng.random(n) < 0.2] = -np.inf  # some bounds absent
        res = solve_qp(H, g, C, d, lb, ub, full_output=True)
        assert res.ok
        stat, infeas, comp, lam_min = _kkt(H, g, C, d, lb, ub, res)
        assert stat < 1e-8 and infeas < 1e-9 and comp < 1e-8 and lam_min >= 0.0


def _brute_force(H, g, C, d):
    """Enumerate active sets; the KKT point with lam >= 0 that is feasible is THE optimum."""
    n, m = g.size, C.shape[0]
    for k in range(0, n + 1):
        for S in itertools.combinations(range(m), k):
            A = C[list(S)]
            K = np.block([[H, A.T], [A, np.zeros((k, k))]]) if k else H
            try:
                sol = np.linalg.solve(K, np.concatenate([-g, d[list(S)]]))
            except np.linalg.LinAlgError:
                continue
            x, lam = sol[:n], sol[n:]
            if np.all(C @ x <= d + 1e-9) and np.all(lam >= -1e-9):
                return x
    return None


def test_matches_brute_force_on_small_problems():
    rng = np.random.default_rng(2)
    checked = 0
    for _ in range(200):
        H, g = _random_pd(rng, 3, 0.1), rng.normal(size=3) * 3
        C, d = rng.normal(size=(6, 3)), rng.normal(size=6)
        ref = _brute_force(H, g, C, d)
        res = solve_qp(H, g, C, d, full_output=True)
        if ref is None:
            assert res.status == "infeasible"
        else:
            assert res.ok and np.allclose(res.x, ref, atol=1e-7)
            checked += 1
    assert checked > 100


def test_infeasible_bounds_flagged_never_nan():
    res = solve_qp(np.eye(1), [0.0], lb=[1.0], ub=[0.0], full_output=True)
    assert res.status == "infeasible" and np.all(np.isfinite(res.x))


def test_infeasible_general_constraints_flagged_never_nan():
    # x1 + x2 <= -1 with x >= 0.
    res = solve_qp(np.eye(2), [0.0, 0.0], C=[[1.0, 1.0]], d=[-1.0], lb=[0.0, 0.0], full_output=True)
    assert res.status == "infeasible" and np.all(np.isfinite(res.x))
    x = solve_qp(np.eye(2), [0.0, 0.0], C=[[1.0, 1.0]], d=[-1.0], lb=[0.0, 0.0])
    assert np.all(np.isfinite(x))


def test_degenerate_duplicate_and_equality_constraints():
    rng = np.random.default_rng(4)
    H, g = _random_pd(rng, 4), rng.normal(size=4) * 5
    row = rng.normal(size=4)
    C = np.vstack([row, row, 2 * row, rng.normal(size=4)])
    d = np.array([0.1, 0.1, 0.2, 1.0])
    lb = np.array([-1.0, 0.3, -np.inf, -1.0])
    ub = np.array([1.0, 0.3, np.inf, 1.0])  # x2 pinned: lb == ub
    res = solve_qp(H, g, C, d, lb, ub, full_output=True)
    assert res.ok
    assert abs(res.x[1] - 0.3) < 1e-9
    stat, infeas, comp, lam_min = _kkt(H, g, C, d, lb, ub, res)
    assert stat < 1e-8 and infeas < 1e-9 and comp < 1e-8 and lam_min >= 0.0


def test_bad_inputs_raise():
    with pytest.raises(ValueError):
        solve_qp(np.array([[1.0, 0.0], [0.0, -1.0]]), [0.0, 0.0])  # not PD
    with pytest.raises(ValueError):
        solve_qp(np.eye(2), [0.0, float("nan")])
    with pytest.raises(ValueError):
        solve_qp(np.eye(2), [0.0, 0.0], C=np.eye(2), d=[1.0])


def test_controller_sized_problem_solves_well_under_2ms():
    rng = np.random.default_rng(5)
    problems = []
    for _ in range(100):
        H, g = _random_pd(rng, 6, 0.1), rng.normal(size=6) * 10
        C = rng.normal(size=(28, 6))
        x0 = rng.normal(size=6)
        problems.append((H, g, C, C @ x0 + rng.uniform(0, 1, 28), x0 - 1, x0 + 1))
    times = []
    for p in problems:
        t = time.perf_counter()
        solve_qp(*p)
        times.append(time.perf_counter() - t)
    assert np.median(times) < 2e-3
