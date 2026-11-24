"""A small dense convex QP solver: Goldfarb-Idnani dual active set, pure numpy.

Solves::

    minimise   1/2 x^T H x + g^T x
    subject to C x <= d,   lb <= x <= ub

for strictly convex ``H`` (symmetric positive definite). Sized for the
controller: n = 6 variables, a few dozen constraints, solved every tick.

Why Goldfarb-Idnani: it starts from the unconstrained minimum, so it needs no
feasible starting point (no phase 1), every iterate is dual feasible, and when
a violated constraint cannot be added it has *proved* the problem infeasible
rather than run out of iterations. The active-set factorisation is recomputed
with a QR of ``L^-1 N`` each iteration - the matrices are at most 6x6, so a
fresh factorisation is cheaper in Python than an update and has no drift.

Reference: D. Goldfarb, A. Idnani, "A numerically stable dual method for
solving strictly convex quadratic programs", Math. Programming 27 (1983).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

OPTIMAL = "optimal"
INFEASIBLE = "infeasible"
MAX_ITER = "max_iter"


@dataclass
class QPResult:
    """Everything :func:`solve_qp` knows about a solve.

    Attributes:
        x: the solution; for ``status != "optimal"`` a best-effort point
            (finite, never NaN) that satisfies the constraints active when the
            solver stopped. Not feasible in general.
        status: ``"optimal"``, ``"infeasible"`` or ``"max_iter"``.
        active: indices of active constraints. ``0 .. m-1`` are the rows of
            ``C``; ``m + i`` is ``lb[i]``; ``m + n + i`` is ``ub[i]``.
        multipliers: Lagrange multipliers (>= 0) in the same order, length
            ``m + 2n``: ``H x + g + C^T lam_C - lam_lb + lam_ub = 0``.
        iterations: inner iterations taken.
        obj: ``1/2 x^T H x + g^T x``.
    """

    x: np.ndarray
    status: str
    active: List[int] = field(default_factory=list)
    multipliers: Optional[np.ndarray] = None
    iterations: int = 0
    obj: float = 0.0

    @property
    def ok(self) -> bool:
        return self.status == OPTIMAL


def solve_qp(H, g, C=None, d=None, lb=None, ub=None, *, tol: float = 1e-9,
             max_iter: Optional[int] = None, full_output: bool = False):
    """Minimise ``1/2 x^T H x + g^T x`` s.t. ``C x <= d``, ``lb <= x <= ub``.

    Args:
        H: (n, n) symmetric positive definite.
        g: (n,).
        C, d: (m, n) and (m,), or both ``None``.
        lb, ub: (n,) or ``None``; entries may be +/-inf.
        tol: relative feasibility tolerance.
        max_iter: cap on inner iterations (default ``10 * (m + 2n) + 20``).
        full_output: return a :class:`QPResult` instead of only ``x``.

    Returns:
        ``x`` (ndarray), or a :class:`QPResult` with ``full_output=True``. An
        infeasible problem never raises and never returns NaN: ``x`` is the
        best-effort point and ``status`` says ``"infeasible"`` (only visible
        with ``full_output=True``).

    Raises:
        ValueError: shapes do not match, an input is NaN, or ``H`` is not
            positive definite.
    """
    H = np.asarray(H, dtype=float)
    g = np.asarray(g, dtype=float).reshape(-1)
    n = g.size
    if H.shape != (n, n):
        raise ValueError(f"H must be ({n}, {n}), got {H.shape}")
    if not (np.all(np.isfinite(H)) and np.all(np.isfinite(g))):
        raise ValueError("H and g must be finite")
    H = 0.5 * (H + H.T)

    if C is None:
        C = np.zeros((0, n))
        d = np.zeros(0)
    C = np.asarray(C, dtype=float).reshape(-1, n)
    d = np.asarray(d, dtype=float).reshape(-1)
    m = C.shape[0]
    if d.shape != (m,):
        raise ValueError(f"d must have {m} entries, got {d.shape}")
    lb = np.full(n, -np.inf) if lb is None else np.asarray(lb, dtype=float).reshape(n)
    ub = np.full(n, np.inf) if ub is None else np.asarray(ub, dtype=float).reshape(n)
    if np.any(np.isnan(C)) or np.any(np.isnan(d)) or np.any(np.isnan(lb)) or np.any(np.isnan(ub)):
        raise ValueError("constraints must not contain NaN")

    # Everything as  A x >= b  (GI's form): -C x >= -d;  x >= lb;  -x >= -ub.
    eye = np.eye(n)
    A = np.vstack([-C, eye, -eye])
    b = np.concatenate([-d, lb, -ub])
    A_norm = np.linalg.norm(A, axis=1)
    A_norm[A_norm == 0.0] = 1.0
    finite = np.isfinite(b)
    # A zero row with b <= 0 is always satisfied; with b > 0 never. Handled by
    # the general path (z = 0, no dual step possible -> infeasible).

    try:
        L = np.linalg.cholesky(H)
    except np.linalg.LinAlgError:
        raise ValueError("H must be positive definite") from None
    Linv = np.linalg.inv(L)          # H^-1 = Linv^T Linv
    x = -(Linv.T @ (Linv @ g))

    total = m + 2 * n
    if max_iter is None:
        max_iter = 10 * total + 20
    active: List[int] = []
    u = np.zeros(0)
    it = 0
    status = OPTIMAL
    scale = tol * (1.0 + np.where(finite, np.abs(b), 0.0))

    while True:
        s = np.where(finite, A @ x - np.where(finite, b, 0.0), np.inf)
        viol = s / A_norm
        if active:
            viol[active] = np.inf
        p = int(np.argmin(viol))
        if not s[p] < -scale[p] * A_norm[p]:
            break  # all satisfied
        n_p = A[p]
        u_p = 0.0
        added = False
        while not added:
            it += 1
            if it > max_iter:
                status = MAX_ITER
                break
            q = len(active)
            w = Linv @ n_p
            if q:
                Q, R = np.linalg.qr(Linv @ A[active].T, mode="complete")
                dv = Q.T @ w
                r = np.linalg.solve(R[:q, :q], dv[:q])
                z = Linv.T @ (Q[:, q:] @ dv[q:])
                dz2 = float(dv[q:] @ dv[q:])
            else:
                r = np.zeros(0)
                z = Linv.T @ w
                dz2 = float(w @ w)
            # Partial (dual) step: the largest t keeping active multipliers >= 0.
            t1, k = np.inf, -1
            for j in range(q):
                if r[j] > 1e-12:
                    tj = u[j] / r[j]
                    if tj < t1:
                        t1, k = tj, j
            # Full (primal) step: makes constraint p exactly active. z = 0 when
            # n_p is linearly dependent on the active normals.
            if dz2 <= 1e-14 * float(w @ w):
                t2 = np.inf
            else:
                t2 = (b[p] - float(n_p @ x)) / dz2
            if not np.isfinite(t1) and not np.isfinite(t2):
                status = INFEASIBLE
                break
            if not np.isfinite(t2):
                # Dual step only: drop the blocking constraint and retry.
                u = u - t1 * r
                u_p += t1
                del active[k]
                u = np.delete(u, k)
                continue
            t = min(t1, t2)
            x = x + t * z
            u = u - t * r
            u_p += t
            if t2 <= t1:
                active.append(p)
                u = np.append(u, u_p)
                added = True
            else:
                del active[k]
                u = np.delete(u, k)
        if status != OPTIMAL:
            break

    if not np.all(np.isfinite(x)):  # defensive: the algebra above cannot produce NaN from finite inputs
        x = np.zeros(n)
        status = INFEASIBLE
    if not full_output:
        return x
    lam = np.zeros(total)
    if status == OPTIMAL:
        lam[active] = np.maximum(u, 0.0)
    obj = float(0.5 * x @ H @ x + g @ x)
    return QPResult(x=x, status=status, active=list(active), multipliers=lam, iterations=it, obj=obj)
