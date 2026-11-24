"""Keyboard Cartesian steps through the QP, for ``mycobot-teleop-keyboard --ik qp``.

A key becomes a short min-jerk ``goal_pose`` chunk; the teleop loop calls
:meth:`QPJog.tick` every ``dt`` and sends the result with ``command_joints``.
Compared with firmware ``send_coords`` the arm moves along a straight line
under the package's own joint limits, and a held key re-targets the running
chunk without stopping.

The I/O (reading joints, the gate, the session) stays in ``teleop/keyboard.py``;
this module is numpy only and imports nothing from ``sdk``.
"""

from __future__ import annotations

import dataclasses
from typing import List, Optional, Sequence

from mycobot_control.qp.chunk import ChunkExecutor
from mycobot_control.qp.controller import ControllerConfig, QPController
from mycobot_control.qp.kinematics import Kinematics, pose_from_xyzrpy, xyzrpy_from_pose

#: Duration of the chunk one key press asks for. A 5 mm step then peaks at
#: ~38 mm/s. Short enough to finish inside the 0.5 s deadman window.
DEFAULT_STEP_S = 0.25


class QPJog:
    """One executor per burst of keys; re-anchored on the measured joints when idle.

    Args:
        rate_hz: tick rate (the controller's ``dt`` is ``1 / rate_hz``).
        step_s: chunk duration per key.
        lookahead_ticks: see :class:`~mycobot_control.qp.chunk.ChunkExecutor`.
        config: controller config; its ``dt`` is overridden by ``rate_hz``.
    """

    def __init__(self, rate_hz: float = 20.0, step_s: float = DEFAULT_STEP_S, lookahead_ticks: int = 2,
                 config: Optional[ControllerConfig] = None, kin: Optional[Kinematics] = None) -> None:
        if rate_hz <= 0:
            raise ValueError("rate_hz must be positive")
        cfg = dataclasses.replace(config if config is not None else ControllerConfig(), dt=1.0 / float(rate_hz))
        self.kin = kin if kin is not None else Kinematics()
        self.ctrl = QPController(self.kin, cfg)
        self.step_s = float(step_s)
        self.lookahead_ticks = int(lookahead_ticks)
        self.ex: Optional[ChunkExecutor] = None

    @property
    def dt(self) -> float:
        return self.ctrl.cfg.dt

    @property
    def active(self) -> bool:
        """A chunk is running (or still settling): keep ticking."""
        return self.ex is not None and not self.ex.done

    def start_coords(self, q_measured_deg: Sequence[float]) -> List[float]:
        """Firmware-style ``[x, y, z, rx, ry, rz]`` a new key step should start from.

        While moving, that is the commanded pose (so a held key does not
        stack steps on a lagging measurement). When idle, the executor is
        re-anchored on the *measured* joints first - the same rule as joint
        teleop: no target is built on a pose the arm never reached.
        """
        if self.ex is None or self.ex.done:
            self.ex = ChunkExecutor(self.ctrl, q_measured_deg, lookahead_ticks=self.lookahead_ticks)
        return xyzrpy_from_pose(self.kin.fk(self.ex.q_cmd)).tolist()

    def goal(self, coords: Sequence[float]) -> None:
        """Retarget to ``coords`` (call :meth:`start_coords` first)."""
        if self.ex is None:
            raise RuntimeError("start_coords() first")
        self.ex.goal_pose(pose_from_xyzrpy(coords), self.step_s)

    def tick(self, q_measured_deg: Optional[Sequence[float]] = None) -> List[float]:
        if self.ex is None:
            raise RuntimeError("no motion to tick")
        return self.ex.tick(q_measured_deg)

    def cancel(self) -> None:
        """Drop the running chunk. The arm holds the last command it was sent."""
        self.ex = None
