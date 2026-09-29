# AGENTS.md

Guidance for coding agents working in this repository.
(`CLAUDE.md` is a symlink to this file. Edit `AGENTS.md`.)

Python control for the Elephant Robotics **myCobot 280 JetsonNano** with the
adaptive gripper, modelled on `~/manipulation/nero-control` (the AgileX NERO
equivalent). Read that repo's `AGENTS.md` for the reasoning behind the shared design.

## Commands

```bash
uv sync --extra dev              # .venv + this package editable; NO pymycobot
uv sync --extra dev --extra arm  # ...and keep pymycobot. Plain `--extra dev`
                                 #    UNINSTALLS it: sync prunes to named extras.
uv run pytest                    # full suite; no hardware, no SDK needed
uv run pytest tests/test_keymap.py::test_joint_opposite_keys_cancel
uv run --isolated --python 3.8 --with pytest --with-editable . pytest   # 3.8 floor
uv build
```

The suite must pass with neither ROS nor pymycobot installed.
`test_calls_bind_to_real_pymycobot_signatures` only runs when pymycobot is
present; run it with the `arm` extra after bumping the SDK.

Entry points: `mycobot` (dispatcher: teleop / info / joints / record),
`mycobot-teleop-keyboard`, `mycobot-info`, `mycobot-move-joints`, `mycobot-record`.

## The invariants (same as nero-control)

1. **No ROS and no pymycobot on the core import path.** `ros/` is the only
   place `rclpy` may appear; `PyMyCobotBackend.connect()` is the only place
   `pymycobot` is imported. `tests/test_core_has_no_ros.py` enforces both, in
   a subprocess.
2. **Hardware is never the default.** `ArmConfig.live=False` builds
   `DryRunBackend`. `--live` is the only route to the arm and asks for a typed
   `yes` (refuses on a non-TTY without `--assume-yes`). `mycobot-record` and
   `mycobot-info` always wrap a live backend in `ReadOnlyBackend`.
3. **Claims are marked.** Anything not confirmed on this arm carries
   `TODO(unverified):` where it matters. SDK signatures are quoted from
   pymycobot 4.0.7 in `sdk/backend.py`; re-check them with `inspect.signature`
   rather than recalling them.

## Architecture

- `sdk/backend.py`: `ArmBackend` Protocol; `DryRunBackend`, `PyMyCobotBackend`
  (serial `MyCobot280` on the Jetson's `/dev/ttyTHS1` @ 1 Mbaud, or
  `MyCobot280Socket` from a PC), `ReadOnlyBackend` (drops commands, forwards
  `stop()`). `valid_vector()` turns pymycobot's `-1` / `None` / partial replies
  into "no reading".
- `sdk/session.py`: `ArmConfig` + `ArmSession` own backend, connection and
  `ControlGate`. Every command goes through `command_joints` / `command_coords`
  / `command_gripper`, all behind `gate.require_open()`. `command_joints`
  refuses targets within 3 deg of the joint table and any jump over 20 deg
  from the *measured* angles (catches rad/deg mix-ups and stale reads).
- `sdk/status.py`: `get_error_information()` + `is_power_on()` →
  `should_halt` (joint limit 1-6, collision 16-19, unknown codes, power off)
  vs `was_rejected` (IK 32-34: reported, not latched).
- `sdk/portlock.py`: takes the vendor's `/tmp/mycobot_lock` (serial only) for
  the whole session, so this package and a vendor ROS node cannot share the UART.
- `safety/`: `ControlGate`, `Deadman` (0.5 s), `limits.py` (degrees and mm -
  the SDK's units - throughout).
- `teleop/keymap.py` is pure; `teleop/keyboard.py` is the loop. Fault latch as
  in nero-control: **the deadman must never override a halting status; only
  the arm reporting error 0 clears it.** Space = gate closed + `stop()`.
- `record/`: fixed-rate CSV (joints in deg *and* rad, coords, gripper, status,
  per-read latency) + `.meta.json`. `t_host` = `time.time()`, same as the
  teleop `--log`, so the two join.

## Deliberate omissions

- Nothing calls `release_all_servos()` or `power_off()`: both drop torque and a
  raised arm falls. Stopping means stop sending, or `stop()`.
- No `--home`/park preset: the zero pose stands the arm upright; whether the
  path is clear depends on the tool and table.
- No continuous/velocity jog yet. Measure the link first (see README bring-up).

## Relation to other repos

- `~/manipulation/mycobot_ros2`: vendor ROS 2 packages and the URDF
  (`mycobot_description/urdf/mycobot_280_jn/`). The shipped URDF is malformed;
  `~/manipulation/arc-mycobot/src/arc_mycobot/assets/robots/mycobot_urdf.py`
  repairs it. Do not edit the vendor checkout.
- `~/manipulation/arc-mycobot`: Isaac Lab sim tasks (Python 3.11, torch). Never
  import it here; exchange files. Joint order matches `record/recorder.py`'s
  `joint_order`. Sim home poses there are sim-only, not measured.

## Conventions

Comments explain *why*. Tests assert behaviour, not wording, and inject clocks
instead of sleeping. Keep Python 3.8 syntax (`typing.List`/`Optional` wherever
evaluated at runtime); the code is meant to run on the Jetson too.
