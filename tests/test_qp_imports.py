"""The qp package's import rules (tests/test_core_has_no_ros.py skips it).

* ``mycobot_control.qp`` may import numpy, but never pymycobot, ROS, serial,
  OpenCV or the ``sdk`` session modules: the link brain imports it on a PC
  with no arm SDK. ``qp.cli`` reaches the session only inside functions.
* Nothing outside ``qp`` pulls numpy in: the dispatcher and teleop import the
  QP stack only when it is used.
"""

import subprocess
import sys

import pytest


def _loaded_after(code):
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    return [n for n in out.stdout.strip().split(",") if n]


def test_qp_loads_no_sdk_no_vendor_code():
    pytest.importorskip("numpy")
    bad = _loaded_after(
        "import sys, pkgutil, importlib, mycobot_control.qp as qp\n"
        "for m in pkgutil.iter_modules(qp.__path__, 'mycobot_control.qp.'):\n"
        "    importlib.import_module(m.name)\n"
        "print(','.join(sorted(n for n in sys.modules if n.split('.')[0] in "
        "{'rclpy', 'pymycobot', 'serial', 'cv2'} or n.startswith('mycobot_control.sdk'))))\n"
    )
    assert bad == []


def test_dispatcher_and_teleop_do_not_load_numpy():
    bad = _loaded_after(
        "import sys\n"
        "from mycobot_control import cli\n"
        "cli._commands()\n"
        "import mycobot_control.teleop.keyboard\n"
        "print(','.join(sorted(n for n in sys.modules if n.split('.')[0] == 'numpy')))\n"
    )
    assert bad == []
