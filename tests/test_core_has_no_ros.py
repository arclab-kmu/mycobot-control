"""Invariant 1: nothing on the core import path imports ROS or pymycobot."""

import ast
import subprocess
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1] / "src" / "mycobot_control"
FORBIDDEN_ROOTS = {"rclpy", "geometry_msgs", "sensor_msgs", "std_msgs", "moveit", "torch", "isaaclab"}


def _imported_roots(path):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name.split(".")[0], node
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.module.split(".")[0], node


def test_no_forbidden_imports_outside_ros():
    offenders = []
    for path in PKG.rglob("*.py"):
        if "ros" in path.relative_to(PKG).parts[:1]:
            continue
        for root, node in _imported_roots(path):
            if root in FORBIDDEN_ROOTS:
                offenders.append(f"{path.relative_to(PKG)}:{node.lineno} imports {root}")
    assert not offenders, offenders


def test_pymycobot_only_imported_inside_functions():
    # Module-level import of the vendor SDK would put hardware on the default path.
    for path in PKG.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in tree.body:
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            assert not any(n.split(".")[0] == "pymycobot" for n in names), path


def test_importing_everything_loads_neither_ros_nor_pymycobot():
    # A subprocess, not sys.modules surgery: clearing modules in-process breaks
    # class identity for the other tests (nero-control learned this).
    code = (
        "import sys, pkgutil, importlib, mycobot_control\n"
        "for m in pkgutil.walk_packages(mycobot_control.__path__, 'mycobot_control.'):\n"
        "    importlib.import_module(m.name)\n"
        "bad = sorted(n for n in sys.modules if n.split('.')[0] in "
        "{'rclpy', 'pymycobot', 'serial', 'numpy'})\n"
        "print(','.join(bad))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == ""
