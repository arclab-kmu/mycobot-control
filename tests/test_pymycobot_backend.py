"""PyMyCobotBackend against a stand-in pymycobot module, and - when the real
SDK is installed - against its actual signatures."""

import inspect
import sys
import types

import pytest

from mycobot_control.sdk.backend import PyMyCobotBackend


class FakeMC:
    instances = []

    def __init__(self, *args, **kwargs):
        self.args, self.kwargs, self.calls = args, kwargs, []
        FakeMC.instances.append(self)

    def __getattr__(self, name):
        def call(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            return {"get_angles": [1, 2, 3, 4, 5, 6], "get_coords": -1, "get_error_information": 0,
                    "is_power_on": 1, "get_gripper_value": 42}.get(name)
        return call


@pytest.fixture
def fake_sdk(monkeypatch):
    FakeMC.instances.clear()
    mod = types.ModuleType("pymycobot")
    mod.MyCobot280 = FakeMC
    mod.MyCobot280Socket = FakeMC
    monkeypatch.setitem(sys.modules, "pymycobot", mod)
    return mod


def test_serial_backend_calls(fake_sdk, tmp_path, monkeypatch):
    monkeypatch.setattr("mycobot_control.sdk.portlock.VENDOR_LOCK_PATH", str(tmp_path / "lock"))
    b = PyMyCobotBackend(lock=False)
    b.connect()
    mc = FakeMC.instances[-1]
    assert mc.args == ("/dev/ttyTHS1", 1000000)
    assert b.get_joint_state().angles == [1, 2, 3, 4, 5, 6]
    assert b.get_coords() is None  # -1 from the SDK is "no reading"
    assert b.get_gripper_value() == 42
    assert b.read_status().is_normal
    b.send_angles([0] * 6, 20)
    b.set_gripper_value(30, 50)
    assert ("send_angles", ([0.0] * 6, 20), {"_async": True}) in mc.calls
    assert ("set_gripper_value", (30, 50), {"gripper_type": 1}) in mc.calls
    b.disconnect()
    assert mc.calls[-1][0] == "close"
    assert not b.is_connected()


def test_socket_backend_needs_host():
    with pytest.raises(ValueError):
        PyMyCobotBackend(transport="socket")


def test_socket_backend_connects_to_host(fake_sdk):
    b = PyMyCobotBackend(transport="socket", host="10.0.0.5")
    b.connect()
    assert FakeMC.instances[-1].args == ("10.0.0.5", 9000)


def test_calls_bind_to_real_pymycobot_signatures():
    real = pytest.importorskip("pymycobot")
    cls = real.MyCobot280
    bind = lambda name, *a, **k: inspect.signature(getattr(cls, name)).bind(None, *a, **k)  # noqa: E731
    inspect.signature(cls).bind("/dev/ttyTHS1", 1000000)
    inspect.signature(real.MyCobot280Socket).bind("10.0.0.5", 9000)
    bind("send_angles", [0.0] * 6, 20, _async=True)
    bind("send_coords", [0.0] * 6, 20, 0, _async=True)
    bind("set_gripper_value", 30, 50, gripper_type=1)
    bind("get_gripper_value", 1)
    for name in ("get_angles", "get_coords", "get_error_information", "is_power_on", "power_on",
                 "stop", "clear_error_information", "close", "get_system_version", "get_fresh_mode"):
        bind(name)
    bind("set_fresh_mode", 1)
    bind("get_joint_min_angle", 1)
