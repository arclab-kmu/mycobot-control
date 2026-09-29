import os
import sys

import pytest

from mycobot_control.sdk.backend import DryRunBackend
from mycobot_control.sdk.session import ArmConfig, ArmSession


class FakeReader:
    """Feeds keys to teleop.run(); None entries are poll timeouts. EOF when exhausted."""

    def __init__(self, keys):
        self._keys = list(keys)

    def read(self, timeout):
        return self._keys.pop(0) if self._keys else ""


@pytest.fixture
def dry():
    return DryRunBackend()


@pytest.fixture
def session(dry):
    s = ArmSession(ArmConfig(), backend=dry)
    s.start()
    yield s
    s.stop()
