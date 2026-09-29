"""Per-keypress CSV log for ``mycobot-teleop-keyboard --log``.

One row per keypress and per interlock decision, whatever the outcome - the
sessions worth reading are the ones where keys did not become commands. Gate
and latch state are stamped *after* the event. ``t_host`` is ``time.time()``,
the same clock as ``mycobot-record``, so the two CSVs join.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import IO, Any, Optional, Union

FIELDS = ("t_host", "event", "key", "action", "outcome", "gate", "latched", "measured", "target", "note")


def default_log_path(directory: Union[str, Path] = "recordings") -> Path:
    return Path(directory) / time.strftime("teleop-%Y%m%d-%H%M%S.csv")


class EventLog:
    def __init__(self, path: Union[str, Path]) -> None:
        self.csv_path = Path(path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh: Optional[IO[str]] = open(self.csv_path, "w", newline="")
        self._writer = csv.DictWriter(self._fh, fieldnames=FIELDS)
        self._writer.writeheader()
        self.rows = 0

    def record(self, event: str, **fields: Any) -> None:
        row = {"t_host": f"{time.time():.6f}", "event": event}
        for name in FIELDS[2:]:
            value = fields.get(name)
            if isinstance(value, (list, tuple)):
                value = json.dumps([round(float(v), 4) for v in value])
            row[name] = "" if value is None else value
        self._writer.writerow(row)
        if self._fh is not None:
            self._fh.flush()
        self.rows += 1

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> "EventLog":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
