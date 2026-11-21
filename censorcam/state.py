# CensorCam - Copyright (C) 2026 Aeternum Logic. Developed by Brian Rawls.
# Licensed under the GNU Affero General Public License v3.0 - see LICENSE.
"""Shared, thread-safe operator state (the mode the streamer selected)."""

from __future__ import annotations

import threading
import time
from enum import Enum
from typing import Callable, List


class Mode(str, Enum):
    NORMAL = "normal"
    PEOPLE = "people"   # BLUR PEOPLE
    ALL = "all"         # BLUR ALL

    @classmethod
    def parse(cls, value: str) -> "Mode":
        v = value.strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "normal": cls.NORMAL, "off": cls.NORMAL, "none": cls.NORMAL, "pass": cls.NORMAL,
            "people": cls.PEOPLE, "blur_people": cls.PEOPLE, "person": cls.PEOPLE, "persons": cls.PEOPLE,
            "all": cls.ALL, "blur_all": cls.ALL, "full": cls.ALL, "panic": cls.ALL,
        }
        if v not in aliases:
            raise ValueError(f"unknown mode {value!r} (use normal | people | all)")
        return aliases[v]

    @property
    def strictness(self) -> int:
        return {Mode.NORMAL: 0, Mode.PEOPLE: 1, Mode.ALL: 2}[self]

    @property
    def label(self) -> str:
        return {Mode.NORMAL: "NORMAL", Mode.PEOPLE: "BLUR PEOPLE", Mode.ALL: "BLUR ALL"}[self]


class OperatorState:
    """Holds the operator-selected mode. Mode changes are only ever made by the
    operator (Stream Deck / HTTP / OBS scene) -- never by the AI."""

    def __init__(self, start_mode: Mode = Mode.ALL):
        self._lock = threading.Lock()
        self._mode = start_mode
        self._prev_non_all = Mode.NORMAL if start_mode == Mode.ALL else start_mode
        self._changed_at = time.monotonic()
        self._listeners: List[Callable[[Mode, Mode], None]] = []
        self.latency_probe = False

    @property
    def mode(self) -> Mode:
        return self._mode  # atomic read of a reference

    @property
    def changed_at(self) -> float:
        return self._changed_at

    def add_listener(self, fn: Callable[[Mode, Mode], None]) -> None:
        self._listeners.append(fn)

    def set_mode(self, mode: Mode, source: str = "api") -> Mode:
        with self._lock:
            old = self._mode
            if mode == old:
                return mode
            if old != Mode.ALL:
                self._prev_non_all = old
            self._mode = mode
            self._changed_at = time.monotonic()
        for fn in list(self._listeners):
            try:
                fn(old, mode)
            except Exception:  # listeners must never break mode switching
                pass
        print(f"[mode] {old.label} -> {mode.label} (via {source})", flush=True)
        return mode

    def toggle_all(self, source: str = "api") -> Mode:
        """BLUR ALL on/off. Turning it off returns to the previous mode."""
        if self._mode == Mode.ALL:
            return self.set_mode(self._prev_non_all, source)
        return self.set_mode(Mode.ALL, source)

    def toggle_people(self, source: str = "api") -> Mode:
        if self._mode == Mode.PEOPLE:
            return self.set_mode(Mode.NORMAL, source)
        return self.set_mode(Mode.PEOPLE, source)
