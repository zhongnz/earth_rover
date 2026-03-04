from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class MissionState(Enum):
    INIT = auto()
    STARTING = auto()
    RUNNING = auto()
    RECOVERING = auto()
    STOPPING = auto()
    STOPPED = auto()
    ERROR = auto()


@dataclass
class MissionFSM:
    """Minimal mission lifecycle state machine."""

    state: MissionState = MissionState.INIT

    def on_start(self) -> None:
        if self.state in (MissionState.INIT, MissionState.STOPPED):
            self.state = MissionState.STARTING

    def on_started(self) -> None:
        if self.state == MissionState.STARTING:
            self.state = MissionState.RUNNING

    def on_recover(self) -> None:
        if self.state == MissionState.RUNNING:
            self.state = MissionState.RECOVERING

    def on_resume(self) -> None:
        if self.state == MissionState.RECOVERING:
            self.state = MissionState.RUNNING

    def on_stop(self) -> None:
        if self.state not in (MissionState.STOPPED, MissionState.ERROR):
            self.state = MissionState.STOPPING

    def on_stopped(self) -> None:
        self.state = MissionState.STOPPED

    def on_error(self) -> None:
        self.state = MissionState.ERROR

