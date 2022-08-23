from enum import Enum, auto


class TimeModes(Enum):
    FORWARD = auto()
    BACKWARD = auto()


class LocalPlannerStatus(Enum):
    TRAPPED = auto()
    ADVANCED = auto()
    REACHED = auto()

