from dataclasses import dataclass


@dataclass
class Range:
    lower: float
    upper: float

    def to_tuple(self):
        return self.lower, self.upper


@dataclass
class WorldData2D:
    x: Range
    y: Range

    def __init__(self, xs, ys):
        self.x = Range(*xs)
        self.y = Range(*ys)


class WorldData3D:
    x: Range
    y: Range
    z: Range
    lower: list
    upper: list

    def __init__(self, xs, ys, zs):
        self.x = Range(*xs)
        self.y = Range(*ys)
        self.z = Range(*zs)
        self.lower = [l for (l, u) in (xs, ys, zs)]
        self.upper = [u for (l, u) in (xs, ys, zs)]
