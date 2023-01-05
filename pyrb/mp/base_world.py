from abc import ABCMeta, abstractmethod
from dataclasses import dataclass

from pyrb.mp.base_agent import MotionPlanningAgent, MotionPlanningAgentActuated
import numpy as np


class BaseMPWorld(metaclass=ABCMeta):

    def __init__(self, data, robot: MotionPlanningAgent, obstacles):
        self.data = data
        self.robot = robot
        self.obstacles = obstacles

    def sample_feasible_config(self):
        return np.random.uniform(self.robot.joint_limits[:, 0], self.robot.joint_limits[:, 1])

    def is_collision_free_config(self, config) -> bool:
        self.robot.set_config(config)
        return not self.robot.collision_manager.in_collision_other(self.obstacles.collision_manager)

    def get_min_distance_to_obstacle(self) -> float:
        return self.robot.collision_manager.min_distance_other(self.obstacles.collision_manager)

    @abstractmethod
    def reset(self):
        raise NotImplementedError()

    @abstractmethod
    def view(self):
        raise NotImplementedError()


class BaseMPTimeVaryingWorld(BaseMPWorld):

    def __init__(self, data, robot: MotionPlanningAgentActuated, obstacles):
        super().__init__(data, robot, obstacles)
        self.t = 0
        self.robot = robot

    def set_time(self, t):
        self.t = t
        self.obstacles.set_time(t)

    def reset(self):
        pass

    def view(self):
        pass


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


@dataclass
class WorldData1D:
    x: Range

    def __init__(self, xs):
        self.x = Range(*xs)
