from abc import ABCMeta, abstractmethod
from dataclasses import dataclass

from pyrb.mp.base_agent import MotionPlanningAgent, MotionPlanningAgentActuated
import numpy as np


class BaseMPWorld(metaclass=ABCMeta):

    def __init__(self, data, robot: MotionPlanningAgent, obstacles):
        self.data = data
        self.robot = robot
        self.obstacles = obstacles

    def sample_collision_free_state(self):
        while True:
            state = np.random.uniform(self.robot.joint_limits[:, 0], self.robot.joint_limits[:, 1])
            if self.is_collision_free_state(state):
                return state

    def is_collision_free_transition(self, state_src, state_dst, nr_coll_steps=10):
        # Assumes state_src is collision free...
        is_collision_free = False
        for i in range(1, nr_coll_steps + 1):
            alpha = i / nr_coll_steps
            state = state_dst * alpha + (1 - alpha) * state_src
            is_collision_free = self.is_collision_free_state(state)
            if not is_collision_free:
                break
        return is_collision_free

    def is_collision_free_state(self, state) -> bool:
        self.robot.set_config(state)
        return not self.robot.collision_manager.in_collision_other(self.obstacles.collision_manager)

    @abstractmethod
    def reset(self):
        raise NotImplementedError()

    @abstractmethod
    def view(self):
        raise NotImplementedError()


class BaseMPTimeVaryingWorld(BaseMPWorld):

    def __init__(self, data, robot: MotionPlanningAgentActuated, obstacles):
        self.t = 0
        super().__init__(data, robot, obstacles)

    def sample_collision_free_state(self):
        while True:
            config = np.random.uniform(self.robot.joint_limits[:, 0], self.robot.joint_limits[:, 1])
            if super().is_collision_free_state(config):
                return config

    def is_collision_free_state(self, state) -> bool:
        config = state[:-1]
        t = state[-1]
        self.set_time(t)
        return super().is_collision_free_state(config)

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

    def __init__(self, xs, ys):
        self.x = Range(*xs)
