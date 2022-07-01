from abc import ABCMeta, abstractmethod


class BaseMPWorld(metaclass=ABCMeta):

    def __init__(self, data, robot, obstacles):
        self.data = data
        self.robot = robot
        self.obstacles = obstacles

    @abstractmethod
    def is_collision_free_transition(self, state_src, state_dst):
        raise NotImplementedError()

    @abstractmethod
    def is_collision_free_state(self, state) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def reset(self):
        raise NotImplementedError()

    @abstractmethod
    def view(self):
        raise NotImplementedError()

