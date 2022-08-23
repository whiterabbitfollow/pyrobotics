import numpy as np


class RealVectorGoalRegion:

    def __init__(self, state=None, radius=.1):
        self.state = state
        self.radius = radius

    def set_goal_state(self, state):
        self.state = state

    def is_within(self, state):
        return np.linalg.norm(state - self.state) < self.radius


class RealVectorTimeGoalRegion(RealVectorGoalRegion):

    def __init__(self, state=None, radius=.1):
        self.state = state
        self.radius = radius

    def set_goal_state(self, state):
        self.state = state

    def is_within(self, state):
        return np.linalg.norm(state[:-1] - self.state[:-1]) < self.radius and state[-1] == self.state[-1]
