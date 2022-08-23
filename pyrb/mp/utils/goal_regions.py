import numpy as np


class RealVectorGoalRegion:

    def __init__(self, state=None, radius=.1):
        self.state = state
        self.radius = radius

    def set_goal_state(self, state):
        self.state = state

    def is_within(self, state):
        return np.linalg.norm(state - self.state) < self.radius

    def mask_is_within(self, states):
        return np.linalg.norm(states - self.state, axis=1) < self.radius


class RealVectorTimeGoalRegion(RealVectorGoalRegion):

    def __init__(self, state=None, radius=.1):
        self.state = state
        self.radius = radius

    def set_goal_state(self, state):
        self.state = state

    def is_within(self, state):
        return np.linalg.norm(state[:-1] - self.state[:-1]) < self.radius and state[-1] == self.state[-1]

    def mask_is_within(self, states):
        return (np.linalg.norm(states[:, :-1] - self.state[:-1], axis=1) < self.radius) & (states[:, -1] == self.state[-1])
