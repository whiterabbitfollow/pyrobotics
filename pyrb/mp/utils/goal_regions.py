import logging

import numpy as np

GOAL_REGIONS_LOGGER_NAME = __file__
logger = logging.getLogger(GOAL_REGIONS_LOGGER_NAME)


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
        self.time_horizon = np.inf

    def set_goal_state(self, state):
        self.state = state
        self.time_horizon = state[-1]

    def is_within(self, state):
        return np.linalg.norm(state[:-1] - self.state[:-1]) < self.radius and state[-1] == self.state[-1]

    def mask_is_within(self, states):
        return (np.linalg.norm(states[:, :-1] - self.state[:-1], axis=1) < self.radius) & (states[:, -1] == self.state[-1])

    def is_config_within(self, state):
        return np.linalg.norm(state[:-1] - self.state[:-1]) < self.radius

    def are_configs_within(self, states):
        return np.linalg.norm(states[:, :-1] - self.state[:-1], axis=1) < self.radius


class RealVectorMinimizingTimeGoalRegion(RealVectorTimeGoalRegion):

    def is_within(self, state):
        within = super().is_config_within(state)
        if within and (np.isinf(self.time_horizon) or state[-1] < self.time_horizon):
            logger.debug("New time horizon %s Old: %s", state[-1], self.time_horizon)
            self.time_horizon = state[-1]
            self.state[-1] = self.time_horizon
        return within

    def mask_is_within(self, states):
        mask_within = super().are_configs_within(states)
        if mask_within.any() and (np.isinf(self.time_horizon) or states[mask_within,-1].min() < self.time_horizon):
            time_horizon = states[mask_within, -1].min()
            logger.debug("New time horizon %s Old: %s", time_horizon, self.time_horizon)
            self.time_horizon = time_horizon
            self.state[-1] = self.time_horizon
        return mask_within
